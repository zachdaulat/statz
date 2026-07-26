
#' Fit a Distributional Synthetic Control
#' 
#' @param data A data frame or tibble.
#' @param response Unquoted column name containing the distribution response values, 
#'   as in the target variable to be replicated by the synthetic control.
#' @param unit_id Unquoted column name containing the unit identifiers.
#' @param treated_unit The specific identifier in `unit_id` represeting the treated unit.
#' @param bucket Optional string (e.g., "1 hour", "1 day") to aggregate time, 
#'   or an unquoted column name containing a string or factor grouping variable (e.g., period_name).
#' @param time Unquoted column name containing the time series.
#' @param scale_method Choose centering method; one of "non", "additive", "multiplicative"
#' @param location Metric for center. Either "median" or "mean"
#' @param n_quantiles Integer. Number of quantiles sampled for the Wasserstein grid
#' @param lambda Numeric. Ridge regularization penalty
#' @param max_iter Integer. Maximum iterations for gradient descent.
#' @param tol Numeric. Convergence tolerance.
#' @return A model object of class `z_dsc`
#' @export
z_dsc <- function(data, response, unit_id, treated_unit, bucket,
                  time = NULL,
                  scale_method = c("none", "additive", "multiplicative"),
                  location = c("median", "mean"),
                  n_quantiles = 100L, lambda = 0.1,
                  max_iter = 10000L, tol = 1e-8) {

  scale_method <- rlang::arg_match(scale_method)
  location <- rlang::arg_match(location)
  
  unit_var <- rlang::enquo(unit_id)
  resp_var <- rlang::enquo(response)
  bucket_var <- rlang::enquo(bucket) 
  time_var <- rlang::enquo(time)

  df <- dplyr::select(data, 
                      .unit = !!unit_var, 
                      .response = !!resp_var)
  
  # --- 3. Hybrid Bucketing Logic ---
  if (!rlang::quo_is_null(time_var)) {
    df$.time <- dplyr::pull(data, !!time_var)
  }
  if (!rlang::quo_is_null(bucket_var)) {
    bucket_expr <- rlang::quo_get_expr(bucket_var)
    
    if (is.character(bucket_expr)) {
      # Path A: User provided a lubridate string (e.g., "1 hour")
      if (!inherits(df$.time, c("POSIXct", "Date"))) {
        rlang::abort("`time` column must be a datetime object when a string is provided for `bucket`.")
      }
      df$.bucket <- lubridate::floor_date(df$.time, unit = bucket_expr)
    } else {
      # Path B: User provided an unquoted column name (e.g., period_name)
      # FIXED: Changed from dplyr::eval_tidy to rlang::eval_tidy
      df$.bucket <- rlang::eval_tidy(bucket_var, data = data) 
    }
  } else {
    # Default: Use the raw time column
    df$.bucket <- df$.time
  }
  
  df$.bucket <- as.factor(df$.bucket)
  df$.unit <- as.factor(df$.unit)
  
  if (any(is.na(df$.response))) {
    rlang::abort("The response column contains NAs. Please impute or remove missing values.")
  }
  
  if (!treated_unit %in% levels(df$.unit)) {
    rlang::abort(sprintf("Treated unit '%s' not found in the unit column.", treated_unit))
  }
  
  bucket_counts <- df |> 
    dplyr::group_by(.bucket, .unit) |> 
    dplyr::tally()
  
  expected_rows <- nlevels(df$.bucket) * nlevels(df$.unit)
  if (nrow(bucket_counts) != expected_rows) {
    rlang::abort("Unbalanced panel: Not all units are present in every time bucket.")
  }
  
  # --- 4. Replace dead code with a minimum observations warning ---
  min_n <- min(bucket_counts$n)
  if (min_n < 5) {
    cli::cli_warn("The smallest bucket has only {min_n} observations. Low sample sizes increase quantile volatility.")
  }

  loc_fn <- switch(location, mean = mean, median = stats::median)

  unit_loc <- df |>
    dplyr::summarise(loc = loc_fn(.response, na.rm = TRUE), .by = .unit) |>
    tibble::deframe()

  if (scale_method == "multiplicative" && any(unit_loc <= 0)) {
    rlang::abort("Multiplicative scaling requires strictly positive unit locations.")
  }

  df <- switch(scale_method,
    none = df,
    additive = dplyr::mutate(df, .response = .response - unit_loc[as.character(.unit)]),
    multiplicative = dplyr::mutate(df, .response = .response / unit_loc[as.character(.unit)])
  )

  df_treated <- dplyr::filter(df, .unit == treated_unit)
  df_donors <- dplyr::filter(df, .unit != treated_unit)
  
  treated_list <- df_treated |> 
    split(f = df_treated$.bucket, drop = TRUE) |> 
    purrr::map(\(x) x$.response)
  
  donor_list <- df_donors |> 
    split(f = df_donors$.bucket, drop = TRUE) |> 
    purrr::map(function(bucket_df) {
      bucket_df |> 
        split(f = bucket_df$.unit, drop = TRUE) |> 
        purrr::map(\(x) x$.response)
    })
  
  n_quantiles <- as.integer(n_quantiles)
  max_iter <- as.integer(max_iter)
  lambda <- as.numeric(lambda)
  tol <- as.numeric(tol)

  results <- z_dsc_rs(treated_list, donor_list, n_quantiles, lambda, max_iter, tol)
  
  # --- 10. Construct the S3 Object ---
  # NOTE: Name alignment depends entirely on factor-level ordering and the balanced-panel check. 
  # Because `split(..., drop = TRUE)` operates on the same factor levels across all buckets, 
  # the inner donor lists are strictly guaranteed to match the order of `levels(droplevels(df_donors$.unit))`.
  donor_names <- levels(droplevels(df_donors$.unit))
  names(results$weights) <- donor_names
  loc_target <- unit_loc[[treated_unit]]
  loc_donors <- unit_loc[donor_names]

  alpha <- if (scale_method == "additive") {
    loc_target - sum(results$weights * loc_donors)
  } else 0

  # Per-donor multiplicative factors: each donor scaled to the treated level.
  # Estimated on pre-treatment data only and held fixed thereafter.
  scale_factors <- if (scale_method == "multiplicative") {
    loc_target / loc_donors
  } else {
    stats::setNames(rep(1, length(donor_names)), donor_names)
  }
  
  out <- list(
    weights = results$weights,
    alpha = alpha,
    scale_factors = scale_factors,
    unit_loc = unit_loc,
    diagnostics = list(
      loss = results$loss,
      loss_penalized = results$loss_penalized,
      converged = results$converged,
      iterations = results$n_iterations,
      effective_rank = results$effective_rank,
      kappa = results$kappa,
      kappa_l2 = results$kappa_l2
    ),
    decomposition = list(
      right_singular_vectors = results$right_singular_vectors,
      svs = results$svs,
      svs_l2 = results$svs_l2
    ),
    params = list(
      treated_unit = treated_unit,
      donor_units = donor_names,
      n_buckets = nlevels(df$.bucket),
      scale_method = scale_method,
      location = location,
      n_quantiles = n_quantiles,
      probs = results$probs,
      lambda = lambda,
      max_iter = max_iter,
      tol = tol
    )
  )
  
  structure(out, class = "z_dsc")
}

#' Print method for Distributional Synthetic Controls
#' @export
print.z_dsc <- function(x, ...) {
  cli::cli_h1("Distributional Synthetic Control")
  
  if (x$diagnostics$converged) {
    cli::cli_alert_success("Optimisation converged in {x$diagnostics$iterations} iterations.")
  } else {
    cli::cli_alert_danger("Optimisation failed to converge (Max iter: {x$diagnostics$iterations}).")
  }
  
  cli::cli_bullets(c(
    "*" = "Treated Unit: {.val {x$params$treated_unit}}",
    "*" = "Buckets: {.val {x$params$n_buckets}}",
    "*" = "Donor Pool: {.val {length(x$params$donor_units)}} total units",
    "*" = "Ridge Penalty (lambda): {.val {x$params$lambda}}",
    "*" = "Scaling Method: {.val {x$params$scale_method}}",
    "*" = "Location Measure: {.val {x$params$location}}"
  ))
  
  cli::cli_h2("Top Contributing Donors")

  if (x$params$scale_method == "additive") {
    cli::cli_alert_info("Offset Constant (alpha): {.val {sprintf('%.3f', x$alpha)}}")
    cat("\n")
  }
  
  w_sorted <- sort(x$weights, decreasing = TRUE)
  top_n <- min(5, length(w_sorted))
  top_weights <- w_sorted[1:top_n]
  
  weight_strings <- sprintf("%.3f", top_weights)
  names(weight_strings) <- names(top_weights)
  
  cli::cli_dl(weight_strings)
  
  if (length(w_sorted) > 5) {
    remaining_weight <- sum(w_sorted[(top_n + 1):length(w_sorted)])
    cli::cli_text(cli::col_grey(
      "... and {length(w_sorted) - 5} other donors sharing {.val {sprintf('%.3f', remaining_weight)}} weight."
    ))
  }
  
  invisible(x)
}

#' Summary method for Distributional Synthetic Controls
#' @export
summary.z_dsc <- function(object, ...) {
  cli::cli_h1("DSC Diagnostic Summary")
  
  cli::cli_h2("Loss Metrics (2-Wasserstein)")
  cli::cli_bullets(c(
    "*" = "Unpenalised Loss: {.val {sprintf('%.5f', object$diagnostics$loss)}}",
    "*" = "Penalised Loss:   {.val {sprintf('%.5f', object$diagnostics$loss_penalized)}}"
  ))
  
  cli::cli_h2("Donor Collinearity Diagnostics")
  
  j <- length(object$params$donor_units)
  rank_pct <- (object$diagnostics$effective_rank / j) * 100
  
  cli::cli_bullets(c(
    "*" = "Effective Rank: {.val {object$diagnostics$effective_rank}} out of {j} ({sprintf('%.1f', rank_pct)}%)"
  ))
  
  if (object$diagnostics$effective_rank < j) {
    cli::cli_alert_warning("Effective rank is strictly less than J. The raw donor pool is perfectly collinear.")
  }
  
  cli::cli_bullets(c(
    "*" = "Raw Condition Number (Kappa): {.val {sprintf('%.2f', object$diagnostics$kappa)}}",
    "*" = "Regularised Kappa (L2):       {.val {sprintf('%.2f', object$diagnostics$kappa_l2)}}"
  ))
  
  if (object$diagnostics$kappa_l2 < object$diagnostics$kappa) {
    cli::cli_alert_success("Ridge penalty successfully improved matrix conditioning.")
  }
  
  invisible(object)
}

#' @export
tidy.z_dsc <- function(x, ...) {
  df_tidy <- tibble::tibble(
    donor = names(x$weights),
    weight = unname(x$weights)
  ) |> 
    dplyr::arrange(dplyr::desc(weight))

  if (x$params$scale_method == "multiplicative") {
    # Dynamically map the scale_factor to the arranged donor column
    df_tidy <- df_tidy |> 
      dplyr::mutate(scale_factor = unname(x$scale_factors[donor]))
  } else if (x$params$scale_method == "additive") {
    # Injecting the alpha intercept at the top of the tidy dataframe
    df_tidy <- dplyr::bind_rows(
      tibble::tibble(donor = "(Intercept)", weight = x$alpha),
      df_tidy
    )
  }
  
  df_tidy
}

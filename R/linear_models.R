#' Fit a linear model via Rust backends
#'
#' Fits an ordinary least squares (OLS) regression using compiled Rust linear
#' algebra engines. Designed as a pedagogical alternative to `stats::lm()`.
#'
#' @param formula An object ob class "formula" (or one that can be coerced to that class).
#' @param data A data frame, tibble, or environment containing the variables in the models.
#' @param engine A character string specifying the computational backend.
#'   Currently supports "naive" (Cholesky factorisation of normal equation).
#'
#' @return A list of class `z_lm` containing coefficients, standard errors,
#'   fitted values, residuals, degrees of freedom, and the residual standard error.
#' @export
#'
#' @examples
#' # Fit a basic model
#' fit <- z_lm(mpg ~ wt + cyl, data = mtcars)
z_lm <- function(formula, data, engine = c("naive", "qr", "svd")) {
  # Matching the engine argument to the provided options
  engine <- rlang::arg_match(engine)

  # --- 1. Evaluating the formula and handle NAs
  # na.fail ensures the function aborts immediately if any missing data is present
  mf <- stats::model.frame(formula, data, na.action = stats::na.fail)

  # --- 2. Extracting components
  # model.matrix automatically adds the intercept column of 1s
  x_mat <- stats::model.matrix(formula, mf)
  y_col <- stats::model.response(mf)

  if (!is.numeric(y_col)) {
    rlang::abort("The response variable must be strictly numeric.")
  }

  # --- 3. Dispatching to specified Rust enginer
  res <- switch(
    engine,
    "naive" = z_lm_naive(x = x_mat, y = as.numeric(y_col)),
    "qr" = rlang::abort("QR decomposition engine not yet implemented."),
    "svd" = rlang::abort("SVD engine not yet implemented.")
  )

  # --- 4. Formatting the output
  # Applying the column names from the design matrix to the output vectors
  # Applying custom class for future S3 methods like print() and summary()
  names(res$coefficients) <- colnames(x_mat)
  names(res$std_errors) <- colnames(x_mat)
  class(res) <- "z_lm"

  res
}

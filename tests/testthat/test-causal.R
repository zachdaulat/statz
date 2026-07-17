library(testthat)
library(statz)
library(quadprog)

# Helper to generate simulated bucket data
# Returns a list containing 'treated' (list of numeric) and 'donors' (list of list of numeric)
generate_mock_buckets <- function(n_buckets = 5, j_donors = 3, n_obs = 50, 
                                  treated_mean = 5, donor_means = c(4, 5, 6), sd = 1) {
  
  treated <- vector("list", n_buckets)
  donors <- vector("list", n_buckets)
  
  for (b in 1:n_buckets) {
    # Treated data for bucket b
    treated[[b]] <- rnorm(n_obs, mean = treated_mean, sd = sd)
    
    # Donor data for bucket b
    donor_b <- vector("list", j_donors)
    for (j in 1:j_donors) {
      donor_b[[j]] <- rnorm(n_obs, mean = donor_means[j], sd = sd)
    }
    donors[[b]] <- donor_b
  }
  
  list(treated = treated, donors = donors)
}

test_that("Independent solver reference test matches quadprog", {

  testthat::skip_if_not_installed("quadprog")
  
  # 1. Setup
  j_donors <- 3
  n_buckets <- 2
  q_quantiles <- 100
  lambda <- 0.1
  
  data <- generate_mock_buckets(n_buckets, j_donors, n_obs = 50)
  
  # 2. Run the Rust implementation
  rust_res <- z_dsc_rs(
    treated = data$treated, 
    donors = data$donors, 
    n_quantiles = q_quantiles, 
    lambda = lambda, 
    max_iter = 10000, 
    tol = 1e-8
  )
  
  # 3. Build the Gram matrix and vector completely in R (The Verification)
  probs <- (seq_len(q_quantiles) - 0.5) / q_quantiles # Midpoint convention
  
  G_raw <- matrix(0, nrow = j_donors, ncol = j_donors)
  c_raw <- numeric(j_donors)
  
  for (b in 1:n_buckets) {
    # stats::quantile type = 7 is standard interpolation
    a_t <- stats::quantile(data$treated[[b]], probs = probs, type = 7, names = FALSE)
    
    D_t <- sapply(1:j_donors, function(j) {
      stats::quantile(data$donors[[b]][[j]], probs = probs, type = 7, names = FALSE)
    })
    
    G_raw <- G_raw + crossprod(D_t) # D_t^T * D_t
    c_raw <- c_raw + crossprod(D_t, a_t)
  }
  
  # Apply normalization and L2 Ridge penalty
  S <- 2 / (n_buckets * q_quantiles)
  G_code <- (S * G_raw) + diag(2 * lambda, j_donors)
  c_code <- S * c_raw
  
  # 4. Solve via quadprog
  # Amat: First col is 1s (sum = 1). Next cols are Identity (w >= 0)
  Amat <- cbind(rep(1, j_donors), diag(j_donors))
  bvec <- c(1, rep(0, j_donors))
  
  qp_res <- quadprog::solve.QP(
    Dmat = G_code, 
    dvec = c_code, 
    Amat = Amat, 
    bvec = bvec, 
    meq = 1 # The first constraint in Amat is an equality constraint
  )
  
  # 5. Assert Equality
  expect_equal(rust_res$weights, qp_res$solution, tolerance = 1e-4)
})

test_that("Ground truth recovery and interpolation work correctly", {
  
  # Scenario A: Exact match
  # Donor 1 is an exact copy of the treated unit.
  data_exact <- generate_mock_buckets(j_donors = 3)
  data_exact$donors <- lapply(seq_along(data_exact$treated), function(b) {
    list(data_exact$treated[[b]], rnorm(50, 100, 1), rnorm(50, 200, 1))
  })
  
  res_exact <- z_dsc_rs(data_exact$treated, data_exact$donors, 100, lambda = 0.001, 1000, 1e-8)
  expect_equal(res_exact$weights, c(1, 0, 0), tolerance = 1e-4)
  
  # Scenario B: Interpolation (N(5,1) treated, N(4,1) and N(6,1) donors)
  data_interp <- generate_mock_buckets(
    treated_mean = 5, 
    donor_means = c(4, 6), 
    j_donors = 2, 
    n_obs = 1000 # High N to reduce empirical sampling noise
  )
  
  res_interp <- z_dsc_rs(data_interp$treated, data_interp$donors, 100, lambda = 0.001, 10000, 1e-8)
  expect_equal(res_interp$weights, c(0.5, 0.5), tolerance = 0.05)
})

test_that("Regularisation forces uniform weights on collinear donors", {
  
  # Two identical donors
  treated <- list(rnorm(50, 5, 1))
  identical_donor <- rnorm(50, 4, 1)
  donors <- list(list(identical_donor, identical_donor))
  
  # Low lambda
  res_low <- z_dsc_rs(treated, donors, 100, lambda = 0.0001, 1000, 1e-8)
  # High lambda
  res_high <- z_dsc_rs(treated, donors, 100, lambda = 100, 1000, 1e-8)
  
  # High lambda should force near-perfect uniformity (0.5, 0.5)
  expect_equal(res_high$weights, c(0.5, 0.5), tolerance = 1e-5)
  
  # The regularized condition number must drop
  expect_true(res_high$kappa_l2 < res_low$kappa_l2)
})

test_that("Simplex invariants hold", {
  data <- generate_mock_buckets()
  res <- z_dsc_rs(data$treated, data$donors, 100, 0.1, 1000, 1e-8)
  
  expect_equal(sum(res$weights), 1.0, tolerance = 1e-8)
  expect_true(all(res$weights >= -1e-12)) # Accounting for float imprecision
  expect_equal(length(res$weights), 3)
})

test_that("Convergence controls limit execution", {
  data <- generate_mock_buckets()
  
  # max_iter = 1 should almost guarantee non-convergence on random data
  res <- z_dsc_rs(data$treated, data$donors, 100, 0.1, max_iter = 1, 1e-8)
  expect_false(res$converged)
  expect_equal(res$n_iterations, 1)
})

test_that("z_dsc() wrapper correctly routes valid data and returns S3 object", {
  valid_df <- tibble::tibble(
    street = rep(c("King", "Queen", "Dundas"), times = 10),
    period = rep(as.POSIXct(c("2026-07-08 08:00:00", "2026-07-08 09:00:00")), each = 15),
    delay = runif(30, 1, 10)
  )
  
  # Notice: bucket is now explicitly placed before time
  res <- z_dsc(
    data = valid_df,
    response = delay,
    unit_id = street,
    treated_unit = "King",
    bucket = "1 hour",
    time = period,
    n_quantiles = 10, 
    lambda = 0.1
  )
  
  expect_s3_class(res, "z_dsc")
  expect_equal(length(res$weights), 2) 
  expect_equal(res$params$treated_unit, "King")
  expect_equal(res$params$n_buckets, 2)
})

test_that("z_dsc() wrapper aggressively guards against degenerate data", {
  base_df <- tibble::tibble(
    street = rep(c("King", "Queen"), times = 2),
    time_col = c(1, 1, 2, 2),
    delay = c(5, 4, 6, 5)
  )
  
  df_na <- base_df
  df_na$delay[1] <- NA
  expect_error(
    z_dsc(df_na, delay, street, "King", time_col),
    regexp = "contains NAs"
  )
  
  expect_error(
    z_dsc(base_df, delay, street, "Richmond", time_col),
    regexp = "not found in the unit column"
  )
  
  df_unbalanced <- base_df[-4, ] 
  expect_error(
    z_dsc(df_unbalanced, delay, street, "King", time_col),
    regexp = "Unbalanced panel"
  )
})

test_that("S3 methods execute correctly", {
  valid_df <- tibble::tibble(
    street = rep(c("King", "Queen", "Dundas"), times = 10),
    period = rep(1:2, each = 15),
    delay = runif(30, 1, 10)
  )
  
  res <- z_dsc(valid_df, delay, street, "King", period)
  
  expect_error(print(res), NA)
  expect_error(summary(res), NA)
  
  res_tidy <- broom::tidy(res)
  expect_s3_class(res_tidy, "tbl_df")
  expect_equal(nrow(res_tidy), 2)
  expect_named(res_tidy, c("donor", "weight"))
})

test_that("z_dsc() safely processes single-observation buckets", {
  valid_df_single <- tibble::tibble(
    street = rep(c("King", "Queen", "Dundas"), times = 2),
    period = rep(1:2, each = 3),
    delay = runif(6, 1, 10)
  )
  
  expect_warning(
    res <- z_dsc(valid_df_single, delay, street, "King", period),
    regexp = "smallest bucket has only 1 observations"
  )
  
  expect_s3_class(res, "z_dsc")
  expect_true(res$diagnostics$converged)
})

test_that("z_dsc() center argument correctly extracts alpha shift", {
  valid_df <- tibble::tibble(
    street = rep(c("King", "Queen", "Dundas"), times = 10),
    period = rep(1:2, each = 15),
    delay = ifelse(street == "King", 
                   runif(30, 15, 20), 
                   runif(30, 5, 10))
  )
  
  res_centered <- z_dsc(valid_df, delay, street, "King", period, center = TRUE)
  res_tidy <- broom::tidy(res_centered)
  
  expect_true(res_centered$params$center)
  expect_true(res_centered$alpha > 5) 
  expect_equal(res_tidy$donor[1], "(Intercept)")
})

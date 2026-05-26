library(testthat)
library(statz) 
library(palmerpenguins)

test_that("z_lm engines correctly solve simple analytic case", {
  # Known analytic dataset: y = 2x + 1
  df <- data.frame(
    x = c(1, 2, 3, 4, 5),
    y = c(3, 5, 7, 9, 11)
  )

  engines <- c("cholesky", "qr")
  
  for (eng in engines) {
    fit <- z_lm(y ~ x, data = df, engine = eng)
    
    # Check coefficients
    expect_equal(unname(fit$coefficients["(Intercept)"]), 1, tolerance = 1e-10, info = eng)
    expect_equal(unname(fit$coefficients["x"]), 2, tolerance = 1e-10, info = eng)
    
    # Perfect fit means residual standard error (sigma) should be functionally zero
    expect_true(fit$sigma < 1e-10, info = eng)
  }
})

test_that("z_lm strictly fails on NA values", {
  df <- data.frame(
    x = c(1, 2, NA, 4),
    y = c(2, 4, 6, 8)
  )

  # Should throw an error due to na.action = na.fail for all engines
  expect_error(z_lm(y ~ x, data = df, engine = "cholesky"))
  expect_error(z_lm(y ~ x, data = df, engine = "qr"))
})

test_that("z_lm engines match stats::lm on palmerpenguins", {
  # Drop NAs explicitly for the comparison test
  penguins_clean <- na.omit(penguins)

  # Fit base R model
  # We use a formula with a continuous and a categorical variable
  f <- body_mass_g ~ flipper_length_mm + species
  fit_r <- stats::lm(f, data = penguins_clean)
  r_sigma <- summary(fit_r)$sigma

  engines <- c("cholesky", "qr")
  
  for (eng in engines) {
    fit_z <- z_lm(f, data = penguins_clean, engine = eng)
    
    # 1. Compare Coefficients 
    expect_equal(fit_z$coefficients, coef(fit_r), ignore_attr = TRUE, tolerance = 1e-8, info = eng)
    
    # 2. Compare Fitted Values
    expect_equal(fit_z$fitted_values, fitted(fit_r), ignore_attr = TRUE, tolerance = 1e-8, info = eng)
    
    # 3. Compare Residuals
    expect_equal(fit_z$residuals, residuals(fit_r), ignore_attr = TRUE, tolerance = 1e-8, info = eng)
  
    # 4. Compare Residual Standard Error (sigma)
    expect_equal(fit_z$sigma, r_sigma, tolerance = 1e-8, info = eng)
  
    # 5. Compare Degrees of Freedom
    expect_equal(fit_z$df_residual, fit_r$df.residual, info = eng)
  }
})

test_that("z_eigen matches base::eigen for symmetric matrices", {
  # Create a symmetric matrix
  set.seed(2001)
  mat <- matrix(rnorm(25), nrow = 5, ncol = 5)
  sym_mat <- crossprod(mat) 
  
  # Compute decompositions
  r_eigen <- base::eigen(sym_mat, symmetric = TRUE)
  rust_eigen <- z_eigen(sym_mat)
  
  # Reverse the faer outputs to match LAPACK's descending sort order
  rust_values_desc <- rev(rust_eigen$values)
  rust_vectors_desc <- rust_eigen$vectors[, ncol(rust_eigen$vectors):1]
  
  # 1. Test Eigenvalues
  expect_equal(rust_values_desc, r_eigen$values, tolerance = 1e-8)
  
  # 2. Test Eigenvectors 
  expect_equal(
    abs(rust_vectors_desc), 
    abs(r_eigen$vectors), 
    tolerance = 1e-8
  )
})

test_that("z_svd matches base::svd", {
  # Create a standard matrix
  set.seed(2001)
  mat <- matrix(rnorm(50), nrow = 10, ncol = 5)
  
  # Compute SVD
  r_svd_full <- base::svd(mat, nu = nrow(mat))
  rust_svd <- z_svd(mat)
  
  # 1. Test Singular Values (d)
  expect_equal(rust_svd$d, r_svd_full$d, tolerance = 1e-8)
  
  # 2. Test U matrix (Subset Rust's Full SVD to match R's Thin SVD)
  expect_equal(
    abs(rust_svd$u), 
    abs(r_svd_full$u), 
    tolerance = 1e-8
  )
  
  # 3. Test V matrix
  expect_equal(
    abs(rust_svd$v), 
    abs(r_svd_full$v), 
    tolerance = 1e-8
  )
})

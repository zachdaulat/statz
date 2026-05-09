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
library(testthat)
library(statz) # Assuming your package is named statz
library(palmerpenguins)

test_that("z_lm naive engine correctly solves simple analytic case", {
  # Known analytic dataset: y = 2x + 1
  df <- data.frame(
    x = c(1, 2, 3, 4, 5),
    y = c(3, 5, 7, 9, 11)
  )

  fit <- z_lm(y ~ x, data = df, engine = "naive")

  # Check coefficients
  expect_equal(unname(fit$coefficients["(Intercept)"]), 1, tolerance = 1e-10)
  expect_equal(unname(fit$coefficients["x"]), 2, tolerance = 1e-10)

  # Perfect fit means residual standard error (sigma) should be functionally zero
  expect_true(fit$sigma < 1e-10)
})

test_that("z_lm strictly fails on NA values", {
  df <- data.frame(
    x = c(1, 2, NA, 4),
    y = c(2, 4, 6, 8)
  )

  # Should throw an error due to na.action = na.fail
  expect_error(z_lm(y ~ x, data = df, engine = "naive"))
})

test_that("z_lm naive engine matches stats::lm on palmerpenguins", {
  # Drop NAs explicitly for the comparison test
  penguins_clean <- na.omit(penguins)

  # Fit both models
  # We use a formula with a continuous and a categorical variable
  f <- body_mass_g ~ flipper_length_mm + species

  fit_z <- z_lm(f, data = penguins_clean, engine = "naive")
  fit_r <- stats::lm(f, data = penguins_clean)

  # 1. Compare Coefficients (ignore attributes like names for strict numeric check)
  expect_equal(fit_z$coefficients, coef(fit_r), ignore_attr = TRUE, tolerance = 1e-8)
  
  # 2. Compare Fitted Values
  expect_equal(fit_z$fitted_values, fitted(fit_r), ignore_attr = TRUE, tolerance = 1e-8)
  
  # 3. Compare Residuals
  expect_equal(fit_z$residuals, residuals(fit_r), ignore_attr = TRUE, tolerance = 1e-8)

  # 4. Compare Residual Standard Error (sigma)
  # R's summary(lm) computes sigma
  r_sigma <- summary(fit_r)$sigma
  expect_equal(fit_z$sigma, r_sigma, tolerance = 1e-8)

  # 5. Compare Degrees of Freedom
  expect_equal(fit_z$df_residual, fit_r$df.residual)
})

test_that("z_pnorm matches R's pnorm for standard normal", {
  test_values <- c(-3, -1.96, -1, 0, 1, 1.96, 3)
  for (z in test_values) {
    expect_equal(z_pnorm(z), pnorm(z), tolerance = 1e-4)
  }
})

test_that("z_pnorm matches R's pnorm (absolute error < 2e-7)", {
  test_values <- c(-3, -1.96, -1, 0, 1, 1.96, 3)
  for (z in test_values) {
    expect_true(
      abs(z_pnorm(z) - pnorm(z)) < 1e-7,
      label = paste("absolute error at z =", z)
    )
  }
})

test_that("z_pnorm handles non-standard normal", {
  expect_equal(z_pnorm(100, mean = 100, sd = 15), 0.5, tolerance = 1e-6)
})

test_that("z_pnorm rejects invalid sd", {
  expect_error(z_pnorm(0, sd = -1))
  expect_error(z_pnorm(0, sd = 0))
})

test_that("z_dnorm matches R's dnorm", {
  expect_equal(z_dnorm(0), dnorm(0), tolerance = 1e-10)
  expect_equal(z_dnorm(1, mean = 1, sd = 2), dnorm(1, 1, 2), tolerance = 1e-10)
})

test_that("z_dpois matches R's dpois", {
  test_cases <- list(
    list(x = 0, lambda = 3),
    list(x = 2, lambda = 3),
    list(x = 5, lambda = 5),
    list(x = 10, lambda = 7)
  )
  for (tc in test_cases) {
    expect_equal(
      z_dpois(tc$x, tc$lambda),
      dpois(tc$x, tc$lambda),
      tolerance = 1e-10
    )
  }
})

test_that("z_ppois matches R's ppois", {
  test_cases <- list(
    list(x = 0, lambda = 3),
    list(x = 5, lambda = 5),
    list(x = 10, lambda = 2),
    list(x = 20, lambda = 10)
  )
  for (tc in test_cases) {
    expect_equal(
      z_ppois(tc$x, tc$lambda),
      ppois(tc$x, tc$lambda),
      tolerance = 1e-10
    )
  }
})

test_that("z_dpois rejects invalid inputs", {
  expect_error(z_dpois(-1, 3))
  expect_error(z_dpois(2.5, 3))
  expect_error(z_dpois(2, -1))
})

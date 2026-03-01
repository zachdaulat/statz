# Normal Distribution tests
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

# Posson Distribution tests
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

test_that("z_dpois rejects invalid inputs", {
  expect_error(z_dpois(-1, 3))
  expect_error(z_dpois(2.5, 3))
  expect_error(z_dpois(2, -1))
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

# Gamma Distribution tests
# --- z_lgamma tests ---

test_that("z_lgamma matches R lgamma() for small integers", {
  for (n in 1:15) {
    expect_equal(
      z_lgamma(n),
      lgamma(n),
      tolerance = 1e-13,
      label = paste0("lgamma(", n, ")")
    )
  }
})

test_that("z_lgamma matches R lgamma() for half-integers", {
  half_ints <- seq(0.5, 10.5, by = 1)
  for (z in half_ints) {
    expect_equal(
      z_lgamma(z),
      lgamma(z),
      tolerance = 1e-13,
      label = paste0("lgamma(", z, ")")
    )
  }
})

test_that("z_lgamma matches R lgamma() across a range of values", {
  test_vals <- c(
    0.01,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    5.0,
    7.5,
    10.0,
    25.0,
    50.0,
    100.0,
    500.0,
    1000.0
  )
  for (z in test_vals) {
    expect_equal(
      z_lgamma(z),
      lgamma(z),
      tolerance = 1e-11,
      label = paste0("lgamma(", z, ")")
    )
  }
})

test_that("z_lgamma handles the reflection formula correctly", {
  # For z < 0.5, lgamma(z) + lgamma(1-z) = ln(pi) - ln|sin(pi*z)|
  test_vals <- c(0.1, 0.2, 0.3, 0.4, 0.49)
  for (z in test_vals) {
    lhs <- z_lgamma(z) + z_lgamma(1 - z)
    rhs <- log(pi) - log(abs(sin(pi * z)))
    expect_equal(
      lhs,
      rhs,
      tolerance = 1e-12,
      label = paste0("reflection at z = ", z)
    )
  }
})

test_that("z_lgamma is negative on (1, 2) and non-negative outside", {
  expect_lt(z_lgamma(1.5), 0)
  expect_lt(z_lgamma(1.9), 0)
  expect_gte(z_lgamma(0.5), 0)
  expect_gte(z_lgamma(3.0), 0)
})

# --- z_dgamma tests ---

test_that("z_dgamma matches R dgamma() for basic cases", {
  # Standard cases
  expect_equal(
    z_dgamma(2, shape = 3, rate = 1),
    dgamma(2, shape = 3, rate = 1),
    tolerance = 1e-10
  )
  expect_equal(
    z_dgamma(1, shape = 2, rate = 2),
    dgamma(1, shape = 2, rate = 2),
    tolerance = 1e-10
  )
  expect_equal(
    z_dgamma(0.5, shape = 0.5, rate = 1),
    dgamma(0.5, shape = 0.5, rate = 1),
    tolerance = 1e-10
  )
})

test_that("z_dgamma with scale parameterisation matches rate", {
  # dgamma(x, shape, scale = s) should equal dgamma(x, shape, rate = 1/s)
  expect_equal(
    z_dgamma(2, shape = 3, scale = 2),
    z_dgamma(2, shape = 3, rate = 0.5),
    tolerance = 1e-14
  )
})

test_that("z_dgamma matches exponential distribution when shape = 1", {
  # Gamma(1, rate) = Exponential(rate)
  rate <- 2
  x <- 1
  expected <- rate * exp(-rate * x)
  expect_equal(z_dgamma(x, shape = 1, rate = rate), expected, tolerance = 1e-10)
})

test_that("z_dgamma log mode is consistent", {
  val <- z_dgamma(2, shape = 3, rate = 1)
  log_val <- z_dgamma(2, shape = 3, rate = 1, log = TRUE)
  expect_equal(log_val, log(val), tolerance = 1e-14)
})

test_that("z_dgamma matches R dgamma() across a grid", {
  shapes <- c(0.5, 1, 2, 5, 10)
  rates <- c(0.5, 1, 2, 5)
  xs <- c(0.1, 0.5, 1, 2, 5, 10)

  for (a in shapes) {
    for (b in rates) {
      for (x in xs) {
        expect_equal(
          z_dgamma(x, shape = a, rate = b),
          dgamma(x, shape = a, rate = b),
          tolerance = 1e-9,
          label = paste0("dgamma(", x, ", ", a, ", ", b, ")")
        )
      }
    }
  }
})

test_that("z_dgamma validates inputs correctly", {
  expect_error(z_dgamma(-1, shape = 1, rate = 1))
  expect_error(z_dgamma(1, shape = -1, rate = 1))
  expect_error(z_dgamma(1, shape = 1, rate = -1))
  expect_error(z_dgamma(1, shape = 1, rate = 1, scale = 1))
})

test_that("z_pgamma matches R pgamma() for basic cases", {
  expect_equal(
    z_pgamma(2, shape = 3, rate = 1),
    pgamma(2, shape = 3, rate = 1),
    tolerance = 1e-10
  )
  expect_equal(
    z_pgamma(1, shape = 2, scale = 0.5),
    pgamma(1, shape = 2, scale = 0.5),
    tolerance = 1e-10
  )
})

test_that("z_pgamma validates inputs correctly", {
  expect_error(z_pgamma("a", shape = 1, rate = 1))
  expect_error(z_pgamma(1, shape = -1, rate = 1))
  expect_error(z_pgamma(1, shape = 1, rate = 1, scale = 1))
})

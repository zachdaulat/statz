# ==============================================================================
# Normal Distribution
# ==============================================================================

test_that("dnorm and pnorm match stats:: on vectorised inputs and defaults", {
  x <- c(-3, -1.96, -1, 0, 1, 1.96, 3)

  # Default arguments (mean = 0, sd = 1)
  expect_equal(dnorm(x), stats::dnorm(x), tolerance = 1e-15)
  expect_equal(pnorm(x), stats::pnorm(x), tolerance = 1e-14)

  # Non-standard parameters
  expect_equal(
    dnorm(x, mean = 2, sd = 1.5),
    stats::dnorm(x, mean = 2, sd = 1.5),
    tolerance = 1e-15
  )
  expect_equal(
    pnorm(x, mean = 2, sd = 1.5),
    stats::pnorm(x, mean = 2, sd = 1.5),
    tolerance = 1e-14
  )
  expect_equal(
    pnorm(x, mean = 2, sd = 1.5, lower_tail = FALSE),
    stats::pnorm(x, mean = 2, sd = 1.5, lower.tail = FALSE),
    tolerance = 1e-14
  )
})

test_that("dnorm and pnorm handle NA and boundary conditions gracefully", {
  expect_true(is.na(dnorm(NA_real_)))
  expect_true(is.na(pnorm(NA_real_)))
  expect_true(is.nan(dnorm(1, sd = -1)))
  expect_true(is.nan(pnorm(1, sd = -1)))

  # Degenerate zero variance point mass
  expect_equal(dnorm(0, mean = 0, sd = 0), Inf)
  expect_equal(dnorm(1, mean = 0, sd = 0), 0)
  expect_equal(pnorm(0, mean = 0, sd = 0), 1)
  expect_equal(pnorm(-1, mean = 0, sd = 0), 0)
})

# ==============================================================================
# Poisson Distribution
# ==============================================================================

test_that("dpois and ppois match stats:: on vectorised counts and defaults", {
  x <- c(0, 1, 2, 5, 10, 20)
  lambda <- 4.5

  expect_equal(
    dpois(x, lambda = lambda),
    stats::dpois(x, lambda = lambda),
    tolerance = 1e-14
  )
  expect_equal(
    ppois(x, lambda = lambda),
    stats::ppois(x, lambda = lambda),
    tolerance = 1e-14
  )
  expect_equal(
    ppois(x, lambda = lambda, lower_tail = FALSE),
    stats::ppois(x, lambda = lambda, lower.tail = FALSE),
    tolerance = 1e-14
  )
})

test_that("dpois and ppois handle domain boundaries and non-integers without crashing", {
  expect_equal(dpois(2.5, lambda = 3), 0.0)
  expect_equal(dpois(-1, lambda = 3), 0.0)
  expect_true(is.nan(dpois(2, lambda = -1)))

  # ppois evaluates floor(x) on continuous values
  expect_equal(
    ppois(2.5, lambda = 3),
    stats::ppois(2.5, lambda = 3),
    tolerance = 1e-14
  )
  expect_true(is.na(dpois(NA_real_, lambda = 3)))
  expect_true(is.na(ppois(NA_real_, lambda = 3)))
})

# ==============================================================================
# Gamma Distribution & lgamma
# ==============================================================================

test_that("lgamma matches base::lgamma on vectors", {
  z <- c(0.01, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 100.0)
  expect_equal(lgamma(z), base::lgamma(z), tolerance = 1e-14)
})

test_that("dgamma and pgamma match stats:: across rate and scale parameterisations", {
  x <- c(0.1, 0.5, 1, 2, 5)
  shape <- 2.5

  # Default rate = 1
  expect_equal(
    dgamma(x, shape = shape),
    stats::dgamma(x, shape = shape),
    tolerance = 1e-14
  )
  expect_equal(
    pgamma(x, shape = shape),
    stats::pgamma(x, shape = shape),
    tolerance = 1e-14
  )

  # Rate vs Scale equivalence
  expect_equal(
    dgamma(x, shape = shape, scale = 2),
    stats::dgamma(x, shape = shape, scale = 2),
    tolerance = 1e-14
  )
  expect_equal(
    pgamma(x, shape = shape, scale = 2),
    stats::pgamma(x, shape = shape, scale = 2),
    tolerance = 1e-14
  )
  expect_equal(
    dgamma(x, shape = shape, rate = 0.5),
    dgamma(x, shape = shape, scale = 2),
    tolerance = 1e-15
  )
})

test_that("dgamma and pgamma R wrappers enforce mutual exclusivity of rate and scale", {
  expect_error(
    dgamma(1, shape = 1, rate = 1, scale = 1),
    "Only one of `rate` or `scale`"
  )
  expect_error(
    pgamma(1, shape = 1, rate = 1, scale = 1),
    "Only one of `rate` or `scale`"
  )
})

# ==============================================================================
# Inverse Gaussian Distribution
# ==============================================================================

test_that("dinvgauss and pinvgauss match statmod on vectorised inputs", {
  skip_if_not_installed("statmod")

  y <- c(0.1, 0.5, 1.0, 2.5, 5.0)
  mu <- 2.0
  lambda <- 1.5

  expect_equal(
    dinvgauss(y, mu = mu, lambda = lambda),
    statmod::dinvgauss(y, mean = mu, shape = lambda),
    tolerance = 1e-14
  )
  expect_equal(
    pinvgauss(y, mu = mu, lambda = lambda),
    statmod::pinvgauss(y, mean = mu, shape = lambda),
    tolerance = 1e-14
  )
  expect_equal(
    pinvgauss(y, mu = mu, lambda = lambda, lower_tail = FALSE),
    statmod::pinvgauss(y, mean = mu, shape = lambda, lower.tail = FALSE),
    tolerance = 1e-14
  )
})

test_that("dinvgauss and pinvgauss return fast paths and NaNs on boundary values", {
  expect_equal(dinvgauss(0, mu = 1, lambda = 1), 0.0)
  expect_equal(dinvgauss(-1, mu = 1, lambda = 1), 0.0)
  expect_true(is.nan(dinvgauss(1, mu = -1, lambda = 1)))
  expect_true(is.nan(pinvgauss(1, mu = 1, lambda = -1)))
})

# ==============================================================================
# Tweedie Distribution
# ==============================================================================

test_that("dtweedie and ptweedie match tweedie package on vectorised inputs", {
  skip_if_not_installed("tweedie")

  y <- c(0.0, 0.5, 1.5, 3.0)
  mu <- 2.0
  phi <- 1.2
  power <- 1.5

  expect_equal(
    dtweedie(y, mu = mu, phi = phi, power = power),
    tweedie::dtweedie(y, mu = mu, phi = phi, power = power),
    tolerance = 1e-10
  )
  expect_equal(
    ptweedie(y, mu = mu, phi = phi, power = power),
    tweedie::ptweedie(y, mu = mu, phi = phi, power = power),
    tolerance = 1e-10
  )
})

test_that("dtweedie and ptweedie throw errors on invalid compound Poisson-Gamma powers", {
  expect_error(
    dtweedie(1, mu = 2, phi = 1, power = 1.0),
    "strictly between 1 and 2"
  )
  expect_error(
    dtweedie(1, mu = 2, phi = 1, power = 2.0),
    "strictly between 1 and 2"
  )
  expect_error(
    ptweedie(1, mu = 2, phi = 1, power = 0.5),
    "strictly between 1 and 2"
  )
  expect_error(dtweedie(1, mu = -1, phi = 1, power = 1.5), "mu must be >= 0")
})

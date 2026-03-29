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

# ------------- Tweedie Distribution Tests -----------------

test_that("z_dtweedie calculates exact zero mass correctly", {
  # For y = 0, it should perfectly match the Poisson point mass
  y <- 0
  mu <- 2.5
  phi <- 1.2
  p <- 1.5

  res_statz <- z_dtweedie(y, mu, phi, p, log = TRUE)
  res_tweedie <- tweedie::dtweedie(y, mu = mu, phi = phi, power = p)

  expect_equal(res_statz, log(res_tweedie), tolerance = 1e-10)
})

test_that("z_dtweedie handles mu = 0 fast path", {
  # Point mass at 0
  expect_equal(z_dtweedie(0, mu = 0, phi = 1, power = 1.5, log = FALSE), 1.0)
  expect_equal(z_dtweedie(5, mu = 0, phi = 1, power = 1.5, log = FALSE), 0.0)
})

test_that("z_dtweedie Dunn-Smyth series matches CRAN tweedie package", {
  # Test a spread of positive y values
  y_vals <- c(0.1, 1.0, 2.5, 10.0)
  mu <- 3.0
  phi <- 1.5
  p <- 1.6

  res_statz <- purrr::map_dbl(y_vals, \(y) {
    z_dtweedie(y, mu, phi, p, log = FALSE)
  })
  res_tweedie <- tweedie::dtweedie(y_vals, mu = mu, phi = phi, power = p)

  # Tolerance is set to 1e-10 to account for the algorithmic difference
  # between Dunn-Smyth series and Fourier inversion.
  expect_equal(res_statz, res_tweedie, tolerance = 1e-10)
})

test_that("z_dtweedie evaluates correctly near the Poisson boundary (p -> 1)", {
  y_vals <- c(0.1, 1.0, 5.0, 10.0)
  mu <- 2.0
  phi <- 1.0
  p <- 1.05

  # p = 1.05 is highly discrete, testing loop efficiency
  res_statz <- purrr::map_dbl(y_vals, \(y) {
    z_dtweedie(y, mu = mu, phi = phi, power = p, log = FALSE)
  })
  res_tweedie <- tweedie::dtweedie(y_vals, mu = mu, phi = phi, power = p)

  expect_equal(res_statz, res_tweedie, tolerance = 1e-14)
})

test_that("z_dtweedie evaluates correctly near the Gamma boundary (p -> 2)", {
  y_vals <- c(0.1, 1.0, 5.0, 10.0)
  mu <- 2.0
  phi <- 1.0
  p <- 1.95

  # p = 1.95 is highly continuous, testing extreme Gamma shapes
  res_statz <- purrr::map_dbl(y_vals, \(y) {
    z_dtweedie(y, mu = mu, phi = phi, power = p, log = FALSE)
  })
  res_tweedie <- tweedie::dtweedie(y_vals, mu = mu, phi = phi, power = p)

  expect_equal(res_statz, res_tweedie, tolerance = 1e-10)
})

test_that("z_dtweedie wrapper blocks degenerate power parameters", {
  expect_error(z_dtweedie(1, 2, 1, power = 1.0), "strictly between 1 and 2")
  expect_error(z_dtweedie(1, 2, 1, power = 2.0), "strictly between 1 and 2")
  expect_error(z_dtweedie(1, 2, 1, power = 0.5), "strictly between 1 and 2")
})

# --- Tweedie CDF (z_ptweedie) Tests ---

test_that("z_ptweedie enforces strict input validation and blocks NA", {
  # y constraints
  expect_error(z_ptweedie(-1, 2, 1, 1.5), ">= 0")
  expect_error(z_ptweedie(NA_real_, 2, 1, 1.5), "single numeric value")
  expect_error(z_ptweedie(c(1, 2), 2, 1, 1.5), "single numeric value") # Blocks vectors

  # Parameter constraints
  expect_error(z_ptweedie(1, -2, 1, 1.5), ">= 0")
  expect_error(z_ptweedie(1, 2, 0, 1.5), "positive numeric")
  expect_error(z_ptweedie(1, 2, 1, NA_real_), "strictly between 1 and 2")

  # Degenerate boundaries
  expect_error(z_ptweedie(1, 2, 1, 1.0), "strictly between 1 and 2")
  expect_error(z_ptweedie(1, 2, 1, 2.0), "strictly between 1 and 2")
})

test_that("z_ptweedie handles exact point mass fast paths", {
  # mu = 0: All mass is at exactly 0.
  # For any y >= 0, cumulative probability is 1.0 (lower) or 0.0 (upper)
  expect_equal(
    z_ptweedie(5.0, mu = 0, phi = 1, power = 1.5, lower.tail = TRUE),
    1.0
  )
  expect_equal(
    z_ptweedie(5.0, mu = 0, phi = 1, power = 1.5, lower.tail = FALSE),
    0.0
  )
  expect_equal(z_ptweedie(5.0, mu = 0, phi = 1, power = 1.5, log.p = TRUE), 0.0) # ln(1) = 0

  # y = 0, mu > 0: Should exactly match the Poisson probability of 0 events
  # lambda = mu^(2-p) / (phi * (2-p))
  mu <- 3.0
  phi <- 1.5
  p <- 1.6
  lambda <- (mu^(2 - p)) / (phi * (2 - p))
  expected_p0 <- exp(-lambda)

  expect_equal(z_ptweedie(0, mu, phi, p, lower.tail = TRUE), expected_p0)
  expect_equal(z_ptweedie(0, mu, phi, p, lower.tail = FALSE), 1.0 - expected_p0)
})

test_that("z_ptweedie upper and lower tails complement perfectly", {
  # Test the mathematical invariant: F(y) + S(y) = 1
  y <- 2.5
  mu <- 3.0
  phi <- 1.5
  p <- 1.6

  p_lower <- z_ptweedie(y, mu, phi, p, lower.tail = TRUE)
  p_upper <- z_ptweedie(y, mu, phi, p, lower.tail = FALSE)

  expect_equal(p_lower + p_upper, 1.0)

  # Check log transformations
  expect_equal(
    z_ptweedie(y, mu, phi, p, lower.tail = TRUE, log.p = TRUE),
    log(p_lower)
  )
})

test_that("z_ptweedie Dunn-Smyth series matches CRAN tweedie package", {
  # Standard parameterization
  y_vals <- c(0.1, 1.0, 2.5, 10.0)
  mu <- 3.0
  phi <- 1.5
  p <- 1.6

  # Use map_dbl because z_ptweedie currently strictly expects scalars
  res_statz <- purrr::map_dbl(y_vals, \(y) {
    z_ptweedie(y, mu, phi, p, lower.tail = TRUE)
  })
  res_tweedie <- tweedie::ptweedie(y_vals, mu = mu, phi = phi, power = p)

  # CDFs are generally more stable than PDFs, so 1e-6 is a very safe tolerance
  expect_equal(res_statz, res_tweedie, tolerance = 1e-6)
})

test_that("z_ptweedie evaluates correctly near the boundaries (p -> 1 and p -> 2)", {
  y_vals <- c(0.1, 1.0, 5.0)
  mu <- 2.0
  phi <- 1.0

  # Near Poisson boundary
  res_statz_pois <- purrr::map_dbl(y_vals, \(y) {
    z_ptweedie(y, mu, phi, power = 1.05)
  })
  res_tweedie_pois <- tweedie::ptweedie(
    y_vals,
    mu = mu,
    phi = phi,
    power = 1.05
  )
  expect_equal(res_statz_pois, res_tweedie_pois, tolerance = 1e-5)

  # Near Gamma boundary
  res_statz_gam <- purrr::map_dbl(y_vals, \(y) {
    z_ptweedie(y, mu, phi, power = 1.95)
  })
  res_tweedie_gam <- tweedie::ptweedie(y_vals, mu = mu, phi = phi, power = 1.95)
  expect_equal(res_statz_gam, res_tweedie_gam, tolerance = 1e-5)
})

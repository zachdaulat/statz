test_that("Neumaier summation outperforms base R on extreme magnitude differences", {
  # 1e22 exceeds both 64-bit and 80-bit float mantissa capacities
  x <- c(1e22, rep(1, 10000), -1e22)

  # Base R will lose the unit increments and evaluate to 0 on all architectures
  expect_equal(base::sum(x), 0)

  # statz Neumaier summation tracks truncated bits in the error accumulator `c`
  expect_equal(statz:::sum(x), 10000, tolerance = 1e-12)
  expect_equal(statz:::mean(x), 10000 / 10002, tolerance = 1e-14)
})

test_that("statz variance and standard deviation match stats::var and stats::sd", {
  set.seed(42)
  x <- rnorm(1000, mean = 50, sd = 15)

  # Checking parity to extremely tight tolerances
  expect_equal(statz:::var(x), stats::var(x), tolerance = 1e-14)
  expect_equal(statz:::sd(x), stats::sd(x), tolerance = 1e-14)
})

test_that("statz covariance and correlation match stats::cov and stats::cor", {
  set.seed(42)
  x <- rnorm(100)
  y <- x * 2 + rnorm(100)

  expect_equal(statz:::cov(x, y), stats::cov(x, y), tolerance = 1e-14)
  expect_equal(statz:::cor(x, y), stats::cor(x, y), tolerance = 1e-14)
})

test_that("Rust Option<f64> correctly maps to R NA_real_ via extendr", {
  # Replacing the old numeric(0) assertions[cite: 6] with FFI boundary tests
  expect_true(is.na(statz:::mean(numeric(0))))
  expect_true(is.na(statz:::var(c(5)))) # Length < 2 should yield NA
  expect_true(is.na(statz:::sd(c(5))))
  expect_true(is.na(statz:::cov(c(1, 2), c(1, 2, 3)))) # Mismatched lengths
  expect_true(is.na(statz:::cor(c(5, 5, 5), c(1, 2, 3)))) # Zero variance denominator
})

test_that("statz correctly propagates R NAs/NaNs to NA_real_ without crashing", {
  x_na <- c(1, 2, NA_real_, 4, 5)
  y <- c(1, 2, 3, 4, 5)

  expect_true(is.na(statz:::var(x_na)))
  expect_true(is.na(statz:::cov(x_na, y)))
  expect_true(is.na(statz:::cor(x_na, y)))
})

test_that("statz:::quantile matches base R Type 7 exactly", {
  set.seed(42)
  x <- runif(500, min = -100, max = 100)
  probs <- c(0, 0.25, 0.5, 0.75, 1)

  statz_q <- statz:::quantile(x, probs)
  base_q <- stats::quantile(x, probs, type = 7, names = FALSE)

  expect_equal(statz_q, base_q, tolerance = 1e-14)
})

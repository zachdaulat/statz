test_that("z_sum computes correctly", {
  expect_equal(z_sum(c(1, 2, 3)), 6)
  expect_equal(z_sum(numeric(0)), 0)
  expect_equal(z_sum(c(-1, 1)), 0)
})

test_that("z_mean computes correctly", {
  expect_equal(z_mean(c(1, 2, 3)), 2)
  expect_equal(z_mean(c(10)), 10)
  expect_true(is.nan(z_mean(numeric(0))))
})

test_that("z_median computes correctly", {
  # Odd length

  expect_equal(z_median(c(3, 1, 2)), 2)
  # Even length
  expect_equal(z_median(c(4, 1, 3, 2)), 2.5)
  # Already sorted
  expect_equal(z_median(c(1, 2, 3, 4, 5)), 3)
})

test_that("z_var matches R's var()", {
  x <- c(2, 4, 4, 4, 5, 5, 7, 9)
  expect_equal(z_var(x), var(x), tolerance = 1e-10)
})

test_that("z_sd matches R's sd()", {
  x <- c(2, 4, 4, 4, 5, 5, 7, 9)
  expect_equal(z_sd(x), sd(x), tolerance = 1e-10)
})

test_that("z_cov matches R's cov()", {
  x <- c(1, 2, 3, 4, 5)
  y <- c(2, 4, 5, 4, 5)
  expect_equal(z_cov(x, y), cov(x, y), tolerance = 1e-10)
})

test_that("z_cov(x, x) equals z_var(x)", {
  x <- c(2, 4, 4, 4, 5, 5, 7, 9)
  expect_equal(z_cov(x, x), z_var(x), tolerance = 1e-10)
})

test_that("z_cor matches R's cor()", {
  x <- c(2, 4, 4, 4, 5, 5, 7, 9)
  y <- c(1, 3, 5, 2, 7, 6, 8, 4)
  expect_equal(z_cor(x, y), cor(x, y), tolerance = 1e-10)
  expect_equal(z_cor_onepass(x, y), cor(x, y), tolerance = 1e-10)
})

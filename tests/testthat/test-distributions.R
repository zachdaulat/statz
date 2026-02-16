test_that("z_pnorm matches R's pnorm for standard normal", {
  test_values <- c(-3, -1.96, -1, 0, 1, 1.96, 3)
  for (z in test_values) {
    expect_equal(z_pnorm(z), pnorm(z), tolerance = 1e-6)
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

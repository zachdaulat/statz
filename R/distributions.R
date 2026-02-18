#' Normal distribution probability density function
#'
#' Computes the density f(x | μ, σ) for X ~ N(mean, sd^2).
#'
#' @param x A numeric value or vector
#' @param mean Mean of the normal distribution (default: 0)
#' @param sd Standard deviation of the normal distribution (default: 1)
#'
#' @return The probability density at x
#' @export
z_dnorm <- function(x, mean = 0, sd = 1) {
  if (!is.numeric(sd) || any(sd <= 0, na.rm = TRUE)) {
    rlang::abort("`sd` must be a positive numeric value.")
  }
  z_dnorm_rs(x, mean, sd)
}

#' Normal distribution cumulative distribution function
#'
#' Computes P(X <= x) for X ~ N(mean, sd^2) using
#' the Abramowitz & Stegun (1964) equation 7.1.26
#' error function approximation.
#'
#' @param x A numeric value quantile
#' @param mean Mean of the normal distribution (default: 0)
#' @param sd Standard deviation of the normal distribution (default: 1)
#'
#' @return Cumulative probability (0 < y < 1)
#' @export
#'
#' @examples
#' z_pnorm(1.96)           # ≈ 0.975
#' z_pnorm(0)              # 0.5
#' z_pnorm(100, 100, 15)   # ≈ 0.5
z_pnorm <- function(x, mean = 0, sd = 1) {
  # Input validation
  if (!is.numeric(sd) || any(sd <= 0, na.rm = TRUE)) {
    rlang::abort("`sd` must be a positive numeric value.")
  }

  # Compute z-score and call Rust backend
  z <- (x - mean) / sd
  z_pnorm_std(z)
}

#' Poisson distribution probability mass function
#'
#' Computes P(X = x)
#'
#' @param x A non-negative whole number
#' @param lambda Mean/variance, non-negative
#'
#' @return Probability mass at x
#' @export
z_dpois <- function(x, lambda) {
  # Input validation
  if (!is.numeric(x) || x < 0 || x != floor(x)) {
    rlang::abort("`x` must be a positive integer")
  }
  if (!is.numeric(lambda) || lambda <= 0) {
    rlang::abort("`lambda` must be a positive numeric value")
  }

  z_dpois_rs(as.integer(x), lambda)
}

#' Poisson cumulative probability function
#'
#' Computes P(X <= x)
#'
#' @param x A non-negative whole number
#' @param lambda Mean/variance, non-negative
#'
#' @return Cumulative probability of values <= x
#' @export
z_ppois <- function(x, lambda) {
  # Input validation
  if (!is.numeric(x) || x < 0 || x != floor(x)) {
    rlang::abort("`x` must be a positive integer")
  }
  if (!is.numeric(lambda) || lambda <= 0) {
    rlang::abort("`lambda` must be a positive numeric value")
  }

  z_ppois_rec(x, lambda)
}

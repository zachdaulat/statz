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
z_dnorm <- function(x, mean = 0, sd = 1, log = FALSE) {
  if (!is.numeric(sd) || any(sd <= 0, na.rm = TRUE)) {
    rlang::abort("`sd` must be a positive numeric value.")
  }
  z_dnorm_rs(x = x, mean = mean, sd = sd, log = log)
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
z_pnorm <- function(x, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE) {
  # Input validation
  if (!is.numeric(sd) || any(sd <= 0, na.rm = TRUE)) {
    rlang::abort("`sd` must be a positive numeric value.")
  }

  # Compute z-score and call Rust backend
  z <- (x - mean) / sd
  z_pnorm_std(z = z, lower_tail = lower.tail, log_p = log.p)
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
z_dpois <- function(x, lambda, log = FALSE) {
  # Input validation
  if (!is.numeric(x) || x < 0 || x != floor(x)) {
    rlang::abort("`x` must be a positive integer")
  }
  if (!is.numeric(lambda) || lambda <= 0) {
    rlang::abort("`lambda` must be a positive numeric value")
  }

  z_dpois_rs(x = as.integer(x), lambda = lambda, log = log)
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
z_ppois <- function(x, lambda, lower.tail = TRUE, log.p = FALSE) {
  # Input validation
  if (!is.numeric(x) || x < 0 || x != floor(x)) {
    rlang::abort("`x` must be a positive integer")
  }
  if (!is.numeric(lambda) || lambda <= 0) {
    rlang::abort("`lambda` must be a positive numeric value")
  }

  z_ppois_rec(x = x, lambda = lambda, lower_tail = lower.tail, log_p = log.p)
}


#' Gamma distribution probability density function
#'
#' Computes the PDF of the Gamma(shape, rate) distribution at x,
#' using log-space arithmetic with a Lanczos approximation for ln Γ(α)
#' adapted from the Boost.Math C++ library's lanczos13m53 parameter set.
#'
#' Supports both rate and scale parameterisations, matching R's
#' \code{\link[stats]{dgamma}} interface. If neither `rate` nor `scale`
#' is provided, defaults to rate = 1.
#'
#' @param x A positive numeric value at which to evaluate the density
#' @param shape The shape parameter (α > 0)
#' @param rate The rate parameter (β > 0). Exactly one of `rate` or
#'   `scale` should be provided.
#' @param scale The scale parameter (θ = 1/β > 0). Exactly one of
#'   `rate` or `scale` should be provided.
#' @param log Logical; if TRUE, return the log-density (default: FALSE)
#'
#' @return The gamma PDF value f(x | α, β), or ln(f) if `log = TRUE`.
#' @export
#'
#' @examples
#' # Exponential distribution (shape = 1)
#' z_dgamma(1, shape = 1, rate = 2)
#'
#' # Compare with R's dgamma
#' z_dgamma(2, shape = 3, rate = 1)
#' dgamma(2, shape = 3, rate = 1)
#'
#' # Scale parameterisation
#' z_dgamma(2, shape = 3, scale = 2)
z_dgamma <- function(x, shape, rate = NULL, scale = NULL, log = FALSE) {
  # Input validation
  if (!is.numeric(x) || x < 0) {
    rlang::abort("`x` must be a positive numeric value")
  }

  if (!is.numeric(shape) || shape <= 0) {
    rlang::abort("`shape` must be a positive numeric value")
  }

  if (!is.null(rate) && !is.null(scale)) {
    rlang::abort("Only one of `rate` or `scale` should be provided, not both")
  }

  if (is.null(rate) && is.null(scale)) {
    rate <- 1
  } else if (is.null(rate)) {
    if (!is.numeric(scale) || scale <= 0) {
      rlang::abort("`scale` must be a positive numeric value")
    }
    rate <- 1 / scale
  } else {
    if (!is.numeric(rate) || rate <= 0) {
      rlang::abort("`rate` must be a positive numeric value")
    }
  }

  z_dgamma_rs(x = x, shape = shape, rate = rate, log = log)
}

#' Gamma cumulative distribution function
#'
#' Computes P(X <= x) for X ~ Gamma(shape, rate), using a Taylor series
#' for the lower regularised incomplete gamma function when x is small
#' relative to shape, and Legendre's continued fraction (via the modified
#' Lentz algorithm) for the upper complement when x is large.
#'
#' @param x A positive numeric value (quantile)
#' @param shape The shape parameter (α > 0)
#' @param rate The rate parameter (β > 0). Exactly one of `rate` or
#'   `scale` should be provided.
#' @param scale The scale parameter (θ = 1/β > 0). Exactly one of
#'   `rate` or `scale` should be provided.
#'
#' @return Cumulative probability P(X <= x)
#' @export
#'
#' @examples
#' z_pgamma(2, shape = 3, rate = 1)
#' pgamma(2, shape = 3, rate = 1)
#'
#' z_pgamma(1, shape = 1, rate = 1)   # Exponential: 1 - exp(-1)
z_pgamma <- function(
  x,
  shape,
  rate = NULL,
  scale = NULL,
  lower.tail = TRUE,
  log.p = FALSE
) {
  # Input validation
  if (!is.numeric(x)) {
    rlang::abort("`x` must be a numeric value")
  }

  if (!is.numeric(shape) || shape <= 0) {
    rlang::abort("`shape` must be a positive numeric value")
  }

  if (!is.null(rate) && !is.null(scale)) {
    rlang::abort("Only one of `rate` or `scale` should be provided, not both")
  }

  if (is.null(rate) && is.null(scale)) {
    rate <- 1
  } else if (is.null(rate)) {
    if (!is.numeric(scale) || scale <= 0) {
      rlang::abort("`scale` must be a positive numeric value")
    }
    rate <- 1 / scale
  } else {
    if (!is.numeric(rate) || rate <= 0) {
      rlang::abort("`rate` must be a positive numeric value")
    }
  }

  z_pgamma_rs(
    x = x,
    shape = shape,
    rate = rate,
    lower_tail = lower.tail,
    log_p = log.p
  )
}

#' Tweedie distribution probability density function
#'
#' Computes the density f(y) for a Tweedie random variable using the
#' Dunn & Smyth (2005) series expansion. This implementation is currently
#' parameterised strictly for the compound Poisson-gamma case where 1 < p < 2.
#'
#' @param y A numeric value of a quantile (y >= 0).
#' @param mu The mean parameter (μ >= 0).
#' @param phi The dispersion parameter (φ > 0).
#' @param power The variance power parameter (1 < p < 2).
#' @param log Logical; if TRUE, probabilities p are given as log(p).
#'
#' @return A numeric vector of densities.
#' @export
#'
#' @examples
#' z_dtweedie(y = 1.5, mu = 2, phi = 1, power = 1.5)
z_dtweedie <- function(y, mu, phi, power, log = FALSE) {
  # Input validation
  if (!is.numeric(y) || any(y < 0, is.na(y) == TRUE)) {
    rlang::abort("`y` must be a numeric vector with values >= 0.")
  }

  if (!is.numeric(mu) || length(mu) != 1 || mu < 0) {
    rlang::abort("`mu` must be a single numeric value >= 0.")
  }

  if (!is.numeric(phi) || length(phi) != 1 || phi <= 0) {
    rlang::abort("`phi` must be a single positive numeric value.")
  }

  if (!is.numeric(power) || length(power) != 1 || power <= 1 || power >= 2) {
    rlang::abort(
      "`power` must be strictly between 1 and 2 for this compound Poisson-gamma implementation."
    )
  }

  z_dtweedie_rs(y = y, mu = mu, phi = phi, power = power, log = log)
}

#' Tweedie distribution cumulative distribution function
#'
#' Computes the cumulative probability F(y) for a Tweedie random variable using the
#' Dunn & Smyth (2005) series expansion. This implementation is currently
#' parameterised strictly for the compound Poisson-gamma case where 1 < p < 2.
#'
#' @param y A numeric quantile (y >= 0). Currently only accepts scalars.
#' @param mu The mean parameter (μ >= 0).
#' @param phi The dispersion parameter (φ > 0).
#' @param power The variance power parameter (1 < p < 2).
#' @param lower.tail Logical; if TRUE (default) probabilities are P(Y < y), otherwise, P(Y > y).
#' @param log.p Logical; if TRUE, probabilities p are given as ln(p).
#'
#' @return A numeric scalar of the cumulative probability.
#' @export
#'
#' @examples
#' z_ptweedie(y = 1.5, mu = 2, phi = 1, power = 1.5)
z_ptweedie <- function(y, mu, phi, power, lower.tail = TRUE, log.p = FALSE) {
  # Input validation for scalars
  if (!is.numeric(y) || length(y) != 1 || is.na(y) || y < 0) {
    rlang::abort("`y` must be a single numeric value >= 0.")
  }

  if (!is.numeric(mu) || length(mu) != 1 || is.na(mu) || mu < 0) {
    rlang::abort("`mu` must be a single numeric value >= 0.")
  }

  if (!is.numeric(phi) || length(phi) != 1 || is.na(phi) || phi <= 0) {
    rlang::abort("`phi` must be a single positive numeric value.")
  }

  if (
    !is.numeric(power) ||
      length(power) != 1 ||
      is.na(power) ||
      power <= 1 ||
      power >= 2
  ) {
    rlang::abort(
      "`power` must be strictly between 1 and 2 for this compound Poisson-Gamma implementation."
    )
  }

  z_ptweedie_rs(
    y = y,
    mu = mu,
    phi = phi,
    power = power,
    lower_tail = lower.tail,
    log_p = log.p
  )
}

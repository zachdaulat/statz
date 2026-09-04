#' Gamma distribution probability density function
#'
#' @description
#' Computes the PDF of the Gamma(shape, rate) distribution at x,
#' using log-space arithmetic with a Lanczos approximation for ln Γ(α)
#' adapted from the Boost.Math C++ library's lanczos13m53 parameter set.
#'
#' Supports both rate and scale parameterisations, matching R's
#' \code{\link[stats]{dgamma}} interface. If neither `rate` nor `scale`
#' is provided, defaults to rate = 1.
#'
#'    f(x | α, β) = β^α / Γ(α) · x^(α-1) · e^(-βx)
#'
#' Computed as:
#'    ln f = α·ln(β) - ln Γ(α) + (α-1)·ln(x) - β·x
#'
#' @param x A numeric (doubles) vector of positive values
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
#' statz::dgamma(1, shape = 1, rate = 2)
#'
#' # Compare with R's dgamma
#' statz::dgamma(2, shape = 3, rate = 1)
#' stats::dgamma(2, shape = 3, rate = 1)
#'
#' # Scale parameterisation
#' statz::dgamma(2, shape = 3, scale = 2)
dgamma <- function(x, shape, rate = NULL, scale = NULL, log = FALSE) {
  if (is.null(rate) && is.null(scale)) {
    rate <- 1
  } else if (is.null(rate)) {
    rate <- 1 / scale
  } else if (!is.null(rate) && !is.null(scale)) {
    rlang::abort("Only one of `rate` or `scale` should be provided, not both.")
  }

  dgamma_rs(x = x, shape = shape, rate = rate, log = log)
}

#' Gamma cumulative distribution function
#'
#' @description
#' Computes P(X <= x) for X ~ Gamma(shape, rate), using a Taylor series
#' for the lower regularised incomplete gamma function when x is small
#' relative to shape, and Legendre's continued fraction (via the modified
#' Lentz algorithm) for the upper complement when x is large. The crossover
#' at z = shape + 1 ensures that the directly computed quantity is always
#' the smaller of P and Q, avoiding precision loss from subtraction near 1.
#'
#' @param x A numeric (doubles) vector of positive values
#' @param shape The shape parameter (α > 0)
#' @param rate The rate parameter (β > 0). Exactly one of `rate` or
#'   `scale` should be provided.
#' @param scale The scale parameter (θ = 1/β > 0). Exactly one of
#'   `rate` or `scale` should be provided.
#' @param lower.tail Logical; if TRUE (default), probabilities are P(X ≤ x), otherwise, P(X > x).
#' @param log.p Logical; if TRUE, probabilities p are given as ln(p) (default: FALSE).
#'
#' @return Cumulative probability P(X <= x)
#' @export
#'
#' @examples
#' statz::pgamma(2, shape = 3, rate = 1)
#' stats::pgamma(2, shape = 3, rate = 1)
#'
#' statz::pgamma(1, shape = 1, rate = 1)   # Exponential: 1 - exp(-1)
pgamma <- function(
  x,
  shape,
  rate = NULL,
  scale = NULL,
  lower.tail = TRUE,
  log.p = FALSE
) {
  if (is.null(rate) && is.null(scale)) {
    rate <- 1
  } else if (is.null(rate)) {
    rate <- 1 / scale
  } else if (!is.null(rate) && !is.null(scale)) {
    rlang::abort("Only one of `rate` or `scale` should be provided, not both.")
  }

  pgamma_rs(
    x = x,
    shape = shape,
    rate = rate,
    lower_tail = lower.tail,
    log_p = log.p
  )
}

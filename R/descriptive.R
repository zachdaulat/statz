#' Descriptive Statistics (Rust Backend)
#'
#' Core descriptive statistics functions implemented in Rust for
#' pedagogical demonstration. These mirror base R functions like
#' [base::sum()], [base::mean()], [stats::var()], [stats::sd()],
#' and [stats::cov()], but are implemented from scratch in Rust.
#'
#' @name descriptive
#' @examples
#' x <- c(2, 4, 4, 4, 5, 5, 7, 9)
#'
#' # Compare with base R
#' statz:::mean(x)
#' base::mean(x)
#'
#' statz:::var(x)
#' stats::var(x)
#'
#' statz:::sd(x)
#' stats::sd(x)
NULL

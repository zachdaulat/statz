#' Fit a linear model via Rust
#'
#' @description
#' Fits an ordinary least squares (OLS) regression using compiled Rust linear
#' algebra engines. Designed as a pedagogical alternative to `stats::lm()`.
#'
#' @param formula An object of class "formula" (or one that can be coerced to that class).
#' @param data A data frame, tibble, or environment containing the model's variables.
#' @param engine A character string specifying the computational backend.
#'   Currently supports "cholesky" (fastest, requires full rank) or "qr" (numerically stable).
#'
#' @return A list of class `statz_lm` containing coefficients, standard errors,
#'   fitted values, residuals, degrees of freedom, and the residual standard error.
#' @export
#'
#' @examples
#' # Fit a basic model
#' fit <- lm(mpg ~ wt + cyl, data = mtcars)
lm <- function(formula, data, engine = c("cholesky", "qr", "svd")) {
  # Matching the engine argument to the provided options
  engine <- rlang::arg_match(engine)

  # --- 1. Evaluating the formula and components
  # na.fail ensures the function aborts immediately if any missing data is present
  # model.matrix automatically adds the intercept column of 1s
  mf <- stats::model.frame(formula, data, na.action = stats::na.fail)
  x_mat <- stats::model.matrix(formula, mf)
  y_col <- stats::model.response(mf)

  if (!is.numeric(y_col)) {
    rlang::abort("The response variable must be strictly numeric.")
  }

  # --- 2. Passing inputs to Rust
  res <- lm_rs(x = x_mat, y = as.double(y_col), engine = engine)

  # --- 3. Formatting the output
  # Applying the column names from the design matrix to the output vectors
  # Applying custom class for future S3 methods like print() and summary()
  names(res$coefficients) <- colnames(x_mat)
  names(res$std_errors) <- colnames(x_mat)
  class(res) <- "statz_lm"

  res
}

//! Inverse Gaussian (Wald) distribution density (`dinvgauss`) and
//! cumulative probability (`pinvgauss`).

use super::normal::pnorm;
use crate::consts::*;
use extendr_api::prelude::*;

extendr_module! {
    mod invgauss;
    fn dinvgauss_r;
    fn pinvgauss_r;
}

/// Compute the Inverse Gaussian probability density function
///
/// @description
/// Evaluated natively in log-space for maximum numerical stability.
///
/// @param y A numeric (double) vector of quantiles (y > 0).
/// @param mu The mean parameter (μ > 0).
/// @param lambda The shape parameter (λ > 0).
/// @param log Logical; if TRUE, probabilities p are given as ln(p).
/// @return A numeric vector of probability densities.
/// @export
#[extendr(r_name = "dinvgauss")]
pub fn dinvgauss_r(
    y: &[f64],
    #[extendr(default = "1")] mu: f64,
    #[extendr(default = "NULL")] lambda: f64,
    #[extendr(default = "FALSE")] log: bool,
) -> Doubles {
    y.iter()
        .map(|&yi| dinvgauss(yi, mu, lambda, log))
        .collect::<Doubles>()
}

pub fn dinvgauss(y: f64, mu: f64, lambda: f64, log: bool) -> f64 {
    // 1: Fast paths
    if mu <= 0.0 || lambda <= 0.0 {
        return f64::NAN;
    }
    if y <= 0.0 || y.is_infinite() {
        return if log { f64::NEG_INFINITY } else { 0.0 };
    }

    // 2: Normalisation constant terms
    let l_lam: f64 = 0.5 * lambda.ln();
    let l_y: f64 = 1.5 * y.ln();

    // 3: Exponential term
    let exp_term: f64 = -lambda * (y - mu).powi(2) / (2.0 * mu.powi(2) * y);

    // 4: Combine terms in log space
    let l_pdf: f64 = l_lam - LN_SQRT_2PI - l_y + exp_term;

    if log {
        l_pdf
    } else {
        l_pdf.exp()
    }
}

/// Compute the Inverse Gaussian cumulative distribution function
///
/// @description
/// Utilizes the standard normal CDF for full machine precision in extreme tails,
/// featuring a structural overflow brake for extreme parameterisations.
///
/// @param y A numeric (double) vector of quantiles (y > 0).
/// @param mu The mean parameter (μ > 0).
/// @param lambda The shape parameter (λ > 0).
/// @param lower_tail Logical; if TRUE (default), probabilities are P(Y <= y).
/// @param log_p Logical; if TRUE, probabilities p are given as ln(p).
/// @return A numeric vector of cumulative probabilities.
/// @export
#[extendr(r_name = "pinvgauss")]
pub fn pinvgauss_r(
    y: &[f64],
    #[extendr(default = "1")] mu: f64,
    #[extendr(default = "NULL")] lambda: f64,
    #[extendr(default = "TRUE")] lower_tail: bool,
    #[extendr(default = "FALSE")] log_p: bool,
) -> Doubles {
    y.iter()
        .map(|&yi| pinvgauss(yi, mu, lambda, lower_tail, log_p))
        .collect::<Doubles>()
}

pub fn pinvgauss(y: f64, mu: f64, lambda: f64, lower_tail: bool, log_p: bool) -> f64 {
    // 0: Fast paths for domain boundaries and safety
    if mu <= 0.0 || lambda <= 0.0 {
        return f64::NAN;
    }
    if y <= 0.0 {
        let p: f64 = if lower_tail { 0.0 } else { 1.0 };
        return if log_p { p.ln() } else { p };
    }
    if y == f64::INFINITY {
        let p: f64 = if lower_tail { 1.0 } else { 0.0 };
        return if log_p { p.ln() } else { p };
    }

    // 1: Define z inputs to normal CDF
    let z_1: f64 = (lambda / y).sqrt() * ((y / mu) - 1.0);
    let z_2: f64 = -(lambda / y).sqrt() * ((y / mu) + 1.0);

    // 2: Evaluating IG CDF terms
    let pnorm_1: f64 = pnorm(z_1, lower_tail, false);
    let l_exp: f64 = 2.0 * lambda / mu;

    // Break for extreme parameterisations when correction term is
    // negligible exploiting pnorm decreasing faster than exp term
    if l_exp > f64::MAX.ln() {
        return if log_p { pnorm_1.ln() } else { pnorm_1 };
    }

    let pnorm_2: f64 = pnorm(z_2, true, false);
    let corr: f64 = l_exp.exp() * pnorm_2;

    // 3: Evaluate tail argument
    let cdf: f64 = if lower_tail {
        pnorm_1 + corr
    } else {
        pnorm_1 - corr
    };

    // 4: log space reeturn check
    if log_p {
        cdf.ln()
    } else {
        cdf
    }
}

// Rust-side unit tests
#[allow(clippy::excessive_precision)]
#[allow(clippy::manual_range_contains)]
#[cfg(test)]
mod tests {
    use super::*;

    // --- dinvgauss tests ---

    #[test]
    fn test_dinvgauss_peak() {
        // When y = mu = 1, lambda = 1, the exponential term is 0.
        // PDF simplifies to 1 / sqrt(2 * PI)
        let expected = 1.0 / SQRT_2PI;
        assert!((dinvgauss(1.0, 1.0, 1.0, false) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_dinvgauss_fast_paths() {
        assert_eq!(dinvgauss(0.0, 1.0, 1.0, false), 0.0);
        assert_eq!(dinvgauss(-1.0, 1.0, 1.0, false), 0.0);
        assert_eq!(dinvgauss(f64::INFINITY, 1.0, 1.0, false), 0.0);
        assert_eq!(dinvgauss(0.0, 1.0, 1.0, true), f64::NEG_INFINITY);
    }

    // --- pinvgauss tests ---

    #[test]
    fn test_pinvgauss_fast_paths() {
        assert_eq!(pinvgauss(0.0, 1.0, 1.0, true, false), 0.0);
        assert_eq!(pinvgauss(0.0, 1.0, 1.0, false, false), 1.0);
        assert_eq!(pinvgauss(f64::INFINITY, 1.0, 1.0, true, false), 1.0);
    }

    #[test]
    fn test_pinvgauss_tail_complementarity() {
        // F(y) + S(y) = 1.0
        let y = 2.5;
        let mu = 1.5;
        let lambda = 2.0;
        let lower = pinvgauss(y, mu, lambda, true, false);
        let upper = pinvgauss(y, mu, lambda, false, false);
        assert!((lower + upper - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_pinvgauss_extreme_parameter_brake() {
        // Triggers the exponent > 709.0 brake.
        // 2 * lambda / mu = 1000.
        // Should safely return the first pnorm term without NaN panics.
        let cdf = pinvgauss(1.0, 1.0, 500.0, true, false);
        assert!(cdf >= 0.0 && cdf <= 1.0 && !cdf.is_nan());
    }
}

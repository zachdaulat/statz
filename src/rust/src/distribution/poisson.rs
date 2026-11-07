//! Poisson distribution probability mass (`dpois`) and cumulative probability (`ppois`).

use extendr_api::prelude::*;

extendr_module! {
    mod poisson;
    fn dpois_r;
    fn ppois_r;
}

/// Compute the Poisson probability mass function
///
/// @description
/// Computes P(X = x) using log-space arithmetic to avoid factorial overflow.
///
/// @param x A numeric (double) vector of non-negative whole numbers.
/// @param lambda The rate parameter (λ > 0).
/// @param log Logical; if TRUE, probabilities p are given as ln(p).
/// @return A numeric vector of probability masses.
/// @export
#[extendr(r_name = "dpois")]
pub fn dpois_r(x: &[f64], lambda: f64, #[extendr(default = "FALSE")] log: bool) -> Doubles {
    // Scalar NA check outside the loop for speed
    if lambda.is_na() {
        return (0..x.len()).map(|_| f64::na()).collect::<Doubles>();
    }

    x.iter()
        .map(|&xi| {
            if xi.is_na() {
                f64::na() // R-specific NA handled at the boundary
            } else {
                dpois(xi, lambda, log)
            }
        })
        .collect::<Doubles>()
}

pub fn dpois(x: f64, lambda: f64, log: bool) -> f64 {
    // Pure mathematical bounds
    if x.is_nan() || lambda.is_nan() || lambda < 0.0 {
        return f64::NAN;
    }

    // R returns 0 for non-integer densities
    if x < 0.0 || x.fract() != 0.0 {
        return if log { f64::NEG_INFINITY } else { 0.0 };
    }

    let x_int = x as u64;
    let mut p = x * lambda.ln() - lambda;

    for i in 1..=x_int {
        p -= (i as f64).ln();
    }

    if log {
        p
    } else {
        p.exp()
    }
}

/// Compute the Poisson cumulative distribution function
///
/// @description
/// Computes P(X <= x) using a recurrence relation for numerical stability.
///
/// @param x A numeric (double) vector of non-negative whole numbers.
/// @param lambda The rate parameter (λ > 0).
/// @param lower_tail Logical; if TRUE (default), probabilities are P(X <= x).
/// @param log_p Logical; if TRUE, probabilities p are given as ln(p).
/// @return A numeric vector of cumulative probabilities.
/// @export
#[extendr(r_name = "ppois")]
pub fn ppois_r(
    x: &[f64],
    lambda: f64,
    #[extendr(default = "TRUE")] lower_tail: bool,
    #[extendr(default = "FALSE")] log_p: bool,
) -> Doubles {
    if lambda.is_na() {
        return (0..x.len()).map(|_| f64::na()).collect::<Doubles>();
    }

    x.iter()
        .map(|&xi| {
            if xi.is_na() {
                f64::na()
            } else {
                ppois(xi, lambda, lower_tail, log_p)
            }
        })
        .collect::<Doubles>()
}

pub fn ppois(x: f64, lambda: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || lambda.is_nan() || lambda < 0.0 {
        return f64::NAN;
    }

    if x < 0.0 {
        let cdf: f64 = 0.0;
        return if lower_tail {
            if log_p {
                f64::NEG_INFINITY
            } else {
                cdf
            }
        } else {
            if log_p {
                0.0
            } else {
                1.0 - cdf
            }
        };
    }

    // ppois evaluates at the floor of continuous values
    let x_floor = x as u64;
    let mut p: f64 = (-lambda).exp();
    let mut cdf: f64 = p;

    for k in 1..=x_floor {
        p *= lambda / (k as f64);
        cdf += p;
    }

    cdf = if lower_tail { cdf } else { 1.0 - cdf };

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

    // --- dpois tests ---

    #[test]
    fn test_dpois_at_zero() {
        // P(X=0 | λ=3) = e^{-3} ≈ 0.0497871
        let expected = (-3.0_f64).exp();
        assert!((dpois(0.0, 3.0, false) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dpois_known_value() {
        // P(X=2 | λ=3) = 3^2 * e^{-3} / 2! = 9 * e^{-3} / 2 ≈ 0.2240418
        let expected = 9.0 * (-3.0_f64).exp() / 2.0;
        assert!((dpois(2.0, 3.0, false) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dpois_sums_to_one() {
        // Sum of PMF over a large range should ≈ 1
        let lambda = 5.0;
        let total: f64 = (0..=50).map(|k| dpois(k as f64, lambda, false)).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dpois_non_integers_and_domain() {
        // Negative lambda returns NaN
        assert!(dpois(2.0, -1.0, false).is_nan());

        // Non-integers and negative counts evaluate to 0.0 density
        assert_eq!(dpois(2.5, 3.0, false), 0.0);
        assert_eq!(dpois(-1.0, 3.0, false), 0.0);
        assert_eq!(dpois(2.5, 3.0, true), f64::NEG_INFINITY);
    }

    // --- ppois tests ---

    #[test]
    fn test_ppois_at_zero() {
        // P(X<=0 | λ=3) = P(X=0) = e^{-3}
        let expected = (-3.0_f64).exp();
        assert!((ppois(0.0, 3.0, true, false) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_ppois_monotone() {
        // CDF must be non-decreasing
        let lambda = 4.0;
        let mut prev = 0.0;
        for k in 0..=20 {
            let current = ppois(k as f64, lambda, true, false);
            assert!(current >= prev);
            prev = current;
        }
    }

    #[test]
    fn test_ppois_approaches_one() {
        // P(X<=50 | λ=5) should be very close to 1
        assert!((ppois(50.0, 5.0, true, false) - 1.0).abs() < 1e-10);
    }
}

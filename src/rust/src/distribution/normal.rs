//! Normal (Gaussian) distribution density (`dnorm`) and cumulative probability (`pnorm`).
//!
//! Evaluates extreme tail probabilities to full `f64` precision via `libm::erfc`.

#![allow(clippy::excessive_precision)]

use crate::consts::*;
use extendr_api::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

extendr_module! {
    mod normal;
    fn dnorm_r;
    fn pnorm_r;
}

/// Compute the normal probability density function
///
/// @description
/// Computes the density f(x | μ, σ) for X ~ N(mean, sd^2).
///
/// @param x A numeric (double) vector of quantiles.
/// @param mean Mean of the normal distribution.
/// @param sd Standard deviation of the normal distribution.
/// @param log Logical; if TRUE, probabilities p are given as ln(p).
/// @return A numeric vector of probability densities.
/// @export
#[extendr(r_name = "dnorm")]
pub fn dnorm_r(
    x: &[f64],
    #[extendr(default = "0.0")] mean: f64,
    #[extendr(default = "1.0")] sd: f64,
    #[extendr(default = "FALSE")] log: bool,
) -> Doubles {
    x.iter()
        .map(|&xi| dnorm(xi, mean, sd, log))
        .collect::<Doubles>()
}

pub fn dnorm(x: f64, mean: f64, sd: f64, log: bool) -> f64 {
    // Fast paths
    if sd < 0.0 {
        return f64::NAN;
    }
    if sd == 0.0 {
        // For dnorm: Infinity at the mean, 0 elsewhere
        if x == mean {
            return f64::INFINITY;
        } else {
            return if log { f64::NEG_INFINITY } else { 0.0 };
        };
    }

    let z: f64 = (x - mean) / sd;
    let pdf: f64 = -sd.ln() - LN_SQRT_2PI - (0.5 * z * z);

    // Evaluating boolean for returning in log or regular space
    if log {
        pdf
    } else {
        pdf.exp()
    }
}

/// Compute the normal cumulative distribution function
///
/// @description
/// Computes Φ(z) = P(X <= x) for X ~ N(mean, sd^2) utilizing the `libm`
/// complementary error function (erfc) for full 64-bit machine precision.
///
/// @param x A numeric (double) vector of quantiles.
/// @param mean Mean of the normal distribution.
/// @param sd Standard deviation of the normal distribution.
/// @param lower_tail Logical; if TRUE (default), probabilities are P(X <= x).
/// @param log_p Logical; if TRUE, probabilities p are given as ln(p).
/// @return A numeric vector of cumulative probabilities.
/// @export
#[extendr(r_name = "pnorm")]
pub fn pnorm_r(
    x: &[f64],
    #[extendr(default = "0.0")] mean: f64,
    #[extendr(default = "1.0")] sd: f64,
    #[extendr(default = "TRUE")] lower_tail: bool,
    #[extendr(default = "FALSE")] log_p: bool,
) -> Doubles {
    // 1. Parameter boundary fast-paths
    if mean.is_na() || sd.is_na() {
        return (0..x.len()).map(|_| f64::na()).collect::<Doubles>();
    }
    if mean.is_nan() || sd.is_nan() || sd < 0.0 {
        return (0..x.len()).map(|_| f64::NAN).collect::<Doubles>();
    }

    x.iter()
        .map(|&xi| {
            if xi.is_na() {
                f64::na()
            } else if sd == 0.0 {
                // CDF point mass logic when variance is 0
                let cdf: f64 = if xi >= mean { 1.0 } else { 0.0 };
                let p: f64 = if lower_tail { cdf } else { 1.0 - cdf };
                if log_p {
                    p.ln()
                } else {
                    p
                }
            } else {
                let z = (xi - mean) / sd;
                pnorm(z, lower_tail, log_p)
            }
        })
        .collect::<Doubles>()
}

pub fn pnorm(z: f64, lower_tail: bool, log_p: bool) -> f64 {
    // 0: NaN fast path
    if z.is_nan() {
        return f64::NAN;
    }

    // 1: Build erfc() input u
    let mut u: f64 = z * FRAC_1_SQRT_2;

    // 2: Tail handling to adjust sign of u
    if lower_tail {
        u = -u;
    }

    // 3: erfc() call for phi
    // Using libm:erfc() directly via path
    let phi: f64 = libm::erfc(u) / 2.0;

    if log_p {
        phi.ln()
    } else {
        phi
    }
}

// Rust-side unit tests
#[allow(clippy::excessive_precision)]
#[allow(clippy::manual_range_contains)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // --- dnorm tests ---

    #[test]
    fn test_dnorm_standard_at_zero() {
        // dnorm(0, 0, 1) = 1/sqrt(2*pi) ≈ 0.3989422804
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((dnorm(0.0, 0.0, 1.0, false) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dnorm_nonstandard() {
        // dnorm(10, 10, 3) should equal dnorm(0, 0, 3)
        assert!((dnorm(10.0, 10.0, 3.0, false) - dnorm(0.0, 0.0, 3.0, false)).abs() < 1e-10);
    }

    #[test]
    fn test_dnorm_degenerate_and_nan() {
        // Negative standard deviation returns NaN
        assert!(dnorm(0.0, 0.0, -1.0, false).is_nan());

        // Zero standard deviation produces Dirac point mass
        assert_eq!(dnorm(2.0, 2.0, 0.0, false), f64::INFINITY);
        assert_eq!(dnorm(1.0, 2.0, 0.0, false), 0.0);
    }

    // --- pnorm tests ---

    #[test]
    fn test_pnorm_at_zero() {
        // Φ(0) = 0.5 exactly. Machine precision.
        assert!((pnorm(0.0, true, false) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_pnorm_symmetry() {
        // Φ(-z) = 1 - Φ(z)
        let z = 1.5;
        assert!((pnorm(-z, true, false) - (1.0 - pnorm(z, true, false))).abs() < 1e-15);
    }

    #[test]
    fn test_pnorm_known_values() {
        // Reference values from WolframAlpha: ex. N[CDF[NormalDistribution[0, 1], 1.96], 25]
        // Φ(1) ≈ 0.8413447460685429
        assert!((pnorm(1.0, true, false) - 0.8413447460685429485852325).abs() < 1e-15);
        // Φ(-1) ≈ 0.15865525393145705
        assert!((pnorm(-1.0, true, false) - 0.1586552539314570514147675).abs() < 1e-15);
        // Φ(1.96) ≈ 0.9750021048517795
        assert!((pnorm(1.96, true, false) - 0.9750021048517795658634157).abs() < 1e-15);
        // Φ(3) ≈ 0.9986501019683699
        assert!((pnorm(3.0, true, false) - 0.9986501019683699054733482).abs() < 1e-15);
    }

    #[test]
    fn test_pnorm_edge_cases() {
        assert_eq!(pnorm(f64::INFINITY, true, false), 1.0);
        assert_eq!(pnorm(f64::NEG_INFINITY, true, false), 0.0);
        assert!(pnorm(f64::NAN, true, false).is_nan());
    }

    #[test]
    fn test_pnorm_deep_tails() {
        // Reference values also generated from WolframAlpha
        // A&S 7.1.26 carries an absolute error of ~1.5e-7.
        // For z = -5.0, the true mathematical probability is ~2.866e-7.
        // A&S would fail catastrophically here. libm::erfc retains full mantissa precision.

        let z: f64 = -5.0;
        let expected: f64 = 2.866515718791939116737523e-7;
        let calc: f64 = pnorm(z, true, false);

        // Use relative error: |calc - expected| / expected
        let relative_error: f64 = (calc - expected).abs() / expected;
        assert!(
            relative_error < 1e-14,
            "Deep tail relative precision lost. Error: {}",
            relative_error
        );

        // Test an even deeper tail where z = -8 (probability ~ 6.22e-16)
        let z_deep: f64 = -8.0;
        let expected_deep: f64 = 6.220960574271784123515995e-16;
        let calc_deep: f64 = pnorm(z_deep, true, false);
        let relative_error_deep: f64 = (calc_deep - expected_deep).abs() / expected_deep;
        assert!(
            relative_error_deep < 1e-14,
            "Extreme deep tail relative precision lost."
        );

        // Test extreme tail where z = -20 (probability ~ 2.75e-89)
        let z_xtr: f64 = -20.0;
        let expected_xtr: f64 = 2.753624118606233695075623e-89;
        let calc_xtr: f64 = pnorm(z_xtr, true, false);
        let relative_error_xtr: f64 = (calc_xtr - expected_xtr).abs() / expected_xtr;
        assert!(
            relative_error_xtr < 1e-13,
            "Extreme deep tail relative precision lost."
        );
    }
}

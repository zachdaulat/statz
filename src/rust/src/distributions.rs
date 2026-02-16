use extendr_api::prelude::*;
use std::f64::consts::{PI, SQRT_2};

// Placeholder module for probability distribution functions.
// I will implement these as I progress through my statistics learning.
//
// Planned functions:
// - dnorm, pnorm         (normal distribution)
// - dpois, ppois         (Poisson distribution)
// - dgamma, pgamma       (gamma distribution)
// - dtweedie, ptweedie   (Tweedie/compound Poisson-gamma)

/// Compute the normal probability density function (PDF)
/// @param x A single numeric value at which to evaluate the density
/// @param mean The mean of the normal distribution (μ)
/// @param sd The standard deviation of the normal distribution (σ > 0)
/// @return The probability density f(x | μ, σ)
/// @export
#[extendr]
pub fn z_dnorm_rs(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    (1.0 / (sd * (2.0 * PI).sqrt())) * (-0.5 * z * z).exp()
}

/// Compute the standard normal cumulative distribution function (CDF)
/// using the Abramowitz and Stegun (1972, 10th ed.) equation 7.1.26
/// error function approximation.
/// Maximum absolute error: |ε| < 1.5 × 10⁻⁷
/// @param z A z-score (standardised value)
/// @return Cumulative probability Φ(z) = P(Z ≤ z) for Z ~ N(0,1)
/// @export
#[extendr]
#[allow(non_upper_case_globals)]
pub fn z_pnorm_std(z: f64) -> f64 {
    // Ask about need to reference variables z, t, u, and use of const

    if z.is_nan() {
        return f64::NAN;
    }
    if z == f64::INFINITY {
        return 1.0;
    }
    if z == f64::NEG_INFINITY {
        return 0.0;
    }

    let u = (z / SQRT_2).abs();

    // Constants defined by A&S, p.299
    const p: f64 = 0.32759_11;
    const a_1: f64 = 0.25482_9592;
    const a_2: f64 = -0.28449_6736;
    const a_3: f64 = 1.42141_3741;
    const a_4: f64 = -1.45315_2027;
    const a_5: f64 = 1.06140_5429;

    let t = 1.0 / (1.0 + (p * u));

    // Horner's method for erfc(|u|)
    let erfc: f64 = a_5
        .mul_add(t, a_4)
        .mul_add(t, a_3)
        .mul_add(t, a_2)
        .mul_add(t, a_1)
        * t
        * (-u * u).exp();

    let erf: f64 = (1.0 - erfc).copysign(z);

    (1.0 + erf) / 2.0
}

extendr_module! {
    mod distributions;
    fn z_dnorm_rs;
    fn z_pnorm_std;
}

// Rust-side unit tests
#[cfg(test)]
mod tests {
    use super::*;

    // --- z_dnorm tests ---

    #[test]
    fn test_dnorm_standard_at_zero() {
        // dnorm(0, 0, 1) = 1/sqrt(2*pi) ≈ 0.3989422804
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((z_dnorm_rs(0.0, 0.0, 1.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dnorm_nonstandard() {
        // dnorm(10, 10, 3) should equal dnorm(0, 0, 3)
        assert!((z_dnorm_rs(10.0, 10.0, 3.0) - z_dnorm_rs(0.0, 0.0, 3.0)).abs() < 1e-10);
    }

    // --- z_pnorm_std tests ---

    #[test]
    fn test_pnorm_at_zero() {
        // Φ(0) = 0.5 exactly
        assert!((z_pnorm_std(0.0) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_pnorm_symmetry() {
        // Φ(-z) = 1 - Φ(z)
        let z = 1.5;
        assert!((z_pnorm_std(-z) - (1.0 - z_pnorm_std(z))).abs() < 1e-7);
    }

    #[test]
    fn test_pnorm_known_values() {
        // Known values from standard normal tables
        // Φ(1) ≈ 0.8413447
        assert!((z_pnorm_std(1.0) - 0.841_344_7).abs() < 1e-6);
        // Φ(-1) ≈ 0.1586553
        assert!((z_pnorm_std(-1.0) - 0.158_655_3).abs() < 1e-6);
        // Φ(1.96) ≈ 0.9750021
        assert!((z_pnorm_std(1.96) - 0.975_002_1).abs() < 1e-6);
        // Φ(3) ≈ 0.9986501
        assert!((z_pnorm_std(3.0) - 0.998_650_1).abs() < 1e-6);
    }

    #[test]
    fn test_pnorm_edge_cases() {
        assert_eq!(z_pnorm_std(f64::INFINITY), 1.0);
        assert_eq!(z_pnorm_std(f64::NEG_INFINITY), 0.0);
        assert!(z_pnorm_std(f64::NAN).is_nan());
    }
}

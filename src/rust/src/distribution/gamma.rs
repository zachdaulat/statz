//! Gamma distribution density (`dgamma`) and cumulative probability (`pgamma`).
//!
//! Dispatches between Taylor series and Legendre continued fractions at $z = \alpha + 1$
//! to avoid subtractive cancellation in the tails.

#![allow(clippy::excessive_precision)]

use super::special::{lgamma, lower_gamma_series, upper_gamma_cf};
use extendr_api::prelude::*;

extendr_module! {
    mod gamma;
    fn dgamma_r;
    fn pgamma_r;
}

/// Gamma PDF Engine
///
/// @description
/// Evaluates the PDF of the Gamma(shape, rate) distribution at x using
/// log-space arithmetic to avoid overflow:
///
///   f(x | α, β) = β^α / Γ(α) · x^(α-1) · e^(-βx)
///
/// Computed as:
///   ln f = α·ln(β) - ln Γ(α) + (α-1)·ln(x) - β·x
///
/// The log-gamma term is evaluated via `lgamma()` (Boost adaptation).
///
/// @param x A positive numeric value
/// @param shape The shape parameter (α > 0)
/// @param rate The rate parameter (β > 0)
/// @param log Logical; if TRUE, return the log-density
/// @return The gamma PDF value at x, or ln(PDF) if log = TRUE
/// @keywords internal
#[extendr(r_name = "dgamma_rs")]
pub fn dgamma_r(x: &[f64], shape: f64, rate: f64, log: bool) -> Doubles {
    x.iter()
        .map(|&xi| dgamma(xi, shape, rate, log))
        .collect::<Doubles>()
}

pub fn dgamma(x: f64, shape: f64, rate: f64, log: bool) -> f64 {
    // Fast path input handling
    if shape <= 0.0 || rate <= 0.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return if log { f64::NEG_INFINITY } else { 0.0 };
    }

    let lnum: f64 = shape * rate.ln();
    let lgam: f64 = lgamma(shape);
    let lpow: f64 = (shape - 1.0) * x.ln();

    let pdf: f64 = lnum - lgam + lpow - (rate * x);

    // Evaluating boolean for returning in log or regular space
    if log {
        pdf
    } else {
        pdf.exp()
    }
}

/// Gamma CDF Engine
///
/// @description
/// Compute the gamma CDF: P(X <= x) for X ~ Gamma(shape, rate).
///
/// Dispatches to `lower_gamma_series` (Taylor series for P) when the
/// scaled argument z = rate · x is small relative to shape, and to
/// `upper_gamma_cf` (Legendre continued fraction for Q) otherwise.
/// The crossover at z = shape + 1 ensures that the directly computed
/// quantity is always the smaller of P and Q, avoiding precision loss
/// from subtraction near 1.
///
/// @param x A positive numeric value (quantile)
/// @param shape The shape parameter (α > 0)
/// @param rate The rate parameter (β > 0)
/// @return The cumulative probability P(X ≤ x | α, β)
/// @keywords internal
#[extendr(r_name = "pgamma_rs")]
pub fn pgamma_r(x: &[f64], shape: f64, rate: f64, lower_tail: bool, log_p: bool) -> Doubles {
    x.iter()
        .map(|&xi| pgamma(xi, shape, rate, lower_tail, log_p))
        .collect::<Doubles>()
}

pub fn pgamma(x: f64, shape: f64, rate: f64, lower_tail: bool, log_p: bool) -> f64 {
    // Checking for R's specific NA_real_ first
    if x.is_na() || shape.is_na() || rate.is_na() {
        return f64::na(); // extendr bridges this back as NA_real_
    }

    if x.is_nan() || shape.is_nan() || rate.is_nan() {
        return f64::NAN; // extendr bridges this back as NaN
    }

    // Domain boundaries (Note: adjusted for log_p)
    if x <= 0.0 {
        let p: f64 = if lower_tail { 0.0 } else { 1.0 };
        return if log_p { p.ln() } else { p };
    }
    if x.is_infinite() {
        let p: f64 = if lower_tail { 1.0 } else { 0.0 };
        return if log_p { p.ln() } else { p };
    }

    // Prepping z variable and dispatch logic assigning if/else to `cdf`
    // Calculated in regular space
    let z: f64 = rate * x;

    let cdf: f64 = if z < shape + 1.0 {
        let p: f64 = lower_gamma_series(shape, z);
        if lower_tail {
            p
        } else {
            1.0 - p
        }
    } else {
        let q: f64 = upper_gamma_cf(shape, z);
        if lower_tail {
            1.0 - q
        } else {
            q
        }
    };

    // 5. Log space return boolean argument check
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

    // --- dgamma tests ---

    #[test]
    fn test_dgamma_exponential_special_case() {
        // Gamma(shape=1, rate=λ) is Exponential(λ)
        // f(x) = λ * exp(-λx), so dgamma(1, 1, 2) = 2 * exp(-2) ≈ 0.2706706
        let result = dgamma(1.0, 1.0, 2.0, false);
        let expected = 2.0 * (-2.0_f64).exp();
        assert!((result - expected).abs() < 1e-15);
    }

    #[test]
    fn test_dgamma_known_values() {
        // Cross-referenced with R: dgamma(2, shape=3, rate=1)
        // = 1^3 / Γ(3) * 2^2 * exp(-2) = 2^2 * exp(-2) / 2 ≈ 0.2706706
        let result = dgamma(2.0, 3.0, 1.0, false);
        let expected = 0.2706705664732254;
        assert!(
            (result - expected).abs() < 1e-15,
            "dgamma(2, 3, 1) = {}, expected {}",
            result,
            expected
        );

        // dgamma(1, shape=2, rate=2) ≈ 0.5413411
        let result2 = dgamma(1.0, 2.0, 2.0, false);
        let expected2 = 0.5413411329464508;
        assert!((result2 - expected2).abs() < 1e-15);

        // dgamma(0.5, shape=0.5, rate=1) ≈ 0.4393913
        let result3 = dgamma(0.5, 0.5, 0.5, false);
        let expected3 = 0.4393912894677223970468620;
        assert!((result3 - expected3).abs() < 1e-15);
    }

    #[test]
    fn test_dgamma_log_mode() {
        // log(dgamma(2, 3, 1)) ≈ ln(0.2706706)
        let log_result = dgamma(2.0, 3.0, 1.0, true);
        let direct_result = dgamma(2.0, 3.0, 1.0, false);
        assert!((log_result - direct_result.ln()).abs() < 1e-14);
    }

    #[test]
    fn test_dgamma_large_shape() {
        // For large shape, gamma approaches normal; density should be well-behaved
        // dgamma(50, shape=50, rate=1) — near the mode at (shape-1)/rate = 49
        let result = dgamma(50.0, 50.0, 1.0, false);
        assert!(result > 0.0);
        assert!(result < 1.0);
        // dgamma(50, 50, 1) ≈ 0.056325...
        assert!((result - 0.05632500632519082541154874).abs() < 1e-15);
    }

    #[test]
    fn test_dgamma_small_shape() {
        // Small shape parameter — density concentrated near zero
        // dgamma(0.01, shape=0.1, rate=1) should be large
        let result = dgamma(0.01, 0.1, 1.0, false);
        assert!(result > 1.0); // Density can exceed 1 for small shape
    }

    #[test]
    fn test_dgamma_integrates_approximately() {
        // Rough numerical integration check: the PDF should integrate to ~1
        // Use a simple Riemann sum over [0.001, 20] with shape=2, rate=1
        let shape = 2.0;
        let rate = 1.0;
        let dx = 0.001;
        let n = 20000;
        let sum: f64 = (1..=n)
            .map(|i| {
                let x = i as f64 * dx;
                dgamma(x, shape, rate, false) * dx
            })
            .sum();
        // Should be close to 1, allowing for discretisation error
        assert!((sum - 1.0).abs() < 0.01, "Integral = {}", sum);
    }

    // --- pgamma tests ---

    #[test]
    fn test_pgamma_exponential_cdf() {
        // Gamma(1, rate) is Exponential(rate): CDF = 1 - exp(-rate * x)
        let rate = 2.0;
        for &x in &[0.1, 0.5, 1.0, 2.0, 5.0] {
            let expected = 1.0 - f64::exp(-rate * x);
            assert!(
                (pgamma(x, 1.0, rate, true, false) - expected).abs() < 1e-10,
                "pgamma({}, 1, {}) = {}, expected {}",
                x,
                rate,
                pgamma(x, 1.0, rate, true, false),
                expected
            );
        }
    }

    #[test]
    fn test_pgamma_integer_shape() {
        // For integer shape n: P(X <= x) = 1 - e^(-x) * Σ_{k=0}^{n-1} x^k / k!
        // Shape = 2, rate = 1: P = 1 - e^(-x)(1 + x)
        let x = 3.0;
        let expected = 1.0 - f64::exp(-x) * (1.0 + x);
        assert!(
            (pgamma(x, 2.0, 1.0, true, false) - expected).abs() < 1e-10,
            "pgamma(3, 2, 1) = {}, expected {}",
            pgamma(x, 2.0, 1.0, true, false),
            expected
        );

        // Shape = 3, rate = 1: P = 1 - e^(-x)(1 + x + x²/2)
        let expected3 = 1.0 - f64::exp(-x) * (1.0 + x + x * x / 2.0);
        assert!(
            (pgamma(x, 3.0, 1.0, true, false) - expected3).abs() < 1e-10,
            "pgamma(3, 3, 1) = {}, expected {}",
            pgamma(x, 3.0, 1.0, true, false),
            expected3
        );
    }

    #[test]
    fn test_pgamma_known_values() {
        // Cross-referenced with R's pgamma()
        // pgamma(1, shape=2, rate=1) ≈ 0.2642411
        assert!((pgamma(1.0, 2.0, 1.0, true, false) - 0.26424111765711533).abs() < 1e-10);
        // pgamma(5, shape=3, rate=1) ≈ 0.8753480
        assert!((pgamma(5.0, 3.0, 1.0, true, false) - 0.8753479805169189).abs() < 1e-10);
        // pgamma(0.5, shape=0.5, rate=1) ≈ 0.6826895
        assert!((pgamma(0.5, 0.5, 1.0, true, false) - 0.6826894921370859).abs() < 1e-9);
    }

    #[test]
    fn test_pgamma_both_branches() {
        // Ensure the series branch and CF branch produce mathematically
        // identical results at the exact crossover boundary (z = a + 1).
        let shape = 5.0;
        let z = 6.0; // Exactly shape + 1.0

        // Manually route the exact same input through both algorithms
        let p_series = lower_gamma_series(shape, z);
        let p_cf = 1.0 - upper_gamma_cf(shape, z);

        // They should match down to floating-point precision
        assert!(
            (p_series - p_cf).abs() < 1e-15,
            "Mismatch at boundary! Series: {}, CF: {}",
            p_series,
            p_cf
        );
    }

    #[test]
    fn test_pgamma_rate_scaling() {
        // pgamma(x, shape, rate) = pgamma(rate*x, shape, 1)
        let x = 2.0;
        let shape = 3.0;
        let rate = 0.5;
        let p1 = pgamma(x, shape, rate, true, false);
        let p2 = pgamma(rate * x, shape, 1.0, true, false);
        assert!((p1 - p2).abs() < 1e-12);
    }

    #[test]
    fn test_pgamma_edge_cases() {
        // P(0) = 0 for any valid shape and rate
        assert_eq!(pgamma(0.0, 2.0, 1.0, true, false), 0.0);
        // Negative x → 0
        assert_eq!(pgamma(-1.0, 2.0, 1.0, true, false), 0.0);
        // Infinity → 1
        assert_eq!(pgamma(f64::INFINITY, 2.0, 1.0, true, false), 1.0);
    }

    #[test]
    fn test_pgamma_monotonicity() {
        // CDF must be monotonically non-decreasing
        let shape = 2.5;
        let rate = 1.0;
        let xs = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0];
        let ps: Vec<f64> = xs
            .iter()
            .map(|&x| pgamma(x, shape, rate, true, false))
            .collect();
        for i in 1..ps.len() {
            assert!(
                ps[i] >= ps[i - 1],
                "Non-monotonic: P({}) = {} < P({}) = {}",
                xs[i],
                ps[i],
                xs[i - 1],
                ps[i - 1]
            );
        }
    }

    #[test]
    fn test_pgamma_bounds() {
        // CDF must be in [0, 1]
        let shapes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        let xs = [0.01, 0.1, 1.0, 5.0, 20.0, 100.0];
        for &a in &shapes {
            for &x in &xs {
                let p = pgamma(x, a, 1.0, true, false);
                assert!(
                    (0.0..=1.0).contains(&p),
                    "Out of bounds: pgamma({}, {}, 1) = {}",
                    x,
                    a,
                    p
                );
            }
        }
    }

    #[test]
    fn test_pgamma_large_shape() {
        // For large shape, gamma CDF is well-approximated near the mean
        // Mean = shape/rate. pgamma(mean, shape, rate) should be near 0.5
        let shape = 100.0;
        let rate = 1.0;
        let p_at_mean = pgamma(shape / rate, shape, rate, true, false);
        assert!(
            (p_at_mean - 0.5).abs() < 0.05,
            "pgamma(100, 100, 1) = {}, expected near 0.5",
            p_at_mean
        );
    }

    #[test]
    fn test_pgamma_small_shape() {
        // Small shape (< 1): density is concentrated near zero
        // pgamma(0.01, 0.1, 1) should be substantial
        let p = pgamma(0.01, 0.1, 1.0, true, false);
        assert!(p > 0.1, "Small shape: pgamma(0.01, 0.1, 1) = {}", p);
    }

    #[test]
    fn test_pgamma_pdf_cdf_consistency() {
        // Approximate derivative of CDF should equal PDF
        let shape = 3.0;
        let rate = 1.0;
        let x = 2.0;
        let h = 1e-7;
        let numerical_pdf = (pgamma(x + h, shape, rate, true, false)
            - pgamma(x - h, shape, rate, true, false))
            / (2.0 * h);
        let analytical_pdf = dgamma(x, shape, rate, false);
        assert!(
            (numerical_pdf - analytical_pdf).abs() < 1e-5,
            "CDF derivative {} vs PDF {}",
            numerical_pdf,
            analytical_pdf
        );
    }
}

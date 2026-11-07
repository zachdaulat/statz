//! Special mathematical functions and numerical stability primitives.
//!
//! Implements the Boost.Math Lanczos `lgamma` approximation ($\ln\Gamma(z)$),
//! regularized incomplete gamma functions ($P(a, z)$ series and $Q(a, z)$ continued
//! fraction), and the `log_sum_exp` trick.

#![allow(clippy::excessive_precision)]

use crate::consts::*;
use extendr_api::prelude::*;
use std::f64::consts::PI;

extendr_module! {
    mod special;
    fn lgamma_r;
}

// ============================================================
// Documentation for lgamma (Boost adaptation)
// ============================================================

/// Compute the natural logarithm of the gamma function: ln Γ(z).
///
/// @description
/// Uses a simplified adaptation of the Boost.Math C++ library's Lanczos
/// approximation (lanczos13m53 parameter set, N=13, G≈6.0247),
/// optimised for IEEE 754 double-precision (f64) arithmetic. The coefficients
/// are from the `lanczos_sum_expG_scaled` variant, which absorbs both the
/// √(2π) constant and the e^G scaling factor into the rational polynomial,
/// eliminating two sources of rounding error.
///
/// The Lanczos sum is evaluated as a ratio of two degree-12 polynomials
/// P(z)/Q(z) via Horner's method with fused multiply-add (FMA) instructions,
/// avoiding the catastrophic cancellation that can occur with the traditional
/// alternating-sign summation formulation.
///
/// Maximum approximation error: ~1.2 × 10^17 (near full f64 precision),
/// except for inputs infinitesimally close to 1 and 2.
///
/// Simplifications relative to the full Boost implementation:
/// - Omits the special Taylor series handling for z near 1 and 2
///   (costs ~1-2 ULPs in those neighbourhoods)
/// - A precomputed `LN_FACTORIALS` array storing integer inputs from ln(0!) to ln(15!)
///
/// @references
/// - Boost.Math library: <https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/lanczos.html>
/// - Pugh, G.R. (2004). "An Analysis of the Lanczos Gamma Approximation."
///   PhD thesis, University of British Columbia.
/// - Lanczos, C. (1964). "A Precision Approximation of the Gamma Function."
///   SIAM Journal on Numerical Analysis, 1(1), 86-96.
///
/// @param z A numeric (double) vector of positive values.
/// @return A numeric vector containing the natural logarithm of the gamma function.
/// @export
#[extendr(r_name = "lgamma")]
pub fn lgamma_r(z: &[f64]) -> Doubles {
    z.iter().map(|&zi| lgamma(zi)).collect::<Doubles>()
}

pub fn lgamma(z: f64) -> f64 {
    // Checking if z is a small whole number to return precomputed values
    if z > 0.0 && z <= 16.0 && z.fract() == 0.0 {
        return LN_FACTORIALS[(z - 1.0) as usize];
    }

    // Applying reflection formula when z < 0.5
    if z < 0.5 {
        let lgam: f64 = LN_PI - (PI * z).sin().abs().ln() - lgamma(1.0 - z);
        return lgam;
    }

    const G: f64 = 6.024680040776729583740234375;

    // Numerator Coefficients (P)
    const P: [f64; 13] = [
        56906521.91347156388090791033559122686859,
        103794043.1163445451906271053616070238554,
        86363131.28813859145546927288977868422342,
        43338889.32467613834773723740590533316085,
        14605578.08768506808414169982791359218571,
        3481712.15498064590882071018964774556468,
        601859.6171681098786670226533699352302507,
        75999.29304014542649875303443598909137092,
        6955.999602515376140356310115515198987526,
        449.9445569063168119446858607650988409623,
        19.51992788247617482847860966235652136208,
        0.5098416655656676188125178644804694509993,
        0.006061842346248906525783753964555936883222,
    ];

    // Denominator Coefficients (Q)
    const Q: [f64; 13] = [
        0.0,
        39916800.0,
        120543840.0,
        150917976.0,
        105258076.0,
        45995730.0,
        13339535.0,
        2637558.0,
        357423.0,
        32670.0,
        1925.0,
        66.0,
        1.0,
    ];

    // Core terms in Boost formulation
    let t: f64 = z - 0.5;
    let mut lgam: f64 = t * (t + G).ln() - t;

    // Only evaluating the Lanczos sum (Boost authors' rational polynomial)
    // if the size of `lgam` doesn't just truncate the precision anyway
    if lgam * f64::EPSILON < 20.0 {
        let numer = P.iter().rev().fold(0.0, |acc: f64, &x| acc.mul_add(z, x));
        let denom = Q.iter().rev().fold(0.0, |acc: f64, &x| acc.mul_add(z, x));

        lgam += (numer / denom).ln();
    }

    lgam
}

/// Compute the regularised lower incomplete gamma function P(a, z)
/// using a Taylor series expansion.
///
/// P(a, z) = (z^a · e^(-z)) / (a · Γ(a)) · Σ_{k=0}^∞ z^k / [(a+1)(a+2)···(a+k)]
///
/// Converges rapidly when z < a + 1 (the lower integral is the smaller quantity).
/// Each successive term is the previous term multiplied by z / (a + k), so the
/// ratio is less than 1 in this domain.
///
/// # Arguments
/// * `a` - Shape parameter (a > 0)
/// * `z` - Evaluation point (z > 0, typically z < a + 1)
pub(crate) fn lower_gamma_series(a: f64, z: f64) -> f64 {
    let ln_prefix: f64 = (a * z.ln()) - z - lgamma(a);

    // Initialise variables
    let mut k: f64 = 1.0;
    let mut curr: f64 = 1.0;
    let mut term: f64 = 1.0;

    // Computing the Taylor series
    while term > curr * f64::EPSILON {
        term *= z / (a + k);
        curr += term;
        k += 1.0;
    }

    (ln_prefix.exp() * curr) / a
}

/// Compute the regularised upper incomplete gamma function Q(a, z)
/// using Legendre's continued fraction representation, evaluated via
/// the modified Lentz algorithm.
///
/// Q(a, z) = 1 - P(a, z) = (z^a · e^(-z)) / Γ(a) · 1/CF
///
/// where CF is the continued fraction with partial numerators
/// a_k = k(a - k) and partial denominators b_k = z - a + 2k + 1.
///
/// Converges rapidly when z >= a + 1 (the upper integral is the smaller quantity).
///
/// # Arguments
/// * `a` - Shape parameter (a > 0)
/// * `z` - Evaluation point (z > 0, typically z >= a + 1)
#[allow(non_snake_case)]
#[allow(unused_assignments)]
pub(crate) fn upper_gamma_cf(a: f64, z: f64) -> f64 {
    let ln_prefix: f64 = (a * z.ln()) - z - lgamma(a);

    // TINY constant for textbook Modified Lentz shock-absorption
    const TINY: f64 = 1e-30;

    // Initialise variables
    // b_0 is guaranteed to be >= 2.0 due to domain split in z_pgamma_rs()
    let mut k: f64 = 0.0;
    let mut a_k: f64 = 0.0;
    let mut b_k: f64 = z - a + 1.0;
    let mut f: f64 = b_k;

    let mut C: f64 = f;
    let mut D: f64 = 0.0;
    let mut del: f64 = 2.0;

    while (del - 1.0).abs() > f64::EPSILON {
        // Update Legendre's coefficients starting at iteration 1
        k += 1.0;
        a_k = k * (a - k);
        b_k += 2.0;

        // Lentz updates
        D = b_k + (a_k * D);
        if D.abs() < TINY {
            D = TINY;
        }

        C = b_k + (a_k / C);
        if C.abs() < TINY {
            C = TINY;
        }

        D = D.recip();
        del = C * D;
        f *= del;
    }

    ln_prefix.exp() / f
}

// Helper function for evaluating a series sum with high dynamic range
pub(crate) fn log_sum_exp(l_terms: &[f64], log: bool) -> f64 {
    // Empty slice safety check
    if l_terms.is_empty() {
        return if log { f64::NEG_INFINITY } else { 0.0 };
    }
    // Finding max element
    // Must copy because reduce and max require owned data
    // .unwrap() is safe because of .is_empty() check above
    let l_max: f64 = l_terms.iter().copied().reduce(f64::max).unwrap();
    // Subtract max l from each log term l_k
    let l_sum: f64 = l_terms.iter().map(|l_k| (l_k - l_max).exp()).sum::<f64>();

    if log {
        l_max + l_sum.ln()
    } else {
        l_max.exp() * l_sum
    }
}

// Rust-side unit tests
#[allow(clippy::excessive_precision)]
#[allow(clippy::manual_range_contains)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::arch_distributions::arch_lgamma_godfrey;

    // --- lgamma tests (Boost adaptation) ---

    #[test]
    fn test_lgamma_small_integers() {
        // ln Γ(n) = ln((n-1)!) for positive integers
        // Γ(1) = 0! = 1, ln(1) = 0
        assert!((lgamma(1.0) - 0.0).abs() < 1e-15);
        // Γ(2) = 1! = 1, ln(1) = 0
        assert!((lgamma(2.0) - 0.0).abs() < 1e-15);
        // Γ(3) = 2! = 2, ln(2) ≈ 0.6931471805599453
        assert!((lgamma(3.0) - 2.0_f64.ln()).abs() < 1e-15);
        // Γ(5) = 4! = 24, ln(24) ≈ 3.178053830347946
        assert!((lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-15);
        // Γ(11) = 10! = 3628800
        assert!((lgamma(11.0) - 3628800.0_f64.ln()).abs() < 1e-15);
    }

    #[test]
    fn test_lgamma_half_integers() {
        // Γ(0.5) = √π, so ln Γ(0.5) = 0.5 * ln(π) ≈ 0.5723649429247001
        let expected = 0.5 * PI.ln();
        assert!((lgamma(0.5) - expected).abs() < 1e-15);
        // Γ(1.5) = 0.5 * √π
        let expected_1_5 = (0.5 * PI.sqrt()).ln();
        assert!((lgamma(1.5) - expected_1_5).abs() < 1e-15);
    }

    // Expected values generated via Wolfram Alpha: N[Log[Gamma[z]], 25]
    // Tolerances set to 1e-15 and 1e-14 to account for standard f64
    // Unit in the Last Place (ULP) accumulation.
    #[test]
    fn test_lgamma_known_values() {
        // N[Log[Gamma[1/10]], 25]
        assert!((lgamma(0.1) - 2.252712651734205959869702).abs() < 1e-15);
        // N[Log[Gamma[25/10]], 25]
        assert!((lgamma(2.5) - 0.2846828704729191596324947).abs() < 1e-15);
        // N[Log[Gamma[103/10]], 25]
        assert!((lgamma(10.3) - 13.48203678613835697061507).abs() < 1e-14);
        // N[Log[Gamma[100]], 25]
        assert!((lgamma(100.0) - 359.1342053695753987760440).abs() < 1e-14);
    }

    // Expected values generated via Wolfram Alpha: N[Log[Gamma[z]], 25]
    // Tolerances set to 1e-15 and 1e-14 to account for standard f64
    // Unit in the Last Place (ULP) accumulation.
    #[test]
    fn test_lgamma_large_z() {
        // N[Log[Gamma[1000]], 25]
        assert!((lgamma(1000.0) - 5905.220423209181211826077).abs() < 1e-11);
        // N[Log[Gamma[10000]], 25]
        assert!((lgamma(10000.0) - 82099.71749644237727264896).abs() < 1e-11);
    }

    #[test]
    fn test_lgamma_near_one_and_two() {
        // Expected values generated via Wolfram Alpha: N[Log[Gamma[z]], 25]
        // Tolerances set to 1e-7 to account for precision loss from catastrophic
        // cancellation near the roots, as the dedicated Taylor series expansions
        // were bypassed for general performance and to simply the adaptation.

        let expected_1 = -0.00005772635749294248232231362;
        assert!((lgamma(1.0001) - expected_1).abs() < 1e-7);

        let expected_2 = -0.00004227364250705751767768638;
        assert!((lgamma(1.9999) - expected_2).abs() < 1e-7);

        // Verify sign: lgamma is negative on (1, 2) and positive outside
        assert!(lgamma(1.5) < 0.0);
        assert!(lgamma(0.5) > 0.0);
        assert!(lgamma(3.0) > 0.0);
    }

    #[test]
    fn test_lgamma_reflection_formula() {
        // For z < 0.5, verify via reflection: Γ(z)Γ(1-z) = π / sin(πz)
        // So lgamma(z) + lgamma(1-z) = ln(π) - ln|sin(πz)|
        let z = 0.25;
        let lhs = lgamma(z) + lgamma(1.0 - z);
        let rhs = PI.ln() - (PI * z).sin().ln();
        assert!((lhs - rhs).abs() < 1e-15);
    }

    #[test]
    fn test_lgamma_smoothness() {
        // A function is smooth if the numerical derivative doesn't jump wildly.
        // We check the transition across z=3.0 (where the integer lookup table lives)
        let eps = 1e-7;
        let z = 3.0;

        // Evaluate just below, exactly at, and just above the integer
        let val_below = lgamma(z - eps);
        let val_exact = lgamma(z);
        let val_above = lgamma(z + eps);

        // The rate of change (slope) approaching from the left should match the right
        let slope_left = (val_exact - val_below) / eps;
        let slope_right = (val_above - val_exact) / eps;

        // If the difference in slopes is extremely small, the function is continuous and smooth
        assert!((slope_left - slope_right).abs() < 1e-5);
    }

    #[test]
    fn test_lgamma_extreme_tails() {
        // Extremely small positive number (approaching the singularity)
        let tiny = 1e-100;
        assert!(lgamma(tiny) > 0.0);
        assert!(!lgamma(tiny).is_nan());

        // Extremely large number (approaching f64 overflow)
        // f64 max is ~1.7e308, lgamma overflows before that, but 1e100 is safe
        let massive = 1e100;
        assert!(lgamma(massive) > 0.0);
        assert!(!lgamma(massive).is_infinite());
    }

    // --- lgamma_godfrey tests ---

    #[test]
    fn test_lgamma_godfrey_matches_boost() {
        // Both implementations should agree to high precision
        let test_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 50.0, 100.0];
        for z in test_values {
            let boost = lgamma(z);
            let godfrey = arch_lgamma_godfrey(z);
            assert!(
                (boost - godfrey).abs() < 1e-12,
                "Mismatch at z = {}: boost = {}, godfrey = {}",
                z,
                boost,
                godfrey
            );
        }
    }

    #[test]
    fn test_lgamma_godfrey_integers() {
        assert!((arch_lgamma_godfrey(1.0) - 0.0).abs() < 1e-14);
        assert!((arch_lgamma_godfrey(5.0) - 24.0_f64.ln()).abs() < 1e-13);
    }

    // --- Log-Sum-Exp Tests ---

    #[test]
    fn test_log_sum_exp_basic() {
        let terms = vec![0.0, 0.0];
        // ln(exp(0) + exp(0)) = ln(2)
        assert!((log_sum_exp(&terms, true) - std::f64::consts::LN_2).abs() < 1e-15);
    }

    #[test]
    fn test_log_sum_exp_empty() {
        // Empty slice should cleanly return -Inf in log space
        let empty: Vec<f64> = vec![];
        assert_eq!(log_sum_exp(&empty, true), f64::NEG_INFINITY);
    }

    #[test]
    fn test_log_sum_exp_underflow_prevention() {
        // If we didn't factor out the max, exp(-1000) would instantly underflow to 0.0,
        // and the log of the sum would be -Inf.
        // Factoring out the max allows this to calculate correctly.
        let terms = vec![-1000.0, -1000.0];
        // ln(exp(-1000) + exp(-1000)) = ln(2 * exp(-1000)) = ln(2) - 1000
        let expected = std::f64::consts::LN_2 - 1000.0;
        assert!((log_sum_exp(&terms, true) - expected).abs() < 1e-14);
    }
}

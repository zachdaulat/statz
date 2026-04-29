use core::f64;
use extendr_api::prelude::*;
use statrs::consts::{LN_PI, LN_SQRT_2PI};
use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

extendr_module! {
    mod distributions;
    fn z_dnorm_rs;
    fn z_pnorm_std;
    fn z_pnorm_as;
    fn z_dpois_rs;
    fn z_ppois_di;
    fn z_ppois_rec;
    fn z_lgamma;
    fn z_dgamma_rs;
    fn z_pgamma_rs;
    fn z_dtweedie_rs;
    fn z_ptweedie_rs;
    fn z_dinvgauss_rs;
    fn z_pinvgauss_rs;
}

// Planned functions:
// - dnorm, pnorm         (normal distribution)
// - dpois, ppois         (Poisson distribution)
// - dgamma, pgamma       (gamma distribution)
// - dtweedie, ptweedie   (Tweedie/compound Poisson-gamma)
// - dinvgauss, -invgauss (inverse gaussian distribution)

/// Compute the normal probability density function (PDF)
/// @param x A single numeric value at which to evaluate the density
/// @param mean The mean of the normal distribution (μ)
/// @param sd The standard deviation of the normal distribution (σ > 0)
/// @return The probability density f(x | μ, σ)
/// @export
#[extendr]
pub fn z_dnorm_rs(x: f64, mean: f64, sd: f64, log: bool) -> f64 {
    let z = (x - mean) / sd;
    let pdf = -sd.ln() - LN_SQRT_2PI - (0.5 * z * z);

    // Evaluating boolean for returning in log or regular space
    if log {
        pdf
    } else {
        pdf.exp()
    }
}

/// Compute the standard normal cumulative distribution function (CDF)
/// using the `libm` crate error function implementations for full f64
/// machine precision.
/// @param z A z-score (standardised value)
/// @return Cumulative probability Φ(z) = P(Z ≤ z) for Z ~ N(0,1)
/// @export
#[extendr]
pub fn z_pnorm_std(z: f64, lower_tail: bool, log_p: bool) -> f64 {
    // 0: Edge cases for fast paths handled implicity by libm::erfc()

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

/// Compute the standard normal cumulative distribution function (CDF)
/// using the Abramowitz and Stegun (1972, 10th ed.) equation 7.1.26
/// error function approximation.
/// Deprecated in favour of the libm::erfc() implementation but kept
/// as an initial pedagogical version.
/// Maximum absolute error: |ε| < 1.5 × 10⁻⁷
/// @param z A z-score (standardised value)
/// @return Cumulative probability Φ(z) = P(Z ≤ z) for Z ~ N(0,1)
/// @export
#[extendr]
#[allow(non_upper_case_globals)]
pub fn z_pnorm_as(z: f64, lower_tail: bool, log_p: bool) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == f64::INFINITY {
        return if log_p { 0.0 } else { 1.0 };
    }
    if z == f64::NEG_INFINITY {
        return if log_p { f64::NEG_INFINITY } else { 0.0 };
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

    // Calulated in regular space
    let mut cdf: f64 = (1.0 + erf) / 2.0;

    // Evaluating tail boolean
    cdf = if lower_tail { cdf } else { 1.0 - cdf };

    // Evaluating log or regular space boolean
    if log_p {
        cdf.ln()
    } else {
        cdf
    }
}

/// Compute the Poisson probability mass function P(X = x)
/// using log-space arithmetic to avoid factorial overflow.
/// @param x A non-negative integer count
/// @param lambda The rate parameter (λ > 0)
/// @return The probability mass P(X = x | λ)
/// @export
#[extendr]
pub fn z_dpois_rs(x: i32, lambda: f64, log: bool) -> f64 {
    let mut p = x as f64 * lambda.ln() - lambda;

    for i in 1..=x {
        p -= f64::from(i).ln();
    }

    // Evaluating boolean for returning in log or regular space
    if log {
        p
    } else {
        p.exp()
    }
}

/// Compute the Poisson cumulative distribution function P(X ≤ x)
/// using log-space PMF evaluation for each term.
/// @param x A non-negative integer count
/// @param lambda The rate parameter (λ > 0)
/// @return The cumulative probability P(X ≤ x | λ)
/// @export
#[extendr]
// Named di for reference to the dpois function and this iterator approach
pub fn z_ppois_di(x: i32, lambda: f64, log_p: bool) -> f64 {
    // Calculated in regular space
    let cdf: f64 = (0..=x).map(|i| z_dpois_rs(i, lambda, false)).sum::<f64>();

    // Evaluating boolean for returning in log or regular space
    if log_p {
        cdf.ln()
    } else {
        cdf
    }
}

/// Compute the Poisson cumulative distribution function P(X ≤ x)
/// using a recurrence relation: P(X = k) = P(X = k-1) · λ/k
/// @param x A non-negative integer count
/// @param lambda The rate parameter (λ > 0)
/// @return The cumulative probability P(X ≤ x | λ)
/// @export
#[extendr]
// Named rec for reference to exploiting its recurrence relation
pub fn z_ppois_rec(x: i32, lambda: f64, lower_tail: bool, log_p: bool) -> f64 {
    let mut p = (-lambda).exp(); // P(X = 0)
    let mut cdf = p;

    for k in 1..=x {
        p *= lambda / k as f64;
        cdf += p;
    }

    // Evaluating tail boolean
    cdf = if lower_tail { cdf } else { 1.0 - cdf };

    // Evaluating log or regular space boolean
    if log_p {
        cdf.ln()
    } else {
        cdf
    }
}

// ============================================================
// Documentation for z_dgamma_rs
// ============================================================

/// Compute the gamma distribution probability density function.
///
/// Evaluates the PDF of the Gamma(shape, rate) distribution at x using
/// log-space arithmetic to avoid overflow:
///
///   f(x | α, β) = β^α / Γ(α) · x^(α-1) · e^(-βx)
///
/// Computed as:
///   ln f = α·ln(β) - ln Γ(α) + (α-1)·ln(x) - β·x
///
/// The log-gamma term is evaluated via `z_lgamma()` (Boost adaptation).
///
/// # Returns
/// The probability density f(x | α, β), or its natural log if `log = true`.
/// @param x A positive numeric value
/// @param shape The shape parameter (α > 0)
/// @param rate The rate parameter (β > 0)
/// @param log Logical; if TRUE, return the log-density
/// @return The gamma PDF value at x, or ln(PDF) if log = TRUE
/// @export
#[extendr]
pub fn z_dgamma_rs(x: f64, shape: f64, rate: f64, log: bool) -> f64 {
    let lnum: f64 = shape * rate.ln();
    let lgam: f64 = z_lgamma(shape);
    let lpow: f64 = (shape - 1.0) * x.ln();

    let pdf: f64 = lnum - lgam + lpow - (rate * x);

    // Evaluating boolean for returning in log or regular space
    if log {
        pdf
    } else {
        pdf.exp()
    }
}

// ============================================================
// Documentation for z_pgamma_rs
// ============================================================

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
/// @export
#[extendr]
pub fn z_pgamma_rs(x: f64, shape: f64, rate: f64, lower_tail: bool, log_p: bool) -> f64 {
    // Checking for R's specific NA_real_ first
    if x.is_na() || shape.is_na() || rate.is_na() {
        return f64::na(); // extendr bridges this back as NA_real_
    }

    if x.is_nan() || shape.is_nan() || rate.is_nan() {
        return f64::NAN; // extendr bridges this back as NaN
    }

    // Domain boundaries (Note: adjusted for log_p)
    if x <= 0.0 {
        return if log_p { f64::NEG_INFINITY } else { 0.0 };
    }
    if x.is_infinite() {
        return if log_p { 0.0 } else { 1.0 };
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

// ============================================================
// Documentation for z_dtweedie_rs
// ============================================================

/// Computing the Tweedie PDF (compound Poisson-Gamma)
/// Internal Rust engine for the Tweedie density.
/// Assumes all inputs (y, mu, phi, power) have been pre-validated by the R wrapper.
/// - `y < 0` is not handled here (should be blocked by R).
/// - Automatically routes exact zeros to point-mass fast paths.
/// - Uses the Dunn & Smyth (2005) series expansion for y > 0.
#[extendr]
pub fn z_dtweedie_rs(y: f64, mu: f64, phi: f64, power: f64, log: bool) -> f64 {
    // Fast path when mu = 0
    if mu == 0.0 {
        return if y == 0.0 {
            // When mu = 0 and y = 0
            if log {
                0.0
            } else {
                1.0
            }
        } else {
            // When mu = 0 and y > 0
            if log {
                f64::NEG_INFINITY
            } else {
                0.0
            }
        };
    }

    // Organize parameter sets
    let tw_params = TwParams { mu, phi, power };
    let PgParams {
        lambda,
        shape,
        rate,
    } = tw_params.to_pg();

    // Fast path when y = 0 but mu > 0
    // Just the Poisson mass at 0
    if y == 0.0 && mu > 0.0 {
        return if log { -lambda } else { (-lambda).exp() };
    }

    dtweedie_series(y, lambda, shape, rate, log)
}

// ============================================================
// Documentation for z_ptweedie_rs
// ============================================================

/// Computing the Tweedie CDF (compound Poisson-Gamma)
/// Internal Rust engine for the Tweedie cumulative distribution.
/// Assumes all inputs (y, mu, phi, power) have been pre-validated by the R wrapper.
/// - `y < 0` is not handled here (should be blocked by R).
/// - Automatically routes exact zeros to point-mass fast paths.
/// - Uses the Dunn & Smyth (2005) series expansion for y > 0.
/// - Supports `lower_tail` and `log_p` evaluation.
#[extendr]
pub fn z_ptweedie_rs(y: f64, mu: f64, phi: f64, power: f64, lower_tail: bool, log_p: bool) -> f64 {
    if mu == 0.0 {
        if lower_tail {
            return if log_p { 0.0 } else { 1.0 };
        } else {
            return if log_p { f64::NEG_INFINITY } else { 0.0 };
        }
    }

    // Organize parameter sets
    let tw_params = TwParams { mu, phi, power };
    let PgParams {
        lambda,
        shape,
        rate,
    } = tw_params.to_pg();

    // Direct space point mass at 0
    let p_zero: f64 = (-lambda).exp();

    // Fast path when y = 0 but mu > 0, condition unnecessary but used for clarity
    // Just the Poisson mass at 0
    if y == 0.0 && mu > 0.0 {
        if lower_tail {
            return if log_p { -lambda } else { p_zero };
        } else {
            // P(Y > 0) = 1 - e^(-lambda)
            // Use exp_m1() to avoid catastrophic cancellation when lambda is near 0.
            // exp_m1(-lambda) calculates e^(-lambda) - 1. We negate it to get 1 - e^(-lambda).
            let p_upper: f64 = -(-lambda).exp_m1();
            return if log_p { p_upper.ln() } else { p_upper };
        }
    }

    // Primary series evaluation for y > 0
    let p_tweedie: f64 = if lower_tail {
        ptweedie_series(y, lambda, shape, rate, lower_tail) + p_zero
    } else {
        ptweedie_series(y, lambda, shape, rate, lower_tail)
    };

    if log_p {
        p_tweedie.ln()
    } else {
        p_tweedie
    }
}

// ============================================================
// Documentation for z_dinvgauss_rs
// ============================================================

/// Computes the Inverse Gaussian probability density function (PDF).
/// Internal Rust engine for `statz`.
/// Assumes parameters (mu, lambda) are strictly positive.
/// - Returns 0.0 (or -Inf in log space) for y <= 0 or y -> Inf.
/// - Evaluates natively in log space to ensure maximum likelihood stability.
#[extendr]
pub fn z_dinvgauss_rs(y: f64, mu: f64, lambda: f64, log: bool) -> f64 {
    // 1: Fast paths
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

// ============================================================
// Documentation for z_pinvgauss_rs
// ============================================================

/// Computes the Inverse Gaussian cumulative distribution function (CDF).
/// Internal Rust engine for `statz`.
/// Assumes parameters (mu, lambda) are strictly positive.
/// - Utilizes `libm::erfc` via the standard normal CDF for full machine precision in extreme tails.
/// - Implements a structural overflow brake for extreme parameterisations (2λ/μ > 709).
#[extendr]
pub fn z_pinvgauss_rs(y: f64, mu: f64, lambda: f64, lower_tail: bool, log_p: bool) -> f64 {
    // 0: Fast paths for domain boundaries and safety
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
    let pnorm_1: f64 = z_pnorm_std(z_1, lower_tail, false);
    let l_exp: f64 = 2.0 * lambda / mu;

    // Break for extreme parameterisations when correction term is
    // negligible exploiting pnorm decreasing faster than exp term
    if l_exp > f64::MAX.ln() {
        return if log_p { pnorm_1.ln() } else { pnorm_1 };
    }

    let pnorm_2: f64 = z_pnorm_std(z_2, true, false);
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

// -------------------- Convenience Functions -------------------

// ============================================================
// Documentation for z_lgamma (Boost adaptation)
// ============================================================

/// Compute the natural logarithm of the gamma function, ln Γ(z).
///
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
/// Maximum approximation error: ~1.2 × 10⁻¹⁷ (near full f64 precision),
/// except for inputs infinitesimally close to 1 and 2.
///
/// Simplifications relative to the full Boost implementation:
/// - Omits the special Taylor series handling for z near 1 and 2
///   (costs ~1-2 ULPs in those neighbourhoods, negligible for statistical use)
/// - Omits the log(tgamma(z)) path for 3 ≤ z < 100
///
/// # Arguments
/// * `z` - A positive real number (z > 0), or z < 0.5 (handled via reflection)
///
/// # Returns
/// The value of ln Γ(z) as an f64.
///
/// # References
/// - Boost.Math library: <https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/lanczos.html>
/// - Pugh, G.R. (2004). "An Analysis of the Lanczos Gamma Approximation."
///   PhD thesis, University of British Columbia.
/// - Lanczos, C. (1964). "A Precision Approximation of the Gamma Function."
///   SIAM Journal on Numerical Analysis, 1(1), 86-96.
/// @param z A positive numeric value
/// @return The natural logarithm of the gamma function at z
/// @export
#[allow(clippy::excessive_precision)]
#[extendr]
pub fn z_lgamma(z: f64) -> f64 {
    // Checking if z is a small whole number to return precomputed values
    if z > 0.0 && z <= 16.0 && z.fract() == 0.0 {
        return LN_FACTORIALS[(z - 1.0) as usize];
    }

    // Applying reflection formula when z < 0.5
    if z < 0.5 {
        let lgam: f64 = LN_PI - (PI * z).sin().abs().ln() - z_lgamma(1.0 - z);
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
    let ln_prefix: f64 = (a * z.ln()) - z - z_lgamma(a);

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
    let ln_prefix: f64 = (a * z.ln()) - z - z_lgamma(a);

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

// ------------- Tweedie Convenience Structs, Methods, & Functions -------------

// Tweedie parameters struct
#[derive(Clone, Copy, Debug)]
pub(crate) struct TwParams {
    pub mu: f64,
    pub phi: f64,
    pub power: f64,
}

#[allow(unused)]
impl TwParams {
    // Smart Constructor might be unnecessary with R-level input validation
    // I can consider removing, or at least ignoring for now
    // Smart constructor ensuring valid Tweedie parameters
    pub fn new(mu: f64, phi: f64, power: f64) -> std::result::Result<Self, &'static str> {
        if mu <= 0.0 {
            return Err("mu (mean) must be greater than 0");
        }
        if phi <= 0.0 {
            return Err("phi (dispersion) must be greater than 0");
        }
        if power <= 1.0 || power >= 2.0 {
            return Err("power parameter 'p' must be in the open interval (1, 2)");
        }

        Ok(TwParams { mu, phi, power })
    }

    // Tweedie to Poisson-Gamma parameter conversion
    pub fn to_pg(self) -> PgParams {
        // Extracting Tweedie parameters
        let TwParams { mu, phi, power } = self;

        let lambda = mu.powf(2.0 - power) / (phi * (2.0 - power));
        let shape = (2.0 - power) / (power - 1.0);
        let rate = mu.powf(1.0 - power) / (phi * (power - 1.0));

        PgParams {
            lambda,
            shape,
            rate,
        }
    }
}

// Poisson-Gamma parameters struct
#[derive(Clone, Copy, Debug)]
pub(crate) struct PgParams {
    pub lambda: f64,
    pub shape: f64,
    pub rate: f64,
}

#[allow(unused)]
impl PgParams {
    // Smart Constructor might be unnecessary with R-level input validation
    // I can consider removing, or at least ignoring for now
    // Smart constructor ensuring valid Poisson-Gamma parameters
    pub fn new(lambda: f64, shape: f64, rate: f64) -> std::result::Result<Self, &'static str> {
        if lambda <= 0.0 {
            return Err("lambda must be > 0");
        }
        if shape <= 0.0 {
            return Err("shape must be > 0");
        }
        if rate <= 0.0 {
            return Err("rate must be > 0");
        }

        Ok(PgParams {
            lambda,
            shape,
            rate,
        })
    }

    // Poisson-Gamma to Tweedie parameter conversion
    pub fn to_tw(self) -> TwParams {
        // Extracting Poisson-Gamma parameters
        let PgParams {
            lambda,
            shape,
            rate,
        } = self;

        let mu = lambda * shape / rate;
        let power = (shape + 2.0) / (shape + 1.0);
        let phi = (lambda.powf(1.0 - power) * (shape / rate).powf(2.0 - power)) / (2.0 - power);

        TwParams { mu, phi, power }
    }
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

/// Dunn-Smyth Series helper function, Tweedie density for y > 0
/// Internal engine for the Tweedie density infinite series (y > 0).
/// Accumulates Poisson-weighted Gamma densities into a vector for log_sum_exp()
pub(crate) fn dtweedie_series(y: f64, lambda: f64, shape: f64, rate: f64, log: bool) -> f64 {
    // 1: Starting index and log-likelihood
    let k_0: f64 = lambda.floor().max(1.0);
    let l_0: f64 = z_dpois_rs(k_0 as i32, lambda, true) + z_dgamma_rs(y, k_0 * shape, rate, true);

    // Evaluating just the right side finding gradient direction
    let k_p1: f64 = k_0 + 1.0;
    let l_p1: f64 =
        z_dpois_rs(k_p1 as i32, lambda, true) + z_dgamma_rs(y, k_p1 * shape, rate, true);

    // 2: Gradient setting the step direction for hill-climb
    let step: f64 = if l_p1 > l_0 { 1.0 } else { -1.0 };
    let mut k_prev: f64 = if step > 0.0 { k_0 } else { k_p1 };
    let mut k_curr: f64 = if step > 0.0 { k_p1 } else { k_0 };
    let mut l_prev: f64 = if step > 0.0 { l_0 } else { l_p1 };
    let mut l_curr: f64 = if step > 0.0 { l_p1 } else { l_0 };

    // 3: Core maximum finding hill-climb loop
    // Max will be at k_prev when loop ends
    while l_curr > l_prev {
        k_prev = k_curr; // Storing the max
        l_prev = l_curr;

        // Update running index with step with safety break
        k_curr += step;
        if k_curr < 1.0 {
            break;
        }

        l_curr =
            z_dpois_rs(k_curr as i32, lambda, true) + z_dgamma_rs(y, k_curr * shape, rate, true);
    }
    let k_max = k_prev;
    let l_max = l_prev;

    // Defining term size limit
    let lim: f64 = l_max + f64::EPSILON.ln();

    // 4: Initialise vector and collect terms
    let mut l_terms: Vec<f64> = Vec::with_capacity(1024);
    l_terms.push(l_max);

    // --- Reverse loop ---
    let mut ki: f64 = k_max - 1.0;
    // Priming read for reverse loop
    let mut ld_pois: f64 = z_dpois_rs(ki as i32, lambda, true);
    let mut ld_gamma: f64 = z_dgamma_rs(y, ki * shape, rate, true);
    let mut li: f64 = ld_pois + ld_gamma;

    while li > lim {
        l_terms.push(li); // Accumulate immediately

        ki -= 1.0;
        if ki < 1.0 {
            break;
        } // Domain safety

        // Calculate next term for condition check
        let ld_pois: f64 = z_dpois_rs(ki as i32, lambda, true);
        let ld_gamma: f64 = z_dgamma_rs(y, ki * shape, rate, true);
        li = ld_pois + ld_gamma;
    }

    // --- Forward loop ---
    ki = k_max + 1.0;
    // Priming read for forward loop
    ld_pois = z_dpois_rs(ki as i32, lambda, true);
    ld_gamma = z_dgamma_rs(y, ki * shape, rate, true);
    li = ld_pois + ld_gamma;

    while li > lim {
        l_terms.push(li); // Accumulating immediately upon entering loop iteration
        ki += 1.0;

        // Calculate next term for condition check
        let ld_pois: f64 = z_dpois_rs(ki as i32, lambda, true);
        let ld_gamma: f64 = z_dgamma_rs(y, ki * shape, rate, true);
        li = ld_pois + ld_gamma;
    }

    // 5: Evaluate log-sum-exp
    log_sum_exp(&l_terms, log)
}

/// Dunn-Smyth Series helper function, Tweedie CDF for y > 0
/// Internal engine for the Tweedie CDF infinite series (y > 0).
/// Accumulates Poisson-weighted Gamma probabilities in direct space.
pub(crate) fn ptweedie_series(y: f64, lambda: f64, shape: f64, rate: f64, lower_tail: bool) -> f64 {
    // 1: Starting index and initialise sum
    let k0: f64 = lambda.floor().max(1.0);
    let lp_pois: f64 = z_dpois_rs(k0 as i32, lambda, true);
    let lp_gamma: f64 = z_pgamma_rs(y, k0 * shape, rate, lower_tail, true);
    let mut term: f64 = (lp_pois + lp_gamma).exp();
    let mut sum: f64 = 0.0;
    let mut ki: f64 = k0;

    // 2: Reverse direction loop
    while term > f64::EPSILON {
        // Accumulate new term
        sum += term;
        ki -= 1.0;

        // Safety check for index
        if ki < 1.0 {
            break;
        };
        let lp_pois: f64 = z_dpois_rs(ki as i32, lambda, true);
        let lp_gamma: f64 = z_pgamma_rs(y, ki * shape, rate, lower_tail, true);
        term = (lp_pois + lp_gamma).exp();
    }

    // 3: Reset indices to next index for forward loop
    ki = k0 + 1.0;
    let lp_pois: f64 = z_dpois_rs(ki as i32, lambda, true);
    let lp_gamma: f64 = z_pgamma_rs(y, ki * shape, rate, lower_tail, true);
    term = (lp_pois + lp_gamma).exp();

    // 4: Enter forward loop with next index
    while term > f64::EPSILON {
        // Accumulating new term
        sum += term;
        ki += 1.0;
        let lp_pois: f64 = z_dpois_rs(ki as i32, lambda, true);
        let lp_gamma: f64 = z_pgamma_rs(y, ki * shape, rate, lower_tail, true);
        term = (lp_pois + lp_gamma).exp();
    }

    sum
}

// ============================================================
// Documentation for z_lgamma_godfrey (pedagogical implementation)
// ============================================================

/// Compute ln Γ(z) using Godfrey's Lanczos coefficient set (g=7, N=9).
///
/// This is a pedagogical implementation of the traditional Lanczos approximation
/// using Paul Godfrey's well-known f64 coefficient set. Unlike the Boost
/// adaptation in `z_lgamma()`, this uses the standard formulation:
///
///   ln Γ(z) = ½ ln(2π) + (z - ½) ln(z + g - ½) - (z + g - ½) + ln S(z)
///
/// where S(z) = c₀ + Σ(k=1..8) cₖ/(z-1+k) is the Lanczos sum with
/// alternating-sign coefficients, and the formula is evaluated after
/// shifting z → z-1 to convert from Γ(z+1) to Γ(z).
///
/// This implementation is less precise than `z_lgamma()` due to potential
/// cancellation in the alternating-sign sum, but is included as a learning
/// exercise. It is not exported to R.
///
/// # Arguments
/// * `z` - A positive real number
///
/// # Returns
/// The value of ln Γ(z) as an f64.
///
/// # References
/// - Godfrey, P. "Lanczos Implementation of the Gamma Function."
///   <http://my.fit.edu/~gabdo/gamma.txt>
/// - <https://www.mrob.com/pub/ries/lanczos-gamma.html>
#[allow(unused)]
#[allow(clippy::excessive_precision)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn z_lgamma_godfrey(mut z: f64) -> f64 {
    // Checking if z is a small whole number to return precomputed values
    if z > 0.0 && z <= 16.0 && z.fract() == 0.0 {
        return LN_FACTORIALS[(z - 1.0) as usize];
    }

    // Godfrey's coefficient set, g = 7, N = 9
    const G: f64 = 7.0;
    const COEFFS: [f64; 9] = [
        0.99999999999980993227684700473478,
        676.520368121885098567009190444019,
        -1259.13921672240287047156078755283,
        771.3234287776530788486528258894,
        -176.61502916214059906584551354,
        12.507343278686904814458936853,
        -0.13857109526572011689554707,
        9.984369578019570859563e-6,
        1.50563273514931155834e-7,
    ];

    // Applying reflection formula when z < 0.5
    if z < 0.5 {
        let lgam: f64 = LN_PI - (PI * z).sin().abs().ln() - z_lgamma_godfrey(1.0 - z);
        return lgam;
    }

    z -= 1.0;

    let mut s: f64 = COEFFS[0];

    for i in 1..9 {
        s += COEFFS[i] / (z + i as f64);
    }

    let t = z + G + 0.5;

    let lgam: f64 = LN_SQRT_2PI + ((z + 0.5) * t.ln()) - t + s.ln();

    lgam
}

// ============================================================
// Documentation for LN_FACTORIALS lookup table
// ============================================================

/// Precomputed values of ln(n!) for n = 0, 1, ..., 15.
///
/// Used by `z_lgamma()` and `z_lgamma_godfrey()` to short-circuit evaluation
/// when the input is a small positive integer. Since Γ(n) = (n-1)! for
/// positive integers, ln Γ(n) = ln((n-1)!), so `LN_FACTORIALS[n-1]` gives
/// the correct result for z = n.
///
/// Index mapping: LN_FACTORIALS[k] = ln(k!) for k = 0..15.
///   - LN_FACTORIALS[0] = ln(0!) = 0.0
///   - LN_FACTORIALS[1] = ln(1!) = 0.0
///   - LN_FACTORIALS[2] = ln(2!) ≈ 0.6931
///   - ...
///   - LN_FACTORIALS[15] = ln(15!) ≈ 27.8993
#[allow(clippy::approx_constant)]
#[allow(clippy::excessive_precision)]
pub(crate) const LN_FACTORIALS: [f64; 16] = [
    0.0,
    0.0,
    0.6931471805599453094172321,
    1.791759469228055000812477,
    3.178053830347945619646942,
    4.787491742782045994247701,
    6.579251212010100995060178,
    8.525161361065414300165531,
    10.60460290274525022841723,
    12.80182748008146961120772,
    15.10441257307551529522571,
    17.50230784587388583928765,
    19.98721449566188614951736,
    22.55216385312342288557085,
    25.19122118273868150009343,
    27.89927138384089156608944,
];

// ------------------ TESTS -----------------
// Rust-side unit tests
#[allow(clippy::excessive_precision)]
#[allow(clippy::manual_range_contains)]
#[cfg(test)]
mod tests {
    use super::*;

    // --- z_dnorm tests ---

    #[test]
    fn test_dnorm_standard_at_zero() {
        // dnorm(0, 0, 1) = 1/sqrt(2*pi) ≈ 0.3989422804
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((z_dnorm_rs(0.0, 0.0, 1.0, false) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dnorm_nonstandard() {
        // dnorm(10, 10, 3) should equal dnorm(0, 0, 3)
        assert!(
            (z_dnorm_rs(10.0, 10.0, 3.0, false) - z_dnorm_rs(0.0, 0.0, 3.0, false)).abs() < 1e-10
        );
    }

    // --- z_pnorm_std tests ---

    #[test]
    fn test_pnorm_at_zero() {
        // Φ(0) = 0.5 exactly. Machine precision.
        assert!((z_pnorm_std(0.0, true, false) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_pnorm_symmetry() {
        // Φ(-z) = 1 - Φ(z)
        let z = 1.5;
        assert!((z_pnorm_std(-z, true, false) - (1.0 - z_pnorm_std(z, true, false))).abs() < 1e-15);
    }

    #[test]
    fn test_pnorm_known_values() {
        // Reference values from WolframAlpha: ex. N[CDF[NormalDistribution[0, 1], 1.96], 25]
        // Φ(1) ≈ 0.8413447460685429
        assert!((z_pnorm_std(1.0, true, false) - 0.8413447460685429485852325).abs() < 1e-15);
        // Φ(-1) ≈ 0.15865525393145705
        assert!((z_pnorm_std(-1.0, true, false) - 0.1586552539314570514147675).abs() < 1e-15);
        // Φ(1.96) ≈ 0.9750021048517795
        assert!((z_pnorm_std(1.96, true, false) - 0.9750021048517795658634157).abs() < 1e-15);
        // Φ(3) ≈ 0.9986501019683699
        assert!((z_pnorm_std(3.0, true, false) - 0.9986501019683699054733482).abs() < 1e-15);
    }

    #[test]
    fn test_pnorm_edge_cases() {
        assert_eq!(z_pnorm_std(f64::INFINITY, true, false), 1.0);
        assert_eq!(z_pnorm_std(f64::NEG_INFINITY, true, false), 0.0);
        assert!(z_pnorm_std(f64::NAN, true, false).is_nan());
    }

    #[test]
    fn test_pnorm_deep_tails() {
        // Reference values also generated from WolframAlpha
        // A&S 7.1.26 carries an absolute error of ~1.5e-7.
        // For z = -5.0, the true mathematical probability is ~2.866e-7.
        // A&S would fail catastrophically here. libm::erfc retains full mantissa precision.

        let z: f64 = -5.0;
        let expected: f64 = 2.866515718791939116737523e-7;
        let calc: f64 = z_pnorm_std(z, true, false);

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
        let calc_deep: f64 = z_pnorm_std(z_deep, true, false);
        let relative_error_deep: f64 = (calc_deep - expected_deep).abs() / expected_deep;
        assert!(
            relative_error_deep < 1e-14,
            "Extreme deep tail relative precision lost."
        );

        // Test extreme tail where z = -20 (probability ~ 2.75e-89)
        let z_xtr: f64 = -20.0;
        let expected_xtr: f64 = 2.753624118606233695075623e-89;
        let calc_xtr: f64 = z_pnorm_std(z_xtr, true, false);
        let relative_error_xtr: f64 = (calc_xtr - expected_xtr).abs() / expected_xtr;
        assert!(
            relative_error_xtr < 1e-13,
            "Extreme deep tail relative precision lost."
        );
    }

    // --- z_dpois_rs tests ---

    #[test]
    fn test_dpois_at_zero() {
        // P(X=0 | λ=3) = e^{-3} ≈ 0.0497871
        let expected = (-3.0_f64).exp();
        assert!((z_dpois_rs(0, 3.0, false) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dpois_known_value() {
        // P(X=2 | λ=3) = 3^2 * e^{-3} / 2! = 9 * e^{-3} / 2 ≈ 0.2240418
        let expected = 9.0 * (-3.0_f64).exp() / 2.0;
        assert!((z_dpois_rs(2, 3.0, false) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dpois_sums_to_one() {
        // Sum of PMF over a large range should ≈ 1
        let lambda = 5.0;
        let total: f64 = (0..=50).map(|k| z_dpois_rs(k, lambda, false)).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    // --- z_ppois_di tests ---

    #[test]
    fn test_ppois_at_zero() {
        // P(X<=0 | λ=3) = P(X=0) = e^{-3}
        let expected = (-3.0_f64).exp();
        assert!((z_ppois_di(0, 3.0, false) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_ppois_monotone() {
        // CDF must be non-decreasing
        let lambda = 4.0;
        let mut prev = 0.0;
        for k in 0..=20 {
            let current = z_ppois_di(k, lambda, false);
            assert!(current >= prev);
            prev = current;
        }
    }

    #[test]
    fn test_ppois_approaches_one() {
        // P(X<=50 | λ=5) should be very close to 1
        assert!((z_ppois_di(50, 5.0, false) - 1.0).abs() < 1e-10);
    }

    // --- z_lgamma tests (Boost adaptation) ---

    #[test]
    fn test_lgamma_small_integers() {
        // ln Γ(n) = ln((n-1)!) for positive integers
        // Γ(1) = 0! = 1, ln(1) = 0
        assert!((z_lgamma(1.0) - 0.0).abs() < 1e-15);
        // Γ(2) = 1! = 1, ln(1) = 0
        assert!((z_lgamma(2.0) - 0.0).abs() < 1e-15);
        // Γ(3) = 2! = 2, ln(2) ≈ 0.6931471805599453
        assert!((z_lgamma(3.0) - 2.0_f64.ln()).abs() < 1e-15);
        // Γ(5) = 4! = 24, ln(24) ≈ 3.178053830347946
        assert!((z_lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-15);
        // Γ(11) = 10! = 3628800
        assert!((z_lgamma(11.0) - 3628800.0_f64.ln()).abs() < 1e-15);
    }

    #[test]
    fn test_lgamma_half_integers() {
        // Γ(0.5) = √π, so ln Γ(0.5) = 0.5 * ln(π) ≈ 0.5723649429247001
        let expected = 0.5 * PI.ln();
        assert!((z_lgamma(0.5) - expected).abs() < 1e-15);
        // Γ(1.5) = 0.5 * √π
        let expected_1_5 = (0.5 * PI.sqrt()).ln();
        assert!((z_lgamma(1.5) - expected_1_5).abs() < 1e-15);
    }

    // Expected values generated via Wolfram Alpha: N[Log[Gamma[z]], 25]
    // Tolerances set to 1e-15 and 1e-14 to account for standard f64
    // Unit in the Last Place (ULP) accumulation.
    #[test]
    fn test_lgamma_known_values() {
        // N[Log[Gamma[1/10]], 25]
        assert!((z_lgamma(0.1) - 2.252712651734205959869702).abs() < 1e-15);
        // N[Log[Gamma[25/10]], 25]
        assert!((z_lgamma(2.5) - 0.2846828704729191596324947).abs() < 1e-15);
        // N[Log[Gamma[103/10]], 25]
        assert!((z_lgamma(10.3) - 13.48203678613835697061507).abs() < 1e-14);
        // N[Log[Gamma[100]], 25]
        assert!((z_lgamma(100.0) - 359.1342053695753987760440).abs() < 1e-14);
    }

    // Expected values generated via Wolfram Alpha: N[Log[Gamma[z]], 25]
    // Tolerances set to 1e-15 and 1e-14 to account for standard f64
    // Unit in the Last Place (ULP) accumulation.
    #[test]
    fn test_lgamma_large_z() {
        // N[Log[Gamma[1000]], 25]
        assert!((z_lgamma(1000.0) - 5905.220423209181211826077).abs() < 1e-11);
        // N[Log[Gamma[10000]], 25]
        assert!((z_lgamma(10000.0) - 82099.71749644237727264896).abs() < 1e-11);
    }

    #[test]
    fn test_lgamma_near_one_and_two() {
        // Expected values generated via Wolfram Alpha: N[Log[Gamma[z]], 25]
        // Tolerances set to 1e-7 to account for precision loss from catastrophic
        // cancellation near the roots, as the dedicated Taylor series expansions
        // were bypassed for general performance and to simply the adaptation.

        let expected_1 = -0.00005772635749294248232231362;
        assert!((z_lgamma(1.0001) - expected_1).abs() < 1e-7);

        let expected_2 = -0.00004227364250705751767768638;
        assert!((z_lgamma(1.9999) - expected_2).abs() < 1e-7);

        // Verify sign: lgamma is negative on (1, 2) and positive outside
        assert!(z_lgamma(1.5) < 0.0);
        assert!(z_lgamma(0.5) > 0.0);
        assert!(z_lgamma(3.0) > 0.0);
    }

    #[test]
    fn test_lgamma_reflection_formula() {
        // For z < 0.5, verify via reflection: Γ(z)Γ(1-z) = π / sin(πz)
        // So lgamma(z) + lgamma(1-z) = ln(π) - ln|sin(πz)|
        let z = 0.25;
        let lhs = z_lgamma(z) + z_lgamma(1.0 - z);
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
        let val_below = z_lgamma(z - eps);
        let val_exact = z_lgamma(z);
        let val_above = z_lgamma(z + eps);

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
        assert!(z_lgamma(tiny) > 0.0);
        assert!(!z_lgamma(tiny).is_nan());

        // Extremely large number (approaching f64 overflow)
        // f64 max is ~1.7e308, lgamma overflows before that, but 1e100 is safe
        let massive = 1e100;
        assert!(z_lgamma(massive) > 0.0);
        assert!(!z_lgamma(massive).is_infinite());
    }

    // --- z_lgamma_godfrey tests ---

    #[test]
    fn test_lgamma_godfrey_matches_boost() {
        // Both implementations should agree to high precision
        let test_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 50.0, 100.0];
        for z in test_values {
            let boost = z_lgamma(z);
            let godfrey = z_lgamma_godfrey(z);
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
        assert!((z_lgamma_godfrey(1.0) - 0.0).abs() < 1e-14);
        assert!((z_lgamma_godfrey(5.0) - 24.0_f64.ln()).abs() < 1e-13);
    }

    // --- z_dgamma_rs tests ---

    #[test]
    fn test_dgamma_exponential_special_case() {
        // Gamma(shape=1, rate=λ) is Exponential(λ)
        // f(x) = λ * exp(-λx), so dgamma(1, 1, 2) = 2 * exp(-2) ≈ 0.2706706
        let result = z_dgamma_rs(1.0, 1.0, 2.0, false);
        let expected = 2.0 * (-2.0_f64).exp();
        assert!((result - expected).abs() < 1e-15);
    }

    #[test]
    fn test_dgamma_known_values() {
        // Cross-referenced with R: dgamma(2, shape=3, rate=1)
        // = 1^3 / Γ(3) * 2^2 * exp(-2) = 2^2 * exp(-2) / 2 ≈ 0.2706706
        let result = z_dgamma_rs(2.0, 3.0, 1.0, false);
        let expected = 0.2706705664732254;
        assert!(
            (result - expected).abs() < 1e-15,
            "dgamma(2, 3, 1) = {}, expected {}",
            result,
            expected
        );

        // dgamma(1, shape=2, rate=2) ≈ 0.5413411
        let result2 = z_dgamma_rs(1.0, 2.0, 2.0, false);
        let expected2 = 0.5413411329464508;
        assert!((result2 - expected2).abs() < 1e-15);

        // dgamma(0.5, shape=0.5, rate=1) ≈ 0.4393913
        let result3 = z_dgamma_rs(0.5, 0.5, 0.5, false);
        let expected3 = 0.4393912894677223970468620;
        assert!((result3 - expected3).abs() < 1e-15);
    }

    #[test]
    fn test_dgamma_log_mode() {
        // log(dgamma(2, 3, 1)) ≈ ln(0.2706706)
        let log_result = z_dgamma_rs(2.0, 3.0, 1.0, true);
        let direct_result = z_dgamma_rs(2.0, 3.0, 1.0, false);
        assert!((log_result - direct_result.ln()).abs() < 1e-14);
    }

    #[test]
    fn test_dgamma_large_shape() {
        // For large shape, gamma approaches normal; density should be well-behaved
        // dgamma(50, shape=50, rate=1) — near the mode at (shape-1)/rate = 49
        let result = z_dgamma_rs(50.0, 50.0, 1.0, false);
        assert!(result > 0.0);
        assert!(result < 1.0);
        // dgamma(50, 50, 1) ≈ 0.056325...
        assert!((result - 0.05632500632519082541154874).abs() < 1e-15);
    }

    #[test]
    fn test_dgamma_small_shape() {
        // Small shape parameter — density concentrated near zero
        // dgamma(0.01, shape=0.1, rate=1) should be large
        let result = z_dgamma_rs(0.01, 0.1, 1.0, false);
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
                z_dgamma_rs(x, shape, rate, false) * dx
            })
            .sum();
        // Should be close to 1, allowing for discretisation error
        assert!((sum - 1.0).abs() < 0.01, "Integral = {}", sum);
    }

    // --- z_pgamma_rs tests ---

    #[test]
    fn test_pgamma_exponential_cdf() {
        // Gamma(1, rate) is Exponential(rate): CDF = 1 - exp(-rate * x)
        let rate = 2.0;
        for &x in &[0.1, 0.5, 1.0, 2.0, 5.0] {
            let expected = 1.0 - f64::exp(-rate * x);
            assert!(
                (z_pgamma_rs(x, 1.0, rate, true, false) - expected).abs() < 1e-10,
                "pgamma({}, 1, {}) = {}, expected {}",
                x,
                rate,
                z_pgamma_rs(x, 1.0, rate, true, false),
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
            (z_pgamma_rs(x, 2.0, 1.0, true, false) - expected).abs() < 1e-10,
            "pgamma(3, 2, 1) = {}, expected {}",
            z_pgamma_rs(x, 2.0, 1.0, true, false),
            expected
        );

        // Shape = 3, rate = 1: P = 1 - e^(-x)(1 + x + x²/2)
        let expected3 = 1.0 - f64::exp(-x) * (1.0 + x + x * x / 2.0);
        assert!(
            (z_pgamma_rs(x, 3.0, 1.0, true, false) - expected3).abs() < 1e-10,
            "pgamma(3, 3, 1) = {}, expected {}",
            z_pgamma_rs(x, 3.0, 1.0, true, false),
            expected3
        );
    }

    #[test]
    fn test_pgamma_known_values() {
        // Cross-referenced with R's pgamma()
        // pgamma(1, shape=2, rate=1) ≈ 0.2642411
        assert!((z_pgamma_rs(1.0, 2.0, 1.0, true, false) - 0.26424111765711533).abs() < 1e-10);
        // pgamma(5, shape=3, rate=1) ≈ 0.8753480
        assert!((z_pgamma_rs(5.0, 3.0, 1.0, true, false) - 0.8753479805169189).abs() < 1e-10);
        // pgamma(0.5, shape=0.5, rate=1) ≈ 0.6826895
        assert!((z_pgamma_rs(0.5, 0.5, 1.0, true, false) - 0.6826894921370859).abs() < 1e-9);
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
        let p1 = z_pgamma_rs(x, shape, rate, true, false);
        let p2 = z_pgamma_rs(rate * x, shape, 1.0, true, false);
        assert!((p1 - p2).abs() < 1e-12);
    }

    #[test]
    fn test_pgamma_edge_cases() {
        // P(0) = 0 for any valid shape and rate
        assert_eq!(z_pgamma_rs(0.0, 2.0, 1.0, true, false), 0.0);
        // Negative x → 0
        assert_eq!(z_pgamma_rs(-1.0, 2.0, 1.0, true, false), 0.0);
        // Infinity → 1
        assert_eq!(z_pgamma_rs(f64::INFINITY, 2.0, 1.0, true, false), 1.0);
    }

    #[test]
    fn test_pgamma_monotonicity() {
        // CDF must be monotonically non-decreasing
        let shape = 2.5;
        let rate = 1.0;
        let xs = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0];
        let ps: Vec<f64> = xs
            .iter()
            .map(|&x| z_pgamma_rs(x, shape, rate, true, false))
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
                let p = z_pgamma_rs(x, a, 1.0, true, false);
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
        let p_at_mean = z_pgamma_rs(shape / rate, shape, rate, true, false);
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
        let p = z_pgamma_rs(0.01, 0.1, 1.0, true, false);
        assert!(p > 0.1, "Small shape: pgamma(0.01, 0.1, 1) = {}", p);
    }

    #[test]
    fn test_pgamma_pdf_cdf_consistency() {
        // Approximate derivative of CDF should equal PDF
        let shape = 3.0;
        let rate = 1.0;
        let x = 2.0;
        let h = 1e-7;
        let numerical_pdf = (z_pgamma_rs(x + h, shape, rate, true, false)
            - z_pgamma_rs(x - h, shape, rate, true, false))
            / (2.0 * h);
        let analytical_pdf = z_dgamma_rs(x, shape, rate, false);
        assert!(
            (numerical_pdf - analytical_pdf).abs() < 1e-5,
            "CDF derivative {} vs PDF {}",
            numerical_pdf,
            analytical_pdf
        );
    }

    // --- Tweedie Parameter Conversion Tests ---

    #[test]
    fn test_tw_to_pg_shape_special_case() {
        // For the Tweedie distribution, when p = 1.5, the underlying Gamma shape
        // parameter α = (2 - p) / (p - 1) evaluates to 0.5 / 0.5 = 1.0.
        // This means the Gamma severities simplify to Exponential severities.
        let tw = TwParams::new(5.0, 2.0, 1.5).expect("Valid parameters");
        let pg = tw.to_pg();

        assert!((pg.shape - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tw_pg_round_trip() {
        // Define arbitrary valid Tweedie parameters
        let orig_mu = 10.0;
        let orig_phi = 1.5;
        let orig_power = 1.7;

        let tw = TwParams::new(orig_mu, orig_phi, orig_power).expect("Valid parameters");

        // Convert to Poisson-Gamma
        let pg = tw.to_pg();

        // 1. Verify the specific algebraic relationship: E[Y] = μ = λ * (α / β)
        let expected_mu = pg.lambda * (pg.shape / pg.rate);
        assert!((expected_mu - orig_mu).abs() < 1e-10);

        // 2. Verify the full reverse conversion back to Tweedie
        let tw_recovered = pg.to_tw();
        assert!((tw_recovered.mu - orig_mu).abs() < 1e-10);
        assert!((tw_recovered.phi - orig_phi).abs() < 1e-10);
        assert!((tw_recovered.power - orig_power).abs() < 1e-10);
    }

    #[test]
    fn test_tw_smart_constructor_validation() {
        // Ensure the smart constructor rejects mathematically invalid parameters
        assert!(
            TwParams::new(0.0, 1.0, 1.5).is_err(),
            "Should reject μ <= 0"
        );
        assert!(
            TwParams::new(1.0, 0.0, 1.5).is_err(),
            "Should reject φ <= 0"
        );
        assert!(
            TwParams::new(1.0, 1.0, 1.0).is_err(),
            "Should reject p = 1 (Poisson degenerate)"
        );
        assert!(TwParams::new(1.0, 1.0, 2.5).is_err(), "Should reject p > 2");
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

    // --- Tweedie Engine Tests ---

    #[test]
    fn test_ds_series_terminates() {
        // Simply verifies the hill-climb and expansion loops do not infinite-loop
        // and that they return a valid, finite f64 value.
        let val = dtweedie_series(2.5, 1.5, 1.0, 1.0, true);
        assert!(val.is_finite());
    }

    #[test]
    fn test_dtweedie_fast_paths() {
        // Fast Path 1: mu = 0, y = 0 -> Probability 1.0 (log = 0.0)
        assert_eq!(z_dtweedie_rs(0.0, 0.0, 1.0, 1.5, true), 0.0);

        // Fast Path 1b: mu = 0, y > 0 -> Probability 0.0 (log = -Inf)
        assert_eq!(z_dtweedie_rs(2.0, 0.0, 1.0, 1.5, true), f64::NEG_INFINITY);

        // Fast Path 2: y = 0, mu > 0 -> Poisson point mass
        // For mu = 5, p = 1.5, phi = 2.0 -> lambda = 5^(0.5) / (2 * 0.5) = sqrt(5)
        let lambda = 5.0_f64.powf(0.5) / 1.0;
        let expected_log_prob = -lambda; // Since Poisson(0; lambda) = e^-lambda

        let calc = z_dtweedie_rs(0.0, 5.0, 2.0, 1.5, true);
        assert!((calc - expected_log_prob).abs() < 1e-14);
    }

    #[test]
    fn test_ptweedie_series_terminates() {
        // Verifies the linear space while-loops converge and return a valid probability
        let val = ptweedie_series(2.5, 1.5, 1.0, 1.0, true);
        assert!(val >= 0.0 && val <= 1.0);
    }

    #[test]
    fn test_ptweedie_fast_paths() {
        // mu = 0 -> Point mass at 0. CDF for any y >= 0 is 1.0 (or 0.0 in log space)
        assert_eq!(z_ptweedie_rs(5.0, 0.0, 1.0, 1.5, true, false), 1.0);
        assert_eq!(z_ptweedie_rs(5.0, 0.0, 1.0, 1.5, true, true), 0.0);

        // y = 0, mu > 0 -> Poisson point mass at 0: exp(-lambda)
        let lambda = 5.0_f64.powf(0.5) / 1.0;
        let expected_p = (-lambda).exp();
        let expected_log_p = -lambda;

        assert!((z_ptweedie_rs(0.0, 5.0, 2.0, 1.5, true, false) - expected_p).abs() < 1e-14);
        assert!((z_ptweedie_rs(0.0, 5.0, 2.0, 1.5, true, true) - expected_log_p).abs() < 1e-14);
    }

    // --- Tweedie Invariant Tests ---

    #[test]
    fn test_dtweedie_regression_y_zero_direct_space() {
        // This explicitly catches the `-lambda.exp()` precedence bug.
        // If the bug exists, this will return a massive negative number instead of the true probability.
        let mu = 5.0;
        let phi = 2.0;
        let p = 1.5;
        // lambda = mu^(2-p) / (phi * (2-p))
        let lambda = 5.0_f64.powf(0.5) / 1.0;
        let expected_p = (-lambda).exp();

        let calc_p = z_dtweedie_rs(0.0, mu, phi, p, false);
        assert!(
            (calc_p - expected_p).abs() < 1e-14,
            "Direct space y=0 fast path failed."
        );
    }

    #[test]
    fn test_ptweedie_monotonicity() {
        // CDF must be strictly non-decreasing: F(y_1) <= F(y_2) for y_1 < y_2
        let y_vals = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0];
        let mut prev_cdf = -1.0;

        for &y in &y_vals {
            let curr_cdf = z_ptweedie_rs(y, 2.0, 1.5, 1.6, true, false);
            assert!(curr_cdf >= prev_cdf, "CDF monotonicity violated at y={}", y);
            prev_cdf = curr_cdf;
        }
    }

    #[test]
    fn test_ptweedie_bounds() {
        // F(y) must be bounded strictly within [0, 1]
        let y_vals = [0.0, 0.5, 5.0, 50.0];

        for &y in &y_vals {
            let cdf = z_ptweedie_rs(y, 3.0, 1.2, 1.4, true, false);
            assert!(
                cdf >= 0.0 && cdf <= 1.0,
                "CDF out of bounds at y={}: {}",
                y,
                cdf
            );
        }
    }

    #[test]
    fn test_tweedie_pdf_cdf_consistency() {
        // The central-difference numerical derivative of the CDF
        // should approximate the PDF.
        // f(y) ≈ (F(y + h) - F(y - h)) / 2h
        let y = 2.5;
        let mu = 2.0;
        let phi = 1.2;
        let p = 1.5;
        let h = 1e-5;

        let pdf = z_dtweedie_rs(y, mu, phi, p, false);
        let cdf_plus = z_ptweedie_rs(y + h, mu, phi, p, true, false);
        let cdf_minus = z_ptweedie_rs(y - h, mu, phi, p, true, false);

        let approx_pdf = (cdf_plus - cdf_minus) / (2.0 * h);

        // Tolerance set to 1e-4 as suggested due to series accumulation error
        assert!(
            (pdf - approx_pdf).abs() < 1e-4,
            "PDF/CDF consistency violated. PDF: {}, Approx: {}",
            pdf,
            approx_pdf
        );
    }

    #[test]
    fn test_ptweedie_upper_tail_consistency() {
        // F(y) + S(y) = 1.0
        let y = 3.5;
        let mu = 2.5;
        let phi = 1.1;
        let p = 1.6;

        let lower = z_ptweedie_rs(y, mu, phi, p, true, false);
        let upper = z_ptweedie_rs(y, mu, phi, p, false, false);

        assert!(
            (lower + upper - 1.0).abs() < 1e-14,
            "Upper/Lower tail consistency violated. Sum: {}",
            lower + upper
        );
    }

    // --- z_dinvgauss_rs tests ---

    #[test]
    fn test_dinvgauss_peak() {
        // When y = mu = 1, lambda = 1, the exponential term is 0.
        // PDF simplifies to 1 / sqrt(2 * PI)
        let expected = 1.0 / statrs::consts::SQRT_2PI;
        assert!((z_dinvgauss_rs(1.0, 1.0, 1.0, false) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_dinvgauss_fast_paths() {
        assert_eq!(z_dinvgauss_rs(0.0, 1.0, 1.0, false), 0.0);
        assert_eq!(z_dinvgauss_rs(-1.0, 1.0, 1.0, false), 0.0);
        assert_eq!(z_dinvgauss_rs(f64::INFINITY, 1.0, 1.0, false), 0.0);
        assert_eq!(z_dinvgauss_rs(0.0, 1.0, 1.0, true), f64::NEG_INFINITY);
    }

    // --- z_pinvgauss_rs tests ---

    #[test]
    fn test_pinvgauss_fast_paths() {
        assert_eq!(z_pinvgauss_rs(0.0, 1.0, 1.0, true, false), 0.0);
        assert_eq!(z_pinvgauss_rs(0.0, 1.0, 1.0, false, false), 1.0);
        assert_eq!(z_pinvgauss_rs(f64::INFINITY, 1.0, 1.0, true, false), 1.0);
    }

    #[test]
    fn test_pinvgauss_tail_complementarity() {
        // F(y) + S(y) = 1.0
        let y = 2.5;
        let mu = 1.5;
        let lambda = 2.0;
        let lower = z_pinvgauss_rs(y, mu, lambda, true, false);
        let upper = z_pinvgauss_rs(y, mu, lambda, false, false);
        assert!((lower + upper - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_pinvgauss_extreme_parameter_brake() {
        // Triggers the exponent > 709.0 brake.
        // 2 * lambda / mu = 1000.
        // Should safely return the first pnorm term without NaN panics.
        let cdf = z_pinvgauss_rs(1.0, 1.0, 500.0, true, false);
        assert!(cdf >= 0.0 && cdf <= 1.0 && !cdf.is_nan());
    }
}

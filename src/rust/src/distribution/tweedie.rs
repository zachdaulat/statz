//! Compound Poisson-Gamma Tweedie ($1 < p < 2$) density (`dtweedie`) and
//! cumulative probability (`ptweedie`).
//!
//! Evaluates the Dunn & Smyth (2005) infinite series expansion via modal hill-climbing
//! and bilateral series truncation.

use super::{
    gamma::{dgamma, pgamma},
    poisson::dpois,
    special::log_sum_exp,
};
use extendr_api::prelude::*;

extendr_module! {
    mod tweedie;
    fn dtweedie_r;
    fn ptweedie_r;
}

/// Computes the Tweedie probability density function
///
/// @description
/// Computes the density for a Tweedie random variable using the
/// Dunn & Smyth (2005) series expansion. Parameterised strictly for
/// the compound Poisson-gamma case (1 < p < 2).
///
/// @param y A numeric (double) vector of quantiles (y >= 0).
/// @param mu The mean parameter (μ >= 0).
/// @param phi The dispersion parameter (φ > 0).
/// @param power The variance power parameter (1 < p < 2).
/// @param log Logical; if TRUE, probabilities p are given as ln(p).
/// @return A numeric vector of probability densities.
/// @export
#[extendr(r_name = "dtweedie")]
pub fn dtweedie_r(
    y: &[f64],
    mu: f64,
    phi: f64,
    power: f64,
    #[extendr(default = "FALSE")] log: bool,
) -> extendr_api::Result<Doubles> {
    if power <= 1.0 || power >= 2.0 {
        return Err(extendr_api::Error::Other(
            "power must be strictly between 1 and 2 for compound Poisson-Gamma".into(),
        ));
    }
    if phi <= 0.0 || mu < 0.0 {
        return Err(extendr_api::Error::Other(
            "mu must be >= 0 and phi must be > 0".into(),
        ));
    }

    let densities: Doubles = y
        .iter()
        .map(|&yi| dtweedie(yi, mu, phi, power, log))
        .collect::<Doubles>();

    Ok(densities)
}

pub fn dtweedie(y: f64, mu: f64, phi: f64, power: f64, log: bool) -> f64 {
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

/// Computes the Tweedie cumulative distribution function
///
/// @description
/// Computes the cumulative probability for a Tweedie random variable using the
/// Dunn & Smyth (2005) series expansion. Parameterised strictly for
/// the compound Poisson-gamma case (1 < p < 2).
///
/// @param y A numeric (double) vector of quantiles (y >= 0).
/// @param mu The mean parameter (μ >= 0).
/// @param phi The dispersion parameter (φ > 0).
/// @param power The variance power parameter (1 < p < 2).
/// @param lower_tail Logical; if TRUE (default), probabilities are P(Y <= y).
/// @param log_p Logical; if TRUE, probabilities p are given as ln(p).
/// @return A numeric vector of cumulative probabilities.
/// @export
#[extendr(r_name = "ptweedie")]
pub fn ptweedie_r(
    y: &[f64],
    mu: f64,
    phi: f64,
    power: f64,
    #[extendr(default = "TRUE")] lower_tail: bool,
    #[extendr(default = "FALSE")] log_p: bool,
) -> extendr_api::Result<Doubles> {
    if power <= 1.0 || power >= 2.0 {
        return Err(extendr_api::Error::Other(
            "power must be strictly between 1 and 2 for compound Poisson-Gamma".into(),
        ));
    }
    if phi <= 0.0 || mu < 0.0 {
        return Err(extendr_api::Error::Other(
            "mu must be >= 0 and phi must be > 0".into(),
        ));
    }

    let masses: Doubles = y
        .iter()
        .map(|&yi| ptweedie(yi, mu, phi, power, lower_tail, log_p))
        .collect::<Doubles>();

    Ok(masses)
}

pub fn ptweedie(y: f64, mu: f64, phi: f64, power: f64, lower_tail: bool, log_p: bool) -> f64 {
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

/// Dunn-Smyth Series helper function, Tweedie density for y > 0
/// Internal engine for the Tweedie density infinite series (y > 0).
/// Accumulates Poisson-weighted Gamma densities into a vector for log_sum_exp()
pub(crate) fn dtweedie_series(y: f64, lambda: f64, shape: f64, rate: f64, log: bool) -> f64 {
    // 1: Starting index and log-likelihood
    let k_0: f64 = lambda.floor().max(1.0);
    let l_0: f64 = dpois(k_0, lambda, true) + dgamma(y, k_0 * shape, rate, true);

    // Evaluating just the right side finding gradient direction
    let k_p1: f64 = k_0 + 1.0;
    let l_p1: f64 = dpois(k_p1, lambda, true) + dgamma(y, k_p1 * shape, rate, true);

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

        l_curr = dpois(k_curr, lambda, true) + dgamma(y, k_curr * shape, rate, true);
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
    let mut ld_pois: f64 = dpois(ki, lambda, true);
    let mut ld_gamma: f64 = dgamma(y, ki * shape, rate, true);
    let mut li: f64 = ld_pois + ld_gamma;

    while li > lim {
        l_terms.push(li); // Accumulate immediately

        ki -= 1.0;
        if ki < 1.0 {
            break;
        } // Domain safety

        // Calculate next term for condition check
        let ld_pois: f64 = dpois(ki, lambda, true);
        let ld_gamma: f64 = dgamma(y, ki * shape, rate, true);
        li = ld_pois + ld_gamma;
    }

    // --- Forward loop ---
    ki = k_max + 1.0;
    // Priming read for forward loop
    ld_pois = dpois(ki, lambda, true);
    ld_gamma = dgamma(y, ki * shape, rate, true);
    li = ld_pois + ld_gamma;

    while li > lim {
        l_terms.push(li); // Accumulating immediately upon entering loop iteration
        ki += 1.0;

        // Calculate next term for condition check
        let ld_pois: f64 = dpois(ki, lambda, true);
        let ld_gamma: f64 = dgamma(y, ki * shape, rate, true);
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
    let lp_pois: f64 = dpois(k0, lambda, true);
    let lp_gamma: f64 = pgamma(y, k0 * shape, rate, lower_tail, true);
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
        let lp_pois: f64 = dpois(ki, lambda, true);
        let lp_gamma: f64 = pgamma(y, ki * shape, rate, lower_tail, true);
        term = (lp_pois + lp_gamma).exp();
    }

    // 3: Reset indices to next index for forward loop
    ki = k0 + 1.0;
    let lp_pois: f64 = dpois(ki, lambda, true);
    let lp_gamma: f64 = pgamma(y, ki * shape, rate, lower_tail, true);
    term = (lp_pois + lp_gamma).exp();

    // 4: Enter forward loop with next index
    while term > f64::EPSILON {
        // Accumulating new term
        sum += term;
        ki += 1.0;
        let lp_pois: f64 = dpois(ki, lambda, true);
        let lp_gamma: f64 = pgamma(y, ki * shape, rate, lower_tail, true);
        term = (lp_pois + lp_gamma).exp();
    }

    sum
}

// Rust-side unit tests
#[allow(clippy::excessive_precision)]
#[allow(clippy::manual_range_contains)]
#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(dtweedie(0.0, 0.0, 1.0, 1.5, true), 0.0);

        // Fast Path 1b: mu = 0, y > 0 -> Probability 0.0 (log = -Inf)
        assert_eq!(dtweedie(2.0, 0.0, 1.0, 1.5, true), f64::NEG_INFINITY);

        // Fast Path 2: y = 0, mu > 0 -> Poisson point mass
        // For mu = 5, p = 1.5, phi = 2.0 -> lambda = 5^(0.5) / (2 * 0.5) = sqrt(5)
        let lambda = 5.0_f64.powf(0.5) / 1.0;
        let expected_log_prob = -lambda; // Since Poisson(0; lambda) = e^-lambda

        let calc = dtweedie(0.0, 5.0, 2.0, 1.5, true);
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
        assert_eq!(ptweedie(5.0, 0.0, 1.0, 1.5, true, false), 1.0);
        assert_eq!(ptweedie(5.0, 0.0, 1.0, 1.5, true, true), 0.0);

        // y = 0, mu > 0 -> Poisson point mass at 0: exp(-lambda)
        let lambda = 5.0_f64.powf(0.5) / 1.0;
        let expected_p = (-lambda).exp();
        let expected_log_p = -lambda;

        assert!((ptweedie(0.0, 5.0, 2.0, 1.5, true, false) - expected_p).abs() < 1e-14);
        assert!((ptweedie(0.0, 5.0, 2.0, 1.5, true, true) - expected_log_p).abs() < 1e-14);
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

        let calc_p = dtweedie(0.0, mu, phi, p, false);
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
            let curr_cdf = ptweedie(y, 2.0, 1.5, 1.6, true, false);
            assert!(curr_cdf >= prev_cdf, "CDF monotonicity violated at y={}", y);
            prev_cdf = curr_cdf;
        }
    }

    #[test]
    fn test_ptweedie_bounds() {
        // F(y) must be bounded strictly within [0, 1]
        let y_vals = [0.0, 0.5, 5.0, 50.0];

        for &y in &y_vals {
            let cdf = ptweedie(y, 3.0, 1.2, 1.4, true, false);
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

        let pdf = dtweedie(y, mu, phi, p, false);
        let cdf_plus = ptweedie(y + h, mu, phi, p, true, false);
        let cdf_minus = ptweedie(y - h, mu, phi, p, true, false);

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

        let lower = ptweedie(y, mu, phi, p, true, false);
        let upper = ptweedie(y, mu, phi, p, false, false);

        assert!(
            (lower + upper - 1.0).abs() < 1e-14,
            "Upper/Lower tail consistency violated. Sum: {}",
            lower + upper
        );
    }
}

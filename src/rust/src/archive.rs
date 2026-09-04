#![allow(unused)]

use extendr_api::prelude::*;
use faer::{ColRef, MatRef};
use std::f64;

// Registering the archived modules
extendr_module! {
    mod archive;
    use arch_descriptive;
    use arch_distributions;
}

// RMatrix to faer Mat conversion function
// Zero-copy view of RMatrix
pub(crate) fn rmatrix_to_matref<'a>(x: &'a RMatrix<f64>) -> MatRef<'a, f64> {
    let nrows: usize = x.nrows();
    let ncols: usize = x.ncols();
    let data: &[f64] = x.data(); // This correctly returns &[f64]

    // Explicitly use MatRef, not Mat
    MatRef::from_column_major_slice(data, nrows, ncols)
}

// Doubles to faer Col conversion function
// Zero-copy view of Doubles
pub(crate) fn doubles_to_colref<'a>(y: &'a Doubles) -> ColRef<'a, f64> {
    // Drop down to the underlying Robj to extract the raw f64 slice safely
    let data: &[f64] = y
        .as_robj()
        .as_real_slice()
        .expect("Vector must be standard real numbers");

    // Explicitly use ColRef, not Col
    ColRef::from_slice(data)
}

pub mod arch_descriptive {
    use extendr_api::prelude::*;

    // Registering archived functions
    extendr_module! {
        mod arch_descriptive;
        fn z_sum;
        fn z_mean;
        fn z_median;
        fn z_var;
        fn z_sd;
        fn z_cov;
        fn z_cor;
        fn z_cor_onepass;
    }

    /// Compute the sum of a numeric vector. Naïve summation with Rust's `.sum()` method
    /// @param x A numeric vector.
    /// @return The sum as a double.
    /// @keywords internal
    #[extendr]
    pub fn z_sum(x: &[f64]) -> f64 {
        x.iter().sum()
    }

    /// Compute the arithmetic mean of a numeric vector.
    /// @param x A numeric vector
    /// @return The mean as a double.
    /// @keywords internal
    #[extendr]
    pub fn z_mean(x: &[f64]) -> f64 {
        if x.is_empty() {
            return f64::NAN;
        }
        let n = x.len() as f64;
        z_sum(x) / n
    }

    /// Compute the median of a numeric vector.
    /// @param x A numeric vector.
    /// @return The median as a double
    /// @keywords internal
    #[extendr]
    pub fn z_median(x: &[f64]) -> f64 {
        if x.is_empty() {
            return f64::NAN;
        }
        let mut sorted = x.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    /// Compute the sample variance of a numeric vector (Bessel-corrected, n-1).
    /// @param x A numeric vector.
    /// @return The sample variance as a double.
    /// @keywords internal
    #[extendr]
    pub fn z_var(x: &[f64]) -> f64 {
        if x.len() < 2 {
            return f64::NAN;
        }
        let mean: f64 = z_mean(x);
        let n = x.len() as f64;
        x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / (n - 1.0)
    }

    /// Compute the sample standard deviation of a numeric vector.
    /// @param x A numeric vector.
    /// @return The sample standard deviation as a double.
    /// @keywords internal
    #[extendr]
    pub fn z_sd(x: &[f64]) -> f64 {
        z_var(x).sqrt()
    }

    /// Compute the sample covariance of two numeric vectors.
    /// @param x A numeric vector.
    /// @param y A numeric vector of the same length.
    /// @return The sample covariance as a double.
    /// @keywords internal
    #[extendr]
    pub fn z_cov(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return f64::NAN;
        }
        let mean_x: f64 = z_mean(x);
        let mean_y: f64 = z_mean(y);
        let n: f64 = x.len() as f64;
        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / (n - 1.0)
    }

    /// Compute the Pearson correlation coefficient of two numeric vectors.
    /// @param x A numeric vector.
    /// @param y A numeric vector.
    /// @return The sample correlation as a double
    /// @keywords internal
    #[extendr]
    pub fn z_cor(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return f64::NAN;
        }
        z_cov(x, y) / (z_sd(x) * z_sd(y))
    }

    /// Compute Pearson correlation coefficient, optimized single-pass
    /// @param x A numeric vector.
    /// @param y A numeric vector.
    /// @return The sample correlation as a double
    /// @keywords internal
    #[extendr]
    pub fn z_cor_onepass(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return f64::NAN;
        }

        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        let mut sum_xy: f64 = 0.0;
        let mut sum_x2: f64 = 0.0;
        let mut sum_y2: f64 = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            sum_x += xi;
            sum_y += yi;
            sum_xy += xi * yi;
            sum_x2 += xi * xi;
            sum_y2 += yi * yi;
        }

        let n: f64 = x.len() as f64;
        let numer: f64 = sum_xy - (sum_x * sum_y / n);
        let denom: f64 = ((sum_x2 - (sum_x.powi(2) / n)) * (sum_y2 - (sum_y.powi(2) / n))).sqrt();

        numer / denom
    }
}

pub mod arch_distributions {
    use crate::consts::*;
    use extendr_api::prelude::*;
    use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

    // Registering archived functions
    extendr_module! {
        mod arch_distributions;
        fn pnorm_as;
        fn ppois_di;
        fn lgamma_godfrey;
    }

    /// Compute the standard normal cumulative distribution function (CDF)
    ///
    /// @description
    /// Uses the Abramowitz and Stegun (1972, 10th ed.) equation 7.1.26
    /// error function approximation. Deprecated in favour of the
    /// libm::erfc() implementation but kept as an initial pedagogical version.
    /// Maximum absolute error: |ε| < 1.5 × 10^7
    ///
    /// @param z A z-score (standardised value)
    /// @param lower_tail Logical; if TRUE (default), probabilities are P(X <= x).
    /// @param log_p Logical; if TRUE, probabilities p are given as ln(p).
    /// @return A numeric vector of cumulative probabilities.
    /// @keywords internal
    #[extendr]
    #[allow(non_upper_case_globals)]
    pub fn pnorm_as(z: f64, lower_tail: bool, log_p: bool) -> f64 {
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

    /// Compute the Poisson probability mass function
    ///
    /// @description
    /// Compute the Poisson cumulative distribution function P(X ≤ x)
    /// using log-space PMF evaluation for each term.
    ///
    /// @param x A numeric (double) vector of non-negative whole numbers.
    /// @param lambda The rate parameter (λ > 0).
    /// @param log Logical; if TRUE, probabilities p are given as ln(p).
    /// @return A numeric vector of probability masses.
    /// @keywords internal
    #[extendr]
    // Named di for reference to the dpois function and this iterator approach
    pub fn ppois_di(x: i32, lambda: f64, log_p: bool) -> f64 {
        // Calculated in regular space
        let cdf: f64 = (0..=x)
            .map(|i| crate::distributions::dpois(i as f64, lambda, false))
            .sum::<f64>();

        // Evaluating boolean for returning in log or regular space
        if log_p {
            cdf.ln()
        } else {
            cdf
        }
    }

    // ============================================================
    // Documentation for lgamma_godfrey (pedagogical implementation)
    // ============================================================

    /// Compute ln Γ(z) using Godfrey's Lanczos coefficient set (g=7, N=9).
    ///
    /// @description
    /// This is a pedagogical implementation of the traditional Lanczos approximation
    /// using Paul Godfrey's well-known f64 coefficient set. Unlike the Boost
    /// adaptation in `lgamma()`, this uses the standard formulation:
    ///
    ///   ln Γ(z) = ½ ln(2π) + (z - ½) ln(z + g - ½) - (z + g - ½) + ln S(z)
    ///
    /// where S(z) = c₀ + Σ(k=1..8) cₖ/(z-1+k) is the Lanczos sum with
    /// alternating-sign coefficients, and the formula is evaluated after
    /// shifting z → z-1 to convert from Γ(z+1) to Γ(z).
    ///
    /// This implementation is less precise than `lgamma()` due to potential
    /// cancellation in the alternating-sign sum, but is included as a learning
    /// exercise.
    ///
    /// @references
    /// - Godfrey, P. "Lanczos Implementation of the Gamma Function."
    ///   <http://my.fit.edu/~gabdo/gamma.txt>
    /// - <https://www.mrob.com/pub/ries/lanczos-gamma.html>
    ///
    /// @param z A numeric (double) vector of positive values.
    /// @return A numeric vector containing the natural logarithm of the gamma function.
    /// @keywords internal
    #[allow(unused)]
    #[allow(clippy::excessive_precision)]
    #[allow(clippy::needless_range_loop)]
    #[extendr]
    pub(crate) fn lgamma_godfrey(mut z: f64) -> f64 {
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
            let lgam: f64 = LN_PI - (PI * z).sin().abs().ln() - lgamma_godfrey(1.0 - z);
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
}

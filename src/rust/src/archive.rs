#![allow(unused)]

use extendr_api::prelude::*;
use faer::{ColRef, MatRef};
use std::f64;

// Registering the archived modules
extendr_module! {
    mod archive;
    use arch_descriptive;
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

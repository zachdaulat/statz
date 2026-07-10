use faer::{Col};
use extendr_api::prelude::*;

// Registering this module's functions
extendr_module! {
    mod descriptive;
    fn z_sum;
    fn z_mean;
    fn z_median;
    fn z_var;
    fn z_sd;
    fn z_cov;
    fn z_cor;
    fn z_cor_onepass;
}

/// Compute the sum of a numeric vector.
/// @param x A numeric vector.
/// @return The sum as a double.
/// @export
#[extendr]
pub fn z_sum(x: &[f64]) -> f64 {
    x.iter().sum()
}

/// Compute the arithmetic mean of a numeric vector.
/// @param x A numeric vector
/// @return The mean as a double.
/// @export
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
/// @export
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
/// @export
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
/// @export
#[extendr]
pub fn z_sd(x: &[f64]) -> f64 {
    z_var(x).sqrt()
}

/// Compute the sample covariance of two numeric vectors.
/// @param x A numeric vector.
/// @param y A numeric vector of the same length.
/// @return The sample covariance as a double.
/// @export
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
/// @export
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
/// @export
#[extendr]
pub fn z_cor_onepass(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }

    let n: f64 = x.len() as f64;
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

    let numer = sum_xy - (sum_x * sum_y / n);
    let denom = ((sum_x2 - (sum_x.powi(2) / n)) * (sum_y2 - (sum_y.powi(2) / n))).sqrt();

    numer / denom
}

// // Future quantile function for export to R using Rust level function
// #[extendr]
// pub fn z_quantile(x: Doubles, probs: Doubles) -> Doubles {
//     // 1. Converting Doubles inputs to slices
//     let x_slice = x.as_slice();
//     let probs_slice = probs.as_slice();

//     // 2. Calling core engine quantile()
//     let quantiles_col: Col<f64> = quantile(x_slice, probs_slice);

//     // 3. Converting Col<f64> to R's Doubles
//     quantiles_col.iter().collect::<Doubles>()
// }

// Internally used empirical quantile function
pub(crate) fn quantile(x: &[f64], probs: &[f64]) -> Col<f64> {
    // Guarding against empty slices to prevent usize underflow
    if x.is_empty() {
        return Col::full(probs.len(), f64::NAN);
    }
    // Get length of sample 0-indexed
    let n: usize = x.len() - 1;
    let n_f64: f64 = n as f64;

    // 1. Copying and sorting the sample
    let mut x_sort: Vec<f64> = x.to_vec();
    x_sort.sort_by(|a: &f64, b: &f64| a.total_cmp(b));

    // 2. Initialize and populate new Col<f64>
    Col::from_fn(probs.len(), |i| {
        let p: f64 = probs[i];
        let index: f64 = p * (n_f64);
        let j: usize = index.floor() as usize;
        let gamma: f64 = index - index.floor();

        // Guard against single-observation buckets and upper boundary p=1
        if j >= n {
            x_sort[n]
        } else {
            (1.0 - gamma) * x_sort[j] + gamma * x_sort[j + 1]
        }
    })
}

// Rust-side unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        assert_eq!(z_sum(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(z_sum(&[]), 0.0);
    }

    #[test]
    fn test_mean() {
        assert!((z_mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-10);
        assert!(z_mean(&[]).is_nan());
    }

    #[test]
    fn test_median_odd() {
        assert_eq!(z_median(&[3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn test_median_even() {
        assert_eq!(z_median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn test_variance() {
        // var(c(2, 4, 4, 4, 5, 5, 7, 9)) in R = 4.571429
        let x = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((z_var(&x) - 4.571428571428571).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_identical() {
        // cov(x, x) should equal var(x)
        let x = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((z_cov(&x, &x) - z_var(&x)).abs() < 1e-10);
    }

    #[test]
    fn test_cor_perfect_positive() {
        // cor(x, x) = 1.0
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((z_cor(&x, &x) - 1.0).abs() < 1e-10);
        assert!((z_cor_onepass(&x, &x) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cor_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((z_cor(&x, &y) - (-1.0)).abs() < 1e-10);
        assert!((z_cor_onepass(&x, &y) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cor_implementations_agree() {
        // Both implementations should produce the same result
        let x = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let y = vec![1.0, 3.0, 5.0, 2.0, 7.0, 6.0, 8.0, 4.0];
        assert!((z_cor(&x, &y) - z_cor_onepass(&x, &y)).abs() < 1e-10);
    }

    // ==========================================
    // Tests for quantile
    // ==========================================

    // Import everything from the parent module
    use faer::{col, Col};

    /// Helper function for safe floating-point comparison of faer columns
    fn assert_col_eq(a: &Col<f64>, b: &Col<f64>, tol: f64) {
        assert_eq!(a.nrows(), b.nrows(), "Dimension mismatch");
        for i in 0..a.nrows() {
            assert!(
                (a[i] - b[i]).abs() < tol,
                "Mismatch at index {}: {} vs {}",
                i,
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_quantile_sorted_data() {
        // Basic test with pre-sorted data checking standard quartiles
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = vec![0.0, 0.5, 1.0];
        
        // Adjust the path to crate::descriptive::quantile if this test 
        // is housed outside the descriptive module
        let q = crate::descriptive::quantile(&data, &probs);
        let expected = col![1.0, 3.0, 5.0];
        
        assert_col_eq(&q, &expected, 1e-9);
    }

    #[test]
    fn test_quantile_unsorted_data() {
        // The quantile function should handle unsorted memory seamlessly
        let data = vec![5.0, 1.0, 4.0, 2.0, 3.0];
        let probs = vec![0.25, 0.75];
        
        let q = crate::descriptive::quantile(&data, &probs);
        
        // Assuming R's Type 7 interpolation: 
        // 25th percentile of 1:5 is 2.0, 75th percentile is 4.0
        let expected = col![2.0, 4.0];
        
        assert_col_eq(&q, &expected, 1e-9);
    }

    #[test]
    fn test_quantile_single_observation() {
        // A bucket with only 1 observation should return that observation 
        // for any requested probability, without triggering an out-of-bounds panic.
        let data = vec![42.0];
        let probs = vec![0.1, 0.5, 0.9];
        
        let q = crate::descriptive::quantile(&data, &probs);
        let expected = col![42.0, 42.0, 42.0];
        
        assert_col_eq(&q, &expected, 1e-9);
    }

    #[test]
    fn test_quantile_empty_slice() {
        // An empty slice should gracefully return a column of NaNs 
        // to prevent usize underflow panics.
        let data: Vec<f64> = vec![];
        let probs = vec![0.25, 0.75];
        
        let q = crate::descriptive::quantile(&data, &probs);
        
        assert_eq!(q.nrows(), 2);
        assert!(q[0].is_nan(), "Expected NaN for empty slice");
        assert!(q[1].is_nan(), "Expected NaN for empty slice");
    }
}

use extendr_api::prelude::*;

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
pub fn z_variance(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let mean = z_mean(x);
    let n = x.len() as f64;
    x.iter()
        .map(|&xi| (xi - mean).powi(2))
        .sum::<f64>() / (n - 1.0)
}

/// Compute the sample standard deviation of a numeric vector.
/// @param x A numeric vector.
/// @return The sample standard deviation as a double.
/// @export
#[extendr]
pub fn z_sd(x: &[f64]) -> f64 {
    z_variance(x).sqrt()
}

/// Compute the sample covariance of two numeric vectors.
/// @param x A numeric vector.
/// @param y A numeric vector of the same length.
/// @return The sample covariance as a double.
/// @export
#[extendr]
pub fn z_covariance(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let mean_x = z_mean(x);
    let mean_y = z_mean(y);
    let n = x.len() as f64;
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / (n - 1.0)
 }

 // Registering this module's functions
extendr_module! {
    mod descriptive;
    fn z_sum;
    fn z_mean;
    fn z_median;
    fn z_variance;
    fn z_sd;
    fn z_covariance;
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
        assert!((z_variance(&x) - 4.571428571428571).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_identical() {
        // cov(x, x) should equal var(x)
        let x = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((z_covariance(&x, &x) - z_variance(&x)).abs() < 1e-10);
    }
}





















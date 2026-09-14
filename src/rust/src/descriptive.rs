#![allow(non_snake_case)]

use extendr_api::prelude::*;
use faer::Col;

// Registering this module's functions
extendr_module! {
    mod descriptive;
    fn sum;
    fn mean;
    fn median;
    fn var;
    fn sd;
    fn cov;
    fn cor;
    fn quantile_r;
}

/// Compute the sum of a numeric vector using Neumaier summation
///
/// @description
/// This function computes the sum of a numeric vector using the
/// Neumaier summation algorithm for improved numerical stability. It tracks and
/// compensates for truncated floating point bits, preventing precision loss when
/// adding values with high magnitude variation or across large datasets.
///
/// @param x A numeric (double) vector.
/// @return The sum as a double.
/// @keywords internal
#[extendr]
pub fn sum(x: &[f64]) -> f64 {
    // Empty vector check
    if x.is_empty() {
        return 0.0;
    };

    // Neumaier summation algorithm
    let mut sum: f64 = 0.0;
    let mut c: f64 = 0.0;

    for &y in x {
        let t: f64 = sum + y;

        // Accumulating lost bits
        if sum.abs() >= y.abs() {
            c += (sum - t) + y;
        } else {
            c += (y - t) + sum;
        }
        sum = t;
    }

    sum + c
}

/// Compute the arithmetic mean of a numeric vector via Neumaier summation
///
/// @description
/// This function calculates the arithmetic mean of a numeric vector via the
/// Neumaier summation algorithm used in this package's `sum()` implementation.
/// It is highly resistant to floating-point rounding errors when summing vectors
/// with high magnitude ranges or across large datasets.
///
/// @param x A numeric (double) vector
/// @return The mean as a double.
/// @keywords internal
#[extendr]
pub fn mean(x: &[f64]) -> Option<f64> {
    if x.is_empty() {
        return None;
    }
    let n: f64 = x.len() as f64;

    Some(sum(x) / n)
}

/// Compute the median of a numeric vector.
///
/// @description
/// This function calculates the median of a numeric vector via an
/// introselect implementation based on the "ipnsort" algorithm by
/// Lukas Bergdoll and Orson Peters. The fallback algorithm is
/// Median of Medians using Tukey’s Ninther. Guarantees linear
/// runtime for all inputs compared to a O(N log N) full-sort.
///
/// @param x A numeric (double) vector.
/// @return The median as a double
/// @keywords internal
#[extendr]
pub fn median(x: &[f64]) -> Option<f64> {
    if x.is_empty() {
        return None;
    }
    if x.iter().any(|v| v.is_nan()) {
        return None;
    }

    let mut x_rs: Vec<f64> = x.to_vec();
    let n: usize = x.len();
    let k: usize = n / 2;
    let (less, upp_mid, _greater) = x_rs.select_nth_unstable_by(k, f64::total_cmp);

    if n % 2 == 0 {
        let low_mid: f64 = less.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some((low_mid + *upp_mid) / 2.0)
    } else {
        Some(*upp_mid)
    }
}

/// Compute the sample variance of a numeric vector
///
/// @description
/// Calculates the sample variance using Welford's online algorithm,
/// updating a running mean and sum of squared deviations in a
/// single pass. This avoids the catastrophic cancellation that affects
/// the textbook `sum(x^2) - sum(x)^2 / n` formulation, where both terms
/// grow with the square of the mean while their difference does not.
/// Relative precision is preserved even when the coefficient of
/// variation is small.
///
/// @param x A numeric (double vector of length 2 or greater.
/// @return The sample variance as a double, or `NA` is `x` has fewer
///   than two elements or contains `NA` or `NaN`.
/// @keywords internal
#[extendr]
pub fn var(x: &[f64]) -> Option<f64> {
    if x.len() < 2 {
        return None;
    }

    // Initializing variables for Welford's
    let mut k: f64 = 0.0;
    let mut Mk: f64 = 0.0;
    let mut Sk: f64 = 0.0;

    // Welford's algorithm
    for &xi in x {
        // Early return if any element is NaN
        if xi.is_nan() {
            return None;
        }

        k += 1.0;
        let dev_prev: f64 = xi - Mk;
        Mk += dev_prev / k;
        let dev_new: f64 = xi - Mk;
        Sk += dev_prev * dev_new;
    }

    let n: f64 = x.len() as f64;
    let var: f64 = Sk / (n - 1.0);
    Some(var)
}

/// Compute the sample standard deviation of a numeric vector.
///
/// @description
/// Calculates the sample standard deviation by taking the
/// square root of the variance, which uses Welford's algorithm.
///
/// @param x A numeric (double) vector of length 2 or greater.
/// @return The sample standard deviation as a double, or `NA` if `x`
///   has fewer than two elements or contains `NA` or `NaN`.
/// @keywords internal
#[extendr]
pub fn sd(x: &[f64]) -> Option<f64> {
    // Relying on `var()` to do Option checks
    var(x).map(|v| v.sqrt())
}

/// Compute the sample covariance of two numeric vectors.
///
/// @description
/// Calculates the sample covariance using Welford's online algorithm.
/// Co-deviations are updated in a single pass, preventing catastrophic
/// cancellation issues present in the naïve formulation.
///
/// @param x A numeric (double) vector.
/// @param y A numeric (double) vector of the same length.
/// @return The sample covariance as a double, or `NA` if the vectors
///   differ in length, have fewer than two elements, or contain `NA`/`NaN`.
/// @keywords internal
#[extendr]
pub fn cov(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    // Initializing variables for streaming iterator
    let mut k: f64 = 0.0;
    let mut Ck: f64 = 0.0;
    let mut Mx: f64 = 0.0;
    let mut My: f64 = 0.0;

    // Welford's algorithm
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        // Early return if any element is NaN
        if xi.is_nan() {
            return None;
        }

        // Deviations from *previous* mean
        let xdev: f64 = xi - Mx;
        let ydev: f64 = yi - My;

        // Update variables
        k += 1.0;
        Mx += xdev / k;
        My += ydev / k;
        // Asymmetric formulation, using new x deviation and old y dev
        Ck += (xi - Mx) * ydev;
    }

    let n: f64 = x.len() as f64;
    let cov: f64 = Ck / (n - 1.0);
    Some(cov)
}

/// Compute Pearson correlation coefficient
///
/// @description
/// Calculates the Pearson correlation coefficient via Welford's online algorithm
/// for the variance and covariance accumulators.
///
/// @param x A numeric (double) vector.
/// @param y A numeric (double) vector of the same length.
/// @return The sample correlation as a double bounded between -1.0 and 1.0,
///   or `NA` if the vectors differ in length, have fewer than two elements,
///   contain `NA`/`NaN`, or have zero variance.
/// @keywords internal
#[extendr]
pub fn cor(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let mut k: f64 = 0.0;
    let mut Mx: f64 = 0.0;
    let mut My: f64 = 0.0;
    let mut Sx: f64 = 0.0;
    let mut Sy: f64 = 0.0;
    let mut Ck: f64 = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        // Early return if any element is NaN
        if xi.is_nan() {
            return None;
        }

        let xdev_prev: f64 = xi - Mx;
        let ydev_prev: f64 = yi - My;

        // Update variables
        k += 1.0;
        Mx += xdev_prev / k;
        My += ydev_prev / k;

        let xdev: f64 = xi - Mx;
        let ydev: f64 = yi - My;

        Sx += xdev * xdev_prev;
        Sy += ydev * ydev_prev;
        Ck += xdev * ydev_prev;
    }

    let denom: f64 = (Sx * Sy).sqrt();
    if denom == 0.0 {
        return None;
    }
    let r: f64 = (Ck / denom).clamp(-1.0, 1.0);

    Some(r)
}

/// Compute sample quantiles for a numeric vector
///
/// @description
/// Calculates sample quantiles for the specified probabilities.
/// This implementation replicates R's default Type 7 continupus sample
/// quantile method (linear interpolation).
///
/// @param x A numeric (double) vector.
/// @param probs A numeric (double) vector of probabilities with values between 0 and 1.
/// @return A numeric (double) vector of calculated quantiles. Returns `NaN`
///   for any requested probability if `x` is empty.
/// @keywords internal
#[extendr(r_name = "quantile")]
pub fn quantile_r(x: &[f64], probs: &[f64]) -> Doubles {
    // Zero-copy slices passed directly to quantile()
    let quantiles_col: Col<f64> = quantile(x, probs);

    // Allocating the return vector to hand back to R
    quantiles_col.iter().collect::<Doubles>()
}

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
        assert_eq!(sum(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(sum(&[]), 0.0);
    }

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0]).unwrap() - 2.0).abs() < 1e-14);
        assert!(mean(&[]).is_none()); // Replaced .is_nan() with .is_none()
    }

    #[test]
    fn test_median_odd() {
        assert_eq!(median(&[3.0, 1.0, 2.0]).unwrap(), 2.0);
    }

    #[test]
    fn test_median_even() {
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
    }

    #[test]
    fn test_variance() {
        let x = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((var(&x).unwrap() - 4.571428571428571).abs() < 1e-14);
        assert!(var(&[1.0]).is_none()); // Checking length < 2 logic
    }

    #[test]
    fn test_sd() {
        let x = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((sd(&x).unwrap() - 2.138089935299395).abs() < 1e-14);
    }

    #[test]
    fn test_var_welford_stability() {
        // High mean, low variance. A naive sum(x^2) - sum(x)^2/n approach
        // suffers catastrophic cancellation here. Welford's handles it perfectly.
        let offset = 1e9;
        let x = vec![offset + 1.0, offset + 2.0, offset + 3.0];
        // Variance of [1, 2, 3] is exactly 1.0. Adding an offset shouldn't change it.
        assert!((var(&x).unwrap() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_covariance_identical() {
        let x = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((cov(&x, &x).unwrap() - var(&x).unwrap()).abs() < 1e-14);
        assert!(cov(&x, &[1.0, 2.0]).is_none()); // Mismatched lengths
    }

    #[test]
    fn test_covariance_welford_stability() {
        let offset = 1e9;
        let x = vec![offset + 1.0, offset + 2.0, offset + 3.0];
        let y = vec![offset + 3.0, offset + 2.0, offset + 1.0];
        // True covariance of [1,2,3] and [3,2,1] is -1.0.
        assert!((cov(&x, &y).unwrap() - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_cor_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((cor(&x, &x).unwrap() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_cor_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((cor(&x, &y).unwrap() - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_cor_zero_variance() {
        let x = vec![5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0];
        // cor() should return None to avoid division by zero when denom is 0.0
        assert!(cor(&x, &y).is_none());
    }

    #[test]
    fn test_cor_welford_stability() {
        let offset = 1e9;
        let x = vec![offset + 1.0, offset + 2.0, offset + 3.0];
        let y = vec![offset + 3.0, offset + 2.0, offset + 1.0];
        // True correlation is perfect negative (-1.0).
        assert!((cor(&x, &y).unwrap() - (-1.0)).abs() < 1e-14);
    }

    // ==========================================
    // Tests for quantile
    // ==========================================
    use faer::{col, Col};

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
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = vec![0.0, 0.5, 1.0];
        let q = crate::descriptive::quantile(&data, &probs);
        let expected = col![1.0, 3.0, 5.0];
        assert_col_eq(&q, &expected, 1e-14);
    }

    #[test]
    fn test_quantile_unsorted_data() {
        let data = vec![5.0, 1.0, 4.0, 2.0, 3.0];
        let probs = vec![0.25, 0.75];
        let q = crate::descriptive::quantile(&data, &probs);
        let expected = col![2.0, 4.0];
        assert_col_eq(&q, &expected, 1e-14);
    }

    #[test]
    fn test_quantile_single_observation() {
        let data = vec![42.0];
        let probs = vec![0.1, 0.5, 0.9];
        let q = crate::descriptive::quantile(&data, &probs);
        let expected = col![42.0, 42.0, 42.0];
        assert_col_eq(&q, &expected, 1e-14);
    }

    #[test]
    fn test_quantile_empty_slice() {
        let data: Vec<f64> = vec![];
        let probs = vec![0.25, 0.75];
        let q = crate::descriptive::quantile(&data, &probs);
        assert_eq!(q.nrows(), 2);
        assert!(q[0].is_nan(), "Expected NaN for empty slice");
        assert!(q[1].is_nan(), "Expected NaN for empty slice");
    }
}

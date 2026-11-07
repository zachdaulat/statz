//! Euclidean projection onto the probability simplex ($\Delta^{J-1}$) in $O(J \log J)$ time.
//!
//! Provides both allocating (`project_simplex`) and zero-allocation in-place
//! (`project_simplex_into`) routines.

use faer::{Col, ColMut, ColRef};

pub(crate) fn project_simplex_into(v: ColMut<f64>, buf: &mut Vec<f64>) {
    // 1. Reset buffer
    buf.clear();
    buf.extend(v.as_ref().iter().copied());

    // 2. Sorting buffer
    buf.sort_unstable_by(|a: &f64, b: &f64| b.total_cmp(a));

    // 3. Finding threshold tau
    let mut sum: f64 = 0.0;
    let mut tau: f64 = 0.0;

    for (k, &u) in buf.iter().enumerate() {
        sum += u;
        let tau_k = (sum - 1.0) / (k + 1) as f64;

        if u - tau_k <= 0.0 {
            break;
        }

        tau = tau_k
    }

    // 4. Applying tau to weights vector in-place
    for val in v.iter_mut() {
        *val = f64::max(*val - tau, 0.0);
    }
}

// Simplex projection helper function
// R-wrapper or calling function must ensure no NAs/NaNs in the input data
pub fn project_simplex(v: ColRef<f64>) -> Col<f64> {
    let j: usize = v.nrows();

    // 1. Sort weights vector in descending order, u = sorted weights vector
    let mut u: Vec<f64> = v.iter().copied().collect::<Vec<f64>>();
    u.sort_unstable_by(|a: &f64, b: &f64| b.total_cmp(a));

    // 2. Initializing loop state
    let mut sum: f64 = 0.0;
    let mut tau: f64 = 0.0;

    // 3. Finding threshold tau
    for (k, &u_k) in u.iter().enumerate() {
        sum += u_k;
        let tau_k = (sum - 1.0) / (k + 1) as f64;

        if u_k - tau_k <= 0.0 {
            break;
        };

        tau = tau_k;
    }

    // 4. Applying tau to unsorted weights vector
    Col::from_fn(j, |i: usize| f64::max(v[i] - tau, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::dsc::test_fixtures::assert_col_eq;
    use faer::{col, col::AsColRef};

    // Helper macro to assert float equality
    macro_rules! assert_col_approx_eq {
        ($left:expr, $right:expr, $tol:expr) => {
            assert_eq!($left.nrows(), $right.nrows());
            for i in 0..$left.nrows() {
                assert!(
                    ($left[i] - $right[i]).abs() < $tol,
                    "Mismatch at index {}: {} != {}",
                    i,
                    $left[i],
                    $right[i]
                );
            }
        };
    }

    // ==========================================
    // Tests for project_simplex
    // ==========================================

    #[test]
    fn test_project_simplex_already_valid() {
        // A vector already on the simplex (sums to 1, all >= 0) should remain unchanged
        let v = col![0.2, 0.3, 0.5];
        let projected = project_simplex(v.as_col_ref());

        assert_col_eq(&projected, &v, 1e-9);
    }

    #[test]
    fn test_project_simplex_clamping() {
        // A vector with negative values and sum > 1 should clamp negatives to 0
        // and adjust the rest to sum to 1.
        // [1.2, -0.2] projects exactly to [1.0, 0.0]
        let v = col![1.2, -0.2];
        let expected = col![1.0, 0.0];
        let projected = project_simplex(v.as_col_ref());

        assert_col_eq(&projected, &expected, 1e-9);
    }

    #[test]
    fn test_project_simplex_uniform_shift() {
        // A vector with valid proportions but sum > 1 should uniformly shift down
        // [0.8, 0.4] (sum = 1.2) should shift down by exactly 0.1 each
        let v = col![0.8, 0.4];
        let expected = col![0.7, 0.3];
        let projected = project_simplex(v.as_col_ref());

        assert_col_eq(&projected, &expected, 1e-9);
    }

    // ==========================================
    // Tests for project_simplex_into
    // ==========================================

    #[test]
    fn test_simplex_basic_projection() {
        let mut v = col![0.4, 0.8, -0.2];
        let mut buf = Vec::new();

        project_simplex_into(v.as_mut(), &mut buf);

        // The expected projection shifts the positive elements down equally
        // to sum to 1, and clamps the negative element to 0.
        let expected = col![0.3, 0.7, 0.0];
        assert_col_approx_eq!(v, expected, 1e-12);
    }

    #[test]
    fn test_simplex_dirty_buffer_reuse() {
        let mut v = col![2.0, 2.0, 2.0];

        // Simulating a worker thread buffer that has leftover garbage data
        // from a previous bootstrap iteration, including a different length.
        let mut buf = vec![99.9, -100.0, f64::NAN, 42.0, 0.0];

        project_simplex_into(v.as_mut(), &mut buf);

        // If `buf.clear()` works correctly, the garbage data is ignored.
        // A uniform input > 1 should project strictly to a uniform simplex.
        let expected = col![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        assert_col_approx_eq!(v, expected, 1e-12);
    }

    #[test]
    fn test_simplex_already_valid() {
        let mut v = col![0.2, 0.5, 0.3];
        let mut buf = Vec::new();

        project_simplex_into(v.as_mut(), &mut buf);

        // If the vector already sums to 1 and is non-negative,
        // the projection should leave it entirely unchanged.
        let expected = col![0.2, 0.5, 0.3];
        assert_col_approx_eq!(v, expected, 1e-12);
    }

    #[test]
    fn test_simplex_all_negative() {
        let mut v = col![-2.0, -1.0, -3.0];
        let mut buf = Vec::new();

        project_simplex_into(v.as_mut(), &mut buf);

        // If all elements are negative, the algorithm must select the least
        // negative element (-1.0) and push it to 1.0, clamping the rest to 0.0.
        let expected = col![0.0, 1.0, 0.0];
        assert_col_approx_eq!(v, expected, 1e-12);
    }
}

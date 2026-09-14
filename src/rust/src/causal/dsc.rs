use crate::ext::RMatrixExt;
use extendr_api::{prelude::*, Error};
use faer::{
    col::AsColRef, linalg::solvers::SelfAdjointEigen, mat::AsMatRef, Col, ColMut, ColRef, Mat,
    MatRef, Side,
};
use std::f64;

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
    // Import everything from the parent module
    use super::*;
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
}

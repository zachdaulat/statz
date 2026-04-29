use std::f64;

use extendr_api::prelude::*;
use faer::{
    linalg::solvers::{Llt, Solve},
    mat::AsMatRef,
    Col, ColRef, Mat, MatRef, Side,
};

extendr_module! {
    mod linear_models;
    fn z_lm_naive;
}

// ------------ EXTENDR INTERFACES -------------
// R-Rust Extendr interface function
/// Internal R wrapper for the naïve Cholesky OLS solver
///
/// Handles R <-> Rust type translation. Dispatches the numeric design matrix
/// and response vector to the Rust Cholesky engine. Returns a list containing
/// coefficients, standard errors, fitted values, residuals, residual degrees of
/// freedom, and residual standard deviation.
///
/// @export
/// @keywords internal
#[extendr]
pub fn z_lm_naive(x: RMatrix<f64>, y: Doubles) -> extendr_api::Result<List> {
    // --- 1. Converting & constructing intputs
    let x_ref: MatRef<f64> = rmatrix_to_matref(&x);
    let y_ref: ColRef<f64> = doubles_to_colref(&y);

    // --- 2. Compute results
    let result: LmResult = lm_cholesky(x_ref, y_ref).map_err(Error::Other)?;

    Ok(list!(
        coefficients = result.theta.iter().collect::<Doubles>(),
        std_errors = result.std_errors.iter().collect::<Doubles>(),
        fitted_values = result.fitted.iter().collect::<Doubles>(),
        residuals = result.resid.iter().collect::<Doubles>(),
        df_residual = result.df,
        sigma = result.sigma
    ))
}

// ------------ RUST ENGINES -------------------

// --- Rust structs standardiizng outputs
// OLS Result struct
// Returns:
// - coefficients
// - fitted values: y_hat = mat * theta
// - residuals: y - y_hat
// - residual variance: var = residuals norm^2 / (n - p)
pub(crate) struct LmResult {
    pub theta: Col<f64>,
    pub std_errors: Col<f64>,
    pub fitted: Col<f64>,
    pub resid: Col<f64>,
    pub df: f64,
    pub sigma: f64,
}

// --- Rust engines
// OLS engine using Cholesky factorisation
pub(crate) fn lm_cholesky(x_mat: MatRef<f64>, y_col: ColRef<f64>) -> Result<LmResult, String> {
    // --- 1. Preparing inputs ---
    let n: usize = x_mat.nrows();
    let p: usize = x_mat.ncols();
    let df: f64 = (n - p) as f64;
    let identity: Mat<f64> = Mat::identity(p, p);

    // --- 2. Computing intermediates ---
    let xtx: Mat<f64> = x_mat.transpose() * x_mat;
    let xty: Col<f64> = x_mat.transpose() * y_col;

    // --- 3. Cholesky Factorisation
    let llt: Llt<f64> = Llt::new(xtx.as_mat_ref(), Side::Lower).map_err(|_| {
        Error::Other("X'X is not positive definite (likely rank-deficient design matrix)".into())
    })?;

    // --- 4. Solve & compute results ---
    let theta: Col<f64> = llt.solve(xty);
    let y_hat: Col<f64> = x_mat * &theta;
    let resid: Col<f64> = y_col - &y_hat;
    let sigma: f64 = (resid.squared_norm_l2() / df).sqrt();

    // --- 5. Computing standard errors ---
    let xtx_inv: Mat<f64> = llt.solve(&identity);
    let std_errors: Col<f64> = (0..p)
        .map(|i| sigma * xtx_inv[(i, i)].sqrt())
        .collect::<Col<f64>>();

    // Collecting outputs
    Ok(LmResult {
        theta,
        std_errors,
        fitted: y_hat,
        resid,
        df,
        sigma,
    })
}

// ---------- Helper Functions -----------

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

// -------------------- TESTS ------------------------
// Rust-side unit tests
#[allow(clippy::excessive_precision)]
#[allow(clippy::manual_range_contains)]
#[cfg(test)]
mod tests {
    use super::*;
    // mat and col, faer's macros for building matrices and columns manually
    use faer::{col, col::AsColRef, mat, mat::AsMatRef};

    #[test]
    fn test_lm_cholesky_simple() {
        // y = 2x_1 + 3x_2
        let x = mat![[1.0, 2.0], [1.0, 4.0], [1.0, 6.0],];
        // Exact fit, no noise
        let y = col![5.0, 9.0, 13.0];

        let res = lm_cholesky(x.as_mat_ref(), y.as_col_ref()).unwrap();

        // Check coefficients
        assert!((res.theta[0] - 1.0).abs() < 1e-10); // Intercept = 1
        assert!((res.theta[1] - 2.0).abs() < 1e-10); // Slope = 2

        // Exact fit means sigma should be functionally zero
        assert!(res.sigma < 1e-10);
    }
}

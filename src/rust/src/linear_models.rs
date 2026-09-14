use crate::ext::{FaerColExt, FaerMatExt, RMatrixExt};
use extendr_api::prelude::*;
use faer::{
    diag::DiagRef,
    linalg::{
        solvers::{Llt, Qr, SelfAdjointEigen, Solve, SolveLstsq, Svd},
        triangular_inverse::*,
    },
    mat::AsMatRef,
    Col, ColRef, Mat, MatRef, Par, Side,
};
use std::f64;

extendr_module! {
    mod linear_models;
    fn lm_r;
    fn eigen;
    fn svd;
}

// ------------ EXTENDR INTERFACES -------------
/// Internal OLS solver
///
/// @description
/// Handles R <-> Rust FFI translation and engine dispatching for
/// OLS linear modelling.
///
/// @param x A numeric design matrix.
/// @param y A numeric response vector
/// @param engine A character string specifying the backend ("cholesky" or "qr").
/// @return A list containing coefficients, standard errors, fitted values,
///   residuals, residual degrees of freedom, and residual standard deviation.
///
/// @keywords internal
#[extendr(r_name = "lm_rs")]
pub fn lm_r(x: RMatrix<f64>, y: Doubles, engine: &str) -> extendr_api::Result<List> {
    // 1. Get faer ref views to R data
    let x_ref: MatRef<f64> = x.as_mat_ref();
    let y_ref: ColRef<f64> = y.as_col_ref();

    // 2. Dispatch to correct Rust engine
    #[rustfmt::skip]
    let result: LmResult = match engine {
        "cholesky" => lm_chol(x_ref, y_ref).map_err(Error::Other)?,
        "qr"       => lm_qr(x_ref, y_ref).map_err(Error::Other)?,
        "svd"      => return Err(Error::Other("SVD engine not yet implemented".into())),
        _          => return Err(Error::Other("Invalid engine selection".into())),
    };

    // 3. Returning formatted list
    Ok(list!(
        coefficients = result.theta.iter().collect::<Doubles>(),
        std_errors = result.std_errors.iter().collect::<Doubles>(),
        fitted_values = result.fitted.iter().collect::<Doubles>(),
        residuals = result.resid.iter().collect::<Doubles>(),
        df_residual = result.df,
        sigma = result.sigma
    ))
}

/// Eigendecomposition via `faer`
///
/// @description
/// Replicates `base::eigen()` for symmetric matrices, providing an interface to
/// the Rust-native `faer` library instead of LAPACK.
///
/// @param x A numeric symmetric matrix
/// @return A list containig the eigenvalues (`values`) and eigenvectors (`vectors`).
/// @export
#[extendr]
pub fn eigen(x: RMatrix<f64>) -> extendr_api::Result<List> {
    // Instantiate eigendecomposition from RMatrix
    let eigen: SelfAdjointEigen<f64> = SelfAdjointEigen::new(x.as_mat_ref(), Side::Lower)
        .map_err(|_| Error::Other("Error during eigendecomposition".into()))?;

    // Extracting views of the values and vectors from the eigendecomposition
    let eigenvalues: DiagRef<f64> = eigen.S();
    let eigenvectors: MatRef<f64> = eigen.U();

    // Converting DiagRef to ColRef view with .columne_vector() so its iterable
    Ok(list!(
        values = eigenvalues.column_vector().iter().collect::<Doubles>(),
        vectors = eigenvectors.as_rmatrix()
    ))
}

/// Singular Value Decomposition via faer
///
/// @description
/// Replicates `base::svd()`, providing an interface to the Rust-native `faer` library
/// instead of LAPACK.
///
/// @param x A numeric matrix.
/// @return A list containing the singular values (`d`), left singular vectors (`u`),
///   and right singular vectors (`v`).
/// @export
#[extendr]
pub fn svd(x: RMatrix<f64>) -> extendr_api::Result<List> {
    // Instantiate singular value decomposition from RMatrix
    let svd: Svd<f64> = Svd::new(x.as_mat_ref())
        .map_err(|_| Error::Other("Error during singular value decomposition".into()))?;

    // Extracting views of the singular values and the U and V factors
    let singular_values: DiagRef<f64> = svd.S();
    let u: MatRef<f64> = svd.U();
    let v: MatRef<f64> = svd.V();

    // Converting DiagRef to ColRef view with .columne_vector() so its iterable
    Ok(list!(
        d = singular_values.column_vector().iter().collect::<Doubles>(),
        u = u.as_rmatrix(),
        v = v.as_rmatrix()
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
pub(crate) fn lm_chol(x_mat: MatRef<f64>, y_col: ColRef<f64>) -> Result<LmResult, String> {
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

// OLS engine using QR decomposition
pub(crate) fn lm_qr(x_mat: MatRef<f64>, y_col: ColRef<f64>) -> Result<LmResult, String> {
    // --- 1. Preparing inputs ---
    let n: usize = x_mat.nrows();
    let p: usize = x_mat.ncols();
    let df: f64 = (n - p) as f64;

    // --- 2. QR Factorisation of X ---
    let qr: Qr<f64> = x_mat.qr();

    // --- 3. Solve least-squares problem directly
    // This computes theta = argmin ||X*theta - y||
    // Internally, applies Q^T to y, then back-substitutes against R
    let theta: Col<f64> = qr.solve_lstsq(y_col);

    // --- 4. Fitted values and residuals
    let y_hat: Col<f64> = x_mat * &theta;
    let resid: Col<f64> = y_col - &y_hat;
    let sigma: f64 = (resid.squared_norm_l2() / df).sqrt();

    // --- 5. Standard errors via R^{-1} ---
    // Get R from the QR factorisation
    let r_ref: MatRef<'_, f64> = qr.thin_R();

    // Explicitly allocate new mutable p x p Mat of zeroes to hold inverse
    let mut r_inv: Mat<f64> = Mat::zeros(p, p);

    // Compute upper triangular inverse
    invert_upper_triangular(r_inv.as_mut(), r_ref, Par::Seq);

    // diag((X'X)^{-1})[i] = sum over k of (R^{-1})[i,k]^2
    let std_errors: Col<f64> = (0..p)
        .map(|i| {
            let row_norm_sq: f64 = (i..p).map(|k| r_inv[(i, k)].powi(2)).sum();
            sigma * row_norm_sq.sqrt()
        })
        .collect::<Col<f64>>();

    Ok(LmResult {
        theta,
        std_errors,
        fitted: y_hat,
        resid,
        df,
        sigma,
    })
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

        let res = lm_chol(x.as_mat_ref(), y.as_col_ref()).unwrap();

        // Check coefficients
        assert!((res.theta[0] - 1.0).abs() < 1e-10); // Intercept = 1
        assert!((res.theta[1] - 2.0).abs() < 1e-10); // Slope = 2

        // Exact fit means sigma should be functionally zero
        assert!(res.sigma < 1e-10);
    }

    #[test]
    fn test_lm_qr_simple() {
        // y = 2x_1 + 3x_2
        let x = mat![[1.0, 2.0], [1.0, 4.0], [1.0, 6.0],];
        // Exact fit, no noise
        let y = col![5.0, 9.0, 13.0];

        // Call the new QR engine
        let res = lm_qr(x.as_mat_ref(), y.as_col_ref()).unwrap();

        // Check coefficients
        assert!((res.theta[0] - 1.0).abs() < 1e-10); // Intercept = 1
        assert!((res.theta[1] - 2.0).abs() < 1e-10); // Slope = 2

        // Exact fit means sigma should be functionally zero
        assert!(res.sigma < 1e-10);
    }
}

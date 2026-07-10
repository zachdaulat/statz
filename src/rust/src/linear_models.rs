use std::f64;
use extendr_api::prelude::*;
use faer::{
    linalg::{solvers::{Llt, Solve, Qr, SolveLstsq, SelfAdjointEigen, Svd}, triangular_inverse::*},
    diag::DiagRef,
    mat::AsMatRef,
    Col, ColRef, Mat, MatRef, Side, Par,
};

extendr_module! {
    mod linear_models;
    fn z_lm_chol;
    fn z_lm_qr;
    fn z_eigen;
    fn z_svd;
}

// ------------ EXTENDR INTERFACES -------------
// R-Rust Extendr interface function
/// Internal Cholesky solver wrapper handling R <-> Rust type translation 
/// bridging extendr interface with R and internal Rust-level faer types.
///
/// Dispatches the numeric design matrix and response vector to the Rust 
/// Cholesky engine. Returns a list containing coefficients, standard errors, 
/// fitted values, residuals, residual degrees of freedom, and residual 
/// standard deviation.
///
/// @export
/// @keywords internal
#[extendr]
pub fn z_lm_chol(x: RMatrix<f64>, y: Doubles) -> extendr_api::Result<List> {
    // --- 1. Extract faer Ref views to R data
    // Uses new faer Ext traits
    let x_ref: MatRef<f64> = x.as_mat_ref();
    let y_ref: ColRef<f64> = y.as_col_ref();

    // --- 2. Compute results
    let result: LmResult = lm_chol(x_ref, y_ref).map_err(Error::Other)?;

    Ok(list!(
        coefficients = result.theta.iter().collect::<Doubles>(),
        std_errors = result.std_errors.iter().collect::<Doubles>(),
        fitted_values = result.fitted.iter().collect::<Doubles>(),
        residuals = result.resid.iter().collect::<Doubles>(),
        df_residual = result.df,
        sigma = result.sigma
    ))
}

/// Dispatches the numeric design matrix and response vector to the Rust 
/// QR decomposition-based OLS engine. Returns a list containing coefficients,
/// standard errors, fitted values, residuals, residual degrees of freedom,
/// and residual standard deviation
/// 
/// @export
/// @keywords internal
#[extendr]
pub fn z_lm_qr(x: RMatrix<f64>, y: Doubles) -> extendr_api::Result<List> {
    // --- 1. Extract faer Ref views to R data
    // Uses new faer Ext traits
    let x_ref: MatRef<f64> = x.as_mat_ref();
    let y_ref: ColRef<f64> = y.as_col_ref();

    // --- 2. Compute results
    let result: LmResult = lm_qr(x_ref, y_ref).map_err(Error::Other)?;

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

// Linear algebra interfaces in R to the `faer` implementations

/// An R interface to Eigendecomposition performed by `faer` in Rust. 
/// Essentially replicates `base::eigen()` for symmetric matrices but 
/// using the Rust-native `faer` utilities instead of LAPACK.
/// 
/// @export
#[extendr]
pub fn z_eigen(x: RMatrix<f64>) -> extendr_api::Result<List> {
    // Instantiate eigendecomposition from RMatrix
    let eigen: SelfAdjointEigen<f64> = SelfAdjointEigen::new(x.as_mat_ref(), faer::Side::Lower).map_err(|_| {
        Error::Other("Error during eigendecomposition".into())
    })?;

    // Extracting views of the values and vectors from the eigendecomposition
    let eigenvalues: DiagRef<f64> = eigen.S();
    let eigenvectors: MatRef<f64> = eigen.U();

    // Converting DiagRef to ColRef view with .columne_vector() so its iterable
    Ok(list!(
        values = eigenvalues.column_vector().iter().collect::<Doubles>(),
        vectors = eigenvectors.as_rmatrix()
    ))
}

/// An R interface to Singular Value Decomposition performed by `faer` in Rust.
/// Essentially replicates `base::svd()` but using the Rust-native 
/// `faer` utilities instead of LAPACK.
/// 
/// @export
#[extendr]
pub fn z_svd(x: RMatrix<f64>) -> extendr_api::Result<List> {
    // Instantiate singular value decomposition from RMatrix
    let svd: Svd<f64> = Svd::new(x.as_mat_ref()).map_err(|_| {
        Error::Other("Error during singular value decomposition".into())
    })?;

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
            let row_norm_sq: f64 = (i..p)
                .map(|k| r_inv[(i, k)].powi(2))
                .sum();
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

// ---------- Extension Traits for R types with faer -----------

// Defining extension traits bridging R and faer types
pub(crate) trait FaerMatExt {
    fn as_mat_ref(&self) -> MatRef<'_, f64>;
}

pub(crate) trait FaerColExt {
    fn as_col_ref(&self) -> ColRef<'_, f64>;
}

pub(crate) trait RMatrixExt {
    fn as_rmatrix(&self) -> RMatrix<f64>;
}

// pub(crate) trait RVectorExt {
//     fn as_doubles(&self) -> Doubles;
// }

// Implement for RMatrix
impl FaerMatExt for RMatrix<f64> {
    // RMatrix to faer Mat conversion function
    // Zero-copy view of RMatrix
    fn as_mat_ref(&self) -> MatRef<'_, f64> {
        let nrows: usize = self.nrows();
        let ncols: usize = self.ncols();
        let data: &[f64] = self.data(); // Correctly returns &[f64]

        // Explicitly use MatRef, not Mat
        MatRef::from_column_major_slice(data, nrows, ncols)
    }
}

// Implement for Doubles
impl FaerColExt for Doubles {
    // Doubles to faer Col conversion function
    // Zero-copy view of Doubles
    fn as_col_ref(&self) -> ColRef<'_, f64> {
        // Drop down to the underlying Robj to extract the raw f64 slice safely
        let data: &[f64] = self
            .as_robj()
            .as_real_slice()
            .expect("Vector must be standard real numbers");

        // Explicitly use ColRef, not Col
        ColRef::from_slice(data)
    }
}

impl RMatrixExt for MatRef<'_, f64> {
    fn as_rmatrix(&self) -> RMatrix<f64> {
        let nrows: usize = self.nrows();
        let ncols: usize = self.ncols();

        // Using new_matrix to dynamically allocate and populate the R matrix.
        // It iterates over the dimensions, calling the closure to pull the 
        // (r, c) value from the faer MatRef
        RMatrix::new_matrix(nrows, ncols, |r, c| self[(r, c)])
    }
}

// impl RVectorExt for DiagRef<'_, f64> {
//     fn as_doubles(&self) -> Doubles {
        
//     }
// }

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

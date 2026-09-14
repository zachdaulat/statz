use extendr_api::prelude::*;
use faer::{ColRef, MatRef};
use std::f64;

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

#![allow(unused)]

use std::f64;
use extendr_api::prelude::*;
use faer::{MatRef, ColRef};

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

//! Pre-allocated thread-local matrix and vector workspaces (`DscScratch`, `ResampleScratch`).
//!
//! Isolates heap allocations outside hot optimization and bootstrap resampling loops.

use crate::causal::panel::{QuantilePanel, ResamplePanel};
use faer::{Col, Mat};

// ------------------- Scratch buffer structs ----------------
// Might add boolean vector scratchpad for things like checking duplicates, like a
// refactor of `DscError::DuplicateDonor` check if used with tens of thousands of donors
#[rustfmt::skip]
pub(crate) struct ResampleScratch {
    pub resample: ResamplePanel,
    pub quantile: QuantilePanel,
    pub dsc:      DscScratch,
    pub pad:      Vec<f64>,
}

#[rustfmt::skip]
pub(crate) struct DscScratch {
    pub donors: Mat<f64>,
    pub gram:   Mat<f64>,
    pub cross:  Col<f64>,
    pub w:      Col<f64>,
    pub w_new:  Col<f64>,
    pub w_buf:  Vec<f64>,
}

impl DscScratch {
    /// Allocatest the thread-local workspace using the global maximum dimensions.
    /// `max_donors` is typically `qp.n_units - 1`.
    #[rustfmt::skip]
    pub fn new(n_probs: usize, max_donors: usize) -> Self {
        Self {
            donors: Mat::zeros(n_probs, max_donors),    // Donor quantile matrix, Q x (J-1)
            gram:   Mat::zeros(max_donors, max_donors), // Gram matrix, (J-1)^2
            cross:  Col::zeros(max_donors),   // J-1 cross validation vector
            w:      Col::zeros(max_donors),   // Weights vector buffer (J-1)
            w_new:  Col::zeros(max_donors),   // Weights vector buffer (J-1)
            w_buf:  Vec::with_capacity(max_donors), // Simplex sorting buffer (J-1)
        }
    }
}

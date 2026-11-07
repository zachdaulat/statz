//! Distributional Synthetic Controls (DSC) estimation, spectral diagnostics, and inference.
//!
//! Re-exports the core Wasserstein ($W_2$) optimization engine, workspace buffers,
//! model representations, and resampling routines.

#![allow(unused)]
#![allow(non_snake_case)]

// child submodules
mod bootstrap;
mod config;
mod diagnostics;
mod error;
mod fit;
mod model;
mod placebo;
mod scratch;
mod simplex;

// Reexportiung public API (facade pattern)
pub use bootstrap::*;
pub use config::*;
pub use diagnostics::*;
pub use error::*;
pub use fit::*;
pub use model::*;
pub use placebo::*;
pub use scratch::*;
pub use simplex::*;

// // Re-export public & crate-internal API
// pub use bootstrap::*;
// pub use config::{BootSpec, DscSpec, LocParams, PlaceboSpec};
// pub use diagnostics::{dsc_diagnostics, DscDiagnostics};
// pub use error::DscError;
// pub use fit::dsc_fit;
// pub(crate) use fit::{dsc_check, DscFit};
// pub use model::*;
// pub use placebo::*;
// pub(crate) use scratch::{DscScratch, ResampleScratch};
// pub use simplex::project_simplex;
// pub(crate) use simplex::project_simplex_into;

// Shared test fixtures accessible across all `dsc/*` submodules
#[cfg(test)]
pub(crate) mod test_fixtures {
    use crate::causal::panel::{LocAdj, LocStat, QuantilePanel};
    use faer::Col;

    /// Helper function for safe floating-point comparison of faer columns
    pub fn assert_col_eq(a: &Col<f64>, b: &Col<f64>, tol: f64) {
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

    /// Generates a strictly controlled 4-unit, 2-period, 2-quantile panel.
    /// Target = Unit 0. Donors = Units 1, 2, 3.
    pub fn mock_panel() -> QuantilePanel {
        let n_periods: usize = 2;
        let probs: Vec<f64> = vec![0.25, 0.75];
        let n_units: usize = 4;
        let q: usize = probs.len();

        let mut values: Vec<f64> = Vec::with_capacity(n_periods * n_units * q);

        // Loop over periods to populate the flat column-major structure
        for _ in 0..n_periods {
            // Unit 0 (Target): Arbitrary data, not used for Gram matrix computation
            values.extend_from_slice(&[10.0, 20.0]);

            // Unit 1 (Donor 1)
            values.extend_from_slice(&[1.0, 2.0]);

            // Unit 2 (Donor 2)
            values.extend_from_slice(&[3.0, 4.0]);

            // Unit 3 (Donor 3)
            values.extend_from_slice(&[5.0, 6.0]);
        }

        QuantilePanel {
            values,
            probs,
            n_units,
            n_periods,
            adj: LocAdj::None,
            stat: LocStat::None,
            locations: Col::<f64>::full(n_units, 1.0),
        }
    }

    /// Generates a panel where the Target unit is a perfect linear combination
    /// of Donor 1 and Donor 2: Target = (0.3 * Donor 1) + (0.7 * Donor 2).
    /// Generates unique data across periods to ensure a full-rank, strictly convex Gram matrix.
    pub fn mock_panel_linear_combo() -> QuantilePanel {
        let n_periods = 2;
        let probs = vec![0.25, 0.75];
        let n_units = 4;
        let q = probs.len();

        let mut values = Vec::with_capacity(n_periods * n_units * q);

        // Period 0
        // D1: [1.0, 2.0], D2: [3.0, 4.0], D3: [5.0, 6.0]
        // Target: 0.3 * D1 + 0.7 * D2 = [2.4, 3.4]
        values.extend_from_slice(&[2.4, 3.4]);
        values.extend_from_slice(&[1.0, 2.0]);
        values.extend_from_slice(&[3.0, 4.0]);
        values.extend_from_slice(&[5.0, 6.0]);

        // Period 1 (Linearly independent from Period 0)
        // D1: [2.0, 1.0], D2: [4.0, 5.0], D3: [1.0, 8.0]
        // Target: 0.3 * D1 + 0.7 * D2 = [3.4, 3.8]
        values.extend_from_slice(&[3.4, 3.8]);
        values.extend_from_slice(&[2.0, 1.0]);
        values.extend_from_slice(&[4.0, 5.0]);
        values.extend_from_slice(&[1.0, 8.0]);

        QuantilePanel {
            values,
            probs,
            n_units,
            n_periods,
            adj: LocAdj::None,
            stat: LocStat::None,
            locations: Col::<f64>::full(n_units, 1.0),
        }
    }
}

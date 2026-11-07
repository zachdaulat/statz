//! Dense 3D tensor (`QuantilePanel`) storing evaluated empirical quantile functions ($T \times U \times Q$).
//!
//! Provides zero-copy `faer::MatRef` period views and in-place location adjustments
//! (translation shift and scaling).

use super::{LocAdj, LocStat, Panel, PanelError};
use faer::{col::AsColMut, reborrow::ReborrowMut, Col, ColRef, MatMut, MatRef};
use std::slice::ChunksExactMut;

/// A dense tensor containing the evaluated quantile functions by cell.
///
/// This struct organizes a strictly uniform three-dimensional layout ($$T \times U \times Q)
/// ensuring structural conformability for downstream matrix operations.
#[derive(Debug)]
pub struct QuantilePanel {
    /// Dense vector of length `n_periods * n_units * probs.len()`
    pub values: Vec<f64>,
    /// The probability grid used to evaluate these quantiles
    pub probs: Vec<f64>,
    pub n_units: usize,
    pub n_periods: usize,
    /// Tracks which location adjustment was applied (or not) to the `values` field's data
    pub adj: LocAdj,
    /// Tracks which location statistic is stored in `locations` field for each unit
    pub stat: LocStat,
    pub locations: Col<f64>,
}

impl Panel for QuantilePanel {
    fn n_units(&self) -> usize {
        self.n_units
    }
    fn n_periods(&self) -> usize {
        self.n_periods
    }
    fn cell(&self, u: usize, t: usize) -> &[f64] {
        let start: usize = (t * self.n_units + u) * self.probs.len();
        let end: usize = start + self.probs.len();

        &self.values[start..end]
    }
    fn cell_range(&self, u: usize, t: usize) -> std::ops::Range<usize> {
        let start: usize = (t * self.n_units + u) * self.probs.len();
        let end: usize = start + self.probs.len();

        start..end
    }
    fn period(&self, t: usize) -> &[f64] {
        let start: usize = t * self.n_units * self.probs.len();
        let end: usize = start + (self.n_units * self.probs.len());

        &self.values[start..end]
    }
    fn period_range(&self, t: usize) -> std::ops::Range<usize> {
        let start: usize = t * self.n_units * self.probs.len();
        let end: usize = start + (self.n_units * self.probs.len());

        start..end
    }
}

impl QuantilePanel {
    // MatRef view of a single period
    /// Constructs a zero-allocation `faer::MatRef` reference for a specific period.
    ///
    /// Maps the underlying column-major flat vector for period `t` into a `MatRef`.
    /// The resulting matrix has dimensions $Q \times U$, where rows represent probabilities
    /// and column represent the observational units.
    pub fn mat_ref(&self, t: usize) -> MatRef<'_, f64> {
        MatRef::from_column_major_slice(self.period(t), self.probs.len(), self.n_units)
    }

    /// Constructs `faer::MatRef` references for all periods in the panel.
    ///
    /// Iterates over all periods to return a vector of `MatRef` views. While this allocates
    /// a vector for the references, the underlying matrix data remains uncopied and is read
    /// directly from the contiguous `QuantilePanel` memory.
    pub fn mat_ref_all(&self) -> Vec<MatRef<'_, f64>> {
        (0..self.n_periods)
            .map(|t| self.mat_ref(t))
            .collect::<Vec<MatRef<f64>>>()
    }

    /// Extracts the evaluated quantiles for a subset of units in a specific period
    /// and copies them into a mutable matrix view.
    ///
    /// The resulting matrix is constructed column-by-column in the exact orders specified
    /// but the `units` slice. Operates entirely in-place.
    pub fn select_units_into(&self, units: &[usize], t: usize, mut out: MatMut<'_, f64>) {
        for (i, &u) in units.iter().enumerate() {
            // Get the 1D slice from contigupus memory
            let cell: &[f64] = self.cell(u, t); // `QuantilePanel` cells must all be same length

            // Wrap the slice in a faer column view and copy it into the matrix buffer
            out.rb_mut().col_mut(i).copy_from(ColRef::from_slice(cell));
        }
    }

    /// Computes unit means from any empirical panel and safely updates the internal state.
    ///
    /// Accepts any struct implementing the `Panel` trait (e.g., `ObsPanel`, `ResamplePanel`)
    /// to guarantee the statistics are calculated on the raw empirical observations.
    pub fn means_from(&mut self, panel: &impl Panel) -> Result<(), PanelError> {
        // 0. Unit and location length mismatches validation check
        if panel.n_units() != self.n_units || self.locations.nrows() != self.n_units {
            return Err(PanelError::UnitLengthMismatch {
                expected: self.n_units,
                actual: panel.n_units(),
            });
        };

        // 1. Dispatch to the zero-allocation `Panel` trait method
        panel.mean_all_into(self.locations.as_col_mut());

        // 2. Lock `QuantilePanel` state with the mutation
        self.stat = LocStat::Mean;
        Ok(())
    }

    /// Computes unit medians from any empirical panel and safely updates the internal state.
    pub fn medians_from(
        &mut self,
        panel: &impl Panel,
        pad: &mut Vec<f64>,
    ) -> Result<(), PanelError> {
        // 0. Unit and location length mismatches validation check
        if panel.n_units() != self.n_units || self.locations.nrows() != self.n_units {
            return Err(PanelError::UnitLengthMismatch {
                expected: self.n_units,
                actual: panel.n_units(),
            });
        };

        // 1. Dispatch to the minimal-allocation `Panel` trait method
        panel.median_all_into(self.locations.as_col_mut(), pad);

        // 2. Lock `QuantilePanel` state with the mutation
        self.stat = LocStat::Median;
        Ok(())
    }

    /// Applies a translation location adjustment in-place across all units.
    ///
    /// Accepts a `faer::Col<f64>` vector of shifts and subtracts the corresponding unit's
    /// shift value from each evalauted quantile. Operates in-place over mutable chunks to
    /// avoid heap allocations. `shifts` are a unit's panel-wide median or mean, which can be
    /// obtained from `median_all` and `mean_all`.
    pub fn shift_all(&mut self) -> Result<(), PanelError> {
        if !matches!(self.adj, LocAdj::None) {
            return Err(PanelError::AlreadyAdjusted {
                current_adj: self.adj,
            });
        }
        if matches!(self.stat, LocStat::None) {
            return Err(PanelError::LocationsNotComputed);
        }
        let n_u: usize = self.n_units;

        // Mutable chunks by cell over the values vector
        let chunks: ChunksExactMut<'_, f64> = self.values.chunks_exact_mut(self.probs.len());
        for (i, chunk) in chunks.enumerate() {
            // Module arithmetic matching chunk with unit
            let u: usize = i % n_u;
            let shift: f64 = self.locations[u];
            for val in chunk {
                *val -= shift; // Dereference on mutable value to write back result
            }
        }
        self.adj = LocAdj::Shift;
        Ok(())
    }

    /// Applies a scaling location adjustment in-place across all units.
    ///
    /// Accepts a `faer::Col<f64>` vector of scalars and divides the corresponding unit's
    /// evaluated quantiles by the scalars. Operates in-place over mutable chunks to avoid
    /// heap allocations. `scalars` are a unit's panel-wide median or mean, which can be
    /// obtained from `median_all` and `mean_all`.
    pub fn scale_all(&mut self) -> Result<(), PanelError> {
        if !matches!(self.adj, LocAdj::None) {
            return Err(PanelError::AlreadyAdjusted {
                current_adj: self.adj,
            });
        }
        if matches!(self.stat, LocStat::None) {
            return Err(PanelError::LocationsNotComputed);
        }
        let n_u: usize = self.n_units;

        // Mutable chunks by cell over the values vector
        let chunks: ChunksExactMut<'_, f64> = self.values.chunks_exact_mut(self.probs.len());
        for (i, chunk) in chunks.enumerate() {
            // Modulo arithmetic matching chunk with unit
            let u: usize = i % n_u;
            let scalar: f64 = self.locations[u];
            for val in chunk {
                *val /= scalar; // Dereference on mutable value to write back result
            }
        }
        self.adj = LocAdj::Scale;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::panel::{setup_test_panel, ObsPanel};

    #[test]
    fn test_quantile_agreement_type7() {
        // Construct a single-cell panel (T=1, U=1)
        let response: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0];
        let units: Vec<i32> = vec![0, 0, 0, 0];
        let periods: Vec<i32> = vec![0, 0, 0, 0];
        let obs = ObsPanel::new(&response, &units, &periods, 1, 1).unwrap();

        let probs: Vec<f64> = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let quantiles: QuantilePanel = obs.to_quantiles(&probs);

        // R Type-7 mathematical manual calculation for [2.0, 4.0, 6.0, 8.0]:
        // index = p * (n - 1). For N=4, n-1 = 3.
        // p=0.00: i=0, gamma=0.00 -> 2.0
        // p=0.25: i=0, gamma=0.75 -> 0.25(2.0) + 0.75(4.0) = 3.5
        // p=0.50: i=1, gamma=0.50 -> 0.50(4.0) + 0.50(6.0) = 5.0
        // p=0.75: i=2, gamma=0.25 -> 0.75(6.0) + 0.25(8.0) = 6.5
        // p=1.00: i=3, gamma=0.00 -> 8.0
        let expected: Vec<f64> = vec![2.0, 3.5, 5.0, 6.5, 8.0];

        assert_eq!(quantiles.values, expected);
    }

    #[test]
    fn test_to_quantiles_into_reuse() {
        let obs: ObsPanel = setup_test_panel(); // T=2, U=2
        let probs: Vec<f64> = vec![0.25, 0.5, 0.75]; // Q=3

        // 1. Verify `empty_quantiles` allocation math
        let mut out: QuantilePanel = obs.empty_quantiles(&probs);

        // Capacity must equal exactly T * U * Q
        let expected_capacity = 2 * 2 * 3;
        assert_eq!(out.values.capacity(), expected_capacity);
        assert!(out.values.is_empty());

        // 2. First evaluation pass
        obs.to_quantiles_into(&probs, &mut out);
        assert_eq!(out.values.len(), expected_capacity);
        let first_pass_values = out.values.clone();

        // 3. Second evaluation pass (simulating a worker loop)
        obs.to_quantiles_into(&probs, &mut out);

        // Invariant: The length must stay identical, and the values must be overwritten, not appended.
        assert_eq!(
            out.values.len(),
            expected_capacity,
            "QuantilePanel grew infinitely! Missing out.values.clear()?"
        );
        assert_eq!(out.values, first_pass_values);
    }

    #[test]
    fn test_quantile_shift_all() {
        // 1. Manually synthesize a panel (T=1, U=2, Q=3)
        let mut qp: QuantilePanel = QuantilePanel {
            values: vec![
                10.0, 20.0, 30.0, // Unit 0
                50.0, 60.0, 70.0, // Unit 1
            ],
            probs: vec![0.25, 0.5, 0.75],
            n_units: 2,
            n_periods: 1,
            adj: LocAdj::None,
            stat: LocStat::Mean,
            locations: faer::col![5.0, 10.0],
        };

        // 2. Apply shifts (Unit 0 shifts down by 5, Unit 1 shifts down by 10)
        // let shifts: Col<f64> = col![5.0, 10.0];
        qp.shift_all().unwrap();

        // 3. Assert exact mathematical correctness
        assert_eq!(qp.values[0..3], [5.0, 15.0, 25.0]);
        assert_eq!(qp.values[3..6], [40.0, 50.0, 60.0]);
    }

    #[test]
    fn test_quantile_scale_all() {
        // Manually synthesize a panel (T=1, U=2, Q=3)
        let mut qp: QuantilePanel = QuantilePanel {
            values: vec![
                10.0, 20.0, 30.0, // Unit 0
                50.0, 60.0, 70.0, // Unit 1
            ],
            probs: vec![0.25, 0.5, 0.75],
            n_units: 2,
            n_periods: 1,
            adj: LocAdj::None,
            stat: LocStat::Mean,
            locations: faer::col![2.0, 10.0],
        };

        // Apply scaling (Unit 0 scales down by 2.0, Unit 1 scales down by 10.0)
        // let scalars: Col<f64> = faer::col![2.0, 10.0];
        qp.scale_all().unwrap();

        // Assert exact mathematical correctness
        assert_eq!(qp.values[0..3], [5.0, 10.0, 15.0]);
        assert_eq!(qp.values[3..6], [5.0, 6.0, 7.0]);
    }
}

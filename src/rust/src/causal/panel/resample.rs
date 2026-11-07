//! Reusable, pre-allocated workspace (`ResamplePanel`) for zero-allocation bootstrap resampling.
//!
//! Supports observation-level, period-level (block), and two-stage resampling designs,
//! constructed from `ObsPanel` data.

use super::{LocAdj, LocStat, Panel, QuantilePanel};
use faer::Col;
use std::iter;

/// A mutable container for zero-allocation bootstrap resampling from `ObsPanel`.
///
/// Maintains the same ragged memory layout as `ObsPanel`, but is a reusable container for resampling.
/// This struct ensures resamples can only be drawn from `ObsPanel`. Methods targeting this struct are
/// designed to overwrite `values` and `offsets` in-place, avoiding the need for a reallocation for
/// every bootstrap resampling iteration.
#[derive(Debug)]
pub struct ResamplePanel {
    pub values: Vec<f64>,
    pub offsets: Vec<usize>,
    pub n_units: usize,
    pub n_periods: usize,
}

impl Panel for ResamplePanel {
    fn n_units(&self) -> usize {
        self.n_units
    }
    fn n_periods(&self) -> usize {
        self.n_periods
    }
    fn cell(&self, u: usize, t: usize) -> &[f64] {
        let c: usize = (t * self.n_units) + u;
        let start: usize = self.offsets[c];
        let end: usize = self.offsets[c + 1];

        &self.values[start..end]
    }
    fn cell_range(&self, u: usize, t: usize) -> std::ops::Range<usize> {
        let c: usize = (t * self.n_units) + u;
        let start: usize = self.offsets[c];
        let end: usize = self.offsets[c + 1];

        start..end
    }
    fn period(&self, t: usize) -> &[f64] {
        let c: usize = t * self.n_units;
        let start: usize = self.offsets[c];
        let end: usize = self.offsets[c + self.n_units];

        &self.values[start..end]
    }
    fn period_range(&self, t: usize) -> std::ops::Range<usize> {
        let c: usize = t * self.n_units;
        let start: usize = self.offsets[c];
        let end: usize = self.offsets[c + self.n_units];

        start..end
    }
}

impl ResamplePanel {
    /// Evaluates empirical quantiles for each cell into a new `QuantilePanel`.
    ///
    /// Accepts a slice of probabilities (`probs`) and allocates a new `QuantilePanel`
    /// alongside a temporary sorting scratchpad to evalute the quantiels for each cell.
    /// Employs linear interpolation (equivalent to R's type 7) to calculate the quantiles.
    /// Lazily evaluates th einterpolation iterator to efficiently populate the tensor.
    pub fn to_quantiles(&self, probs: &[f64]) -> QuantilePanel {
        // 1. Getting variables needed for allocation size
        let max_n: usize = (0..self.n_cells())
            .map(|i| self.offsets[i + 1] - self.offsets[i])
            .max()
            .unwrap_or(0);

        // 2. Allocatting capacity for dense B x J x Q tensor
        // Pre-allocated capacity for zero-allocation quantile evaluation
        let mut values: Vec<f64> = Vec::with_capacity(self.n_periods * self.n_units * probs.len());
        let mut pad: Vec<f64> = Vec::with_capacity(max_n);

        // 3. Sequential cell evaluation pass
        for t in 0..self.n_periods {
            for u in 0..self.n_units {
                let cell: &[f64] = self.cell(u, t);
                // Empty cell check
                if cell.is_empty() {
                    values.extend(iter::repeat(f64::NAN).take(probs.len()));
                    continue;
                }
                // Length of sample 0 indexed
                let n: usize = cell.len() - 1;
                let n_f64: f64 = n as f64;

                // Sorting on pre-allocated scratchpad vector
                pad.clear();
                pad.extend_from_slice(cell);
                pad.sort_by(|a: &f64, b: &f64| a.total_cmp(b));

                // Iterator over each probability level
                // R's type 7, replication of linear interpolation quantile function
                let quantiles = probs.iter().map(|&p| {
                    let index: f64 = p * n_f64;
                    let j: usize = index.floor() as usize;
                    let gamma: f64 = index - index.floor();

                    if j >= n {
                        return pad[n];
                    }

                    ((1.0 - gamma) * pad[j]) + (gamma * pad[j + 1])
                });

                values.extend(quantiles);
            }
        }

        QuantilePanel {
            values: values,
            probs: probs.to_vec(),
            n_units: self.n_units,
            n_periods: self.n_periods,
            adj: LocAdj::None,
            stat: LocStat::None,
            locations: Col::<f64>::full(self.n_units, f64::NAN),
        }
    }

    /// Evaluates empirical quantiles for each cell into a mutable `QuantilePanel` container.
    ///
    /// Accepts a `QuantilePanel` pre-allocated by `empty_quantiles` and a slice of
    /// probabilities (`probs`). Operates in-place by extending the pre-allocated container to
    /// avoid heap allocations across evaluation iterations. Employs linear interpolation
    /// (equivalent to R's type 7) to calculate the quantiles via a lazily evaluated iterator
    /// into a dense, column-major tensor layout.
    pub fn to_quantiles_into(&self, probs: &[f64], out: &mut QuantilePanel) {
        // 0. Reset Quantiles
        out.values.clear();
        out.adj = LocAdj::None;
        out.stat = LocStat::None;

        // 1. Getting variables needed for allocations
        let max_n: usize = (0..self.n_cells())
            .map(|i| self.offsets[i + 1] - self.offsets[i])
            .max()
            .unwrap_or(0);

        // 2. Allocating capacity for quantile evaluation
        let mut pad: Vec<f64> = Vec::with_capacity(max_n);

        // 3. Sequential cell evaluation pass
        for t in 0..self.n_periods {
            for u in 0..self.n_units {
                let cell: &[f64] = self.cell(u, t);
                // Empty cell check
                if cell.is_empty() {
                    out.values.extend(iter::repeat(f64::NAN).take(probs.len()));
                    continue;
                }
                // Length of sample 0-indexed
                let n: usize = cell.len() - 1;
                let n_f64: f64 = n as f64;

                // Sorting on pre-allocated scratchpad vector
                pad.clear();
                pad.extend_from_slice(cell);
                pad.sort_by(|a: &f64, b: &f64| a.total_cmp(b));

                // Iterator over each probability level
                // R's type 7, replication of linear interpolation quantile function
                let quantiles = probs.iter().map(|&p| {
                    let index: f64 = p * n_f64;
                    let j: usize = index.floor() as usize;
                    let gamma: f64 = index - index.floor();

                    if j >= n {
                        return pad[n];
                    }

                    ((1.0 - gamma) * pad[j]) + (gamma * pad[j + 1])
                });

                out.values.extend(quantiles);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::panel::{setup_test_panel, ObsPanel};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_resample_obs_invariants() {
        let obs: ObsPanel = setup_test_panel();
        let mut out: ResamplePanel = obs.empty_resample();
        let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(2001);

        obs.resample_obs_into(&mut out, &mut rng);

        for t in 0..obs.n_periods() {
            for u in 0..obs.n_units() {
                let obs_cell: &[f64] = obs.cell(u, t);
                let out_cell: &[f64] = out.cell(u, t);

                // Invariant 1: Sizes match exactly
                assert_eq!(obs_cell.len(), out_cell.len());

                // Invariant 2: Every drawn value belongs to the original cell
                for &val in out_cell {
                    assert!(obs_cell.contains(&val));
                }
            }
        }
    }

    #[test]
    fn test_resample_periods_invariants() {
        let obs: ObsPanel = setup_test_panel();
        let mut out: ResamplePanel = obs.empty_resample();
        let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(2001);

        obs.resample_periods_into(&mut out, &mut rng);

        for t_out in 0..out.n_periods() {
            // Reconstruct the entire output period
            let mut out_period_data: Vec<f64> = Vec::new();
            for u in 0..out.n_units() {
                out_period_data.extend_from_slice(out.cell(u, t_out));
            }

            // Must perfectly match at least one original period in `obs`
            let mut matched: bool = false;
            for t_obs in 0..obs.n_periods() {
                let mut obs_period_data: Vec<f64> = Vec::new();
                for u in 0..obs.n_units() {
                    obs_period_data.extend_from_slice(obs.cell(u, t_obs));
                }

                if out_period_data == obs_period_data {
                    matched = true;
                    break;
                }
            }
            assert!(
                matched,
                "Output period did not match any original input period"
            );
        }
    }

    #[test]
    fn test_resample_twostage_invariants() {
        let obs: ObsPanel = setup_test_panel();
        let mut out: ResamplePanel = obs.empty_resample();
        let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(2001);

        obs.resample_twostage_into(&mut out, &mut rng);

        for t_out in 0..out.n_periods() {
            // Find which period was drawn by comparing cell sizes
            let sizes_out: Vec<usize> = (0..out.n_units())
                .map(|u: usize| out.cell(u, t_out).len())
                .collect();

            let mut matched: bool = false;
            for t_obs in 0..obs.n_periods() {
                let sizes_obs: Vec<usize> = (0..obs.n_units())
                    .map(|u: usize| obs.cell(u, t_obs).len())
                    .collect();

                // If sizes match, verify observations are drawn from that exact cell
                if sizes_out == sizes_obs {
                    matched = true;
                    for u in 0..out.n_units() {
                        for &val in out.cell(u, t_out) {
                            assert!(obs.cell(u, t_obs).contains(&val));
                        }
                    }
                    break;
                }
            }
            assert!(
                matched,
                "Two-stage output sizes did not match any original period"
            );
        }
    }

    #[test]
    fn test_rng_determinism() {
        let obs: ObsPanel = setup_test_panel();

        let mut out1: ResamplePanel = obs.empty_resample();
        let mut rng1: ChaCha8Rng = ChaCha8Rng::seed_from_u64(2001);
        obs.resample_twostage_into(&mut out1, &mut rng1);

        let mut out2: ResamplePanel = obs.empty_resample();
        let mut rng2: ChaCha8Rng = ChaCha8Rng::seed_from_u64(2001);
        obs.resample_twostage_into(&mut out2, &mut rng2);

        // Identical seeds produce identical output
        assert_eq!(out1.values, out2.values);

        let mut out3: ResamplePanel = obs.empty_resample();
        let mut rng3: ChaCha8Rng = ChaCha8Rng::seed_from_u64(2001);
        rng3.set_stream(1); // Diverge the stream
        obs.resample_twostage_into(&mut out3, &mut rng3);

        // Changing the stream alters the sequence
        assert_ne!(out1.values, out3.values);
    }
}

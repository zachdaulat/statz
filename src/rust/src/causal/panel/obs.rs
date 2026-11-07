//! Primary ragged empirical observation panel (`ObsPanel`) backed by prefix-sum cell offsets.
//!
//! Handles validated construction from raw vectors, probability grid resolution,
//! pre/post-treatment panel splitting, and transition into resampled or quantile panels.

use super::{
    probs_grid, LocAdj, LocStat, Panel, PanelError, ProbsMode, QuantilePanel, ResamplePanel,
};
use faer::Col;
use rand::distr::{Distribution, Uniform};
use std::iter;

/// The initial structure organizing a distributional panel dataset.
///
/// Stores ragged distributional panel data, where units and periods can contain varying observation
/// counts. Uses a high-performance 1D flat vector layout and prefix-sum offsets vector storing cell indices.
/// This layout minimizes pointer chasing and ensures optimal CPU cache behaviour during iterations.
#[derive(Debug)]
pub struct ObsPanel {
    /// All observations grouped by cell (two-tier structure by period and unit).
    pub values: Vec<f64>,
    /// Prefix-sum array of length `n_cells + 1` dictating cell boundaries.
    pub offsets: Vec<usize>,
    pub n_units: usize,
    pub n_periods: usize,
}

impl Panel for ObsPanel {
    // Defining primary accessor functions
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

// Core constructors and state meanagement for `ObsPanel`
impl ObsPanel {
    /// Constructs a new `ObsPanel` from flat input observation vectors.
    ///
    /// This constructor performs three passes over the data to construct the prefix-sum
    /// ragged array layout. It validates dimensions, checks for missing data, and ensures
    /// identifier boundaries are respected.
    ///
    /// Performance: This is the primary intialization phase and allocates three heap vectors,
    /// (`counts`, `offsets`, and `values`)
    pub fn new(
        response: &[f64],
        unit_ids: &[i32],
        period_ids: &[i32],
        n_units: i32,
        n_periods: i32,
    ) -> Result<Self, PanelError> {
        // Dimensions and length checks
        if n_units <= 0 || n_periods <= 0 {
            return Err(PanelError::ZeroDims { n_units, n_periods });
        }
        if response.len() != unit_ids.len() {
            return Err(PanelError::DataLengthMismatch {
                expected: response.len(),
                actual: unit_ids.len(),
                vector: "unit_ids",
            });
        }
        if response.len() != period_ids.len() {
            return Err(PanelError::DataLengthMismatch {
                expected: response.len(),
                actual: period_ids.len(),
                vector: "period_ids",
            });
        }

        // Allocate and instantiate helper arrays
        let n_cells: usize = n_units as usize * n_periods as usize;
        let mut counts: Vec<usize> = vec![0; n_cells];
        let mut offsets: Vec<usize> = vec![0; n_cells + 1];
        let mut values: Vec<f64> = vec![0.0; response.len()];

        // Pass 1: Cell counts and input validity checks
        for (((i, &y), &u_id), &t_id) in response
            .iter()
            .enumerate()
            .zip(unit_ids.iter())
            .zip(period_ids.iter())
        {
            // Check valid IDs
            if u_id < 0 || u_id >= n_units {
                #[rustfmt::skip]
                return Err(PanelError::IdOutOfRange {
                    id: u_id, n: n_units, vector: "unit_ids", at: i,
                });
            }
            if t_id < 0 || t_id >= n_periods {
                #[rustfmt::skip]
                return Err(PanelError::IdOutOfRange {
                    id: t_id, n: n_periods, vector: "period_ids", at: i,
                });
            }
            // NaN guard
            if y.is_nan() {
                return Err(PanelError::HasNaN { at: i });
            }

            // Identify cell and increment
            let c: usize = (t_id as usize * n_units as usize) + u_id as usize;
            counts[c as usize] += 1;
        }

        // Pass 2: Prefix-sum offsets
        for ci in 1..=n_cells {
            offsets[ci] = offsets[ci - 1] + counts[ci - 1];
        }

        // Pass 3: Insert response to values
        let mut cursors: Vec<usize> = offsets.clone();
        for ((&y, &u_id), &t_id) in response.iter().zip(unit_ids.iter()).zip(period_ids.iter()) {
            // Get cell index
            let c: usize = (t_id as usize * n_units as usize) + u_id as usize;
            // Insert response into values
            values[cursors[c]] = y;
            // Increment cursor
            cursors[c] += 1;
        }

        Ok(ObsPanel {
            values: values,
            offsets: offsets,
            n_units: n_units as usize,
            n_periods: n_periods as usize,
        })
    }

    /// Resolves a `ProbsMode` configuration into a concrete probability grid.
    ///
    /// Performance: Allocates a new `Vec<f64>` containing the standardized grid.
    pub fn probs_from(&self, mode: ProbsMode) -> Vec<f64> {
        match mode {
            ProbsMode::Grid(p) => p,
            ProbsMode::Fixed(n) => probs_grid(n),
            ProbsMode::MinCell => {
                let min_n: usize = (0..self.n_cells())
                    .map(|i: usize| self.offsets[i + 1] - self.offsets[i])
                    .min()
                    .unwrap_or(0);
                probs_grid(min_n)
            }
            ProbsMode::MaxCell => {
                let max_n: usize = (0..self.n_cells())
                    .map(|i: usize| self.offsets[i + 1] - self.offsets[i])
                    .max()
                    .unwrap_or(0);
                probs_grid(max_n)
            }
            ProbsMode::MedianCell => {
                let n: usize = self.n_cells();
                let k: usize = n / 2;
                let mut sizes: Vec<usize> = (0..n)
                    .map(|i| self.offsets[i + 1] - self.offsets[i])
                    .collect::<Vec<usize>>();
                let (less, upp_mid, _) = sizes.select_nth_unstable(k);

                let mid: usize = if n % 2 == 0 {
                    let low_mid: usize = less.iter().copied().max().unwrap();
                    (low_mid + *upp_mid) / 2
                } else {
                    *upp_mid
                };

                probs_grid(mid)
            }
            ProbsMode::MeanCell => {
                let total_obs: usize = (0..self.n_cells())
                    .map(|i: usize| self.offsets[i + 1] - self.offsets[i])
                    .sum();
                let mean: usize = total_obs / self.n_cells();
                probs_grid(mean)
            }
        }
    }

    /// Splits the panel at treatment period `t0` into separate pre-treatment and
    /// post-treatment `ObsPanel` instances.
    ///
    /// Consumes the original panel and partitions the underlying contiguous data
    /// at period `t0`. Periods `0..t0` are retained in the returned pre-treatment
    /// panel, while periods `t0..self.n_periods` are moved into the newly constructed
    /// post-treatment panel with its prefix-sum `offsets` re-indexed to start at `0`.
    ///
    /// # Arguments
    /// * `t0` - The zero-indexed period at which treatment begins. Must lie strictly
    ///   within `(0, self.n_periods)` so that both resulting panels contain at least
    ///   one period.
    ///
    /// # Errors
    /// Returns [`PanelError::InvalidTreatedPeriod`] if `t0 == 0` or `t0 >= self.n_periods`.
    pub fn split_treatment(mut self, t0: usize) -> Result<(ObsPanel, ObsPanel), PanelError> {
        // 1. Validate treatment ID within periods
        if t0 == 0 || t0 >= self.n_periods {
            return Err(PanelError::InvalidTreatedPeriod {
                period: t0,
                n_periods: self.n_periods,
            });
        };

        // 2, Compute split boundary indices for cells and observations
        let c: usize = t0 * self.n_units;
        let v: usize = self.offsets[c];

        // 3. Construct post-treatment panel in single pass over trailing offsets
        let mut post: ObsPanel = ObsPanel {
            values: self.values.split_off(v),
            offsets: self.offsets[c..]
                .iter()
                .map(|&ci| ci - v)
                .collect::<Vec<usize>>(),
            n_units: self.n_units,
            n_periods: self.n_periods - t0,
        };

        // 4. Truncate pre-treatment offsets (retining the final boundary at `c`)
        // and update period count
        self.offsets.truncate(c + 1);
        self.n_periods = t0;

        Ok((self, post))
    }
}

// Methods for interacting with `ResamplePanel`
impl ObsPanel {
    /// Allocates memory for a reusable, mutable resample container.
    ///
    /// Allocates sufficient space for a worst-case scenario of the largest period(s) being
    /// all of the resamples. Intended to be called once to allocate the capacity, then cleared,
    /// repopulated, and reused every resampling iteration.
    pub fn empty_resample(&self) -> ResamplePanel {
        // Get size of largest period
        let max_period: usize = (0..self.n_periods)
            .map(|t: usize| self.period_size(t))
            .max()
            .unwrap_or(0);

        // Allocate "worst-case" sampling via largest period
        let capacity: usize = max_period * self.n_periods;

        ResamplePanel {
            values: Vec::with_capacity(capacity),
            offsets: vec![0 as usize; self.offsets.len()], // Must not be 0-length to be indexed by resampling fucntions
            n_units: self.n_units,
            n_periods: self.n_periods,
        }
    }

    /// Resamples observations by cell into mutable `ResamplePanel` container.
    ///
    /// Accepts a `ResamplePanel` pre-allocated by `empty_resample`, and `rand::Rng` instance,
    /// resampling each cell to populate the container. `ChaCha8Rng` is the currently preferred `Rng`.
    /// Lazily evaluates sampling iterator to efficiently populate the container.
    pub fn resample_obs_into(&self, out: &mut ResamplePanel, rng: &mut impl rand::Rng) {
        // Reset state for next iteration.
        out.values.clear();
        out.offsets.clear();
        out.offsets.extend_from_slice(&self.offsets);

        // Loop over cells
        for t in 0..self.n_periods {
            for u in 0..self.n_units {
                // Get cell slice
                let cell: &[f64] = self.cell(u, t);
                if cell.len() == 0 {
                    continue;
                }

                // Populate output panel with sampled observations
                // Uses lazily evaluated iterator and extend
                let draws = Uniform::new(0, cell.len())
                    .unwrap()
                    .sample_iter(&mut *rng)
                    .take(cell.len())
                    .map(|i| cell[i]);

                out.values.extend(draws);
            }
        }
    }

    /// Resamples periods in a cluster random sampling design into a mutable `ResamplePanel` container.
    ///
    /// Accepts a `ResamplePanel` pre-allocated by `empty_resample`, and `rand::Rng` instange. Samples from
    /// periods in a clsuter random sampling design as blocks to populate the container.
    /// `ChaCha8Rng` is the currently preferred `Rng`. Bulk-copying slices via `extend_from_slice`
    /// similar to `memcpy` in C.
    /// NOTE: Does not maintain temporal ordering of the periods.
    pub fn resample_periods_into(&self, out: &mut ResamplePanel, rng: &mut impl rand::Rng) {
        // 1. Reset state for this resample
        out.values.clear();
        out.offsets[0] = 0;
        let mut curr_offset: usize = 0;
        let mut out_cell_idx: usize = 0;

        // 2. Draw samples of periods
        for t in Uniform::new(0, self.n_periods)
            .unwrap()
            .sample_iter(&mut *rng)
            .take(self.n_periods)
        {
            // 3. Reconstruct cell boundaries for output offsets
            for u in 0..self.n_units {
                // Calculate cell sizes
                let cell_size: usize = self.cell_size(u, t);

                // Accumulate and assign
                curr_offset += cell_size;
                out_cell_idx += 1;
                out.offsets[out_cell_idx] = curr_offset;
            }

            // 4. Bulk copying all observations from period
            out.values.extend_from_slice(self.period(t));
        }
    }

    /// Resamples both periods and observations by cell, into a mutable `ResamplePanel` container.
    ///
    /// Accepts a `ResamplePanel` pre-allocated by `empty_resample`, and `rand::Rng` instance.
    /// `ChaCha8Rng` is the currently preferred `Rng`. Interleaves period draws with draws from
    /// the cells in that period, processing all draws in a single pass to populate `ResamplePanel`.
    /// NOTE: Does not maintain temporal ordering of the periods.
    pub fn resample_twostage_into(&self, out: &mut ResamplePanel, rng: &mut impl rand::Rng) {
        // 1. Reset state for this resample and instantiation
        out.values.clear();
        out.offsets[0] = 0;
        let mut curr_offset: usize = 0;
        let mut out_cell_idx: usize = 0;
        let period_dist: Uniform<usize> = Uniform::new(0, self.n_periods).unwrap();

        // 2. Drawing random periods
        for _ in 0..self.n_periods {
            // Single quick "micro-borrow"/reborrow from rng instance
            let t: usize = period_dist.sample(&mut *rng);
            let t_start: usize = t * self.n_units;

            // 3. Iterate over units in drawn period
            for u in 0..self.n_units {
                let val_start: usize = self.offsets[t_start + u];
                let val_end: usize = self.offsets[t_start + u + 1];
                let n: usize = val_end - val_start;

                // 4. Update offsets array and iteration state
                curr_offset += n;
                out_cell_idx += 1;
                out.offsets[out_cell_idx] = curr_offset;

                if n == 0 {
                    continue;
                }

                // 5. Draw observations from the target cell
                let draws = Uniform::new(val_start, val_end)
                    .unwrap()
                    .sample_iter(&mut *rng)
                    .take(n)
                    .map(|i: usize| self.values[i]);

                // 6. Populate ouput from draws iterator
                out.values.extend(draws);
            }
        }
    }
}

// Methods for interacting with `QuantilePanel`
impl ObsPanel {
    /// Allocates memory for a reusable, mutable quantile container.
    ///
    /// Allocates exact memory capacity for a dense tensor layout bsed on the length of the
    /// provided `probs` slice. Intended to be called once to allocate the capacity,
    /// then repopulated and reused every resampling iteration via `to_quantiles_into`.
    pub fn empty_quantiles(&self, probs: &[f64]) -> QuantilePanel {
        let capacity: usize = self.n_periods * self.n_units * probs.len();

        QuantilePanel {
            values: Vec::with_capacity(capacity),
            probs: probs.to_vec(),
            n_units: self.n_units,
            n_periods: self.n_periods,
            adj: LocAdj::None,
            stat: LocStat::None,
            locations: Col::<f64>::full(self.n_units, f64::NAN),
        }
    }

    /// Evaluates empirical quantiles for each cell into a new `QuantilePanel`.
    ///
    /// Accepts a slice of probabilities (`probs`) and allocates a new `QuantilePanel`
    /// alongside a temporary sorting scratchpad to evalute the quantiels for each cell.
    /// Employs linear interpolation (equivalent to R's type 7) to calculate the quantiles.
    /// Lazily evaluates th einterpolation iterator to efficiently populate the tensor.
    pub fn to_quantiles(&self, probs: &[f64]) -> QuantilePanel {
        // 1. Getting variables needed for allocation size
        let max_n: usize = (0..self.n_cells())
            .map(|i: usize| self.offsets[i + 1] - self.offsets[i])
            .max()
            .unwrap_or(0);

        // 2. Allocate capacity for dense B x J x Q tensor
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

    #[test]
    fn test_probs_from_variants() {
        // Construct an unbalanced panel (U=3, T=1) to distinctly separate summary statistics.
        // Unit 0 size: 1
        // Unit 1 size: 4
        // Unit 2 size: 10
        // Total obs: 15. Mean size: 15/3 = 5. Median size: 4.
        let response: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        let units: Vec<i32> = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        let periods: Vec<i32> = vec![0; 15];
        let panel: ObsPanel = ObsPanel::new(&response, &units, &periods, 3, 1).unwrap();

        // Pinming the exact arithmetic logic to the variant
        assert_eq!(panel.probs_from(ProbsMode::MinCell).len(), 1);
        assert_eq!(panel.probs_from(ProbsMode::MedianCell).len(), 4);
        assert_eq!(panel.probs_from(ProbsMode::MeanCell).len(), 5);
        assert_eq!(panel.probs_from(ProbsMode::MaxCell).len(), 10);

        // Fixed and manual grid bypass cell sizes entirely
        assert_eq!(panel.probs_from(ProbsMode::Fixed(7)).len(), 7);
        let custom_grid: Vec<f64> = vec![0.1, 0.9];
        assert_eq!(
            panel.probs_from(ProbsMode::Grid(custom_grid.clone())),
            custom_grid
        );
    }

    #[test]
    fn test_round_trip_naive_filter() {
        let response: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let units: Vec<i32> = vec![0, 0, 1, 0, 1, 1, 1];
        let periods: Vec<i32> = vec![0, 0, 0, 1, 1, 1, 1];
        let n_units: i32 = 2;
        let n_periods: i32 = 2;

        let panel: ObsPanel =
            ObsPanel::new(&response, &units, &periods, n_units, n_periods).unwrap();

        // Compare every cell against a naive O(N) filter
        for t in 0..n_periods {
            for u in 0..n_units {
                let naive_cell: Vec<f64> = response
                    .iter()
                    .zip(units.iter())
                    .zip(periods.iter())
                    .filter_map(
                        |((&y, &ui), &ti)| {
                            if ui == u && ti == t {
                                Some(y)
                            } else {
                                None
                            }
                        },
                    )
                    .collect();

                assert_eq!(panel.cell(u as usize, t as usize), naive_cell.as_slice());
            }
        }
    }

    #[test]
    fn test_obspanel_round_trip() {
        // U = 2, T = 2
        // Cell (0,0): [1.0, 2.0]
        // Cell (1,0): [3.0]
        // Cell (0,1): [] -> Empty cell test!
        // Cell (1,1): [4.0, 5.0, 6.0]
        let response: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let units: Vec<i32> = vec![0, 0, 1, 1, 1, 1];
        let periods: Vec<i32> = vec![0, 0, 0, 1, 1, 1];

        let panel = ObsPanel::new(&response, &units, &periods, 2, 2).unwrap();

        assert_eq!(panel.cell(0, 0), &[1.0, 2.0]);
        assert_eq!(panel.cell(1, 0), &[3.0]);
        assert_eq!(panel.cell(0, 1), &[] as &[f64]);
        assert_eq!(panel.cell(1, 1), &[4.0, 5.0, 6.0]);
    }
}

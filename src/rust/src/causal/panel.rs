#![allow(unused)]
#![allow(non_snake_case)]

use crate::ext::RMatrixExt;
use core::f64;
use extendr_api::{prelude::*, Error};
use faer::{
    col::AsColRef, linalg::solvers::SelfAdjointEigen, mat::AsMatRef, Col, ColMut, ColRef, Mat,
    MatRef, Side,
};
use rand::distr::{Distribution, Uniform};
use rand_chacha::ChaCha8Rng;
use std::{iter, slice::ChunksExactMut};

impl std::fmt::Display for PanelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PanelError::LengthMismatch {
                expected,
                actual,
                vector,
            } => write!(
                f,
                "`{vector}` has length {actual}, expected {expected} to match `response`"
            ),
            PanelError::IdOutOfRange { id, n, vector, at } => write!(
                f,
                "`{vector}[{}]` is {id}; identifiers must lie in [0, {n})",
                at + 1
            ),
            PanelError::InvalidTreatedId { id, n_units } => {
                write!(f, "`treated_id` is {id}; must lie in [0, {n_units})")
            }
            PanelError::HasNaN { at } => write!(
                f,
                "`response[{}]` is NA or NaN; the panel requires complete observations",
                at + 1
            ),
            PanelError::EmptyCell { unit, period } => write!(
                f,
                "cell (unit {}, period {}) contains no observations",
                unit + 1,
                period + 1
            ),
            PanelError::ZeroDims { n_units, n_periods } => write!(
                f,
                "panel dimensions must be strictly positive; received n_units = {n_units} and n_periods = {n_periods}"
            ),
        }
    }
}

impl std::error::Error for PanelError {}

/// Defines the location adjustment applied to the empirical distributions.
pub enum LocAdj {
    /// Raw empirical distributions with no location adjustment.
    None,
    /// Translation adjustment based on a central location statistic.
    Shift,
    /// Scaling adjustment based on a central location statistic.
    Scale,
}

/// Defines the central location statistic used when applying `LocAdj`.
pub enum LocStat {
    Mean,
    Median,
}

/// Defines the resolution strategy for the probability grid used in quantile evaluation.
///
/// To ensure matrix conformability in downstream `faer` linear algebra, every bootstrap
/// resampling iteration must project onto the same $Q$-dimensional quantile space. This
/// enum dictates how that global $Q$ is determined before resampling begins.
pub enum ProbsMode {
    /// A manually provided, pre-computed vector fo probability levels.
    Grid(Vec<f64>),
    /// A fixed number of quantiles, yielding a uniform probability grid.
    Fixed(usize),
    /// Generates a grid matching the observation count of the smallest cell in the panel.
    MinCell,
    /// Generates a grid matching the median observation count across all cells.
    MedianCell,
    /// Generates a grid matching the mean observation count across all cells.
    MeanCell,
    /// Generates a grid matching the observation count of the largest cell in the panel.
    MaxCell,
}

/// Represents validtaion failures encountered during Panel data structure initialization.
///
/// The panel structures rely on dense contiguous memory and matrix conformability,
/// so these errors and checks enforce invalid inputs before any heavy allocations occur.
#[derive(Debug, Clone, PartialEq)]
pub enum PanelError {
    /// One of the ID vector lengths do not match the `response` vector length.
    LengthMismatch {
        expected: usize,
        actual: usize,
        vector: &'static str,
    },
    /// An ID in the data exceeds the user-specified ID range
    IdOutOfRange {
        id: i32,
        n: i32,
        vector: &'static str,
        at: usize,
    },
    /// The specified treated unit index does not exist within the panel dimensions.
    InvalidTreatedId { id: i32, n_units: i32 },
    /// The dataset contains missing values. The Distributional Synthetic Control
    /// pipeline requires complete observations for evaluation.
    HasNaN { at: usize },
    /// A given unit-period cell contains no observations.
    EmptyCell { unit: usize, period: usize },
    /// The panel was initialized with a dimension of 0.
    ZeroDims { n_units: i32, n_periods: i32 },
}

/// A general interface for ragged and rectangular distributional panel data.
///
/// This trait abstracts the indexing math required to interact with contiguous 1D memory
/// as a $U \times T$ distributional panel data. Because observations counts can vary
/// across units and periods, methods prioritize lazily-evaluated iterators, in-place
/// mutations, and numerically stable algorithms for higher performance.
pub trait Panel {
    // --------------- Required functions

    /// Returns the total number of units ($U$) in the panel.
    fn n_units(&self) -> usize;
    /// Returns the total number of time periods ($T$) in the panel.
    fn n_periods(&self) -> usize;
    /// Returns a borrowed slice of the observations for a specific unit-period cell.
    fn cell(&self, u: usize, t: usize) -> &[f64];
    /// Returns the $O(1)$ index boundaries of a cell within the underlying 1D flat vector.
    fn cell_range(&self, u: usize, t: usize) -> std::ops::Range<usize>;
    /// Returns a borrowed slice of the observations for a specific period.
    fn period(&self, t: usize) -> &[f64];
    /// Returns the index boundaries of a period within the underlying 1D flat vector.
    fn period_range(&self, t: usize) -> std::ops::Range<usize>;
    // fn data(&self) -> (Vec<f64>, Vec<i32>, Vec<i32>);

    // -------------- Inspection & derived functions
    /// Determines if a cell contains zero observations
    fn is_empty_cell(&self, u: usize, t: usize) -> bool {
        self.cell_range(u, t).is_empty()
    }
    fn n_cells(&self) -> usize {
        self.n_units() * self.n_periods()
    }
    fn cell_size(&self, u: usize, t: usize) -> usize {
        self.cell_range(u, t).len()
    }
    fn cell_col_ref(&self, u: usize, t: usize) -> ColRef<'_, f64> {
        ColRef::from_slice(self.cell(u, t))
    }
    fn unit_size(&self, u: usize) -> usize {
        (0..self.n_periods())
            .map(|t: usize| self.cell_size(u, t))
            .sum()
    }
    fn period_size(&self, t: usize) -> usize {
        self.period_range(t).len()
    }
    /// Allocates and returns a flat vector containing all observations for a unit across all periods.
    /// Performance: Requires a heap allocation. Prefer `unit_iter` for zero-allocation
    /// lazily-evaluate iterators.
    fn unit_data(&self, u: usize) -> Vec<f64> {
        // Get number of observations for that unit
        let mut data: Vec<f64> = Vec::with_capacity(self.unit_size(u));
        for t_id in 0..self.n_periods() {
            let cell: &[f64] = self.cell(u, t_id);
            data.extend_from_slice(cell);
        }
        data
    }
    fn unit_data_all(&self) -> Vec<Vec<f64>> {
        (0..self.n_units())
            .map(|u: usize| self.unit_data(u))
            .collect::<Vec<Vec<f64>>>()
    }
    /// Returns a zero-allocation, lazily evaluated iterator over all observations for a given unit.
    fn unit_iter(&self, u: usize) -> impl Iterator<Item = &f64> + '_ {
        (0..self.n_periods()).flat_map(move |t: usize| self.cell(u, t).iter())
    }

    // ---------------- Unit-level descriptive statistics
    /// Calculates the sum of a unit's observations using Neumaier compensated summation.
    /// This prevents catastrophic cancellation and precision loss in large, high-magnitude
    /// range datasets.
    fn sum(&self, u: usize) -> f64 {
        let mut sum: f64 = 0.0;
        let mut c: f64 = 0.0;

        // Neumaier summation
        for &y in self.unit_iter(u) {
            let t: f64 = sum + y;

            // Accumulating lost bits
            if sum.abs() >= y.abs() {
                c += (sum - t) + y;
            } else {
                c += (y - t) + sum;
            }
            sum = t;
        }
        sum + c
    }
    /// Returns a `faer::Col<f64>` containing the `sum` of every unit.
    fn sum_all(&self) -> Col<f64> {
        let mut sum: Col<f64> = Col::zeros(self.n_units());
        let mut c: Col<f64> = Col::zeros(self.n_units());

        for t in 0..self.n_periods() {
            for u in 0..self.n_units() {
                for &y in self.cell(u, t).iter() {
                    let t: f64 = sum[u] + y;

                    // Accumulating lost bits
                    if sum[u].abs() >= y.abs() {
                        c[u] += (sum[u] - t) + y;
                    } else {
                        c[u] += (y - t) + sum[u];
                    }
                    sum[u] = t;
                }
            }
        }
        for u in 0..self.n_units() {
            sum[u] += c[u];
        }
        sum
    }
    /// Calculates the arithmetic mean using Neumaier compensated summation.
    /// Returns `f64::NAN` if the unit contains zero observations.
    fn mean(&self, u: usize) -> f64 {
        let mut sum: f64 = 0.0;
        let mut c: f64 = 0.0;
        let mut n: usize = 0;

        // Neumaier summation & counter for ragged vector
        for &y in self.unit_iter(u) {
            let t: f64 = sum + y;
            n += 1;

            // Accumulating lost bits
            if sum.abs() >= y.abs() {
                c += (sum - t) + y;
            } else {
                c += (y - t) + sum;
            }
            sum = t;
        }
        if n == 0 {
            f64::NAN
        } else {
            (sum + c) / n as f64
        }
    }
    /// Returns a `faer::Col<f64>` containing the arithmetic means of all units via
    /// Neumaier compensated summation.
    fn mean_all(&self) -> Col<f64> {
        // Use mutable vector to apply the division to each element, for_each?
        let mut sum: Col<f64> = Col::zeros(self.n_units());
        let mut c: Col<f64> = Col::zeros(self.n_units());
        let mut n: Vec<usize> = vec![0; self.n_units()];

        // Nested for loops for neumaier summation over periods and units
        for t in 0..self.n_periods() {
            for u in 0..self.n_units() {
                // Cell slice and length extraction
                let cell: &[f64] = self.cell(u, t);
                n[u] += cell.len();
                // Summation loop over the sequential cell
                for &y in cell.iter() {
                    let t: f64 = sum[u] + y;
                    if sum[u].abs() >= y.abs() {
                        c[u] += (sum[u] - t) + y;
                    } else {
                        c[u] += (y - t) + sum[u];
                    }
                    sum[u] = t;
                }
            }
        }

        // Applying bit loss compensation and calculating means
        for u in 0..self.n_units() {
            if n[u] == 0 {
                sum[u] = f64::NAN; // Guarding against division by zero for empty units
            } else {
                sum[u] = (sum[u] + c[u]) / n[u] as f64;
            }
        }
        sum
    }
    /// Calculates the median of a unit's observations via an introselect implementation
    /// based on the "ipnsort" algorithm by Lukas Bergdoll and Orson Peters. The fallback
    /// algorithm is Median of Medians using Tukey’s Ninther.
    ///
    /// Performance: Triggers a heap allocation to build the contiguous array before sorting.
    fn median(&self, u: usize) -> f64 {
        let mut x: Vec<f64> = self.unit_data(u);
        let n: usize = x.len();
        if n == 0 {
            return f64::NAN;
        }
        let k: usize = n / 2;
        let (less, upp_mid, _) = x.select_nth_unstable_by(k, f64::total_cmp);
        if n % 2 == 0 {
            let low_mid: f64 = less.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (low_mid + *upp_mid) / 2.0
        } else {
            *upp_mid
        }
    }
    /// Computes the median across all units returning a `faer::Col<f64>`, via an introselect
    /// implementation based on the "ipnsort" algorithm by Lukas Bergdoll and Orson Peters.
    /// The fallback algorithm is Median of Medians using Tukey’s Ninther.
    ///
    /// Performance: Highly optimized, reuses a single pre-allocated scratchpad for all units,
    /// performing exactly one heap allocation.
    fn median_all(&self) -> Col<f64> {
        // Identify max cell size via nested for loops
        let mut size_max: usize = 0;
        for ui in 0..self.n_units() {
            let size: usize = std::cmp::max(self.unit_size(ui), size_max);
            size_max = size;
        }
        let mut pad: Vec<f64> = Vec::with_capacity(size_max);

        Col::from_fn(self.n_units(), |u: usize| {
            pad.clear();
            for t in 0..self.n_periods() {
                pad.extend_from_slice(self.cell(u, t));
            }
            let n: usize = pad.len();
            if n == 0 {
                return f64::NAN;
            }
            let k: usize = n / 2;
            let (less, upp_mid, _) = pad.select_nth_unstable_by(k, f64::total_cmp);
            if n % 2 == 0 {
                let low_mid: f64 = less.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (low_mid + *upp_mid) / 2.0
            } else {
                *upp_mid
            }
        })
    }
    /// Calculates the sample variance using Welford's online algorithm.
    fn var(&self, u: usize) -> f64 {
        let mut k: f64 = 0.0;
        let mut Mk: f64 = 0.0;
        let mut Sk: f64 = 0.0;

        // Welford's algorithm
        for &xi in self.unit_iter(u) {
            k += 1.0;
            let dev_prev: f64 = xi - Mk;
            Mk += dev_prev / k;
            let dev_new: f64 = xi - Mk;
            Sk += dev_prev * dev_new;
        }

        if k < 2.0 {
            return f64::NAN;
        }
        let var: f64 = Sk / (k - 1.0);
        var
    }
    /// Returns a `faer::Col<f64>` containing the variance of all units via Welford's online algorithm.
    fn var_all(&self) -> Col<f64> {
        let mut k: Col<f64> = Col::zeros(self.n_units());
        let mut Mk: Col<f64> = Col::zeros(self.n_units());
        let mut Sk: Col<f64> = Col::zeros(self.n_units());

        for t in 0..self.n_periods() {
            for u in 0..self.n_units() {
                for &xi in self.cell(u, t).iter() {
                    k[u] += 1.0;
                    let dev_prev: f64 = xi - Mk[u];
                    Mk[u] += dev_prev / k[u];
                    let dev_new: f64 = xi - Mk[u];
                    Sk[u] += dev_prev * dev_new;
                }
            }
        }
        for u in 0..self.n_units() {
            if k[u] < 2.0 {
                Sk[u] = f64::NAN;
            } else {
                Sk[u] = Sk[u] / (k[u] - 1.0);
            }
        }
        Sk
    }
    /// Returns the standard deviation of a given unit via Welford's online algorithm.
    fn sd(&self, u: usize) -> f64 {
        self.var(u).sqrt()
    }
    /// Returns a `faer::Col<f64>` containing the standard deviation of all units via Welford's online algorithm.
    fn sd_all(&self) -> Col<f64> {
        let mut res: Col<f64> = self.var_all();
        for u in res.iter_mut() {
            *u = u.sqrt();
        }
        res
    }
}

/// The initial structure organizing a distributional panel dataset.
///
/// Stores ragged distributional panel data, where units and periods can contain varying observation
/// counts. Uses a high-performance 1D flat vector layout and prefix-sum offsets vector storing cell indices.
/// This layout minimizes pointer chasing and ensures optimal CPU cache behaviour during iterations.
#[derive(Debug)]
pub struct ObsPanel {
    /// All observations grouped by cell (two-tier structure by period and unit).
    values: Vec<f64>,
    /// Prefix-sum array of length `n_cells + 1` dictating cell boundaries.
    offsets: Vec<usize>,
    n_units: usize,
    n_periods: usize,
}

/// A mutable container for zero-allocation bootstrap resampling from `ObsPanel`.
///
/// Maintains the same ragged memory layout as `ObsPanel`, but is a reusable container for resampling.
/// This struct ensures resamples can only be drawn from `ObsPanel`. Methods targeting this struct are
/// designed to overwrite `values` and `offsets` in-place, avoiding the need for a reallocation for
/// every bootstrap resampling iteration.
#[derive(Debug)]
pub struct ResamplePanel {
    values: Vec<f64>,
    offsets: Vec<usize>,
    n_units: usize,
    n_periods: usize,
}

/// A dense tensor containing the evaluated quantile functions by cell.
///
/// This struct organizes a strictly uniform three-dimensional layout ($$T \times U \times Q)
/// ensuring structural conformability for downstream matrix operations.
#[derive(Debug)]
pub struct QuantilePanel {
    /// Dense vector of length `n_periods * n_units * probs.len()`
    values: Vec<f64>,
    /// The probability grid used to evaluate these quantiles
    probs: Vec<f64>,
    n_units: usize,
    n_periods: usize,
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
            return Err(PanelError::LengthMismatch {
                expected: response.len(),
                actual: unit_ids.len(),
                vector: "unit_ids",
            });
        }
        if response.len() != period_ids.len() {
            return Err(PanelError::LengthMismatch {
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

    /// Resamples both periods, and observations by cell, into a mutable `ResamplePanel` container.
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
            // Single quick "micro-borrow" from rng instange
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
        }
    }

    /// Evaluates empirical quantiles for each cell into a mutable `QuantilePanel` container.
    ///
    /// Accepts a `QuantilePanel` pre-allocated by `empty_quantiles` and a slice of
    /// probabilities (`probs`). Operates in-place by extending the pre-allocated container to
    /// avoid heap allocations across evaluation iterations. Employs linear interpolation
    /// (equivalent to R's type 7) to calculate the quantiles via a lazily evaluated iterator
    /// into a dense, column-major tensor layout.
    pub fn to_quantiles_into(&self, out: &mut QuantilePanel, probs: &[f64]) {
        // 0. Reset Quantiles
        out.values.clear();

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
        }
    }

    /// Evaluates empirical quantiles for each cell into a mutable `QuantilePanel` container.
    ///
    /// Accepts a `QuantilePanel` pre-allocated by `empty_quantiles` and a slice of
    /// probabilities (`probs`). Operates in-place by extending the pre-allocated container to
    /// avoid heap allocations across evaluation iterations. Employs linear interpolation
    /// (equivalent to R's type 7) to calculate the quantiles via a lazily evaluated iterator
    /// into a dense, column-major tensor layout.
    pub fn to_quantiles_into(&self, out: &mut QuantilePanel, probs: &[f64]) {
        // 0. Reset Quantiles
        out.values.clear();

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

    /// Applies a translation location adjustment in-place across all units.
    ///
    /// Accepts a `faer::Col<f64>` vector of shifts and subtracts the corresponding unit's
    /// shift value from each evalauted quantile. Operates in-place over mutable chunks to
    /// avoid heap allocations. `shifts` are a unit's panel-wide median or mean, which can be
    /// obtained from `median_all` and `mean_all`.
    pub fn shift_all(&mut self, shifts: &Col<f64>) {
        let n_u: usize = self.n_units;

        // Mutable chunks by cell over the values vector
        let chunks: ChunksExactMut<'_, f64> = self.values.chunks_exact_mut(self.probs.len());
        for (i, chunk) in chunks.enumerate() {
            // Module arithmetic matching chunk with unit
            let u: usize = i % n_u;
            let shift: f64 = shifts[u];
            for val in chunk {
                *val -= shift; // Dereference on mutable value to write back result
            }
        }
    }

    /// Applies a scaling location adjustment in-place across all units.
    ///
    /// Accepts a `faer::Col<f64>` vector of scalars and divides the corresponding unit's
    /// evaluated quantiles by the scalars. Operates in-place over mutable chunks to avoid
    /// heap allocations. `scalars` are a unit's panel-wide median or mean, which can be
    /// obtained from `median_all` and `mean_all`.
    pub fn scale_all(&mut self, scalars: &Col<f64>) {
        let n_u: usize = self.n_units;

        // Mutable chunks by cell over the values vector
        let chunks: ChunksExactMut<'_, f64> = self.values.chunks_exact_mut(self.probs.len());
        for (i, chunk) in chunks.enumerate() {
            // Modulo arithmetic matching chunk with unit
            let u: usize = i % n_u;
            let scalar: f64 = scalars[u];
            for val in chunk {
                *val /= scalar; // Dereference on mutable value to write back result
            }
        }
    }
}

// ---------------------- Trait impls ---------------------

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
        let end: usize = start + self.probs.len();

        &self.values[start..end]
    }
    fn period_range(&self, t: usize) -> std::ops::Range<usize> {
        let start: usize = t * self.n_units * self.probs.len();
        let end: usize = start + (self.n_units * self.probs.len());

        start..end
    }
}

// ----------------------- Helper functions -------------------

/// Generates a uniform probability grid of length `n`.
///
/// Computes the grid using the formula $(i + 0.5) / n$.
/// Allocates and returns a new vector containing the probability levels.
pub(crate) fn probs_grid(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i: usize| (i as f64 + 0.5) / (n as f64))
        .collect::<Vec<f64>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::col;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_probs_grid_edge_cases() {
        // A length of 0 resolves to an empty vector, preventing downstream
        // quantile interpolation from panicking, but resulting in empty tensors.
        assert_eq!(probs_grid(0), Vec::<f64>::new());

        // A length of 1 defaults to the exact 50th percentile (median)
        assert_eq!(probs_grid(1), vec![0.5]);

        // A length of 2 evenly divides the probability space
        assert_eq!(probs_grid(2), vec![0.25, 0.75]);
    }

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
    fn test_trait_defaults() {
        let panel: ObsPanel = setup_test_panel(); // U=2, T=2 (Total cells = 4)

        assert_eq!(panel.n_cells(), 4);

        // Validating cell_size mapping
        // Period 0: U0=[1.0, 2.0], U1=[3.0]
        // Period 1: U0=[4.0], U1=[5.0, 6.0, 7.0]
        assert_eq!(panel.cell_size(0, 0), 2);
        assert_eq!(panel.cell_size(1, 0), 1);
        assert_eq!(panel.cell_size(0, 1), 1);
        assert_eq!(panel.cell_size(1, 1), 3);

        // Validating period_size prefix-sum derivations
        assert_eq!(panel.period_size(0), 3); // 2 + 1
        assert_eq!(panel.period_size(1), 4); // 1 + 3
    }

    #[test]
    fn test_panel_errors() {
        let res: Vec<f64> = vec![1.0, 2.0];
        let u: Vec<i32> = vec![0, 1];
        let p: Vec<i32> = vec![0, 0];

        // 1. ZeroDims Error
        assert_eq!(
            ObsPanel::new(&res, &u, &p, 0, 1).unwrap_err(),
            PanelError::ZeroDims {
                n_units: 0,
                n_periods: 1
            }
        );

        // 2. LengthMismatch for unit IDs
        assert_eq!(
            ObsPanel::new(&res, &vec![0], &p, 2, 1).unwrap_err(),
            PanelError::LengthMismatch {
                expected: 2,
                actual: 1,
                vector: "unit_ids"
            }
        );

        // 3. LengthMismatch for period IDs
        assert_eq!(
            ObsPanel::new(&res, &u, &vec![0, 0, 0], 2, 1).unwrap_err(),
            PanelError::LengthMismatch {
                expected: 2,
                actual: 3,
                vector: "period_ids"
            }
        );

        // 4. IdOutOfRange for units (e.g., unit '2' provided but max is 1)
        assert_eq!(
            ObsPanel::new(&res, &vec![0, 2], &p, 2, 1).unwrap_err(),
            PanelError::IdOutOfRange {
                id: 2,
                n: 2,
                vector: "unit_ids",
                at: 1
            }
        );

        // 5. IdOutOfRange for periods
        assert_eq!(
            ObsPanel::new(&res, &u, &vec![0, 1], 2, 1).unwrap_err(),
            PanelError::IdOutOfRange {
                id: 1,
                n: 1,
                vector: "period_ids",
                at: 1
            }
        );

        // 6. HasNaN guard check
        assert_eq!(
            ObsPanel::new(&vec![1.0, f64::NAN], &u, &p, 2, 1).unwrap_err(),
            PanelError::HasNaN { at: 1 }
        );
    }

    // Helper to generate a small, predictable ragged panel
    fn setup_test_panel() -> ObsPanel {
        let response: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let units: Vec<i32> = vec![0, 0, 1, 0, 1, 1, 1];
        let periods: Vec<i32> = vec![0, 0, 0, 1, 1, 1, 1];
        // Period 0: U0=[1.0, 2.0], U1=[3.0]
        // Period 1: U0=[4.0], U1=[5.0, 6.0, 7.0]
        ObsPanel::new(&response, &units, &periods, 2, 2).unwrap()
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
        obs.to_quantiles_into(&mut out, &probs);
        assert_eq!(out.values.len(), expected_capacity);
        let first_pass_values = out.values.clone();

        // 3. Second evaluation pass (simulating a worker loop)
        obs.to_quantiles_into(&mut out, &probs);

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
        };

        // 2. Apply shifts (Unit 0 shifts down by 5, Unit 1 shifts down by 10)
        let shifts: Col<f64> = col![5.0, 10.0];
        qp.shift_all(&shifts);

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
        };

        // Apply scaling (Unit 0 scales down by 2.0, Unit 1 scales down by 10.0)
        let scalars: Col<f64> = faer::col![2.0, 10.0];
        qp.scale_all(&scalars);

        // Assert exact mathematical correctness
        assert_eq!(qp.values[0..3], [5.0, 10.0, 15.0]);
        assert_eq!(qp.values[3..6], [5.0, 6.0, 7.0]);
    }
}

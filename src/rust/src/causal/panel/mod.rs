//! Contiguous 1D memory representations for ragged and rectangular distributional panel data.
//!
//! Defines the core [`Panel`] trait alongside unit-level descriptive statistics,
//! zero-allocation iterators, and submodule re-exports.

#![allow(unused)]
#![allow(non_snake_case)]

// child submodules
mod config;
mod error;
mod grid;
mod obs;
mod quantile;
mod resample;

// Reexportiung public API (facade pattern)
pub use config::*;
pub use error::*;
pub use obs::*;
pub use quantile::*;
pub use resample::*;

// Reexprting internal helper
pub(crate) use grid::probs_grid;

// External crate imports
use faer::{Col, ColMut, ColRef};

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
    /// Calculates the arithmetic means of all units via Neumaier compensated summation
    /// and writes them into the provided buffer.
    fn mean_all_into(&self, mut out: ColMut<f64>) {
        // Use mutable vector to apply the division to each element, for_each?
        let mut c: Col<f64> = Col::zeros(self.n_units());
        let mut n: Vec<usize> = vec![0; self.n_units()];
        out.fill(0.0);

        // Nested for loops for neumaier summation over periods and units
        for t in 0..self.n_periods() {
            for u in 0..self.n_units() {
                // Cell slice and length extraction
                let cell: &[f64] = self.cell(u, t);
                n[u] += cell.len();
                // Summation loop over the sequential cell
                for &y in cell.iter() {
                    let tval: f64 = out[u] + y;
                    if out[u].abs() >= y.abs() {
                        c[u] += (out[u] - tval) + y;
                    } else {
                        c[u] += (y - tval) + out[u];
                    }
                    out[u] = tval;
                }
            }
        }

        // Applying bit loss compensation and calculating means
        for u in 0..self.n_units() {
            if n[u] == 0 {
                out[u] = f64::NAN; // Guarding against division by zero for empty units
            } else {
                out[u] = (out[u] + c[u]) / n[u] as f64;
            }
        }
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
    /// Computes the median across all units and writes them into the provided buffer.
    /// via an introselect implementation based on the "ipnsort" algorithm by
    /// Lukas Bergdoll and Orson Peters. The fallback algorithm is Median of Medians
    /// using Tukey’s Ninther.
    ///
    /// Performance: Highly optimized, reuses a single pre-allocated scratchpad for all units,
    /// performing exactly one heap allocation for the scratchpad itself.
    fn median_all_into(&self, mut out: ColMut<f64>, pad: &mut Vec<f64>) {
        // 1. Identify the maximum unit size across the panel
        // Calculating the maximum unit size (observations across all periods) to
        // allocate the scratchpad exactly once to handle the worst-case unit.
        let mut size_max: usize = 0;
        for ui in 0..self.n_units() {
            let size: usize = std::cmp::max(self.unit_size(ui), size_max);
            size_max = size;
        }
        pad.clear();
        pad.reserve(size_max);

        // 2. Iterate over units and populate the pre-allocated output buffer directly
        for u in 0..self.n_units() {
            // Gather all observations for this unit across all periods into the contiguous pad
            for t in 0..self.n_periods() {
                pad.extend_from_slice(self.cell(u, t));
            }
            let n: usize = pad.len();
            // Handle missing or empty unit data cleanly using the poison value
            if n == 0 {
                out[u] = f64::NAN;
                continue;
            }

            // 3. Introselect algorithm for O(N) median finding
            let k: usize = n / 2;
            let (less, upp_mid, _) = pad.select_nth_unstable_by(k, f64::total_cmp);

            // 4. Direct assignment to the output buffer
            if n % 2 == 0 {
                let low_mid: f64 = less.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                out[u] = (low_mid + *upp_mid) / 2.0;
            } else {
                out[u] = *upp_mid;
            }

            pad.clear();
        }
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

// Shared test helper accessible to all `panel/*` submodules via `super::setup_test_panel()`
#[cfg(test)]
pub(crate) fn setup_test_panel() -> ObsPanel {
    let response: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let units: Vec<i32> = vec![0, 0, 1, 0, 1, 1, 1];
    let periods: Vec<i32> = vec![0, 0, 0, 1, 1, 1, 1];
    // Period 0: U0=[1.0, 2.0], U1=[3.0]
    // Period 1: U0=[4.0], U1=[5.0, 6.0, 7.0]
    ObsPanel::new(&response, &units, &periods, 2, 2).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

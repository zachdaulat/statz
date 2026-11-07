//! Domain validation and runtime state errors (`PanelError`) for panel initialization and mutation.

use super::LocAdj;

/// Represents validtaion failures encountered during Panel data structure initialization.
///
/// The panel structures rely on dense contiguous memory and matrix conformability,
/// so these errors and checks enforce invalid inputs before any heavy allocations occur.
#[derive(Debug, Clone, PartialEq)]
pub enum PanelError {
    /// One of the ID vector lengths do not match the `response` vector length.
    DataLengthMismatch {
        expected: usize,
        actual: usize,
        vector: &'static str,
    },
    /// Locations vector length does not match the panel's unit count.
    UnitLengthMismatch { expected: usize, actual: usize },
    /// Attempted to shift of scale a `QuantilePanel` before computing unit locations.
    LocationsNotComputed,
    /// An ID in the data exceeds the user-specified ID range
    IdOutOfRange {
        id: i32,
        n: i32,
        vector: &'static str,
        at: usize,
    },
    /// The specified treated unit index does not exist within the panel dimensions.
    InvalidTreatedUnit { unit: i32, n_units: i32 },
    /// The specified treatment period does not allow for a valid pre/post split.
    InvalidTreatedPeriod { period: usize, n_periods: usize },
    /// The dataset contains missing values. The Distributional Synthetic Control
    /// pipeline requires complete observations for evaluation.
    HasNaN { at: usize },
    /// A given unit-period cell contains no observations.
    EmptyCell { unit: usize, period: usize },
    /// The panel was initialized with a dimension of 0.
    ZeroDims { n_units: i32, n_periods: i32 },
    /// The panel has already been transformed. Multiple location
    /// adjustments on the same panel are mathematically invalid.
    AlreadyAdjusted { current_adj: LocAdj },
}

impl std::error::Error for PanelError {}

impl std::fmt::Display for PanelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PanelError::DataLengthMismatch {
                expected,
                actual,
                vector,
            } => write!(
                f,
                "`{vector}` has length {actual}, expected {expected} to match `response`"
            ),
            PanelError::UnitLengthMismatch { expected, actual } => write!(
                f,
                "unit location adjustments lengths mismatch: expected {expected} units, received {actual}"
            ),
            PanelError::LocationsNotComputed => write!(
                f,
                "cannot apply a location adjustment before unit locations (mean or median) have been computed"
            ),
            PanelError::IdOutOfRange { id, n, vector, at } => write!(
                f,
                "`{vector}[{}]` is {id}; identifiers must lie in [0, {n})",
                at + 1
            ),
            PanelError::InvalidTreatedUnit { unit, n_units } => {
                write!(f, "`treated_unit` is {unit}; must lie in [0, {n_units})")
            }
            PanelError::InvalidTreatedPeriod { period, n_periods } => {
                write!(f, "`t0` is {period}; must lie in strictly (0, {n_periods}) to generate valid pre- and post-treatment panels")
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
            PanelError::AlreadyAdjusted { current_adj } => {
                // Map the internal enum to a user-friendly string for the R console
                let adj_str = match current_adj {
                    LocAdj::Shift => "translation (shift)",
                    LocAdj::Scale => "scaling",
                    LocAdj::None => "none", // Theoretically unreachable in an error state
                };
                write!(
                    f,
                    "quantile panel has already received a {} adjustment; applying multiple transformations to the same panel is mathematically invalid",
                    adj_str
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::panel::ObsPanel;

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
            PanelError::DataLengthMismatch {
                expected: 2,
                actual: 1,
                vector: "unit_ids"
            }
        );

        // 3. LengthMismatch for period IDs
        assert_eq!(
            ObsPanel::new(&res, &u, &vec![0, 0, 0], 2, 1).unwrap_err(),
            PanelError::DataLengthMismatch {
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
}

//! Error variants (`DscError`) for DSC input validation, workspace bounds,
//! and numerical solver failures.

use crate::causal::panel::PanelError;

#[derive(Debug, Clone, PartialEq)]
pub enum DscError {
    /// Panel has no periods, or the probability grid is empty
    EmptyPanel { n_periods: usize, n_probs: usize },
    /// No donors provided to construct the synthetic control.
    NoDonors,
    /// Target unit index is outside the panel unit bounds [0, n_units).
    TargetOutOfRange { target: usize, n_units: usize },
    /// A donor unit index is outside the panel unit bounds [0, n_units).
    DonorOutOfRange { donor: usize, n_units: usize },
    /// The target unit appears within the donor pool.
    TargetInDonors { target: usize },
    /// A donor unit appears multiple times in the donor pool.
    DuplicateDonor { donor: usize },
    /// Pre-allocated scratchpad dimensions are not sufficient for the required problem size.
    ScratchMismatch {
        required_q: usize,
        allocated_q: usize,
        required_donors: usize,
        allocated_donors: usize,
    },
    /// NaN or infinity reached a quantity that must be finite.
    NonFinite { context: &'static str },
    /// Penalty must be finite and non-negative.
    InvalidPenalty(f64),
    /// Tolerance must be finite and strictly positive.
    InvalidTolerance(f64),
    /// Maximum iterations must be strictly positive.
    InvalidMaxIter(i32),
    /// Self-adjoint eigendecomposition of the Gram matrix failed.
    EigenFailure,
    /// Placebo filter rejected all candidate units.
    NoValidPlacebos {
        n_candidates: usize,
        n_passed: usize,
    },
    /// Upstream panel construction or access error.
    Panel(PanelError),
}

impl From<PanelError> for DscError {
    fn from(e: PanelError) -> Self {
        DscError::Panel(e)
    }
}

impl std::error::Error for DscError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DscError::Panel(e) => Some(e),
            _ => None,
        }
    }
}

impl std::fmt::Display for DscError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DscError::EmptyPanel { n_periods, n_probs } => write!(
                f,
                "panel is empty; received {n_periods} periods and {n_probs} probability levels"
            ),
            DscError::NoDonors => write!(f, "donor pool is empty; at least one donor is required"),
            DscError::TargetOutOfRange { target, n_units } => write!(
                f,
                "target unit index {target} is out of bounds; panel contains {n_units} units"
            ),
            DscError::DonorOutOfRange { donor, n_units } => write!(
                f,
                "donor unit index {donor} is out of bounds; panel contains {n_units} units"
            ),
            DscError::TargetInDonors { target } => write!(
                f,
                "target unit {target} cannot be included in the donor pool"
            ),
            DscError::DuplicateDonor { donor } => write!(
                f,
                "donor unit {donor} appears multiple times in the donor pool"
            ),
            DscError::ScratchMismatch {
                required_q,
                allocated_q,
                required_donors,
                allocated_donors,
            } => write!(
                f,
                "scratchpad capacity insufficient: required at least ({required_q} x {required_donors}), but allocated buffer is ({allocated_q} x {allocated_donors})"
            ),
            DscError::NonFinite { context } => write!(
                f,
                "non-finite value (NA, NaN, or Inf) encountered during: {context}"
            ),
            DscError::InvalidPenalty(p) => write!(
                f,
                "penalty must be finite and non-negative; received {p}"
            ),
            DscError::InvalidTolerance(tol) => write!(
                f,
                "convergence tolerance must be finite and strictly positive; received {tol}"
            ),
            DscError::InvalidMaxIter(iter) => write!(
                f,
                "max iterations must be strictly positive; received {iter}"
            ),
            DscError::EigenFailure => write!(
                f,
                "error in the self-adjoint eigendecomposition of the Gram matrix"
            ),
            DscError::NoValidPlacebos { n_candidates, n_passed } => write!(
                f,
                "no valid placebo donors available: {n_passed} of {n_candidates} candidates passed the pre-fit filter"
            ),
            DscError::Panel(e) => write!(f, "{e}"),
        }
    }
}

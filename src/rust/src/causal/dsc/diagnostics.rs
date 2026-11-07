//! Spectral diagnostics (`DscDiagnostics`, `dsc_diagnostics`) for the donor Gram matrix.
//!
//! Computes self-adjoint eigendecompositions, singular value spectra, condition
//! numbers ($\kappa$), and numerical effective rank for regularized and unregularized fits.

use super::{DscError, DscScratch};
use faer::{linalg::solvers::SelfAdjointEigen, mat::AsMatRef, Col, ColRef, Mat, MatRef, Side};

#[rustfmt::skip]
pub struct DscDiagnostics {
    pub evecs:     Mat<f64>,
    pub svs:       Col<f64>,
    pub svs_l2:    Col<f64>,
    pub kappa:     f64,
    pub kappa_l2:  f64,
    pub sv_max:    f64,
    pub sv_min:    f64,
    pub eff_rank:  i32,
}

pub(crate) fn dsc_diagnostics(
    pad: &DscScratch,
    j: usize,
    penalty: f64,
    is_regularized: bool,
) -> Result<DscDiagnostics, DscError> {
    #![allow(non_snake_case)]
    // 0.0 - Input validity checks
    // 0.1 - Donor count check
    if j == 0 {
        return Err(DscError::NoDonors);
    }
    // 0.2 - Scratchpad capacity check (guarding against panics when taking submatrix)
    if pad.gram.nrows() < j || pad.gram.ncols() < j {
        return Err(DscError::ScratchMismatch {
            required_q: 0,
            allocated_q: 0,
            required_donors: j,
            allocated_donors: pad.gram.ncols(),
        });
    }
    // 0.3 - Optimization parameter checks
    if !penalty.is_finite() || penalty < 0.0 {
        return Err(DscError::InvalidPenalty(penalty));
    }

    // 0.4 - Extract active submatrix for the decomposition
    let G: MatRef<f64> = pad.gram.as_mat_ref().submatrix(0, 0, j, j);

    // 0.5 - Set L2 penalty and mathematical shifts on Gram matrix state
    let l2_penalty: f64 = 2.0 * penalty;
    let (shift_l2, shift_unreg) = if is_regularized {
        (0.0, -l2_penalty) // Evals already have penalty; back it out for unreg
    } else {
        (l2_penalty, 0.0) // Evals lack penalty; add it for l2
    };

    // 1.0 - Self-Adjoint Eigendecomposition of the Gram matrix
    let eigen: SelfAdjointEigen<f64> =
        SelfAdjointEigen::new(G, Side::Upper).map_err(|_| DscError::EigenFailure)?;

    // 1.1 - Extracting decomposition outputs
    let evals: ColRef<f64> = eigen.S().column_vector();
    let evecs: MatRef<f64> = eigen.U();

    // 1.2 - Diagnostics: Regularized singular values and condition number of D
    let svs_l2: Col<f64> = Col::from_fn(j, |i| f64::max(evals[i] + shift_l2, 0.0).sqrt());
    let sv_l2_min: f64 = svs_l2.min().ok_or(DscError::EigenFailure)?;
    let sv_l2_max: f64 = svs_l2.max().ok_or(DscError::EigenFailure)?;
    let kappa_l2: f64 = sv_l2_max / sv_l2_min;

    // 1.3 - Unregularized singular values and condition number of D
    let svs: Col<f64> = Col::from_fn(j, |i| f64::max(evals[i] + shift_unreg, 0.0).sqrt());
    let sv_min: f64 = svs.min().ok_or(DscError::EigenFailure)?;
    let sv_max: f64 = svs.max().ok_or(DscError::EigenFailure)?;
    let kappa: f64 = sv_max / sv_min;

    // 1.4 - Diagnostics: Effective rank
    let rank_tol: f64 = sv_max * (j as f64) * f64::EPSILON;
    let effective_rank: i32 = svs.iter().filter(|&sigma| sigma > &rank_tol).count() as i32;

    // 2.0 - Populate output struct with diagnostics
    Ok(DscDiagnostics {
        svs,
        svs_l2,
        evecs: evecs.to_owned(),
        kappa,
        kappa_l2,
        sv_max,
        sv_min,
        eff_rank: effective_rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // Tests for dsc_diagnostics
    // ==========================================
    #[test]
    fn test_dsc_diagnostics_regularization_shifts() {
        let mut pad = DscScratch::new(2, 3);
        let j = 3;
        let penalty = 2.0; // l2_penalty = 4.0

        // 1. Evaluate the UNREGULARIZED matrix
        pad.gram[(0, 0)] = 10.0;
        pad.gram[(1, 1)] = 20.0;
        pad.gram[(2, 2)] = 30.0;
        let diag_unreg = dsc_diagnostics(&pad, j, penalty, false).unwrap();

        // 2. Evaluate the REGULARIZED matrix
        // Manually simulate `dsc_fit` adding `l2_penalty` (4.0) to the diagonal
        pad.gram[(0, 0)] = 14.0;
        pad.gram[(1, 1)] = 24.0;
        pad.gram[(2, 2)] = 34.0;
        let diag_reg = dsc_diagnostics(&pad, j, penalty, true).unwrap();

        // Now the mathematical truths should perfectly align across both branches
        for i in 0..j {
            // Check that diag_reg.svs (which backs out the penalty) equals diag_unreg.svs (which did nothing)
            assert!(
                (diag_reg.svs[i] - diag_unreg.svs[i]).abs() < 1e-9,
                "Penalty subtraction failed at index {}",
                i
            );

            // Check that diag_unreg.svs_l2 (which adds the penalty) equals diag_reg.svs_l2 (which did nothing)
            assert!(
                (diag_unreg.svs_l2[i] - diag_reg.svs_l2[i]).abs() < 1e-9,
                "Penalty addition failed at index {}",
                i
            );
        }
    }
}

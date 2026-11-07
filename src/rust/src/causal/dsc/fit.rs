//! Core DSC optimization engine (`dsc_fit`), input pre-flight checks (`dsc_check`), and solver state (`DscFit`).
//!
//! Solves for optimal donor weights minimizing squared 2-Wasserstein ($W_2$) distance
//! via projected gradient descent over pre-treatment quantile panels.

use super::{project_simplex_into, DscError, DscScratch};
use crate::causal::panel::QuantilePanel;
use faer::{
    col::AsColMut,
    linalg::matmul::{
        self,
        triangular::{self, BlockStructure},
    },
    mat::AsMatMut,
    reborrow::{Reborrow, ReborrowMut},
    Accum, Col, ColMut, ColRef, MatMut, MatRef, Par,
};

// ------------------- Output struct -------------------------
#[rustfmt::skip]
pub(crate) struct DscFit {
    pub weights:   Col<f64>,      // length J-1, in `donors` order
    pub loss:      f64,           // unpenalized — the W2 distance
    pub loss_pen:  f64,
    pub a_sq:      f64,
    pub n_iter:    i32,
    pub converged: bool,
}

impl DscFit {
    pub fn new(max_donors: usize) -> Self {
        Self {
            weights: Col::zeros(max_donors),
            loss: 0.0,
            loss_pen: 0.0,
            a_sq: 0.0,
            n_iter: 0,
            converged: false,
        }
    }
}

// ------------------- DSC fit engine -------------------------------
pub fn dsc_fit(
    qp: &QuantilePanel,
    target: usize,
    donors: &[usize],
    penalty: f64,
    max_iter: i32,
    tol: f64,
    pad: &mut DscScratch,
    fit: &mut DscFit,
) -> Result<(), DscError> {
    #![allow(non_snake_case)]

    // 0.1 - Initialize variables
    let n_probs: usize = qp.probs.len();
    let T_0: usize = qp.n_periods;
    let j: usize = donors.len();

    // 0.2 - Establish the mutable active views of the scratchpad buffers
    // By shadowing the allocated memory to mutable views, I isolate the math code
    // from the allocated memory capacity
    let mut D: MatMut<f64> = pad.donors.as_mat_mut().submatrix_mut(0, 0, n_probs, j);
    let mut G: MatMut<f64> = pad.gram.as_mat_mut().submatrix_mut(0, 0, j, j);
    let mut c: ColMut<f64> = pad.cross.as_col_mut().subrows_mut(0, j);
    let mut w: ColMut<f64> = pad.w.as_col_mut().subrows_mut(0, j);
    let mut w_new: ColMut<f64> = pad.w_new.as_col_mut().subrows_mut(0, j);
    let mut a_sq: f64 = 0.0; // Target unit squared L2 norm

    // 0.3 - Reset, clear, and initialize buffers
    fit.converged = false; // reset converged flag
    w.fill((j as f64).recip()); // Uniform weights
    G.fill(0.0); // reset Gram with 0s
    c.fill(0.0); // reset cross with 0s

    // 1 - Map-reduce donor matrix accumulation loop
    for t in 0..T_0 {
        // 1.1 - Collect donors into buffers and get treated view
        let t_mat: MatRef<f64> = qp.mat_ref(t);
        let a: ColRef<f64> = t_mat.col(target);

        qp.select_units_into(donors, t, D.rb_mut());

        // 1.2 - matmul accumulation steps
        triangular::matmul(
            G.rb_mut(),
            BlockStructure::TriangularUpper,
            Accum::Add,
            D.rb().transpose(),
            BlockStructure::Rectangular,
            D.rb(),
            BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );

        matmul::matmul(
            // Cross correlation vector accumulation
            c.rb_mut(),
            Accum::Add,
            D.rb().transpose(),
            a,
            1.0,
            Par::Seq,
        );

        // Accumulate target squared norm
        a_sq += a.squared_norm_l2();
    }

    // 2 - Preparing Gram matrix, cross vector, and eigendecomposition diagnostics
    // 2.1 - Reflecting computed upper side to the lower side triangle
    // unsafe just kept because it works,
    for col in 1..j {
        for row in 0..col {
            unsafe {
                // Immutable borrow first prevents simultaneous veiws to same matrix G?
                let val: f64 = *G.rb().get_unchecked(row, col);
                // Mutable borrow to assign value
                *G.rb_mut().get_mut_unchecked(col, row) = val;
            };
        }
    }

    // 2.2 - Applying normalization constant and regularization
    let scalar: f64 = 2.0 / (T_0 * n_probs) as f64;
    faer::zip!(G.rb_mut()).for_each(|faer::unzip!(g)| {
        *g *= scalar;
    });
    faer::zip!(c.rb_mut()).for_each(|faer::unzip!(ci)| {
        *ci *= scalar;
    });
    // Zero-allocation diagonal mutation applying L2 penalty to Gram matrix
    let l2_penalty: f64 = 2.0 * penalty;
    faer::zip!(G.rb_mut().diagonal_mut()).for_each(|faer::unzip!(d)| {
        *d += l2_penalty;
    });

    // 2.3 - Step size via max column 1-norm from largest eigenvalue upper bound
    let mut g_norm: f64 = 0.0;
    for col in G.rb().col_iter() {
        let sum: f64 = col.norm_l1();
        g_norm = f64::max(g_norm, sum);
    }
    let eta: f64 = g_norm.recip();

    // Guard against degenerate zero-matrices yielding infinite step sizes
    if !eta.is_finite() {
        return Err(DscError::NonFinite {
            context: "step size computation (Gram matrix is degenerate/zero)",
        });
    }

    // 3. - Projected Gradient Descent
    // G and c are scaled and L2 penalty applied to G
    // ∇f(w) = Gw - c
    for iter in 1..=max_iter {
        fit.n_iter = iter;

        // 3.1 - Prime w_new with element wise step expansion (w + eta * c)
        // Formula: w_new = w + (eta * c)
        faer::zip!(w_new.rb_mut(), w.rb(), c.rb())
            .for_each(|faer::unzip!(w_ni, wi, ci)| *w_ni = wi + (eta * ci));

        // 3.2 - Matrix multiplication stepping weights directly into primed w_new
        // matmul formula: acc = (alpha * lhs * rhs) + (Accum::Add * acc)
        // Formula: w_new -= eta * G * w
        matmul::matmul(
            // Accumulate downhill step into w_new
            w_new.rb_mut(),
            Accum::Add,
            G.rb(),
            w.rb(),
            -eta,
            Par::Seq,
        );

        // 3.3 - Project onto simplex in-place
        project_simplex_into(w_new.rb_mut(), &mut pad.w_buf);

        // 3.4 - Check convergence
        // Manual L2 norm of weights differences for better vectorization
        let mut sum: f64 = 0.0;
        for i in 0..j {
            let diff = w_new[i] - w[i];
            sum += diff * diff;
        }
        let delta: f64 = sum.sqrt();

        // Ping-pong buffer swap between weights vectors
        std::mem::swap(&mut w, &mut w_new);

        if delta < tol {
            fit.converged = true;
            break;
        }
    }

    // 4. - Final objective computations
    let a_sq_scaled: f64 = a_sq / (T_0 as f64 * n_probs as f64);
    // Compute w^T * G * w
    matmul::matmul(
        w_new.rb_mut(),
        Accum::Replace,
        G.rb(),
        w.rb(),
        1.0,
        Par::Seq,
    );
    let mut wGw: f64 = 0.0;
    for i in 0..j {
        wGw += w[i] * w_new[i];
    }

    // Compute C^t * w
    let mut cw: f64 = 0.0;
    for i in 0..j {
        cw += c[i] * w[i];
    }

    fit.a_sq = a_sq_scaled;
    fit.loss_pen = a_sq_scaled - cw + (0.5 * wGw);
    fit.loss = fit.loss_pen - (penalty * w.squared_norm_l2());
    fit.weights.as_col_mut().subrows_mut(0, j).copy_from(w.rb());

    Ok(())
}

/// Function to validate an input `QuantilePanel` a single time to enable faster reuse
pub(crate) fn dsc_check(
    qp: &QuantilePanel,
    target: usize,
    donors: &[usize],
    penalty: f64,
    max_iter: i32,
    tol: f64,
    pad: &DscScratch,
) -> Result<(), DscError> {
    let n_units: usize = qp.n_units;
    let n_periods: usize = qp.n_periods;
    let n_probs: usize = qp.probs.len();
    let j: usize = donors.len();

    // 0. -------------- Input Validity Checks ------------

    // 0.1 Structural panel checks
    if n_periods == 0 || n_probs == 0 {
        return Err(DscError::EmptyPanel { n_periods, n_probs });
    }
    if j == 0 {
        return Err(DscError::NoDonors);
    }
    if target >= n_units {
        return Err(DscError::TargetOutOfRange { target, n_units });
    }

    // 0.2 Optimization parameter checks
    if !penalty.is_finite() || penalty < 0.0 {
        return Err(DscError::InvalidPenalty(penalty));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(DscError::InvalidTolerance(tol));
    }
    if max_iter <= 0 {
        return Err(DscError::InvalidMaxIter(max_iter));
    }

    // 0.3 Scratchpad conformability checks
    if pad.donors.nrows() < n_probs || pad.donors.ncols() < j {
        return Err(DscError::ScratchMismatch {
            required_q: n_probs,
            allocated_q: pad.donors.nrows(),
            required_donors: j,
            allocated_donors: pad.donors.ncols(),
        });
    }

    // 0.4 Donor set integrity checks
    for (i, &d) in donors.iter().enumerate() {
        if d >= n_units {
            return Err(DscError::DonorOutOfRange { donor: d, n_units });
        }
        if d == target {
            return Err(DscError::TargetInDonors { target });
        }
        if donors[..i].contains(&d) {
            return Err(DscError::DuplicateDonor { donor: d });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::dsc::test_fixtures::{mock_panel, mock_panel_linear_combo};
    use faer::{col, mat, mat::AsMatRef};

    // ==========================================
    // Tests for dsc_fit
    // ==========================================

    #[test]
    fn test_dsc_fit_gram_by_hand() {
        #![allow(non_snake_case)]
        // 1. Setup mock panel: 2 periods, 2 quantiles, 4 units (target = 0, donors = 1, 2, 3)
        // Let Donor 1 = [1.0, 2.0], Donor 2 = [3.0, 4.0], Donor 3 = [5.0, 6.0] for both periods.
        let qp = mock_panel();

        let mut pad = DscScratch::new(2, 3);
        let mut fit = DscFit::new(3);

        // 2. Run fit
        dsc_fit(&qp, 0, &[1, 2, 3], 0.0, 10, 1e-6, &mut pad, &mut fit).unwrap();

        // 3. Hand-calculate expected Gram matrix
        // D^T * D for one period =
        // [1  2]   [1  3  5]   [ 5  11  17]
        // [3  4] * [2  4  6] = [11  25  39]
        // [5  6]               [17  39  61]
        // Across 2 periods, the sum doubles.
        // The scaling factor is 2.0 / (T_0 * n_probs) = 2.0 / (2 * 2) = 0.5.
        // Therefore, the final scaled G is exactly equal to the single period D^T * D.
        let expected_G = mat![[5.0, 11.0, 17.0], [11.0, 25.0, 39.0], [17.0, 39.0, 61.0]];

        let j = 3;
        let G = pad.gram.as_mat_ref().submatrix(0, 0, j, j);

        // 4. Assert full matrix to ensure reflection loop covers both triangles
        for row in 0..j {
            for col in 0..j {
                assert!(
                    (G.get(row, col) - expected_G.get(row, col)).abs() < 1e-9,
                    "Gram matrix mismatch at ({}, {})",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_dsc_fit_known_answer() {
        // 1. Setup mock panel where Target is a perfect linear combination of Donors 1 and 2.
        // Target = 0.3 * Donor 1 + 0.7 * Donor 2 + 0.0 * Donor 3
        let qp = mock_panel_linear_combo();

        let mut pad = DscScratch::new(qp.probs.len(), 3);
        let mut fit = DscFit::new(3);

        // 2. Run fit with NO penalty, high max_iter, and tight tolerance
        dsc_fit(&qp, 0, &[1, 2, 3], 0.0, 1000, 1e-8, &mut pad, &mut fit).unwrap();

        // 3. Assert the weights recovered the exact linear combination
        let expected_weights = col![0.3, 0.7, 0.0];
        let j = 3;

        for i in 0..j {
            assert!(
                (fit.weights.get(i) - expected_weights.get(i)).abs() < 1e-5,
                "Failed to recover known weights at index {}",
                i
            );
        }

        // 4. Assert W2 distance (loss) collapsed effectively to zero
        assert!(
            fit.loss < 1e-7,
            "Perfect fit did not result in near-zero loss"
        );
    }

    #[test]
    fn test_dsc_fit_convergence_reset() {
        let qp = mock_panel();
        let mut pad = DscScratch::new(qp.probs.len(), 3);
        let mut fit = DscFit::new(3);

        // 1. First run: Generous iterations, should converge easily
        dsc_fit(&qp, 0, &[1, 2, 3], 0.1, 1000, 1e-4, &mut pad, &mut fit).unwrap();
        assert!(fit.converged, "First run failed to converge as expected");

        // 2. Second run: Impossible tolerance and only 1 iteration
        dsc_fit(&qp, 0, &[1, 2, 3], 0.1, 1, 1e-12, &mut pad, &mut fit).unwrap();

        // 3. Assert the flag dropped correctly
        assert!(!fit.converged, "Convergence flag leaked from previous run");
    }

    // ==========================================
    // Tests for dsc_check
    // ==========================================

    #[test]
    fn test_dsc_check_errors() {
        let qp = mock_panel(); // n_units = 4
        let pad = DscScratch::new(qp.probs.len(), 3);

        // 1. Target Out of Range
        assert_eq!(
            dsc_check(&qp, 99, &[1, 2, 3], 0.1, 100, 1e-4, &pad).unwrap_err(),
            DscError::TargetOutOfRange {
                target: 99,
                n_units: 4
            }
        );

        // 2. Target inside Donor Pool
        assert_eq!(
            dsc_check(&qp, 0, &[1, 0, 3], 0.1, 100, 1e-4, &pad).unwrap_err(),
            DscError::TargetInDonors { target: 0 }
        );

        // 3. Duplicate Donors
        assert_eq!(
            dsc_check(&qp, 0, &[1, 2, 1], 0.1, 100, 1e-4, &pad).unwrap_err(),
            DscError::DuplicateDonor { donor: 1 }
        );

        // 4. Invalid Optimization Parameters
        assert_eq!(
            dsc_check(&qp, 0, &[1, 2, 3], -5.0, 100, 1e-4, &pad).unwrap_err(),
            DscError::InvalidPenalty(-5.0)
        );
        assert_eq!(
            dsc_check(&qp, 0, &[1, 2, 3], 0.1, 0, 1e-4, &pad).unwrap_err(),
            DscError::InvalidMaxIter(0)
        );

        // 5. Scratchpad Mismatch (Passing 4 donors to a 3-capacity scratchpad)
        assert_eq!(
            dsc_check(&qp, 0, &[1, 2, 3, 4], 0.1, 100, 1e-4, &pad).unwrap_err(),
            DscError::ScratchMismatch {
                required_q: 2, // Your struct currently passes 0 here in `dsc_diagnostics` but `dsc_check` uses n_probs
                allocated_q: pad.donors.nrows(),
                required_donors: 4,
                allocated_donors: 3,
            }
        );
    }
}

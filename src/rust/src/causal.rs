use std::{f64};
use extendr_api::{Error, prelude::*};
use faer::{
    Col, ColMut, ColRef, Mat, MatRef, Side, 
    col::AsColRef, mat::AsMatRef,
    linalg::solvers::SelfAdjointEigen,
};
use crate::linear_models::RMatrixExt;

extendr_module! {
    mod causal;
    fn z_dsc_rs;
}

// Future refinement when doing bootstrapping, modularize into separate functions
// to enable crate-level testing with wrapper doing R object view conversion?
// Distributional synthetic controls implementation
#[extendr]
#[allow(non_snake_case)]
pub fn z_dsc_rs(
    treated: List,      // Each element: Doubles vectors of treated unit obs by bucket
    donors: List,       // Each element: List of Doubles, one per donor by bucket
    n_quantiles: i32,   // Q, number of quantiles to sample from each bucket
    lambda: f64,        // L2 penalty
    max_iter: i32,      // Maximum number of iterations for gradient descent loop
    tol: f64,           // Threshold for change in weights vector norm
) -> extendr_api::Result<List> {
    // 1. Construct probability grid
    let probs: Vec<f64> = (0..n_quantiles)
        .map(|i: i32| (i as f64 + 0.5) / (n_quantiles as f64))
        .collect::<Vec<f64>>();

    // 2. References construction to R the R bucket data
    let T_0 = treated.len();
    
    // Owning all of the Robj wrappers to ensure compiler lifetimes
    let treated_objs: Vec<Robj> = treated.values().collect();
    let donor_objs: Vec<Vec<Robj>> = donors
        .values()
        .map(|d| {
            d.as_list()
             .map(|l| l.values().collect())
        })
        .collect::<Option<Vec<Vec<Robj>>>>()
        .ok_or(Error::from("Each donor bucket must be a nested list"))?;

    // Safely borrowing slices from the owned wrappers
    let mut buckets: Vec<BucketData> = Vec::with_capacity(T_0);

    // Iterating over treated and donors lists to populate slices container
    for (i, tr_obj) in treated_objs.iter().enumerate() {

        let tr_slice: &[f64] = tr_obj
            .as_real_slice()
            .ok_or(Error::from("Treated data must be numeric doubles"))?;

        let mut donors_slices: Vec<&[f64]> = Vec::with_capacity(donor_objs[i].len());

        for d_obj in &donor_objs[i] {
            let dn_slice: &[f64] = d_obj
                .as_real_slice()
                .ok_or(Error::from("Donor data must be numeric doubles"))?;
            donors_slices.push(dn_slice);
        }

        buckets.push(BucketData {
            treated: tr_slice,
            donors: donors_slices,
        })
    }

    // 3. Map-Reduce step constructing Gram matrix G and cross-correlation vector c
    let (G_sum, c_sum, a_sq_sum): (Mat<f64>, Col<f64>, f64) = buckets
        .iter()
        .map(|b| {
            // --- MAP PHASE ---

            // 3.1 Evaluating treated unit quantiles (vector `a`)
            let a_b: Col<f64> = crate::descriptive::quantile(b.treated, &probs);

            // 3.2 Evaluating donor quantiles (matrix `D`)
            let j: usize = b.donors.len();
            let mut D_b: Mat<f64> = Mat::zeros(n_quantiles as usize, j);
            for (idx, slice) in b.donors.iter().enumerate() {
                let q_donor: Col<f64> = crate::descriptive::quantile(*slice, &probs);
                D_b.col_mut(idx).copy_from(&q_donor);
            }

            // 3.3 Computing Gram matrix, cross-correlation vector, treated sq L2 norm
            let G_b: Mat<f64> = D_b.transpose() * &D_b;
            let c_b: Col<f64> = D_b.transpose() * &a_b;
            let a_sq_b: f64 = a_b.squared_norm_l2();

            (G_b, c_b, a_sq_b)
        })
        .reduce(|acc, new| {
            // --- REDUCE PHASE ---

            // 3.4 Accumulating Gram matrices and vectors
            let acc_G: Mat<f64> = &acc.0 + &new.0;
            let acc_c: Col<f64> = &acc.1 + &new.1;
            let acc_a: f64 = &acc.2 + new.2;

            (acc_G, acc_c, acc_a)
        })
        .ok_or(Error::from("Mat-reduce failed: Bucket data iterator was empty"))?;

    // 4. Preparing Gram matrix and cross vector and eigendecomposition diagnostics
    // 4.1 Applying normalization constant
    let scalar: f64 = 2.0 / (T_0 as f64 * n_quantiles as f64);
    let mut G: Mat<f64> = &G_sum * scalar;
    let c: Col<f64> = &c_sum * scalar;
    let j: usize = c.nrows();

    // 4.2 Eigendecomposition of the normalized Gram matrix
    let eigen: SelfAdjointEigen<f64> = SelfAdjointEigen::new(G.as_mat_ref(), Side::Upper).map_err(|_| {
        Error::from("Error during Gram matrix eigendecomposition")
    })?;

    // Extracting decomposition outputs
    let evals: ColRef<f64> = eigen.S().column_vector();
    let evecs: MatRef<f64> = eigen.U();
    
    // Diagnostics: Singular values and condition number of D
    let svs: Col<f64> = Col::from_fn(j, |i| f64::max(evals[i], 0.0).sqrt());
    let sv_min: f64 = svs.min().ok_or(Error::from("Singular values `Col<f64>` is empty"))?;
    let sv_max: f64 = svs.max().ok_or(Error::from("Singular values `Col<f64>` is empty"))?;
    let kappa: f64 = sv_max / sv_min;

    // Diagnostics: Effective rank
    let rank_tol: f64 = sv_max * (j as f64) * f64::EPSILON;
    let effective_rank: i32 = svs
        .iter()
        .filter(|&sigma| sigma > &rank_tol)
        .count() as i32;

    // 4.3 Regularization
    let l2_penalty: f64 = 2.0 * lambda;

    // Zero-allocation diagonal mutation to apply L2 penalty to Gram matrix
    let mut G_diag: ColMut<f64> = G.diagonal_mut().column_vector_mut();
    for i in 0..j {
        G_diag[i] += l2_penalty;
    }

    // Diagnostics: Regularized singular value spectrum
    let svs_l2: Col<f64> = Col::from_fn(j, |i| {
        f64::max(evals[i] + l2_penalty, 0.0).sqrt()
    });
    let sv_min_l2: f64 = svs_l2.min().ok_or(Error::from("Regularized singular values `Col<f64>` is empty"))?;
    let sv_max_l2: f64 = svs_l2.max().ok_or(Error::from("Regularized singular values `Col<f64>` is empty"))?;
    let kappa_l2: f64 = sv_max_l2 / sv_min_l2;

    // 4.4 Step size (1 / max regularized eigenvalue)
    let step: f64 = (sv_max.powi(2) + l2_penalty).recip();


    // 5. Projected Gradient Descent
    // G and c are scaled and L2 penalty applied to G
    // ∇f(w) = Gw - c
    // 5.1 Initialize parameters, w at uniform weights
    let mut w: Col<f64> = Col::full(j, (j as f64).recip());
    let mut converged: bool = false;
    let mut n_iter: i32 = 0;

    // Gradient descent loop
    for _iter in 0..max_iter {
        n_iter += 1;
        // 5.2 Compute gradient
        let grad: Col<f64> = (&G * &w) - &c;

        // 5.3 Step weights downhill
        let mut w_new = &w - (step * grad);

        // 5.4 Projection onto simplex
        w_new = project_simplex(w_new.as_col_ref());

        // 5.5 Check convergence
        let delta: f64 = (&w_new - &w).norm_l2();

        w = w_new;
        
        if delta < tol { 
            converged = true; 
            break; 
        }
    }

    // 6. Final objective computations and R exports

    // Scaling treated sum of squares
    let a_sq_scaled: f64 = a_sq_sum / (T_0 as f64 * n_quantiles as f64);
    // Compute w^T * G * w
    let wGw: f64 = w.transpose() * &G * &w;
    // Compute C^t * w
    let cw: f64 = c.transpose() * &w;
    // Penalized objective the optimizer saw
    let obj_penalized: f64 = a_sq_scaled - cw + (0.5 * wGw);
    // Unpenalized objective (mean squared 2-Wasserstein Distance)
    let obj_unpenalized: f64 = obj_penalized - (lambda * w.squared_norm_l2());

    Ok(list!(
        weights = w.iter().collect::<Doubles>(),
        loss = obj_unpenalized,
        loss_penalized = obj_penalized,
        converged = converged,
        n_iterations = n_iter,
        probs = probs.iter().collect::<Doubles>(),
        effective_rank = effective_rank,
        right_singular_vectors = evecs.as_rmatrix(),
        svs = svs.iter().collect::<Doubles>(),
        kappa = kappa,
        svs_l2 = svs_l2.iter().collect::<Doubles>(),
        kappa_l2 = kappa_l2,
    ))
}

// Treated-Donor bucket pair helper struct
struct BucketData<'a> {
    treated: &'a [f64],
    donors: Vec<&'a [f64]>,
}

// Simplex projection helper function
// R-wrapper or calling function must ensure no NAs/NaNs in the input data
pub(crate) fn project_simplex(v: ColRef<f64>) -> Col<f64> {
    let j: usize = v.nrows();

    // 1. Sort weights vector in descending order, u = sorted weights vector
    let mut u: Vec<f64> = v.iter().copied().collect::<Vec<f64>>();
    u.sort_unstable_by(|a: &f64, b: &f64| b.total_cmp(a));

    // 2. Initializing loop state
    let mut sum: f64 = 0.0;
    let mut tau: f64 = 0.0;

    // 3. Finding threshold tau
    for (k, &u_k) in u.iter().enumerate() {
        sum += u_k;
        let tau_k = (sum - 1.0) / (k + 1) as f64;

        if u_k - tau_k <= 0.0 {
            break
        };

        tau = tau_k;
    };

    // 4. Applying tau to unsorted weights vector
    Col::from_fn(j, |i| {
        f64::max(v[i] - tau, 0.0)
    })
}

// Old version of the Robj slice container for loop using unsafe blocks
    // // Iterating over treated and donors lists to populate slices container
    // for (tr_b, donors_b) in treated.values().zip(donors.values()) {

    //     // Extracting &[f64] slice from treated bucket Robj
    //     let tr_temp: &[f64] = tr_b
    //         .as_real_slice()
    //         .ok_or(Error::from("Treated data must be numeric doubles"))?;

    //     // Previous implementation used unsafe blocks
    //     // UNSAFE BLOCK: memory is locked by the `treated` function argument
    //     // Rebuilding the slice to detach lifetime from temporary `tr_b` wrapper
    //     let tr_slice: &[f64] = unsafe {
    //         std::slice::from_raw_parts(tr_temp.as_ptr(), tr_temp.len())
    //     };

    //     // Converting the donor bucket's Robj to a list
    //     let donors_list: List = donors_b
    //         .as_list()
    //         .ok_or(Error::from("Each donor bucket must be a nested list"))?;

    //     let mut donors_slices: Vec<&[f64]> = Vec::with_capacity(donors_list.len());

    //     for d_obj in donors_list.values() {
    //         let d_temp: &[f64] = d_obj
    //             .as_real_slice()
    //             .ok_or(Error::from("Donor data must be numeric doubles"))?;

    //         // Previous implementation used unsafe blocks
    //         // UNSAFE BLOCK: Rebuilding donor slice detached from the Robj
    //         let donor: &[f64] = unsafe {
    //             std::slice::from_raw_parts(d_temp.as_ptr(), d_temp.len())
    //         };
    //         donors_slices.push(donor);
    //     }

    //     buckets.push(BucketData {
    //         treated: tr_slice,
    //         donors: donors_slices,
    //     })
    // }


#[cfg(test)]
mod tests {
    // Import everything from the parent module
    use super::*;
    use faer::{col, Col};

    /// Helper function for safe floating-point comparison of faer columns
    fn assert_col_eq(a: &Col<f64>, b: &Col<f64>, tol: f64) {
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

    // ==========================================
    // Tests for project_simplex
    // ==========================================

    #[test]
    fn test_project_simplex_already_valid() {
        // A vector already on the simplex (sums to 1, all >= 0) should remain unchanged
        let v = col![0.2, 0.3, 0.5];
        let projected = project_simplex(v.as_col_ref());
        
        assert_col_eq(&projected, &v, 1e-9);
    }

    #[test]
    fn test_project_simplex_clamping() {
        // A vector with negative values and sum > 1 should clamp negatives to 0 
        // and adjust the rest to sum to 1. 
        // [1.2, -0.2] projects exactly to [1.0, 0.0]
        let v = col![1.2, -0.2];
        let expected = col![1.0, 0.0];
        let projected = project_simplex(v.as_col_ref());
        
        assert_col_eq(&projected, &expected, 1e-9);
    }

    #[test]
    fn test_project_simplex_uniform_shift() {
        // A vector with valid proportions but sum > 1 should uniformly shift down
        // [0.8, 0.4] (sum = 1.2) should shift down by exactly 0.1 each
        let v = col![0.8, 0.4];
        let expected = col![0.7, 0.3];
        let projected = project_simplex(v.as_col_ref());
        
        assert_col_eq(&projected, &expected, 1e-9);
    }
}

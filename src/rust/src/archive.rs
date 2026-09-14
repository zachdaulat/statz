#![allow(unused)]

use extendr_api::prelude::*;
use faer::{ColRef, MatRef};
use std::f64;

// Registering the archived modules
extendr_module! {
    mod archive;
    use arch_descriptive;
    use arch_distributions;
    use arch_causal;
}

// RMatrix to faer Mat conversion function
// Zero-copy view of RMatrix
pub(crate) fn rmatrix_to_matref<'a>(x: &'a RMatrix<f64>) -> MatRef<'a, f64> {
    let nrows: usize = x.nrows();
    let ncols: usize = x.ncols();
    let data: &[f64] = x.data(); // This correctly returns &[f64]

    // Explicitly use MatRef, not Mat
    MatRef::from_column_major_slice(data, nrows, ncols)
}

// Doubles to faer Col conversion function
// Zero-copy view of Doubles
pub(crate) fn doubles_to_colref<'a>(y: &'a Doubles) -> ColRef<'a, f64> {
    // Drop down to the underlying Robj to extract the raw f64 slice safely
    let data: &[f64] = y
        .as_robj()
        .as_real_slice()
        .expect("Vector must be standard real numbers");

    // Explicitly use ColRef, not Col
    ColRef::from_slice(data)
}

pub mod arch_descriptive {
    use extendr_api::prelude::*;

    // Registering archived functions
    extendr_module! {
        mod arch_descriptive;
        fn arch_sum;
        fn arch_mean;
        fn arch_median;
        fn arch_var;
        fn arch_sd;
        fn arch_cov;
        fn arch_cor;
        fn arch_cor_onepass;
    }

    /// Compute the sum of a numeric vector. Naïve summation with Rust's `.sum()` method
    /// @param x A numeric vector.
    /// @return The sum as a double.
    /// @keywords internal
    #[extendr]
    pub fn arch_sum(x: &[f64]) -> f64 {
        x.iter().sum()
    }

    /// Compute the arithmetic mean of a numeric vector.
    /// @param x A numeric vector
    /// @return The mean as a double.
    /// @keywords internal
    #[extendr]
    pub fn arch_mean(x: &[f64]) -> f64 {
        if x.is_empty() {
            return f64::NAN;
        }
        let n = x.len() as f64;
        arch_sum(x) / n
    }

    /// Compute the median of a numeric vector.
    /// @param x A numeric vector.
    /// @return The median as a double
    /// @keywords internal
    #[extendr]
    pub fn arch_median(x: &[f64]) -> f64 {
        if x.is_empty() {
            return f64::NAN;
        }
        let mut sorted = x.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    /// Compute the sample variance of a numeric vector (Bessel-corrected, n-1).
    /// @param x A numeric vector.
    /// @return The sample variance as a double.
    /// @keywords internal
    #[extendr]
    pub fn arch_var(x: &[f64]) -> f64 {
        if x.len() < 2 {
            return f64::NAN;
        }
        let mean: f64 = arch_mean(x);
        let n = x.len() as f64;
        x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / (n - 1.0)
    }

    /// Compute the sample standard deviation of a numeric vector.
    /// @param x A numeric vector.
    /// @return The sample standard deviation as a double.
    /// @keywords internal
    #[extendr]
    pub fn arch_sd(x: &[f64]) -> f64 {
        arch_var(x).sqrt()
    }

    /// Compute the sample covariance of two numeric vectors.
    /// @param x A numeric vector.
    /// @param y A numeric vector of the same length.
    /// @return The sample covariance as a double.
    /// @keywords internal
    #[extendr]
    pub fn arch_cov(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return f64::NAN;
        }
        let mean_x: f64 = arch_mean(x);
        let mean_y: f64 = arch_mean(y);
        let n: f64 = x.len() as f64;
        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / (n - 1.0)
    }

    /// Compute the Pearson correlation coefficient of two numeric vectors.
    /// @param x A numeric vector.
    /// @param y A numeric vector.
    /// @return The sample correlation as a double
    /// @keywords internal
    #[extendr]
    pub fn arch_cor(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return f64::NAN;
        }
        arch_cov(x, y) / (arch_sd(x) * arch_sd(y))
    }

    /// Compute Pearson correlation coefficient, optimized single-pass
    /// @param x A numeric vector.
    /// @param y A numeric vector.
    /// @return The sample correlation as a double
    /// @keywords internal
    #[extendr]
    pub fn arch_cor_onepass(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return f64::NAN;
        }

        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        let mut sum_xy: f64 = 0.0;
        let mut sum_x2: f64 = 0.0;
        let mut sum_y2: f64 = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            sum_x += xi;
            sum_y += yi;
            sum_xy += xi * yi;
            sum_x2 += xi * xi;
            sum_y2 += yi * yi;
        }

        let n: f64 = x.len() as f64;
        let numer: f64 = sum_xy - (sum_x * sum_y / n);
        let denom: f64 = ((sum_x2 - (sum_x.powi(2) / n)) * (sum_y2 - (sum_y.powi(2) / n))).sqrt();

        numer / denom
    }
}

pub mod arch_distributions {
    use crate::consts::*;
    use extendr_api::prelude::*;
    use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

    // Registering archived functions
    extendr_module! {
        mod arch_distributions;
        fn arch_pnorm_as;
        fn arch_ppois_di;
        fn arch_lgamma_godfrey;
    }

    /// Compute the standard normal cumulative distribution function (CDF)
    ///
    /// @description
    /// Uses the Abramowitz and Stegun (1972, 10th ed.) equation 7.1.26
    /// error function approximation. Deprecated in favour of the
    /// libm::erfc() implementation but kept as an initial pedagogical version.
    /// Maximum absolute error: |ε| < 1.5 × 10^7
    ///
    /// @param z A z-score (standardised value)
    /// @param lower_tail Logical; if TRUE (default), probabilities are P(X <= x).
    /// @param log_p Logical; if TRUE, probabilities p are given as ln(p).
    /// @return A numeric vector of cumulative probabilities.
    /// @keywords internal
    #[extendr]
    #[allow(non_upper_case_globals)]
    pub fn arch_pnorm_as(z: f64, lower_tail: bool, log_p: bool) -> f64 {
        if z.is_nan() {
            return f64::NAN;
        }
        if z == f64::INFINITY {
            return if log_p { 0.0 } else { 1.0 };
        }
        if z == f64::NEG_INFINITY {
            return if log_p { f64::NEG_INFINITY } else { 0.0 };
        }

        let u = (z / SQRT_2).abs();

        // Constants defined by A&S, p.299
        const p: f64 = 0.32759_11;
        const a_1: f64 = 0.25482_9592;
        const a_2: f64 = -0.28449_6736;
        const a_3: f64 = 1.42141_3741;
        const a_4: f64 = -1.45315_2027;
        const a_5: f64 = 1.06140_5429;

        let t = 1.0 / (1.0 + (p * u));

        // Horner's method for erfc(|u|)
        let erfc: f64 = a_5
            .mul_add(t, a_4)
            .mul_add(t, a_3)
            .mul_add(t, a_2)
            .mul_add(t, a_1)
            * t
            * (-u * u).exp();

        let erf: f64 = (1.0 - erfc).copysign(z);

        // Calulated in regular space
        let mut cdf: f64 = (1.0 + erf) / 2.0;

        // Evaluating tail boolean
        cdf = if lower_tail { cdf } else { 1.0 - cdf };

        // Evaluating log or regular space boolean
        if log_p {
            cdf.ln()
        } else {
            cdf
        }
    }

    /// Compute the Poisson probability mass function
    ///
    /// @description
    /// Compute the Poisson cumulative distribution function P(X ≤ x)
    /// using log-space PMF evaluation for each term.
    ///
    /// @param x A numeric (double) vector of non-negative whole numbers.
    /// @param lambda The rate parameter (λ > 0).
    /// @param log Logical; if TRUE, probabilities p are given as ln(p).
    /// @return A numeric vector of probability masses.
    /// @keywords internal
    #[extendr]
    // Named di for reference to the dpois function and this iterator approach
    pub fn arch_ppois_di(x: i32, lambda: f64, log_p: bool) -> f64 {
        // Calculated in regular space
        let cdf: f64 = (0..=x)
            .map(|i| crate::distributions::dpois(i as f64, lambda, false))
            .sum::<f64>();

        // Evaluating boolean for returning in log or regular space
        if log_p {
            cdf.ln()
        } else {
            cdf
        }
    }

    // ============================================================
    // Documentation for lgamma_godfrey (pedagogical implementation)
    // ============================================================

    /// Compute ln Γ(z) using Godfrey's Lanczos coefficient set (g=7, N=9).
    ///
    /// @description
    /// This is a pedagogical implementation of the traditional Lanczos approximation
    /// using Paul Godfrey's well-known f64 coefficient set. Unlike the Boost
    /// adaptation in `lgamma()`, this uses the standard formulation:
    ///
    ///   ln Γ(z) = ½ ln(2π) + (z - ½) ln(z + g - ½) - (z + g - ½) + ln S(z)
    ///
    /// where S(z) = c₀ + Σ(k=1..8) cₖ/(z-1+k) is the Lanczos sum with
    /// alternating-sign coefficients, and the formula is evaluated after
    /// shifting z → z-1 to convert from Γ(z+1) to Γ(z).
    ///
    /// This implementation is less precise than `lgamma()` due to potential
    /// cancellation in the alternating-sign sum, but is included as a learning
    /// exercise.
    ///
    /// @references
    /// - Godfrey, P. "Lanczos Implementation of the Gamma Function."
    ///   <http://my.fit.edu/~gabdo/gamma.txt>
    /// - <https://www.mrob.com/pub/ries/lanczos-gamma.html>
    ///
    /// @param z A numeric (double) vector of positive values.
    /// @return A numeric vector containing the natural logarithm of the gamma function.
    /// @keywords internal
    #[allow(unused)]
    #[allow(clippy::excessive_precision)]
    #[allow(clippy::needless_range_loop)]
    #[extendr]
    pub fn arch_lgamma_godfrey(mut z: f64) -> f64 {
        // Checking if z is a small whole number to return precomputed values
        if z > 0.0 && z <= 16.0 && z.fract() == 0.0 {
            return LN_FACTORIALS[(z - 1.0) as usize];
        }

        // Godfrey's coefficient set, g = 7, N = 9
        const G: f64 = 7.0;
        const COEFFS: [f64; 9] = [
            0.99999999999980993227684700473478,
            676.520368121885098567009190444019,
            -1259.13921672240287047156078755283,
            771.3234287776530788486528258894,
            -176.61502916214059906584551354,
            12.507343278686904814458936853,
            -0.13857109526572011689554707,
            9.984369578019570859563e-6,
            1.50563273514931155834e-7,
        ];

        // Applying reflection formula when z < 0.5
        if z < 0.5 {
            let lgam: f64 = LN_PI - (PI * z).sin().abs().ln() - arch_lgamma_godfrey(1.0 - z);
            return lgam;
        }

        z -= 1.0;

        let mut s: f64 = COEFFS[0];

        for i in 1..9 {
            s += COEFFS[i] / (z + i as f64);
        }

        let t = z + G + 0.5;

        let lgam: f64 = LN_SQRT_2PI + ((z + 0.5) * t.ln()) - t + s.ln();

        lgam
    }
}

pub mod arch_causal {
    use crate::{causal::project_simplex, ext::RMatrixExt};
    use extendr_api::{prelude::*, Error};
    use faer::{
        col::AsColRef, linalg::solvers::SelfAdjointEigen, mat::AsMatRef, Col, ColMut, ColRef, Mat,
        MatRef, Side,
    };
    use std::f64;

    extendr_module! {
        mod arch_causal;
        fn arch_dsc;
    }

    // Future refinement when doing bootstrapping, modularize into separate functions
    // to enable crate-level testing with wrapper doing R object view conversion
    // Distributional synthetic controls implementation
    #[extendr]
    #[allow(non_snake_case)]
    pub fn arch_dsc(
        treated: List,    // Each element: Doubles vectors of treated unit obs by bucket
        donors: List,     // Each element: List of Doubles, one per donor by bucket
        n_quantiles: i32, // Q, number of quantiles to sample from each bucket
        penalty: f64,     // L2 penalty
        max_iter: i32,    // Maximum number of iterations for gradient descent loop
        tol: f64,         // Threshold for change in weights vector norm
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
            .map(|d| d.as_list().map(|l| l.values().collect()))
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
            .ok_or(Error::from(
                "Mat-reduce failed: Bucket data iterator was empty",
            ))?;

        // 4. Preparing Gram matrix and cross vector and eigendecomposition diagnostics
        // 4.1 Applying normalization constant
        let scalar: f64 = 2.0 / (T_0 as f64 * n_quantiles as f64);
        let mut G: Mat<f64> = &G_sum * scalar;
        let c: Col<f64> = &c_sum * scalar;
        let j: usize = c.nrows();

        // 4.2 Eigendecomposition of the normalized Gram matrix
        let eigen: SelfAdjointEigen<f64> = SelfAdjointEigen::new(G.as_mat_ref(), Side::Upper)
            .map_err(|_| Error::from("Error during Gram matrix eigendecomposition"))?;

        // Extracting decomposition outputs
        let evals: ColRef<f64> = eigen.S().column_vector();
        let evecs: MatRef<f64> = eigen.U();

        // Diagnostics: Singular values and condition number of D
        let svs: Col<f64> = Col::from_fn(j, |i| f64::max(evals[i], 0.0).sqrt());
        let sv_min: f64 = svs
            .min()
            .ok_or(Error::from("Singular values `Col<f64>` is empty"))?;
        let sv_max: f64 = svs
            .max()
            .ok_or(Error::from("Singular values `Col<f64>` is empty"))?;
        let kappa: f64 = sv_max / sv_min;

        // Diagnostics: Effective rank
        let rank_tol: f64 = sv_max * (j as f64) * f64::EPSILON;
        let effective_rank: i32 = svs.iter().filter(|&sigma| sigma > &rank_tol).count() as i32;

        // 4.3 Regularization
        let l2_penalty: f64 = 2.0 * penalty;

        // Zero-allocation diagonal mutation to apply L2 penalty to Gram matrix
        let mut G_diag: ColMut<f64> = G.diagonal_mut().column_vector_mut();
        for i in 0..j {
            G_diag[i] += l2_penalty;
        }

        // Diagnostics: Regularized singular value spectrum
        let svs_l2: Col<f64> = Col::from_fn(j, |i| f64::max(evals[i] + l2_penalty, 0.0).sqrt());
        let sv_min_l2: f64 = svs_l2.min().ok_or(Error::from(
            "Regularized singular values `Col<f64>` is empty",
        ))?;
        let sv_max_l2: f64 = svs_l2.max().ok_or(Error::from(
            "Regularized singular values `Col<f64>` is empty",
        ))?;
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
        let obj_unpenalized: f64 = obj_penalized - (penalty * w.squared_norm_l2());

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
}

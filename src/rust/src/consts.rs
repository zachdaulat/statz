#![allow(unused)]
#![allow(clippy::approx_constant)]
#![allow(clippy::excessive_precision)]

// These constants and their values were copied directly from the `statrs::consts`
// module in the `statrs` crate
// The `statrs` crate is governed by the MIT License
pub(crate) const EULER_MASCHERONI: f64 =
    0.5772156649015328606065120900824024310421593359399235988057672348849;
pub(crate) const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452223455184457816472122518527279025978;
pub(crate) const LN_PI: f64 = 1.1447298858494001741434273513530587116472948129153;
pub(crate) const LN_SQRT_2PI: f64 = 0.91893853320467274178032973640561763986139747363778;
pub(crate) const LN_SQRT_2PIE: f64 = 1.4189385332046727417803297364056176398613974736378;
pub(crate) const SQRT_2PI: f64 = 2.5066282746310005024157652848110452530069867406099;
pub(crate) const TWO_SQRT_E_OVER_PI: f64 = 1.8603827342052657173362492472666631120594218414085755;

// ============================================================
// Documentation for LN_FACTORIALS lookup table
// ============================================================

/// Precomputed values of ln(n!) for n = 0, 1, ..., 15.
///
/// Used by `lgamma()` and `lgamma_godfrey()` to short-circuit evaluation
/// when the input is a small positive integer. Since Γ(n) = (n-1)! for
/// positive integers, ln Γ(n) = ln((n-1)!), so `LN_FACTORIALS[n-1]` gives
/// the correct result for z = n.
///
/// Index mapping: LN_FACTORIALS[k] = ln(k!) for k = 0..15.
///   - LN_FACTORIALS[0] = ln(0!) = 0.0
///   - LN_FACTORIALS[1] = ln(1!) = 0.0
///   - LN_FACTORIALS[2] = ln(2!) ≈ 0.6931
///   - ...
///   - LN_FACTORIALS[15] = ln(15!) ≈ 27.8993
pub(crate) const LN_FACTORIALS: [f64; 16] = [
    0.0,
    0.0,
    0.6931471805599453094172321,
    1.791759469228055000812477,
    3.178053830347945619646942,
    4.787491742782045994247701,
    6.579251212010100995060178,
    8.525161361065414300165531,
    10.60460290274525022841723,
    12.80182748008146961120772,
    15.10441257307551529522571,
    17.50230784587388583928765,
    19.98721449566188614951736,
    22.55216385312342288557085,
    25.19122118273868150009343,
    27.89927138384089156608944,
];

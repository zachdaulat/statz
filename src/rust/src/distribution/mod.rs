//! High-precision probability distributions (PDF/PMF, CDF) and special mathematical functions.
//!
//! Exposes both pure Rust scalar kernels and vectorized `extendr` R wrappers.

#![allow(clippy::excessive_precision)]

use extendr_api::prelude::*;

pub mod gamma;
pub mod invgauss;
pub mod normal;
pub mod poisson;
pub mod special;
pub mod tweedie;

// // Re-export public Rust functions and #[extendr] wrappers into `distribution::*`
// pub use gamma::*;
// pub use invgauss::*;
// pub use normal::*;
// pub use poisson::*;
// pub use special::*;
// pub use tweedie::*;

extendr_module! {
    mod distribution;
    use normal;
    use poisson;
    use gamma;
    use invgauss;
    use tweedie;
    use special;
}

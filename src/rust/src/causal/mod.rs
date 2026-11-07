//! Causal inference data structures, estimators, and inference pipelines.
//!
//! Organizes contiguous panel data abstractions (`panel`) and Distributional
//! Synthetic Controls (`dsc`) routines.

pub mod dsc;
pub mod panel;

pub use dsc::*;
pub use panel::*;

extendr_api::extendr_module! {
    mod causal;
}

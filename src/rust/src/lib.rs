use extendr_api::prelude::*;

// Module declarations
mod archive;
mod causal;
mod consts;
mod descriptive;
mod distributions;
mod linear_models;
// Future modules
// mod spatial;

// Macro to export modules to R
extendr_module! {
    mod statz;
    use archive;
    use causal;
    use descriptive;
    use distributions;
    use linear_models;
}

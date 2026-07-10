use extendr_api::prelude::*;

// Module declarations
mod descriptive;
mod distributions;
mod linear_models;
mod causal;
// Future modules
// mod spatial;
mod archive;

// Macro to export modules to R
extendr_module! {
    mod statz;
    use descriptive;
    use distributions;
    use linear_models;
    use causal;
}

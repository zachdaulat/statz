use extendr_api::prelude::*;

// Module declarations
mod descriptive;
mod distributions;
mod linear_models;
// Future modules
// mod spatial;

// Macro to export functions for R
// I need to register functions here as I add to each module
extendr_module! {
    mod statz;
    use descriptive;
    use distributions;
    use linear_models;
}

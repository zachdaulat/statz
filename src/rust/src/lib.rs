use extendr_api::prelude::*;

// Module declarations
mod descriptive;
// mod distributions;
// Future modules
// mod regression;
// mod spatial;
// mod linalg;

// Macro to export functions for R
// I need to register functions here as I add to each module
extendr_module! {
    mod statz;
    use descriptive;
}

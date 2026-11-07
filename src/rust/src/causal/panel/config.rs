//! Configuration and state-tracking enums for distributional panel transformations.
//!
//! Includes probability grid resolution strategies (`ProbsMode`), location
//! statistics (`LocStat`), location adjustments (`LocAdj`), and bootstrap modes (`ResampleMode`).

/// Defines the location adjustment applied to the empirical distributions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocAdj {
    /// Raw empirical distributions with no location adjustment.
    None,
    /// Translation adjustment based on a central location statistic.
    Shift,
    /// Scaling adjustment based on a central location statistic.
    Scale,
}

/// Defines the central location statistic used when applying `LocAdj`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocStat {
    None,
    Mean,
    Median,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResampleMode {
    Observations,
    Periods,
    TwoStage,
}

/// Defines the resolution strategy for the probability grid used in quantile evaluation.
///
/// To ensure matrix conformability in downstream `faer` linear algebra, every bootstrap
/// resampling iteration must project onto the same $Q$-dimensional quantile space. This
/// enum dictates how that global $Q$ is determined before resampling begins.
#[derive(Debug, Clone, PartialEq)]
pub enum ProbsMode {
    /// A manually provided, pre-computed vector fo probability levels.
    Grid(Vec<f64>),
    /// A fixed number of quantiles, yielding a uniform probability grid.
    Fixed(usize),
    /// Generates a grid matching the observation count of the smallest cell in the panel.
    MinCell,
    /// Generates a grid matching the median observation count across all cells.
    MedianCell,
    /// Generates a grid matching the mean observation count across all cells.
    MeanCell,
    /// Generates a grid matching the observation count of the largest cell in the panel.
    MaxCell,
}

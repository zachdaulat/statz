//! Probability grid generation utilities for uniform empirical quantile evaluation.

/// Generates a uniform probability grid of length `n`.
///
/// Computes the grid using the formula $(i + 0.5) / n$.
/// Allocates and returns a new vector containing the probability levels.
pub(crate) fn probs_grid(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i: usize| (i as f64 + 0.5) / (n as f64))
        .collect::<Vec<f64>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probs_grid_edge_cases() {
        // A length of 0 resolves to an empty vector, preventing downstream
        // quantile interpolation from panicking, but resulting in empty tensors.
        assert_eq!(probs_grid(0), Vec::<f64>::new());

        // A length of 1 defaults to the exact 50th percentile (median)
        assert_eq!(probs_grid(1), vec![0.5]);

        // A length of 2 evenly divides the probability space
        assert_eq!(probs_grid(2), vec![0.25, 0.75]);
    }
}

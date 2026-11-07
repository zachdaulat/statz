//! Parameter specifications (`DscSpec`, `BootSpec`, `PlaceboSpec`) and location
//! adjustment parameters (`LocParams`).

use crate::causal::panel::{LocAdj, LocStat, ProbsMode, QuantilePanel, ResampleMode};
use faer::{Col, ColRef};

pub(crate) struct DscSpec {
    pub probs: ProbsMode,
    pub loc_stat: LocStat,
    pub loc_adj: LocAdj,
    pub penalty: f64,
    pub max_iter: usize,
    pub tol: f64,
}

pub(crate) struct BootSpec {
    pub n_boot: usize,
    pub mode: ResampleMode,
    pub seed: u64,
}

pub(crate) struct PlaceboSpec {
    pub pre_fit_filter: f64,
}

pub enum LocParams {
    None,
    Shift { gamma: f64 },
    Scale { rho: Col<f64> }, // length j, aligned with `donors`
}

impl LocParams {
    /// Pre-allocates the appropriate variant for the chosen location adjustment.
    /// Only allocates heap memory when `LocAdj::Scale` is selected.
    pub fn new(adj: LocAdj, n_donors: usize) -> Self {
        match adj {
            LocAdj::None => Self::None,
            LocAdj::Shift => Self::Shift { gamma: f64::NAN },
            LocAdj::Scale => Self::Scale {
                rho: Col::full(n_donors, f64::NAN),
            },
        }
    }

    /// Computes and writes the location parameter(s) in-place without reallocating.
    /// Locations are computed from the empirical panels
    ///
    /// * For `Shift`: $\gamma = \mu_{\text{target}} - \sum_{i=1}^j w_i \mu_{\text{donor}_i}$
    /// * For `Scale`: $\rho_i = \mu_{\text{target}} / \mu_{\text{donor}_i}$
    pub fn compute_from(
        &mut self,
        qp: &QuantilePanel,
        target: usize,
        donors: &[usize],
        weights: ColRef<f64>,
    ) {
        let mu_target: f64 = qp.locations[target];

        match self {
            Self::None => {}
            Self::Shift { gamma } => {
                let mut weighted_donor_loc: f64 = 0.0;
                for (i, &d) in donors.iter().enumerate() {
                    weighted_donor_loc += weights[i] * qp.locations[d];
                }
                *gamma = mu_target - weighted_donor_loc;
            }
            Self::Scale { rho } => {
                for (i, &d) in donors.iter().enumerate() {
                    rho[i] = mu_target / qp.locations[d];
                }
            }
        }
    }
}

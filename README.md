# statz

An R package with a Rust computational backend, built as a pedagogical implementation of core statistical functions. Beginning with descriptive statistics, through to probability distributions, linear models, and generalised linear models. The Rust computational core is bridged to R via [`extendr`](https://extendr.rs/).

This is an in-development project intended to be a structured and focused but extendable statistics and data science learning plan, progressing toward causal inference techniques. Each function is implemented to facilitate my learning and understanding.

## Table of Contents

- [Architecture](#architecture)
- [Design Choices](#design-choices)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Install from GitHub](#install-from-github)
- [Implemented Functions](#implemented-functions)
  - [Part 1 — Descriptive Statistics](#part-1)
  - [Part 2 — Probability Distributions](#part-2)
    - [Normal Distribution](#normal-distribution)
    - [Poisson Distribution](#poisson-distribution)
    - [Gamma Distribution](#gamma-distribution)
    - [Tweedie Distribution *(in progress)*](#tweedie-distribution-in-progress)
- [Planned Work](#planned-work)
  - [Part 3 — Linear Models](#part-3)
  - [Part 4 — Generalised Linear Models](#part-4)
  - [Parts 5+ — Causal Inference and Spatial Statistics](#parts-5)
- [Testing](#testing)
  - [Notable Rust Tests](#notable-rust-tests)
- [References](#references)
- [Licence](#licence)

## Architecture

The package follows a two-layer architecture. Following the same strategy as base R and R packages that use C/C++ or Fortran backends, `statz` delegates numerical computation to a compiled language. R provides the user-facing wrapper functions with input validation, documentation, and interfaces similar to base R and the native `stats` package (`distributions.R`), with numerical computation handled by compiled Rust code (`distributions.rs`, `descriptive.rs`). The `extendr` / `rextendr` toolchain bridges the two layers via R's C foreign function interface, compiling the Rust source into a shared library that R loads at runtime.

```
statz/
├── src/rust/src/
│   ├── lib.rs                # Module declarations and extendr registration
│   ├── descriptive.rs        # Descriptive statistics (Rust)
│   └── distributions.rs      # Probability distributions (Rust)
├── R/
│   └── distributions.R       # R wrappers with validation and documentation
├── tests/testthat/
│   └── test-distributions.R  # R-level tests 
└── ...
```

## Design Choices

Currently the most significant interface departure from the R-native implementations is that the distribution functions are not yet vectorised. The descriptive statistics functions (`z_mean()`, `z_cor()`, etc.) are, but I decided against vectorising the distributions for now to focus on learning their respective PDFs/PMF and CDFs and the numerical methods. I may vectorise them eventually as an exercise in learning Rust's ownership system and borrow checker, zero-allocation iterator techniques, and memory safety; potentially useful prep for the OLS and GLM implementations planned in parts 3 and 4.

Functions annotated with the `#[extendr]` attribute are exported to R via the `extendr_module!` macro. Most internal helper functions like `lower_gamma_series()` and `upper_gamma_cf()` use `pub(crate)` to remain accessible within the Rust crate but are hidden from R. Most exported Rust functions like `z_pgamma_rs()` have an associated R wrapper like `z_pgamma()` that adds input validation and parameterisation options. The exported Rust functions are usually not intended to be called directly by a user.

The initial implementation prioritises the core analytical functions (`d*` and `p*`) over quantile (`q*`) and random generation (`r*`) functions to maintain a focused pedagogical scope.

## Installation

Installing this package from source requires the Rust toolchain and standard R build tools.

### Prerequisites

- **R** (≥ 4.0)
- **Rust toolchain**: Install via [rustup.rs](https://rust-lang.org/tools/install/). Verify with `rustc --version` and `cargo --version`.
- **C/C++ Build Tools**: 
  - **Windows**: Install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version.
  - **macOS**: Run `xcode-select --install` in the terminal.
  - **Linux**: Install standard R development tools (e.g., `sudo apt install r-base-dev` on Ubuntu).

### Install from GitHub

You can compile and install the package using `pak` (recommended) or `devtools`:

```r
# Using pak
install.packages("pak")
pak::pkg_install("zachdaulat/statz")

# Or using devtools
install.packages("devtools")
devtools::install_github("zachdaulat/statz")
```

Alternatively, clone and build locally:

```bash
$ git clone https://github.com/zachdaulat/statz.git
$ cd statz
```

```r
# From R, with working directory set to the package root
install.packages(c("rextendr", "devtools"))
rextendr::document()
devtools::install()
```

## Implemented Functions

### <a id="part-1"></a>Part 1 — Descriptive Statistics

This first part of the project was intentionally brief. It served primarily as an introduction to Rust, the Rust-R bridge via `extendr`, and to set up the testing and build workflow. 

Basic sample statistics implemented in Rust, using Bessel's correction (*n* − 1 denominator) for the sample variance and standard deviation to match R's `var()` and `sd()`.

| Function | Description |
|---|---|
| `z_sum(x)` | Sum of a numeric vector |
| `z_mean(x)` | Arithmetic mean |
| `z_median(x)` | Median (sorts internally) |
| `z_var(x)` | Sample variance (Bessel-corrected) |
| `z_sd(x)` | Sample standard deviation |
| `z_cov(x, y)` | Sample covariance |
| `z_cor(x, y)` | Pearson correlation coefficient |
| `z_cor_onepass(x, y)` | Pearson correlation, single-pass Welford-style accumulation |

### <a id="part-2"></a>Part 2 — Probability Distributions

Following R’s standard naming convention, the package implements density/mass and cumulative probability distribution functions using the `d*` (density/mass) and `p*` (distribution) prefixes. R wrappers provide input validation and, where applicable, dual parameterisation (e.g., `rate` and `scale` for the gamma distribution). All density/mass computations use log-space arithmetic to avoid overflow.

The probability density/mass functions accept a `log` argument to return the log-density/log-probability directly, and the CDF functions accept a `log.p` argument to return cumulative log-probabilities. The CDF functions also accept a `lower.tail` argument; when `FALSE`, the upper tail probability is returned.

| Function | Description | Method |
|----------|-------------|-----------|
| `z_dnorm(x, mean, sd)` | Normal PDF | Log-space: $-\ln\sigma - \frac{1}{2}\ln(2\pi) - \frac{z^2}{2}$ |
| `z_pnorm(x, mean, sd)` | Normal CDF | Abramowitz & Stegun (1972) eq. 7.1.26 error function approximation; Horner's method |
| `z_dpois(x, lambda)` | Poisson PMF | Log-space: $x \ln\lambda - \lambda - \sum_{i=1}^{x}\ln i$ |
| `z_ppois(x, lambda)` | Poisson CDF | Recurrence relation: $P(X = k) = P(X = k-1) \cdot \lambda/k$ |
| `z_lgamma(z)` | $\ln\Gamma(z)$ | Boost.Math Lanczos adaptation (see below) |
| `z_dgamma(x, shape, rate, scale)` | Gamma PDF | Log-space using `z_lgamma()` for the $\ln\Gamma(\alpha)$ term |
| `z_pgamma(x, shape, rate, scale)` | Gamma CDF | Regularised incomplete gamma function (see below) |

<!-- TODO: Add Tweedie functions when implemented. -->

The current normal and Poisson CDF upper-tail implementations compute $1 - P$, and therefore lose precision when $P$ is near 1. A production update would compute the upper tail directly, but I simplified this for a pedagogical project. The gamma PDF and CDF provide dual parameterisation options through providing either `rate` or `scale`, but not both.

#### Normal Distribution

`z_pnorm()` computes the normal cumulative probability $\Phi(z)$ through its relationship to the error function:

$$
\Phi(z) = \frac{1}{2} \left[ 1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right) \right]
$$

Equation 7.1.26 from Abramowitz & Stegun (1972) approximates the error function from its complement $\text{erf}(z) = 1 - \text{erfc}(z)$:

$$
\text{erf}(x) \approx 1 - \left(a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5\right) e^{-x^2}
$$

Where,

- $t = 1/(1 + px)$, 
- $p, a_1,...,a_5$ are provided constants
- The maximum absolute error is $|\epsilon(x)| \le 1.5 \times 10^{-7}$

The normal CDF takes the z-score as its input, so the error function evaluates $x = |z/\sqrt{2}|$. The A&S 7.1.26 polynomial is efficiently evaluated using Horner's method with Rust's `.mul_add()` for fused multiply-add instructions. 

#### Poisson Distribution

The PMF `z_dpois()` uses log-space arithmetic to avoid factorial overflow. The log-mass $x\ln\lambda - \lambda - \sum \ln i$ is computed directly and exponentiated only at the final step.

The CDF `z_ppois()` exploits the Poisson recurrence relation $P(X = k) = P(X = k-1) \cdot \lambda / k$, accumulating the sum in a single pass without recomputing each PMF term independently. The package also retains the less efficient direct-iteration implementation (`z_ppois_di()`).

#### Gamma Distribution

##### `z_lgamma()` — Log-Gamma via Lanczos Approximation

Computes $\ln\Gamma(z)$ using a simplified adaptation of the [Boost.Math C++ library](https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/lanczos.html)'s Lanczos approximation, drawing from the `lanczos.hpp` and `gamma.hpp` source files. It uses the `lanczos13m53` parameters tuned for `f64` arithmetic ($N = 13$, $G \approx 6.0247$).

The standard computational form of the Lanczos approximation expresses the gamma function as $\Gamma(z+1)$ and uses a partial fraction expansion:

$$
\Gamma(z+1) = \sqrt{2\pi} \left( z + g + \frac{1}{2} \right)^{z+1/2} e^{-(z+g+1/2)} \sum_{k=0}^N \frac{c_k}{z+k}
$$

`z_lgamma()` evaluates a log-space adaptation of the Boost.Math formulation:

$$
\ln \Gamma(z) = t \ln\left(z + g - \frac{1}{2}\right) - t + \ln L_{g,e}(z)
$$

$$
L_{g,e}(z) = \frac{P_{12}(z)}{Q_{12}(z)} = \frac{\sum_{k=0}^{12} p_k z^k}{\sum_{k=0}^{12} q_k z^k}, \quad t = z - \frac{1}{2}
$$

This implementation adapts the `lanczos13m53::lanczos_sum_expG_scaled` coefficient set from the `lanczos.hpp` source, which absorbs both the $\sqrt{2\pi}$ normalisation constant and the $e^{G}$ scaling factor directly into the $p_k$ coefficients. Below is a list of additional details about the Boost authors' optimizations and my implementation choices.

**Optimizations & Implementation Decisions**

- **(Boost) Algorithmic stability:** The Lanczos sum ($L_{g,e}(z)$) is evaluated as a ratio of two degree-12 polynomials $P(z)$ and $Q(z)$ using Horner's method with `.mul_add()`. This required the Boost authors to compute twice as many coefficient values for a given N, but all of the coefficients can now be positive instead of alternating positive and negative, avoiding catastrophic cancellation risks.
- **(Boost) Precision simplification:** The Lanczos sum is not evaluated when the primary term (`lgam`) is so large that machine epsilon would truncate the addition anyway. The condition determining whether to evaluate it is `lgam * f64::EPSILON < 20.0`, the same condition found directly in the Boost.Math source `gamma.hpp`.
- **Whole number lookups:** I constructed a precomputed `LN_FACTORIALS` array for small positive integers (`z <= 16`) to return the stored value, bypassing the Lanczos approximation entirely for these common inputs.
- **Root handling:** The Boost implementation uses Taylor series expansions for inputs near 1 and 2. I omitted this as a deliberate simplification at the expense of reduced precision near the roots.
- **Domain handling:** For $z < 0.5$, the function automatically applies the standard reflection formula: $\ln \Gamma(z) = \ln \pi - \ln|\sin(\pi z)| - \ln \Gamma(1 - z)$.

The package also includes `z_lgamma_godfrey()`, a secondary implementation using Paul Godfrey's traditional formulation with alternating-sign coefficients (g = 7, N = 9). This is retained internally as a pedagogical comparison. The Godfrey formulation is simpler but less precise due to potential cancellation in the alternating sum.

##### `z_dgamma()` — Gamma PDF

Computes the probability density function $f(x | \alpha, \beta)$ via the shape-rate parameterisation in log-space with `z_lgamma()`:

$$\ln f = \alpha \ln(\beta) - \ln \Gamma(\alpha) + (\alpha-1)\ln(x) - \beta x$$

##### `z_pgamma()` — Gamma CDF via Incomplete Gamma Function

Computes the cumulative probability for $X \sim \text{Gamma}(\alpha, \beta)$ by approximating the regularised incomplete gamma functions. 

`z_pgamma_rs()` dispatches between the two helper functions `lower_gamma_series()` and `upper_gamma_cf()` based on the domain boundary $z = \alpha + 1$ to directly compute the smaller of the two tails and avoid cancellation risks for probabilities near 1.

- **Series Expansion** ($z < \alpha + 1$): Evaluates a Taylor series to compute the lower-tail probability $P(\alpha, z)$.
- **Continued Fraction** ($z \ge \alpha + 1$): Evaluates Legendre's continued fraction via the Modified Lentz Algorithm to compute the upper-tail probability $Q(\alpha, z)$.

The `lower_tail` boolean argument determines whether the directly computed probability or its complement is returned.

#### Tweedie Distribution *(in progress)*

<!-- Placeholder for the Tweedie unit. Key framing: the Tweedie family is the "capstone" of the distributions phase, tying together the exponential dispersion model framework and connecting the normal, Poisson, and gamma distributions as special cases. -->

The Tweedie family is the final milestone of the distributions part. It is parameterised by a variance power *p* and a dispersion parameter $\phi$, together dictating the variance–mean relationship $\text{Var}(Y) = \phi \mu^p$. For the special case $1 < p < 2$, the Tweedie distribution is a compound Poisson–gamma process: the sum of the gamma-distributed severities of a Poisson-distributed number of events. This produces a distribution with a point mass at zero and a continuous positive density—naturally modelling zero-inflated positive continuous data like transit delay durations.

<!-- Topics to cover when implementing:
     - Exponential dispersion models and the exponential family
     - How Normal (p = 0), Poisson (p = 1), Gamma (p = 2), etc are special cases
     - The compound Poisson–gamma interpretation for 1 < p < 2
     - Connection to the tweedie_variance_power parameter in LightGBM
     - The Dunn & Smyth series evaluation for the Tweedie density (if implementing dtweedie)
     - This is the conceptual bridge to the GLM phase: the Tweedie family unifies the fitting procedure via IRLS
-->

## Planned Work

The linear algebra implementations in parts 3 and 4 will use the [`faer`](https://crates.io/crates/faer) Rust crate. `faer` is a general-purpose, Rust-native linear algebra library optimised for large and dense matrix operations at the scale typical for statistical computing, rather than the low-dimensional operations common in graphics or game development.

### <a id="part-3"></a>Part 3 — Linear Models (`z_lm()`)

Three implementations of ordinary least squares regression, progressively improving in numerical stability as I learn linear algebra:

1. **Naïve normal equations:** $(X^TX)^{-1}X^Ty$ — expected to fail on the NIST Filip dataset
2. **QR decomposition:** To replicate R's own `lm()` implementation
3. **SVD:** Performance-stability tradeoff and pedagogical exercise

I also have a three-tier testing strategy in mind:

- Baseline correctness: palmerpenguins dataset + simple exact cases
- Numerical stability: NIST Filip dataset
- Scalability: Statistics Canada Census PUMF

This will be benchmarked against R's native `stats::lm()` and possibly Python OLS implementations like from `statsmodels` and `sklearn`.

### <a id="part-4"></a>Part 4 — Generalised Linear Models (`z_glm()`)

IRLS (Iteratively Reweighted Least Squares) fitting for:

- Gaussian
- Poisson
- Quasi-Poisson
- Gamma
- Tweedie (compound Poisson-Gamma)

<!-- Topics:
     - Fisher scoring / Newton-Raphson connection
     - Working weights and working responses
     - Deviance residuals
     - Dispersion parameter estimation
     - Connection to Module 2 (King Street DiD): the z_glm() with gamma family will be used for travel time reliability modelling
-->

### <a id="parts-5"></a>Parts 5+ — Causal Inference and Spatial Statistics (planned)

Difference-in-differences estimation, spatial weight matrices, and spatial econometrics tools.

## Testing

The package uses a dual-layer testing strategy:

- **Rust unit tests** (`#[cfg(test)]`): verifies computational logic in isolation, independent of R. Run with `cargo test` from `src/rust/` or from script ui with a compatible IDE and Rust language tools.
- **R integration tests** (`tests/testthat/`): verify the full R → FFI → Rust → R pipeline, including input validation in the R wrappers. Run with `devtools::test()`.

### Notable Rust Tests

**`z_lgamma` — WolframAlpha benchmarks**: Known values computed via `N[Log[Gamma[z]], 25]` in WolframAlpha, verified to $10^{-15}$ tolerance (near full f64 precision). Includes integer, half-integer, and arbitrary real inputs.

**`z_lgamma` — Reflection formula identity**: Verifies the functional equation $\ln\Gamma(z) + \ln\Gamma(1 - z) = \ln\pi - \ln|\sin(\pi z)|$ to $10^{-15}$, confirming that the reflection branch and the main branch produce mutually consistent results.

**`z_lgamma` — Smoothness across lookup table boundary**: Computes left and right numerical derivatives at $z = 3.0$ (a whole number boundary of the `LN_FACTORIALS` precomputed table) and verifies the slopes agree, confirming continuous differentiability. Critical for the solver in the planned GLM phase.

**`z_lgamma` — Extreme tails**: Tests inputs at $z = 10^{-100}$ and $z = 10^{100}$, verifying finite, non-NaN results.

**`z_pgamma` — PDF–CDF consistency**: Computes the numerical derivative of the CDF at a point and verifies it matches the analytical PDF from `z_dgamma()`, confirming the fundamental theorem of calculus relationship $f(x) = F'(x)$ holds across the two independent implementations.

**`z_pgamma` — Exponential special case**: Since Gamma(1, β) is the Exponential(β) distribution, the CDF has a closed-form solution $1 - e^{-\beta x}$, providing an exact analytical check across multiple $x$ values.

**`z_dgamma` — Riemann integration**: A brute-force numerical integration of the PDF over [0.001, 20] with dx = 0.001 verifies the density integrates to approximately 1. A basic sanity check that the normalisation is correct.

**`z_lgamma_godfrey` — Cross-validation**: The Godfrey and Boost implementations are compared across a range of inputs, verifying agreement to $10^{-12}$ and confirming that two independent formulations of the Lanczos approximation converge to the same values.

## References

- Gautschi, W. (1972). Error Function and Fresnel Integrals. In M. Abramowitz & I. A. Stegun (Eds.), Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables (Tenth Printing, with corrections, Vol. 1–1, pp. 295–330). U.S. Government Printing Office. (Original work published 1964, U.S. Government Printing Office)
- Godfrey, P. (2001). A note on the computation of the convergent Lanczos complex Gamma approximation [Unpublished manuscript].
- Jørgensen, B. (1987). Exponential Dispersion Models. Journal of the Royal Statistical Society: Series B (Methodological), 49(2), 127–162. https://doi.org/10.1111/j.2517-6161.1987.tb01685.x
- Lanczos, C. (1964). A Precision Approximation of the Gamma Function. Journal of the Society for Industrial and Applied Mathematics: Series B, Numerical Analysis, 1, 86–96.
- Maddock, J. & Boost.Math Authors. (2025). Boost.Math C++ Library (Version 1.90.0) [C++]. https://www.boost.org/doc/libs/latest/libs/math/doc/html/index.html
- Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). Numerical Recipes: The Art of Scientific Computing (3rd ed.). Cambridge University Press.
- Pugh, G. R. (2004). An analysis of the Lanczos Gamma Approximation [Ph.D.]. University of British Columbia.


<!-- TODO: Add Tweedie references like Dunn & Smyth 2005, etc. when applicable -->

## Licence

MIT

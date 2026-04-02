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
    - [Inverse Gaussian Distribution](#inverse-gaussian-distribution)
    - [Tweedie Distribution](#tweedie-distribution)
- [Planned Work](#planned-work)
  - [Part 3 — Linear Models](#part-3)
  - [Part 4 — Generalised Linear Models](#part-4)
  - [Parts 5+ — Causal Inference and Spatial Statistics](#parts-5)
- [Testing](#testing)
  - [Notable Tests](#notable-tests)
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

Functions annotated with the `#[extendr]` attribute are exported to R via the `extendr_module!` macro. Most internal helper functions like `lower_gamma_series()` and `upper_gamma_cf()` use `pub(crate)` to remain accessible within the Rust crate but are hidden from R. Most of the functions like `z_pgamma_rs()` flagged with a suffix (usually `*_rs()`) are the exported Rust function, and have an associated R wrapper like `z_pgamma()` that adds input validation and parameterisation options. The exported Rust functions are usually not intended to be called directly by a user.

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
# install.packages("pak")
pak::pkg_install("zachdaulat/statz")

# Or using devtools
# install.packages("devtools")
devtools::install_github("zachdaulat/statz")
```

Alternatively, clone and build locally:

```bash
$ git clone https://github.com/zachdaulat/statz.git
$ cd statz
```

```r
# From R, with working directory set to the package root
# install.packages(c("rextendr", "devtools"))
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

Following R’s standard naming convention, the package implements density/mass and cumulative probability distribution functions using the `d*` (density/mass) and `p*` (CDF) prefixes. R wrappers provide input validation and, where applicable, dual parameterisation (e.g., `rate` and `scale` for the gamma distribution). All density/mass computations use log-space arithmetic to avoid overflow.

The probability density/mass functions also accept a `log` argument to return the log-density/log-probability directly, and the CDF functions accept a `log.p` argument to return cumulative log-probabilities. The CDF functions also accept a `lower.tail` argument; the default is `TRUE`, and the upper tail probability is returned when `FALSE`.

| Function | Description | Method |
|----------|-------------|-----------|
| `z_dnorm(x, mean, sd)` | Normal PDF | Log-space: $-\ln\sigma - \frac{1}{2}\ln(2\pi) - \frac{z^2}{2}$ |
| `z_pnorm(x, mean, sd)` | Normal CDF | `libm::erfc()`-based for deep tails: $\frac{1}{2}\text{erfc}(-z/\sqrt{2})$ |
| `z_dpois(x, lambda)` | Poisson PMF | Log-space: $x \ln\lambda - \lambda - \sum_{i=1}^{x}\ln i$ |
| `z_ppois(x, lambda)` | Poisson CDF | Recurrence relation: $P(X = k) = P(X = k-1) \cdot \lambda/k$ |
| `z_lgamma(z)` | $\ln\Gamma(z)$ | Boost.Math Lanczos adaptation (see below) |
| `z_dgamma(x, shape, rate, scale)` | Gamma PDF | Log-space using `z_lgamma()` for the $\ln\Gamma(\alpha)$ term |
| `z_pgamma(x, shape, rate, scale)` | Gamma CDF | Regularised incomplete gamma function (see below) |
| `z_dinvgauss(y, mu, lambda)` | Inverse Gaussian PDF | Log-space: $\frac{1}{2}\ln\lambda-\frac{1}{2}\ln(2\pi)-\frac{3}{2}\ln y-\frac{\lambda(y-\mu)^2}{2\mu^2y}$ |
| `z_pinvgauss(y, mu, lambda)` | Inverse Gaussian CDF | `libm::erfc()`-based normal CDF; $\Phi(z_1) + \exp\left(\frac{2\lambda}{\mu}\right) \Phi(z_2)$ |
| `z_dtweedie(y, mu, phi, power)` | Tweedie PDF ($1 < p < 2$) | Dunn & Smyth (2005) series expansion; log-sum-exp trick with hill-climbing |
| `z_ptweedie(y, mu, phi, power)` | Tweedie CDF ($1 < p < 2$) | Linear accumulation of Poisson-weighted gamma probabilities |

The Poisson CDF upper-tail implementation currently computes $1 - P$, and therefore loses precision when $P$ is near 1. This is a pedagogical simplification for now. The Normal, Gamma, Inverse Gaussian, and Tweedie CDFs each evaluate their respective upper tails directly, preserving near full machine precision deep into the right tails.

The gamma PDF and CDF provide dual parameterisation options through providing either `rate` or `scale`, but not both. The Tweedie implementations are currently scoped specifically to the $1 < p < 2$ compound Poisson-Gamma special case, as this is the interesting case relevant for modelling zero-inflated continuous data.

#### Normal Distribution

`z_pnorm()` computes the normal cumulative probability $P(Z \le z)$ for $Z \sim N(0,1)$ from the lower tail via the complementary error function $\text{erfc}(x)$:

$$
\Phi(z) = \frac{1}{2} \text{erfc}\left(-\frac{z}{\sqrt{2}}\right)
$$

An intermediate variable `u` is calculated from the z-score input before being passed into `libm::erfc()`. To compute the upper tail, the sign of `u` is flipped passing $u = z / \sqrt{2}$ instead.

The complementary error function implementation used here is from the `libm` crate, a Rust port of C math libraries. Using this robust `erfc()` implementation enables a very simple internal structure for `z_pnorm_rs()` to compute from either tail of the normal CDF with nearly full significand precision even for exceedingly small probabilities at a magnitude of $10^{-89}$.

Note: This `libm::erfc()`-based implementation of the normal CDF is an update over my initial pedagogical version retained as `z_pnorm_as()`, which uses the Abramowitz & Stegun eq. 7.1.26 polynomial approximation of the error function. This update was motivated by the need for better tail precision for the Inverse Gaussian CDF, due to the A&S approximation's $|\epsilon(x)| \le 1.5 \times 10^{-7}$ maximum absolute error.

#### Poisson Distribution

**PMF (`z_dpois`)**

Computes the probability mass function $P(X = x | \lambda)$ using log-space arithmetic to avoid factorial overflow. The log-mass $x\ln\lambda - \lambda - \sum \ln i$ is computed directly and exponentiated only at the final step.

**CDF (`z_ppois`)**

Computes the cumulative probability for $X \sim \text{Poisson}(\lambda)$ by exploiting the recurrence relation $P(X = k) = P(X = k-1) \cdot \lambda / k$, accumulating the sum in a single pass without recomputing each PMF term independently. The package also retains the less efficient direct-iteration implementation (`z_ppois_di()`).

#### Gamma Distribution

**Log-gamma (`z_lgamma`)** — via Lanczos Approximation

Computes $\ln\Gamma(z)$ using a simplified adaptation of the [Boost.Math C++ library](https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/lanczos.html)'s Lanczos approximation, drawing from the `lanczos.hpp` and `gamma.hpp` source files. It uses the `lanczos13m53` parameters tuned for `f64` arithmetic ($N = 13$, $G \approx 6.0247$).

To contextualise the Boost.Math authors' optimisations, the standard computational form of the Lanczos approximation expresses the gamma function as $\Gamma(z+1)$ and with a partial fraction expansion:

$$
\Gamma(z+1) = \sqrt{2\pi} \left( z + g + \frac{1}{2} \right)^{z+1/2} e^{-(z+g+1/2)} \sum_{k=0}^N \frac{c_k}{z+k}
$$

`z_lgamma()` evaluates a log-space adaptation of the Boost.Math formulation:

$$
\ln \Gamma(z) = t \ln\left(t + g\right) - t + \ln L_{g,e}(z)
$$

$$
t = z - \frac{1}{2}, \quad L_{g,e}(z) = \frac{P_{12}(z)}{Q_{12}(z)} = \frac{\sum_{k=0}^{12} p_k z^k}{\sum_{k=0}^{12} q_k z^k}
$$

For $z < 0.5$, the function applies the log-space reflection formula: $\ln \Gamma(z) = \ln \pi - \ln|\sin(\pi z)| - \ln \Gamma(1 - z)$.

This implementation adapts the `lanczos13m53::lanczos_sum_expG_scaled` coefficient set from the `lanczos.hpp` source, which absorbs both the $\sqrt{2\pi}$ normalisation constant and the $e^{G}$ scaling factor directly into the $p_k$ coefficients. Below is a list of additional details about the Boost authors' optimizations and my implementation choices.

**Optimizations & Implementation Decisions**

- **(Boost) Algorithmic stability:** The Lanczos sum $L_{g,e}(z)$, is evaluated as a ratio of two degree-12 polynomials $P(z)$ and $Q(z)$ using Horner's method with `.mul_add()` for FMA operations. This required the Boost authors to compute twice as many coefficient values for a given N, but all of the coefficients can now be positive instead of alternating positive and negative, avoiding catastrophic cancellation risks.
- **(Boost) Precision simplification:** The Lanczos sum is not evaluated when the primary term (`lgam`) is so large that the addition would be entirely truncated anyway. The condition determining whether to evaluate it is `lgam * f64::EPSILON < 20.0`, the same condition found directly in the Boost.Math source `gamma.hpp`.
- **Whole number lookups:** I constructed a precomputed `LN_FACTORIALS` array storing $\ln(0!)$ to $\ln(15!)$ for small positive integer inputs ($\ln\Gamma(z)$ inputs $1 \le z \le 16$). The stored value is returned directly for these common cases, bypassing the Lanczos approximation.
- **Root handling:** The Boost implementation uses Taylor series expansions for inputs near 1 and 2. I omitted this as a deliberate simplification at the expense of reduced precision near these inputs, although I might revisit this.

The underlying Rust crate includes `z_lgamma_godfrey()`, which uses Paul Godfrey's traditional formulation (g = 7, N = 9) but suffers from cancellation risks in its alternating sign coefficients. This is retained internally as an initial pedagogical version before I wrote the Boost.Math adaptation with its more robust rational polynomial approach.

**PDF (`z_dgamma`)**

Computes the probability density function $f(x | \alpha, \beta)$ via the shape-rate parameterisation in log-space with `z_lgamma()`:

$$\ln f(x) = \alpha \ln(\beta) - \ln \Gamma(\alpha) + (\alpha-1)\ln(x) - \beta x$$

**CDF (`z_pgamma`)**

Computes the cumulative probability for $X \sim \text{Gamma}(\alpha, \beta)$ by approximating the regularised incomplete gamma functions. 

`z_pgamma_rs()` dispatches between the two helper functions `lower_gamma_series()` and `upper_gamma_cf()` based on the domain boundary $z = \alpha + 1$ to directly compute the smaller of the two tails and avoid cancellation risks for probabilities near 1.

- **Series Expansion** ($z < \alpha + 1$): Evaluates a Taylor series to compute the lower-tail probability $P(\alpha, z)$.
- **Continued Fraction** ($z \ge \alpha + 1$): Evaluates Legendre's continued fraction via the Modified Lentz Algorithm to compute the upper-tail probability $Q(\alpha, z)$.

The `lower_tail` boolean argument then determines whether the directly computed probability or its complement must be returned.

#### Inverse Gaussian Distribution 

Adding the Inverse Gaussian PDF and CDF was motivated by it representing the last remaining major case of the Tweedie family and its superior applicability for travel times and heavy-tailed data compared to the Gamma distribution.

**PDF (`z_dinvgauss`)**

The probability density function $f(y | \mu, \lambda)$ is evaluated in log-space to ensure numerical stability in future MLE loops:

$$
\ln f(y) = \frac{1}{2}\ln(\lambda) - \frac{1}{2}\ln(2\pi) - \frac{3}{2}\ln(y) - \frac{\lambda(y - \mu)^2}{2\mu^2y}
$$

**CDF (`z_pinvgauss`)**

Computes the cumulative probability for $Y \sim \text{IG}(\mu, \lambda)$, evaluated  through its relationship to the normal CDF with intermediate variables $z_1$ and $z_2$:

$$
F(y) = \Phi_{lower}(z_1) + \exp\left(\frac{2\lambda}{\mu}\right) \Phi_{lower}(z_2)
$$

Where:

$$
z_1 = \sqrt{\frac{\lambda}{y}} \left(\frac{y}{\mu} - 1\right), \quad z_2 = -\sqrt{\frac{\lambda}{y}} \left(\frac{y}{\mu} + 1\right)
$$

The absolute error magnification risk in the IG CDF correction term, with a very large exponential multiplying a very small normal CDF with a maximum absolute error magnitude $~10^{-7}$, motivated the need for the more precise `z_pnorm_std()` implementation using `libm::erfc()`. Even more extreme $\mu$ and $\lambda$ parameterisations that produce an overflowing exponential term are protected against by checking its log-space value against `f64::MAX.ln()`, and safely ignored if larger due to the correction term converging to 0 anyway via the shrinking normal CDF.

`z_pinvgauss_rs()` computes the lower or upper tail by passing the boolean to the first normal CDF term and whether the following correction term is subsequently added or subtracted. The lower tail evaluation is shown above and so the upper tail is as follows:

$$
S(y) = \Phi_{upper}(z_1) - \exp\left(\frac{2\lambda}{\mu}\right) \Phi_{lower}(z_2)
$$

#### Tweedie Distribution

The Tweedie family represents the capstone to the current pedagogical goals for implementing the normal, Poisson, Gamma, and Inverse Gaussian distributions. Their place within the Tweedie family helps me learn how their mean-variance relationships reflect a general power variance function $\text{Var}(Y) = \phi \mu^p$ emerging from the derivatives of their cumulant functions, and how they are unified under the exponential dispersion model framework.

This implementation covers the $1 < p < 2$ special case, the compound Poisson-gamma distribution. The impetus leading me to initially learn about this special case and the Tweedie family was my search for a way to model positive continuous data with a large exact point mass at zero.

Internal helper structs handle parameter conversions between the standard Tweedie parameters $(\mu, \phi, p)$ and the Poisson-gamma parameters $(\lambda, \alpha, \beta)$.

**PDF (`z_dtweedie`)**

Because the density $f(y | \lambda, \alpha, \beta)$ lacks a closed analytical form for $y > 0$, it is evaluated via the Dunn & Smyth (2005) infinite series expansion. The density is an infinite sum of Poisson-weighted gamma densities:

$$
f(y) = \sum_{k=1}^{\infty} \left( \frac{e^{-\lambda} \lambda^k}{k!} \right) \frac{\beta^{k\alpha}}{\Gamma(k\alpha)} y^{k\alpha - 1} e^{-\beta y}
$$

To handle the extreme dynamic range of the gamma densities across the series, all terms are computed in log-space. The algorithm begins with a hill-climbing search starting from $k = \lfloor\lambda\rfloor$ to locate the dominant term ($l_{\max}$). The series expands in both directions from this peak until terms drop below the machine-epsilon threshold relative to the maximum, and the accumulated log-terms are combined via a log-sum-exp evaluation. A fast-path point mass $e^{-\lambda}$ evaluates exactly $y = 0$.

**CDF (`z_ptweedie`)**

The cumulative distribution function $F(y)$ decomposes into the Poisson mass at zero plus the sum of Poisson-weighted Gamma cumulative probabilities:

$$
F(y) = e^{-\lambda} + \sum_{k=1}^{\infty} \left( \frac{e^{-\lambda} \lambda^k}{k!} \right) G(y; k\alpha, \beta)
$$

Where $G$ represents the regularised incomplete gamma function. Because each term is the product of a Poisson probability and a Gamma CDF and both are strictly bounded in $[0, 1]$, the series is safely accumulated in direct linear space, avoiding the dynamic range complexities of the density.

The `lower_tail` argument is passed directly through to the underlying `z_pgamma_rs()` calls. Because `z_pgamma_rs()` dispatches between a Taylor series and Legendre's continued fraction based on the $z = \alpha + 1$ domain boundary to maximize precision, passing the tail flag down ensures the upper tail is computed securely without naively evaluating $1 - F(y)$ when $F(y)$ is close to 1.

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

### Notable Tests

All of the reference values used in extreme case precision checks like the `z_pnorm` deep tail sigfig test were generated from WolframAlpha. The tests are also not comprehensive, domain-wide validations of the whole parameter spaces.

**`z_pnorm` — Deep tail significand preservation**: Verifies that the updated normal CDF implementation using `libm::erfc()` maintains relative precision in the extreme tails. The probability at $z = -20$ at a magnitude of $~10^{-89}$ maintains accuracy to $10^{-13}$ relative error. 

**`z_lgamma` — WolframAlpha benchmarks**: Known values computed via `N[Log[Gamma[z]], 25]` in WolframAlpha, verified to $10^{-15}$ tolerance (near full f64 precision). Includes integer, half-integer, and arbitrary real inputs.

**`z_lgamma` — Reflection formula identity**: Verifies the functional equation $\ln\Gamma(z) + \ln\Gamma(1 - z) = \ln\pi - \ln|\sin(\pi z)|$ to $10^{-15}$, confirming that the reflection branch and the main branch produce mutually consistent results.

**`z_lgamma` — Smoothness across lookup table boundary**: Computes left and right numerical derivatives at $z = 3.0$ (a whole number boundary of the `LN_FACTORIALS` precomputed table) and verifies the slopes agree, confirming continuous differentiability. Critical for the solver in the planned GLM phase.

**`z_lgamma` — Extreme tails**: Tests inputs at $z = 10^{-100}$ and $z = 10^{100}$, verifying finite, non-NaN results.

**`z_pgamma` — PDF–CDF consistency**: Computes the numerical derivative of the CDF at a point and verifies it matches the analytical PDF from `z_dgamma()`, confirming the fundamental theorem of calculus relationship $f(x) = F'(x)$ holds across the two independent implementations.

**`z_pgamma` — Exponential special case**: Since Gamma(1, β) is the Exponential(β) distribution, the CDF has a closed-form solution $1 - e^{-\beta x}$, providing an exact analytical check across multiple $x$ values.

**`z_dgamma` — Riemann integration**: A brute-force numerical integration of the PDF over [0.001, 20] with dx = 0.001 verifies the density integrates to approximately 1. A basic sanity check that the normalisation is correct.

**`z_lgamma_godfrey` — Cross-validation**: The Godfrey and Boost implementations are compared across a range of inputs, verifying agreement to $10^{-12}$ and confirming that two independent formulations of the Lanczos approximation converge to the same values.

**`z_pinvgauss` - Extreme parameter safety**: Tests the correction term safety check with an aggressive exponential parameterisation ($\frac{2\lambda}{\mu} = 1000$).

**`z_dtweedie` & `z_ptweedie` — PDF/CDF numerical consistency**: Computes the central-difference numerical derivative of the CDF and verifies it matches the independently computed PDF. Agreement to $10^{-4}$ cross-validates two independent infinite series implementations (density via log-sum-exp vs. CDF via direct accumulation).

**`z_ptweedie` — Tail complementarity**: Verifies the mathematical invariant $F(y) + S(y) = 1.0$ to a tolerance of $10^{-14}$. This confirms that passing the tail-dispatch flag through to the underlying Poisson-weighted Gamma CDF series produces cohering tail results.

**R-Level CRAN Cross-Validation**: Some R wrappers are tested against established CRAN reference packages. The Inverse Gaussian implementations match `statmod::dinvgauss` and `statmod::pinvgauss` to $10^{-15}$ and $10^{-15}$, respectively, for the tested inputs. 

The Tweedie density series implementation matches the `tweedie` package down to $10^{-10}$ for most of the $p$ space, and $10^{-14}$ near the Poisson boundary. The Tweedie CDF `z_ptweedie` matches `ptweedie` to $10^{-13}$ and $10^{-15}$ depending on the test.

<!-- ...proving mathematical soundness across extreme parameter spaces and independent algorithmic approaches (e.g., matching Dunn-Smyth series against Fourier inversion algorithms). -->

## References

- Dunn, P. K., & Smyth, G. K. (2005). Series evaluation of Tweedie exponential dispersion model densities. Statistics and Computing, 15(4), 267–280. https://doi.org/10.1007/s11222-005-4070-y
- Gautschi, W. (1972). Error Function and Fresnel Integrals. In M. Abramowitz & I. A. Stegun (Eds.), Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables (Tenth Printing, with corrections, Vol. 1–1, pp. 295–330). U.S. Government Printing Office. (Original work published 1964, U.S. Government Printing Office)
- Godfrey, P. (2001). A note on the computation of the convergent Lanczos complex Gamma approximation [Unpublished manuscript].
- Jørgensen, B. (1987). Exponential Dispersion Models. Journal of the Royal Statistical Society: Series B (Methodological, 49(2), 127–162. https://doi.org/10.1111/j.2517-6161.1987.tb01685.x
- Lanczos, C. (1964). A Precision Approximation of the Gamma Function. Journal of the Society for Industrial and Applied Mathematics: Series B, Numerical Analysis, 1, 86–96.
- Maddock, J. & Boost.Math Authors. (2025). Boost.Math C++ Library (Version 1.90.0) [C++]. https://www.boost.org/doc/libs/latest/libs/math/doc/html/index.html
- Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). Numerical Recipes: The Art of Scientific Computing (3rd ed.). Cambridge University Press.
- Pugh, G. R. (2004). An analysis of the Lanczos Gamma Approximation [Ph.D.]. University of British Columbia.
- Smyth, G., Chen, L., Hu, Y., Dunn, P., Phipson, B., & Chen, Y. (2025). statmod: Statistical Modeling (Version 1.5.1) [Computer software]. https://cran.r-project.org/web/packages/statmod/index.html
- The Rust Project Developers. (2026). `libm`: Libm in pure Rust (Version 0.2.16) [Rust]. https://crates.io/crates/libm

## Licence

MIT

# AI4BayesCode

> **Install this as an agent skill** — once installed and your agent reloaded, type
> **`/AI4BayesCode`** to launch it. See **[Installation](#installation-as-an-agent-skill)** below
> (Claude Code · Codex · Cursor · AI-assisted). The repository root IS the skill package: it bundles
> `SKILL.md`, the router `start.md`, the phase skills, and the full C++ block library under `include/`.

A header-only C++ library for **stateful, block-wise MCMC** inside a
larger Gibbs loop, plus an AI code-generation skill for turning
natural-language model descriptions into validated samplers. **Supports
R (Rcpp), Python (pybind11), and standalone C++ backends from the same
source.**

## Installation (as an agent skill)

The repository root IS the skill package — it contains `SKILL.md` plus the bundled C++ library under
`include/`. After installing and reloading your agent, type **`/AI4BayesCode`** to launch it (it loads
`start.md` and routes you to the codegen or block_design flow).

```bash
git clone https://github.com/zjg540066169/AI4BayesCode.git AI4BayesCode
cd AI4BayesCode
test -f SKILL.md && test -d include
```

### Option A — manual install for Claude Code

```bash
CLAUDE_SKILLS_DIR="$HOME/.claude/skills"
TARGET="$CLAUDE_SKILLS_DIR/AI4BayesCode"
mkdir -p "$CLAUDE_SKILLS_DIR"
[ -e "$TARGET" ] && mv "$TARGET" "$TARGET.backup.$(date +%Y%m%d-%H%M%S)"
rsync -a --exclude '.git/' --exclude '.block_design_staging/' --exclude 'generated/' --exclude '*.bak*' ./ "$TARGET"/
```

Reload Claude Code, then type `/AI4BayesCode`.

### Option B — manual install for Codex

```bash
CODEX_SKILLS_DIR="${CODEX_HOME:-$HOME/.codex}/skills"
TARGET="$CODEX_SKILLS_DIR/AI4BayesCode"
mkdir -p "$CODEX_SKILLS_DIR"
[ -e "$TARGET" ] && mv "$TARGET" "$TARGET.backup.$(date +%Y%m%d-%H%M%S)"
rsync -a --exclude '.git/' --exclude '.block_design_staging/' --exclude 'generated/' --exclude '*.bak*' ./ "$TARGET"/
```

Restart Codex.

### Option C — AI-assisted installation (let your AI do it)

Send the install guide to a local AI coding assistant and ask it to follow the steps:

- GitHub: `https://github.com/zjg540066169/AI4BayesCode/blob/main/AI_AGENT_INSTALL.md`
- Raw: `https://raw.githubusercontent.com/zjg540066169/AI4BayesCode/main/AI_AGENT_INSTALL.md`

Suggested prompt:

```text
Please install AI4BayesCode by following this guide:
https://github.com/zjg540066169/AI4BayesCode/blob/main/AI_AGENT_INSTALL.md
Do not overwrite existing files without backup.
```

### Option D — Cursor project rule

See `AI_AGENT_INSTALL.md` §6 — it writes a `.cursor/rules/ai4bayescode.mdc` pointing at the installed skill.

### After install — quick start

```text
/AI4BayesCode
```

Then describe your model in plain typeable text — e.g. *"y ~ N(Xbeta, sigma^2) with Jeffreys prior p(sigma^2) propto 1/sigma^2"* (ordinary notation is fine; just spell Greek letters out as `beta`, `sigma`) —
to GENERATE a sampler by composing existing blocks, or say *"design a new block"* to build a new
primitive.

**Backends.** The **standalone C++** backend needs only a C++17 compiler — the bundled headers under
this skill's `include/` ARE its local install (compile with `-I <install>/include`), **no R or Python
required**. The **R (Rcpp)** and **Python (pybind11)** backends are optional add-ons (see
[Dependencies](#dependencies)); the installable **R** and **Python** packages (below) resolve the header
path automatically via `system.file("include", package = "AI4BayesCode")` / the Python package path.
Either way the headers are always local — installing the skill (or a `git clone`) is all a pure-C++ user needs.

## Install as an R or Python package

The same library also ships as installable **R** and **Python** packages from this repo.
You can install **straight from GitHub**, or **clone it locally first** and install from
the clone (handy for offline installs or development).

### Clone (for local install / development)
```bash
git clone https://github.com/zjg540066169/AI4BayesCode.git
cd AI4BayesCode
```

### R  (needs a C++17 compiler and `Rcpp`, `RcppArmadillo`; `httr2` + `jsonlite` for `generate()`)
```r
# straight from GitHub:
remotes::install_github("zjg540066169/AI4BayesCode", subdir = "r-pkg")
# …or from a local clone:
remotes::install_local("AI4BayesCode/r-pkg")     # or, from inside the clone:  R CMD INSTALL r-pkg

library(AI4BayesCode)
ai4bayescode_example("GaussianLocationScale")                  # run a built-in
# describe the model in plain typeable text (spell Greek out: beta, sigma, propto):
ai4bayescode_generate("y ~ N(Xbeta, sigma^2) with Jeffreys prior p(sigma^2) propto 1/sigma^2")
```

### Python  (≥ 3.11; a C++ compiler; `brew install armadillo`)
```bash
# straight from GitHub:
pip install "git+https://github.com/zjg540066169/AI4BayesCode.git#subdirectory=python"
# …or from a local clone:
pip install ./AI4BayesCode/python                # or, from inside the clone:  pip install ./python
```
```python
import AI4BayesCode
# plain typeable text -- spell Greek out (beta, sigma), no special symbols:
AI4BayesCode.generate("y ~ N(Xbeta, sigma^2) with Jeffreys prior p(sigma^2) propto 1/sigma^2")
```

Both packages are **self-contained** — the C++ headers and the AI skill corpus ship
inside them (no checkout needed at runtime). Describe the model in **plain typeable text**:
ordinary notation is fine (`y ~ N(Xbeta, sigma^2)`, `propto`), just spell out Greek
letters (`beta`, `sigma`, `mu`, `tau`) since they can't be typed. `generate()` then
uses an LLM (Claude **or** OpenAI) to write and validate the sampler; pass your key with
`API_key=` or set it once with `set_key()`.

> **Maintainers:** edit only the canonical `include/` + `skills/` at the repo root, then
> run `bash tools/sync.sh` to regenerate the copies the two packages ship.

## What it is

`AI4BayesCode` gives you four things that Stan and raw MCMClib do not:

1. **Stateful, resumable NUTS leaf blocks** with persistent mini-warmup
   dual-averaging. Each block is an object that holds its own MCMC state
   (current draw, step size, adaptation counters) across calls to
   `step()`, so you can update some parameters with NUTS and other
   parameters with BART / Gibbs / anything else in the same outer loop
   without reinitializing.
2. **A recursive block interface.** Every sampler — leaf or composite —
   satisfies the same four-method contract (`set_context`, `step`,
   `current`, `set_current`). Composites are themselves blocks, so a
   generated hierarchical sampler can be dropped into a larger outer
   MCMC as a single child.
3. **A three-layer validator + code-generation skill chain.** The
   generator (`skills/codegen.md`) converts a user's prose model into a
   self-contained C++ file for the chosen backend (R / Python / C++ /
   Both). The validator (`skills/validator.md`) then audits it at
   Syntactic (compile), Semantic (the numbered-check registry in
   `skills/validator.md` — see "Check number registry"), and Runtime
   (smoke + 2-chain R-hat + posterior-predictive p-values + PSIS-LOO)
   layers.

4. **Backend-neutral library types + dual-module examples.** The
   library's user-facing accessors (`get_current`, `get_history`,
   `get_dag`, `set_current`, `predict_at`) return backend-neutral types
   (`state_map`, `history_map`, `dag_info`) defined in
   `include/AI4BayesCode/types.hpp`. Rcpp auto-wraps them to R lists
   via `rcpp_wrap.hpp`; pybind11 auto-casts them to Python dicts +
   numpy arrays via `pybind_casters.hpp`. Shipped examples with
   `#ifdef AI4BAYESCODE_RCPP_MODULE` / `#ifdef AI4BAYESCODE_PYBIND_MODULE`
   guards compile under either or both backends from the same `.cpp`.

## Sampler types shipped

| Strategy | When to use | Block header |
|---|---|---|
| NUTS on a single continuous parameter (default) | Most continuous parameters | `nuts_block.hpp` |
| Joint NUTS over several parameters (real and/or constrained) | Tightly-coupled continuous parameters (shift invariance, additive linear mean, random effects sharing mean); supports mixed REAL + POSITIVE (and other per-slice constraints) directly | `joint_nuts_block.hpp` |
| Gaussian-mean BART (CRAN BART R package backfitting) | `y = f(x) + noise` with tree prior | `bart_block.hpp` (GPL) |
| Generalized BART via RJMCMC (Linero 2022) | Any likelihood: Normal / Logistic / Poisson / NB / Heteroscedastic / AFT log-logistic / AFT generalized gamma / Gamma shape / Beta / Beta-Binomial / user-supplied (subclass `genbart::likelihood`) | `genbart_block.hpp` (GPL) |
| Poisson-multinomial gamma augmentation | Baker 1994 / Forster 2010 gamma trick; pairs with C-1 `genbart_block(poisson_lik)` for multinomial logistic BART in Murray 2021 Sec 3.1 C-1 identified architecture | `poisson_multinomial_aug_block.hpp` |
| Reversible-jump MCMC (variable selection, Dirac spike-and-slab, change-points) | Trans-dimensional targets; identity / linear / affine 1D transforms in v0.5 | `rjmcmc_block.hpp` |
| Finite-state Hidden Markov Model (exact FFBS) | HMM latent state sequence z_1:T | `hmm_block.hpp` |
| Pólya-Gamma augmented logistic (fast Gibbs) | **Linear logistic regression only** (not logistic BART — use `genbart_block + logistic_lik` for that) | `pg_logistic_block.hpp` |
| Elliptical Slice Sampling (latent Gaussian) | Any latent-Gaussian + arbitrary likelihood model (GP regression / classification, CAR / ICAR / GMRF, intrinsic smoothing) | `elliptical_slice_sampling_block.hpp` |
| Univariate slice sampling (Neal 2003) | 1-D continuous parameter where NUTS can't be used (non-diff log p, black-box library call, prohibitively expensive gradient) | `univariate_slice_sampling_block.hpp` |
| 1-D time-series GP via celerite O(N) Cholesky | Astronomical / financial / climate time-series; spline-like 1-D extrapolation for large N | `celerite_gp_block.hpp` + `celerite_marginal_likelihood.hpp` |
| Exact Dirichlet conjugate | Dirichlet-Categorical, LDA | `dirichlet_gibbs_block.hpp` |
| Exact Beta conjugate | Spike-slab mixing, Beta-Binomial | `beta_gibbs_block.hpp` |
| Closed-form Bernoulli | Spike-slab, variable selection | `binary_gibbs_block.hpp` |
| Closed-form categorical | Mixture labels, LDA assignments | `categorical_gibbs_block.hpp` |
| InverseGamma conjugate (discouraged) | Legacy; see Gelman 2006 critique in `skills/codegen.md §2a` | `inv_gamma_gibbs_block.hpp` |

The generator picks strategies in this priority order:

1. Specialized shipped blocks (BART, genBART, HMM FFBS, PG logistic,
   Dirichlet-conj, Beta-conj, binary / categorical conjugate,
   Poisson-multinomial gamma augmentation) wherever they apply —
   always faster AND correct-by-construction.
2. **Modular `nuts_block` per continuous parameter** (default).
3. **`joint_nuts_block`** as an opt-in performance escape hatch when
   coupling analysis (see `skills/codegen.md` §4a) indicates "High"
   coupling (mixed REAL + POSITIVE slices handled directly).
4. **`rjmcmc_block`** for trans-dimensional targets (variable selection
   via Dirac spike-and-slab, change-points, unknown model order). See
   `examples/SpikeSlabRJMCMC.cpp` reference template.

## Layout

```
AI4BayesCode/
├── skills/                              AI generator + validator skill chain
│   ├── codegen.md                         natural-language → Rcpp .cpp + R runner
│   ├── block_catalogue/index.md                 which block to pick per parameter
│   ├── constraints.md                     the 16 constraint transforms + joint_constraint
│   ├── validator.md                       3-layer audit (syntactic / semantic / runtime)
│   └── rcpp_api.md                        Rcpp / arma type + API pitfalls
├── R/
│   └── AI4BayesCode_helpers.R              ai4bayescode_sourceCpp + ai4bayescode_perf_hint
├── include/
│   ├── AI4BayesCode/                       Layer-1 header-only library
│   │   ├── block_sampler.hpp              abstract 4-method interface
│   │   │                                  + current_named_outputs() for joint blocks
│   │   ├── shared_data.hpp                DAG-aware shared state
│   │   │                                  (Gibbs DAG + predict DAG + stochastic refreshers)
│   │   ├── nuts_block.hpp                 NUTS wrapper w/ persistent adapt
│   │   ├── joint_nuts_block.hpp           joint NUTS over several sub-params
│   │   │                                  (REAL + POSITIVE and other per-slice constraints)
│   │   ├── composite_block.hpp            recursive Gibbs scheduler + predict_at(rng)
│   │   ├── constraints.hpp                16 constraint transforms
│   │   ├── binary_gibbs_block.hpp         closed-form Gibbs for z ∈ {0,1}
│   │   ├── beta_gibbs_block.hpp           exact Beta(a, b)
│   │   ├── dirichlet_gibbs_block.hpp      exact Dirichlet via gamma-norm
│   │   ├── categorical_gibbs_block.hpp    closed-form Gibbs for z ∈ {1..K}
│   │   ├── inv_gamma_gibbs_block.hpp      InvGamma conjugate (DISCOURAGED — Gelman 2006)
│   │   ├── rjmcmc_block.hpp               reversible-jump MCMC (Dirac spike-and-slab)
│   │   ├── rjmcmc_transforms.hpp          library 1D bijections for rjmcmc_block v0.5
│   │   ├── hmm_block.hpp                  finite-state HMM via forward-filter backward-sample
│   │   ├── pg_logistic_block.hpp          Pólya-Gamma logistic regression (LINEAR only)
│   │   ├── poisson_multinomial_aug_block.hpp Baker/Forster gamma aug for C-1 genBART multinomial
│   │   ├── elliptical_slice_sampling_block.hpp   Murray-Adams-MacKay 2010 ESS (latent Gaussian)
│   │   ├── univariate_slice_sampling_block.hpp   Neal 2003 section 4.1 slice (1-D, NUTS fallback)
│   │   ├── celerite_gp_block.hpp          celerite O(N) Cholesky for 1-D time-series GP
│   │   ├── celerite_marginal_likelihood.hpp stateless helper: celerite log-marginal-likelihood
│   │   ├── bart_block.hpp                 Gaussian-mean BART (GPL, Rcpp-only)
│   │   └── genbart_block.hpp              Linero 2022 generalized-BART RJMCMC with pluggable likelihoods (GPL, Rcpp-only)
│   ├── mcmclib/                         vendored MCMClib + patches
│   ├── eigen/                           vendored Eigen 3.4 (MPL-2.0)
│   └── autodiff/                        vendored autodiff 1.1.2 (MIT, gen-time Check #12)
├── libgp_kernels/                       vendored libgp kernel subsystem (BSD-3)
├── celerite/                            vendored celerite core (MIT, 1-D O(N) GP)
├── bart_pure_cpp/                       vendored BART-family tree kernels
│   └── src/
│       ├── BART/                           CRAN BART R package C++ sources (McCulloch/Sparapani/Spanbauer)
│       ├── GENBART/                         genBART tree infra + RJMCMC + likelihoods (Linero 2022, namespace genbart::)
│       └── SOFTBART_VENDOR/                 SoftBART soft-decision trees (Linero & Yang 2018)
├── tests/                               pure-C++ unit tests (round-trips, FD gradients)
├── examples/                            39 reference examples (standalone C++ demos)
│   ├── *.cpp                              39 reference examples (Gaussian, logistic, BART,
│   │                                      GP, HMM, mixtures, spatial GMRF, IRT, splines,
│   │                                      VI, ...) — each a self-contained `int main()` demo
│   ├── run_*.R                            runner scripts (wall-clock + R3 + perf_hint)
│   ├── test_*.R                           R-level diagnostic tests
│   └── Makevars                           paths for R_MAKEVARS_USER
└── README.md                            this file
```

## Supported backends

| Backend | Module macro | Helper | Python/R call |
|---|---|---|---|
| **R (default)** | `RCPP_MODULE(...)` | `R/AI4BayesCode_helpers.R` | `ai4bayescode_sourceCpp(...)` |
| **Python** | `PYBIND11_MODULE(...)` | `python/AI4BayesCode/` | `AI4BayesCode.source(...)` |
| **Standalone C++** | `int main()` | user-supplied Makefile/CMake | direct compile |
| **Dual (R + Python)** | both, guarded by `#ifdef` | both helpers | either, from the same `.cpp` |

The R helper sets `-DAI4BAYESCODE_RCPP_MODULE`; the Python helper sets
`-DAI4BAYESCODE_PYBIND_MODULE`. A generated sampler can guard an
`RCPP_MODULE` and/or `PYBIND11_MODULE` block with these macros so the
same `.cpp` compiles for either backend (see the backend-neutral types
described above).

## Quick start — R

```r
source("AI4BayesCode/R/AI4BayesCode_helpers.R")
ai4bayescode_sourceCpp("AI4BayesCode/examples/GaussianLocationScale.cpp",
                    AI4BayesCode_path = "AI4BayesCode")

set.seed(1)
y <- rnorm(100, 2.0, 1.5)

m <- new(GaussianLocationScale, y, seed = 42L, keep_history = TRUE)
m$step(4000L)   # warmup
m$step(4000L)   # sampling

h  <- m$get_history()
pp <- m$predict_at(list())   # posterior-predictive y_rep

cat("mu:    mean =", mean(h$mu),    " sd =", sd(h$mu),    "\n")
cat("sigma: mean =", mean(h$sigma), " sd =", sd(h$sigma), "\n")
cat("y_rep shape:", paste(dim(pp$y_rep), collapse = " x "), "\n")
```

## Quick start — Python

Install (one-time):

```bash
brew install armadillo            # macOS; or `apt install libarmadillo-dev` on Linux
pip install pybind11 numpy
# optional extras:
pip install networkx matplotlib arviz       # DAG viz + diagnostics
```

Use:

```python
import sys
sys.path.insert(0, "AI4BayesCode/python")    # or `pip install ./AI4BayesCode/python`
import AI4BayesCode
import numpy as np

mod = AI4BayesCode.source(             # sourceCpp(...) is a back-compat alias
    "AI4BayesCode/examples/GaussianLocationScale.cpp",
    ai4bayescode_path="AI4BayesCode"
)

np.random.seed(1)
y = np.random.normal(2.0, 1.5, 100).astype(np.float64)
m = mod.GaussianLocationScale(y, rng_seed=42, keep_history=True)
m.step(4000)    # warmup
m.step(4000)    # sampling

hist = m.get_history()      # dict[str, np.ndarray], each (n_draws, dim)
mu_draws    = hist["mu"][:, 0]
sigma_draws = hist["sigma"][:, 0]

# Diagnostics (built into AI4BayesCode)
print(f"R-hat(mu):    {AI4BayesCode.rhat(mu_draws):.4f}")
print(f"ESS(mu):      {AI4BayesCode.ess_bulk(mu_draws):.0f}")
print(AI4BayesCode.posterior_summary(mu_draws))

# Parallel multi-chain
def build(seed):
    return mod.GaussianLocationScale(y, rng_seed=seed, keep_history=True)
chains = AI4BayesCode.run_chains(build,
                                  seeds=[1, 2, 3, 4],
                                  n_burn=2000, n_keep=2000, n_jobs=4)

# DAG visualization
AI4BayesCode.plot_dag(m, out_path="my_dag.png")
```

## Shipped examples (frontend-independent)

All **39** shipped examples under `examples/*.cpp` are
**frontend-independent standalone C++ demos** — each one is a
self-contained `int main()` that builds the data, runs the sampler,
and checks recovery with **no R or Python required** (compile with a
C++17 compiler and `-I include`). They cover Gaussian, logistic,
probit, BART / genBART / SoftBART, GP regression / classification /
time-series, HMM, Dirichlet / DP / Pitman-Yor / finite / HDP mixtures,
spatial GMRF (ICAR / GMRF prior), IRT, B-spline / HSGP, ODE-SIR,
reversible-jump spike-and-slab, and several mean-field / structured VI
models.

To use one of these models from **R** or **Python**, point the
corresponding helper at the `.cpp` so it is wrapped and imported as a
module:

- **R** — `ai4bayescode_sourceCpp("examples/GaussianLocationScale.cpp",
  AI4BayesCode_path = "AI4BayesCode")` (see "Quick start — R").
- **Python** — `AI4BayesCode.source("examples/GaussianLocationScale.cpp",
  ai4bayescode_path="AI4BayesCode")` (see "Quick start — Python").

The two installable packages (R `r-pkg/`, Python `python/`) ship the
same library headers and skill corpus, so `generate()` can emit a
wrapper-bearing `.cpp` for the backend you choose.

## Posterior-predictive + PSIS-LOO (Layer 3 R3)

Every wrapper with a registered stochastic refresher (all shipped
examples, including all 4 shipped `GBart*` wrappers) emits `y_rep` uniformly via
`predict_at(list())`. The generated R runner wires this into the
validator's Layer 3 R3:

```r
pp      <- m$predict_at(list())            # n_draws × N posterior predictive
bp_stat <- list(mean = mean, sd = sd, ...) # 6 summary stats
pv      <- sapply(bp_stat, function(f)
    mean(apply(pp$y_rep, 1, f) >= f(y_obs)))
# pv ∈ (0.05, 0.95) → OK

library(loo)
LL <- pointwise_loglik(h, y_obs)           # generated per observation family
# stack across 2 chains, call loo::loo(LLarr, r_eff = ...)
```

See `skills/codegen.md` §6a for the per-observation-family templates
(Gaussian / Bernoulli / Poisson / Multinomial / BART-noise).

## Joint NUTS — performance escape hatch

When the coupling analysis flags a High-coupling pattern (IRT-style
shift invariance, additive linear mean, fixed + random effect sharing
mean), the generator emits a `joint_nuts_block`:

```cpp
joint_nuts_block_config cfg;
cfg.name = "theta_b_joint";
cfg.sub_params.push_back({ "theta", N });
cfg.sub_params.push_back({ "b",     J });
cfg.initial_cat = arma::join_cols(theta_init, b_init);
cfg.log_density_grad = &joint_theta_b_log_density;
impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
```

See `examples/IRT1PL_joint.cpp`, `HierarchicalLM_joint.cpp` for
reference. Mixed REAL + POSITIVE slices (e.g. joint (alpha, beta,
sigma)) are handled by `joint_nuts_block` directly — set the per-slice
`constraint` to `POSITIVE` for the positive sub-params; see
`examples/LinearRegJointMixed.cpp`.

Joint NUTS has a higher semantic-bug surface (concatenate-and-slice
code). The validator's **Check #11** specifically audits: gradient
slice alignment, prior completeness per sub-param, scale consistency,
Jacobian, write-back offsets, and construction-time dim asserts.
R3's Bayesian-p-value threshold tightens to (0.02, 0.98) when joint
NUTS is used.

### Known limitation (v1.1 roadmap)

`joint_nuts_block` currently uses mcmclib's default **identity mass
matrix** by default. Practical dimension envelope:

- Works well up to `~dim 70` — validated end-to-end (IRT at N=60, J=12;
  LinearRegJointMixed at dim 7; HierarchicalLM_joint at dim 14).
- Expect slow cross-chain mixing at dim ≥ 100 — chains find the right
  region (posterior recovery 100% within 3sd, LOO k<0.5 ≥ 93%) but
  do not cross-chain-align within a few hundred warmup iterations.

An opt-in online dense-metric (Welford) adaptation
(`cfg.use_dense_metric = true`, gated by validator Check #18) is
available for ill-conditioned targets; see `skills/validator.md` §18.

## Validator (Layer 1 / 2 / 3)

Every generated sampler passes through three layers of audit:

1. **Syntactic** — `sourceCpp` compiles.
2. **Semantic** — the numbered code-review checks in the
   `skills/validator.md` "Check number registry" (most always-on, a few
   conditional — e.g. #11 joint NUTS, #13 RNG separation; #12 is the one
   gen-time gradient-verification check). These catch bugs that compile
   and run but produce the wrong posterior.
3. **Runtime** — executes the sampler once; three sub-steps share a
   single 2-chain `keep_history = TRUE` run:
   - **R1 smoke test** (~10 steps): finite values, predict_at doesn't
     mutate state.
   - **R2 R-hat / ESS** (4000 burn + 4000 keep per chain): does the
     sampler converge to *some* stationary distribution?
   - **R3 posterior check**: is that stationary distribution the
     *right* posterior? 6 Bayesian p-values + PSIS-LOO via `loo::loo()`.

The generated R runner emits all three layers plus a
`ai4bayescode_perf_hint()` call at the end that nudges the user toward
`joint_nuts_block` if per-sweep time is slow.

See `skills/validator.md` for the full semantic-check registry and
failure-routing guide.

## Dependencies

- C++17 compiler
- RcppArmadillo (or a standalone Armadillo install)
- R packages: `Rcpp`, `RcppArmadillo`, `posterior`, `loo`
- On macOS, the bundled Accelerate framework is used for BLAS/LAPACK;
  on Linux, link against `-lblas -llapack`.

## Licensing

**The combined, distributed work — AI4BayesCode plus the vendored
dependencies it compiles against — is licensed as a whole under the
GNU General Public License, version 3.0 or later
(GPL-3.0-or-later).** This is the project's elected license; the full
text is in `LICENSE` at the repo root. Project code is
`Copyright (C) 2026 AI4BayesCode`. Any sampler you generate inherits
GPL-3.0-or-later.

- GPL **v3** (not v2) is required because the vendored Apache-2.0
  components (mcmclib / BaseMatrixOps, used by the NUTS backend) are
  one-way compatible with GPL-3 but not with GPL-2.0-only. The
  BART-family kernels are GPL-2.0-or-later upstream, and the "or-later"
  clause resolves to the GPL-3 branch for the combined work.
- Per-component upstream licenses are unchanged and remain in force:
  the vendored BART kernel under `bart_pure_cpp/src/BART/` is the **CRAN BART R
  package** (Copyright (C) 2017-2018 McCulloch / Sparapani /
  Spanbauer; header-only adaptation Copyright (C) 2024-2026 Jungang
  Zou), GPL-2.0-or-later; `bart_pure_cpp/src/GENBART/` (genBART, Linero 2022)
  and `bart_pure_cpp/src/SOFTBART_VENDOR/` (SoftBART, Linero & Yang 2018) are
  GPL-2.0-or-later. Any translation unit that `#include`s
  `bart_block.hpp`, `genbart_block.hpp`, or `softbart_block.hpp`
  transitively pulls the corresponding GPL kernel.
- Other vendored code retains its own GPL-compatible upstream
  license: Eigen (MPL-2.0), autodiff (MIT), mcmclib / BaseMatrixOps
  (Apache-2.0), celerite (MIT), libgp (BSD-3-Clause), librjmcmc
  (CeCILL-B). Each keeps its original LICENSE / COPYING file.

See `THIRD_PARTY_LICENSES.md` for the full per-file attributions,
modification notes, and the complete compatibility analysis.

## Notes on the vendored mcmclib

This repository ships a modified copy of
[MCMClib](https://github.com/kthohr/mcmc) under Apache 2.0. The
modifications are documented in the headers of
`include/mcmclib/mcmc/nuts.hpp` and `include/mcmclib/mcmc/nuts.ipp`,
and include:

- Persistent mini-warmup adaptation mode (`use_persistent_adapt`,
  `epsilon_bar_persist`, etc.).
- Overflow / underflow clamp in `nuts_find_initial_step_size`.
- Tree-doubling fix: the inner loop now uses the `draw_neg` /
  `draw_pos` endpoints instead of the stale `prev_draw`, matching
  Hoffman & Gelman 2014 Algorithm 6.
- A fix to the step-size search direction in
  `nuts_find_initial_step_size` so it matches Hoffman & Gelman 2014
  Algorithm 4 (the original only doubled upward and could not halve
  when the initial epsilon was too large).

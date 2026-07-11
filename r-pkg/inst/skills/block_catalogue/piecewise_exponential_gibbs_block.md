## piecewise_exponential_gibbs_block

**Reference example:** `examples/PehSurvival.cpp` (piecewise-exponential baseline
hazard, no covariates -- the smallest working demo). Compose with `nuts_block(beta)`
for Cox-style linear regression (see the SelectWhen "USAGE" block below).

Exact Gamma-conjugate Gibbs update for the K piecewise-constant baseline hazard
rates of a piecewise-exponential (PEH) survival model with an EXTERNALLY-supplied
per-subject log-relative-hazard `f_i` (Cox-style proportional-hazards structure).

**Model** (Ibrahim, Chen & Sinha 2001 Sec.3.2; Kalbfleisch 1978; Prentice & Gloeckler
1978): time axis is partitioned into `K` bins by edges `0 = e_0 < e_1 < ... < e_K`.
Baseline hazard is piecewise constant `h_0(t) = lambda_k` for `t in (e_{k-1}, e_k]`.
Per-subject hazard multiplies by `exp(f_i)` from another block (linear predictor
`x_i^T beta`, BART, GP, spline, ... -- this block does NOT know or care):

```
h_i(t) = lambda_{k(t)} * exp(f_i)
```

Observed data: event / censoring time `t_i > 0`, event indicator `delta_i in {0,1}`,
optional left-truncation entry time `v_i >= 0` (subject at-risk on `[v_i, t_i]`).

**Full-conditional (Gamma-Poisson conjugacy)** -- the joint log-likelihood
factors across bins into `K` independent Poisson kernels in `lambda_k`:

```
E_k = sum_i delta_i * I(t_i in bin k)                  (events in bin k)
R_k = sum_i exp(f_i) * Delta_k(t_i, v_i)               (weighted at-risk time)
Delta_k(t_i, v_i) = max(0, min(t_i, e_k) - max(v_i, e_{k-1}))

lambda_k | rest ~ Gamma(a0 + E_k,  b0 + R_k)           (independent across k)
```

One exact Gibbs step draws all K rates. O(n + K) per step. K=10..500 comfortable.

```cpp
#include "AI4BayesCode/piecewise_exponential_gibbs_block.hpp"

piecewise_exponential_gibbs_block_config cfg;
cfg.name  = "lambda";                              // shared_data + history key (writes length-K)
cfg.edges = arma::vec({0.0, 1.0, 3.0, 6.0, 12.0}); // K = 4 bins on [0, 12]
cfg.a0    = 0.01;                                  // Gamma(a0, b0) prior on each lambda_k
cfg.b0    = 0.01;                                  // (0.01, 0.01) weakly informative
// time_key="t", event_key="delta" (defaults) -- must be in shared_data.
// offset_key: OPTIONAL -- sibling block writes exp(x_i^T beta) each sweep here.
cfg.offset_key = "exp_lp";                         // "" or key-missing => baseline-only PEH
// entry_time_key: OPTIONAL for left-truncation.
composite->add_child(std::make_unique<piecewise_exponential_gibbs_block>(std::move(cfg)));
```

**Parameterization**: Gamma shape-RATE (matches `gamma_gibbs_block` /
`inv_gamma_gibbs_block`, R's `rgamma(n, shape, rate)`, JAGS / NIMBLE / Stan).
Internal `std::gamma_distribution` is shape-SCALE; conversion is commented at
the call site.

**Bin convention**: bins are `(e_{k-1}, e_k]`, k = 1..K (1-based math, 0-based
storage). A time `t_i = e_k` exactly lands in bin `k`. `edges[K]` must exceed
`max(t_i)` observed follow-up (throws otherwise).

**USAGE (PEH + Cox linear log-hazard-ratio beta):**

```cpp
// Sibling block that samples beta and writes exp(x_i^T beta) each sweep.
// See examples/PehCoxRegression.cpp for the wired-up template.
```

**JUSTIFICATION (Check #16): Exception 3** (textbook multivariate conjugate
with NUTS-wasteful efficiency profile). `joint_nuts_block` on `log lambda[K]`
mixes slowly at K=50-200 under any diagonal metric (Gamma-tailed posterior on
each `lambda_k`), and a full-dense metric is O(K^2) per sample. The Gibbs
update is 50-100x faster and exact (no warmup / no autocorrelation on
`lambda`).

**Check #15** parity:
`tests_autodiff/block_tests/test_piecewise_exponential_gibbs_block.cpp`
verifies BOTH E_k / R_k accounting and the Gamma(a0 + E_k, b0 + R_k) sample
distribution on a hand-computed 4-obs / 3-bin fixture at 20 000 draws,
within 5 % mean / 15 % variance tolerance.

**Do NOT use** when:
- baseline hazard is smoothly varying (e.g. spline / GP prior on `log lambda`):
  the GMRF / GP prior breaks the Gamma-Poisson conjugacy; use `joint_nuts_block`
  on `log lambda[K]` with the smoothness prior instead;
- observations are interval-censored (`t_i` only known to lie in `[L_i, U_i]`):
  compose with `interval_censored_survival_augmentation_block` first (that block
  imputes `t_i` from a truncated PEH survival distribution given lambda, then
  this block updates lambda as usual);
- competing risks / multi-state models: this block handles single-event
  right-censored (or left-truncated + right-censored) survival only.

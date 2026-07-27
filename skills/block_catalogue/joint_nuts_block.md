## joint_nuts_block

A SINGLE NUTS block that owns multiple named real-valued sub-parameters
and samples them JOINTLY on a concatenated unconstrained vector. This
is the performance escape hatch for models whose likelihood tightly
couples several continuous parameters (shift-invariance, additive
linear mean, fixed+random effect sharing the mean structure).

```cpp
#include "AI4BayesCode/joint_nuts_block.hpp"
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;

joint_nuts_block_config cfg;
cfg.name = "theta_b_joint";           // block name (not a parameter name)

// Sub-parameter layout. Offsets are assigned in the order listed below.
cfg.sub_params.push_back({ "theta", N });   // shared_data key "theta", dim N
cfg.sub_params.push_back({ "b",     J });   // shared_data key "b",     dim J

// Concatenated initial value, same layout order.
cfg.initial_cat = arma::join_cols(theta_init, b_init);

// Joint log-density on the concatenated REAL-valued vector. The lambda
// MUST write the full-length gradient, slicing at the sub_params
// offsets (validator Check #11.1).
cfg.log_density_grad = &joint_theta_b_log_density;

// Joint blocks usually need more warmup than per-param NUTS because
// the (N+J)-dim mass matrix needs more runway.
cfg.n_warmup_first_call = 1000;

impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));

// Dependencies are keyed by BLOCK name, not sub-param names.
impl_->data().declare_dependencies(
    "theta_b_joint", {"Y", "M", "sigma_b"});  // union of sub-param reads
```

### What you gain

- A single NUTS trajectory over the whole (N+J)-dim space per Gibbs
  sweep. Joint sampling sees the cross-parameter dependence structure
  (shift invariance, additive-linear-mean correlation, ...) that
  Gibbs-wise NUTS can't express, so at small-to-moderate dimension the
  chains mix well and recover posteriors that modular NUTS would
  need orders of magnitude more sweeps to reach.
- After each step the block splits the concatenated draw back into the
  individual shared_data keys (`theta`, `b`, ...) via
  `current_named_outputs()`, so downstream blocks that read those keys
  (e.g. a separate sigma-NUTS) work unchanged.

### What you pay for

- Higher semantic-bug surface: one monolithic log-density with
  slicing, offsets, and prior assembly. Validator Check #11 exists
  specifically to audit this.
- Per-slice constraints are supported: each sub-parameter may declare
  its own constraint (REAL / POSITIVE / SIMPLEX / CHOLESKY_CORR /
  COV_MATRIX / ...) and the block applies the transform + log|Jacobian|
  internally. A positive / simplex parameter can be co-sampled in the
  same block; no need to concatenate by hand.
- **The block defaults to an IDENTITY mass matrix** (mcmclib's default).
  Step-size dual averaging alone is not enough on highly-correlated
  posteriors -- chains find the right region but may require very
  long warmup to agree across seeds. **Welford-based diagonal mass matrix adaptation (SHIPPED
  2026-04-20) adds opt-in dense-metric adaptation** via
  `cfg.use_dense_metric = true`, plus
  `cfg.dense_metric_pilot_iters` and `cfg.dense_metric_adapt_iters`.
  When enabled, the FIRST `step()` call runs a pilot NUTS phase with
  identity metric, collects samples, computes their sample
  covariance (Stan-style `n/(n+5)*Sigma + 1e-3*(5/(n+5))*I` regularization),
  inverts to the precision matrix (mcmclib's `precond_mat` IS the mass
  matrix, so we pass Sigma^-^1), and uses dense metric for all subsequent
  sampling. Test coverage:
  `tests_autodiff/test_joint_nuts_dense_metric.cpp` -- on a 10-dim
  rho=0.95 correlated Gaussian, dense metric gives max split-R-hat
  1.002 vs identity metric's 1.19.

  **The practical ceiling is model-specific, not dim-specific.** The
  key property is how strongly the posterior correlates across sub-
  parameters; identity-metric NUTS handles weakly-correlated
  high-dim fine, but struggles with strongly-correlated modest-dim.
  Validated envelope (10k+10k x 2 chains):
   - IRT1PL_joint is **well-conditioned**: clean R-hat < 1.01 at
     dim 38, 72, and **115** (N=100, J=15, 10k+10k). The earlier
     "dim 70 ceiling" was set at a shorter-budget run and is revised
     up -- IRT is not the bottleneck.
   - HierarchicalLM_joint is **more correlated** (fixed effects and
     random effects share the mean structure): R-hat 1.004 at G=10
     (dim 14), 1.037 at G=20 (dim 24), but **2.1 at G=50 (dim 54)**
     on u and tau. Fails well below IRT's passing dim.
   - The same code geometry is fine in IRT at dim 115 and fails in
     HierLM at dim 54. Treat dim as a rough guide only; the real
     test is per-parameter R-hat at 10k+10k.
   - **HierLM-style joint blocks:** when sub_params contain a positive
     scale parameter (sigma / tau) AND a real vector of random effects
     with large dim (e.g. `(sigma, eps_raw[M])`, or `(tau, u[J])`),
     identity-metric NUTS often fails (rhat ~= 1.28 at dim 42, ~= 2.1 at
     dim 54), so dense metric is FREQUENTLY needed -- but START DIAGONAL
     and escalate to `cfg.use_dense_metric = true` as a Check #18 step
     when per-parameter R-hat at 10k+10k (or R2/R3) shows diagonal is
     inadequate. Measure; do NOT gate dense on a dimension threshold
     (consistent with the "dim is a rough guide only" note above; there
     is no "Check #11.7").
   - For other joint patterns (shift-invariance, additive linear mean
     without per-unit random effects), the DEFAULT is still ONE
     `joint_nuts_block` over the coupled continuous params -- do NOT split
     "to keep it simple" (coupled params mix ~10x slower when split and
     freeze on funnels; `codegen_cpp.md` Sec.4a). Reserve modular only for
     genuinely scalar params, post-NCR branches, or obvious conditional
     independence.

### Stan-style 3-phase warmup (SHIPPED 2026-06-03)

Stan-style windowed warmup; **OPT-IN (default OFF since 2026-06-20)** -- set
`cfg.use_three_phase_warmup = true` only as an EXTREME-COND escalation. The
default for dense is the single-phase pilot above (broad-corpus 5-23x faster
keep-phase ESS/s; converges across all realistic cond). Reach for three-phase
only when single-pilot dense still gives R-hat > 1.05 on a high-curvature /
ridge-trapping target (sparse spatial / ICAR). When enabled, the warmup is split
into Stan-style expanding windows so the mass matrix is re-estimated multiple
times as the chain enters the typical set:

- **Phase I** (`cfg.tp_phase1_iters`, default **75 iter**): identity-
  metric step-size dual-averaging only -- no mass-matrix updates.
- **Phase II** (`cfg.tp_phase2_windows`, default **{25, 50, 100, 200,
  500} iter**, total 875): per window, collect samples, compute the
  Welford covariance with Stan-style `n/(n+5)*Sigma_window + 1e-3*(5/(n+5))*I`
  regularisation, install as the mass matrix for the NEXT window, and
  re-tune step size under the new metric. The expanding-window schedule
  prioritises the latest window's geometry over the noisy early region.
- **Phase III** (`cfg.tp_phase3_iters`, default **50 iter**): final
  identity-metric step-size dual-averaging under the locked mass
  matrix; no further mass-matrix updates.

Total default budget: 75 + 875 + 50 = **1000 warmup iter** (matches
Stan default).

**When to use vs single-phase pilot:**
- Single-phase (default): well-conditioned targets, low correlation,
  a single Welford pass on the typical set converges quickly.
- 3-phase (opt-in): high-curvature or multi-scale targets where the
  pilot covariance changes substantially between burn-in and the
  stationary region (e.g. sparse spatial / temporal random effects
  with strong long-range correlation, hierarchical funnels under
  non-centered parameterisation). Empirical envelope: on small
  spatial random-effect fixtures the three-phase schedule reduces
  wall-clock roughly 2x over single-phase with comparable-or-better
  ESS while preserving posterior identity (cross-mode R-hat
  agreement).

**Limitation**: existing examples that decompose into multiple
separate `nuts_block`s instead of a single `joint_nuts_block` do
NOT benefit from 3-phase warmup without refactor or regeneration.
The 3-phase machinery operates on the joint block's mass matrix;
per-slice NUTS blocks each have their own scalar adaptation.

### When to pick joint vs modular

See `skills/codegen_cpp.md` Sec.4a (Coupling analysis). The short
version:

- **Specialized / structural blocks FIRST.** Always use a specialized
  block (`bart_block`, `genbart_block`, `dirichlet_gibbs_block`,
  `beta_gibbs_block`, `binary_gibbs_block`, `categorical_gibbs_block`,
  `hmm_block`, `pg_logistic_block`, `elliptical_slice_sampling_block`,
  `rjmcmc_block`, GMRF / GP blocks) when it GENUINELY applies -- faster AND
  correct-by-construction. For some (discrete latent, random parameter-
  space dimension) NUTS is structurally inapplicable, so the specialized
  block is mandatory, not just preferred.
- **`joint_nuts_block` is the DEFAULT** for every continuous parameter NOT
  claimed by a specialized block. The Sec.4a coupling table flags where joint
  is most critical (shift-invariance, additive linear mean, fixed+random
  effect, hierarchical scales -- they FREEZE or bias if split), but joint is
  the default regardless of coupling level -- do NOT split continuous
  parameters into per-parameter blocks to "keep it simple".
- single `nuts_block` is **LOW priority** -- only a genuinely scalar
  continuous parameter, a post-NCR funnel branch, or obvious conditional
  independence the generator chooses to isolate.

### Reference examples

- `examples/IRT1PL_joint.cpp` -- (theta, b) joint with separate sigma_b.
- `examples/HierarchicalLM_joint.cpp` -- (alpha, beta, u) joint with
  separate sigma, tau.

## joint_nuts_block -- per-slice constraints

`joint_nuts_block` supports **per-slice constraint kinds** (REAL,
POSITIVE, SIMPLEX, CHOLESKY_CORR, COV_MATRIX, ...). The block handles
the unconstrained-vs-natural transformation and log-Jacobian
bookkeeping internally, so the user's log-density stays on the natural
scale.

```cpp
#include "AI4BayesCode/joint_nuts_block.hpp"
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;

joint_nuts_block_config cfg;
cfg.name = "abs_joint";
cfg.sub_params.push_back({"alpha", 1, joint_constraint::REAL});
cfg.sub_params.push_back({"beta",  p, joint_constraint::REAL});
cfg.sub_params.push_back({"sigma", 1, joint_constraint::POSITIVE});

// Natural-scale initial values (sigma > 0).
cfg.initial_cat = arma::join_cols(arma::vec{alpha_init}, beta_init,
                                    arma::vec{sigma_init});

// User-written NATURAL-scale log-density. NO Jacobian terms -- the
// block adds them per-slice. Gradient is w.r.t. natural-scale
// parameters; the block chain-rules it back to the unconstrained scale.
cfg.log_density_grad = &joint_log_density_natural;

impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
```

### Constraint kinds

- `joint_constraint::REAL` -- identity transform, zero Jacobian.
- `joint_constraint::POSITIVE` -- log transform. Block stores
  `unc = log(nat)`, user's log-density sees `nat = exp(unc) > 0`.
  Block adds `log|Jacobian| = sum(unc)` across the slice and applies
  chain rule: `grad_unc_i = grad_nat_i * nat_i + 1`.

When a block has a positive hyperparameter (like sigma) alongside
real-valued parameters (alpha, beta), declare sigma's slice as
POSITIVE rather than splitting it into a separate `nuts_block` --
co-sampling sigma with the location parameters materially improves
mixing when alpha/beta and sigma are posterior-correlated.

### dense_metric for HierLM coupling (Check #18 escalation, measured)

When sub_params include both a POSITIVE scale parameter AND a REAL
vector of per-unit random effects at large dim -- e.g.
`(sigma, eps_raw[M])` for `eps = sigma * eps_raw` non-centered
parameterization, or `(tau, u[J])` -- the `u_i` and `sigma` are
strongly multiplicatively correlated. Identity-metric NUTS mixes
poorly here (see joint_nuts_block Sec.1289 dim-vs-rhat envelope), so the
diagonal->dense escalation frequently lands on dense.

**Start diagonal; escalate to `cfg.use_dense_metric = true` (Check #18)
when diagnostics show diagonal is inadequate** -- measure, don't gate on
dimension. Example (with the Check #18 minimums applied):

```cpp
joint_nuts_block_config cfg;
cfg.name = "gamma_sigma_eps_joint";
cfg.sub_params.push_back({"gamma",   1, joint_constraint::REAL});
cfg.sub_params.push_back({"sigma",   1, joint_constraint::POSITIVE});
cfg.sub_params.push_back({"eps_raw", M, joint_constraint::REAL});
cfg.initial_cat = arma::join_cols(arma::vec{gamma_init},
                                   arma::vec{sigma_init},
                                   eps_raw_init);
cfg.log_density_grad = &joint_log_density_natural;

// Dense metric is FREQUENTLY needed for HierLM coupling at large dim, but it is a
// Check #18 ESCALATION (measured) -- START DIAGONAL and enable this only if R2/R3
// shows diagonal is inadequate; do NOT gate on dimension (there is no Check #11.7).
// When you DO enable it, Check #18 minimums apply.
const std::size_t d = 1 + 1 + M;   // joint-block unconstrained dim
// JUSTIFICATION (Check #18): documented diagonal R-hat failure on this coupling.
cfg.use_dense_metric         = true;
cfg.dense_metric_pilot_iters = std::max<std::size_t>(2000, 100 * d);
cfg.dense_metric_adapt_iters = 2000;
cfg.n_warmup_first_call      = cfg.dense_metric_pilot_iters + 1000;

impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
```

### Reference example

- `examples/LinearRegJointMixed.cpp` -- (alpha, beta, sigma) joint with
  REAL + POSITIVE per-slice constraints. (Linear mean coupling, no
  per-unit random effects -> dense_metric not required.)
- `examples/HierarchicalLM_MultivariateRE.cpp` -- (beta, z_flat, log_tau,
  log_sigma) joint with `cfg.use_dense_metric = true`. Canonical
  HierLM-with-random-effects example.

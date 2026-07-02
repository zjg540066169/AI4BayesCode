## normal_gamma_cluster_gibbs_block (diagonal Gaussian clusters)

Closed-form vectorized Gibbs leaf that samples per-cluster diagonal
Normal-Gamma parameters (mu_k, lambda_k) for k = 1..K_trunc under

    lambda_kd ~ Gamma(a_lambda_0, rate=b_lambda_0)
    mu_kd | lambda_kd ~ N(mu_0_d, 1 / (kappa_0 * lambda_kd))
    y_i ~ N(mu_{z_i}, diag(1 / lambda_{z_i}))

Posterior conjugate update per (k, d) (Bishop PRML §2.3.6 / Murphy 2007
§4). Empty clusters draw from the prior — matches Ishwaran-James 2001
§3.2. Two named outputs: cfg.mu_name (length K_trunc * d, cluster-major
row order) and cfg.lambda_name (same shape; PRECISIONS, not variances).

```cpp
normal_gamma_cluster_gibbs_block_config cfg;
cfg.name        = "cluster_params";   // block label, NOT a data key
cfg.K_trunc     = 20;
cfg.d           = 2;
cfg.N           = 200;
cfg.z_key       = "z";
cfg.y_key       = "y";                // length N * d, row-major
cfg.mu_name     = "mu";               // output key #1
cfg.lambda_name = "lambda";           // output key #2
// DATA-DRIVEN weakly-informative hypers (REQUIRED — see CRITICAL below).
// Per dimension j: mu_0_j = mean(y[,j]); b_lambda_0 = mean_j var(y[,j]).
cfg.mu_0        = col_means_of_y;     // length d, = column means of y
cfg.kappa_0     = 0.01;               // Var[mu] = 100 * E[1/lambda]
cfg.a_lambda_0  = 2.0;                // heavy-tailed, weakly informative
cfg.b_lambda_0  = mean_col_var_of_y;  // Gamma RATE; E[sigma^2] ~ data scale
cfg.initial_mu      = ...;            // data anchors (spread y rows)
cfg.initial_lambda  = ...;            // a_lambda_0 / b_lambda_0
```

> **CRITICAL — DP/BNP mixture cluster-prior hyperparameters MUST be
> data-driven, NOT fixed constants.** A fixed Normal-Gamma prior (e.g.
> `mu_0 = 0`, `b_lambda_0 = 1`) is mis-scaled for any data whose
> location/spread is not ~unit, and the truncated-SBP single-site
> categorical Gibbs then converges to a WRONG over-segmented posterior
> (verified: on variance≈270 data it recovers ≈6 spurious clusters
> instead of the true 3). **R-hat does NOT catch this** — the chains
> stably agree on the wrong posterior; only a posterior-predictive /
> occupied-cluster-count check exposes it. Always compute, in the
> wrapper constructor from `y`: `mu_0_j = mean(y[,j])`,
> `kappa_0 = 0.01`, `a_lambda_0 = 2.0`,
> `b_lambda_0 = mean_j var(y[,j])`, `alpha ~ Gamma(1,1)`. Expose a
> `(y, K_trunc, seed, keep_history)` constructor that does this as the
> DEFAULT; an explicit-hyperparameter constructor may exist for
> advanced control but must not be the documented default. Gold
> reference: `examples/DPGaussianMixture.cpp` (data-driven constructor).

**JUSTIFICATION (Check #16): Exception 1 from codegen_priors.md §2b**
(NEW Tier-B block; conjugate cluster update is the textbook Murphy
2007 §4 formula). **Check #15** parity:
`tests_autodiff/block_tests/test_normal_gamma_cluster_gibbs_block.cpp`
verifies prior-recovery (empty cluster) and posterior-recovery
(populated single cluster) within 5% on 10k draws.

**FULL covariance NIW** is shipped as `niw_cluster_gibbs_block` (NEW
2026-05-03) — see its catalogue entry below. Use the NIW block when
the data has off-diagonal correlation inside clusters; use the
diagonal Normal-Gamma block when dimensions inside a cluster are
approximately independent (smaller state, faster).

**Reference examples**: `DPGaussianMixture.cpp`, `PYGaussianMixture.cpp`,
`DPGaussianMixture_DerivedAlpha.cpp`.

## GP composition recipes (heteroscedastic / hierarchical / multi-output)

The shipped GP reference examples (`GPRegression.cpp`,
`GPClassification.cpp`, `GPTimeSeries.cpp`) cover **single-latent**
GP models. For multi-latent / multi-output / hierarchical GPs, the
agent must compose multiple primitives. The recipes below show the
composition pattern that the agent should follow when generating a
new GP wrapper. These recipes, together with the existing single-GP
examples, are sufficient for an agent that has read this skill set to
generate a multi-latent / multi-output / hierarchical GP model.

### Recipe A -- Heteroscedastic GP

**Model:** `y_i ~ Normal(f_mu(x_i), exp(f_sigma(x_i)))` with TWO
independent GP priors: `f_mu ~ GP(0, K_mu)`, `f_sigma ~ GP(0, K_sigma)`.

**Composition** (in `composite_block` `add_child` order):

1. `joint_nuts_block({log_amp_mu [real], log_ell_mu [real]})`
   -- hyperparameters for `K_mu`. Add Jacobian `+ log_amp_mu +
   log_ell_mu` to the prior contribution.
2. `joint_nuts_block({log_amp_sigma [real], log_ell_sigma [real]})`
   -- hyperparameters for `K_sigma`. Same Jacobian pattern.
3. shared_data refresher computing
   `L_mu = chol(K_mu(X, X; amp_mu, ell_mu) + jitter * I)`.
4. shared_data refresher computing `L_sigma` (independently).
5. `elliptical_slice_sampling_block` on `f_mu` reading `L_mu`. The
   log-likelihood lambda sums
   `dnorm(y_i, f_mu_i, exp(f_sigma_i), log = TRUE)` (current
   `f_sigma` from context, holds `f_mu` as the proposed slice).
6. `elliptical_slice_sampling_block` on `f_sigma` reading `L_sigma`.
   SAME log-likelihood expression but now `f_mu` is held fixed and
   `f_sigma` is the proposed slice.
7. `y_rep` stochastic refresher:
   `y_rep_i = rnorm(1, f_mu_i, exp(f_sigma_i))` using the wrapper's
   `predict_rng_`.

**Predict DAG**: `X -> L_mu, L_sigma`; `L_mu, f_mu_init -> f_mu`;
`L_sigma, f_sigma_init -> f_sigma`; `f_mu, f_sigma -> y_rep`.

**Why joint NUTS on each pair**: `(log_amp_mu, log_ell_mu)` and
`(log_amp_sigma, log_ell_sigma)` each have their own banana ridge
(see escalation ladder above). Modular NUTS on 4 separate
hyperparameters is the most common failure mode for this model.

### Recipe B -- Hierarchical GP

**Model:** Observations `y_n` indexed by group label `g(n)`; each
group `g` has its own GP `f_g ~ GP(0, K(x_g; amp_g, ell_g))`;
group-level hyperparameters share a hyperprior
`(log_amp_g, log_ell_g) ~ Normal(mu_h, Sigma_h)`.

**Composition:**

1. `joint_nuts_block({mu_h [real vector], Sigma_h_chol [real lower-triangular]})`
   -- hyperprior parameters. Use a positive constraint on the
   diagonal of `Sigma_h_chol` (REAL on log-scale, then back-
   transform; or the POSITIVE slice of `joint_nuts_block`).
2. **Per group g**:
   `joint_nuts_block({log_amp_g [real], log_ell_g [real]})`
   reading `mu_h, Sigma_h_chol` from context -- group-level GP
   hyperparameters. Prior contribution is the multivariate
   Normal(mu_h, Sigma_h) on `(log_amp_g, log_ell_g)`.
3. **Per group g**: shared_data refresher computing
   `L_g = chol(K(x_g, x_g; amp_g, ell_g) + jitter * I)`.
4. **Per group g**: `elliptical_slice_sampling_block` on `f_g`
   reading `L_g`. The log-likelihood sums
   `dnorm(y_n, f_g[i_n], sigma_obs, log = TRUE)` for `n` in group `g`.
5. `nuts_block(log_sigma_obs)` (REAL) -- observation noise.
6. y_rep stochastic refresher.

**Predict DAG**: `mu_h, Sigma_h_chol -> log_amp_g, log_ell_g for all g`;
`log_amp_g, log_ell_g -> L_g`; `L_g -> f_g`;
`f_g (n in group g) -> y_rep_n`.

**Composite size**: with G groups, this composite has
`1 + G + G + G + 1 = 3G + 2` child blocks. For large G, consider
batching the per-group blocks into a single ESS over a block-
diagonal `L = diag(L_1, ..., L_G)` to reduce composite overhead.

### Recipe C -- Multi-output / Kronecker GP

**Model:** `y_{i,j}` for `i = 1..N` inputs and `j = 1..M` outputs:
`vec(Y) ~ Normal(0, K_input (x) K_output + sigma^2 * I)` with LKJ-style
prior on the output correlation matrix.

**Composition:**

1. `joint_nuts_block({log_amp [real], log_ell [real]})` --
   input kernel hyperparameters.
2. `nuts_block(log_sigma)` (REAL) -- noise scale.
3. `nuts_block` on the LKJ-style `Lambda` correlation matrix for
   `K_output` (parameterise via Cholesky of the correlation matrix).
4. shared_data refresher computing
   `L = chol(K_input (x) K_output + sigma^2 * I)`. **Use Kronecker
   shortcuts when possible**: when the noise term is also separable
   (e.g., per-output noise), `chol(K1 (x) K2) = chol(K1) (x) chol(K2)`
   gives O(N^3 + M^3) instead of O((NM)^3); otherwise use
   eigendecomposition (Saatci 2011 PhD Sec.5).
5. `elliptical_slice_sampling_block` on `vec(F)` reading `L`.
6. y_rep refresher: `y_{i,j} = F_{i,j} + sigma * randn()`.

**Predict DAG**: hyperparams -> `L`; `L -> F`; `F + sigma -> y_rep`.

**Caveats**:
- The full `NM x NM` Cholesky in step 4 is the bottleneck. For
  large N, M (>100 each), consider Kronecker structure or low-rank
  approximations. The current `elliptical_slice_sampling_block`
  reads `L` as flat column-major; you may need to hand-roll the
  Cholesky-Kronecker structure outside the block.
- LKJ priors on `Lambda` benefit from the Cholesky-factor
  parameterization to avoid positive-definiteness rejections.

## GP composition recipes (heteroscedastic / hierarchical / multi-output)

The shipped GP reference examples (`GPRegression.cpp`,
`GPClassification.cpp`, `GPTimeSeries.cpp`) cover **single-latent**
GP models. For multi-latent / multi-output / hierarchical GPs, the
agent must compose multiple primitives. The recipes below show the
composition pattern that the agent should follow when generating a
new GP wrapper. These recipes, together with the existing single-GP
examples, are sufficient for an agent that has read this skill set to
generate a multi-latent / multi-output / hierarchical GP model.

**The rule every recipe below obeys.** A latent GP whose covariance
depends on parameters you are also sampling must be WHITENED and put in
the SAME `joint_nuts_block` as those parameters:

    f_g = L(theta_g) z_g,   z_g ~ N(0, I)
    joint_nuts_block: [{theta_g...}, {z_g, N_g, REAL}]
    density: loglik(y | f = L z) - 0.5 z'z + log p(theta_g)

Two failure modes this avoids, both of which look healthy in
diagnostics:

- Putting the hyperparameters in their own block with only their PRIOR
  in the log-density. With an `elliptical_slice_sampling_block` on the
  latent, the prior factor `log p(f | theta)` cancels in the elliptical
  acceptance ratio, so it belongs to no block at all -- and the
  hyperparameters are then drawn from their priors on every dataset
  while the latent fit and every R-hat still look fine.
- Whitening but still alternating (hyperparameter block, then ESS on
  z). With z held fixed, `f = L(theta) z` is a deterministic function
  of theta, so `p(theta | z, y)` is far sharper than `p(theta | y)` and
  the alternation random-walks. Measured on `GPClassification.cpp`'s
  dataset at 4 chains x (1000 + 2000): amplitude R-hat 1.62 (ESS 7)
  diagonal / 1.10 (ESS 28) dense, against 1.001 (ESS 1937) for one
  joint block.

Gibbs-alternating between two DIFFERENT latents (recipe A's `f_mu` and
`f_sigma`, recipe B's groups) is fine -- the pathology is between a
latent and its own covariance hyperparameters, not across latents.

Whenever the observation likelihood is Gaussian, prefer marginalising
the latent out entirely (`GPRegression.cpp`) over sampling it at all.

### Recipe A -- Heteroscedastic GP

**Model:** `y_i ~ Normal(f_mu(x_i), exp(f_sigma(x_i)))` with TWO
independent GP priors: `f_mu ~ GP(0, K_mu)`, `f_sigma ~ GP(0, K_sigma)`.

**Composition** (in `composite_block` `add_child` order):

1. `joint_nuts_block({log_amp_mu [real], log_ell_mu [real],
   z_mu [N, real]})` -- the `K_mu` hyperparameters AND the whitened
   `f_mu` latent, together. Its density is
   `sum_i dnorm(y_i, f_mu_i, exp(f_sigma_i), log = TRUE)
   - 0.5 z_mu'z_mu + log p(amp_mu) + log p(ell_mu)`, with
   `f_mu = L_mu(amp_mu, ell_mu) z_mu` rebuilt INSIDE the density at the
   proposed hyperparameters, and `f_sigma` read from context. Add the
   `+ log_amp_mu + log_ell_mu` Jacobian for the REAL log-scale slices.
2. `joint_nuts_block({log_amp_sigma, log_ell_sigma, z_sigma [N, real]})`
   -- same structure for `K_sigma`; the same likelihood expression, now
   with `f_mu` read from context and `f_sigma = L_sigma z_sigma`
   proposed.
3. shared_data refreshers publishing `f_mu = L_mu z_mu` and
   `f_sigma = L_sigma z_sigma` at block boundaries, so each block can
   read the other's current latent from context. (`L_mu`, `L_sigma`
   refreshers feed these.)
4. `y_rep` stochastic refresher:
   `y_rep_i = rnorm(1, f_mu_i, exp(f_sigma_i))` using the wrapper's
   `predict_rng_`.

**Predict DAG**: `X -> L_mu, L_sigma`; `L_mu, z_mu -> f_mu`;
`L_sigma, z_sigma -> f_sigma`; `f_mu, f_sigma -> y_rep`.

**Why each latent rides with its own hyperparameters**: see the rule at
the top of this file. Alternating between block 1 and block 2 is fine --
`f_mu` and `f_sigma` are different latents.

**Cheaper alternative for `f_mu`**: given `f_sigma`, the likelihood is
Gaussian in `f_mu` with known heteroscedastic noise `exp(f_sigma)`, so
`f_mu` admits a closed-form integral. Marginalising it -- sampling
`(amp_mu, ell_mu)` from `N(y; 0, K_mu + diag(exp(f_sigma)))` with the
Rasmussen & Williams Eq. (5.9) gradient, as in `GPRegression.cpp` --
removes N dimensions from the sampler. `f_sigma` still needs the
whitened treatment above.

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
   `joint_nuts_block({log_amp_g [real], log_ell_g [real],
   z_g [N_g, real]})` reading `mu_h, Sigma_h_chol` from context --
   group-level GP hyperparameters AND that group's whitened latent,
   together. Density: `sum_{n in g} dnorm(y_n, f_g[i_n], sigma_obs,
   log = TRUE) - 0.5 z_g'z_g + MVN(mu_h, Sigma_h) on
   (log_amp_g, log_ell_g)`, with `f_g = L_g(amp_g, ell_g) z_g` rebuilt
   inside the density.
3. **Per group g**: shared_data refresher publishing
   `f_g = L_g z_g` (and an `L_g` refresher feeding it) so downstream
   blocks and `y_rep` see the current latent.
4. `nuts_block(log_sigma_obs)` (REAL) -- observation noise. Its
   density DOES read `y` and the current `f_g`, so it is the one
   hyperparameter here that is written correctly by default.
5. y_rep stochastic refresher.

**Predict DAG**: `mu_h, Sigma_h_chol -> log_amp_g, log_ell_g for all g`;
`log_amp_g, log_ell_g -> L_g`; `L_g, z_g -> f_g`;
`f_g (n in group g) -> y_rep_n`.

**Composite size**: with G groups, this composite has
`1 + G + 1 + 1 = G + 3` child blocks (each group's hyperparameters and
latent share ONE block). Alternating across groups is fine -- they are
different latents.

### Recipe C -- Multi-output / Kronecker GP

**Model:** `y_{i,j}` for `i = 1..N` inputs and `j = 1..M` outputs:
`vec(Y) ~ Normal(0, K_input (x) K_output + sigma^2 * I)` with LKJ-style
prior on the output correlation matrix.

**Composition:** this likelihood is GAUSSIAN, so there is no latent to
sample -- `vec(F)` is integrated out and the model IS its own marginal.
Follow `GPRegression.cpp`, not the whitened route.

1. ONE `joint_nuts_block` over
   `{log_amp [real], log_ell [real], log_sigma [real],
   Lambda_chol [CHOLESKY_CORR]}` -- the input-kernel hyperparameters,
   the noise scale, and the LKJ-style output correlation, all together.
   Add the `+ log_amp + log_ell + log_sigma` Jacobians for the REAL
   log-scale slices; `joint_nuts_block` supplies the `CHOLESKY_CORR`
   slice's Jacobian itself.
2. Its density is the marginal likelihood
   `log N(vec(Y); 0, K_input (x) K_output + sigma^2 I)`, with the
   Rasmussen & Williams Sec.5.5 Eq. (5.9) gradient
   `0.5 tr((alpha alpha' - Ky^-1) dKy/dtheta)`, `alpha = Ky^-1 vec(Y)`.
   Build `Ky` INSIDE the density at the proposed hyperparameters -- a
   shared_data refresher only fires at block boundaries, never inside a
   NUTS trajectory.
3. **Use Kronecker shortcuts when possible**: when the noise term is
   also separable (e.g. per-output noise),
   `chol(K1 (x) K2) = chol(K1) (x) chol(K2)` gives `O(N^3 + M^3)`
   instead of `O((NM)^3)`; otherwise use eigendecomposition
   (Saatci 2011 PhD Sec.5), which also gives `log|Ky|` and `Ky^-1 v`
   cheaply for the isotropic-noise case.
4. `F` is recovered at PREDICT time from the closed-form posterior
   conditional, exactly as in `GPRegression.cpp` -- never sampled.
5. y_rep refresher: draw `F_star` from its Gaussian posterior, then
   `y_{i,j} = F_star_{i,j} + sigma * randn()`.

**Predict DAG**: hyperparams -> `Ky`; `Ky + Y -> F_mean`; `F_mean + sigma -> y_rep`.

**Caveats**:
- The full `NM x NM` Cholesky inside the density is the bottleneck,
  and it runs once per gradient evaluation, not once per sweep. For
  large N, M (>100 each) the Kronecker or eigendecomposition route in
  step 3 is not an optimisation, it is the difference between minutes
  and days.
- LKJ priors on `Lambda` benefit from the Cholesky-factor
  parameterization to avoid positive-definiteness rejections; that is
  what the `CHOLESKY_CORR` slice gives you.

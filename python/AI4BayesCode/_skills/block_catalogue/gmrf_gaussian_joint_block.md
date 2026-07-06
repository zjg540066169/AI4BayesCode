## gmrf_gaussian_joint_block (Knorr-Held & Rue 2002 joint (x, kappa) update)

Joint block update of a Gaussian-data GMRF field **and** its smoothing
precision, for the model

    y_i = x_i + eps_i,  eps ~ N(0, sigma2 I)      (Gaussian observation)
    x   ~ N(0, (kappa R)^{-1})                     (GMRF prior, R fixed)
    kappa ~ Gamma(a, b)

**Why (vs `gmrf_precision_block` + a sibling kappa block):** the naive Gibbs
`[x | kappa, y ; kappa | x]` mixes catastrophically because x and kappa are
strongly a-posteriori dependent (Knorr-Held & Rue 2002, *Scand. J. Statist.*
**29**: 597-614). This block removes that coupling: it updates `kappa` from
its **collapsed marginal** `p(kappa | y)` (x integrated out) via a
multiplicative log-scale random walk (Knorr-Held "scheme 3"), evaluating the
marginal-likelihood ratio from the sparse-Cholesky log-determinant of the
full-conditional precision `Q = kappa R + sigma2^{-1} I`; then it draws
`x | kappa, y` directly (Rue 2001 sparse-Cholesky sampler). Outputs `x` under
`name` and `kappa` under `name`+`"_kappa"`.

**JUSTIFICATION (Check #16):** fixed-dim continuous Gaussian latent + strongly
coupled scale hyperparameter (`system_design.md` §11.1). The collapsed joint
update is the textbook remedy for the (x, kappa) mixing pathology. Check #15
parity panel under `tests/`:
- `test_gmrf_gaussian_joint_block.cpp` — sampled `kappa` posterior mean+sd and
  `E[x|y]` matched to a **dense fine-grid** computation of the exact posterior
  `p(kappa|y) ∝ Gamma(kappa;a,b) p(y|kappa)` and
  `E[x|y] = ∫ mu(kappa) p(kappa|y) dkappa`; plus two-chain Gelman-Rubin
  R-hat < 1.01 on kappa from over-dispersed inits. Empirically E[kappa] within
  0.8%, sd within 2.3%, E[x] within 0.005, R-hat 1.00012. Ground truth is
  dense linear algebra only — zero external dependency.

**Scope (v1.2.1 shipped):** Gaussian observation `y ~ N(x, sigma2 I)` with
known `sigma2`; single smoothing precision `kappa`; proper or IGMRF `R`
(`sum_to_zero = true` for rank-deficient R). **Deferred:** non-Gaussian
(Poisson/Binomial) likelihoods via the Taylor-GMRF proposal (Rue 2001 §quadratic
approx), jointly-sampled regression coefficients in the mean, and unknown
`sigma2`.

```cpp
gmrf_gaussian_joint_block_config cfg;
cfg.name            = "x";       // x under "x", precision under "x_kappa"
cfg.n               = N;
cfg.R               = R;         // fixed sparse structure (arma::sp_mat)
cfg.y               = y;         // Gaussian data (length N)
cfg.sigma2          = 0.25;      // known observation variance
cfg.kappa_a         = 1.0;       // Gamma shape
cfg.kappa_b         = 1.0;       // Gamma rate
cfg.kappa_init      = 1.0;
cfg.log_kappa_rw_sd = 0.4;       // MH proposal width (scheme 3)
cfg.sum_to_zero     = false;     // true for IGMRF (RW1/RW2/ICAR) R
```

**Vendored kernel:** Eigen 3.4 `Eigen/SparseCholesky` (header-only, MPL-2.0),
same as `gmrf_precision_block`. Build flag `-I include/eigen`.

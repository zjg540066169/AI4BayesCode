## genbart_block (GPL-2.0+)

**Generic tree-ensemble primitive** for Bayesian BART with ANY
plug-in likelihood. Implements Linero 2022 (arXiv:2202.09924)
reversible-jump MCMC with Laplace-approximated BIRTH / DEATH / CHANGE
tree proposals; user supplies only `log_f(y, lambda, obs_i)` and
(optionally) `score` / `obs_info` on a subclass of
`genbart::likelihood`. Tree-ensemble output is `r(x)` on the
LINEAR-PREDICTOR scale -- interpretation depends on the attached
likelihood.

`current()` returns a length-N arma::vec of r(X_i).

### When to use -- PREFER `bart_block` first

`genbart_block` is the GENERIC engine: Linero 2022 reversible-jump MCMC with
Laplace-approximated leaf proposals, **substantially slower** than `bart_block`'s
conjugate Gaussian-leaf backfitting (`score` / `obs_info` are called n times per
tree update). Do NOT reach for it by default.

- **Plain Gaussian mean, constant noise -> `bart_block`** (the `BartNoise`
  example). Using `genbart_block + normal_lik` for standard homoscedastic
  Gaussian regression is a common mistake: identical model, far slower sampler.
  `normal_lik` exists for composition, not as the Gaussian default.
- **Composite / varying-coefficient / additive-ensemble BART** (VC-BART, where
  each coefficient function is its own BART) is ALSO a `bart_block` job --
  compose one `bart_block` per ensemble via backfitting, using `weights_key` for
  the per-observation covariate scaling. See the `bart_block` card, "Composite /
  varying-coefficient BART via BACKFITTING". Do NOT write a custom VC likelihood.
- **Use `genbart_block` ONLY when `bart_block` cannot express the model** -- a
  genuinely NON-Gaussian response (Logistic / Poisson / NB / Heteroscedastic /
  AFT / Beta / Gamma_shape / Beta-Binomial / custom).

### Shipped likelihoods (`genbart::lik::*`)

| Likelihood class | Response | Nuisance param(s) | Reference example |
|---|---|---|---|
| `normal_lik(sigma_init, nu, lambda)`      | y ∈ ℝ, Normal                 | sigma              | (Linero 2022 §4.1 regression; use BartNoise for the tuned CRAN BART R package alternative) |
| `logistic_lik()`                          | y ∈ {0,1}, Bernoulli-sigmoid  | —                  | `examples/GBartLogistic.cpp` |
| `poisson_lik()`                           | y ∈ ℕ, Poisson-log-link       | —                  | `examples/GBartPoisson.cpp`  |
| `negative_binomial_lik(kappa_init)`       | y ∈ ℕ, NB(mu, kappa)           | kappa (RW-MH)      | (compose from `GBartPoisson.cpp` + this likelihood header) |
| `heteroscedastic_lik(phi_init, a0, b0)`    | y ∈ ℝ, mean=variance=exp(r)   | phi (Gamma Gibbs)  | `examples/GBartHeteroscedastic.cpp` |
| `aft_log_logistic_lik(sigma_init, ...)`   | y_log = log time + censoring  | sigma (RW-MH)      | (compose from `GBartPoisson.cpp` + this likelihood header) |
| `aft_generalized_gamma_lik(...)`           | log time + censoring          | 2 nuisances        | (planned) |
| `gamma_shape_lik(...)`                    | y > 0, Gamma with shape r(x)   | rate (nuisance)    | (planned) |
| `beta_lik(phi_init, ...)`                 | y ∈ (0,1), Beta               | phi (RW-MH)        | (planned) |
| `beta_binomial_lik(...)`                  | overdispersed counts          | rho (RW-MH)        | (planned) |

Beyond these, users can write custom likelihoods: subclass
`genbart::likelihood` and override at minimum `log_f`; the base class
supplies finite-difference `score` / `obs_info` defaults. Analytic
overrides are strongly recommended for speed since Laplace leaf
proposals call them n times per tree update.

### Configuration

```cpp
genbart_block_config cfg;
cfg.name        = "r";           // shared_data key storing r(X)
cfg.x_train     = X;
cfg.y_init      = y;
cfg.offset_init = Rcpp::NumericVector(0);  // empty = zeros
cfg.lik         = likelihood_.get();       // non-owning pointer
cfg.ntrees      = 50;                       // Linero 2022: 50 for softer
                                            // likelihoods (logistic/NB/
                                            // beta/AFT); 200 for well-
                                            // identified (normal/Poisson/
                                            // heteroscedastic)
cfg.y_key       = "";            // empty = fixed training data; non-
                                 // empty = refresh Y from shared_data
                                 // each sweep (nested MCMC use case)
cfg.offset_key  = "";            // similarly for offset
// cfg.hypers default to Linero 2022 §3.3 (adaptive half-Cauchy on
// sigma_mu with c = 1/sqrt(ntrees); DART off).
```

### Nested MCMC

`GBart*` wrappers' `set_current(Rcpp::List)` accepts any subset of
`{X, y, offset}` (where applicable) -- see `examples/GBartPoisson.cpp`
for the canonical pattern. Outer samplers push updated working
responses / imputed covariates between sweeps:

```r
for (iter in seq_len(n_iter)) {
    # ... outer sampler produces y_new / X_new ...
    m$set_current(list(X = X_new, y = y_new))
    m$step(1L)
}
```

### DART sparsity

`genbart::rjmcmc_hypers::dart_active = true` enables Linero 2018's
sparse-Dirichlet prior on split-variable probabilities. Same "ask the
user; expose constructor args only on opt-in" recipe as `bart_block`
-- see `skills/codegen.md` follow-up questions. Recommended defaults
when opted-in: `dart_active = TRUE`, `dart_const_theta = FALSE`.

### RNG

genBART uses R's global RNG via `genbart::arn` (same pattern as the
legacy `bart_block`). Reproducibility
requires `set.seed()` in R BEFORE the MCMC loop; the construction-time
seed on `GBart*` wrappers only affects the mutable `predict_rng_`
stream used by stochastic refreshers.

### Performance

End-to-end 4-chain diagnostics on `GBartPoisson(N=200, p=3, ntrees=200)`:
per-sweep wall time ~0.01 s; 4 × (4000 burnin + 10000 keep) sequential
total ~441 s; max Rhat = 1.004; all 200 per-obs R-hat < 1.05 and
ESS_bulk > 400.

Any file including `bart_block.hpp` or `genbart_block.hpp` inherits
GPL-3.0-or-later (the project license; the vendored BART / genBART
kernels are GPL-2.0-or-later upstream, which is GPL-3-compatible).

## pg_logistic_block (T12, Polson-Scott-Windle 2013 PG augmentation)

Bayesian logistic regression via Polya-Gamma data augmentation.
Exact Gibbs: alternates PG(1, X_i' beta) auxiliary draws (library-
internal truncated series, K=128 terms -> 1e-8 relative tail bias on
PG mean) with Gaussian beta | omega update. 10-100x faster than NUTS-on-
logistic for p < ~1000.

WARNING **Hard scope limitation: LINEAR LOGISTIC ONLY -- NOT LOGISTIC BART.**
PG augmentation's exact-Gibbs advantage depends on the linear
predictor `X_i' beta` being a parametric linear combination. Substituting
a BART mean function f(X_i) breaks both (i) the Gaussian beta | omega
conjugacy and (ii) BART's tree-location identifiability (the PG-
augmented pseudo-response kappa = y - 0.5 does not anchor BART's
Gaussian-observation tree scale). For logistic BART, use
**`genbart_block` + `genbart::lik::logistic_lik`** (see
`examples/GBartLogistic.cpp`) -- the RJMCMC tree kernel handles the
non-conjugate sigmoid likelihood directly via Laplace leaf
proposals, with no augmentation required. Never combine `bart_block` +
`pg_logistic_block`.

WARNING **Use this block ONLY with a PROPER, INFORMATIVE Gaussian `prior_cov`.**
The `beta | omega` draw is `N(., (X'Omega X + prior_cov^{-1})^{-1})`; the prior
precision `prior_cov^{-1}` is the stabilizer. Route to a `joint_nuts_block` on
the natural-scale logistic log-density instead when the coefficient prior is:
(a) **flat / improper** (incl. a fake-flat `prior_cov = ~1e8 * I`) -- on a
(quasi-)separated dataset `omega -> 0`, `X'Omega X -> 0`, `beta`'s conditional
variance -> inf, and `beta` runs off to `+/-1e9` (absorbing state; R-hat
explodes, coverage collapses); (b) **weakly-informative** -- the floor is thin,
prefer NUTS for robustness; (c) **non-Gaussian** -- this block takes only a fixed Gaussian `prior_cov`. (A
Gaussian scale-mixture prior -- Student-t / Laplace / horseshoe / Cauchy -- can
keep PG via an added scale latent, but this system routes such shrinkage priors
to NUTS, `joint_nuts_failure.md` Form G; a non-scale-mixture prior needs NUTS.)
NUTS needs no precision floor and is stable in all three.

**JUSTIFICATION (Check #16): Exception 1** (discrete/augmented
measure; NUTS cannot target PG latent directly). Library-blessed
block: user writes no PG sampling code. Check #15 parity test at
`tests_autodiff/test_pg_logistic_block.cpp` verifies PG(1, z) mean
against analytical (1/(2z)) tanh(z/2) + end-to-end beta recovery
on synthetic logistic regression (p=5, N=500).

Reference template: `examples/LogisticRegression.cpp`.

```cpp
pg_logistic_block_config cfg;
cfg.name = "beta";
cfg.p = p;
cfg.y_key = "y";   // N-vector of 0/1
cfg.X_key = "X";   // N*p column-major flat
cfg.prior_mean = arma::vec(p, arma::fill::zeros);
cfg.prior_cov  = prior_sd*prior_sd * arma::eye<arma::mat>(p, p);
cfg.initial_beta = arma::vec(p, arma::fill::zeros);
```

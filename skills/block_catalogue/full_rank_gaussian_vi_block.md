## full_rank_gaussian_vi_block (v1 SHIPPED 2026-05-26)

```cpp
full_rank_gaussian_vi_block_config cfg;
cfg.name        = "<shared_data key>";
cfg.initial_unc = arma::vec{...};                              // K-dim mean
cfg.initial_L   = arma::mat(K, K, arma::fill::eye) * 0.1;      // L init (lower-triangular)
cfg.constrain   = constraints::positive::constrain;            // omit for real
cfg.unconstrain = constraints::positive::unconstrain;
cfg.log_density_grad = /* same lambda as nuts_block / mean-field VI */;
cfg.dependencies     = {"y", "X", ...};
cfg.optimizer        = vi_optimizer::raabbvi_config{};
auto blk = std::make_shared<full_rank_gaussian_vi_block>(cfg);
impl_->add_block(blk);
```

**What it does**: maintains q(eta) = N(eta; mu, LL^T) on the unconstrained
scale, with L lower-triangular Cholesky factor -- captures FULL posterior
correlations among the K coordinates. Parameter count = K + K(K+1)/2,
quadratic in K. Use when mean-field's independence assumption fails:
- Regression coefficients with collinear predictors
- BNN output-layer weights conditional on hidden representations
- Hierarchical scale x parameter funnel where MF underestimates the
  marginal variance
- Any target where the posterior has off-diagonal covariance you want
  to capture

**Empirical comparison** (tests/test_vi_full_rank_correlated.cpp, 2D
rho=0.95 correlated Gaussian, mean-field-vs-full-rank apples-to-apples):

| Quantity | Truth | Mean-field VI | Full-rank VI |
|---|---:|---:|---:|
| marginal sd | 1.00 | 0.31 (conditional, biased) | **0.99 (matches truth)** |
| correlation rho | 0.95 | 0 (by construction) | **0.948** |
| PSIS-k-hat | -- | 0.80 (FAIL -- q != p) | **-0.14 (q ~= p exact)** |

The 0.31 mean-field value is exactly sqrt(1-rho^2) = the CONDITIONAL sd,
which is what mean-field collapses to -- textbook Bishop Sec.10.1.2.

**Caps**: auto-suggest K <= 50; warn 50 < K <= 200; reject K > 500
(too expensive -- switch back to mean-field or restructure).

**Public interface**: identical to `mean_field_gaussian_vi_block`,
plus `get_L() -> const arma::mat&` (current lower-triangular Cholesky
factor).

**Constraints + validator obligations**: same as
`mean_field_gaussian_vi_block`.

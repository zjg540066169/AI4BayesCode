## stick_breaking_block (truncated SBP -- DP / PY / custom)

Closed-form Gibbs leaf that samples a length-K_trunc simplex pi via
truncated stick-breaking (Ishwaran & James 2001). User supplies the
per-stick Beta parameters `a_fn(k, counts, ctx)` and `b_fn(k, counts, ctx)`
as closures; the block knows nothing about which stochastic process
it implements. This makes a single block primitive cover

- **Dirichlet Process (Sethuraman 1994)**: a_k = 1 + n_k,
  b_k = alpha + sum_{j>k} n_j
- **Pitman-Yor (Pitman & Yor 1997)**: a_k = 1 + n_k - discount,
  b_k = alpha + (k+1) * discount + sum_{j>k} n_j
- **Hierarchical / kernel SBP**: any user-defined a_fn, b_fn that
  reads other ctx variables.

State: pi length K_trunc (last stick V_{K_trunc-1} = 1 forced per
Ishwaran-James truncation). Optionally also exposes V_1..V_{K_trunc-1}
under cfg.v_name (set to a non-empty string).

```cpp
stick_breaking_block_config cfg;
cfg.name        = "pi";
cfg.K_trunc     = 20;
cfg.counts_key  = "cluster_counts";   // typically a refresher of z
cfg.v_name      = "stick_V";          // optional output of stick fractions
cfg.a_fn = [](std::size_t k, const arma::vec& counts,
              const block_context& /*ctx*/) -> double {
    return 1.0 + counts[k];           // DP
};
cfg.b_fn = [](std::size_t k, const arma::vec& counts,
              const block_context& ctx) -> double {
    const double a = ctx.at("alpha")[0];
    double tail = 0.0;
    for (std::size_t j = k + 1; j < counts.n_elem; ++j) tail += counts[j];
    return a + tail;
};
cfg.initial_pi = arma::vec(20, arma::fill::value(1.0 / 20));
```

**JUSTIFICATION (Check #16): Exception 1** (per-stick Beta conditional
on a NEW Tier-B block -- Ishwaran-James textbook scheme).
**Check #15** parity: `tests_autodiff/block_tests/test_stick_breaking_block.cpp`
verifies analytic E[pi_k] under both empty-counts (GEM(alpha) regime)
and populated-counts DP regimes within 5% on 20k draws.

**Reference examples**: `DPGaussianMixture.cpp`, `PYGaussianMixture.cpp`,
`DPGaussianMixture_DerivedAlpha.cpp`.

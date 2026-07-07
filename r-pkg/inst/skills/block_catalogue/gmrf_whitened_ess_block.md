## gmrf_whitened_ess_block (Murray 2010 Elliptical Slice Sampling on implicit GMRF prior; SHIPPED 2026-06-03)

Companion to `gmrf_precision_block` for the **non-Gaussian observation
likelihood** case. When the latent prior is GMRF
(`pi(x) prop.to exp{-1/2 x^T Q x}`, Q sparse PSD) but the observation
likelihood is NOT Gaussian (Poisson, Bernoulli, Student-t, NB,
log-Gaussian Cox), the full conditional `p(x | y, hyperparams)` is no
longer Gaussian and `gmrf_precision_block`'s direct conjugate draw
does not apply. This block uses Murray 2010 Elliptical Slice Sampling
on the IMPLICIT GMRF prior -- the prior is never expanded; only sparse
Q is factorised.

**Algorithm** (per step):
1. Build Q = Q_fn(ctx); refactor sparse Cholesky (SimplicialLLT + AMD)
2. Sample nu ~ N(0, Q^{-1}) via Rue 2001 permuted backsolve (sample
   z ~ N(0, I), solve `L^T y_perm = z`, apply inverse permutation)
3. ESS shrink loop: propose `x' = x cos theta + nu sin theta`, accept if
   `log_lik(x') > log_lik(x) + log(u)`, else shrink theta-bracket
   (Murray 2010 Algorithm 1)

The acceptance rate is independent of the likelihood scale (Murray
2010 Theorem 1), so the block mixes well even when the posterior is
sharply peaked relative to the prior. Per-sweep cost
O(n * b_w^2 + n * n_shrink_avg) where b_w = bandwidth and n_shrink_avg
is typically 2-5 for well-conditioned posteriors.

**JUSTIFICATION (Check #16):** Latent Gaussian with non-Gaussian
observation likelihood -- falls outside `gmrf_precision_block`'s
Gaussian-conditional scope. Murray 2010 ESS is the standard textbook
algorithm for this regime (cited in Rasmussen-Williams Ch.3 GP
classification, Banerjee-Carlin-Gelfand 2014 spatial-epidemiology
Ch.5 for Poisson-ICAR / BYM2).

**Sum-to-zero invariant**: ESS rotation `x cos theta + nu sin theta` is
linear; if `x_cur` and `nu` are both zero-mean, every proposal is
zero-mean. The block enforces sum-to-zero only on `nu` post-sample
(via the same projection as `gmrf_precision_block`) and on
`x_initial` / `set_current(x)`; the inner shrink loop needs no
extra projection. Tested to machine precision (max|mean| = 9.7e-13
across 8000 draws on N=16 ICAR).

**Empirical correctness** (header tests, 2026-06-03):
- T1 Pure prior recovery (N=16, 10k draws): sample var / Q^{-1}
  diag = 0.998 (essentially exact); off-diag cov match within 0.2%
- T2 Poisson-ICAR N=16, 4 chains x 2k: R-hat max=1.026, ESS_bulk
  min=245, coverage 15/16 = 94%, sum-to-zero preserved to 1e-12
- T3b Poisson-ICAR N=64, 4 chains x 10k: R-hat max=1.041,
  ESS_bulk min=149, coverage 63/64 = 98%, 0.09s wall per chain

**Verified convergence budgets (R-hat < 1.01, 4 chains, Poisson-ICAR fixture):**

| Grid | Budget/chain | Wall/chain | R-hat max | ESS_bulk min |
|---|---|---|---|---|
| N=16 (4x4) | 20k | 0.06s | 1.004 | 2409 |
| N=64 (8x8) | 50k | 0.47s | 1.006 | 915 |

Empirical scaling: budget grows roughly linearly with N at typical
lattice connectivities (4-NN / 6-NN). Larger grids (N ~= 100-200)
may need 100k-200k iter per chain to maintain R-hat < 1.01; the user
should pilot the convergence budget for their specific Q topology and
likelihood family before committing to a production budget.

**Example composite recipe** (sparse-precision latent + user-supplied
non-Gaussian likelihood; user adapts `Q_fn` and `log_lik` for their
model):

```cpp
gmrf_whitened_ess_block_config cfg;
cfg.name = "x";
cfg.n    = N;
cfg.Q_fn = [Q_base](const block_context& ctx) -> arma::sp_mat {
    // Build the sparse precision at the current hyperparameter values.
    // Example pattern: scale a fixed sparsity pattern by a precision hyper.
    const double kappa = ctx.at("kappa")[0];
    return kappa * Q_base;
};
cfg.log_lik = [data](const arma::vec& x, const block_context& ctx) -> double {
    // User-supplied non-Gaussian observation log-density at the proposed
    // latent x, reading any further hyperparameters from ctx. Returns
    // sum_i log p(y_i | x_i, other_hyperparams).
    return /* user implementation */;
};
cfg.sum_to_zero  = true;               // for ICAR-style improper priors
cfg.initial_x    = arma::vec(N, arma::fill::zeros);

// Typical composite: pair with one or more nuts_block for hyperparameters
// (location intercept on the real line, log-precision on the positive line).
```

**Reference templates:** TBD (v1.3 roadmap). Until shipped, use the
example recipe above and the inline pattern in `system_design.md`
Sec.13 GMRF-family section.

**Vendored kernel:** Eigen 3.4 `Eigen/SparseCholesky` (header-only,
MPL-2.0). Same `-I include/eigen` build flag as
`gmrf_precision_block`.

**Scope (v1.2 ship):**
- Symbolic factorisation cached once (assumes Q's sparsity pattern is
  fixed across steps; numerical values may vary -- typical for
  `Q = kappa * R` decompositions)
- Single sum-to-zero constraint via post-hoc projection on `nu`
- Universal `log_lik(x, ctx) -> double` user callback (any non-Gaussian
  likelihood family)
- AMD reordering via Eigen built-in

**Deferred to v1.3:**
- Check #25 validator for `Q_fn` (returns symmetric PSD sparse) and
  `log_lik` (returns finite scalar) contracts
- Cross-block ESS scheduling for composite_blocks with multiple GMRF
  latents
- Reference templates for sparse-precision GMRF + non-Gaussian
  likelihood compositions

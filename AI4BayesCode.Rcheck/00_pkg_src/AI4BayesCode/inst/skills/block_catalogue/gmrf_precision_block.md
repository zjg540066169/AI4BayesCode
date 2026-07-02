## gmrf_precision_block (Rue 2001 sparse-Cholesky direct sampler)

Direct sampler for Gaussian Markov Random Fields specified by a sparse
precision matrix Q (Rue 2001 JRSSB *"Fast sampling of Gaussian Markov
random fields"*). Target distribution in canonical form
(Rue eq. 4 / §3.1.2):

  pi(x) ∝ exp{ -½ x^T Q x + b^T x },   x ~ N(Q^{-1} b, Q^{-1})

Algorithm: P Q P^T = L L^T (AMD reordering + sparse Cholesky via Eigen
SimplicialLLT) → solve `L^T y_perm = z`, `z ~ N(0, I)` → apply inverse
permutation. Mean shift `mu = Q^{-1} b` via the cached factorisation.
Per-sweep cost O(n · b_w^2) for the numerical re-factorisation
(b_w = bandwidth ≈ O(√n) for typical 2D lattice GMRFs); the symbolic
factorisation + AMD ordering are computed once and amortised.

**JUSTIFICATION (Check #16):** Fixed-dim continuous Gaussian with
sparse precision (`system_design.md` §11.1 class 1, specialised
efficiency path). Direct conjugate draw — alternative to NUTS on
high-dim Gaussian latents in hierarchical models (spatial smoothing,
RW1 / RW2 splines, ICAR / BYM2 disease mapping, lattice GP
approximations). `gmrf_precision_block` is the library-blessed
sparse-Cholesky direct sampler. Check #15 parity panel under
`tests/`:
- `test_gmrf_precision_block.cpp` — 5 sub-tests: diagonal Q sanity,
  AR(1) n=5 Cov vs dense Q^{-1}, canonical b ≠ 0 mean shift, IGMRF
  1D random walk sum-to-zero (exact constraint + projected-Cov
  match), two-init R-hat across 4 chains on n=50.

Ground truth is closed-form / dense-inverse only — zero external
dependency.

**Reference templates:**
- `examples/GMRFPrior.cpp` — pure-prior 2D ICAR demo on a
  rectangular lattice; sum-to-zero enforced; kappa settable via
  `set_current("kappa")` for downstream hierarchical composition.
- `examples/ICARSpatialGMRF.cpp` — full Bayesian ICAR with Gaussian
  observations (hybrid composite: gmrf_precision_block for phi +
  separate nuts_block instances for Intercept / tau / sigma).

**Scope (v1.2 ship):**
- Symbolic factorisation cached once (assumes Q's sparsity pattern
  is fixed across steps; numerical values may vary — typical for
  `Q = kappa · R` decompositions)
- Single sum-to-zero constraint via post-hoc projection on a
  ridge-regularised Q (simplified Rue §3.1.3, the spam / R-INLA
  approach)
- AMD reordering via Eigen built-in

**Deferred to v1.2.1:**
- Arbitrary linear constraints A x = b (Rue §3.1.3 exact kriging)
- METIS reordering for very large graphs
- Conditional sampling x_A | x_B (Rue §3.1.1)
- Knorr-Held & Rue (2002) joint block update of hyperparam + x
- Divide-and-conquer for very large GMRFs (Rue §4)

```cpp
gmrf_precision_block_config cfg;
cfg.name = "x";
cfg.n    = N;
cfg.Q_fn = [R](const block_context& ctx) -> arma::sp_mat {
    const double kappa = ctx.at("kappa")[0];
    return kappa * R;                  // R = fixed sparse Laplacian
};
// Optional canonical "b":
cfg.b_fn = [y, sigma2](const block_context& ctx) -> arma::vec {
    const double Intercept = ctx.at("Intercept")[0];
    return (y - Intercept) / sigma2;   // canonical form b
};
cfg.sum_to_zero  = true;               // for ICAR / IGMRF
cfg.ridge_epsilon = 0.0;               // auto-bumped to 1e-8 when sum_to_zero
cfg.initial_x    = arma::vec(N, arma::fill::zeros);  // optional
```

**Vendored kernel:** Eigen 3.4 `Eigen/SparseCholesky` (header-only,
MPL-2.0). Build flag: `-I include/eigen` (added to `tests/Makefile`
CPPFLAGS and `examples/Makevars` PKG_CPPFLAGS as part of the v1.2
Block 2 ship).

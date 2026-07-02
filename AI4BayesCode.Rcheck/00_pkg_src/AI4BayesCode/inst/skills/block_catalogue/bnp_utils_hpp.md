## bnp_utils.hpp (Bayesian-nonparametric utility helpers)

Header-only namespace `AI4BayesCode::bnp` with five functions used
by `stick_breaking_block` and any user-written `log_probs_fn` /
refresher / Neal-Algorithm-2/8 composition:

- `counts_from_z(z, K)` — histogram of 1-indexed cluster assignments.
- `crp_log_prior(k, n_minus_i, alpha, N_minus_i)` — CRP allocation
  log-prior for cluster k (or NEW slot at k = K_current).
- `py_log_prior(k, n_minus_i, K_current, alpha, discount, N_minus_i)`
  — Pitman-Yor variant.
- `crp_sample_new_assignment(counts, alpha, rng)` — draw a 0-indexed
  cluster id under CRP weights (returns K to signal NEW).
- `py_sample_new_assignment(counts, alpha, discount, K_current, rng)`
  — PY variant.

These are READ-ONLY pure functions; no class, no state. Useful
when a user writes a refresher that needs to draw a CRP assignment
at predict time, or when implementing a Neal-Alg-2 categorical_gibbs
log_probs_fn.

**Check #15** parity: `tests_autodiff/block_tests/test_bnp_utils.cpp`
verifies all five functions within Check #15 tolerance (analytic +
empirical).

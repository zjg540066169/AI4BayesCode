## bnp_utils.hpp (Bayesian-nonparametric utility helpers)

Header-only namespace `AI4BayesCode::bnp` with six functions used
by `stick_breaking_block` and any user-written `log_probs_fn` /
refresher / Neal-Algorithm-2/8 composition:

- `counts_from_z(z, K)` -- histogram of 1-indexed cluster assignments.
- `crp_log_prior(k, n_minus_i, alpha, N_minus_i)` -- CRP allocation
  log-prior for cluster k (or NEW slot at k = K_current).
- `py_log_prior(k, n_minus_i, K_current, alpha, discount, N_minus_i)`
  -- Pitman-Yor variant.
- `crp_sample_new_assignment(counts, alpha, rng)` -- draw a 0-indexed
  cluster id under CRP weights (returns K to signal NEW).
- `py_sample_new_assignment(counts, alpha, discount, K_current, rng)`
  -- PY variant.
- `sample_alpha_escobar_west(k, n, a, b, alpha_current, rng)` -- one
  Escobar-West (1995) auxiliary-variable draw of the DP concentration
  `alpha` from a Gamma(a, b) prior, given k occupied clusters out of n
  observations. Exact for the DP: it conditions on (k, n) via the
  auxiliary `eta ~ Beta(alpha + 1, n)`, so the truncated-SBP tail sticks
  never enter. Use this ONLY when the model really is a plain DP with a
  Gamma prior on alpha; for a swappable prior, or anything Pitman-Yor,
  put `alpha` on a `nuts_block` over the Antoniak (k, n) marginal
  instead (see `DPGaussianMixture.cpp`).

These are READ-ONLY pure functions; no class, no state. Useful
when a user writes a refresher that needs to draw a CRP assignment
at predict time, or when implementing a Neal-Alg-2 categorical_gibbs
log_probs_fn.

**Check #15** parity: NO dedicated parity test ships for this header. The six
functions are exercised indirectly through the BNP examples' recovery checks.
When Check #15 fires on a model using these helpers, write the parity test as
part of that generation rather than citing one that does not exist.

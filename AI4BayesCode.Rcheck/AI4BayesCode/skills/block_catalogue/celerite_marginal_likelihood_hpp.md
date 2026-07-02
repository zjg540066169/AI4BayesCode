## celerite_marginal_likelihood.hpp (pure-function helper)

Stateless helper that computes log p(y | celerite kernel params) for a
1-D time series, via a temporary `celerite::solver::CholeskySolver`.
Used by slice-sampler lambdas in `GPTimeSeries.cpp` to evaluate the
marginal at PROPOSED hyperparameter values without mutating any
block's internal solver state.

Two signatures:
- `celerite_log_marginal(t, y, a_real, c_real, a_comp, b_comp, c_comp, d_comp, obs_diag, jitter=1e-10)` -- full kernel (real + quasi-periodic terms).
- `celerite_log_marginal_real(t, y, a_real, c_real, obs_diag, jitter)` -- convenience wrapper for real-exponential terms only (OU / Matern 1/2 / single-scale RW).

On numerical failure (non-PD kernel, NaN input, etc.) returns
`-std::numeric_limits<double>::infinity()` rather than throwing.

Per-call cost is O(N) dominated by the celerite `compute()`. Calling
this helper from a slice-sampler lambda (~3-10 evaluations per step)
is feasible for N up to ~10,000.

## celerite_gp_block (1-D time-series O(N) GP)

**1-D time-series Gaussian Process** with O(N) Cholesky via the
semi-separable algorithm of Foreman-Mackey et al. 2017. Wraps
`celerite::solver::CholeskySolver<double>` from the vendored celerite
core at `AI4BayesCode/celerite/`.

Kernel class: sums of real-exponential + quasi-periodic terms
(SHO-style). NOT a general multi-dim GP; for that use
`elliptical_slice_sampling_block` + libgp_kernels.

**Reference example**: `examples/GPTimeSeries.cpp` (slice-sampling, single
real-exponential / OU / Matern 1/2 kernel, with
`univariate_slice_sampling_block` driving amp/tau/sigma on the
celerite marginal likelihood).

**Current role**: in GPTimeSeries.cpp the celerite block is placed LAST
in the composite's Gibbs order so that its internal
`CholeskySolver` reflects the post-sweep state. Its `predict_at`
does not call `predict_mean_var(t_new)`, which gives marginal means
and variances only, in O(N_new * N): for its single exponential (OU)
kernel, a Kalman filter, RTS smoother and backward sampling give the
marginal means and sds and one joint draw of f per stored draw, in
O(N + N_new). Slice blocks do
NOT read `celerite_logp` from ctx (they compute the marginal at
proposed values via `celerite_marginal_likelihood.hpp`); the block's
cached `celerite_logp` is kept for downstream inspection only.

Use for:
- Astronomical time-series (Foreman-Mackey et al. 2017 original)
- Financial / climate time-series
- Spline-like 1-D extrapolation for N > 2000 where dense GP is too slow

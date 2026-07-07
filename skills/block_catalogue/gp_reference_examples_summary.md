## GP reference examples summary

| Example | Architecture | Hyperparam sampler | Latent sampler | Likelihood |
|---|---|---|---|---|
| `GPRegression.cpp` | **MARGINAL** (f integrated out) | `joint_nuts_block({amp, ell, sigma})` (POSITIVE x 3) | n/a (f recovered at predict time) | Gaussian |
| `GPClassification.cpp` | **WHITENED** (f = L * z, z ~ N(0, I)) | `joint_nuts_block({amp, ell})` (POSITIVE x 2) -- log-density includes Bernoulli-logit lik at proposed (amp, ell) | `elliptical_slice_sampling_block` on **z** (prior `L_identity`) | Bernoulli-logit |
| `GPTimeSeries.cpp` | celerite (1-D, O(N)) | `univariate_slice_sampling_block` x 3 (amp, tau, sigma) | marginalized out (semi-separable solver) | Gaussian |

**The architectural rule for GPs**:

| Observation likelihood | f admits closed-form integral? | Architecture |
|---|---|---|
| Gaussian | YES -- `p(y\|theta) = N(y \| 0, K + sigma^2I)` | **Marginal**: sample only hyperparameters; no ESS, no explicit f |
| Bernoulli, Poisson, Student-t, Negative-Binomial, ... | NO | **Whitened ESS**: sample `z ~ N(0, I)`, recover `f = L(amp, ell) * z` |

This is the same architectural choice every reference GP library makes
(Stan, libgp, GaussianProcesses, GPy, GPflow): marginalize f when you
can; sample a whitened latent when you must.

The whitened parameterization (Murray & Adams 2010) is preferred over
centered ESS because the centered conditional `p(amp, ell | f)` has a
prior factor `p(f | amp, ell)` that pulls `(amp, ell)` toward small
values when f is small, which causes the chain to collapse to
`(amp ~= 0, f ~= 0)` for weakly informative data. Whitening puts the
prior on z (independent of `(amp, ell)`) so the hyperparameter
conditional sees the data only via the Bernoulli-logit likelihood
`Bernoulli(y | sigmoid(L(amp, ell) * z))`.

`GPRegression.cpp` follows the marginal route; `GPClassification.cpp`
uses whitened ESS because Bernoulli-logit has no closed-form latent
integral.

Pick by structure:
- Multi-D continuous covariates + Gaussian response: GPRegression (marginal)
- Multi-D + binary response: GPClassification (whitened ESS + joint NUTS on amp, ell)
- 1-D time-series, N moderate-to-large: GPTimeSeries (celerite O(N) advantage)
- 1-D time-series, N small: GPRegression also works (O(N^3) is fine for N <= 500)

**When the shipped examples DON'T fit the prompt:**
- Heteroscedastic / hierarchical / multi-output / Kronecker GP ->
  see "**GP composition recipes**" below.
- Non-Gaussian likelihood NOT covered by `GPClassification.cpp`
  (Poisson, Student-t, ...) -> adapt `GPClassification.cpp` by swapping
  the `log_lik` callback in its `elliptical_slice_sampling_block`.

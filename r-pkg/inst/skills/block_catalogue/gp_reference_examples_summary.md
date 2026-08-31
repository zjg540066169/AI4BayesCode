## GP reference examples summary

| Example | Architecture | Hyperparam sampler | Latent sampler | Likelihood |
|---|---|---|---|---|
| `GPRegression.cpp` | **MARGINAL** (f integrated out) | `joint_nuts_block({amp, ell, sigma})` (POSITIVE x 3) | n/a (f recovered at predict time) | Gaussian |
| `GPClassification.cpp` | **WHITENED** (f = L * z, z ~ N(0, I)) | ONE `joint_nuts_block` over **(amp, ell, z)** -- `[{amp,1,POSITIVE},{ell,1,POSITIVE},{z,N,REAL}]`, diagonal metric | same block (no separate latent sampler) | Bernoulli-logit |
| `GPTimeSeries.cpp` | celerite (1-D, O(N)) | `univariate_slice_sampling_block` x 3 (amp, tau, sigma) | marginalized out (semi-separable solver) | Gaussian |

**The architectural rule for GPs**:

| Observation likelihood | f admits closed-form integral? | Architecture |
|---|---|---|
| Gaussian | YES -- `p(y\|theta) = N(y \| 0, K + sigma^2I)` | **Marginal**: sample only hyperparameters; no ESS, no explicit f |
| Bernoulli, Poisson, Student-t, Negative-Binomial, ... | NO | **Whitened, one block**: sample `(amp, ell, z)` jointly with `z ~ N(0, I)`, recovering `f = L(amp, ell) * z` inside the density |

This is the same architectural choice every reference GP library makes
(Stan, libgp, GaussianProcesses, GPy, GPflow): marginalize f when you
can; sample a whitened latent when you must.

The whitened parameterization (Murray & Adams 2010) is preferred over
the centered one because the centered conditional `p(amp, ell | f)` has
a prior factor `p(f | amp, ell)` that pulls `(amp, ell)` toward small
values when f is small, which causes the chain to collapse to
`(amp ~= 0, f ~= 0)` for weakly informative data. Whitening puts the
prior on z (independent of `(amp, ell)`) so the hyperparameters see the
data only via the likelihood `Bernoulli(y | sigmoid(L(amp, ell) * z))`.

**Whitening is necessary but not sufficient -- do not then Gibbs.**
Sampling z in an `elliptical_slice_sampling_block` and `(amp, ell)` in
another block leaves the two tightly coupled: with z held fixed,
`f = L(theta) z` is a deterministic function of theta, so
`p(theta | z, y)` is much sharper than `p(theta | y)` and the
alternation random-walks. Measured on `GPClassification.cpp`'s own
dataset, 4 chains x (1000 burn + 2000 keep), worst cross-chain rank
R-hat / bulk ESS:

| sampler | amp | ell | f |
|---|---|---|---|
| ESS-Gibbs, diagonal metric | 1.623 / 7 | 1.179 / 41 | 1.279 / 11 |
| ESS-Gibbs, dense metric | 1.102 / 28 | 1.159 / 17 | 1.067 / 49 |
| ONE joint block over (amp, ell, z) | 1.001 / 1937 | 1.002 / 1539 | 1.003 / 3074 |

`GPRegression.cpp` follows the marginal route; `GPClassification.cpp`
uses the whitened single-block route because Bernoulli-logit has no
closed-form latent integral.

Pick by structure:
- Multi-D continuous covariates + Gaussian response: GPRegression (marginal)
- Multi-D + binary response: GPClassification (whitened, one joint NUTS block over amp, ell, z)
- 1-D time-series, N moderate-to-large: GPTimeSeries (celerite O(N) advantage)
- 1-D time-series, N small: GPRegression also works (O(N^3) is fine for N <= 500)

**When the shipped examples DON'T fit the prompt:**
- Heteroscedastic / hierarchical / multi-output / Kronecker GP ->
  see "**GP composition recipes**" below.
- Non-Gaussian likelihood NOT covered by `GPClassification.cpp`
  (Poisson, Student-t, ...) -> adapt `GPClassification.cpp` by swapping
  the likelihood term (and its `r = d loglik / d f`) inside the joint
  block's log-density. Everything else -- the whitening, the Cholesky
  derivative, the multiplicative jitter -- carries over unchanged.

## elliptical_slice_sampling_block

Generic Elliptical Slice Sampler (Murray, Adams, MacKay 2010) for any
**latent Gaussian model** with arbitrary likelihood. Takes a prior-
covariance Cholesky L from context + a user-supplied log_lik lambda,
returns posterior draws of the latent vector f. No gradient needed, no
step-size tuning, handles arbitrary cross-correlation in the prior
covariance.

**Use cases** -- latent-Gaussian models with **non-Gaussian** likelihoods:
- GP **classification** with Bernoulli-logit (see `examples/GPClassification.cpp`)
- GP regression with non-Gaussian observation noise (Student-t, Poisson, etc.)
- CAR / ICAR / GMRF spatial models with non-Gaussian observation
- Intrinsic GMRF time-series smoothing with non-Gaussian likelihood
- Any latent-Gaussian-with-non-Gaussian-likelihood model (Rue & Held
  2005 book scope)

**DO NOT use for GP regression with Gaussian observations.** When the
observation likelihood is Gaussian, the latent f admits a closed-form
integral
   `p(y | hyperparams) = N(y | 0, K + sigma^2 I)`
and the right architecture is the **marginal-likelihood approach** --
sample only the hyperparameters from the 3-dim marginal posterior, no
explicit f, no ESS. See `examples/GPRegression.cpp` for that pattern;
it matches Stan, libgp, and GaussianProcesses (every reference GP
implementation marginalizes f for Gaussian observations).

**Name disambiguation**: we use the full name `elliptical_slice_sampling`
instead of "ESS" to avoid collision with "Effective Sample Size" in
MCMC diagnostics vocabulary.

**Reference example**: `examples/GPClassification.cpp` (Bernoulli-logit
likelihood -- the canonical case where ESS is the correct choice).

**JUSTIFICATION (Check #16)**: Exception 1 -- specialized latent-
Gaussian sampler; NUTS on f with strongly-correlated Sigma suffers
from step-size collapse. Library parity test at
`tests_autodiff/block_tests/test_elliptical_slice_sampling_block.cpp`
(fix L = I, Gaussian likelihood, 50k draws; variance dead-on match
0.5000 vs analytical, per-point mean within 3-sigma multiple-testing
band).

```cpp
#include "AI4BayesCode/elliptical_slice_sampling_block.hpp"

elliptical_slice_sampling_block_config cfg;
cfg.name = "f";
cfg.N    = N;
cfg.L_chol_key = "L_chol";   // prior Cholesky N*N flat column-major
cfg.log_lik = [&](const arma::vec& f, const block_context& ctx) {
    // User log p(y | f, ctx). Any likelihood.
};
```

**CRITICAL INVARIANT -- `L_chol_key` must point at the Cholesky of the
LATENT prior covariance only.** When this block IS used (i.e. with a
non-Gaussian likelihood that you cannot marginalize), the L_chol fed
in must be the Cholesky of the latent f's prior `K(hyperparams)`, NOT
of any marginal covariance that contains observation-noise terms:

```cpp
// CORRECT -- latent f has prior N(0, K). sigma enters only via log_lik.
M.diag() += jitter;          // chol(K + jitter*I)

// WRONG -- would only make sense for the marginal-likelihood approach
// (no ESS); inside this block it double-counts sigma.
M.diag() += sigma*sigma + jitter;   // chol(K + sigma^2*I + jitter*I)
```

If any parameter (most commonly the observation noise `sigma`) appears
in this block's `log_lik` callback, it MUST NOT also appear in the
dependency chain that produces `L_chol`. Double-counting the same
parameter in both the latent prior and the likelihood targets a wrong
joint posterior. Cross-implementation R-hat against a marginalized
reference (Stan / libgp) reliably catches this silent bug.

Gibbs order: place AFTER hyperparameter blocks; composite
`declare_invalidates` chain ensures L_chol is fresh when ESS runs.

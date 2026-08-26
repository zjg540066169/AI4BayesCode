## elliptical_slice_sampling_block

Generic Elliptical Slice Sampler (Murray, Adams, MacKay 2010) for any
**latent Gaussian model** with arbitrary likelihood. Takes a prior-
covariance Cholesky L from context + a user-supplied log_lik lambda,
returns posterior draws of the latent vector f. No gradient needed, no
step-size tuning, handles arbitrary cross-correlation in the prior
covariance.

**Use cases** -- latent-Gaussian models with **non-Gaussian**
likelihoods, where the prior covariance L is **FIXED** (a graph
Laplacian, a random-walk penalty, a kernel at hyperparameters that are
themselves held fixed or updated far from the latent):
- CAR / ICAR / GMRF spatial models with non-Gaussian observation
- Intrinsic GMRF time-series smoothing with non-Gaussian likelihood
- Any latent-Gaussian-with-non-Gaussian-likelihood model (Rue & Held
  2005 book scope)

**DO NOT use when L depends on parameters you are also sampling.**
This is the case that looks like ESS's home turf and is not: a GP whose
`(amplitude, lengthscale)` are sampled. Two things go wrong.

First, correctness. The elliptical rotation proposes from the Gaussian
prior and accepts on `log_lik` alone, so the prior factor CANCELS --
`log N(f; 0, L L')` appears in NO block's log-density. A hyperparameter
block that does not write that term in by hand is sampled FROM ITS
PRIOR, on every dataset, while the latent fit and every R-hat still
look healthy.

Second, mixing. Even done right -- whitened, with the hyperparameter
block carrying the likelihood at the proposed `(amp, ell)` -- the
Gibbs alternation between "theta given z" and "z given theta" is slow,
because with z held fixed `f = L(theta) z` is a deterministic function
of theta, so `p(theta | z, y)` is much sharper than `p(theta | y)`.
Measured on `examples/GPClassification.cpp`'s dataset, 4 chains x
(1000 + 2000), worst cross-chain rank R-hat / bulk ESS on amplitude:
1.623 / 7 (diagonal metric), 1.102 / 28 (dense), against 1.001 / 1937
for putting `(amp, ell, z)` in ONE `joint_nuts_block`.

For a sampled-hyperparameter GP, use the architecture in
`codegen_cpp.md` Sec.4a instead: marginalise f out for a Gaussian
likelihood, or whiten and sample `(amp, ell, z)` in a single
`joint_nuts_block` for a non-Gaussian one.

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

**Reference example**: none of the shipped examples uses this block --
both GP examples moved to the architectures above. The behavioural
reference is the parity test below, which is also where to look for the
config contract.

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

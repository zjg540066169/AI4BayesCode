## rjmcmc_block ( identity + library 1D transforms + custom AD bijection)

Trans-dimensional MCMC kernel. Samples a paired (gamma, beta)
state where gamma is a binary inclusion vector and beta is a
continuous vector with beta[j] = 0 exactly when gamma[j] = 0
(Dirac spike-and-slab geometry). Each sweep, for each j in
random order: optionally Gibbs-update beta[j] when gamma[j]=1
(via `continuous_update` hook), then propose gamma[j] flip with
a birth-death proposal.

**Scope:**
- **Identity-coordinate** (default): birth draws `beta_new` directly
  from user's `propose_sample`. Jacobian = 1 by construction.
  Users write no Jacobian formula.
- **Library 1D transforms** ( SHIPPED 2026-04-20): optional
  `cfg.transform` accepts a
  `rjmcmc_transforms::transform_1d_base` wrapping one of
  `identity_transform_1d`, `diagonal_linear_transform_1d(scale)`,
  `diagonal_affine_transform_1d(scale, offset)`. When set, the
  birth pipeline routes through the transform:
    u ~ propose_sample(rng, j, ctx)
    beta_new = transform.apply_forward(u, &beta_new)
    |det J| computed by the transform (library-side, not user-side)
  MH accept ratio includes the Jacobian correctly per Green 1995.
  When NOT set (default), the identity path is used.
- **Custom bijection with runtime AD Jacobian** (SHIPPED): optional
  `cfg.transform = make_templated_bijection_1d(Forward{}, Inverse{})`
  where `Forward` is a single TEMPLATED callable struct and `Inverse`
  is a non-templated `double -> double` analytic inverse. Framework
  instantiates Forward at both `double` (for sampling) and
  `autodiff::var` (for runtime AD computation of `|dbeta/du|`). Users
  STILL write no Jacobian formula. See
  `include/AI4BayesCode/rjmcmc_custom_bijection.hpp` for the API and
  `validator.md` Sec.14 for the bijection sanity probes
  (round-trip / Jacobian non-singularity / forward-reverse Jacobian
  inverse-pair).

See `system_design.md` Sec.10.2 for the full three-tier story (
identity / library 1D transforms / custom AD bijection).

**Supported model classes:** Dirac spike-and-slab variable
selection (the canonical use case), change-point insertion with
prior-sampled values, mixture-component birth/death for finite
unknown-K (non-BNP).

**NOT appropriate for:**
- BNP mixtures (DP, PY, HDP) -- use the **truncated SBP** path
  (`stick_breaking_block` + `normal_gamma_cluster_gibbs_block` +
  `categorical_gibbs_block`); see `examples/DPGaussianMixture.cpp`
  / `examples/PYGaussianMixture.cpp` /
  `examples/DPGaussianMixture_DerivedAlpha.cpp`. **The cluster-prior
  hyperparameters MUST be data-driven weakly-informative (see the
  CRITICAL note under `normal_gamma_cluster_gibbs_block` below) -- fixed
  hypers silently produce a wrong over-segmented posterior R-hat won't
  flag.** CRP-marginal Neal Alg 2/8 and Jain-Neal split-merge are not
  in the current scope; they remain optimisations rather than
  correctness gates (note: `split_merge_block` co-composed with
  `categorical_gibbs_block` is now unblocked via
  `composite_block::declare_shared_history("z")` if the truncated-SBP
  partition posterior is multimodal for a given dataset).
- Multi-dim bijections `R^n -> R^n` for `n > 1` (e.g., Stephens
  2000 split-merge for finite mixtures with unknown K) -- current scope ships
  the 1D-scalar case only (`templated_bijection_1d`); multi-dim
  custom bijections require a future block class.
- Multi-coefficient block birth (birthing a cluster of related
  coefs at once via one transform) -- wait for a future multi-dim
  transforms); the current scope is 1D-per-coefficient only.
- HMM / Markov-structured discrete -- use `hmm_block` (T10,
  SHIPPED 2026-04-20; see the hmm_block entry above).

**Critical for good mixing:** supply `continuous_update` function
in the config. This is typically a closed-form Gibbs draw for
beta[j] | gamma[j]=1 under the linear conditional. Without this
hook, beta[j] stays at its birth-time value as long as gamma[j]=1
-- posterior for beta is badly biased and sigma^2 is inflated.

**Reference templates:**
- `examples/SpikeSlabRJMCMC.cpp` -- canonical Dirac spike-and-slab
  under the **Ishwaran & Rao 2005 sigma-scaled slab** form
  (`beta_j | gamma_j=1, sigma, tau ~ N(0, sigma^2 tau^2)`, so tau
  is dimensionless). 4-block composite:
    1. `beta_gibbs_block(pi)` -- Exception 3 (scalar Beta-Bernoulli
       conjugate); covered by library parity test Check #15.
    2. `nuts_block(sigma)` with Jeffreys `p(sigma) prop.to 1/sigma`.
    3. `nuts_block(tau)` with Jeffreys `p(tau) prop.to 1/tau` +
       k=0 fallback to half-Normal(0, 1) pin (inside the natural-
       scale log-density).
    4. `rjmcmc_block(gamma, beta)` with hand-written Gibbs
       `continuous_update` for `beta_j | gamma_j=1` -- Exception 2
       (kernel contract); covered by per-usage parity test
       Check #15.
  `gamma_init` via marginal-OLS screening
  (`gamma_init[argmax_j |X_j'y|] = 1`) guarantees k >= 1 at iter 1,
  avoiding the improper-posterior transient on tau under Jeffreys.
  Constructor takes only `(X, y, a_pi, b_pi, seed, keep_history)` --
  no hyperparameters on sigma/tau thanks to scale-invariant
  Jeffreys priors. See `skills/codegen_priors.md Sec.2a` for the variance
  prior discipline rationale and Sec.3a Class 2b for the Dirac
  spike-and-slab decision tree.
- `examples/SpikeSlabSinhBijection.cpp` -- minimal demo using
  the templated bijection path (`make_templated_bijection_1d` with
  a `sinh` forward + `asinh` inverse). Single-coefficient toy
  model with fixed hyperparameters; the whole point is to exercise
  the custom-bijection plumbing end-to-end with a non-linear bijection that
  the identity and library transforms cannot fit. See `tests_autodiff/audit_rjmcmc_custom_bijection.R`
  for the multi-chain MCMC audit (rhat + posterior comparison
  against the closed-form Bayes-factor answer).

```cpp
rjmcmc_block_config cfg;
cfg.name              = "gamma_beta_rj";
cfg.gamma_key         = "gamma";
cfg.beta_key          = "beta";
cfg.p                 = p;
cfg.log_joint         = &spike_slab_log_joint;
cfg.propose_sample    = &spike_slab_propose_sample;
cfg.propose_logq      = &spike_slab_propose_logq;
cfg.continuous_update = &spike_slab_continuous_update;  // closed-form Gibbs
cfg.gamma_init        = arma::vec(p, arma::fill::zeros);
cfg.beta_init         = arma::vec(p, arma::fill::zeros);
```

**R-hat caveat on gamma R-hat:** similar to DirichletSparse,
gamma[j] that are consistently 0 across chains can show
numerical-noise R-hat (>1.05 or higher). Inspect per-component
R-hat only on j with mean(gamma[,j]) > 0.01 (the "active subset"
-- j with any non-trivial inclusion probability). True-zero j
should NOT contribute to the R-hat diagnostic.

**sigma/tau cross-chain R-hat under spike-and-slab (empirical
finding, 2026-04-20 long-chain audit):** sigma and tau's cross-
chain R-hat is a **secondary, unreliable** diagnostic for this
model class. Different MCMC chains settle at different valid
gamma patterns (all with 100% true-signal recovery but possibly
differing on false-positive nuisance inclusions); sigma and tau's
conditional posteriors depend on which gamma pattern is active,
so cross-chain sigma/tau R-hat stays > 1.5 even at 30k+30k chain
length. The 2026-04-20 long-chain experiment
(`tests_autodiff/audit_rjmcmc_longchain.R`) confirmed this is
structural, NOT slow mixing: X6 sigma R-hat went from 1.10 at
10k+10k to 1.73 at 30k+30k (got worse as each chain settles
more firmly into its own local gamma mode).

**Recommended spike-slab diagnostic protocol:**
- PRIMARY: R-hat on active subset of gamma and beta (< 1.05
  target). These DO converge with longer chains and are the
  diagnostics that reflect inference correctness.
- SECONDARY (informational only): R-hat on sigma and tau. Treat
  values > 1.05 as evidence of gamma multimodality, NOT a
  sampler bug. Report but don't block on.
- INCLUSION: `min_tpos` (min true-positive inclusion prob) and
  `max_fpos` (max true-negative inclusion prob) are the
  inference-level metrics that matter.

## BNP example summary

| Example | Process | Hyperparam sampler | Cluster sampler | discount |
|---|---|---|---|---|
| `DPGaussianMixture.cpp`             | DP truncated SBP | `nuts_block` on `log(alpha)` | `normal_gamma_cluster_gibbs_block` | n/a |
| `PYGaussianMixture.cpp`             | PY truncated SBP | `nuts_block` on `log(alpha)` (digamma terms) | same as DP | FIXED at construction |
| `DPGaussianMixture_DerivedAlpha.cpp` | DP truncated SBP, alpha = exp(phi) via refresher | `nuts_block` on `phi` (REAL) | same as DP | n/a |
| `FiniteGaussianMixture.cpp`         | **Finite K** (NOT BNP); pi via Dirichlet(alpha/K) symmetric prior | n/a (alpha fixed at construction) | same as DP | n/a |

**The architectural rule for BNP mixtures with truncated SBP**:

| Parameter | Block | Why |
|---|---|---|
| z (length N) | `categorical_gibbs_block` | Class-1 conditional independence given (pi, mu, lambda) |
| pi (length K_trunc) | `stick_breaking_block` | Per-stick Beta conjugate; user supplies a_fn/b_fn |
| (mu, lambda) | `normal_gamma_cluster_gibbs_block` | Per-(k,d) Normal-Gamma conjugate; empty clusters draw from prior |
| alpha (scalar) | `nuts_block` on `log(alpha)` | Beta likelihood x Gamma prior; closed-form Gamma alternative requires a future `gamma_gibbs_block` |
| (alpha derived) | `register_refresher("alpha", ...)` | Pattern shown in `DPGaussianMixture_DerivedAlpha.cpp` |

Truncation choice: default `K_trunc = max(20, ceil(N / 5))`. The
truncation error decays exponentially in K_trunc for moderate alpha
(Ishwaran & James 2001 Sec.2). For data whose true cluster count is
> ~K_trunc/2 supply a larger K_trunc explicitly.

**LABEL SWITCHING**: BNP mixtures are exchangeable in (mu, lambda,
pi). Per `skills/label_switching.md`, post-MCMC relabeling is required
for per-component summaries. Audits using R-hat on per-component
parameters MUST apply Stephens 2000 (or equivalent) before computing.
The shipped `audit_dp_gaussian_4chain.R` only checks GLOBALLY-IDENTIFIED
quantities (alpha, K_active, predictive moments) for this reason;
per-component R-hat is left to user post-processing.

**Marginal-Neal-Alg-2/8 (CRP) alternative**: not shipped at this version. Would
require a CRP-specific categorical_gibbs that handles birth/death of
clusters in the partition (z_i sampled from a (K+1)-way mixture where
the K+1-th outcome creates a new cluster). The truncated SBP approach
chosen here keeps state fixed-dim and reuses existing
`categorical_gibbs_block` directly -- that is the current Q5 lean
(current design lean).

**Split-merge (Jain & Neal 2004)**: not shipped at this version. The existing
`rjmcmc_block` is structured for spike-and-slab (gamma, beta) partitions, not
for cluster-allocation split-merge proposals. A future
`split_merge_block` would implement Jain-Neal's restricted-Gibbs +
MH-on-partitions directly. Truncated SBP mixes well enough on toy and
moderate-N problems that split-merge is an optimisation rather than a
correctness requirement.

---

# VI-family (variational inference blocks)

VI is an ENGINE alternative to NUTS / joint NUTS for the SAME kinds of
continuous parameters, not a different parameter-kind specialization.
The "Block type table" at the top of this catalogue lists the standard
MCMC choice per parameter kind; the VI alternatives below replace
those choices when the user has selected VI in the codegen.md Sec.3 VI
engine trigger. See `system_design.md Sec.18` for the full architectural
backing.

**When to reach for a VI block instead of nuts_block / joint_nuts_block**

- The model has known symmetries that fragment the MCMC posterior into
  modes the chains can't mix between (BNN sign/permutation symmetry,
  deep-network weights with hidden-unit permutation, embedding
  parameters), AND the user has accepted "one mode is good enough"
  semantics. MCMC's R-hat will routinely fail at any chain length; VI
  picks one mode cleanly.
- The model has many tightly-coupled continuous parameters where joint
  NUTS would need full-rank metric estimation (warmup-expensive, see
  Check #18 / `system_design.md Sec.13` "Metric + warmup decision"). The VI
  alternative is a SINGLE `full_rank_gaussian_vi_block` over the
  concatenated coupled params (merged dim <= 50) -- the direct analog of
  `joint_nuts_block`. **Do NOT split coupled continuous params into
  separate per-parameter VI blocks** (`q(beta)*q(sigma^2)` coupled through the
  composite by q-sample passing): the cross-block mean-field
  factorization cannot represent the joint covariance, so it
  underestimates marginal variance (measured ~66% too small on a
  collinear pair in the conjugate-NIG regression demo), and the
  per-block PSIS-k-hat is a FALSE reassurance -- it scores each block's
  CONDITIONAL fit (given a sibling q-sample), not the joint marginal, so
  a too-narrow q shows a low k-hat while its variance is wrong. Above merged
  dim 50, a single mean-field block over the concatenated vector still
  pays the variance-underestimation cost but gives ONE well-defined
  joint k-hat -- at one optimizer step per outer iteration instead of one HMC
  trajectory. See `codegen.md Sec.VI sub-flow Step 2a` for the codegen rule.
- The user has explicitly accepted point-estimate-of-q semantics in
  exchange for tractable speed (deterministic prediction, no chain-
  mixing concerns).

**When NOT to reach for VI** (default to nuts_block / joint_nuts_block):

- Low-dim posteriors with no identification issues -- NUTS works.
- Models needing honest credible intervals -- mean-field underestimates
  variance.
- Multimodal posteriors where ALL modes matter -- VI picks one.
- Discrete latent variables with strong dependence -- VI marginals
  collapse to deterministic 0/1 under exclusive KL.

See `system_design.md Sec.18.9` for full caveats and `codegen.md Sec.3 "VI
engine trigger"` for how the trigger question surfaces these to the
user.

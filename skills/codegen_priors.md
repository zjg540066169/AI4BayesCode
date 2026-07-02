---
name: AI4BayesCode-codegen-priors
description: |
  Prior elicitation + block selection for AI4BayesCode code generation.
  Extracted from codegen.md (sections 2a, 2b, 2b.1, 2c-e, 3a). Covers:
  variance / scale prior discipline (Gelman 2006 + Jeffreys default),
  block selection priority (specialized / structural blocks by
  applicability FIRST -> joint_nuts_block DEFAULT for the rest ->
  single nuts_block LOW -> gibbs avoided), the univariate slice sampler
  NUTS fallback, Gibbs-block
  Checks #15/#16/#17, and the discrete-variable decision tree
  (Class 1-5) that governs when *_gibbs_block / rjmcmc_block /
  hmm_block is appropriate. The entry-point skill `codegen.md`
  points here for all prior + block decisions.
---

# AI4BayesCode codegen — priors + block selection

Companion skill to `codegen.md`. Load this when resolving priors,
picking block types, auditing Gibbs usage, or classifying discrete
latents.

**MANDATORY — every prior-clarifying question MUST be a structured
labeled-option prompt** (the mechanism is environment-dependent, per `start.md`
§0 and `codegen.md` §0): **in Claude Code, use the `AskUserQuestion` tool — never
an inline "(a)/(b)/(c)" list; in an environment WITHOUT a structured-question tool
(Codex Default mode, Gemini CLI, plain chat), use the markdown labeled-option
fallback from `start.md` §0.** Either way, NEVER skip the ask or silently use a
default. This applies to every "ask the user" moment below, including: prior on
variance / scale (§2a), parameterization ambiguity (§"Parameterization
ambiguity"), missing hyperparameters on Gamma / Beta / Laplace / etc.,
conjugate-vs-NUTS block choice, and the discrete-variable decision tree (§3a).
When this skill says "ask the user", "ask only when ...", or "propose them as
options", it is shorthand for "emit one structured labeled-option prompt". In
Claude Code specifically, inline option bullets are forbidden — re-emit as
`AskUserQuestion`. The same rule is in `codegen.md` §0.

**MANDATORY — every prior-related `AskUserQuestion` call MUST include
a "Literature-informed" option** (or "LLM-informed" when web search
is unavailable). See the "Literature-informed prior elicitation"
section below for the elicitation workflow. Examples of questions
that must offer this option:
- §2a: scale-prior choice (Jeffreys / HalfNormal / Inverse-Gamma /
  literature-informed).
- "Parameterization ambiguity": rate vs scale vs precision choice
  (defaults / user-specified / literature-informed).
- Missing hyperparameters on any informative-family prior (Gamma,
  Beta, Laplace, half-Cauchy, etc.).
- Any "(a) default / (b) alternative" picker.
Users should never have to ask "can you also try a literature-based
prior?" — that option is always on offer.

**MANDATORY — when prior elicitation completes and the workflow
moves to the model-confirmation step (DAG image + summary table in
`codegen.md` §4), the DAG shown to the user MUST be a PREDICTION
DAG (generative / causal direction), NEVER a Gibbs / dependency
DAG.** This is the same MANDATORY rule stated in `codegen.md` §4(b);
it is repeated here so that prior-elicitation transitions cannot
accidentally produce a Gibbs-style diagram. Concretely: edges go
from parents to children in the data-generating story, sibling
priors get no edges between them even when their Gibbs full
conditionals read each other (canonical counter-example:
spike-and-slab Laplace, where σ and τ² are siblings — both parents
of β — with no σ↔τ² edge despite the slab term σ²τ² making the
Gibbs conditionals mutually dependent). See `codegen.md` §4(b) for
the full rule and `system_design.md` §4 for the Gibbs-vs-Predict
DAG distinction.

---

## 2a. Variance / scale parameter prior discipline (Gelman 2006)

**DEFAULT for any scale parameter sigma > 0 in any model** (noise
variance in regression, group-level variance in hierarchical models,
etc.):

    p(sigma) ∝ 1/sigma    (Jeffreys, scale-invariant noninformative)

Equivalently p(log sigma) = constant. Parameterize as η = log(sigma)
and the Jacobian of the transform exactly cancels the prior term —
the user's NATURAL-SCALE log-density contains the Jeffreys prior as
a `-log(sigma)` term, and `constraints::positive::wrap` adds the
`+log(sigma)` Jacobian, yielding a clean likelihood on η-space.

**Precise arithmetic** (beware of common confusion): the likelihood
itself contains a `-N log(sigma)` term from the Gaussian normalization
`(2π sigma²)^{-N/2}`. That -N log(sigma) is part of the LIKELIHOOD,
NOT the prior; it does NOT cancel with the Jacobian. Only ONE
`-log(sigma)` (the Jeffreys prior term) cancels the ONE `+log(sigma)`
(the Jacobian). Example for Gaussian regression:

    natural-scale lp(sigma) = -(N+1) log(sigma) - SSE / (2 sigma²)
                              ^^^^^^^^^^^^^^^^
                              -N log(sigma)  (likelihood normalization)
                              -log(sigma)    (Jeffreys prior)
                              = -(N+1) log(sigma)

    After wrap adds +log(sigma) Jacobian:
    unconstrained lp(η)      = -N log(sigma) - SSE / (2 sigma²)
                              = -N·η - SSE / (2 exp(2η))

See `examples/SpikeSlabRJMCMC.cpp` and
`examples/GaussianLocationScale.cpp` for reference implementations
of sigma with Jeffreys prior on `nuts_block` + `constraints::positive::wrap`.

**PROHIBITED as default "noninformative":**

- `InverseGamma(ε, ε)` with small ε (Gelman 2006 Bayesian Analysis
  1(3):515-533 critique: improper at ε → 0; posterior is highly
  sensitive to the choice of small ε; not actually noninformative).
- Any proper prior claiming to be "noninformative" without scale-
  invariance. `half-Cauchy` and `half-Normal` are WEAKLY informative,
  not noninformative — use only as escape for sparse-info cases.

**ESCAPE when Jeffreys gives improper posterior (k=0 fallback):**

A scale parameter can receive ZERO effective data transiently —
e.g., slab variance in spike-and-slab when all gamma = 0
(sum_active beta² = 0); group-level variance when a group has 0
members. Under Jeffreys, the conditional posterior in those states
is the improper prior itself and the chain is unstable.

**Recommended pattern** (used in `examples/SpikeSlabRJMCMC.cpp`):

1. **Keep Jeffreys as the main prior** — whenever k ≥ 1 (nonzero
   effective data), the conditional is proper and NUTS is happy.
2. **Detect k=0 inside the natural-scale log-density** and return
   a **half-Normal(0, 1) fallback density**:
   ```cpp
   if (k == 0) {
       lp = -0.5 * tau * tau;   // half-Normal(0, 1)
       grad = -tau;
       return lp;
   }
   // normal case with k >= 1:
   lp = -(k+1) log(tau) - sum_b2 / (2 * sigma^2 * tau^2);
   ```
   The fallback softly pins tau near its reference scale (for the
   spike-and-slab σ²τ² parametrization, tau=1 is "signal scale =
   noise scale"). This is a transient state; as soon as any γ_j
   activates, the main Jeffreys+likelihood density takes over and
   tau's posterior concentrates on the data-driven value.
3. **Initial-value guard** — set initial state so k ≥ 1 at
   iteration 1 (e.g., marginal-OLS screening: pick the most-
   correlated predictor as the initial active signal).

This pattern keeps Jeffreys as the principled default for well-fed
posteriors and contains the degenerate state with minimal
intervention. It is the pattern the AI agent should use by default.

**For genuinely-sparse-info models** where k remains small
(< 5) throughout MCMC (e.g., hierarchical models with 2-3 groups),
half-Normal(0, A) with A ≈ 2.5 × data-scale is Gelman 2006's
default weakly-informative choice. Don't use half-Cauchy unless
the scale parameter truly is hard to identify — its polynomial tail
is slower for NUTS.

`examples/SpikeSlabRJMCMC.cpp` tau block uses Jeffreys + k=0
fallback. The Ishwaran & Rao 2005 JASA σ²τ² parametrization (tau
dimensionless) makes Jeffreys the natural choice on tau.

**Subtle warmup concern (known limitation).** If k=0 is entered
during NUTS's initial dual-averaging warmup, the step size adapts
against the half-Normal(0, 1) fallback geometry (tight, low
curvature), then may be suboptimal when k flips to ≥ 1 and the
geometry changes to Jeffreys + slab likelihood. In practice the
marginal-OLS `gamma_init` guarantees k ≥ 1 from iteration 1, so
the first warmup iteration is already in the "real" regime and
this concern is rare. If you write a new model where k=0 is a
likely early state (e.g., no good marginal-screening signal), run
an extra sanity warmup step before letting the chain sample.

**INFORMATIVE IG prior** is ONLY appropriate when the user explicitly
provides hyperparameters from external knowledge (e.g., BART's
calibrated IG using sigest — see below). NEVER use IG as a default.

---

## 2b. Block selection priority

Selection is TWO-PHASE — NOT "the first item in the list wins":

**Phase 1 — specialized / structural blocks claim what they match
(checked FIRST, by applicability).** If the model's structure is targeted
by a specialized block — discrete latent (HMM / Ising / Potts), random
parameter-space dimension (variable selection / model averaging →
`rjmcmc_block`, per §3a), sparse GMRF / CAR / ICAR, GP latent, tree
ensemble (BART), or a blessed conjugate structure — route those parameters
to that block. For these, joint NUTS is at best slower and for some
(discrete latent, random dimension) is structurally INAPPLICABLE (NUTS
cannot sample discrete states or a point-mass spike). A specialized block
that genuinely matches the structure ALWAYS outranks joint NUTS.

**Phase 2 — `joint_nuts_block` is the DEFAULT for everything left.** Every
continuous parameter NOT claimed in Phase 1 goes into ONE `joint_nuts_block`.

The numbered list below is a CATALOGUE of block types, NOT a priority
rank — do NOT read "item 1" as "try this first":

1. **`joint_nuts_block`** — **DEFAULT for continuous parameters.**
   Collect ALL continuous parameters not claimed by a specialized
   block (RJ / BART / HMM / GP / GMRF / whitelist Gibbs — items 3+
   below, which are checked first by applicability) into ONE
   `joint_nuts_block` sampled over the concatenated unconstrained
   vector. Per-slice `joint_constraint` (REAL / POSITIVE) handles
   mixed support in a single block; the user's log-density stays
   natural-scale (the block adds each slice's Jacobian via
   `constraints::<kind>::wrap` — system_design.md §10.1).

   **Do NOT force a decomposition to fit existing blocks.** Block-
   decomposed per-parameter NUTS was a tooling concession; the
   generator can now write a correct joint natural-scale log-density
   directly. When a model has no matching specialized block, WRITE
   THE JOINT LOG-DENSITY rather than contorting the model into our
   block menu — a single `joint_nuts_block` over a hand-written
   log-density is the flexible default. Coupled parameters
   (correlation → 0.9 in collinear designs, hierarchical scales,
   VAR / spectral coupling) mix ~10× slower per ESS when split, and
   FREEZE outright on funnels — so joint is the default, not an
   optimization.

   **Funnel pattern (positive scale + raw effects) → MANDATORY
   non-centered reparameterization.** If a positive scale parameter
   (τ, σ) governs the spread of raw effects (`raw_j ~ Normal(·,
   scale)`), the centered form FREEZES even joint NUTS (empirically:
   eight_schools_centered → random-effect ESS = NA, R-hat 2.23).
   Apply NCR and follow **`skills/joint_nuts_failure.md` (Mode 1)**
   for the recipe + constraint-kind layout. Validator Check #24
   enforces it. (Verified reference:
   `tests/test_joint_nuts_ncr_funnel.cpp`.)

   Fill in `joint_nuts_block_config` directly (fields
   `log_density_grad` / `initial_cat`); POSITIVE and all per-slice
   constraints are fully supported, alongside the dense / 3-phase /
   diagonal metric paths. See `examples/HSGPRegression.cpp` and
   `examples/HierarchicalLM_MultivariateRE.cpp` for POSITIVE-slice +
   dense-metric templates. Validator Check #11 (+ #24).
2. **`nuts_block`** — single-parameter NUTS. **LOW priority.** Most
   continuous parameters belong in the joint block (item 1); reach for
   a standalone `nuts_block` only for: a genuinely scalar continuous
   parameter; a post-NCR funnel branch that cannot share the joint
   metric; or a high-dim block with obvious conditional independence
   the generator chooses to isolate. Always correct as a fallback (NUTS
   on a Jacobian-wrapped log-density is never wrong-target), just slower
   per ESS than joint when parameters are coupled. Validator Check #11
   does not apply (single parameter).
3. **`rjmcmc_block`** — trans-dimensional. Class 4; only applicable
   when the parameter space dimension is itself a random variable
   (state space is not a fixed-dim manifold). See §3a below and
   §11.2(a) of `system_design.md`.
4. **`bart_block` / `genbart_block`** — tree-based nonparametric
   regression. Specialized; both pull in GPL-2.0-or-later.
   - `bart_block`: vanilla Gaussian BART via Bayesian backfitting
     (Chipman et al. 2010).
   - `genbart_block`: Linero 2022 generalized-BART via RJMCMC with
     Laplace leaf proposals; accepts a plug-in `genbart::likelihood`
     for Normal / Logistic / Poisson / NB / Heteroscedastic / AFT /
     Beta / Gamma_shape / Beta-Binomial / user-supplied. Use this
     whenever the response family is non-Gaussian OR when the user
     needs a response family outside the 10 shipped ones (subclass
     `genbart::likelihood`; see `codegen_cpp.md §6c`).
5. **`hmm_block`** — exact forward-filter backward-sample for
   finite-state Hidden Markov Models.

   **Bayesian-nonparametric mixture infrastructure (SHIPPED 2026-05-02):**
   - **`stick_breaking_block`** — truncated stick-breaking simplex,
     covers DP / PY / ... / custom Beta-stick processes via user-supplied
     `a_fn` / `b_fn`. See `block_catalogue.md`.
   - **`normal_gamma_cluster_gibbs_block`** — vectorized diagonal
     Normal-Gamma cluster sampler over K_trunc clusters with empty-
     cluster prior draws.
   - **`bnp_utils.hpp`** — header-only CRP / PY weight + log-prior
     helpers for user-written log_probs_fn / refreshers.

   Reference examples: `DPGaussianMixture.cpp`, `PYGaussianMixture.cpp`,
   `DPGaussianMixture_DerivedAlpha.cpp`. The composition recipe is in
   `block_catalogue.md` "BNP example summary"; the planned (not v0)
   pieces are `gamma_gibbs_block` (closed-form alpha update),
   `niw_cluster_gibbs_block` (full-covariance NIW), and
   `split_merge_block` (Jain-Neal 2004 partition acceleration).
6. **`pg_logistic_block`** — Bayesian logistic regression via
   Polya-Gamma data augmentation, 10-100× faster than NUTS-on-logistic
   for p < 1000.
   ⚠️ **LINEAR LOGISTIC ONLY.** Hard-scope limitation: the predictor
   must be `X' beta`. **DO NOT use for logistic BART** (where f(X_i)
   is a tree ensemble) — PG augmentation breaks BART's Gaussian-
   observation tree-location identifiability. For logistic BART,
   use `genbart_block + genbart::lik::logistic_lik` (see
   `examples/GBartLogistic.cpp`) instead.
7. **Gaussian Process blocks** — architecture chosen by the
   observation likelihood:

   ### Architectural rule (mandatory)

   | Observation likelihood | f admits closed-form integral? | Architecture |
   |---|---|---|
   | Gaussian | YES (`p(y\|θ) = N(y \| 0, K + σ² I)`) | **Marginal**: sample only hyperparameters |
   | Bernoulli, Poisson, Student-t, Negative-Binomial, … | NO | **Whitened ESS**: sample `z ~ N(0, I)`, recover `f = L(amp, ell) · z` |

   Every reference GP library (Stan, libgp, GaussianProcesses, GPy,
   GPflow) follows the same rule: marginalize f when you can; sample
   a whitened latent when you must. AI4BayesCode follows this
   convention.

   ### Marginal architecture (Gaussian observations)

   See `examples/GPRegression.cpp`. One `joint_nuts_block` over
   the three positive hyperparameters (amplitude, lengthscale, sigma)
   targets `y ~ N(0, K + σ² I)`. Analytic gradient via Rasmussen &
   Williams §5.5 Eq. (5.9). f is recovered at predict time via the
   closed-form posterior conditional. NO `elliptical_slice_sampling_block`,
   NO latent f or z in the sampling state.

   ### Whitened ESS architecture (non-Gaussian observations)

   See `examples/GPClassification.cpp`. The latent is reparameterised
   as `z ~ N(0, I)`; we recover `f = L(amp, ell) · z` whenever a
   likelihood evaluation needs f. `elliptical_slice_sampling_block`
   samples z with prior `L_identity` (the chol of I); the block's
   `log_lik` callback computes `f = L_chol · z` and evaluates the
   non-Gaussian likelihood (Bernoulli-logit, Poisson, …). The
   hyperparameters are sampled together by one
   `joint_nuts_block({amp, ell})` (POSITIVE × 2) whose log-
   density includes `Bernoulli(y | sigmoid(L(amp, ell) · z))` —
   NOT a `p(f | amp, ell)` prior factor — so the `(amp, ell)` chain
   sees the data via the likelihood only and does NOT collapse to
   `(amp ≈ 0, f ≈ 0)`. Joint NUTS is the default because the
   `(amp, ell)` posterior has a banana-shaped ridge that modular
   NUTS slow-mixes along (typically dropping `amp` ESS by 5–10×).

   When `(amp, ell)` ESS is still inadequate at the extended budget
   (typically `ell` lingering at low ESS due to bulk autocorrelation
   from finite-difference gradient noise), the next escalation is to
   replace the FD gradient inside the joint log-density with a
   reverse-mode Cholesky-AD analytic gradient (Murray 2016). See
   `block_catalogue.md` **"GP convergence troubleshooting ladder"**
   for the full escalation order, and **"GP composition recipes"**
   for heteroscedastic / hierarchical / multi-output GP patterns.
8. **`gmrf_precision_block`** — Gaussian Markov Random Field latent
   (sparse precision matrix Q). Rue 2001 sparse-Cholesky direct
   sampler via Eigen SimplicialLLT + AMD reordering, with optional
   sum-to-zero constraint. Use for the latent component when its
   prior takes the form `pi(x) ∝ exp{-½ x^T Q x + b^T x}` with Q
   sparse PSD. Reference templates: `examples/GMRFPrior.cpp`,
   `examples/ICARSpatialGMRF.cpp`.
9. **`ising_cluster_block`** — Discrete Markov Random Field on a
   user-supplied undirected graph. Swendsen-Wang 1987 bond-augmented
   cluster moves. Use when latent is x ∈ {0..Q-1}^n with prior
   pi(x) ∝ exp{β Σ_{i~j} I[x_i = x_j]}. Reference template:
   `examples/IsingPrior.cpp`.
10. **`gmrf_whitened_ess_block`** — Gaussian Markov Random Field latent
    (sparse precision Q) with **non-Gaussian** observation likelihood.
    Murray 2010 Elliptical Slice Sampling on the implicit GMRF prior:
    per step, refactor sparse Cholesky of Q, sample
    nu ~ N(0, Q^{-1}) via Rue 2001 backsolve, then ESS rotate
    `x' = x cos θ + nu sin θ` against the user-supplied
    log-likelihood. Optional sum-to-zero (preserved invariantly by the
    ESS rotation). Use whenever the latent's prior is
    `pi(x) ∝ exp{-½ x^T Q x}` (Q sparse PSD) AND the observation
    likelihood is non-Gaussian (Poisson, Bernoulli, Student-t, etc.) —
    i.e., the cases that `gmrf_precision_block` cannot handle as a
    direct conjugate draw. Example use case: spatial / temporal
    sparse random-effect models with non-Gaussian observation
    likelihoods. Empirical correctness: 95-98% coverage on simulated
    DGP across small-grid pilot fixtures.

**Note — hidden discrete latent (HMM / Ising / Potts / MRF) + Normal
emission routing.** When the model has a discrete latent z with a
specialized prior (HMM / Ising / Potts / general MRF) AND a Gaussian
emission likelihood `y_i | z_i = k ~ N(mu_k, sigma_k)` (possibly with
per-class sigma_k), route as a hybrid composite: the specialized
prior block for z (`hmm_block`, `binary_gibbs_block`,
`categorical_gibbs_block` per the prior structure) + cluster-conjugate
`normal_gamma_cluster_gibbs_block` for per-class `(mu_k, lambda_k)`
(treat z as the cluster partition). DO NOT use NUTS on `(mu_k,
sigma_k)` with an identifying ordering constraint (e.g. `delta > 0`
to force mu_0 < mu_1) — NUTS dual-averaging interacts badly with
slow-mixing z and silently biases the posterior on `(mu_k, sigma_k)`.
Label identification is handled post-hoc by sorting the posterior
means (the standard finite-K mixture convention). See
`codegen_cpp.md` §4a "Hidden discrete latent + Normal emission" row.

11. **`*_gibbs_block`** — LAST RESORT. Use ONLY under a documented
   Exception (see below). Carries the highest silent-correctness
   risk in the library, because derivation errors in the `params_fn`
   produce wrong posteriors that pass all MCMC diagnostics.

**Exceptions that justify a `*_gibbs_block`:**

- **Exception 1 — Discrete parameter.** NUTS cannot target discrete
  measures. Use `categorical_gibbs_block`, `binary_gibbs_block`,
  `dirichlet_gibbs_block`, `beta_gibbs_block` as appropriate.
- **Exception 2 — `rjmcmc_block` kernel hook.** The `continuous_update`
  and `propose_sample` hooks of `rjmcmc_block` are contractually
  required to be direct iid conditional draws (NUTS would break
  detailed balance of the RJ chain). Typically a hand-written
  closed-form conditional sample (+ Check #17 whitelist + Check #15
  parity test).
- **Exception 3 — Scalar / vector textbook conjugate with NUTS-wasteful
  efficiency profile.** Example: π in spike-and-slab has only a
  length-p sufficient statistic and posterior Beta(a + Σγ,
  b + p - Σγ) — NUTS warmup on 1-D tight-posterior scalar is
  wasteful. Another example: heavy-tail IG conditional triggering
  NUTS adaptation pathology (but prefer the §2a pattern —
  Jeffreys + k=0 half-Normal(0,1) fallback — over IG + Gibbs).
  **Vector-valued case: Albert-Chib (1993) latent z for any probit
  binary likelihood** (`y_i ~ Bernoulli(Phi(mu_i + offset_i))` →
  `z_i ~ TN(mu_i + offset_i, 1, sign(2 y_i - 1) > 0)`). The conditional
  is a product of N independent truncated Normals — a length-N
  closed-form vector conjugate sample where NUTS would be silly. Use
  `probit_aug_block` (whitelisted under Exception 3); see
  `examples/ProbitRegression.cpp` (Albert-Chib + NUTS on beta) for a
  reference composition (for a BART probit mean, compose `probit_aug_block`
  with a `bart_block` using `cfg.binary = true`). Do NOT inline the
  truncated-normal sampling in a wrapper — use the library-blessed
  block.

**Hand-written Gibbs (inline distribution samplers outside of
blessed library blocks) is BANNED** except in Exception 2 whitelist.
Validator Check #17 is a static grep for `std::gamma_distribution`,
`Rcpp::rbeta`, etc. in example code and flags non-whitelisted usage.
Rationale: hand-rolled Gibbs requires the AI to correctly judge
conjugacy AND derive the update's parameters — a silent-correctness
failure mode that is MORE error-prone than writing a joint NUTS
log-density (which the validator can check term-by-term). That is
precisely why Gibbs stays rare and `joint_nuts_block` is the default.

**Exception 4 — AI-authored custom block/sampler (structural-gap last
resort).** When — judged AT CODE-GENERATION TIME from the model's
structure, NOT by running and observing a bad R-hat — (a) no
specialized or conjugate-Gibbs block fits the model, AND (b) neither
`joint_nuts_block` nor `nuts_block` is structurally appropriate (e.g. a
bespoke tree ensemble, or a novel trans-dimensional / discrete-structure
move that NUTS cannot express), the generator MAY author its own
block/sampler. This is the LAST resort — exhaust the specialized blocks
and the joint-NUTS default first. It MUST carry an inline justification:
`// JUSTIFICATION (Check #17): Exception 4 — no blessed block fits because <...>; joint/single NUTS structurally inapplicable because <...>; custom scheme = <...>; targets the correct posterior because <...>.`
Within Exception 4, still PREFER a NUTS-based custom log-density over any
hand-written Gibbs — hand-rolled conjugacy stays the most error-prone
path and remains governed by the Check #17 ban; Exception 4 is NOT a
backdoor for it. Custom code is gated by the full validator (compile +
2-chain R-hat / coverage / ESS + gradient Check #12 where applicable).
The block choice is fixed at generation time and is NEVER swapped at
runtime — a runtime try-fail-swap would waste a full MCMC run.

### 2b.1 univariate_slice_sampling_block — NUTS fallback for 1-D non-diff / black-box targets

**Prerequisite: NUTS (item 1 above) is NOT applicable** because ONE of:

  (a) log p is non-differentiable (piecewise, floor/ceil, kink);
  (b) log p is a black-box library call whose gradient would require
      re-implementing that library with autodiff::var types (e.g.,
      celerite marginal log-likelihood via
      `celerite_marginal_likelihood.hpp`);
  (c) gradient evaluation cost is prohibitively high relative to lp
      eval (e.g., O(N^3) Cholesky per gradient with no shared
      factorization).

**Slice is NOT a default substitute for NUTS.** If lp is differentiable
and the gradient is reasonable to write (hand-derived or via autodiff),
ALWAYS prefer `nuts_block` -- NUTS mixes better on smooth continuous
targets and is the library's first choice.

**AI-safety profile (identical to NUTS).** The user writes ONLY a
natural-scale log-density lambda. No conditional-posterior derivation
(unlike Gibbs). Library-level Check #15 parity test
(`tests_autodiff/block_tests/test_univariate_slice_sampling_block.cpp`)
verifies 10k draws match analytical mean/variance on three fixtures
(Normal / Gamma / Beta via real / positive / interval constraints).

**Scope: strictly univariate.** `initial_unc` MUST have length 1. For
multi-dim parameters use `nuts_block` / `joint_nuts_block` (or add a
future `hyperrectangle_slice_sampling_block` / similar variant as a
separate block).

**Current reference use case:** `examples/GPTimeSeries.cpp` v0.5 --
hyperparameters amp, tau, sigma on celerite-marginalized likelihood,
trigger (b) (celerite is a black-box C++ library without exposed
autodiff-through-Cholesky).

**JUSTIFICATION (Check #16):** Exception 1 from the table above --
specialized sampler for 1-D continuous parameters whose log-density
lacks an accessible gradient (violates NUTS prerequisite).

Canonical usage pattern:

```cpp
#include "AI4BayesCode/univariate_slice_sampling_block.hpp"

univariate_slice_sampling_block_config cfg;
cfg.name         = "amp";
cfg.initial_unc  = arma::vec{std::log(amp_init)};  // length 1, unc scale
cfg.constrain    = constraints::positive::constrain;
cfg.unconstrain  = constraints::positive::unconstrain;
cfg.w            = 1.0;    // initial bracket width on unc scale
cfg.log_density  = [](const arma::vec& t_unc, const block_context& ctx) {
    return constraints::positive::wrap(t_unc, nullptr,
        [&](const arma::vec& t_nat, arma::vec* /*unused*/) -> double {
            // natural-scale lp (prior + likelihood); NO Jacobian, NO grad.
            // Jacobian handled by wrap; slice block does no FD.
            return lp;
        });
};
impl_->add_child(
    std::make_unique<univariate_slice_sampling_block>(std::move(cfg)));
```

NOTE: Do NOT add the slice block to the Check #17 whitelist (inline
distribution samplers) -- the slice block is a library-provided
specialized sampler, same category as `elliptical_slice_sampling_block`
and `*_gibbs_block` library blocks. Users writing examples don't emit
inline slice-sampling code; they use the library block.

---

## 2c. Check #15 — Gibbs-block parity test (Option A library-level)

Every `*_gibbs_block` type shipped in the library has ONE block-level
parity test in `tests_autodiff/block_tests/test_<blockname>_gibbs_block.cpp`
that verifies the SAMPLING MECHANISM: given fixed hyperparameters
(e.g., {alpha=5, beta=10} for a Beta distribution), 10,000 draws
match the analytic distribution's mean and variance within
tolerances 5% (mean) and 10% (variance).

Any example using a library `*_gibbs_block` with a **textbook
`params_fn`** (matches one of the canonical patterns below)
inherits correctness from the block-level test — no per-usage
parity test is required.

Canonical textbook `params_fn` patterns:
- `beta_gibbs_block`: `{a_prior + sum(counts), b_prior + n - sum(counts)}`
- `dirichlet_gibbs_block`: `{alpha + counts}` (element-wise)
- `inv_gamma_gibbs_block` (DISCOURAGED as default — see §2a):
  `{a_prior + n/2, b_prior + sse/2}` for Gaussian-IG conjugate
- `categorical_gibbs_block`: likelihood-weighted probability vector

**Escape clause (per-usage parity test required):** if the
`params_fn` is non-textbook (uses conditional logic like
`if gamma[j] == 1`, uses active-subset indexing, or any other
derivation beyond plug-in of counts into a standard posterior), the
example must ALSO ship a per-usage parity test at
`tests_autodiff/test_<model>_<blockname>_gibbs_parity.cpp`.
Same 10,000-draw mean/variance comparison, same tolerances.

If validator finds a `*_gibbs_block` usage without a corresponding
parity test (block-level OR per-usage, depending on `params_fn`),
Layer 2 FAIL.

Check #15 also covers other specialized library samplers:
`elliptical_slice_sampling_block`,
`univariate_slice_sampling_block`,
`poisson_multinomial_aug_block`. Each has a corresponding parity
test in `tests_autodiff/block_tests/`.

## 2d. Check #16 — Inline justification comment for every Gibbs usage

Every `*_gibbs_block` usage and every hand-written Gibbs step inside
an `rjmcmc_block` hook must have an inline comment above the block
construction / hook definition stating which Exception (from §2b
above) justifies the choice. Format:

    // JUSTIFICATION (Check #16): Exception <N> from codegen_priors.md §2b —
    // <short one-line reason>. Per Check #15, covered by
    // <path-to-parity-test>.

Validator static check: grep for `*_gibbs_block` usage without an
accompanying "JUSTIFICATION" or "Exception" mention within 20 lines
above → Layer 2 FAIL.

## 2e. Check #17 — No hand-written Gibbs samplers outside whitelist

Direct inline use of distribution samplers (e.g.,
`std::gamma_distribution`, `std::normal_distribution` returning a
posterior draw, `Rcpp::rbeta`, `Rcpp::rgamma`, `Rcpp::rdirichlet`,
`R::rgamma`) is PROHIBITED in example code, with a narrow whitelist:

- Inside `rjmcmc_block::propose_sample` (kernel contract)
- Inside `rjmcmc_block::continuous_update` (kernel contract, +
  Check #15 per-usage parity test required)
- Inside `register_stochastic_refresher` for posterior-predictive
  `y_rep` generation (NOT part of the Markov chain; no correctness
  impact on posterior)
- Inside `predict_at(Rcpp::List)` wrapper methods (NOT part of the
  Markov chain; same reasoning as stochastic refreshers)
- Inside library internal code (`include/AI4BayesCode/*_gibbs_block.hpp`
  and friends)
- Inside a §2c Exception 4 custom block/sampler carrying the
  `// JUSTIFICATION (Check #17): Exception 4 — …` comment (structural-gap
  last resort: no blessed block fits AND NUTS is structurally
  inapplicable). Hand-rolled conjugate Gibbs where a blessed block or
  NUTS would work is NOT covered by this carve-out.

Validator static check: grep for distribution samplers in
`examples/*.cpp` + filter by source context → list non-whitelisted
hits → Layer 2 FAIL.

**Documented legacy exception**: `examples/ARDLasso.cpp` is a
self-contained Gibbs sampler that pre-dates the block system
(Park & Casella 2008 Bayesian LASSO with Jeffreys-conjugate
conditionals). It uses inline `std::gamma_distribution` and
`std::normal_distribution` extensively, all wrapping textbook
closed-form conditionals. It remains in `examples/` as a reference
for the pre-block ARD pattern. The Check #17 static grep MAY
whitelist this file explicitly; the cpp header carries the legacy
note. DO NOT generate new samplers following this pattern — for
new continuous parameters, use `joint_nuts_block` (default) or
`nuts_block`. The only carve-out is §2c Exception 4 (a genuine
structural gap where NUTS is inapplicable), and even there prefer a
NUTS-based custom log-density over hand-rolled Gibbs.

Why: hand-written Gibbs is the highest-risk silent-correctness bug
class in this library. An AI agent writing a one-off posterior draw
can get the formula wrong without any diagnostic catching the
error — misjudging conjugacy or the update's parameters is easier than
getting a joint NUTS log-density wrong. Library-blessed `*_gibbs_block`
(with parity tests, inline justification) is the safe path. For any
continuous param not covered by Exceptions 1-3 of §2b: USE
`joint_nuts_block` (default) / `nuts_block` INSTEAD. Hand-written
sampling is permitted ONLY under §2c Exception 4 (no block fits AND
NUTS structurally inapplicable), with the Check #17 justification
comment and full validator gating.

---

## 2f. VI engine considerations (only when VI is selected)

The codegen.md §3 VI engine trigger lets the user opt into pure VI
or hybrid (MCMC + VI). When VI is selected for ANY parameter,
revisit the prior choices for those VI'd parameters using this
checklist. Priors for parameters that stay on MCMC are unchanged.

Foundations: VI optimizes a finite ELBO over the variational
parameters λ; if the prior contributes a non-finite term at the
typical η scale, the optimizer diverges. MCMC tolerates a much
wider class of priors (improper, scale-invariant, etc.) because
the sampler only ever evaluates ratios. Implications:

**(1) Avoid genuinely improper priors on VI'd parameters.**
- Jeffreys `p(σ) ∝ 1/σ` on a VI'd scale: improper, but conditionally
  proper when the data effective count k ≥ 1. For VI, this typically
  WORKS at convergence but is fragile during early optimization
  when the natural-scale gradient explodes near σ → 0. Use the
  §2a inline k=0 half-Normal(0, 1) fallback pattern, OR switch the
  VI'd σ to half-Normal(0, A) with A ≈ 2.5 × data-scale. For pure
  MCMC the §2a defaults stand unchanged.
- Flat / uniform on R for a location parameter: never well-defined
  for VI without a data-informed reparameterization. Use a weakly
  informative N(0, 5²) or similar.
- "Diffuse" priors with very large scale (N(0, 1e6²), Gamma(1e-3,
  1e-3) on a precision): mathematically proper but pathological
  for VI initialization — the unconstrained-space scale of the
  posterior is many orders of magnitude smaller than the prior
  scale, so initial σ values are far from a useful neighborhood.
  Use weakly informative defaults (N(0, 5²) or N(0, 10²) on
  regression coefs; half-Normal(0, 2.5 × data-scale) on σ).

**(2) Non-centered / unit-scale reparameterizations help VI.**
- For hierarchical models on MCMC, non-centering is sometimes a
  fix for Neal's funnel; for VI, non-centering is **almost
  always** preferred because mean-field on a centered hierarchical
  posterior fails badly (the prior-scale coupling between θ_j and
  τ creates posterior dependence that mean-field cannot capture).
  Default to non-centered for any VI'd hierarchical parameter.
- For BNN weights: keep them in the natural N(0, σ²) form (no
  reparameterization needed); the VI block sees them as
  `q(η_jk) = N(η_jk; μ_jk, σ_var,jk²)` directly. The prior σ²
  is a separate parameter (often kept on MCMC in hybrid mode).

**(3) Constraint transforms don't change.** `constraints::positive::wrap`
and friends work identically under VI — VI's unconstrained-space
η is exactly the same as NUTS's unconstrained-space draw, with
the same log|J|. No new prior-elicitation work is needed for the
transform.

**(4) Hyperpriors on hyperparameters (sigma2_alpha, sigma2_beta
in BNN, τ in Lasso, etc.)** typically belong on MCMC even in
hybrid mode — they're scalar, well-behaved, and VI mean-field
underestimates their variance which propagates through to all
the dependent VI'd parameters. Default: hyperparams stay on
NUTS / NUTS-positive; only their many-coef children go on VI.

**(5) Spike-and-slab (§3a Class 2a Gaussian-mixture spike or §3a
Class 2b Dirac spike) under VI**: not supported in v1. Mean-field
VI cannot represent the inclusion-indicator posterior properly
(the q(γ_j) marginal collapses to a deterministic 0 or 1 under
exclusive-KL optimization). If the user picks VI for a model with
spike-and-slab structure, the codegen agent MUST default the
inclusion-indicator block back to MCMC (rjmcmc or
`binary_gibbs_block`) and put VI only on the continuous parts.
Document this in the model summary.

See `system_design.md §18.5` (variational family) and §18.9
(caveats — when to NOT use VI) for the architectural backing of
these rules.

---

## Parameterization ambiguity

Some distributions have multiple common parameterizations. When the
user specifies one of these, always confirm which form they mean
before writing the log-density:

| Distribution | Form A | Form B | How to ask |
|-------------|--------|--------|------------|
| Gamma | shape-**rate**: E[X] = shape/rate | shape-**scale**: E[X] = shape*scale | "Gamma(a, b) — is b the rate or scale?" |
| Inverse-Gamma | shape-**scale**: E[X] = scale/(shape-1) | shape-**rate**: E[X] = 1/(rate*(shape-1)) | "InvGamma(a, b) — is b the scale or rate?" |
| Normal | mean-**sd** | mean-**variance** | "Normal(0, 10) — is 10 the sd or variance?" |
| Half-Normal | **sd** | **variance** | "HalfNormal(0, 10) — is 10 the sd or variance?" |
| Wishart | df-**scale matrix** | df-**rate matrix** (inverse scale) | "Which Wishart convention?" |

The table above uses **rate** parameterization for Gamma as the default
(matching R's `dgamma(x, shape, rate)`). If the user says "Gamma(2, 3)"
without clarification, ask. Getting this wrong silently produces a
different prior with potentially very different behavior.

### Skip clarification at numeric fixed points

Before triggering a clarification question, check whether the specific
numeric values would yield *the same distribution* under both
parameterizations. If so, the ambiguity is vacuous and no question is
needed. The identity fixed points for all scale-family distributions
above are **0** and **1**.

| User input | Check | Ask? |
|---|---|---|
| `Normal(0, 1)`        | σ=1 ⇔ σ²=1 (since 1²=1)              | **No** |
| `Normal(0, 4)`        | σ=4 vs σ²=4 (i.e. σ=2) — different   | **Yes** |
| `Gamma(1, 1)`         | rate=1 ⇔ scale=1                     | **No** |
| `Gamma(2, 1)`         | rate=1 ⇔ scale=1, shape irrelevant   | **No** |
| `Gamma(2, 3)`         | rate=3 vs scale=3 → 9× different     | **Yes** |
| `HalfNormal(1)`       | σ=1 ⇔ σ²=1                           | **No** |
| `InverseGamma(2, 1)`  | rate=1 ⇔ scale=1                     | **No** |
| `LogNormal(0, 1)`     | same as Normal fixed-point logic     | **No** |

**Rule of thumb:** for each scale-like argument, if its value is 0 or 1,
treat that argument as unambiguous. Ask only when at least one scale
argument is a non-trivial positive number.

This check belongs in the **pre-generation validation** phase: agents
that ask about every `Normal(0, 1)` or `Gamma(2, 1)` they see become
annoying without preventing any actual posterior errors. The question
is raised only when misinterpretation would measurably change the
prior.

## BART sigma override (default for BART-structured models)

For any model whose likelihood is `y ~ N(BART(X), sigma^2)` (or contains
a BART mean component), **override the generic Jeffreys `p(sigma) ∝ 1/sigma`
scale default** (the default-priors table) and use BART::wbart's calibrated
conjugate inverse-gamma:

    sigma^2 ~ InverseGamma(nu/2, nu*lambda/2),
    lambda  = sigest^2 * qchisq(0.1, nu) / nu,   (sigquant = 0.9 default)

where `sigest` is the OLS residual sd, computed automatically by
`bart_model` — retrieve via `bart_block::current_sigma()` after the
BART child is added to the composite. Default `nu = 3`.

This is NOT weakly informative; it's deliberately informative because
the naive model `y = BART(X) + N(0, sigma^2)` with a non-informative
prior on sigma has a soft identifiability problem (a larger tree fit
trades off against a smaller sigma, and neither is pinned down by the
likelihood alone). Without this calibration, sigma can drift anywhere
that BART's under-fit bias permits.

Follow the exact same **decision flow** as for any other parameter:
- Present BART's calibrated IG as option (b) "Default"
- Offer option (c) literature-informed / alternative (HalfNormal, Gamma, etc.)
- Still ask the user — don't silently force it
- But DO tell the user why it's the recommendation (identifiability)

The block sampling this is still a standard `nuts_block`; only the
prior inside the log-density changes. See `examples/BartNoise.cpp` for
the reference template, and `skills/block_catalogue.md` for the bart_block
usage recipe.

## Literature-informed prior elicitation

When the user picks option (c) in the prior decision flow (see
`codegen.md §2`), use WebSearch and/or PubMed tools to find
domain-relevant studies that report effect sizes, parameter ranges, or
meta-analyses. Follow this workflow (inspired by
[Riegler et al. 2025, Sci. Rep.](https://doi.org/10.1038/s41598-025-18425-9)):

1. **Search**: query for the parameter's domain context (e.g. "typical
   regression coefficients BMI heart disease meta-analysis").
2. **Extract**: identify reported effect sizes, confidence intervals, or
   typical parameter ranges from the search results.
3. **Propose two prior sets**: generate at least two alternatives:
   - A **moderately informative** prior centered on the literature values
     with a standard deviation reflecting the reported uncertainty.
   - A **weakly informative** prior with wider tails as a conservative
     fallback.
4. **Present to user**: show both proposals with the source references,
   your reasoning, and a recommendation. Let the user pick.
5. **Report confidence**: be transparent about limitations — LLM-suggested
   priors tend toward overconfidence in the width (Riegler et al. 2025
   found moderately informative priors were often worse than weakly
   informative ones). When in doubt, widen the prior.

Example interaction:

> User: "y ~ N(mu + beta * BMI, sigma^2), no prior on beta"
>
> Claude: "I searched for typical BMI-outcome associations. Based on
> [Smith et al. 2020, JAMA], a 1-unit BMI increase is associated with
> a 0.5-2.0 unit change in outcome Y. I suggest:
>   (a) Moderately informative: beta ~ Normal(1.0, 1.0)
>   (b) Weakly informative: beta ~ Normal(0, 10)
> I recommend (b) unless you have strong domain knowledge, since
> literature-derived priors can be overconfident. Which do you prefer?"

## Exposing hyperparameters as constructor arguments

When the user picks option (b) "expose as arguments" in the prior
decision flow, add the hyperparameters to the constructor signature
and store them in `shared_data_t` so the log-density lambda can read
them from `ctx`:

```cpp
// Constructor: pass hyperparameters
MyModel(const arma::vec& y, double prior_mu_sd, double prior_sigma_sd, int seed)

// Store in shared_data
impl_->data().set("prior_mu_sd", arma::vec{prior_mu_sd});

// Read in log-density
const double prior_sd = ctx.at("prior_mu_sd")[0];
lp += -0.5 * mu * mu / (prior_sd * prior_sd);
```

---

## 3a. Discrete-variable decision tree (RUN THIS BEFORE Workflow §4)

If the user's model has ANY discrete latent variable (a variable
$z$ that takes values in a finite / countable set, NOT a continuous
real / simplex / interval), classify it via this tree and follow
the branch. **You MUST NOT improvise a new pattern.** The
architectural reasons live in `skills/system_design.md` §11 —
you don't need to understand the math, just follow the rule.

```
Discrete latent z found?
└── z's structure ─────────────────────────────────────────────────

    CLASS 1: per-observation K-way, {z_i} conditionally independent
             given continuous θ.  Examples: Gaussian mixture,
             latent-class, ZIP / ZINB, mixture regression, LCA.
    → USE `categorical_gibbs_block` (or `binary_gibbs_block` if K=2).
      π can go in `dirichlet_gibbs_block` (conjugate) or nuts_block
      on simplex. Component params (μ, σ, β) go in nuts_block.
      z is sampled directly; it appears in get_current() / get_history().
      Do NOT marginalize z.

    CLASS 1b: BINARY outcome y_i with PROBIT link via Albert-Chib (1993)
              data augmentation. Latent z_i is CONTINUOUS, not discrete,
              but is structurally the "discrete-augmenting" variable.
              Examples: probit linear regression, probit BART, probit
              GLMs with shrinkage / hierarchical structure.
    → USE `probit_aug_block` for the latent z (Exception 3 — closed-
      form vector conjugate sample). Compose with the appropriate
      Gaussian-likelihood block for the mean structure:
        - linear mean (β):    nuts_block on β with N(0, prior_sd^2) prior;
                              see `examples/ProbitRegression.cpp`
        - BART mean f(X):     bart_block with `cfg.binary = true` so the
                              leaf prior tau matches BART::pbart's
                              `3 / (k * sqrt(ntrees))`
        - GP latent f:        elliptical_slice_sampling_block reading
                              z as Gaussian "data"
      sigma is FIXED at 1 (probit identifiability) — `probit_aug_block`
      hardcodes this; do NOT add a sigma block.

    CLASS 2a: continuous spike-and-slab. Both components have positive
              Gaussian variance (tight spike τ_0 > 0 plus wider slab τ_1).
    → USE `binary_gibbs_block` for γ_j + nuts_block for β_j.
      Gibbs is valid (both Gaussians have positive density everywhere).
      When τ_0/τ_1 < 0.01 (VERY tight spike), the Gibbs chain mixes
      slowly; in that specific case you may marginalize γ inside β's
      NUTS lambda using a log-sum-exp on the 2-component prior. Only
      do this if the user explicitly asks or τ_0 is pathologically
      tight. Default: plain Gibbs.

    CLASS 2b: Dirac spike-and-slab (β_j is EXACTLY 0 when γ_j = 0;
              point-mass component, not just a tight Gaussian).
    → *** SUPPORTED via `rjmcmc_block` v0. ***
      Use `examples/SpikeSlabRJMCMC.cpp` as the reference template.
      Model — Ishwaran & Rao 2005 JASA sigma-scaled slab form:
        beta_j | gamma_j=1, sigma, tau ~ N(0, sigma^2 * tau^2)
      (tau is dimensionless, signal-to-noise-ratio scale; sigma and
       tau are posterior-decorrelated under this parametrization →
       clean NUTS geometry on both.)
      Structure (see §2a, §2b):
        - pi:    beta_gibbs_block (Exception 3 — scalar textbook
                 conjugate; has library parity test Check #15)
        - sigma: nuts_block with Jeffreys p(sigma) ∝ 1/sigma
        - tau:   nuts_block with Jeffreys p(tau) ∝ 1/tau, with
                 k=0 fallback to half-Normal(0,1) pin (see §2a's
                 "ESCAPE" rule — tau is dimensionless so 1 is the
                 natural reference scale)
        - (gamma, beta): rjmcmc_block with hand-written Gibbs in
                 continuous_update hook (Exception 2 — kernel
                 contract; has per-usage parity test Check #15)
      gamma_init: marginal-OLS screening — set gamma[argmax_j |X_j'y|]
      = 1. Guarantees k >= 1 from iteration 1, avoiding the improper-
      posterior transient under Jeffreys on tau.
      Constructor takes only (X, y, a_pi, b_pi, seed, keep_history) —
      no hyperparameters on sigma/tau thanks to scale-invariant
      Jeffreys priors.
      Do NOT attempt:
        - `binary_gibbs_block` + fixing β=0 when γ=0: state space
          $\{(0,0)\} \cup \{1\}\times\mathbb{R}$ is not a fixed-dim
          manifold; Gibbs is NOT irreducible.
        - marginalizing γ + NUTS on β: marginal prior is a mixed
          Lebesgue + atomic measure; NUTS silently samples the slab
          only, producing a wrong posterior (R-hat / LOO still look
          clean — silent correctness bug).
      Critical pattern for good mixing: supply the `continuous_update`
      hook to rjmcmc_block_config (Gibbs from conditional posterior
      of beta_j | gamma_j=1). Closed-form for linear Gaussian
      models — see spike_slab_continuous_update in
      SpikeSlabRJMCMC.cpp. Without this, beta_j is stuck at its
      birth-time value and sigma^2 is inflated.
      See system_design.md §10.2 for the Jacobian story (identity-
      coordinate means no user Jacobian even here).
      If the user prefers continuous relaxation or horseshoe, those
      alternatives are still perfectly valid and simpler — propose
      them as options (a), (b); v0 rjmcmc_block is option (c).
      **Statically enforced by validator Check #25**: once a model is
      classified Class 2b (Dirac spike), the generated sampler MUST use
      `rjmcmc_block` for (gamma, beta). Check #25 flags the two ways this
      goes wrong — `binary_gibbs_block`/fix-beta=0 (reducible: the gamma
      indicator ratchets to all-in and FREEZES — runtime-detectable) and
      marginalize-gamma-into-a-NUTS-lambda (NOT detectable at runtime: it
      silently samples the slab only, R-hat/LOO stay clean). The second is
      the reason a STATIC check is required, not just runtime diagnostics.

    CLASS 3: Markov-structured discrete (HMM z_{1:T}, change-point,
             regime-switching, anything where z_t depends on z_{t-1}).
    → *** SUPPORTED via `hmm_block`. ***
      Use `examples/HMMGaussian2State.cpp` as the reference template —
      2-state Gaussian-emission HMM demonstrating the block with
      fixed A / pi / emission params. For a full Bayesian HMM where
      A / pi / emission params are also sampled, add sibling blocks:
        - A rows: `dirichlet_gibbs_block` per row
        - pi     : `dirichlet_gibbs_block`
        - emission means / variances: `nuts_block` (Jeffreys, §2a)
        - z      : `hmm_block` (this entry)
      Exact O(T*K^2) forward-filter backward-sample (Fruhwirth-Schnatter
      2006 Ch. 11). Per-site Gibbs would mix catastrophically on the
      coupled latent z_{1:T}; FFBS samples the full sequence jointly.
      Parity test: `tests_autodiff/test_hmm_block.cpp` verifies
      empirical marginals against analytical Baum-Welch smoothing
      (max_abs_err < 0.2% on 10k draws).

    CLASS 4: model-size unknown (# mixture components, # change points,
             # basis functions). Parameter DIM itself varies.
    → Two sub-branches depending on whether the problem is a BNP
      mixture (Dirichlet Process / Pitman-Yor / HDP):
        (a) BNP mixture — TRUNCATED stick-breaking infrastructure
            SHIPPED 2026-05-02. Use the composition pattern from
            `DPGaussianMixture.cpp` / `PYGaussianMixture.cpp` /
            `DPGaussianMixture_DerivedAlpha.cpp`:
              - z          : `categorical_gibbs_block`
              - pi         : `stick_breaking_block` (DP / PY / custom
                              via user-supplied a_fn / b_fn)
              - cluster_params : `normal_gamma_cluster_gibbs_block`
                                 (diagonal Gaussian) — or future
                                 `niw_cluster_gibbs_block` for full cov
              - alpha      : `nuts_block` on log(alpha) with Gamma prior
              - alpha derived: `register_refresher("alpha", ...)`
                              (DPGaussianMixture_DerivedAlpha.cpp)
            CRP-marginal Neal Alg 2/3/8 and Jain-Neal split-merge are
            NOT shipped; they remain future extensions. Truncated SBP
            mixes well for moderate-N problems and avoids the discrete-
            partition state-space changes.
        (b) Non-BNP (unknown # change points / basis functions /
            submodel structure) — `rjmcmc_block` v0 (identity-
            coordinate) covers most cases. Use SpikeSlabRJMCMC.cpp
            as template; adapt log_joint, propose_sample,
            continuous_update for the new model.
      If the bijection is non-identity (e.g. Stephens 2000 split-
      merge with parameter rescaling), wait for v0.5 / v1 rjmcmc
      extensions (todo T5 / T6).

    CLASS 5a: Bayesian-network structure learning (categorical /
              discrete data, DAG topology G is the latent of interest,
              conditional probability tables θ marginalised analytically).
    → *** SUPPORTED via `order_mcmc_block` (SHIPPED 2026-05-31). ***
      Use `examples/OrderMCMCBN.cpp` as the reference template; full
      end-to-end demo in `examples/OrderMCMCBN.cpp` + audit in
      `tests/audit_OrderMCMCBN_vs_BiDAG.R`. Scope (v1.2):
        - Discrete categorical observations (no BGe / Gaussian / mixed)
        - n ≤ 64 variables (FK 2003 §4.2 64-bit parent-set bitmask)
        - max_parents k_max ≤ 5 default (tractability cliff at k≈7-8)
        - BDeu scoring (Heckerman-Geiger-Chickering 1995 + Buntine 1991)

      **MANDATORY clarification — DAG prior P(G).** order_mcmc_block
      exposes `cfg.use_structure_prior` (default TRUE) which selects
      between two distinct priors on G. These give MEASURABLY
      DIFFERENT posteriors — silent disagreement is a real risk if the
      user expected one and the codegen picks the other. ALWAYS ask:

      ```
      AskUserQuestion: "DAG prior P(G) on the Bayesian-network
      structure — which form?"
        (a) Uniform DAG prior — every DAG (with in-degree ≤ k_max)
            carries equal prior weight P(G) ∝ 1. Matches BiDAG
            (edgepf=1), bnlearn, and most reference BN packages.
            → cpp: `cfg.use_structure_prior = false`
        (b) Friedman-Koller 2003 per-family balancing — penalises
            high fan-in, P(G) ∝ ∏_j 1/C(p−1, |Pa_j|). FK paper's
            preferred default; argued to reflect domain priors better
            for empirical BN learning. → cpp: `cfg.use_structure_prior
            = true`
      ```

      If the spec says "Uniform DAG prior" (e.g. `G ~ Uniform(DAGs)`)
      this maps to (a). If the spec says "Friedman-Koller prior" or
      "per-family balancing" it maps to (b). If unstated, ASK — do
      NOT default to the library flag's default (TRUE = FK Eq 2)
      without confirming, because the flag NAME suggests "structure
      prior" but the FALSE state IS itself a structural prior (the
      uniform one) — the naming is unfortunately ambiguous.

      Document the chosen value with an inline cite in the generated
      .cpp wrapper, e.g.:
        `cfg.use_structure_prior = false;   // per spec: uniform P(G)`

      *Known limitations* (deferred to v1.2.1; see project roadmap):
        - Kuipers-Moffa 2017 partition_mcmc_block (removes FK §4.1
          induced-structure-prior bias inside Markov equivalence
          classes)
        - BGe Gaussian score (continuous data)
        - Mixed conditional-Gaussian BN (Lauritzen 1992)
        - Edge-specific structural prior
        - Tempered / parallel-tempered chains

    CLASS 5b: Ising / MRF / undirected graphical model structure
              learning over discrete states.
    → *** NOT SUPPORTED. *** Action: decline and explain.
      Deferred — no `ising_block` / `mrf_block` shipped in v1.2.

    LARGE K (K > 100) per observation:
    → Gibbs per-obs sampling is still valid but slow. Use
      `categorical_gibbs_block`; warn user about per-sweep cost
      O(NK). No marginalization shortcut available in the framework.
```

**Key operational rules:**
- Class 2b IS supported via rjmcmc_block; always supply a
  `continuous_update` for good mixing.
- Class 3 IS supported via `hmm_block` (FFBS, O(T·K²)).
- Class 5a (Bayesian-network structure learning) IS supported via
  `order_mcmc_block`; **MUST `AskUserQuestion` on the DAG prior**
  (uniform vs FK Eq 2) — do NOT default to library flag default
  without confirming spec intent (the flag's name is ambiguous).
- Class 5b (Ising / MRF) declined — no library support in v1.2.
- Class 4 splits into BNP (truncated SBP via `stick_breaking_block` +
  `normal_gamma_cluster_gibbs_block`; see `DPGaussianMixture.cpp` /
  `PYGaussianMixture.cpp` / `DPGaussianMixture_DerivedAlpha.cpp`) vs
  non-BNP (`rjmcmc_block` v0 covers most).

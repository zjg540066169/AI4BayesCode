---
name: AI4BayesCode-blocks
description: |
  Reference catalogue of all block types in AI4BayesCode. Consult this when
  choosing which block to use for a parameter, or when configuring a block.
---

# Block catalogue

**License note.** The entire AI4BayesCode project (including every block
header listed here and every example that uses them) is distributed
under **GPL-3.0-or-later**. The "(GPL-2.0+)" tags next to
`bart_block` / `genbart_block` below are a legacy reminder that those
blocks transitively pull in the GPL-licensed BART / genBART tree
kernels under `bart_pure_cpp/src/BART` and `bart_pure_cpp/src/GENBART`;
the project-wide license is the same regardless of which block you use.
See `LICENSE` / `THIRD_PARTY_LICENSES.md` at the repo root.

## Contributed blocks — search local + downloaded BEFORE core

The blocks in THIS file are the **core** tier. Two more tiers may add blocks that
are ALSO usable in a generated sampler (both are already on the compile `-I` path,
so `#include "<Block>.hpp"` resolves):

| Tier | Where | Trust |
|---|---|---|
| **local** | `blocks_local/<Block>/` — your work-in-progress blocks | self |
| **downloaded** | `blocks_download/<Block>/` — installed from the registry | provenance |
| **core** | this file | maintainer-vetted |

**Priority when several fit: local > downloaded > core** (prefer the user's own /
reviewed block over a generic core one). Block names are globally unique (core
names are reserved; local + downloaded are checked at creation / install), so
there is never a name collision to resolve.

**Two-stage, token-bounded selection** — do this BEFORE settling on a core block;
it stays cheap no matter how many blocks are installed (bounded by relevance, not
by block count):

1. **Index (cheap).** Build a one-line-per-block index from the manifests:
   `glob blocks_local/*/manifest.dcf`, then `glob blocks_download/*/manifest.dcf`,
   and read each manifest's `Block` + `RoutingKey` (one short line each). Do NOT
   read the full cards yet. (No local/downloaded blocks present → skip to core.)
2. **Filter.** From that index (local rows first, then downloaded), pick the 5–10
   blocks whose `RoutingKey` plausibly matches the model.
3. **Read cards — survivors only.** For each candidate read its `SelectWhen` and
   full card `blocks_local/<Block>/skills/<Block>.md` (or
   `blocks_download/<Block>/skills/<Block>.md`); choose the best fit.
4. If NO local / downloaded block fits, use a **core** block from the table below.
5. `#include "<Block>.hpp"` for the chosen block (already on the `-I` path).

**When you USE a contributed block, also load its bundle files (from its manifest):**
- **Example** (`Example:` field → `<tier>/<Block>/examples/<Model>.cpp`): read it as
  the concrete reference composition for THIS block — the analogue of a core
  `examples/*.cpp`, showing the exact wiring, `set_current`, `predict_at`, runner.
- **ValidationSkill** (`ValidationSkill:` field → `<tier>/<Block>/skills/<Block>_validation.md`)
  + **`ChecksApplicable:`**: apply these block-specific checks at validation time —
  see `validator.md` ("Contributed-block checks").

**Confirm it.** Surface the selected block + the alternative candidates in the
existing model-confirmation gate (`codegen.md §2` / `start.md` Stage 3) so the user
can switch. (Full design: `block_design_skills/contrib.md`.)

## Block type table

| Parameter kind | Block type | Constraint wrap |
|---------------|-----------|----------------|
| real scalar/vector | `nuts_block` | `constraints::real::wrap` |
| strictly positive (sigma, tau) | `nuts_block` | `constraints::positive::wrap` |
| simplex, non-conjugate | `nuts_block` | `constraints::simplex::wrap` |
| simplex, exact Dirichlet | `dirichlet_gibbs_block` | (none) |
| **probability in (0,1), conjugate Beta** | **`beta_gibbs_block`** | **(none — exact draw)** |
| probability in (0,1), non-conjugate | `nuts_block` | `constraints::interval::wrap` |
| lower-bounded (x > lo) | `nuts_block` | `constraints::lower_bounded::wrap` |
| upper-bounded (x < up) | `nuts_block` | `constraints::upper_bounded::wrap` |
| interval (lo < x < up) | `nuts_block` | `constraints::interval::wrap` |
| ordered K-vector | `nuts_block` | `constraints::ordered::wrap` |
| Cholesky of corr matrix | `nuts_block` | `constraints::cholesky_corr::wrap` |
| unit vector (K-sphere) | `nuts_block` | `constraints::unit_vector::wrap` |
| binary z in {0,1} | `binary_gibbs_block` | (none) |
| **Albert-Chib latent z for any probit binary likelihood** (z_i ~ TN(mu_i + offset_i, 1, sign(2 y_i - 1) > 0)) | **`probit_aug_block`** | **(none — closed-form truncated Normal)** |
| categorical z in {1..K} | `categorical_gibbs_block` | (none) |
| **HMM state sequence z_1:T in {0..K-1}** | **`hmm_block` (FFBS)** | **(none)** |
| **discrete MRF / Ising / Potts on a graph** (`pi(x) ∝ exp{β Σ_{i~j} I[x_i = x_j]}`, x ∈ {0..Q-1}^n) | **`ising_cluster_block`** (Swendsen-Wang 1987 cluster moves) | **(none — bond augmentation + per-cluster recolor)** |
| **sparse-precision Gaussian MRF** (`pi(x) ∝ exp{-½ x^T Q x + b^T x}`, Q sparse PSD, optional sum-to-zero) | **`gmrf_precision_block`** (Rue 2001 sparse Cholesky + AMD) | **(none — direct conjugate draw via Eigen SimplicialLLT)** |
| **sparse-precision GMRF + non-Gaussian likelihood** (`pi(x) ∝ exp{-½ x^T Q x}` · `log_lik(x)`; Poisson / Bernoulli / Student-t / NB / log-Gaussian Cox observation) | **`gmrf_whitened_ess_block`** (Murray 2010 ESS on implicit GMRF prior; Rue 2001 backsolve for prior draws) | **(none — Eigen SimplicialLLT + permuted backsolve + likelihood-free ESS shrink)** |
| **Bayesian-network structure learning** (discrete data, n ≤ 64, BDeu score; output = total order ≺ + Bayesian-model-averaged DAG over Pa(i) ⊂ Pred(i, ≺)) | **`order_mcmc_block`** (Friedman-Koller 2003 order MCMC + Heckerman-Geiger-Chickering 1995 BDeu + FK §4.2 three-tier cache) | **(none — combinatorial MH on permutations)** |
| **LDA token topic assignment z_n in {1..K} + theta_d (M-doc simplex) + phi_k (V-vocab simplex), Dirichlet hyperpriors** | **`lda_collapsed_gibbs_block`** (Griffiths-Steyvers 2004 collapsed Gibbs) | **(none — joint output of z, theta, phi)** |
| **Bernoulli response (logistic reg)** | **`pg_logistic_block` (PG augmentation)** | **(none)** |
| Gaussian mean f (BART) | `bart_block` | (none, GPL-2.0+) |
| **Any likelihood with log_f + score + obs_info (Poisson / NB / logistic / heteroscedastic / AFT / beta / gamma_shape / beta_binomial / custom)** | **`genbart_block` + `genbart::lik::*`** | **(none, GPL-2.0+)** |
| **Poisson-multinomial gamma augmentation (log_phi ~ Gamma)** | **`poisson_multinomial_aug_block`** | **(none)** |
| **tightly-coupled real parameters (shift-invariance, additive linear mean, fixed+random effect)** | **`joint_nuts_block`** | **(none — identity; current scope is real only)** |
| **hierarchical random effects** (`u_i ~ N(mu_u, tau)` with group-level `mu_u`, `tau`) — Gaussian, binomial, Poisson, or any GLMM family | **`joint_nuts_block`** with NC reparameterization (read `skills/hierarchical_re.md` BEFORE coding) | **(positive constraints on `tau, sigma_y`; real on `mu_u, u_raw, beta`)** |
| **truncated stick-breaking simplex (DP / PY / HDP weights)** | **`stick_breaking_block`** | **(none — per-stick Beta gamma-trick)** |
| **diagonal-Gaussian cluster (mu_k, lambda_k) with conjugate Normal-Gamma prior across K_trunc clusters** | **`normal_gamma_cluster_gibbs_block`** | **(none — per-cluster, per-dim conjugate)** |
| **full-covariance Gaussian cluster (mu_k, Sigma_k) with conjugate NIW prior across K_trunc clusters** | **`niw_cluster_gibbs_block`** | **(none — per-cluster Bartlett decomposition)** |
| **scalar positive with conjugate Gamma posterior (DP α under truncated SBP, Normal-Gamma marginal precision, …)** | **`gamma_gibbs_block`** | **(none — exact draw via gamma-trick)** |
| **cluster-partition split / merge MH proposal (accelerated mixing for DPMM)** | **`split_merge_block`** | **(none — Jain-Neal 2004)** |
| **discrete categorical latents `z_i in {0..K_i-1}` (general n-variable mean-field VI)** | **`mean_field_categorical_vi_block`** (Bishop §10.1 + RAABBVI) | **(none — internal anchored softmax)** |
| **discrete latents with user-specified CLIQUE partition (intra-clique joint, inter-clique factorised — refines Block 4)** | **`structured_categorical_vi_block`** (Saul-Jordan 1996 + RAABBVI) | **(none — per-clique anchored softmax)** |

### When to use beta_gibbs_block vs NUTS for (0,1) parameters

⚠️ **READ `skills/codegen_priors.md` §2b "Block selection priority" first.**
The default for continuous parameters is `joint_nuts_block` (it collects
the continuous parameters NOT claimed by a specialized / structural block);
single `nuts_block` is LOW priority. `beta_gibbs_block` is a LAST-RESORT
option that must be justified by an Exception from §2b. Misuse carries silent-correctness risk (wrong `params_fn`
derivation → wrong posterior that passes all MCMC diagnostics).

**Correct use case — Exception 3 (scalar textbook conjugate with
NUTS-wasteful efficiency profile):**

- Spike-and-slab mixing proportion: `gamma_j ~ Bernoulli(pi)`, `pi ~ Beta(a, b)`
  → `pi | gamma ~ Beta(a + sum(gamma), b + p - sum(gamma))`
  Scalar parameter with only a length-p sufficient statistic
  (Σ gamma). NUTS warmup on 1-D tight-posterior scalar is
  wasteful → `beta_gibbs_block` is justified.
- Beta-Binomial: `y ~ Binomial(n, p)`, `p ~ Beta(a, b)`
  → `p | y ~ Beta(a + y, b + n - y)`. Similar scalar conjugate.

**Correctness obligations (Checks #15, #16 from codegen_priors.md):**

1. The block's sampling mechanism is covered by the library-level
   parity test
   `tests_autodiff/block_tests/test_beta_gibbs_block.cpp` (Option A).
2. The `params_fn` you write must match a textbook pattern above;
   if it has any conditional logic (e.g., `if gamma[j] == 1`) or
   active-subset indexing, you ALSO need a per-usage parity test.
3. Inline justification comment (Check #16) required at the block
   construction site, stating which Exception applies.

Prefer `nuts_block` with `constraints::interval::wrap(0, 1)` when:
- The conditional is NOT exactly Beta (e.g. p enters a logistic link,
  a nonlinear function, or has non-conjugate factors)
- You're not sure the conditional is exactly Beta (when in doubt,
  NUTS)
- The parameter is not scalar / not tight-posterior (NUTS is faster
  on vectors even in conjugate cases)

## nuts_block

```cpp
nuts_block_config cfg;
cfg.name        = "<shared_data key>";
cfg.initial_unc = arma::vec{...};        // unconstrained starting point
cfg.constrain   = constraints::positive::constrain;   // omit for real
cfg.unconstrain = constraints::positive::unconstrain;  // omit for real
cfg.log_density_grad = /* lambda */;
// For simplex/ordered/cholesky_corr:
cfg.initial_step_size   = 0.05;
cfg.n_warmup_first_call = 400;
```

### Configuration discipline

The `nuts_block_config` exposes several knobs that affect MCMC correctness;
not all are free for code-generation to set. The table below is the
authoritative codegen-time guide.

| Field | Code-gen may set? | Allowed values | Notes |
|---|---|---|---|
| `name` | YES | any unique string within the composite | shared_data key the block writes to |
| `initial_unc` | YES | `arma::vec` of correct dim | unconstrained starting point; from data-driven init (OLS, MOM) when possible |
| `constrain` / `unconstrain` | YES | one of the `constraints::<kind>::*` pairs | must match the parameter's natural-space constraint |
| `log_density_grad` | YES | `[](theta_unc, ctx, grad){ return constraints::<kind>::wrap(...); }` | natural-scale lp + grad inside the wrap; BLAS per `codegen_cpp.md` §6.1 |
| `n_warmup_first_call` | YES | 200 (default) - 3000 | longer first-call warmup is the right escape valve when initial adaptation needs more runway; bumping this is **always** preferred over per-step adaptation |
| `initial_step_size` | YES | 0.01 - 0.1 | only set explicitly for simplex / ordered / cholesky_corr where the default 1.0 is too aggressive |
| `n_draws_per_step` | NO | leave at default 1 | larger values thin-but-recompute; not needed for outer-Gibbs composition |
| **`n_warmup_per_step`** | **NEVER** | **leave at default 0** | see "n_warmup_per_step is mandatory 0" below |

### `n_warmup_per_step` is mandatory 0

**Code-gen agents must NOT set `cfg.n_warmup_per_step` to any value other
than the default 0.** Do not write `cfg.n_warmup_per_step = 5` or any
other non-zero value, regardless of any "5-8% variance bias acceptable"
language an older comment in `nuts_block.hpp` might suggest.

**Why.** Setting `n_warmup_per_step > 0` re-enables a chain-state
corruption mechanism that the 2026-04-12 mcmclib bugfix removed.
Per-step adaptation re-runs dual averaging against the same draws being
kept, violating detailed balance and progressively shrinking the step
size into a neighborhood the chain cannot leave. **The runtime symptom
is the "stuck-fast" pattern:** sim1 reports `rhat_max ≈ 2.2`,
`ess_bulk_AI = NA / single digits`, and AI walltime SHORTER than Stan's
(stuck chains do almost no tree exploration). L3 single-dataset checks
PASS because the agent's calibration init lands in a friendly region;
sim1 cross-dataset checks FAIL catastrophically.

**Common temptation: "the sigma block keeps rejecting at default 0".**
This is a real symptom but the actual problem is a non-centered
hierarchical funnel between `(sigma_*, raw_effect_*)` — the scale and
its raw-effect partner are in separate blocks but their conditionals are
multiplicatively coupled. The correct fix is methodological:
1. **`joint_nuts_block`** over `(sigma_*, z_*)` per
   `codegen_cpp.md §4a` row "scale + raw effect, non-centered". The
   joint block samples the multiplicative coupling exactly.
2. **Bump `n_warmup_first_call`** to 1500-3000 so the first-call
   adaptation lands in the typical set.
3. **Better init values** via OLS / method of moments in the wrapper
   constructor.

Use one or more of (1)-(3). Do NOT escape to `n_warmup_per_step > 0` —
that produces silently wrong posteriors that pass L3 while failing sim1.

The same reasoning applies to other blocks where the AI might be tempted
to "patch up rejection by re-adapting": same answer applies (joint
block, longer first-call warmup, better init).

## dirichlet_gibbs_block

**Reference example:** `examples/FiniteGaussianMixture.cpp` (shipped
2026-05-03) uses `dirichlet_gibbs_block` for the symmetric Dirichlet
posterior on the K-mixing-weight vector π. The DirichletSimplex
example uses `nuts_block` + `constraints::simplex::wrap` instead — that
is the more general option for non-conjugate factors on the simplex.
`dirichlet_gibbs_block` is also the right choice for the "A rows / pi"
of a full-Bayesian HMM (mentioned in `HMMGaussian2State.cpp`'s header
comments; the minimal 2-state example ships with fixed A / pi so it
doesn't actually use this block).
Codegen agents targeting Dirichlet-Categorical / LDA should compose
this block inline.

Exact iid draws via gamma-normalization. Use when the full conditional is
a clean Dirichlet (Dirichlet-Categorical, LDA proportions, etc.).

```cpp
dirichlet_gibbs_block_config cfg;
cfg.name           = "theta";
cfg.n_categories   = K;
cfg.initial_values = arma::vec(K, arma::fill::value(1.0 / K));
cfg.alpha_post_fn  = [](const block_context& ctx) -> arma::vec {
    return ctx.at("alpha") + counts;  // posterior concentration
};
```

**Do NOT use** when theta has non-conjugate factors. Fall back to
`nuts_block` + `constraints::simplex::wrap`.

### R-hat caveat on near-zero simplex components

`dirichlet_gibbs_block` is exact, so R-hat on each theta component is
clean regardless of posterior mass. But for **`nuts_block` +
`constraints::simplex::wrap`** (e.g. the Linero / sparse-Dirichlet
pattern in `DirichletSparse`), any component with near-zero posterior
mean is susceptible to a well-known R-hat artifact: the sampler
explores the flat/unidentified tail of the unconstrained stick-
breaking coordinate, and on the natural scale both chains produce
"effectively zero" values (e.g. 1e-6 in one chain, 1e-10 in another)
that give a legitimate R-hat of 2+ because *between-chain variance /
within-chain variance is numerically large* — even though both chains
agree the component is 0. Always report per-component R-hat, not just
the max; the max is dominated by unidentified / data-less components
in sparse-Dirichlet settings.

## beta_gibbs_block

Exact draw from Beta(alpha, beta) via the Gamma trick. Use when the full
conditional is exactly Beta — typically mixing proportions in
spike-and-slab, or probabilities in Beta-Binomial models.

```cpp
#include "AI4BayesCode/beta_gibbs_block.hpp"

beta_gibbs_block_config cfg;
cfg.name = "pi";
cfg.initial_value = 0.5;
cfg.params_fn = [](const block_context& ctx) -> beta_dist_params {
    const arma::vec& gamma = ctx.at("gamma");
    double a = ctx.at("a_pi")[0] + arma::sum(gamma);
    double b = ctx.at("b_pi")[0] + gamma.n_elem - arma::sum(gamma);
    // Beta(a, b), E[pi] = a / (a + b)
    return {a, b};
};
```

**Do NOT use** when the conditional is not exactly Beta. Fall back to
`nuts_block` + `constraints::interval::wrap(0, 1)`.


## binary_gibbs_block

**Reference example:** NONE shipped — library-only. The classical use
case (continuous spike-and-slab with binary indicator γ_j) is
intentionally covered in `examples/SpikeSlabRJMCMC.cpp` via the
`rjmcmc_block` (Dirac spike-and-slab is the geometry the project
targets), not via `binary_gibbs_block`. If a codegen agent needs the
continuous-spike Class 2a pattern (continuous relaxation instead of
Dirac), it should compose `binary_gibbs_block` + `nuts_block` inline
following the header-comment pattern in `binary_gibbs_block.hpp`.

Closed-form Bernoulli for z in {0,1}. Provide `log_odds_fn(ctx) -> arma::vec`.
Use for spike-and-slab, variable selection.

**Target-geometry caveat:** `binary_gibbs_block` is the right tool for
*continuous* spike-and-slab (both spike and slab have positive Gaussian
density) and for plain binary indicators. It is **NOT** the right tool
for **Dirac** spike-and-slab (β_j = 0 exactly when γ_j = 0) — that
target has a dimension-changing state space and Gibbs on it is not
irreducible. See `skills/system_design.md` §11 and
`skills/codegen_priors.md` §3a. For Dirac spike-and-slab, use the
`rjmcmc_block` identity path (see the rjmcmc_block entry below and
`examples/SpikeSlabRJMCMC.cpp` reference template). Alternatively,
a continuous relaxation (tight-Gaussian-spike) with `nuts_block` on
beta works too when the point-mass is not semantically required.

```cpp
binary_gibbs_block_config cfg;
cfg.name     = "gamma";
cfg.n_binary = p;
cfg.initial_values = arma::vec(p, arma::fill::zeros);
cfg.log_odds_fn = [](const block_context& ctx) -> arma::vec { ... };
```

## probit_aug_block

**Reference example:** `examples/ProbitRegression.cpp` (Bayesian probit
linear regression via Albert-Chib + NUTS-on-beta). The same
Albert-Chib augmentation pattern composes naturally with `bart_block`
when `cfg.binary = true` (probit BART), but no shipped example
combines them — see `bart_block`'s "Binary mode" docstring for the
recipe.

Closed-form Gibbs leaf for the **Albert-Chib (1993)** data-augmentation
latent z in any probit-link binary likelihood:

```
y_i        ~ Bernoulli(p_i),  p_i = Phi(mu_i + offset_i)
z_i | rest ~ N(mu_i + offset_i, 1) truncated to:
                 (0, +inf)   if y_i = 1
                 (-inf, 0)   if y_i = 0
```

Compose with any Gaussian-likelihood downstream block (the downstream
block sees `z` — or `z - offset` if you bake the offset out — as its
working response). Standard pattern:

```
composite "ProbitWhatever":
  child(0) z      probit_aug_block (this block)
  child(1) <mean> nuts_block / bart_block (binary=true) /
                   linear_block / GP / hierarchical / ...
```

**Why it exists.** Albert-Chib augmentation is *the* textbook closed-
form Gibbs step for probit. Before this block shipped, probit examples
had to inline the truncated-normal step in the wrapper's `step()`
method, breaking uniformity. `probit_aug_block` makes the Gibbs leaf
library-blessed (Exception 3 of `codegen_priors.md` §2b — closed-form
vector conjugate sample, NUTS-wasteful for N independent truncated
normals).

**Sigma is FIXED at 1.0** by probit identifiability. There is no config
knob for it. Anyone wanting `sigma != 1` is targeting a different
likelihood (Tobit / censored Gaussian) and should use a different
block.

```cpp
probit_aug_block_config cfg;
cfg.name        = "z";          // shared_data key for output z (length N)
cfg.n_obs       = N;
cfg.y_key       = "y";          // length-N {0, 1}
cfg.mu_key      = "mu_lin";     // length-N linear predictor (Gaussian
                                //   block writes this — bart_block's
                                //   "f_bart", or a refresher computing
                                //   X*beta, etc.)
cfg.offset_key  = "";           // optional; "" = no offset; else scalar
                                //   (length 1) or per-obs (length N)
cfg.initial_z   = arma::vec();  // optional warm start; default snaps to
                                //   sign(2 y - 1)
```

**Conditional independence (no sequential update needed).** Each `z_i`
depends only on `(y_i, mu_i, offset_i)`. The block draws all `z_i`'s
in a single vectorised pass per `step()` — no inner loop dependency,
no need to refresh the context mid-sweep. (Contrast with
`binary_gibbs_block` where sequential update IS required.)

**Composition with bart_block for probit BART.** Set `cfg.binary = true`
on the sibling `bart_block` so its leaf-prior tau matches BART::pbart's
`3 / (k * sqrt(ntrees))` formula. Without that, `bart_block` falls
back to the Gaussian formula based on the working-response range and
over-shrinks the leaves by a factor of 3. See `bart_block_config::binary`
in this catalogue.

**Check #15 parity test:**
`tests_autodiff/block_tests/test_probit_aug_block.cpp` — verifies the
empirical mean of drawn z's against the closed-form `TN(mu, 1, [a, b])`
expectation across 5 regimes (centred / shifted positive / shifted
negative / mixed-y vector with scalar offset / per-obs offset
broadcast) at 5% relative tolerance, 10000 draws each.

## categorical_gibbs_block

**Reference example:** NONE shipped — library-only. A finite-mixture or
LDA example using `categorical_gibbs_block` + `dirichlet_gibbs_block`
is on the future roadmap (todo T7-T9: DP mixture via Neal Alg 2/3/8).
For now, codegen agents targeting Class 1 mixture / LCA / LDA should
compose this block inline following the header-comment pattern in
`categorical_gibbs_block.hpp`.

Per-observation closed-form Gibbs for z in {1..K}. Provide
`log_probs_fn(ctx) -> arma::mat(N, K)` of unnormalized log-probs.
Use for mixture labels, LDA assignments.

**Target-geometry caveat:** valid only when the {z_i} are
**conditionally independent** across i given the continuous
parameters (the standard Class 1 assumption — mixture models, LCA,
ZIP). It is **NOT** valid for Markov-structured latents (HMM
z_{1:T}, change-point) — per-site Gibbs mixes catastrophically
slowly on those. See `skills/system_design.md` §11 and
`skills/codegen_priors.md` §3a. HMM / state-space support is provided by
`hmm_block` (T10, SHIPPED 2026-04-20) via forward-filter
backward-sample; see the hmm_block entry below.

```cpp
categorical_gibbs_block_config cfg;
cfg.name         = "z";
cfg.n_obs        = N;
cfg.n_categories = K;
cfg.initial_labels = arma::vec(N, arma::fill::ones);
cfg.log_probs_fn = [](const block_context& ctx) -> arma::mat { ... };
```

## lda_collapsed_gibbs_block

**Reference example:** `examples/LdaCollapsedGibbs.cpp` (shipped 2026-05-08).
Use this block for any LDA-style model: token-level latent topic
assignment `z_n ∈ {1..K}` paired with per-document Dirichlet topic
proportions `θ_d` and per-topic Dirichlet word distributions `φ_k`,
with FIXED-K and FIXED Dirichlet hyperparameters `α, β`. Replaces
the naïve composition `categorical_gibbs_block(z) + dirichlet_gibbs_block(θ_d) × M + dirichlet_gibbs_block(φ_k) × K`,
which is correct in the limit but mixes catastrophically because of
the strong cross-block coupling between (z, θ, φ) — this is the
canonical §11.2(b) "discrete target with strong local dependence"
case where a specialized algorithm is mandatory.

The block samples z via the Griffiths & Steyvers (2004) collapsed
Gibbs update with (θ, φ) integrated out:

  P(z_n = k | z_{-n}, w, α, β)
       ∝ (n_{d,k}^{-n} + α_k)
         × (n_{k,w_n}^{-n} + β_{w_n})
         / (n_{k,*}^{-n} + sum(β))

After the z sweep, θ and φ are sampled directly from their Dirichlet
conjugate posteriors via gamma-normalization (same mechanism as
`dirichlet_gibbs_block`). The block exposes all three under
`current_named_outputs()` so the composite writes z, θ, φ to
shared_data after each sweep.

**Output layout** (column-major flat, recover via `matrix(., M, K)`
or `matrix(., K, V)` in R):
- `z`     : length-N integer vector of topic assignments (1-indexed)
- `theta` : length-(M*K), entry [d + k*M] = θ_{d,k}
- `phi`   : length-(K*V), entry [k + v*K] = φ_{k,v}

**Label switching**: this block does NOT canonicalize topics
internally (system_design.md guarantees deterministic block output
for given seed + init). Apply Stephens 2000 in R-level alignment
code per `skills/label_switching.md` if the reference implementation
also has unconstrained topics.

**JUSTIFICATION (Check #16):** Exception 1 from `codegen_priors.md` §2b
— z is discrete; NUTS cannot target it. Library Check #15 parity
test is at
`tests_autodiff/block_tests/test_lda_collapsed_gibbs_block.cpp`
(5 regimes: count integrity, theta conjugate parity, phi conjugate
parity, cross-impl parity vs in-test reference via label-invariant
statistics, recovery from synthetic truth at K=2 with Hungarian
match).

```cpp
lda_collapsed_gibbs_block_config cfg;
cfg.name           = "lda";
cfg.M              = M;          // number of documents
cfg.V              = V;          // vocabulary size
cfg.K              = K;          // number of topics (FIXED)
cfg.alpha          = arma::vec(K, arma::fill::ones);  // default Dir(1,..,1)
cfg.beta           = arma::vec(V, arma::fill::ones);
cfg.w_key          = "w";        // ctx key for length-N word ids (1..V)
cfg.doc_key        = "doc";      // ctx key for length-N doc ids (1..M)
cfg.z_out_key      = "z";        // output keys for shared_data
cfg.theta_out_key  = "theta";
cfg.phi_out_key    = "phi";
// cfg.z_init left empty -> deterministic (i mod K) + 1 init.
impl_->add_child(
    std::make_unique<lda_collapsed_gibbs_block>(std::move(cfg)));
```

The Tier-A wrapper exposes only the canonical six R-methods. The
block's name (`"lda"`) is a label for Gibbs DAG bookkeeping; it is
NOT a shared_data key (the joint-block pattern from
system_design.md §3 / §13). Downstream consumers read z, theta,
phi via the OUTPUT keys above.

## gamma_gibbs_block

**Reference example:** library-only initially; designed as a drop-in
replacement for `nuts_block` on log(α) inside the BNP examples
(`DPGaussianMixture.cpp` etc.) for users who want exact iid α draws
under the truncated-SBP DP closed-form posterior. The shipped DP
example uses NUTS for simplicity; switching to this block is a 1-2
line edit.

Closed-form Gibbs leaf for a SCALAR positive parameter whose full
conditional is exactly Gamma(shape, rate). Companion / dual of
`inv_gamma_gibbs_block`. Use when the conditional posterior is an
exact Gamma — common cases:

- DP concentration α under truncated stick-breaking with
  `α ~ Gamma(a, b)` prior → posterior `α | V_1, …, V_{T−1}
  ~ Gamma(a + T − 1, b − Σ log(1 − V_k))`.
- Scalar precision λ when the Normal-Gamma joint is integrated to
  the marginal Gamma on λ.
- Any scalar positive parameter with Gamma posterior conjugate.

```cpp
#include "AI4BayesCode/gamma_gibbs_block.hpp"

gamma_gibbs_block_config cfg;
cfg.name          = "alpha";
cfg.initial_value = a_prior / b_prior;
cfg.params_fn = [a_prior, b_prior](const block_context& ctx) {
    const arma::vec& V = ctx.at("stick_V");
    const std::size_t T = V.n_elem;
    double Tsum = 0.0;
    for (std::size_t k = 0; k + 1 < T; ++k) Tsum += std::log(1.0 - V[k]);
    return AI4BayesCode::gamma_params{
        a_prior + static_cast<double>(T - 1),
        b_prior - Tsum
    };
};
```

**Parameterization**: shape-RATE (matches `inv_gamma_gibbs_block`,
R's `rgamma(n, shape, rate)`, JAGS / NIMBLE / Stan). Internally
draws via `std::gamma_distribution<double>(shape, scale = 1/rate)`
which is shape-SCALE in C++. The conversion is commented at the
call site (`rcpp_api.md §11`).

**JUSTIFICATION (Check #16): Exception 3** (scalar textbook conjugate
with NUTS-wasteful efficiency profile).
**Check #15** parity:
`tests_autodiff/block_tests/test_gamma_gibbs_block.cpp` covers three
regimes (small shape, large shape, DP-style closure) at 10 000 draws,
within 5 % mean / 10 % variance tolerance.

**Do NOT use** when the conditional is not exactly Gamma (e.g., α with
non-conjugate likelihood factors). Fall back to `nuts_block` with
`constraints::positive::wrap`.

## inv_gamma_gibbs_block

**Reference example:** NONE shipped, and none planned — this block is
intentionally library-only because it is DISCOURAGED as default (see
below). The preferred pattern for scale parameters is Jeffreys on
`nuts_block` with inline k=0 half-Normal(0,1) fallback; see
`examples/SpikeSlabRJMCMC.cpp` tau block for the reference template.

⚠️ **STRONGLY DISCOURAGED AS DEFAULT.** See `codegen_priors.md` §2a "Variance
/ scale parameter prior discipline (Gelman 2006)": the default prior
for variance/scale is Jeffreys `p(sigma) ∝ 1/sigma` implemented via
`nuts_block` + `constraints::positive::wrap`, NOT InverseGamma.
`Gamma(ε, ε)` as "noninformative" is the specific prior Gelman 2006
Bayesian Analysis 1(3):515-533 refuted.

**Valid use cases** (both require explicit documented justification
inline per Check #16):
- Exception 3 with genuinely heavy-tail-pathological NUTS conditional —
  but first try the Jeffreys + k=0 half-Normal(0, 1) fallback
  pattern (`codegen_priors.md §2a`), which implements weakly-informative on
  natural scale without the IG(ε, ε) critique. If even that pattern
  fails to mix, only then consider IG with documented justification.
- INFORMATIVE IG prior where the hyperparameters come from external
  knowledge (e.g., BART's calibrated IG via `sigest`, where the prior
  is deliberately informative to resolve an identifiability issue).

**If you are tempted to use this block, first reconsider whether your
goal is actually better served by `nuts_block` with Jeffreys
`p(sigma) ∝ 1/sigma` on the natural scale.** Most of the time, yes.
The pattern in `examples/SpikeSlabRJMCMC.cpp` is the canonical
template:
- sigma: `nuts_block` with Jeffreys prior
- tau  : `nuts_block` with Jeffreys prior + k=0 fallback to
  half-Normal(0, 1) pin (inside the natural-scale log-density)
under the Ishwaran & Rao 2005 sigma-scaled slab
(`beta_j | gamma_j=1, sigma, tau ~ N(0, sigma^2 tau^2)`).

Closed-form Gibbs for a scalar positive parameter whose conditional
is InverseGamma(shape, rate). Provide `params_fn(ctx) ->
inv_gamma_params{shape, rate}`.

```cpp
inv_gamma_gibbs_block_config cfg;
cfg.name          = "sigma2";
cfg.initial_value = var(y);
cfg.params_fn     = [N](const block_context& ctx) {
    const arma::vec& y    = ctx.at("y");
    const arma::vec& Xb   = ctx.at("Xbeta_cache");
    const double     a_pr = ctx.at("a_sigma_prior")[0];
    const double     b_pr = ctx.at("b_sigma_prior")[0];
    double sse = arma::accu(arma::square(y - Xb));
    // Conditional: IG(a + N/2, b + SSE/2). E[sigma^2] = rate / (shape-1).
    return inv_gamma_params{ a_pr + N/2.0, b_pr + sse/2.0 };
};
```

**Parameterization:** shape-RATE. To sample, the block draws
g ~ Gamma(shape, 1/rate) internally and inverts.

**When to use:** any scalar with a closed-form InverseGamma
conditional — sigma^2 in Gaussian regression, tau^2 in a slab prior,
any scale parameter with conjugate InvGamma prior. Do NOT use for
vector-valued positive parameters; that's `nuts_block` with
`constraints::positive::wrap` territory.

## pg_logistic_block (T12, Polson-Scott-Windle 2013 PG augmentation)

Bayesian logistic regression via Polya-Gamma data augmentation.
Exact Gibbs: alternates PG(1, X_i' beta) auxiliary draws (library-
internal truncated series, K=128 terms → 1e-8 relative tail bias on
PG mean) with Gaussian β | ω update. 10-100× faster than NUTS-on-
logistic for p < ~1000.

⚠️ **Hard scope limitation: LINEAR LOGISTIC ONLY — NOT LOGISTIC BART.**
PG augmentation's exact-Gibbs advantage depends on the linear
predictor `X_i' beta` being a parametric linear combination. Substituting
a BART mean function f(X_i) breaks both (i) the Gaussian β | ω
conjugacy and (ii) BART's tree-location identifiability (the PG-
augmented pseudo-response κ = y - 0.5 does not anchor BART's
Gaussian-observation tree scale). For logistic BART, use
**`genbart_block` + `genbart::lik::logistic_lik`** (see
`examples/GBartLogistic.cpp`) — the RJMCMC tree kernel handles the
non-conjugate sigmoid likelihood directly via Laplace leaf
proposals, with no augmentation required. Never combine `bart_block` +
`pg_logistic_block`.

**JUSTIFICATION (Check #16): Exception 1** (discrete/augmented
measure; NUTS cannot target PG latent directly). Library-blessed
block: user writes no PG sampling code. Check #15 parity test at
`tests_autodiff/test_pg_logistic_block.cpp` verifies PG(1, z) mean
against analytical (1/(2z)) tanh(z/2) + end-to-end beta recovery
on synthetic logistic regression (p=5, N=500).

Reference template: `examples/LogisticRegression.cpp`.

```cpp
pg_logistic_block_config cfg;
cfg.name = "beta";
cfg.p = p;
cfg.y_key = "y";   // N-vector of 0/1
cfg.X_key = "X";   // N*p column-major flat
cfg.prior_mean = arma::vec(p, arma::fill::zeros);
cfg.prior_cov  = prior_sd*prior_sd * arma::eye<arma::mat>(p, p);
cfg.initial_beta = arma::vec(p, arma::fill::zeros);
```

## hmm_block (T10, forward-filter backward-sample)

Exact FFBS sampler for finite-state Hidden Markov Models. Samples
the latent state sequence z_1:T jointly from its full conditional
p(z_1:T | y, A, pi, theta) via log-space forward filter + backward
sampling. K and T fixed per construction; A (K*K row-major), pi
(length K), and emission log-density are read from context each
sweep (typically sampled by sibling `dirichlet_gibbs_block` for A
and pi, and `nuts_block` / `beta_gibbs_block` for emission theta).

**JUSTIFICATION (Check #16): Exception 1** (discrete state
sequence; NUTS cannot target). Algorithm is Baum-Welch / Fruhwirth-
Schnatter 2006 Ch. 11 standard FFBS. Check #15 parity test at
`tests_autodiff/test_hmm_block.cpp` verifies empirical marginals
P(z_t = k | y) against analytical Baum-Welch smoothing to
max_abs_err < 0.2% (10k draws on K=2, T=5 fixture).

Reference template: `examples/HMMGaussian2State.cpp` (minimal 2-
state Gaussian-emission demo with fixed A, pi, emission params).

```cpp
hmm_block_config cfg;
cfg.name = "z";
cfg.T    = T; cfg.K = K;
cfg.A_key = "A"; cfg.pi_key = "pi";
cfg.emission_logp =
    [](std::size_t t, std::size_t k, const block_context& ctx) {
        const arma::vec& y = ctx.at("y");
        // log p(y_t | z_t = k, theta)
        return ...;
    };
cfg.initial_z = arma::vec(T, arma::fill::zeros);
```

## ising_cluster_block (Swendsen-Wang cluster sampler)

Specialised non-conjugate Gibbs sweep for the Ising / Potts target on
a user-supplied undirected graph:

  pi(x) ∝ exp{β · Σ_{i~j} I[x_i = x_j]},  x ∈ {0..Q-1}^n

via Swendsen-Wang (1987): (i) bond augmentation per like-coloured edge
with probability `1 − exp(−β)`; (ii) union-find cluster identification;
(iii) per-cluster uniform recolour over ALL Q states (stay-prob `1/Q`,
detailed-balance-preserving). For strongly-coupled discrete MRFs this
is the standard remedy for per-site Gibbs's catastrophic mixing
(`system_design.md` §11.2(b)).

**JUSTIFICATION (Check #16):** discrete-MRF target with strong local
dependence (Exception §11.2(b)). Algorithm is Swendsen-Wang 1987
(physics) / Higdon 1998 (statistician framing). Check #15 parity panel
under `tests/`:
- `test_ising_cluster_block.cpp` — 4×4 enumeration vs MC at β=0.5 and
  β=1.0; β=0 iid-Uniform boundary; two-init mixing; Q=3 Potts
  symmetry. 5 sub-tests.
- `test_ising_cluster_block_diagnostics.cpp` — split-R-hat across
  4 chains (incl. ordered-phase mode-mixing test on signed m);
  batch-means ESS; 17-bucket Pearson χ² vs enumeration; energy
  moments. 8 sub-tests.
- `test_ising_sw_vs_single_site.cpp` — quantified ≥ 5× per-sweep
  efficiency advantage over single-site Metropolis at β=1.0 ordered
  phase (empirically 6× on 10×10).

Ground truth is closed-form enumeration / textbook only — **zero
external package dependency** in the shipped tree.

Reference template: `examples/IsingPrior.cpp` (pure-prior 2D Ising /
Potts demo wiring `ising_cluster_block` through `composite_block` +
`RCPP_MODULE`).

**Scope (v1.2 ship):** h = 0 (no external field), scalar β (no per-
edge β_ij), no partial decoupling (Higdon §2.3). Each deferred to
v1.2.1 (deferred; see project roadmap).

```cpp
ising_cluster_block_config cfg;
cfg.name         = "x";
cfg.n_vertices   = L_x * L_y;
cfg.n_states     = 2;                  // 2 = Ising; ≥ 3 = Potts
cfg.edges        = AI4BayesCode::make_2d_lattice_edges(
                       L_x, L_y, /*periodic=*/false, /*eight_nn=*/false);
cfg.beta_key     = "beta";              // optional ctx slot for β
cfg.beta_default = 0.44;                // used if beta_key missing
cfg.initial_state = arma::vec(N, arma::fill::zeros);  // optional
```

**Helper:** `make_2d_lattice_edges(L_x, L_y, periodic, eight_nn)`
builds the 4-NN / 8-NN edge list for rectangular lattices. Supply
arbitrary `arma::umat (2 × n_edges)` for non-rectangular topologies.

## gmrf_precision_block (Rue 2001 sparse-Cholesky direct sampler)

Direct sampler for Gaussian Markov Random Fields specified by a sparse
precision matrix Q (Rue 2001 JRSSB *"Fast sampling of Gaussian Markov
random fields"*). Target distribution in canonical form
(Rue eq. 4 / §3.1.2):

  pi(x) ∝ exp{ -½ x^T Q x + b^T x },   x ~ N(Q^{-1} b, Q^{-1})

Algorithm: P Q P^T = L L^T (AMD reordering + sparse Cholesky via Eigen
SimplicialLLT) → solve `L^T y_perm = z`, `z ~ N(0, I)` → apply inverse
permutation. Mean shift `mu = Q^{-1} b` via the cached factorisation.
Per-sweep cost O(n · b_w^2) for the numerical re-factorisation
(b_w = bandwidth ≈ O(√n) for typical 2D lattice GMRFs); the symbolic
factorisation + AMD ordering are computed once and amortised.

**JUSTIFICATION (Check #16):** Fixed-dim continuous Gaussian with
sparse precision (`system_design.md` §11.1 class 1, specialised
efficiency path). Direct conjugate draw — alternative to NUTS on
high-dim Gaussian latents in hierarchical models (spatial smoothing,
RW1 / RW2 splines, ICAR / BYM2 disease mapping, lattice GP
approximations). `gmrf_precision_block` is the library-blessed
sparse-Cholesky direct sampler. Check #15 parity panel under
`tests/`:
- `test_gmrf_precision_block.cpp` — 5 sub-tests: diagonal Q sanity,
  AR(1) n=5 Cov vs dense Q^{-1}, canonical b ≠ 0 mean shift, IGMRF
  1D random walk sum-to-zero (exact constraint + projected-Cov
  match), two-init R-hat across 4 chains on n=50.

Ground truth is closed-form / dense-inverse only — zero external
dependency.

**Reference templates:**
- `examples/GMRFPrior.cpp` — pure-prior 2D ICAR demo on a
  rectangular lattice; sum-to-zero enforced; kappa settable via
  `set_current("kappa")` for downstream hierarchical composition.
- `examples/ICARSpatialGMRF.cpp` — full Bayesian ICAR with Gaussian
  observations (hybrid composite: gmrf_precision_block for phi +
  separate nuts_block instances for Intercept / tau / sigma).

**Scope (v1.2 ship):**
- Symbolic factorisation cached once (assumes Q's sparsity pattern
  is fixed across steps; numerical values may vary — typical for
  `Q = kappa · R` decompositions)
- Single sum-to-zero constraint via post-hoc projection on a
  ridge-regularised Q (simplified Rue §3.1.3, the spam / R-INLA
  approach)
- AMD reordering via Eigen built-in

**Deferred to v1.2.1:**
- Arbitrary linear constraints A x = b (Rue §3.1.3 exact kriging)
- METIS reordering for very large graphs
- Conditional sampling x_A | x_B (Rue §3.1.1)
- Knorr-Held & Rue (2002) joint block update of hyperparam + x
- Divide-and-conquer for very large GMRFs (Rue §4)

```cpp
gmrf_precision_block_config cfg;
cfg.name = "x";
cfg.n    = N;
cfg.Q_fn = [R](const block_context& ctx) -> arma::sp_mat {
    const double kappa = ctx.at("kappa")[0];
    return kappa * R;                  // R = fixed sparse Laplacian
};
// Optional canonical "b":
cfg.b_fn = [y, sigma2](const block_context& ctx) -> arma::vec {
    const double Intercept = ctx.at("Intercept")[0];
    return (y - Intercept) / sigma2;   // canonical form b
};
cfg.sum_to_zero  = true;               // for ICAR / IGMRF
cfg.ridge_epsilon = 0.0;               // auto-bumped to 1e-8 when sum_to_zero
cfg.initial_x    = arma::vec(N, arma::fill::zeros);  // optional
```

**Vendored kernel:** Eigen 3.4 `Eigen/SparseCholesky` (header-only,
MPL-2.0). Build flag: `-I include/eigen` (added to `tests/Makefile`
CPPFLAGS and `examples/Makevars` PKG_CPPFLAGS as part of the v1.2
Block 2 ship).

## gmrf_whitened_ess_block (Murray 2010 Elliptical Slice Sampling on implicit GMRF prior; SHIPPED 2026-06-03)

Companion to `gmrf_precision_block` for the **non-Gaussian observation
likelihood** case. When the latent prior is GMRF
(`pi(x) ∝ exp{-½ x^T Q x}`, Q sparse PSD) but the observation
likelihood is NOT Gaussian (Poisson, Bernoulli, Student-t, NB,
log-Gaussian Cox), the full conditional `p(x | y, hyperparams)` is no
longer Gaussian and `gmrf_precision_block`'s direct conjugate draw
does not apply. This block uses Murray 2010 Elliptical Slice Sampling
on the IMPLICIT GMRF prior — the prior is never expanded; only sparse
Q is factorised.

**Algorithm** (per step):
1. Build Q = Q_fn(ctx); refactor sparse Cholesky (SimplicialLLT + AMD)
2. Sample nu ~ N(0, Q^{-1}) via Rue 2001 permuted backsolve (sample
   z ~ N(0, I), solve `L^T y_perm = z`, apply inverse permutation)
3. ESS shrink loop: propose `x' = x cos θ + nu sin θ`, accept if
   `log_lik(x') > log_lik(x) + log(u)`, else shrink θ-bracket
   (Murray 2010 Algorithm 1)

The acceptance rate is independent of the likelihood scale (Murray
2010 Theorem 1), so the block mixes well even when the posterior is
sharply peaked relative to the prior. Per-sweep cost
O(n · b_w^2 + n · n_shrink_avg) where b_w = bandwidth and n_shrink_avg
is typically 2-5 for well-conditioned posteriors.

**JUSTIFICATION (Check #16):** Latent Gaussian with non-Gaussian
observation likelihood — falls outside `gmrf_precision_block`'s
Gaussian-conditional scope. Murray 2010 ESS is the standard textbook
algorithm for this regime (cited in Rasmussen-Williams Ch.3 GP
classification, Banerjee-Carlin-Gelfand 2014 spatial-epidemiology
Ch.5 for Poisson-ICAR / BYM2).

**Sum-to-zero invariant**: ESS rotation `x cos θ + nu sin θ` is
linear; if `x_cur` and `nu` are both zero-mean, every proposal is
zero-mean. The block enforces sum-to-zero only on `nu` post-sample
(via the same projection as `gmrf_precision_block`) and on
`x_initial` / `set_current(x)`; the inner shrink loop needs no
extra projection. Tested to machine precision (max|mean| = 9.7e-13
across 8000 draws on N=16 ICAR).

**Empirical correctness** (header tests, 2026-06-03):
- T1 Pure prior recovery (N=16, 10k draws): sample var / Q^{-1}
  diag = 0.998 (essentially exact); off-diag cov match within 0.2%
- T2 Poisson-ICAR N=16, 4 chains × 2k: R-hat max=1.026, ESS_bulk
  min=245, coverage 15/16 = 94%, sum-to-zero preserved to 1e-12
- T3b Poisson-ICAR N=64, 4 chains × 10k: R-hat max=1.041,
  ESS_bulk min=149, coverage 63/64 = 98%, 0.09s wall per chain

**Verified convergence budgets (R-hat < 1.01, 4 chains, Poisson-ICAR fixture):**

| Grid | Budget/chain | Wall/chain | R-hat max | ESS_bulk min |
|---|---|---|---|---|
| N=16 (4×4) | 20k | 0.06s | 1.004 | 2409 |
| N=64 (8×8) | 50k | 0.47s | 1.006 | 915 |

Empirical scaling: budget grows roughly linearly with N at typical
lattice connectivities (4-NN / 6-NN). Larger grids (N ≈ 100-200)
may need 100k-200k iter per chain to maintain R-hat < 1.01; the user
should pilot the convergence budget for their specific Q topology and
likelihood family before committing to a production budget.

**Example composite recipe** (sparse-precision latent + user-supplied
non-Gaussian likelihood; user adapts `Q_fn` and `log_lik` for their
model):

```cpp
gmrf_whitened_ess_block_config cfg;
cfg.name = "x";
cfg.n    = N;
cfg.Q_fn = [Q_base](const block_context& ctx) -> arma::sp_mat {
    // Build the sparse precision at the current hyperparameter values.
    // Example pattern: scale a fixed sparsity pattern by a precision hyper.
    const double kappa = ctx.at("kappa")[0];
    return kappa * Q_base;
};
cfg.log_lik = [data](const arma::vec& x, const block_context& ctx) -> double {
    // User-supplied non-Gaussian observation log-density at the proposed
    // latent x, reading any further hyperparameters from ctx. Returns
    // sum_i log p(y_i | x_i, other_hyperparams).
    return /* user implementation */;
};
cfg.sum_to_zero  = true;               // for ICAR-style improper priors
cfg.initial_x    = arma::vec(N, arma::fill::zeros);

// Typical composite: pair with one or more nuts_block for hyperparameters
// (location intercept on the real line, log-precision on the positive line).
```

**Reference templates:** TBD (v1.3 roadmap). Until shipped, use the
example recipe above and the inline pattern in `system_design.md`
§13 GMRF-family section.

**Vendored kernel:** Eigen 3.4 `Eigen/SparseCholesky` (header-only,
MPL-2.0). Same `-I include/eigen` build flag as
`gmrf_precision_block`.

**Scope (v1.2 ship):**
- Symbolic factorisation cached once (assumes Q's sparsity pattern is
  fixed across steps; numerical values may vary — typical for
  `Q = kappa · R` decompositions)
- Single sum-to-zero constraint via post-hoc projection on `nu`
- Universal `log_lik(x, ctx) -> double` user callback (any non-Gaussian
  likelihood family)
- AMD reordering via Eigen built-in

**Deferred to v1.3:**
- Check #25 validator for `Q_fn` (returns symmetric PSD sparse) and
  `log_lik` (returns finite scalar) contracts
- Cross-block ESS scheduling for composite_blocks with multiple GMRF
  latents
- Reference templates for sparse-precision GMRF + non-Gaussian
  likelihood compositions

## order_mcmc_block (Friedman-Koller 2003 order MCMC for Bayesian-network structure)

Specialised MCMC kernel for **Bayesian-network structure learning**
over a state space of total orderings on n discrete random variables.
Target distribution (Friedman & Koller 2003, *Machine Learning*
**50**: 95-125):

    p(≺ | D) ∝ p(≺) Σ_{G ≺ ≺} p(D | G) p(G)
            = p(≺) Π_{i=1..n} Σ_{Pa_i ⊂ Pred(i, ≺)} score(i, Pa_i)

where the sum over DAGs compatible with order ≺ factorises into n
independent sums because each variable's parent set lives in its
predecessor set. We sample orders via Metropolis-Hastings with a
mixture of any-pair swaps and adjacent swaps, and at each step
sample a DAG by drawing each variable's parent set from its
posterior conditional on the current order. Discrete data + BDeu
score (Heckerman-Geiger-Chickering 1995, Eq 28) is the only
likelihood family in v1.2.

**JUSTIFICATION (Check #16):** discrete DAG-structure learning — a
combinatorial state space that's intractable with NUTS / Gibbs (no
gradient, factorial-many DAGs even for n = 10). Order MCMC reduces
the search to the n! permutations (one order ~ many DAGs) and uses
BDeu Bayesian model averaging within each order. This is the
textbook efficiency path for DAG learning. Algorithm is
Friedman-Koller 2003; the BDe / BDeu score is Heckerman-Geiger-
Chickering 1995. Check #15 parity panel under `tests/`:
- `test_bde_scorer.cpp` — 12 sub-tests across T1-T8: 2-node
  Heckerman hand-computed BDe (empty + single-parent families);
  likelihood-equivalence within a Markov class (X→Y vs Y→X give
  identical marginal scores); empty-parent closed form (BDe
  reduces to product of Beta-Bernoulli marginals); structure
  prior FK Eq 2 toggling (uniform vs fan-in-penalised); BDeu
  α-scaling asymptotics; single-edge score reduces to two-bin
  Beta-Bernoulli; top-C candidate-parent selection by single-edge
  score; validation rejects (cardinality mismatch, α ≤ 0,
  out-of-range data values).
- `test_score_cache.cpp` — 11 sub-tests across T1-T8: cached
  families sorted descending by score; `order_node_score`
  consistency with explicit Σ `family_score`; total order
  log-score factorises across nodes; `sample_parent_set` returns
  feasible (Pa ⊂ predecessor) sets; γ-pruning at default 10-nat
  cap; FK §4.2 top-C candidate cap respected; Markov-equivalence
  stability (identity vs reverse order on a chain gives similar
  log-scores); `sample_dag` returns size-n feasible parent-set
  bitmasks.
- `test_order_mcmc_block.cpp` — 11 sub-tests: construction smoke;
  reproducibility under fixed seed; permutation invariant under
  step; round-trip `set_current ∘ get_current`; non-trivial MH
  accept rate (0.02 < α < 0.98); long-run log-score plateau on
  chain BN; sampled DAG respects current order (parents are
  predecessors); `current_named_outputs` returns 3 keys
  (order / order_sampled_DAG / order_log_score).

Ground truth is hand-computed BDe / Markov-equivalence + textbook
formulas — zero external package dependency in the unit tier.

**Stress / robustness (9 sub-tests):** `tests/test_order_mcmc_block_stress.cpp` —
R1 bitwise reproducibility, R2 no-signal max edge marginal < 0.30,
R3 max_parents enforced, R4 BDeu α-asymptotics, R5 n = 20
stability, **R6 4-chain Gelman-Rubin R-hat < 1.01 strict (Vehtari
2021) on unimodal target**, R7 post-conv std/|mean| < 1%, R8
r_i = 4 cardinality recovery, R9 n = 2 edge case.

**Exact-posterior gold-standard diagnostics (4 sub-tests):**
`tests/test_order_mcmc_block_diagnostics.cpp` —
- **D1 n = 3:** enumerate all 25 DAGs, compute exact
  P(G | D) ∝ p(D | G) · |LE(G)| · Π_i ρ(|Pa_i|), marginalise to
  edges, compare to 20000 MCMC samples; HARD max |Δ| < 0.05
  achieves ≈ 0.001.
- **D2 n = 4:** same comparison on 543 DAGs, 30000 samples;
  achieves ≈ 0.009.
- **D3 conditional P(Pa_i | order, D):** direct subset
  enumeration vs sample_parent_set 10000 draws; HARD max
  |Δ| < 0.03 achieves ≈ 0.001.
- **D4 ESS(log_score) > 200** on 10000 post-burn samples (Geyer
  initial-positive-sequence).

**bnlearn ASIA cross-check:**
`tests/audit_OrderMCMCBN_bnlearn_cross.R` — On ASIA (8 nodes,
N = 5000), OrderMCMCBN's top-8 inclusion frequencies recover
**7 / 8** true Markov-equivalent edges with **7 / 7 perfect
skeleton match** against `bnlearn::hc`. The miss (A → T) is also
missed by bnlearn::hc — low data evidence, not an algorithm
failure.

**BiDAG reference-implementation comparison:**
`tests/audit_OrderMCMCBN_vs_BiDAG.R` — head-to-head with
`BiDAG::orderMCMC` (Kuipers-Moffa reference, source of the v1.2.1
partition-MCMC track) on identical data and 4 matched chains.
Both implementations achieve **R-hat < 1.01** (ours 1.00073,
BiDAG 1.00000). **Outcome (A): our code converges to the same
target as the reference.**

**Reference template:** `examples/OrderMCMCBN.cpp` — Tier A R
wrapper (`new(OrderMCMCBN, D, cardinalities, bdeu_alpha,
max_parents, candidate_top_C, family_cache_F, gamma_prune_nats,
prob_adjacent_swap, initial_order, rng_seed, keep_history)`)
with the six-method R contract (step / get_current /
set_current / predict_at / get_dag / get_history).

**Scope (v1.2 ship):**
- **Discrete data only** with per-column cardinality vector (BDeu
  with user-settable equivalent sample size α). Continuous /
  mixed / hybrid BN deferred to v1.2.1.
- **Order MCMC**: any-pair + adjacent-swap mixture (default
  prob_adjacent_swap = 0.5).
- FK §4.2 three-tier candidate-parent heuristic (top-C parents
  per node by single-edge score; top-F whole families cached
  globally; γ-pruning in nats of remaining families during the
  sum step).
- **Per-order BDeu Bayesian model averaging** for sampled DAG —
  parent sets drawn from the FK §4.2 posterior conditional on
  the current order.
- **Documented FK §4.1 bias** within Markov equivalence: order
  MCMC's induced *structure* prior is not hypothesis-equivalent
  (Markov-equivalent DAGs receive different prior weights). The
  algorithm faithfully recovers the **skeleton** (7 / 7 vs
  bnlearn on ASIA) but may flip directions within an equivalence
  class. Fix is Kuipers-Moffa 2017 partition MCMC — deferred to
  v1.2.1 — see "Deferred to v1.2.1".

**Deferred to v1.2.1:**
- Kuipers & Moffa (2017) **partition MCMC** to remove the FK §4.1
  structure-prior bias inside Markov equivalence classes.
- **Continuous / mixed-type data** (BGe Gaussian score per
  Geiger-Heckerman 1994; mixed-discrete-Gaussian conditional
  Gaussian networks per Lauritzen 1992).
- **Edge-specific prior** (currently uniform DAG prior with hard
  max-parents cap).
- **Tempered / parallel-tempered chains** for very multimodal
  posteriors.
- **Predict_at v2**: forward simulation under sampled DAG (v1 is
  a stub returning the current sampled DAG).

```cpp
order_mcmc_block_config cfg;
cfg.name              = "order";       // shared_data key for the
                                       //   permutation (length n)
cfg.data              = D;              // arma::imat (N × n) of
                                       //   non-negative integers
                                       //   in {0, ..., r_i - 1}
cfg.cardinalities     = arma::uvec(n, arma::fill::value(2));
cfg.bdeu_alpha        = 1.0;           // BDeu equivalent sample size
cfg.max_parents       = 5;             // hard cap (FK §4.2 typical 4-6)
cfg.candidate_top_C   = 7;             // FK §4.2 top-C candidates
cfg.family_cache_F    = 200;           // FK §4.2 top-F cached families
cfg.gamma_prune_nats  = 10.0;          // γ-prune sum in nats
cfg.prob_adjacent_swap = 0.5;          // 0.5 = balanced any-pair /
                                       //   adjacent-swap mixture
cfg.initial_order     = arma::uvec();  // empty = random; otherwise
                                       //   a permutation of 0..n-1
cfg.init_rng_seed     = 42;
```

**Tier C kernels:**
- `include/AI4BayesCode/bde_scorer.hpp` — BDe / BDeu family-score
  kernel. O(N + q_i · r_i) per family with explicit count
  scatter; per-cell log-Gamma differences.
- `include/AI4BayesCode/score_cache.hpp` — FK §4.2 three-tier
  cache. `order_node_score(i, order)` = log Σ_{Pa_i ⊂ Pred(i)}
  exp(score(i, Pa_i)) over the cached + γ-pruned family list.
  `sample_parent_set(i, order, rng)` draws a parent set from the
  same posterior.

**Six-method R contract** (Tier A `OrderMCMCBN`):
- `step(M)` — M MH order-swap steps.
- `get_current()` — `list(order = 1:n permutation, sampled_DAG =
  n × n {0, 1} matrix, log_score = scalar)`.
- `set_current(list(order = ...))` — only `order` is settable;
  derived `sampled_DAG` / `log_score` are read-only (rejected
  on input).
- `predict_at(list())` — v1 stub returning the current
  `sampled_DAG`.
- `get_dag()` — sampling DAG topology (block-level).
- `get_history()` — `list(order = M × n matrix, ...)` when
  `keep_history = TRUE`.

**Vendored kernel:** none. Pure C++17 + Armadillo + 64-bit
bitmasks for parent sets (caps n ≤ 64; FK §4.2 already requires
small n for tractable enumeration).

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
    beta_new = transform.apply_forward(u, &β_new)
    |det J| computed by the transform (library-side, not user-side)
  MH accept ratio includes the Jacobian correctly per Green 1995.
  When NOT set (default), the identity path is used.
- **Custom bijection with runtime AD Jacobian** (SHIPPED): optional
  `cfg.transform = make_templated_bijection_1d(Forward{}, Inverse{})`
  where `Forward` is a single TEMPLATED callable struct and `Inverse`
  is a non-templated `double -> double` analytic inverse. Framework
  instantiates Forward at both `double` (for sampling) and
  `autodiff::var` (for runtime AD computation of `|dβ/du|`). Users
  STILL write no Jacobian formula. See
  `include/AI4BayesCode/rjmcmc_custom_bijection.hpp` for the API and
  `validator.md` §14 for the bijection sanity probes
  (round-trip / Jacobian non-singularity / forward-reverse Jacobian
  inverse-pair).

See `system_design.md` §10.2 for the full three-tier story (
identity / library 1D transforms / custom AD bijection).

**Supported model classes:** Dirac spike-and-slab variable
selection (the canonical use case), change-point insertion with
prior-sampled values, mixture-component birth/death for finite
unknown-K (non-BNP).

**NOT appropriate for:**
- BNP mixtures (DP, PY, HDP) — use the **truncated SBP** path
  (`stick_breaking_block` + `normal_gamma_cluster_gibbs_block` +
  `categorical_gibbs_block`); see `examples/DPGaussianMixture.cpp`
  / `examples/PYGaussianMixture.cpp` /
  `examples/DPGaussianMixture_DerivedAlpha.cpp`. **The cluster-prior
  hyperparameters MUST be data-driven weakly-informative (see the
  CRITICAL note under `normal_gamma_cluster_gibbs_block` below) — fixed
  hypers silently produce a wrong over-segmented posterior R-hat won't
  flag.** CRP-marginal Neal Alg 2/8 and Jain-Neal split-merge are not
  in the current scope; they remain optimisations rather than
  correctness gates (note: `split_merge_block` co-composed with
  `categorical_gibbs_block` is now unblocked via
  `composite_block::declare_shared_history("z")` if the truncated-SBP
  partition posterior is multimodal for a given dataset).
- Multi-dim bijections `R^n -> R^n` for `n > 1` (e.g., Stephens
  2000 split-merge for finite mixtures with unknown K) — current scope ships
  the 1D-scalar case only (`templated_bijection_1d`); multi-dim
  custom bijections require a future block class.
- Multi-coefficient block birth (birthing a cluster of related
  coefs at once via one transform) — wait for a future multi-dim
  transforms); the current scope is 1D-per-coefficient only.
- HMM / Markov-structured discrete — use `hmm_block` (T10,
  SHIPPED 2026-04-20; see the hmm_block entry above).

**Critical for good mixing:** supply `continuous_update` function
in the config. This is typically a closed-form Gibbs draw for
beta[j] | gamma[j]=1 under the linear conditional. Without this
hook, beta[j] stays at its birth-time value as long as gamma[j]=1
— posterior for beta is badly biased and sigma^2 is inflated.

**Reference templates:**
- `examples/SpikeSlabRJMCMC.cpp` — canonical Dirac spike-and-slab
  under the **Ishwaran & Rao 2005 sigma-scaled slab** form
  (`beta_j | gamma_j=1, sigma, tau ~ N(0, sigma^2 tau^2)`, so tau
  is dimensionless). 4-block composite:
    1. `beta_gibbs_block(pi)` — Exception 3 (scalar Beta-Bernoulli
       conjugate); covered by library parity test Check #15.
    2. `nuts_block(sigma)` with Jeffreys `p(sigma) ∝ 1/sigma`.
    3. `nuts_block(tau)` with Jeffreys `p(tau) ∝ 1/tau` +
       k=0 fallback to half-Normal(0, 1) pin (inside the natural-
       scale log-density).
    4. `rjmcmc_block(gamma, beta)` with hand-written Gibbs
       `continuous_update` for `beta_j | gamma_j=1` — Exception 2
       (kernel contract); covered by per-usage parity test
       Check #15.
  `gamma_init` via marginal-OLS screening
  (`gamma_init[argmax_j |X_j'y|] = 1`) guarantees k ≥ 1 at iter 1,
  avoiding the improper-posterior transient on tau under Jeffreys.
  Constructor takes only `(X, y, a_pi, b_pi, seed, keep_history)` —
  no hyperparameters on sigma/tau thanks to scale-invariant
  Jeffreys priors. See `skills/codegen_priors.md §2a` for the variance
  prior discipline rationale and §3a Class 2b for the Dirac
  spike-and-slab decision tree.
- `examples/SpikeSlabSinhBijection.cpp` — minimal demo using
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
— j with any non-trivial inclusion probability). True-zero j
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

## bart_block (GPL-2.0+)

For `y = f(x) + epsilon`. One tree-sweep per `step()`. Sigma is handled
by a sibling block.

**Vendor (2026-05-03 onward):** uses the unified BART_unified library at
`bart_pure_cpp/src/`. Internally wraps `stdbart::bart_model` (the namespace name
is `stdbart` to avoid clash with the engine class also named `bart`;
the project-facing alias `bart::bart_model` is also valid). The Tier B
plug-in API (`set_X / set_Y / set_data / set_offset / set_s / get_s /
update_step / current_sigma / current_var_probs / current_var_counts /
get_invchi / startdart`) is uniform with `genbart_block`. softBART
(`softbart::softbart_model`) is also vendored under the same `bart_pure_cpp/`
tree (`bart_pure_cpp/src/SOFTBART_VENDOR`), and `softbart_block` ships with the same uniform interface —
see `examples/SoftBartNoise.cpp` for a worked example.

**For nested MCMC**: `BartNoise::set_current(Rcpp::List)` already
accepts any subset of `{X, y, sigma}` and routes them into the block
— use this when composing BART inside an outer Gibbs. No other R
methods are exposed; the unified six-method interface is the whole
API. (System-design agents modifying `bart_block` itself should read
`skills/system_design.md`; code-generation agents do NOT need
to — see the top of `skills/codegen.md`.)

```cpp
bart_block_config cfg;
cfg.name                 = "f_bart";
cfg.x_train              = X;            // Rcpp::NumericMatrix
cfg.y_init               = y;            // Rcpp::NumericVector
cfg.working_response_key = "bart_target";
cfg.sigma_key            = "sigma";
cfg.ntrees               = 50;
cfg.sigma_init           = sd(y);   // placeholder, see note below

// --- Optional: sparse variable selection (Linero 2018 DART) ---
cfg.dart                 = false;   // enable DART sparsity prior
cfg.aug                  = false;   // Linero 2018 Section 4 augmentation

// --- Optional: probit-BART leaf prior tau formula (Albert-Chib regime) ---
cfg.binary               = false;   // see "Binary leaf prior" below
```

### Binary leaf prior (Albert-Chib probit BART)

`bart_block_config::binary` controls which leaf-prior tau formula
`bart_model` uses internally:

| `cfg.binary` | tau formula | When to use |
|--------------|-------------|-------------|
| `false` (default) | `(max(y_init) - min(y_init)) / (2 * k * sqrt(ntrees))` | Gaussian regression `y ~ N(BART(X), sigma^2)`. The data range gives a sensible leaf scale. |
| `true` | `3 / (k * sqrt(ntrees))` | **Probit BART via Albert-Chib data augmentation.** The BART working response is a latent z whose effective range under truncation is ~6 (≈ ±3 SD). Using the Gaussian formula on an arbitrary y_init scale (e.g. ±1) over-shrinks the leaves by a factor of 3. This formula matches `BART::pbart` / `cpbart.cpp` line 244 exactly. |

Non-Gaussian likelihoods that are NOT data augmentation (e.g. logistic
BART via `genbart_block + genbart::lik::logistic_lik`) do not use
`bart_block` at all — see the `genbart_block` section below.

When generating a probit-BART wrapper that uses Albert-Chib (the user
prompt says "data augmentation"), set `cfg.binary = true`. The
recipe is to compose `probit_aug_block` (latent z) with `bart_block`
(`cfg.binary = true`) in a single composite. The shipped
`examples/ProbitRegression.cpp` shows the Albert-Chib half; the
`bart_block` docstring describes the binary mode flag.

Uses R's RNG. `set.seed()` for reproducibility.

### Sparse variable selection (DART)

`bart_block_config` exposes two Linero (2018) flags for **implicit
variable selection** in high-p settings:

- **`dart = true`** — replaces BART's uniform split-variable probabilities
  with a sparse Dirichlet prior. Variables with no signal have their
  split probability shrunk toward zero. Posterior-mean split frequency
  per variable gives a usable variable-importance / inclusion score
  without a separate spike-and-slab block.
- **`aug = true`** — only active when `dart = true`. Enables the data-
  augmentation scheme in Linero 2018 Section 4 that accelerates mixing
  of the sparse-Dirichlet concentration parameter. Adds a small
  per-sweep cost and is not always beneficial — leave OFF by default;
  flip ON only if DART's concentration parameter is visibly slow to mix.

**When to recommend this to the user:** the model is of the form
`y = f(X) + noise` with p moderate-to-large (say p >= 10) AND the user
indicates they care about which features matter (variable selection,
feature importance, interpretability). DART is often a cleaner fit than
wrapping BART inside a spike-and-slab selector — it gets sparsity "for
free" from the tree prior.

**Decision-flow recipe** for the codegen skill:
- If the user's model description includes words like "variable selection",
  "which features matter", "sparse", "feature importance", or "high-
  dimensional", AskUserQuestion whether to enable DART.
- Recommended defaults when the user opts in: **`dart = TRUE, aug = FALSE`**.
  Describe `aug` in one sentence (mixing accelerator for the
  concentration parameter, adds per-sweep cost, turn on only if DART
  mixing is slow) and let the user pick. Offer `dart = FALSE` when the
  goal is pure prediction.
- Expose `dart` and `aug` as constructor arguments to the generated
  R wrapper **only if the user opts into DART** — otherwise hardcode
  both to `false` inside the class body and do NOT surface them to R.
  See `examples/BartNoise.cpp` — they are positional args 9 and 10 in
  that reference template's constructor.

All other BART hyperparameters (`ntrees`, `k`, `power`, `base`, `nu`,
`numcut`) keep BART::wbart's defaults and are passed through unchanged.

### Sigma initialization in bart_model (automatic)

**You do NOT need to compute hat_sigma in R or in AI-generated code.**
The `bart_model` constructor automatically:

1. Fits OLS internally (`arma::solve(X, y)`) when `p < n`
2. Computes `sigest = sqrt(RSS / (n - p))` from OLS residuals
3. Falls back to `sd(y)` only when `p >= n` (high-dimensional case)
4. Sets `lambda = sigest^2 * qchisq(0.1, nu) / nu` for the conjugate
   sigma prior (using a pure C++ lookup table, no R function calls)
5. Uses `sigest` as the initial sigma value

`cfg.sigma_init` is NOT used to override this — `bart_block` does not
call `set_sigma` after construction. The internal OLS-based estimate
matches `BART::wbart`'s `sigest` default exactly.

### Recommended sigma prior for BART-structured models

In any model with likelihood `y ~ N(BART(X), sigma^2)`, sigma has a soft
identifiability problem — BART's flexible f(x) can absorb signal that
should go into the noise variance, and with a weakly informative prior
sigma can drift anywhere the under-fit bias permits. The
default for this class of model is BART::wbart's **calibrated conjugate
inverse-gamma** prior (Chipman, George & McCulloch 2010):

    sigma^2 ~ InverseGamma(nu/2, nu*lambda/2)
    lambda  = sigest^2 * qchisq(0.1, nu) / nu      (sigquant = 0.9)
    nu = 3 (default)

`sigest` is the OLS residual SD, computed automatically by `bart_model`
and retrievable via `bart_block::current_sigma()` after the BART child
has been added to the composite.

**Decision-flow recipe** (follow `skills/codegen.md` Section 2):
- When parsing a user model with a BART mean and no explicit sigma prior,
  run the AskUserQuestion decision flow exactly as for any other
  parameter, but substitute this calibrated IG in place of the generic
  Jeffreys `p(sigma) ∝ 1/sigma` scale default (the default-priors table).
- Present it as option (b) "Default" and tell the user WHY
  (identifiability / under-fit absorption).
- Offer `HalfNormal(0, 10)` or a user-specified alternative as option (c).
- Still accept an explicit user-supplied prior verbatim.

**Sampling stays uniform:** sigma is still a standard `nuts_block` — only
the prior term in the log-density changes. Do NOT introduce a custom
conjugate-Gibbs block for sigma; mixing closed-form conjugate draws with
a NUTS-first generator is explicitly prohibited (see
`skills/codegen.md` Hard Rules). The only Gibbs blocks the generator may
use are the `*_gibbs_block` headers already shipped in
`include/AI4BayesCode/` — see the "Gibbs blocks" sections of this catalogue
for the current roster.

See `examples/BartNoise.cpp` for the reference template — both the
`R::qchisq` calibration in the constructor and the IG log-density in the
`nuts_block` lambda.

## genbart_block (GPL-2.0+)

**Generic tree-ensemble primitive** for Bayesian BART with ANY
plug-in likelihood. Implements Linero 2022 (arXiv:2202.09924)
reversible-jump MCMC with Laplace-approximated BIRTH / DEATH / CHANGE
tree proposals; user supplies only `log_f(y, lambda, obs_i)` and
(optionally) `score` / `obs_info` on a subclass of
`genbart::likelihood`. Tree-ensemble output is `r(x)` on the
LINEAR-PREDICTOR scale -- interpretation depends on the attached
likelihood.

`current()` returns a length-N arma::vec of r(X_i).

### Shipped likelihoods (`genbart::lik::*`)

| Likelihood class | Response | Nuisance param(s) | Reference example |
|---|---|---|---|
| `normal_lik(sigma_init, nu, lambda)`      | y ∈ ℝ, Normal                 | sigma              | (Linero 2022 §4.1 regression; use BartNoise for the tuned CRAN BART R package alternative) |
| `logistic_lik()`                          | y ∈ {0,1}, Bernoulli-sigmoid  | —                  | `examples/GBartLogistic.cpp` |
| `poisson_lik()`                           | y ∈ ℕ, Poisson-log-link       | —                  | `examples/GBartPoisson.cpp`  |
| `negative_binomial_lik(kappa_init)`       | y ∈ ℕ, NB(mu, kappa)           | kappa (RW-MH)      | (compose from `GBartPoisson.cpp` + this likelihood header) |
| `heteroscedastic_lik(phi_init, a0, b0)`    | y ∈ ℝ, mean=variance=exp(r)   | phi (Gamma Gibbs)  | `examples/GBartHeteroscedastic.cpp` |
| `aft_log_logistic_lik(sigma_init, ...)`   | y_log = log time + censoring  | sigma (RW-MH)      | (compose from `GBartPoisson.cpp` + this likelihood header) |
| `aft_generalized_gamma_lik(...)`           | log time + censoring          | 2 nuisances        | (planned) |
| `gamma_shape_lik(...)`                    | y > 0, Gamma with shape r(x)   | rate (nuisance)    | (planned) |
| `beta_lik(phi_init, ...)`                 | y ∈ (0,1), Beta               | phi (RW-MH)        | (planned) |
| `beta_binomial_lik(...)`                  | overdispersed counts          | rho (RW-MH)        | (planned) |

Beyond these, users can write custom likelihoods: subclass
`genbart::likelihood` and override at minimum `log_f`; the base class
supplies finite-difference `score` / `obs_info` defaults. Analytic
overrides are strongly recommended for speed since Laplace leaf
proposals call them n times per tree update.

### Configuration

```cpp
genbart_block_config cfg;
cfg.name        = "r";           // shared_data key storing r(X)
cfg.x_train     = X;
cfg.y_init      = y;
cfg.offset_init = Rcpp::NumericVector(0);  // empty = zeros
cfg.lik         = likelihood_.get();       // non-owning pointer
cfg.ntrees      = 50;                       // Linero 2022: 50 for softer
                                            // likelihoods (logistic/NB/
                                            // beta/AFT); 200 for well-
                                            // identified (normal/Poisson/
                                            // heteroscedastic)
cfg.y_key       = "";            // empty = fixed training data; non-
                                 // empty = refresh Y from shared_data
                                 // each sweep (nested MCMC use case)
cfg.offset_key  = "";            // similarly for offset
// cfg.hypers default to Linero 2022 §3.3 (adaptive half-Cauchy on
// sigma_mu with c = 1/sqrt(ntrees); DART off).
```

### Nested MCMC

`GBart*` wrappers' `set_current(Rcpp::List)` accepts any subset of
`{X, y, offset}` (where applicable) -- see `examples/GBartPoisson.cpp`
for the canonical pattern. Outer samplers push updated working
responses / imputed covariates between sweeps:

```r
for (iter in seq_len(n_iter)) {
    # ... outer sampler produces y_new / X_new ...
    m$set_current(list(X = X_new, y = y_new))
    m$step(1L)
}
```

### DART sparsity

`genbart::rjmcmc_hypers::dart_active = true` enables Linero 2018's
sparse-Dirichlet prior on split-variable probabilities. Same "ask the
user; expose constructor args only on opt-in" recipe as `bart_block`
-- see `skills/codegen.md` follow-up questions. Recommended defaults
when opted-in: `dart_active = TRUE`, `dart_const_theta = FALSE`.

### RNG

genBART uses R's global RNG via `genbart::arn` (same pattern as the
legacy `bart_block`). Reproducibility
requires `set.seed()` in R BEFORE the MCMC loop; the construction-time
seed on `GBart*` wrappers only affects the mutable `predict_rng_`
stream used by stochastic refreshers.

### Performance

End-to-end 4-chain diagnostics on `GBartPoisson(N=200, p=3, ntrees=200)`:
per-sweep wall time ~0.01 s; 4 × (4000 burnin + 10000 keep) sequential
total ~441 s; max Rhat = 1.004; all 200 per-obs R-hat < 1.05 and
ESS_bulk > 400.

Any file including `bart_block.hpp` or `genbart_block.hpp` inherits
GPL-3.0-or-later (the project license; the vendored BART / genBART
kernels are GPL-2.0-or-later upstream, which is GPL-3-compatible).

## poisson_multinomial_aug_block

Gamma-augmentation block for **multinomial / binary logistic BART via
genBART** under the **reference-category identified parameterization**
(category 0 fixed as reference, f^(0)(x) := 1, with C-1 non-reference
generalized-BART functions r_1, ..., r_{C-1}).

Implements the classical Poisson-multinomial gamma trick:

    phi_i | y, r  ~  Gamma(n_i, 1 + sum_{j=1..C-1} exp(r_j(x_i)))

For single-observation data (n_i = 1) this reduces to
Exp(rate_i). Given phi_i, the augmented likelihood factorises into
C-1 conditionally independent Poisson-like likelihoods that each
match `genbart::lik::poisson_lik` with **offset = log(phi_i)**:

    likelihood_j ~ prod_i exp(u_{i,j} * r_j(x_i) - phi_i * exp(r_j(x_i)))

where u_{i,j} = [y_i == j]. The block writes log_phi under its own
name (so downstream genbart_blocks reference it via offset_key) and
each u_j under the configured u_keys[j-1].

### Attribution

The gamma-trick augmentation is classical (predates BART):
**Baker 1994** (Statistician 43(4):495-504), **Forster 2010**
(Stat. Meth. 7(3):210-224), **Walker 2011**, **Caron & Doucet 2012**.
**Murray 2021** (JASA 116(534):756-769 §3.1) introduced the C-1
reference-category identified multinomial parameterization for tree
ensembles. This implementation preserves that architecture but uses
Linero 2022's RJMCMC tree kernel (via `genbart_block`) as the
per-ensemble sampler rather than backfitting + GIG conjugate prior.

**JUSTIFICATION (Check #16): Exception 1** (discrete/augmented
measure; NUTS cannot target the continuous phi with its y-dependent
rate).

### Use with `genbart_block` children

Reference templates:
- `examples/GBartLogistic.cpp` -- binary (C = 2) via **direct**
  `genbart_block + logistic_lik` (simpler; no augmentation).
- `examples/GBartMultinomial.cpp` -- C >= 2 via C-1
  `genbart_block(poisson_lik, offset_key = "log_phi_aug")` +
  one `poisson_multinomial_aug_block`.

```cpp
poisson_multinomial_aug_block_config cfg;
cfg.name   = "log_phi_aug";   // downstream offset_key = this name
cfg.N      = N;
cfg.C      = C;               // total classes incl. reference (>= 2)
cfg.y_key  = "y";
cfg.n_key  = "";              // empty = n_i = 1 per obs
cfg.r_keys = { "r_1", ..., "r_{C-1}" };
cfg.u_keys = { "u_1", ..., "u_{C-1}" };
cfg.initial_y = y_arma;       // integer labels 0..C-1 as doubles
// cfg.initial_log_phi defaults to zeros (phi = 1, neutral start).
```

Gibbs sweep order inside the composite:
`poisson_multinomial_aug_block` FIRST (writes u_j and log_phi from
current r_j), then each `genbart_block` in sequence (each reads its
own u_j via y_key and the shared log_phi via offset_key).

## elliptical_slice_sampling_block

Generic Elliptical Slice Sampler (Murray, Adams, MacKay 2010) for any
**latent Gaussian model** with arbitrary likelihood. Takes a prior-
covariance Cholesky L from context + a user-supplied log_lik lambda,
returns posterior draws of the latent vector f. No gradient needed, no
step-size tuning, handles arbitrary cross-correlation in the prior
covariance.

**Use cases** — latent-Gaussian models with **non-Gaussian** likelihoods:
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
and the right architecture is the **marginal-likelihood approach** —
sample only the hyperparameters from the 3-dim marginal posterior, no
explicit f, no ESS. See `examples/GPRegression.cpp` for that pattern;
it matches Stan, libgp, and GaussianProcesses (every reference GP
implementation marginalizes f for Gaussian observations).

**Name disambiguation**: we use the full name `elliptical_slice_sampling`
instead of "ESS" to avoid collision with "Effective Sample Size" in
MCMC diagnostics vocabulary.

**Reference example**: `examples/GPClassification.cpp` (Bernoulli-logit
likelihood — the canonical case where ESS is the correct choice).

**JUSTIFICATION (Check #16)**: Exception 1 — specialized latent-
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

**CRITICAL INVARIANT — `L_chol_key` must point at the Cholesky of the
LATENT prior covariance only.** When this block IS used (i.e. with a
non-Gaussian likelihood that you cannot marginalize), the L_chol fed
in must be the Cholesky of the latent f's prior `K(hyperparams)`, NOT
of any marginal covariance that contains observation-noise terms:

```cpp
// CORRECT — latent f has prior N(0, K). σ enters only via log_lik.
M.diag() += jitter;          // chol(K + jitter*I)

// WRONG — would only make sense for the marginal-likelihood approach
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

## univariate_slice_sampling_block

Neal 2003 univariate slice sampler (stepping-out + shrinkage,
section 4.1, with random step-out budget split for reversibility per
eq. 5). **Strictly 1-D / scalar.** User writes ONLY a
natural-scale log-density lambda; sampler machinery is textbook and
library-owned. Same AI-safety profile as `nuts_block` (no
conditional-posterior derivation, unlike Gibbs).

**When to use vs nuts_block.** ALWAYS prefer `nuts_block` for
smooth differentiable targets -- NUTS mixes better and is the
default. Choose `univariate_slice_sampling_block` only when NUTS
cannot:
- log p is non-differentiable (piecewise, kink, floor/ceil)
- log p is a black-box library call whose gradient would require
  re-implementing that library with autodiff::var (e.g., celerite
  marginal log-likelihood via `celerite_marginal_likelihood.hpp`)
- gradient prohibitively expensive relative to lp eval.

See `skills/codegen_priors.md` §2b.1 for the decision tree.

**Reference example**: `examples/GPTimeSeries.cpp` (using slice sampling:
hyperparameters amp/tau/sigma sampled via slice on celerite's
marginal log-likelihood).

**JUSTIFICATION (Check #16): Exception 1** -- specialized sampler for
1-D continuous parameters whose log-density lacks an accessible
gradient (violates NUTS's prerequisite). Library parity test at
`tests_autodiff/block_tests/test_univariate_slice_sampling_block.cpp`
(three fixtures: Normal via identity, Gamma via positive, Beta via
interval; 10k draws each, mean/variance within 5%/10% of analytical).

```cpp
#include "AI4BayesCode/univariate_slice_sampling_block.hpp"

univariate_slice_sampling_block_config cfg;
cfg.name         = "amp";
cfg.initial_unc  = arma::vec{std::log(amp_init)};  // unc scale, length 1
cfg.constrain    = constraints::positive::constrain;
cfg.unconstrain  = constraints::positive::unconstrain;
cfg.w            = 1.0;    // initial slice-bracket width on unc scale
cfg.log_density  = [](const arma::vec& t_unc, const block_context& ctx) {
    return constraints::positive::wrap(t_unc, nullptr,
        [&](const arma::vec& t_nat, arma::vec* /*unused*/) -> double {
            // natural-scale lp (prior + likelihood); NO Jacobian, NO grad.
            return lp;
        });
};
```

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
`CholeskySolver` reflects the post-sweep state. The `predict_at`
path then calls `predict_mean_var(t_new)` directly. Slice blocks do
NOT read `celerite_logp` from ctx (they compute the marginal at
proposed values via `celerite_marginal_likelihood.hpp`); the block's
cached `celerite_logp` is kept for downstream inspection only.

Use for:
- Astronomical time-series (Foreman-Mackey et al. 2017 original)
- Financial / climate time-series
- Spline-like 1-D extrapolation for N > 2000 where dense GP is too slow

## GP reference examples summary

| Example | Architecture | Hyperparam sampler | Latent sampler | Likelihood |
|---|---|---|---|---|
| `GPRegression.cpp` | **MARGINAL** (f integrated out) | `joint_nuts_block({amp, ell, sigma})` (POSITIVE × 3) | n/a (f recovered at predict time) | Gaussian |
| `GPClassification.cpp` | **WHITENED** (f = L · z, z ~ N(0, I)) | `joint_nuts_block({amp, ell})` (POSITIVE × 2) — log-density includes Bernoulli-logit lik at proposed (amp, ell) | `elliptical_slice_sampling_block` on **z** (prior `L_identity`) | Bernoulli-logit |
| `GPTimeSeries.cpp` | celerite (1-D, O(N)) | `univariate_slice_sampling_block` x 3 (amp, tau, sigma) | marginalized out (semi-separable solver) | Gaussian |

**The architectural rule for GPs**:

| Observation likelihood | f admits closed-form integral? | Architecture |
|---|---|---|
| Gaussian | YES — `p(y\|θ) = N(y \| 0, K + σ²I)` | **Marginal**: sample only hyperparameters; no ESS, no explicit f |
| Bernoulli, Poisson, Student-t, Negative-Binomial, … | NO | **Whitened ESS**: sample `z ~ N(0, I)`, recover `f = L(amp, ell) · z` |

This is the same architectural choice every reference GP library makes
(Stan, libgp, GaussianProcesses, GPy, GPflow): marginalize f when you
can; sample a whitened latent when you must.

The whitened parameterization (Murray & Adams 2010) is preferred over
centered ESS because the centered conditional `p(amp, ell | f)` has a
prior factor `p(f | amp, ell)` that pulls `(amp, ell)` toward small
values when f is small, which causes the chain to collapse to
`(amp ≈ 0, f ≈ 0)` for weakly informative data. Whitening puts the
prior on z (independent of `(amp, ell)`) so the hyperparameter
conditional sees the data only via the Bernoulli-logit likelihood
`Bernoulli(y | sigmoid(L(amp, ell) · z))`.

`GPRegression.cpp` follows the marginal route; `GPClassification.cpp`
uses whitened ESS because Bernoulli-logit has no closed-form latent
integral.

Pick by structure:
- Multi-D continuous covariates + Gaussian response: GPRegression (marginal)
- Multi-D + binary response: GPClassification (whitened ESS + joint NUTS on amp, ell)
- 1-D time-series, N moderate-to-large: GPTimeSeries (celerite O(N) advantage)
- 1-D time-series, N small: GPRegression also works (O(N³) is fine for N ≤ 500)

**When the shipped examples DON'T fit the prompt:**
- Heteroscedastic / hierarchical / multi-output / Kronecker GP →
  see "**GP composition recipes**" below.
- Non-Gaussian likelihood NOT covered by `GPClassification.cpp`
  (Poisson, Student-t, …) → adapt `GPClassification.cpp` by swapping
  the `log_lik` callback in its `elliptical_slice_sampling_block`.

## GP convergence troubleshooting ladder

This ladder applies to GP samplers that retain an explicit latent
(non-Gaussian likelihood — `GPClassification.cpp` and its
adaptations). For Gaussian likelihoods, use the marginal architecture
in `GPRegression.cpp` directly; it does not have these mixing pitfalls
because f is integrated out.

The shipped `GPClassification.cpp` already applies the two most
important fixes in this ladder — **whitening** (step 4 below) **and
joint NUTS over `(amp, ell)`** (step 3) — by default. The remaining
steps are escalations on top of those when posterior recovery / ESS
is still poor.

When hyperparameter R-hat fails or ESS is too low on a whitened-ESS
GP sampler, escalate:

1. **Tighten the lengthscale prior.** Half-Normal(0, scale_x/4) or an
   InverseGamma fit to the data's distance-quantile range localises
   the posterior away from extreme regions. (Stan's GP case study
   uses InverseGamma(5, 5) for unit-scale `x`.)

2. **Reparameterise to log-scale.** Sample `log_amp`, `log_ell` as
   REAL via the joint block (or fall back to `nuts_block` per slice),
   with the +log(amp)+log(ell) Jacobian added to the log-density.
   Removes boundary-related step-size collapse.

3. **Joint NUTS over `(amp, ell)` — DEFAULT in `GPClassification.cpp`.**
   The `(amp, ell)` banana ridge requires JOINT trajectory steps —
   modular NUTS on each parameter slow-mixes along the ridge,
   typically dropping `amp` ESS into the dozens at long chains. The
   shipped example uses one `joint_nuts_block({amp, ell})`
   (POSITIVE × 2) so the two hyperparameters are sampled together
   under a shared mass matrix. If you split them back into separate
   `nuts_block`s for any reason, expect 5–10× ESS loss on `amp`.

4. **Whitening (Murray & Adams 2010).** Already applied in
   `GPClassification.cpp`. Replaces explicit `f` with `z ~ N(0, I)`
   via `f = L · z`, so z's prior does NOT depend on `(amp, ell)`.
   The hyperparameter conditional sees the data only via the
   likelihood (Bernoulli-logit, Poisson, …), avoiding the
   `(amp ≈ 0, f ≈ 0)` collapse mode of the centered parameterisation.

5. **Switch the joint log-density to an analytic gradient.** The
   shipped example uses finite differences (4 K-builds + chols per
   gradient call). Reverse-mode Cholesky AD (Murray 2016) gives a
   noise-free gradient at O(N³) one-shot + O(N²) per θ-component;
   typically lifts `ell` ESS from the dozens to the hundreds. Worth
   it when the whitened-ESS bulk autocorrelation on `ell` is the
   bottleneck.

6. **Marginalise the latent if you can.** If the likelihood happens
   to be Gaussian (or any likelihood that admits a closed-form latent
   integral), drop the ESS block entirely and sample only the
   hyperparameters from the marginal posterior. This is the route
   taken by `GPRegression.cpp` for Gaussian observations.

**References**: Rasmussen & Williams 2006 §5 (marginal-likelihood
gradient formula `0.5 tr((α α' − K⁻¹) ∂K/∂θ)`, used in
`GPRegression.cpp`); Murray & Adams 2010 *Slice Sampling Covariance
Hyperparameters of Latent Gaussian Models* (whitening, used in
`GPClassification.cpp`); Betancourt 2017 *Robust Gaussian Processes
in Stan* (priors + reparameterisations).

## GP composition recipes (heteroscedastic / hierarchical / multi-output)

The shipped GP reference examples (`GPRegression.cpp`,
`GPClassification.cpp`, `GPTimeSeries.cpp`) cover **single-latent**
GP models. For multi-latent / multi-output / hierarchical GPs, the
agent must compose multiple primitives. The recipes below show the
composition pattern that the agent should follow when generating a
new GP wrapper. These recipes, together with the existing single-GP
examples, are sufficient for an agent that has read this skill set to
generate a multi-latent / multi-output / hierarchical GP model.

### Recipe A — Heteroscedastic GP

**Model:** `y_i ~ Normal(f_mu(x_i), exp(f_sigma(x_i)))` with TWO
independent GP priors: `f_mu ~ GP(0, K_mu)`, `f_sigma ~ GP(0, K_sigma)`.

**Composition** (in `composite_block` `add_child` order):

1. `joint_nuts_block({log_amp_mu [real], log_ell_mu [real]})`
   — hyperparameters for `K_mu`. Add Jacobian `+ log_amp_mu +
   log_ell_mu` to the prior contribution.
2. `joint_nuts_block({log_amp_sigma [real], log_ell_sigma [real]})`
   — hyperparameters for `K_sigma`. Same Jacobian pattern.
3. shared_data refresher computing
   `L_mu = chol(K_mu(X, X; amp_mu, ell_mu) + jitter * I)`.
4. shared_data refresher computing `L_sigma` (independently).
5. `elliptical_slice_sampling_block` on `f_mu` reading `L_mu`. The
   log-likelihood lambda sums
   `dnorm(y_i, f_mu_i, exp(f_sigma_i), log = TRUE)` (current
   `f_sigma` from context, holds `f_mu` as the proposed slice).
6. `elliptical_slice_sampling_block` on `f_sigma` reading `L_sigma`.
   SAME log-likelihood expression but now `f_mu` is held fixed and
   `f_sigma` is the proposed slice.
7. `y_rep` stochastic refresher:
   `y_rep_i = rnorm(1, f_mu_i, exp(f_sigma_i))` using the wrapper's
   `predict_rng_`.

**Predict DAG**: `X → L_mu, L_sigma`; `L_mu, f_mu_init → f_mu`;
`L_sigma, f_sigma_init → f_sigma`; `f_mu, f_sigma → y_rep`.

**Why joint NUTS on each pair**: `(log_amp_mu, log_ell_mu)` and
`(log_amp_sigma, log_ell_sigma)` each have their own banana ridge
(see escalation ladder above). Modular NUTS on 4 separate
hyperparameters is the most common failure mode for this model.

### Recipe B — Hierarchical GP

**Model:** Observations `y_n` indexed by group label `g(n)`; each
group `g` has its own GP `f_g ~ GP(0, K(x_g; amp_g, ell_g))`;
group-level hyperparameters share a hyperprior
`(log_amp_g, log_ell_g) ~ Normal(mu_h, Sigma_h)`.

**Composition:**

1. `joint_nuts_block({mu_h [real vector], Sigma_h_chol [real lower-triangular]})`
   — hyperprior parameters. Use a positive constraint on the
   diagonal of `Sigma_h_chol` (REAL on log-scale, then back-
   transform; or the POSITIVE slice of `joint_nuts_block`).
2. **Per group g**:
   `joint_nuts_block({log_amp_g [real], log_ell_g [real]})`
   reading `mu_h, Sigma_h_chol` from context — group-level GP
   hyperparameters. Prior contribution is the multivariate
   Normal(mu_h, Sigma_h) on `(log_amp_g, log_ell_g)`.
3. **Per group g**: shared_data refresher computing
   `L_g = chol(K(x_g, x_g; amp_g, ell_g) + jitter * I)`.
4. **Per group g**: `elliptical_slice_sampling_block` on `f_g`
   reading `L_g`. The log-likelihood sums
   `dnorm(y_n, f_g[i_n], sigma_obs, log = TRUE)` for `n` in group `g`.
5. `nuts_block(log_sigma_obs)` (REAL) — observation noise.
6. y_rep stochastic refresher.

**Predict DAG**: `mu_h, Sigma_h_chol → log_amp_g, log_ell_g for all g`;
`log_amp_g, log_ell_g → L_g`; `L_g → f_g`;
`f_g (n in group g) → y_rep_n`.

**Composite size**: with G groups, this composite has
`1 + G + G + G + 1 = 3G + 2` child blocks. For large G, consider
batching the per-group blocks into a single ESS over a block-
diagonal `L = diag(L_1, …, L_G)` to reduce composite overhead.

### Recipe C — Multi-output / Kronecker GP

**Model:** `y_{i,j}` for `i = 1..N` inputs and `j = 1..M` outputs:
`vec(Y) ~ Normal(0, K_input ⊗ K_output + sigma² · I)` with LKJ-style
prior on the output correlation matrix.

**Composition:**

1. `joint_nuts_block({log_amp [real], log_ell [real]})` —
   input kernel hyperparameters.
2. `nuts_block(log_sigma)` (REAL) — noise scale.
3. `nuts_block` on the LKJ-style `Lambda` correlation matrix for
   `K_output` (parameterise via Cholesky of the correlation matrix).
4. shared_data refresher computing
   `L = chol(K_input ⊗ K_output + sigma² · I)`. **Use Kronecker
   shortcuts when possible**: when the noise term is also separable
   (e.g., per-output noise), `chol(K1 ⊗ K2) = chol(K1) ⊗ chol(K2)`
   gives O(N³ + M³) instead of O((NM)³); otherwise use
   eigendecomposition (Saatçi 2011 PhD §5).
5. `elliptical_slice_sampling_block` on `vec(F)` reading `L`.
6. y_rep refresher: `y_{i,j} = F_{i,j} + sigma · randn()`.

**Predict DAG**: hyperparams → `L`; `L → F`; `F + sigma → y_rep`.

**Caveats**:
- The full `NM × NM` Cholesky in step 4 is the bottleneck. For
  large N, M (>100 each), consider Kronecker structure or low-rank
  approximations. The current `elliptical_slice_sampling_block`
  reads `L` as flat column-major; you may need to hand-roll the
  Cholesky-Kronecker structure outside the block.
- LKJ priors on `Lambda` benefit from the Cholesky-factor
  parameterization to avoid positive-definiteness rejections.

## Reduced-rank / spline / spatial patterns (HSGP, B-spline, ICAR)

The recipes above cover full-rank GPs (Cholesky of `K(X, X)` per draw).
For **basis-expanded / non-centered structured priors** -- HSGP, splines,
ICAR, BYM2 -- a different architectural pattern applies:

| Example | Pattern | When to use |
|---|---|---|
| `HSGPRegression.cpp` | 1-D Hilbert-space reduced-rank GP. `f(x) = sum_m sqrt(spd(ell)_m) * z_m * phi_m(x)`. | Smooth function regression with O(M) cost (vs O(N^3) full GP). M = 20-50 basis. brms-style. |
| `BSplineRegression.cpp` | Penalised B-spline. `f(x) = Bsp(x) . (sds * z)` non-centered. | 1-D smoothing without GP cost. Smoothing parameter `sds` learned from data. |

**Architectural rule for these patterns** (different from the full-rank
GP recipes above):

1. Reparameterise EVERY positive scalar to log scale. Treat them as
   REAL parameters of `joint_nuts_block`.
2. Put ALL parameters (`Intercept`, all log-scales, all latent vectors)
   into ONE `joint_nuts_block`.
3. Set `cfg.use_dense_metric = true`. The Welford pilot covariance
   handles the (Intercept, basis-coef-mean) ridge AND the
   (positive-scale, latent vec) funnel in one pass.
4. Manually add `log|Jacobian| = +log_scale` for each log-transformed
   positive scalar inside the user log-density. The block does NOT add
   Jacobians for REAL sub-params.

**Why all-REAL + dense metric (not per-slice POSITIVE + identity)**:
on the funnel/ridge geometry of HSGP/spline/ICAR an identity metric
consistently produces ESS = 1-3 on the smoothing scale or chain-stuck
behaviour at moderate chain lengths. Log-transform the positive scalars
to REAL and turn on the dense metric so the Welford pilot covariance can
capture the off-diagonal coupling.

**Why not separate blocks per parameter**: blocking
`(Intercept | log_amp | log_ell | z)` separately preserves the
within-block strong correlations as cross-block conditionals, and the
per-block step size adapts to those tight correlations. The chain
mixes within each block but takes O(rho) sweeps to traverse the joint
posterior, where rho is the cross-block correlation; for HSGP/spline/
ICAR rho is large enough that this dominates the wall time.

**Pattern check before writing the cpp** (metric is a Check #18 escalation,
MEASURED — there is no "Check #11.7"):
- Joint-block dim >= 5 with at least one POSITIVE scalar
  AND a basis-coefficient or latent vector?
  -> use `joint_nuts_block` (real-only), log-transform the POSITIVE scalars
     (manually add log|J|); START DIAGONAL and escalate to
     `use_dense_metric = true` (Check #18) ONLY if R2/R3 shows diagonal is
     inadequate — do NOT gate dense on the dimension.
- Joint-block dim < 5, all POSITIVE, no latent vector?
  -> `joint_nuts_block` with identity metric is fine
     (this is what `GPRegression.cpp` and `GPClassification.cpp` do
     for the 2-3 hyperparameters).

## Vendored libraries summary

| Library | Path | License | Purpose |
|---|---|---|---|
| Eigen 3.4 | `include/eigen/` | MPL-2.0 | Linear algebra |
| mcmclib | `include/mcmclib/` | Apache-2.0 | NUTS kernel |
| autodiff 1.1.2 | `include/autodiff/` | MIT | Check #12 |
| libgp kernels | `libgp_kernels/` | BSD-3 | GP kernel evaluators (SE, Matern, Periodic, RQ, ARD, noise + sum/prod composition) |
| celerite core | `celerite/include/celerite/` | MIT | 1-D time-series O(N) Cholesky |
| BART (legacy Gaussian) | `bart_pure_cpp/src/BART` | GPL-2.0+ | CRAN BART R package Gaussian tree kernel (backfitting) |
| genBART | `bart_pure_cpp/src/GENBART` | GPL-2.0+ | Linero 2022 generalized-BART RJMCMC with pluggable likelihoods (Normal / Logistic / Poisson / NB / Heteroscedastic / AFT / Beta / Gamma_shape / Beta-Binomial + user-supplied) |
| librjmcmc transforms | `include/AI4BayesCode/rjmcmc_transforms.hpp` | CeCILL-B | RJMCMC bijections |

## joint_nuts_block

A SINGLE NUTS block that owns multiple named real-valued sub-parameters
and samples them JOINTLY on a concatenated unconstrained vector. This
is the performance escape hatch for models whose likelihood tightly
couples several continuous parameters (shift-invariance, additive
linear mean, fixed+random effect sharing the mean structure).

```cpp
#include "AI4BayesCode/joint_nuts_block.hpp"
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;

joint_nuts_block_config cfg;
cfg.name = "theta_b_joint";           // block name (not a parameter name)

// Sub-parameter layout. Offsets are assigned in the order listed below.
cfg.sub_params.push_back({ "theta", N });   // shared_data key "theta", dim N
cfg.sub_params.push_back({ "b",     J });   // shared_data key "b",     dim J

// Concatenated initial value, same layout order.
cfg.initial_cat = arma::join_cols(theta_init, b_init);

// Joint log-density on the concatenated REAL-valued vector. The lambda
// MUST write the full-length gradient, slicing at the sub_params
// offsets (validator Check #11.1).
cfg.log_density_grad = &joint_theta_b_log_density;

// Joint blocks usually need more warmup than per-param NUTS because
// the (N+J)-dim mass matrix needs more runway.
cfg.n_warmup_first_call = 1000;

impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));

// Dependencies are keyed by BLOCK name, not sub-param names.
impl_->data().declare_dependencies(
    "theta_b_joint", {"Y", "M", "sigma_b"});  // union of sub-param reads
```

### What you gain

- A single NUTS trajectory over the whole (N+J)-dim space per Gibbs
  sweep. Joint sampling sees the cross-parameter dependence structure
  (shift invariance, additive-linear-mean correlation, ...) that
  Gibbs-wise NUTS can't express, so at small-to-moderate dimension the
  chains mix well and recover posteriors that modular NUTS would
  need orders of magnitude more sweeps to reach.
- After each step the block splits the concatenated draw back into the
  individual shared_data keys (`theta`, `b`, ...) via
  `current_named_outputs()`, so downstream blocks that read those keys
  (e.g. a separate sigma-NUTS) work unchanged.

### What you pay for

- Higher semantic-bug surface: one monolithic log-density with
  slicing, offsets, and prior assembly. Validator Check #11 exists
  specifically to audit this.
- Per-slice constraints are supported: each sub-parameter may declare
  its own constraint (REAL / POSITIVE / SIMPLEX / CHOLESKY_CORR /
  COV_MATRIX / ...) and the block applies the transform + log|Jacobian|
  internally. A positive / simplex parameter can be co-sampled in the
  same block; no need to concatenate by hand.
- **The block defaults to an IDENTITY mass matrix** (mcmclib's default).
  Step-size dual averaging alone is not enough on highly-correlated
  posteriors — chains find the right region but may require very
  long warmup to agree across seeds. **Welford-based diagonal mass matrix adaptation (SHIPPED
  2026-04-20) adds opt-in dense-metric adaptation** via
  `cfg.use_dense_metric = true`, plus
  `cfg.dense_metric_pilot_iters` and `cfg.dense_metric_adapt_iters`.
  When enabled, the FIRST `step()` call runs a pilot NUTS phase with
  identity metric, collects samples, computes their sample
  covariance (Stan-style `n/(n+5)·Σ + 1e-3·(5/(n+5))·I` regularization),
  inverts to the precision matrix (mcmclib's `precond_mat` IS the mass
  matrix, so we pass Σ⁻¹), and uses dense metric for all subsequent
  sampling. Test coverage:
  `tests_autodiff/test_joint_nuts_dense_metric.cpp` — on a 10-dim
  ρ=0.95 correlated Gaussian, dense metric gives max split-R-hat
  1.002 vs identity metric's 1.19.

  **The practical ceiling is model-specific, not dim-specific.** The
  key property is how strongly the posterior correlates across sub-
  parameters; identity-metric NUTS handles weakly-correlated
  high-dim fine, but struggles with strongly-correlated modest-dim.
  Validated envelope (10k+10k × 2 chains):
   - IRT1PL_joint is **well-conditioned**: clean R-hat < 1.01 at
     dim 38, 72, and **115** (N=100, J=15, 10k+10k). The earlier
     "dim 70 ceiling" was set at a shorter-budget run and is revised
     up — IRT is not the bottleneck.
   - HierarchicalLM_joint is **more correlated** (fixed effects and
     random effects share the mean structure): R-hat 1.004 at G=10
     (dim 14), 1.037 at G=20 (dim 24), but **2.1 at G=50 (dim 54)**
     on u and tau. Fails well below IRT's passing dim.
   - The same code geometry is fine in IRT at dim 115 and fails in
     HierLM at dim 54. Treat dim as a rough guide only; the real
     test is per-parameter R-hat at 10k+10k.
   - **HierLM-style joint blocks:** when sub_params contain a positive
     scale parameter (sigma / tau) AND a real vector of random effects
     with large dim (e.g. `(sigma, eps_raw[M])`, or `(tau, u[J])`),
     identity-metric NUTS often fails (rhat ≈ 1.28 at dim 42, ≈ 2.1 at
     dim 54), so dense metric is FREQUENTLY needed — but START DIAGONAL
     and escalate to `cfg.use_dense_metric = true` as a Check #18 step
     when per-parameter R-hat at 10k+10k (or R2/R3) shows diagonal is
     inadequate. Measure; do NOT gate dense on a dimension threshold
     (consistent with the "dim is a rough guide only" note above; there
     is no "Check #11.7").
   - For other joint patterns (shift-invariance, additive linear mean
     without per-unit random effects), the DEFAULT is still ONE
     `joint_nuts_block` over the coupled continuous params — do NOT split
     "to keep it simple" (coupled params mix ~10× slower when split and
     freeze on funnels; `codegen_cpp.md` §4a). Reserve modular only for
     genuinely scalar params, post-NCR branches, or obvious conditional
     independence.

### Stan-style 3-phase warmup (SHIPPED 2026-06-03)

Stan-style windowed warmup; **OPT-IN (default OFF since 2026-06-20)** — set
`cfg.use_three_phase_warmup = true` only as an EXTREME-COND escalation. The
default for dense is the single-phase pilot above (broad-corpus 5-23× faster
keep-phase ESS/s; converges across all realistic cond). Reach for three-phase
only when single-pilot dense still gives R-hat > 1.05 on a high-curvature /
ridge-trapping target (sparse spatial / ICAR). When enabled, the warmup is split
into Stan-style expanding windows so the mass matrix is re-estimated multiple
times as the chain enters the typical set:

- **Phase I** (`cfg.tp_phase1_iters`, default **75 iter**): identity-
  metric step-size dual-averaging only — no mass-matrix updates.
- **Phase II** (`cfg.tp_phase2_windows`, default **{25, 50, 100, 200,
  500} iter**, total 875): per window, collect samples, compute the
  Welford covariance with Stan-style `n/(n+5)·Σ_window + 1e-3·(5/(n+5))·I`
  regularisation, install as the mass matrix for the NEXT window, and
  re-tune step size under the new metric. The expanding-window schedule
  prioritises the latest window's geometry over the noisy early region.
- **Phase III** (`cfg.tp_phase3_iters`, default **50 iter**): final
  identity-metric step-size dual-averaging under the locked mass
  matrix; no further mass-matrix updates.

Total default budget: 75 + 875 + 50 = **1000 warmup iter** (matches
Stan default).

**When to use vs single-phase pilot:**
- Single-phase (default): well-conditioned targets, low correlation,
  a single Welford pass on the typical set converges quickly.
- 3-phase (opt-in): high-curvature or multi-scale targets where the
  pilot covariance changes substantially between burn-in and the
  stationary region (e.g. sparse spatial / temporal random effects
  with strong long-range correlation, hierarchical funnels under
  non-centered parameterisation). Empirical envelope: on small
  spatial random-effect fixtures the three-phase schedule reduces
  wall-clock roughly 2× over single-phase with comparable-or-better
  ESS while preserving posterior identity (cross-mode R-hat
  agreement).

**Limitation**: existing examples that decompose into multiple
separate `nuts_block`s instead of a single `joint_nuts_block` do
NOT benefit from 3-phase warmup without refactor or regeneration.
The 3-phase machinery operates on the joint block's mass matrix;
per-slice NUTS blocks each have their own scalar adaptation.

### When to pick joint vs modular

See `skills/codegen_cpp.md` §4a (Coupling analysis). The short
version:

- **Specialized / structural blocks FIRST.** Always use a specialized
  block (`bart_block`, `genbart_block`, `dirichlet_gibbs_block`,
  `beta_gibbs_block`, `binary_gibbs_block`, `categorical_gibbs_block`,
  `hmm_block`, `pg_logistic_block`, `elliptical_slice_sampling_block`,
  `rjmcmc_block`, GMRF / GP blocks) when it GENUINELY applies — faster AND
  correct-by-construction. For some (discrete latent, random parameter-
  space dimension) NUTS is structurally inapplicable, so the specialized
  block is mandatory, not just preferred.
- **`joint_nuts_block` is the DEFAULT** for every continuous parameter NOT
  claimed by a specialized block. The §4a coupling table flags where joint
  is most critical (shift-invariance, additive linear mean, fixed+random
  effect, hierarchical scales — they FREEZE or bias if split), but joint is
  the default regardless of coupling level — do NOT split continuous
  parameters into per-parameter blocks to "keep it simple".
- single `nuts_block` is **LOW priority** — only a genuinely scalar
  continuous parameter, a post-NCR funnel branch, or obvious conditional
  independence the generator chooses to isolate.

### Reference examples

- `examples/IRT1PL_joint.cpp` — (theta, b) joint with separate sigma_b.
- `examples/HierarchicalLM_joint.cpp` — (alpha, beta, u) joint with
  separate sigma, tau.

## joint_nuts_block — per-slice constraints

`joint_nuts_block` supports **per-slice constraint kinds** (REAL,
POSITIVE, SIMPLEX, CHOLESKY_CORR, COV_MATRIX, ...). The block handles
the unconstrained-vs-natural transformation and log-Jacobian
bookkeeping internally, so the user's log-density stays on the natural
scale.

```cpp
#include "AI4BayesCode/joint_nuts_block.hpp"
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;

joint_nuts_block_config cfg;
cfg.name = "abs_joint";
cfg.sub_params.push_back({"alpha", 1, joint_constraint::REAL});
cfg.sub_params.push_back({"beta",  p, joint_constraint::REAL});
cfg.sub_params.push_back({"sigma", 1, joint_constraint::POSITIVE});

// Natural-scale initial values (sigma > 0).
cfg.initial_cat = arma::join_cols(arma::vec{alpha_init}, beta_init,
                                    arma::vec{sigma_init});

// User-written NATURAL-scale log-density. NO Jacobian terms — the
// block adds them per-slice. Gradient is w.r.t. natural-scale
// parameters; the block chain-rules it back to the unconstrained scale.
cfg.log_density_grad = &joint_log_density_natural;

impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
```

### Constraint kinds

- `joint_constraint::REAL` — identity transform, zero Jacobian.
- `joint_constraint::POSITIVE` — log transform. Block stores
  `unc = log(nat)`, user's log-density sees `nat = exp(unc) > 0`.
  Block adds `log|Jacobian| = sum(unc)` across the slice and applies
  chain rule: `grad_unc_i = grad_nat_i * nat_i + 1`.

When a block has a positive hyperparameter (like sigma) alongside
real-valued parameters (alpha, beta), declare sigma's slice as
POSITIVE rather than splitting it into a separate `nuts_block` —
co-sampling sigma with the location parameters materially improves
mixing when alpha/beta and sigma are posterior-correlated.

### dense_metric for HierLM coupling (Check #18 escalation, measured)

When sub_params include both a POSITIVE scale parameter AND a REAL
vector of per-unit random effects at large dim — e.g.
`(sigma, eps_raw[M])` for `eps = sigma * eps_raw` non-centered
parameterization, or `(tau, u[J])` — the `u_i` and `sigma` are
strongly multiplicatively correlated. Identity-metric NUTS mixes
poorly here (see joint_nuts_block §1289 dim-vs-rhat envelope), so the
diagonal→dense escalation frequently lands on dense.

**Start diagonal; escalate to `cfg.use_dense_metric = true` (Check #18)
when diagnostics show diagonal is inadequate** — measure, don't gate on
dimension. Example (with the Check #18 minimums applied):

```cpp
joint_nuts_block_config cfg;
cfg.name = "gamma_sigma_eps_joint";
cfg.sub_params.push_back({"gamma",   1, joint_constraint::REAL});
cfg.sub_params.push_back({"sigma",   1, joint_constraint::POSITIVE});
cfg.sub_params.push_back({"eps_raw", M, joint_constraint::REAL});
cfg.initial_cat = arma::join_cols(arma::vec{gamma_init},
                                   arma::vec{sigma_init},
                                   eps_raw_init);
cfg.log_density_grad = &joint_log_density_natural;

// Dense metric is FREQUENTLY needed for HierLM coupling at large dim, but it is a
// Check #18 ESCALATION (measured) — START DIAGONAL and enable this only if R2/R3
// shows diagonal is inadequate; do NOT gate on dimension (there is no Check #11.7).
// When you DO enable it, Check #18 minimums apply.
const std::size_t d = 1 + 1 + M;   // joint-block unconstrained dim
// JUSTIFICATION (Check #18): documented diagonal R-hat failure on this coupling.
cfg.use_dense_metric         = true;
cfg.dense_metric_pilot_iters = std::max<std::size_t>(2000, 100 * d);
cfg.dense_metric_adapt_iters = 2000;
cfg.n_warmup_first_call      = cfg.dense_metric_pilot_iters + 1000;

impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
```

### Reference example

- `examples/LinearRegJointMixed.cpp` — (alpha, beta, sigma) joint with
  REAL + POSITIVE per-slice constraints. (Linear mean coupling, no
  per-unit random effects → dense_metric not required.)
- `examples/HierarchicalLM_MultivariateRE.cpp` — (beta, z_flat, log_tau,
  log_sigma) joint with `cfg.use_dense_metric = true`. Canonical
  HierLM-with-random-effects example.

## stick_breaking_block (truncated SBP — DP / PY / custom)

Closed-form Gibbs leaf that samples a length-K_trunc simplex pi via
truncated stick-breaking (Ishwaran & James 2001). User supplies the
per-stick Beta parameters `a_fn(k, counts, ctx)` and `b_fn(k, counts, ctx)`
as closures; the block knows nothing about which stochastic process
it implements. This makes a single block primitive cover

- **Dirichlet Process (Sethuraman 1994)**: a_k = 1 + n_k,
  b_k = alpha + sum_{j>k} n_j
- **Pitman-Yor (Pitman & Yor 1997)**: a_k = 1 + n_k - discount,
  b_k = alpha + (k+1) * discount + sum_{j>k} n_j
- **Hierarchical / kernel SBP**: any user-defined a_fn, b_fn that
  reads other ctx variables.

State: pi length K_trunc (last stick V_{K_trunc-1} = 1 forced per
Ishwaran-James truncation). Optionally also exposes V_1..V_{K_trunc-1}
under cfg.v_name (set to a non-empty string).

```cpp
stick_breaking_block_config cfg;
cfg.name        = "pi";
cfg.K_trunc     = 20;
cfg.counts_key  = "cluster_counts";   // typically a refresher of z
cfg.v_name      = "stick_V";          // optional output of stick fractions
cfg.a_fn = [](std::size_t k, const arma::vec& counts,
              const block_context& /*ctx*/) -> double {
    return 1.0 + counts[k];           // DP
};
cfg.b_fn = [](std::size_t k, const arma::vec& counts,
              const block_context& ctx) -> double {
    const double a = ctx.at("alpha")[0];
    double tail = 0.0;
    for (std::size_t j = k + 1; j < counts.n_elem; ++j) tail += counts[j];
    return a + tail;
};
cfg.initial_pi = arma::vec(20, arma::fill::value(1.0 / 20));
```

**JUSTIFICATION (Check #16): Exception 1** (per-stick Beta conditional
on a NEW Tier-B block — Ishwaran-James textbook scheme).
**Check #15** parity: `tests_autodiff/block_tests/test_stick_breaking_block.cpp`
verifies analytic E[pi_k] under both empty-counts (GEM(alpha) regime)
and populated-counts DP regimes within 5% on 20k draws.

**Reference examples**: `DPGaussianMixture.cpp`, `PYGaussianMixture.cpp`,
`DPGaussianMixture_DerivedAlpha.cpp`.

## normal_gamma_cluster_gibbs_block (diagonal Gaussian clusters)

Closed-form vectorized Gibbs leaf that samples per-cluster diagonal
Normal-Gamma parameters (mu_k, lambda_k) for k = 1..K_trunc under

    lambda_kd ~ Gamma(a_lambda_0, rate=b_lambda_0)
    mu_kd | lambda_kd ~ N(mu_0_d, 1 / (kappa_0 * lambda_kd))
    y_i ~ N(mu_{z_i}, diag(1 / lambda_{z_i}))

Posterior conjugate update per (k, d) (Bishop PRML §2.3.6 / Murphy 2007
§4). Empty clusters draw from the prior — matches Ishwaran-James 2001
§3.2. Two named outputs: cfg.mu_name (length K_trunc * d, cluster-major
row order) and cfg.lambda_name (same shape; PRECISIONS, not variances).

```cpp
normal_gamma_cluster_gibbs_block_config cfg;
cfg.name        = "cluster_params";   // block label, NOT a data key
cfg.K_trunc     = 20;
cfg.d           = 2;
cfg.N           = 200;
cfg.z_key       = "z";
cfg.y_key       = "y";                // length N * d, row-major
cfg.mu_name     = "mu";               // output key #1
cfg.lambda_name = "lambda";           // output key #2
// DATA-DRIVEN weakly-informative hypers (REQUIRED — see CRITICAL below).
// Per dimension j: mu_0_j = mean(y[,j]); b_lambda_0 = mean_j var(y[,j]).
cfg.mu_0        = col_means_of_y;     // length d, = column means of y
cfg.kappa_0     = 0.01;               // Var[mu] = 100 * E[1/lambda]
cfg.a_lambda_0  = 2.0;                // heavy-tailed, weakly informative
cfg.b_lambda_0  = mean_col_var_of_y;  // Gamma RATE; E[sigma^2] ~ data scale
cfg.initial_mu      = ...;            // data anchors (spread y rows)
cfg.initial_lambda  = ...;            // a_lambda_0 / b_lambda_0
```

> **CRITICAL — DP/BNP mixture cluster-prior hyperparameters MUST be
> data-driven, NOT fixed constants.** A fixed Normal-Gamma prior (e.g.
> `mu_0 = 0`, `b_lambda_0 = 1`) is mis-scaled for any data whose
> location/spread is not ~unit, and the truncated-SBP single-site
> categorical Gibbs then converges to a WRONG over-segmented posterior
> (verified: on variance≈270 data it recovers ≈6 spurious clusters
> instead of the true 3). **R-hat does NOT catch this** — the chains
> stably agree on the wrong posterior; only a posterior-predictive /
> occupied-cluster-count check exposes it. Always compute, in the
> wrapper constructor from `y`: `mu_0_j = mean(y[,j])`,
> `kappa_0 = 0.01`, `a_lambda_0 = 2.0`,
> `b_lambda_0 = mean_j var(y[,j])`, `alpha ~ Gamma(1,1)`. Expose a
> `(y, K_trunc, seed, keep_history)` constructor that does this as the
> DEFAULT; an explicit-hyperparameter constructor may exist for
> advanced control but must not be the documented default. Gold
> reference: `examples/DPGaussianMixture.cpp` (data-driven constructor).

**JUSTIFICATION (Check #16): Exception 1 from codegen_priors.md §2b**
(NEW Tier-B block; conjugate cluster update is the textbook Murphy
2007 §4 formula). **Check #15** parity:
`tests_autodiff/block_tests/test_normal_gamma_cluster_gibbs_block.cpp`
verifies prior-recovery (empty cluster) and posterior-recovery
(populated single cluster) within 5% on 10k draws.

**FULL covariance NIW** is shipped as `niw_cluster_gibbs_block` (NEW
2026-05-03) — see its catalogue entry below. Use the NIW block when
the data has off-diagonal correlation inside clusters; use the
diagonal Normal-Gamma block when dimensions inside a cluster are
approximately independent (smaller state, faster).

**Reference examples**: `DPGaussianMixture.cpp`, `PYGaussianMixture.cpp`,
`DPGaussianMixture_DerivedAlpha.cpp`.

## niw_cluster_gibbs_block (full-covariance Gaussian clusters)

Closed-form vectorized Gibbs leaf that samples per-cluster
(mu_k, Sigma_k) from a conjugate Normal-Inverse-Wishart prior across
K_trunc clusters. Companion to `normal_gamma_cluster_gibbs_block`
(diagonal Σ); use this when within-cluster correlation matters.

```cpp
#include "AI4BayesCode/niw_cluster_gibbs_block.hpp"

niw_cluster_gibbs_block_config cfg;
cfg.name        = "cluster_params";   // block label
cfg.K_trunc     = 20;
cfg.d           = 3;
cfg.N           = 200;
cfg.z_key       = "z";
cfg.y_key       = "y";                // length N * d, row-major
cfg.mu_name     = "mu";               // output: K * d, cluster-major
cfg.sigma_name  = "sigma";            // output: K * d * d, cluster-major,
                                      //         row-major within d×d block
cfg.mu_0        = arma::vec(3, arma::fill::zeros);
cfg.kappa_0     = 0.1;
cfg.Psi_0_flat  = arma::vectorise(arma::eye<arma::mat>(3, 3)); // I_3 row-major
cfg.nu_0        = 5.0;                 // > d - 1 (here > 2). nu - d - 1 > 0
                                       // for E[Sigma|prior] to exist.
cfg.initial_mu      = ...;             // K * d
cfg.initial_sigma   = ...;             // K * d * d (PSD per cluster)
```

**Posterior** (per cluster, n_k > 0): the standard NIW conjugate update
(Murphy 2007 §4 eqs 4-11). **IW sampling**: Bartlett decomposition
(Bartlett 1933, Anderson 2003) — numerically stable, no matrix inverse
on the hot path beyond one inv_sympd of `Psi_n` per cluster per step.

**Empty cluster** (n_k == 0): draws Σ_k ~ IW(Ψ_0, ν_0); μ_k | Σ_k ~
N(μ_0, Σ_k / κ_0). Matches `normal_gamma_cluster_gibbs_block` and
Ishwaran & James 2001 §3.2.

**Output convention**: `sigma_name` exposes COVARIANCE (NOT precision).
This differs from `normal_gamma_cluster_gibbs_block` whose `lambda_name`
is precision. The names disambiguate: `lambda` for precision, `sigma`
for covariance. User downstream blocks (`categorical_gibbs_block`'s
`log_probs_fn`) need to know which they're consuming.

**JUSTIFICATION (Check #16): Exception 1 from codegen_priors.md §2b**
(NEW Tier-B block; conjugate cluster update is the textbook Murphy
2007 §4 formula; Bartlett decomposition is the standard IW sampling
method).

**Check #15** parity:
`tests_autodiff/block_tests/test_niw_cluster_gibbs_block.cpp` covers
empty-cluster prior recovery, populated single-cluster posterior
recovery (mu within 0.4% rel err, Sigma diagonal within 1% rel err
on 4000 draws), and multi-cluster PSD-shape sanity.

**When to pick NIW over Normal-Gamma**: NIW for d ≥ 2 with off-diagonal
structure; Normal-Gamma for d == 1 or for diagonal covariance assumption.
At d = 1, NIW reduces to NormalGamma but normal_gamma_cluster is
cheaper there.

## split_merge_block (Jain-Neal 2004 cluster-partition MH)

Metropolis-Hastings proposal block that toggles a cluster allocation
vector `z` between **merged** and **split** versions in a single
proposal. Accelerates mixing on partitions vs single-i Gibbs.

**Algorithm**: Jain & Neal 2004 "A Split-Merge Markov Chain Monte Carlo
Procedure for the Dirichlet Process Mixture Model". This block
implements the algorithm in the **truncated SBP regime** (NOT
CRP-marginal): `π` is held fixed during the MH ratio, and the prior on
`z` is product-of-categorical Cat(π).

**Algorithm summary** (per step):
1. Sample (i, j) uniformly w/o replacement from {0..N-1}.
2. If `z[i] == z[j]` → SPLIT proposal:
   - Pick `c_new` uniformly from EMPTY slots.
   - Initialise S = {k : z[k] == z[i], k ≠ i, j} randomly to `{s_i, c_new}`.
   - Run T restricted-Gibbs scans; final iteration accumulates log q(z*|z).
   - log A = log q(z|z*) - log q(z*|z) + log_prior + log_lik
3. Else → MERGE proposal:
   - Set z*[k] = s_i for all k in {j} ∪ S.
   - Compute reverse-split q via launch state + T restricted scans
     accumulating at the final iter where target = z[k].

**Empty-slot rule**: SPLIT proposals when no empty slots are available
return immediately as rejected. So K_active is bounded by K_trunc.

```cpp
#include "AI4BayesCode/split_merge_block.hpp"

split_merge_block_config cfg;
cfg.name = "split_merge_z";        // child label
cfg.N = N;
cfg.K_trunc = K_trunc;
cfg.d = d;
cfg.z_name = "z";                  // writes back to shared_data["z"]
cfg.y_key = "y";
cfg.pi_key = "pi";
cfg.mu_key = "mu";
cfg.lambda_key = "lambda";         // exactly ONE of lambda_key / sigma_key
// cfg.sigma_key = "sigma";        // (full covariance path)
cfg.n_restricted_gibbs_iters = 5;  // Jain-Neal §3.1 typical
cfg.initial_z = z_init;
impl_->add_child(std::make_unique<split_merge_block>(std::move(cfg)));

// REQUIRED: `categorical_gibbs_block` and `split_merge_block` both
// drive — and both record to history — the SAME allocation `z`.
// shared_data writes coexist fine (last writer in the sweep wins),
// but composite_block::get_history() rejects a history key emitted by
// two children UNLESS it is explicitly declared shared. Declare it,
// using the SAME key both blocks write (categorical_gibbs `cfg.name`
// == split_merge `cfg.z_name`, here "z"):
impl_->declare_shared_history("z");
```

**Composite integration**: add `split_merge_block` as a child AFTER
`categorical_gibbs_block` in the Gibbs sweep, and call
`impl_->declare_shared_history("z")` (see above). Both blocks write
shared_data key "z" (coexist fine — last writer in the sweep wins) AND
both record "z" to history; the `declare_shared_history` call makes
`get_history()` keep the LAST contributor in scan order — i.e. the
post-split-merge, scan-END `z` draw, which is the correct posterior
chain. Without the declaration `get_history()` throws
`duplicate key 'z' contributed by multiple children`. Each step
alternates: per-i Gibbs → split-merge proposal → cluster_params
update → π update.

**Acceptance asymmetry note**: Jain-Neal's q-density is asymmetric
(merge is deterministic; split is random over restricted Gibbs). So
SPLIT acceptance rates can be high (favoured) and MERGE acceptance rates
can be near 0 even when statistically equivalent. This is **textbook
behavior**, not a bug. The block is designed to accelerate breaking
into well-fitted sub-clusters; merging away of redundant clusters
relies on the categorical Gibbs sweeps depopulating clusters.

**JUSTIFICATION (Check #16): Exception 1** (discrete-allocation MH that
NUTS cannot target). Library-blessed block; user writes no MH
acceptance code.

**Check #15** parity: `tests_autodiff/block_tests/test_split_merge_block.cpp`
verifies (a) smoke + valid output on identical-cluster fixture; (b)
truth-recovery K_active >= 2 on a 2-cluster split-favored fixture;
(c) full-covariance sigma_key path runs.

**No hand-written Jacobian** (Check #5 vacuous — discrete-allocation MH).

## bnp_utils.hpp (Bayesian-nonparametric utility helpers)

Header-only namespace `AI4BayesCode::bnp` with five functions used
by `stick_breaking_block` and any user-written `log_probs_fn` /
refresher / Neal-Algorithm-2/8 composition:

- `counts_from_z(z, K)` — histogram of 1-indexed cluster assignments.
- `crp_log_prior(k, n_minus_i, alpha, N_minus_i)` — CRP allocation
  log-prior for cluster k (or NEW slot at k = K_current).
- `py_log_prior(k, n_minus_i, K_current, alpha, discount, N_minus_i)`
  — Pitman-Yor variant.
- `crp_sample_new_assignment(counts, alpha, rng)` — draw a 0-indexed
  cluster id under CRP weights (returns K to signal NEW).
- `py_sample_new_assignment(counts, alpha, discount, K_current, rng)`
  — PY variant.

These are READ-ONLY pure functions; no class, no state. Useful
when a user writes a refresher that needs to draw a CRP assignment
at predict time, or when implementing a Neal-Alg-2 categorical_gibbs
log_probs_fn.

**Check #15** parity: `tests_autodiff/block_tests/test_bnp_utils.cpp`
verifies all five functions within Check #15 tolerance (analytic +
empirical).

## BNP example summary

| Example | Process | Hyperparam sampler | Cluster sampler | discount |
|---|---|---|---|---|
| `DPGaussianMixture.cpp`             | DP truncated SBP | `nuts_block` on `log(alpha)` | `normal_gamma_cluster_gibbs_block` | n/a |
| `PYGaussianMixture.cpp`             | PY truncated SBP | `nuts_block` on `log(alpha)` (digamma terms) | same as DP | FIXED at construction |
| `DPGaussianMixture_DerivedAlpha.cpp` | DP truncated SBP, alpha = exp(phi) via refresher | `nuts_block` on `phi` (REAL) | same as DP | n/a |
| `FiniteGaussianMixture.cpp`         | **Finite K** (NOT BNP); π via Dirichlet(α/K) symmetric prior | n/a (α fixed at construction) | same as DP | n/a |

**The architectural rule for BNP mixtures with truncated SBP**:

| Parameter | Block | Why |
|---|---|---|
| z (length N) | `categorical_gibbs_block` | Class-1 conditional independence given (pi, mu, lambda) |
| pi (length K_trunc) | `stick_breaking_block` | Per-stick Beta conjugate; user supplies a_fn/b_fn |
| (mu, lambda) | `normal_gamma_cluster_gibbs_block` | Per-(k,d) Normal-Gamma conjugate; empty clusters draw from prior |
| alpha (scalar) | `nuts_block` on `log(alpha)` | Beta likelihood × Gamma prior; closed-form Gamma alternative requires a future `gamma_gibbs_block` |
| (alpha derived) | `register_refresher("alpha", ...)` | Pattern shown in `DPGaussianMixture_DerivedAlpha.cpp` |

Truncation choice: default `K_trunc = max(20, ceil(N / 5))`. The
truncation error decays exponentially in K_trunc for moderate alpha
(Ishwaran & James 2001 §2). For data whose true cluster count is
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
`categorical_gibbs_block` directly — that is the current Q5 lean
(current design lean).

**Split-merge (Jain & Neal 2004)**: not shipped at this version. The existing
`rjmcmc_block` is structured for spike-and-slab (γ, β) partitions, not
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
those choices when the user has selected VI in the codegen.md §3 VI
engine trigger. See `system_design.md §18` for the full architectural
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
  Check #18 / `system_design.md §13` "Metric + warmup decision"). The VI
  alternative is a SINGLE `full_rank_gaussian_vi_block` over the
  concatenated coupled params (merged dim ≤ 50) — the direct analog of
  `joint_nuts_block`. **Do NOT split coupled continuous params into
  separate per-parameter VI blocks** (`q(β)·q(σ²)` coupled through the
  composite by q-sample passing): the cross-block mean-field
  factorization cannot represent the joint covariance, so it
  underestimates marginal variance (measured ~66% too small on a
  collinear pair in the conjugate-NIG regression demo), and the
  per-block PSIS-k̂ is a FALSE reassurance — it scores each block's
  CONDITIONAL fit (given a sibling q-sample), not the joint marginal, so
  a too-narrow q shows a low k̂ while its variance is wrong. Above merged
  dim 50, a single mean-field block over the concatenated vector still
  pays the variance-underestimation cost but gives ONE well-defined
  joint k̂ — at one optimizer step per outer iteration instead of one HMC
  trajectory. See `codegen.md §VI sub-flow Step 2a` for the codegen rule.
- The user has explicitly accepted point-estimate-of-q semantics in
  exchange for tractable speed (deterministic prediction, no chain-
  mixing concerns).

**When NOT to reach for VI** (default to nuts_block / joint_nuts_block):

- Low-dim posteriors with no identification issues — NUTS works.
- Models needing honest credible intervals — mean-field underestimates
  variance.
- Multimodal posteriors where ALL modes matter — VI picks one.
- Discrete latent variables with strong dependence — VI marginals
  collapse to deterministic 0/1 under exclusive KL.

See `system_design.md §18.9` for full caveats and `codegen.md §3 "VI
engine trigger"` for how the trigger question surfaces these to the
user.

## vi_block (abstract base — NOT used directly)

Abstract base for the VI family. Derives from `block_sampler` and
overrides `engine_kind() == engine_kind_t::VI`. Two concrete
subclasses ship in v1: `mean_field_gaussian_vi_block` and
`full_rank_gaussian_vi_block`. Users don't construct `vi_block`
directly.

## mean_field_gaussian_vi_block (primary v1)

```cpp
mean_field_gaussian_vi_block_config cfg;
cfg.name           = "<shared_data key>";
cfg.initial_unc    = arma::vec{...};                // unconstrained mean init
cfg.initial_log_sd = arma::vec(K, arma::fill::value(-2.0));  // log σ init
cfg.constrain      = constraints::positive::constrain;       // omit for real
cfg.unconstrain    = constraints::positive::unconstrain;     // omit for real
cfg.log_density_grad = /* same lambda you'd write for nuts_block */;
cfg.dependencies     = {"y", "X", ...};   // shared_data keys this block reads
cfg.optimizer        = vi_optimizer::raabbvi_config{};  // default: RAABBVI
auto blk = std::make_shared<mean_field_gaussian_vi_block>(cfg);
impl_->add_block(blk);
```

**What it does**: maintains q(η) = ∏_i N(η_i; μ_i, σ_i²) on the
unconstrained scale η = constraints::unconstrain(θ). Each step()
call runs ONE RAABBVI optimizer step (see `vi_optimizer.hpp` below
for the algorithm; `system_design.md §18.7` for the why). After
step(), `composite_block` writes a q-SAMPLE θ = constrain(η_draw)
to shared_data under `cfg.name`; siblings reading that key see a
fresh draw each outer iteration. The R-level `get_current()`
returns the q-mean (point estimate), not a sample.

**Public interface (Tier B)**:

| Method | Returns | Notes |
|---|---|---|
| `step(rng)` | void | one RAABBVI optimizer step |
| `current()` | `const arma::vec&` | q-mean = constrain(μ); deterministic |
| `current_sample(rng)` | `arma::vec` | q-sample = constrain(μ + σ⊙ε), ε~N(0,I) |
| `set_current(μ)` | void | overwrites variational mean ONLY |
| `set_variational_state(μ, log_sd)` | void | overwrites both |
| `get_log_sd()` | `arma::vec` | current log σ |
| `current_elbo()` | double | last-step ELBO |
| `history()` | `const vi_history_t&` | per-step (elbo, μ, log_sd, γ, epoch) + final_khat |

**Constraints**: uses `constraints::*::wrap` identical to
`nuts_block`. The user's log-density lambda lives on the natural
scale; no hand-written Jacobian. Validator Check #5 unchanged.

**Validator obligations**:
- Layer 2 Check #21 (VI block contract conformance) — mandatory.
- Layer 2 Check #22 (VI optimizer = RAABBVI) — mandatory.
- The Layer-3 R2-VI PSIS-k̂ diagnostic (joint PSIS-k̂ < 0.7) — mandatory
  pre-merge.

**Reference tests** (`tests/`, non-shipped but illustrative — show the
hybrid composition + non-centered reparameterization pattern):

- `test_vi_glmm_logistic.cpp` — Hierarchical Bayesian logistic GLMM with
  random intercepts. VI on the J random effects (non-centered z_j);
  NUTS on β, α_0, log σ_α. The canonical applied-stats VI use case
  (multi-site / multi-school / multi-group binary outcomes).
- `test_vi_bnn_regression.cpp` — 1-hidden-layer Bayesian neural network
  for continuous regression. VI on the non-centered weights
  (α_0, Ã, β_0, β̃); NUTS on log σ_α, log σ_β, log σ_y. The
  canonical ML VI use case (high-dim weights with hierarchical scale
  funnel — Bishop 2006 §10.1, Welandawe 2022, see system_design §18).
- `test_vi_hybrid_composition.cpp` — Bayesian ridge regression with
  sampled coefficient-prior scale. The minimal hybrid demo.
- `test_vi_mean_field_gaussian.cpp`, `test_vi_correlated_gaussian.cpp`
  — pure-VI baselines on Gaussian targets (k̂ PASS / k̂ FAIL respectively,
  documenting Bishop §10.1.2 variance-underestimation caveat).

## full_rank_gaussian_vi_block (v1 SHIPPED 2026-05-26)

```cpp
full_rank_gaussian_vi_block_config cfg;
cfg.name        = "<shared_data key>";
cfg.initial_unc = arma::vec{...};                              // K-dim mean
cfg.initial_L   = arma::mat(K, K, arma::fill::eye) * 0.1;      // L init (lower-triangular)
cfg.constrain   = constraints::positive::constrain;            // omit for real
cfg.unconstrain = constraints::positive::unconstrain;
cfg.log_density_grad = /* same lambda as nuts_block / mean-field VI */;
cfg.dependencies     = {"y", "X", ...};
cfg.optimizer        = vi_optimizer::raabbvi_config{};
auto blk = std::make_shared<full_rank_gaussian_vi_block>(cfg);
impl_->add_block(blk);
```

**What it does**: maintains q(η) = N(η; μ, LL^T) on the unconstrained
scale, with L lower-triangular Cholesky factor — captures FULL posterior
correlations among the K coordinates. Parameter count = K + K(K+1)/2,
quadratic in K. Use when mean-field's independence assumption fails:
- Regression coefficients with collinear predictors
- BNN output-layer weights conditional on hidden representations
- Hierarchical scale × parameter funnel where MF underestimates the
  marginal variance
- Any target where the posterior has off-diagonal covariance you want
  to capture

**Empirical comparison** (tests/test_vi_full_rank_correlated.cpp, 2D
ρ=0.95 correlated Gaussian, mean-field-vs-full-rank apples-to-apples):

| Quantity | Truth | Mean-field VI | Full-rank VI |
|---|---:|---:|---:|
| marginal sd | 1.00 | 0.31 (conditional, biased) | **0.99 (matches truth)** |
| correlation ρ | 0.95 | 0 (by construction) | **0.948** |
| PSIS-k̂ | — | 0.80 (FAIL — q ≠ p) | **−0.14 (q ≈ p exact)** |

The 0.31 mean-field value is exactly √(1−ρ²) = the CONDITIONAL sd,
which is what mean-field collapses to — textbook Bishop §10.1.2.

**Caps**: auto-suggest K ≤ 50; warn 50 < K ≤ 200; reject K > 500
(too expensive — switch back to mean-field or restructure).

**Public interface**: identical to `mean_field_gaussian_vi_block`,
plus `get_L() → const arma::mat&` (current lower-triangular Cholesky
factor).

**Constraints + validator obligations**: same as
`mean_field_gaussian_vi_block`.

## mean_field_categorical_vi_block (v1.2 SHIPPED 2026-05-31)

```cpp
mean_field_categorical_vi_block_config cfg;
cfg.name          = "z";                          // shared_data key for q-sample
cfg.cardinalities = arma::uvec{2, 3, 2, 4};       // K_i per latent variable
cfg.log_density   = [](const arma::uvec& z,
                       const block_context& ctx) -> double {
    // user-supplied log p~(z, ctx), evaluated on integer-encoded z
};
cfg.dependencies     = {"y", "X"};
cfg.exact_enumeration = false;                    // MC by default
cfg.n_mc_samples      = 16;                       // gradient MC samples
cfg.optimizer         = vi_optimizer::raabbvi_config{};
auto blk = std::make_shared<mean_field_categorical_vi_block>(cfg);
impl_->add_child(std::move(blk));
```

**What it does**: maintains `q(z) = ∏_i Categorical(z_i; φ_i)` over
`n` discrete latents (cardinalities `K_i` per variable), with internal
anchored-softmax parameterisation `η_i ∈ R^{K_i−1}` and `φ_{i,0}`
anchored to `1 − Σ_{l>0} φ_{i,l}`. Optimises `−ELBO` via RAABBVI on
the analytically-correct gradient `∂ELBO/∂φ_{i,k} = G_{i,k} − log
φ_{i,k} − 1` where `G_{i,k} = E_{q\i}[log p̃(z_{-i}, z_i = k)]`.

**Two gradient modes** (selected by `cfg.exact_enumeration`):
- **Exact**: enumerate full joint state space `∏_i K_i` (capped at
  `cfg.exact_state_cap`, default 4096). Used for unit tests and small
  models. Zero variance.
- **Monte Carlo** (default): sample `S = cfg.n_mc_samples` joint
  draws `z_s ~ q`; for each `(i, k)`, substitute `z_s[i] := k` and
  evaluate `log_density`. Cost = `S × Σ_i K_i` evaluations per step.

**JUSTIFICATION (Check #16)**: Discrete latents with strong local
dependence (`system_design.md` §11.2(b)). Per-site Gibbs irreducible
but mixes catastrophically on coupled chains; categorical mean-field
VI is the deterministic deterministic alternative — converges cleanly
to a (biased) joint with correct marginals in many regimes, with
known underestimate-of-joint-variance caveat (Bishop §10.1.2)
correctly diagnosed by PSIS-k̂.

**Empirical validation** (`tests/test_mean_field_categorical_vi_chain.cpp`
on 4-node K=3 Potts chain at β=0.8 vs 81-state exact enumeration):

| Test | Result |
|---|---|
| Symmetric chain, per-var KL(q‖p) | < 1e-5 (all 4 vars) |
| Asymmetric chain (h ≠ 0), per-var KL | max 0.00144 |
| Asymmetric chain, Pearson cor(q,p) on 12-vec | **0.999** |
| ELBO ≤ log Z (variational bound) | PASS |
| Joint PSIS-k̂ on coupled chain | 5.78 (FAIL — correctly diagnoses MF bias) |

**Public interface** (`vi_block` contract):
- `current()` returns concatenated φ (length `Σ_i K_i`) — the q-mean
  point estimate
- `current_sample(rng)` returns one z draw (length `n` integer
  indices in `{0..K_i-1}`)
- `get_log_sd()` returns the unconstrained η vector (length `Σ_i (K_i−1)`)
- `set_current(phi_concat)` accepts a probability vector and recovers
  η via inverse softmax
- composite_block writes integer-indexed q-samples to shared_data
  under `cfg.name` (per §18.4 hybrid-correctness invariant)

**Reference template:**
- `examples/CategoricalIsingChainVI.cpp` — n-node K-state Potts chain
  with optional per-node external field `h`, RAABBVI-converged VI
  marginals + predict_at sampling. R-level audit
  (`tests/audit_CategoricalIsingChainVI.R`) cross-checks per-var KL
  < 0.05 vs 81-state exact enumeration on n=4 K=3 chain.

**Caveats (Bishop §10.1.2, validated by k̂ diagnostic)**:
- Marginals recovered well; joint covariance NOT captured by MF
- For tightly coupled targets (large β, dense graph), use Block 5
  (`structured_categorical_vi_block`, v1.2 future) with user-specified
  clique factorisation, OR drop to MCMC

**Scope (v1.2 ship):**
- Cardinality `K_i ≥ 2` per variable, arbitrary mix across nodes
- Generic `log_density` callback — no exponential-family assumption
- Both exact and Monte Carlo gradient modes
- RAABBVI optimizer + PSIS-k̂ Layer-3 diagnostic

**Deferred to v1.2.1:**
- Structured factorisation via user-specified cliques (this is Block 5)
- CAVI closed-form updates when `log_density` is exp-family (faster
  alternative for that special case)
- Continuous-relaxation alternatives (Concrete / Gumbel-softmax) —
  explicitly rejected by plan §4 (analytical-KL is more accurate)

**Constraints + validator obligations**: same as
`mean_field_gaussian_vi_block` (Checks #21–#22 plus the Layer-3 R2-VI
PSIS-k̂ diagnostic).

## structured_categorical_vi_block (v1.2 SHIPPED 2026-05-31)

```cpp
structured_categorical_vi_block_config cfg;
cfg.name             = "z";
cfg.cardinalities    = arma::uvec{3, 3, 3, 3};
cfg.clique_partition = { {0, 1}, {2, 3} };    // partition of {0..n-1}
cfg.log_density      = /* same callback as Block 4 */;
cfg.dependencies     = {"y", "X"};
cfg.exact_enumeration = false;
cfg.n_mc_samples      = 16;
cfg.optimizer         = vi_optimizer::raabbvi_config{};
auto blk = std::make_shared<structured_categorical_vi_block>(cfg);
impl_->add_child(std::move(blk));
```

**What it does**: implements **Saul-Jordan 1996** "structured" (a.k.a.
**partially-factorised**) mean-field VI. Refines Block 4 by preserving
intra-clique correlation exactly while factorising ACROSS user-defined
cliques:

```
Block 4 (fully factorised):    q(z) = ∏_i Cat(z_i; φ_i)
Block 5 (structured):          q(z) = ∏_C Cat(z_C; φ_C)
```

For clique C with joint state count `S_C = ∏_{i∈C} K_i`, q_C is an
anchored-softmax Categorical over `S_C` states. Total free parameters
= `∑_C (S_C - 1)` — exponential in clique size, so cliques must be
small (4-5 nodes is typical).

**Gradient**: analytical sum-over-CLIQUE-state (vs Block 4's
sum-over-variable-state). Exact enumeration mode for small joint state
spaces; Monte Carlo otherwise. Closed-form chain rule through the
anchored softmax exactly mirrors Block 4.

**JUSTIFICATION (Check #16)**: discrete latents with strong local
dependence (`system_design.md` §11.2(b)). Structured MF gives
dramatically better approximations than Block 4 when the user can
identify strong-coupling clusters; degenerates exactly to Block 4
under singleton cliques, and to exact inference under a single
all-encompassing clique (both unit-tested).

**Degeneracies (correctness invariants, unit-tested)**:
- **Singleton cliques** `{{0},{1},...,{n-1}}` ⇒ block IS Block 4.
  Test `S2` verifies per-node marginals match Block 4 to < 0.01.
- **Single grand-clique** `{{0,1,...,n-1}}` ⇒ block performs EXACT
  inference (joint φ matches normalised `exp(log p~)`). Test `S3`
  verifies max joint diff < 0.001 over 81 states.

**Empirical head-to-head with Block 4** (4-node K=3 chain with strong
β_intra=1.5 and weak β_inter=0.3, cliques {{0,1},{2,3}}):

| Metric | Block 4 (MF) | Block 5 (structured) | Improvement |
|---|---:|---:|---:|
| KL(joint ‖ exact) | 0.465 | 0.010 | **46×** |
| KL(per-node ‖ exact) | 0.025 | 5e-6 | **~5000×** |
| KL(pairwise ‖ exact) | 0.219 | 5e-7 | **~400000×** |
| ELBO | 5.958 | **6.414** | +0.456 |
| log Z (truth) | — | — | 6.425 |

Block 5's ELBO is **0.011 below log Z** — the variational bound is
essentially tight. Within-clique pairwise marginal `P(z_0, z_1)` is
captured EXACTLY in clique C1, vs Block 4's forced product-of-marginals
that washes out the correlation.

**Public interface** (`vi_block` contract):
- `current()` returns concatenated per-clique joint φ_C (length `∑_C S_C`)
- `current_sample(rng)` returns one z draw (length n integer indices)
- `per_node_marginals()` helper computes per-node marginal matrix
  by marginalising the clique joint φ
- `set_current(phi_concat)` accepts per-clique joint probabilities and
  recovers η via inverse anchored softmax
- composite_block writes integer-indexed q-samples to shared_data
  under `cfg.name` (per §18.4 hybrid-correctness invariant)

**Reference template**:
- `examples/StructuredPottsVI.cpp` — n-node K-state Potts on arbitrary
  graph (user-supplied edges + per-edge couplings + per-node external
  field + clique partition).
- R audit `tests/audit_StructuredPottsVI.R` — Pearson cor 1.00000 vs
  exact 81-state enumeration over both cliques.

**Validation against independent paths** (`tests/test_structured_*.cpp`):
| Path | Algorithm | Status |
|---|---|---|
| #1 | RAABBVI gradient (block under test) | reference |
| #2 | Bishop CAVI lifted to clique level (closed-form fixed point) | matches VI to 1e-4 |
| #3 | Single-site Gibbs MCMC + R-hat across 4 chains | R-hat 1.00003 |
| #4 | Exact 81-state enumeration | matches VI to 2e-4 |

VI multi-init consistency across 4 init seeds: max deviation **0.000026**.

**Scope (v1.2 ship)**:
- Arbitrary user-specified clique partition (each node in exactly one
  clique)
- Cardinality `K_i ≥ 2` per node, arbitrary mix
- Generic `log_density` callback — no exponential-family assumption
- Both exact and Monte Carlo gradient modes
- RAABBVI optimizer + PSIS-k̂ Layer-3 diagnostic

**Deferred to v1.2.1**:
- Saul-Jordan §2.3 "inducing partial factorizability" — auxiliary
  variables W_ij to decouple intra-clique intractable couplings
- Overlapping cliques (junction-tree style) — current v1 requires a
  proper partition
- Closed-form CAVI alternative when `log_density` is exp-family

**Constraints + validator obligations**: same as
`mean_field_gaussian_vi_block` (Checks #21–#22 plus the Layer-3 R2-VI
PSIS-k̂ diagnostic).

## vi_optimizer.hpp (helper utilities, NOT a block)

Header-only port of Welandawe et al. 2022 RAABBVI. Exposes:

- `vi_optimizer::raabbvi_config` — POD struct with `gamma_0` (initial
  learning rate, default 0.1), `rho` (geometric decay between epochs,
  default 0.5), `tau` (SKL termination threshold, default 0.1),
  `W_min` (minimum R̂-convergence window, default 100), `max_epochs`
  (cap on outer loops, default 50), `S_khat` (samples for terminal
  k̂ computation, default 1000).
- `vi_optimizer::raabbvi<GradFn>` — templated struct taking a
  gradient functor and orchestrating avgAdam + Polyak-Ruppert
  iterate averaging + R̂ convergence at fixed γ + SKL termination.
- `vi_optimizer::psis_khat(log_weights) → double` — Pareto-smoothed
  importance-sampling k̂ computation; used by VI blocks at SKL
  termination and by `psis_diagnostic.hpp` for Layer-3 R2-VI.

Users do NOT construct `vi_optimizer::raabbvi` directly; configure
via the VI block's `cfg.optimizer` field.

See `system_design.md §18.7` for full algorithm details.

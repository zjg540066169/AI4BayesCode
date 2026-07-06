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
are ALSO usable in a generated sampler — the compile step puts both on the `-I`
path so `#include "<Block>.hpp"` resolves:

| Tier | Where | Trust |
|---|---|---|
| **local** | `./blocks_local/<Block>/` — **project-relative** (resolved against the current working directory): blocks you are developing or keep with this project | self |
| **downloaded** | `~/.AI4BayesCode/blocks_download/<Block>/` — **user-global**, installed via `ai4bayescode_install_block()`; one shared store across R / Python / C++ **and** across all your projects (override root with `$AI4BAYESCODE_DATA_HOME`) | provenance |
| **core** | this file | maintainer-vetted |

**Priority when several fit: local > downloaded > core** (prefer the user's own /
reviewed block over a generic core one). Names are unique across DISTINCT blocks
(core names reserved; new local / downloaded names are checked at creation /
install). The ONE deliberate same-name case: after you develop a block locally,
publish it, and re-download it, `./blocks_local/<Block>/` and
`~/.AI4BayesCode/blocks_download/<Block>/` hold the SAME block. That is not a
conflict to resolve — the tooling **deduplicates by name and local wins** (your
live working copy shadows the published snapshot) for BOTH discovery below AND
the compile `-I` path. So list local rows first and, when a name also appears
under downloaded, keep only the local row.

**Two-stage, token-bounded selection** — do this BEFORE settling on a core block;
it stays cheap no matter how many blocks are installed (bounded by relevance, not
by block count):

1. **Index (cheap).** Build a one-line-per-block index from the manifests:
   `glob ./blocks_local/*/manifest.dcf`, then
   `glob ~/.AI4BayesCode/blocks_download/*/manifest.dcf`, and read each manifest's
   `Block` + `RoutingKey` (one short line each). Do NOT read the full cards yet.
   (No local/downloaded blocks present → skip to core.)
2. **Filter.** From that index (local rows first, then downloaded), pick the 5–10
   blocks whose `RoutingKey` plausibly matches the model.
3. **Read cards — survivors only.** For each candidate read its `SelectWhen` and
   full card `./blocks_local/<Block>/skills/<Block>.md` (or
   `~/.AI4BayesCode/blocks_download/<Block>/skills/<Block>.md`); choose the best fit.
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
| **Gaussian-data GMRF with UNKNOWN smoothing precision** (`y ~ N(x, σ²I)`, `x ~ N(0,(κR)⁻¹)`, `κ ~ Gamma`; slow (x,κ) Gibbs mixing) | **`gmrf_gaussian_joint_block`** (Knorr-Held & Rue 2002 collapsed joint (x,κ) update) | **(none — κ from its marginal via sparse-Cholesky log-det, then x\|κ,y)** |
| **sparse-precision GMRF + non-Gaussian likelihood** (`pi(x) ∝ exp{-½ x^T Q x}` · `log_lik(x)`; Poisson / Bernoulli / Student-t / NB / log-Gaussian Cox observation) | **`gmrf_whitened_ess_block`** (Murray 2010 ESS on implicit GMRF prior; Rue 2001 backsolve for prior draws) | **(none — Eigen SimplicialLLT + permuted backsolve + likelihood-free ESS shrink)** |
| **Bayesian-network structure learning** (discrete data, n ≤ 64, BDeu score; output = total order ≺ + Bayesian-model-averaged DAG over Pa(i) ⊂ Pred(i, ≺)) | **`order_mcmc_block`** (Friedman-Koller 2003 order MCMC + Heckerman-Geiger-Chickering 1995 BDeu + FK §4.2 three-tier cache) | **(none — combinatorial MH on permutations)** |
| **LDA token topic assignment z_n in {1..K} + theta_d (M-doc simplex) + phi_k (V-vocab simplex), Dirichlet hyperpriors** | **`lda_collapsed_gibbs_block`** (Griffiths-Steyvers 2004 collapsed Gibbs) | **(none — joint output of z, theta, phi)** |
| **Bernoulli response (logistic reg)** | **`pg_logistic_block` (PG augmentation)** | **(none)** |
| Gaussian mean f, **constant** noise (standard BART) — **DEFAULT for any real-valued response with constant variance; the FAST conjugate BART** | `bart_block` | (none, GPL-2.0+) |
| **NON-Gaussian response ONLY** (Poisson / NB / logistic / heteroscedastic / AFT / beta / gamma_shape / beta_binomial / custom) — a likelihood `bart_block` cannot express **even after the reduction ladder** (Gaussian direct / augmentation / backfitting / known-weights / working-response — see the `genbart_block` card "When to use") | **`genbart_block` + `genbart::lik::*`** — ⚠️ generic RJMCMC (Laplace leaf proposals), **much slower** than `bart_block`; **NEVER use `genbart_block + normal_lik` for plain Gaussian regression — use `bart_block`** | **(none, GPL-2.0+)** |
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

## Core block cards — read ONLY the one you need

After the Block type table above routes you to a block, read that block's card
for its full configuration / discipline (do NOT read them all):

- **`nuts_block`** -> `block_catalogue/nuts_block.md`
- **`dirichlet_gibbs_block`** -> `block_catalogue/dirichlet_gibbs_block.md`
- **`beta_gibbs_block`** -> `block_catalogue/beta_gibbs_block.md`
- **`binary_gibbs_block`** -> `block_catalogue/binary_gibbs_block.md`
- **`probit_aug_block`** -> `block_catalogue/probit_aug_block.md`
- **`categorical_gibbs_block`** -> `block_catalogue/categorical_gibbs_block.md`
- **`lda_collapsed_gibbs_block`** -> `block_catalogue/lda_collapsed_gibbs_block.md`
- **`gamma_gibbs_block`** -> `block_catalogue/gamma_gibbs_block.md`
- **`inv_gamma_gibbs_block`** -> `block_catalogue/inv_gamma_gibbs_block.md`
- **`pg_logistic_block`** -> `block_catalogue/pg_logistic_block.md`
- **`hmm_block`** -> `block_catalogue/hmm_block.md`
- **`ising_cluster_block`** -> `block_catalogue/ising_cluster_block.md`
- **`gmrf_precision_block`** -> `block_catalogue/gmrf_precision_block.md`
- **`gmrf_gaussian_joint_block`** -> `block_catalogue/gmrf_gaussian_joint_block.md`
- **`gmrf_whitened_ess_block`** -> `block_catalogue/gmrf_whitened_ess_block.md`
- **`order_mcmc_block`** -> `block_catalogue/order_mcmc_block.md`
- **`rjmcmc_block`** -> `block_catalogue/rjmcmc_block.md`
- **`bart_block`** -> `block_catalogue/bart_block.md`
- **`genbart_block`** -> `block_catalogue/genbart_block.md`
- **`poisson_multinomial_aug_block`** -> `block_catalogue/poisson_multinomial_aug_block.md`
- **`elliptical_slice_sampling_block`** -> `block_catalogue/elliptical_slice_sampling_block.md`
- **`univariate_slice_sampling_block`** -> `block_catalogue/univariate_slice_sampling_block.md`
- **`celerite_marginal_likelihood.hpp`** -> `block_catalogue/celerite_marginal_likelihood_hpp.md`
- **`celerite_gp_block`** -> `block_catalogue/celerite_gp_block.md`
- **`GP reference examples summary`** -> `block_catalogue/gp_reference_examples_summary.md`
- **`GP convergence troubleshooting ladder`** -> `block_catalogue/gp_convergence_troubleshooting_ladder.md`
- **`GP composition recipes`** -> `block_catalogue/gp_composition_recipes.md`
- **`Reduced-rank / spline / spatial patterns`** -> `block_catalogue/reduced_rank_spline_spatial_patterns.md`
- **`Vendored libraries summary`** -> `block_catalogue/vendored_libraries_summary.md`
- **`joint_nuts_block`** -> `block_catalogue/joint_nuts_block.md`
- **`joint_nuts_block — per-slice constraints`** -> `block_catalogue/joint_nuts_block_per_slice_constraints.md`
- **`stick_breaking_block`** -> `block_catalogue/stick_breaking_block.md`
- **`normal_gamma_cluster_gibbs_block`** -> `block_catalogue/normal_gamma_cluster_gibbs_block.md`
- **`niw_cluster_gibbs_block`** -> `block_catalogue/niw_cluster_gibbs_block.md`
- **`split_merge_block`** -> `block_catalogue/split_merge_block.md`
- **`bnp_utils.hpp`** -> `block_catalogue/bnp_utils_hpp.md`
- **`BNP example summary`** -> `block_catalogue/bnp_example_summary.md`
- **`vi_block`** -> `block_catalogue/vi_block.md`
- **`mean_field_gaussian_vi_block`** -> `block_catalogue/mean_field_gaussian_vi_block.md`
- **`full_rank_gaussian_vi_block`** -> `block_catalogue/full_rank_gaussian_vi_block.md`
- **`mean_field_categorical_vi_block`** -> `block_catalogue/mean_field_categorical_vi_block.md`
- **`structured_categorical_vi_block`** -> `block_catalogue/structured_categorical_vi_block.md`
- **`vi_optimizer.hpp`** -> `block_catalogue/vi_optimizer_hpp.md`

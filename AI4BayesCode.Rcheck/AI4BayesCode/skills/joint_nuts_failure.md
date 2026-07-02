---
name: AI4BayesCode-joint-nuts-failure
description: |
  Failure modes of joint_nuts_block (the default sampler for coupled
  continuous parameters) and how to fix them BEFORE shipping. Covers the
  three v1 modes — (1) hierarchical funnel + the mandatory non-centered
  reparameterization (NCR) recipe, (2) multi-modal / label-switching,
  (3) high-dim joint slowness — plus the escalation ladder. The funnel
  NCR recipe lives HERE (codegen_priors.md and validator Check #24 point
  here). Consult this skill BEFORE emitting a joint_nuts_block, and when a
  joint model shows pathology (max tree depth, divergences, R-hat large,
  a scale parameter stuck near zero, or random-effect ESS = NA).
---

# joint_nuts_block — failure modes + fixes

`joint_nuts_block` samples all coupled continuous parameters jointly over a
single concatenated unconstrained vector. It is the **default** for coupled
continuous targets (block-decomposed per-parameter NUTS mixes ~10× slower per
ESS when parameters are correlated, and FREEZES outright on hierarchical
funnels — see Mode 1 evidence). But joint NUTS has three well-characterized
failure modes. Each is detectable and fixable; this skill is the reference.

## When to consult

- **Codegen agent**: BEFORE emitting any `joint_nuts_block`, check the prompt
  against Mode 1 (funnel) — NCR is **mandatory** when the pattern matches.
- **Validator**: target reference when Check #24 (joint-NUTS pre-flight) fails.
- **Debug**: lookup when a joint model shows pathology at runtime.

---

## Failure Mode 1 — Hierarchical funnel (MOST COMMON; pre-codegen detectable)

### Symptom
- Tree depth hits max / frequent divergences.
- The scale parameter (τ, σ) gets stuck near 0.
- Random-effect parameters report **ESS = NA** (a chain has ~zero within-chain
  variance — it is FROZEN) and **R-hat ≫ 1.05**; coverage collapses.

### Why
Centered parameterization `θ_j ~ Normal(μ, τ)` couples θ to τ: when τ is small
the θ_j are squeezed into a tight neck, when τ is large they spread out. A
single global step size cannot navigate both the wide body and the tight neck,
so the sampler either diverges (step too big at the neck) or freezes (step too
small in the body). Per-block Gibbs is even worse: each block's step is tuned
to one slice of the funnel and is wrong for the rest.

### Detection (from the prompt, before codegen)
Match this structural pattern:
```
scale  ~ HalfNormal / HalfCauchy / InvGamma / Exponential   (a POSITIVE hyper-scale)
for j:  raw_j ~ Normal(<loc>, scale * <const>)               (effects drawn at that scale)
```
`<loc>` may be 0, a global mean μ, or a linear predictor.

### Fix — MANDATORY non-centered reparameterization (NCR)
Do **not** keep the centered form on a joint_nuts_block. Centered + joint NUTS
on a funnel diverges or maxes tree depth on essentially every iteration.

```
CENTERED  (BAD — funnel):
    τ      ~ HalfNormal(s)                 # POSITIVE
    θ_j    ~ Normal(μ, τ)                  # tightly coupled to τ

NON-CENTERED  (GOOD — flat geometry):
    τ      ~ HalfNormal(s)                 # POSITIVE slice
    η_j    ~ Normal(0, 1)                  # REAL slice (standardized)
    θ_j   := μ + τ · η_j                   # DETERMINISTIC — NOT sampled
```

Rules:
- The `joint_nuts_block` contains `(μ [REAL], τ [POSITIVE], η_1..p [REAL])`.
- `θ_j` is **computed inside the natural-scale log-density** (to evaluate the
  likelihood) and, if a downstream block needs it, in a **deterministic
  refresher** reading μ/τ/η — it is NEVER a sampled sub-parameter.
- The natural-scale log-density writes the priors on μ, τ, η plus the
  likelihood `y ~ f(μ + τ·η)`. It writes **NO Jacobian term for τ** — the
  POSITIVE slice adds `log|J|` automatically (`constraints::positive::wrap`;
  system_design.md §10.1). Check #5 still forbids a hand-written `+ log(τ)`.
- NCR does NOT change τ's own prior — keep the Jeffreys / half-normal choice
  per system_design.md §11.6.

### API (unified joint_nuts_block; constraint kind per slice)
```cpp
joint_nuts_block_config cfg;
cfg.name = "hier_joint";
cfg.sub_params = {
    {"mu",  1, joint_constraint::REAL},      // global mean
    {"tau", 1, joint_constraint::POSITIVE},  // hyper-scale  (block adds log|J|)
    {"eta", J, joint_constraint::REAL},      // standardized raw effects
};
// initial values are NATURAL-scale (tau > 0). The block maps to the
// unconstrained sampler state internally.
cfg.initial_cat = arma::join_cols(arma::vec{0.0}, arma::vec{1.0},
                                   arma::zeros(J));
// natural-scale log p + grad. theta_j = mu + tau*eta_j computed HERE for the
// likelihood; gradient is w.r.t. (mu, tau, eta) on the natural scale.
cfg.log_density_grad = [/*data*/](const arma::vec& nat,
                                  const block_context& ctx,
                                  arma::vec* g) -> double { /* ... */ };
```

### Empirical evidence (this repo, 2026-06-11)
`eight_schools_centered` sampled per-block + centered FROZE: all 8 θ at
ESS = NA, R-hat ≈ 2.23, coverage 0.10 — while the reference (mcmclib, same
centered model) reached coverage 0.90. Lowering the step size (target_accept
0.8→0.95) gave a **bit-identical** failure: the chains were frozen, not
mis-stepped. A diagonal-mass (`adapt_mass`) workaround rescued *this* model
(cov 0.10→0.90) but REGRESSED tight time-series (arma11) and a harder funnel
(Mh) — it is a per-model band-aid, not a fix. **The real fix is joint + NCR**
(this mode): it removes the funnel geometry at the source, model-independently.

---

## Failure Mode 2 — Multi-modal / label-switching (partially detectable)

### Symptom
Chains split into different modes; R-hat is huge on the symmetric parameters,
yet each chain looks well-mixed internally; ESS may look fine per-chain.

### Detection
- Mixture components (label switching across the K components).
- Factor / loading models (sign flips).
- Any exact permutation/sign symmetry in the prompt's parameterization.

### Fix
Add an **identifying constraint** that breaks the symmetry, e.g.:
- `ordered` constraint on component means (μ_1 < μ_2 < …),
- positivity on the first factor loading,
- a fixed reference category.
If an identifying constraint is unnatural, ACCEPT the symmetry and report with
**label-invariant summaries** (e.g. sorted component parameters) rather than
per-label R-hat.

### Empirical evidence (this repo)
`hmm_drive_0`: φ[1] and φ[2] are exchangeable transition rows → R-hat ≈ 2.23
from label switching, NOT a sampler bug. Diagnosed via per-parameter R-hat:
the symmetric pair is high, the rest fine.

---

## Failure Mode 3 — High-dimensional joint, slow (detectable from param count)

### Symptom
Wall-time per iteration is very large; R-hat is OK.

### Detection
Param count above ~50–100 AND the likelihood **factorizes** (conditional
independence across a grouping index).

### Fix
Split the single joint group into smaller joint groups along the conditional-
independence structure (each group < ~50 params). Automated "obvious
independence" auto-split is a v1.2 codegen feature (F4); until then, split
manually when the factorization is clear, or accept the cost. Do NOT split
across a genuine coupling — that silently biases inference.

---

## Escalation ladder

```
Level 1: NCR (funnel)  OR  reparameterize (constraint boundary)
Level 2: split into smaller joint groups via conditional independence
Level 3: fall back to per-parameter nuts_block (correct, just slower per ESS)
Level 4: declare "needs a specialized algorithm" (HMM / GMRF / order_mcmc)
         or "needs problem reformulation"
```

Always prefer the lowest level that resolves the pathology. Level 3 (per-param
nuts_block) is always available as a correctness fallback — it is slower but
never wrong-target — EXCEPT on funnels, where per-param NUTS itself freezes
(Mode 1); there, Level 1 (NCR) is required, not optional.

---

## Relationship to validator Check #24

Check #24 (joint-NUTS pathology pre-flight) is the static guard. It verifies,
when a `joint_nuts_block` is present:
1. **Funnel NCR done** — if the prompt declares the Mode-1 pattern, the cpp
   must contain the non-centered form (`eta_j` declared; `theta_j := mu +
   tau*eta_j` deterministic), not the centered form.
2. **Constraint kinds match** — each slice's `joint_constraint` matches the
   prompt's support (scale → POSITIVE, mean/effect → REAL).
3. **Lambda completeness** — every declared sub-parameter is read inside the
   joint log-density (a missing read = silent prior-only sampling).

A Mode-1 match with a centered cpp is a **FAIL_24(a)**: the fix is this skill's
NCR recipe.

---

## v1.2 deferred modes (documented for forward reference, not yet shipped)

- **Mode 4 — Mixed-scale posteriors**: parameter scales spanning orders of
  magnitude; the single mass matrix can't adapt to both. Fix: per-parameter
  rescaling, or fall back to per-param nuts_block (Mode-3 escalation).
- **Mode 5 — Constraint-boundary hits**: a positive parameter hugging 0;
  divergences at the boundary. Fix: tighter / offset prior.
- **Mode 6 — Weak identification / plateau**: high autocorrelation, posterior
  ≈ prior on some parameter. Fix: tighten the prior, or report
  "non-identifiable" in the model card.

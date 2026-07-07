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
Match EITHER form — the funnel geometry is identical whether the positive
hyperparameter is a standard deviation OR a variance:
```
(A) SD-parameterized:
    scale ~ HalfNormal / HalfCauchy / Exponential            (a POSITIVE sd)
    for j:  raw_j ~ Normal(<loc>, scale * <const>)           (sd used DIRECTLY)

(B) VARIANCE- or PRECISION-parameterized — the Gaussian-hierarchical / ARD /
    Neal-1996 BNN standard (weight-variance priors; e.g. nn_rbm, ridge/ARD):
    var  ~ InvGamma / Scaled-Inv-χ²   (POSITIVE variance)   ->  sd = sqrt(var)
    prec ~ Gamma                      (POSITIVE precision)  ->  sd = 1 / sqrt(prec)
    for j:  raw_j ~ Normal(<loc>, sd * <const>)             (sd = sqrt(var) OR 1/sqrt(prec))
```
`<loc>` may be 0, a global mean μ, or a linear predictor.

**Form (B) is at least as common as (A) and is the one that gets missed** — the
positive slice is a VARIANCE and the sd entering the Normal is its `sqrt`, so the
literal word "scale" never appears. `InvGamma` / `Gamma`-on-precision almost
always signal form (B). Do NOT dismiss `var ~ InvGamma; θ ~ Normal(0, sqrt(var))`
as a non-funnel because it is variance-parameterized — it IS Mode 1, and centered
+ joint NUTS (or Gibbs on the conjugate `var`) freezes on it exactly as in (A).

### Fix — non-centered reparameterization (NCR); the DEFAULT for weak-data funnels
Centered + joint NUTS on a weak-data funnel diverges or maxes tree depth on
essentially every iteration, so NCR is the default — but it is NOT unconditional
(see the caveat below).

```
CENTERED  (BAD on a weak-data funnel):
    τ ~ HalfNormal   # sd   -- OR --   σ² ~ InvGamma   # variance   -- OR --   ω ~ Gamma   # precision
    θ_j ~ Normal(μ, τ)                 θ_j ~ Normal(μ, sqrt(σ²))                θ_j ~ Normal(μ, 1/sqrt(ω))

NON-CENTERED  (flat geometry):  same POSITIVE slice (τ / σ² / ω) + η_j ~ Normal(0,1)  # REAL, standardized
    θ_j := μ + τ · η_j    OR    μ + sqrt(σ²) · η_j    OR    μ + η_j / sqrt(ω)    # DETERMINISTIC, NOT sampled
```
Materialize `θ_j` inside the natural-scale log-density; the POSITIVE slice (τ, σ²,
OR ω) supplies its own `log|J|` — never hand-write a Jacobian (`+ log τ`,
`+ log sqrt(σ²)`, …).

**Caveat — NCR is not unconditional (data informativeness).** CP and NCP are
mirror geometries of the SAME posterior (Betancourt & Girolami 2013). NCR helps
only when the per-group likelihood is WEAK (few obs/group, shrinkage / tiny-scale
priors — the common auto-gen case, so NCR is usually right). For STRONGLY-
informative per-group data (many obs, small noise) centered is better and blanket
NCP re-introduces an inverted funnel. Tell-tale: an already-NCP model that STILL
funnels (tree-depth saturation, low τ-ESS) needs that scale CENTERED, not "more
NCR". The general dominant form is partial non-centering `θ_j := μ + a·τ·η_j`,
`a ∈ [0,1]` (Papaspiliopoulos-Roberts-Sköld 2007) — `a` a deterministic constant,
no Jacobian.

Rules:
- The `joint_nuts_block` contains `(μ [REAL], scale-or-variance [POSITIVE], η_1..p [REAL])`.
  The POSITIVE slice is `τ` (sd form) OR `σ²` (variance form); `θ_j` is built with
  `τ·η_j` or `sqrt(σ²)·η_j` accordingly.
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
// VARIANCE form (nn_rbm / ARD): make the POSITIVE slice the variance and take
// its sqrt when building theta —
//     {"sigma2", 1, joint_constraint::POSITIVE},   // a VARIANCE, not an sd
//     ... theta_j = mu + sqrt(sigma2) * eta_j;      // sqrt() inside the log-density
// Keep sigma2's InvGamma prior; the POSITIVE slice still supplies log|J|.
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
Usually NOT a sampler bug. The **preferred / default** handling is at the
**R-hat / post-draw** level, not inside the sampler:
1. **Diagnose** on **label-invariant** quantities — R-hat of the sorted /
   order-statistic component parameters (or of post-hoc relabeled draws), NOT
   raw per-label R-hat. If the label-invariant R-hat converges, the sampler is
   fine and the raw per-label blow-up is the benign symmetry artifact.
2. **Resolve** POST-HOC — relabel the posterior draws (simple-sort /
   Stephens 2000 / Hungarian) in the diagnostics / analysis layer, per
   `label_switching.md`. Let the sampler explore all K! symmetric modes freely.

An in-sampler `ordered` / identifying constraint (`μ_1 < μ_2 < …`), monotone
reparameterization, or positivity-on-the-first-loading is a **NOT-RECOMMENDED
fallback — discouraged, not forbidden.** It CAN break the symmetry, but tends
to interact badly with slow-mixing discrete-allocation companions, is
error-prone, can hurt mixing, and may bias the natural-scale posterior
(`codegen_cpp.md` §205). Reach for it only when a model genuinely cannot be
resolved post-hoc (some models legitimately need it).

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
1. **Funnel NCR done** — if the prompt declares the Mode-1 pattern (SD form (A)
   OR variance form (B) `var ~ InvGamma; θ ~ Normal(0, sqrt(var))`), the cpp
   must contain the non-centered form (`eta_j` declared; `theta_j := mu +
   tau*eta_j` OR `theta_j := mu + sqrt(sigma2)*eta_j` deterministic), not the
   centered form and not Gibbs on the conjugate variance.
2. **Constraint kinds match** — each slice's `joint_constraint` matches the
   prompt's support (scale → POSITIVE, mean/effect → REAL).
3. **Lambda completeness** — every declared sub-parameter is read inside the
   joint log-density (a missing read = silent prior-only sampling).

A Mode-1 match with a centered cpp is a **FAIL_24(a)**: the fix is this skill's
NCR recipe.

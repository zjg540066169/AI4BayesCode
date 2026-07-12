---
name: AI4BayesCode-joint-nuts-failure
description: |
  Failure modes of joint_nuts_block (the default sampler for coupled
  continuous parameters) and how to fix them BEFORE shipping. Covers the
  three v1 modes -- (1) hierarchical funnel + the mandatory non-centered
  reparameterization (NCR) recipe, (2) multi-modal / label-switching,
  (3) high-dim joint slowness -- plus the escalation ladder. The funnel
  NCR recipe lives HERE (codegen_priors.md and validator Check #24 point
  here). Consult this skill BEFORE emitting a joint_nuts_block, and when a
  joint model shows pathology (max tree depth, divergences, R-hat large,
  a scale parameter stuck near zero, or random-effect ESS = NA).
---

# joint_nuts_block -- failure modes + fixes

`joint_nuts_block` samples all coupled continuous parameters jointly over a
single concatenated unconstrained vector. It is the **default** for coupled
continuous targets (block-decomposed per-parameter NUTS mixes ~10x slower per
ESS when parameters are correlated, and FREEZES outright on hierarchical
funnels -- see Mode 1 evidence). But joint NUTS has three well-characterized
failure modes. Each is detectable and fixable; this skill is the reference.

## When to consult

- **Codegen agent**: BEFORE emitting any `joint_nuts_block`, check the prompt
  against Mode 1 (funnel) -- NCR is **mandatory** when the pattern matches.
- **Validator**: target reference when Check #24 (joint-NUTS pre-flight) fails.
- **Debug**: lookup when a joint model shows pathology at runtime.

---

## Failure Mode 1 -- Hierarchical funnel (MOST COMMON; pre-codegen detectable)

### Symptom
- Tree depth hits max / frequent divergences.
- The scale parameter (tau, sigma) gets stuck near 0.
- Random-effect parameters report **ESS = NA** (a chain has ~zero within-chain
  variance -- it is FROZEN) and **R-hat >> 1.05**; coverage collapses.

### Why
Centered parameterization `theta_j ~ Normal(mu, tau)` couples theta to tau: when tau is small
the theta_j are squeezed into a tight neck, when tau is large they spread out. A
single global step size cannot navigate both the wide body and the tight neck,
so the sampler either diverges (step too big at the neck) or freezes (step too
small in the body). Per-block Gibbs is even worse: each block's step is tuned
to one slice of the funnel and is wrong for the rest.

### Detection (from the prompt, before codegen)
Match EITHER form -- the funnel geometry is identical whether the positive
hyperparameter is a standard deviation OR a variance:
```
(A) SD-parameterized:
    scale ~ HalfNormal / HalfCauchy / Exponential            (a POSITIVE sd)
    for j:  raw_j ~ Normal(<loc>, scale * <const>)           (sd used DIRECTLY)

(B) VARIANCE- or PRECISION-parameterized -- the Gaussian-hierarchical / ARD /
    Neal-1996 BNN standard (weight-variance priors; e.g. nn_rbm, ridge/ARD):
    var  ~ InvGamma / Scaled-Inv-chi^2   (POSITIVE variance)   ->  sd = sqrt(var)
    prec ~ Gamma                      (POSITIVE precision)  ->  sd = 1 / sqrt(prec)
    for j:  raw_j ~ Normal(<loc>, sd * <const>)             (sd = sqrt(var) OR 1/sqrt(prec))
```
`<loc>` may be 0, a global mean mu, or a linear predictor.

**Form (B) is at least as common as (A) and is the one that gets missed** -- the
positive slice is a VARIANCE and the sd entering the Normal is its `sqrt`, so the
literal word "scale" never appears. `InvGamma` / `Gamma`-on-precision almost
always signal form (B). Do NOT dismiss `var ~ InvGamma; theta ~ Normal(0, sqrt(var))`
as a non-funnel because it is variance-parameterized -- it IS Mode 1, and centered
+ joint NUTS (or Gibbs on the conjugate `var`) freezes on it exactly as in (A).

### Fix -- non-centered reparameterization (NCR); the DEFAULT for weak-data funnels
Centered + joint NUTS on a weak-data funnel diverges or maxes tree depth on
essentially every iteration, so NCR is the default -- but it is NOT unconditional
(see the caveat below).

```
CENTERED  (BAD on a weak-data funnel):
    tau ~ HalfNormal   # sd   -- OR --   sigma^2 ~ InvGamma   # variance   -- OR --   omega ~ Gamma   # precision
    theta_j ~ Normal(mu, tau)                 theta_j ~ Normal(mu, sqrt(sigma^2))                theta_j ~ Normal(mu, 1/sqrt(omega))

NON-CENTERED  (flat geometry):  same POSITIVE slice (tau / sigma^2 / omega) + eta_j ~ Normal(0,1)  # REAL, standardized
    theta_j := mu + tau * eta_j    OR    mu + sqrt(sigma^2) * eta_j    OR    mu + eta_j / sqrt(omega)    # DETERMINISTIC, NOT sampled
```
Materialize `theta_j` inside the natural-scale log-density; the POSITIVE slice (tau, sigma^2,
OR omega) supplies its own `log|J|` -- never hand-write a Jacobian (`+ log tau`,
`+ log sqrt(sigma^2)`, ...).

**Caveat -- NCR is not unconditional (data informativeness).** CP and NCP are
mirror geometries of the SAME posterior (Betancourt & Girolami 2013). NCR helps
only when the per-group likelihood is WEAK (few obs/group, shrinkage / tiny-scale
priors -- the common auto-gen case, so NCR is usually right). For STRONGLY-
informative per-group data (many obs, small noise) centered is better and blanket
NCP re-introduces an inverted funnel. Tell-tale: an already-NCP model that STILL
funnels (tree-depth saturation, low tau-ESS) needs that scale CENTERED, not "more
NCR". The general dominant form is partial non-centering `theta_j := mu + a*tau*eta_j`,
`a  in  [0,1]` (Papaspiliopoulos-Roberts-Skold 2007) -- `a` a deterministic constant,
no Jacobian.

Rules:
- The `joint_nuts_block` contains `(mu [REAL], scale-or-variance [POSITIVE], eta_1..p [REAL])`.
  The POSITIVE slice is `tau` (sd form) OR `sigma^2` (variance form); `theta_j` is built with
  `tau*eta_j` or `sqrt(sigma^2)*eta_j` accordingly.
- `theta_j` is **computed inside the natural-scale log-density** (to evaluate the
  likelihood) and, if a downstream block needs it, in a **deterministic
  refresher** reading mu/tau/eta -- it is NEVER a sampled sub-parameter.
- The natural-scale log-density writes the priors on mu, tau, eta plus the
  likelihood `y ~ f(mu + tau*eta)`. It writes **NO Jacobian term for tau** -- the
  POSITIVE slice adds `log|J|` automatically (`constraints::positive::wrap`;
  system_design.md Sec.10.1). Check #5 still forbids a hand-written `+ log(tau)`.
- NCR does NOT change tau's own prior -- keep the Jeffreys / half-normal choice
  per system_design.md Sec.11.6.

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
// its sqrt when building theta --
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
`eight_schools_centered` sampled per-block + centered FROZE: all 8 theta at
ESS = NA, R-hat ~= 2.23, coverage 0.10 -- while the reference (mcmclib, same
centered model) reached coverage 0.90. Lowering the step size (target_accept
0.8->0.95) gave a **bit-identical** failure: the chains were frozen, not
mis-stepped. A diagonal-mass (`adapt_mass`) workaround rescued *this* model
(cov 0.10->0.90) but REGRESSED tight time-series (arma11) and a harder funnel
(Mh) -- it is a per-model band-aid, not a fix. **The real fix is joint + NCR**
(this mode): it removes the funnel geometry at the source, model-independently.

### Form C -- STATE-SPACE / random-walk / Markov innovation funnel

Same Mode-1 funnel geometry, index `t` (time) instead of `j` (level). The shared
scale is an INNOVATION variance / SD; the "children" are `T` time-indexed states
linked by a Markov recurrence. Detection patterns (fire on ANY t-indexed Markov
prior, not just j-indexed exchangeable hierarchical):

    mu_t     ~ Normal(mu_{t-1},                        sigma_level)   (level RW)
    mu_t     ~ Normal(mu_{t-1} + gamma_t,              sigma_level)   (level + trend)
    gamma_t  ~ Normal(gamma_{t-1},                     sigma_trend)   (trend RW)
    s_t      ~ Normal(-sum_{h=1..P-1} s_{t-h},         sigma_seas)    (period-P sum-to-zero seasonal)
    x_t      ~ Normal(phi * x_{t-1},                   sigma)         (AR(1))

Variance-parameterized versions (`sigma^2 ~ InvGamma`) stack Form C with Form B --
treat as Form C for the reparam decision and Form B for the `sqrt()` in the
log-density.

**Same Mode-1 escalation applies -- Level 1 NCR is REQUIRED, not optional.**
Reparam the INNOVATION on the raw scale, then materialise the state
deterministically from the Markov recurrence:

    // level RW
    mu_raw_t   ~ Normal(0, 1)                  for t = 2..n            // sampled
    mu_t       = mu_{t-1} + sigma_level * mu_raw_t                     // deterministic

    // sum-to-zero seasonal (denote S_t = s_t + sum_{h=1..P-1} s_{t-h})
    S_raw_t    ~ Normal(0, 1)                  for t = P..n            // sampled
    S_t        = sigma_seas * S_raw_t
    s_t        = S_t - sum_{h=1..P-1} s_{t-h}                          // recurrence

### Why dense metric alone is NOT sufficient here

A centered state-space joint block has TWO stacked pathologies -- diagonal metric
fails on both, dense fails on the second:

1. **State-state RIDGE.** Adjacent states are near-perfectly correlated when the
   innovation SD is smaller than the observation SD -- the posterior lies on a
   near-linear ridge in `(state_1, ..., state_T)` space. Diagonal metric cannot
   rotate to the ridge direction; **dense metric CAN**.
2. **Sigma-state FUNNEL.** The shared innovation SD forms a Neal funnel with the
   T states (same geometry as Form A / B). Dense metric assumes LOCAL-Gaussian
   geometry; the funnel is intrinsically non-Gaussian, so **dense metric does NOT
   fix this**.

Reference implementations that use dense metric + centered states (e.g. the
standard "dense_e" recipe on a random-walk or seasonal DLM) still report many
divergences at long T or tight signal, precisely because the funnel remains.
NCR of every scale-governed state is the one fix that removes the funnel at the
source. **Best combined fix**: `cfg.use_dense_metric = true` AND NCR every
innovation-scale-governed state -- Check #18 dense escalation without NCR does
NOT clear Check #24(a).

**Check #24(a) FAIL rule**: a centered `state_t ~ Normal(prev_state_expr,
sigma^2)` under `joint_nuts_block` is a HARD FAIL regardless of the metric --
the codegen agent MUST rewrite as NCR before shipping, same as Form A / B.

---

## Mode-1 EXTENSIONS -- variants that still bite joint_nuts_block

All variants below reduce to Form A / B / C geometry (shared scale governs a
vector of Gaussian children) with the same NCR fix. Listed to close DETECTION
gaps. Patterns fully handled by a shipped specialized block (DP concentration
via `stick_breaking_block`; DP cluster atoms via `normal_gamma_cluster_gibbs_block`;
CAR/ICAR/BYM2/GP/1-D-GP via `gmrf_precision_block`/`gmrf_whitened_ess_block`/
`elliptical_slice_sampling_block`/`celerite_gp_block` -- see `codegen_cpp.md`
Sec.4a routing table; Dirac spike-slab via `examples/SpikeSlabRJMCMC.cpp` --
see `codegen_priors.md` Sec.3a Class 2b) are OUT of scope -- they never reach
`joint_nuts_block`.

**META-RULE.** When multiple scales exist at DIFFERENT LEVELS, NCR must be
applied at EVERY level -- common auto-gen bug is fixing only the deepest.
**Anti-rule:** do NOT NCR any level whose per-group data is strongly informative
(see base Mode-1 caveat above); centered wins there, partial NCR is the general form.

### Form-A extensions (same NCR as A, different detection signature)

- **A.1 IRT 2PL.** `theta_i ~ N(0, sigma_theta)` is Form-A over persons; anchor
  `sigma_theta := 1` (standard 2PL identification -- rotation-invariance,
  same trick as Form E) then NCR both hierarchies `log a_j ~ N(mu_a, sigma_a)`
  and `b_j ~ N(mu_b, sigma_b)`. Do NOT combine with `a_1 := 1` -- over-identifies.
  Ref: Curtis 2010 JSS 36 CS1; Fox 2010 "Bayesian Item Response Modeling" Ch.4.

- **A.2 Nested / stacked hierarchies** (radon-style).
  `alpha_j ~ N(gamma, sigma_alpha)` AND `gamma ~ N(mu_0, sigma_gamma)`.
  Two stacked Form-A funnels. Ref: Gelman-Hill 2007 Ch.12 (radon);
  Papaspiliopoulos-Roberts-Skold 2007 Stat.Sci. 22(1):59-73 (partial NCR).

- **A.3 Crossed random effects** (subject x item).
  `s_i ~ N(0, sigma_s)` AND `t_j ~ N(0, sigma_t)` -- two orthogonal Form-A
  funnels. Two-joint_nuts_block Gibbs recipe: `hierarchical_re.md` Sec.6.
  Ref: Sorensen-Hohenstein-Vasishth 2016 arXiv:1506.06201 ("Correlated varying
  intercepts, varying slopes" section / Listing 8).

- **A.4 Measurement error / latent covariates** (classical `W = X + u` OR
  Berkson `X = W + u`). `X_i ~ N(mu_x, tau_x^2)`, `u ~ N(0, tau_u^2)`.
  NCR: `X_i = mu_x + tau_x * z_i`, `z_i ~ N(0,1)`.
  Ref: Richardson-Gilks 1993 Am.J.Epi. 138(6):430-442; Carroll et al. 2006
  "Measurement Error in Nonlinear Models" Ch.9.

### Form-C extensions (same NCR as C, different detection or twist)

- **C.1 Stochastic volatility.** `y_t | h_t ~ N(0, exp(h_t/2))`,
  `h_t ~ N(mu + phi*(h_{t-1} - mu), sigma_eta)`. `exp(h/2)` observation
  SHARPENS the pinch. NCR the innovation.
  Ref: Kim-Shephard-Chib 1998 RES 65(3):361-393; Stan User's Guide "Time-series
  models" > "Stochastic volatility".

- **C.2 AR(1) stationary initial state.**
  `h_1 ~ N(mu, sigma / sqrt(1 - phi^2))` is a THIRD funnel over `(sigma, phi, h_1)`.
  Standard Form-C NCR on `t >= 2` innovations misses `h_1`; reparam it too:
  `h_1 = mu + sigma/sqrt(1-phi^2) * h_std_1`, `h_std_1 ~ N(0,1)`.
  Ref: Stan User's Guide "Time-series models" > "Autoregressive models"
  (stationary initialization).

- **C.3 Poisson state-space (temporal).**
  `y_t ~ Poisson(exp(lambda_t))`, `lambda_t ~ N(lambda_{t-1}, sigma_lambda)`.
  Form-C on log-rate; log-link sharpens the pinch (cf. C.1). Same NCR as C.
  **Route disambiguation**: `t`-indexed Markov recurrence -> apply here;
  spatial LGCP (neighborhood prior on `f_i`) -> `gmrf_whitened_ess_block`, NOT
  this fix.

- **C.4 RW2 / P-spline innovation.**
  `d_t := f_t - 2*f_{t-1} + f_{t-2} ~ N(0, sigma_f)`. NCR on the SECOND
  DIFFERENCE `d_t = sigma_f * d_std_t`, NOT on states `f_t` directly
  (sigma_f enters the intrinsic precision, not the states). When used as
  a pure GMRF smoother -> `gmrf_precision_block` (out of scope here).
  Ref: Lang-Brezger 2004 JCGS 13(1):183-212; Rue-Held 2005 Ch.3.4.

- **C.5 Parallel Form-C (BSTS local-linear-trend).**
  `mu_t ~ N(mu_{t-1} + delta_{t-1}, sigma_mu)`; `delta_t ~ N(delta_{t-1}, sigma_delta)`.
  Two innovation scales govern different innovation vectors -- NCR BOTH.

- **C.6 Panel state-space (Form A x Form C).**
  `mu_{j,t} ~ N(mu_{j,t-1}, sigma_j)` AND `sigma_j ~ HalfNormal(0, tau)`.

### Form D -- Multivariate hierarchical (LKJ / Wishart)

`b_g ~ MVN(0, Sigma)` -- each `tau_k` stacks a Form-A funnel against `b_{g,k}`
across G. NCR (Cholesky): `z_g ~ N(0, I_K)`; `b_g = diag(tau) * L * z_g`.
Full config: `codegen_cpp.md Sec.4a` "Multivariate hierarchical" row.

- **Detection.** `MVN(0, Sigma)` with `Sigma = diag(tau) L L^T diag(tau)`
  (LKJ + scale) -> Cholesky NCR above. `Sigma ~ Inv-Wishart(nu, S_0)` ->
  decompose to `L * L^T` via Bartlett and NCR the diagonal scales; do NOT
  sample raw `Sigma` inside `joint_nuts_block` (see Barnard-McCulloch-Meng 2000
  for LKJ preference).

- **D.1 Multivariate DLM / MSV.** Form D per time step: `eta_t ~ MVN(0, Sigma)`,
  `t = 1..T`. `eta_t = diag(tau) * L * z_t`, `z_t ~ MVN(0, I_K)`. Never
  instantiate Sigma.

- **D.2 NLME / population PK.** Form D + nonlinear likelihood (ODE, algebraic).
  Same D fix; nonlinearity sharpens the pinch as in C.1.
  Ref: Wakefield 1996 JASA 91(433):62-75; Margossian-Zhang-Gillespie 2022
  CPT:PSP 11(9):1151-1169 (Stan+Torsten).

Refs: BDA3 Ch.15; Barnard-McCulloch-Meng 2000 Statistica Sinica 10:1281-1311;
LKJ 2009 J.Multivar.Anal. 100(9):1989-2001.

### Form E -- Factor / ARD / CFA loading-scale funnel

`y_i = W * z_i + eps`, `z_i ~ N(0, I_K)`, `W_{d,k} ~ N(0, tau_k^2)` with
`tau_k ~ HalfCauchy` (ARD column scale). K stacked column funnels.

**Disambiguation vs Form G**: Form E is coefficients W in a BILINEAR term
(`y = W z + eps`, W multiplies a LATENT factor) -- rotation-invariant.
Form G is coefficients beta in a LINEAR predictor (`y ~ X beta`, beta multiplies
OBSERVED covariates) -- no rotation invariance.

**NCR.** `omega_{d,k} ~ N(0,1)`; `W_{d,k} = tau_k * omega_{d,k}`.
Rotation invariance `(W, z) <-> (W A, A^{-1} z)` is a SEPARATE Mode-2-family
issue -- fix by lower-triangular `W` with positive diagonal (or for CFA,
anchor `psi := 1`), only if the user model doesn't already impose it.

Refs: Bishop 1999 NIPS 11:382-388 (Bayesian PCA); Ghosh-Dunson 2009
JCGS 18(2):306-320; Piironen-Vehtari 2017 EJS 11(2):5018-5051 Sec.3 (ARD).

### Form F -- HSGP reduced-rank GP

CAR/ICAR/BYM2/GP/1-D-GP are routed to specialized blocks -- see the section
preamble. HSGP is the one latent-Gaussian family that legitimately reaches
`joint_nuts_block` (no shipped specialized block).

Spectral basis expansion:
`f(x) = sum_{j=1..M} sqrt(S_j(sigma_f, ell)) * beta_j * phi_j(x)`.
**Detection cue**: Laplacian-Dirichlet eigenfunction basis on `[-L, L]`,
`M`-truncation, `beta_j ~ N(0,1)` on spectral coefficients.

**NCR pre-baked**: `beta_j ~ N(0, 1)`; amplitude and lengthscale enter
MULTIPLICATIVELY at likelihood evaluation via `sqrt(S_j(sigma_f, ell))`, not
as variance of `beta`.

**Config**: `sub_params = {(sigma_f, 1, POSITIVE), (ell, 1, POSITIVE), (beta, M, REAL)}`;
`use_dense_metric = true` (amplitude/lengthscale banana ridge).

Ref: Riutort-Mayol et al. 2023 Stat.Comput. 33(1) Sec.3.3;
Solin-Sarkka 2020 Stat.Comput. 30(2).

### Form G -- Global-local shrinkage / horseshoe family

`beta_j ~ N(0, tau^2 * lambda_j^2)`, `tau ~ HalfCauchy(0, tau_0)` (global),
`lambda_j ~ HalfCauchy(0, 1)` (local). TWO nested Form-A funnels: `(tau, beta)`
globally and `(lambda_j, beta_j)` locally. Half-Cauchy tails give common neck
at the origin -- reported to freeze NUTS under any step size when centered.

**NCR.** `z_j ~ N(0, 1)`; `beta_j := tau * lambda_j * z_j`. Priors unchanged.

**Family generalizes** (same `z_j`; substitute the family-specific product):
Regularized/Finnish HS (`sd = tau * lambda_tilde_j`), HS+ (`sd = tau * lambda_j`,
NCR every scale-parent recursively), R2D2 (`sd = sigma * sqrt(W * phi_j)`),
Dirichlet-Laplace (`sd = tau * phi_j * sqrt(psi_j)`), Bayesian LASSO
(`sd = sigma * tau_j`).

- **G.1 Normal-Gamma prior** (PARTIAL NCR).
  Detection: `psi_j ~ Gamma(lambda, gamma^2/2)`, `beta_j ~ N(0, psi_j)`,
  `gamma^2 ~ Gamma(...)` (Griffin-Brown 2010 Bayesian Anal. 5(1):171-188).
  Non-center Normal ONLY: `beta_j = sqrt(psi_j) * z_j`. The `(gamma^2, psi_j)`
  Gamma-Gamma parent STAYS CENTERED (non-centering destabilises).

- **G.2 Continuous spike-and-slab.**
  `gamma_j` continuous (relaxed inclusion): standard Form-G NCR
  `beta_j = tau * sqrt(gamma_j + (1-gamma_j)*v0) * z_j`.
  Dirac `gamma_j` -> out of scope (see preamble; use `SpikeSlabRJMCMC.cpp`).

Refs: Carvalho-Polson-Scott 2010 Biometrika 97(2):465-480; Piironen-Vehtari
2017 EJS 11(2):5018-5051 Sec.3 + App.C (Stan code, family compendium);
Bhadra et al. 2017 Bayesian Anal. 12(4):1105-1131 (HS+); Zhang et al. 2022
JASA 117(538):862-874 (R2D2); Bhattacharya et al. 2015 JASA 110(512):1479-1490
(DL); Ishwaran-Rao 2005 AoS 33(2):730-773 (spike-slab).

---

## Failure Mode 2 -- Multi-modal / label-switching (partially detectable)

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
1. **Diagnose** on **label-invariant** quantities -- R-hat of the sorted /
   order-statistic component parameters (or of post-hoc relabeled draws), NOT
   raw per-label R-hat. If the label-invariant R-hat converges, the sampler is
   fine and the raw per-label blow-up is the benign symmetry artifact.
2. **Resolve** POST-HOC -- relabel the posterior draws (simple-sort /
   Stephens 2000 / Hungarian) in the diagnostics / analysis layer, per
   `label_switching.md`. Let the sampler explore all K! symmetric modes freely.

An in-sampler `ordered` / identifying constraint (`mu_1 < mu_2 < ...`), monotone
reparameterization, or positivity-on-the-first-loading is a **NOT-RECOMMENDED
fallback -- discouraged, not forbidden.** It CAN break the symmetry, but tends
to interact badly with slow-mixing discrete-allocation companions, is
error-prone, can hurt mixing, and may bias the natural-scale posterior
(`codegen_cpp.md` Sec.205). Reach for it only when a model genuinely cannot be
resolved post-hoc (some models legitimately need it).

### Empirical evidence (this repo)
`hmm_drive_0`: phi[1] and phi[2] are exchangeable transition rows -> R-hat ~= 2.23
from label switching, NOT a sampler bug. Diagnosed via per-parameter R-hat:
the symmetric pair is high, the rest fine.

---

## Failure Mode 3 -- High-dimensional joint, slow (detectable from param count)

### Symptom
Wall-time per iteration is very large; R-hat is OK.

### Detection
Param count above ~50-100 AND the likelihood **factorizes** (conditional
independence across a grouping index).

### Fix
Split the single joint group into smaller joint groups along the conditional-
independence structure (each group < ~50 params). Automated "obvious
independence" auto-split is a v1.2 codegen feature (F4); until then, split
manually when the factorization is clear, or accept the cost. Do NOT split
across a genuine coupling -- that silently biases inference.

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
nuts_block) is always available as a correctness fallback -- it is slower but
never wrong-target -- EXCEPT on funnels, where per-param NUTS itself freezes
(Mode 1); there, Level 1 (NCR) is required, not optional.

---

## Relationship to validator Check #24

Check #24 (joint-NUTS pathology pre-flight) is the static guard. It verifies,
when a `joint_nuts_block` is present:
1. **Funnel NCR done** -- if the prompt declares the Mode-1 pattern (SD form (A)
   OR variance form (B) `var ~ InvGamma; theta ~ Normal(0, sqrt(var))`), the cpp
   must contain the non-centered form (`eta_j` declared; `theta_j := mu +
   tau*eta_j` OR `theta_j := mu + sqrt(sigma2)*eta_j` deterministic), not the
   centered form and not Gibbs on the conjugate variance.
2. **Constraint kinds match** -- each slice's `joint_constraint` matches the
   prompt's support (scale -> POSITIVE, mean/effect -> REAL).
3. **Lambda completeness** -- every declared sub-parameter is read inside the
   joint log-density (a missing read = silent prior-only sampling).

A Mode-1 match with a centered cpp is a **FAIL_24(a)**: the fix is this skill's
NCR recipe.

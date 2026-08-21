---
name: AI4BayesCode-joint-nuts-failure
description: |
  Failure modes of joint_nuts_block (the default sampler for coupled
  continuous parameters) and how to fix them BEFORE shipping. LEADS with a
  general funnel/ridge signature (S1 scale x latent; S2 correlated-latent
  ridge) + a universal fix ladder (marginalize / NCR / QR), so a model matching
  the PATTERN but no named Form is still covered. The named cases -- (1)
  hierarchical funnel + NCR, (2) multi-modal / label-switching, (3) high-dim
  slowness, plus Forms A-H -- are WORKED EXAMPLES. The escalation ladder and the
  funnel NCR recipe live HERE (codegen_priors.md and validator Check #24 point
  here). READ MODES 1-4 BEFORE EMITTING ANY joint_nuts_block -- they are how
  the block's membership is decided, and Mode 4 leaves no runtime signal to
  come back on. Also consult when a joint model shows pathology (max tree
  depth, divergences, R-hat large, a scale parameter stuck near zero, or
  random-effect ESS = NA).
---

# joint_nuts_block -- failure modes + fixes

`joint_nuts_block` samples all coupled continuous parameters jointly over a
single concatenated unconstrained vector. It is the **default** for coupled
continuous targets (block-decomposed per-parameter NUTS mixes ~10x slower per
ESS when parameters are correlated, and FREEZES outright on hierarchical
funnels -- see Mode 1 evidence). Being the default is not a licence to put
everything in one block: four failure modes are characterized below, and
Modes 1-4 are read BEFORE emitting the block, to decide what it contains.

## When to consult

**Codegen agent -- READ MODES 1-4 BEFORE EMITTING ANY `joint_nuts_block`.**
Not "consult when something looks wrong": the four modes are how you DECIDE
the block, so they are read while you are choosing its membership. Mode 4 has
no runtime signal to come back on, so there is no second chance.

The membership decision has two halves, and both can apply to one block --
scales pair with their raw effects INSIDE, the residual scale stays OUTSIDE:

1. **What goes IN** -- Mode 1 (funnel). NCR is **mandatory** on a match.
2. **What stays OUT** -- Mode 4. An uncoupled parameter costs the block a
   shared step size and buys nothing.

Modes 2 (multi-modal) and 3 (high-dim) shape the same decision: if the target
is label-switching or the block is about to exceed the dimension where one
metric can serve, that is decided now, not after a bad run.

**Debug**: lookup when a joint model shows pathology at runtime -- max tree
depth, divergences, large R-hat, a scale stuck near zero, random-effect
ESS = NA. Modes 1-3 leave these traces. Mode 4 does not.

---

## The general signature -- match THIS, not just a named Form

The Forms/Modes below are WORKED EXAMPLES, not an exhaustive catalogue. If a
model's funnel/ridge matches no named Form, do NOT conclude "unfixable" or
"multimodal" -- match it to one of two structural signatures (they cover the
great majority of joint_nuts_block pathologies) and apply the fix ladder.

**(S1) Scale x latent funnel.** A sampled scale-like `s` (variance / sd /
precision / lengthscale / amplitude / spectral density / concentration) sets the
scale of a VECTOR of latents `z` (z's prior sd is a function of s, OR z is
multiplied by a function of s before the likelihood); `(s, z)` funnels as
`s -> 0`. (Forms A, B, C, D, E, F, G.)

**(S2) Correlated-latent ridge.** Latents have a highly correlated posterior -- a
ridge where different coefficient vectors fit equally well -- from either (a) a
collinear design/basis matrix `X` (spline / RBF / poly -- Form H), or (b) a
prior-induced correlation (random-walk / AR states, no `X` -- Form C). S1 and S2
co-occur in heteroscedastic latent-function (F/H) and state-space (C) models.

**Fix ladder (by SIGNATURE, not Form name):**
1. **MARGINALIZE** a conjugate Gaussian-linear latent (Gaussian latent + Gaussian
   obs through a linear map, even heteroscedastic KNOWN noise) in closed form --
   kills S1 AND S2 at once; try first. (A non-Gaussian GLM latent is NOT
   closed-form marginalizable even with a linear predictor -- use NCR / QR.)
2. **NCR** (S1): factor the scale out, `z = g(s) * z_raw`, `z_raw ~ N(0,1)`,
   priors unchanged; substitute the family-specific product (see the Forms).
3. **DECORRELATE** (S2): QR the design (`X = Q R`; sample in `Q`, map back by
   `R^{-1}`); for a prior-induced state ridge (no `X`), a dense metric is the
   in-sampler analogue (see Form C).
4. **Data-strength**: NCR under weak data; the centered form can win under strong
   data (the per-group likelihood dominates the prior).

Only AFTER this ladder, and only with `validator.md`'s R2 evidence (over-
dispersed chains in SEPARATED modes with a gap, from a KNOWN symmetry), is
"genuine multimodality" (Mode 2) acceptable -- a missing Form is not evidence.
Genuinely NONLINEAR geometries (banana, dynamical-system couplings) are neither
S1 nor S2: reparameterize toward near-Gaussian coordinates or escalate (ladder
at the end). These are less common in this library than S1/S2.

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
      -- OR a BOUNDED sd: scale ~ Uniform(0, U) / any INTERVAL(lo, U) prior
         (the classic multilevel default). The funnel geometry is IDENTICAL:
         the scale slice is INTERVAL not POSITIVE, but raw_j ~ Normal(loc,
         scale) still forms the neck. Do NOT dismiss a bounded/Uniform scale
         prior as a non-funnel because the slice is INTERVAL -- it IS Mode 1,
         and the NCR fix + funnel-mixing risk apply exactly as in the
         POSITIVE-sd case.
    for j:  raw_j ~ Normal(<loc>, scale * <const>)           (sd used DIRECTLY)

(B) VARIANCE- or PRECISION-parameterized -- the Gaussian-hierarchical / ARD /
    Neal-1996 BNN standard (weight-variance priors; BNN / ridge / ARD):
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
- The `joint_nuts_block` contains `(mu [REAL], scale-or-variance [POSITIVE or INTERVAL], eta_1..p [REAL])`.
  The scale slice is `tau` (sd form) OR `sigma^2` (variance form); constraint is POSITIVE
  for an unbounded scale, or INTERVAL(lo, U) for a bounded/Uniform-prior scale. `theta_j`
  is built with `tau*eta_j` or `sqrt(sigma^2)*eta_j` accordingly; the slice's own
  constraint (`positive::wrap` OR `interval::wrap`) supplies `log|J|` automatically.
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
// VARIANCE form (weight-variance BNN / ARD): make the POSITIVE slice the variance and take
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
A centered scale-governed hierarchical model sampled per-block + centered
FROZE: all group-level effects at ESS = NA, R-hat >> 1, coverage collapsed
-- while the reference (mcmclib, same
centered model) reached coverage 0.90. Lowering the step size (target_accept
0.8->0.95) gave a **bit-identical** failure: the chains were frozen, not
mis-stepped. A diagonal-mass (`adapt_mass`) workaround rescued *this* model
(cov 0.10->0.90) but REGRESSED a tight AR/ARMA time-series model and a harder funnel
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

**Heteroscedastic / coupled GP (BOTH mean AND log-sigma are HSGPs): NCP +
dense metric is often NOT enough.** Two coupled spectral paths, plus a
persistent `(sdgp <-> beta)` funnel and inter-`beta` ridge when the lengthscale
is small (wiggly `f`), can leave R-hat > 1.01 on the `beta` vector after every
metric / target / warmup tweak. That is Mode-1-like (reparameterize /
marginalize), NOT Mode 2 -- do NOT report it as "multimodal". Escalate:
1. **MARGINALIZE the Gaussian-linear mean-GP coefficients `beta`** analytically:
   conditional on the sigma path, `sum sqrt(S_j) beta_j phi_j` is Gaussian-linear
   even under a heteroscedastic KNOWN noise covariance, so `beta` integrates out
   in closed form -- removing the funnel + ridge entirely and leaving only the
   hyperparameters + the sigma path to sample.
2. the **GP convergence troubleshooting ladder** in `block_catalogue/index.md`
   (Cholesky-AD gradient, marginal GP, celerite).
Also sanity-check the lengthscale prior against the data x-scale -- a mismatched
too-small lengthscale forces pathological wiggliness that masquerades as
multimodality.

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

### Form H -- Penalized-spline / linear-basis coefficient ridge

Detection cue: a spline / B-spline / RBF / polynomial design matrix `X` with
coefficients `s ~ N(0, tau^2 I)` (smoothing sd `tau`) and mean = `X s`. The
`(tau, s)` pair is a Form-A funnel AND the columns of `X` are near-collinear, so
`s` rides a highly correlated ridge -- two chains fit the SAME `mu = X s` with
different `s`, giving persistent R-hat on `s` while `tau` and everything else
converge.

NCR fixes the `(tau, s)` funnel (`s = tau * z`, `z ~ N(0,1)`); it does NOT fix
the inter-coefficient ridge. For that:
1. **QR-decorrelate the design** (`X = Q R`; sample coefficients in the `Q`
   basis, map back by `R^{-1}`) -- the standard Stan/brms fix; makes the coefficient
   posterior near-isotropic.
2. or **MARGINALIZE the Gaussian-linear `s`** analytically (as in Form F;
   `X s` is Gaussian-linear even under a heteroscedastic known noise
   covariance) -- integrates the ridge out.

A persistent R-hat on a linear-basis coefficient vector is Mode-1-like
(reparameterize / marginalize), NOT Mode 2 -- there is no permutation or sign
symmetry, so do NOT diagnose it as label-switching / "multimodal".

---

## Failure Mode 2 -- Multi-modal / label-switching (partially detectable)

### Symptom
Chains split into different modes; R-hat is huge on the symmetric parameters,
yet each chain looks well-mixed internally; ESS may look fine per-chain.

### Detection
- Mixture components (label switching across the K components).
- Factor / loading models (sign flips).
- Any exact permutation/sign symmetry in the prompt's parameterization.
- **NOT this mode**: a GP / spline / linear-basis coefficient ridge (Forms
  F/H) has NO permutation or sign symmetry -- do NOT diagnose it as
  label-switching / "multimodal"; route to Forms F/H (reparameterize /
  marginalize). Genuine multimodality needs the evidence in validator.md's
  R2 note (separated modes with a gap, from a known symmetry).

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
(`codegen_cpp.md` Sec.4a). Reach for it only when a model genuinely cannot be
resolved post-hoc (some models legitimately need it).

### Empirical evidence (this repo)
An HMM with exchangeable transition rows: phi[1] and phi[2] give R-hat >> 1
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

## Failure Mode 4 -- An UNCOUPLED parameter inside the joint block

**Decide this BEFORE codegen, from the model. Runtime will not tell you.** The
penalty is proportional to the spread of the data, so a well-scaled validation
dataset shows nothing: measured below, at gap 1 the ESS cost is zero. A model
carrying this defect passes L3 with high ESS and clean R-hat, and only starves
on data whose scale is large. There is no runtime gate to fall back on.

### The question to ask
Of every parameter you are about to put in one joint block: **is it coupled to
its blockmates in the POSTERIOR?**

- **In** -- a non-centered `(sigma_k, z_k)` pair, where the scale multiplies
  the raw effect. That is Mode 1 and the block is the fix.
- **Out** -- an observation-level residual scale beside locations it does not
  multiply. In `y ~ N(alpha + X beta, sigma^2)`, `Cov(beta_j, sigma) = 0`
  analytically, so membership buys nothing. Give it its own `nuts_block`.

Both can apply to one sampler. `codegen_cpp.md` Sec.4a's table is the short
form: joint(alpha, beta), **sigma separate**.

### Why membership costs anything
One step size serves the whole block, and it settles near what the scale wants
-- far too large for the locations. The gap grows without bound because a
POSITIVE slice is sampled as `log(sigma)`, whose posterior sd is ~`1/sqrt(2n)`
at ANY magnitude, while an unconstrained location's sd grows with the data.

### Empirical evidence (synthetic, one variable changed)
`y = X beta + eps`, N=80, p=7, OLS start, 2000 warmup + 4000 draws, same data
and seed per arm; "gap" multiplies the response.

| gap | sigma in / diag | sigma in / dense | sigma OUT |
|---|---|---|---|
| 1 | beta ESS 1635 | -- | 1814 |
| 1e3 | 786 | 791 | 1575 |
| 1e6 | **92** | **17** | 1811 |

At gap 1e6 the joint step size is 1.647 while a separate beta block wants 0.244
and a separate sigma block 1.965 -- ~7x too large for beta, which is why
sigma's own ESS inside stays 2215 while beta's is 92. **A dense metric makes it
worse** (92 -> 17): the problem is not correlation, and dense has more to
estimate from the same warmup. Do not escalate here.

### If it already shipped
Symptom on wide-scale data: R-hat and coverage FINE (the posterior is right),
ESS collapsed on some slices while the offending parameter's own ESS is
healthy. Confirm by moving it to its own block and comparing adapted step
sizes; if they differ a lot, the joint block was serving the wrong coordinate.

---

## Escalation ladder

```
Level 0: MARGINALIZE a conjugate Gaussian-linear latent in closed form --
         removes the scale x latent funnel (S1) AND the correlated-latent
         ridge (S2) at once
Level 1: NCR (funnel S1)  OR  QR-decorrelate / dense metric (ridge S2)  OR
         reparameterize (constraint boundary)
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

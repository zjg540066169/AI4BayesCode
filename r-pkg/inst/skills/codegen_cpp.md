---
name: AI4BayesCode-codegen-cpp
description: |
  C++ file emission for AI4BayesCode samplers -- modular vs joint NUTS
  coupling analysis, GPL-3.0+ boilerplate header, natural-scale
  log-density lambda template with constraints::wrap pattern,
  posterior-predictive y_rep stochastic refresher (per-observation-
  family templates: Gaussian / Bernoulli / Poisson / Multinomial /
  BART-noise), Check #12 gradient verification via autodiff (with
  throwaway verify_<ClassName>.cpp companion file), Assembly order
  inside constructor, Class shape (6-method contract + mutable
  predict_rng_). Extracted from codegen.md (sections 4a, 5, 6, 6a,
  6b, 7, 8). The entry-point skill `codegen.md` points here for all
  C++-emission guidance.
---

# AI4BayesCode codegen -- C++ file emission

Companion skill to `codegen.md`. Load this when writing the generated
`.cpp` file: boilerplate, log-density lambdas, posterior-predictive
layer, Check #12 gradient verification, assembly order, class shape.

For prior elicitation and block selection, see `codegen_priors.md`.
For the R runner template, see `codegen_r_runner.md`.

---

## 4a. Coupling analysis: modular vs joint NUTS

The framework supports both `nuts_block` (one parameter, one NUTS
trajectory per Gibbs sweep) and `joint_nuts_block` (several named
sub-parameters, one joint NUTS trajectory per sweep). **`joint_nuts_block`
is the DEFAULT** for continuous parameters not owned by a specialized
block: you write ONE joint natural-scale log-density (likelihood + all
priors + per-slice constraints) and the block runs NUTS over the
concatenated unconstrained vector. Modular `nuts_block` is a LOW-priority
fallback -- for a genuinely scalar parameter, a post-NCR funnel branch, or
deliberate isolation. Coupled continuous parameters mix ~10x slower per
ESS when split and FREEZE outright on funnels, so joint is the default,
not an optimization. (Specialized / conjugate-Gibbs blocks still take
precedence WHEN THEY GENUINELY APPLY -- for EFFICIENCY, not correctness:
an AI-written joint log-density is validator-checkable term by term, so
correctness no longer requires picking from the built-in menu. Never
force a poorly-matching block; if none fits AND NUTS is structurally
inapplicable, see `codegen_priors.md Sec.2c` Exception 4.)

### Coupling heuristics (apply during code generation)

Work through each pair / group of continuous parameters in the
likelihood and classify:

| Pattern in likelihood | Coupling | Default |
|---|---|---|
| `y ~ f(theta_i - b_j)`, `y ~ f(alpha - beta)` -- shift invariance | **High** | `joint_nuts_block(theta, b)` |
| `y ~ N(alpha + X beta, sigma^2)` -- additive linear mean | **High** | `joint_nuts_block(alpha, beta)`, sigma separate |
| `y ~ N(X beta + Z u, sigma^2)`, `u ~ N(0, tau^2)` -- fixed + random effect sharing mean | **High** | `joint_nuts_block(alpha, beta, u)`, sigma + tau separate |
| Correlation between continuous params visible in the generative model | **High** | joint |
| Hierarchical hyperparameter over vector of children: `u_i ~ N(mu_u, tau^2)` -- hyper and children both continuous | **High** (REQUIRED) | **MANDATORY: read `skills/hierarchical_re.md` in full BEFORE generating code.** Single `joint_nuts_block` over `(mu_u, tau, [sigma_y if Gaussian likelihood], u_raw, beta)` with **non-centered (NC) reparameterization** `u_i = mu_u + tau * z_i, z_i ~ N(0, 1)`. NEVER split these across separate `nuts_block`s -- the funnel between `(tau, u_i)` returns immediately, even with NC, because separated Gibbs steps re-create the inverse funnel (Betancourt-Girolami 2015 Sec.3.3). Empirical failure mode: cov_AI ~= 0.07-0.75 vs cov_REF ~= 0.94 across radon_*_centered, radon_*_noncentered, surgical_model -- all four caused by separate-block composition, not by the parameterization label. The rule applies regardless of G (no "small enough G" exception). Set `cfg.use_dense_metric = true` as a Check #18 escalation when 2-chain rhat > 1.05 at 20k+20k on the joint posterior -- START DIAGONAL, measure, then escalate; do NOT gate dense on a dimension threshold (large G makes dense *likely*, not mandatory -- confirm via R-hat). For CROSSED effects (e.g. pilots-style `a_g + b_s` shared `sigma_y`): TWO joint_nuts_block instances, one per RE level, sharing `sigma_y` via shared_data; see `hierarchical_re.md Sec.6`. |
| **Multivariate hierarchical** with d-dim per-unit random effects: `u_i ~ MultiNormal(0, Sigma)` where `Sigma` has d scale parameters `sigma_1..sigma_d` and possibly correlations `rho_{jk}` (e.g., bivariate normal latent species effect, random intercept + slope, multi-output GLMM) | **High** | **`joint_nuts_block` over `(sigma_1..sigma_d` POSITIVE`, [rho_jk` REAL via z-transform`], u_raw[G*d]` REAL`)`** -- the d scales, the d*(d-1)/2 correlations, and the G*d non-centered raw effects are all multiplicatively / Cholesky-coupled and CANNOT be separated into independent NUTS blocks without catastrophic cross-impl bias (same bug class as univariate hierarchical-LM: separated POSITIVE-scale + REAL-vec joint NUTS gives wrong cross-impl posteriors even when single-chain rhat looks fine). at large `G*d` dense metric is often needed, but **start diagonal and escalate to `cfg.use_dense_metric = true` as a Check #18 step only when R2/R3 diagnostics show diagonal is inadequate** (measure, don't predict -- no fixed dimension threshold). For `Sigma` parameterization: the simplest correct path is `Sigma = diag(sigma) * L_corr * L_corr' * diag(sigma)` with `L_corr` from `cholesky_corr` constraint sampled in its own block, leaving `(sigma, u_raw)` jointly POSITIVE+REAL in the joint block. For d=2 with a single correlation `rho`, a reparameterization `rho = 2*sigmoid(z_rho) - 1` keeps `z_rho` in the joint block as REAL. |
| BART tree ensemble + residual sigma (BART has its own kernel) | Low | `bart_block` + `nuts_block(sigma)` (specialized, NOT joint) |
| Polya-Gamma augmented logistic (conjugate after augmentation) | Low | **`pg_logistic_block`** -- 10-100x faster than NUTS-on-logistic for p < 1000. **LINEAR LOGISTIC ONLY -- NOT for logistic BART** (use `genbart_block + logistic_lik` instead; PG breaks BART's tree-scale identifiability). |
| Poisson / NB / heteroscedastic / AFT / beta / gamma_shape BART | Medium | **`genbart_block` with the matching `genbart::lik::*`** (Linero 2022 RJMCMC with Laplace leaf proposals; the single primitive handles all 10 shipped likelihoods + user-supplied). WARNING **genbart's RJMCMC is much slower than `bart_block` -- this row is for genuinely NON-Gaussian responses ONLY. A plain Gaussian mean is `bart_block` (NOT `genbart_block + normal_lik`); an additive / varying-coefficient Gaussian BART (VC-BART) is backfitting `bart_block`s + `weights_key`, see the `bart_block` card -- NOT a custom genbart likelihood.** Reference examples shipped: `examples/GBartPoisson.cpp` (count regression), `GBartHeteroscedastic.cpp` (mean = variance = exp(r)). For NB / AFT / beta / gamma_shape / beta_binomial, the agent should compose from `GBartPoisson.cpp` + the corresponding `genbart::lik::*` likelihood header in `bart_pure_cpp/src/GENBART/likelihoods/`. |
| Bernoulli BART (binary classification) | Medium | **`genbart_block + logistic_lik`** directly (no augmentation) -- see `examples/GBartLogistic.cpp`. Do NOT use `pg_logistic_block + bart_block` -- PG's kappa = y - 0.5 pseudo-response breaks BART's Gaussian-observation tree-scale identifiability. |
| Multinomial logistic BART (C >= 2 classes) | Medium | **C-1 x `genbart_block(poisson_lik, offset_key="log_phi_aug")` + one `poisson_multinomial_aug_block`** via the Baker 1994 / Forster 2010 gamma trick in the Murray 2021 Sec.3.1 reference-category identified architecture. See `examples/GBartMultinomial.cpp`. For binary (C=2) prefer the simpler `GBartLogistic` direct path. |
| Dirichlet-Categorical conjugate | Low | `dirichlet_gibbs_block` + `categorical_gibbs_block` |
| GP regression with **Gaussian** observation likelihood | (no latent f sampled) | **Marginal-likelihood architecture**: integrate f out, sample only `(amplitude, lengthscale, sigma)` from `y ~ N(0, K + sigma^2 I)` via one `joint_nuts_block` (3 POSITIVE slices). Analytic gradient via Rasmussen & Williams Sec.5.5 Eq. (5.9) `0.5 tr((alpha alpha' - K^-^1) dK/dtheta)`. NO ESS, NO latent f or z. See `examples/GPRegression.cpp`. Matches Stan / libgp / GaussianProcesses. |
| Latent **dense GP** + **non-Gaussian** likelihood (GP classification, GP regression with Poisson / Student-t / NB observations) | N/A for the latent f | **Whitened ESS via z**: sample `z ~ N(0, I)` via `elliptical_slice_sampling_block` with `L_chol_key = "L_identity"`; recover `f = L(amp, ell) * z` inside the block's `log_lik`. Hyperparameter blocks include the non-Gaussian likelihood at proposed `(amp, ell)` in their log-density. See `examples/GPClassification.cpp`. |
| Latent **sparse GMRF** (Q sparse PSD: CAR / ICAR / RW1 / RW2 / 2D lattice GMRF) + **non-Gaussian** likelihood (Poisson / Bernoulli / Student-t / NB / log-Gaussian Cox) | N/A for the latent x | **`gmrf_whitened_ess_block`** (Murray 2010 ESS on the IMPLICIT GMRF prior; Rue 2001 sparse-Cholesky permuted backsolve for prior draws). User supplies `Q_fn(ctx) -> arma::sp_mat` and `log_lik(x, ctx) -> double`. Strictly more efficient than the dense `elliptical_slice_sampling_block` path when Q is sparse (e.g. 4-NN / 6-NN lattice -- Q has O(n) non-zeros, dense Sigma has n^2 non-zeros). Sum-to-zero preserved exactly by ESS rotation linearity. Compose with `nuts_block` for hyperparameters (linear-predictor intercept on the real line, log-precision on the positive line, ...). See `block_catalogue/index.md` `gmrf_whitened_ess_block` section for the example recipe and verified convergence budgets at N=16 / N=64. |
| GP hyperparameters under the **whitened ESS** path -- known `(amp, ell)` **banana ridge** | **High** for `(log_amp, log_ell)` | Default: one `joint_nuts_block({amp, ell})` (POSITIVE x 2), reading the latent `z` and computing the likelihood at proposed `(amp, ell)` via `f = L(amp, ell) * z`. Modular per-slice NUTS slow-mixes along the banana ridge (5-10x ESS loss on `amp` at long chains) and is NOT recommended even as a starting point. If `ell` ESS is still inadequate at the extended budget, escalate to a reverse-mode Cholesky-AD analytic gradient inside the joint log-density. See `block_catalogue/index.md` "GP convergence troubleshooting ladder". |
| 1-D time-series GP | N/A | `celerite_gp_block` + `univariate_slice_sampling_block` on hyperparameters |
| **Reduced-rank GP (HSGP / Hilbert-space GP)** -- `f(x) = sum_m sqrt(spd(ell)_m) * z_m * phi_m(x)` | **High** -- `(amp, ell, z[M])` is a 3-way funnel + (Intercept, sum z phi) ridge | Reparameterise log_amp, log_ell, log_sigma to REAL; ONE `joint_nuts_block` (real-only) with `cfg.use_dense_metric = true` over `(Intercept, log_amp, log_ell, log_sigma, z[M])`. Add `+log_amp + log_ell` Jacobians manually. Identity metric on this geometry produces ESS=1-3 on amp/ell, so the dense metric is required here. See `examples/HSGPRegression.cpp`. |
| **Penalised B-spline / smoothing spline** -- `f(x) = Bsp(x) . (sds * z)` non-centered | **High** -- (Intercept, basis-coef-mean) ridge + (sds, z) funnel | Same pattern as HSGP: ONE `joint_nuts_block` (real-only) with `use_dense_metric = true` over `(Intercept, log_sds, log_sigma, z[K_s])`. Manual Jacobian for log_sds. See `examples/BSplineRegression.cpp`. |
| **ICAR / BYM / spatial CAR random effect** + **Gaussian** observation | Improper prior on phi (sum-to-zero needed) + (Intercept, mean phi) ridge | **Hybrid composite**: `gmrf_precision_block` over `phi[N]` (Rue 2001 sparse-Cholesky direct draw + hard sum-to-zero projection) + three separate `nuts_block` for Intercept (real), `log_tau` (positive), `log_sigma` (positive). The GMRF block samples phi via the exact Gaussian full-conditional in canonical form `Q = tau R + (1/sigma^2) diag(n_i)`; the NUTS blocks each see phi via shared_data. ~60x faster than a NUTS-only joint workaround. Use Half-Normal sigma prior (NOT Jeffreys) -- ICAR can absorb all variance, Half-Normal pushes back at sigma -> 0. See `examples/ICARSpatialGMRF.cpp`. |
| **ICAR / CAR / RW1 / RW2 sparse-precision random effect** + **non-Gaussian** observation (Poisson / Bernoulli / NB) | Improper prior on the latent (sum-to-zero needed for ICAR-style) + (Intercept, mean latent) ridge + likelihood-induced extra curvature | **Hybrid composite**: **`gmrf_whitened_ess_block`** over the latent (Murray 2010 ESS on the implicit GMRF prior via Rue 2001 sparse-Cholesky backsolve; sum-to-zero preserved by ESS rotation linearity) + separate `nuts_block` for the linear-predictor intercept (real), log-precision (positive), and any non-spatial random effect scales. User supplies `Q_fn(ctx) -> arma::sp_mat` and `log_lik(x, ctx) -> double` (the user's observation log-density evaluated at the proposed latent). See `block_catalogue/index.md` `gmrf_whitened_ess_block` section for the example recipe and verified convergence budgets at N=16 / N=64. |
| **Hidden discrete latent** (HMM / Ising / Potts / MRF on graph) **+ Normal emission** (Gaussian observation per latent class, possibly with per-class sigma_k) | Low for emission params given z (conjugate); High for z under spatial / sequential prior | **Hybrid composite**: specialized prior block for z (`hmm_block` for HMM, `binary_gibbs_block` for binary Ising/MRF, `categorical_gibbs_block` for K-state Potts) + **`normal_gamma_cluster_gibbs_block`** for per-class `(mu_k, lambda_k)` (Normal-Gamma conjugate; treat z as the partition). Label switching handled via post-hoc sort in runner. AVOID (not recommended) a `joint_nuts_block` with a `delta > 0` ordering constraint here -- NUTS dual-averaging interacts badly with slow-mixing z and can bias the posterior for (mu_k, sigma_k). |
| **BNP mixture (Dirichlet Process / Pitman-Yor) -- unknown number of components K** | **Allocation z is discrete; pi is a stick-breaking simplex (correlated by construction); (mu, lambda) per cluster are conjugate** | **Truncated SBP (Ishwaran-James 2001)**: `categorical_gibbs_block` (z) + `stick_breaking_block` (pi, with DP or PY a_fn / b_fn) + `normal_gamma_cluster_gibbs_block` (mu, lambda diagonal Normal-Gamma) + `nuts_block` on log(alpha). For alpha as a derived function of other parameters, register a `register_refresher("alpha", ...)`. CRP-marginal Neal Alg 2/8 and Jain-Neal split-merge NOT shipped. See `examples/DPGaussianMixture.cpp` / `examples/PYGaussianMixture.cpp` / `examples/DPGaussianMixture_DerivedAlpha.cpp`. **Caveat**: DP truncated SBP intrinsically over-clusters on well-separated fixtures (see the DP block notes in `block_catalogue/index.md`). When K is known, prefer the finite-K wrapper below. |
| **Finite-K Gaussian mixture (K known)** | Allocation z discrete; pi Dirichlet conjugate; (mu, lambda) Normal-Gamma conjugate | `categorical_gibbs_block` (z) + `dirichlet_gibbs_block` (pi, posterior alpha/K + counts) + `normal_gamma_cluster_gibbs_block` (mu, lambda diagonal). K and alpha are CONSTRUCTOR ARGS. See `examples/FiniteGaussianMixture.cpp`. **Recovers truth mu within 0.21 sigma on the same fixture where DP over-clusters** -- use this when K is known via domain knowledge or model selection. |
| **Hierarchical DP (HDP) Gaussian mixture -- clustering with G groups sharing atoms** | Multi-level: top-level beta + per-group pi_j; atoms shared across groups | **Truncated HDP** (Wang-Paisley-Blei 2011 simplified after Teh et al. 2006): `categorical_gibbs_block` (z) + `niw_cluster_gibbs_block` (mu, Sigma shared atoms) + `stick_breaking_block` (beta top-level, **HEURISTIC** update on combined counts) + G x `dirichlet_gibbs_block` (pi_j per group, posterior alpha*beta + counts_j). See `examples/HDPGaussianMixture.cpp`. **V0 caveat**: beta update is heuristic, not the rigorous Antoniak-table CRF (BayesMix is the porting reference for full HDP). |
| **DPMM with split-merge acceleration (mode-escape via Jain-Neal 2004)** | Cluster partition has slow single-i Gibbs mixing | Add `split_merge_block` as a child AFTER `categorical_gibbs_block` in any DPMM composite (DP / PY / Finite). Both write to `z`; composite allows multi-children writing the same key. See `block_catalogue/index.md` `split_merge_block` Sec. for details and acceptance asymmetry note. |
| Independent prior branches with no shared latent (e.g. two disjoint submodels) | Low | modular per parameter |

### Decision procedure

1. Use a **specialized shipped block** when it GENUINELY applies (BART,
   genbart, PG, Dirichlet / categorical conjugate, ESS for latent
   Gaussian, GMRF, celerite for 1-D GP, ...). These win on EFFICIENCY
   (and are correct-by-construction). Never force a poorly-matching one.
2. For recognized High-coupling patterns in the table above, group the
   parameters into ONE `joint_nuts_block`.
3. **All remaining continuous parameters -> ONE `joint_nuts_block` over a
   hand-written joint log-density. This is the DEFAULT.** Do NOT fragment
   a coupled model into separate `nuts_block`s, and do NOT contort it to
   fit a built-in block.
4. A standalone `nuts_block` is the **LOW-priority** fallback: a
   genuinely scalar parameter, a post-NCR funnel branch, or deliberate
   isolation.
5. **Structural gap (`codegen_priors.md Sec.2c` Exception 4):** if -- judged
   at generation time -- no block fits AND NUTS is structurally
   inapplicable (e.g. a bespoke tree ensemble, a novel trans-dimensional
   move), author a custom block/sampler with the Check #17 justification
   comment. Last resort; never a runtime swap.

### Coupling outside the table (no Sec.4a match)

When Sec.4a lists no matching pattern, the DEFAULT is still ONE
`joint_nuts_block` over a hand-written joint log-density for the coupled
/ unclaimed continuous parameters. You do NOT need to recognize a named
model class to justify joint -- writing the model's own log-density is
the point. Just satisfy the correctness checklist below.

**The general principle.** Identify whether the likelihood, as written,
makes two (or more) parameters jointly identifiable but each
individually weakly identified -- i.e., the data constrains some
function `g(theta_i, theta_j, ...)` more tightly than it constrains
the individual parameters. When this holds, the posterior is correlated
regardless of prior anchoring, and joint sampling may be warranted. Any
structural argument tracing back to this property counts.

**Non-exhaustive examples** of common likelihood-level patterns that
produce posterior correlation. These are illustrations of the kind of
structural argument expected, **not an enumeration**:

1. **Linear-combination invariance.** The likelihood depends on
   `(theta_i + theta_j)` or `(theta_i - theta_j)` but not strongly on
   either alone -- the data constrains the combination more tightly than
   the individuals.
2. **Multiplicative coupling.** The likelihood enters as a product
   `f(theta_i * theta_j)` (or `f(theta_i / theta_j)`) and is only weakly
   affected by holding the product fixed while varying the factors.
3. **Shared-role-in-derived-quantity.** Two parameters appear inside
   the same internal quantity (a variance, a mean, a scale, a
   probability) and the observable contribution of that quantity is
   what the data informs.
4. **Cancellation in a recurrence.** In any time-iterated or recursive
   expression involving past states / errors / values, two coefficients
   applied at different positions can produce a near-zero net effect
   over many steps, leaving them only weakly identified relative to each
   other.
5. **Explicit reparameterization invariance.** There exists a
   transformation `T(theta_i, theta_j) -> (theta_i', theta_j')` such
   that the likelihood is invariant up to data-scale anchoring (prior
   pulls them apart, but data does not).

**Other structural arguments are equally valid** as long as they trace
back to the likelihood as written. **Do not invoke a coupling pattern
by recognizing the model's name** -- that argument is not portable
across novel models the catalogue hasn't seen.

Joint NUTS IS the default for coupled / unclaimed continuous parameters.
The list below is the **correctness checklist you MUST satisfy when
hand-writing the joint log-density** (enforced by Check #11 / #24 / #12)
-- it is not a reason to fall back to modular.

### Joint NUTS correctness checklist (satisfy when writing a joint log-density)

Writing the joint log-posterior by hand concentrates the **semantic-bug
surface** into one function. Verify each item (the validator enforces
them):

1. **Missing prior term.** A joint log-posterior over
   `(theta_1, ..., theta_k)` must include `log p(theta_i)` for EVERY
   `i`. Forget one and the sampler silently targets a different
   posterior. Modular keeps each prior local to its block's lambda, so
   the failure mode does not exist.
2. **Missing Jacobian.** Each constrained slice (positive, simplex,
   correlation) needs `+log|d theta_natural / d theta_unc|` added to
   the joint log-density. Easy to drop one in a mixed block.
3. **Gradient bugs.** A single k-dim gradient is harder to verify by
   inspection than `k` individual 1-D gradients. Off-by-one, sign, and
   chain-rule errors are common.
4. **Initial-value compatibility.** All `k` parameters must
   simultaneously be in valid support; modular blocks each have local
   init checks.
5. **Identity-metric joint NUTS still struggles on tilted ridges.**
   Joint NUTS makes correlated proposals only via the gradient
   direction; severely anisotropic geometries need a dense metric
   (Check #18). Enable it on `joint_nuts_block` with
   `cfg.use_dense_metric = true`.

### Writing the joint log-density (mandatory checks)

`joint_nuts_block` is the default; when you write its log-density you MUST:

- (a) State the coupling / model structure in one sentence (header
  comment).
- (b) Include ALL prior terms and ALL Jacobians, and verify by
  inspection (count `log p` terms = number of parameters; count
  Jacobians = number of constrained slices). Check #11 audits this.
- (c) Either identity metric is geometrically acceptable, OR
  reparameterize to all-REAL and set `cfg.use_dense_metric = true` with
  a Check #18 justification.

(b) is mandatory and is NOT a reason to fragment a coupled model into
modular blocks -- it is the same prior / Jacobian content you would need
either way. A standalone `nuts_block` is fine only for a genuinely
separable / scalar parameter (the low-priority fallback).

Document in the header comment:

- If shipping joint (the default), write:
  `// JOINT JUSTIFICATION: <coupling argument>; subset = {<names>}; metric = <identity/dense>; log-posterior terms verified: <list>.`
- If shipping a standalone modular `nuts_block` (fallback), write:
  `// MODULAR NOTE: <param> is genuinely scalar / separable because <reason>.`

### Label switching: prefer post-hoc; in-sampler constraints are a discouraged fallback

Label switching (the symmetry between exchangeable component labels in
mixture / clustering / discrete-allocation models) should PREFERABLY be
resolved **POST-HOC in the runner**, using Stephens 2000 + Hungarian
assignment alignment per `label_switching.md`, with convergence judged on
label-invariant (order-statistic) R-hat. Breaking label switching inside the
C++ sampler via a structural constraint -- such as:

- Ordered K-vector constraint on a labeled parameter (e.g.,
  `mu_1 < mu_2 < ... < mu_K`)
- Monotone reparameterization between labeled parameters
- Positive-cut (`delta > 0`) reparam intended to enforce a specific
  labeling

-- is a **NOT-RECOMMENDED fallback (discouraged, not forbidden)**. Such
constraints interact badly with slow-mixing discrete-allocation companion
blocks (per `system_design.md` Sec.11.2(b)): the chain mixes on the constrained
scale and L2 / single-chain diagnostics may pass, but the natural-scale
posterior can be silently biased (invisible to R-hat / ESS but visible in
coverage and cross-implementation comparison). So reach for an in-sampler
constraint ONLY when a model genuinely cannot be resolved post-hoc -- some
models legitimately require it.

Choose your sampler (modular NUTS / joint NUTS / conjugate Gibbs / etc.)
based on Sec.4a + Sec.4b purely from coupling / conjugacy / efficiency
considerations -- independent of the label-switching question; prefer handling
label switching downstream in the runner.

### Warn the user when using joint_nuts_block

Whenever the generated sampler uses `joint_nuts_block`, add a comment
at the top of the `.cpp` stating which parameters are joint and why,
and emit in the R runner's comments:

```
# NOTE: this sampler uses joint_nuts_block for (<names>) due to tight
# coupling in the likelihood. Joint NUTS has a higher semantic-bug
# surface than modular NUTS; verify with validator Check #11
# (skills/validator.md) before relying on results for production.
```

This tightens the validator burden (Check #11 explicitly audits the
slicing, prior completeness, Jacobian, and write-back offsets in any
joint block).

### Post-run performance hint (in the R runner)

Always emit the following helper call at the end of the R runner so
that when a runner is slow (e.g. one that stayed modular as a fallback),
the user gets a clear joint-escape suggestion:

```r
ai4bayescode_perf_hint(
    wall_sec = total_wall_sec,                # actual elapsed time across stages
    n_sweeps_total = 2 * (n_burnin + n_keep),
    uses_joint_nuts = FALSE                   # TRUE if this runner already uses joint
)
```

`total_wall_sec` is set in the R2 parallel block of the runner (see
`codegen_r_runner.md` and `validator.md Sec.R2`). Do NOT use
`c1$wall_sec + c2$wall_sec` -- that double-counts under parallel execution.

See `codegen_r_runner.md` for the full runner template (the helper is
emitted into the runner, not part of AI4BayesCode core).

### Metric choice: prefer the adapted DIAGONAL metric (full dense = last resort)

The step size is ALWAYS capped (`max_step_size = pi`, default), so a
low-curvature boundary (Beta/binomial near 0/1, an overdispersed start)
can no longer run the dual averaging into a frozen chain -- that freeze
guard is automatic and needs no flag.

For the METRIC of a joint block, the adapted DIAGONAL metric is the
recommended default (the Stan / NumPyro default; the robust middle
ground). Do NOT enable the FULL-DENSE metric speculatively -- it is a
known fragility (2026-05-01: 70% of sim1 replicates stuck while every
single-dataset gate passed; corroborated 2026-06-16 by an external
developer: dense froze sd=0 / ESS~2 while a diagonal/identity metric
mixed cleanly).

**Decision tree:**

1. **Reparameterize first.** Centered hierarchical -> non-center; funnel ->
   Betancourt non-centered; heavy-tailed positive -> log-transform;
   multiplicative noise -> additive log-space. Removes anisotropy at the
   model level.

2. **Use the adapted DIAGONAL metric** for any non-trivial joint block:
   `cfg.use_diagonal_metric = true` (keep `use_dense_metric = false`). It
   rescales each axis by its posterior SD (windowed Welford), fixing
   axis-aligned ill-conditioning -- positive params spanning orders of
   magnitude in scale, the common "one tight direction" case -- WITHOUT
   the full-dense path's fragility. Measured on SDs (1, 10, 100): the
   wide axis goes from ESS ~350 (identity, biased SD) to ESS ~6500
   (diagonal, correct SD). This is the recommended default for generated
   joint samplers. For a tiny, already-unit-scale block, the IDENTITY
   metric (`use_diagonal_metric = false`; step-size only, NOT diagonal)
   is fine and skips the 1000-iter windowed warmup. Emit in the runner:
   ```
   # NOTE: joint_nuts_block metric = adapted diagonal (use_diagonal_metric).
   # If R-hat > 1.05 at 20k+20k, the geometry may be ROTATED (off-diagonal
   # correlation a diagonal metric can't capture) -- only then consider the
   # full-dense path (Check #18).
   ```

3. **Enable dense metric only when** ONE of:
   - The user explicitly requests it after seeing diagonal R-hat
     failure;
   - A documented diagonal-failure case applies. Currently three
     such cases are recognized:
     1. GP banana ridge in heteroscedastic / hierarchical / multi-
        output GP per `codegen_priors.md Sec.11.4` + `block_catalogue/index.md`
        "GP convergence troubleshooting ladder";
     2. **Centered hierarchical with G >= 3 groups inside a joint
        block** (e.g., `joint_nuts_block` over `(mu_u, u_1..u_G)`
        with `u_i ~ N(mu_u, tau)` directly, NOT
        `u_i = mu_u + tau * z_i`). This produces a (mu_u, u) ridge
        whose diagonal-metric step size collapses to the narrow
        per-axis width -- empirically observed to cause
        ESS_AI/ESS_Stan ratios of 1/100 to 1/500 even when rhat
        looks fine. **REQUIRED FIX is non-centered reparameterization**
        `u_i = mu_u + tau * z_i, z_i ~ N(0, 1)` which keeps diagonal
        sufficient. A centered scale-governed effect is a hard validator
        Check #24(a) FAIL -- dense metric is NOT an accepted substitute for
        NCR (the funnel freezes under any static metric). This is therefore
        a "reparameterize" case, NOT a dense-justification case;
     3. **Multivariate hierarchical** (large `G*d`) per the Sec.4a coupling
        table row "Multivariate hierarchical", WHERE R2/R3 diagnostics show
        the diagonal metric is inadequate (measured, not gated on a fixed
        dimension threshold);
   - The model class is documented to require dense in
     `block_catalogue/index.md`.

   Enabling triggers Check #18, which requires:
   - inline `// JUSTIFICATION (Check #18): ...` comment
   - `dense_metric_pilot_iters >= max(2000, 100 * d)`
   - `n_warmup_first_call >= pilot_iters + 1000`
   - `dense_metric_adapt_iters >= 2000`

**Three-phase warmup (`cfg.use_three_phase_warmup`) -- OPT-IN extreme-cond
escalation (default OFF since 2026-06-20).** Single-pilot is the default and
recommended warmup for dense: a broad 31-model corpus showed single-pilot gives
5-23x higher keep-phase ESS/s than three-phase and converges across all realistic
cond (the earlier "3-phase is the dense default" was REVERTED). Set
`cfg.use_three_phase_warmup = true` ONLY for a documented extreme-cond bootstrap
failure -- a high-curvature / ridge-trapping target (sparse spatial / temporal
random effects on large grids, ICAR) where the single identity pilot's Welford
covariance is biased by transient trapping on a low-dim ridge AND single-pilot
dense still gives R-hat > 1.05 at production budget. Three-phase is Stan-style
windowed warmup: Phase I 75 iter (step-size only, identity mass), Phase II 5
expanding windows 25->50->100->200->500 (each resets Welford + updates mass; recursive
self-correction across windows fixes ridge trapping), Phase III 50 iter (final
step-size tune, frozen mass). Total 1000 warmup iter. It is SLOWER in keep-phase
for ordinary targets (its short Phase III + dual-averaging epsilon_bar lag
under-tune the step), so it is an escalation, not a default. HARD RULE: three-phase
requires the dense metric -- diagonal + three-phase is gated off (under-tunes the
step ~38x). NOTE (2026-06-20 flip validation): the shipped HSGP example's
three-phase path was actually BROKEN (R-hat 1.017) and single-pilot FIXED it
(1.0015, 6x ESS). See system_design.md Sec.13.

**Why not enable dense (or three-phase) speculatively:** dense metric is single-
shot Welford with no windowed self-correction; speculative dense fails silently in
30-70% of replicates while single-dataset 2-chain validators pass. The 2026-05-01
retrospective on a piecewise-trend time-series model showed every L2 (13/13) and L3
(4k+4k 2-chain R-hat < 1.005, BPV in (0.02, 0.98)) gate passing while sim1 cross-
impl R-hat median was 2.0 with 70% replicates having `ess_bulk = NA` (chain stuck) --
the model was identifiable; the metric choice was the bug, and DIAGONAL recovered it.
The lever that catches this pre-ship: start diagonal + escalate on diagnostics
(Check #18); do not turn on dense or three-phase by guess.

**Existing examples to migrate:** any wrapper currently shipping
with `cfg.use_dense_metric = true` AND no Check #18 justification
must be retrofitted (add justification or flip to false) before
the next merge. Audit grep:
`grep -rn "use_dense_metric *= *true" examples/`.

---

## 5. File boilerplate (copy verbatim into every generated .cpp)

Every generated `.cpp` MUST begin with the project's GPL-3.0-or-later
license header (AI4BayesCode is distributed under GPL-3.0+ -- see
`LICENSE` at the repo root; any file linking against the vendored
BART kernel in `bart_pure_cpp/` inherits GPL regardless). Do NOT emit an
Apache-2.0 header; the core is no longer Apache.

```cpp
// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  <ClassName>.cpp -- <one-line description of the model>
//
//  <2-4 line model description: likelihood, parameters, priors>
//
//  @example:R / @example:python  -- MANDATORY runnable DGP block(s) for the
//  chosen backend(s); see "Header @example block" below.
// ============================================================================

// [[Rcpp::depends(RcppArmadillo)]]
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <RcppArmadillo.h>
#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
namespace constraints = AI4BayesCode::constraints;
```

For BART blocks, also `#include "AI4BayesCode/bart_block.hpp"` (the BART
kernel is GPL-2.0-or-later upstream; this is a transitive dependency,
not an additional license obligation beyond the project's own
GPL-3.0+).  For joint NUTS blocks, also
`#include "AI4BayesCode/joint_nuts_block.hpp"`. For GP models, add the
relevant headers (`elliptical_slice_sampling_block.hpp`,
`celerite_gp_block.hpp`, `celerite_marginal_likelihood.hpp`,
`libgp_kernels_unity.h`, `univariate_slice_sampling_block.hpp`) as
needed. Put all log-density functions in an anonymous `namespace { }`.

### Header `@example` block (mandatory) -- runnable DGP for `doc()`

The header MUST carry a small, **runnable** example with a self-contained
data-generating process (DGP), tagged by language, placed at the END of the
top comment block (after the model description, before `// [[Rcpp::depends`).
`ai4bayescode_doc()` (R) / `AI4BayesCode.doc()` (Python) parse this and show it
as the copy-paste Example; if the block for the caller's language is absent,
`doc()` falls back to an auto-generated `new(...)` stub -- so a missing block
never breaks the card, it just degrades to the stub.

Format (every line is a `//` comment; `doc()` strips the `// ` prefix):

```cpp
// @example:R
//   library(AI4BayesCode)
//   set.seed(1); N <- 200
//   <DGP: simulate data VALID for THIS model's likelihood>
//   ai4bayescode_sourceCpp("<ClassName>.cpp")   # compile+load; RELATIVE path ONLY
//   m <- new(<ClassName>, <ctor args>, 1L, TRUE)   # comment what each arg is
//   m$step(2000); str(m$get_current())
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(1); N = 200
//   <DGP: the same simulation in numpy>
//   Mod = AI4BayesCode.source("<ClassName>.cpp")   # RELATIVE path ONLY
//   m = Mod.<ClassName>(<ctor args>, 1, True); m.step(2000); print(m.get_current())
// @example:end
```

Rules:

- **Emit ONLY the block(s) for the runtime backend chosen in Sec.1.** R -> just
  `@example:R`; Python -> just `@example:python`; Both R+Python -> both. (The
  shipped library examples carry both; an R-bound model -- e.g. a `bart_block` /
  `genbart` kernel that has no Python binding -- carries `@example:R` only, and
  Python `doc()` falls back to the stub.)
- **The DGP must produce data VALID for the model's likelihood** -- Bernoulli
  outcomes in {0,1}, Poisson / NB counts as non-negative integers, a simplex
  that sums to 1, a connected graph for ICAR (node_idx + edges + N_nodes
  mutually consistent), design matrix and response with matching N, etc. A
  generic `rnorm` is correct ONLY for a Gaussian response. The whole point is
  copy-paste-runnable: it must actually run and return finite draws.
- **Keep it compact** (<= ~8 lines per backend) and **use the installed-package API
  with a RELATIVE path**: R -> `ai4bayescode_sourceCpp("<ClassName>.cpp")`
  then `new(<ClassName>, ...)`; Python -> `Mod = AI4BayesCode.source("<ClassName>.cpp")`
  then `Mod.<ClassName>(...)`. The path is ALWAYS the relative
  `<ClassName>.cpp` -- **NEVER an absolute `/Users/...` path, NEVER
  `AI4BayesCode_path=`, NEVER a `source(".../AI4BayesCode_helpers.R")` runner line.**
  Absolute paths break on another machine or if the folder moves, and the
  `@example` `doc()` shows runs inside the installed package.
- **Single source of the DGP**: the toy data here is the SAME simulation the
  runner skill (`codegen_r_runner.md` / `codegen_python_runner.md`) uses for the
  Layer-3 harness -- write it once and mirror it here so the two cannot drift.

### Authoring order -- C++ demo FIRST, then runners, then the comment

The example DGP has THREE views that must stay consistent; author them in this
order:

1. **C++ standalone demo first.** Write the `int main()` that simulates data
   from known parameters, fits the block, and checks recovery (the
   frontend-independent demo). This `int main()` is the **canonical DGP** -- the
   single source of truth for the example data. In a shipped library example it
   lives at the end of the file, guarded so it is active ONLY as a plain binary:

   ```cpp
   #if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
   #include <cstdio>
   int main() { /* simulate -> fit -> check recovery */ }
   #endif
   ```

2. **Then the R / Python runner**, translating that same simulation.
3. **Then distill the language-specific runner into the `@example:<lang>`
   comment** above (a compact single-chain copy).

**Validation keeps all three in sync.** If the R/Python example changes (a
different sample size, prior, or data shape), update the `int main()` demo AND
the `@example` comment to match -- they are three renderings of ONE DGP. A
validator pass must confirm the `@example` block actually runs and (for library
examples) that two independent chains converge (rank-normalized R-hat); the
embedded block itself stays single-chain (the 2-chain convergence run is a
validation step, never shipped in the example).

### Joint NUTS log-density template (the default path; see Sec.4a)

When the coupling analysis triggers `joint_nuts_block`, write ONE joint
log-density function that takes the CONCATENATED vector and fills the
full gradient. The slicing convention must match the `sub_params`
order declared in the config.

```cpp
// Example: joint (theta, b) with layout [theta (N); b (J)]
double joint_theta_b_log_density(const arma::vec& theta_cat,
                                 const block_context& ctx,
                                 arma::vec* grad) {
    const std::size_t N = /* ... from ctx or captured ... */;
    const std::size_t J = /* ... from ctx or captured ... */;
    if (theta_cat.n_elem != N + J) {
        return -std::numeric_limits<double>::infinity();  // Check #11.6
    }
    auto theta = theta_cat.subvec(0,     N - 1);        // Check #11.1 slice align
    auto b     = theta_cat.subvec(N, N + J - 1);

    double lp = 0.0;
    if (grad) { grad->set_size(N + J); grad->zeros(); }

    // (1) every sub-param's prior contributes -- Check #11.2
    // (2) all entries on REAL scale (no mixed scales) -- Check #11.3
    // (3) identity transform -> no Jacobian -- Check #11.4
    // (4) likelihood: write d/dtheta_i to grad[i],
    //     write d/db_j to grad[N + j]                  // Check #11.1
    // (5) compute likelihood and gradient via BLAS per Sec.6.1; the slice
    //     convention in (4) applies to where the gradient is ASSIGNED,
    //     NOT to how it's computed. See Sec.6.1 "Joint case" for the
    //     canonical (regular BLAS slice + irregular gather/scatter)
    //     pattern. Validator Check #19.

    if (!std::isfinite(lp)) return -std::numeric_limits<double>::infinity();
    return lp;
}

// Assembly inside constructor:
joint_nuts_block_config cfg;
cfg.name = "theta_b_joint";                        // block name, not parameter name
cfg.sub_params.push_back({ "theta", N });          // offsets [0, N)
cfg.sub_params.push_back({ "b",     J });          // offsets [N, N+J)
cfg.initial_cat = arma::join_cols(theta_init, b_init);
cfg.log_density_grad = &joint_theta_b_log_density;
cfg.n_warmup_first_call = 1000;                    // joint blocks need more runway
impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
```

The `declare_dependencies` call is keyed by the **block name**, not by
individual sub-parameter names:

```cpp
impl_->data().declare_dependencies(
    "theta_b_joint", {"Y", "M", "sigma_b"});  // block name, union of sub-param reads
```

Sub-parameter names (`theta`, `b`, ...) appear in shared_data as
individual keys automatically -- the joint block writes each slice back
to its own key via `current_named_outputs()`.

**Hard rules for joint_nuts_block:**
- Every sub-parameter name must be unique within the block AND must be
  the same name downstream blocks read via `declare_dependencies`.
- The slicing order in the log-density MUST match the `sub_params`
  order in the config. Document the layout in a comment above the
  log-density function (validator Check #11.1).
- Per-slice constraints are supported directly -- each sub-parameter
  may declare its own constraint (REAL / POSITIVE / SIMPLEX /
  CHOLESKY_CORR / COV_MATRIX / ...) and the block applies the
  transform + log|Jacobian| internally. Mixing constraint types within
  a single block is fine; you do NOT need to split a positive / simplex
  parameter into a separate `nuts_block`.
- In the R runner's comments and in the `.cpp` header, note that the
  sampler uses joint NUTS and requires validator Check #11.

---

## 6. Log-density lambda template

```cpp
cfg.log_density_grad =
    [](const arma::vec& theta_unc, const block_context& ctx,
       arma::vec* grad) {
        return constraints::<KIND>::wrap(
            theta_unc, grad,
            [&](const arma::vec& theta_nat, arma::vec* grad_nat) -> double {
                // AI-WRITTEN: natural-scale log p + gradient
                // Read from ctx. Write lp and *grad_nat.
                // DO NOT include Jacobian. DO NOT touch theta_unc.
                return lp;
            });
    };
```

---

## 6.1 Vectorized gradient computation -- armadillo / BLAS (mandatory for GLMs)

The natural-scale log-density inside the `wrap(...)` lambda **MUST** be
written using armadillo's high-level matrix / vector operators rather
than nested `for` loops over `(i, j)` index pairs. armadillo's `*`, `.t()`,
`solve(...)`, `chol(...)`, `dot(...)` operators dispatch to the linked
BLAS / LAPACK (Apple Accelerate on macOS, OpenBLAS / MKL elsewhere),
which is 10-50x faster than a hand-rolled scalar inner loop on the
exact same math.

The cost of a slow gradient is multiplicative across the whole MCMC:
each NUTS step makes O(10-100) gradient calls. A 10x slow gradient =
10x slow chain. Concrete observed gap on a small (N=80, p=7)
Bernoulli logistic regression: scalar-loop AI 45.9s vs Stan
`bernoulli_logit_glm` 4.2s, an 11x gap that is entirely from this rule
being violated.

### Storage rule

Pre-store the design matrix and any token / observation index arrays
as native armadillo types in `shared_data` once, in the constructor.
Do NOT store flat 1-D vectors that the log-density has to "reshape"
mentally on every call.

```cpp
// In constructor -- once, at construction:
impl_->data().set("X_mat", X);            // arma::mat (N x p), not flat
impl_->data().set("y",     y);            // arma::vec (N)

// In the log-density lambda -- fast path:
const arma::mat& X = ctx.at_mat("X_mat"); // zero-copy reference
const arma::vec& y = ctx.at("y");

arma::vec eta = X * theta_nat;            // (N x p)(p x 1) = (N x 1) -- BLAS gemv
```

If `shared_data_t` does not yet have `at_mat()` and the existing
`ctx.at("X")` returns a flat `arma::vec`, reconstruct a non-owning
mat **once** at the top of the lambda (zero copy):

```cpp
const arma::vec& X_flat = ctx.at("X_mat_flat");
const arma::mat X(const_cast<double*>(X_flat.memptr()),
                  N, p, /*copy_aux=*/false, /*strict=*/true);
arma::vec eta = X * theta_nat;            // BLAS gemv from here on
```

(`const_cast` is safe because `strict=true` makes the mat read-only.)

### Dimension comment rule

Every BLAS expression in the log-density MUST be annotated with the
shape of each operand and the result. This forces the AI to compute
dimensions before writing, and gives the human reviewer a one-second
sanity check:

```cpp
arma::vec eta   = X * theta_nat;          // (N x p)(p x 1) = (N x 1)
arma::vec p_hat = 1.0 / (1.0 + arma::exp(-eta));   // (N x 1) elementwise
double    lp    = arma::dot(y, eta) - arma::accu(arma::log1p(arma::exp(eta)));
                                          //  scalar - scalar = scalar
if (grad_nat) {
    *grad_nat = X.t() * (y - p_hat);      // (p x N)(N x 1) = (p x 1)
}
```

If a line lacks a dimension comment, the AI is operating without
verifying its shape contract -- a leading source of silent gradient
bugs. This is enforced as part of Check #12 review (the dimension
comments must be present and correct).

### Per-observation-family templates

Use these as starting points; substitute the right `theta_nat`
slicing and the model's prior contribution. Every line is annotated
with shapes. All three patterns achieve Stan-tier per-iteration speed
for typical GLM dimensions (N <= 10^4, p <= 10^2).

#### Bernoulli logistic regression

```cpp
// y in {0, 1}^N, X is N x p, theta_nat = beta (p x 1)
const arma::mat& X = X_mat_view;          // (N x p), pre-stored
const arma::vec& y = ctx.at("y");         // (N x 1)

arma::vec eta = X * theta_nat;            // (N x p)(p x 1) = (N x 1)
// log p(y) = sum_i [ y_i eta_i - log(1 + exp(eta_i)) ]
//          = dot(y, eta) - accu(log1p(exp(eta)))
double lp = arma::dot(y, eta);
for (auto& e : eta) lp -= (e > 0
        ? e + std::log1p(std::exp(-e))     // numerically stable softplus
        : std::log1p(std::exp(e)));

if (grad_nat) {
    arma::vec p_hat = 1.0 / (1.0 + arma::exp(-eta));   // (N x 1)
    *grad_nat = X.t() * (y - p_hat);      // (p x N)(N x 1) = (p x 1)
}
```

#### Gaussian regression (known sigma)

```cpp
// y, X, theta_nat = beta. sigma scalar from ctx.
arma::vec eta  = X * theta_nat;            // (N x 1)
arma::vec res  = y - eta;                   // (N x 1)
double inv_s2  = 1.0 / (sigma * sigma);
double lp      = -0.5 * arma::dot(res, res) * inv_s2 - N * std::log(sigma);
if (grad_nat) {
    *grad_nat = X.t() * res * inv_s2;       // (p x 1)
}
```

#### Poisson log-regression

```cpp
// y in N_0^N (counts), eta = log rate, theta_nat = beta
arma::vec eta = X * theta_nat;              // (N x 1)
arma::vec mu  = arma::exp(eta);             // (N x 1)
double lp     = arma::dot(y, eta) - arma::accu(mu);   // -sum(lgamma(y+1)) is constant; drop
if (grad_nat) {
    *grad_nat = X.t() * (y - mu);           // (p x 1)
}
```

#### Gaussian likelihood with sigma sampled

When sigma is also a parameter (gradient w.r.t. sigma), add the sigma
contribution; X * theta_nat etc. are the same.

```cpp
arma::vec eta = X * beta_nat;               // (N x 1)
arma::vec res = y - eta;                    // (N x 1)
double sse    = arma::dot(res, res);        // scalar
double lp     = -0.5 * sse / (sigma * sigma) - N * std::log(sigma);
if (grad_beta)  *grad_beta  = X.t() * res / (sigma * sigma);   // (p x 1)
if (grad_sigma) *grad_sigma = -N / sigma + sse / std::pow(sigma, 3);
```

#### Joint case (`joint_nuts_block`)

In a joint log-density the gradient is a CONCATENATED vector -- the
slice convention from Check #11.1 says where each piece is written,
NOT how it is computed. Compute via BLAS, then assign into the
appropriate slice. Use scalar loops ONLY for irregular gather /
scatter (e.g., random-effects indexed by a per-observation group ID).

Canonical pattern -- joint `(alpha, beta, u)` for a hierarchical
linear model with `g_idx[n]  in  {0..G-1}` mapping observation n to
its group, `theta_cat = [alpha (1); beta (p); u (G)]`:

```cpp
// Slice the concatenated vector -- Check #11.1 layout.
const double alpha = theta_cat[0];
auto beta = theta_cat.subvec(1,         1 + p - 1);
auto u    = theta_cat.subvec(1 + p, 1 + p + G - 1);

// Irregular gather: u_n[i] = u[g_idx[i]]. Scalar loop is correct.
arma::vec u_n(N);
for (std::size_t n = 0; n < N; ++n) {
    u_n[n] = u[static_cast<std::size_t>(g_idx[n])];
}

// Linear predictor and residual via BLAS.
arma::vec mu  = alpha + X * arma::vec(beta) + u_n;       // (N x p)(p x 1) + (N x 1)
arma::vec res = y - mu;                                  // (N x 1)
double lp     = -0.5 * arma::dot(res, res) / sigma2;     // scalar

if (grad) {
    arma::vec g_mu = res / sigma2;                       // (N x 1)
    (*grad)[0]                  += arma::accu(g_mu);     // alpha slice -- sum
    grad->subvec(1, 1 + p - 1)  += X.t() * g_mu;         // beta slice -- BLAS gemv
    // Irregular scatter into u slice. Scalar loop is correct.
    for (std::size_t n = 0; n < N; ++n) {
        (*grad)[1 + p + static_cast<std::size_t>(g_idx[n])] += g_mu[n];
    }
}
```

Two anti-patterns this rules out: (i) a nested `for (i) for (j)`
over `X_flat[i + j * N] * beta[j]` to compute `X * beta` -- replace
with `X * arma::vec(beta)`; (ii) a scalar accumulation of
`grad[1 + j] += g_mu[i] * X_flat[i + j * N]` to scatter the beta
gradient -- replace with `grad->subvec(1, 1 + p - 1) += X.t() * g_mu`.

### What NOT to write

```cpp
// ANTI-PATTERN -- DO NOT EMIT THIS
for (size_t i = 0; i < N; ++i) {
    double eta_i = 0.0;
    for (size_t j = 0; j < p; ++j) eta_i += X_flat[i + j * N] * theta_nat[j];
    lp += y[i] * eta_i - softplus(eta_i);
    double resid = y[i] - sigmoid(eta_i);
    for (size_t j = 0; j < p; ++j) (*grad_nat)[j] += resid * X_flat[i + j * N];
}
```

Two scalar inner loops over `j` per observation. No SIMD, no BLAS,
no cache blocking. ~10-50x slower than the `X * theta_nat` /
`X.t() * (y - p)` two-line equivalent.

### Elementwise helper functions -- armadillo expressions, NOT scalar loops in disguise

A second anti-pattern that silently undoes the BLAS gain: writing
helper functions like `softplus_vec`, `sigmoid_vec`, `inv_logit_vec`
as a `for` loop inside a `arma::vec`-returning wrapper. The function
LOOKS vectorized from the outside but is fully scalar inside.

```cpp
// ANTI-PATTERN -- DO NOT EMIT THIS
inline arma::vec sigmoid_vec(const arma::vec& z) {
    arma::vec out(z.n_elem);                             // heap alloc
    for (std::size_t i = 0; i < z.n_elem; ++i) {         // scalar loop
        out[i] = (z[i] >= 0.0)
            ? 1.0 / (1.0 + std::exp(-z[i]))
            : std::exp(z[i]) / (1.0 + std::exp(z[i]));
    }
    return out;
}
```

Two costs: (a) the inner loop is scalar (no SIMD); (b) every call
allocates a fresh `arma::vec out(N)` (heap allocation, ~50-100ns
overhead each). Each gradient evaluation typically calls these
helpers 2-4 times, so 2-4 heap allocations per call. Across 40k
NUTS iterations x ~500 leapfrog steps that's tens of millions of
allocations -- measurable slowdown.

Use armadillo's elementwise primitives instead. Each operator
(`arma::exp`, `arma::log1p`, `arma::abs`, `arma::clamp`, scalar
arithmetic on vec) is implemented as a SIMD-vectorized expression
template. The whole expression evaluates lazily with ONE allocation
of the final result.

```cpp
// PREFERRED -- armadillo expression, lazy evaluation, SIMD
inline arma::vec sigmoid_arma(const arma::vec& z) {
    return 1.0 / (1.0 + arma::exp(-z));                  // 1 alloc, SIMD
}

inline arma::vec softplus_arma(const arma::vec& z) {
    // log(1 + exp(z)) numerically stable: |z| + log1p(exp(-|z|)) for z<=0,
    // z + log1p(exp(-z)) for z>0 -- combine via abs:
    arma::vec az = arma::abs(z);                         // 1 alloc
    return arma::log1p(arma::exp(-az))                   // expression, 1 alloc total
         + arma::clamp(z, 0.0, arma::datum::inf);
}

// log_inv_logit and log1m_inv_logit (numerically stable):
inline arma::vec log_inv_logit_arma(const arma::vec& z) {
    return -softplus_arma(-z);                           // log(sigma(z))
}
```

Even simpler -- when the helper is called once and the result is used
once, INLINE the expression directly in the log-density to skip the
function-call boundary entirely:

```cpp
arma::vec eta = X * theta_nat;
arma::vec p_hat = 1.0 / (1.0 + arma::exp(-eta));         // sigmoid inline
double lp = arma::dot(y, eta) - arma::accu(arma::log1p(arma::exp(arma::abs(eta))))
           - arma::accu(arma::clamp(eta, 0.0, arma::datum::inf));   // softplus inline
if (grad_nat) *grad_nat = X.t() * (y - p_hat);
```

### Why this isn't dangerous (despite "BLAS feels risky")

Three reasons the BLAS form is at least as safe as the loop form for
AI-generated code:

1. **Dimension errors are LOUD with armadillo, SILENT with loops.**
   `X * theta_nat` where `X.n_cols != theta_nat.n_elem` throws an
   immediate `std::logic_error` with a message naming the mismatched
   dimensions. A scalar loop with the wrong inner bound will happily
   read past the end of `theta_nat` (UB) or simply compute the wrong
   sum without complaint. The smoke test `worker(1)` catches all
   armadillo-thrown dim errors in seconds; loop-bound errors only
   surface when downstream metrics (rhat, coverage) drift.

2. **Check #12 (autodiff verify) catches semantic errors regardless
   of writing style.** A wrong gradient written as `X.t() * z` vs.
   `-X.t() * z` (sign flipped) fails Check #12 the same way as the
   loop equivalent. The dimension-comment rule above adds a second
   layer: AI verifies shape contract before writing.

3. **Cross-impl rhat catches anything subtle.** Stan is the
   reference; if the AI gradient is wrong, rhat against Stan goes >
   1.5 instantly.

Mandatory companion: **Check #12 verify file MUST also be vectorized**
the same way (autodiff::var supports armadillo `*`, `.t()`, `dot`).
Don't write the verify in scalar loops while the production cpp is
vectorized -- the verify must exercise the same code path.

### Performance gain in production

| Pattern (Bernoulli logistic, N=80, p=7) | wall (40k iter) | Notes |
|---|---|---|
| Stan `bernoulli_logit_glm` | 4.2s | reference |
| AI scalar nested-loop | 45.9s | observed |
| AI armadillo `X * theta` | ~5-10s (projected) | matches Stan to within constant factor |

For larger problems (e.g., a multilevel logistic with N~=1000 and ~50
total parameters), the gap widens -- scalar loop scales worse than
BLAS. The rule above is the cheapest single intervention for sampler
throughput.

---

## 6a. Posterior-predictive layer (PPC)

Every generated sampler MUST register a **stochastic refresher** for the
observation layer in the wrapper's constructor, and declare the predict
DAG edges from the parameters that enter the likelihood into a `y_rep`
node. This is what makes `predict_at(list())` return a posterior-
predictive `y_rep` matrix -- the input to validator Layer 3 R3.

### Why this is mandatory

The validator's Layer 3 R3 (Bayesian p-values + PSIS-LOO) consumes
`pp$y_rep` from `model$predict_at(list())`. If the wrapper has no
stochastic refresher for y_rep, R3 cannot run, and the generator has
shipped a sampler whose posterior the validator cannot check. Failing
to register the refresher is a semantic hole equivalent to failing
Check #1 (wrong observation distribution).

### What to add, step by step

For a model with observation `y_i | theta ~ p(y | theta, ...)`:

**1. In the wrapper constructor**, right after the Gibbs-DAG declarations:

```cpp
// Predict DAG: every likelihood parent -> y_rep
impl_->data().declare_predict_edges("mu",    {"y_rep"});
impl_->data().declare_predict_edges("sigma", {"y_rep"});
// (for models with covariates, X -> derived -> y_rep too)

// Reserve a y_rep slot so downstream refreshers have somewhere to write.
// Length matches training y.
impl_->data().set("y_rep", arma::vec(y.n_elem, arma::fill::zeros));

// Stochastic refresher for the observation layer. Signature:
//   arma::vec(const shared_data_t& d, std::mt19937_64& rng)
impl_->data().register_stochastic_refresher(
    "y_rep",
    [N_obs = y.n_elem](const AI4BayesCode::shared_data_t& d,
                        std::mt19937_64& rng) {
        const double mu_    = d.get("mu")[0];
        const double sigma_ = d.get("sigma")[0];
        std::normal_distribution<double> norm(0.0, 1.0);
        arma::vec y_rep(N_obs);
        for (std::size_t i = 0; i < N_obs; ++i) {
            y_rep[i] = mu_ + sigma_ * norm(rng);
        }
        return y_rep;
    });
```

**2. Add the mutable predict RNG member** to the wrapper class (see Sec.8).
Thread it into every `impl_->predict_at(replaced, predict_rng_)` call.
`predict_at` stays `const`.

**3. In the Rcpp `predict_at(Rcpp::List)` method**, the `y_rep` key will
already be in the returned `block_context` -- just convert to
`Rcpp::NumericVector` and add it to the output list. No special case
needed on the wrapper side.

### Per-observation-family templates (refresher + pointwise_loglik)

Pick the row matching the model's observation likelihood. The left
column is the stochastic refresher body (goes inside the lambda in
step 1 above); the right column is the R-side `pointwise_loglik` helper
(emitted into the runner, consumed by Layer 3 R3).

#### Gaussian: `y_i ~ N(mu_i, sigma^2)`

Refresher:
```cpp
std::normal_distribution<double> norm(0.0, 1.0);
arma::vec y_rep(N_obs);
for (std::size_t i = 0; i < N_obs; ++i) {
    y_rep[i] = mu_vec[i] + sigma * norm(rng);
}
return y_rep;
```
Pointwise log-lik (R):
```r
pointwise_loglik <- function(hist, y) {
    mu_d    <- hist$mu      # (iter,) or (iter x N)
    sigma_d <- hist$sigma   # (iter,)
    S <- length(sigma_d); N <- length(y)
    LL <- matrix(NA_real_, S, N)
    for (d in seq_len(S)) {
        mu_row <- if (is.matrix(mu_d)) mu_d[d, ] else rep(mu_d[d], N)
        LL[d, ] <- dnorm(y, mu_row, sigma_d[d], log = TRUE)
    }
    LL
}
```

#### Bernoulli: `y_i ~ Bern(p)`

Refresher:
```cpp
std::bernoulli_distribution bern(p);
arma::vec y_rep(N_obs);
for (std::size_t i = 0; i < N_obs; ++i) {
    y_rep[i] = bern(rng) ? 1.0 : 0.0;
}
return y_rep;
```
Pointwise log-lik (R):
```r
pointwise_loglik <- function(hist, y) {
    p_d <- hist$p
    S <- length(p_d); N <- length(y)
    LL <- matrix(NA_real_, S, N)
    for (d in seq_len(S)) LL[d, ] <- dbinom(y, 1, p_d[d], log = TRUE)
    LL
}
```

#### Poisson: `y_i ~ Poisson(f_i)`, f_i = exp(log_f_i)

Refresher:
```cpp
arma::vec y_rep(N_obs);
for (std::size_t i = 0; i < N_obs; ++i) {
    std::poisson_distribution<int> pois(std::exp(log_f[i]));
    y_rep[i] = static_cast<double>(pois(rng));
}
return y_rep;
```
Pointwise log-lik (R):
```r
pointwise_loglik <- function(hist, y) {
    lf_d <- hist$log_f        # (iter x N) matrix
    S <- nrow(lf_d); N <- length(y)
    LL <- matrix(NA_real_, S, N)
    for (d in seq_len(S)) {
        LL[d, ] <- dpois(y, exp(lf_d[d, ]), log = TRUE)
    }
    LL
}
```

#### Multinomial counts: `y ~ Multinomial(N_tot, theta)`

Refresher:
```cpp
const arma::vec& theta = d.get("theta");
std::discrete_distribution<int> cat(theta.begin(), theta.end());
arma::vec counts(theta.n_elem, arma::fill::zeros);
for (int n = 0; n < N_total; ++n) {
    counts[static_cast<std::size_t>(cat(rng))] += 1.0;
}
return counts;
```
Pointwise log-lik (R): use category-level log-multinomial,
`dmultinom(y, size = sum(y), prob = theta, log = TRUE)`, broadcast per
draw. Note: LOO is per-observation; for Multinomial the "observation
unit" is the count vector, so LL has shape `iter x 1`.

#### BART-noise: `y_i ~ N(f_bart_i, sigma^2)`

Refresher: same shape as Gaussian, use `f_bart[i]` as the mean.

Pointwise log-lik (R):
```r
pointwise_loglik <- function(hist, y) {
    F_mat <- hist$f_bart   # iter x N
    sig_d <- hist$sigma
    S <- length(sig_d); N <- length(y)
    LL <- matrix(NA_real_, S, N)
    for (d in seq_len(S)) LL[d, ] <- dnorm(y, F_mat[d, ], sig_d[d], log = TRUE)
    LL
}
```

### Hard rules

1. **Every sampler registers a stochastic refresher for y_rep** unless
   the model genuinely has no observation layer (rare). If the
   observation likelihood is not in the table above, pick the closest
   analogue and document the choice in a header comment in the `.cpp`.
2. **Length of y_rep matches the training y** by default; wrappers
   that accept new covariates via `predict_at(list(X = X_new))` may
   resize y_rep to match X_new's row count inside the lambda (read
   N from shared_data if needed).
3. **The refresher MUST NOT use `std::random_device{}()`** inside its
   body -- always use the supplied `rng` argument. See validator
   Check #13 (RNG separation).
4. **A sampler may skip y_rep** only if the code-gen can prove the
   observation is deterministic given parameters (vanishingly rare).
5. **NEVER collapse intermediate deterministic latents into the y_rep
   refresher.** The Gibbs sampler MAY marginalize / collapse an
   intermediate latent for efficiency (that's a sampling decision and
   belongs to the Gibbs DAG, not the predict DAG). But the predict
   DAG MUST reconstruct the full generative chain as shown to the user
   for verification -- every node from the user-verified DAG appears
   in `shared_data`, every edge appears in `declare_predict_edges`,
   and every deterministic intermediate gets its own
   `register_refresher` so `predict_at` walks the chain explicitly.
   The acceptance test: after codegen, `get_dag()$predict_edges`
   keys  U  values,  U  `data_inputs`, MUST equal the node set of the
   user-verified DAG (modulo the `y_rep` renaming of the observed
   `y` node). See Sec.6c below for the canonical multi-layer pattern
   and validator.md Check #6 for the structural check.

---

## 6c. Multi-layer predict DAGs -- preserve every intermediate

**The single most failure-prone codegen pattern.** When the user-
verified DAG has an intermediate deterministic latent like

```
flowchart TD
    X      --> f_BART
    Z_mat  --> Zbeta
    beta   --> Zbeta
    f_BART --> theta
    Zbeta  --> theta
    tau    --> theta
    theta_raw --> theta
    theta  --> y
    v2     --> y
```

the AI is strongly tempted to inline `theta = f_BART + Zbeta + tau *
theta_raw` directly into the `y_rep` refresher body and collapse the
DAG to

```
declare_predict_edges("f_BART", {"y_rep"});
declare_predict_edges("Zbeta",  {"y_rep"});
declare_predict_edges("tau",    {"y_rep"});
declare_predict_edges("v2",     {"y_rep"});
// theta and theta_raw silently dropped
```

This is **WRONG** for two reasons:

1. **DAG visualization lies to the user.** `ai4bayescode_plot_dag(model)` no longer
   matches the DAG they approved at the verification step. The
   collapsed DAG drops `theta` and `theta_raw`. The user signed off
   on a multi-layer generative story; the code ships a single-layer
   story.

2. **`predict_at` with intermediate replacement breaks.** If a future
   user wants to call `predict_at(list(theta = theta_new))` (e.g. to
   propagate a posterior summary of theta through the observation
   layer without resampling), there is no `theta` node in the DAG to
   replace. The framework's DAG walk has nothing to anchor on.

### Correct pattern: two-layer refreshers

Reconstruct **every node** in the user-verified DAG. Pick the right
refresher kind per node:

- **Deterministic intermediate** (a function of its parents, no
  randomness given parents): `register_refresher(key, fn)`. Example:
  `theta = f_bart + Zbeta + tau * theta_raw`.
- **Stochastic intermediate** (a draw from a distribution given
  parents -- rare; usually only the terminal y_rep is stochastic):
  `register_stochastic_refresher(key, fn)`.
- **Stochastic terminal** (y_rep): `register_stochastic_refresher`,
  reads ONLY direct generative parents from the user-verified DAG
  (here: `theta` and `v2`, NOT `f_BART`, `Zbeta`, `tau`, `theta_raw`).

```cpp
// === Shared-data slots -- one per node in the user-verified DAG. ===
impl_->data().set("theta",  arma::vec(N, arma::fill::zeros));
impl_->data().set("y_rep",  arma::vec(N, arma::fill::zeros));
// (X, Z_mat_flat, v2 enter via .set or declare_data_input upstream;
//  f_bart, beta, Zbeta, tau, theta_raw come from their respective blocks.)

// === Predict DAG -- every edge from the user-verified DAG. ===
// Layer 1: feeders into the deterministic intermediate.
impl_->data().declare_predict_edges("f_bart",    {"theta"});
impl_->data().declare_predict_edges("Zbeta",     {"theta"});
impl_->data().declare_predict_edges("tau",       {"theta"});
impl_->data().declare_predict_edges("theta_raw", {"theta"});
// Layer 2: theta + observation noise feeders into y_rep.
impl_->data().declare_predict_edges("theta",     {"y_rep"});
impl_->data().declare_predict_edges("v2",        {"y_rep"});
// (X is a data_input. Z_mat_flat is a data_input. v2 is a
//  data_input -- they all enter the predict DAG via the next layer.)
impl_->data().declare_data_input("X");
impl_->data().declare_data_input("Z_mat_flat");
impl_->data().declare_data_input("v2");

// === Deterministic refresher for the intermediate theta. ===
impl_->data().register_refresher(
    "theta",
    [](const AI4BayesCode::shared_data_t& d) {
        const arma::vec& f_bart    = d.get("f_bart");
        const arma::vec& Zbeta     = d.get("Zbeta");
        const double     tau_      = d.get("tau")[0];
        const arma::vec& theta_raw = d.get("theta_raw");
        // theta = f_bart + Zbeta + tau * theta_raw  (NCP).
        return f_bart + Zbeta + tau_ * theta_raw;
    });

// === Stochastic refresher for y_rep -- reads ONLY theta and v2. ===
impl_->data().register_stochastic_refresher(
    "y_rep",
    [](const AI4BayesCode::shared_data_t& d, std::mt19937_64& rng) {
        const arma::vec& theta = d.get("theta");
        const arma::vec& v2    = d.get("v2");
        const std::size_t N    = theta.n_elem;
        std::normal_distribution<double> norm(0.0, 1.0);
        arma::vec y_rep(N);
        for (std::size_t i = 0; i < N; ++i) {
            y_rep[i] = theta[i] + std::sqrt(v2[i]) * norm(rng);
        }
        return y_rep;
    });
```

### Acceptance test (run mentally before shipping)

1. List every node in the user-verified mermaid DAG (here: `X`,
   `Z_mat`, `beta`, `tau`, `tau_smooth`, `theta_raw`, `f_BART`,
   `Zbeta`, `theta`, `y`, `v2`).
2. Read the generated wrapper. Collect:
   - keys appearing as source or target of `declare_predict_edges`,
   - keys appearing in `declare_data_input`.
3. Rename `y` -> `y_rep` in the user-verified set (this is the only
   permitted renaming).
4. Take the symmetric difference. If non-empty, FIX before shipping
   -- never let the DAG drift from what the user approved.

### Sampling vs prediction -- Gibbs may collapse, predict may NOT

You may legitimately marginalize an intermediate at SAMPLING time
for efficiency. Example: if `theta = f_bart + Zbeta + tau *
theta_raw` is a non-centered parameterization, the Gibbs sweep can
target `(f_bart, Zbeta, tau, theta_raw)` directly without ever
materializing a `theta` block -- that's a coupling decision on the
Gibbs DAG, NOT a license to collapse the predict DAG. **The predict
DAG always reconstructs the full generative chain**, with
`register_refresher("theta", ...)` providing the explicit
recomputation rule. Conversely, a centered parameterization that
HAS a `theta` block in the Gibbs DAG is also fine -- both
parameterizations produce the same generative DAG. The predict DAG
is invariant to sampler parameterization choices.

### Generative-DAG context edges (`declare_context_edges`) -- required

After the predict-DAG edges, emit `declare_context_edges(from, {to})`
for the model's **prior / hyperprior** structure so `ai4bayescode_plot_dag(model)`
renders the full generative story (solid predict sub-DAG + faded
prior context). These are VIZ-ONLY: predict_at's BFS never reads
`context_edges_` (shared_data.hpp). Rules:

1. Topology is **read from the model's own prior code, never
   invented**. For every named shared_data slot that parameterizes a
   sampled parameter's prior, add `hyperslot -> param`. Examples:
   `sigma^2 ~ IG(nu/2,nu*lambda/2)` with slots `sigma_nu, sigma_lambda`
   => `sigma_nu->sigma`, `sigma_lambda->sigma`. `theta ~ Dir(alpha)`
   with slot `alpha` => `alpha->theta`.
2. A sampled "forest/kernel" parameter feeding a deterministic mean
   node gets a context edge too: `BART->f_bart`, `genBART->r`,
   `amplitude->K_matrix`, `lengthscale->K_matrix` (parallel to the
   user-approved MetaRegBartSpline `BART->f_bart`).
3. If a model's priors are hardcoded constants with NO named slot
   (`mu ~ N(0,100^2)`, `sigma ~ Jeffreys`, `beta ~ N(0,10^2)`),
   emit NO context edges -- the generative DAG then correctly equals
   the predict DAG. Do not fabricate constant-prior nodes.
4. **Disjointness invariant:** a context edge `hyperparam -> param`
   MUST NOT also appear in `declare_predict_edges`. Posterior
   prediction conditions on the MCMC draws of `param`, not on
   re-sampling its prior; putting the prior edge in the predict DAG
   would wrongly trigger BFS recomputation of `param`. Keep
   `predict_edges_` and `context_edges_` disjoint.

Gold standard: MetaRegBartSpline.cpp. The 28 shipped examples with
named hyperprior slots all declare context edges; see
PREDICT_AT_AUDIT for the per-example topology table.

---

## 6b. Gradient verification via autodiff (mandatory -- validator Check #12)

The production `.cpp` carries only the hand-written log-density +
analytic gradient. **No autodiff / Eigen code goes into the delivered
file.** Check #12 happens in a THROWAWAY companion file that the AI
creates, uses, and DELETES during generation. The user never sees
verification code.

### Workflow

```
                 AI GENERATION STEP
                 ------------------

  1. AI writes production example:      examples/MyModel.cpp
     (hand-written log-density + analytic grad only; no verification code)

  2. AI writes verify companion:        tests_autodiff/verify_MyModel.cpp
     - Copies the hand-written log-density functions verbatim from step 1
       (both come from the same LLM think, so drift is negligible)
     - Adds templated (autodiff::var-compatible) versions of the same math
     - Adds a single `verify_MyModel_grad(...)` function with [[Rcpp::export]]

  3. AI compiles + runs:
       ai4bayescode_sourceCpp("tests_autodiff/verify_MyModel.cpp", ...)
       r <- verify_MyModel_grad(synthetic_data, n_points = 5, seed = 12345)
     Assert every `max_diff_<block> < 1e-8` for AD-differentiable
     blocks, `< 1e-5` for FD-based blocks (see "Fallback" below).

  4. On PASS:   AI deletes tests_autodiff/verify_MyModel.cpp AND
                runs `rmdir tests_autodiff/` to remove the (now empty)
                directory. Final user workspace contains ONLY the
                production .cpp + the R / Python runner -- no
                tests_autodiff/ folder of any kind.
     On FAIL:   AI pinpoints which block's diff exceeds, fixes the
                hand-written gradient in the production file, and
                re-runs from step 3. Common bugs are listed at the
                bottom of this section.

                       PRODUCTION TIME
                       ---------------

  User runs Rcpp::sourceCpp("MyModel.cpp") -- the file contains only
  production code. No autodiff runtime dep, no performance overhead,
  no verification artifacts. 100% clean.
```

### Companion file template (tests_autodiff/verify_<ClassName>.cpp)

```cpp
// Check #12 (gradient verification) for <ClassName>.cpp. THROWAWAY -- deleted after
// PASS confirmed.

// [[Rcpp::depends(RcppArmadillo)]]
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <RcppArmadillo.h>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/autodiff_wrap.hpp"

// For blocks needing AD verification (no special functions in lp body):
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include <cmath>
#include <limits>
#include <random>

using AI4BayesCode::block_context;
namespace cs  = AI4BayesCode::constraints;
namespace adw = AI4BayesCode::autodiff_wrap;

// ---- (1) COPIES of hand-written log-density functions from production ----
// VERBATIM from examples/<ClassName>.cpp -- same math, same grad formula.
// AI pastes these in at generation; no drift because both versions come
// from the same prompt.
namespace {

double <block1>_log_density(const arma::vec& theta_nat,
                             const block_context& ctx,
                             arma::vec* grad_nat) {
    // ... same body as in production file ...
}
// ... and every other NUTS block's log-density ...

// ---- (2) Templated (AD-compatible) versions of the SAME math ----
// Returns only the scalar lp; no gradient. T = autodiff::var at call
// time. Math functions use unqualified `log`, `exp`, etc. for ADL.
template <typename Vec>
auto <block1>_lp_ad(const Vec& theta_nat, const block_context& ctx) {
    using T = typename std::decay<decltype(theta_nat[0])>::type;
    T lp = T(0);
    // ... identical lp math, no grad writes, use `lp = lp - ...` style ...
    return lp;
}

// ---- (3) FD helper for non-AD blocks (lgamma / digamma etc.) ----
template <typename WrapFn>
arma::vec fd_grad(const arma::vec& theta_unc, WrapFn wrap_fn,
                   double h = 1e-5) {
    const std::size_t n = theta_unc.n_elem;
    arma::vec g(n);
    for (std::size_t i = 0; i < n; ++i) {
        arma::vec tp = theta_unc; tp[i] += h;
        arma::vec tm = theta_unc; tm[i] -= h;
        g[i] = (wrap_fn(tp, nullptr) - wrap_fn(tm, nullptr)) / (2.0 * h);
    }
    return g;
}

} // anonymous namespace

// [[Rcpp::export]]
Rcpp::List verify_<ClassName>_grad(/* data args... */,
                                    int n_points = 5,
                                    int seed     = 12345) {
    // Build one block_context per NUTS block, populate with data + any
    // other blocks' current values (fixed to reasonable defaults).

    std::mt19937_64 rng(static_cast<std::uint64_t>(seed));
    std::uniform_real_distribution<double> unif_real(-2.0, 2.0);
    std::uniform_real_distribution<double> unif_log (-1.0, 1.5);

    double max_diff_<block1> = 0.0;
    // ...

    for (int k = 0; k < n_points; ++k) {
        // --- AD verification for blocks without special functions ---
        {
            arma::vec th{ /* random unconstrained */ };
            arma::vec g_hw(dim), g_ad(dim);
            double lp_hw = cs::<kind>::wrap(th, &g_hw,
                [&](const arma::vec& t, arma::vec* g) {
                    return <block1>_log_density(t, ctx_<block1>, g);
                });
            double lp_ad = adw::wrap_<kind>(th, &g_ad,
                [&](const auto& t) { return <block1>_lp_ad(t, ctx_<block1>); });
            double d = std::max(std::abs(lp_hw - lp_ad),
                                arma::max(arma::abs(g_hw - g_ad)));
            if (d > max_diff_<block1>) max_diff_<block1> = d;
        }

        // --- FD verification for blocks whose lp uses lgamma / digamma etc. ---
        // autodiff.hpp does NOT overload these special functions, so we
        // fall back to central finite-difference of the hand-written
        // wrap. FD precision ~= 1e-5.
        {
            arma::vec th{ /* random unconstrained */ };
            auto wf = [&](const arma::vec& t, arma::vec* g) {
                return cs::<kind>::wrap(t, g,
                    [&](const arma::vec& tn, arma::vec* gn) {
                        return <block_special>_log_density(tn, ctx_<bs>, gn);
                    });
            };
            arma::vec g_hw(dim);
            wf(th, &g_hw);
            arma::vec g_fd = fd_grad(th, wf);
            double d = arma::max(arma::abs(g_hw - g_fd));
            if (d > max_diff_<block_special>) max_diff_<block_special> = d;
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("max_diff_<block1>") = max_diff_<block1>,
        // ...one entry per block...
        Rcpp::Named("n_points") = n_points,
        Rcpp::Named("seed")     = seed);
}
```

### Constraint-kind selector (picks the right AD wrap)

`cs::<kind>` / `adw::wrap_<kind>` follow the block's constraint:

| Block's constraint | hand-written wrap | AD wrap |
|---|---|---|
| `real` (any dim)            | `cs::real::wrap`            | `adw::wrap_real` |
| `positive` scalar           | `cs::positive::wrap`        | `adw::wrap_positive` |
| `simplex` K-dim             | `cs::simplex::wrap`         | `adw::wrap_simplex` |
| `lower_bounded(lo)`         | `cs::lower_bounded::wrap`   | `adw::wrap_mixed` with `slice_spec::lower(dim, lo)` |
| `upper_bounded(hi)`         | `cs::upper_bounded::wrap`   | `adw::wrap_mixed` with `slice_spec::upper(dim, hi)` |
| `interval(lo, hi)`          | `cs::interval::wrap`        | `adw::wrap_mixed` with `slice_spec::interval(dim, lo, hi)` |
| **mixed joint** (e.g. real + positive in one block) | n/a (hand-written signs each slice) | `adw::wrap_mixed({slice_spec::real(d1), slice_spec::positive(d2), ...})` |

**Joint blocks with real-only slices** (`joint_nuts_block`): use direct
`autodiff::gradient` in the verify function -- the concatenated vector
has identity transform so no wrap needed; just build `VectorXvar`,
call the templated lp, compute `gradient()`.

**Joint blocks with mixed constraint slices** (`joint_nuts_block` with
per-slice constraints, e.g. real + positive): the production block
handles transforms + log|Jacobian| internally; the user-written
`joint_log_density` operates on the natural scale. Verify by direct
`autodiff::gradient` on the natural-scale lp (no Jacobian needed in the
verify -- the user's lp doesn't include one either).

### Fallback: FD verification for lgamma / digamma

`autodiff.hpp` does not provide overloads for `lgamma`, `digamma`,
`beta`, or similar R-specific special functions. Blocks whose lp body
uses `R::lgammafn`, `R::digamma`, etc. cannot be verified via autodiff --
use **central finite-difference** instead (see the `fd_grad` helper
above). Tolerance: `< 1e-5` (FD precision ceiling with `h = 1e-5`).

Both verification mechanisms catch the main bug class (arithmetic errors in
the hand-written gradient): FD confirms grad matches numerical
derivative of lp; AD confirms grad matches reverse-mode autodiff.

### At generation time, the AI runs

```r
ai4bayescode_sourceCpp("tests_autodiff/verify_<ClassName>.cpp")

# Synthetic data matching the model's shape
set.seed(1)
y <- rnorm(100)   # adapt per model

r <- verify_<ClassName>_grad(y, 20L, 42L)
cat("Check #12 results:\n"); print(r)

# Pass: AD-backed blocks < 1e-8; FD-backed blocks < 1e-5.
ad_diffs <- r[grepl("^max_diff_.*(?<!_fd)$", names(r), perl = TRUE)]
fd_diffs <- r[grepl("^max_diff_.*_fd$",   names(r))]
if (any(unlist(ad_diffs) >= 1e-8) || any(unlist(fd_diffs) >= 1e-5)) {
    stop("Check #12 FAILED -- see which max_diff exceeded threshold")
}
cat("Check #12 PASS\n")
# => AI deletes tests_autodiff/verify_<ClassName>.cpp at this point.
```

If any `max_diff` exceeds its threshold -> AI pinpoints the block,
opens the hand-written gradient, and fixes. Typical bugs:
- sign error in a term (`d/dx(-0.5 x^2) = -x`, not `+x`)
- factor missing (forgot `2` from `d/dx x^2 = 2x`)
- wrong denominator (`sigma^2` vs `sigma^3` in sigma gradient)
- double-counted Jacobian (user added `+ log(sigma)` inside the
  natural-scale function while `constraints::positive::wrap` is
  already adding it)

Once `Check #12 PASS`, AI **deletes `tests_autodiff/verify_<ClassName>.cpp`**.
The production `<ClassName>.cpp` is delivered unchanged -- no #ifdef,
no scaffolding, no Eigen, no autodiff references.

### Templated function gotchas

When you write the templated (AD) version:

- Use `T` (the template parameter) as the scalar type, not `double`.
- Use `lp = lp - ...` / `lp = lp - 0.5 * r * r / sigma2;` style
  instead of `lp += ...;` -- some autodiff types don't support compound
  assignment robustly.
- Initialise `T lp = T(0);` (not `double lp = 0.0`).
- Math functions: `log`, `exp`, `sqrt`, `pow` -- these are
  ADL-overloaded for `autodiff::var`; plain `std::log` etc. will NOT
  see autodiff overloads. Just call `log(x)` unqualified.
- **No mutation of `ctx`**. Read-only. The same ctx is used across
  verify iterations.
- For positive / simplex / ordered / etc. parameters, the templated
  function always sees the NATURAL-scale value (e.g. sigma > 0, not
  log(sigma)) -- just like the hand-written function.

---

## 6c. Custom genBART likelihood (when the 10 shipped families don't fit)

When the user's model has a likelihood that is NOT one of the 10
shipped `genbart::lik::*` classes (Normal / Logistic / Poisson / NB /
Heteroscedastic / AFT log-logistic / AFT generalized gamma /
Gamma_shape / Beta / Beta-Binomial), the codegen agent can emit a
custom subclass of `genbart::likelihood` inside the generated `.cpp`.
The contract is minimal -- only `log_f` and `name()` are mandatory;
`score` and `obs_info` have finite-difference defaults but analytic
overrides are strongly recommended for speed (called n times per tree
update).

### Template (anonymous namespace inside the `.cpp`)

```cpp
namespace {
class my_custom_lik : public genbart::likelihood {
public:
    explicit my_custom_lik(/* user-supplied nuisance init, prior hypers */)
        : sigma_(...) {}

    double log_f(double y, double lambda, std::size_t /*obs_i*/) const override {
        // NATURAL-scale log p(y | lambda). Must return -inf
        // (not NaN) off-support.
        const double z = (y - lambda) / sigma_;
        return -0.5 * std::log(2.0 * M_PI) - std::log(sigma_) - 0.5 * z * z;
    }

    double score(double y, double lambda, std::size_t /*obs_i*/) const override {
        // d log_f / d lambda. Defaults to central FD if omitted.
        return (y - lambda) / (sigma_ * sigma_);
    }

    double obs_info(double /*y*/, double lambda, std::size_t /*obs_i*/) const override {
        // -d^2 log_f / d lambda^2 (observed info; primary input to the
        // Laplace leaf posterior). Must be non-negative; lower bound
        // handled by the framework's v_precision_lb clamp.
        return 1.0 / (sigma_ * sigma_);
    }

    // Optional: update nuisance parameter(s) in a Gibbs / RW-MH step
    // after each full tree sweep. See normal_lik / negative_binomial_lik
    // for examples.
    void update_nuisance(const std::vector<double>& y,
                         const std::vector<double>& lambda,
                         genbart::rn& gen) override {
        // ... RW-MH or conjugate Gibbs on sigma_ ...
    }

    // Optional: precompute per-observation constants when Y is set.
    void prepare(const std::vector<double>& y) override {
        // ... cache lgamma, lchoose, delta lookups, etc. ...
    }

    const char* name() const override { return "my_custom"; }
    std::size_t num_nuisance() const override { return 1; }
    std::vector<double> nuisance_snapshot() const override {
        return std::vector<double>{sigma_};
    }

private:
    double sigma_;
};
} // namespace
```

### Wiring into the wrapper

Inside the wrapper constructor, own the likelihood via a
`std::unique_ptr<genbart::likelihood>` member and pass `.get()` to
`genbart_block_config::lik`:

```cpp
likelihood_ = std::make_unique<my_custom_lik>(sigma_init);

genbart_block_config cfg;
cfg.name     = "r";
cfg.x_train  = X;
cfg.y_init   = y;
cfg.lik      = likelihood_.get();
cfg.ntrees   = ntrees;
cfg.hypers   = genbart::rjmcmc_hypers{};
impl_->add_child(std::make_unique<genbart_block>(std::move(cfg)));
```

The y_rep stochastic refresher (Sec.6a) still needs to know how to sample
from the likelihood; the codegen agent writes the family-specific draw
body (`rnorm`, `rpois`, rejection sampler for a general-purpose
custom family, etc.) and captures the `likelihood_.get()` pointer
in the lambda to read nuisance parameters at PPC time.

### Safety checklist for custom likelihoods

- `log_f` MUST return `-std::numeric_limits<double>::infinity()`
  (never NaN) outside the support.
- `obs_info` MUST be non-negative.
- Any `exp`/`log` on unbounded lambda should use the same clamp pattern
  as `genbart::lik::poisson_lik` (safe_exp: clamp at +/-50) to avoid
  overflow inside the Laplace proposal.
- If the likelihood needs per-observation metadata (censoring
  indicators, trial counts, etc.), store in subclass state (vector
  keyed by `obs_i`) and consult via the `obs_i` argument.
- If update_nuisance uses RW-MH, tune `prop_sd` so acceptance rate
  lands in [0.2, 0.5] per Linero 2022 Sec.3.4 recommendations.

---

## 7. Assembly order (inside constructor)

1. Install fixed data + initial values: `impl_->data().set("y", y);`
2. Declare dependencies: `impl_->data().declare_dependencies("mu", {"y", "sigma"});`
3. (Optional) Register refreshers for deterministic derived quantities.
4. **Declare predict DAG edges** into `y_rep` and **register the y_rep
   stochastic refresher** (see Sec.6a above).
5. Add child blocks in Gibbs-sweep order: `impl_->add_child(...)`.

---

## 7a. set_current(X, y) dispatcher -- dynamic-N canonical pattern

**Hard rule (system_design.md Sec.7 rule 4 + rule 6).** If `X` or `y`
appears as a `set_current` key:
- **`p` (number of predictors) is strict** -- internal block state
  (beta, gamma, tau2, mixture cluster vectors, etc.) is allocated
  with length `p` at construction. Changing `p` requires
  reconstructing the wrapper; the dispatcher MUST reject p
  mismatches with a clear "reconstruct the wrapper" error message.
- **`N` (number of observations) is dynamic** -- Gibbs blocks read
  `N` dynamically from `y.n_elem` / `X.n_elem / p` inside their
  log-density / refresher lambdas. The dispatcher MUST update the
  cached `N_` field whenever `X` changes, resize the `y_rep` slot,
  and clear history if mid-run in `keep_history_` mode.
- **Cross-check** `X.nrow` against `y.length` when both are in the
  same call.

### Canonical template

```cpp
void set_current(Rcpp::List params) {
    // ... handle non-data keys first (sigma, beta, gamma, tau2, ...) ...

    const bool has_X = params.containsElementNamed("X");
    const bool has_y = params.containsElementNamed("y");

    // -------- X branch --------
    if (has_X) {
        Rcpp::NumericMatrix X_new =
            Rcpp::as<Rcpp::NumericMatrix>(params["X"]);

        if (static_cast<std::size_t>(X_new.ncol()) != p_) {
            Rcpp::stop("set_current: X has %d columns but model was "
                       "constructed with p = %zu. p is fixed by internal "
                       "block state; reconstruct the wrapper to change p.",
                       X_new.ncol(), p_);
        }
        const std::size_t N_new = static_cast<std::size_t>(X_new.nrow());

        // Cross-check y if also in this call.
        if (has_y) {
            Rcpp::NumericVector y_new =
                Rcpp::as<Rcpp::NumericVector>(params["y"]);
            if (static_cast<std::size_t>(y_new.size()) != N_new) {
                Rcpp::stop("set_current: X has %zu rows but y has length "
                           "%d -- provide matching dimensions.",
                           N_new, y_new.size());
            }
        }

        arma::mat X_mat(X_new.begin(), X_new.nrow(), X_new.ncol(), false);
        impl_->data().set("X", arma::vectorise(X_mat));

        if (N_new != N_) {
            // y_rep slot resize so refresher writes into the right-sized
            // buffer. The refresher reads N dynamically from X.n_elem /
            // p; do NOT capture N from constructor scope (validator
            // Check #6 / #19).
            impl_->data().set("y_rep", arma::vec(N_new, arma::fill::zeros));
            // History buffers from previous N are now incomparable.
            if (keep_history_ && impl_->history_size() > 1) {
                Rcpp::warning("set_current: X row count changed from %zu "
                              "to %zu -- clearing history buffers (mixed-N "
                              "history is unsupported).", N_, N_new);
                impl_->clear_history();
            }
        }
        N_ = N_new;   // refresh cached N (system_design.md Sec.7 rule 6)
    }

    // -------- y branch --------
    if (has_y) {
        arma::vec y_new = Rcpp::as<arma::vec>(params["y"]);
        if (y_new.n_elem != N_) {
            Rcpp::stop("set_current: y length %zu != current N = %zu",
                       y_new.n_elem, N_);
        }
        impl_->data().set("y", y_new);
    }
}
```

### Refresher capture rule (must hold)

The y_rep stochastic refresher (and any other refresher reading X/y)
must compute N dynamically -- capture `p` (a true constant) from the
constructor scope but read N from data each time:

```cpp
// CORRECT -- N is derived from the scratch context dynamically
impl_->data().register_stochastic_refresher(
    "y_rep",
    [p](const AI4BayesCode::shared_data_t& d, std::mt19937_64& rng) {
        const arma::vec& X_flat = d.get("X");
        const std::size_t N_cur = X_flat.n_elem / p;   // DYNAMIC
        // ...
    });

// WRONG -- N captured from constructor scope; predict_at(list(X=X_new))
// with different N reads / writes out of bounds.
impl_->data().register_stochastic_refresher(
    "y_rep",
    [N, p](const AI4BayesCode::shared_data_t& d, std::mt19937_64& rng) {
        // N is the construction-time value; X_new size is ignored
        // ...
    });
```

### Anti-pattern (legacy wrappers -- DO NOT REPLICATE)

```cpp
// LEGACY (system_design Sec.7 rule 4 violation):
if (static_cast<std::size_t>(X_new.nrow()) != N_ ||
    static_cast<std::size_t>(X_new.ncol()) != p_)
    Rcpp::stop("X dimensions must match construction");
// -- compares against frozen N_; rejects any legitimate N change.
```

If you ABSOLUTELY must keep N strict (e.g. because the model's prior
or block-state genuinely depends on training N -- rare; HMM hidden-state
length is one example), then say so explicitly in the error:

```cpp
// STRICT-N legitimate use (HMM, ARMA, change-point -- N is a model
// invariant, not just a data dimension):
if (X_new.nrow() != N_)
    Rcpp::stop("set_current: this model fixes N at construction "
               "because <REASON>. To change N, reconstruct the wrapper.");
```

The validator (Check #19) accepts EITHER the dynamic-N canonical
template OR the strict-N pattern with documented reason.

---

## 8. Class shape

```cpp
class <ClassName> {
public:
    <ClassName>(/* data + int rng_seed + bool keep_history = false */);
    void                      step();                             // no-arg = 1 sweep; body is just `{ step(1); }`
    void                      step(int n_steps);                  // loops impl_->step(rng_) n_steps times
    AI4BayesCode::state_map   get_current() const;                // backend-neutral; Rcpp/pybind auto-convert
    void                      set_current(const AI4BayesCode::state_map& params);
    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const;
    AI4BayesCode::dag_info    get_dag() const;                    // = impl_->get_dag()
    AI4BayesCode::history_map get_history() const;                // = impl_->get_history()
    // CONDITIONAL 7th method -- emit ONLY if the composite contains any
    // nuts_block / joint_nuts_block child (see Sec.9). Then also add the
    // readapt_rng_ member below.
    void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1);
private:
    std::mt19937_64                             rng_;
    mutable std::mt19937_64                     predict_rng_;  // Sec.6a
    mutable std::mt19937_64                     readapt_rng_;  // Sec.9, NUTS-family only
    std::unique_ptr<AI4BayesCode::composite_block> impl_;
    bool                                        keep_history_ = false;
};
```

**Constructor body (canonical):** construct `impl_` with a NAME string --
`impl_(std::make_unique<composite_block>("<ClassName>"))` in the init list
(the ONLY ctor is `composite_block(std::string name)`; NEVER pass a bool) --
then, as the LAST statement of the constructor body, enable history with
`if (keep_history_) impl_->set_keep_history(true);`. The wrapper's
`keep_history` argument is stored in `keep_history_`, NOT forwarded to the
`composite_block` constructor. See `examples/GaussianLocationScale.cpp`
(ctor init list ~line 162, body ~line 211).

**Return BACKEND-NEUTRAL types -- NEVER `Rcpp::List` in the class methods.** They
return `AI4BayesCode::state_map` (= `unordered_map<string, arma::vec>`),
`history_map` (= `unordered_map<string, arma::mat>`), and `dag_info`; Rcpp **and**
pybind11 auto-convert them (codegen.md Sec.1). Declaring them `Rcpp::List` is the
obsolete R-only style and forces a manual conversion that does not compile.

**CONSTRUCT `impl_` with a NAME string -- the ONLY constructor is
`composite_block(std::string name)`:**

```cpp
// In the wrapper constructor init list:
impl_(std::make_unique<composite_block>("<ClassName>")),
keep_history_(keep_history)
```

- `std::make_unique<composite_block>("<ClassName>")` -- the **ONLY**
  constructor is `composite_block(std::string name)`. **NEVER** pass a
  `bool` / `keep_history` to it: `std::make_unique<composite_block>(keep_history)`
  **DOES NOT COMPILE** (there is no `composite_block(bool)` ctor). The
  wrapper's `keep_history` argument is stored in the `keep_history_`
  field, NOT forwarded to the `composite_block` constructor.
- **Enable per-draw history via a SEPARATE call in the constructor body**
  (NOT a ctor argument): `if (keep_history_) impl_->set_keep_history(true);`
  as the LAST statement of the constructor.
  Ground truth: `examples/GaussianLocationScale.cpp` line ~162
  (`std::make_unique<composite_block>("GaussianLocationScale")`) and
  line ~211 (`if (keep_history_) impl_->set_keep_history(true);`).

**The engine `impl_` (a `composite_block`) exposes ONLY these methods -- do not
invent others:**
- `impl_->step(rng_)` -- ONE Gibbs sweep; **LOOP it** for `n_steps`. It takes the
  rng **by reference**, NOT `impl_->step(n, rng)`.
- `impl_->set_keep_history(true)` -- enable per-draw history (call in the
  constructor body when `keep_history_` is true; see above).
- `impl_->current_named_outputs()` -> `state_map` (the current draw).
- `impl_->get_history()` -> `history_map`; `impl_->get_dag()` -> `dag_info`.
- `impl_->data()` (the shared DataContext: `.set(...)`, `.declare_dependencies(...)`,
  `.declare_data_input(...)`, `.register_stochastic_refresher(...)`),
  `impl_->add_child(std::make_unique<...block>(...))`.

There is **NO `impl_->get_current()`** -- build the class's `get_current()` by
`return impl_->current_named_outputs();` (or assemble from the child blocks).
The canonical, copy-this reference is `examples/GaussianLocationScale.cpp`
(its `get_current` / `get_history` / `predict_at` / `get_dag` bodies).

**NUTS-family note:** if the composite contains ANY `nuts_block` or
`joint_nuts_block` child, the wrapper MUST ALSO expose a conditional 7th
method `void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1);`
and carry a 3rd RNG member `mutable std::mt19937_64 readapt_rng_;` (seeded
once in the ctor init list). See Sec.9 "Conditional 7th method `readapt_NUTS`"
below for the exact wrapper-class additions and the
`examples/GaussianLocationScale.cpp` reference (lines ~158, ~288, ~296).

Expose via `RCPP_MODULE(<ClassName>_module) { ... }`.

### Why `mutable std::mt19937_64 predict_rng_`

The wrapper owns TWO RNG streams:
- `rng_` -- used by MCMC `step()`. Every `step()` call advances this stream.
- `predict_rng_` -- used by stochastic refreshers invoked via
  `predict_at` (y_rep sampling, see Sec.6a). Declared `mutable` so that
  `predict_at` can stay `const` while still advancing the RNG.

Both are seeded ONCE in the constructor from the user-provided seed
(with `predict_rng_` receiving the seed XORed with the golden-ratio
constant so the two streams are decoupled but reproducible):

```cpp
<ClassName>(..., int rng_seed, bool keep_history = false)
    : rng_(rng_seed == 0
               ? std::mt19937_64{std::random_device{}()}
               : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
      predict_rng_(rng_seed == 0
               ? std::mt19937_64{std::random_device{}()}
               : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                 ^ 0x9E3779B97F4A7C15ULL}),
      ...
{ ... }
```

**Never call `std::random_device{}()` inside `predict_at`** -- it is
slow and non-reproducible on macOS (see validator Check #13). Seed
predict_rng_ once at construction and let it advance naturally across
predict_at calls.

### Initial values: NEVER expose by default

**Do NOT surface per-parameter initial values (e.g. `mu_init`,
`sigma_init`, `theta_init`, `beta_init`) as constructor arguments in
the generated class.** The wrapper must choose initial values
internally, deterministically, from the data and/or the prior -- for
example:
- Location (mu, alpha, beta): `mean(y)`, `0`, or `OLS(y ~ X)`
- Scale (sigma, tau): OLS residual SD when well-defined, else `sd(y)`
  (BART blocks: `bart_block::current_sigma()` which is the calibrated
  `sigest`)
- Probability (p): the prior mean `a / (a + b)`, clamped away from 0/1
- Simplex (theta): the prior mean `alpha / sum(alpha)`
- Concentration (kappa, theta): `1.0` or the prior mean

Rationale: initial values affect MCMC warmup but not the target
posterior; exposing them pollutes the R-facing interface with knobs
that 99% of users don't need and can't set intelligently. A user who
genuinely wants overdispersed inits for R-hat diagnostics can call
`model$set_current(list(...))` **after** construction -- that's what
`set_current` is for.

**Exception (and the ONLY exception):** if the user explicitly asks
("expose `mu_init` as a constructor argument so I can tune it") during
the prior-elicitation flow, add it -- clearly marked `[init,
exposed by user request]` in the summary table. Do NOT offer it
proactively; only honor an explicit ask.

### predict_at method

Every generated class MUST include a `predict_at(Rcpp::List) const`
method. The behavior depends on (1) whether the model has data
inputs and (2) whether the class supports `keep_history`.

**WARNING Critical #1 (data_input forwarding):** if the constructor calls
`declare_data_input("...")` for ANY key, the `predict_at` wrapper
MUST accept that key in `new_data` and forward to
`impl_->predict_at(replaced, predict_rng_)`. **Do NOT hard-reject
non-empty `new_data` with `Rcpp::stop` if any data_input exists** --
that silently breaks posterior predictive at new covariates
(compile + R-hat pass, but `predict_at(list(X = X_new))` errors or
uses the OLD X, producing meaningless predictions).

**WARNING Critical #2 (keep_history -> history-mode predict):** if the
class has a `keep_history_` field (constructor accepts `keep_history`
and stores per-draw history), `predict_at` MUST include a separate
branch that **loops over all posterior draws** and returns an
`n_draws x N` matrix per refreshed key. A single-draw `predict_at`
under `keep_history_ = TRUE` is the silent-wrong-posterior-predictive
bug: the user expects the full posterior predictive distribution and
gets a single point estimate at the last MCMC iteration instead.

**Refresher should use dynamic N**, not a captured constructor-time
`N`. Use `X_flat.n_elem / p` (or equivalent) so the refresher works
correctly when `predict_at(list(X = X_new))` provides a different
sample size:

```cpp
impl_->data().register_stochastic_refresher(
    "y_rep",
    [p](const AI4BayesCode::shared_data_t& d, std::mt19937_64& rng) {
        const arma::vec& X_flat = d.get("X");
        const std::size_t N_cur = X_flat.n_elem / p;   // DYNAMIC, not captured
        const arma::vec& beta   = d.get("beta");
        const double sigma      = d.get("sigma")[0];
        std::normal_distribution<double> norm(0.0, 1.0);
        arma::vec y_rep(N_cur);
        for (std::size_t i = 0; i < N_cur; ++i) { /* ... */ }
        return y_rep;
    });
```

DO NOT capture `N` from the constructor scope (`[N, p](...)`) -- the
refresher then can't handle `predict_at(list(X = X_smaller))`.

#### Models with covariates (e.g. X in regression / BART)

**Required pattern: every covariate read by the y_rep refresher
must be BOTH `declare_data_input("X")` AND a source of
`declare_predict_edges("X", {"y_rep"})` (or to an intermediate
non-data-input).** As of the Pass-2 availability rule relaxation
(shared_data.hpp), these are NO LONGER mutually exclusive -- a
data_input with a default training value in shared_data counts as
"available" for any stochastic refresher that lists it as a
predict-DAG parent. predict_at(list()) samples y_rep at the
training X; predict_at(list(X = X_new)) replaces X in the scratch
context and the refresher samples at the new X. Both code paths
go through the SAME predict-DAG walk.

Why declare the edge AT ALL if the refresher reads `X` via
`d.get("X")` anyway? Two reasons, both critical:

1. **ai4bayescode_plot_dag visualisation.** Without the edge, `X` appears as an
   orphan node in the rendered DAG -- the user can't see that `X`
   produces `y_rep` in the generative story.
2. **Full-Reconstruction Discipline (validator Check #6 (B)/(C)/(D)).**
   Every key the refresher reads must be a declared direct parent
   of the refresher's output node. Reading an undeclared key is a
   silent contract violation; the validator's audit script flags
   it.

**Correct pattern (Gaussian regression):**

```cpp
impl_->data().declare_data_input("X");                       // X is replaceable
impl_->data().declare_predict_edges("X",     {"y_rep"});     // X -> y_rep (NEW)
impl_->data().declare_predict_edges("beta",  {"y_rep"});     // beta -> y_rep
impl_->data().declare_predict_edges("sigma", {"y_rep"});     // sigma -> y_rep
// Refresher reads d.get("X"), d.get("beta"), d.get("sigma") -- exactly
// the three declared parents. Check #6 condition (C) satisfied.
```

**Also correct** (X feeds an intermediate non-data-input node, BART-
style): X -> f_bart (deterministic propagation via the bart_block's
`predict`), f_bart -> y_rep. Here X is on the predict DAG, but its
direct child is a *non*-data-input node, so y_rep's parents are all
available. See `examples/BartNoise.cpp` for this pattern.

Constructor declaration template:

Canonical wrapper template (non-BART; for BART see `BartNoise.cpp`,
which uses the same two-branch pattern with `bart_child.predict()`
and `bart_child.predict_history()` for history-mode):

```cpp
Rcpp::List predict_at(Rcpp::List new_data) const {
    // ---- Parse `new_data` once (shared by both modes) -------------
    bool has_X = new_data.size() > 0;
    arma::vec x_flat;
    if (has_X) {
        Rcpp::CharacterVector names = new_data.names();
        for (R_xlen_t i = 0; i < new_data.size(); ++i) {
            std::string key = Rcpp::as<std::string>(names[i]);
            // Membership test: shared_data_t exposes data_input_keys() (a
            // std::set of the declared data-input keys); there is NO
            // is_data_input(key) member. Test membership with .count(key).
            if (impl_->data().data_input_keys().count(key) == 0u) {
                Rcpp::stop("<ClassName>::predict_at: unknown key '%s'. "
                           "Valid keys: <list>.", key.c_str());
            }
        }
        Rcpp::NumericMatrix X_new =
            Rcpp::as<Rcpp::NumericMatrix>(new_data["X"]);
        if (X_new.ncol() != static_cast<int>(p_)) {
            Rcpp::stop("<ClassName>::predict_at: X_new has %d cols, "
                       "training X has %zu.", X_new.ncol(), p_);
        }
        x_flat = arma::vec(X_new.begin(), X_new.nrow() * X_new.ncol());
    }

    if (!keep_history_) {
        // ---- Stateful mode: single predict at current draw --------
        block_context replaced;
        if (has_X) replaced["X"] = x_flat;
        auto result = impl_->predict_at(replaced, predict_rng_);
        Rcpp::List out;
        for (const auto& kv : result) {
            Rcpp::NumericVector v(kv.second.n_elem);
            for (std::size_t i = 0; i < kv.second.n_elem; ++i)
                v[i] = kv.second[i];
            out[kv.first] = v;
        }
        return out;
    }

    // ---- History mode: loop over all posterior draws --------------
    // **Critical:** `composite_block::predict_at` only accepts replaced
    // keys that are in `(data_inputs  U  block_names)`. Sub-outputs of
    // composite blocks (e.g. `beta` from a rjmcmc block named
    // `gamma_beta_rj`) are NOT block names -- `replaced["beta"] = ...`
    // would fail validation. The right pattern is to compute y_rep
    // MANUALLY per draw from the history, without going through
    // `impl_->predict_at`:
    AI4BayesCode::history_map hist = impl_->get_history();
    const arma::mat& beta_hist  = hist.at("beta");   // n_draws x p
    const arma::mat& sigma_hist = hist.at("sigma");  // n_draws x 1
    const std::size_t n_draws = beta_hist.n_rows;

    // X to use this round: training X by default, user's new X if given.
    const arma::vec& X_use = has_X ? x_flat : impl_->data().get("X");
    const std::size_t N_pred = X_use.n_elem / p_;

    Rcpp::NumericMatrix yrep_mat(n_draws, N_pred);
    std::normal_distribution<double> norm01(0.0, 1.0);
    for (std::size_t d = 0; d < n_draws; ++d) {
        const double sigma_d = sigma_hist(d, 0);
        for (std::size_t i = 0; i < N_pred; ++i) {
            double xb = 0.0;
            for (std::size_t j = 0; j < p_; ++j)
                xb += X_use[i + j * N_pred] * beta_hist(d, j);
            yrep_mat(d, i) = xb + sigma_d * norm01(predict_rng_);
        }
    }

    Rcpp::List out;
    out["y_rep"] = yrep_mat;
    return out;
}
```

**When `impl_->predict_at` IS usable in history mode**: only when every
sampled key the refresher reads is *also* the name of a child block
(e.g. `BetaBernoulli`'s "p" is both a sampled key and the block name).
In that case you can use the cleaner `replaced[key] = ...` pattern --
see `BetaBernoulli.cpp`, `DirichletSimplex.cpp`, `DirichletSparse.cpp`.

If any refresher-read key is a sub-output of a composite block (rjmcmc,
joint_nuts_block, etc.), fall back to the manual-compute pattern above
(SpikeSlabRJMCMC.cpp, ARDLasso.cpp).

The history-mode branch needs ALL the sampled keys that downstream
refreshers read. Enumerate each refresher's `d.get(...)` calls, pull
each key from `impl_->get_history()`, and set them in `replaced`
per draw. (Constants and `declare_data_input` keys are pulled from
`new_data` once, OUTSIDE the loop.)

Validate keys and dimensions. Throw with the valid-key list on error.
For BART children, call `bart_block::predict(X_new)` directly and
inject the result into the `replaced` context before forwarding.

#### Models WITHOUT covariates (no `declare_data_input` calls)

Use this template ONLY when the constructor has no
`declare_data_input(...)` call:

```cpp
Rcpp::List predict_at(Rcpp::List new_data) const {
    if (new_data.size() > 0) {
        Rcpp::stop("This model has no covariate inputs.");
    }
    return Rcpp::List::create();
}
```

If you find yourself emitting this template even though the
constructor declares a data input, STOP -- that's the
silent-broken-predict_at bug. Switch to the canonical wrapper above.

See `examples/BartNoise.cpp` for the canonical BART predict_at pattern,
`examples/GPRegression.cpp` for the GP + libgp kernel pattern, and
`examples/GPClassification.cpp` for the Bernoulli-logit variant.

**Important:** predict_at is `const` -- it must NOT modify MCMC state.

---

## 9. Backend module declarations (R / Python / C++)

The `.cpp` file's tail declares how the class is exposed to the host
language. The same class body can serve all three backends as long as
all user-facing accessors return **backend-neutral types** from
`AI4BayesCode::types.hpp`:

| Accessor | Return type |
|---|---|
| `get_current()`  | `AI4BayesCode::state_map` |
| `set_current()`  | takes `AI4BayesCode::state_map` |
| `get_history()`  | `AI4BayesCode::history_map` |
| `get_dag()`      | `AI4BayesCode::dag_info` |
| `predict_at()`   | returns `AI4BayesCode::state_map` or `AI4BayesCode::history_map` |
| `get_adaptation()` | `AI4BayesCode::adaptation_info` |

`Rcpp::wrap()` specializations (in `AI4BayesCode/rcpp_wrap.hpp`) and
pybind11 type casters (in `AI4BayesCode/pybind_casters.hpp`) convert these
to R lists / Python dicts automatically. Include the one that matches
your backend at the tail of the `.cpp`.

### Tail template -- choose based on Sec.1 Q0 "Runtime backend"

#### Conditional 7th method `readapt_NUTS` (NUTS-family composites only)

**Rule:** if the wrapper's composite contains at least one
`nuts_block` or `joint_nuts_block`
child, the wrapper MUST expose a 7th R-level method
`readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1)` and carry a
3rd RNG member `readapt_rng_`. If the composite has no NUTS-family
child (pure BART, pure Gibbs, pure VI, pure HMM, pure SBP, pure
RJMCMC, pure slice), the wrapper has exactly the 6 core methods
and 2 RNGs. See `system_design.md Sec.1` (conditional 7th method),
Sec.8 (3rd RNG), Sec.13 NUTS-family contract, and `validator.md Sec.24`.

**ALWAYS register the 3-arg form.** Use EXACTLY
`readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1)` (the code
template below). Some shipped examples show a shorter 2-arg
`readapt_NUTS(int n, bool reset = false)` -- do NOT copy that shorter form; the
3-arg signature is the current standard, and the R runner calls it with all
three arguments (Rcpp drops C++ defaults, so the caller passes all three).
`max_tree_depth = -1` = "use each block's configured depth" -- this is the value
the runner passes (`readapt_NUTS(N, FALSE, -1L)`) for essentially EVERY model, so
you do NOT need to think about tree depth. Tuning it to a positive value (faster
re-adaptation on a stiff / ill-conditioned target) is RARE and decided per-model
during generation; leave it at `-1L` unless a model specifically needs it.

**Wrapper-class additions (NUTS-using wrappers only):**

```cpp
// Add to constructor init list, after predict_rng_(...)
readapt_rng_(rng_seed == 0
         ? std::mt19937_64{std::random_device{}()}
         : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                           ^ 0xBF58476D1CE4E5B9ULL}),

// Add to private member declarations, after predict_rng_
mutable std::mt19937_64 readapt_rng_;   // readapt_NUTS only (7th method)

// Add to the wrapper's public method section (before private:)
/// 7th R-level method: re-tune NUTS metric (mass matrix + step size +
/// dual averaging) without advancing chain state. Available because
/// the composite contains NUTS-family children. See system_design.md
/// Sec.13 NUTS-family + validator.md Sec.24.
void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1) {
    if (n < 0) Rcpp::stop("readapt_NUTS: n must be non-negative");
    // max_tree_depth: -1 (default) = use each block's configured depth;
    // >0 = temporarily cap NUTS tree depth for these n adaptation iters
    // (faster re-adaptation on high-dim / ill-conditioned targets).
    impl_->readapt_NUTS(static_cast<std::size_t>(n), reset, readapt_rng_,
                        max_tree_depth > 0
                            ? static_cast<std::size_t>(max_tree_depth) : 0);
}
```

Mix constant `0xBF58476D1CE4E5B9ULL` (SplitMix64) MUST be the same
across all NUTS-using examples for consistency. Do not invent
alternates.

#### Emit BOTH module blocks -- the `.cpp` is ALWAYS dual-module

Always emit **both** the `RCPP_MODULE` block AND the `PYBIND11_MODULE` block
below, each under its own `#ifdef`, regardless of the runtime-backend answer
(codegen.md Sec.1). The same `.cpp` is then source-able in R
(`ai4bayescode_source` sets `-DAI4BAYESCODE_RCPP_MODULE`) AND in Python
(`AI4BayesCode.source` sets `-DAI4BAYESCODE_PYBIND_MODULE`) -- the backend
choice selects only the runner file, never which binding block(s) to write.

**`RCPP_MODULE` block (R -- active when `AI4BAYESCODE_RCPP_MODULE` is defined):**

```cpp
#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(<ClassName>_module) {
    Rcpp::class_<ClassName>("<ClassName>")
        .constructor<...args...>("<docstring>")
        // Rcpp modules IGNORE C++ default args, so expose step() and step(int) as
        // two arity-dispatched overloads -> both `m$step()` and `m$step(n)` work.
        .method("step", (void (ClassName::*)())    &ClassName::step)
        .method("step", (void (ClassName::*)(int)) &ClassName::step)
        .method("get_current",  &ClassName::get_current)
        .method("set_current",  &ClassName::set_current)
        .method("predict_at",   &ClassName::predict_at)
        .method("get_dag",      &ClassName::get_dag)
        .method("get_history",  &ClassName::get_history)
        // CONDITIONAL -- only emit the line below if the composite
        // contains any NUTS-family child (nuts_block /
        // joint_nuts_block).
        .method("readapt_NUTS", &ClassName::readapt_NUTS);
}
#endif
```

Include `AI4BayesCode/rcpp_wrap.hpp` alongside the other AI4BayesCode headers.

**`PYBIND11_MODULE` block (Python -- active when `AI4BAYESCODE_PYBIND_MODULE` is
defined). Emit this too, always -- right after the RCPP block:**

```cpp
#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"

PYBIND11_MODULE(<ClassName>, m) {
    AI4BayesCode::register_ai4bayescode_types(m);  // one-time DagInfo/AdaptationInfo bindings

    pybind11::class_<ClassName>(m, "<ClassName>")
        .def(pybind11::init<...args...>(),
             pybind11::arg("arg1"), pybind11::arg("arg2") = default_val, ...,
             "<docstring>")
        .def("step", (void (ClassName::*)())    &ClassName::step)          // model.step() = 1 sweep
        .def("step", (void (ClassName::*)(int)) &ClassName::step, pybind11::arg("n_steps"))
        .def("get_current",  &ClassName::get_current)
        .def("set_current",  &ClassName::set_current, pybind11::arg("params"))
        .def("predict_at",   &ClassName::predict_at, pybind11::arg("new_data"))
        .def("get_dag",      &ClassName::get_dag)
        .def("get_history",  &ClassName::get_history)
        // CONDITIONAL -- only emit if composite has NUTS-family child
        .def("readapt_NUTS", &ClassName::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false,
             pybind11::arg("max_tree_depth") = -1);
}
#endif
```

Python users invoke via `AI4BayesCode.source("MyModel.cpp")` (packaged; no path),
which sets `-DAI4BAYESCODE_PYBIND_MODULE` at compile time.

#### Dual-module (R + Python from the same .cpp -- the 6 shipped examples)

Guard both blocks; the active backend's `-D` define picks which one
compiles. See `examples/ODE_SIR.cpp` for a full dual-module reference.

#### C++ standalone (no host language -- testing or deployment)

Instead of a module block, emit a `main()` function at the end:

```cpp
int main(int argc, char** argv) {
    arma::vec y = load_from_csv(argv[1]);  // user-defined
    MyModel m(y, /*seed=*/42, /*keep_history=*/true);

    m.step(2000);  // warmup
    m.step(2000);  // keep

    auto hist = m.get_history();        // history_map
    auto dag  = m.get_dag();            // dag_info

    // Example: compute posterior mean of "mu"
    const auto& mu_hist = hist.at("mu");
    const double mu_pm = arma::mean(mu_hist.col(0));
    std::cout << "posterior mean(mu) = " << mu_pm << "\n";
    return 0;
}
```

Compile:
```
clang++ -std=c++17 -O2 \
    -I/path/to/AI4BayesCode/include \
    -I/path/to/AI4BayesCode/include/mcmclib \
    -I/path/to/AI4BayesCode/include/mcmclib/BaseMatrixOps/include \
    -DMCMC_ENABLE_ARMA_WRAPPERS -DARMA_DONT_USE_WRAPPER \
    MyModel.cpp -o MyModel -framework Accelerate
```

No RCPP_MODULE, no PYBIND11_MODULE, no Rcpp dependency at all. Useful
for benchmarking, embedding into a larger C++ program, or Docker
deployment.

### BART examples: R-only

BART blocks (`bart_block`, `genbart_block`) store data as
`Rcpp::NumericMatrix`/`NumericVector` internally because the vendored
CRAN BART R package / genBART kernels use these types directly, and call
into R's global RNG (via `arn`) for tree-structure proposals. A Python
port would require refactoring both kernels to use `arma::mat`/
`arma::vec` AND swapping `arn` for a portable RNG -- deferred for v1.
For now, all BART-using examples (BartNoise, SoftBartNoise, and the 4 GBart* wrappers)
ship with `AI4BAYESCODE_RCPP_MODULE` only. Python users who need
BART-like nonparametric mean functions should use `GPRegression` or
`GPClassification` as substitutes.

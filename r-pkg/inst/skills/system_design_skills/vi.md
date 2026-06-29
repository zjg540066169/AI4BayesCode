<!-- system_design MODULE — Variational Inference + rationale (extracted 2026-06-21 from system_design.md §18-§19).
     SINGLE LIVE SOURCE: edit HERE, not the monolith. system_design.md is now a thin index
     (§N -> module).
     Cross-refs keep the "§N" scheme, resolved via the system_design.md index. -->

## 18. Variational Inference — architectural extension

This section defines how the VI family fits the AI4BayesCode
invariants (§§1–17). It is the deep reference for the brief
family entry in §13's "VI-family" subsection and for
`skills/codegen.md`'s VI trigger question. Read it before
designing or modifying any `vi_block` subclass.

VI is OPT-IN. Default for every codegen-produced sampler is MCMC
(see `skills/codegen.md` §2 VI trigger question). VI is the
escape hatch for models MCMC handles poorly: high-dim posteriors
with explicit symmetries (BNN), tractable continuous parameters
with no useful conditional structure, anything where the user has
already accepted point-estimate-of-q semantics. When MCMC works,
prefer MCMC — VI's bias (§18.9) is not free.

### 18.1 Architectural insight — VI factor ≡ Gibbs block

Bishop §10.1 derives mean-field variational inference as the
coordinate-ascent fixed point of the factorized family
`q(Z) = ∏_i q_i(Z_i)`. The optimal q*_j satisfies

  ln q*_j(Z_j) = E_{i≠j}[ ln p(X, Z) ] + const          (Bishop 10.9)

This is structurally identical to a Gibbs block's full conditional:
both define a slice of the posterior conditional on the other
slices' current values. The difference is what gets MAINTAINED
between sweeps:

| | Gibbs block | VI block |
|---|---|---|
| State maintained | last sample θ̂ ∈ Θ | variational params λ = (μ, σ) or (μ, L) |
| `step(rng)` does | one sample from p(θ\|others) | one optimizer step on −ELBO(λ\|others) |
| `current()` returns | last sample | q-mean (point estimate) |
| What is written to shared_data | last sample | q-sample η~q then θ=T(η) |

The composite framework is **engine-agnostic by design**: a
`composite_block` containing only VI children IS standalone VI;
one containing a mix IS the hybrid mode the user picks via
codegen. No special "hybrid" framework code exists or is needed.

### 18.2 `block_sampler::engine_kind()` enum

`include/AI4BayesCode/block_sampler.hpp` adds (Phase 2):

```cpp
enum class engine_kind_t { MCMC, VI };
virtual engine_kind_t engine_kind() const = 0;  // override per family
```

All existing blocks (`nuts_block`, `joint_nuts_block(_mixed)`,
`bart_block`, `genbart_block`, `*_gibbs_block`, `rjmcmc_block`,
`hmm_block`, `stick_breaking_block`, `composite_block` over
all-MCMC children, etc.) override to return `MCMC`. The new
`vi_block` base returns `VI`. `composite_block::engine_kind()`
returns `MCMC` if all children are MCMC, `VI` if all are VI, and
`MCMC` (conservative default) if mixed (the hybrid case).

**The §1 six-method R-level contract is preserved unchanged**:
`engine_kind()` is C++ machinery, not Rcpp-exposed. R sees only
the same `step / get_current / set_current / predict_at /
get_dag / get_history`. The wrapper class chooses children at
construction time; outer R/Python code is identical for MCMC,
VI, and hybrid wrappers.

### 18.3 VI block contract — extension of §1 semantics

`vi_block` extends `block_sampler` with no public R-side changes
and the following Tier B additions:

```cpp
class vi_block : public block_sampler {
public:
  // === inherited from block_sampler ===
  void  set_context(const block_context&) override;
  void  step(std::mt19937_64& rng) override;          // one optimizer step
  const arma::vec& current() const override;          // q-mean
  void  set_current(const arma::vec& mu) override;    // overwrite μ only
  const std::string& name() const override;
  size_t dim() const override;
  engine_kind_t engine_kind() const override { return engine_kind_t::VI; }

  // === VI-specific extensions (callable from Tier A dispatcher) ===
  arma::vec current_sample(std::mt19937_64& rng) const;     // draw θ ~ q
  void  set_variational_state(const arma::vec& mu,
                              const arma::vec& log_sd);
  arma::vec get_log_sd() const;                              // current log σ
  double current_elbo() const;                               // last-step ELBO
  // history hook: pushes (elbo, μ, log σ, γ, epoch) when keep_history=true
  const vi_history_t& history() const;
};
```

Key contract differences from MCMC blocks:

- **`step(rng)`** runs ONE optimizer step (avgAdam from RAABBVI,
  §18.7), not a sampling step. The rng is used inside `step()`
  for the reparameterization draw ε ~ N(0, I) and (optionally)
  data subsampling.
- **`current()`** returns the q-mean E_q[θ] ≈ T(μ) (apply the
  natural-scale transform to the unconstrained mean μ). For most
  use cases this is what users want as a point estimate.
- **`set_current(μ)`** overwrites the variational mean ONLY; log
  σ is left intact. To overwrite both, Tier A dispatcher routes
  the per-component keys via `set_variational_state(...)`.
- **`current_sample(rng)`** is the new method invoked by the
  composite AFTER each VI step() to write a q-sample (not the
  q-mean) into shared_data. See §18.4 for the rationale.
- **`get_history()`** in Tier A wrapper, for any VI child, returns
  the ELBO + variational-parameter trajectory (per `step()` call,
  when `keep_history=true`). NOT posterior draws. See §18.7 for
  the storage shape.

### 18.4 shared_data semantics — VI writes q-sample, not q-mean

**The critical hybrid-mode invariant.** After every VI step, the
composite calls `child.current_sample(rng)` (NOT `current()`)
and writes that q-sample into shared_data under the child's
`name()`. MCMC siblings that read the key during their own sweep
see a fresh draw each outer iteration — a natural Monte Carlo
integration over q for the marginal-over-VI conditional they
target.

If a VI block wrote its q-MEAN to shared_data instead, MCMC
siblings would condition on a point estimate and **silently
underestimate** their own posterior variance whenever they depend
on the VI block's output. This is the hybrid-mode equivalent of
the §6 working-response correctness invariant: the value that
flows between blocks must encode the right thing.

**Pure-VI standalone mode**: same behavior — VI children write
q-samples to shared_data. R-level `get_current()` bypasses
shared_data (it's a separate aggregation in the Tier A wrapper
that calls each child's `current()` directly). This means R-level
`get_current()` is deterministic / reproducible for pure-VI
wrappers, even though shared_data carries q-samples for
inter-block reads.

| Read path | Returns |
|---|---|
| `ctx.at(name)` from a sibling block during sweep | q-sample (fresh per outer iter) |
| `m$get_current()` from R | q-mean (deterministic) |
| `m$predict_at(list(...))` from R | uses `predict_rng_` to draw q-samples (see §18.7) |

### 18.5 Variational family — mean-field PRIMARY, full-rank opt-in

**v1 primary: mean-field Gaussian on unconstrained R^K.**
Each user-declared parameter θ_i is wrapped via existing
`constraints::*::wrap`: the unconstrained η_i = T^{-1}(θ_i)
lives in R, and

  q(η) = ∏_i q(η_i) = ∏_i N(η_i; μ_i, σ_i²)

with variational parameters λ = {(μ_i, log σ_i)} (2K scalars for
a K-dim parameter). This is the Bishop §10.1.3 univariate-Gaussian
factorization extended over all latent parameters.

**v1 opt-in: full-rank Gaussian (SHIPPED 2026-05-26 via
`full_rank_gaussian_vi_block`).** Mean-field's diagonal q cannot
represent posterior off-diagonal covariance — for correlated targets
(regression with collinear predictors, BNN output-layer weights
conditional on hidden representations, hierarchical scale × parameter
funnels), the posterior marginal variance is collapsed to the
CONDITIONAL variance (Bishop §10.1.2). The full-rank block computes

  q(η) = N(η; μ, L L^T)        L lower triangular

with variational parameters λ = {μ, log L_diag, vec_lower_offdiag(L)}
(K + K(K+1)/2 scalars). Reparameterization gradient:
  η = μ + L ε,   ε ~ N(0, I)
  ∂(-ELBO)/∂μ_k       = -E_ε[(∂log p̃/∂η_k)]
  ∂(-ELBO)/∂L_ij (i>j) = -E_ε[(∂log p̃/∂η_i) ε_j]
  ∂(-ELBO)/∂(log L_jj) = -E_ε[(∂log p̃/∂η_j) ε_j · L_jj] - 1

**When to choose mean-field vs full-rank**:
- mean-field: posterior approximately factorizes → fast, fewer params
- full-rank: posterior has non-trivial off-diagonal covariance → use it
- For MIXED cases (some coordinates correlated, others independent),
  use TWO blocks in composite: one mean-field over the independent
  coords, one full-rank over the correlated subset

**Caps**: auto-suggest K ≤ 50; warn 50 < K ≤ 200; reject K > 500.
Beyond 500 dims the K(K+1)/2 Cholesky parameter cost (>125k entries)
makes the optimizer slow; consider splitting parameters or using
mean-field over the bulk.

**Validated** by `tests/test_vi_full_rank_correlated.cpp` on a 2D
ρ=0.95 bivariate Gaussian: mean-field collapses marginal sd to
√(1-ρ²)=0.31 (conditional variance), full-rank recovers marginal
sd ≈ 1.0 AND correlation ≈ 0.95, with PSIS-k̂ ≈ -0.14 (q ≈ p exact).
Same correctness signature on the BNN regression hyperparams
(σ_y/σ_α/σ_β reach R̂<1.20 vs NUTS-truth, vs mean-field's R̂>1.36
on σ_α and >1.51 on σ_β).

**No new Jacobian discipline (§10).** `constraints::*::wrap`
already handles the natural → unconstrained transform with
log|J|. The user-supplied log-density lambda lives on the
natural scale exactly as for `nuts_block`. Check #5 (no
hand-written `+ log(σ)`) applies unchanged.

### 18.6 Objective — exclusive KL, reparameterization gradient

**Default objective: exclusive KL** (ELBO maximization):

  −ELBO(λ) = E_{η~q(·;λ)}[ ln q(η; λ) − ln p̃(η) ]

where p̃(η) = p(θ=T(η), y) · |det dT/dη| is the natural-scale
joint density with the constraint Jacobian, both supplied by the
user's existing log-density lambda + `constraints::*::wrap`.
Dhaka et al. 2021 found exclusive KL dominates inclusive KL in
moderate-to-high-dim VI; no inclusive-KL variant is shipped in
v1.

**Gradient via reparameterization trick:** for mean-field Gaussian,

  η_i = μ_i + σ_i · ε_i,   ε_i ~ N(0, 1)
  ∇_μ_i  [−ELBO]      = − E_ε[ ∇_η_i ln p̃(η) ]
  ∇_logσ_i [−ELBO]    = − E_ε[ ε_i · σ_i · ∇_η_i ln p̃(η) ] − 1

(Kucukelbir et al. 2017, ADVI eqs. 4–6). The inner
∇_η_i ln p̃ is exactly the user's existing natural-scale gradient
that `nuts_block` consumes — **already Check #12-verified at
gen-time via autodiff**. NO new gradient infrastructure is
needed. Full-rank gets the analogous chain rule through L.

**Always PSIS-reweight expectations** (Dhaka 2021): when
computing E_q[f(θ)] for any posterior expectation (predictive,
LOO, BPV), draw S samples η_s ~ q, compute importance weights
w_s = p̃(η_s) / q(η_s), apply PSIS to stabilize w_s, then take
the weighted average. Layer-3 R3 and the `predict_at` stochastic
refresher path both implement this.

### 18.7 Optimizer — RAABBVI (Welandawe 2022)

**v1 ships RAABBVI as the default and only optimizer.** The
plain-ADVI step-size schedule of Kucukelbir 2017 (ELBO-window
heuristic) is known-brittle; RAABBVI is the principled
replacement. It combines:

1. **avgAdam**: like Adam, but the running squared-gradient
   denominator uses a cumulative average (not EMA), giving
   asymptotically SGD-like behavior; Polyak-Ruppert iterate
   averaging for the final λ̄_γ estimate.
2. **R̂-based convergence detection at fixed γ**: within each
   learning-rate epoch γ_k, treat iterates as a single chain and
   compute `R̂_max(W) < 1.1` for an adaptive window
   W ∈ [W_min, 0.95·k]. On detection, declare epoch convergence.
3. **Geometric γ decay**: γ_{k+1} = ρ · γ_k between epochs
   (default ρ = 0.5, initial γ_0 = 0.1).
4. **SKL inefficiency-index termination**: I = RSKL × RI < τ
   stops the entire optimization. RSKL = symmetric KL between
   successive averaged iterates λ̄_{γ_k} and λ̄_{γ_{k−1}} (closed
   form for Gaussians), RI = relative inefficiency (variance of
   λ̄_γ estimate vs Monte Carlo bound), default τ = 0.1.

Reference implementation: Python package `viabel` (Welandawe et
al.). v1 ports the algorithm to header-only C++ in
`include/AI4BayesCode/vi_optimizer.hpp` (Phase 3).

**get_history() shape for VI blocks** (per child):

```r
list(
  elbo   = numeric(n_steps),        # ELBO at each step
  mu     = matrix(n_steps × K),     # variational mean trajectory
  log_sd = matrix(n_steps × K),     # variational log-sd trajectory
  gamma  = numeric(n_steps),        # learning-rate at each step
  epoch  = integer(n_steps)         # which γ-epoch the step belongs to
)
```

These let R-side plotting do loss curves, per-epoch convergence
diagnostics, and the SKL termination trace.

### 18.8 Layer-3 diagnostic — PSIS-k̂ replaces R-hat for VI blocks

For MCMC blocks, Layer-3 R2 is "two chains, 4k+4k draws, R-hat
< 1.05 on every parameter" (see `skills/validator.md`). For VI
blocks this is not the right diagnostic — the variational q is a
single point in parameter space at convergence, not a
distribution over chains.

**v1 ships joint Pareto-Smoothed Importance Sampling k̂** (Yao
et al. 2018, Dhaka 2021) as the VI-block Layer-3 R2 substitute:

1. Draw S samples η_s ~ q (default S = 1000).
2. Compute log-importance-weights log w_s = log p̃(η_s) − log q(η_s).
3. Fit a generalized Pareto to the upper tail of w_s; the shape
   parameter k̂ is the diagnostic.
4. **Joint** (not marginal): k̂ is computed on the full η vector,
   not per parameter. Marginal k̂ misses joint failure modes
   (Yao et al. 2018).

**3-tier thresholds** (Yao 2018 + Dhaka 2021 confirmed):

| k̂ range | Verdict | Action |
|---|---|---|
| < 0.5 | PASS | report directly; PSIS-reweight optional |
| 0.5 – 0.7 | CAUTION | report with PSIS-reweighted expectations only; flag in summary |
| ≥ 0.7 | FAIL | retry up to 5 attempts per `start.md` failure-recovery policy: try full-rank on largest |k̂_marginal|, switch to non-centered reparam, increase optimizer epochs, drop τ to 0.05, init from multiple seeds and pick best ELBO |

**Hybrid mode Layer-3 R2**: split the diagnostic by child kind:
MCMC children get R-hat (< 1.05), VI children get PSIS-k̂
(< 0.7), plus an aggregated **ELBO trajectory plot** rendered
inline in the runner R script. There is intentionally no joint
hybrid convergence criterion in v1 (§18.10).

**Layer-3 R2-VI PSIS-k̂ diagnostic** (a Layer-3 runtime check, NOT a
numbered semantic check): PSIS-k̂ < 0.7 for every VI-family block.
Layer-3 fails fast with the action checklist above on k̂ ≥ 0.7. See
`validator.md §R2-VI` for the gating R code (3-tier verdict, CAUTION
warns + flags, FAIL stops).

### 18.9 Caveats — when to NOT use VI

Documented for the codegen agent to show during the VI trigger
question (`skills/codegen.md` §2). All from Bishop §10.1.2 /
Blei 2017:

1. **Mean-field underestimates posterior variance.** The q*
   minimizing exclusive KL is mode-seeking; it shrinks toward
   modal regions and ignores tails. If the user needs honest
   credible intervals on a low-dim parameter, MCMC is correct
   and VI is biased small.
2. **Multimodal posteriors get one mode.** Exclusive KL is
   mode-seeking, not mode-averaging. If the user model has
   genuine label-switching or symmetry-induced multimodality
   (BNN, mixture model component re-labeling, sign-symmetric
   embedding), VI returns ONE mode — usefully tractable for
   prediction, but not the full posterior. Document this in the
   trigger question.
3. **Label-switching differs from MCMC.** A VI run on a mixture
   model with no identifiability constraint converges to ONE
   component assignment; multiple seeds give different
   assignments. MCMC bounces between them (and produces useless
   marginals unless re-labeled). Both are diseases of the
   underlying non-identification; the symptoms look different.
4. **High dim ≠ automatic VI win.** Dhaka 2021 showed VI in
   high dim is fine WITH exclusive KL + reparam grad + RAABBVI
   + PSIS. Naive mean-field VI on a high-dim multi-mode posterior
   is bad either way. The v1 defaults (§18.6, §18.7) are the
   "works in high dim" recipe.
5. **Conditional structure matters.** For models where MCMC has
   a clean factorization (Gibbs blocks for everything, low
   between-block coupling), MCMC is fast and correct. Reaching
   for VI here is over-engineering and accepts unnecessary
   bias.

The codegen agent surfaces (1)–(3) in the VI trigger question
prompt and requires explicit confirmation before proceeding.

### 18.10 What is NOT in v1 scope

Deferred to v2+ (each individually a substantial project):

- **Normalizing-flow q families**. Dhaka 2021: NF is hard to
  optimize; few real-world wins over RAABBVI mean-field.
- **HMC-inside-q** (Salimans, Kingma, Welling 2015). Different
  use case (improve VI gradient estimate via short HMC chains);
  orthogonal to our hybrid.
- **MCMC-aware ELBO / nested SSVI** (Hoffman 2017). Different
  decomposition (global VI + local MCMC E-step over per-data
  latents); v1 hybrid is symmetric peer-block composition.
- **Conjugate VI blocks** (e.g., `dirichlet_vi_block`,
  `normal_normal_vi_block`). The unconstrained-Gaussian-family
  default covers most cases; conjugate VI is a future
  optimization for specific block families with closed-form
  natural-parameter updates.
- **Joint convergence criterion for hybrid.** v1 runs hybrid
  for a user-specified outer iteration budget and reports
  per-child diagnostics (R-hat for MCMC, k̂ for VI, ELBO
  trace). A joint criterion needs research.
- **Pathfinder warm-start for MCMC** (Zhang, Carpenter, Gelman,
  Vehtari 2022). Adjacent direction; not blocking.
- **Stein VI / SVGD, stochastic-gradient MCMC, and other
  exotica.** Not blocking.

### 18.11 Citations

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*,
  §10.1 (modular variational inference).
- Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei,
  D. M. (2017). Automatic Differentiation Variational Inference
  (ADVI). *JMLR* 18(14):1–45.
- Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Yes,
  but did it work?: Evaluating variational inference. *ICML*
  2018.
- Dhaka, A. K., Catalina, A., Welandawe, M., Andersen, M. R.,
  Huggins, J. H., & Vehtari, A. (2021). Challenges and
  opportunities in high-dimensional variational inference.
  *NeurIPS* 2021.
- Welandawe, M., Andersen, M. R., Vehtari, A., & Huggins, J. H.
  (2022). Robust, Automated, and Accurate Black-box Variational
  Inference. *JMLR* (RAABBVI).
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017).
  Variational Inference: A Review for Statisticians. *JASA*
  112(518):859–877.

---

## 19. Design rationale / history (why the architecture is shaped this way)

- **Stateful modules over pass-stateless.** Early prototype had
  step() return a fresh state dict; outer MCMC threaded state
  through every call. This made composition fragile (state
  shape changed per block) and expensive (serialize-deserialize
  each sweep). The stateful contract (each wrapper owns its
  internal sampler; outer MCMC just tells it what changed) won.

- **Six-method contract.** Tried exposing `$get_parameters`,
  `$set_parameter`, `$get_n_draws`, per-parameter `$set_<name>`,
  etc. Users forgot which method to call. The minimal uniform
  contract reduced the cognitive load and made outer
  composition code uniform.

- **set_current as Rcpp::List, not typed method.** Rcpp modules
  don't support C++ function overloading — multi-type
  dispatch needs a list / variant. Rcpp::List with key-based
  routing is the most ergonomic match.

- **Three-tier split.** Pre-2026-04-12, Tier A and Tier B were
  sometimes merged ("just expose the underlying kernel directly
  to R"). That broke the six-method contract and made every
  wrapper class different from R's perspective. The split is
  now strictly enforced.

- **get_dag explicit.** Alternative: infer the DAG from reads/
  writes at runtime. Rejected because (a) validator Layer 2
  needs the DAG statically at code review, and (b) predict_at
  needs it to know which derived keys to recompute.

- **Gibbs DAG vs Predict DAG split.** Single-DAG attempts conflated
  "who depends on whom for sampling" and "who directly produces
  whom generatively". They are different. Separated 2026-04-12.

- **Check #12 at gen-time only, not runtime.** Tried runtime
  autodiff gradient via autodiff.hpp; 97–562× slowdown killed
  the approach. Hand-written gradient + throwaway gen-time
  verify turned out to be the right engineering balance (2026-04).

- **BART set_current(X, y) support (2026-04-19).** See §15
  case study.

New system-design agents: read this history before proposing an
architectural change. Many obvious-looking alternatives have been
tried and rejected for specific reasons.

<!-- system_design MODULE -- block-family conventions (extracted 2026-06-21 from system_design.md Sec.13).
     SINGLE LIVE SOURCE: edit HERE, not the monolith. system_design.md is now a thin index
     (Sec.N -> module).
     Cross-refs keep the "Sec.N" scheme, resolved via the system_design.md index. -->

## 13. Block families -- design conventions

Each family has its own setter naming; keep within the family.

### Freeze semantics per family (kernel-control, interface.md Sec.1)

Every wrapper exposes `freeze(names)` / `unfreeze(names = <missing>)` /
`get_frozen()` as kernel-control methods (see interface.md Sec.1 amendment +
DESIGN_NOTES_FREEZE_UNFREEZE_2026-07-19.md for the full contract). The per-family
semantics of what "freezing" does are uniform in principle -- skip the child's
`step()` and hold its shared_data key at the last-set value -- but the
WHITELIST/BLACKLIST admission varies:

**WHITELIST (freeze supported):**

| Family | Freeze effect | Notes |
|---|---|---|
| `nuts_block` | `step()` skipped; kernel state (mass matrix, step size) preserved; `readapt_NUTS` also no-op on this child | canonical use case (probit / sigma-fixed extension) |
| `joint_nuts_block` | **v1: BOTH whole-block AND slot-level** (scope-bumped 2026-07-19). Passing the block's own name freezes all slots atomically; passing a slot name (sub-parameter in `slot_specs_`) freezes ONLY that slot -- the remaining slots continue to sample jointly, with the mass matrix re-projected onto the free-slot subspace (Welford covariance skips frozen rows/cols; leapfrog zeros momentum/gradient on frozen dims; constraint chain rule outputs are masked). This unblocks the probit / partial-fix extension pattern (freeze `log_sigma` slot inside `joint_nuts_block({beta, log_sigma})` to pin sigma at 1). Dim-changing slots (SIMPLEX / CHOLESKY_* / CORR_ / COV_) freeze via the `unc_offset` / `unc_dim` dual-offset scheme. See DESIGN_NOTES_FREEZE_UNFREEZE_2026-07-19.md Sec.10.a for the integrator + Welford + constraint chain rule details. | `readapt_NUTS` on a slot-frozen joint block still runs; mass-matrix re-adaptation skips frozen dims. Whole-block freeze -> `readapt_NUTS` also no-op. |
| `*_gibbs_block` (dirichlet / beta / binary / categorical / survival trio) | `step()` skipped; last draw held in shared_data | uniform; no derived-state hazard |
| `gmrf_precision_block` / `gmrf_whitened_ess_block` | `step()` skipped; latent `x` held at last draw | mild derived state via `Q_fn(ctx)` -- if a hyperparam feeding Q moves while `x` is frozen, the effective conditional on `x` still makes sense (Gibbs-condition-on-x variant) |
| `stick_breaking_block` / `normal_gamma_cluster_gibbs_block` | `step()` skipped; stick lengths / cluster params held | valid conditional; alpha etc. still sample |
| `rjmcmc_block` | **v1 supports BOTH whole-block AND sub-key** (sub-key scope-bumped 2026-07-19). Whole-block via `freeze("<name>")` atomically suppresses BOTH the trans-dim gamma sweep AND the `continuous_update` beta refresh. Sub-key via dot-path: `freeze("<name>.gamma")` freezes ONLY the gamma sweep (active betas continue to sample); `freeze("<name>.beta")` freezes ONLY the beta refresh (gamma sweep continues). Enables post-model-selection inference (fix active-set at MAP, keep sampling active betas). `dim()` compile-time-fixed at 2p keeps concatenation type-stable through any freeze/unfreeze. See DESIGN_NOTES Sec.6 + Sec.10.d. | The rjmcmc_block override on `freeze()` accepts local sub-keys `"gamma"` / `"beta"` (routed from composite dot-path resolver Sec.10.c). |

**BLACKLIST (freeze -> Rcpp::stop with reason):**

| Family | Reason |
|---|---|
| `bart_block` | `set_current(arma::vec)` is `Rcpp::stop` -- no invertibility. Freezing means "hold current forest" but user cannot inject a specific forest state. Additionally: frozen BART leaves `f_bart` in shared_data stale under any subsequent `set_current(X = X_new)`, causing downstream sigma NUTS to adapt on stale residuals. Silent bias. Canonical error string (matches DESIGN_NOTES Sec.6): `"freezing bart_block not supported (forest is non-invertible + derived-state hazard on subsequent set_current(X=...) updates); use predict_at for held-out prediction, drop the BART child from composite for ablation"`. |
| `genbart_block` | Same class as `bart_block`. Canonical error string (matches DESIGN_NOTES Sec.6): `"freezing genbart_block not supported (forest is non-invertible + derived-state hazard); same rationale as bart_block"`. |
| `hmm_block` | Latent state sequence z frozen while emission parameters sample yields mismatched conditioning (Baum-Welch forward pass depends on emissions). Silent bias. Canonical error string (matches DESIGN_NOTES Sec.6): `"freezing hmm_block not supported (latent state sequence conditioning breaks when emission parameters sample); model the HMM inside a two-composite pattern with the emission-freeze at the outer level"`. |
| `vi_block` subclasses | Sec.18.4 invariant: composite writes `current_sample(rng)` (fresh q-draw) to shared_data each step, NOT `current()` (q-mean). Freezing VI breaks the hybrid q-sample stream -> MCMC siblings silently underestimate posterior variance. Error string: `"freezing VI blocks not supported (breaks q-sample stream invariant); freeze the entire VI wrapper at composite level instead"`. |

**All FOUR error strings above contain the substring `"not supported"` verbatim** -- validator Check #26(b) uses `grepl("not supported", ...)` as a uniform blacklist-error test.

**Composite-of-composite (nested wrappers).** freeze/unfreeze accept
dot-path names for descending into nested composite children:
`m$freeze("outer_composite_child.inner_leaf_block")` walks
`composite_block` -> child (itself a composite) -> child's leaf. See
DESIGN_NOTES Sec.10.c for the exact resolution rules; get_frozen()
returns dot-path names for nested frozen children so round-trip via
`freeze(get_frozen(), quiet=TRUE)` works uniformly across depths.

**`quiet=TRUE` for batch refreeze.** `freeze(names, quiet=TRUE)`
suppresses the redundant-refreeze warning path (unknown-name /
blacklist errors still fire). Use for checkpoint restore where the
saved `frozen` set is expected to overlap the current frozen set.
See DESIGN_NOTES Sec.10.b.

Validator Check #26 tests presence + gate + refreeze warning + stale-derived warning.

### NUTS-family (continuous parameters)

`nuts_block` (single parameter, real or constrained) and
`joint_nuts_block(_mixed)` (concatenated sub-parameters). Tier B
setter: `set_current(arma::vec)` -- writes the unconstrained vector
and updates the kernel's internal state. Tier A wrapper's
`set_current(list(key = value))` routes `value` (arma::vec or
scalar) through the constraint's `unconstrain(...)` before
forwarding.

Data inputs for NUTS blocks: typically the response vector `y` and
design matrix `X`. Updates happen via shared_data only -- NUTS
kernels don't own X / y internally (they read from ctx). So no
special Tier B setter for X / y; just push via shared_data.

**Kernel-tuning method -- `readapt_NUTS(n, reset = false)`.** The 7th
R-level method on any wrapper whose composite contains at least
one NUTS-family child. Re-tunes the NUTS step size + dual-averaging
state WITHOUT advancing the chain state. **Mass matrix (dense
metric `precond_mat`) is preserved as-is** -- mcmclib's underlying
`nuts()` does not auto-adapt the mass matrix during the n internal
adaptation iterations; a separate Welford-based pilot (single-pilot
default, opt-in 3-phase for extreme cond -- see the "Stan-style 3-phase
warmup" subsection below) is the route for dense-metric adaptation.

**Use case.** Sequential / online update where new data shifts the
posterior geometry; the old metric is mis-tuned for the new target.
`readapt_NUTS(500)` re-tunes from the current state in 500 internal
iterations and restores state.

**Contract.**

- **State preserved.** `m$get_current()` before == `m$get_current()`
  after (bitwise on all sampled parameters). Implementation:
  snapshot shared_data + each block's current draw, run n internal
  step()s with metric adaptation ON, restore from snapshot.
- **Kernel updated.** Mass matrix + step size + dual-averaging
  counter retain their new values.
- **`reset = false` (default)**: Welford running covariance
  continues from previous state -- efficient for small target
  shifts (sequential update with informative previous metric).
  **`reset = true`**: re-initialize Welford and dual-averaging from
  scratch -- use for large target shifts or hot-start from far
  state where the previous metric is misleading.
- **RNG.** Uses `readapt_rng_`, the third RNG stream separate from
  `rng_` (MCMC) and `predict_rng_` (predict_at). See Sec.8. Same seed
  -> bit-identical readapt trajectory regardless of
  predict_at / step() history.
- **Composite dispatch.**
  `composite_block::readapt_NUTS(int n, bool reset, std::mt19937_64& rng)`
  iterates children; each block reports `supports_readapt()`. Non-
  NUTS-family blocks inherit default `supports_readapt() == false`
  and are silently skipped. NUTS-family blocks override to
  `true` and implement `readapt(n, reset, rng)`.
- **History.** The n ghost iterations are NOT recorded in
  `get_history()` -- they're internal and discarded along with state
  restore. Only `step()`'s on-record samples enter history.
- **Not a const method.** `readapt_NUTS` modifies kernel state and
  is therefore NOT marked `const`. Cannot be called from inside
  `predict_at`.
- **Not available on VI / BART / Gibbs / RJMCMC / HMM / SBP / Slice
  wrappers.** Those families return `supports_readapt() == false`.
  VI has its own RAABBVI optimizer that adapts step size on every
  `step()` -- no separate readapt API needed in v1. If a future use
  case requires it, design as `readapt_VI` (different semantics:
  variational params advance -- should NOT share method name with
  readapt_NUTS). Tracked in Sec.18.10 deferred-scope list.
- **Long-tail concentration / scale caveat.** For hierarchical
  models with heavy-tailed concentration parameters (Dirichlet
  alpha with Jeffreys prior, hierarchical SD tau, BNP mixture concentration),
  calling `readapt_NUTS` mid-chain may temporarily re-tune step
  size to a value that is suboptimal for exploring the parameter's
  long tail. The kernel remains CORRECT (detailed balance preserved
  w.r.t. the same target); only mixing efficiency on that specific
  parameter may be reduced for the next few hundred to thousand
  steps. Mitigation: allow ample post-readapt sampling steps
  (e.g., 2000+) before computing R-hat / ESS. This is an inherent
  property of dual-averaging step-size adaptation on heavy-tailed
  posteriors, not a readapt-specific issue. Validated empirically
  on `DirichletHierarchical` 2026-05-26 -- alpha concentration's R-hat
  reaches 44 with 1500-step post-readapt budget, but other 6 params
  stay R-hat ~= 1.00.
- **Hybrid-composite stale-derived-state caveat.** `readapt_NUTS`
  only runs on NUTS-family children (others are silently skipped
  per `supports_readapt() = false`). In **hybrid composites** that
  combine NUTS with BART / specialised Gibbs blocks whose outputs
  (`f_bart`, `f_softbart`, working residuals, etc.) are written to
  shared_data only inside the non-NUTS block's `step()`, a NUTS
  sibling's readapt adapter will read the STALE derived state if
  `set_current` changed upstream data without an intervening
  `step()`. The wrong residual distribution then produces a
  mis-tuned step size for the NUTS sibling.

  **Workflow rule per composite type:**

  | Composite | After `set_current` data change |
  |---|---|
  | Pure NUTS-family only | `readapt_NUTS()` immediately [OK] |
  | NUTS + BART / SoftBart hybrid | `step(1L)` first to refresh `f_bart`, THEN `readapt_NUTS()` |
  | NUTS + Gibbs that writes derived state in step() | `step(1L)` first, THEN `readapt_NUTS()` |
  | All-non-NUTS (pure Gibbs / pure BART / pure VI) | `readapt_NUTS()` is no-op anyway |

  Concrete example -- `BartNoise` composite: `set_current(list(X=X_new, y=y_new))`
  updates BART's internal X / y and shared_data's `bart_target`, but
  shared_data's `f_bart` stays at the last forest's prediction on OLD
  X. The sigma `nuts_block` sibling's readapt adapter reads `f_bart` ->
  uses stale residual `y - f_bart_stale`. Inserting one `step(1L)`
  between `set_current` and `readapt_NUTS` refreshes `f_bart`
  (BART's `step()` writes it back to shared_data), and sigma's readapt
  then adapts on the correct geometry.

  This caveat does NOT apply to BART/Gibbs blocks themselves (they
  never readapt). The kernel correctness of the NUTS sibling is
  preserved either way (detailed balance still holds); it's an
  EFFICIENCY caveat -- wrong step size = slower mixing -- not a bias
  one.

**Validator Check #23 (NEW)**: state-preservation grep + RNG
separation grep + R-level round-trip test. See
`validator.md Sec.23` for full audit template.

### GMRF-family -- non-Gaussian observation companion (SHIPPED 2026-06-03)

Companion to `gmrf_precision_block` (Gaussian conditional only):
`gmrf_whitened_ess_block` handles the non-Gaussian-likelihood case via
Murray 2010 Elliptical Slice Sampling on the implicit GMRF prior. The
two blocks cover the GMRF latent family completely:

| Block | Use case | Algorithm |
|---|---|---|
| `gmrf_precision_block` | GMRF latent + Gaussian observation (conditional is Gaussian) | Rue 2001 direct conjugate draw |
| `gmrf_whitened_ess_block` | GMRF latent + non-Gaussian observation (Poisson, Bernoulli, Student-t, ...) | ESS on implicit prior; user provides `log_lik(x, ctx)` |

**Example composite** -- sparse-precision latent + user-supplied
non-Gaussian observation likelihood:

```
gmrf_whitened_ess_block("x", Q_fn = [k](ctx) -> k * Q_base, sum_to_zero=true,
                              log_lik = <user's non-Gaussian log p(y | x, hyperparams)>)
+ nuts_block("<location intercept>", real)
+ nuts_block("<log precision>",      positive)
```

The ESS block handles the latent (the slow-mixing high-dim GMRF)
efficiently per Murray 2010 (acceptance rate independent of likelihood
scale); the NUTS blocks handle smooth scalar hyperparameters.

**Sum-to-zero invariant.** ESS rotation `x cos theta + nu sin theta` is linear,
so if both `x_cur` and `nu` are zero-mean, every proposal is also
zero-mean. The block enforces sum-to-zero only on `nu` post-sample
(via the same projection as `gmrf_precision_block`) and on
`x_initial` / `set_current(x)`; the inner shrink loop needs no extra
projection.

**Stateful semantics.** Identical to `gmrf_precision_block`:
`set_context` copies ctx; `step(rng)` advances `x_` one ESS sweep;
`current()` / `current_named_outputs()` expose latest `x_`;
`set_current(x)` overrides (re-projects under sum_to_zero);
`keep_history` appends per-step copy.

**Empirical correctness** (header verification, 2026-06-03):
- N=16 ICAR 4x4 grid, pure-prior recovery: sample variance / Q^{-1}
  diagonal = 0.998 (essentially exact).
- N=16 Poisson-ICAR, 4 chains x 2k draws: max R-hat 1.026,
  coverage 15/16 = 94%, sum-to-zero invariant preserved to
  9.7e-13.
- N=64 Poisson-ICAR, 4 chains x 10k draws: max R-hat 1.041,
  coverage 63/64 = 98%, 0.09s wall per chain.

**Performance.** Per-step cost is dominated by sparse Cholesky
factorization of Q (O(n * b_w^2) with b_w ~= O(sqrtn) for 2D lattice
GMRFs) plus an ESS shrink loop (typically 2-5 likelihood evaluations
per step at well-conditioned posteriors, independent of likelihood
scale per Murray 2010). At lattice sizes typical of spatial random-
effect applications, this is substantially faster than `joint_nuts_block`
over the full hierarchical parameter set, because joint NUTS on the
same geometry requires expensive leapfrog steps + dense-mass-matrix
inversion (per-step cost grows much faster in the latent dimension).
Pilot the budget for the user's specific Q topology and likelihood
family before committing to production.

**Validator Check #18.** GMRF-whitened ESS does NOT participate in
the dense-metric pilot machinery (it has no metric -- ESS is metric-
free); Check #18 is silent for `gmrf_whitened_ess_block`. The block's
correctness contract is the standard `block_sampler` interface plus
Q_fn callback signature; future v1.3 Check #25 could verify Q_fn
returns symmetric PSD sparse and log_lik returns finite scalar on
valid x.

### NUTS-family -- Stan-style 3-phase warmup (SHIPPED 2026-06-03)

**Metric + warmup decision (READ FIRST -- broad-corpus revision 2026-06-20).**
This is GUIDANCE, not a rigid spec. The robust strategy is ESCALATION DRIVEN BY
DIAGNOSTICS, mirroring Stan (default metric `diag_e`; it never auto-switches to
dense -- escalation is diagnostic/human-driven). Treat the table and the
condition-number figures below as STARTING POINTS and rough intuition, NOT hard
cutoffs. **If runtime diagnostics disagree with the heuristic, follow the
diagnostics and adjust** -- you (the agent) are expected to re-tune, not to obey a
number. The only firm, code-enforced invariant is the diagonal+3-phase ban.

**Escalation ladder (the actual decision procedure):**
1. START: **diagonal + single-pilot** -- cheapest, and sufficient for the common
   case (uncorrelated / weakly-correlated / merely scale-heterogeneous posteriors).
2. If runtime VALIDATION shows it is inadequate -- R-hat > 1.01, low ESS,
   max-tree-depth saturation, or low E-BFMI -- **try DENSE (still single-pilot).**
   Dense rotates a correlation ridge a diagonal metric cannot; this is the main
   escalation and resolves the large majority of correlated targets.
3. If dense + single-pilot ALSO fails to converge (rare -- extreme cond ~thousands,
   where one identity pilot can't bootstrap the covariance, or high-curvature
   ridge-trapping spatial/ICAR), **try dense + 3-phase**
   (`use_three_phase_warmup = true`) -- OR reparameterize (often the better fix).
4. If the geometry is NON-GAUSSIAN (funnel, centered-hierarchical, curvature), no
   metric helps -- reparameterize (below). The tell: it looks like a metric problem
   but does NOT improve under dense.

| posterior (starting heuristic) | metric | warmup |
|--------------------------------|--------|--------|
| uncorrelated / weakly-correlated / scale-heterogeneous | diagonal | single-pilot |
| strongly correlated (the common correlated case) | dense | single-pilot |
| dense single-pilot can't bootstrap (extreme cond / ridge-trapping) | dense | 3-phase |

**The one FIRM rule (code-enforced, not a heuristic): diagonal + 3-phase is
BANNED.** 3-phase gives step-size dual-averaging only ~1000 iters but our DA's
epsilon_bar averaging lag needs ~2500; on a diagonal (still ill-conditioned)
metric the step UNDERSHOOTS and mixing collapses (pl2: ~38x worse; 6-13x across
ar1/eqc/funnel). The gating enforces it (3-phase runs only when `use_dense_metric`).
Everything else here is advisory -- adjust on diagnostics.

WHY dense defaults to single-pilot (not 3-phase): on the broad 31-model corpus
dense+single-pilot gave 5-23x higher keep-phase ESS/s than dense+3-phase (3-phase's
short Phase III under-tunes the step via the same DA lag) and converged across all
realistic cond (corpus max 609, 5/5). 3-phase only wins at extreme synthetic cond
(8D equicorr rho=.999 cond~=8000: single-pilot 0/8 vs 3-phase 8/8) -- rare, usually a
reparameterization smell. So reach for 3-phase only when step 3 above demands it.
The metric-escape guard (reset to identity on a stiff-boundary warmup escape, e.g.
ARMA's |theta|>1 ridge) is automatic on every metric path.

**Non-Gaussian geometry is NOT a metric problem.** Funnels / centered-hierarchical
/ curvature do not converge under ANY metric (diag or dense); the fix is
REPARAMETERIZATION -- NCR (non-centered) for funnels/hierarchical, untwist for
curvature. Demonstrated 2026-06-20: centered funnel 2/5 -> NCR 5/5; centered hier_J30
0/5 -> NCR 5/5; twisted banana 2/5 -> untwist 5/5. See joint_nuts_failure.md Mode 1.

(`auto_select_metric` -- pilot estimates cond(R) [now with a Marchenko-Pastur noise
floor so high-dim uncorrelated targets are not wrongly picked dense], picks dense if
cond exceeds the floor-adjusted threshold -- is an opt-in EAGER-DENSE shortcut, NOT
the recommended default. Prefer the diagnostic-driven ladder above: a fixed cond
threshold PREDICTS; diagnostics MEASURE. Use auto only when you knowingly want eager
dense without a validation round-trip.)

**CODEGEN NOTE -- the warmup default now matches the policy (2026-06-20 flip).**
The config field `use_three_phase_warmup` now DEFAULTS TO `false` (single-pilot, the
recommended default), so setting `use_dense_metric = true` alone correctly gives the
FAST single-pilot path -- no extra config needed (the earlier "defaults true" foot-gun
is gone). Set `use_three_phase_warmup = true` ONLY for a documented extreme-cond
bootstrap failure (step 3 of the ladder). The 3 dense-using shipped examples (HSGP,
BSpline, HierarchicalLM_MultivariateRE) were re-validated under the new default and
all converge -- notably HSGP's 3-phase path was actually BROKEN (R-hat 1.017) and
single-pilot FIXED it (1.0015, 6x ESS). Full C++ suite green after the flip.

`joint_nuts_block` supports **two** dense-metric pilot modes (selected by
`use_three_phase_warmup`, default `false` = single-phase):

- **Single-phase** (DEFAULT + RECOMMENDED; the field defaults `false`; original v1.1
  behavior): identity-NUTS
  burn-in for `dense_metric_pilot_iters` iter, then collect
  `dense_metric_adapt_iters` samples, compute one Welford covariance,
  install as mass matrix. Re-warm with `n_warmup_first_call` under
  the dense metric (re-tune step size). Simple, FASTER in keep-phase
  (corpus: 5-23x vs three-phase) and converges across all realistic cond.

- **Three-phase** (OPT-IN extreme-cond escalation only -- slower in keep-phase; set
  `use_three_phase_warmup = true` only for a documented extreme-cond bootstrap
  failure): Stan-style windowed warmup (Carpenter et al. 2017, Stan ref manual Sec.16.2):
  - **Phase I** (default `tp_phase1_iters = 75`): step-size
    adaptation only, identity mass matrix. Lets the sampler find
    the typical set before any mass-matrix estimation.
  - **Phase II** (default `tp_phase2_windows = {25, 50, 100, 200,
    500}`, total 875): expanding windows of mass-matrix adaptation.
    Each window runs sampler under the previous window's mass
    matrix, computes Welford covariance across the whole window,
    installs regularized precision (Stan-style shrinkage
    `Sigma_reg = (n/(n+5))*Sigma + 1e-3*(5/(n+5))*I`) as the new mass
    matrix. The recursive self-correction -- window k uses M from
    window k-1 which let the chain explore more broadly than
    window k-1's biased start -- fixes ICAR-style ridge trapping.
  - **Phase III** (default `tp_phase3_iters = 50`): final
    step-size tune with frozen mass matrix. **("Frozen" here is
    the internal numerical term for a fixed mass matrix -- NOT to be
    confused with the user-facing `m$freeze()` kernel-control method
    in interface.md Sec.1.)**

  After all three phases, `first_call_` is set to FALSE so the main
  step path runs no additional warmup; sampling begins immediately.

**Use case.** High-curvature targets (sparse spatial / temporal
random effects on large grids, hierarchical funnels) where single-
phase pilot's Welford covariance is biased by the initial transient
trapping on a low-dimensional ridge.

**Default (2026-06-20 flip).** `use_three_phase_warmup` defaults to `false`:
`use_dense_metric = true` blocks use the SINGLE-PILOT path by default (the
recommended, faster choice -- see the decision rule at the top of this section and
the CODEGEN NOTE above). 3-phase is OPT-IN for extreme-cond bootstrap failures
only; set `cfg.use_three_phase_warmup = true` for those. DIAGONAL blocks ignore
this flag and always use the single pilot (3-phase is gated to dense-only;
diagonal + 3-phase is banned). All `nuts_block` blocks (no dense path) are
unaffected.

**Compatibility with `readapt_NUTS`.** Composes naturally -- Phases
I and III reuse the same step-size dual-averaging primitive that
`readapt_NUTS` exercises (the `n_adapt_draws` path on mcmclib's
`nuts()`); Phase II adds a windowed-Welford primitive at the same
architectural level. No conceptual machinery beyond what
`readapt_NUTS` already demonstrates is achievable in the stateful
API.

**Stateful semantics.** Cumulative-iter-driven; user-facing
`step(n)` still returns one keep draw per call. The first `step()`
call when `use_three_phase_warmup` is on runs the full 1000-iter
warmup schedule internally; subsequent calls are pure sampling.
Mid-3-phase `get_current()` / `predict_at` are not exposed (the
schedule runs atomically inside the first `step()`).

**Empirical motivation.** Backwards-compat verified on a small
spatial random-effect fixture (single joint NUTS, dense metric):
single-phase warmup shows the expected dense-metric burn-in cost;
three-phase warmup reduces wall-clock by roughly 2x on the same
fixture while preserving posterior identity (cross-mode R-hat < 1.01,
output finite, posterior state plausible). On harder fixtures where
single-phase pilot's Welford covariance is dominated by initial-
transient ridge geometry, windowed self-correction is necessary to
reach R-hat < 1.05 at production budgets; pilot the new feature on
the user's own target before committing.

**Limitations.** Three-phase is implemented for `joint_nuts_block`
only (the family that exposes `use_dense_metric`). The single-block
`nuts_block` does not have a dense-metric path and is unaffected.
Cross-block 3-phase scheduling for composite_blocks containing
multiple independent `nuts_block` children is future work; for
such composites, the per-block default single-NUTS adaptation
already runs independently.

### Gibbs-family (closed-form conjugate draws)

`*_gibbs_block` headers (dirichlet, beta, binary, categorical). No
gradient, no NUTS. Tier B setter: `set_current(arma::vec)` writes
the current draw. Tier A dispatcher routes `list(key = value)` to
the block's `set_current`.

Data inputs: whatever the block's params_fn / log_odds_fn /
alpha_post_fn / log_probs_fn reads from ctx. Refresh via
shared_data.

### BART-family (tree ensembles)

`bart_block` (Gaussian BART, CRAN BART R package backfitting) and `genbart_block`
(Linero 2022 RJMCMC with pluggable likelihoods). **Selection: `bart_block` is the
default for any real-valued response with constant noise (conjugate leaves, fast);
`genbart_block`'s RJMCMC is much slower, use it ONLY for non-Gaussian likelihoods
`bart_block` cannot express. Additive / varying-coefficient ensembles (VC-BART)
stay in `bart_block` via backfitting + `weights_key` -- see the `bart_block`
catalogue card.** Tier B setters:
- `bart_block`: `set_X`, `set_Y`, `set_data`.
- `genbart_block`: `set_X`, `set_Y`, `set_offset`, `set_data`.
`set_current(arma::vec)` is `Rcpp::stop` on both -- the tree forest
has no unique inverse from fitted values. Tier A dispatcher's
`set_current(list(...))` routes X / y / offset keys into these
C++-only setters. Rejected keys: `f_bart` for `bart_block`; `r`
(or `r_j` for multinomial) for `genbart_block`.

Case study: see Sec.15 (BART case study, lifecycle.md) for the full fix history (originally
written for `bart_block`; the same pattern applies to `genbart_block`
which uses `(Y, offset)` keys).

**BART-family carve-out to interface.md Sec.1.** BART-family wrappers
historically expose three tree-serialization methods NOT in the core-six
+ kernel-control formula: `get_tree` / `set_tree` / `get_tree_history`.
These are BART-specific (non-BART wrappers must not expose them) and
are formally acknowledged as a carve-out in interface.md Sec.1 so the
grep-audit does not flag them as violations. They are not part of the
kernel-control category and not the target of Check #26.

**BART freeze blacklist.** BART-family blocks are on the freeze
blacklist -- `m$freeze("<bart_block_name>")` raises `Rcpp::stop`. See
the "Freeze semantics per family" table at the top of Sec.13 for
rationale (non-invertible forest + stale derived state on subsequent
`set_current(X = X_new)`). Sibling `nuts_block` children in a BART
composite (e.g., `sigma`) can still be frozen normally.

### Joint-NUTS-family

`joint_nuts_block` (per-slice constraints: REAL / POSITIVE / SIMPLEX
/ CHOLESKY_CORR / COV_MATRIX / ...). Tier B setter: `set_current(arma::vec)`
writes the concatenated unconstrained draw. Tier A dispatcher
routes `list(joint_name = value)` OR per-sub-param keys (the
composite auto-splits on write).

Validator Check #11 (joint-slice alignment) is mandatory for any
joint-family block.

### VI-family (variational inference blocks)

`mean_field_gaussian_vi_block` (primary v1: factorized univariate
Gaussian over all sub-parameters) and `full_rank_gaussian_vi_block`
(v1 opt-in: joint Gaussian via Cholesky LL^T over a user-named
subset). Both derive from `vi_block`, which derives from
`block_sampler` with `engine_kind() == VI` (NUTS / Gibbs / BART /
RJMCMC / HMM / SBP all return `MCMC`). Tier B setter:
`set_current(arma::vec)` writes the variational mean mu on the
unconstrained scale; log sigma (mean-field) or vec(L) (full-rank)
keeps its current value. Tier A dispatcher routes per-component
keys `list(name_mean = mu_vec, name_log_sd = ls_vec)` (full-state
overwrite is rare -- see Sec.18.3).

Data inputs and the natural-scale log-density lambda are
identical to `nuts_block`: shared via shared_data, transformed
via `constraints::*::wrap`, NO hand-written Jacobian (Check #5
unchanged). `step(rng)` is ONE RAABBVI optimizer step (Welandawe
2022); the rng is used for the reparameterization draw and any
optional data subsampling. After step(), the composite writes a
q-SAMPLE (eta ~ q, theta = T(eta)) -- NOT the q-mean -- into shared_data
under the block's `name()`. This propagates the VI block's
posterior uncertainty to any MCMC sibling reading it in hybrid
mode. The R-level `get_current()` returns the q-mean (point
estimate), bypassing shared_data.

`get_history()` records per-step ELBO + variational-parameter
trajectory (for the loss curve), NOT posterior draws.

Validator **Check #21 (VI block contract conformance) + Check #22
(VI optimizer = RAABBVI)** plus **the Layer-3 R2-VI PSIS-k-hat
diagnostic (k-hat < 0.7)** are all mandatory before ship -- see Sec.18 for
the diagnostic and the full VI architecture.

### Survival-family -- lifetime / time-to-event primitives (SHIPPED 2026-07-10)

Three composable Gibbs blocks for piecewise-exponential-hazard (PEH)
survival models with right / interval censoring and shared frailty.
All three share the Gibbs-family convention (Tier B `set_current(arma::vec)`
writes current draw; Tier A dispatcher routes `list(key = value)`); no
gradient, no NUTS.

| Block | Role | Algorithm |
|---|---|---|
| `piecewise_exponential_gibbs_block` | log-hazard rates lambda_k on user break-points | Gamma-Poisson conjugate (Kalbfleisch 1978; Ibrahim/Chen/Sinha 2001 Sec.3.2) |
| `interval_censored_survival_augmentation_block` | event-time augmentation for interval-censored subjects | Tanner-Wong 1987, exact inverse-CDF truncated PEH (Sinha-Chen-Ghosh 1999; Lin-Wang 2010) |
| `frailty_gamma_gibbs_block` | per-group multiplicative frailty w_g > 0 | Gamma-Gamma conjugate (Clayton 1991; Ibrahim Sec.4.3) |

Reference compositions: `examples/PehSurvival.cpp` (right-censored)
and `examples/PehSharedFrailty.cpp` (right + interval + shared frailty).
Each block verified via `tests_autodiff/block_tests/test_*_block.cpp`
with hand-computed fixtures, sample mean / variance vs analytic within
5% / 15% tolerance.

---


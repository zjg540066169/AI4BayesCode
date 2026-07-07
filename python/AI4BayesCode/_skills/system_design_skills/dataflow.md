<!-- system_design MODULE -- data flow + wiring (extracted 2026-06-21 from system_design.md Sec.3-Sec.9).
     SINGLE LIVE SOURCE: edit HERE, not the monolith. system_design.md is now a thin index
     (Sec.N -> module).
     Cross-refs keep the "Sec.N" scheme, resolved via the system_design.md index. -->

## 3. shared_data + block_context -- mental model

`shared_data_t` is the composite's in-memory key-value store, shared
across all child blocks. It holds:
- **parameters** that blocks sample. For single-parameter blocks the
  block's `name()` IS the data key it writes (`set_current` writes to
  `data().values_[name]`). For joint blocks (`joint_nuts_block`)
  the block writes multiple sub-parameter
  keys via `current_named_outputs()`; the block's own `name()` is a
  label used by `declare_dependencies` / `declare_invalidates` only,
  not a data key.
- **data inputs** that outer MCMC may refresh via predict_at or
  set_current (keys declared via `declare_data_input(key)`);
- **derived / intermediate quantities** maintained by registered
  refreshers (e.g. `bart_target = y - X_other * beta_other`);
- **hyperparameters / constants** (e.g. `alpha_prior`, `nu`, `N`, `K`).

`block_context` is the read-only view of shared_data that
`set_context(ctx)` passes to each block per sweep. A block reads
`ctx.at(key)` to fetch whatever it needs (declared via
`declare_dependencies(block_name, {keys...})`).

**Invariant -- shared_data must stay in sync.** After any change via
`set_current`, any entry the composite advertises in shared_data
(especially data inputs and derived quantities) MUST reflect the new
state BEFORE the next `step()`. A Tier A dispatcher that calls
`bart_blk->set_X(X_new)` but forgets to also
`impl_->data().set("X", new_flat_vec)` creates a silent inconsistency:
DAG traversal and predict_at will see the OLD X.

---

## 4. Gibbs DAG vs Predict DAG

These are **two separate DAGs**. Confusing them is a common bug.

### Gibbs DAG -- sampling dependencies

Edges declared via `declare_dependencies(block_name, {keys read})`
and `declare_invalidates(block_name, {keys whose values become stale})`.
Read-direction: "block X reads keys K". Used by the composite to
build `block_context` for each block's sweep.

### Predict DAG -- generative / causal dependencies

Edges declared via `declare_predict_edges(producer, {consumers})`.
Read-direction: "producer DIRECTLY produces consumer". Used by
`predict_at` to do BFS from replaced data inputs forward to derived
quantities (y_rep, y_hat, f_bart, ...). Only direct edges, no
skipping -- BFS inserts a node only if ALL of its parents in the
predict DAG are available.

Example (BartNoise):
- Gibbs DAG: `f_bart` reads `bart_target`; `sigma` reads `y, f_bart`
- Predict DAG: `X -> f_bart`, `f_bart -> y_rep`, `sigma -> y_rep`

If you add a new derived quantity (e.g. `linear_predictor`), declare
BOTH its Gibbs invalidation (which parameter updates stale it) AND
its predict-DAG edge (which data input produces it). Missing either
silently breaks Layer 3 validator checks.

### Generative DAG -- context edges (VIZ-ONLY, third edge-set)

Edges declared via `declare_context_edges(from, {to})`. These encode
the **prior / hyperprior** structure: a sampled parameter's prior
parents (e.g. `sigma_nu -> sigma` for `sigma^2 ~ IG(nu/2, nu*lambda/2)`),
or a sampled "forest/kernel" parameter feeding a deterministic mean
node (`BART -> f_bart`, `amplitude -> K_matrix`).

**These edges are NEVER traversed by predict_at's BFS.**
`predict_downstream_of` / `predict_stochastic_sampleable` in
shared_data.hpp do not read `context_edges_`. They exist solely so
`get_dag()` (R: `ai4bayescode_plot_dag`) can render the full **generative DAG**:
the solid predict sub-DAG (exactly the set `predict_at` recomputes)
plus the faded prior/hyperprior context.

Hard rule: a context edge `hyperparam -> param` MUST NEVER also appear
in `predict_edges_`. Posterior prediction conditions on the MCMC
draws of the parameters, not on re-sampling their priors -- so adding
the prior edge to the predict DAG would wrongly make the BFS
recompute the parameter when the hyperparam is replaced. Keep the two
maps disjoint. Zero correctness risk by construction (viz-only); the
only failure mode is a misleading paper figure, caught by the
ai4bayescode_plot_dag visual check and validator Check #6.

Topology is **read from the model's own prior code, not invented**:
if the wrapper sets a slot `a_smooth` and the log-density uses it as
the IG shape of `sigma_rw2`, the context edge is
`a_smooth -> log_sigma_rw2`. Examples whose priors are hardcoded
constants with no named slot (e.g. `mu ~ N(0, 100^2)`,
`sigma ~ Jeffreys`) correctly have NO context edges -- the generative
DAG then equals the predict DAG.

Gold standard: MetaRegBartSpline.cpp. Canonical pattern across the
shipped examples: hyperprior-slot -> sampled-param, plus
forest/kernel-param -> deterministic-mean-node.

---

## 5. Refreshers -- deterministic vs stochastic

Refreshers are functions registered with shared_data that recompute a
derived key when upstream keys change.

### Deterministic refresher

`register_refresher(key, fn)`: pure function of current shared_data.
Called automatically during set_context when any upstream key is
invalidated. Example: `bart_target = y - X_other * beta_other`.

### Stochastic refresher

`register_stochastic_refresher(key, fn)`: takes an additional
`std::mt19937_64&` so it can sample. Called ONLY by `predict_at`
(never during a regular MCMC step -- that's the sampler's job).
Example: `y_rep ~ N(X * beta + alpha, sigma^2)`.

**Invariant -- every wrapper that expects Layer 3 R3 (posterior-
predictive p-values + LOO) to run MUST register a stochastic
refresher for `y_rep`.** Documented exceptions:
- `SpikeSlabSinhBijection` -- a pedagogical TOY (single coefficient;
  sigma/pi/slab_sd all fixed) demonstrating the sinh RJMCMC
  bijection. It has NO R binding and NO predict DAG by design;
  ratified as a documented exception (a 1-coef predictive has no
  scientific value). Do NOT replicate this in real samplers.
- All shipped `GBart*` wrappers register y_rep refreshers
  appropriate to their observation family.

**Invariant -- R3.b PSIS-LOO is DIAGNOSTIC ONLY, never pass/fail.**
Sampler-correctness gates are R2 (R-hat, ESS) and R3.a (Bayesian
p-values, `stopifnot`-gated). GP latent-variable and many
hierarchical models with informative per-observation latents
systematically fail the (k<0.5: >=50%, k>=0.7: <10%) targets even with
correctly sampled posteriors -- Pareto-$\hat{k}$ reflects how strongly
each leave-one-out posterior differs from the full-data posterior,
which is a property of the model class, not a sampler bug. The
R-runner template MUST emit `warning()`, never `stopifnot()`, on the
Pareto-$\hat{k}$ thresholds. See Vehtari, Simpson, Gelman, Yao, Gabry
-- *Pareto Smoothed Importance Sampling*, JMLR 2024 (arXiv:1507.02646),
esp. Sec.1 and the GP / random-effects examples. Operational rules:
`validator.md` "R3.b -- DIAGNOSTIC ONLY" subsection and
`codegen_r_runner.md` Sec.10 R3 emission.

**History-mode shadow carve-out (Q5 Ruling A, 2026-05-15).**
`keep_history=TRUE` `predict_at` MAY use a manual per-draw loop ONLY
when the per-draw quantities are tree forests (bart/genbart/softbart)
or `joint_nuts_block` sub-params, which the framework's
replaced-key validation (data_inputs  U  block names) structurally
cannot accept. The loop MUST use the SAME generative formula as the
registered refreshers (verified by an equivalence / R-reference
gate). The STATEFUL (`keep_history=FALSE`) path MUST still route
through `impl_->predict_at` -- no shadow there -- and `ai4bayescode_plot_dag` must
still reflect the fully reconstructed predict DAG. Gold standard
MetaRegBartSpline is conformant under this carve-out. See
validator.md Check #6.

**Delivered-validation-harness opt-out (CORRECTNESS INVARIANT).**
A pre-generation question (`codegen.md` "Delivered-code validation
harness") lets the user choose whether the Layer-3 harness ships in
the delivered code; **default = No (deliver a minimal
`example_<ClassName>.R`)**. This NEVER affects whether validation
runs: the validator (Layer-3 R1/R2/R3 + Checks #1-20) ALWAYS runs and
must PASS during generation regardless of the answer -- the harness is
a throwaway gating tool by default (same rule as the smoke test /
the Check #12 autodiff-verify file, jacobian.md Sec.10.1), retained in `run_<ClassName>.R` only if
the user opts in. "default: no harness" != "skip validation". The
chain helper is always named `run_chain_<ClassName>()`
(model-specific, never bare `run_chain`).

---

## 6. Working-response / data-injection pattern

Any block whose log-density / kernel uses an "input vector" (partial
residual, working response, sufficient statistic, imputed covariate
matrix) MUST support injection via TWO paths, and they must agree:

### Path A -- shared_data + `declare_dependencies` + `set_context`

The canonical in-sweep path. An outer block writes to a key (e.g.
`bart_target`), declares invalidation on upstream parameter
updates, and the inner block's `set_context` reads that key on
every sweep.

Used for working residuals, NB/ZIP u & v, etc.

### Path B -- set_current(list(...)) dispatcher

The R-user / outer-process path. The Tier A wrapper's set_current
routes a key (e.g. `y` for bart_block) into Tier B's direct setter
AND updates the corresponding shared_data entry so Path A stays
consistent.

Both paths must work. An inner MCMC driven from C++ will use Path A;
an outer MCMC driven from R will use Path B. They converge on the
same kernel state.

---

## 7. set_current(Rcpp::List) -- dispatcher contract

Every user-facing wrapper implements `set_current` as a pure
dispatcher. The body is a sequence of `if
(params.containsElementNamed("key"))` branches. The contract:

1. **Accept any subset of the supported keys in a single call.**
   `set_current(list(X = X_new, y = r, sigma = s))` runs all three
   atomically.

2. **Silently ignore unknown keys.** Callers often round-trip
   `set_current(get_current())`, which carries every tracked scalar
   / vector. Don't `Rcpp::stop` on unknown keys -- that breaks the
   round-trip.

3. **Reject ONLY keys that would be semantically impossible to
   overwrite.** Examples:
   - `f_bart` / `log_f` -- tree forest has no unique inverse from
     fitted values.
   - any output of a deterministic refresher -- caller should
     update the UPSTREAM key instead.
   Error message must be clear and point at the valid keys.

4. **Validate type and dimension** for every accepted key; emit
   `Rcpp::stop("...")` with a precise mismatch message ("row count
   50 does not match current n = 100"). Do NOT let bad input
   propagate into the kernel.

5. **Keep shared_data in sync** after the Tier B call (see Sec.3).

6. **Refresh cached Tier A state** (fit vectors, dimension caches)
   atomically with the Tier B update, so a subsequent
   `get_current()` (WITHOUT an intervening step) returns values
   consistent with the new input -- no stale-read bug.

7. **Never call `step()` from inside `set_current`.** set_current
   is pure rebinding; the outer sampler decides when to advance.

8. **Document supported / rejected / ignored keys** in a leading
   comment block on the method. The codegen agent reads these
   comments when generating derived wrappers.

---

## 8. RNG discipline

Every wrapper class MUST have TWO distinct RNG members, plus a
THIRD when the wrapper exposes `readapt_NUTS` (i.e., contains any
NUTS-family child), plus an additional BART-specific rule (see below):

```cpp
std::mt19937_64         rng_;          // MCMC; ADVANCED by step()
mutable std::mt19937_64 predict_rng_;  // predict_at only
mutable std::mt19937_64 readapt_rng_;  // readapt_NUTS only;
                                       // present ONLY on NUTS-family wrappers
```

Seeded once in the constructor:

| RNG | Seed |
|---|---|
| `rng_` | `rng_seed` |
| `predict_rng_` | `rng_seed ^ 0x9E3779B97F4A7C15ULL` |
| `readapt_rng_` | `rng_seed ^ 0xBF58476D1CE4E5B9ULL` (SplitMix64 mix constant; distinct from predict_rng_'s constant) |

The `mutable` qualifier lets `const` methods (`predict_at`)
advance their RNG. `readapt_rng_` is also `mutable` for symmetry,
though `readapt_NUTS` is NOT a const method (it modifies kernel
state -- see Sec.13).

### Why three RNGs (when NUTS-family is present)

- Sharing `rng_` with `predict_at` (called many times for posterior
  predictive diagnostics) advances MCMC's trajectory, destroying
  reproducibility of `step()`. It also violates `predict_at`'s
  documented contract of being pure w.r.t. MCMC state.
- Sharing `rng_` (or `predict_rng_`) with `readapt_NUTS` similarly
  contaminates either MCMC sampling or predictive diagnostics.
  `readapt_NUTS(n)` runs n internal adaptation iterations that
  consume many random draws; these must NOT bleed into either of
  the other two streams.
- Three streams keep all three operations independently
  reproducible from `rng_seed`: same seed -> bit-identical
  `step()` trajectory regardless of how many `predict_at` /
  `readapt_NUTS` calls have been made.

### BART-family additional rule

BART uses R's RNG via `arn` (bart_pure_cpp/src/BART/rn.h). The mt19937 argument
to bart_block::step is IGNORED. Reproducibility requires R-level
`set.seed()` before the MCMC loop AND the two-or-three mt19937
streams for the NON-BART siblings.

### Validator checks

- **Check #13**: any wrapper with a stochastic refresher must be
  audited for `rng_` / `predict_rng_` separation. Silent
  correctness bug if the two streams are conflated.
- **Check #23** (NEW): any wrapper exposing `readapt_NUTS` must be
  audited for `readapt_rng_` separation (not aliased to `rng_` or
  `predict_rng_`) + state preservation across the call. See
  `validator.md Sec.23`.

---

## 9. History / keep_history / serialization

Every wrapper accepts `keep_history = false` (default) or `true` via
the constructor. When true:
- every `step()` call pushes the current state into a per-block
  history buffer,
- BART-family blocks ALSO serialize the live tree ensemble
  (`serialize_live_trees()`) so predict_at can replay per-draw
  predictions;
- memory grows ~linearly with iteration count -- document this in
  the wrapper header so users don't OOM themselves.

`get_history()` returns an aggregated `Rcpp::List` with one entry
per block (scalar history -> `NumericVector` of length n_draws,
vector history -> `NumericMatrix` of (n_draws x dim)).

### Invariants

- `keep_history` is set ONCE at construction; there is no runtime
  toggle. Adding one would require careful burnin semantics.
- When `keep_history = true`, `predict_at` on a BART-family wrapper
  can return EITHER the current-state prediction OR the full
  (n_draws x n_test) matrix over history -- document which in the
  wrapper's header.
- History buffers are NEVER cleared by user calls. If a test needs
  to drop them, reconstruct the wrapper.

---


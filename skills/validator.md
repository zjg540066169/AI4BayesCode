---
name: AI4BayesCode-validator
description: |
  Post-generation validation checklist for AI4BayesCode samplers. Run this
  after the codegen skill produces a .cpp file to catch common silent bugs.
  The validator reads the generated code and checks each item below.
---

# AI4BayesCode code validator

> **HEADER PATCH IN FLIGHT (2026-07-19) -- Check #26 is FORWARD-LOOKING.**
> Check #26 (kernel-control conformance) tests the presence of
> `freeze / unfreeze / get_frozen` methods bound via the
> `AI4BAYESCODE_BIND_KERNEL_CONTROL` macro from
> `include/AI4BayesCode/kernel_control_mixin.hpp`. That header + the
> underlying `block_sampler` / `composite_block` / `joint_nuts_block`
> implementations SHIP IN A SEPARATE FOLLOW-UP PATCH. Until that patch
> lands, Check #26 is DISABLED -- do NOT run its (a)/(b)/(c)/(d)
> sub-checks against generated code (they will fail structurally because
> the required header + methods do not yet exist). Layer-3 R2.f (frozen
> parameter exclusion) is also inert until then, since no wrapper will
> have any frozen parameter to exclude. All other checks (#1-#25 +
> Layer-3 R1 / R2 non-.f / R3) run as before. See
> `DESIGN_NOTES_FREEZE_UNFREEZE_2026-07-19.md` for the migration
> timeline.

Validation of a generated sampler proceeds in **three** layers, from
cheapest to most expensive. Sub-steps within Layer 3 share the same
2-chain run, so running them in sequence is essentially free after the
chains finish.

> **Contributed-block checks (local / downloaded blocks).** If the sampler uses a
> LOCAL (`blocks_local/`) or DOWNLOADED (`blocks_download/`) block, that block ships
> its OWN validation skill. In ADDITION to the core checks below, load the block's
> `ValidationSkill` (its manifest's `ValidationSkill:` field ->
> `<tier>/<Block>/skills/<Block>_validation.md`) and apply exactly the checks its
> manifest names in `ChecksApplicable:` (e.g. `#17; BL1..BL6` -- the `BL*` entries are
> the block's OWN checks, defined in that validation skill). The block's bundled
> example + library test (`test_<Block>.cpp`) are the reference for what a passing
> run looks like. See `block_catalogue/index.md` ("Contributed blocks") for how the block
> is selected + its Example loaded.

1. **Syntactic** -- compilation. The C++ compiler does this for free; no
   manual checklist. If `sourceCpp` fails, fix and retry.
2. **Semantic** -- code-level review. Twenty-six checks (#1-#26), all
   Layer-2 semantic. Most are static code review; a few -- #12 (gradient
   verification via autodiff), #14 (bijection probes), and #15 (library
   parity) -- run a throwaway compile at generation, but they are still
   semantic checks, not a separate category. Checks #11 (joint-NUTS)
   and #13 (RNG separation) fire conditionally; Check #26 (kernel-control
   conformance) fires on every wrapper. These catch bugs that compile
   and run without error but produce a wrong posterior.
3. **Runtime** -- all execution-based checks, sharing one pair of chains
   produced with `keep_history = TRUE`:
   - **R1. Smoke test** (~10 steps) -- catches immediate failures, non-
     finite values, and predict_at state mutation.
   - **R2. 2-chain R-hat / ESS** (4000 burnin + 4000 keep per chain) --
     does the sampler converge to some stationary distribution?
   - **R3. Posterior check** (Bayesian p-values + PSIS-LOO, reuses R2's
     chains) -- is that stationary distribution the right posterior?

Work through the layers in order. If a layer fails, fix the code and
re-run that layer before moving on. Within Layer 3, run R1 -> R2 -> R3
in order; failure at an earlier step means don't bother with the next.

---

## Check number registry (all 26)

Checks #1-#26 are Layer-2 semantic audits (#14-#17 are defined in
sibling skill files; the rest in this file). The Layer-3 R2-VI
PSIS-k-hat diagnostic (no registry number) is also defined here. This
registry is the authoritative cross-reference.

| # | Name | Layer | Defined in | Trigger |
|--:|---|---|---|---|
| 1 | Distribution parameterization | Semantic | validator.md Sec.1 | always |
| 2 | Parallel vs sequential update | Semantic | validator.md Sec.2 | any Gibbs block with vector parameter |
| 3 | Dead parameters | Semantic | validator.md Sec.3 | always |
| 4 | Missing intercept / offset | Semantic | validator.md Sec.4 | regression-style likelihood |
| 5 | Jacobian handling | Semantic | validator.md Sec.5 | any `nuts_block` with a constraint wrap |
| 6 | DAG consistency | Semantic | validator.md Sec.6 | always |
| 7 | Dependency declaration | Semantic | validator.md Sec.7 | always |
| 8 | Rcpp API correctness | Semantic | validator.md Sec.8 | always |
| 9 | Numerical stability | Semantic | validator.md Sec.9 | always |
| 10 | State mutation in predict_at | Semantic | validator.md Sec.10 | always |
| 11 | joint_nuts_block extra audit | Semantic | validator.md Sec.11 | sampler uses `joint_nuts_block` |
| 12 | Gradient verification via autodiff | Semantic | validator.md Sec.12 | any `nuts_block` with a hand-written gradient |
| 13 | RNG separation (MCMC vs predict_at) | Semantic | validator.md Sec.13 | any wrapper with a stochastic refresher |
| 14 | Bijection sanity probes (round-trip + Jacobian non-singularity + fwd/rev inverse-pair) | Semantic | validator.md Sec.14 | `rjmcmc_block` with `templated_bijection_1d` (user-supplied custom bijection) |
| 15 | Library parity test (Gibbs + specialized samplers) | Semantic | codegen_priors.md Sec.2c | any example using a `*_gibbs_block`, `*_slice_sampling_block`, `elliptical_slice_sampling_block`, `poisson_multinomial_aug_block`, or hand-written Gibbs in an rjmcmc hook |
| 16 | Inline Gibbs-exception justification comment | Semantic | codegen_priors.md Sec.2d | any `*_gibbs_block` construction site or hand-written Gibbs in rjmcmc hook |
| 17 | No hand-written Gibbs samplers outside whitelist | Semantic | codegen_priors.md Sec.2e | always |
| 18 | Dense metric justification + pilot scaling | Semantic | validator.md Sec.18 | joint block with `cfg.use_dense_metric = true` |
| 19 | Vectorized gradient computation (BLAS compliance) | Semantic | validator.md Sec.19 | any `nuts_block` / `joint_nuts_block*` whose log-density reads a design matrix from ctx |
| 20 | n_warmup_per_step must stay 0 | Semantic | validator.md Sec.20 | any `nuts_block_config` in the generated cpp |
| 21 | VI block contract conformance | Semantic | validator.md Sec.21 | any `vi_block` subclass in the wrapper's composite (inheritance + engine_kind + q-sample-write + history shape) |
| 22 | VI optimizer = RAABBVI | Semantic | validator.md Sec.22 | any `vi_block` subclass (avgAdam + iterate averaging + R-hat-conv + SKL termination) |
| 23 | readapt_NUTS state-preservation + RNG separation | Semantic | validator.md Sec.23 | wrapper exposes the kernel-control method `readapt_NUTS` (i.e., contains any `nuts_block` / `joint_nuts_block` child) |
| 24 | Joint-NUTS pathology pre-flight (funnel NCR / constraint kind / lambda completeness) | Semantic | validator.md Sec.24 + joint_nuts_failure.md | cpp constructs a `joint_nuts_block` |
| 25 | Trans-dimensional / Dirac-spike must use `rjmcmc_block` (reducibility + silent-slab guard) | Semantic | validator.md Sec.25 + codegen_priors.md Sec.3a Class 2b/4 | model is Sec.3a Class 2b (Dirac point-mass spike) or Class 4 (parameter-space dimension is a random variable) -- i.e. posterior support is a union of manifolds of different dimension |
| 26 | Kernel-control conformance (freeze / unfreeze / get_frozen present + whitelist/blacklist gate + refreeze warning + stale-derived warning) | Semantic | validator.md Sec.26 + DESIGN_NOTES_FREEZE_UNFREEZE_2026-07-19.md | always (kernel-control is a universal wrapper category per interface.md Sec.1) |

**Check #12 status note:** Check #12 is the ONE execution-based Layer-2
check -- the AI writes a throwaway `tests_autodiff/verify_<ClassName>.cpp`
during generation, confirms max-diff tolerances, and deletes the file
on PASS. See Sec.12 below for the full template.

**Checks #15-17** are pre-merge obligations for any example using
`*_gibbs_block` or hand-written Gibbs; the `block_catalogue/index.md` and
`system_design.md Sec.11.7 / Sec.16` both cite them. See the pre-merge
checklist in `system_design.md Sec.16` for how they roll up into the
overall gate.

---

## Layer 2 -- Semantic

Twenty-six checks. **Checks #1-#11, #13, #18, #19, #20, #21, #22,
#23, #25, and #26 are static** -- the AI audits them by reading the
generated `.cpp` (plus, for #23 readapt_NUTS and #26 freeze/unfreeze,
R-level round-trip tests in the runner), no execution required. Checks #14-#17 are
defined in sibling skill files. **Check #12 (gradient verification via autodiff) is
the one execution-based semantic check** -- the AI writes a throwaway
companion file `tests_autodiff/verify_<ClassName>.cpp` that copies the
production hand-written log-density verbatim and adds
`autodiff::var`-compatible templated versions, compiles it, runs a
5-20-point autodiff-vs-handwritten comparison (finite-difference
fallback for blocks using `lgamma` / `digamma`), and confirms max abs
diff < 1e-8 (AD) or < 1e-5 (FD) per block. On PASS the AI **deletes**
the verify file. The production `.cpp` is shipped with no `#ifdef`, no
autodiff code, no Eigen, no verification scaffolding; the user compiles
it as-is with `Rcpp::sourceCpp(...)`. See Check #12 body below for the
full template.

Checks #11 (joint-NUTS), #13 (RNG separation), #18 (dense metric),
#19 (BLAS compliance), #20 (n_warmup_per_step), #21 (VI block
contract), #22 (VI optimizer), #23 (readapt_NUTS), #24 (joint-NUTS
pathology pre-flight), and #25 (rjmcmc / Dirac-spike) fire
conditionally -- only when the relevant pattern is present (see each
check's **Trigger**). Check #26 (kernel-control conformance) fires on
EVERY wrapper -- freeze/unfreeze/get_frozen is a universal category.

### 1. Distribution parameterization

**Check:** every call to a random distribution has a comment stating the
exact parameterization and expected mean.

**What to look for:**
```cpp
// WRONG -- no comment, ambiguous:
std::gamma_distribution<double> gam(a, b);

// RIGHT -- annotated:
std::gamma_distribution<double> gam(shape, scale);
// ^ C++ std uses shape-SCALE, E[X] = shape * scale
```

**Common traps:**
- `std::gamma_distribution(shape, SCALE)` vs R `rgamma(n, shape, RATE)`
  -- **they are inverses**. To match R's `rgamma(n, a, rate=r)` in C++:
  `std::gamma_distribution<double>(a, 1.0/r)`.
- `std::normal_distribution(mean, STDDEV)` -- not variance.
- InvGamma: draw `Gamma(shape, 1/scale)` then invert. NOT `Gamma(shape, scale)`.

**Fix:** add comment with E[X] next to every distribution call.

### 2. Parallel vs sequential update

**Check:** when a Gibbs block updates a vector of parameters (gamma_1..p
or z_1..N), does it update each component sequentially (write back after
each flip) or all at once (compute all, then flip all)?

**What to look for:**
```cpp
// WRONG -- parallel update:
arma::vec log_odds = compute_all_log_odds(context);
for (int j = 0; j < p; ++j) {
    values[j] = (uniform(rng) < sigmoid(log_odds[j])) ? 1.0 : 0.0;
}

// RIGHT -- sequential update:
for (int j = 0; j < p; ++j) {
    arma::vec log_odds = compute_all_log_odds(context);  // re-read
    values[j] = (uniform(rng) < sigmoid(log_odds[j])) ? 1.0 : 0.0;
    context[block_name] = values;  // write back before next j
}
```

**When parallel is OK:** only when components are conditionally independent
given the rest (e.g. z_i in a mixture model with fixed component params).

**When sequential is required:** whenever component j's conditional depends
on component k's current value (e.g. spike-and-slab with correlated X).

**Fix:** add `context_[name] = values_;` inside the loop after each update.

### 3. Dead parameters

**Check:** every parameter declared in shared_data is actually updated by
some block during the Gibbs sweep.

**What to look for:**
- A key is `set()` in the constructor but no block updates that key.
  A block "updates" a key when:
  - the key matches the block's `name()` (single-parameter blocks), OR
  - the block is a joint block (e.g. `joint_nuts_block`) that writes
    the key as a sub-parameter
    via `current_named_outputs()`. The joint block's own `name()` is
    a label, not a data key, and will not match any sub-parameter key.
- A block is `add_child()`'d but none of the keys it writes back is
  read by any other block's `declare_dependencies` (i.e. the block's
  output is genuinely orphaned).
- A parameter appears in `get_current()` output but its value never changes
  across `step()` calls

**Fix:** ensure the parameter has a block that samples it, or remove it
from the model if it's meant to be fixed.

### 4. Missing intercept / offset

**Check:** if the model has `Y ~ f(X) + epsilon`, is there an intercept
term `alpha`?

**What to look for:**
- The log-density uses `Y - X*beta` without subtracting an intercept
- No block updates a key named "alpha" or "intercept" -- neither a
  single-parameter block of that name, nor a joint block that writes
  such a sub-parameter via `current_named_outputs()`.
- The user's model description mentions "regression" but the generated
  code has no constant term

**Fix:** add an intercept block (scalar, real NUTS or normal Gibbs).

### 5. Jacobian handling

**Check:** all NUTS blocks use `constraints::*::wrap` for the Jacobian.
No hand-written Jacobian terms in the log-density.

**What to look for:**
```cpp
// WRONG -- hand-written Jacobian:
lp += std::log(sigma);  // Jacobian for log transform

// RIGHT -- handled by wrap:
return constraints::positive::wrap(theta_unc, grad,
    [&](const arma::vec& sigma_nat, arma::vec* grad_nat) {
        // natural-scale lp only, NO Jacobian
        return lp;
    });
```

**Fix:** remove any manual `+= log(x)` Jacobian terms from the lambda
body. The wrap function adds them automatically.

### 6. DAG consistency

**Check:** the predict DAG edges match the actual data flow in the
code, AND every key consumed inside a predict refresher is reachable
from the predict DAG (so `predict_at(list(K = new))` actually feeds
into the recomputation).

**What to look for:**
- `declare_predict_edges("X", {"f_bart"})` but the code doesn't
  actually compute f_bart from X (e.g. it uses a different key name).
- A node is in the predict DAG but not in shared_data.
- `declare_data_input("X")` but X is never used in any predict
  computation.
- **MISSING predict-edges or data-input declarations.** For every
  key `K` that a `register_stochastic_refresher` /
  `register_refresher` lambda reads via `d.get("K")` (or
  equivalent), at least one of the following must hold:
    1. `K` is registered as a replaceable input via
       `declare_data_input("K")`, OR
    2. `K` appears as the source of a `declare_predict_edges("K", {...})`
       chain that reaches the refreshed output.

  **Common failure pattern (regression):** the `y_rep` refresher reads
  `X` via `d.get("X")` to compute `X @ beta`, but the code only
  declares `declare_predict_edges("beta", {"y_rep"})` and forgets
  `declare_predict_edges("X", {"y_rep"})` AND does not
  `declare_data_input("X")`. Result: `model$predict_at(list(X = X_new))`
  silently runs with the OLD X (or an empty stub), producing
  meaningless predictions. Compile + R-hat all pass.

**Fix:** for every key read inside a predict refresher, either
register it as a `declare_data_input` (if it's user-replaceable
data) or add a `declare_predict_edges` arrow from it to the
refresher's output (if it's a sampled / derived upstream node).

**Sibling check -- silent default-substitution inside refresher
bodies is forbidden.** A stochastic / deterministic refresher's
lambda body MUST assume the framework has guaranteed that every
declared parent is available; it MAY NOT synthesise a fallback
for a missing parent. The framework's BFS rule (Pass 1
deterministic + Pass 2 stochastic in `shared_data.hpp`) is the
SINGLE source of truth for "should this refresher fire": when any
declared parent is missing and has no valid default in
`values_`, the BFS must SKIP the refresher and the output key
must be absent from `predict_at(newdata)`'s returned list.

**Forbidden patterns inside a refresher lambda:**

```cpp
// FORBIDDEN -- silently substitutes the training mean when v2 is missing:
const arma::vec& v2 = d.get("v2");
const arma::vec v2_eff = (v2.n_elem == N_cur)
                            ? v2
                            : arma::ones<arma::vec>(N_cur) * arma::mean(v2);
// FORBIDDEN -- silently copies training X when X is empty at predict time:
const arma::vec& X_flat = d.has("X") ? d.get("X") : x_train_flat_;
// FORBIDDEN -- silently inserts zeros / NaNs for missing entries:
arma::vec sigma_per_obs(N_cur, arma::fill::value(0.1));   // hardcoded
```

These patterns make `predict_at(newdata)` return downstream keys
(`y_rep`, etc.) that the user did not authorise the framework to
compute -- the framework's correctness contract is that **a missing
data_input means "skip the downstream refresher"**, not "improvise
a default and continue". This bug is silent: compile passes,
R-hat / ESS / LOO pass, but the posterior predictive at new
covariates is computed with WRONG variances / WRONG covariates.

**Concrete failure example.** Model:

```
y_i ~ N(theta_i, v2_i),    v2_i  = known per-observation variance
theta_i ~ N(f(X_i, Z_i), tau^2),    f(X, Z) = BART(X) + spline(Z)
```

User-verified predict DAG: `X -> f_bart -> theta; Z -> f_spline ->
theta; tau -> theta; theta -> y_rep; v2 -> y_rep`. User calls
`predict_at(list(X = X_new, Z = Z_new))`. Correct behaviour:
`theta` recomputed at the new (X, Z); `y_rep` ABSENT from the
result because `v2_new` was not supplied. Incorrect (silent-
substitution) behaviour: refresher computes
`v2_eff = mean(v2_train) * ones(N_new)` and returns a `y_rep`
that uses constant per-point variance -- wrong posterior
predictive.

**How to detect:** the R1 smoke test (Layer 3) cycles through
`predict_at(<subset>)` for every reasonable subset of
`data_inputs` and compares the actual output keys to the keys
predicted by the declared DAG (using the same BFS availability
rule). Any extra key in the actual output -> silent substitution
in the refresher -> fix the refresher body.

**Fix template.** Drop the fallback; let the framework decide:

```cpp
// CORRECT -- read declared parents only; framework guarantees they're
// available. If they're not, the BFS rule skips this refresher and
// the output key won't appear in predict_at().
const arma::vec& v2 = d.get("v2");      // size-N, no fallback
const arma::vec& theta = d.get("theta");
arma::vec y_rep(theta.n_elem);
for (std::size_t i = 0; i < theta.n_elem; ++i) {
    y_rep[i] = theta[i] + std::sqrt(v2[i]) * norm(rng);
}
return y_rep;
```

**Sibling check -- predict DAG FULL-RECONSTRUCTION rule (every
intermediate node must be a first-class participant in the predict
walk, not collapsed into a terminal refresher's lambda body).**

This is the most failure-prone codegen pattern, severe enough that
it deserves its own discipline beyond the simpler "node set must
match" check. The full discipline has FOUR mandatory conditions per
intermediate node, ALL of which must hold. Missing any one of them
is a bug.

### The Full-Reconstruction Discipline

For every node `N` in the user-verified DAG that is NEITHER a root
(prior / data input) NOR the terminal observed node, **`N` must be
fully reconstructable from its declared parents at predict time**.
"Fully reconstructable" means ALL FOUR conditions below hold:

**(1) `N` has a dedicated slot in shared_data.**
The constructor calls `impl_->data().set("N", arma::vec(...))` (or
sets a scalar) at construction time. Without a slot, the predict
walk has nowhere to write the recomputed value, and downstream
refreshers reading `d.get("N")` see stale or absent data.

**(2) `N` has incoming `declare_predict_edges` from EXACTLY its
direct parents in the user-verified DAG.**
Not transitive ancestors, not the union of its block's Gibbs reads.
Direct generative parents only. If the user's mermaid shows
`f_bart, Zbeta, tau, theta_raw -> theta`, then the four edges
`declare_predict_edges("f_bart", {"theta"})`, ...,
`declare_predict_edges("theta_raw", {"theta"})` must all be
present, AND there must be NO other edges feeding into `theta`.

**(3) `N` has its OWN dedicated refresher.**
`register_refresher("N", ...)` for deterministic intermediates,
`register_stochastic_refresher("N", ...)` for stochastic ones. The
refresher body reads `N`'s direct parents from `d.get(...)` and
writes the recomputed value. **`N`'s computation must NOT live
inside any other node's refresher**, even if it would be more
"efficient" to inline. Inlining breaks the predict-DAG walk's
ability to replace `N` (e.g. `predict_at(list(theta = theta_new))`)
and breaks `ai4bayescode_plot_dag(model)`'s ability to visualize the generative
story.

**(4) `N`'s outgoing edges go to nodes that read ONLY `N`, not
`N`'s parents.**
If `theta -> y_rep` is declared, the `y_rep` stochastic refresher
must read `d.get("theta")`, NOT `d.get("f_bart")` /
`d.get("Zbeta")` / etc. Reading `N`'s parents inside `N`'s child
refresher is the symptom of an inlined intermediate -- even if the
edge `N -> child` IS declared, the child's lambda is still doing
`N`'s job. Look at every stochastic refresher's lambda body: every
`d.get(K)` call must correspond to a direct predict-DAG parent of
the refresher's output node.

### Why "fully reconstruct"?

The predict DAG is a **callable computation graph**, not a
diagram. At `predict_at(list(X = X_new))` time, the framework:
1. writes `X = X_new` into the scratch context,
2. BFS-walks the predict DAG from `changed = {X}` forward,
3. at each node N along the walk, runs N's registered refresher,
   which writes the recomputed `N` value into scratch,
4. terminal stochastic refreshers (y_rep) sample from their
   parents' fresh scratch values.

If `theta` has no slot / no edges / no refresher / its job is done
inside `y_rep`'s lambda, then `theta` is **not part of the
computation graph** -- it is a private intermediate in one lambda.
The user can't query it, replace it, plot it, or audit it. The
sampler's posterior may still be correct (sampling lives on a
different DAG), but the predict-time generative model is silently
incomplete.

Sampling vs prediction discipline (codegen_cpp.md Sec.6c, Hard Rule #5):
- **Sampling DAG** (Gibbs reads + invalidates) MAY collapse
  intermediates for efficiency (NCP, marginalization, full-
  conditional shortcuts) -- that's a sampler implementation choice.
- **Predict DAG** (`declare_predict_edges` + refreshers) MUST
  fully reconstruct every node the user signed off on. The predict
  DAG is **invariant to sampler parameterization choices**: both
  centered and non-centered parameterizations of the same model
  produce the same predict DAG.

### Acceptance test (machine-checkable, must pass before shipping)

**(A) Node-set equality.**
1. Extract node set from user-verified mermaid in chat: `U`.
2. Extract from generated `get_dag()`:
   `G = keys(predict_edges)  U  values(predict_edges)  U  data_inputs`.
3. Apply the permitted renames (`y -> y_rep`, snake_case
   conversions like `Z_mat -> Z_mat_flat`). Optional hyperparent
   nodes that the user explicitly marked as prior-only (e.g.
   `tau_smooth` feeding only an RW2 penalty inside the prior, not
   the generative chain) MAY be omitted; otherwise keep them.
4. `G == U`. Symmetric difference must be empty.

**(B) Refresher presence per intermediate.**
For every node `N  in  U \ {roots, terminals}` (i.e. every
intermediate), grep the generated `.cpp` for
`register_refresher("N", ...` OR `register_stochastic_refresher("N", ...`.
Exactly one such call must exist. Missing => `N` is inlined
somewhere => bug.

**(C) Direct-parent rule per refresher body.**
For every `register_*refresher("N", lambda)` call, walk the lambda
body. Collect every `d.get("K")` call. The set of `K`s must equal
the direct predict-DAG parents of `N` (read off
`declare_predict_edges` with target `N`). Reading a transitive
ancestor (a parent's parent) => `N` is doing its parent's job => the
parent is inlined => bug.

**(D) Replacement-roundtrip smoke test.**
For every intermediate `N`, calling
`model$predict_at(list(N = N_new))` (where `N_new` is a plausible-
shape replacement) must:
- not error,
- return a result where downstream nodes' values reflect `N_new`
  (NOT the original `N`).

This is the strongest check and exercises the full reconstruction
pipeline. Add it to the model's test suite alongside the standard
`predict_at(list())` and `predict_at(list(X = X_new))` checks.

### Concrete failure example (the meta-regression case)

User-verified mermaid:
```
X --> f_BART; Z_mat --> Zbeta; beta --> Zbeta;
f_BART --> theta; Zbeta --> theta; tau --> theta; theta_raw --> theta;
theta --> y; v2 --> y
U = {X, Z_mat, beta, tau, theta_raw, f_BART, Zbeta, theta, y, v2}
```

Generated `get_dag()$predict_edges`:
```
X -> f_bart, Z_mat_flat -> Zbeta, beta -> Zbeta,
f_bart -> y_rep, Zbeta -> y_rep, tau -> y_rep, v2 -> y_rep
G = {X, f_bart, Z_mat_flat, Zbeta, beta, tau, v2, y_rep}
```

After renames `y -> y_rep`, `Z_mat -> Z_mat_flat`:
```
U' = {X, Z_mat_flat, beta, tau, theta_raw, f_bart, Zbeta, theta, y_rep, v2}
G \ U'  = {}
U' \ G  = {theta_raw, theta}     # <- FAIL (A): missing nodes
```

Diagnosing each missing piece:
- (A) `theta`, `theta_raw` missing from node set.
- (B) No `register_refresher("theta", ...)` in the generated `.cpp`.
- (C) The `y_rep` refresher's lambda body grep shows it calling
      `d.get("f_bart")`, `d.get("Zbeta")`, `d.get("tau")`,
      `d.get("theta_raw")`, `d.get("v2")` -- five reads, but
      `y_rep`'s direct parents per the user DAG are only
      `{theta, v2}`. The four extra reads are exactly the parents
      of the dropped `theta` node.
- (D) `model$predict_at(list(theta = some_theta_new))` errors with
      "unknown key 'theta'" (no node to replace).

**All four signs of inlining present. Fix per codegen_cpp.md Sec.6c:**
- Add `impl_->data().set("theta", arma::vec(N, fill::zeros))`.
- Add the four incoming edges:
  `declare_predict_edges("f_bart", {"theta"})` etc.
- Add `declare_predict_edges("theta", {"y_rep"})`.
- Add `register_refresher("theta", lambda)` whose lambda reads
  `f_bart, Zbeta, tau, theta_raw` and writes `theta`.
- Strip the four `d.get("f_bart")` etc. calls from `y_rep`'s
  lambda; it now reads ONLY `theta` and `v2`.

After the fix, all four conditions (A)-(D) pass and `ai4bayescode_plot_dag()`
matches the user-approved DAG exactly.

**Sibling check -- `predict_at` Rcpp wrapper must forward to the
framework BFS; must not hard-reject partial newdata or shadow-
implement the predict computation.** Three forbidden patterns
all violate the same rule: the framework's BFS
(`shared_data.hpp::predict_downstream_of` +
`predict_stochastic_sampleable`) is the SOLE authority on which
downstream keys appear in the output. The wrapper's only job is
to parse `new_data`, place the supplied keys in a
`block_context replaced`, and call
`impl_->predict_at(replaced, predict_rng_)`.

**Forbidden pattern A -- reject ALL non-empty newdata:**

```cpp
Rcpp::List predict_at(Rcpp::List new_data) const {
    if (new_data.size() > 0) {
        Rcpp::stop("<ClassName>: predict_at takes empty list");
    }
    return Rcpp::wrap(impl_->predict_at({}, predict_rng_));
}
```

When the model declares `declare_data_input("X")` but the wrapper
hard-rejects non-empty `new_data`, calling
`model$predict_at(list(X = X_new))` errors out. Compile + R-hat
all pass; the silent break is only discovered when the user tries
to use it.

**Forbidden pattern B -- reject PARTIAL newdata (force user to
supply ALL siblings):**

```cpp
if (!has_v) {
    Rcpp::stop("predict_at: v_sq must be supplied when X / Z are "
               "supplied for new points.");
}
```

This pre-empts the framework's selective-output rule, forcing the
user to either supply ALL N-scaled inputs or get no output. The
correct behaviour: forward whatever the user supplied; if `v_sq`
is missing, the BFS will skip `y_rep`, and the output will contain
the upstream nodes (`theta_pred`, `theta_mean`, etc.) that ARE
computable from the supplied subset.

**Forbidden pattern C -- shadow-implement the predict computation
(typically inside an `if (keep_history_)` history-mode branch):**

```cpp
if (keep_history_) {
    for (std::size_t d = 0; d < n_draws; ++d) {
        // Manually compute spline_d, theta_mean_d, theta_pred_d,
        // y_rep_d. The framework's BFS is NEVER consulted; the
        // wrapper has reinvented the predict-DAG walk and ignores
        // the user's newdata-induced availability constraints.
    }
    return out;
}
```

A shadow implementation always emits the FULL set of downstream
keys (because the manual loop never checks availability), masking
silent default-substitution bugs and contradicting `ai4bayescode_plot_dag()`'s
promise to faithfully reflect the predict-DAG. The history-mode
branch MUST iterate posterior draws, install each draw's
parameters into a fresh `replaced` context, and call
`impl_->predict_at(replaced, predict_rng_)` inside the loop. The
framework's BFS then runs ONCE per draw and produces only the keys
it considers reachable for that draw's `newdata`.

**Static grep (yellow flags):**

```
grep -nE 'Rcpp::stop\([^)]*supplied when' examples/*.cpp
grep -nE 'predict_at[^{]*\{[^}]*keep_history_[^}]*for[^}]*draws' examples/*.cpp
```

**Runtime detection (R1 smoke).** Layer 3 R1 cycles through
`predict_at(<subset>)` for every smoke-fixture newdata combination
and (i) requires the call to NOT throw on legitimate partial
inputs, (ii) compares the actual output keys against the strict
BFS rule (non-empty newdata: missing data_inputs are unavailable;
no auto-substitution from training defaults). A wrapper that
hard-rejects, or returns a key whose declared parent is unavailable,
fails R1.

**Concrete failure example** (one model, three regressions of the
same wrapper). `MetaRegBartSpline` -- meta-regression:
```
Y ~ N(theta, v2), theta ~ N(BART(X) + spline(Z), tau^2)
```
The user calls `predict_at(list(X = X_new, Z = Z_new))` with the
intent "recompute `theta_pred` at the new covariates; do NOT
predict `y_rep` because per-observation variances `v_sq` aren't
available at the new points". Three AI-generated regressions of
this wrapper have been observed:

1. **Silent default-substitution** (forbidden pattern in the
   sibling check above): refresher uses
   `mean(v_sq_train) * ones(N_new)` and returns `y_rep`.
2. **Hard-reject** (pattern B): wrapper throws
   `"v_sq must be supplied when X / Z are supplied"`.
3. **Shadow history-mode predict** (pattern C): wrapper's
   `if (keep_history_)` branch manually loops draws and computes
   `y_rep_d` regardless of whether `v_sq` is in newdata.

The correct behaviour: forward to
`impl_->predict_at(replaced, predict_rng_)` and return whatever
the BFS produces. With `replaced = {X, Z_mat_flat}` and `v_sq`
absent, the output contains `f_bart`, `spline`, `theta_mean`,
`theta_pred` but NOT `y_rep` -- matching the user's intent without
any extra wrapper logic. The same code path serves history mode
by looping draws and calling
`impl_->predict_at(replaced, predict_rng_)` inside the loop with
per-draw `replaced["f_bart"]`, `replaced["tau"]`, etc.

**Sibling check -- `set_current(X, y)` dispatcher must use the
CURRENT n, not the construction-time N.** If a Tier-A wrapper's
`set_current(Rcpp::List)` accepts an `X` or `y` key, the dispatcher
MUST validate against the current working n (refreshed every time
X changes), not against a frozen `N_` cached at construction
(system_design.md Sec.7 rule 4: "row count 50 does not match
**current n** = 100"; rule 6: "refresh cached Tier A state --
dimension caches").

**Forbidden anti-pattern:**

```cpp
// WRONG: compares against frozen N_; rejects any legitimate N
// change without explaining why.
if (X_new.nrow() != N_ || X_new.ncol() != p_)
    Rcpp::stop("X dimensions must match construction");
```

**Static grep:**

```
grep -E '(X|y).*\bnrow\s*\(\s*\)\s*!=\s*N_|(\.n_elem|\.size\(\))\s*!=\s*N_' examples/*.cpp
```

For every hit, the wrapper must either:

- **(A) Dynamic-N branch (PREFERRED).** Pattern from codegen_cpp.md
  Sec.7a canonical template:
  - Check `X.ncol() != p_` (strict, p is fixed by internal block
    state -- gamma, beta, tau2, cluster vectors).
  - Update `N_ = X_new.nrow()` after committing X.
  - Resize the `y_rep` slot when N changes.
  - Clear history if `keep_history_ && N_changed`.
  - Cross-check `X.nrow == y.length` when both keys appear in the
    same call.

- **(B) Strict-N branch (LEGACY, DISCOURAGED).** Permitted ONLY when
  the model genuinely requires fixed N (e.g. HMM hidden-state
  length, ARMA / change-point models where N is a model invariant,
  Albert-Chib probit augmentation where the latent z is length-N).
  Error message MUST name the reason and tell the user to
  reconstruct:
  ```
  Rcpp::stop("This model fixes N at construction because <REASON>. "
             "To change N, reconstruct the wrapper.");
  ```

**Common failure trace.** User calls
`m$set_current(list(X = X_obs, y = y_obs))` where `X_obs` and
`y_obs` are dimensionally consistent but `X_obs.nrow()` differs
from the construction-time N. The dispatcher errors with
`"X dimensions must match construction"`. The user expected the
wrapper to update its working N. Fix: apply the canonical template
from codegen_cpp.md Sec.7a.

**Sibling check -- refresher must read N dynamically (not capture
constructor-time `N`).** Any stochastic refresher whose output
(`y_rep` etc.) depends on N MUST derive N inside the lambda body
from `X_flat.n_elem / p` or `y.n_elem`, NOT capture it from
construction scope:

```cpp
// CORRECT:
[p](const shared_data_t& d, std::mt19937_64& rng) {
    const arma::vec& X_flat = d.get("X");
    const std::size_t N_cur = X_flat.n_elem / p;     // DYNAMIC
    // ...
}

// WRONG -- N captured at construction scope:
[N, p](const shared_data_t& d, std::mt19937_64& rng) {
    // ... uses N captured at lambda-definition time
}
```

The wrong pattern silently breaks `predict_at(list(X = X_new))`
with a different sample size -- refresher reads / writes out of
bounds on the wrongly-sized `y_rep` buffer. Often invisible until
the user exercises new-X prediction.

**Sibling check -- `keep_history_` requires a history-mode branch
in `predict_at` that LOOPS draws (and inside the loop, forwards
to the framework BFS).** If the class has a `keep_history_` field
(a constructor parameter `keep_history` stored as a class field),
the `predict_at` wrapper MUST include an `if (keep_history_)`
branch that loops over ALL posterior draws and returns an
`n_draws x N` matrix per refreshed key (e.g., `y_rep`). **Forbidden
pattern (no loop, single-draw output):**

```cpp
Rcpp::List predict_at(Rcpp::List new_data) const {
    block_context replaced;
    /* ... parse new_data ... */
    auto result = impl_->predict_at(replaced, predict_rng_);
    /* converts result to a length-N vector unconditionally */
}
```

When the class supports `keep_history_ = TRUE` but `predict_at`
only returns the current (last) draw, the user gets a single
point estimate dressed up as a posterior-predictive output --
**a silent statistical bug**. Compile + R-hat all pass; the
caller never knows their `predict_at` result is NOT the posterior
predictive distribution over the MCMC chain. (Inside the loop, the
forbidden-pattern-C rule above still applies: each iteration must
forward to `impl_->predict_at()` rather than reinvent the DAG walk.)

**Sibling check -- refresher should derive size dynamically, not
capture constructor-time `N`.** If the refresher lambda captures
`N` at construction (e.g., `[N, p](...) { arma::vec y_rep(N); ... }`),
`predict_at(list(X = X_new))` with a different `N_new` will
read/write out of bounds. Compute `N_cur = X_flat.n_elem / p` (or
equivalent) inside the lambda so the refresher adapts to whatever
size `predict_at` provides.

See `codegen_cpp.md Sec.predict_at method` for the canonical two-branch
template (stateful + history mode) and the dynamic-N refresher
pattern.

**Sibling check -- generative-DAG context edges
(`declare_context_edges`) must match the model's prior structure and
stay DISJOINT from `predict_edges`.** This is part of Check #6 (DAG
consistency), not a separate numbered check -- it verifies the
`get_dag()$context_edges` map that `ai4bayescode_plot_dag` renders faded.

`declare_context_edges(from, {to})` declares the prior / hyperprior
parents of sampled parameters (and forest/kernel params feeding a
deterministic mean node). It is VIZ-ONLY: `predict_downstream_of` /
`predict_stochastic_sampleable` in shared_data.hpp never read
`context_edges_`, so a wrong context edge cannot corrupt any
posterior -- the only failure mode is a misleading `ai4bayescode_plot_dag` figure.

Acceptance conditions (all must hold):

1. **Read-from-code, not invented.** Every context edge
   `hyperslot -> param` must correspond to an actual shared_data slot
   used as that parameter's prior hyperparameter in the wrapper /
   log-density. Grep: for each `declare_context_edges("K", {"P"})`,
   `K` must be a `data().set("K", ...)` slot AND the log-density / prior
   for `P` must reference `K`. Inventing an edge with no code basis
   is a bug.
2. **No-fabricated-constant-nodes.** If a prior is a hardcoded
   constant with no named slot (`mu ~ N(0,100^2)`,
   `sigma ~ Jeffreys`), there must be NO context edge for it. The
   generative DAG then equals the predict DAG -- correct, not a gap.
3. **Disjointness.** `keys/values of context_edges_`  intersect  the
   corresponding `predict_edges_` arrow set must be empty for any
   `hyperparam -> param` pair: a prior edge must NEVER also be a
   predict edge (else BFS wrongly recomputes the param when the
   hyperparam is replaced). Static check:
   `for (from,to) in context_edges: assert (from,to) not in predict_edges`.
4. **Forest/kernel convention.** A sampled tree-ensemble / kernel
   parameter feeding a deterministic mean node is declared as a
   context edge (`BART->f_bart`, `genBART->r`, `amplitude->K_matrix`),
   parallel to the user-approved MetaRegBartSpline `BART->f_bart`.

Detection: `ai4bayescode_plot_dag(model)` must render the prior context faded and
the solid predict sub-DAG unchanged; `predict_at(list())` output keys
must be IDENTICAL with and without context edges (proves BFS
isolation). See codegen_cpp.md Sec.6c "Generative-DAG context edges" and
system_design.md Sec.4 "Generative DAG -- context edges".

**Sibling check -- history-mode shadow carve-out (Q5 Ruling A) +
predict_at must NOT echo training data.** Part of Check #6 (DAG
consistency); not a new numbered check.

(a) **No data-echo stubs.** `predict_at` MUST NOT return the observed
training `y` as `y_rep`. Detection: with a model fit to data `y`,
`max|predict_at(list())$y_rep - y|` must be > 0 (a real
posterior-predictive draw differs from the data). Two shipped
examples (BSplineRegression, HSGPRegression) historically
had stub `predict_at` returning `impl_->data().get("y")` -- a silent
correctness bug (fixed 2026-05-15). Any wrapper whose
`predict_at` body is `out["y_rep"] = ...get("y")` fails this check.

(b) **History-mode shadow carve-out (Ruling A).** `keep_history=TRUE`
`predict_at` MAY use a manual per-draw loop ONLY when the per-draw
quantities are tree forests (bart/genbart/softbart) or
`joint_nuts_block` sub-params (alpha/beta/u, Intercept/log_*, z) that
the framework's replaced-key validation (data_inputs  U  block names)
cannot accept. Conditions: the loop computes `y_rep` with the SAME
generative formula as the registered refreshers (gate: equivalence /
R-reference); the STATEFUL path still routes through
`impl_->predict_at` (no shadow there); `ai4bayescode_plot_dag` still reflects the
fully reconstructed predict DAG. Forbidden pattern C does NOT apply
to these. Gold standard MetaRegBartSpline is conformant under this
carve-out. See system_design.md Sec.5 (Q5 ruling).

### 7. Dependency declaration

**Check:** every key read by a block's log-density lambda via `ctx.at("key")`
is listed in `declare_dependencies(block_name, {...})`.

**What to look for:**
- The lambda calls `ctx.at("sigma")` but "sigma" is not in the
  dependency list -> build_context_for will throw at runtime
- A dependency is listed but never read -> harmless but misleading

**Joint blocks:** `declare_dependencies` for a `joint_nuts_block`
is keyed by the **block name**, not by the
individual sub-parameter names. The list should be the **union** of
everything every sub-parameter's log-density reads. Example:

```cpp
// Joint block owns (theta, b), sub-params share one log-density lambda
// that reads Y, M, and sigma_b from ctx.
impl_->data().declare_dependencies(
    "theta_b_joint",            // block name, NOT "theta" or "b"
    {"Y", "M", "sigma_b"});     // union of reads across all sub-params
```

Downstream blocks whose dependencies refer to the sub-parameters
themselves (`"theta"`, `"b"`) work unchanged -- the composite writes
each sub-parameter into shared_data under its own key via
`current_named_outputs()`.

**Fix:** ensure declare_dependencies lists exactly the keys the lambda reads.

### 8. Rcpp API correctness

**Check:** see `skills/rcpp_api.md` for the full list. Key items:
- `Rcpp::List` names checked with `.size()` before `.names()`
- `NumericMatrix` is column-major
- `Rcpp::stop()` used instead of `throw` in Rcpp context
- No Rcpp calls in pure C++ headers

These compile but behave wrongly at runtime if mis-used.

### 9. Numerical stability

**Check:** the log-density and gradient don't produce NaN/Inf for
edge-case parameter values.

**What to look for:**
- `log(x)` where x can be 0 or negative -> add `if (x <= 0) return -inf`
- Division by `sigma` or `sigma^2` where sigma can be very small
- `exp(large_number)` overflow -> use log-sum-exp
- `lgamma(x)` for very small x

### 10. State mutation in predict_at

**Check:** `predict_at` does not read-modify-write MCMC state.

**What to look for:**
- The Rcpp wrapper's `predict_at` calls methods that mutate members of
  the block (e.g. writes to `shared_data_` on the live model)
- `composite_block::predict_at` should operate on a scratch copy of
  `shared_data_t`, never the live one

### 11. joint_nuts_block extra audit (conditional)

**Trigger:** this check ONLY runs if the generated sampler uses
`joint_nuts_block` (or any block that concatenates multiple
sub-parameters into one NUTS vector). Modular NUTS (one parameter
per block) skips this check entirely.

**Why:** joint_nuts_block merges the log-density, gradient, and
unconstrained transform of several parameters into a single concatenate-
and-slice lambda. This is the single highest-risk code pattern in
AI-generated AI4BayesCode samplers -- it compiles, runs, and often even
converges, but silently produces the wrong posterior if any of the
slicing is off by one.

**What to look for (all six must pass):**

1. **Gradient slice alignment.** For a concatenated vector
   `theta_cat = [theta; b]` with `theta.n_elem = N` and `b.n_elem = J`:
   - `d log p / d theta_i` must be written to `grad[i]` for `i = 0..N-1`
   - `d log p / d b_j` must be written to `grad[N + j]` for `j = 0..J-1`
   - Off-by-one or offset swap is the most common bug here.

2. **Every sub-parameter's prior is included, with correct sign.**
   Missing a sub-parameter's prior term turns that parameter into an
   improper flat prior. Posterior moments are then dominated by the
   likelihood with no shrinkage -- often still "converges" under R-hat
   but to the wrong stationary distribution.

3. **Per-slice constraint kinds are explicit in the config.** For
   `joint_nuts_block`, per-slice constraint kinds are declared in the
   config (`joint_constraint::REAL` vs `joint_constraint::POSITIVE`
   vs `SIMPLEX` / `CHOLESKY_CORR` / `COV_MATRIX` / ...), and the block
   itself applies transforms and log-Jacobians before calling the
   user's NATURAL-scale log-density. The user's log-density MUST NOT
   include any log|Jacobian| terms (the block adds them); audit the
   log-density body to confirm no `log(sigma)` terms appear from manual
   Jacobian bookkeeping. If every slice is REAL the concatenated vector
   is already on the identity scale and no transform is applied.

4. **Log |Jacobian| added once, per sub-parameter, on the right slice.**
   If a sub-param is positive (log transform) or simplex (stick-
   breaking), its Jacobian term appears exactly once and reads only
   from its own slice of `theta_cat`. Double-counting or cross-slice
   reads both produce silent miscalibration.

5. **set_current / write-back splits by the same offset map used in
   log_density.** After each NUTS update, the concatenated draw is
   split back into the individual shared_data keys (`theta`, `b`, ...).
   The offset map here MUST match the one used inside the log-density
   lambda. A mismatch means the next Gibbs sweep reads swapped
   parameters and the chain silently produces garbage.

6. **Construction-time dimension asserts.** The constructor must
   assert `theta_cat.n_elem == sum(sub_param_dims)`; the log-density
   lambda must assert on its input dim on first call. These cheap
   asserts catch the majority of bugs 1, 3, and 5 before any chain
   runs.

**R3 tightening when joint_nuts_block is used:** the posterior-
predictive p-value threshold in R3.a tightens from (0.05, 0.95) to
(0.02, 0.98), because silent miscalibration bugs in joint_NUTS are
more insidious than in modular code and need a stricter filter.

**Layer 2 routing:** failures in this check most often masquerade as
Check #1 (wrong parameterization), #4 (missing intercept / offset),
or #5 (Jacobian). When the model uses joint_NUTS, audit Check #11
FIRST before blaming #1/#4/#5.

### 12. Gradientverification via autodiff

**Trigger:** any NUTS-based block with a hand-written gradient -- i.e.,
every generated sampler that uses `nuts_block` with a
`constraints::*::wrap`-based `log_density_grad` lambda. Specialised
blocks that don't carry a hand-written gradient (`bart_block`,
`*_gibbs_block`) are not subject to this check.

**Mechanism (the one execution-based semantic check):** the AI writes
a throwaway companion file `tests_autodiff/verify_<ClassName>.cpp`
alongside the production `.cpp`. The companion contains:
  - copies (verbatim) of the production hand-written log-density
    functions (both versions come from the same LLM think; drift is
    negligible);
  - templated (`autodiff::var`-compatible) versions of the same math;
  - a verify function `verify_<ClassName>_grad(...)` that samples
    random theta points and compares hand-written grad vs autodiff grad
    for each NUTS block.

For blocks whose log-density uses `lgamma` / `digamma` / other
special functions that `autodiff.hpp` doesn't overload, the AI falls
back to **finite-differenceverification** (central-difference of the hand-
written wrap). FD precision is ~1e-5 vs AD's ~1e-8.

The AI compiles the verify file, calls the verify function, and
asserts every `max_diff < 1e-8` for AD-backed blocks / `< 1e-5` for
FD-backed blocks. On PASS, the AI **deletes** the verify file. The
production `.cpp` is delivered unchanged -- no `#ifdef`, no autodiff
code, no Eigen, no verification artifacts. User never sees verification code.

**Joint blocks (`joint_nuts_block`):** the
verify function compares the FULL concatenated gradient vector
(`grad[0..N+J-1]` for a `[theta(N); b(J)]` joint) hand-written-vs-AD
in one call, **not** per sub-parameter slice. A correct joint
gradient is necessarily correct on every slice; verifying the
concatenated vector at once also catches the off-by-one slice
misalignment that per-slice verification would miss (see Check #11).
The verify-file recipe for joint blocks (build `VectorXvar`, call the
templated lp, compute `gradient()` on the whole vector) is in
`codegen_cpp.md` Sec."Constraint-kind selector".

**What to look for in the AI's workflow (as validator):**

1. Production `examples/<ClassName>.cpp` contains ONLY hand-written
   natural-scale log-densities wrapped by `constraints::*::wrap`. No
   autodiff includes, no `#ifdef AI4BAYESCODE_VERIFY_GRADIENTS`, no
   templated variants.
2. AI generated `tests_autodiff/verify_<ClassName>.cpp` at some point
   during generation.
3. AI ran the verify function successfully (generation log should
   record the max_diff values per block).
4. AI deleted `tests_autodiff/verify_<ClassName>.cpp` after PASS (the
   file should NOT be in the delivered artifact).

**What the user sees:** nothing. They `Rcpp::sourceCpp("<ClassName>.cpp")`
and get the hand-written sampler at full speed.

**Failure routing at generation:**
- `max_diff_<block> > threshold` -> bug in that block's hand-written
  gradient. Common classes:
  - sign error (`d/dx(-0.5 x^2) = -x`, not `+x`)
  - factor missing (forgot `2` from `d/dx x^2 = 2x`)
  - wrong denominator (`sigma^2` vs `sigma^3`)
  - double-counted Jacobian (user added `+ log(sigma)` inside the
    natural-scale function while `constraints::positive::wrap` is
    already adding it -- see Check #5)
- Multiple blocks diverging by similar magnitude -> suspect wrong
  parameterization (sd vs variance, rate vs scale) -- check Layer 2
  Check #1 before fixing per-block gradients.

### 13. RNG separation (MCMC vs predict_at)

**Trigger:** any wrapper that registers a stochastic refresher (for
posterior-predictive y_rep or similar). Modular samplers without any
stochastic refresher can skip this check.

**Check:** the wrapper class has TWO RNG members: `rng_` for MCMC
sampling and `mutable std::mt19937_64 predict_rng_` for stochastic
refreshers invoked from `predict_at`. They must not share state.

**What to look for:**
```cpp
// WRONG: predict_at advances the MCMC rng_, so posterior-predictive
//        calls change MCMC trajectory and destroy reproducibility.
Rcpp::List predict_at(Rcpp::List) const {
    return impl_->predict_at(replaced, rng_);  // rng_ is the MCMC RNG
}

// RIGHT: dedicated mutable predict_rng_ seeded ONCE at construction.
//        const method + mutable member preserves const-correctness.
class MyModel {
    std::mt19937_64         rng_;          // MCMC
    mutable std::mt19937_64 predict_rng_;  // predict_at only
    // ...
    Rcpp::List predict_at(Rcpp::List) const {
        return impl_->predict_at(replaced, predict_rng_);  // separate stream
    }
};

// Seed once in the constructor, derived from rng_seed with the golden-
// ratio constant so the two streams are decoupled but reproducible.
predict_rng_(rng_seed == 0
             ? std::mt19937_64{std::random_device{}()}
             : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                               ^ 0x9E3779B97F4A7C15ULL})
```

**Why:** posterior-predictive draws must be reproducible given a stable
construction seed, independent of how many MCMC steps have been taken.
Sharing `rng_` also violates the const contract of `predict_at` (which
is documented as pure/side-effect-free on MCMC state).

**Never** call `std::random_device{}()` in the body of `predict_at` -- R3
samples y_rep O(n_draws) times per call, and `random_device` is slow
and non-reproducible on macOS.

**Fix:** add `mutable std::mt19937_64 predict_rng_`, seed it once in the
constructor with `rng_seed ^ 0x9E3779B97F4A7C15ULL`, thread it into
every `impl_->predict_at(..., predict_rng_)` call.

### 14. Bijection sanity probes for `templated_bijection_1d`

**Trigger:** any cpp that wraps a custom user-supplied bijection in
`AI4BayesCode::rjmcmc_transforms::make_templated_bijection_1d(fwd,
inv)` and assigns the result to `rjmcmc_block_config::transform`.
Does NOT trigger for the identity-coordinate path (no `transform`),
the library 1D transforms (`identity_transform_1d`,
`diagonal_linear_transform_1d`, `diagonal_affine_transform_1d`), nor
for any block other than `rjmcmc_block`.

**What it checks (three numerical probes on a small grid of u
points, all derived from the SINGLE templated forward + inverse pair
the user wrote):**

1. **Round-trip residual** -- `|inv(fwd(u)) - u| < 1e-10` at every
   probe point. Catches typos in the user's analytic inverse.
2. **Jacobian non-singularity** -- `|dbeta/du| > 1e-12` at every probe
   point (auto-computed via reverse-mode autodiff on the templated
   forward instantiated at `autodiff::var`). Catches degenerate
   bijections (constant maps, pole points, sign flips that produce
   `det J <= 0`).
3. **Forward / reverse Jacobian inverse-pair** --
   `|fwd_J(u) * rev_J(T(u)) - 1| < 1e-10`. The bijection invariant.
   Catches inverse implementations that compute the WRONG inverse
   (a typo in the analytic inverse that happens to round-trip but
   not satisfy `J_fwd * J_rev = 1`).

**Why these three (and not the original "verify two bijection
implementations agree"):** the current design uses the SINGLE-template
pattern (one templated forward, instantiated by the framework at
both `double` and `autodiff::var`). The original "two function"
typo class is eliminated by construction -- there is no second
function. What can still go wrong is (a) a wrong analytic inverse,
or (b) a degenerate forward at the working point. Both are caught by
the three probes above.

**Mechanism (companion file pattern, mirroring Check #12):**

The AI generates a throwaway `tests_autodiff/verify_<ClassName>_v1.cpp`
alongside the production `.cpp`. The companion contains:

  - `#include` of the production cpp's templated `Forward` and
    `Inverse` types (or copies them inline if they are local types
    in the production cpp's anonymous namespace);
  - construction of the same `templated_bijection_1d` instance the
    production cpp uses;
  - a verify function that loops over a 10-point grid of u values
    spanning the bijection's working domain and asserts all three
    probes above.

The AI compiles the verify file, runs it, asserts all
`max_residual < tolerance`, and on PASS deletes the verify file.
The production cpp ships unchanged (no autodiff include in
production, no probe scaffolding visible to the user).

**Companion file template:**

```cpp
#include <RcppArmadillo.h>
#include "AI4BayesCode/rjmcmc_custom_bijection.hpp"
#include <cmath>
#include <iostream>

using AI4BayesCode::rjmcmc_transforms::make_templated_bijection_1d;
using AI4BayesCode::rjmcmc_transforms::check_roundtrip;
using AI4BayesCode::rjmcmc_transforms::check_jacobian_inverse_pair;

// Paste the production Forward / Inverse types verbatim:
struct UserForward {
    template <typename T>
    T operator()(T u) const { /* user code from production cpp */ }
};
struct UserInverse {
    double operator()(double beta) const { /* user code from production cpp */ }
};

// [[Rcpp::export]]
Rcpp::List verify_<ClassName>_bijection_v1() {
    auto bij = make_templated_bijection_1d(UserForward{}, UserInverse{});
    const double TOL_ROUND   = 1e-10;
    const double TOL_JAC_MIN = 1e-12;
    const double TOL_INVPAIR = 1e-10;
    // Probe grid: representative u values for the bijection's domain.
    // For unbounded R domain, use a logspaced grid in (-3, 3).
    arma::vec us = arma::linspace(-3.0, 3.0, 10);
    double max_round = 0.0, max_invpair = 0.0;
    double min_jac   = std::numeric_limits<double>::infinity();
    for (double u : us) {
        max_round   = std::max(max_round,   check_roundtrip(*bij, u));
        max_invpair = std::max(max_invpair, check_jacobian_inverse_pair(*bij, u));
        double beta;  double j = bij->apply_forward(u, beta);
        if (std::isfinite(j)) min_jac = std::min(min_jac, j);
    }
    bool pass = (max_round < TOL_ROUND) &&
                (min_jac   > TOL_JAC_MIN) &&
                (max_invpair < TOL_INVPAIR);
    return Rcpp::List::create(
        Rcpp::Named("pass")         = pass,
        Rcpp::Named("max_round")    = max_round,
        Rcpp::Named("min_jac")      = min_jac,
        Rcpp::Named("max_invpair")  = max_invpair);
}
```

**Failure routing at generation:**
- `max_round > 1e-10` -> user's Inverse is wrong. Check the analytic
  inverse formula matches the forward.
- `min_jac < 1e-12` -> forward is degenerate at some probe point
  (constant section, pole, sign flip). Re-derive forward.
- `max_invpair > 1e-10` -> forward and inverse disagree on the
  bijection invariant (`J_fwd * J_rev = 1`). Same root cause as
  round-trip failure or as a sign error in the inverse.

On any failure, the agent fixes the bijection BEFORE shipping; the
companion file stays in `tests_autodiff/` until all three probes
PASS.

**Joint blocks / multi-dim bijections:** Out of scope at this version.
`templated_bijection_1d` is strictly 1D scalar. Multi-dim bijections
(e.g., split-merge for finite mixtures with unknown K) require a
distinct block class and are not yet shipped; if a future block
adds multi-dim support, sanity check generalises to `det J` non-
singularity and the multi-dim inverse-pair invariant.

### 18. Dense metric justification + pilot scaling

**Trigger:** any cpp file with `cfg.use_dense_metric = true` on a
`joint_nuts_block_config`.

**Default policy (2026-06-20): prefer diagonal; escalate to dense on diagnostics;
escalate to 3-phase only at extreme cond.** This is GUIDANCE -- if diagnostics
disagree with the heuristic, follow the diagnostics and re-tune. The recommended
default for any new / unknown joint model is DIAGONAL + single-pilot warmup --
cheapest, and best whenever diagonal suffices (the common case). Escalate by
MEASURING, not predicting (do NOT gate on a `cond(R)` threshold -- it is dimension-
dependent and not calibratable from limited data):

1. Enable `use_dense_metric = true` ONLY when runtime validation (R2/R3) shows
   diagonal is inadequate -- R-hat > 1.01, low ESS, max-tree-depth saturation, or
   low E-BFMI (the signature of a correlation ridge a diagonal metric cannot
   rotate away). **Dense + single-pilot is the right next step** and resolves the
   large majority of correlated targets (broad 31-model corpus: converges to
   cond 609; 5-23x faster keep-phase ESS/s than dense+3-phase -- so do NOT reach
   for 3-phase first). Only DENSE is genuinely REQUIRED (not just faster) when
   correlation is so strong any axis-aligned metric fails to converge.
2. Add `use_three_phase_warmup = true` ONLY if dense + single-pilot ITSELF fails
   to converge -- the rare extreme-cond bootstrap failure (8D equicorr rho=.999
   cond~=8000: single-pilot 0/8 vs 3-phase 8/8). Extreme cond is usually a
   reparameterization smell -- prefer reparameterizing.

The ONE firm, code-enforced rule:
- **NEVER set `use_three_phase_warmup = true` with a DIAGONAL metric** -- 3-phase
  under-tunes the step on diagonal (~38x worse ESS/s on pl2); the code gates
  3-phase to dense-only.
Full decision table + escalation ladder: system_design.md Sec.13 "Metric + warmup decision".

**Why a single pilot can still go wrong (and when 3-phase is the cure):** a
single-pilot dense metric is single-shot Welford adaptation -- no windowed self-
correction. It is the fast default and reliable at realistic cond, but a BAD pilot
estimate freezes the chain (the mass matrix is fixed after the pilot): this happens
at EXTREME cond, when the pilot hasn't explored the support, or on a non-Gaussian
posterior where sample covariance is the wrong statistic. That extreme-cond case is
exactly where 3-phase's Stan-style 5-window adaptation (v1.2, 2026-06-03) pays off --
escalate per step 2, OR reparameterize; Check #18's single-shot concern is relaxed
for the three-phase path. Pilot quality wants (a) length scaling with `d^2`, (b) the
pilot exploring full support before the metric is fixed, (c) near-Gaussian posterior.
The bigger real-world risk, though, is enabling dense SPECULATIVELY when diagonal
would do. 2026-05-01 retrospective on a piecewise-trend time-series model: codegen
turned dense on by guess on an 11-dim posterior; 70% of sim1 replicates failed
(`rhat_max ~= 2.23`, `ess_bulk_AI_min = NA` -- chain stuck under a frozen mass matrix
from a 500-iter pilot), while Stan + DIAGONAL + 20k warmup recovered truth at 88%
coverage on the same data. Every L2 (13/13) and L3 R1/R2/R3 single-dataset gate
passed before sim1 caught it. LESSON: don't enable dense speculatively -- start
diagonal, escalate on diagnostics.

**Static review -- all four must hold:**

1. **Inline justification comment** within 20 lines above the
   `cfg.use_dense_metric = true` line:
   ```
   // JUSTIFICATION (Check #18): <reason> -- <specific evidence>.
   ```
   Acceptable reasons:
   - **Reparameterization explored and ruled out** (preferred fix
     for anisotropy; see `codegen_cpp.md Sec.4a` "Dense metric: opt-in
     only"). E.g., "non-centered hierarchical tried, group-level
     R-hat 1.4 on tau at 20k+20k".
   - **Documented R-hat failure on joint+diagonal at extended
     budget.** E.g., "joint+diagonal 20k warmup gave max R-hat 1.6
     on a representative dataset; dense improves to 1.02 at the
     same budget". Cite dataset / parameter / budget.
   - **Posterior class known to be intrinsically anisotropic.**
     E.g., "GP banana ridge per `codegen_priors.md Sec.11.4`". Cite the
     documentation. **NOTE -- a CENTERED scale-governed hierarchical
     effect is NOT a valid dense justification:** it is a hard Check
     #24(a) FAIL, and the only accepted fix is non-centered
     reparameterization (NCR), not dense metric. Dense cannot rescue a
     funnel that freezes under any static metric (eight_schools_centered
     ESS=NA; joint_nuts_failure.md Mode-1). Do NOT keep a centered
     scale-governed effect and justify it with dense.

2. **Pilot scaling** -- `cfg.dense_metric_pilot_iters >= max(2000,
   100 * d)` where `d = sum(s.dim for s in cfg.sub_params)`. The
   default `500` is undersized for any d >= 5. For an 11-dim joint
   block, that's `max(2000, 1100) = 2000`.

3. **Warmup overhead** -- `cfg.n_warmup_first_call >=
   cfg.dense_metric_pilot_iters + 1000`. Step-size dual averaging
   under the new dense metric needs space.

4. **Adapt iters** -- `cfg.dense_metric_adapt_iters >= 2000`.
   Single-shot adaptation cannot recover from a bad initial
   estimate.

**Failure routing:**
- Missing justification (1) -> most likely speculative dense; flip
  to `use_dense_metric = false` and ship with diagonal. If R2 then
  passes at 4k+4k, dense was unnecessary.
- Undersized (2)/(3)/(4) -> bump to the minimum. The warmup overhead
  is non-trivial (10k-20k iter for high-d joints) but is the actual
  cost of dense metric being correct.

**Limits of static review:** cannot verify the *substance* of the
justification -- the agent's claim "non-centered was tried" is taken
at face value. The companion mitigation is in `codegen_cpp.md Sec.4a`
("Dense metric: opt-in only") which constrains codegen-time policy.
Cannot detect Welford-inversion failures at runtime; those manifest
as `ess_bulk = NA` in Layer 3 R2, and the validator should treat
any Layer 3 R2 stall accompanied by `use_dense_metric = true` as
automatic dense-metric escalation back to this check.

---

### 19. Vectorized gradient computation (BLAS compliance -- Sec.6.1)

**Trigger:** any `nuts_block` / `joint_nuts_block`
whose log-density lambda reads a design matrix from `ctx` (key `X`,
`X_flat`, `X_mat`, or any equivalent flat / matrix design key).

**Why:** the natural-scale log-density and gradient run inside every
NUTS leapfrog step (~10-100 calls per MCMC step). A scalar nested
`for (i) for (j) X_flat[i + j * N] * beta[j]` rather than a BLAS gemv
`X * beta` is 10-50x slower at typical GLM dimensions. Concrete
observed gap on a small (N ~= 80, p ~= 7) Bernoulli logistic GLM:
scalar-loop AI ~46s vs Stan `bernoulli_logit_glm` ~4s -- an 11x gap
entirely from this rule being violated. See `codegen_cpp.md Sec.6.1` for
the per-family templates and the joint case.

**What to look for** (static grep inside the `wrap(...)` lambda body
ONLY -- NOT inside `register_stochastic_refresher` /
`register_refresher` / `rjmcmc_block::continuous_update` hooks, which
are cold paths or per-coefficient by contract):

```cpp
// ANTI-PATTERN -- fail the check
for (size_t i = 0; i < N; ++i) {
    double xb = 0.0;
    for (size_t j = 0; j < p; ++j)
        xb += X_flat[i + j * N] * beta[j];     // <-- inner scalar loop
    ...
    for (size_t j = 0; j < p; ++j)
        (*grad_nat)[j] += resid * X_flat[i + j * N];   // <-- inner scalar loop
}
```

Replace with `arma::vec eta = X * beta` / `*grad = X.t() * (y - mu)`
or the full joint pattern in `codegen_cpp.md Sec.6.1` "Joint case".

**Acceptable exceptions (must be one of):**

1. **Irregular gather / scatter** -- the outer scalar loop indexes via
   a per-observation lookup vector matching regex
   `[a-z_][a-z0-9_]*_idx\[` (e.g., `g_idx[n]`, `cluster_idx[n]`,
   `time_idx[n]`). Pattern: `u_n[n] = u[g_idx[n]]` (gather),
   `(*grad)[1+p+g_idx[n]] += g_mu[n]` (scatter). These cannot be
   vectorized via BLAS; scalar is correct. No justification comment
   required -- the `_idx[` pattern is self-documenting.

2. **Inline justification comment** within 5 lines above the loop:
   ```
   // JUSTIFICATION (Check #19): <reason>.
   ```
   Acceptable reasons:
   - Per-coefficient slicing in an `rjmcmc_block` `continuous_update`
     hook (per-`j` `xtx_j` / `xtr_j` accumulation); but the hook
     itself is excluded by the trigger, so this exception applies
     only if the same code is mistakenly placed in the main
     log-density lambda.
   - Custom likelihood whose per-observation contribution is non-
     linear in a way arma's elementwise primitives cannot express
     (rare -- first try `arma::exp`, `arma::log1p`, `arma::clamp`).

**Failure routing:**
- Bare nested loop without exception (1) or comment (2) -> rewrite
  per `codegen_cpp.md Sec.6.1` per-family or joint template.
- Comment present but unsupported reason -> reject; either find a
  BLAS form or replace the reason with one of the acceptable ones.
- Slow per-iteration gradient observed in a sim1 run (e.g., AI
  10x slower than Stan reference at same N/p) accompanied by a
  passing #19 -> escalate to look for the second anti-pattern in
  `codegen_cpp.md Sec.6.1` "Elementwise helper functions" (helpers
  that hide a scalar loop inside an `arma::vec`-returning wrapper).

**What this check does NOT cover:**
- Cold-path stochastic refreshers (y_rep): per-iteration RNG sampling
  is naturally scalar; not a Sec.6.1 violation.
- Helper functions like `softplus_vec` / `sigmoid_vec` that are
  scalar loops in disguise -- that's a separate `codegen_cpp.md Sec.6.1`
  "Elementwise helper functions" rule, enforced at codegen time
  rather than via static greppable pattern.

---

### 20. n_warmup_per_step must stay 0

**Trigger:** any `nuts_block_config` in the generated cpp.

**Why:** setting `cfg.n_warmup_per_step > 0` re-enables a chain-state
corruption mechanism that the 2026-04-12 mcmclib NUTS bugfix removed
("n_adapt_draws included kept draws, causing ongoing adaptation during
sampling. Fix: n_warmup_per_step default changed from 20 to 0.
Variance now correct across all dimensions"). With ongoing adaptation,
dual averaging is recomputed against the same draws being kept in the
chain, violating detailed balance AND progressively shrinking the step
size into a region the chain cannot leave. **Runtime symptom:** sim1
reports `rhat_max ~= 2.2`, `ess_bulk_AI ~= NA / single digits`, AI
walltime SHORTER than Stan's because the locked chain does almost no
tree exploration. L3 single-dataset checks PASS because the agent's
calibration init (per-county OLS, balanced floor, etc.) lands the chain
in a friendly region; sim1 cross-dataset checks FAIL catastrophically.

This check is the safety net. The codegen-time prevention is in
`skills/codegen.md` Hard Rules + `skills/block_catalogue/index.md` "nuts_block
-> Configuration discipline" + the doc comment in
`include/AI4BayesCode/nuts_block.hpp`. If those have been read, this
check should never fire.

**What to look for** (static grep, anywhere in the generated cpp):

```cpp
// ANTI-PATTERN -- fail the check
cfg.n_warmup_per_step = 5;       // any non-zero literal
cfg.n_warmup_per_step = 10u;
cfg.n_warmup_per_step   = 20;
```

The agent's older justifying language ("5-8% variance bias acceptable
per nuts_block docs") is no longer accurate -- that comment in
`nuts_block.hpp` predated the 2026-04-12 bugfix and has been rewritten.
A regen reading the current header will not produce that text.

**Acceptable in production examples: NONE.** No code-gen agent should
ever write `cfg.n_warmup_per_step = <non-zero>`. If a sigma block
keeps rejecting at the default 0 (the symptom that tempted the
override), the actual problem is a non-centered hierarchical funnel --
fix it methodologically:
1. `joint_nuts_block` over `(sigma_*, z_*)` per
   `codegen_cpp.md Sec.4a` row "scale + raw effect, non-centered".
2. Bump `n_warmup_first_call` to 1500-3000.
3. Better init via OLS / method of moments in the wrapper constructor.

**Failure routing:**
- Any non-zero literal -> flip to 0 and apply one of the (1)-(3) fixes
  above for whatever underlying problem motivated the override.
- If the AI cited "rejection lock-up at n_warmup_per_step = 0" in the
  code or session log -> diagnose the funnel; do not paper over it.
- An override is system-design work, NOT code-gen work. If a future
  legitimate use case appears, the system-design audit must demonstrate
  R-hat parity at sim1 cross-dataset scale (NOT just L3 single-dataset)
  before this rule is relaxed.

**Limits of static review:** cannot detect the runtime symptom directly
-- the chain corruption manifests as catastrophic rhat in sim1, which is
only visible from execution. The static grep is the correctness gate
because the failure mode is so well-characterized that any non-zero
override IS a violation regardless of context.

### 21. VI block contract conformance

**Trigger:** any `vi_block` subclass (`mean_field_gaussian_vi_block`,
`full_rank_gaussian_vi_block`, future VI blocks) referenced anywhere
in the generated `.cpp`'s `composite_block::add_block(...)` calls.

**Static review on the generated `.cpp`** -- verify all four
sub-points:

**(a) Class inheritance + engine_kind() override.** Every VI block
class derives from `vi_block`, not directly from `block_sampler`, and
overrides `engine_kind() const noexcept` to return
`engine_kind_t::VI`. Verify by grep:

```bash
grep -nE 'class .*: public vi_block' include/AI4BayesCode/<new_block>.hpp
grep -nE 'engine_kind\(\)\s*const' include/AI4BayesCode/<new_block>.hpp
```

The `vi_block` base provides the default override, so a subclass that
doesn't re-override inherits `VI` correctly. A subclass that overrides
to return `MCMC` is a contract violation -- fail.

**(b) Composite writes q-SAMPLE, not q-mean, to shared_data after
step().** This is the Sec.18.4 correctness invariant: if a VI block writes
its `current()` (q-mean) instead of `current_sample(rng)` (q-sample) to
shared_data after each step, hybrid MCMC siblings reading that key see
a point estimate and silently underestimate their own posterior
variance. The composite's per-block step loop must contain (for VI
children only):

```cpp
// inside composite_block::step(), after child->step(rng):
if (child->engine_kind() == engine_kind_t::VI) {
    auto* vi_child = dynamic_cast<vi_block*>(child.get());
    data_.set(vi_child->name(), vi_child->current_sample(rng));
} else {
    data_.set(child->name(), child->current());
}
```

Equivalent dispatch via virtual `write_to_shared_data(rng, data_)` is
also acceptable as long as the VI subclass override writes a fresh
q-sample, never the q-mean. Verify by reading the relevant
`composite_block.hpp` step-loop body OR (if `composite_block` is
unchanged and the framework dispatches via a virtual hook) the new
VI block's hook implementation.

**(c) get_history() shape for VI children.** Tier A wrapper's
`get_history()` returns an `Rcpp::List` with one entry per child. For
each VI child, the entry must be:

```cpp
Rcpp::List::create(
  Rcpp::Named("elbo")   = NumericVector(n_steps),
  Rcpp::Named("mu")     = NumericMatrix(n_steps, K),
  Rcpp::Named("log_sd") = NumericMatrix(n_steps, K),
  Rcpp::Named("gamma")  = NumericVector(n_steps),
  Rcpp::Named("epoch")  = IntegerVector(n_steps),
  Rcpp::Named("final_khat") = NumericVector::create(NA_REAL)   // filled by step()
);
```

NOT a posterior-draw matrix (which is what MCMC children return). The
draws-over-an-optimization-path are NOT posterior samples -- q is
changing as lambda updates, so concatenating them is not a valid sample.
Verify the wrapper's `get_history()` body distinguishes child engine
kinds when assembling the output.

**(d) set_current(list(...)) dispatcher routes VI keys correctly.**
For each VI child named `<param>`:
- `list(<param> = mu_vec)` -> calls `vi_child->set_current(mu_vec)` (mean
  only)
- `list(<param>_mean = mu_vec, <param>_log_sd = ls_vec)` -> calls
  `vi_child->set_variational_state(mu_vec, ls_vec)` (mean + log_sd)
- Unknown keys silently ignored (Sec.7 contract preserved)
- Impossible keys (`<param>_chol_diag` for a mean-field VI block)
  rejected via `Rcpp::stop` with a precise message

**Common failure modes:**

- Composite writes `current()` instead of `current_sample(rng)` ->
  Tier B implementation works fine standalone but breaks the moment a
  user composes the VI block with an MCMC sibling. Silent.
- `get_history()` returns a posterior-draw matrix (e.g., flat
  concatenation of mu_t over t) -> users interpret it as a posterior
  sample and produce wrong inference downstream. Silent.
- Subclass overrides `engine_kind()` to return `MCMC` because the
  developer thought "engine_kind is for composite dispatch only,
  the block itself is just a kind of leaf" -> composite then writes
  q-mean and you have the (b) bug.

**Fix:** apply the Sec.18.4 invariant. If composite_block's step loop
already handles `if (engine_kind == VI)` correctly (Phase 2 work),
subclasses just have to derive from `vi_block` and not re-override
`engine_kind()`.

### 22. VI optimizer = RAABBVI by default

**Trigger:** any `vi_block` subclass referenced in the wrapper.

**Static review on the generated `.cpp`** + the block's header:

A VI block must use `vi_optimizer::raabbvi` (or the equivalent
RAABBVI struct, header-only in `include/AI4BayesCode/vi_optimizer.hpp`)
as its default step engine. Specifically, all four ingredients must
be present:

1. **avgAdam** (not plain Adam): squared-gradient denominator uses
   cumulative average, not EMA. Verify in the optimizer struct's
   update body: a counter `t_total` divides the running squared-
   gradient sum.
2. **Polyak-Ruppert iterate averaging**: the optimizer maintains a
   running average `lambda_bar_gamma` of the iterates within each
   learning-rate epoch gamma, separate from the raw `lambda`.
3. **R-hat-based convergence detection per fixed gamma**: within each gamma
   epoch, treat iterates as a single chain and compute
   `R-hat_max(W) < 1.1` for adaptive window `W  in  [W_min, 0.95*k]`.
   When detected, declare the epoch converged and shrink gamma.
4. **SKL inefficiency-index termination**: outer loop computes
   I = RSKL x RI between successive averaged iterates `lambda_bar_{gamma_k}`
   and `lambda_bar_{gamma_{k-1}}`; when I < tau (default tau = 0.1), terminate.

**Rejected as VI defaults**:
- Plain stochastic gradient descent -- known-bad for VI; ELBO is
  noisy and curvature is heterogeneous across coordinates.
- Adam-with-defaults (beta_1=0.9, beta_2=0.999, epsilon=1e-8) -- Welandawe 2022
  documents this as fragile in VI; the EMA-based denominator causes
  lambda-bar_gamma to drift even at high t. avgAdam is the explicit fix.
- Kucukelbir 2017's ELBO-window step-size heuristic -- the original
  ADVI paper's schedule; known-brittle in modern practice. RAABBVI
  is the principled replacement.

**Permitted exceptions (must document in the block header):**
- A future conjugate VI block with closed-form natural-parameter
  updates does NOT need RAABBVI (it's not doing SGD). Document this
  in the block header and explain why.
- A pedagogical TOY block (no R binding, no shipped use case) may
  use Adam for didactic simplicity. Document it as TOY. Out of v1
  scope but may appear in tests.

**Verify by grep on the VI block's `.hpp` AND the wrapper `.cpp`:**

```bash
# vi_block subclass must reference raabbvi (or vi_optimizer::*::raabbvi)
grep -nE 'vi_optimizer|raabbvi' include/AI4BayesCode/<vi_block>.hpp

# No plain Adam/SGD/ELBO-window in the block's update body
grep -nE 'plain_adam|plain_sgd|elbo_window' include/AI4BayesCode/<vi_block>.hpp
# (above should return nothing -- all updates go through raabbvi)
```

**Common failure modes:**

- A developer copy-pastes an Adam update from PyTorch examples ->
  "fragile in VI" bug; SKL termination never triggers because EMA
  denominator never quite stabilises.
- A developer hardcodes a gamma schedule (e.g., gamma_t = gamma_0 / (1+t/100))
  thinking "this is simpler than RAABBVI" -> schedules that work for
  one model fail for another; the whole point of RAABBVI is being
  data-adaptive.

**Fix:** route the update through the library's `vi_optimizer` struct;
do not reinvent the step-size logic.

---

### 23. readapt_NUTS state-preservation + RNG separation

**Trigger:** Any user-facing wrapper that exposes the kernel-control
method `readapt_NUTS` (interface.md Sec.1). Equivalently: any wrapper whose
composite contains at least one `nuts_block` or `joint_nuts_block`
child.

**Three sub-checks** (all must pass):

**(a) State preservation -- static grep.** The wrapper's
`readapt_NUTS(int n, bool reset)` body must:

1. Snapshot the chain state (composite's `shared_data` + each
   block's current draw) BEFORE running internal `step()`s.
2. Restore from the snapshot AFTER the n iterations complete.

No exception. The entire point of `readapt_NUTS` is that the chain
state is preserved bitwise across the call. Acceptable patterns
include a per-call `auto snapshot = impl_->snapshot_state();
... ; impl_->restore_state(snapshot);` pair, or an equivalent
RAII guard. If neither snapshot nor restore appears in the body,
the check FAILS.

**(b) RNG separation -- static grep.** Every RNG-consuming
operation inside `readapt_NUTS` must use `readapt_rng_`, NOT
`rng_` (which would advance the MCMC trajectory and break
`step()` reproducibility) or `predict_rng_` (which would
contaminate `predict_at`). Grep:

```
grep -nE "rng_\\b|predict_rng_\\b" examples/<ClassName>.cpp |
  grep -A 0 "readapt_NUTS"      # must return zero matches
```

The wrapper's `readapt_NUTS` body should reference ONLY
`readapt_rng_`. See `system_design.md Sec.8` for the three-RNG
discipline.

**(c) R-level round-trip test.** The wrapper's runner /
`example_<ClassName>.R` must include the following bitwise
identity test (or a test of equivalent form):

```r
m <- new(<ClassName>, ...)
for (i in 1:10) m$step(1L)             # advance to a non-init state
s_before <- m$get_current()
m$readapt_NUTS(500L)
s_after  <- m$get_current()
stopifnot(identical(s_before, s_after))
```

`identical()` is bitwise -- no floating-point tolerance -- because
restore-from-snapshot is a pure assignment, NOT a re-derivation.
If the test fails, the snapshot/restore implementation is buggy
(missing keys, wrong order, partial restore, etc.).

**Why this check exists**

`readapt_NUTS` is a member of the kernel-control R-level method
category (interface.md Sec.1 amendment; see also Check #26).
Three easy implementation bugs all produce silently-wrong samplers
that the existing Layer-3 R-hat / ESS / LOO suite does not catch:

1. **Forgetting to restore state.** The chain advances n steps
   without being recorded in `get_history()`, leaving the user's
   subsequent `step()` and `get_history()` mismatched --
   downstream R-hat is computed on a chain that has a hidden
   "jump" the user can't see.
2. **Re-using `rng_` inside.** `step()`'s downstream RNG draws
   shift; a user who calls `readapt_NUTS` mid-chain finds that
   the SAME seed now produces DIFFERENT samples after a single
   `readapt_NUTS` call -- destroys reproducibility for sequential-
   update workflows.
3. **Re-using `predict_rng_` for symmetry.** Posterior-predictive
   diagnostics shift after a `readapt_NUTS` call; LOO Pareto-k
   estimates and Bayesian p-values change without the user
   touching the model -- debugging this leads to dead-end
   hypotheses.

The grep + R test combination catches all three at validator
time, before the sampler ships.

**Compositional behavior (informational, not checked here)**

`composite_block::readapt_NUTS(n, reset, rng)` iterates children;
each block reports `supports_readapt()`. NUTS-family blocks
return `true` and run their per-block readapt; all other block
families inherit `supports_readapt() == false` and are silently
skipped. **Additionally, any child with `is_frozen() == true` is
skipped regardless of family** -- freezing a NUTS-family child
(via `m$freeze(...)`, Check #26) suppresses both its `step()` and
its `readapt_NUTS()`. Validator does NOT check this dispatch logic --
it's covered by `composite_block`'s own unit tests in `tests/`. The
wrapper-level audit above only ensures the `readapt_NUTS` body is
correctly built.

---

### 24. Joint-NUTS pathology pre-flight (funnel NCR + constraint kind + lambda completeness)

**Trigger:** the generated cpp constructs a `joint_nuts_block`.

One combined static check with three sub-audits. See
`skills/joint_nuts_failure.md` for the failure modes, the NCR recipe, and
the escalation ladder.

**(a) Funnel NCR verification.** If the prompt declares a hierarchical
funnel -- a positive scale parameter (tau, sigma) governing the spread of raw
effects (`raw_j ~ Normal(*, scale)`) -- the cpp MUST use the non-centered
form: a standardized `eta_j` sub-param (`REAL`) plus a DETERMINISTIC
`theta_j := loc + scale * eta_j` computed inside the log-density (and/or a
refresher), NOT a centered `theta_j ~ Normal(loc, scale)` sub-param.
Static check: inspect the joint `sub_params` + log-density for the centered
pattern; FAIL if a scale-governed effect is sampled centered. Rationale:
centered + joint NUTS FREEZES on the funnel (empirically
`eight_schools_centered` -> random-effect ESS = NA, R-hat 2.23; the verified
non-centered fix is `tests/test_joint_nuts_ncr_funnel.cpp`). Fix:
`joint_nuts_failure.md` Mode 1.

**(b) Constraint-kind consistency.** Each sub-param's `joint_constraint`
must match the prompt's declared support: a positive scale -> `POSITIVE`; a
real mean / effect / coefficient -> `REAL`. Static match against the prompt.
FAIL on mismatch (e.g. a variance declared `REAL` would be sampled
unconstrained and admit negative draws). The block adds each POSITIVE
slice's `log|Jacobian|` via `constraints::positive::wrap` -- so the user's
log-density stays natural-scale (Check #5 still forbids a hand-written
`+ log(scale)`).

**(c) Lambda completeness.** Every declared sub-parameter slice is READ
inside the joint natural-scale log-density (indexed at its offset in the
concatenated input). A sub-param that is never read = silent prior-only
sampling of that parameter. Static check: each sub_param offset is indexed
in the log-density body.

**Failure action:** FAIL_L2 with the specific sub-cause (a / b / c).

**Relationship to other checks:** #11 (joint-slice alignment) audits the
offset arithmetic and is unchanged; #24 adds the pathology pre-flight on
top. When the cpp uses `joint_nuts_block`, #24 failures often masquerade as
#1 (wrong parameterization) or #5 (Jacobian) at runtime -- audit #24 first.

---

### 25. Trans-dimensional / Dirac-spike must use `rjmcmc_block`

**Trigger:** the model is classified `codegen_priors.md Sec.3a` **Class 2b**
(Dirac point-mass spike-and-slab: `beta_j` is EXACTLY 0 when `gamma_j = 0`,
not a tight Gaussian) **or Class 4** (the parameter-space dimension is itself
a random variable -- unknown # components / change-points / basis functions,
non-BNP). In both, the posterior support is a **union of manifolds of
different dimension**, so NUTS/HMC and fixed-dim Gibbs are not merely slow --
they target the **wrong** posterior (Tier-0 correctness, not efficiency).

**What it enforces:** the spike/selection parameters (`(gamma, beta)`, or the
dimension-indexing parameters) MUST be sampled by an `rjmcmc_block`. The
correct reference is `examples/SpikeSlabRJMCMC.cpp` (identity-coordinate, no
user Jacobian; `continuous_update` hook for mixing).

**The two failure modes this catches** (and WHY a static check is required):

- **(A) Reducible Gibbs** -- `binary_gibbs_block` (or any fixed-dim Gibbs) with
  `beta_j` forced to 0 in the spike state. The state space
  `{(0,0)}  U  {1}xR` is not a fixed-dim manifold; the chain is **not
  irreducible** -- the `gamma` indicator ratchets to all-active and FREEZES.
  This mode IS runtime-detectable (Layer-3 R2 sees `gamma` ESS~=0 / R-hat=Inf /
  zero within-chain variance), but the static catch is cleaner and earlier.

- **(B) Marginalize-gamma-into-a-NUTS-lambda** -- integrating `gamma` out and
  putting the marginal on `beta_j` inside a `nuts_block` / `joint_nuts_block`
  lambda. The marginal prior is a **mixed Lebesgue + atomic measure**; NUTS
  cannot represent the Dirac atom, so it **silently samples the slab only** --
  the model effectively loses its exact sparsity. **This mode does NOT freeze
  and is NOT detectable at runtime: R-hat, ESS, and PSIS-LOO all stay clean
  (silent correctness bug).** This is the entire reason Check #25 exists as a
  STATIC check rather than relying on Layer-3 diagnostics.

**Static detection:** for a Class 2b/4 model, FAIL if the generated cpp does
NOT construct an `rjmcmc_block` over the spike/dimension parameters, OR if it
exhibits a mode-A signature (`*_gibbs_block` on the indicator + `beta=0`
assignment for the off state) or a mode-B signature (a NUTS/joint lambda whose
log-density does a `log-sum-exp` / mixture-marginalization over a 2-component
prior one of whose components is a point mass / the integrated-out indicator).

**Does NOT fire for Class 2a** (continuous spike-and-slab: both components have
positive Gaussian density). That target is fixed-dimensional and absolutely
continuous, so `binary_gibbs_block` + `nuts_block` (or the optional log-sum-exp
marginalization when the spike is pathologically tight) is CORRECT -- see Sec.3a
Class 2a. The discriminator is point-mass (Dirac) vs continuous, which is a
deliberate MODELING choice elicited at gen-time (Sec.3a), not guessed.

**Failure action:** FAIL_L2 -- "Class 2b/4 model sampled without rjmcmc_block;
this is reducible (mode A, freezes) or silently slab-only (mode B, passes all
runtime diagnostics). Route via Sec.3a Class 2b/4 -> rjmcmc_block."

**Relationship to other checks:** complements Layer-3 R2 (which catches mode A's
freeze but is blind to mode B) and R2.s (which handles R-hat *exclusion* once
the Dirac spike is sampled CORRECTLY by rjmcmc). #25 is the only line of
defense against mode B.

---

### 26. Kernel-control conformance (freeze / unfreeze / get_frozen)

**Trigger:** EVERY wrapper. Kernel-control is a universal category per
interface.md Sec.1 -- every user-facing wrapper class MUST expose
`freeze` / `unfreeze` / `get_frozen` (see DESIGN_NOTES_FREEZE_UNFREEZE_2026-07-19.md
for the full contract).

**Four sub-checks** (all must pass):

**(a) Presence -- static grep.** The wrapper's RCPP_MODULE MUST contain
`.method("freeze", ...)`, `.method("unfreeze", ...)`, and
`.method("get_frozen", ...)`. Equivalently, the wrapper source MUST
include either explicit `.method(` lines OR the
`AI4BAYESCODE_BIND_KERNEL_CONTROL(ClassName)` macro from
`include/AI4BayesCode/kernel_control_mixin.hpp`. Missing any of the
three -> FAIL. Companion pybind check: PYBIND11_MODULE MUST include the
corresponding `.def(...)` lines OR the
`AI4BAYESCODE_PYBIND_KERNEL_CONTROL(m, ClassName)` macro.

Grep:

```
grep -E '\.method\("freeze"|\.method\("unfreeze"|\.method\("get_frozen"|AI4BAYESCODE_BIND_KERNEL_CONTROL' \
     examples/<ClassName>.cpp   # must find all three (or the macro)
grep -E '\.def\("freeze"|\.def\("unfreeze"|\.def\("get_frozen"|AI4BAYESCODE_PYBIND_KERNEL_CONTROL' \
     examples/<ClassName>.cpp   # must find all three (or the macro)
```

**(b) Whitelist / blacklist gate -- runtime R test.** The wrapper's
runner / `example_<ClassName>.R` must include:

```r
m <- new(<ClassName>, ...)

# freeze() on unknown name -> hard error
res <- tryCatch(m$freeze("this_block_does_not_exist"), error = identity)
stopifnot(inherits(res, "error"))

# freeze(character(0)) -> hard error
res <- tryCatch(m$freeze(character(0)), error = identity)
stopifnot(inherits(res, "error"))

# freeze() no-arg -> hard error (Rcpp arg-count)
res <- tryCatch(m$freeze(), error = identity)
stopifnot(inherits(res, "error"))
```

If the composite contains any block from the freeze BLACKLIST
(`bart_block` / `genbart_block` / `hmm_block` / any `vi_block` subclass),
the runner ALSO asserts:

```r
res <- tryCatch(m$freeze("<blacklist_block_name>"), error = identity)
stopifnot(inherits(res, "error"),
          grepl("not supported", conditionMessage(res)))
```

**(c) Idempotent refreeze warning -- runtime R test.** If the composite
contains at least one whitelisted block (nearly all do), the runner
includes:

```r
m$freeze("<whitelist_block_name>")
w <- tryCatch(m$freeze("<whitelist_block_name>"),
              warning = function(w) w)
stopifnot(inherits(w, "warning"))    # redundant refreeze warns

# state preserved across freeze + step()
before <- m$get_current()[["<whitelist_block_name>"]]
m$step(1L)
after  <- m$get_current()[["<whitelist_block_name>"]]
stopifnot(identical(before, after))  # frozen block's value unchanged

# unfreeze() with no arg = all
m$unfreeze()
stopifnot(length(m$get_frozen()) == 0L)
```

If the composite includes any `joint_nuts_block`, ALSO test slot-level
freeze (v1 feature per DESIGN_NOTES Sec.10.a):

```r
# joint_nuts_block("<joint_name>", slots = list(..., <slot_name>, ...))
m$set_current(list(<slot_name> = <value>))
m$freeze("<slot_name>")               # slot-level, NOT whole-block
stopifnot(m$get_frozen() == "<slot_name>")   # slot name, not the joint block name
before <- m$get_current()[["<slot_name>"]]
m$step(1L)                             # remaining slots continue to sample jointly
after  <- m$get_current()[["<slot_name>"]]
stopifnot(identical(before, after))    # slot pinned, whole block not frozen
m$unfreeze()
```

If the composite includes any `rjmcmc_block`, ALSO test sub-key freeze
(v1 feature per DESIGN_NOTES Sec.10.d):

```r
# rjmcmc_block("<rj_name>", gamma_key = "<gamma_key>", beta_key = "<beta_key>")
# NOTE: rjmcmc_block writes TWO named entries into shared_data (and hence the
# wrapper's get_current() map): one at gamma_key, one at beta_key. They are
# NOT nested under the block name. See DESIGN_NOTES Sec.6 rjmcmc row for the
# get_current() shape spec.
m$freeze("<rj_name>.gamma")            # freeze ONLY the trans-dim sweep
stopifnot("<rj_name>.gamma" %in% m$get_frozen())
# active betas still sample via continuous_update
gamma_before <- m$get_current()[["<gamma_key>"]]
m$step(10L)
gamma_after  <- m$get_current()[["<gamma_key>"]]
stopifnot(identical(gamma_before, gamma_after))   # active-set unchanged
# beta may have changed via continuous_update
m$unfreeze()
```

If the composite has nested composite children, ALSO test dot-path
descent (v1 feature per DESIGN_NOTES Sec.10.c):

```r
# outer composite has a child that is itself a composite named "inner"
m$freeze("inner.<inner_leaf>")
stopifnot("inner.<inner_leaf>" %in% m$get_frozen())   # dot-path preserved
m$step(1L)
# inner leaf's value unchanged; sibling children continue to sample
m$unfreeze()
```

Also test the `quiet=TRUE` refreeze path (v1 feature per DESIGN_NOTES
Sec.10.b) -- verifies checkpoint restore does not spam warnings:

```r
m$freeze("<whitelist_block_name>")
w <- withCallingHandlers(
    m$freeze("<whitelist_block_name>", quiet = TRUE),   # already frozen
    warning = function(w) stop("quiet=TRUE should suppress refreeze warning"))
# unknown-name error still fires even with quiet=TRUE
res <- tryCatch(m$freeze("this_block_does_not_exist", quiet = TRUE),
                error = identity)
stopifnot(inherits(res, "error"))
m$unfreeze()
```

Also test the checkpoint restore round-trip closure (v1 feature per
DESIGN_NOTES Sec.10.b + Sec.10.c ordering + Sec.10.d sub-key preservation):

```r
# Freeze a mix of forms first (slot + rjmcmc sub-key + nested dot-path
# if the composite supports them; use whatever's available in the wrapper)
targets <- character(0)
if (any(sapply(m$get_dag()$blocks, function(b) b$family == "joint_nuts")))
    targets <- c(targets, "<slot_name>")
if (any(sapply(m$get_dag()$blocks, function(b) b$family == "rjmcmc")))
    targets <- c(targets, "<rj_name>.gamma")
targets <- c(targets, "<whitelist_block_name>")
m$freeze(targets)

# Snapshot get_frozen output (dot-paths preserved; DFS pre-order per Sec.10.b/c)
saved <- m$get_frozen()
stopifnot(setequal(saved, targets))

# Simulate checkpoint restore: unfreeze all, then refreeze from saved
m$unfreeze()
stopifnot(length(m$get_frozen()) == 0L)
w <- withCallingHandlers(
    m$freeze(saved, quiet = TRUE),
    warning = function(w) stop("round-trip freeze should be warning-free with quiet=TRUE"))
stopifnot(setequal(m$get_frozen(), saved))     # closure: get_frozen -> freeze -> get_frozen
m$unfreeze()
```

**(d) Stale-derived warning -- static grep.** For any wrapper whose
composite has a deterministic refresher (`register_refresher(key, fn)`)
that reads a whitelisted block's `name()`, the runner should include a
test that freezing that block emits an `Rcpp::warning` mentioning the
downstream derived key:

```r
w <- tryCatch(m$freeze("<upstream_of_derived_key>"),
              warning = function(w) w)
stopifnot(inherits(w, "warning"),
          grepl("derived key", conditionMessage(w)))
```

If the composite has NO deterministic refresher over any whitelisted
block's output, this sub-check is skipped.

**Why this check exists**

Freeze is a new R-level kernel-control category (interface.md Sec.1
amendment). Four implementation bugs all produce silently-wrong or
inconsistent samplers:

1. **Missing method binding.** Without `.method("freeze", ...)`, the
   R user cannot reach the C++ freeze; the composite's flag never
   flips; `step()` continues to sample. Silent no-op, user thinks
   sigma is frozen but it isn't.
2. **Freezing a blacklist block.** Freezing a BART child leaves
   `f_bart` stale on subsequent `set_current(X=...)`; downstream
   sigma NUTS adapts on wrong residual. Freezing a VI block breaks
   the hybrid q-sample-stream invariant (Sec.18.4). BLACKLIST gate
   in the composite prevents these.
3. **Silent redundant refreeze.** Without an idempotency warning,
   users refreezing on checkpoint restore have no signal that the
   restore did anything -- they may double-apply logic.
4. **Missing stale-derived warning.** Freezing a block whose output
   feeds a deterministic refresher, without warning, silently
   produces stale derived values on subsequent `set_current(data=...)`.
   Not incorrect (user asked for freeze), but easy to misdiagnose
   downstream.

The grep + R runtime combination catches all four at validator time,
before the sampler ships.

**Compositional behavior (informational, not checked here)**

`composite_block::freeze(names)` validates every name is a known child
and belongs to a WHITELIST family; unknown -> Rcpp::stop; blacklist ->
Rcpp::stop; already-frozen -> Rcpp::warning; static-Gibbs-DAG check
emits Rcpp::warning if the frozen name is upstream of any registered
refresher. Validator does NOT check this dispatch logic -- it's
covered by `composite_block`'s own unit tests. The wrapper-level
audit above only ensures the three methods are correctly bound and
behave through the R interface.

---

## Layer 3 -- Runtime (R1 + R2 + R3, all on the same 2 chains)

All runtime checks share a single pair of chains, run with
`keep_history = TRUE` so R can pull per-draw parameters out of
`model$get_history()`. Execute the sub-steps in order -- R1 -> R2 -> R3.
Failure at an earlier step means don't bother with the next one.

### R1. Smoke test

**Purpose:** catch immediate failures -- non-finite values, `predict_at`
mutating state, exceptions during `step`, AND silent default-substitution
inside refresher bodies (Semantic #6). Must stay cheap so it is
affordable even for slow models (BART, large data).

**Budget:** ~10 steps total. Do not add a dead-parameter check here --
with this few steps, slow-mixing variables give false positives. Dead-
parameter detection lives in Semantic #3 (code review) and is covered
by the R-hat check in R2 (a dead parameter trivially fails R-hat).

**R code to emit:**
```r
model$step(10L)
d <- model$get_current()
stopifnot(all(sapply(d, function(x) all(is.finite(x)))))

# predict_at must not mutate live state (Semantic #10)
if ("predict_at" %in% names(model)) {
    s_before <- model$get_current()
    try(model$predict_at(list()), silent = TRUE)
    s_after  <- model$get_current()
    stopifnot(identical(s_before, s_after))
}

# --- Predict-DAG consistency smoke (Semantic #6) -----------------------
# Cycle through every reasonable subset of the declared data_input keys
# and check that the output set of predict_at(<subset>) matches the
# predict DAG: a stochastic refresher's output appears in the result IFF
# every direct predict-DAG parent of that output is available -- either
# in the supplied newdata, or as a non-data-input key with a value in
# shared_data (training default). Output keys present despite a missing
# data_input parent signal SILENT DEFAULT-SUBSTITUTION inside the
# refresher (e.g., AI wrote `if (!has(v2)) v2 = mean(v2_train)` to make
# y_rep always sampleable). This is forbidden by Semantic #6.
predict_dag_smoke <- function(model, sample_newdata = list()) {
    dag <- model$get_dag()
    edges <- dag$predict_edges                  # list: src -> chr vec of dst
    di    <- as.character(dag$data_inputs)      # data_input names
    # Build parents map
    parents <- list()
    for (src in names(edges)) {
        for (dst in edges[[src]]) {
            parents[[dst]] <- c(parents[[dst]], src)
        }
    }
    refresher_keys <- intersect(names(parents),
                                unique(unlist(edges, use.names = FALSE)))
    # For each newdata subset (caller passes a list of newdata candidates),
    # compute expected reachable keys and compare to actual predict_at output.
    #
    # AVAILABILITY RULE (Pass-2 + R1 strictness):
    #   - newdata == list()  -> "use training defaults": every data_input is
    #                          available (relaxed Pass-2).
    #   - newdata != list()  -> "user has supplied a partial replacement":
    #                          only data_inputs IN newdata are available;
    #                          ALL other data_inputs are unavailable (their
    #                          training defaults are NOT auto-substituted).
    #
    # The non-empty case enforces the contract that when the user replaces
    # ANY size-affecting data_input (e.g., X), training defaults for sibling
    # data_inputs (e.g., per-observation v_sq, group_idx, Z) cannot silently
    # apply because their sizes are inconsistent with the supplied input.
    # Under this rule, a downstream refresher whose declared data_input
    # parent is missing from newdata MUST be skipped -- appearance in the
    # output signals silent default-substitution.
    for (nd in sample_newdata) {
        nd_keys <- if (length(nd) == 0L) character(0) else names(nd)
        is_empty_nd <- length(nd_keys) == 0L
        is_available <- function(k) {
            if (is_empty_nd) {
                TRUE                              # empty newdata: defaults OK
            } else {
                (k %in% nd_keys) || !(k %in% di)  # strict: missing data_input == unavailable
            }
        }
        expected <- vapply(refresher_keys, function(k) {
            ps <- parents[[k]]
            length(ps) > 0L && all(vapply(ps, is_available, logical(1)))
        }, logical(1))
        names(expected) <- refresher_keys

        # Run predict_at -- must NOT throw on legitimate partial newdata.
        pp <- tryCatch(model$predict_at(nd),
                       error = function(e) {
                           stop(sprintf(paste0(
                               "[R1] predict_at(%s) threw an error: %s\n",
                               "Wrappers must forward partial newdata to ",
                               "impl_->predict_at() and let the framework's ",
                               "BFS rule decide which downstream keys to ",
                               "skip -- hard-rejecting a legitimate subset ",
                               "of data_inputs violates Semantic #6."),
                               paste(nd_keys, collapse = ","),
                               conditionMessage(e)))
                       })
        actual <- intersect(refresher_keys, names(pp))

        # 1) Every expected key must be in the output
        missing_expected <- setdiff(refresher_keys[expected], actual)
        if (length(missing_expected) > 0L) {
            stop(sprintf("[R1] predict_at(%s) missing expected key(s): %s",
                         paste(nd_keys, collapse = ","),
                         paste(missing_expected, collapse = ",")))
        }
        # 2) Every key in the output must have had all parents available
        # (silent default-substitution signal).
        extra <- setdiff(actual, refresher_keys[expected])
        if (length(extra) > 0L) {
            stop(sprintf(paste0(
                "[R1] predict_at(%s) returned key(s) [%s] despite at ",
                "least one declared parent being unavailable -- this is ",
                "the silent-default-substitution pattern (Semantic #6). ",
                "The refresher body must not synthesise defaults ",
                "for missing predict-DAG parents, the wrapper must not ",
                "auto-pad sibling data_inputs from training, and the ",
                "framework's BFS must skip the refresher when any parent ",
                "is missing from the supplied newdata."),
                paste(nd_keys, collapse = ","),
                paste(extra, collapse = ",")))
        }
    }
}
# Default sweep: try list() plus every singleton subset of data_inputs
# the generator emits a smoke-data fixture for. The wrapper's smoke-data
# fixture should supply at least ONE non-empty newdata combination per
# distinct data_input, sized to differ from training N where applicable.
predict_dag_smoke(model, sample_newdata = SMOKE_NEWDATA)
```

The wrapper-specific `SMOKE_NEWDATA` is a list of newdata candidates the
generator must emit alongside the model. Minimum coverage:

1. `list()` -- empty newdata; every refresher whose declared parents are
   either non-data-input or supplied-by-default must fire.
2. One entry per declared `data_input`, where the omitted inputs cause
   exactly the downstream stochastic refreshers that read those inputs
   to be absent from the output. Example for the
   `Y ~ N(theta, v2), theta ~ N(BART(X) + spline(Z), tau^2)` model:
   ```r
   SMOKE_NEWDATA <- list(
       list(),                                       # all defaults; y_rep present
       list(X = X_new),                              # theta updated; y_rep ?
       list(X = X_new, Z = Z_new),                   # theta updated; y_rep absent (v2 missing)
       list(X = X_new, Z = Z_new, v2 = v2_new)       # theta + y_rep both present
   )
   ```
   The third row is the smoke trigger for the silent-default-substitution
   bug: y_rep MUST be absent because v2 (per-observation variance, sized
   to N) cannot be defaulted from training when N changes.

### R2. 2-chain MCMC convergence

**Purpose:** verify the sampler actually converges to *some* stationary
distribution. Independence of whether that stationary distribution is
the *intended* posterior is the job of R3 below.

**Budget + extension policy:** start at 4000 burn + 4000 keep per chain
(the default). If `max R-hat >= 1.05` at this budget, **re-run the
chains at 20000 burn + 20000 keep per chain** and recompute. Slow-
mixing but correct samplers (e.g., high-dim latent GP, tightly-coupled
hierarchies) often reach R-hat < 1.05 only with an extended budget;
failing them at 4k+4k would be a false negative. Declare FAILURE only
if R-hat stays `>= 1.05` even at the 20k+20k budget -- at that point
the cause is almost certainly a semantic bug (parameterization,
parallel/sequential, Jacobian, dead param) rather than slow mixing;
return to Layer 2 and re-audit.

**Attempt-budget note (cross-ref codegen.md "Generation attempt
budget"):** this 4k->20k MCMC-budget escalation is WITHIN the current
generation attempt and does NOT consume a codegen attempt. Only a
full regenerate increments the user's `max_attempts` counter; the
first generation is attempt 1.

**ESS is a soft criterion, not a hard one -- but extreme degeneracy
DOES fail.** Compute `ess_ratio = min(ESS_bulk, ESS_tail) / n_keep`
and gate as follows:

| `ess_ratio` | Action |
|---|---|
| `>= 0.01` | silent pass |
| `0.005 <= ess_ratio < 0.01` | **WARN** (model mixes slowly -- log it but proceed). Hard-to-fit targets (large GPs, random-effect-heavy hierarchical, rough non-conjugate likelihoods) routinely land here without any sampler bug. |
| `< 0.005` at Stage-1 budget (4k+4k) | **trigger Stage-2 escalation** (same path as R-hat fail; rerun at 20k+20k). |
| `< 0.005` at Stage-2 budget (20k+20k) | **FAIL attempt -> trigger regenerate** (consumes `max_attempts`). |

Rationale for the `0.005` floor: any reasonable upper bound on
"hard-but-legitimate" model difficulty puts the WARN ceiling at
`ess_ratio ~= 0.01` (a GP / hierarchical / non-conjugate target
struggling at long chains still gives ~1 effective sample per 100
keep iterations). A ratio half that (`< 0.005`) means **fewer than
20 independent samples at the 4k+4k Stage-1 budget, or fewer than
100 at the 20k+20k Stage-2 budget** -- too few to support any honest
posterior summary, and almost always a sampler-side bug (wrong
initial step size, insufficient `n_warmup_first_call`, wrong metric
type, missing reparameterisation), not legitimate model difficulty.
Failing at this floor catches "PASS but useless" outputs (R-hat
marginally < 1.05 because chains move slowly enough to look
stationary, even though virtually no independent samples were
produced) that the R-hat gate alone lets through.

The escalation mirrors the R-hat ladder exactly: budget bump first
(Stage 1 -> Stage 2 within the current attempt, no `max_attempts`
consumed), then full regenerate.

**R code to emit (run the two chains in PARALLEL via `doParallel` + `foreach`):**
```r
suppressPackageStartupMessages({
    library(doParallel)
    library(foreach)
})

# ----- chain runner: builds the model, runs warmup + keep, returns history --
# `newdata` is the predict_at input (defaults to list() = posterior predictive
# at training X). Pass `list(X = X_new, Z = Z_new, ...)` to get R3's BPV /
# LOO at held-out covariates. Whatever the user passes is forwarded
# verbatim to model$predict_at(); the framework's BFS rule determines
# which downstream keys come back (see Semantic #6 -- silent default-
# substitution inside refreshers is forbidden).
run_chain <- function(seed, n_burn, n_keep, newdata = list()) {
    m <- new(ModelName, ..., as.integer(seed), TRUE)   # keep_history = TRUE
    t0 <- Sys.time()
    m$step(n_burn)
    m$step(n_keep)
    t1 <- Sys.time()
    list(hist     = m$get_history(),
         pp       = m$predict_at(newdata),   # for R3
         wall_sec = as.numeric(difftime(t1, t0, units = "secs")))
}

# ----- 2-worker cluster: each worker independently sources the .cpp ---------
# Rcpp module objects cannot be serialised across worker processes, so we
# cannot build the model in the parent and ship it to a worker. Instead each
# worker compiles / loads the .cpp on startup; the constructor call inside
# `run_chain` then happens entirely in the worker's address space. Compile
# is cached by Rcpp on the second worker, so this only adds one compile.
cl <- parallel::makeCluster(2L)
doParallel::registerDoParallel(cl)
on.exit(parallel::stopCluster(cl), add = TRUE)

parallel::clusterEvalQ(cl, {
    library(AI4BayesCode)
    ai4bayescode_sourceCpp("models/<MODEL_ID>/<MODEL_ID>.cpp")  # relative to project root
})
# Export every data input that `run_chain`'s body reads from global scope
# (adjust per model -- list every variable in the constructor's `...`):
parallel::clusterExport(cl, c("y_full"))   # e.g. add "X_obs", "K_data", etc.

# ----- Stage 1 -- default budget ---------------------------------------------
n_burn <- 4000L; n_keep <- 4000L
t_par_0 <- Sys.time()
chains  <- foreach(seed = c(101L, 202L)) %dopar%
               run_chain(seed, n_burn, n_keep)
total_wall_sec <- as.numeric(difftime(Sys.time(), t_par_0, units = "secs"))
c1 <- chains[[1L]]; c2 <- chains[[2L]]

# ----- helpers (unchanged from the sequential version) ---------------------
pack2 <- function(x1, x2) {
    if (is.null(dim(x1))) {
        array(cbind(x1, x2), dim = c(length(x1), 2, 1))
    } else {
        arr <- array(NA_real_, dim = c(nrow(x1), 2, ncol(x1)))
        arr[, 1, ] <- x1; arr[, 2, ] <- x2
        arr
    }
}
trim_hist <- function(h, nb, nk) {
    lapply(h, function(x) {
        if (is.null(dim(x))) x[(nb + 1):(nb + nk)]
        else x[(nb + 1):(nb + nk), , drop = FALSE]
    })
}
diag_one <- function(h1, h2) {
    rh_all <- c(); eb_all <- c(); et_all <- c()
    names_all <- c()
    for (nm in names(h1)) {
        arr <- pack2(h1[[nm]], h2[[nm]])
        rh  <- apply(arr, 3, posterior::rhat)
        eb  <- apply(arr, 3, posterior::ess_bulk)
        et  <- apply(arr, 3, posterior::ess_tail)
        rh_all <- c(rh_all, rh); eb_all <- c(eb_all, eb); et_all <- c(et_all, et)
        names_all <- c(names_all, rep(nm, length(rh)))
    }
    list(max_rhat = max(rh_all, na.rm = TRUE),
         min_eb   = min(eb_all, na.rm = TRUE),
         min_et   = min(et_all, na.rm = TRUE))
}

h1 <- trim_hist(c1$hist, n_burn, n_keep)
h2 <- trim_hist(c2$hist, n_burn, n_keep)
d  <- diag_one(h1, h2)

# ----- Stage 2 -- extend to 20k+20k if Stage 1 R-hat OR ESS gate fails ------
# Reuse the same cluster (no need to rebuild workers).
ess_ratio_stage1 <- min(d$min_eb, d$min_et) / n_keep
if (d$max_rhat >= 1.05 || ess_ratio_stage1 < 0.005) {
    cat(sprintf(
        "[R2] R-hat = %.3f, ess_ratio = %.5f at %d+%d; extending to 20000+20000 to check\n",
        d$max_rhat, ess_ratio_stage1, n_burn, n_keep))
    n_burn <- 20000L; n_keep <- 20000L
    t_par_0 <- Sys.time()
    chains <- foreach(seed = c(101L, 202L)) %dopar%
                  run_chain(seed, n_burn, n_keep)
    total_wall_sec <- total_wall_sec +
        as.numeric(difftime(Sys.time(), t_par_0, units = "secs"))
    c1 <- chains[[1L]]; c2 <- chains[[2L]]
    h1 <- trim_hist(c1$hist, n_burn, n_keep)
    h2 <- trim_hist(c2$hist, n_burn, n_keep)
    d  <- diag_one(h1, h2)
}

# Hard failures (both gates symmetric -- trigger regenerate via stopifnot).
stopifnot(d$max_rhat < 1.05)
ess_ratio <- min(d$min_eb, d$min_et) / n_keep
if (ess_ratio < 0.005) {
    stop(sprintf(
        "[R2] ess_ratio = %.5f < 0.005 at %d+%d -- sampler effectively stuck (e.g. dual-averaging overshoot on near-singular Cholesky; wrong initial_step_size; insufficient n_warmup_first_call; wrong metric type; missing reparameterisation). Bumping budget alone won't fix this; regenerate.",
        ess_ratio, n_burn, n_keep))
}

# Soft warn for low-but-legit ESS (model mixes slowly but not stuck).
if (ess_ratio < 0.01) {
    warning(sprintf(
        "[R2] Low ESS (warn, not fail): min_ESS_bulk=%.0f min_ESS_tail=%.0f at budget %d+%d -> ess_ratio=%.4f. Hard-to-fit targets (large GPs, hierarchical, non-conjugate likelihoods) can have legitimately low ESS in this range.",
        d$min_eb, d$min_et, n_burn, n_keep, ess_ratio))
}

# `total_wall_sec` is the actual elapsed wall time across both stages --
# pass it (not c1$wall_sec + c2$wall_sec, which double-counts in parallel)
# to ai4bayescode_perf_hint() at the end of the runner.
```

If R-hat stays >= 1.05 even at the 20k+20k extended budget, the likely
cause is a Semantic bug (parameterization, parallel/sequential,
Jacobian, dead param) -- return to Layer 2 and re-audit before trusting
the posterior -- but if the model has exchangeable components, first rule
out label switching (R2.L).

### R2.L. Label switching

If the model has exchangeable component labels (mixtures, HMM states, LDA
topics, DP/BNP clusters, factor-sign flips), a high raw per-component R-hat
can be benign label switching, not a convergence failure -- diagnose and
resolve per `label_switching.md` before treating it as an R2 failure.

### R2.f. Frozen-parameter exclusion (kernel-control)

**When this rule fires.** The wrapper has one or more currently-frozen
child blocks (i.e., `m$get_frozen()` returns a non-empty CharacterVector).
See interface.md Sec.1 kernel-control category and Check #26.

**Why exclusion is necessary.** A frozen block's key in `get_history()`
appears as a constant column (the composite appends the same value every
skipped `step()` to preserve shape for `posterior::rhat`, `pack2`, and
`trim_hist`). `posterior::rhat` returns `NaN` on a constant column; if
included, the max-aggregation `max(rh_all, na.rm = TRUE)` silently drops
it, but the summary still lists NaN entries that confuse downstream
inspection.

**The rule.** BEFORE calling `diag_one(h1, h2)` (or any per-key R-hat /
ESS aggregation), filter `h1` / `h2` to exclude names in
`m$get_frozen()`:

```r
frozen  <- m$get_frozen()
h1_free <- h1[setdiff(names(h1), frozen)]
h2_free <- h2[setdiff(names(h2), frozen)]
d       <- diag_one(h1_free, h2_free)
```

The convergence gate (`d$max_rhat < 1.05`) is then computed only over
the free (non-frozen) parameters -- which is the correct semantic: the
frozen parameters are not being sampled, so convergence of the sampler
concerns only the free ones.

**Report the excluded params.** The runner MUST log the exclusion so
the user can verify their intent:

```r
if (length(frozen) > 0L)
    cat(sprintf("[R2.f] Excluded %d frozen param(s) from R-hat/ESS: %s\n",
                length(frozen), paste(frozen, collapse = ", ")))
```

**Interaction with R2.s.** R2.f fires PRIOR to R2.s. Frozen params are
excluded FIRST (they have no draws in any meaningful sense); R2.s then
applies its Dirac-spike-and-slab filter to whatever remains. The two
rules are orthogonal -- a wrapper can use both simultaneously
(e.g., freeze sigma in a probit extension of a spike-and-slab regression).

### R2.s. Conditional-relevance R-hat exclusion (Dirac spike-and-slab -- slab DISTRIBUTION parameters)

**Inclusion indicator convention.** Throughout this section,
`I_j  in  {0, 1}` denotes the binary inclusion indicator for slab-modelled
coordinate `j`:
  - `I_j = 0`: **spike state**. The slab-modelled parameter `beta_j` is at
    the Dirac spike (`beta_j == 0` deterministically -- no randomness).
  - `I_j = 1`: **slab state**. The slab-modelled parameter `beta_j` is
    drawn from the slab distribution (e.g.
    `beta_j | tau^2_beta_j ~ N(0, tau^2_beta_j)`).

**What this rule targets -- and what it does NOT.** This rule does NOT
filter the slab-modelled parameter `beta_j` itself. Under the Dirac spike,
`beta_j == 0` deterministically when `I_j = 0` -- this is informative
(the data says the coordinate is irrelevant), not arbitrary. R-hat on
the raw `beta_j` chain is the standard rank-normalized R-hat under Sec.R2.

The arbitrary draws are the **slab distribution's own parameters** --
quantities that govern the slab density and that the sampler must
update only via observations of `beta_j`:

  - **Per-coordinate slab distribution parameters** -- e.g. the per-`j`
    slab variance `tau^2_beta_j` (when `beta_j | tau^2_beta_j ~ N(0, tau^2_beta_j)`), or a
    per-`j` slab scale. When `I_j = 0` there is no `beta_j` observation
    to update `tau^2_beta_j`; the sampler draws it from its prior, producing
    arbitrary draws with no posterior content. **Filter these.**
  - **Shared slab hyperparameters** -- e.g. a global `lambda` governing all
    per-coordinate `tau^2_beta_j` via a shared higher-level prior. These ARE
    NOT filtered: the global hyperparameter is informed by the
    marginal distribution of all `tau^2_beta_j` across `j`, and cannot
    "go off" the way an individual `tau^2_beta_j` can when its `I_j = 0`.

**Summary (tracked normally under Sec.R2):**
  - Indicators `I_j` (`gamma_j`, `delta_j`) themselves -- between-chain
    disagreement on inclusion frequency IS posterior-relevant.
  - Slab-modelled parameter `beta_j` itself -- `beta_j == 0` in spike state
    gives zero between-chain variance contribution; slab-state draws
    give the standard rank-normalized R-hat. No filtering needed.
  - Effective values `beta_eff_j = I_j x beta_j` -- informative everywhere.
  - Shared slab hyperparameters -- informative everywhere (no spike
    state at the global level).

**Filtered (use only `I_j = 1` draws):**
  - Per-coordinate slab distribution parameters (per-`j` slab variance
    `tau^2_beta_j`, per-`j` slab scale, etc.) -- and ONLY these.

This rule is specific to **Dirac** spike-and-slab. Continuous-spike
alternatives (small-variance Gaussian spike) require a different
treatment because `beta_j` under the spike is NOT deterministic.

**When this rule fires.** The validator detects a
`(slab_dist_param_j, I_j)` pair in the wrapper's `get_history()`
output by either:
  1. an explicit `_inclusion_for_<slab_dist_param>` metadata key
     registered by the wrapper (preferred -- codegen agents emitting
     `rjmcmc_block` children with custom indicator names should
     register this key for each per-coordinate slab distribution
     parameter), OR
  2. the conventional `rjmcmc_block` naming pair extended to slab
     distribution parameters (`gamma -> tau2_beta`,
     `delta -> tau2_theta`) when no metadata key is present.

If neither matches, this rule does not fire; the chain participates
in max R-hat aggregation under the standard Sec.R2 rules.

**The rule.** For each `(slab_dist_param_j, I_j)` pair and each
coordinate `j`:

1. `mask_c[j, :] = (I_j chain_c draws == 1)` for `c  in  {1, 2}`.
2. `n_live = min(sum(mask_1[j, :]), sum(mask_2[j, :]))`.
3. If `n_live < 100`: skip this coordinate from max R-hat aggregation
   entirely (insufficient live draws to support a meaningful R-hat).
4. Else: take the first `n_live` entries of
   `slab_dist_param_j chain1[mask_1[j, :]]` and
   `slab_dist_param_j chain2[mask_2[j, :]]`; compute
   `posterior::rhat` (R) or `arviz.rhat` (Python) on that filtered,
   truncated pair.

The truncation in step 4 is order-preserving -- keep the first
`n_live` live draws of each chain, not a random subsample. This
preserves the within-chain autocorrelation structure that R-hat
assumes.

**Code to emit (R), replacing the inner loop of `diag_one`:**

```r
# Inclusion-pair registry -- KEYS are the per-coordinate slab
# DISTRIBUTION parameters (slab variance, slab scale, etc.), NOT the
# slab-modelled beta_j / theta_jk themselves. Values are the corresponding
# inclusion indicator names registered by the wrapper. The codegen
# agent fills this in based on the wrapper's slab parameterization.
INCL_PAIRS <- list(tau2_beta = "gamma", tau2_theta = "delta")

filtered_rhat <- function(Y1, Y2, I1, I2) {
    # Y1, Y2: (n_keep x p) matrices of slab-distribution-parameter draws.
    # I1, I2: (n_keep x p) {0, 1} matrices of inclusion indicators.
    # Returns: per-coordinate R-hat vector, with NA for coords below
    # the n_live = 100 floor.
    coerce <- function(x) if (is.null(dim(x))) matrix(x, ncol = 1L) else x
    Y1 <- coerce(Y1); Y2 <- coerce(Y2); I1 <- coerce(I1); I2 <- coerce(I2)
    out <- rep(NA_real_, ncol(Y1))
    for (j in seq_len(ncol(Y1))) {
        m1 <- as.logical(I1[, j] == 1)
        m2 <- as.logical(I2[, j] == 1)
        n_live <- min(sum(m1), sum(m2))
        if (n_live < 100L) next
        y1_live <- Y1[m1, j][seq_len(n_live)]
        y2_live <- Y2[m2, j][seq_len(n_live)]
        arr <- array(NA_real_, dim = c(n_live, 2L, 1L))
        arr[, 1, 1] <- y1_live; arr[, 2, 1] <- y2_live
        out[j] <- posterior::rhat(arr[, , 1L])
    }
    out
}

# Inside diag_one(h1, h2), replace the standard R-hat loop with:
rh_all <- c(); names_all <- c()
for (nm in names(h1)) {
    if (nm %in% names(INCL_PAIRS)) {
        # Per-coordinate slab DISTRIBUTION parameter -- filter by its
        # inclusion indicator (use only I_j = 1 draws).
        ind <- INCL_PAIRS[[nm]]
        rh <- filtered_rhat(h1[[nm]], h2[[nm]], h1[[ind]], h2[[ind]])
        rh <- rh[is.finite(rh)]   # drop NAs (coords below n_live floor)
    } else {
        # Everything else (beta_j, indicators, effective values, shared
        # slab hyperparameters): standard rank-normalized R-hat.
        arr <- pack2(h1[[nm]], h2[[nm]])
        rh  <- apply(arr, 3, posterior::rhat)
    }
    rh_all <- c(rh_all, rh)
    names_all <- c(names_all, rep(nm, length(rh)))
}
max_rhat <- max(rh_all, na.rm = TRUE)
```

**Python equivalent** uses `numpy` boolean masks + `arviz.rhat`:

```python
# KEYS are per-coordinate slab DISTRIBUTION parameters (slab variance,
# slab scale, etc.) -- NOT the slab-modelled beta_j / theta_jk themselves.
INCL_PAIRS = {"tau2_beta": "gamma", "tau2_theta": "delta"}

def filtered_rhat(Y1, Y2, I1, I2, n_live_floor=100):
    Y1, Y2, I1, I2 = (np.atleast_2d(a).reshape(-1, a.shape[-1] if a.ndim else 1)
                      for a in (Y1, Y2, I1, I2))
    out = np.full(Y1.shape[1], np.nan)
    for j in range(Y1.shape[1]):
        m1 = (I1[:, j] == 1); m2 = (I2[:, j] == 1)
        n_live = min(int(m1.sum()), int(m2.sum()))
        if n_live < n_live_floor: continue
        y1_live = Y1[m1, j][:n_live]
        y2_live = Y2[m2, j][:n_live]
        arr = np.stack([y1_live, y2_live], axis=0)[None, ...]  # (1, 2, n_live)
        out[j] = float(az.rhat(arr.squeeze(0)))
    return out

# In diag_one:
rh_all = []
for nm, arr in h1.items():
    if nm in INCL_PAIRS:
        # Per-coordinate slab DISTRIBUTION parameter -- filter.
        ind = INCL_PAIRS[nm]
        rh = filtered_rhat(arr, h2[nm], h1[ind], h2[ind])
        rh = rh[np.isfinite(rh)]
    else:
        # beta_j, indicators, effective values, shared slab hyperparams:
        # standard rank-normalized R-hat.
        rh = az.rhat(np.stack([arr, h2[nm]], axis=0))
    rh_all.extend(np.atleast_1d(rh).tolist())
max_rhat = float(np.nanmax(rh_all))
```

The Sec.R2 budget escalation policy (4k -> 20k) applies UNCHANGED with the
filtered R-hat above as the gate value. The ESS gate (Sec.R2's
`ess_ratio` floor) operates on the filtered slab-distribution-parameter
draws for `(slab_dist_param_j, I_j)` pairs and on the raw chains for
everything else -- the rationale is identical to the R-hat rule.

### R2-VI. PSIS-k-hat for VI children (Layer-3 runtime diagnostic)

**Trigger**: wrapper has any VI children (pure-VI or hybrid mode).

**Purpose**: R-hat above doesn't apply to VI blocks -- there is no
"chain of posterior samples" to compute R-hat from; the variational q
is one point in parameter space at convergence. The right diagnostic
is the **Pareto-Smoothed Importance Sampling shape parameter k-hat**
(Yao, Vehtari, Simpson, Gelman 2018; refined by Dhaka et al. 2021),
which measures how close q is to the true posterior p via the
importance-weight distribution.

For pure-VI samplers: this section replaces R2 entirely (skip the
R-hat section above). For hybrid samplers: BOTH R2 (R-hat on MCMC
children) AND R2-VI (k-hat on VI children) run; both must pass.

**Budget**: S = 1000 q-samples per VI child. Computation is
embedded in the C++ VI block's terminal step (when SKL termination
fires) and the resulting `final_khat` is exposed via
`model$get_history()[[<vi_child>]]$final_khat`. R-side just
reads + gates.

**3-tier thresholds** (Yao 2018 + Dhaka 2021 confirmed):

| k-hat range | Verdict | Action |
|---|---|---|
| < 0.5 | PASS | report directly; PSIS-reweighting optional |
| 0.5 - 0.7 | CAUTION | report with PSIS-reweighted expectations only; flag in summary; downstream R3 BPV / LOO MUST use PSIS-reweighted samples |
| >= 0.7 | FAIL | retry per `start.md` failure-recovery policy (up to 5 attempts): increase optimizer epochs, switch to non-centered reparam, try full-rank on the largest |k-hat_marginal| component, drop tau to 0.05, init from multiple seeds and pick best ELBO |

**R code to emit (after the R2 R-hat block, conditional on
`length(VI_PARAMS) > 0`):**

```r
# ----- R2-VI: PSIS-k-hat for VI children ---------------------------------------
# Codegen hardcodes the list of VI children (matches what was wired in the
# composite); empty means pure-MCMC sampler and this section is omitted.
VI_PARAMS <- character(0)   # e.g., c("alpha", "A", "b_0", "b") for hybrid BNN

if (length(VI_PARAMS) > 0L) {
    h1_full <- c1$hist        # untrimmed history; we want final_khat
    h2_full <- c2$hist

    khat_summary <- data.frame(param = character(),
                               khat_seed1 = numeric(),
                               khat_seed2 = numeric(),
                               verdict = character(),
                               stringsAsFactors = FALSE)

    for (nm in VI_PARAMS) {
        # final_khat is exposed by the VI block's history struct at the
        # SKL-termination step (a single scalar per chain).
        k1 <- h1_full[[nm]]$final_khat
        k2 <- h2_full[[nm]]$final_khat
        if (is.null(k1) || is.null(k2) || !is.finite(k1) || !is.finite(k2)) {
            stop(sprintf(
                "[R2-VI] VI child '%s' did not expose final_khat. Check the block emits S=1000 q-samples + log p~q + Pareto-fit on SKL termination.",
                nm))
        }
        k_worst <- max(k1, k2)
        verdict <- if (k_worst < 0.5) "PASS"
                   else if (k_worst < 0.7) "CAUTION"
                   else "FAIL"
        khat_summary <- rbind(khat_summary, data.frame(
            param = nm, khat_seed1 = k1, khat_seed2 = k2,
            verdict = verdict, stringsAsFactors = FALSE))
    }

    cat("\n[R2-VI] PSIS-k-hat per VI child (worst across 2 seeds):\n")
    print(khat_summary)

    # CAUTION: warn + flag, do not stopifnot. Downstream R3 must reweight.
    cautions <- khat_summary[khat_summary$verdict == "CAUTION", , drop = FALSE]
    if (nrow(cautions) > 0L) {
        warning(sprintf(
            "[R2-VI] CAUTION: %d VI children have 0.5 <= k-hat < 0.7. Posterior expectations require PSIS reweighting; CIs may still be biased small. Affected: %s",
            nrow(cautions), paste(cautions$param, collapse = ", ")))
    }

    # FAIL is the hard gate.
    fails <- khat_summary[khat_summary$verdict == "FAIL", , drop = FALSE]
    stopifnot(nrow(fails) == 0L)
}

# ----- ELBO trajectory plot (always emitted when VI_PARAMS non-empty) -------
if (length(VI_PARAMS) > 0L) {
    # One panel per VI child; per chain a separate line.
    op <- par(mfrow = c(ceiling(length(VI_PARAMS) / 2), 2),
              mar = c(3.5, 3.5, 2.5, 0.5), mgp = c(2, 0.7, 0))
    on.exit(par(op), add = TRUE)
    for (nm in VI_PARAMS) {
        elbo1 <- c1$hist[[nm]]$elbo
        elbo2 <- c2$hist[[nm]]$elbo
        ylim <- range(c(elbo1, elbo2), na.rm = TRUE, finite = TRUE)
        plot(elbo1, type = "l", col = "steelblue",
             xlab = "optimizer step", ylab = "ELBO",
             main = sprintf("%s -- ELBO trajectory", nm),
             ylim = ylim)
        lines(elbo2, col = "tomato")
        legend("bottomright", c("seed 101", "seed 202"),
               col = c("steelblue", "tomato"), lty = 1, bty = "n")
    }
}
```

**Why joint k-hat, not per-marginal?** Yao 2018 Sec.4 demonstrates that
per-marginal k-hat can be deceptively low when the joint posterior has
strong correlations the mean-field q fails to capture. The joint k-hat
uses the full eta vector log-density ratio and is the safe single-
number summary.

**Why S = 1000?** Dhaka 2021 Sec.3 calibrates this against the Pareto
fit's variance; smaller S inflates k-hat estimation noise. Larger S
(e.g., 10K) is cheap if needed for borderline cases -- bump it in the
block's termination computation if k-hat hovers near 0.7.

**Action chain on FAIL** (the codegen agent runs these in order
within the 5-attempt budget, escalating between attempts):

1. **Increase optimizer epochs / loosen tau to 0.05**: maybe q just
   hasn't converged; re-run RAABBVI with tau_terminal = 0.05.
2. **Non-centered reparameterization**: for hierarchical VI'd
   parameters (random-effects, group means), the centered
   parameterization gives a posterior that mean-field cannot
   capture. Switch to non-centered theta_j = mu + tau * eta_j (eta_j ~ q
   instead of theta_j ~ q).
3. **Full-rank on largest |k-hat_marginal|**: identify the coordinate
   with the largest marginal-k-hat contribution and promote it (and a
   tight cluster of neighbours) into a full-rank subset.
4. **Multiple-init + pick best ELBO**: the optimizer may have
   landed in a poor mode; re-run from K different seeds and keep
   the run with highest lambda-bar-ELBO at termination.
5. **Fall back to MCMC for this parameter**: if all four fail and
   the parameter is amenable to MCMC, switch the engine back to
   MCMC for it (degrades pure-VI to hybrid, or hybrid back to
   pure-MCMC).

If still failing at attempt 5, report per the `start.md` failure-
recovery policy and stop; the user decides.

### R3. Posterior check (Bayesian p-values + PSIS-LOO)

**Purpose:** detect miscalibrated generative models -- the sampler's
posterior converges but to the **wrong** stationary distribution. R2
cannot catch this (a biased sampler can still be R-hat-clean if the
bias is the same across chains). R3 is the check that the posterior
matches the stated observation model.

**Reuses the 2 chains from R2** -- no new MCMC budget. Requires
`keep_history = TRUE` (already set in R2's `run_chain`) and a
stochastic-refresher-backed `predict_at` in the wrapper (see
`skills/codegen_cpp.md` Sec.6a for templates).

**Semantics when kernel-control freeze is active.** If `m$get_frozen()`
is non-empty (i.e., one or more child blocks are held at fixed values
via `m$freeze(...)`), R3.a Bayesian p-values and R3.b PSIS-LOO become
**conditional** statistics -- conditional on the frozen parameter values,
not marginal over their posterior. `pointwise_loglik(y_i | theta_current)`
with (say) sigma held at 1 is a conditional log-likelihood; LOO computed
from it is conditional-LOO, and BPV is conditional-BPV. This is NOT
incorrect -- it answers a different question ("how well does the model
with sigma pinned at 1 fit the data?") -- but the runner MUST surface
the change of semantics to the user:

```r
if (length(m$get_frozen()) > 0L) {
    cat(sprintf(
        "[R3] NOTE: %d frozen param(s) (%s) -- R3.a BPV / R3.b LOO are\n",
        length(m$get_frozen()), paste(m$get_frozen(), collapse = ", ")))
    cat("      CONDITIONAL on the frozen values, not marginal. Interpret\n")
    cat("      as posterior predictive given the pinned parameters, not\n")
    cat("      unconditional model-vs-data fit.\n")
}
```

The Bayesian p-value thresholds and PSIS-k-hat diagnostic targets are
unchanged; the interpretation is.

#### R3.a Bayesian posterior-predictive p-values

```r
pp <- model$predict_at(list())       # no-input posterior predictive
y_rep <- pp$y_rep                    # n_draws x N matrix
```

works **uniformly** for every model -- no-input (DirichletSimplex,
BetaBernoulli, etc.) or has-input (BartNoise, ARDLasso). The
`predict_rng_` member in each wrapper gives reproducible samples given
the same construction seed.

**Summary statistics** -- pick by observation support:
- continuous y: `mean, sd, min, max, q25, q75`
- binary y:  `mean`
- count y:   `mean, sd, max`

**Bayesian p-value:** `pv(T) = mean(T(y_rep_d) >= T(y_obs))` for each T.

**Verdict:**
- pv  in  (0.05, 0.95) -> OK
- outside [0.05, 0.95] -> WARN (one statistic outside is usually fine;
  two or more simultaneously is a semantic bug)
- **Tighter threshold (0.02, 0.98) when the sampler uses
  `joint_nuts_block`** -- silent
  miscalibration bugs in concatenate-and-slice code are more insidious
  (see Check #11).

#### R3.b PSIS-LOO via `loo::loo()`

Pointwise log-likelihood must be provided by a `pointwise_loglik(hist,
y, ...)` helper that lives next to `run_chain` in the generated
runner. See `skills/codegen_cpp.md` Sec.6a for per-observation-family
templates. Examples:

- Gaussian: `dnorm(y_i, mu^(d), sigma^(d), log = TRUE)`
- BART: `dnorm(y_i, f_bart^(d)[i], sigma^(d), log = TRUE)`
- Bernoulli: `dbinom(y_i, 1, p^(d), log = TRUE)`
- Poisson: `dpois(y_i, exp(log_f^(d)[i]), log = TRUE)`

```r
library(loo)
LL1 <- pointwise_loglik(c1$hist, y_obs)   # n_iter x N, after burnin trim
LL2 <- pointwise_loglik(c2$hist, y_obs)
LLarr <- array(NA_real_, dim = c(nrow(LL1), 2, ncol(LL1)))
LLarr[, 1, ] <- LL1; LLarr[, 2, ] <- LL2
rel_n_eff <- loo::relative_eff(exp(LLarr))
lo        <- loo::loo(LLarr, r_eff = rel_n_eff, cores = 1)
```

**Verdict (DIAGNOSTIC ONLY -- does NOT fail R3):**
- Pareto-$\hat{k}$ measures LOO importance-weight reliability, **not**
  sampler correctness or posterior calibration. R3.b records
  `pct_k_lo` and `pct_k_hi` and emits a `warning()` when below the
  targets below, but the runner does **not** `stopifnot` on Pareto-$\hat{k}$.
- Diagnostic targets (warning thresholds, not pass/fail):
  >=50% k<0.5; <10% k>=0.7.
- GP latent-variable and many hierarchical models with informative
  per-observation latents systematically fail these targets even with
  correctly sampled posteriors -- Pareto-$\hat{k}$ reflects how strongly
  each leave-one-out posterior differs from the full-data posterior,
  which is a property of the model class, not a sampler bug. See
  **Vehtari, Simpson, Gelman, Yao, Gabry -- *Pareto Smoothed Importance
  Sampling*, JMLR 2024 (arXiv:1507.02646)**, esp. Sec.1 and the GP /
  random-effects examples.

#### R3 R code to emit (appended to the runner, after R2's chain build)

```r
# Uses c1, c2 produced by R2.

# --- R3.a Bayesian p-values ---
bp_stat <- c(mean = mean, sd = sd, min = min, max = max,
             q25 = function(x) quantile(x, 0.25, names = FALSE),
             q75 = function(x) quantile(x, 0.75, names = FALSE))
pv1 <- sapply(bp_stat, function(f)
    mean(apply(c1$pp$y_rep, 1, f) >= f(y_obs)))
# Threshold tightens to (0.02, 0.98) if the sampler uses joint_nuts_block.
pv_lo <- if (USES_JOINT_NUTS) 0.02 else 0.05
pv_hi <- 1 - pv_lo
stopifnot(all(pv1 > pv_lo & pv1 < pv_hi))

# --- R3.b PSIS-LOO (DIAGNOSTIC ONLY -- does NOT fail R3) ---
# Pareto-k_hat measures LOO importance-weight reliability, NOT sampler
# correctness. GP latent-variable and hierarchical-latent models are
# known to fail this diagnostic even when the posterior is correctly
# sampled (Vehtari, Simpson, Gelman, Yao, Gabry, JMLR 2024,
# arXiv:1507.02646). Sampler-correctness gates are R-hat (R2) and
# Bayesian p-values (R3.a); R3.b is recorded + warned, never stopifnot.
LL1 <- pointwise_loglik(c1$hist, y_obs)
LL2 <- pointwise_loglik(c2$hist, y_obs)
LLarr <- array(NA_real_, dim = c(nrow(LL1), 2, ncol(LL1)))
LLarr[, 1, ] <- LL1; LLarr[, 2, ] <- LL2
rel_n_eff <- loo::relative_eff(exp(LLarr))
lo <- loo::loo(LLarr, r_eff = rel_n_eff, cores = 1)
pct_k_lo <- mean(lo$diagnostics$pareto_k <  0.5) * 100
pct_k_hi <- mean(lo$diagnostics$pareto_k >= 0.7) * 100
cat(sprintf(
    "[R3.b] PSIS-LOO Pareto-k (diagnostic): %.1f%% k<0.5, %.1f%% k>=0.7\n",
    pct_k_lo, pct_k_hi))
if (pct_k_lo < 50 || pct_k_hi >= 10) {
    warning(sprintf(
        "[R3.b] PSIS-LOO Pareto-k DIAGNOSTIC ONLY (NOT a failure): %.1f%% k<0.5, %.1f%% k>=0.7. Pareto-k indicates LOO importance-weight reliability, NOT sampler correctness; GP and hierarchical-latent models routinely fail this diagnostic with correctly sampled posteriors. See Vehtari et al. (JMLR 2024).",
        pct_k_lo, pct_k_hi))
}
```

**Failure routing:**
- Bayesian p-value at the boundary on multiple stats -> Semantic #1
  (wrong observation-distribution parameterization) or #4 (missing
  intercept/offset).
- Pareto-$\hat{k}$ heavy tail (>10% k >= 0.7) -> **DIAGNOSTIC ONLY**, does
  NOT fail R3. May reflect Semantic #9 (numerical stability on edge
  observations) or data outliers, but for GP / hierarchical-latent
  models it is most often intrinsic LOO unreliability rather than a
  sampler bug (Vehtari et al., JMLR 2024).
- R2 passes but R3 fails -> posterior converged but to the wrong
  distribution; almost always Semantic #1 or #5 (Jacobian).

---

## Running the validator

- **Layer 1 (Syntactic):** run `sourceCpp`. Compilation errors -> fix
  and retry.
- **Layer 2 (Semantic):** walk the generated `.cpp` through the 26
  semantic checks (#1-#26; #14-#17 are defined in sibling skill
  files). Check #12 (gradient verification) is execution-based and
  done at generation time via a throwaway
  `tests_autodiff/verify_<ClassName>.cpp` companion file (see the
  Check #12 body above for the template); the rest are static code
  review. Checks #11 (joint_nuts_block), #13 (RNG separation), #18
  (dense metric), #23 (readapt_NUTS), #24 (joint-NUTS pathology), and
  #25 (rjmcmc / Dirac-spike) fire conditionally -- only when the
  relevant pattern is present. **Check #26 (kernel-control conformance)
  fires on EVERY wrapper.**
- **Layer 3 (Runtime):** emit R1, R2, R3 into the R runner. They share
  the same 2 chains (keep_history = TRUE). R1 is a smoke test; R2 is
  R-hat + ESS; R3 is posterior-predictive p-values + PSIS-LOO.

If an R-layer assertion fails, the likely cause is Layer 2. Specifically:
- R1 fails -> Rcpp API (Semantic #8) or numerical stability (#9).
- R2 fails -> Check #2 (parallel update), #3 (dead param), #5 (Jacobian),
  #12 (gradient verification -- was the AI-generated hand-written grad actually
  validated?), or -- if joint_NUTS is used -- #11 (joint-block slicing),
  or -- if `use_dense_metric = true` -- #18 (dense metric pilot
  inadequately tuned; common symptom is `ess_bulk = NA`).
- R3 fails but R2 passes -> Check #1 (parameterization) or #5 (Jacobian)
  in the observation / likelihood piece specifically. If Check #12
  passed then the gradient arithmetic is validated -- the remaining
  error is in the distribution's natural-scale formulation (wrong mean
  parameter, wrong observation family, ...).

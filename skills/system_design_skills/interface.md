<!-- system_design MODULE -- interface contract (extracted 2026-06-21 from system_design.md Sec.0-Sec.2).
     SINGLE LIVE SOURCE: edit HERE, not the monolith. system_design.md is now a thin index
     (Sec.N -> module).
     Cross-refs keep the "Sec.N" scheme, resolved via the system_design.md index. -->

## 0. Audience disambiguation

Two completely different agents interact with this codebase. They must
NOT read the same docs.

### The code-generating agent is NOT the reader of this file.

Given a user model description ("hierarchical linear regression with
centered predictors and a half-normal scale prior"), the code-gen
agent reads `skills/codegen.md` and emits:
- one `.cpp` file that composes EXISTING blocks, and
- one `.R` runner that calls the produced Rcpp module.

That agent never modifies `include/AI4BayesCode/*.hpp`, never edits
`bart_pure_cpp/src/*.h`, never adds `.method(` lines to an RCPP_MODULE, never
defines a new C++ class. It only uses what exists.

From the code-gen agent's point of view, every block is opaque:
- Construct with `new(ClassName, args..., rng_seed, keep_history)`.
- Advance with `m$step(n)`.
- Read with `m$get_current()`, `m$get_history()`, `m$get_dag()`.
- Push new state with `m$set_current(list(key = value, ...))`.
- Predict with `m$predict_at(list(X = X_new, ...))`.

That's the whole vocabulary. This file is two levels deeper than that.

### The system-design agent IS the reader.

You are changing / extending the LAYERS that back the core-6 state
contract + kernel-control category (see Sec.1 amendment below). Every
invariant below must still hold after your change, or nested MCMC
composition silently breaks.

---

## 1. The non-negotiable invariant -- unified stateful-module interface

**Every user-facing wrapper class in `examples/*.cpp` exposes the
SAME set of R-level methods, organized in two categories: the CORE-SIX
state contract (always present, uniform across every wrapper) and the
KERNEL-CONTROL category (per-wrapper members determined by composite
content):**

**Core six (always present, every wrapper):**

```
step         (int n)                -> void
get_current  ()                     -> Rcpp::List
set_current  (Rcpp::List params)    -> void
predict_at   (Rcpp::List new_data)  -> Rcpp::List
get_dag      ()                     -> Rcpp::List
get_history  ()                     -> Rcpp::List
```

**Kernel-control category (per-wrapper, presence dictated by composite):**

```
freeze       (Rcpp::CharacterVector names)                    -- always
   * present on EVERY wrapper (composite-level operation);
   * names must be non-empty valid child block names OR joint_nuts_block
     sub-parameter (slot) names -- slot-level freeze on joint blocks is
     supported in v1 via the joint_nuts_block::freeze() override
     (DESIGN_NOTES Sec.10.a);
   * unknown-name / blacklist-family / no-arg -> Rcpp::stop;
   * already-frozen name -> Rcpp::warning (idempotent);
   * see DESIGN_NOTES_FREEZE_UNFREEZE_2026-07-19.md for full contract
     and Sec.13 "Freeze semantics per family" for the whitelist /
     blacklist per block family.

unfreeze     (Rcpp::Nullable<Rcpp::CharacterVector> = R_NilValue) -- always
   * present on EVERY wrapper;
   * no-arg = unfreeze all; other invalid forms
     (NULL, character(0), unknown name) -> Rcpp::stop;
   * not-currently-frozen name -> Rcpp::warning (idempotent).

get_frozen   () -> Rcpp::CharacterVector -- always
   * present on EVERY wrapper;
   * returns names of currently-frozen children, in composite child
     order (matches `get_dag()` ordering).

readapt_NUTS (int n, bool reset = false, int max_tree_depth = -1) -> void
   * present ONLY on wrappers whose composite contains at least
     one `nuts_block` or `joint_nuts_block` child;
   * pure NUTS metric re-adaptation (mass matrix + step size +
     dual-averaging state); chain state is preserved (snapshot
     + restore around the n internal adaptation iterations);
   * `max_tree_depth = -1` means "keep current value"; a non-negative
     value overrides the sampler's tree-depth cap for the adaptation
     iterations;
   * skips any frozen NUTS-family children (no-op on those);
   * see Sec.13 "NUTS-family" for the full contract and Sec.23 of
     `validator.md` for the conformance check.
```

**Semantic separation.** Core-six methods are for STATE INJECTION and QUERY.
Kernel-control methods are for KERNEL BEHAVIOR CONFIGURATION (which
sub-kernels run, how they adapt); they MUST NOT mutate parameter state --
that goes through `set_current`. Adding a new kernel-control method (e.g.,
a hypothetical `readapt_VI` for VI blocks) requires an explicit Sec.1
amendment and a listed inclusion criterion.

Specifically: there is NEVER a `$set_X()`, `$set_Y()`,
`$set_sigma()`, `$set_uv()`, `$update_working_response()`,
`$refresh_y_rep()`, or any other state-method extension. All
state-injection operations go through `set_current(list(...))` with
keys routed internally.

**BART-family carve-out (historical exception, documented).** Wrappers
whose composite includes any `bart_block` / `genbart_block` / `softbart_block`
child ALSO expose three tree-serialization methods:

```
get_tree          () -> Rcpp::List
set_tree          (Rcpp::List tree) -> void
get_tree_history  () -> Rcpp::List
```

These are historical exceptions to the "core-6 + kernel-control" rule,
predating the formal categorization. They are BART-family specific
(non-BART wrappers must not expose them). Documented here so grep-audit
does not flag them as violations. Not treated as a formal
"tree-serialization category"; any expansion (e.g., forest introspection
on other tree kernels) requires an explicit Sec.1 amendment.

**Naming disambiguation -- user-facing `freeze` vs internal numerical concepts.**
The kernel-control method `m$freeze(...)` is UNRELATED to internal
numerical terms also called "freeze" in the codebase:
- funnel freeze (joint_nuts_failure.md Mode 1) -- a mixing pathology
- phase-III frozen mass matrix (families.md 3-phase warmup) -- Stan-style windowed warmup terminology
- dual-averaging freeze fix (numerical patch) -- an internal step-size adaptation guard

Skill docs disambiguate by context: `m$freeze(` and
`.method("freeze", ...)` always refer to the user API; bare "freeze" in
the context of mixing / mass matrix / adaptation / warmup refers to the
internal numerical concept.

### Why

The outer MCMC does not (and must not have to) know whether a
module's underlying sampler is continuous NUTS, a Dirichlet Gibbs
leaf, a BART tree forest, a joint NUTS block, or a future kernel we
haven't built yet. Every module accepts `set_current(list(...))` and
dispatches by key. The outer composition loop stays uniform:

```r
for (iter in seq_len(n_iter)) {
    for (mod in modules) {
        mod$set_current(list(...))   # push any subset of parameters
        mod$step(1L)                  # advance one sweep
    }
}
```

A new module that adds a "helpful" extra R method (`$set_X` etc.)
breaks this loop for every downstream user. Uniformity is the whole
point of AI4BayesCode -- it is what distinguishes this library from
using dbarts / mcmclib / etc. directly.

### How to verify the invariant

- Grep every `examples/*.cpp` for `\.method(` inside its
  `RCPP_MODULE(..._module)`. The set of method names must exactly
  match the following formula:
  ```
  { step, get_current, set_current, predict_at, get_dag, get_history }
    ALWAYS (core-6)
  U { freeze, unfreeze, get_frozen }
    ALWAYS (kernel-control, via AI4BAYESCODE_BIND_KERNEL_CONTROL macro)
  U { readapt_NUTS }
    iff composite contains any nuts_block / joint_nuts_block child
  U { get_tree, set_tree, get_tree_history }
    iff composite contains any bart_block / genbart_block / softbart_block child
  ```
  Any name outside this formula is a bug. Missing a member the composite
  requires is also a bug (e.g., a NUTS-family wrapper without
  `readapt_NUTS`, or ANY wrapper without freeze/unfreeze/get_frozen).
- In R: `names(m)` on any constructed wrapper should show only the
  reference-class internals (`.self`, `.module`, `.pointer`,
  `.refClassDef`, `.cppclass`, `initialize`, `finalize`) and
  NOTHING else. Rcpp's `.method` shows up as callable via `$`,
  but not in `names()`. Either way, no extra names / functions
  outside the formula above should appear.
- Validator Check #23 verifies `readapt_NUTS` presence/absence + state
  preservation; validator Check #26 verifies freeze/unfreeze/get_frozen
  presence + whitelist/blacklist gate + refreeze warning + stale-derived
  warning.

---

## 2. Three-tier architecture

The invariant above is enforced by a strict three-tier split. Every
BART-family, NUTS-family, Gibbs-family block follows it.

```
 +------------------------------------------------------------+
 | TIER A -- user-facing wrapper (examples/MyModel.cpp)         |
 |   * exposed to R via RCPP_MODULE                            |
 |   * R sees ONLY core-6 state + kernel-control (Sec.1)        |
 |   * set_current(Rcpp::List) is a pure DISPATCHER; routes   |
 |     keys (X, y, theta, sigma, u, v, ...) to Tier B setters   |
 |   * owns a `composite_block impl_` with child blocks added |
 +----------------+-------------------------------------------+
                  | C++-only calls into blocks (not R-visible)
                  v
 +------------------------------------------------------------+
 | TIER B -- block layer (include/AI4BayesCode/*.hpp)              |
 |   * nuts_block, bart_block, genbart_block, *_gibbs_block,  |
 |     joint_nuts_block(_mixed), composite_block,              |
 |     poisson_multinomial_aug_block                          |
 |   * exposes FINE-GRAINED C++ setters for Tier A's use:     |
 |       nuts_block::set_current(arma::vec)                    |
 |       bart_block::set_X / set_Y / set_data                  |
 |       genbart_block::set_X / set_Y / set_offset / set_data |
 |       (name depends on the kernel; see Sec.14 for the rules)  |
 |   * NONE of these have .method() entries -- Tier B is C++    |
 |     implementation detail                                   |
 +----------------+-------------------------------------------+
                  | forward to vendored kernel (no direct Rcpp)
                  v
 +------------------------------------------------------------+
 | TIER C -- vendored kernel (bart_pure_cpp/src/*.h with        |
 |             subtrees BART/, GENBART/, SOFTBART_VENDOR/,     |
 |             include/mcmclib/*.h)                            |
 |   * bart_model::set_data / set_Y / set_X / set_sigma        |
 |   * genbart::genbart_model::set_X / set_Y / set_offset      |
 |   * mcmclib::nuts internals                                 |
 |   * touched ONLY from Tier B; never from Tier A directly    |
 +------------------------------------------------------------+
```

When designing a new block, update all three tiers together. Skipping
Tier A means R cannot drive it. Skipping Tier B means Tier A leaks
into Tier C (abstraction broken). Skipping Tier C means the kernel
has no way to accept the new data / parameter update.

---


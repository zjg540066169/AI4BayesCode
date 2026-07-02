<!-- system_design MODULE — interface contract (extracted 2026-06-21 from system_design.md §0-§2).
     SINGLE LIVE SOURCE: edit HERE, not the monolith. system_design.md is now a thin index
     (§N -> module).
     Cross-refs keep the "§N" scheme, resolved via the system_design.md index. -->

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
- Push new state with `m$set_current(list(key = value, …))`.
- Predict with `m$predict_at(list(X = X_new, …))`.

That's the whole vocabulary. This file is two levels deeper than that.

### The system-design agent IS the reader.

You are changing / extending the LAYERS that back the six-method
interface. Every invariant below must still hold after your change,
or nested MCMC composition silently breaks.

---

## 1. The non-negotiable invariant — unified stateful-module interface

**Every user-facing wrapper class in `examples/*.cpp` exposes the
SAME six core R-level methods, and a 7th kernel-tuning method
ONLY when the wrapper's composite contains at least one NUTS-family
child block:**

**Core six (always present, every wrapper):**

```
step         (int n)                -> void
get_current  ()                     -> Rcpp::List
set_current  (Rcpp::List params)    -> void
predict_at   (Rcpp::List new_data)  -> Rcpp::List
get_dag      ()                     -> Rcpp::List
get_history  ()                     -> Rcpp::List
```

**Optional 7th (kernel-tuning category, conditional on NUTS-family child):**

```
readapt_NUTS (int n, bool reset = false) -> void
   * present ONLY on wrappers whose composite contains at least
     one `nuts_block` or `joint_nuts_block` child;
   * pure NUTS metric re-adaptation (mass matrix + step size +
     dual-averaging state); chain state is preserved (snapshot
     + restore around the n internal adaptation iterations);
   * see §13 "NUTS-family" for the full contract and §24 of
     `validator.md` for the conformance check.
```

Specifically: there is NEVER a `$set_X()`, `$set_Y()`,
`$set_sigma()`, `$set_uv()`, `$update_working_response()`,
`$refresh_y_rep()`, or any other state-method extension. All
state-injection operations go through `set_current(list(...))` with
keys routed internally. The 7th method `readapt_NUTS` is a
KERNEL-tuning method (separate category from state methods) and is
the ONLY R-level extension allowed beyond the core six. New
kernel-tuning methods (e.g., a future `readapt_VI`) require an
explicit §1 amendment.

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
point of AI4BayesCode — it is what distinguishes this library from
using dbarts / mcmclib / etc. directly.

### How to verify the invariant

- Grep every `examples/*.cpp` for `\.method(` inside its
  `RCPP_MODULE(…_module)`. For wrappers WITHOUT any NUTS-family
  child the count is **exactly 6** (core methods only). For
  wrappers WITH at least one `nuts_block` or `joint_nuts_block`
  child, the count is **exactly 7**
  (core six plus `readapt_NUTS`). Any other count or method name
  outside the core 6 / conditional 7 contract is a bug.
- In R: `names(m)` on any constructed wrapper should show only the
  reference-class internals (`.self`, `.module`, `.pointer`,
  `.refClassDef`, `.cppclass`, `initialize`, `finalize`) and
  NOTHING else. Rcpp's `.method` shows up as callable via `$`,
  but not in `names()`. Either way, no extra names / functions
  outside the core 6 / conditional 7 contract should appear.

---

## 2. Three-tier architecture

The invariant above is enforced by a strict three-tier split. Every
BART-family, NUTS-family, Gibbs-family block follows it.

```
 ┌────────────────────────────────────────────────────────────┐
 │ TIER A — user-facing wrapper (examples/MyModel.cpp)         │
 │   * exposed to R via RCPP_MODULE                            │
 │   * R sees ONLY the six-method contract                     │
 │   * set_current(Rcpp::List) is a pure DISPATCHER; routes   │
 │     keys (X, y, theta, sigma, u, v, …) to Tier B setters   │
 │   * owns a `composite_block impl_` with child blocks added │
 └────────────────┬───────────────────────────────────────────┘
                  │ C++-only calls into blocks (not R-visible)
                  ▼
 ┌────────────────────────────────────────────────────────────┐
 │ TIER B — block layer (include/AI4BayesCode/*.hpp)              │
 │   * nuts_block, bart_block, genbart_block, *_gibbs_block,  │
 │     joint_nuts_block(_mixed), composite_block,              │
 │     poisson_multinomial_aug_block                          │
 │   * exposes FINE-GRAINED C++ setters for Tier A's use:     │
 │       nuts_block::set_current(arma::vec)                    │
 │       bart_block::set_X / set_Y / set_data                  │
 │       genbart_block::set_X / set_Y / set_offset / set_data │
 │       (name depends on the kernel; see §14 for the rules)  │
 │   * NONE of these have .method() entries — Tier B is C++    │
 │     implementation detail                                   │
 └────────────────┬───────────────────────────────────────────┘
                  │ forward to vendored kernel (no direct Rcpp)
                  ▼
 ┌────────────────────────────────────────────────────────────┐
 │ TIER C — vendored kernel (bart_pure_cpp/src/*.h with        │
 │             subtrees BART/, GENBART/, SOFTBART_VENDOR/,     │
 │             include/mcmclib/*.h)                            │
 │   * bart_model::set_data / set_Y / set_X / set_sigma        │
 │   * genbart::genbart_model::set_X / set_Y / set_offset      │
 │   * mcmclib::nuts internals                                 │
 │   * touched ONLY from Tier B; never from Tier A directly    │
 └────────────────────────────────────────────────────────────┘
```

When designing a new block, update all three tiers together. Skipping
Tier A means R cannot drive it. Skipping Tier B means Tier A leaks
into Tier C (abstraction broken). Skipping Tier C means the kernel
has no way to accept the new data / parameter update.

---


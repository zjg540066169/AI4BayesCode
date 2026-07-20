<!-- system_design MODULE -- design lifecycle + checklist (extracted 2026-06-21 from system_design.md Sec.14-Sec.17).
     SINGLE LIVE SOURCE: edit HERE, not the monolith. system_design.md is now a thin index
     (Sec.N -> module).
     Cross-refs keep the "Sec.N" scheme, resolved via the system_design.md index. -->

## 14. How to design a new block type (step-by-step)

Suppose you want to add `gp_block` (Gaussian Process regression with
closed-form conjugate draws on the latent function values).

### Step 1 -- Tier C (kernel)

Decide what the kernel looks like. For GP, that's:
- `gp_kernel` class (or external dependency) that holds the
  covariance matrix + its inverse, exposes `set_X(X)`,
  `set_y(y)`, `sample_latent(rng) -> arma::vec`, etc.

Vendor it under `include/gpmcmc/` (or wherever fits) with a clear
license file. If the kernel uses Eigen / other dependencies,
document in `THIRD_PARTY_LICENSES.md`.

**License-compatibility check before vendoring.** AI4BayesCode is
distributed under GPL-3.0-or-later (`LICENSE` at repo root). A new
vendored kernel is acceptable only if its upstream license is
compatible with GPL-3 under the "or-later" flexibility:
- GPL-2-or-later / GPL-3 upstream: no new compatibility constraint.
- MIT / BSD / MPL-2.0 upstream: compatible with GPL-2 and GPL-3;
  preserve the upstream LICENSE file verbatim inside the vendored
  directory.
- Apache-2.0 upstream: compatible with GPL-3 branch of the
  "or-later" only (NOT with GPL-2-only). Acceptable, but document
  the combined-work effect in `THIRD_PARTY_LICENSES.md` (we already
  have precedent for this with MCMClib and BaseMatrixOps).
- Any non-free / proprietary / unclear license: reject. Do not
  vendor.

Do NOT change the upstream's own license headers; Copyright /
`LICENSE` / `COPYING` files inside the vendored directory stay as
they are. The project-wide GPL-3.0+ only applies to code we
authored under `include/AI4BayesCode/`, `examples/`, `R/`, `skills/`,
`tests_autodiff/`, and the repo-root docs.

### Step 2 -- Tier B (block)

Create `include/AI4BayesCode/gp_block.hpp`:
- `struct gp_block_config` with `name`, `x_train`, `y_train_key`,
  any hyperparameters.
- `class gp_block : public block_sampler`:
  - `set_context(const block_context&)`: read any refreshable data
    (y_train_key from ctx).
  - `step(std::mt19937_64&)`: one kernel sweep.
  - `current() const -> const arma::vec&`: current latent f.
  - `set_current(arma::vec)`: write latent (if invertible;
    otherwise `Rcpp::stop` and expose `set_X / set_Y` C++ setters
    for Tier A's dispatcher).
  - `name()`, `dim()`.
  - Any block-specific setters: `set_X`, `set_Y`, `set_lengthscale`, etc.

### Step 3 -- Tier A (one or more wrapper examples)

Write `examples/GPGaussian.cpp`:
- Constructor takes `X`, `y`, any hyperparams + `rng_seed`,
  `keep_history`.
- Owns a `composite_block impl_`.
- Adds `gp_block` + any sibling blocks (e.g. a sigma `nuts_block`).
- Implements the **core-6 state contract** (`step / get_current /
  set_current / predict_at / get_dag / get_history`).
  `set_current(Rcpp::List)` dispatches on keys (`X`, `y`, `sigma`,
  any GP hyperparams) into the relevant child block's C++ setters.
- Inherits `AI4BayesCode::kernel_control_mixin<GPGaussian>` (CRTP)
  so it automatically exposes the **kernel-control category**
  (`freeze / unfreeze / get_frozen`) via the
  `AI4BAYESCODE_BIND_KERNEL_CONTROL(GPGaussian)` macro in RCPP_MODULE.
- RCPP_MODULE exposes ONLY core-6 + kernel-control (plus `readapt_NUTS`
  iff any NUTS-family child, plus BART tree-serialization carve-out iff
  any BART-family child). DO NOT add `.method("set_X", ...)` or any
  other state-method extension.

### Step 4 -- skills + catalogue

- Add an entry in `skills/block_catalogue/index.md` describing `gp_block`:
  when to use, config fields, Gibbs DAG conventions.
- If any hyperparameter is non-obvious, mention it in
  `skills/codegen_priors.md`.
- Add `GPGaussian` to the list of production examples in
  `README.md`.

### Step 5 -- tests

- `tests_autodiff/verify_GPGaussian.cpp` -- Check #12 if any block
  has a hand-written gradient. Delete after PASS.
- An R-level test under `tests_autodiff/` that exercises
  `set_current(list(X=..., y=...))` for the new block:
  round-trip, untouched-key preservation, rejected-key errors,
  multi-iteration loop. Model after
  `tests_autodiff/test_bart_setcurrent.R`.
- Extend `tests_autodiff/audit_compile_smoke.R` with the new
  example.
- Extend `tests_autodiff/audit_predict_at.R` with the new example.

### Step 6 -- validator

- If the new block has a hand-written log-density, add it to the
  Check #12 AD-verify list in `skills/validator.md`.
- If it uses a new constraint kind, extend `skills/constraints.md`
  and `include/AI4BayesCode/constraints.hpp` + add a primitive test
  in `tests_autodiff/`.

### Step 7 -- cross-chain audit

Run `tests_autodiff/audit_xl.R` adding a new example block -- 3
seeds x 10k+10k. Expected: R-hat < 1.05 on every parameter, 95%
coverage on a coverage simulation where applicable.

---

## 15. Case study -- BART nested-MCMC set_current fix (2026-04-19)

A concrete illustration of the invariants above.

### The bug

`BartNoise::set_current(Rcpp::List)` accepted only `sigma`.
`bart_block::set_current(arma::vec)` hard-stopped. Passing
`list(X = X_new)` or `list(y = r)` had no effect -- silently
ignored. This made nested BART (the core use case for this
library) impossible without reconstructing the wrapper on every
outer iteration.

### The fix (applied across three tiers atomically)

**Tier C:** added `bart_model::set_X(NumericMatrix)` in
`bart_pure_cpp/src/bart_model.h` -- keeps current y / fmean / center_y intact,
rebinds the heterbart X pointer. The genBART kernel
(`bart_pure_cpp/src/genbart_model.h`) ships with `set_X` from day 1, so no
Tier C change was needed there.

**Tier B:** added public
`bart_block::set_X`, `set_Y`, `set_data` -- C++-only methods that
forward to Tier C and update cached Tier B state (`cfg_.x_train`,
`cfg_.y_init`, `current_fit_`). NO `.method()` exposure. The new
`genbart_block` (2026-04-24) was written directly in this style and
ships with `set_X`, `set_Y`, `set_offset`, `set_data` from its
first commit.

**Tier A:** extended `BartNoise::set_current(Rcpp::List)` to route
X / y / sigma keys into Tier B setters, atomically handle
combinations (`list(X, y, sigma)`), reject `f_bart`, silently ignore
unknown keys, and update shared_data (`bart_target`, `X`) in sync.
The 4 `GBart*` wrappers (2026-04-24) follow the same dispatcher
template: route `X` / `y` / `offset` into `genbart_block` setters,
reject `r` (or `r_j` for multinomial), silently ignore unknown keys.

### Invariants preserved

- RCPP_MODULE entries: still exactly 6 on both wrappers (per the
  contract in force at the time of this 2026-04-19 case study; today
  the invariant is core-6 + kernel-control per interface.md Sec.1).
  `names(m)`
  from R shows no new methods.
- Unified interface: outer MCMC uses `m$set_current(list(X = X_imp,
  y = r))` uniformly, same as for every other block.
- Pre-merge checklist passes: 8-test functional suite
  (`tests_autodiff/test_bart_setcurrent.R`), 11/11 compile-smoke,
  11/11 predict_at state preservation.

### Takeaway for future block designers

The ONLY reason this fix was two days of work instead of five
minutes is that the old design hard-coded a "BART can't be
refreshed" assumption into Tier B. Every new block type should
ask: what data / parameter updates will the outer MCMC want to
push? Then expose those via Tier B setters and Tier A dispatcher,
even if YOU don't see an immediate use case -- users always find
one, and retrofitting is painful.

---

## 16. Pre-merge checklist (universal, applies to every change)

Before any PR touching Tier A / B / C lands:

- [ ] **Unified-interface grep.** `grep -n '\.method(' examples/*.cpp`:
      for every wrapper class, the set of method names must match the
      interface.md Sec.1 formula: core-6 + kernel-control
      (`freeze`/`unfreeze`/`get_frozen`) + `readapt_NUTS` iff NUTS-family
      child + BART tree-serialization carve-out (`get_tree`/`set_tree`/
      `get_tree_history`) iff BART-family child. No extras outside this
      formula; no renames.
- [ ] **R-level names check.** Construct the wrapper in R, run
      `names(m)`. Should show only reference-class internals
      (`.self`, `.module`, etc.) -- no extra methods leaked.
- [ ] **Tier B public setters not leaked to R.** Grep:
      `grep -rn '\.method("set_X"\|\.method("set_Y"\|\.method("set_data"\|\.method("set_uv"' .` -- should return zero.
- [ ] **shared_data sync after set_current.** For every key your new
      set_current dispatcher writes to Tier B, verify the
      corresponding shared_data entry is also updated (grep the
      dispatcher body for `impl_->data().set(...)`).
- [ ] **Cached state sync.** After the Tier B call,
      `refresh_current_fit_()` / `refresh_current_logf_()` /
      equivalent is called so `get_current()` returns consistent
      values with no intervening `step()`.
- [ ] **Dimension / type validation.** Every setter does
      `Rcpp::stop` on mismatch with a precise error message.
- [ ] **Unknown key tolerance.**
      `m$set_current(list(random_key = rnorm(5)))` does NOT error.
- [ ] **Impossible keys rejected.** Where applicable
      (`f_bart`, `log_f`, refresher outputs), a clear `Rcpp::stop`
      fires.
- [ ] **Round-trip property.**
      `s <- m$get_current(); m$set_current(s)` does not crash and
      leaves state unchanged.
- [ ] **predict_at state preservation.** Snapshot `get_current()`
      before and after a predict_at call -- identical.
- [ ] **Layer 2 Check #5 -- no hand-written Jacobian** in any new
      log-density lambda.
- [ ] **Check #12 AD verify file (if any new NUTS block with
      hand-written grad).** Generate, run, PASS, delete.
- [ ] **Check #15 -- Gibbs-block parity test.** If the example uses
      any `*_gibbs_block`: verify
      `tests_autodiff/block_tests/test_<blockname>_gibbs_block.cpp`
      exists and passes. If the `params_fn` is non-textbook (uses
      conditional logic, active-subset indexing, or any derivation
      beyond plug-in counts), ALSO verify
      `tests_autodiff/test_<model>_<...>_gibbs_parity.cpp` per-usage
      test exists and passes. Same applies to hand-written Gibbs in
      `rjmcmc_block::continuous_update` hooks.
- [ ] **Check #16 -- inline Gibbs justification.** Every
      `*_gibbs_block` construction site and every hand-written Gibbs
      call inside an `rjmcmc_block` hook has an inline comment
      naming which Exception 1/2/3 from `codegen_priors.md Sec.2b` justifies
      the choice.
- [ ] **Check #17 -- no hand-written Gibbs samplers in examples.**
      `grep -E "std::gamma_distribution|std::normal_distribution.*\(.+\)|Rcpp::r(beta|gamma|dirichlet|invgamma)"`
      on `examples/*.cpp`: any hit must be inside the whitelist
      (`rjmcmc_block::propose_sample`, `continuous_update`,
      `register_stochastic_refresher`).
- [ ] **Variance prior discipline (Sec.11.6).** Any new example with a
      scale parameter uses Jeffreys `p(sigma) prop.to 1/sigma` by default
      (NUTS on log-transformed) with inline k=0 half-Normal(0, 1)
      fallback pattern; for genuinely-sparse-info cases, use
      half-Normal(0, A) proper weakly-informative. No
      `InverseGamma(epsilon, epsilon)` as "noninformative" (Gelman 2006 critique).
- [ ] **R-level functional test.** A test script under
      `tests_autodiff/` exercises the new setter keys:
      untouched-key preservation, atomic combinations,
      multi-iteration stability. Add to
      `audit_compile_smoke.R` and `audit_predict_at.R`.
- [ ] **Two-chain R-hat sanity.** 10k+10k x 2 chains at the new
      block's typical use case; R-hat < 1.05 on every param.
- [ ] **Documentation.** `skills/block_catalogue/index.md` entry. If new
      architectural concept, update this file.
- [ ] **VI-family Layer-3 R2-VI PSIS-k-hat diagnostic (k-hat < 0.7).** If the
      change adds or modifies any `vi_block` subclass: a Layer-3 R2-VI
      run (S=1000 q-samples -> joint k-hat) is in the pre-merge test
      suite, and k-hat < 0.7 on the canonical use case. If
      0.5 <= k-hat < 0.7, expectations are PSIS-reweighted in any
      shipped runner. See Sec.18.8 and `validator.md Sec.R2-VI`.
- [ ] **VI Check #21 (block contract conformance).** Covers four
      sub-points (validator.md Sec.21): inheritance from `vi_block`,
      `engine_kind() == VI` override, composite writes
      `current_sample(rng)` (NOT `current()` / q-mean) to
      shared_data after each VI step, and `get_history()` returns
      `list(elbo, mu, log_sd, gamma, epoch, final_khat)` per child
      (NEVER posterior-draw matrices). See Sec.18.3 and Sec.18.4.
- [ ] **VI Check #22 (optimizer = RAABBVI by default).** Any new VI
      block uses `vi_optimizer::raabbvi` (avgAdam + Polyak-Ruppert
      averaging + R-hat convergence + SKL termination). Plain SGD,
      Adam-with-defaults, and Kucukelbir 2017's ELBO-window
      schedule are all explicitly rejected as defaults. See Sec.18.7
      and `validator.md Sec.22`.
- [ ] **Check #23 (readapt_NUTS state-preservation + RNG separation).**
      If the change adds or modifies the `readapt_NUTS` kernel-control method on
      any wrapper (i.e., wrapper containing NUTS-family child): verify
      (a) snapshot-before / restore-after pattern present in
      `readapt_NUTS` body, (b) only `readapt_rng_` consumed inside
      (NOT `rng_` or `predict_rng_`), (c) R-level round-trip test
      `identical(get_current(), { readapt_NUTS(N); get_current() })`
      passes bitwise. See Sec.13 NUTS-family + `validator.md Sec.23`.
- [ ] **Check #26 (a) -- kernel-control methods bound.** RCPP_MODULE
      contains `.method("freeze", ...)`, `.method("unfreeze", ...)`,
      and `.method("get_frozen", ...)` -- either explicitly or via the
      `AI4BAYESCODE_BIND_KERNEL_CONTROL(ClassName)` macro from
      `include/AI4BayesCode/kernel_control_mixin.hpp`. Companion
      PYBIND11_MODULE contains the three `.def(...)` lines or the
      `AI4BAYESCODE_PYBIND_KERNEL_CONTROL(m, ClassName)` macro. See
      Sec.1 kernel-control category + `validator.md Sec.26 (a)`.
- [ ] **Check #26 (b) -- freeze() strict error paths.** R-level test:
      `m$freeze("unknown_block_name")` -> error; `m$freeze(character(0))`
      -> error; `m$freeze()` (no arg) -> error. If composite contains
      any blacklist-family block (`bart_block` / `genbart_block` /
      `hmm_block` / `vi_block`): `m$freeze("<blacklist_name>")` -> error
      with reason string mentioning "not supported". See DESIGN_NOTES_FREEZE_UNFREEZE_2026-07-19.md Sec.6.
- [ ] **Check #26 (c) -- state preservation across freeze + step.**
      R-level test on any WHITELIST block: `m$set_current(list(<name> = v))`
      -> `m$freeze("<name>")` -> `m$step(1L)` -> `m$get_current()[["<name>"]]`
      returns `v` unchanged. Redundant `m$freeze("<name>")` on already-
      frozen block emits `Rcpp::warning`. `m$unfreeze()` with no arg
      -> `m$get_frozen()` returns `character(0)`.
- [ ] **Check #26 (d) -- stale-derived warning (only if applicable).**
      For any wrapper whose composite has a deterministic refresher over
      a whitelisted block's output: `m$freeze("<upstream_of_derived>")`
      emits `Rcpp::warning` mentioning the downstream derived key.
      Skip this sub-check if no such refresher exists.
- [ ] **Layer-3 R2.f exclusion.** R runner emits the frozen-param
      exclusion filter (`h1 <- h1[setdiff(names(h1), m$get_frozen())]`)
      BEFORE calling `diag_one` / `posterior::rhat`. See
      `validator.md Sec.R2.f`.

---

## 17. Anti-patterns to reject in review

- `$set_X`, `$set_Y`, `$refresh_*`, or any other **state-method
  extension** on a user-facing wrapper (a new method that pushes
  parameter / data state into the block). The answer is ALWAYS:
  route it through `set_current(list(...))`.
  **EXCEPTION**: kernel-control methods (separate category, NOT
  state methods) are allowed per Sec.1 / Sec.13. Kernel-control
  members in v1: `freeze` / `unfreeze` / `get_frozen` (always) +
  `readapt_NUTS(n, reset, max_tree_depth)` (iff any NUTS-family
  child). Future kernel-control methods (e.g., a hypothetical
  `readapt_VI`) require an explicit Sec.1 amendment.
- `Rcpp::stop("set_current not supported")` in a BART-family or any
  block that has data inputs an outer MCMC might update. If the
  block has NO updatable inputs, say that in the error message and
  route users at construction-time alternatives.
- Hand-written `+ log(sigma)` inside a natural-scale lambda.
  `constraints::positive::wrap` already adds the Jacobian. Layer 2
  Check #5.
- `std::random_device{}()` called inside `predict_at`. Always use
  the mutable `predict_rng_` seeded once at construction. Validator
  Check #13.
- Calling a Tier C kernel function from Tier A (e.g. wrapper calls
  `impl_->child(0).underlying()->set_X(...)`). Go through Tier B's
  public setter.
- Assuming `std::mt19937_64 rng_` passed to BART's step does anything.
  It doesn't -- BART uses R's RNG. Document this in the wrapper.
- Keeping a reference/pointer to `shared_data_t` inside a leaf. Copy
  or use context references bound to the current sweep.
- Adding a key to shared_data WITHOUT declaring its Gibbs dependency
  edges. The composite will build an empty context for readers.
- Modifying `bart_pure_cpp/src/bart_model.h` destructively (changing signatures
  of existing methods). Only additive changes are acceptable.
- VI block writing its `current()` (q-mean) into shared_data.
  Hybrid siblings then condition on a point estimate and
  underestimate posterior variance. Always write
  `current_sample(rng)`. See Sec.18.4.
- VI block exposing `$set_log_sd` or `$set_chol` as an R-level
  method on the Tier A wrapper. Route through
  `set_current(list(name_log_sd = ...))` per Sec.1.
- Using plain SGD, Adam-with-defaults, or Kucukelbir 2017's
  ELBO-window step-size schedule as the VI optimizer instead of
  RAABBVI. v1 commits to RAABBVI for the practical-defaults
  reasons in Sec.18.7 / Sec.18.10. Adam-with-defaults is a known-
  fragile choice in VI (avgAdam is the fix).
- Using R-hat to diagnose a VI block, or PSIS-k-hat to diagnose an
  MCMC block. The two diagnostics target different failure modes;
  mixing them masks real bugs. See Sec.18.8.
- Applying exclusive KL or PSIS reweighting "selectively". Both
  are v1 defaults applied uniformly; cherry-picking based on a
  specific model's empirics is over-tuning.
- VI block returning posterior draws from `get_history()` instead
  of the (elbo, mu, log sigma, gamma, epoch) trajectory. Draws over the
  optimization path are NOT posterior samples (q is changing);
  use `predict_at` + `current_sample` for posterior expectations.

---


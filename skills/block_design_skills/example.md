# module: example.md
# role: block_design EXAMPLE phase -- author ONE TRI-MODULE C++ example
# (examples/<Model>.cpp) that USES the new block via a composite: a fenced int main()
# standalone demo + an #ifdef AI4BAYESCODE_RCPP_MODULE block + an #ifdef AI4BAYESCODE_PYBIND_MODULE
# block + BOTH @example:R and @example:python doc blocks. The SAME file runs standalone, in R, AND
# in Python -- a contributed block must ship usable from both.
# loaded: lazily, only when the EXAMPLE phase is entered (after Stage 4 impl is locked).
# cite, don't restate: block_sampler/composite semantics = system_design Sec.1/Sec.2 (interface.md);
# set_current dispatch / DAG wiring = system_design Sec.3-Sec.9 (dataflow.md); the tri-module file
# skeleton (include preamble, the two module blocks, the int-main fence, the @example format) =
# codegen_cpp.md (Sec. "Class shape", "Header @example block", "Authoring order"); the canonical worked
# file to copy verbatim for SHAPE = examples/GaussianLocationScale.cpp.

## !!! GATE CHECK FIRST -- did the user accept the example gate? !!!

This module loads ONLY after the user accepted the EXAMPLE gate (asked at the VALIDATE->EXAMPLE
transition -- `validate.md Sec.6` / `00_flow.md Sec.4`: `(a) Yes (default) / (b) No -- skip to SKILL`).
If you reached this module WITHOUT that gate having been asked -- you transitioned straight out of
VALIDATE and "loaded the module" -- **STOP: fire the gate NOW** before doing anything here, and if
the user says "no", go to `skill.md` and skip the example entirely. An example is **OPTIONAL** --
some shipped core blocks have none; a block + passing library test is already a complete bundle.

## !!! TRI-MODULE RULE (non-negotiable) !!!

The example is ONE **tri-module** C++ file -- the SAME shape as every shipped core example (copy
`examples/GaussianLocationScale.cpp`). One file, FOUR renderings of ONE model:

1. a fenced `int main()` standalone demo, guarded by
   `#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)` -- the plain-C++
   smoke test (active ONLY when neither module macro is defined);
2. an `#ifdef AI4BAYESCODE_RCPP_MODULE` ... `RCPP_MODULE(<Model>_module){ ... }` ... `#endif` block (R);
3. an `#ifdef AI4BAYESCODE_PYBIND_MODULE` ... `PYBIND11_MODULE(<Model>, m){ ... }` ... `#endif` block (Python);
4. BOTH `@example:R` and `@example:python` doc blocks in the header comment.

**Emit ALL FOUR -- always.** The `.cpp` BINDINGS are dual-module exactly like codegen: codegen also
emits BOTH `RCPP_MODULE` and `PYBIND11_MODULE` regardless of the runtime backend (`codegen_cpp.md Sec.1`
+ its "Class shape" section), so every generated model is `source`-able in R and Python. The ONE
thing that differs is the `@example` DOC block: codegen writes only the chosen backend's `@example`
(that was the runner it actually tested), whereas block_design writes **AND tests BOTH** `@example:R`
and `@example:python` -- because block_design specifies no runner and a **contributed block must ship
usable from both R and Python**. So here both `@example` are always present and TESTED (see STAGING
below), never one, never transcribed-but-unrun. (The ONLY permanent exception is a block whose kernel
has no Python binding -- the BART / genbart family -- which stays R-only: `RCPP_MODULE` + `@example:R`,
no pybind.)

Why the `int main()` is STILL here: it is the cheapest end-to-end check (compiles + runs as a plain
binary, no R/Python toolchain) AND the smoke target the VALIDATE phase compiles. The two module
blocks are inert unless their macro is defined, so the standalone compile is UNCHANGED from the old
frontend-independent example -- tri-module is a SUPERSET, not a replacement.

**Types**: the driver-class methods use NEUTRAL C++ types only -- `AI4BayesCode::state_map` /
`history_map` / `arma::vec` -- which auto-convert to `Rcpp::List` / a Python dict via
`backend_neutral.hpp`. Errors preferably use **`ai4b::stop(...)`** (backend-neutral: `Rcpp::stop`
under R, a Python exception under pybind, `std::runtime_error` standalone); a plain
`throw std::runtime_error(...)` is also backend-safe (Rcpp and pybind both catch and convert it) and
is what `GaussianLocationScale.cpp` itself uses. What is BANNED is a raw `Rcpp::stop` or any `Rcpp::`
/ `pybind11::` call OUTSIDE its own module block -- that breaks the standalone and Python builds.

## What this phase produces

ONE small, runnable, TRI-MODULE worked model -- `blocks_local/<Block>/examples/<Model>.cpp` -- that
COMPOSES the new block into a `composite_block`, DEMONSTRATES it via an `int main()`, AND BINDS it
for R and Python. It is the human-readable "here is how you use this block" demo, the smoke target
the test phase compiles, AND the artifact a registry submission needs. Keep it minimal: the SMALLEST
model that exercises the block's distinctive feature, not a kitchen-sink showcase.

> The recurring villain: a block that compiles and runs but produces the WRONG posterior. A
> worked example that visibly recovers a known parameter (sim -> fit -> compare) is your first,
> cheapest line of defense against that silent failure -- long before the heavier
> recovery/parity tests in `test_<Block>.cpp`.
>
> **Tune the demo so it DISCRIMINATES.** Pick the data/noise regime so the naive baseline
> (raw data, independent fit, ignoring the structure this block exploits) has VISIBLE nonzero
> error, and the block visibly beats it. A demo where both the baseline and the block score
> "perfect" (e.g. a too-easy, well-separated field) passes its assertion but demonstrates
> NOTHING -- the example exists to SHOW the block's value, so make the baseline visibly fail.

## How the example drives the block

The driver reads state back with the neutral C++ API (`comp.data().get("<key>")`, or
`dynamic_cast<Block&>(comp.child(0)).current()`); `comp.step(rng)` advances the sampler. Wrap the
model in a small **driver class** (the Tier-A wrapper) whose methods are NEUTRAL-typed only --
`state_map` / `history_map` / `arma::vec`, NEVER `Rcpp::List` or a pybind type -- exposing the
six-method contract (`step` / `get_current` / `set_current` / `predict_at` / `get_dag` /
`get_history`). If the block wraps a NUTS child, add the **7th method `readapt_NUTS`** (forwarding to
the child -- `design.md Sec....`). The `block_sampler` semantics are owned by **system_design Sec.1**
(`interface.md`) -- do not re-derive them. The two frontend module blocks bind THIS driver class; the
`int main()` drives it directly.

## Structure of the file (tri-module) -- copy GaussianLocationScale.cpp

Sections, in order. The include preamble, the two module blocks, and the `int main()` fence are
MECHANICAL -- copy them verbatim from `examples/GaussianLocationScale.cpp` (the module-block idioms
are owned by `codegen_cpp.md`; do not re-derive). **Copy the STRUCTURE, but do NOT copy three lines
verbatim from that reference file:** (a) its top-of-file **license header** -- use the form the
License gate below fixes, not the reference's; (b) its **`@example` loader line** -- `GaussianLocationScale`
is a name-loaded SHIPPED example (`ai4bayescode_example("...")` / `AI4BayesCode.example("...")`),
but a contributed block is RELATIVE-path source-loaded (`ai4bayescode_source("<Model>.cpp")` /
`AI4BayesCode.source("<Model>.cpp")`); (c) nothing else is special -- the includes, the two module
blocks, and the fence copy as-is.

1. **Copyright header** (GPL-3.0-or-later; see License gate below).
2. **Header block comment**: the model (likelihood + priors, one display line each), the block
   decomposition (which `composite_block` children, with `<Block>` named), and one line on why THIS
   block (the `SelectWhen` trigger in prose). THEN, at the END of the comment, the
   **`@example:R` / `@example:python` / `@example:end`** blocks -- mandatory, runnable, each using the
   installed-package API with a RELATIVE path: R -> `ai4bayescode_source("<Model>.cpp")` then
   `new(<Model>, ...)`; Python -> `Mod = AI4BayesCode.source("<Model>.cpp")` then `Mod.<Model>(...)`.
   Format = `codegen_cpp.md` "Header @example block". The two `@example` blocks and the `int main()`
   are THREE renderings of ONE DGP -- write the DGP once, mirror it (so they cannot drift).
3. **Include preamble** (copy GaussianLocationScale.cpp ~L67-88): `// [[Rcpp::depends(RcppArmadillo)]]`,
   the `MCMC_ENABLE_ARMA_WRAPPERS` / `ARMA_DONT_USE_WRAPPER` defines, the
   `#ifdef AI4BAYESCODE_RCPP_MODULE #include <RcppArmadillo.h> #else #include <armadillo> #endif`
   switch, then `block_sampler.hpp`, `backend_neutral.hpp`, `shared_data.hpp`, the NEW block header
   (by its bundle path -- the compile phase puts `blocks_local/<Block>/` on `-I`), `composite_block.hpp`,
   `constraints.hpp`, `rcpp_wrap.hpp`, plus `<random>` / `<cmath>` / `<cstdio>`. Other C++ libraries
   (Eigen, celerite, libgp, autodiff) are fine if the block needs them.
4. **(optional) anonymous-namespace free functions** -- e.g. a log-density/grad closure, file-local.
5. **The neutral driver class** -- wires data + DAG + the new block child; the six/seven-method
   contract; neutral types only.
6. **RCPP_MODULE block** -- `#ifdef AI4BAYESCODE_RCPP_MODULE` ... `RCPP_MODULE(<Model>_module){
   Rcpp::class_<<Model>>("<Model>") .constructor<...>() ... .method("step", ...) ... }` ... `#endif`.
7. **PYBIND11_MODULE block** -- `#ifdef AI4BAYESCODE_PYBIND_MODULE` ... `#include
   "AI4BayesCode/pybind_casters.hpp"` ... `PYBIND11_MODULE(<Model>, m){
   AI4BayesCode::register_ai4bayescode_types(m); pybind11::class_<<Model>>(m,"<Model>") .def(init<...>())
   ... }` ... `#endif`.
8. **`int main()`** -- `#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)`
   -> simulate data from a KNOWN truth -> build the composite/block -> warmup + sample -> compute a
   posterior mean -> compare to truth (or a naive baseline) -> `printf` the result -> `return ok ? 0 : 1;`
   -> `#endif`. (For a prior-only block, check the sampled draws against the prior's known moments.)

### Constructor -- minimal wiring

- Build `impl_ = std::make_unique<composite_block>("<Model>")`.
- `impl_->data.set(key, value)` for observed data + initial parameter values.
- Declare the Gibbs-DAG dependencies (`declare_dependencies`) and predict-DAG edges
  (`declare_predict_edges`) + any refreshers -- semantics at **system_design Sec.3-Sec.9** (`dataflow.md`).
  Do NOT restate DAG rules here; wire the minimal set this small model needs (often a single child
  block + one `y_rep` predict edge). Signatures you will need: a **stochastic refresher** is
  `register_stochastic_refresher(key, [](const shared_data_t& d, std::mt19937_64& rng) -> arma::vec {...})`
  (e.g. the `y_rep` posterior-predictive draw); a **deterministic refresher** is
  `register_refresher(key, [](const shared_data_t& d) -> arma::vec {...})`.
- `impl_->add_child(std::make_unique<<Block>>(...))` -- construct the new block with its config.
- `if (keep_history_) impl_->set_keep_history(true);`
- RNG discipline: separate `rng_`, `predict_rng_`, `readapt_rng_` (seed-derived, deterministic when
  seed != 0) -- copy the pattern from `GaussianLocationScale.cpp`; see system_design Sec.8.
- Validation errors use `ai4b::stop(...)` (or a plain `throw std::runtime_error`); never a raw
  `Rcpp::stop`.

### `set_current` is a pure DISPATCHER (Tier A -> Tier B)

`set_current(const state_map& params)` must NOT contain sampler logic. It only:

1. looks up each KEY it owns in `params` (skip keys not present -- partial updates are legal),
2. validates the value (e.g. a positive-constrained parameter must be `> 0` -> `ai4b::stop` on
   violation),
3. routes the value to the corresponding Tier-B setter on the child block (e.g.
   `dynamic_cast<<Block>&>(impl_->child(0)).set_current(vec)` or the block's fine-grained
   `set_X`/`set_Y`-style C++ setter), and
4. mirrors the pushed value back into `impl_->data.set(key,...)` so downstream reads see it.

Which keys, and how they map to Tier-B setters, is the dispatch table -- that is system_design Sec.7
(`dataflow.md`); the example just instantiates it for this one model. See the `set_current` body in
`GaussianLocationScale.cpp` (lines ~238-256) for the canonical "find key -> validate ->
`jblk.set_current(cat)` -> mirror into `data`" shape.

> **Foot-gun -- COPY, do not bind-by-reference, the result of `get_current()`.** `get_current()`
> returns a `state_map` BY VALUE (a temporary); binding `const arma::vec& v = get_current().at(key)`
> leaves `v` dangling the instant the temporary dies -> a `0x0` crash on first use. Take the map
> into a named local first (`auto s = impl_->get_current(); const arma::vec& v = s.at(key);`), or
> copy the vector out. Same applies to any `.at(key)` on a returned-by-value container.

### Optional driver-class methods (`predict_at` / `get_dag` / `get_history`)

Keep them NEUTRAL-typed: `predict_at(const state_map&) -> history_map` (return changed downstream
nodes, predict-DAG BFS -- system_design Sec.4), `get_dag` / `get_history` thin forwards. The demo
`main()` usually does NOT need them -- a recovery demo only needs `step` + reading the current draw.

### The two frontend module blocks + `@example` (both are REQUIRED)

Write BOTH module blocks, binding the driver's six/seven methods -- copy the exact idiom from
`GaussianLocationScale.cpp` (L314-352): register both constructors (legacy + full), both `step`
overloads, and `readapt_NUTS` only if the driver has it. The `PYBIND11_MODULE` block opens with
`AI4BayesCode::register_ai4bayescode_types(m)` so `state_map` / `history_map` cross the boundary.
These bindings are exactly what makes the contributed block `source`-able in R and Python -- they are
NOT deferred to "codegen/packaging elsewhere" the way the old frontend-independent example did.

## License gate (the example file is GPL-3.0-or-later; manifest `License:` governs vendored code)

Every file under `examples/` (including `blocks_local/<Block>/examples/`) is part of AI4BayesCode and
is **GPL-3.0-or-later** -- `system_design Sec.0`: "all code under `examples/` ... is GPL-3.0-or-later;
every file you add or edit MUST carry a matching license header." UNIFORM -- it does NOT depend on
whether the example pulls BART / mcmclib. The example header MUST be the canonical three-line form:

 ```
 // Copyright (C) 2026 <Author>.
 // Licensed under the GNU General Public License v3.0 or later
 // (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
 ```

The per-license header form is `skills/codegen_cpp.md Sec.5` (cited, not restated). The bundle's
`manifest.dcf` `License:` field governs any **vendored** third-party code under `vendor/` (which
keeps its own upstream header verbatim -- see `intake.md` Step 5) -- NOT the example file, which is
always GPL-3.0-or-later. No need to ask the user -- the example license is fixed by project policy.

## ASK-THE-USER for this phase

Use a structured labeled-option prompt (AskUserQuestion in Claude Code; the markdown labeled-option
fallback from `start.md Sec.0` elsewhere). Mark the standing choice **(default)**, never
"(Recommended)"; always include "Other". Elicit:

- **Which demo model?** -- a labeled menu of small models that exercise the block's distinctive
  feature; "(default)" = the smallest one (mirrors the block's `SelectWhen`). "Other" = user
  describes their own.

There is **NO "frontend?" / "Python module?" question** -- the example is ALWAYS tri-module with BOTH
`@example` blocks, because a contributed block ships usable from R and Python by design (unlike
codegen, which asks a backend). The **posterior-predictive check is MANDATORY, not asked**: `main()`
ALWAYS computes a `y_rep`-style draw and checks predicted vs observed. The ONLY carve-out is a
**prior-only block** with no observation model -- no `y_rep`, so it checks sampled draws against the
prior's known moments; that carve-out is mechanism-determined, NOT a user choice. The demo's numerics
(N, the true parameter values) use sensible DEFAULTS recorded in the file header -- surface them in the
recap and let the user override, do NOT interrogate them one at a time (same policy as `validate.md Sec.5`).

## STAGING + AUTOMATION (TOP RULE)

Staging is the delivery GUARD, not a gate. Staging the file, all three checks, and the final move to
`blocks_local/` happen **automatically** -- nothing gated on a "go" (same rule as `validate.md`); the
move happens once all checks pass, then the final path is reported. A failed example is never moved.

- PROPOSE the full tri-module `<Model>.cpp` and STAGE it to the staging dir; show it for review.
- **SMOKE (automatic):** compile the `int main()` as a plain binary -- NEITHER module macro defined;
  link `<armadillo>` (`-framework Accelerate` on macOS / `-lblas -llapack` on Linux) -- and run it.
  The cheapest end-to-end recovery check. Compile recipe = `validate.md Sec.3` minus any Rcpp/R flags.
  (Unchanged by tri-module -- the module blocks are inert without their macro.)
- **BOTH `@example` TESTED (automatic):** because a contributed block ships for both frontends, BOTH
  `@example` blocks must be VERIFIED, not merely written. Source the block and run each example
  against the STAGING copy: R -> `ai4bayescode_source("<Model>.cpp")` then run the `@example:R` body;
  Python -> `AI4BayesCode.source("<Model>.cpp")` then run the `@example:python` body. Each must load,
  run, and return finite draws that recover the truth. Surface the command + expected runtime as you
  START each (a heads-up, not a gate). A frontend whose toolchain is genuinely unavailable is
  surfaced EXPLICITLY (never silently skipped) -- do not claim an untested `@example`.
- MOVE to `blocks_local/<Block>/examples/<Model>.cpp` AUTOMATICALLY once all three (int main + both
  `@example`) pass (optionally record the path in the manifest's OPTIONAL `Example:` field --
  discovery also finds `examples/*.cpp` by convention).
- The fuller recovery / parity / cross-chain R-hat checks live in the VALIDATE phase
  (`test_<Block>.cpp`), not here; this example is the SMOKE target plus a visible sanity recovery.

## Done-when

- File present at `blocks_local/<Block>/examples/<Model>.cpp` with the GPL-3.0-or-later license header.
- **TRI-MODULE**: fenced `int main()` + `#ifdef AI4BAYESCODE_RCPP_MODULE ... RCPP_MODULE ...` block +
  `#ifdef AI4BAYESCODE_PYBIND_MODULE ... PYBIND11_MODULE ...` block + BOTH `@example:R` and
  `@example:python`. (BART / genbart-kernel blocks are the R-only exception: `RCPP_MODULE` +
  `@example:R`, no pybind.)
- Driver methods NEUTRAL-typed (`state_map` / `arma`); `set_current` is a pure key-routing dispatcher
  (no sampler logic) that mirrors into `data`; errors via `ai4b::stop`.
- **`int main()` compiles + RUNS** as a plain binary (automatic, no "go") -- prints a recovery/smoke
  result and returns 0 on pass.
- **BOTH `@example` VERIFIED** -- sourced + run in R AND Python (automatic), each loads and recovers.
- **Posterior-predictive check present** -- `main()` draws `y_rep` and checks predicted vs observed
  (MANDATORY; the ONLY exception is a prior-only block, which checks draws against prior moments).
- License header == **GPL-3.0-or-later** (uniform project policy for `examples/`); the
  `manifest.dcf` `License:` field governs the block + any `vendor/` code, NOT this example file.

This example doubles as the smoke target for the VALIDATE phase's `test_<Block>.cpp`. Next phase:
SKILL (`skill.md` -- the block-local skill card + `manifest.dcf`), per the `00_flow.md` backbone.

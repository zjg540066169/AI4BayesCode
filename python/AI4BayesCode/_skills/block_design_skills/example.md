# module: example.md
# role: block_design EXAMPLE phase — author ONE FRONTEND-INDEPENDENT C++ example
# (examples/<Model>.cpp) that USES the new block via a composite and DEMONSTRATES it with an
# int main(). NO R / Python binding (no Rcpp, no pybind).
# loaded: lazily, only when the EXAMPLE phase is entered (after Stage 4 impl is locked).
# cite, don't restate: block_sampler/composite semantics = system_design §1/§2 (interface.md);
# set_current dispatch / DAG wiring = system_design §3–§9 (dataflow.md).

## !!! GATE CHECK FIRST — did the user accept the example gate? !!!

This module loads ONLY after the user accepted the EXAMPLE gate (asked at the VALIDATE→EXAMPLE
transition — `validate.md §6` / `00_flow.md §4`: `(a) Yes (default) / (b) No — skip to SKILL`).
If you reached this module WITHOUT that gate having been asked — you transitioned straight out of
VALIDATE and "loaded the module" — **STOP: fire the gate NOW** before doing anything here, and if
the user says "no", go to `skill.md` and skip the example entirely. An example is **OPTIONAL** —
some shipped core blocks have none; a block + passing library test is already a complete bundle.

## !!! FRONTEND-INDEPENDENT RULE (non-negotiable) !!!

The example is a **frontend-INDEPENDENT C++ program** — it is bound to NEITHER R NOR Python.
**FORBIDDEN in the example:** `RCPP_MODULE`, `PYBIND11_MODULE`, `#include <RcppArmadillo.h>`,
`#include "AI4BayesCode/rcpp_wrap.hpp"`, `pybind_casters.hpp`, `[[Rcpp::depends(...)]]`, any
`Rcpp::` / `R::` type or call, any `pybind11::` call, and the `AI4BAYESCODE_RCPP_MODULE` /
`AI4BAYESCODE_PYBIND_MODULE` guards. The block must be demonstrable as plain C++.

"Frontend-independent" does NOT mean "dependency-free." The example MAY (and should) freely use
ordinary **C++ libraries** — `<armadillo>`, Eigen, the `celerite` / `libgp` / `autodiff` headers,
and of course the AI4BayesCode headers themselves. The ONLY thing banned is the R/Python BINDING
layer (Rcpp / pybind), not C++ math libraries. Errors use `throw std::runtime_error(...)`, never
`Rcpp::stop` / `ai4b::stop`.

## What this phase produces

ONE small, runnable, FRONTEND-INDEPENDENT worked model — `blocks_local/<Block>/examples/<Model>.cpp`
— that COMPOSES the new block into a `composite_block` and DEMONSTRATES it via an `int main()`
that simulates data, fits, and checks recovery. It is BOTH the human-readable "here is how you use
this block" demo AND the smoke target the test phase compiles. Keep it minimal: the SMALLEST model
that exercises the block's distinctive feature, not a kitchen-sink showcase.

This phase runs ONLY if the user accepted the **example gate** asked at the VALIDATE→EXAMPLE
transition (`00_flow.md §4`: `(a) Yes — write the example (default) / (b) No — skip to SKILL`). If
they declined, you are not here. When it runs, it produces exactly ONE example — a plain C++ binary
(no R, no Python), the cheapest end-to-end check that the block actually works, and what a future
registry submission needs. It is RECOMMENDED (hence the default "yes"), but skippable on request —
a block + passing library test is a valid bundle without it.

> The recurring villain: a block that compiles and runs but produces the WRONG posterior. A
> worked example that visibly recovers a known parameter (sim → fit → compare) is your first,
> cheapest line of defense against that silent failure — long before the heavier
> recovery/parity tests in `test_<Block>.cpp`.
>
> **Tune the demo so it DISCRIMINATES.** Pick the data/noise regime so the naive baseline
> (raw data, independent fit, ignoring the structure this block exploits) has VISIBLE nonzero
> error, and the block visibly beats it. A demo where both the baseline and the block score
> "perfect" (e.g. a too-easy, well-separated field) passes its assertion but demonstrates
> NOTHING — the example exists to SHOW the block's value, so make the baseline visibly fail.

## How the example drives the block (C++ interface only — no frontend)

The example uses the block through its **C++ interface**, NOT through any R/Python method layer.
The composite's `step(rng)` advances the sampler; you read state back with the neutral C++ API
(`comp.data().get("<key>")`, or `dynamic_cast<Block&>(comp.child(0)).current()`). The
`block_sampler` semantics are owned by **system_design §1** (`interface.md`) — do not re-derive
them.

If you wrap the model in a small **driver class** for readability, its methods MUST use NEUTRAL
C++ types only — `AI4BayesCode::state_map` / `history_map` / `arma::vec` — NEVER `Rcpp::List`,
`Rcpp::NumericMatrix`, or any pybind type. But a driver class is OPTIONAL: the simplest example is
just `int main()` driving the composite directly. There is **NO `RCPP_MODULE` and NO
`PYBIND11_MODULE`** — the R/Python binding (the six/seven-method Tier-A wrapper) is a SEPARATE
codegen/packaging concern, NOT part of a frontend-independent example.

## Structure of the file (frontend-independent)

Sections, in order:

1. **Copyright header** (GPL-3.0-or-later; see License gate below).
2. **Header block comment**: the model (likelihood + priors, one display line each), the block
 decomposition (which `composite_block` children, with `<Block>` named), and one line on why
 THIS block (the `SelectWhen` trigger in prose). This comment is the human-readable demo.
3. **Includes**: `<armadillo>` (plain — NOT `<RcppArmadillo.h>`), `block_sampler.hpp`,
 `shared_data.hpp`, `composite_block.hpp`, `constraints.hpp`, AND the new block header (by its
 bundle path, e.g. `#include "<Block>.hpp"` — the compile phase puts `blocks_local/<Block>/` on
 `-I`), plus `<random>` / `<cmath>` / `<cstdio>` for the demo. **NO `rcpp_wrap.hpp`, NO
 `pybind_casters.hpp`, NO `backend_neutral.hpp`, NO `[[Rcpp::depends]]`.** Other C++ libraries
 (Eigen, celerite, libgp, autodiff) are fine if the block needs them.
4. **(optional) anonymous-namespace free functions**: e.g. a log-density/grad closure, file-local.
5. **(optional) a small neutral driver class** wiring data + DAG + children — neutral types only.
6. **`int main()`**: simulate data from a KNOWN truth → build the composite/block → warmup +
 sample → compute a posterior mean → compare to truth (or a naive baseline) → `printf` the result
 → `return ok ? 0 : 1;`. This is the demo AND the smoke target. (For a prior-only block, check the
 sampled draws against the prior's known moments instead of a likelihood recovery.)

### Constructor — minimal wiring

- Build `impl_ = std::make_unique<composite_block>("<Model>")`.
- `impl_->data.set(key, value)` for observed data + initial parameter values.
- Declare the Gibbs-DAG dependencies (`declare_dependencies`) and predict-DAG edges
 (`declare_predict_edges`) + any refreshers — semantics at **system_design §3–§9**
 (`dataflow.md`). Do NOT restate DAG rules here; just wire the minimal set this small model
 needs (often a single child block + one `y_rep` predict edge). The concrete signatures you
 will need (so you don't have to grep `shared_data.hpp`): a **stochastic refresher** is
 `register_stochastic_refresher(key, [](const shared_data_t& d, std::mt19937_64& rng) -> arma::vec {…})`
 (e.g. the `y_rep` posterior-predictive draw); a **deterministic refresher** is
 `register_refresher(key, [](const shared_data_t& d) -> arma::vec {…})`.
- `impl_->add_child(std::make_unique<<Block>>(...))` — construct the new block with its config
 (whatever Stage 4 fixed: name, sub-params/constraints, log-density, hyperparameters).
- `if (keep_history_) impl_->set_keep_history(true);`
- RNG discipline: separate `rng_`, `predict_rng_`, `readapt_rng_` (seed-derived, deterministic
 when seed != 0) — copy the pattern from `GaussianLocationScale.cpp`; see system_design §8.

### `set_current` is a pure DISPATCHER (Tier A → Tier B)

`set_current(const state_map& params)` must NOT contain sampler logic. It only:

1. looks up each KEY it owns in `params` (skip keys not present — partial updates are legal),
2. validates the value (e.g. a positive-constrained parameter must be `> 0` → `throw` on
 violation),
3. routes the value to the corresponding Tier-B setter on the child block (e.g.
 `dynamic_cast<<Block>&>(impl_->child(0)).set_current(vec)` or the block's fine-grained
 `set_X`/`set_Y`-style C++ setter), and
4. mirrors the pushed value back into `impl_->data.set(key,...)` so downstream reads see it.

Which keys, and how they map to Tier-B setters, is the dispatch table — that is system_design
§7 (`dataflow.md`); the example just instantiates it for this one model. See the
`set_current` body in `GaussianLocationScale.cpp` (lines ~209–227) for the canonical
"find key → validate → `jblk.set_current(cat)` → mirror into `data `" shape.

> **Foot-gun — COPY, do not bind-by-reference, the result of `get_current()`.** `get_current()`
> returns a `state_map` BY VALUE (a temporary); binding `const arma::vec& v = get_current().at(key)`
> leaves `v` dangling the instant the temporary dies → a `0x0` crash on first use. Take the map
> into a named local first (`auto s = impl_->get_current(); const arma::vec& v = s.at(key);`), or
> copy the vector out. Same applies to any `.at(key)` on a returned-by-value container.

### Optional driver-class methods (`predict_at` / `get_dag` / `get_history`)

These exist ONLY if you wrote a driver class (optional). If so, keep them NEUTRAL-typed:
`predict_at(const state_map&) -> history_map` (return changed downstream nodes, predict-DAG BFS —
system_design §4), `get_dag` / `get_history` thin forwards. The demo `main()` usually does NOT
need them — a recovery demo only needs `step` + reading the current draw.

### NO frontend module

Do **NOT** write an `RCPP_MODULE` or a `PYBIND11_MODULE`. The example ends with `int main()`. The
six/seven-method R/Python binding (`step`/`get_current`/`set_current`/`predict_at`/`get_dag`/
`get_history` [+ `readapt_NUTS`]) is generated ELSEWHERE (codegen / packaging), never in a
frontend-independent example.

## License gate (the example file is GPL-3.0-or-later; manifest `License:` governs vendored code)

Every file under `examples/` (including `blocks_local/<Block>/examples/`) is part of AI4BayesCode
and is **GPL-3.0-or-later** — `system_design §0` (project license): "all code under `examples/` …
is GPL-3.0-or-later; every file you add or edit MUST carry a matching license header." This is
UNIFORM — it does NOT depend on whether the example pulls BART / mcmclib (the shipped Gibbs-only
examples carry GPL-3.0-or-later too). The example header MUST be the canonical three-line form:

 ```
 // Copyright (C) 2026 <Author>.
 // Licensed under the GNU General Public License v3.0 or later
 // (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
 ```

The per-license header form is `skills/codegen_cpp.md §5` (cited, not restated). The bundle's
`manifest.dcf` `License:` field governs any **vendored** third-party code under `vendor/` (which
keeps its own upstream header verbatim — see `intake.md` Step 5) — NOT the example file, which is
always GPL-3.0-or-later. (A vendored MIT routine stays MIT inside `vendor/`; the example that uses it is
still GPL-3.0-or-later.) No need to ask the user — the example license is fixed by project policy.

## ASK-THE-USER for this phase

Use a structured labeled-option prompt (AskUserQuestion in Claude Code; the markdown
labeled-option fallback from `start.md §0` elsewhere). Mark the standing choice **(default)**,
never "(Recommended)"; always include "Other". Elicit:

- **Which demo model?** — a labeled menu of small models that exercise the block's distinctive
 feature; "(default)" = the smallest one (mirrors the block's `SelectWhen`). "Other" = user
 describes their own.
The **posterior-predictive check is MANDATORY, not asked**: `main()` ALWAYS computes a
`y_rep`-style draw and checks predicted vs observed (it demonstrates the predict path AND gives a
discriminating correctness check — both of which a library block must show). The ONLY carve-out is
a **prior-only block** with no observation model: there is no `y_rep`, so it instead checks sampled
draws against the prior's known moments. That carve-out is mechanism-determined (no likelihood ⇒ no
PPC), NOT a user choice.

(No "Python module?" question and no license question — the example is frontend-independent by rule,
and its license is fixed GPL-3.0-or-later by project policy.) The demo's numerics (sim sample size N, the
true parameter values for the recovery check) use sensible DEFAULTS recorded in the file header —
do NOT interrogate them one at a time; surface them in the recap and let the user override if they
care (same policy as `validate.md` §5).

## STAGING + AUTOMATION (TOP RULE)

Staging is the delivery GUARD, not a gate. Staging the file, compiling, running the demo, AND the
final move to `blocks_local/` all happen **automatically** — nothing is gated on a "go" (same rule
as `validate.md`); the move happens once all checks pass, then the final path is reported. A failed
block is never moved.

- PROPOSE the full `<Model>.cpp` and STAGE it to the staging dir; show it for review.
- MOVE it to `blocks_local/<Block>/examples/<Model>.cpp` AUTOMATICALLY with the bundle on all-pass (optionally record
 the path in the manifest's OPTIONAL `Example:` field — discovery also finds `examples/*.cpp` by convention).
- COMPILE + RUN the demo `main()` **AUTOMATICALLY — no "go" gate.** Once the example is written,
 of course you test it (you can't ship a demo you never ran), so just run it — do NOT ask a "go"
 question. It runs against the STAGING copy (nothing moves), and is the bundle's first end-to-end
 check that the block actually works as plain C++. Surface the command + expected runtime as you
 START (a heads-up, not a gate). Compile recipe = `validate.md` §3 minus any Rcpp/R flags
 (no RcppArmadillo include, no R defines); link plain `<armadillo>` (`-framework Accelerate` on
 macOS / `-lblas -llapack` on Linux).
- The fuller recovery / parity / cross-chain R-hat checks live in the VALIDATE phase (`test_<Block>.cpp`),
 not here; this example is the SMOKE target plus a visible sanity recovery.

## Done-when

- File present at `blocks_local/<Block>/examples/<Model>.cpp` with the GPL-3.0-or-later license header.
- **FRONTEND-INDEPENDENT**: NO `RCPP_MODULE`, NO `PYBIND11_MODULE`, NO `rcpp_wrap.hpp` /
 `pybind_casters.hpp` / `<RcppArmadillo.h>` / `[[Rcpp::depends]]`, NO `Rcpp::` / `R::` / `pybind11::`
 token (except inside a comment). Ends with an `int main()`.
- If a driver class is used, its methods are NEUTRAL-typed (`state_map` / `arma`), and `set_current`
 (if present) is a pure key-routing dispatcher (no sampler logic) that mirrors into `data`.
- **Compiles + RUNS as a plain C++ binary (run automatically, no "go")** — the demo prints a recovery/smoke result and
 returns 0 on pass; the example lives at `examples/<Model>.cpp` (optionally recorded in the manifest's OPTIONAL `Example:` field).
- **Posterior-predictive check present** — `main()` draws `y_rep` and checks predicted vs observed
 (MANDATORY; the ONLY exception is a prior-only block, which checks draws against the prior's known
 moments instead).
- License header == **GPL-3.0-or-later** (uniform project policy for `examples/`); the
 `manifest.dcf` `License:` field governs the block + any `vendor/` code, NOT this example file.

This example doubles as the smoke target for the VALIDATE phase's `test_<Block>.cpp`. Next phase:
SKILL (`skill.md` — the block-local skill card + `manifest.dcf`), per the `00_flow.md` backbone.

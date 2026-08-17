---
name: AI4BayesCode-codegen-python-runner
description: |
  Python runner template for AI4BayesCode samplers -- AI4BayesCode.sourceCpp
  setup, constructor-argument reference block, the delivered
  inline-constructor-lambda + shipped AI4BayesCode.run_chains /
  rhat_summary / diagnose flow (NO generated helpers in the example;
  harness-internal run_chain_<ClassName>() with keep_history=True is
  NOT shipped), Layer 3 validator wiring (R1 smoke
  check, R2 rank-normalized R-hat + ESS (AI4BayesCode numpy helpers), R3 Bayesian p-values
  + PSIS-LOO via arviz), AI4BayesCode.perf_hint call, joint-NUTS threshold
  tightening, and the reference-template catalogue (examples/*.cpp).
  Mirror of codegen_r_runner.md for the Python (pybind11) backend.
  The entry-point skill `codegen.md` and `start.md` Phase 6 point here
  for Python runner emission when the chosen runtime is Python (or
  Both R+Python dual-module).
---

# AI4BayesCode codegen -- Python runner + reference templates

**PREREQUISITE GATE -- L2 first.** Do not load this file, and do not
emit or run any runner / Layer-3 harness, until the L2 semantic
checklist + AD-twin (Check #12) have PASSED with the per-check verdict
table printed (`codegen.md` Sec.11 Step 2, HARD ORDERING GATE). This
file hands you executable runtime machinery; if L2 has not been
printed yet, you are in the runtime-before-semantic reversal.

Companion skill to `codegen.md`. Load this when writing the generated
`.py` runner: `AI4BayesCode.sourceCpp` setup, constructor-argument
reference block, the delivered inline-constructor-lambda + shipped
`AI4BayesCode.run_chains` / `rhat_summary` / `diagnose` flow, the
harness-internal `run_chain_<ClassName>()` helper, Layer 3 validator
wiring (R-hat, ESS, Bayesian p-values, PSIS-LOO), `AI4BayesCode.perf_hint`,
and the reference-template catalogue.

For prior elicitation + block selection, see `codegen_priors.md`.
For the C++ file emission (PYBIND11_MODULE block, type casters), see
`codegen_cpp.md` Sec.9 "Backend module declarations" (Python-only and
dual R+Python forms).

This skill mirrors `codegen_r_runner.md` for the Python backend. Where
the logic is identical between R and Python (prior emission, L2
semantic check, R3 BPV semantics, perf_hint thresholds), refer to the R
skill rather than duplicating content. This file covers only
Python-specific patterns (`AI4BayesCode.sourceCpp`, `multiprocessing`
parallel chains, `arviz` diagnostics, numpy / dict idioms).

---

## 0. Audience disambiguation

This skill is for the codegen agent emitting a Python runner when the
chosen runtime backend is **Python** or **Both R+Python**. If the user
chose R-only, load `codegen_r_runner.md` instead. If the user chose
dual R+Python, load BOTH this file and `codegen_r_runner.md` -- the
two runners share the same .cpp file (with `#ifdef`-guarded
RCPP_MODULE and PYBIND11_MODULE blocks per `codegen_cpp.md` Sec.9), and
the runner deliverables are independent (`.R` + `.py`).

## 9. Output

1. Create output folder if missing (default `./generated/<ClassName>/`).
2. Write `<folder>/<ClassName>.cpp`. **It MUST carry BOTH the
   `@example:python` AND the `@example:R` header blocks** (per
   `codegen_cpp.md` Sec.5 "Header `@example` block") -- the `.cpp` is ALWAYS
   dual-module, so an R user who takes the same file must still see the R
   example via `ai4bayescode_doc()`, and a Python user the Python example via
   `AI4BayesCode.doc()`. Both blocks are the SAME toy DGP this runner uses,
   distilled to <= ~8 runnable lines using each language's packaged API
   (`AI4BayesCode.source(...)` -> `Mod(...)` for Python; `new(<ClassName>,
   ...)` for R). Write the DGP once and mirror it into BOTH header blocks so
   the runner and both doc cards cannot drift.
3. Write the Python runner `<folder>/run_<ClassName>.py` following the
   template below and **use it to gate generation** (it carries the
   Layer-3 R1/R2/R3 harness -- same CORRECTNESS INVARIANT as the R
   side: Layer-3 always runs and must PASS regardless of the user's
   harness choice).
4. **Deliverable depends on the "Delivered-code validation harness"
   answer** (same rule as R version):
   - **No / minimal example (default):** on validator PASS, **DELETE**
     `run_<ClassName>.py` and write a minimal
     `<folder>/example_<ClassName>.py` instead.
   - **Yes (opt-in):** keep `run_<ClassName>.py` as the delivered
     artifact.
   On validator FAIL within `max_attempts`: stop-and-report; do NOT
   write/ship the usage example.
5. Do NOT modify anything outside the output folder.
6. Tell the user: path, class name, exact Python commands (for the
   default case, the commands are just the usage example).

### Usage-example template (`example_<ClassName>.py`, default deliverable)

**READABILITY IS THE HIGHEST RULE for the delivered example.** The
audience is applied researchers with NO programming background: one
linear script, one visible call per action, no user-defined functions,
no abstractions, and as short as correctness allows. Whenever a
structural choice trades completeness or cleverness against
readability, readability wins. All comments and strings in the
delivered files are ENGLISH, regardless of the conversation language
(codegen.md Sec.0b). **When in doubt about a comment, LEAVE IT OUT**:
a comment you are not sure the user needs, or that could confuse them
(undefined acronyms, internal machinery, absence-explanations,
risk talk), is worse than no comment -- the default is omission.

Same rule as the R version: the delivered example drives chains
EXCLUSIVELY through the SHIPPED helpers (`AI4BayesCode.run_chains`,
`AI4BayesCode.rhat_summary`, `AI4BayesCode.diagnose`), with NO
generated helper functions at all -- the constructor is an inline
lambda argument.
The `run_chain_<ClassName>()` helper defined in the runner template
below is HARNESS-INTERNAL (Layer-3 R1/R2/R3 + sim1 need its
`pp = predict_at(...)` collection) and is NEVER copied into the
delivered example; the whole harness (R2 R-hat/ESS, R3 BPV, PSIS-LOO,
R1 smoke wiring) is throwaway and NOT shipped in the default case.

**Emitting the advanced-control block (agent-facing rules -- these do NOT
go into the file).** Same as the R skill: `readapt_NUTS` lines are
CONDITIONAL on a NUTS-family child; `freeze()` is strict (empty /
unknown / blacklisted family -- BART, genBART, softbart, VI, HMM --
raise) while `unfreeze()` is permissive (`unfreeze()` releases all,
`unfreeze([])` raises); element-level freeze works only on PER-ELEMENT
components, coupled ones (ORDERED / SIMPLEX / CORR_MATRIX / COV_MATRIX)
raise; nested dot-paths and rjmcmc sub-keys are also valid names; batch
refreeze takes `quiet=True`. Show only what THIS model supports.

The delivered `example_<ClassName>.py` MUST contain, in order:

1. **Header comment** -- what the model is, plus the short list of the
   shipped helpers the example uses (`AI4BayesCode.run_chains` /
   `rhat_summary` / `diagnose`, one plain-language line each). No
   harness talk, no validator vocabulary (PSIS-LOO, BPV, Layer-3,
   check numbers), no explanations of what the example does NOT
   contain -- an acronym with no context, or an absence-explanation,
   only confuses the audience.
2. **AI4BayesCode.sourceCpp call** -- exact same form as in the runner
   template below.
3. **Compact constructor reference + doc() pointer** -- a SHORT
   comment block: the `mod.<ClassName>(...)` call shape with one brief
   comment per data argument, then ONE pointer line:

   ```python
   # Full constructor / methods / prior reference:
   #   AI4BayesCode.doc("<ClassName>")
   ```

   The FULL 30-40 line argument + methods documentation lives in the
   `.cpp` header (codegen_cpp.md Sec.5) -- that is exactly what
   `AI4BayesCode.doc()` parses and prints, so do NOT duplicate it into
   the example (the full block stays in the harness runner only). A
   wall of reference comments is what makes the example unreadable for
   the non-programmer audience.
4. **NO generated helper functions -- the constructor goes INLINE.**
   The example defines NOTHING between the data and the
   `AI4BayesCode.run_chains` call. The constructor is passed as an
   inline lambda, exactly like the library examples' own `@example`
   headers:

   ```python
   chains = AI4BayesCode.run_chains(
       lambda seed: mod.<ClassName>(<data_args>, rng_seed=int(seed),
                                    keep_history=True),
       seeds=[101, 202, 303, 404], n_burn=4000, n_keep=4000)
   ```

   HARD RULE (the audience is users with NO programming background:
   one visible call, zero abstractions): do NOT emit a named
   constructor factory (`make_<ClassName>` or ANY
   function-returning-function), do NOT emit a bespoke chain driver
   (`run_chain_<ClassName>()` or any burn/collect/return wrapper in
   the deliverable), and do NOT emit
   `for i in range(n_keep): m.step(1); m.get_current()` accumulation
   loops (draws come from `keep_history=True` + `get_history()`, which
   `AI4BayesCode.run_chains` already does internally). The lambda
   reading the data variables defined in the simulation block above is
   fine and intended -- an example script is one linear file. The ONLY
   reason to expand it -- still inline via a small `def` used ONCE,
   only when a kernel-control pin must run inside each worker:

   ```python
   def _new_model(seed):   # only when the model pins a block
       m = mod.<ClassName>(<data_args>, rng_seed=int(seed), keep_history=True)
       m.freeze(["sigma"])
       return m
   chains = AI4BayesCode.run_chains(_new_model,
                                    seeds=[101, 202, 303, 404],
                                    n_burn=4000, n_keep=4000)
   ```

   The harness-internal `run_chain_<ClassName>()` stays in the
   throwaway runner only.
5. **Synthetic data block** -- produces the same fixtures the harness
   used so the example runs as-is.
6. **Multi-chain usage + diagnostics (the everyday flow)** -- the
   shipped helpers, nothing else:

   ```python
   chains = AI4BayesCode.run_chains(
       lambda seed: mod.<ClassName>(<data_args>, rng_seed=int(seed),
                                    keep_history=True),
       seeds=[101, 202, 303, 404], n_burn=4000, n_keep=4000)
   print(AI4BayesCode.rhat_summary(chains))   # convergence across chains
   tbl, plot = AI4BayesCode.diagnose(chains[0]["hist"])
   ```

   Single-chain use is `seeds=[1]` (or the stateful API below); never
   a bespoke wrapper.
7. **Stateful-API usage**, in order: `model.step()`,
   `model.get_history()`, `model.get_current()`,
   `model.predict_at({"<X>": X_test})`, then
   `model.set_current({...})` LAST, AFTER `predict_at`, and COMMENTED
   OUT (its updated value comes from an outer Gibbs composition, so as
   live code it would error in a standalone run).

Skeleton (parameterized; mirror this structure, fill placeholders):

```python
# Usage example for <ClassName> -- <one line: what the model is>.
# Everything runs through the shipped library helpers:
#   AI4BayesCode.run_chains   -- run several MCMC chains in parallel
#   AI4BayesCode.rhat_summary -- convergence check across the chains
#   AI4BayesCode.diagnose     -- posterior summaries + diagnostic plots
import numpy as np
import AI4BayesCode

# Compile + load the model.
mod = AI4BayesCode.source("<ClassName>.cpp")

# <compact constructor reference: the mod.<ClassName>(...) call shape +
#  one brief comment per data argument -- keep it SHORT>
# Full constructor / methods / prior reference:
#   AI4BayesCode.doc("<ClassName>")

# Simulate a toy data set
# <commented synthetic generation of <data_args> + a held-out X test>

# Multi-chain run + diagnostics (the everyday flow; shipped helpers only).
# The model constructor is the inline lambda -- it builds one fresh model
# per chain from the data simulated above.
chains = AI4BayesCode.run_chains(
    lambda seed: mod.<ClassName>(<data_args>, rng_seed=int(seed),
                                 keep_history=True),
    seeds=[101, 202, 303, 404], n_burn=4000, n_keep=4000)

print(AI4BayesCode.rhat_summary(chains))   # convergence across the chains

# Per-chain summaries + trace + autocorrelation + density plots.
tbl, plot = AI4BayesCode.diagnose(chains[0]["hist"])
print(tbl)
# matplotlib is an optional extra (pip install "AI4BayesCode[viz]"), and
# diagnose returns plot = None without it -- calling None raises TypeError.
if plot is not None:
    plot()

# Examples of stateful functions

# Initialize with full history (keep_history=False keeps only the last draw)
model = mod.<ClassName>(..., rng_seed=42, keep_history=True)
model.step(200)                              # burn-in
model.step(200)                              # keep
hist = model.get_history()                   # dict of arrays, one row per STEPPED
                                             # iteration (here 400 = burn + keep)
y_rep = model.predict_at({})["y_rep"]        # posterior predictive at training X

# set_current() updates the sampler statefully (e.g. an outcome
# refreshed by other blocks) WITHOUT reinitialization. Shown
# COMMENTED OUT -- uncomment in a real stateful-composition context:
# model.set_current({"<a data input>": <updated value>})
# model.step(1)                              # one iteration after set_current
# model.get_history()

# ---- Advanced control (uncomment what you need) ---------------------
#
# Re-tune the sampler without advancing the chain -- useful after
# set_current() changes the data enough that the old tuning no longer
# fits. (Only present when the model uses NUTS.)
# model.readapt_NUTS(500)
#
# Hold one parameter fixed instead of sampling it -- e.g. pin sigma = 1
# for a probit-style model, or fix a parameter for a sensitivity check.
# model.set_current({"sigma": 1.0})
# model.freeze(["sigma"])     # sigma stays 1.0; everything else keeps sampling
# model.step(1)
# model.get_frozen()          # -> ["sigma"]
# model.unfreeze(["sigma"])   # resume sampling it
#
# Same thing in one line at construction:
# model = AI4BayesCode.new_frozen(mod.<ClassName>, <data_args>, rng_seed=42,
#                                 keep_history=True,
#                                 fixed={"sigma": 1.0})
#
# You can also hold a single component of a jointly-sampled parameter,
# or one element of it:
# model.freeze(["<component_name>"])
# model.freeze(["<component_name>[2]"])
```

### Python runner template -- standard body (DEFAULT; harness-internal)

This is the DEFAULT `run_chain_<ClassName>` body for the THROWAWAY
Layer-3 harness (R1/R2/R3 + sim1 need the `pp = predict_at(...)`
collection below, which the shipped `AI4BayesCode.run_chains` does not
do). It is NEVER copied into the delivered example. Use it unless the
composite has a NUTS-family child whose conditional posterior shifts
across outer Gibbs sweeps -- in that case use the periodic-readapt body
in the next subsection instead.

```python
def run_chain_<ClassName>(<data_args>, *, seed, n_burn, n_keep,
                          newdata=None, diagnosis=False):
    if newdata is None:
        newdata = {}
    model = mod.<ClassName>(<data_args>, rng_seed=int(seed), keep_history=True)
    # Kernel-control (freeze/unfreeze): must be re-issued per worker.
    # run_chain constructs a fresh model per chain (via multiprocessing.Pool);
    # freeze state does NOT auto-propagate from the outer scope. If you
    # want to pin sigma (probit) or hold any block, add the calls HERE:
    #   model.set_current({"sigma": 1.0})
    #   model.freeze(["sigma"])
    # Then also set `frozen_names = ["sigma"]` at runner scope for R2.f
    # exclusion. See validator.md Sec.R2.f
    t0 = time.time()
    model.step(int(n_burn))
    model.step(int(n_keep))
    wall = time.time() - t0
    # get_history()/predict_at() return per-iteration draws for EVERY iteration
    # stepped (burn-in + keep). Slice off burn-in so downstream sees only kept draws.
    keep = slice(int(n_burn), int(n_burn) + int(n_keep))
    def _slice(d):
        return {k: (np.asarray(v)[keep] if np.asarray(v).ndim == 1 else np.asarray(v)[keep, ...])
                for k, v in d.items()}
    out = {"hist": _slice(model.get_history()),
           "pp":   _slice(model.predict_at(newdata)),   # {} = posterior predictive at training X
           "wall_sec": wall}
    if diagnosis:
        out["diagnosis"], out["diagnosis_plot"] = \
            AI4BayesCode.diagnose(out["hist"], n_burn=0)   # already sliced -> n_burn=0
    return out
```

**Note -- pre-slice vs. n_burn.** The standard body above pre-slices the
history inside the runner, so it calls `AI4BayesCode.diagnose(..., n_burn=0)`
(the draws are already burn-in-stripped). The periodic-readapt body below
does NOT pre-slice and passes `n_burn=int(n_burn)` -- mixing these up (calling
`diagnose(..., n_burn=int(n_burn))` on already-sliced draws, or vice versa) is
an easy mistake that silently drops or double-strips warmup.

**Return-shape contract (harness helper).** `run_chain_<ClassName>`
MUST return a dict whose `"hist"` is a dict of numpy arrays keyed by
parameter name -- scalars as `(n_keep,)` 1-D arrays, vectors as
`(n_keep, dim)` 2-D arrays. NEVER a list-of-per-step-dicts (the R2
`_stack_param` helper branches on `arr.ndim == 1`, so a list breaks
it). This is exactly what `model.get_history()` returns. Multi-chain
R-hat in the DELIVERED example is always the shipped pair
`AI4BayesCode.run_chains(lambda seed: mod.<ClassName>(...), seeds=...)` +
`AI4BayesCode.rhat_summary(chains)` (each chain's `"hist"` is keep-only;
run_chains strips the warmup draws) -- never a hand-rolled loop or a
hand-assembled arviz call. Over-dispersed
initial values, when needed, go through `model.set_current(...)`
inside the inline constructor.

### Constructor reference block

The constructor block must list ALL arguments the user can pass, their
types, and brief descriptions. If hyperparameters are exposed as
constructor arguments (see `codegen.md` Sec.2 and `codegen_priors.md`),
document those too with their defaults. Also document the history keys
and array shapes `get_history()` returns. `AI4BayesCode.run_chains`
puts the KEEP-ONLY version in each chain's `"hist"` (it strips the
warmup draws); a raw `get_history()` call on a stateful model still
spans every stepped iteration.

Concrete example -- the constructor block for a BART model:

```python
# -------------------------------------------------------------------------
#   BartNoise(
#       X:            (N, p) float array   -- predictor matrix
#       y:            (N,)   float array   -- response vector
#       ntrees:       int    = 200         -- number of trees
#       rng_seed:     int                  -- RNG seed (0 = random_device)
#       keep_history: bool   = False       -- record per-iter draws
#   )
#   NOTE: sigma is initialized internally to bart_model's OLS-based sigest.
#         For an overdispersed start (R-hat diagnostics), call
#         model.set_current({"sigma": ...}) AFTER construction.
#
#   Methods:
#     .step(n)         -- run n Gibbs sweeps (one BART + one NUTS-sigma)
#     .get_current()   -> {"f_bart": (N,), "sigma": float}
#     .set_current(d)  -- overwrite sigma; f_bart is read-only
#
#   get_history() keys (run_chains returns these KEEP-ONLY in chain["hist"];
#   a raw get_history() call spans burn + keep):
#     hist["f_bart"] -- (n_iter, N) posterior draws of f
#     hist["sigma"]  -- (n_iter,)   posterior draws of sigma
# -------------------------------------------------------------------------
```

### Modular NUTS in composite -- periodic readapt schedule (CONDITIONAL)

Beyond the sequential-update use case above, `readapt_NUTS` is ALSO
required for modular NUTS-in-composite where a `nuts_block` samples a
parameter whose conditional posterior shifts across outer Gibbs
iterations (typical when sigma^2, regression coefficients, hyper-
parameters, or RJMCMC inclusion indicators update in a sibling block).
Without periodic re-adaptation, the persistent metric from the initial
warmup becomes mis-tuned for later iterations' conditional,
manifesting as the **stuck-fast pattern** (R-hat above 2, ESS in
single digits, chain values bit-identical across hundreds of `step()`
calls).

If the wrapper's composite has any NUTS-family child sampling a
parameter whose conditional posterior is shifted by a sibling block
(Gibbs, rjmcmc, VI, etc.), runners MUST emit a periodic readapt
schedule INSIDE the chain runner. The R-side equivalent rule is in
`codegen_r_runner.md` "Modular NUTS in composite -- periodic readapt
schedule".

**Defaults.** `readapt_every = 500` outer iters, `readapt_n = 50`
re-adapt iters per call (~10% overhead). Tighten to
`readapt_every = 100` if R-hat fails at 4k+4k; loosen to
`readapt_every = 1000` if the chain mixes well and wall is critical.
There is no published canonical value for modular NUTS-in-composite
re-adaptation frequency -- standard frameworks (Stan, PyMC, NumPyro)
use single-warmup + frozen-metric and do not address this regime; the
defaults here are empirically calibrated.

Python emit pattern (note: `readapt_every`, `readapt_n` are
**function arguments**, not script-scope globals, so the helper stays
self-contained -- sim1 / cross-impl harnesses can vary them per
replicate without touching the function body):

```python
def run_chain_<ClassName>(<data_args>, *, seed, n_burn, n_keep,
                          readapt_every=500, readapt_n=50,
                          newdata=None, diagnosis=False):
    if newdata is None: newdata = {}
    model = mod.<ClassName>(<data_args>, rng_seed=int(seed),
                             keep_history=True)
    t0 = time.time()
    # Periodic readapt schedule -- covers BOTH burn-in and keep, since the
    # conditional keeps shifting throughout sampling under Gibbs siblings.
    total = int(n_burn + n_keep)
    full = total // readapt_every
    for _ in range(full):
        model.readapt_NUTS(readapt_n, False)
        model.step(readapt_every)
    remainder = total - full * readapt_every
    if remainder > 0:
        model.readapt_NUTS(readapt_n, False)
        model.step(remainder)
    # Strip warmup, exactly as the default helper does. Without this the R2
    # R-hat and R3 gates are computed over burn-in draws, and ess_ratio (which
    # divides by n_keep downstream) can exceed 1, making the 0.005 floor
    # meaningless.
    keep = slice(int(n_burn), int(n_burn) + int(n_keep))
    def _slice(d):
        return {k: (np.asarray(v)[keep] if np.asarray(v).ndim == 1
                    else np.asarray(v)[keep, ...])
                for k, v in d.items()}
    out = {"hist": _slice(model.get_history()),
           "pp":   _slice(model.predict_at(newdata)),
           "wall_sec": time.time() - t0}
    # diagnosis=True attaches model-INDEPENDENT posterior diagnostics via the
    # SHIPPED library function AI4BayesCode.diagnose() -- do NOT reimplement:
    #   out["diagnosis"]      -> per-parameter table (R-hat / ESS / mean / sd / 90% CI)
    #   out["diagnosis_plot"] -> a callable drawing trace + autocorrelation + density
    # get_history() returns burn-in + keep, so pass n_burn to strip it.
    if diagnosis:
        out["diagnosis"], out["diagnosis_plot"] = \
            AI4BayesCode.diagnose(out["hist"], n_burn=int(n_burn))
    return out

# NOTE: AI4BayesCode.diagnose (the diagnostics table + the trace/ACF/density plot) is a
# SHIPPED function in the AI4BayesCode package -- the runner CALLS it (above);
# it does NOT define its own copy. The summary uses rank-normalized
# split-R-hat / ESS (numpy),
# correct for the single chain a runner produces, and the plot uses matplotlib.
```

**HARD RULE -- the `diagnosis=True` path is non-negotiable.** The diagnostics
AND the plot are a SHIPPED library function, `AI4BayesCode.diagnose(hist,
n_burn=...)`, which returns `(summary_table, plot_fn)` where `plot_fn()` draws
trace + autocorrelation + density with matplotlib. Every generated runner MUST
(1) take a `diagnosis=False` argument; (2) when `diagnosis=True`, CALL
`AI4BayesCode.diagnose()` and attach BOTH `out["diagnosis"]` (table) AND
`out["diagnosis_plot"]` (callable). Do NOT reimplement it inline. This is
independent of how the runner collects draws: pass whatever named dict of kept
draws you built as `AI4BayesCode.diagnose(draws, n_burn=...)` -- use
`n_burn=0` when the draws are already burn-in-stripped (`get_history()` returns
burn-in + keep, so pass the burn-in length there). FORBIDDEN -- an inline
summary-only diagnosis that drops the plot (e.g. `out["summary"] =
az.summary(...)` with no `diagnosis_plot`): that returns a table with NO
trace/ACF/density plot and renames the field. ALWAYS route through
`AI4BayesCode.diagnose()` and expose `out["diagnosis"]` +
`out["diagnosis_plot"]`.

Skip the periodic schedule (use plain `model.step(n_burn);
model.step(n_keep)`) ONLY when the composite is pure-NUTS-on-a-fixed-
conditional (e.g. a single `joint_nuts_block` sampling everything
jointly with no Gibbs siblings) -- in that case the conditional does
not shift and the initial warmup metric stays correct.

### Python runner template

The generated `run_<ClassName>.py` must include:
1. A **constructor reference block** documenting every argument to
   `mod.<ClassName>(...)` with its type, description, default/valid
   range.
2. A burnin phase and a draw-collection loop (the periodic readapt
   pattern above when applicable).
3. The `diagnosis=False` parameter on `run_chain_<ClassName>`, and when
   `diagnosis=True` a CALL to the shipped `AI4BayesCode.diagnose()`
   giving `chain["diagnosis"]` (table) and `chain["diagnosis_plot"]`
   (trace + ACF + density). Do NOT write your own helper and never inline
   a summary-only substitute (see the HARD RULE below).

Follow this structure:

**Path-resolution rule (mirror R skill -- no runtime detection).** The
generated runner uses HARDCODED RELATIVE PATHS. No `inspect.stack`,
no `__file__`-walking, no `os.getcwd` checks. The contract: the user
runs Python from **the project root** -- the directory containing both
`AI4BayesCode/` AND the `<folder>/` where the generator wrote the
`.cpp`. If invoked elsewhere, the user gets a clear filesystem error
and the fix is documented.

```python
# === run_<ClassName>.py -- Layer 3 harness (R1 + R2 + R3) ===
import os, sys, time
import numpy as np

# Packaged API -- install once (`pip install ai4bayescode`); no checkout,
# no sys.path hack, no ai4bayescode_path. Headers travel in the package.
import AI4BayesCode

# Compile + load the generated .cpp; returns the module (the class is an
# attribute, e.g. mod.<ClassName>).
mod = AI4BayesCode.source("<folder>/<ClassName>.cpp")

# Constructor reference (one row per argument; every codegen produces this)
# -------------------------------------------------------------------------
#   <ClassName>(
#       <arg_1>:        <type>      <range/default>     # <description>
#       <arg_2>:        ...
#       ...
#       rng_seed:       int         0 = random_device   # PRNG seed
#       keep_history:   bool        False               # record per-iter draws
#   )
# -------------------------------------------------------------------------

# (Synthetic data block -- exactly matches the codegen_priors.md "Validation
# fixture" guidance; see Sec.6 of that skill for the shape conventions.)
# ... <data setup> ...
#
# HIERARCHICAL / random-effects / weight-variance models -- draw the scale or
# variance hyperparameter FROM ITS PRIOR, then draw the effects at that scale;
# do NOT hard-code the scale to an arbitrary "moderate" value (e.g. sd=0.6).
# A fixed moderate scale can CONFLICT with a tight or heavy-tailed prior -- the
# classic case is a tiny-scale InvGamma weight-variance prior (Neal-1996 / ARD
# weight-variance BNN, where the true variance can be ~1e-6): fixed sd 0.6 makes the DATA say
# "sigma^2 ~ 0.36" while the PRIOR insists "sigma^2 ~ 1e-6", an artificially HARD
# prior-data-conflict posterior (R-hat stuck ~1.02, tiny ESS) wrongly blamed on
# the sampler. Prior-drawn hyperparameters keep the L3 self-test calibrated
# (SBC-style). If the model ships a reference simulator (simulate_data(seed)),
# PREFER it over an invented DGP.

# Per-chain runner helper (with periodic readapt -- see Sec. above).
# Signature MUST include diagnosis=False; when diagnosis=True, CALL the
# shipped AI4BayesCode.diagnose() -- do NOT write your own helper.
def run_chain_<ClassName>(<data_args>, *, seed, n_burn, n_keep,
                          readapt_every=500, readapt_n=50, newdata=None,
                          diagnosis=False):
    ...  # verbatim from the template above, INCLUDING the attach:
    #   if diagnosis:
    #       out["diagnosis"], out["diagnosis_plot"] = \
    #           AI4BayesCode.diagnose(out["hist"], n_burn=int(n_burn))
# NOTE: AI4BayesCode.diagnose (table + trace/ACF/density plot) is SHIPPED in the
# AI4BayesCode package -- there is NO helper to define here.

# === R1. Smoke test ===
m_smoke = mod.<ClassName>(<data_args>, rng_seed=42, keep_history=False)
m_smoke.step(10)
cur = m_smoke.get_current()
assert all(np.all(np.isfinite(np.asarray(v))) for v in cur.values()), \
    "R1: get_current contains non-finite values after 10 steps"
cur_before = {k: np.asarray(v).copy() for k, v in m_smoke.get_current().items()}
pp = m_smoke.predict_at({})
assert "y_rep" in pp and np.all(np.isfinite(np.asarray(pp["y_rep"]))), \
    "R1: predict_at must produce finite y_rep"
cur_after = m_smoke.get_current()
for k, v_before in cur_before.items():
    v_after = np.asarray(cur_after[k])
    assert np.allclose(v_before, v_after), f"R1: predict_at mutated {k}"
print("R1 smoke OK")

# === R2. 2-chain MCMC convergence (sequential; see _run_two_chains) ===
# Defer to validator.md Sec.R2 for the budget/escalation policy + the
# Dirac spike-and-slab Sec.R2.s exclusion rule for per-coordinate slab
# DISTRIBUTION parameters (e.g. per-j slab variance tau2_beta /
# tau2_theta) -- NOT the slab-modelled beta_j / theta_jk themselves.

def _worker(args):
    seed, n_burn, n_keep = args
    return run_chain_<ClassName>(<data_args>, seed=seed,
                                  n_burn=n_burn, n_keep=n_keep)

n_burn = 4000; n_keep = 4000

# Flip to True if this runner's composite contains a joint_nuts_block
# (narrows the R3 BPV pass band from (0.05, 0.95) to (0.10, 0.90) --
# mirrors USES_JOINT_NUTS in codegen_r_runner.md).
USES_JOINT_NUTS = <True if composite has joint_nuts_block else False>

def _run_two_chains(n_burn, n_keep):
    """Run the two diagnostic chains SEQUENTIALLY; return (c1, c2, wall).

    Sequential ON PURPOSE: this runner is a standalone script run via
    `python runner.py`. A module-level process Pool under the 'spawn' start
    method (the macOS / Windows default) re-imports the runner in every worker
    -> a bootstrapping RuntimeError, so the runner never reaches its
    AI4BAYES_VALIDATE line (validation fails at stage `incomplete`). Two chains
    run fast enough sequentially, and this mirrors the R runner, which also
    runs its two validation chains sequentially.
    """
    t0 = time.time()
    c1 = _worker((101, n_burn, n_keep))
    c2 = _worker((202, n_burn, n_keep))
    return c1, c2, time.time() - t0

c1, c2, total_wall_sec = _run_two_chains(n_burn, n_keep)

# --- R-hat / ESS aggregation (rank-normalized, via arviz) ----------------
# Matches validator.md Sec.R2 and codegen_r_runner.md r2_diag(): rank-normalized
# R-hat is a HARD gate (< 1.05); ESS is a SOFT criterion via
# ess_ratio = min(ESS_bulk, ESS_tail) / n_keep -- >= 0.01 silent,
# [0.005, 0.01) WARN and proceed, < 0.005 escalate. AI4BayesCode.rhat is the
# rank-normalized split-R-hat of Vehtari et al. (2021) and matches
# posterior::rhat R-side, so the 1.05 gate means the same thing on both.
# AI4BayesCode.ess_bulk / ess_tail use a simpler autocorrelation estimator
# and are indicative, not identical to posterior::ess_bulk / ess_tail.
#
# Sec.R2.s conditional-relevance exclusion (Dirac spike-and-slab): for per-j
# slab DISTRIBUTION parameters (per-coordinate slab variance tau2_beta /
# tau2_theta and similar -- NOT the slab-modelled beta_j / theta_jk
# themselves), mask draws by inclusion indicator I_j == 1, truncate the two
# chains to the common min retained count, then feed to AI4BayesCode.rhat. See
# validator.md Sec.R2.s for the precise rule and the I_j  in  {0, 1} convention.

def _stack_param(c1, c2, key):
    """Stack two chains' draws for one parameter into an arviz-shaped array.

    Returns a list of per-component (chain, draw) arrays so a vector
    parameter (n_keep x dim) is diagnosed component-by-component, exactly
    like the R-side `apply(arr, 3, posterior::rhat)`.
    """
    a1 = np.asarray(c1["hist"][key]); a2 = np.asarray(c2["hist"][key])
    if a1.ndim == 1:                       # scalar parameter -> single component
        return [np.stack([a1, a2], axis=0)]            # shape (2, n_keep)
    return [np.stack([a1[:, j], a2[:, j]], axis=0)     # shape (2, n_keep)
            for j in range(a1.shape[1])]

def r2_diag(c1, c2, n_keep_used):
    worst_rhat = 0.0
    worst_ess_ratio = np.inf
    for nm in c1["hist"].keys():
        comp_rhat = 0.0
        comp_min_ess = np.inf
        for arr in _stack_param(c1, c2, nm):           # arr: (chains=2, draws)
            x = np.asarray(arr).T                       # -> (draws, chains)
            rh = float(AI4BayesCode.rhat(x))            # rank-normalized split-R-hat,
            eb = float(AI4BayesCode.ess_bulk(x))        # numpy (matches posterior::rhat /
            et = float(AI4BayesCode.ess_tail(x))        # arviz) -> no arviz dep, no
            #                                             az.rhat()->Dataset scalar footgun
            comp_rhat = max(comp_rhat, rh)
            comp_min_ess = min(comp_min_ess, eb, et)
        ess_ratio = comp_min_ess / n_keep_used
        print(f"  {nm:<14}  max Rhat={comp_rhat:.3f}  "
              f"min ESS={comp_min_ess:.0f}  ess_ratio={ess_ratio:.4f}")
        worst_rhat = max(worst_rhat, comp_rhat)
        worst_ess_ratio = min(worst_ess_ratio, ess_ratio)
    return {"rhat": worst_rhat, "ess_ratio": worst_ess_ratio}

# R2.f: exclude frozen params (kernel-control) from R-hat / ESS aggregation.
# See validator.md Sec.R2.f. Frozen params appear as constant columns in
# get_history() (composite appends a repeat every skipped step to preserve
# shape); their R-hat is undefined (NaN). Excluding them keeps the diagnostic
# honest and logs the exclusion for user verification.
# NOTE: if run_chain_<ClassName>() calls model.freeze(...) inside the worker,
# set `frozen_names` here to the same list so R2.f can filter.
frozen_names: list[str] = []   # <-- set to ["sigma", ...] if run_chain freezes
def _apply_frozen_filter(chain):
    if not frozen_names:
        return chain
    filt = {k: v for k, v in chain["hist"].items() if k not in frozen_names}
    return {**chain, "hist": filt}
if frozen_names:
    print(f"[R2.f] Excluding {len(frozen_names)} frozen param(s) from R2: "
          f"{', '.join(frozen_names)}")

# Stage 1 diagnostic at the template's existing budget (4k + 4k).
d = r2_diag(_apply_frozen_filter(c1), _apply_frozen_filter(c2), n_keep)

# Stage-2 escalation (within-attempt): re-run at 20000 + 20000 if Stage-1
# shows max R-hat >= 1.05 OR a severely low ess_ratio (< 0.005), then
# recompute. A slow-but-correct model gets the bigger budget BEFORE being
# declared a failure (validator.md Sec.R2). Do NOT hard-fail ESS at Stage 1.
if d["rhat"] >= 1.05 or d["ess_ratio"] < 0.005:
    print("  [R2] Stage-1 inadequate -> escalating to 20000 + 20000 ...")
    n_burn = 20000; n_keep = 20000
    c1, c2, total_wall_sec = _run_two_chains(n_burn, n_keep)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in c1["hist"].values())
    assert all(np.all(np.isfinite(np.asarray(v))) for v in c2["hist"].values())
    d = r2_diag(_apply_frozen_filter(c1), _apply_frozen_filter(c2), n_keep)

# worst rank-normalized R-hat across all parameters -- drives the final gate.
worst_rhat = d["rhat"]

# Final R2 gates: R-hat HARD (< 1.05); ESS SOFT (only ess_ratio < 0.005 at the
# escalated budget is a FAILURE; [0.005, 0.01) only warns -- legitimate for slow
# GP / hierarchical models).
assert worst_rhat < 1.05, f"R2 FAIL: worst rank-Rhat {worst_rhat:.4f} >= 1.05"
if d["ess_ratio"] < 0.005:
    raise RuntimeError(
        f"R2 FAIL: worst ess_ratio {d['ess_ratio']:.4f} < 0.005 "
        f"even at the escalated budget")
elif d["ess_ratio"] < 0.01:
    import warnings
    warnings.warn(
        f"R2: worst ess_ratio {d['ess_ratio']:.4f} in [0.005, 0.01) -- "
        f"model mixes slowly, proceeding")

# === R3. Posterior check ===
# Bayesian posterior-predictive p-values on up to 6 summary stats + an OPTIONAL
# PSIS-LOO diagnostic. See validator.md Sec.R3 and codegen_r_runner.md Sec.R3.
#
# R3.a Bayesian p-values on (up to) 6 summary statistics. y_rep is the
# posterior-predictive draw matrix (n_keep x N) from predict_at(). The p-value
# for statistic T is P(T(y_rep) >= T(y_obs)) over posterior-predictive draws.
# This IS a gate (unlike R3.b PSIS-LOO below), but only on the CENTRAL
# statistics: a posterior-predictive p-value is ~Uniform(0, 1) even for a
# perfectly-sampled, CORRECTLY-specified model, and the order statistics
# (min / max) are legitimately extreme, so gating on "any statistic outside"
# would fail correct samplers. Gate rule: min / max are printed only; among
# the CENTRAL statistics (mean, sd, q25, q75) one outside is a WARN and TWO OR
# MORE simultaneously FAIL R3 (validator.md R3.a).
# The statistic SET depends on the support of y (validator.md R3.a). For a
# BINARY y every replicate's q25 and q75 are 0 or 1, so both p-values pin at
# 0 or 1 and two central statistics land outside -- failing a demonstrably
# correct sampler. Pick the set from the data, not a fixed list.
_y = np.asarray(y_obs, dtype=float)
_uniq = np.unique(_y)
if np.all(np.isin(_uniq, (0.0, 1.0))):                        # binary
    bp_stat = {"mean": np.mean}
elif np.all(_uniq >= 0) and np.all(_uniq == np.floor(_uniq)):  # count
    bp_stat = {"mean": np.mean, "sd": lambda x: np.std(x, ddof=1),
               "max": np.max}
else:                                                          # continuous
    bp_stat = {
        "mean": np.mean, "sd": lambda x: np.std(x, ddof=1),
        "min": np.min,   "max": np.max,
        "q25": lambda x: np.quantile(x, 0.25),
        "q75": lambda x: np.quantile(x, 0.75),
    }
y_rep = np.asarray(c1["pp"]["y_rep"])      # (n_keep x N)
pv = {nm: float(np.mean(np.array([f(row) for row in y_rep]) >= f(y_obs)))
      for nm, f in bp_stat.items()}
print("\n  Bayesian p-values: " +
      "  ".join(f"{nm}={p:.2f}" for nm, p in pv.items()))
# Gate on the CENTRAL statistics only; min / max stay diagnostic.
pv_lo = 0.10 if USES_JOINT_NUTS else 0.05
pv_hi = 1.0 - pv_lo
_central = {nm: p for nm, p in pv.items() if nm in ("mean", "sd", "q25", "q75")}
_n_out = sum(1 for p in _central.values() if p <= pv_lo or p >= pv_hi)
# Verdict scales with HOW MANY central statistics this support gives us.
# With several, one outside is expected under a correct model, so one WARNs
# and two or more FAIL. With exactly ONE (binary y -> "mean" only) there is
# no such slack: the ">= 2 to fail" rule would make R3.a unfailable.
_fail_at = 2 if len(_central) >= 2 else 1
if 0 < _n_out < _fail_at:
    import warnings
    warnings.warn(f"[R3.a WARN] one central Bayesian p-value outside "
                  f"({pv_lo}, {pv_hi}): {_central}")
assert _n_out < _fail_at, f"[R3.a FAIL] {_n_out} central Bayesian p-values outside ({pv_lo}, {pv_hi}): {_central}"

# R3.b PSIS-LOO (DIAGNOSTIC ONLY -- NEVER gates). Pareto-k_hat measures LOO
# importance-weight reliability, NOT sampler correctness; GP latent-variable
# and hierarchical-latent models routinely fail this diagnostic even with a
# correctly sampled posterior (Vehtari, Simpson, Gelman, Yao, Gabry, JMLR
# 2024, arXiv:1507.02646). Sampler correctness is gated by R-hat (R2) AND the
# Bayesian p-values (R3.a, the assert above). This one is diagnostic only --
# recorded and warned, never a gate.
#
# Emit the per-observation log-likelihood that matches the model's
# observation family (Gaussian example below; replace the body -- see
# codegen_cpp.md Sec.6a per-family templates). Build an InferenceData with a
# log_likelihood group of shape (chain, draw, obs) and call az.loo.
try:
    import arviz as az                          # lazy: PSIS-LOO is the ONLY arviz user
    #                                             and is diagnostic-only, so a missing /
    #                                             broken arviz never gates validation.
    def pointwise_loglik(hist, y):
        # ...replace with the model's per-observation log density,
        #    shape (n_keep, N); see codegen_cpp.md Sec.6a templates...
        raise NotImplementedError
    ll1 = pointwise_loglik(c1["hist"], y_obs)   # (n_keep, N)
    ll2 = pointwise_loglik(c2["hist"], y_obs)   # (n_keep, N)
    ll = np.stack([ll1, ll2], axis=0)           # (chain=2, draw, obs)
    idata_loo = az.from_dict(
        posterior={"_": np.zeros(ll.shape[:2])},
        log_likelihood={"y": ll})
    loo_res = az.loo(idata_loo, pointwise=True)
    khat = np.asarray(loo_res.pareto_k)
    pct_k_lo = float(np.mean(khat < 0.5) * 100)
    pct_k_hi = float(np.mean(khat >= 0.7) * 100)
    print(f"  LOO elpd={float(loo_res.elpd_loo):.1f} "
          f"(se={float(loo_res.se):.1f})  "
          f"pct_k<0.5={pct_k_lo:.1f}%  pct_k>=0.7={pct_k_hi:.1f}%")
    if pct_k_lo < 50 or pct_k_hi >= 10:
        import warnings
        warnings.warn(
            f"[R3.b] PSIS-LOO Pareto-k DIAGNOSTIC ONLY (NOT a failure): "
            f"{pct_k_lo:.1f}% k<0.5, {pct_k_hi:.1f}% k>=0.7. Pareto-k indicates "
            f"LOO importance-weight reliability, NOT sampler correctness; GP and "
            f"hierarchical-latent models routinely fail this diagnostic with "
            f"correctly sampled posteriors. See Vehtari et al. (JMLR 2024).")
except Exception as _loo_err:               # no loglik, no arviz, arviz API change, ...
    print(f"  [R3.b] PSIS-LOO skipped ({type(_loo_err).__name__}) -- "
          "diagnostic only, does not gate.")

# === Performance hint ===
# total_wall_sec is the true elapsed wall time from _run_two_chains; the two
# chains run SEQUENTIALLY, so it already reflects total work -- use it directly
# rather than re-deriving from c1["wall_sec"] + c2["wall_sec"].
AI4BayesCode.perf_hint(
    wall_sec=total_wall_sec,
    n_sweeps_total=2 * (n_burn + n_keep),
    uses_joint_nuts=<True if composite has joint_nuts_block else False>)

# === Final validation verdict (MUST be the VERY LAST line printed) ===
# The generator greps stdout for this exact sentinel. worst_rhat is from R2 above.
# The PASS/FAIL token uses the DEFAULT HARD gate = 1.05 (matches the assert above
# and validator.md Sec.R2); it MUST match the gate, else it prints FAIL while the run
# continues at 1.05 (the confusing "FAIL-but-SUCCESS"). Keep the token text EXACT.
if worst_rhat < 1.05:
    print("AI4BAYES_VALIDATE: PASS")
else:
    print(f"AI4BAYES_VALIDATE: FAIL maxRhat={worst_rhat:.4f}")
# R-hat GRAY ZONE (1.01, 1.05]: PASSES the default gate but not the strict
# Vehtari-2021 1.01 bar. Do NOT auto-escalate -- follow start.md's "R-hat gray
# zone -- opt-in stricter convergence" protocol: surface the structured question
# (a: ship as-is at 1.05 [default] / b: extend budget 20k->40k->80k toward <1.01 /
# c: AI-proposed structural fix / d: other). A heavy-tailed hierarchical variance
# (InvGamma) can legitimately sit at 1.01-1.02 even when the posterior is CORRECT
# (predictive matches a reference); there, option (b) or a higher joint_nuts
# max_tree_depth (10-12) is the remedy, NOT a different model. This line SIGNALS
# the gray zone to the agent.
if worst_rhat >= 1.01:
    print(f"  [convergence] gray zone: PASS at the default 1.05 gate, strict 1.01 "
          f"NOT met (maxRhat={worst_rhat:.4f}) -> apply start.md's opt-in stricter-convergence question.")
```

### Special case: per-step outputs NOT in `get_history()`

Same rule as the R version. When a per-step posterior summary (e.g. an
intermediate quantity computed inside `step()` but not registered with
shared_data) needs to flow into R3, generate a Python helper that
recomputes from `get_history()` outputs and the cached `predict_at`
result. Do NOT add `.method()` entries to `PYBIND11_MODULE` to expose
intermediate scratch -- that breaks the Sec.1 invariant.

**Codegen LLMs MUST NOT hallucinate history keys.** Before emitting
`hist['<key>']` in a generated runner, verify `<key>` is in the block's
documented `get_history()` output. If a field is only under
`get_current()`, route it through `get_current()` per step -- do NOT
pretend it lives in `hist`. (Motivating bug: a runner referenced
`hist['order_sampled_DAG']`, which does not exist -- it compiled but
crashed at first use.)

**Known case:** `order_mcmc_block` exposes `sampled_DAG` (a pxp adjacency
matrix) ONLY via `get_current()`; its `get_history()` returns only
`order`, `order_log_score`, and (when a Tier-A wrapper adds it) `y_rep`.
The DAG is history-omitted because it is heavy: p=20, T=40000 ~= 130 MB;
p=64 (block ceiling) ~= 1.3 GB.

**Collection-loop pattern** -- step through burn-in, then collect the
per-step `get_current()` field draw-by-draw, storing the result at the
TOP LEVEL of `out` (parallel to `"hist"`, NOT inside it):

```python
model.step(int(n_burn))
p_var = data_obs.shape[1]
dags = np.zeros((int(n_keep), p_var, p_var), dtype=int)
for s in range(int(n_keep)):
    model.step(1)
    # sampled_DAG[i, j] = 1 iff j is a parent of i. get_current() returns a FLAT
# length-n^2 vector filled column-major, so reshape with order="F".
    dags[s] = np.asarray(model.get_current()["sampled_DAG"])
out = {"hist": _slice(model.get_history()),
       "dags": dags,                                    # top-level, NOT inside "hist"
       "wall_sec": time.time() - t0}
```

## 9a. Model-specific Python-side preprocessing

The R-skill Sec.9a documents the SoftBart `sigma_hat` recipe (R-side
cross-validated lasso + variance scaling). See `codegen_r_runner.md`
Sec.9a for the full recipe + hard rules (the math is language-agnostic).
The Python translation uses `sklearn.linear_model.LassoCV`, but three
specifics must be reproduced by hand -- sklearn does NOT match `glmnet`
out of the box:

1. **`LassoCV` has NO native 1-SE selection.** R's SoftBart uses
   `lambda.1se` (more regularized than `lambda.min`). Implement 1-SE
   manually from the per-alpha cross-validation MSE path: for each
   alpha compute the mean and standard error of the CV MSE across
   folds (`LassoCV` exposes `mse_path_`), find the minimum-mean alpha,
   then pick the LARGEST alpha (strongest regularization) whose mean
   MSE is within one SE of that minimum.
2. **`sigma_hat` is RMSE, not `sd`.** Compute
   `sqrt(mean(resid**2))` (denominator N), NOT `sd(resid)`
   (denominator N-1), from the in-sample residuals at the 1-SE alpha.
3. **Min-max normalize Y to [-0.5, 0.5] BEFORE the lasso**, and
   un-normalize predictions on the way out. Save
   `(a, b) = (min(y), max(y))`, fit the lasso on
   `(y - a) / (b - a) - 0.5`, and inverse-transform any posterior
   summary via `(z + 0.5) * (b - a) + a`. The C++ kernel side does not
   change.

## 10. Reference templates

The shipped examples are TRI-MODULE: one `.cpp` carries a standalone
`int main`, an `RCPP_MODULE` block, AND a `PYBIND11_MODULE` block, so there
is no separate Python port to look up. All 43 of them expose the same
core-6 API from Python as from R, and each carries a tested
`@example:python` block in its header comment.

When unsure about a Python runner pattern (constructor calling conventions,
predict_at usage, DAG inspection), read the `@example:python` block of the
nearest example. In an installed wheel they live under
`AI4BayesCode/_examples/`; reach them with
`AI4BayesCode.examples_path("<Name>")` rather than a hard-coded path.

For backend selection rules (R-only / Python-only / dual R+Python),
load `codegen.md` Sec.1 and `codegen_cpp.md` Sec.9.

For the Layer-3 validator details (budget escalation, ESS gate, BPV
gate, PSIS-LOO gate, Dirac spike-and-slab Sec.R2.s exclusion rule), load
`validator.md`.

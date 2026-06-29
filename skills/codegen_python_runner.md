---
name: AI4BayesCode-codegen-python-runner
description: |
  Python runner template for AI4BayesCode samplers — AI4BayesCode.sourceCpp
  setup, constructor-argument reference block, run_chain_<ClassName>()
  helper with keep_history=True, Layer 3 validator wiring (R1 smoke
  check, R2 rank-normalized R-hat + ESS via arviz, R3 Bayesian p-values
  + PSIS-LOO via arviz), AI4BayesCode.perf_hint call, joint-NUTS threshold
  tightening, and the reference-template catalogue (examples/*.cpp).
  Mirror of codegen_r_runner.md for the Python (pybind11) backend.
  The entry-point skill `codegen.md` and `start.md` Phase 5 point here
  for Python runner emission when the chosen runtime is Python (or
  Both R+Python dual-module).
---

# AI4BayesCode codegen — Python runner + reference templates

Companion skill to `codegen.md`. Load this when writing the generated
`.py` runner: `AI4BayesCode.sourceCpp` setup, constructor-argument
reference block, `run_chain_<ClassName>()` helper, Layer 3 validator
wiring (R-hat, ESS, Bayesian p-values, PSIS-LOO), `AI4BayesCode.perf_hint`,
and the reference-template catalogue.

For prior elicitation + block selection, see `codegen_priors.md`.
For the C++ file emission (PYBIND11_MODULE block, type casters), see
`codegen_cpp.md` §9 "Backend module declarations" (Python-only and
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
dual R+Python, load BOTH this file and `codegen_r_runner.md` — the
two runners share the same .cpp file (with `#ifdef`-guarded
RCPP_MODULE and PYBIND11_MODULE blocks per `codegen_cpp.md` §9), and
the runner deliverables are independent (`.R` + `.py`).

## 9. Output

1. Create output folder if missing (default `./generated/<ClassName>/`).
2. Write `<folder>/<ClassName>.cpp`. **It MUST carry the
   `@example:python` header block** (per `codegen_cpp.md` §5 "Header
   `@example` block"): the SAME toy DGP this runner uses, distilled to
   ≤ ~8 runnable lines using the packaged API
   (`Mod = AI4BayesCode.source_AI4BayesCode("<ClassName>.cpp")` →
   `Mod(...)`). This is what `AI4BayesCode.doc()` shows as the Example.
   If the backend is Both R+Python, also emit `@example:R`. Write the
   DGP once and mirror it into the header so runner and card cannot
   drift.
3. Write the Python runner `<folder>/run_<ClassName>.py` following the
   template below and **use it to gate generation** (it carries the
   Layer-3 R1/R2/R3 harness — same CORRECTNESS INVARIANT as the R
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

Same cut-point rule as the R version: everything up to (but
EXCLUDING) the first chain of the two-chain diagnostic
(`chain1 = run_chain_<ClassName>(<data_args>, seed=101)`) is the
delivered example; everything from there onward (R2 R-hat/ESS, R3
BPV, PSIS-LOO, R1 smoke wiring) is the throwaway Layer-3 harness and
is NOT shipped in the default case.

The delivered `example_<ClassName>.py` MUST contain, in order:

1. **Header comment** — what the model is + a note that generation
   was validated by the Layer-3 harness (not shipped here). For everyday
   use, point at the built-in `diagnosis=True` option on `run_chain_<ClassName>`
   (model-independent R-hat / ESS / MCSE / summaries + trace+ACF+density via
   arviz); regenerate with the harness only if you specifically need PSIS-LOO
   (model-specific).
2. **AI4BayesCode.sourceCpp call** — exact same form as in the runner
   template below.
3. **Constructor reference block** — identical to the runner.
4. **Synthetic data block** — produces the same fixtures the harness
   used so the example runs as-is.
5. **One call to the runner helper** — typically
   `chain = run_chain_<ClassName>(<data_args>, seed=42, n_burn=200,
   n_keep=200)` (short budget so the example runs in seconds). Then show the
   built-in model-independent diagnostics:
   `chain = run_chain_<ClassName>(<data_args>, seed=42, n_burn=200, n_keep=200, diagnosis=True)`,
   then `chain["diagnosis"]` (per-parameter R-hat / ESS / MCSE / summaries) and
   `chain["diagnosis_plot"]()` (trace + autocorrelation + density; needs `arviz`).
   No LOO (that is model-specific).
6. **A `model.predict_at(...)` call** demonstrating posterior
   predictive use.

```python
# Initialize with full history (keep_history=False keeps only the last draw)
model = mod.<ClassName>(..., rng_seed=42, keep_history=True)
model.step(200)                              # burn-in
model.step(200)                              # keep
hist = model.get_history()                   # dict of (n_keep × dim) numpy arrays
y_rep = model.predict_at({})["y_rep"]        # posterior predictive at training X

# set_current() updates the sampler statefully (e.g. an outcome
# refreshed by other blocks) WITHOUT reinitialization. Shown
# COMMENTED OUT — uncomment in a real stateful-composition context:
# model.set_current({"<a data input>": <updated value>})
# model.step(1)                              # one iteration after set_current
# model.get_history()

# readapt_NUTS() — CONDITIONAL — emit ONLY if the wrapper's composite
# contains any NUTS-family child (nuts_block / joint_nuts_block).
# Skip the whole block if pure BART /
# pure Gibbs / pure VI / pure HMM / pure SBP / pure RJMCMC /
# pure slice.
#
# readapt_NUTS re-tunes the NUTS metric (mass matrix + step size +
# dual averaging) WITHOUT advancing chain state.
#
# WORKFLOW RULE (system_design.md §13 hybrid-composite caveat):
#   - Pure NUTS-family composite (no BART / no specialised Gibbs):
#       set_current(...) -> readapt_NUTS(N) immediately is fine
#   - Hybrid composite with BART / SoftBart / specialised Gibbs whose
#     outputs (f_bart, working residuals, etc.) refresh inside step():
#       set_current(...) -> step(1) -> readapt_NUTS(N)
#     The intervening step(1) refreshes shared_data so the NUTS
#     sibling's adapter sees fresh derived state, not stale values.
#
# Example (uncomment in real sequential-update use):
# model.set_current({"<data input>": <updated value>})
# model.step(1)             # refresh derived state — hybrid composites only
# model.readapt_NUTS(500)   # re-tune metric (default: reset=False)
# model.readapt_NUTS(500, True)   # use reset=True if data change is dramatic
```

### Modular NUTS in composite — periodic readapt schedule (CONDITIONAL)

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
`codegen_r_runner.md` "Modular NUTS in composite — periodic readapt
schedule".

**Defaults.** `readapt_every = 500` outer iters, `readapt_n = 50`
re-adapt iters per call (~10% overhead). Tighten to
`readapt_every = 100` if R-hat fails at 4k+4k; loosen to
`readapt_every = 1000` if the chain mixes well and wall is critical.
There is no published canonical value for modular NUTS-in-composite
re-adaptation frequency — standard frameworks (Stan, PyMC, NumPyro)
use single-warmup + frozen-metric and do not address this regime; the
defaults here are empirically calibrated.

Python emit pattern (note: `readapt_every`, `readapt_n` are
**function arguments**, not script-scope globals, so the helper stays
self-contained — sim1 / cross-impl harnesses can vary them per
replicate without touching the function body):

```python
def run_chain_<ClassName>(<data_args>, *, seed, n_burn, n_keep,
                          readapt_every=500, readapt_n=50,
                          newdata=None, diagnosis=False):
    if newdata is None: newdata = {}
    model = mod.<ClassName>(<data_args>, rng_seed=int(seed),
                             keep_history=True)
    t0 = time.time()
    # Periodic readapt schedule — covers BOTH burn-in and keep, since the
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
    out = {"hist": model.get_history(),
           "pp":   model.predict_at(newdata),
           "wall_sec": time.time() - t0}
    # diagnosis=True attaches model-INDEPENDENT posterior diagnostics via the
    # SHIPPED library function AI4BayesCode.ai4b_diagnose() -- do NOT reimplement:
    #   out["diagnosis"]      -> per-parameter table (R-hat / ESS / mean / sd / 90% CI)
    #   out["diagnosis_plot"] -> a callable drawing trace + autocorrelation + density
    # get_history() returns burn-in + keep, so pass n_burn to strip it.
    if diagnosis:
        out["diagnosis"], out["diagnosis_plot"] = \
            AI4BayesCode.ai4b_diagnose(out["hist"], n_burn=int(n_burn))
    return out

# NOTE: ai4b_diagnose (the diagnostics table + the trace/ACF/density plot) is a
# SHIPPED function in the AI4BayesCode package -- the runner CALLS it (above);
# it does NOT define its own copy. The summary uses split-R-hat / ESS (numpy),
# correct for the single chain a runner produces, and the plot uses matplotlib.
```

**HARD RULE — the `diagnosis=True` path is non-negotiable.** The diagnostics
AND the plot are a SHIPPED library function, `AI4BayesCode.ai4b_diagnose(hist,
n_burn=...)`, which returns `(summary_table, plot_fn)` where `plot_fn()` draws
trace + autocorrelation + density with matplotlib. Every generated runner MUST
(1) take a `diagnosis=False` argument; (2) when `diagnosis=True`, CALL
`AI4BayesCode.ai4b_diagnose()` and attach BOTH `out["diagnosis"]` (table) AND
`out["diagnosis_plot"]` (callable). Do NOT reimplement it inline. This is
independent of how the runner collects draws: pass whatever named dict of kept
draws you built as `AI4BayesCode.ai4b_diagnose(draws, n_burn=...)` — use
`n_burn=0` when the draws are already burn-in-stripped (`get_history()` returns
burn-in + keep, so pass the burn-in length there). FORBIDDEN — an inline
summary-only diagnosis that drops the plot (e.g. `out["summary"] =
az.summary(...)` with no `diagnosis_plot`): that returns a table with NO
trace/ACF/density plot and renames the field. ALWAYS route through
`AI4BayesCode.ai4b_diagnose()` and expose `out["diagnosis"]` +
`out["diagnosis_plot"]`.

Skip the periodic schedule (use plain `model.step(n_burn);
model.step(n_keep)`) ONLY when the composite is pure-NUTS-on-a-fixed-
conditional (e.g. a single `joint_nuts_block` sampling everything
jointly with no Gibbs siblings) — in that case the conditional does
not shift and the initial warmup metric stays correct.

### Python runner template

The generated `run_<ClassName>.py` must include:
1. A **constructor reference block** documenting every argument to
   `mod.<ClassName>(...)` with its type, description, default/valid
   range.
2. A burnin phase and a draw-collection loop (the periodic readapt
   pattern above when applicable).
3. The `diagnosis=False` parameter on `run_chain_<ClassName>`, and when
   `diagnosis=True` a CALL to the shipped `AI4BayesCode.ai4b_diagnose()`
   giving `chain["diagnosis"]` (table) and `chain["diagnosis_plot"]`
   (trace + ACF + density). Do NOT write your own helper and never inline
   a summary-only substitute (see the HARD RULE below).

Follow this structure:

**Path-resolution rule (mirror R skill — no runtime detection).** The
generated runner uses HARDCODED RELATIVE PATHS. No `inspect.stack`,
no `__file__`-walking, no `os.getcwd` checks. The contract: the user
runs Python from **the project root** — the directory containing both
`AI4BayesCode/` AND the `<folder>/` where the generator wrote the
`.cpp`. If invoked elsewhere, the user gets a clear filesystem error
and the fix is documented.

```python
# === run_<ClassName>.py — Layer 3 harness (R1 + R2 + R3) ===
import os, sys, time
import numpy as np

# Packaged API — install once (`pip install ai4bayescode`); no checkout,
# no sys.path hack, no ai4bayescode_path. Headers travel in the package.
import AI4BayesCode

# Compile + load the generated .cpp; returns the module (the class is an
# attribute, e.g. mod.<ClassName>).
mod = AI4BayesCode.source_AI4BayesCode("<folder>/<ClassName>.cpp")

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

# (Synthetic data block — exactly matches the codegen_priors.md "Validation
# fixture" guidance; see §6 of that skill for the shape conventions.)
# ... <data setup> ...

# Per-chain runner helper (with periodic readapt — see § above).
# Signature MUST include diagnosis=False; when diagnosis=True, CALL the
# shipped AI4BayesCode.ai4b_diagnose() — do NOT write your own helper.
def run_chain_<ClassName>(<data_args>, *, seed, n_burn, n_keep,
                          readapt_every=500, readapt_n=50, newdata=None,
                          diagnosis=False):
    ...  # verbatim from the template above, INCLUDING the attach:
    #   if diagnosis:
    #       out["diagnosis"], out["diagnosis_plot"] = \
    #           AI4BayesCode.ai4b_diagnose(out["hist"], n_burn=int(n_burn))
# NOTE: ai4b_diagnose (table + trace/ACF/density plot) is SHIPPED in the
# AI4BayesCode package — there is NO helper to define here.

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

# === R2. 2-chain MCMC convergence (parallel via multiprocessing) ===
# Defer to validator.md §R2 for the budget/escalation policy + the
# Dirac spike-and-slab §R2.s exclusion rule for per-coordinate slab
# DISTRIBUTION parameters (e.g. per-j slab variance tau2_beta /
# tau2_theta) — NOT the slab-modelled beta_j / theta_jk themselves.
import multiprocessing as mp
import arviz as az

def _worker(args):
    seed, n_burn, n_keep = args
    return run_chain_<ClassName>(<data_args>, seed=seed,
                                  n_burn=n_burn, n_keep=n_keep)

n_burn = 4000; n_keep = 4000
t0 = time.time()
ctx = mp.get_context("spawn")
with ctx.Pool(2) as pool:
    chains = pool.map(_worker, [(101, n_burn, n_keep), (202, n_burn, n_keep)])
total_wall_sec = time.time() - t0
c1, c2 = chains

# (R-hat / ESS aggregation, including the §R2.s conditional-relevance
# exclusion rule for Dirac spike-and-slab: filter SLAB DISTRIBUTION
# PARAMETER draws (per-j slab variance tau2_beta / tau2_theta and
# similar; NOT the slab-modelled beta_j / theta_jk themselves) by
# inclusion indicator == 1, truncate to common min count, then
# arviz.rhat. See validator.md §R2.s for the precise rule, the
# I_j ∈ {0, 1} convention, and the canonical numpy-mask code to emit.)
# ... <Stage 1 diag> ...
# ... <Stage 2 budget bump to 20000 + 20000 if R-hat >= 1.05 or
#      ess_ratio < 0.005 — see validator.md §R2> ...

# === R3. Posterior check ===
# Bayesian p-values on summary statistics + PSIS-LOO. See validator.md §R3
# for the gate (BPV in (0.05, 0.95); k̂ < 0.7 acceptable for hierarchical /
# latent models per Vehtari et al. 2024).
# Use arviz.loo for PSIS-LOO; numpy for BPV.

# === Performance hint ===
AI4BayesCode.perf_hint(
    wall_sec=total_wall_sec,
    n_sweeps_total=2 * (n_burn + n_keep),
    uses_joint_nuts=<True if composite has joint_nuts_block else False>)
```

### Special case: per-step outputs NOT in `get_history()`

Same rule as R version: when a per-step posterior summary (e.g. an
intermediate quantity computed inside `step()` but not registered with
shared_data) needs to flow into R3, generate a Python helper that
recomputes from `get_history()` outputs and the cached `predict_at`
result. Do NOT add `.method()` entries to `PYBIND11_MODULE` to expose
intermediate scratch — that breaks the §1 invariant.

## 9a. Model-specific Python-side preprocessing

The R-skill §9a documents the SoftBart `sigma_hat` recipe (R-side
cross-validated lasso + variance scaling). The Python equivalent uses
`sklearn.linear_model.LassoCV` (cross-validated lasso) followed by the
same scale-by-sd-of-residuals step. See `codegen_r_runner.md` §9a for
the exact recipe + hard rules (the math is language-agnostic); the
Python translation is mechanical (`numpy` + `sklearn` for what R
does with `glmnet` + base R).

## 10. Reference templates

Python ports of selected `examples/*.cpp` live in
`AI4BayesCode/examples_py/` when shipped. When unsure about a Python
runner pattern (constructor calling conventions, predict_at usage,
DAG inspection, etc.), grep that folder first. The 6 examples with
`PYBIND11_MODULE` blocks (see `README.md` "Examples with Python
support" for the current list) all expose the same 6-method API as R
and serve as reference templates.

For backend selection rules (R-only / Python-only / dual R+Python),
load `codegen.md` §1 and `codegen_cpp.md` §9.

For the Layer-3 validator details (budget escalation, ESS gate, BPV
gate, PSIS-LOO gate, Dirac spike-and-slab §R2.s exclusion rule), load
`validator.md`.

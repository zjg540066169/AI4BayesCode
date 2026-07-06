---
name: AI4BayesCode-codegen-python-runner
description: |
  Python runner template for AI4BayesCode samplers — AI4BayesCode.sourceCpp
  setup, constructor-argument reference block, run_chain_<ClassName>()
  helper with keep_history=True, Layer 3 validator wiring (R1 smoke
  check, R2 rank-normalized R-hat + ESS (AI4BayesCode numpy helpers), R3 Bayesian p-values
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
2. Write `<folder>/<ClassName>.cpp`. **It MUST carry BOTH the
   `@example:python` AND the `@example:R` header blocks** (per
   `codegen_cpp.md` §5 "Header `@example` block") — the `.cpp` is ALWAYS
   dual-module, so an R user who takes the same file must still see the R
   example via `ai4bayescode_doc()`, and a Python user the Python example via
   `AI4BayesCode.doc()`. Both blocks are the SAME toy DGP this runner uses,
   distilled to ≤ ~8 runnable lines using each language's packaged API
   (`AI4BayesCode.source(...)` → `Mod(...)` for Python; `new(<ClassName>,
   ...)` for R). Write the DGP once and mirror it into BOTH header blocks so
   the runner and both doc cards cannot drift.
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
4. **The FULL `run_chain_<ClassName>()` body VERBATIM** (not a stub) —
   including the `diagnosis` parameter and the `AI4BayesCode.diagnose()`
   call. This is the SAME definition as in the runner template above;
   ship the whole function body so the example runs self-contained.
5. **Synthetic data block** — produces the same fixtures the harness
   used so the example runs as-is.
6. **Monolithic (non-stateful) single call** —
   `mono = run_chain_<ClassName>(..., seed=1, n_burn=4000, n_keep=4000)`
   with a `# Monolithic chain (non-stateful use)` comment, plus the
   `diagnosis=True` variant showing `mono["diagnosis"]` (per-parameter
   R-hat / ESS / MCSE / summaries) and `mono["diagnosis_plot"]()`
   (trace + autocorrelation + density; needs `arviz`). No LOO (that is
   model-specific).
7. **Stateful-API usage**, in order: `model.step()`,
   `model.get_history()`, `model.get_current()`,
   `model.predict_at({"<X>": X_test})`, then
   `model.set_current({...})` LAST, AFTER `predict_at`, and COMMENTED
   OUT (its updated value comes from an outer Gibbs composition, so as
   live code it would error in a standalone run).

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

### Python runner template — standard body (DEFAULT)

This is the DEFAULT `run_chain_<ClassName>` body. Use it unless the
composite has a NUTS-family child whose conditional posterior shifts
across outer Gibbs sweeps — in that case use the periodic-readapt body
in the next subsection instead.

```python
def run_chain_<ClassName>(<data_args>, *, seed, n_burn, n_keep,
                          newdata=None, diagnosis=False):
    if newdata is None:
        newdata = {}
    model = mod.<ClassName>(<data_args>, rng_seed=int(seed), keep_history=True)
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

**Note — pre-slice vs. n_burn.** The standard body above pre-slices the
history inside the runner, so it calls `AI4BayesCode.diagnose(..., n_burn=0)`
(the draws are already burn-in-stripped). The periodic-readapt body below
does NOT pre-slice and passes `n_burn=int(n_burn)` — mixing these up (calling
`diagnose(..., n_burn=int(n_burn))` on already-sliced draws, or vice versa) is
an easy mistake that silently drops or double-strips warmup.

**Return-shape contract.** `run_chain_<ClassName>` MUST return a dict
whose `"hist"` is a dict of numpy arrays keyed by parameter name —
scalars as `(n_keep,)` 1-D arrays, vectors as `(n_keep, dim)` 2-D
arrays. NEVER a list-of-per-step-dicts (the R2 `_stack_param` helper
branches on `arr.ndim == 1`, so a list breaks it). This is exactly what
`model.get_history()` returns. For multi-chain R-hat, run 2+ chains with
overdispersed init via `model.set_current(...)` after construction, then
use `arviz` OR the shipped `AI4BayesCode.rhat_summary` (which exists but
the rest of this skill never mentions).

### Constructor reference block

The constructor block must list ALL arguments the user can pass, their
types, and brief descriptions. If hyperparameters are exposed as
constructor arguments (see `codegen.md` §2 and `codegen_priors.md`),
document those too with their defaults. Also document what
`run_chain_<ClassName>()` returns (which keys, each array's shape).

Concrete example — the constructor block for a BART model:

```python
# -------------------------------------------------------------------------
#   BartNoise(
#       X:            (N, p) float array   — predictor matrix
#       y:            (N,)   float array   — response vector
#       ntrees:       int    = 200         — number of trees
#       rng_seed:     int                  — RNG seed (0 = random_device)
#       keep_history: bool   = False       — record per-iter draws
#   )
#   NOTE: sigma is initialized internally to bart_model's OLS-based sigest.
#         For an overdispersed start (R-hat diagnostics), call
#         model.set_current({"sigma": ...}) AFTER construction.
#
#   Methods:
#     .step(n)         — run n Gibbs sweeps (one BART + one NUTS-sigma)
#     .get_current()   -> {"f_bart": (N,), "sigma": float}
#     .set_current(d)  — overwrite sigma; f_bart is read-only
#
#   run_chain_<ClassName>() returns:
#     out["hist"]["f_bart"] — (n_keep, N) posterior draws of f
#     out["hist"]["sigma"]  — (n_keep,)   posterior draws of sigma
# -------------------------------------------------------------------------
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
    # SHIPPED library function AI4BayesCode.diagnose() -- do NOT reimplement:
    #   out["diagnosis"]      -> per-parameter table (R-hat / ESS / mean / sd / 90% CI)
    #   out["diagnosis_plot"] -> a callable drawing trace + autocorrelation + density
    # get_history() returns burn-in + keep, so pass n_burn to strip it.
    if diagnosis:
        out["diagnosis"], out["diagnosis_plot"] = \
            AI4BayesCode.diagnose(out["hist"], n_burn=int(n_burn))
    return out

# NOTE: ai4bayescode_diagnose (the diagnostics table + the trace/ACF/density plot) is a
# SHIPPED function in the AI4BayesCode package -- the runner CALLS it (above);
# it does NOT define its own copy. The summary uses split-R-hat / ESS (numpy),
# correct for the single chain a runner produces, and the plot uses matplotlib.
```

**HARD RULE — the `diagnosis=True` path is non-negotiable.** The diagnostics
AND the plot are a SHIPPED library function, `AI4BayesCode.diagnose(hist,
n_burn=...)`, which returns `(summary_table, plot_fn)` where `plot_fn()` draws
trace + autocorrelation + density with matplotlib. Every generated runner MUST
(1) take a `diagnosis=False` argument; (2) when `diagnosis=True`, CALL
`AI4BayesCode.diagnose()` and attach BOTH `out["diagnosis"]` (table) AND
`out["diagnosis_plot"]` (callable). Do NOT reimplement it inline. This is
independent of how the runner collects draws: pass whatever named dict of kept
draws you built as `AI4BayesCode.diagnose(draws, n_burn=...)` — use
`n_burn=0` when the draws are already burn-in-stripped (`get_history()` returns
burn-in + keep, so pass the burn-in length there). FORBIDDEN — an inline
summary-only diagnosis that drops the plot (e.g. `out["summary"] =
az.summary(...)` with no `diagnosis_plot`): that returns a table with NO
trace/ACF/density plot and renames the field. ALWAYS route through
`AI4BayesCode.diagnose()` and expose `out["diagnosis"]` +
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
   `diagnosis=True` a CALL to the shipped `AI4BayesCode.diagnose()`
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

# (Synthetic data block — exactly matches the codegen_priors.md "Validation
# fixture" guidance; see §6 of that skill for the shape conventions.)
# ... <data setup> ...

# Per-chain runner helper (with periodic readapt — see § above).
# Signature MUST include diagnosis=False; when diagnosis=True, CALL the
# shipped AI4BayesCode.diagnose() — do NOT write your own helper.
def run_chain_<ClassName>(<data_args>, *, seed, n_burn, n_keep,
                          readapt_every=500, readapt_n=50, newdata=None,
                          diagnosis=False):
    ...  # verbatim from the template above, INCLUDING the attach:
    #   if diagnosis:
    #       out["diagnosis"], out["diagnosis_plot"] = \
    #           AI4BayesCode.diagnose(out["hist"], n_burn=int(n_burn))
# NOTE: ai4bayescode_diagnose (table + trace/ACF/density plot) is SHIPPED in the
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

# === R2. 2-chain MCMC convergence (sequential; see _run_two_chains) ===
# Defer to validator.md §R2 for the budget/escalation policy + the
# Dirac spike-and-slab §R2.s exclusion rule for per-coordinate slab
# DISTRIBUTION parameters (e.g. per-j slab variance tau2_beta /
# tau2_theta) — NOT the slab-modelled beta_j / theta_jk themselves.

def _worker(args):
    seed, n_burn, n_keep = args
    return run_chain_<ClassName>(<data_args>, seed=seed,
                                  n_burn=n_burn, n_keep=n_keep)

n_burn = 4000; n_keep = 4000

# Flip to True if this runner's composite contains a joint_nuts_block
# (tightens the R3 BPV interval from (0.05, 0.95) to (0.02, 0.98) —
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
# Matches validator.md §R2 and codegen_r_runner.md r2_diag(): rank-normalized
# R-hat is a HARD gate (< 1.05); ESS is a SOFT criterion via
# ess_ratio = min(ESS_bulk, ESS_tail) / n_keep — >= 0.01 silent,
# [0.005, 0.01) WARN and proceed, < 0.005 escalate. AI4BayesCode.rhat / ess_* use the
# rank-normalized split-R-hat and bulk/tail ESS of Vehtari et al. (2021),
# the same estimators posterior::rhat / ess_bulk / ess_tail compute R-side.
#
# §R2.s conditional-relevance exclusion (Dirac spike-and-slab): for per-j
# slab DISTRIBUTION parameters (per-coordinate slab variance tau2_beta /
# tau2_theta and similar — NOT the slab-modelled beta_j / theta_jk
# themselves), mask draws by inclusion indicator I_j == 1, truncate the two
# chains to the common min retained count, then feed to AI4BayesCode.rhat. See
# validator.md §R2.s for the precise rule and the I_j ∈ {0, 1} convention.

def _stack_param(c1, c2, key):
    """Stack two chains' draws for one parameter into an arviz-shaped array.

    Returns a list of per-component (chain, draw) arrays so a vector
    parameter (n_keep × dim) is diagnosed component-by-component, exactly
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

# Stage 1 diagnostic at the template's existing budget (4k + 4k).
d = r2_diag(c1, c2, n_keep)

# Stage-2 escalation (within-attempt): re-run at 20000 + 20000 if Stage-1
# shows max R-hat >= 1.05 OR a severely low ess_ratio (< 0.005), then
# recompute. A slow-but-correct model gets the bigger budget BEFORE being
# declared a failure (validator.md §R2). Do NOT hard-fail ESS at Stage 1.
if d["rhat"] >= 1.05 or d["ess_ratio"] < 0.005:
    print("  [R2] Stage-1 inadequate -> escalating to 20000 + 20000 ...")
    n_burn = 20000; n_keep = 20000
    c1, c2, total_wall_sec = _run_two_chains(n_burn, n_keep)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in c1["hist"].values())
    assert all(np.all(np.isfinite(np.asarray(v))) for v in c2["hist"].values())
    d = r2_diag(c1, c2, n_keep)

# worst rank-normalized R-hat across all parameters — drives the final gate.
worst_rhat = d["rhat"]

# Final R2 gates: R-hat HARD (< 1.05); ESS SOFT (only ess_ratio < 0.005 at the
# escalated budget is a FAILURE; [0.005, 0.01) only warns — legitimate for slow
# GP / hierarchical models).
assert worst_rhat < 1.05, f"R2 FAIL: worst rank-Rhat {worst_rhat:.4f} >= 1.05"
if d["ess_ratio"] < 0.005:
    raise RuntimeError(
        f"R2 FAIL: worst ess_ratio {d['ess_ratio']:.4f} < 0.005 "
        f"even at the escalated budget")
elif d["ess_ratio"] < 0.01:
    import warnings
    warnings.warn(
        f"R2: worst ess_ratio {d['ess_ratio']:.4f} in [0.005, 0.01) — "
        f"model mixes slowly, proceeding")

# === R3. Posterior check ===
# Bayesian posterior-predictive p-values on up to 6 summary stats + an OPTIONAL
# PSIS-LOO diagnostic. See validator.md §R3 and codegen_r_runner.md §R3.
#
# R3.a Bayesian p-values on (up to) 6 summary statistics. y_rep is the
# posterior-predictive draw matrix (n_keep × N) from predict_at(). The p-value
# for statistic T is P(T(y_rep) >= T(y_obs)) over posterior-predictive draws.
# DIAGNOSTIC ONLY -- print, WARN on an EGREGIOUS excursion, but NEVER assert /
# gate on it. A posterior-predictive p-value is ~Uniform(0, 1) even for a
# perfectly-sampled, CORRECTLY-specified model, so across 6 statistics ~22% of
# CORRECT samplers would land at least one outside (0.02, 0.98) by chance, and
# order statistics (min / max) are legitimately extreme. Sampler correctness is
# gated by rank-R-hat (R2); a Bayesian p-value is a MODEL-FIT check the user
# owns, not a sampler gate. (Mirrors R3.b PSIS-LOO, also diagnostic-only.)
bp_stat = {
    "mean": np.mean, "sd": lambda x: np.std(x, ddof=1),
    "min": np.min,   "max": np.max,
    "q25": lambda x: np.quantile(x, 0.25),
    "q75": lambda x: np.quantile(x, 0.75),
}
y_rep = np.asarray(c1["pp"]["y_rep"])      # (n_keep × N)
pv = {nm: float(np.mean(np.array([f(row) for row in y_rep]) >= f(y_obs)))
      for nm, f in bp_stat.items()}
print("\n  Bayesian p-values: " +
      "  ".join(f"{nm}={p:.2f}" for nm, p in pv.items()))
_bpv_extreme = {nm: p for nm, p in pv.items() if p < 0.005 or p > 0.995}
if _bpv_extreme:
    import warnings
    warnings.warn(
        f"[R3.a] Bayesian p-value(s) near 0/1 (DIAGNOSTIC, NOT a failure): "
        f"{_bpv_extreme}. Expected for order statistics; investigate model fit "
        f"only if a CENTRAL statistic (mean / sd) is extreme AND R-hat flags.")

# R3.b PSIS-LOO (DIAGNOSTIC ONLY — NEVER gates). Pareto-k_hat measures LOO
# importance-weight reliability, NOT sampler correctness; GP latent-variable
# and hierarchical-latent models routinely fail this diagnostic even with a
# correctly sampled posterior (Vehtari, Simpson, Gelman, Yao, Gabry, JMLR
# 2024, arXiv:1507.02646). Sampler correctness is gated by R-hat (R2) ONLY;
# the Bayesian p-values (R3.a) and this are diagnostics -- recorded + warned,
# never asserted.
#
# Emit the per-observation log-likelihood that matches the model's
# observation family (Gaussian example below; replace the body — see
# codegen_cpp.md §6a per-family templates). Build an InferenceData with a
# log_likelihood group of shape (chain, draw, obs) and call az.loo.
try:
    import arviz as az                          # lazy: PSIS-LOO is the ONLY arviz user
    #                                             and is diagnostic-only, so a missing /
    #                                             broken arviz never gates validation.
    def pointwise_loglik(hist, y):
        # ...replace with the model's per-observation log density,
        #    shape (n_keep, N); see codegen_cpp.md §6a templates...
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
    print(f"  [R3.b] PSIS-LOO skipped ({type(_loo_err).__name__}) — "
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
if worst_rhat < 1.01:
    print("AI4BAYES_VALIDATE: PASS")
else:
    print(f"AI4BAYES_VALIDATE: FAIL maxRhat={worst_rhat:.4f}")
```

### Special case: per-step outputs NOT in `get_history()`

Same rule as the R version. When a per-step posterior summary (e.g. an
intermediate quantity computed inside `step()` but not registered with
shared_data) needs to flow into R3, generate a Python helper that
recomputes from `get_history()` outputs and the cached `predict_at`
result. Do NOT add `.method()` entries to `PYBIND11_MODULE` to expose
intermediate scratch — that breaks the §1 invariant.

**Codegen LLMs MUST NOT hallucinate history keys.** Before emitting
`hist['<key>']` in a generated runner, verify `<key>` is in the block's
documented `get_history()` output. If a field is only under
`get_current()`, route it through `get_current()` per step — do NOT
pretend it lives in `hist`. (Motivating bug: a runner referenced
`hist['order_sampled_DAG']`, which does not exist — it compiled but
crashed at first use.)

**Known case:** `order_mcmc_block` exposes `sampled_DAG` (a p×p adjacency
matrix) ONLY via `get_current()`; its `get_history()` returns only
`order`, `order_log_score`, and (when a Tier-A wrapper adds it) `y_rep`.
The DAG is history-omitted because it is heavy: p=20, T=40000 ≈ 130 MB;
p=64 (block ceiling) ≈ 1.3 GB.

**Collection-loop pattern** — step through burn-in, then collect the
per-step `get_current()` field draw-by-draw, storing the result at the
TOP LEVEL of `out` (parallel to `"hist"`, NOT inside it):

```python
model.step(int(n_burn))
p_var = data_obs.shape[1]
dags = np.zeros((int(n_keep), p_var, p_var), dtype=int)
for s in range(int(n_keep)):
    model.step(1)
    # sampled_DAG[i, j] = 1 iff j is a parent of i (row-major).
    dags[s] = np.asarray(model.get_current()["sampled_DAG"])
out = {"hist": _slice(model.get_history()),
       "dags": dags,                                    # top-level, NOT inside "hist"
       "wall_sec": time.time() - t0}
```

## 9a. Model-specific Python-side preprocessing

The R-skill §9a documents the SoftBart `sigma_hat` recipe (R-side
cross-validated lasso + variance scaling). See `codegen_r_runner.md`
§9a for the full recipe + hard rules (the math is language-agnostic).
The Python translation uses `sklearn.linear_model.LassoCV`, but three
specifics must be reproduced by hand — sklearn does NOT match `glmnet`
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

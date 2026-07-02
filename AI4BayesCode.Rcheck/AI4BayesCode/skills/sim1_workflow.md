---
name: AI4BayesCode-sim1-workflow
description: |
  Governs the lifecycle of model directories in new/: when to write sim1
  scripts, when to move a model out to small_example/, and when to route
  it to fail/. Models stay in new/ until their sim1 script is written and
  the smoke test passes. new/ is never deleted.
---

# Sim1 workflow — model lifecycle in new/

## Directory layout

```
small_example/
├── new/             ← staging area; models live here until sim1 is ready
├── fail/            ← models that cannot have a sim1 script written
├── AI4BayesCode/    ← shared library
└── <id>/            ← graduated models with working sim1 scripts
```

**`new/` is a permanent staging directory. It must never be deleted.**

---

## Step-by-step lifecycle for each model in new/

### 1. Triage

For every model directory `new/<id>/`, check for the three prerequisites:

| Prerequisite | Where to look |
|---|---|
| `new/<id>/<id>.cpp` exists and compiled | STATUS file says SUCCESS or check manually |
| `new/<id>/<id>.r` exists (DGP script + reference fitter call) | `find new/<id> -name "*.r"` |
| **Reference fitter** is identifiable. EITHER:<br>(a) `<id>.stan` exists → cmdstanr-based reference, OR<br>(b) `<id>.r` calls an R-package fitter (e.g., `BART::wbart`, `SoftBart::softbart`, `mixtools::normalmixEM`) → R-package reference | `find new/<id> -name "*.stan"` and inspect `<id>.r` for `library(.)` / `<pkg>::<fitter>(.)` calls |

**If any prerequisite is missing → route to `fail/` immediately (see §3).**

The reference type determines the cross-impl harness (§6 below):
- **Stan reference**: sim1 uses `cmdstanr` to compile `<id>.stan` and run a parallel chain.
- **R-package reference**: sim1 calls the R package directly inside `run_REF_chain` (no Stan compile, no `.stan` file required). The reference call lives in `<id>.r` and the sim1 mirrors its argument list.

This is determined by the codegen agent at model-creation time. Newer model
families (continuous-response BART, soft BART, etc.) have NO Stan
implementation and use R-package references exclusively.

The STATUS file gives a quick read:
- `SUCCESS` → proceed to step 2
- `FAILED_*` → route to `fail/`
- Missing STATUS → investigate before doing anything

### 2. Write and verify the sim1 script

Only after triage passes:

1. Read `<id>.stan`, `<id>.r`, `<id>_runner.R` to understand parameters and
   constructor signature.
2. Write `new/<id>/sim1_<id>.R` following the `AI4BayesCode-sim1` skill.
3. Run the **smoke test only** (worker(1), 1 replicate) to confirm the
   script compiles and aligns without errors:
   ```bash
   cd small_example
   Rscript new/<id>/sim1_<id>.R 1 1
   ```
4. If the smoke test fails, fix the script. Do **not** move the model until
   smoke passes.

### 3. Route to fail/ — when sim1 cannot be written

Move `new/<id>/` → `small_example/fail/<id>/` when ANY of the following
hold:

| Condition | Reason |
|---|---|
| STATUS is `FAILED_*` (AI sampler failed L3) | Underlying sampler is broken |
| Smoke test fails after 2 attempts and root cause is the model, not the script | Not fixable at the sim1 layer |

**No external reference is NOT a fail/ condition.** When neither a `.stan`
file nor an R-package fitter exists, fall back to the self-only
single-chain split-rhat pattern (§6 below) — `posterior::rhat()` on a
single AI chain is the convergence diagnostic, and DGP truth gives
coverage. Reserve `fail/` for models whose AI sampler is itself broken.

```bash
mv new/<id> fail/<id>
```

**Never move to `fail/` if the only problem is a fixable script bug.** Fix
the script and retry the smoke test first.

### 4. Move graduated model to small_example/

Only after the sim1 script smoke test **passes**:

```bash
mv new/<id> small_example/<id>
```

The `new/<id>` directory is gone; `small_example/<id>/sim1_<id>.R` exists
and is verified.

### 5. Run 100 replicates

After moving:

```bash
cd small_example
Rscript <id>/sim1_<id>.R 100 8 > <id>/sim1_run.log 2>&1
```

---

## Rules summary

1. **Write sim1 first, move second.** A model stays in `new/` until its
   smoke test passes.
2. **Cannot do sim1 → `fail/`, not `small_example/`.** No `.stan` file,
   FAILED status, or unfixable smoke failure all route to `fail/`.
3. **Never delete `new/`.** It is the permanent staging area. Even when
   empty, it stays.
4. **Ask before any move or delete.** All `mv` and `rm` on model directories
   require explicit user confirmation unless the user has pre-authorized a
   batch.

---

## Quick checklist before touching any model in new/

```
[ ] STATUS = SUCCESS?
[ ] Reference fitter present? — EITHER .stan file OR R-package call in .r
[ ] .r file present?
[ ] sim1 script written?
[ ] smoke test (1 replicate) passes?
→ Only then: mv new/<id> small_example/<id>
```

If any box is unchecked → either fix it or mv to fail/.

---

## 6. Reference implementations: Stan, R package, or self-only single-chain

The sim1 cross-impl harness needs a way to detect convergence and
correctness. Three patterns are supported, ranked by strength:

1. **Stan reference** — strongest; cross-implementation rhat against an
   independently-coded sampler.
2. **R-package reference** — equivalent to Stan when the package is the
   community's standard implementation (e.g., BART pkg for Gaussian BART).
3. **Self-only single-chain split-rhat** — fallback when no external
   reference exists for the model family. Runs ONE chain of our
   implementation and uses `posterior::rhat()` (which splits the chain
   in half internally) to detect within-chain stationarity.

**MCMC budget — fixed at 20k warmup + 20k post for all models.** This
convention applies regardless of what hyperparameters the model's `<id>.r`
file uses for its one-shot reference call. The `.r` file's `nskip` /
`ndpost` / `iter_warmup` / `iter_sampling` are illustrative defaults the
codegen agent picked for a single fit — sim1 needs full convergence on
EVERY one of the 100 replicates, including the slow / unlucky ones, so
both AI and reference chains run at 20k+20k:

```r
N_BURN <- 20000L
N_KEEP <- 20000L
```

Hyperparameters that ARE model-spec (ntree, prior settings, normalization,
etc.) MUST mirror the `.r` reference exactly per Trap 5 below. Only the
chain-length budget is overridden upward.

### Stan reference (legacy / GLM / mixed effects / state-space, etc.)

`new/<id>/<id>.stan` exists. The sim1 script:

```r
# Setup
suppressPackageStartupMessages({ library(cmdstanr); library(posterior) })
stan_mod <- cmdstanr::cmdstan_model("new/<id>/<id>.stan")

# Per-replicate
run_REF_chain <- function(data, seed) {
    fit <- silent_eval(stan_mod$sample(
        data = list(...),                      # Stan-data shape
        seed = as.integer(seed), chains = 1L,
        iter_warmup = N_BURN, iter_sampling = N_KEEP,
        refresh = 0, show_messages = FALSE,
        show_exceptions = FALSE, output_dir = tempdir()))
    list(fit = fit, ...)                       # caller calls posterior::as_draws_matrix
}
```

Alignment via `posterior::as_draws_matrix(fit$draws())` then column
name matching.

### R-package reference (BART family, soft-BART, mixture R packages, etc.)

`new/<id>/<id>.stan` does NOT exist. `new/<id>/<id>.r` invokes a fitter
from an R package (BART, SoftBart, mixtools, ...) directly. The sim1
script:

```r
# Setup — load the package, no .stan to compile
suppressPackageStartupMessages({ library(BART); library(posterior) })

# Per-replicate
run_REF_chain <- function(X, y, seed) {
    set.seed(as.integer(seed))                 # MOST R-pkg fitters use R's
                                               # global RNG; pin it per call
    t0 <- Sys.time()
    fit <- BART::wbart(x.train = X, y.train = y,
                       ntree = 300L, nskip = N_BURN, ndpost = N_KEEP)
    list(fit = fit,
         wall = as.numeric(difftime(Sys.time(), t0, units = "secs")),
         status = "ok", err = NA_character_,
         n_divergent = NA_integer_,            # not applicable for non-NUTS refs
         n_max_treedepth = NA_integer_)
}
```

Alignment via the package's own draw-extraction API. For BART family:

```r
align_REF_draws <- function(ref_run) {
    fit <- ref_run$fit
    # BART::wbart returns an n_keep × N matrix in $yhat.train and a
    # length-n_keep vector in $sigma.
    list(
        f_bart = as.matrix(fit$yhat.train),    # (n_keep × N), per-obs f
        sigma  = matrix(as.numeric(fit$sigma), ncol = 1L)
    )
}
```

The AI's `align_AI_draws` should produce identically-shaped lists with
the SAME named keys, on the SAME parameterization.

### Self-only single-chain split-rhat (fallback)

When NEITHER a Stan implementation NOR a community-standard R-package
implementation exists for a model family — for example, generalized BART
(logistic / Poisson / NB / heteroscedastic BART) where no `BART::*` /
`SoftBart::*` analogue ships — the sim1 validates convergence via the
**rank-normalized split-rhat** computed on a SINGLE AI chain.

**Why this works**: `posterior::rhat()` (Vehtari et al. 2021) splits the
input chain in half internally and compares the two halves' rank-
normalized within- and between-half variance to produce a split-rhat
diagnostic. A single chain is therefore sufficient — split-rhat ≈ 1 is
evidence that the second half of the chain has explored the same
posterior region as the first half. This is the standard "stationarity"
diagnostic in MCMC.

**Caveats**:
- Detects within-chain stationarity but NOT correctness. If our sampler
  has a systematic bug (wrong gradient, wrong prior) or got stuck in one
  posterior mode, split-rhat is still ≈ 1 and the chain looks converged.
  Coverage of truth (from the DGP) is the only correctness signal.
- Does NOT replace cross-impl validation when an external reference IS
  available. Use this fallback only when no Stan / R-package reference
  exists for the model family.

**Code template** — 1 chain, no `run_REF_chain`:

```r
N_BURN <- 20000L
N_KEEP <- 20000L

run_AI_chain <- function(X, y, seed) {
    set.seed(as.integer(seed))                       # BART arn RNG
    m <- new(<ClassName>, X, y, as.integer(seed), TRUE)
    m$step(N_BURN); m$step(N_KEEP)
    list(model_obj = m, ...)
}

# Per-replicate worker — ONE chain, split-rhat per parameter dim.
worker <- function(r) {
    set.seed(r); dgp <- simulate_data_with_truth(seed = r)
    ai_run  <- run_AI_chain(dgp$data$X, dgp$data$y, seed = r)
    aligned <- align_AI_draws(ai_run$model_obj)
    rhat    <- compute_split_rhat(aligned)           # see below
    cov     <- compute_coverage(aligned, dgp$truth) # standard helper
    ...
}

# Single-chain split-rhat: posterior::rhat() splits internally.
compute_split_rhat <- function(draws) {
    out <- list()
    for (key in names(draws)) {
        d     <- draws[[key]]                         # (n_keep × dim)
        dim_p <- ncol(d)
        rhats <- numeric(dim_p)
        for (j in seq_len(dim_p)) rhats[j] <- posterior::rhat(d[, j])
        names(rhats) <- if (dim_p == 1L) key
                        else paste0(key, "[", seq_len(dim_p), "]")
        out[[key]] <- rhats
    }
    unlist(out, use.names = TRUE)
}
```

**csv schema**: same as Stan / R-package patterns, but `wall_sec_REF`,
`status_REF`, `coverage_REF_mean`, all ESS_REF columns are NA. The
`rhat` column means "split-rhat within the single AI chain" instead of
"cross-impl AI vs reference rhat".

### Detecting which pattern applies

```bash
# 1. Stan reference iff a .stan file exists
test -f "new/<id>/<id>.stan"   # → Stan pattern

# 2. R-package reference iff <id>.r calls a non-base / non-Stan fitter
grep -E "library\(|BART::|SoftBart::|mixtools::|bayesm::" "new/<id>/<id>.r"
# → R-package pattern (mirror the call in run_REF_chain)

# 3. Otherwise, self-only single-chain split-rhat fallback. Verify the
#    .r file has NO fitter call (only simulate_data + truth attrs):
grep -E "wbart\(|pbart\(|softbart|::sample\(|::rmix" "new/<id>/<id>.r"
# → no hits → self-only single-chain split-rhat pattern
```

The codegen agent should set this at model-creation time. The sim1
author mirrors whichever pattern applies.

### Common run_REF_chain template choices by family

| Model family | R package + fitter | Notes |
|---|---|---|
| Continuous BART, Gaussian outcome | `BART::wbart(x.train, y.train, ntree, nskip, ndpost)` | Returns `$yhat.train` (n_keep × N), `$sigma` (n_keep). Default `ntree = 200`; use the value from `<id>.r`. |
| Probit/Bernoulli BART | `BART::pbart(x.train, y.train, ntree, nskip, ndpost)` | Returns `$prob.train`; AI's f is on logit/probit scale — back-transform for cross-impl |
| Soft BART, Gaussian | `SoftBart::softbart(X, Y, X_test, hypers, opts)` | Per-package `Hypers()` includes `normalize_bart(Y)` + `cv.glmnet`-derived `sigma_hat` (see `codegen_r_runner.md §9a`); the sim1's REF call MUST use the same `Hypers()` to keep the prior match |
| Soft BART, probit | `SoftBart::softbart_probit` | Same hypers caveat as above |
| Gaussian mixture (K known) | `mixtools::normalmixEM` (EM, not MCMC; less suitable) or `bayesm::rmixGibbs` (MCMC) | Label switching applies — see `label_switching.md` |
| **Generalized BART (logistic / Poisson / NB / heteroscedastic / AFT / etc.)** | **NO R package available** | **Use self-only single-chain split-rhat pattern (above)** |
| **Soft BART variants beyond Gaussian / probit** | **NO R package available for arbitrary likelihoods** | **Use self-only single-chain split-rhat pattern** |

When the model family has neither an R-package implementation nor a Stan
implementation, the self-only single-chain split-rhat pattern is the
right call — NOT `fail/`. Routing to `fail/` is reserved for models
whose AI sampler itself is broken (FAILED_L3 status).

---

## Common sim1 alignment traps

### Trap 1: joint_nuts_block history keys

When the AI cpp uses `joint_nuts_block`,
the `get_history()` keys are **NOT** the same as `get_current()` keys.
`get_current()` is a user-facing wrapper that may split / transform the
joint slice; `get_history()` returns RAW per-block history under the
`cfg.sub_params[k].name` (the block-level name).

**Always check `cfg.sub_params` (or `cfg.name` for non-joint blocks) in
the .cpp before writing `align_AI_draws`.** Do NOT trust the
`get_current()` list names.

Patterns seen in practice (described abstractly to keep the skill
benchmark-agnostic):

| Pattern | get_current() key | get_history() key (actual) |
|---------|-------------------|------------------------------|
| Whitened-latent GP-regression: `f` derived from `(z, alpha, rho)` at predict time | `f` | NOT IN HISTORY (drop from align) |
| Joint NUTS over a 3-vec on log-scale (e.g., several POSITIVE PK params) | `k_a, K_m, V_m` (split + transformed) | `theta_log` (n_iter × 3, log-scale; need `exp()`) |
| Joint NUTS over a simplex with auxiliary slack (e.g., GARCH-style `(alpha1, beta1, slack)`) | `alpha1, beta1` (split) | `ab_simplex` (n_iter × 3) — first two columns are `alpha1, beta1` |

The symptom is `align_failed: 'data' must be of a vector type, was 'NULL'`
because `h$<wrong_key>` is NULL. To recover:

1. `grep -n "sub_params.push_back\|cfg.name" <id>.cpp` — find the actual
   block names
2. `grep -n "register_refresher\|impl_->data().set" <id>.cpp` — find any
   derived keys (these usually show up in `get_current()` but NOT in
   `get_history()`)
3. Update `align_AI_draws` to read the block-level keys, then transform
   to the natural / Stan-comparable scale (e.g., `exp(theta_log)`,
   `ab_simplex[, 1:2]`, etc.)

### Trap 2: AI cpp constructor argument order

The AI cpp constructor argument order in the .cpp is the SOURCE OF TRUTH.
The `<id>_runner.R` comment block (`new(<id>, ...)`) is usually correct
but not always — when in doubt, `grep` for the C++ class constructor:

```
grep -n "<id>(" <id>.cpp | head
```

A wrong order silently passes (e.g., `dist` values in the `switched`
slot) until a runtime check fires (`switched must be 0/1`). Always
verify the order before writing `run_AI_chain`.

### Trap 3: Stan parameter vs generated-quantities (Stan reference only)

Some Stan models expose useful quantities only in `generated quantities`
(e.g., back-transformed regression coefficients on the un-adjusted
covariate scale, or recovered discrete latents `z` for marginalized
HMM / mixture models). These ARE in `as_draws_matrix()` and look
identical to parameter draws — but cross-impl rhat against an AI
parameter that uses a DIFFERENT parameterization (e.g., AI samples on
adjusted scale, Stan generates un-adjusted) is meaningless.

When in doubt, align both samplers on the SAME-PARAMETERIZATION
quantity (typically the actual `parameters` block). Use generated
quantities only as a sanity check, not for cross-impl rhat.

### Trap 4: R-package reference RNG semantics (R-package reference only)

Most R-package fitters use **R's global RNG** (`set.seed()` controls
everything). They do NOT accept an integer-seed argument like Stan does.
Consequence: in the `worker(r)` function, BOTH the AI chain seeding AND
the R-package call must be preceded by `set.seed(seed)` so per-replicate
reproducibility is preserved. Failing to do this makes the per-replicate
seed argument a no-op for the reference and you get different reference
posteriors across re-runs of the same replicate.

```r
run_REF_chain <- function(X, y, seed) {
    set.seed(as.integer(seed))                 # ← REQUIRED
    fit <- BART::wbart(...)
    ...
}
```

Family-specific exceptions:
- `SoftBart::softbart()` accepts `set.seed()` similarly.
- `bayesm::rmixGibbs()` and a few other packages accept their own seed
  argument; check the package docs.

### Trap 5: R-package reference's hyperparameters MUST match `<id>.r`

The reference call in `<id>.r` is the agent-curated specification of
"how the package should be invoked for THIS model" — its arguments
encode the model spec (ntree, nskip, ndpost, prior calibration choices,
whether to normalize Y, etc.). The sim1 `run_REF_chain` MUST mirror the
same call verbatim, including all hyperparameters. Do not silently
substitute a "more standard" default.

Common gotcha for SoftBart references: `<id>.r` calls
`SoftBart::softbart(X, Y, hypers = Hypers(X, Y, ...))`. The `Hypers()`
constructor applies `normalize_bart(Y)` and `cv.glmnet`-derived
`sigma_hat` per-call (see `codegen_r_runner.md §9a`). If sim1's
`run_REF_chain` reconstructs the call by hand, those preprocessing
steps are silently dropped and the reference posterior shifts.

---
name: AI4BayesCode-sim2-workflow
description: |
  Governs Setting 2 of the AI4Bayes evaluation -- generation-variability
  assessment. Five independent AI4BayesCode samplers are generated from the
  SAME natural-language model description, each evaluated on the same 20
  simulated datasets. This skill specifies the directory layout, the
  per-replicate one-reference-shared-across-five-samplers protocol, the
  naming-collision fix for sourcing 5 cpps in a single R session, and the
  CSV schema (sim1 columns + sampler_id).
---

# sim2 workflow -- generation-variability assessment

## Purpose

In sim1 we measure correctness / perf of ONE AI sampler against a reference
across many datasets. In sim2 we measure **variability across 5 INDEPENDENT
regenerations** of the same natural-language model description. Each of the
5 generated samplers is evaluated on the SAME 20 datasets, and per-sampler
metrics (rhat vs reference, ESS, runtime) are summarized as
median [min--max] across the 5.

## Directory layout

```
sim2/
+-- AI4BayesCode/    <- shared library (copied here for sim2-local sourceCpp)
+-- models1/         <- regeneration 1 of each model (one cpp per model)
|   +-- Mt_model/
|   |   +-- Mt_model.cpp
|   |   +-- STATUS, l2_report.md, session.log
|   |   +-- ...
|   +-- <other models>/
+-- models2/         <- regeneration 2 (independent codegen run)
+-- models3/
+-- models4/
+-- models5/
+-- results/         <- one subfolder per model, holds the sim2 script + outputs
    +-- Mt_model/
        +-- sim2_Mt_model.R
        +-- results.csv
        +-- results.rds
```

The 5 `models<k>/<model>/<model>.cpp` files all implement the same
natural-language model description (`<model>.txt`) but were produced by
independent codegen runs. They may differ in block layout, initial values,
hyperparameter choices, numerical implementation -- that variability is
exactly what sim2 measures.

**Shared resources** (NOT under sim2/, lifted from the canonical `small_example/<model>/` directory):
- `<model>.r` -- the data-generating process
- `<model>.stan` (if Pattern A) -- Stan reference; or the R-package reference
  call lives in the .r file

## Conventions

| Setting | sim1 | **sim2** |
|---|---|---|
| MCMC budget | `N_BURN = N_KEEP = 20000` | **same** |
| Replicates per model (`M`) | 100 | **20** |
| AI chains per replicate | 1 | **5** (one per regeneration) |
| Reference chains per replicate | 1 | **1** (REUSED across 5 AI comparisons) |
| Total rows in csv per model | 100 (N_REPLICATES x 1) | 100 (N_REPLICATES x N_SAMPLERS = 20 x 5) |
| Extra csv column | -- | `sampler_id  in  {1..5}` |

**The reference chain runs ONCE per dataset and is REUSED for all 5 AI
comparisons** (paper Setting 2 protocol). This is a substantial compute
saving: total reference work per model is 20 chains, not 100.

Per-row interpretation:
- `rhat`: AI sampler `k` cross-impl rhat vs the SAME reference draws (NOT
  AI-vs-AI). The 5 rows of one replicate share `aligned_ref` but each gets
  its own `aligned_ai_k`.
- `ess_bulk_AI`, `ess_tail_AI`, `wall_sec_AI`: per-AI-chain -- DIFFERENT for
  each `sampler_id` even within the same replicate.
- `ess_bulk_REF`, `ess_tail_REF`, `wall_sec_REF`, `n_divergent_REF`,
  `n_max_td_REF`: SHARED across all 5 rows of the same replicate (computed
  once on `aligned_ref`).
- `coverage_AI`: per-sampler (each sampler's CIs).
- `coverage_REF`: shared across the 5 rows.

## Naming-collision fix (mandatory)

Sourcing all 5 `models<k>/<model>/<model>.cpp` files in a single R session
causes Rcpp to register the SAME class name 5 times -- the LAST sourceCpp
wins, so only sampler 5 is reachable via `new(<model>, ...)`. Sampler 1-4
become invisible.

**Fix: sed-rename each class to `<MODEL>_v{k}` before sourcing.** Use a
word-boundary regex so the Rcpp module-registration name (which contains
`<MODEL>_module` -- `_` is a word character) is left untouched, but
standalone class identifiers and the `Rcpp::class_<...>("...")` strings
ARE renamed. Code idiom:

```r
N_SAMPLERS <- 5L
MODEL_NAME <- "Mt_model"

ai_classes <- character(N_SAMPLERS)
for (k in seq_len(N_SAMPLERS)) {
    src_path  <- file.path("sim2", paste0("models", k), MODEL_NAME,
                           paste0(MODEL_NAME, ".cpp"))
    new_class <- paste0(MODEL_NAME, "_v", k)
    tmp_cpp   <- tempfile(fileext = ".cpp")
    cpp_text  <- readLines(src_path)
    # \bMODEL\b matches: standalone identifiers, Rcpp::class_<MODEL>,
    # "MODEL" string literal. Does NOT match MODEL_module (because _
    # is a word character and there's no \b between MODEL and _).
    cpp_text  <- gsub(paste0("\\b", MODEL_NAME, "\\b"), new_class, cpp_text)
    writeLines(cpp_text, tmp_cpp)
    AI4BayesCode_sourceCpp(tmp_cpp,
                           AI4BayesCode_path = "sim2/AI4BayesCode")
    ai_classes[k] <- new_class
}

# Dispatch in run_AI_chain -- sampler_id picks which renamed class to construct:
run_AI_chain_k <- function(data, seed, sampler_id) {
    cls <- get(ai_classes[sampler_id])  # symbol -> Rcpp module class
    m   <- new(cls, /* model-specific args */, as.integer(seed), TRUE)
    ...
}
```

The 5 cpps go into separate dylibs (one per sourceCpp call), so the
`RCPP_MODULE(<MODEL>_module)` collisions across files are at the C++
symbol-table level only -- different dylibs don't share symbol tables, so
there's no link conflict. Only the R-side `Rcpp::class_<>("string")`
registrations matter, and our sed renames those to be unique.

## Worker structure (one-reference-five-AI-chains)

```r
N_REPLICATES <- 20L
N_SAMPLERS   <- 5L

worker <- function(r) {
    set.seed(r)
    dgp     <- simulate_data_with_truth(seed = r)

    # ONE reference chain per dataset -- REUSED for all 5 AI chains.
    ref_run     <- run_REF_chain(dgp$data, seed = r)
    aligned_ref <- if (ref_run$status == "ok")
        tryCatch(align_REF_draws(ref_run$fit), error = function(e) NULL)
        else NULL

    # 5 independent AI chains, each compared against the SAME aligned_ref.
    out_rows <- vector("list", N_SAMPLERS)
    for (k in seq_len(N_SAMPLERS)) {
        ai_run <- run_AI_chain_k(dgp$data, seed = r, sampler_id = k)
        out_rows[[k]] <- build_row(r, k, ai_run, ref_run,
                                    aligned_ref, dgp$truth)
    }
    out_rows  # list of N_SAMPLERS row-lists
}

# mclapply over replicates returns list-of-list (each of N_SAMPLERS).
# Flatten before summarize_to_row + rbind:
results       <- par_lapply(seq_len(N_REPLICATES), worker, ...)
results_flat  <- unlist(results, recursive = FALSE)   # length = N_REPLICATES * N_SAMPLERS
csv_df        <- do.call(rbind, lapply(results_flat, summarize_to_row))
```

## CSV schema

Same as sim1 (32 columns) PLUS `sampler_id` (after `replicate`). 33 columns
total, 100 rows per model (20 reps x 5 samplers).

```
replicate, sampler_id, status, status_AI, status_REF,
wall_sec_AI, wall_sec_REF, n_divergent_REF, n_max_td_REF, n_params,
rhat_max, rhat_min, rhat_mean, rhat_median,
ess_bulk_AI_max, ess_bulk_AI_min, ess_bulk_AI_mean, ess_bulk_AI_median,
ess_bulk_REF_max, ess_bulk_REF_min, ess_bulk_REF_mean, ess_bulk_REF_median,
ess_tail_AI_max, ess_tail_AI_min, ess_tail_AI_mean, ess_tail_AI_median,
ess_tail_REF_max, ess_tail_REF_min, ess_tail_REF_mean, ess_tail_REF_median,
coverage_AI_mean, coverage_REF_mean
```

Crash-robust schema (sim1_workflow lessons): the `summarize_to_row` function
must always return the SAME 33-column data.frame even on `try-error`
worker results, otherwise rbind fails when even one worker dies.
Mclapply over a list-of-5 may return either a list-of-5-rows OR a single
try-error if the WHOLE worker died -- handle both.

## Where the script lives + auto-chdir

- Script path: `sim2/results/<model>/sim2_<model>.R`
- Auto-chdir to project root (`block_MCMC`) -- that's `dirname x 4` from the
  script path:

```r
.script_path <- (function() {
    args <- commandArgs(trailingOnly = FALSE); fi <- grep("^--file=", args)
    if (length(fi) > 0L) return(normalizePath(sub("^--file=", "", args[fi[1L]]),
                                              mustWork = FALSE))
    for (i in seq_along(sys.frames())) {
        of <- sys.frames()[[i]]$ofile
        if (!is.null(of)) return(normalizePath(of, mustWork = FALSE))
    }
    NULL
})()
if (!is.null(.script_path))
    setwd(dirname(dirname(dirname(dirname(.script_path)))))
```

From the project root the canonical paths are:
- AI cpps: `sim2/models{1..5}/<MODEL>/<MODEL>.cpp`
- AI4BayesCode: `sim2/AI4BayesCode`
- DGP source: `small_example/<MODEL>/<MODEL>.r` (or lift the body verbatim
  from the existing `sim1_<MODEL>.R`)
- Stan reference: `small_example/<MODEL>/<MODEL>.stan` (or R-package fitter)
- Output: `sim2/results/<MODEL>/{results.csv, results.rds}`

## Routing

Same as sim1_workflow Sec.3 (FAILED_L3 -> fail/), but the routing decision is
made on the **canonical models1 cpp** as the agent's "first attempt".
If models1 says SUCCESS but models3 says FAILED_L3, the model still goes
to sim2 -- the variability across the 5 samplers IS what we want to measure.
The 5 sampler chains may individually fail at L3-level reasons (too high
rhat, divergent transitions); these failures are what sim2 quantifies, so
they should NOT route the whole model to fail/.

## Common sim2-specific pitfalls (in addition to sim1 traps)

### Trap A: forgot to sed-rename -> only sampler 5 is reachable

If you sourceCpp the 5 raw cpps without renaming, only the LAST registered
class is callable -- `new(MODEL, ...)` constructs sampler 5 every time.
You'll see all 5 sampler_id rows produce IDENTICAL chains. Detection: in
the smoke output, all 5 AI walls are identical or rhat across the 5
samplers is exactly 1.000 (suspicious for "independent" generations).

### Trap B: re-running the reference chain 5 times

The ref chain is the slowest part for many models. Running it 5 times per
replicate (one per AI comparison) defeats the compute optimization. Verify
that `ref_run` and `aligned_ref` are computed ONCE per worker call and
referenced inside the per-sampler loop.

### Trap C: mclapply returns list-of-list -- must flatten before rbind

Each worker returns N_SAMPLERS rows (a list of 5 lists). After mclapply,
results is a list-of-list-of-5. Calling rbind directly fails. Always:
```r
results_flat <- unlist(results, recursive = FALSE)
# now length = N_REPLICATES * N_SAMPLERS
```

### Trap D: per-replicate seed for DGP, distinct seed for each AI chain

The DGP uses `set.seed(r)` so all 5 samplers see the SAME data. Each AI
chain uses its own seed (typically also `r`, since the cpp's RNG is
seeded by the constructor; identical AI seeds across samplers are fine
because the underlying cpp implementations differ -- the chain trajectories
will still diverge).

### Trap E: shared aligned_ref but per-sampler ESS_REF metric

`compute_ess(aligned_ref, ...)` returns the SAME number for all 5 sampler_id
rows of one replicate. This is correct (it's the ref's ESS, not a
per-sampler quantity) -- the redundancy is documented and useful for
downstream summarization (median across rows = single value when the
replicate's ref converged).

---
name: AI4BayesCode-codegen-r-runner
description: |
  R runner template for AI4BayesCode samplers -- ai4bayescode_source
  setup, constructor-argument reference block, the delivered
  inline-constructor-closure + shipped ai4bayescode_run_chains /
  ai4bayescode_rhat_summary / ai4bayescode_diagnose flow (NO generated
  helpers in the example; the harness-internal run_chain_<ClassName>()
  with keep_history = TRUE is NOT shipped), Layer 3 validator wiring
  (R1 smoke check, R2
  rank-normalized R-hat + ESS via posterior, R3 Bayesian p-values +
  PSIS-LOO via loo), run timing, joint-NUTS threshold
  tightening, and the reference-template catalogue (examples/*.cpp).
  Extracted from codegen.md (sections 9 and 10). The entry-point
  skill `codegen.md` points here for R runner emission.
---

# AI4BayesCode codegen -- R runner + reference templates

**PREREQUISITE GATE -- L2 first.** Do not load this file, and do not
emit or run any runner / Layer-3 harness, until the L2 semantic
checklist + AD-twin (Check #12) have PASSED with the per-check verdict
table printed (`codegen.md` Sec.11 Step 2, HARD ORDERING GATE). This
file hands you executable runtime machinery; if L2 has not been
printed yet, you are in the runtime-before-semantic reversal.

Companion skill to `codegen.md`. Load this when writing the generated
`.R` runner: ai4bayescode_source setup, constructor-argument reference
block, the delivered inline-constructor-closure + shipped
`ai4bayescode_run_chains` / `ai4bayescode_rhat_summary` /
`ai4bayescode_diagnose` flow, the harness-internal
run_chain_<ClassName>() helper, Layer 3 validator wiring
(R-hat, ESS,
Bayesian p-values, PSIS-LOO), run timing, and the
reference-template catalogue.

For prior elicitation + block selection, see `codegen_priors.md`.
For the C++ file emission, see `codegen_cpp.md`.

---

## 9. Output

1. Create output folder if missing (default `./generated/<ClassName>/`).
2. Write `<folder>/<ClassName>.cpp`. **It MUST carry BOTH the `@example:R`
   AND the `@example:python` header blocks** (per `codegen_cpp.md` Sec.5 "Header
   `@example` block") -- the `.cpp` is ALWAYS dual-module, so a Python user who
   takes the same file must still see the Python example via
   `AI4BayesCode.doc()`, and an R user the R example via `ai4bayescode_doc()`.
   Both blocks are the SAME toy DGP this runner uses, distilled to <= ~8
   runnable lines using each language's packaged API (`library(AI4BayesCode)`
   -> `new(<ClassName>, ...)` for R; `AI4BayesCode.source(...)` -> `Mod(...)`
   for Python). Write the DGP once and mirror it into BOTH header blocks so
   the runner and both doc cards cannot drift.
3. Write the R runner `<folder>/run_<ClassName>.R` following the
   template below and **use it to gate generation** (it carries the
   Layer-3 R1/R2/R3 harness -- see the CORRECTNESS INVARIANT in
   `codegen.md` "Delivered-code validation harness": Layer-3 ALWAYS
   runs and must PASS regardless of the user's harness choice).
4. **Deliverable depends on the "Delivered-code validation harness"
   answer:**
   - **No / minimal example (default):** on validator PASS, **DELETE**
     `run_<ClassName>.R` (it is a throwaway -- same rule as the Step-2
     smoke test and the Sec.6b autodiff-verify file) and instead write a
     minimal `<folder>/example_<ClassName>.R` (template below). The
     production `.cpp` already contains zero validation code.
   - **Yes (opt-in):** keep `run_<ClassName>.R` as the delivered
     artifact (today's behaviour).
   On validator FAIL within `max_attempts`: stop-and-report; do NOT
   write/ship the usage example (it is produced ONLY on PASS).
5. Do NOT modify anything outside the output folder.
6. Tell the user: path, class name, exact R commands (for the
   default case, the commands are just the usage example).

### Usage-example template (`example_<ClassName>.R`, default deliverable)

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

NOT a stripped golden-path snippet, and NOT a copy of the Layer-3
harness prefix. The delivered example drives chains EXCLUSIVELY
through the SHIPPED multi-chain helpers (`ai4bayescode_run_chains`,
`ai4bayescode_rhat_summary`, `ai4bayescode_diagnose`), with NO
generated helper functions at all -- the constructor is an inline
closure argument. The
`run_chain_<ClassName>()` chain helper defined in the R runner
template below is HARNESS-INTERNAL (Layer-3 R1/R2/R3 + sim1 need its
`pp = predict_at(...)` collection) and is NEVER copied into the
delivered example; the whole harness is throwaway and NOT shipped in
the default case.

The delivered `example_<ClassName>.R` MUST contain, in order, with
the comments retained (do not strip comments -- they are the
documentation a first-time user reads):

1. **Header comment** -- what the model is, plus the short list of the
   shipped helpers the example uses (see the skeleton below). No
   harness talk, no validator vocabulary (PSIS-LOO, BPV, Layer-3, check
   numbers), no explanations of what the example does NOT contain --
   an acronym with no context, or an absence-explanation, only
   confuses the audience.
2. **Compile** -- two branches, decided by the `start.md` Sec.1b install
   detection (never assume the package is installed):
   - **Package installed (recommended):** the packaged API compiles from a bare
     RELATIVE filename -- `library(AI4BayesCode)` then
     `ai4bayescode_source("<ClassName>.cpp")` (equivalently
     `ai4bayescode_source(...)`). Here NEVER emit `AI4BayesCode_path=`, a
     `source(".../*_helpers.R")` line, or an absolute `/Users/...` path (all
     three break on another machine or if the folder moves).
   - **R checkout mode (package NOT installed):** emit
     `source("<path>/AI4BayesCode/R/AI4BayesCode_helpers.R")` followed by
     `ai4bayescode_source_checkout("<ClassName>.cpp", AI4BayesCode_path = "<path>/AI4BayesCode")`.
     The checkout compiler is `ai4bayescode_source_checkout()`, NOT
     `ai4bayescode_source()` -- it needs the path because there is no installed
     package to take headers from, and the distinct name keeps sourcing the
     helpers from ever shadowing the packaged API.

   Everything AFTER the compile is identical in both branches: the checkout
   helpers ship `ai4bayescode_run_chains` / `ai4bayescode_rhat_summary` /
   `ai4bayescode_diagnose` too, so the example's shape never changes -- only
   these first one or two lines do.
   IMPORTANT -- where the class binds: `ai4bayescode_source()` /
   `ai4bayescode_source()` bind the compiled class into the CALLER'S frame
   (`env = parent.frame()`). So compile at the TOP LEVEL of the runner. If you wrap
   the compile inside a helper FUNCTION, pass `env = globalenv()` explicitly, else
   `new(<ClassName>, ...)` fails with 'object <ClassName> not found'.
3. **Compact constructor reference + doc() pointer** -- a SHORT
   comment block: the `new(<ClassName>, ...)` call shape with one
   brief comment per data argument, then ONE pointer line:

   ```r
   # Full constructor / methods / prior reference:
   #   ai4bayescode_doc(<ClassName>)
   ```

   The FULL 30-40 line argument + methods documentation lives in the
   `.cpp` header (codegen_cpp.md Sec.5) -- that is exactly what
   `ai4bayescode_doc()` parses and prints, so do NOT duplicate it
   into the example. A wall of reference comments is what makes the
   example unreadable for the non-programmer audience.
4. **NO generated helper functions -- the constructor goes INLINE.**
   The example defines NOTHING between the data and the
   `ai4bayescode_run_chains` call. The constructor is passed as an
   inline one-argument closure, exactly like the library examples'
   own `@example` headers:

   ```r
   run <- ai4bayescode_run_chains(
       function(seed) new(<ClassName>, <data_args>, as.integer(seed), TRUE),
       n_chains = 4L, n_burn = 4000L, n_keep = 4000L)
   ```

   HARD RULE (the audience is users with NO programming background:
   one visible call, zero abstractions): do NOT emit a named
   constructor factory (`make_<ClassName>` or ANY
   function-returning-function), do NOT emit a bespoke chain driver
   (`run_chain_<ClassName>()` or any burn/collect/return wrapper in
   the deliverable), and do NOT emit
   `for (i in ...) { m$step(1L); m$get_current() }` per-iteration
   accumulation loops (draw collection is `keep_history = TRUE` +
   `get_history()`, which `ai4bayescode_run_chains` already does
   internally). The closure reading the data variables defined in the
   simulation block above is fine and intended -- an example script is
   one linear file. The ONLY reason to expand the closure to a
   multi-line body -- still inline, still unnamed -- is a
   kernel-control pin, which must run inside each worker:

   ```r
   run <- ai4bayescode_run_chains(
       function(seed) {
           m <- new(<ClassName>, <data_args>, as.integer(seed), TRUE)
           m$freeze("sigma")   # only when the model pins a block
           m
       },
       n_chains = 4L, n_burn = 4000L, n_keep = 4000L)
   ```

   Chain running, cross-chain R-hat/ESS, and per-chain diagnostics ALL
   come from the SHIPPED library functions; the harness-internal
   `run_chain_<ClassName>()` stays in the throwaway runner only.
5. **Simulation / toy-data code** -- generate `<data_args>` (+ a
   held-out `<X>_test` for the predict example), commented.
6. **Multi-chain usage + diagnostics (the everyday flow)** -- the
   shipped helpers, nothing else:

   ```r
   run <- ai4bayescode_run_chains(
       function(seed) new(<ClassName>, <data_args>, as.integer(seed), TRUE),
       n_chains = 4L, n_burn = 4000L, n_keep = 4000L)
   ai4bayescode_rhat_summary(run)   # convergence check across the chains
   dg <- ai4bayescode_diagnose(run$histories[[1]])
   dg$summary; dg$plot
   ```

   Single-chain use is `n_chains = 1L` (or the stateful API below);
   never a bespoke wrapper.
7. **Stateful-API usage examples** -- commented, in this order:
   `new(<ClassName>, ..., keep_history = TRUE)`, `model$step()`,
   `model$get_history()`, `model$get_current()`, then
   `model$predict_at(list(<X> = <X>_test))`. The
   `model$set_current(list(<a data input> = ...))` block comes
   **LAST, AFTER predict_at, and is itself COMMENTED OUT** (`## `
   on every line): its `<updated value>` is supplied by an outer
   Gibbs composition / another block, so as live code it would error
   in a standalone run. Commenting it keeps the delivered example
   runnable end-to-end while still documenting the stateful-update
   pattern (no reinitialization).

**Writing the simulation block (agent-facing -- this guidance does NOT go
into the file).** For HIERARCHICAL / random-effects / weight-variance
models, draw the scale or variance hyperparameter FROM ITS PRIOR, then
draw the effects at that scale. Do NOT hard-code the scale to an
arbitrary "moderate" value (e.g. sd = 0.6): a fixed moderate scale can
CONFLICT with a tight or heavy-tailed prior -- the classic case is a
tiny-scale InvGamma weight-variance prior (Neal-1996 / ARD weight-variance
BNN, where the true variance can be ~1e-6), where fixed sd 0.6 makes the
DATA say sigma^2 ~= 0.36 while the PRIOR insists sigma^2 ~= 1e-6. That
prior-data conflict is an artificially hard, poorly-mixing posterior
(R-hat stuck ~1.02, tiny ESS) that then gets misread as a sampler
problem. Prior-drawn hyperparameters keep the generation-time self-test
calibrated and on the geometry the model actually targets. If the model
ships its own reference simulator (e.g. a `simulate_data(seed)` in the
model card), PREFER it over an invented DGP.

Skeleton (parameterized; mirror this structure, fill placeholders):

```r
# Usage example for <ClassName> -- <one line: what the model is>.
# Everything runs through the shipped library helpers:
#   ai4bayescode_run_chains   -- run several MCMC chains in parallel
#   ai4bayescode_rhat_summary -- convergence check across the chains
#   ai4bayescode_diagnose     -- posterior summaries + diagnostic plots

# Compile + load the model (the class becomes available by name).
library(AI4BayesCode)
ai4bayescode_source("./<ClassName>/<ClassName>.cpp")

# <compact constructor reference: the new() call shape + one brief
#  comment per data argument -- keep it SHORT>
# Full constructor / methods / prior reference:
#   ai4bayescode_doc(<ClassName>)

# Simulate a toy data set
# <commented synthetic generation of <data_args> + a held-out X test>

# Multi-chain run + diagnostics (the everyday flow; shipped helpers only).
# The model constructor is the inline function(seed) argument -- it builds
# one fresh model per chain from the data simulated above.
run <- ai4bayescode_run_chains(
    function(seed) new(<ClassName>, <data_args>, as.integer(seed), TRUE),
    n_chains = 4L, n_burn = 4000L, n_keep = 4000L)

ai4bayescode_rhat_summary(run)   # convergence check across the chains

# Per-chain summaries + trace + autocorrelation + density plots.
dg <- ai4bayescode_diagnose(run$histories[[1]])
dg$summary   # per-parameter table: mean/sd/median/90% CI, R-hat, ESS
dg$plot      # trace + autocorrelation + density (prints; needs 'bayesplot')

# Examples of stateful functions

## Initialize with full history (keep_history = FALSE keeps only the
## last draw).
model <- new(<ClassName>, <data_args>, seed = 1, keep_history = TRUE)

model$step(10)         ## run 10 iterations
model$get_history()    ## 10 posterior draws (all params)
model$step(100)        ## run 100 more
model$get_history()    ## 110 posterior draws
model$get_current()    ## the last (110th) draw

## predict_at() for predictions.
model$predict_at(list(<X> = <X>_test))

## set_current() updates the sampler statefully (e.g. an outcome
## refreshed by other blocks) WITHOUT reinitialization. Shown
## COMMENTED OUT: <updated value> would be supplied by an outer Gibbs
## composition / another block, so this is illustrative only and is
## NOT executed in the standalone example (keeps the script runnable
## as-is). Uncomment in a real stateful-composition context.
## model$set_current(list(<a data input> = <updated value>))
## model$step(1)          ## one iteration after set_current
## model$get_history()    ## 111 posterior draws

## ---- Advanced control (uncomment what you need) --------------------
##
## Re-tune the sampler without advancing the chain -- useful after
## set_current() changes the data enough that the old tuning no longer
## fits. (Only present when the model uses NUTS.)
## model$readapt_NUTS(500L, FALSE, -1L)
##
## Hold one parameter fixed instead of sampling it -- e.g. pin sigma = 1
## for a probit-style model, or fix a parameter for a sensitivity check.
## model$set_current(list(sigma = 1))
## model$freeze("sigma")     ## sigma stays 1; everything else keeps sampling
## model$step(1L)
## model$get_frozen()        ## -> "sigma"
## model$unfreeze("sigma")   ## resume sampling it
##
## Same thing in one line at construction:
## model <- ai4bayescode_new_frozen(<ClassName>, <data_args>, rng_seed = 42L,
##                                  keep_history = TRUE,
##                                  fixed = list(sigma = 1))
##
## You can also hold a single component of a jointly-sampled parameter,
## or one element of it:
## model$freeze("<component_name>")
## model$freeze("<component_name>[2]")
```

**Emitting the advanced-control block (agent-facing rules -- these do NOT
go into the file).**

- The `readapt_NUTS` lines are CONDITIONAL: emit them ONLY if the
  composite has a NUTS-family child (`nuts_block` / `joint_nuts_block`).
  Drop them entirely for pure BART / Gibbs / VI / HMM / SBP composites.
- `freeze()` is strict and `unfreeze()` is permissive: `freeze()` with no
  argument, an unknown name, or a blacklisted family (BART / genBART /
  softbart / VI / HMM) raises; `unfreeze()` with no argument releases
  all, while `unfreeze(character(0))` raises.
- Element-level freeze (`"<component>[k]"`, 1-based, column-major for
  matrices) works only on PER-ELEMENT components (REAL / POSITIVE /
  LOWER_BOUNDED / UPPER_BOUNDED / INTERVAL). Coupled components
  (ORDERED / SIMPLEX / CORR_MATRIX / COV_MATRIX / ...) raise -- freeze
  the whole component. `get_frozen()` reports it back as
  `"<component>[k]"`.
- Other valid names: `"<outer>.<inner_leaf>"` for a nested composite,
  `"<rjmcmc_name>.gamma"` / `".beta"` for rjmcmc sub-keys. Batch
  refreeze on checkpoint restore takes `quiet = TRUE` to suppress
  redundant-refreeze warnings.
- Show only what THIS model supports. Never paste the whole taxonomy
  into the delivered example.

The example must run as-is on the synthetic shapes generated in the
simulation block. The ENTIRE Layer-3 harness -- including its internal
`run_chain_<ClassName>()` chain helper and everything from
`c1 <- run_chain_<ClassName>(... seed = 101L ...)` onward -- is
omitted from the default deliverable and deleted on PASS (shipped
only under the opt-in). The delivered example never contains a
bespoke chain driver; it uses `ai4bayescode_run_chains` +
`ai4bayescode_rhat_summary` + `ai4bayescode_diagnose`.

### R runner template

The generated `run_<ClassName>.R` must include:
1. A **constructor reference block** documenting every argument to `new()`
   with its type, description, and default/valid range.
2. A burnin phase and a draw-collection loop.

Follow this structure:

**Path-resolution rule (KISS principle -- no runtime detection).** The
generated runner uses HARDCODED RELATIVE PATHS that the codegen agent
fills in at generation time. No `sys.frame(1)$ofile`, no
`commandArgs(trailingOnly = FALSE)` parsing, no `rstudioapi` calls, no
walking directories. Runtime path detection is fragile under
`source()` vs `Rscript` vs RStudio click-run, and statisticians
without a CS background find the resulting code hard to read and
debug.

The contract: the runner assumes **the R session's cwd is the
project root** -- the directory that contains BOTH `AI4BayesCode/` AND
the `<folder>/` where the generator wrote the `.cpp`. If the user
invokes R from anywhere else, they get a clear filesystem error
(`Could not find expected AI4BayesCode directory`) and the fix is
obvious: `setwd()` to the project root, or edit the two paths.

Emit this header at the top of every generated runner, filling
in `<folder>` and `<ClassName>` from your generation context:

```r
# ============================================================================
# run_<ClassName>.R
#
# DEFAULT (installed package): load the package and compile the .cpp with a
# bare RELATIVE path. Run this script from the directory that contains the
# `<folder>/` where the .cpp lives (via `Rscript` or R's `source()`). If
# you're elsewhere, `setwd()` there or edit the path below. NO local
# AI4BayesCode checkout and NO helper sourcing are needed.
# ============================================================================

library(AI4BayesCode)
ai4bayescode_source("<folder>/<ClassName>.cpp")   # relative path; no AI4BayesCode_path=

# ===========================================================================
#  Constructor arguments for <ClassName>
# ===========================================================================
#  new(<ClassName>,
#      <arg1>,        # <type> -- <description>
#      <arg2>,        # <type> -- <description>
#      ...
#      seed           # int   -- RNG seed (0 = random)
#  )
#
#  Methods:
#    $step(n)          -- run n Gibbs sweeps
#    $get_current()    -- named list of current parameter draws
#    $set_current(lst) -- overwrite parameters (partial OK)
# ===========================================================================

# ---- Helper: run ONE chain with history, return (hist, pp, wall_sec) -----
# The runner uses keep_history = TRUE so validator Layer 3 (R1/R2/R3) can
# work on the stored draws. Wall-clock time is captured for the timing line.
#
# HARD RULE (naming) -- everything is named after the model, i.e. the
# Rcpp class `<ClassName>` (the same identifier used in
# `new(<ClassName>, ...)` and `<ClassName>.cpp`). NO snake_case stems
# anywhere. The HARNESS-INTERNAL chain helper is
# `run_chain_<ClassName>()` (e.g. `run_chain_SpikeSlabLaplace` for
# `new(SpikeSlabLaplace, ...)`); the delivered files are
# `run_<ClassName>.R` / `example_<ClassName>.R` -- all consistent with
# the existing `<ClassName>.cpp` convention. NOT bare `run_chain()`.
# Rationale: a runner or example may be sourced alongside others in
# one R session; model-specific function names prevent a later
# source() from clobbering an earlier one.
# All call sites below use the same `run_chain_<ClassName>` name.
# REMINDER: run_chain_<ClassName>() exists ONLY for the throwaway
# Layer-3 harness + sim1 (they need pp = predict_at() collected in the
# worker); it is NEVER shipped. The delivered example defines NO helper
# functions at all -- it drives chains via ai4bayescode_run_chains with
# an INLINE function(seed) constructor closure.
#
# HARD RULE -- ALL data inputs MUST be function parameters of
# run_chain_<ClassName>, NOT global-scope variables read from the
# enclosing R session. Replace `<data_args>` below with the actual
# argument list (e.g., `y, X` or `y, X, group, J`) -- same names that
# appear in the constructor's argument list. Do NOT close over
# `y_obs`, `X_obs`, etc. from the script's outer scope.
#
# Why this matters:
#   * sim1 cross-impl tests call run_chain_<ClassName> with a NEW
#     dataset per replicate (`worker(r) -> simulate_data_with_truth(r)
#     -> run_chain_<ClassName>(...)`). A helper that closes over
#     `y_obs` in the enclosing scope picks up the WRONG dataset on
#     every replicate after the first.
#   * Users running predictions on held-out data want
#     `run_chain_<ClassName>(y_holdout, X_holdout, ...)` to work
#     without rewriting the function body.
#   * The function becomes self-contained and unit-testable.
#
# WRONG (data closed over from outer scope, and bare name):
#     run_chain <- function(seed, n_burnin, n_keep) {            # no data args, bad name
#         model <- new(<ClassName>, y_obs, X_obs,                 # outer-scope grab
#                      as.integer(seed), TRUE)
#         ...
#     }
#
# RIGHT (data passed in; model-specific name):
run_chain_<ClassName> <- function(<data_args>, seed, n_burnin, n_keep, diagnosis = FALSE) {
    model <- new(<ClassName>, <data_args>, as.integer(seed), TRUE)
    # Kernel-control (freeze/unfreeze): must be re-issued per worker.
    # run_chain constructs a fresh model per chain (via foreach %dopar%);
    # freeze state does NOT auto-propagate from the outer scope. If you
    # want to pin sigma (probit) or hold any block, add the calls HERE:
    #   model$set_current(list(sigma = 1))
    #   model$freeze("sigma")
    # Then also set `frozen_names <- c("sigma")` at runner scope for R2.f
    # exclusion. See validator.md Sec.R2.f
    t0 <- Sys.time()
    model$step(n_burnin)
    model$step(n_keep)
    t1 <- Sys.time()
    # Both `get_history()` and `predict_at()` return per-iteration draws
    # for every iteration stepped (burn-in + keep). Slice off burn-in
    # inside the runner so downstream code sees only the kept draws.
    keep_idx <- (n_burnin + 1L):(n_burnin + n_keep)
    slice <- function(lst) lapply(lst, function(x) {
        if (is.null(dim(x))) x[keep_idx]
        else x[keep_idx, , drop = FALSE]
    })
    out <- list(hist     = slice(model$get_history()),
                pp       = slice(model$predict_at(list())),  # no-input posterior predictive
                wall_sec = as.numeric(difftime(t1, t0, units = "secs")))
    # diagnosis = TRUE attaches model-INDEPENDENT posterior diagnostics via the
    # SHIPPED library function ai4bayescode_diagnose() -- do NOT reimplement:
    #   out$diagnosis      -> per-parameter table (R-hat / ESS / MCSE / mean / sd /
    #                         median / 90% CI)
    #   out$diagnosis_plot -> trace + autocorrelation + density plot
    if (isTRUE(diagnosis)) {
        dg <- ai4bayescode_diagnose(out$hist)
        out$diagnosis      <- dg$summary
        out$diagnosis_plot <- dg$plot
    }
    out
}

# ======================================================================
#  HARD RULE -- the `diagnosis = TRUE` path is NON-NEGOTIABLE
# ======================================================================
# The diagnostics AND the plot are a SHIPPED function `ai4bayescode_diagnose()` -- you do
# NOT write them. It is provided by `library(AI4BayesCode)`; call it UNqualified
# (no `AI4BayesCode::`).
# `ai4bayescode_diagnose(<named list of draws>)` returns
#   list(summary = <per-param R-hat / ESS / MCSE / mean / sd / median / 90% CI
#                   table, via posterior>,
#        plot    = <trace + autocorrelation + density, via bayesplot, with a
#                   base-R fallback>)
# (PSIS-LOO is NOT included -- it needs a model-specific pointwise
# log-likelihood, codegen_cpp.md Sec.6a.)
#
# Every generated runner MUST:
#   (1) take a `diagnosis = FALSE` argument;
#   (2) when diagnosis = TRUE, CALL ai4bayescode_diagnose() and attach
#       BOTH out$diagnosis (= dg$summary) AND out$diagnosis_plot (= dg$plot).
#
# Do NOT reimplement the diagnostics or the plot inline. The whole point of
# shipping ai4bayescode_diagnose() is that the runner stops re-emitting it (earlier
# generations kept rewriting a summary-only helper and DROPPING the plot).
# This is INDEPENDENT of how the runner collects draws: pass WHATEVER named
# list of kept draws you built, e.g.
#   ai4bayescode_diagnose(list(beta = beta, sigma = sigma)).
# The contract does not depend on the field being called `hist`.
#
# FORBIDDEN -- an inline summary-only diagnosis that drops the plot:
#   ## WRONG: no plot, renamed field, library function not used
#   if (diagnosis && requireNamespace("posterior", quietly = TRUE)) {
#       out$summary <- posterior::summarise_draws(...)
#   }
# ALWAYS route through ai4bayescode_diagnose() and expose
# out$diagnosis + out$diagnosis_plot.
# ======================================================================

# ---- Pointwise log-likelihood for LOO (see codegen_cpp.md Sec.6a templates) ----
# EMIT THE ROW THAT MATCHES THE MODEL'S OBSERVATION LIKELIHOOD.
# Example below is Gaussian. Replace with the family-specific body.
pointwise_loglik <- function(hist, y) {
    # ...see codegen_cpp.md Sec.6a per-observation-family templates...
}

# ---- Configuration ----
# Data lives at script scope ONLY so the user can edit it without touching
# the function body. It is PASSED INTO run_chain as arguments -- never read
# from inside run_chain. Replace the placeholders with concrete loaders
# (`read.csv`, `data.frame(...)`, etc.). The variable names below
# (`y_obs`, `X_obs`, ...) are conventions for the script scope only;
# they do NOT appear inside run_chain's body -- only as arguments at the
# call sites.
n_burnin <- 4000L
n_keep   <- 4000L
y_obs    <- <user's response vector>
# X_obs   <- <user's design matrix>     # uncomment + add as needed
# group   <- <length-N integer vector>  # uncomment + add as needed
# ... add one variable per data input the constructor takes

# ---- Flip if this runner uses joint_nuts_block (tightens R3 threshold) ---
USES_JOINT_NUTS <- FALSE

# ---- Run two chains ------------------------------------------------------
# Pass every data variable from the Configuration block above as a
# function argument to run_chain_<ClassName>. The first positional
# args MUST match the order of `<data_args>` in its signature.
# Common patterns:
#   run_chain_<ClassName>(y_obs, X_obs, seed = 101L, n_burnin, n_keep)
#   run_chain_<ClassName>(y_obs, X_obs, group, J, seed = 101L, n_burnin, n_keep)
#   run_chain_<ClassName>(Y_obs, seed = 101L, n_burnin, n_keep)   # response-only
c1 <- run_chain_<ClassName>(<data_args>, seed = 101L, n_burnin, n_keep)
c2 <- run_chain_<ClassName>(<data_args>, seed = 202L, n_burnin, n_keep)

# Helpers
pack2 <- function(x1, x2) {
    if (is.null(dim(x1))) {
        array(cbind(x1, x2), dim = c(length(x1), 2, 1))
    } else {
        arr <- array(NA_real_, dim = c(nrow(x1), 2, ncol(x1)))
        arr[, 1, ] <- x1; arr[, 2, ] <- x2
        arr
    }
}

# ---- Layer 3 R1: smoke test ----------------------------------------------
# `c1$hist` / `c2$hist` are already burn-in-stripped (sliced inside
# run_chain) so checks below only see kept draws.
stopifnot(all(sapply(c1$hist, function(x) all(is.finite(x)))))
stopifnot(all(sapply(c2$hist, function(x) all(is.finite(x)))))

# ---- Layer 3 R2: 2-chain R-hat + ESS (rank-normalized), Stage-1 -> Stage-2 ---
# Matches validator.md Sec.R2: R-hat is a HARD gate (< 1.05); ESS is a SOFT
# criterion via ess_ratio = min(ESS)/n_keep -- >= 0.01 silent, [0.005,0.01) WARN
# and proceed, < 0.005 escalate. A slow-but-correct model gets the 20k+20k budget
# BEFORE being declared a failure. Do NOT hard-fail ESS at the 4k stage.
suppressPackageStartupMessages(library(posterior))

r2_diag <- function(h1, h2, n_keep_used) {
    worst_rhat <- 0; worst_ess_ratio <- Inf
    for (nm in names(h1)) {
        arr <- pack2(h1[[nm]], h2[[nm]])
        rh  <- apply(arr, 3, posterior::rhat)
        eb  <- apply(arr, 3, posterior::ess_bulk)
        et  <- apply(arr, 3, posterior::ess_tail)
        ess_ratio <- min(eb, et, na.rm = TRUE) / n_keep_used
        cat(sprintf("  %-14s  max Rhat=%.3f  min ESS=%.0f  ess_ratio=%.4f\n",
                    nm, max(rh, na.rm = TRUE),
                    min(eb, et, na.rm = TRUE), ess_ratio))
        worst_rhat      <- max(worst_rhat, max(rh, na.rm = TRUE))
        worst_ess_ratio <- min(worst_ess_ratio, ess_ratio)
    }
    list(rhat = worst_rhat, ess_ratio = worst_ess_ratio)
}

# R2.f: exclude frozen params (kernel-control) from R-hat / ESS aggregation.
# See validator.md Sec.R2.f. Frozen params appear as constant columns in
# get_history() (composite appends a repeat every skipped step to preserve
# shape); their R-hat is undefined (NaN). Excluding them keeps the diagnostic
# honest and logs the exclusion for user verification.
# NOTE: if run_chain_<ClassName>() calls model$freeze(...) inside the worker,
# set `frozen_names` here to the same character vector so R2.f can filter.
# For a default (no-freeze) runner, this is character(0) and a no-op.
frozen_names <- character(0)   # <-- set to c("sigma", ...) if run_chain freezes
apply_frozen_filter <- function(h) {
    if (length(frozen_names) == 0L) h else h[setdiff(names(h), frozen_names)]
}
if (length(frozen_names) > 0L) {
    cat(sprintf("[R2.f] Excluding %d frozen param(s) from R2: %s\n",
                length(frozen_names), paste(frozen_names, collapse = ", ")))
}

d <- r2_diag(apply_frozen_filter(c1$hist), apply_frozen_filter(c2$hist), n_keep)

# Stage-2 escalation (within-attempt): re-run at 20k+20k if Stage-1 (4k+4k) shows
# max R-hat >= 1.05 OR a severely low ess_ratio (< 0.005), then recompute.
if (d$rhat >= 1.05 || d$ess_ratio < 0.005) {
    cat("  [R2] Stage-1 inadequate -> escalating to 20k+20k ...\n")
    n_burnin <- 20000L; n_keep <- 20000L
    c1 <- run_chain_<ClassName>(<data_args>, seed = 101L, n_burnin, n_keep)
    c2 <- run_chain_<ClassName>(<data_args>, seed = 202L, n_burnin, n_keep)
    stopifnot(all(sapply(c1$hist, function(x) all(is.finite(x)))))
    stopifnot(all(sapply(c2$hist, function(x) all(is.finite(x)))))
    d <- r2_diag(apply_frozen_filter(c1$hist), apply_frozen_filter(c2$hist), n_keep)
}

# Final gates: R-hat HARD; ESS SOFT (only ess_ratio < 0.005 at the escalated
# budget is a FAILURE; [0.005, 0.01) only warns -- legitimate for slow GP/hier).
stopifnot(d$rhat < 1.05)
if (d$ess_ratio < 0.005) {
    stop(sprintf("R2 FAIL: worst ess_ratio %.4f < 0.005 even at the escalated budget",
                 d$ess_ratio))
} else if (d$ess_ratio < 0.01) {
    warning(sprintf("R2: worst ess_ratio %.4f in [0.005, 0.01) -- model mixes slowly, proceeding",
                    d$ess_ratio))
}

# ---- MANDATORY validator contract line (R-hat gate) ---------------------
# The generation harness greps stdout for exactly this token. `d$rhat` is the
# worst rank-normalized R-hat from r2_diag() above. Emit it verbatim.
max_rhat <- d$rhat
# The PASS/FAIL token uses the DEFAULT HARD gate = 1.05 (matches the stopifnot
# above and validator.md Sec.R2). It MUST match the gate: a stricter 1.01 threshold
# here would print FAIL while the run continues at 1.05 -- the confusing
# "FAIL-but-SUCCESS" seen in practice. Keep the token text EXACT (the harness
# greps for it).
if (max_rhat < 1.05) cat("AI4BAYES_VALIDATE: PASS\n") else cat(sprintf("AI4BAYES_VALIDATE: FAIL maxRhat=%.4f\n", max_rhat))
# R-hat GRAY ZONE (1.01, 1.05]: the run PASSES the default gate but not the strict
# Vehtari-2021 1.01 bar. Do NOT auto-escalate. Instead follow start.md's "R-hat
# gray zone -- opt-in stricter convergence" protocol: surface the structured
# question (a: ship as-is at 1.05 [default] / b: extend chain budget 20k->40k->80k
# toward <1.01 / c: AI-proposed structural fix / d: other). A heavy-tailed
# hierarchical variance (InvGamma) can legitimately sit at 1.01-1.02 even when the
# posterior is CORRECT (predictive matches a Stan/reference cross-check) -- there,
# option (b) or a higher joint_nuts `max_tree_depth` (10-12) is the remedy, NOT a
# different model. Emitting this line is what SIGNALS the gray zone to the agent.
if (max_rhat >= 1.01)
    cat(sprintf("  [convergence] gray zone: PASS at the default 1.05 gate, strict 1.01 NOT met (maxRhat=%.4f) -> apply start.md's opt-in stricter-convergence question.\n", max_rhat))

# ---- Layer 3 R3: posterior predictive p-values + PSIS-LOO ---------------
suppressPackageStartupMessages(library(loo))

# R3.a Bayesian p-values. This IS a gate (unlike R3.b PSIS-LOO below), but
# only on the CENTRAL statistics: a posterior-predictive p-value is
# ~Uniform(0, 1) even for a perfectly-sampled, CORRECTLY-specified model, and
# the order statistics (min / max) are legitimately extreme, so gating on
# "any statistic outside" would fail correct samplers. Gate rule: min / max are
# printed only; among the CENTRAL statistics (mean, sd, q25, q75) one outside
# is a WARN and TWO OR MORE simultaneously FAIL R3 (validator.md R3.a).
# The statistic SET depends on the support of y (validator.md R3.a). For a
# BINARY y every replicate's q25 and q75 are 0 or 1, so both p-values pin at
# 0 or 1 and two central statistics land outside -- failing a demonstrably
# correct sampler. Pick the set from the data, not a fixed list.
.y_uniq <- unique(as.numeric(y_obs))
bp_stat <- if (all(.y_uniq %in% c(0, 1))) {
    list(mean = mean)                                   # binary
} else if (all(.y_uniq >= 0 & .y_uniq == floor(.y_uniq))) {
    list(mean = mean, sd = sd, max = max)               # count
} else {
    list(mean = mean, sd = sd, min = min, max = max,    # continuous
         q25 = function(x) quantile(x, 0.25, names = FALSE),
         q75 = function(x) quantile(x, 0.75, names = FALSE))
}
pv <- sapply(bp_stat, function(f)
    mean(apply(c1$pp$y_rep, 1, f) >= f(y_obs)))
cat("\n  Bayesian p-values: ",
    paste(sprintf("%s=%.2f", names(pv), pv), collapse = "  "), "\n")
# Gate on the CENTRAL statistics only; min / max stay diagnostic.
pv_lo   <- if (USES_JOINT_NUTS) 0.10 else 0.05
pv_hi   <- 1 - pv_lo
.central <- pv[intersect(names(pv), c("mean", "sd", "q25", "q75"))]
.n_out   <- sum(.central <= pv_lo | .central >= pv_hi)
# Verdict scales with HOW MANY central statistics this support gives us.
# With several (continuous / count), one outside is expected under a correct
# model -- each has ~2*pv_lo chance -- so one is a WARN and two or more FAIL.
# With exactly ONE (binary y -> `mean` only) there is no such slack: keeping
# the ">= 2 to fail" rule would make R3.a unfailable for every binary model.
.fail_at <- if (length(.central) >= 2L) 2L else 1L
if (.n_out > 0L && .n_out < .fail_at)
    warning(sprintf("[R3.a WARN] one central Bayesian p-value outside (%.2f, %.2f): %s",
                    pv_lo, pv_hi, paste(sprintf("%s=%.3f", names(.central), .central), collapse = ", ")))
stopifnot(.n_out < .fail_at)

# R3.b PSIS-LOO (DIAGNOSTIC ONLY -- does NOT fail R3).
# Pareto-k_hat measures LOO importance-weight reliability, NOT sampler
# correctness. GP latent-variable and hierarchical-latent models are
# known to fail this diagnostic even when the posterior is correctly
# sampled (Vehtari, Simpson, Gelman, Yao, Gabry, JMLR 2024,
# arXiv:1507.02646). Sampler correctness is gated by R-hat (R2) AND the
# Bayesian p-values (R3.a, the stopifnot above). R3.b is diagnostic only --
# recorded and warned, never a gate.
LL1 <- pointwise_loglik(c1$hist, y_obs)
LL2 <- pointwise_loglik(c2$hist, y_obs)
LLarr <- array(NA_real_, dim = c(nrow(LL1), 2, ncol(LL1)))
LLarr[, 1, ] <- LL1; LLarr[, 2, ] <- LL2
rel_n_eff <- loo::relative_eff(exp(LLarr))
lo <- loo::loo(LLarr, r_eff = rel_n_eff, cores = 1)
pct_k_lo <- mean(lo$diagnostics$pareto_k <  0.5) * 100
pct_k_hi <- mean(lo$diagnostics$pareto_k >= 0.7) * 100
cat(sprintf("  LOO elpd=%.1f (se=%.1f)  pct_k<0.5=%.1f%%  pct_k>=0.7=%.1f%%\n",
            lo$estimates["elpd_loo", "Estimate"],
            lo$estimates["elpd_loo", "SE"],
            pct_k_lo, pct_k_hi))
if (pct_k_lo < 50 || pct_k_hi >= 10) {
    warning(sprintf(
        "[R3.b] PSIS-LOO Pareto-k DIAGNOSTIC ONLY (NOT a failure): %.1f%% k<0.5, %.1f%% k>=0.7. Pareto-k indicates LOO importance-weight reliability, NOT sampler correctness; GP and hierarchical-latent models routinely fail this diagnostic with correctly sampled posteriors. See Vehtari et al. (JMLR 2024).",
        pct_k_lo, pct_k_hi))
}

# ---- Post-run performance hint (see codegen_cpp.md Sec.4a) -----------------
# Wall time actually spent sampling. run_chain_<ClassName>() returns
# wall_sec per chain and the chains here run SEQUENTIALLY, so the sum is
# the elapsed time. (If you switch the harness to parallel workers,
# measure the elapsed time around the parallel block instead -- summing
# per-chain times would then over-count.)
total_wall_sec <- c1$wall_sec + c2$wall_sec
n_sweeps_total <- 2L * (n_burnin + n_keep)
message(sprintf("[timing] %.1fs across %d sweeps (%.3fs / sweep)",
                total_wall_sec, n_sweeps_total,
                total_wall_sec / n_sweeps_total))

# ---- (Optional) Predict at new data ------------------------------------
# pred <- c1 model-call equivalent at new X:
# model <- new(<ClassName>, <data_args>, 42L, TRUE)
# model$step(n_burnin); model$step(n_keep)
# pred <- model$predict_at(list(X = X_test))  # see wrapper's predict_at doc
```

### Modular NUTS in composite -- periodic readapt schedule (CONDITIONAL)

**When this section applies.** The wrapper's composite contains any
NUTS-family child (`nuts_block`, `joint_nuts_block`)
AND that child samples a parameter whose
**conditional posterior shifts across outer Gibbs iterations** (typical
when sigma^2, other regression coefficients, hyperparameters, or
RJMCMC inclusion indicators update in a sibling block). If the
composite is pure-NUTS-on-a-fixed-conditional (a single
`joint_nuts_block` sampling everything jointly with no Gibbs siblings),
use the standard `run_chain_<ClassName>` template above instead -- the
initial warmup metric stays correct and a periodic schedule only adds
overhead.

**Why this is needed.** `nuts_block`'s initial-call warmup adapts the
metric (mass matrix + step size + dual averaging) for the conditional
posterior **at that time**. In a modular composite where sibling
blocks shift the conditional between outer iters, the persistent
metric becomes mis-tuned for the new conditional. This manifests as
the **stuck-fast pattern**: R-hat above 2, ESS in single digits, and
the chain values bit-identical across hundreds of `step()` calls. The
shipped solution is to periodically call `readapt_NUTS(n)` between
`step()` chunks -- this re-tunes the metric WITHOUT advancing chain
state.

**Defaults.** `readapt_every = 500L` outer iters,
`readapt_n = 50L` re-adapt iters per call (~10% overhead). Tighten
to `readapt_every = 100L` if R-hat fails at 4k+4k; loosen to
`readapt_every = 1000L` if the chain mixes well and wall is critical.
There is no published canonical value for modular NUTS-in-composite
re-adaptation frequency -- standard frameworks (Stan, PyMC, NumPyro)
use single-warmup + frozen-metric and do not address this regime; the
defaults here are empirically calibrated.

**Replacement `run_chain_<ClassName>` body** (replaces the two
`model$step(...)` calls in the standard template above; everything
else -- wall timing, slicing burn-in, `get_history()`, `predict_at()`,
return shape -- is identical):

```r
run_chain_<ClassName> <- function(<data_args>, seed, n_burnin, n_keep,
                                  readapt_every = 500L,
                                  readapt_n     = 50L) {
    model <- new(<ClassName>, <data_args>, as.integer(seed), TRUE)
    t0 <- Sys.time()
    # Periodic readapt schedule -- covers BOTH burn-in and keep, since the
    # conditional keeps shifting throughout sampling under Gibbs siblings.
    total <- as.integer(n_burnin + n_keep)
    full  <- total %/% readapt_every
    for (i in seq_len(full)) {
        model$readapt_NUTS(readapt_n, FALSE, -1L)
        model$step(as.integer(readapt_every))
    }
    remainder <- total - full * readapt_every
    if (remainder > 0L) {
        model$readapt_NUTS(readapt_n, FALSE, -1L)
        model$step(as.integer(remainder))
    }
    t1 <- Sys.time()
    keep_idx <- (n_burnin + 1L):(n_burnin + n_keep)
    slice <- function(lst) lapply(lst, function(x) {
        if (is.null(dim(x))) x[keep_idx]
        else x[keep_idx, , drop = FALSE]
    })
    list(hist     = slice(model$get_history()),
         pp       = slice(model$predict_at(list())),
         wall_sec = as.numeric(difftime(t1, t0, units = "secs")))
}
```

`readapt_every` and `readapt_n` are exposed as `run_chain_<ClassName>`
arguments (not script-scope globals) so the helper stays
self-contained: a sim1 / cross-impl harness can vary them per
replicate without touching the function body, and a downstream user
can copy-paste the helper into their own analysis without inheriting
hidden state.

### Special case: per-step outputs NOT in `get_history()`

**Default assumption** (what the template above bakes in): every
parameter the user cares about lives in `get_history()`. Most blocks
honour this. A few **do not** -- they expose certain per-step states
ONLY through `get_current()`, never through history, because storing
the state every step under `keep_history = TRUE` would dominate
memory (see `system_design.md` Sec.9 "memory grows linearly with
iteration count").

**Codegen LLMs MUST NOT hallucinate history fields.** Before
emitting `hist$<key>` in a generated runner, verify that `<key>` is
in the block's documented `get_history()` output. If the block's
header lists a field only under `current_named_outputs()` /
`get_current()` (not under `get_history()`), the runner MUST route
that field through `get_current()` per step -- NOT pretend it lives
in `hist`. Hallucinating non-existent history keys is the bug class
that motivated this section (a generated `water_bif_runner.R` once
referenced `hist$order_sampled_DAG`, which does not exist -- the
runner compiled but the dependent function would crash at first
call).

**Known cases (v1.2):**

| Block | Field exposed only via `get_current()` | Per-step size | Why history-omitted |
|---|---|---|---|
| `order_mcmc_block` | `sampled_DAG` (p x p adjacency) | p^2 ints / step | p=20, T=40000 ~= 130 MB; p=64 (block ceiling) ~= 1.3 GB. Most users only need `order_log_score` R-hat or edge marginals (computable from `get_current()` on demand). |

`order_mcmc_block::get_history()` (verbatim) returns ONLY:

- `order` (T x p)
- `order_log_score` (T x 1)
- `y_rep` (T x N*p)   <- only when the Tier-A wrapper adds a `y_rep`
   refresher; for a bare `order_mcmc_block` it is absent.

The sampled DAG is **NEVER** in history.

**Canonical pattern for collecting per-draw DAGs** -- used in the v1.2
`examples/OrderMCMCBN.cpp` companion runner:

```r
# Inside run_chain_<ClassName>() -- collect DAGs in a step-by-step
# loop AFTER burn-in, while still letting get_history() track the
# cheap state (order + log_score + y_rep).
model$step(n_burnin)
p_var <- ncol(data_obs)
dags  <- array(0L, dim = c(as.integer(n_keep), p_var, p_var))
for (s in seq_len(as.integer(n_keep))) {
    model$step(1L)
    # sampled_DAG[i, j] = 1 iff j is parent of i (row-major).
    dags[s, , ] <- as.matrix(model$get_current()$sampled_DAG)
}
list(hist     = slice(model$get_history()),
     dags     = dags,                                     # NEW: per-step DAGs
     wall_sec = as.numeric(difftime(Sys.time(), t0, units = "secs")))
```

The `dags` field is at top-level (parallel to `hist`), NOT inside
`hist`, so downstream code cannot mistake an R-side collection for
a cpp-side history field.

**Performance.** Step-by-step adds one Rcpp boundary crossing per
iteration (~1-10 mus) plus an n^2-int memcpy. For p=20, T=40000 that
is < 1 s extra wall time and ~= 130 MB memory. Document the memory
cost in the wrapper header so users don't OOM themselves at larger
p.

**Extending this table.** Whenever a future block exposes a heavy
per-step-only output (per-step BART tree ensemble, per-step
HMM-forward probability cube, etc.), add it here AND cross-reference
the entry in the Tier-A wrapper's header comment.

---

**Concrete example** -- what the constructor block looks like for a BART
model:

```r
# ===========================================================================
#  Constructor arguments for BartNoise
# ===========================================================================
#  new(BartNoise,
#      X,              # NumericMatrix (N x p) -- predictor matrix
#      y,              # NumericVector (N)     -- response vector
#      ntrees,         # int                  -- number of trees (default 200)
#      seed,           # int                  -- RNG seed (0 = random)
#      keep_history    # logical              -- record every draw (default FALSE)
#  )
#  NOTE: sigma is initialized internally to bart_model's OLS-based sigest.
#        If you want an overdispersed start for R-hat diagnostics, call
#        model$set_current(list(sigma = ...)) AFTER construction.
#
#  Methods:
#    $step(n)           -- run n Gibbs sweeps (one BART + one NUTS-sigma)
#    $get_current()     -- list(f_bart = <N-vec>, sigma = <scalar>)
#    $set_current(lst)  -- overwrite sigma; f_bart is read-only
#
#  Notes:
#    - sigma prior: InverseGamma calibrated from OLS residuals (automatic)
#    - BART uses R's RNG; call set.seed() before new() for reproducibility
#    - ntrees=200 is the BART default; more trees = smoother f, slower
# ===========================================================================
#
#  run_chain_<ClassName>() returns:
#    $f_bart  -- matrix (n_keep x N) of posterior draws of f
#    $sigma   -- vector (n_keep) of posterior draws of sigma
# ===========================================================================
```

The constructor block must list ALL arguments the user can pass, their
types, and brief descriptions. If hyperparameters are exposed as
constructor arguments (see `codegen.md Sec.2` and `codegen_priors.md`),
document those too with their defaults.

The `run_chain_<ClassName>()` helper must always return a **named list
of posterior
draws**, where scalar parameters are `numeric(n_keep)` vectors and
vector parameters are `matrix(n_keep, dim)` matrices. Never return a
list-of-lists that the user has to manually aggregate.

Multi-chain R-hat in the DELIVERED example is always the shipped pair
`ai4bayescode_run_chains(function(seed) new(<ClassName>, ...), n_chains = ...)`
+ `ai4bayescode_rhat_summary(run)` (histories are keep-only; run_chains
strips the warmup draws) -- never a hand-rolled loop or a
hand-assembled `posterior::rhat` call.
Over-dispersed initial values, when needed, go through `set_current()`
inside the inline constructor closure. (The Layer-3 harness computes its R2
R-hat with the validator.md recipe -- that stays harness-internal.)

---

## 9a. Model-specific R-side preprocessing -- SoftBart `sigma_hat` recipe

When the wrapper is `SoftBartNoise` (or any future SoftBart-derived
example), the R-side runner MUST compute `sigma_hat` exactly as the
SoftBart R package does (Linero & Yang 2018) and pass it to the C++
constructor. The C++ kernel (`bart_pure_cpp/src/softbart_model.h`) accepts
`sigma_hat` but its default of `-1.0` falls back to `sd(Y)`, which is
the wrong scale for the half-Cauchy prior `sigma ~ Cauchy_+(0, sigma-hat)`.

### Why R-side, not C++ side

Computing sigma-hat requires `glmnet::cv.glmnet`. Putting that call in the C++
constructor would force a Tier-C -> R callback (architectural smell) and
would block a future Python port. The SoftBart R package itself does
this in R; mirror that boundary.

### The exact procedure (paper-faithful, package-faithful)

Mirrors `Hypers()` + `GetSigma()` from the upstream SoftBART R
package (Linero & Yang 2018; github.com/theodds/SoftBART). Three
steps in this order:

```r
library(glmnet)

# 1. Min-max normalize Y to [-0.5, 0.5]. Save (a, b) for un-normalizing
#    predictions on the way out.
softbart_normalize_Y <- function(y) {
    a <- min(y); b <- max(y)
    list(y_norm = (y - a) / (b - a) - 0.5, a = a, b = b)
}
softbart_unnormalize <- function(z, a, b) (z + 0.5) * (b - a) + a

# 2. Compute sigma_hat via cv.glmnet on the NORMALIZED Y. The package
#    uses lambda.1se (more regularized than lambda.min) and the in-sample
#    RMSE at that lambda -- not lambda.min, not df-adjusted sd. These
#    specifics matter; the package's prior calibration depends on both.
softbart_sigma_hat <- function(X, y_norm, weights = rep(1, length(y_norm))) {
    stopifnot(is.matrix(X) || is.data.frame(X))
    if (is.data.frame(X)) X <- model.matrix(~ . - 1, data = X)
    fit       <- glmnet::cv.glmnet(x = X, y = y_norm, alpha = 1,
                                   weights = weights)
    fitted    <- as.numeric(predict(fit, newx = X, s = "lambda.1se"))
    sqrt(mean((fitted - y_norm)^2))     # RMSE, NOT sd()
}

# 3. Construct SoftBartNoise with the lasso-derived sigma_hat.
norm_info <- softbart_normalize_Y(y)
sigma_hat <- softbart_sigma_hat(X, norm_info$y_norm)
m <- new(SoftBartNoise, X, norm_info$y_norm,
         /* ntrees */ 50L,
         /* k     */ 2.0,
         /* sigma_hat */ sigma_hat,
         /* ... other hypers per the wrapper signature ... */,
         /* seed */ 1L,
         /* keep_history */ TRUE)

# 4. After MCMC, un-normalize any prediction / posterior summary the
#    user sees on the natural Y scale.
y_rep_norm <- m$predict_at(...)$y_rep   # on [-0.5, 0.5]
y_rep      <- softbart_unnormalize(y_rep_norm, norm_info$a, norm_info$b)
```

### Hard rules for SoftBart R-side runners

1. **NEVER** let `sigma_hat` fall back to the C++ default. Always
   compute the lasso sigma-hat in R and pass it explicitly.
2. **`s = "lambda.1se"`** (the default in SoftBart's `predict_glmnet`),
   not `"lambda.min"`. The 1se rule gives a sparser, more regularized
   fit; sigma-hat is therefore larger and the half-Cauchy prior is wider --
   matching the package's calibration.
3. **`sqrt(mean(residuals^2))`** (denominator N), not `sd(residuals)`
   (denominator N-1). Small but real difference; the package uses RMSE.
4. **`alpha = 1`** (lasso, not ridge or elastic net).
5. **Apply `normalize_bart(Y)` BEFORE `cv.glmnet`**, not after. sigma-hat is
   computed on the [-0.5, 0.5]-scaled response so it's on the same
   scale the C++ kernel sees during MCMC. Save `(a, b) = (min(y), max(y))`
   so predictions can be un-normalized.
6. **Un-normalize on the way out.** Posterior-predictive `y_rep`,
   posterior mean of `f(X)`, etc. are on the normalized scale inside
   the C++ kernel. The wrapper / runner is responsible for inverse-
   transforming via `(z + 0.5) * (b - a) + a` before reporting.
7. **`weights = rep(1, length(y))`** by default. If the wrapper
   eventually supports heteroscedastic weights, pass them through to
   `cv.glmnet` as well so the sigma-hat matches the kernel's weighted draw.

### Python port note

When the Python version is built, the same 3-step recipe applies on the
Python side via `sklearn.linear_model.LassoCV` (cross-validated lasso
with default 1-SE rule via `selection='1se'` not natively available;
implement the 1-SE selection manually from `cv.glmnet`-style mean +/-
SE per lambda). The C++ kernel side does not change.

### Status

`SoftBartNoise` ships (see `examples/SoftBartNoise.cpp`); its R runner
follows this recipe verbatim (softBART noise has a different prior
than BART, so the preprocessing differs). No other shipped example
uses this preprocessing.

---

## 10. Reference templates

Modular (codegen default):
- `examples/GaussianLocationScale.cpp` -- real + positive blocks (Jeffreys sigma)
- `examples/BetaBernoulli.cpp` -- interval block (binary y_rep refresher)
- `examples/DirichletSimplex.cpp` -- simplex block (Multinomial y_rep)
- `examples/DirichletSparse.cpp` -- simplex + positive (Multinomial y_rep)
- `examples/DirichletHierarchical.cpp` -- 3-block hierarchical (Dirichlet y_rep)
- `examples/BartNoise.cpp` -- Gaussian BART (CRAN BART R package backfitting) + slice on sigma (Gaussian y_rep)
- `examples/GBartPoisson.cpp` -- genBART Poisson regression via RJMCMC
  (Linero 2022).
- `examples/GBartLogistic.cpp` -- genBART Bernoulli classification via
  `logistic_lik` directly (no augmentation needed).
- `examples/GBartMultinomial.cpp` -- genBART multi-class logistic BART
  via C-1 coupled `genbart_block(poisson_lik, offset=log_phi_aug)` +
  `poisson_multinomial_aug_block`. Handles C >= 2; for binary prefer
  the simpler `GBartLogistic`.
- `examples/GBartHeteroscedastic.cpp` -- genBART heteroscedastic Normal
  with mean = variance = exp(r(x)). Linero 2022 Sec.4.2 showcase.
- For genBART variants NOT shipped as standalone examples (negative
  binomial, AFT survival, beta, gamma-shape, beta-binomial), compose
  from `examples/GBartPoisson.cpp` as the skeleton + the matching
  `genbart::lik::*` header under `bart_pure_cpp/src/GENBART/likelihoods/`.
  The remaining wrapper code (constructor, set_current dispatcher,
  predict_at, RCPP_MODULE) is structurally identical; only the
  likelihood subclass and any nuisance-parameter NUTS block change.
- `examples/ARDLasso.cpp` -- **LEGACY** self-contained conjugate Gibbs;
  Check #17 exception documented (don't follow this pattern for new code)
- `examples/SpikeSlabRJMCMC.cpp` -- Dirac spike-and-slab via rjmcmc_block +
  double-Jeffreys + sigma^2tau^2 Ishwaran-Rao slab (Class 2b reference template)
- `examples/LogisticRegression.cpp` -- Bayesian logistic regression via
  `pg_logistic_block` (Polya-Gamma augmentation; linear predictor only)
- `examples/HMMGaussian2State.cpp` -- 2-state Gaussian-emission HMM via
  `hmm_block` (forward-backward)
- `examples/GPRegression.cpp` -- multi-D GP regression with Gaussian
  observation likelihood. **Marginal-likelihood architecture**: f is
  integrated out analytically; the chain samples only
  (amplitude, lengthscale, sigma) jointly via `joint_nuts_block`
  from the 3-dim marginal posterior `y ~ N(0, K + sigma^2 I)` with
  analytic gradient (Rasmussen & Williams Sec.5.5). half-Normal(0, sd(y))
  on amp, InverseGamma(5, s_med) on ell, Jeffreys on sigma.
- `examples/GPClassification.cpp` -- multi-D GP binary classification
  with Bernoulli-logit likelihood. **Whitened parameterization**
  (Murray & Adams 2010) in ONE block: the latent is `z ~ N(0, I)`,
  recovered as `f = L(amp, ell) * z` wherever the likelihood is
  evaluated, and `joint_nuts_block` samples
  `[{amp,1,POSITIVE},{ell,1,POSITIVE},{z,N,REAL}]` together under a
  diagonal metric. The density is
  `sum y_i f_i - softplus(f_i) - 0.5 z'z + log p(amp) + log p(ell)`;
  the GP prior's log-determinant cancels against the whitening
  Jacobian. Sampling the hyperparameters and z in one trajectory
  rather than alternating is what makes it converge -- the alternation
  leaves amp R-hat above 1.1 at this budget. half-Normal(0, 1) on amp,
  InverseGamma(5, s_med) on ell. No sigma block. `get_current()`
  reports amplitude / lengthscale / z / f (f derived as L z).
- `examples/GPTimeSeries.cpp` -- 1-D time-series GP via celerite
  O(N) Cholesky + `univariate_slice_sampling_block` for amp, tau,
  sigma on celerite-marginalized likelihood. Latent f marginalized
  out (conjugate Gaussian-Gaussian); predict_at uses
  `celerite_gp_block::predict_mean_var`.
- `examples/DPGaussianMixture.cpp` -- Dirichlet-Process Gaussian
  mixture via the truncated stick-breaking representation
  (Ishwaran-James 2001). Composes `categorical_gibbs_block` (z) +
  `stick_breaking_block` (pi, DP a_fn / b_fn) +
  `normal_gamma_cluster_gibbs_block` (mu, lambda diagonal
  Normal-Gamma) + `nuts_block` on log(alpha). Default K_trunc =
  max(20, ceil(N / 5)). DP concentration alpha ~ Gamma(a_alpha,
  b_alpha).
- `examples/PYGaussianMixture.cpp` -- Pitman-Yor variant. Identical
  composition; only stick_breaking_block's a_fn / b_fn change to
  the PY weights (a_k = 1 + n_k - discount, b_k = alpha + (k+1) *
  discount + tail). discount FIXED at construction in v0 (sampling
  it requires an interval(0, 1) constraint; trivially added via separate
  `nuts_block` + interval constraint when needed).
- `examples/DPGaussianMixture_DerivedAlpha.cpp` -- DP mixture where
  alpha = exp(phi) is REGISTERED as a deterministic refresher of
  phi; phi has Normal(0, 1) prior and is sampled by `nuts_block`.
  Demonstrates the alpha-as-derived composition pattern; downstream
  blocks read alpha from ctx unchanged. Reference for the user's
  use case "alpha = complex_function(phi)".
- `examples/FiniteGaussianMixture.cpp` -- **Finite-K** Gaussian mixture
  (NOT BNP). K is FIXED at construction; symmetric Dirichlet(alpha/K) prior
  on pi, sampled exactly via `dirichlet_gibbs_block` (this is the FIRST
  shipped reference example using that block). Use when K is known
  (or chosen via model selection); for unknown K use the DP variant
  but understand its prior-data tradeoff (see the DP block notes in
  `block_catalogue/index.md`).
  4-chain audit recovers truth mu within 0.21 sigma on the 6-cluster
  N=600 fixture (R-hat 1.0004) -- proves the shared infrastructure
  is unbiased when given correct K.

Joint (opt-in via coupling analysis in `codegen_cpp.md Sec.4a`):
- `examples/IRT1PL_joint.cpp` -- joint (theta, b) + separate sigma_b
  (shift-invariance coupling)
- `examples/HierarchicalLM_joint.cpp` -- joint (alpha, beta, u) + separate
  sigma, tau (additive linear mean + random effect)
- `examples/LinearRegJointMixed.cpp` -- joint (alpha, beta, sigma) with
  `joint_nuts_block` (REAL + POSITIVE per-slice)

Copy structure, change only model-specific parts (log-density body,
priors, shared_data keys, class name).

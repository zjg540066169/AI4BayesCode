---
name: AI4BayesCode-sim1
description: |
  Generate a per-model Simulation 1 script that runs 100 replicates of a
  cross-implementation comparison between the AI4BayesCode-generated
  sampler and the Stan reference. Each replicate samples a fresh DGP,
  fits both samplers (single chain each, 20k warmup + 20k sampling),
  aligns shared parameters, and records cross-implementation R-hat,
  per-implementation ESS_bulk / ESS_tail, and 95% credible-interval
  coverage against the data-generating values.
---

# AI4BayesCode Simulation 1 — per-model script generator

## When to invoke

The user asks for a "Simulation 1 script" or "100-replicate cross-impl
comparison script" for a benchmark model `<id>`. The model already has:

1. `<root>/<id>/<id>.cpp`         — AI4BayesCode-generated sampler (passed validator)
2. `<root>/<id>/<id>.stan`        — Stan reference implementation
3. `<root>/<id>/<id>.r`           — DGP script (auto-generated alongside the Stan model;
                                    contains a `simulate_data(seed, args = list(...))` function)
4. `<root>/<id>/<id>_runner.R`    — the AI4BayesCode runner (constructor signature + `get_history()`
                                    output shape are documented here)
5. `<root>/<id>/<id>.txt`         — natural-language model description

Here `<root>` is the project root that also contains an `AI4BayesCode/`
subdirectory (`small_example/` for the prototype, `simulations/` for the
full benchmark). Output is **one file**:

```
<root>/<id>/sim1_<id>.R
```

The script is self-contained, auto-chdirs to `<root>` at startup, and
writes outputs to `<root>/<id>/sim1_results/{results.rds, results.csv}`.

## What the script does

```
For r = 1, ..., 100:
  1. Generate (data, truth) from the DGP with seed = r.
  2. Fit AI4BayesCode chain  (20k burn + 20k keep, seed = r).
  3. Fit Stan reference chain (20k warmup + 20k sampling, seed = r).
  4. Align both chains' draws to the shared parameter set
     (intersection of names).
  5. Compute cross-implementation rank-normalized R-hat (AI as chain 1,
     Stan as chain 2), per-implementation ESS_bulk / ESS_tail, and
     95% CrI coverage indicators against the DGP truth.
  6. Return per-replicate summary + per-parameter raw vectors.

After mclapply: write `results.rds` (full per-param raw) and
`results.csv` (one aggregated row per replicate).
```

The script does **not** persist posterior draws — only the metrics
above plus per-parameter ESS / R-hat / coverage indicators and the
DGP truth.

## Required template structure

The script has 11 sections in this order. Sections (5)–(8) contain the
per-model logic; the rest is boilerplate that should be copied verbatim
modulo `<id>` substitutions.

1. Shebang + auto-chdir to `<root>` (= `dirname(dirname(.script_path))`)
2. `Sys.setenv(OMP_NUM_THREADS = "1")` + `options(mc.cores = 4L)`
3. Library loads + `source("AI4BayesCode/R/AI4BayesCode_helpers.R")`
4. Helpers: `silent_eval()` (mute Stan output) + `par_lapply` (pbmcapply
   if available, else mclapply — used to give a progress bar)
5. **Compile AI4BayesCode and Stan once** (parent process)
6. **DGP wrapper** that returns `list(data = ..., truth = ...)` — per model
7. **Per-implementation chain runners** `run_AI_chain` and `run_Stan_chain`
8. **Alignment helpers** `align_AI_draws`, `align_Stan_draws` — per model
9. Generic metric helpers `compute_cross_rhat`, `compute_ess`,
   `compute_coverage` — copy verbatim
10. Generic worker + CSV summarizer — copy verbatim
11. Smoke test (sequential `worker(1)`) → main `par_lapply` loop → save
    `results.rds` / `results.csv` → readable summary

## The five per-model regions — what to fill in

Use any existing graduated `sim1_<id>.R` in the project root (a
non-label-switching hierarchical Gaussian regression example is the
simplest reference) as a structural template. The five regions below
are the only ones that vary per model.

### (A) DGP wrapper — `simulate_data_with_truth`

The auto-generated `<id>.r` file ships a `simulate_data(seed, args = list(...))`
function that returns ONLY the Stan data list — the latent parameters
(true `mu`, `sigma`, etc.) are local variables and are dropped.

Copy the body of `simulate_data` verbatim, but **return both the data
and the truth**:

```r
simulate_data_with_truth <- function(seed) {
    set.seed(seed)
    # ... copy DGP body verbatim ...
    list(
        data = list(<fields matching <id>.stan data block>),
        truth = list(<every latent parameter that the DGP samples>))
}
```

The `truth` list keys must match the AI4BayesCode and Stan parameter
names (after alignment in (D) and (E)) — coverage matches by name.

### (B) Constructor call — inside `run_AI_chain`

Read `<id>_runner.R` for the constructor signature (look for the
`Constructor arguments for <id>` comment block). The signature is
typically `new(<id>, <data fields...>, seed, keep_history)`. Match it
exactly. `keep_history` MUST be `TRUE` so that `get_history()` returns
the post-burnin draws.

**Important — ignore the example dataset in the runner.** `<id>_runner.R`
ships with a small hard-coded dataset that is **just an example used
during generation validation** (to confirm the AI4BayesCode-generated
sampler compiles and produces sensible output on a known input). It is
not the data for sim1. Do **not** copy it into `sim1_<id>.R`, do **not**
use it as a fallback, and do **not** infer data shapes from it. The data
for every sim1 replicate comes from `simulate_data_with_truth(seed)`
(region (A)), re-drawn fresh on each replicate against a known DGP
truth. Read the runner ONLY for the constructor argument names, types,
and order.

```r
m <- new(<id>,
         <data fields in the order the .cpp constructor expects>,
         as.integer(seed), TRUE)
m$step(N_BURN)
m$step(N_KEEP)
```

### (C) Stan data — inside `run_Stan_chain`

Build `stan_data` from `data` to match the `data { ... }` block in
`<id>.stan`. Field names and types must match exactly.

### (D) `align_AI_draws` — normalize AI4BayesCode `get_history()`

`get_history()` returns a named list. Each entry is either a vector
(scalar parameter, length = total iterations) or a matrix
(vector parameter, rows = iterations, columns = dimensions).

Build a list-of-(iter × dim) matrices for every parameter that is
ALSO present in the Stan output, **trimming the burn-in rows**:

```r
align_AI_draws <- function(model_obj) {
    h <- model_obj$get_history()
    n_total  <- if (is.null(dim(h$<one_param>))) length(h$<one_param>) else nrow(h$<one_param>)
    keep_idx <- seq.int(N_BURN + 1L, n_total)
    standardize <- function(x) {
        if (is.null(dim(x))) x <- matrix(x, ncol = 1L)
        x[keep_idx, , drop = FALSE]
    }
    list(
        <param1> = standardize(h$<param1>),
        ...
    )
}
```

Skip `y_rep` and any other posterior-predictive / latent diagnostic
output — only include parameters Stan also samples.

### (E) `align_Stan_draws` — normalize Stan output

cmdstanr's `fit$draws()` returns a `draws_array`. Use
`posterior::as_draws_matrix()` to flatten and then extract by name.
Stan vector parameters appear as `name[1]`, `name[2]`, ... — sort by
index and bind into an iter × dim matrix:

```r
align_Stan_draws <- function(stan_fit) {
    d  <- posterior::as_draws_matrix(stan_fit$draws())
    cn <- colnames(d)
    sort_by_idx <- function(prefix) {
        cols <- grep(paste0("^", prefix, "\\[\\d+\\]$"), cn, value = TRUE)
        idx  <- as.integer(sub(paste0(prefix, "\\[(\\d+)\\]"), "\\1", cols))
        cols[order(idx)]
    }
    asmat <- function(x) matrix(x, ncol = 1L)
    list(
        <scalar_param> = asmat(d[, "<scalar_param>"]),
        ...
        <vector_param> = unname(d[, sort_by_idx("<vector_param>"), drop = FALSE]),
        ...
    )
}
```

**Stan-only parameters** (e.g., `lp__`, `alpha0` in `generated quantities`,
or anything Stan marginalizes that AI4BayesCode does not produce): do
**not** include in the output list. The downstream `compute_cross_rhat`
takes the intersection of names.

**Parameters AI4BayesCode samples but Stan does not** (e.g., a latent
field that Stan integrates out analytically): do not include in
`align_AI_draws` either, OR include and accept that they will be
dropped from cross-impl R-hat. Coverage and ESS for AI-only parameters
can still be computed if you prefer; just record them under names that
match `truth$<name>`.

## Label-switching models

For mixture / HMM / LDA / latent-class models, the shared K-dimensional
parameters (component means, mixing weights, emission distributions,
etc.) are only identified up to a permutation of the K labels. Direct
cross-impl R-hat and per-component coverage are meaningless without
relabeling.

### Detect

Set `has_label_switching <- TRUE` if `<id>.txt` describes any of:

- A finite mixture without an order constraint (`mu_1 < mu_2 < ...`)
- An HMM with K hidden states and unconstrained transitions / emissions
- LDA / topic models
- Multinomial latent-class models

If the model imposes an explicit order constraint on the K-dim parameter
(e.g., `ordered[K] mu` in Stan), label switching is structurally
prevented and you can skip the relabeling pipeline.

### Recipe (Stephens 2000)

Add two pieces to `align_AI_draws` / `align_Stan_draws` (or as a
separate hook called between alignment and metrics):

**(L1)** A function that computes per-iteration allocation
probabilities `p(z_i = k | y_i, theta_d)` for the model's likelihood.
Output shape: `iter × N × K`.

```r
compute_alloc_probs <- function(draws, data) {
    # Example for a Gaussian mixture:
    iter <- nrow(draws$mu)
    N <- length(data$y)
    K <- ncol(draws$mu)
    p <- array(NA_real_, dim = c(iter, N, K))
    for (d in seq_len(iter)) {
        for (i in seq_len(N)) {
            log_p <- log(draws$pi[d, ]) +
                     dnorm(data$y[i], draws$mu[d, ], draws$sigma[d, ], log = TRUE)
            log_p <- log_p - matrixStats::logSumExp(log_p)
            p[d, i, ] <- exp(log_p)
        }
    }
    p
}
```

For HMMs use forward–backward to compute marginal smoothing
probabilities. For LDA use the per-token topic-assignment posterior
`p(z_n=k | doc_n, w_n, theta_d, phi_d) ∝ theta_d[doc_n,k] * phi_d[k,w_n]`
at each draw. **These are non-trivial and easy to get wrong; surface the
formula in a comment block at the top of the alignment file and verify
on a ground-truth toy example before trusting.**

**Simple-sort relabeling is NOT sufficient for LDA, even at K=2**, when
the AI sampler is z-augmented (slow-mixing). The 2-permutation simple
sort matches the wrong labels often enough to inflate cross-impl rhat;
Stephens with the topic-assignment posterior is required. See
`label_switching.md §3 (LDA / topic model)` for the explicit formula
and a copy-pasteable allocation-probability function.

**(L2)** Stack both chains' allocation probabilities, run Stephens
once on the union, and apply the per-iteration permutation to every
label-dependent parameter:

```r
label_switching_components <- c("<param1>", "<param2>", ...)  # K-dim params

p_AI  <- compute_alloc_probs(ai_draws,  replicate_data)
p_REF <- compute_alloc_probs(ref_draws, replicate_data)
p_stk <- abind::abind(p_AI, p_REF, along = 1)
perm  <- label.switching::stephens(p_stk)$permutations

n_AI <- nrow(p_AI)
perm_AI  <- perm[1:n_AI, ]
perm_REF <- perm[(n_AI + 1):nrow(perm), ]

apply_perm <- function(draws_param, perm_mat) {
    out <- draws_param
    for (d in seq_len(nrow(out))) out[d, ] <- draws_param[d, perm_mat[d, ]]
    out
}
for (nm in label_switching_components) {
    ai_draws[[nm]]  <- apply_perm(ai_draws[[nm]],  perm_AI)
    ref_draws[[nm]] <- apply_perm(ref_draws[[nm]], perm_REF)
}
```

**(L3)** After Stephens, the two chains share an internal labeling
that may not match the DGP truth. Apply a global Hungarian match on
the posterior mean of one label-dependent parameter (typically the
component mean) so coverage indicators line up with truth:

```r
ai_mean <- colMeans(ai_draws$<key_param>)
match_perm <- as.integer(clue::solve_LSAP(
    abs(outer(ai_mean, truth$<key_param>, "-"))))
for (nm in label_switching_components) {
    ai_draws[[nm]]  <- ai_draws[[nm]] [, match_perm, drop = FALSE]
    ref_draws[[nm]] <- ref_draws[[nm]][, match_perm, drop = FALSE]
}
```

### Fall-back

If `compute_alloc_probs` is too hard (e.g., an exotic LDA variant with
hyperparameters that don't expose a closed-form alloc prob), drop the
label-dependent parameters from `align_AI_draws` / `align_Stan_draws`
entirely and report cross-impl agreement on label-invariant statistics
only (e.g., `sigma`, `log_lik`, total-mixture-density summaries). Note
the limitation in a `# NOTE:` comment at the top of the script.

## Output schema

### `results.rds` — list of length 100; each element is a named list with

```
$ replicate         : integer index
$ status            : "ok" | "ai_failed" | "ref_failed" | "align_failed" | "metrics_failed"
$ status_AI         : "ok" | "failed"
$ status_REF        : "ok" | "failed"
$ err_AI            : NA or message
$ err_REF           : NA or message
$ err_align         : NA or message
$ err_metrics       : NA or message
$ wall_sec_AI       : numeric
$ wall_sec_REF      : numeric
$ n_divergent_REF   : integer (Stan-specific)
$ n_max_td_REF      : integer (Stan-specific)
$ rhat              : named numeric vector — cross-impl R-hat per scalar slot
$ ess_bulk_AI       : named numeric vector
$ ess_bulk_REF      : named numeric vector
$ ess_tail_AI       : named numeric vector
$ ess_tail_REF      : named numeric vector
$ coverage_AI       : named integer 0/1 vector (one entry per scalar slot)
$ coverage_REF      : named integer 0/1 vector
$ truth             : per-replicate truth list returned by simulate_data_with_truth
```

For vector parameters, the named slots expand to `<param>[1]`,
`<param>[2]`, ... — names are consistent across `rhat`, `ess_*`,
`coverage_*` and the `truth` list.

### `results.csv` — one row per replicate, 31 columns

```
replicate, status, status_AI, status_REF,
wall_sec_AI, wall_sec_REF, n_divergent_REF, n_max_td_REF, n_params,
rhat_max, rhat_min, rhat_mean, rhat_median,
ess_bulk_AI_max, ess_bulk_AI_min, ess_bulk_AI_mean, ess_bulk_AI_median,
ess_bulk_REF_max, ess_bulk_REF_min, ess_bulk_REF_mean, ess_bulk_REF_median,
ess_tail_AI_max, ess_tail_AI_min, ess_tail_AI_mean, ess_tail_AI_median,
ess_tail_REF_max, ess_tail_REF_min, ess_tail_REF_mean, ess_tail_REF_median,
coverage_AI_mean, coverage_REF_mean
```

The four R-hat / ESS aggregations (`max`, `min`, `mean`, `median`)
are taken across the per-parameter scalar slots within each replicate.
`coverage_*_mean` is the mean of the 0/1 coverage indicators across
all scalar slots within the replicate.

## Smoke test — non-negotiable

Before launching `par_lapply` on the 100 replicates, run `worker(1)`
**sequentially** at the full 20k+20k budget. If it does not return
`status == "ok"`, halt the script and print every error field. This
catches:

- Compile failures (AI4BayesCode or Stan)
- Constructor mismatches (wrong arg order)
- Stan data list mismatches
- Alignment mismatches (parameter shape, missing key)
- Metric computation crashes (length mismatch, NaN propagation)

The smoke replicate runs in ~5 minutes for a small model; this is a
worthwhile insurance for a 100-replicate batch.

## Common pitfalls

1. **AI4BayesCode `get_history()` shape** — scalar parameters MAY come
   back as numeric vectors (length = iter) or as `iter × 1` matrices.
   Standardize to matrices before computing R-hat / ESS.

2. **Stan vector index ordering** — `as_draws_matrix(...)` may emit
   columns in alphabetic order (`alpha[10]` before `alpha[2]`). Sort
   columns by integer index before binding.

3. **Stan generated quantities** — `lp__`, `alpha0`-style derived
   quantities, and parameters Stan marginalizes (in `model { ... }`
   without a corresponding `parameters` declaration) are NOT
   sampled-parameter posterior draws. Exclude them from the alignment
   output to avoid noisy R-hat.

4. **Truth length mismatch** — for vector parameters, `truth$<name>`
   must have length equal to the number of columns in the aligned
   draws. If the DGP uses a different N than the Stan / AI4BayesCode
   sampler expects, fix the DGP wrapper before proceeding.

5. **mclapply + Rcpp module** — the Rcpp module class (`new(<id>, ...)`)
   is registered in the parent process via `ai4bayescode_sourceCpp` and
   inherited by fork. Do NOT call `ai4bayescode_sourceCpp` inside the
   worker — it will recompile in every fork.

6. **cmdstanr verbosity** — wrap every `stan_mod$sample(...)` and
   `fit$diagnostic_summary()` in `silent_eval(...)` or stdout from 4
   parallel workers will interleave incomprehensibly.

7. **DGP determinism** — `set.seed(seed)` MUST be the first call in
   `simulate_data_with_truth`, before any RNG draw. The Stan and
   AI4BayesCode chains use separate seeds (`as.integer(seed)`) and so
   do not race for the same RNG state.

## Run from terminal

```bash
cd <root>                                        # small_example/ or simulations/
Rscript <id>/sim1_<id>.R [N_REPLICATES] [N_CORES]
```

The script accepts two positional CLI arguments, both optional:

- `N_REPLICATES` — number of replicates to run (default `100`)
- `N_CORES`      — number of `mclapply` workers (default `4`)

Examples:

```bash
Rscript <id>/sim1_<id>.R              # 100 replicates, 4 cores  (full run)
Rscript <id>/sim1_<id>.R 5 1          #   5 replicates, 1 core   (debug)
Rscript <id>/sim1_<id>.R 100 8        # 100 replicates, 8 cores  (more parallel)
```

The script auto-chdirs to `<root>` if invoked with a full path.
Successful completion writes `<id>/sim1_results/{results.rds, results.csv}`
and prints a multi-section summary block to stdout. Failed smoke tests
`stop()` and so propagate as a non-zero exit code, so a shell loop over
many models can detect failures with `if [ $? -ne 0 ]; then ...`.

### Required CLI parsing block (insert right after auto-chdir)

```r
# -- Parse CLI args (positional): N_REPLICATES, N_CORES --------------------
.cli_args <- commandArgs(trailingOnly = TRUE)
N_REPLICATES <- if (length(.cli_args) >= 1L) as.integer(.cli_args[1L]) else 100L
N_CORES      <- if (length(.cli_args) >= 2L) as.integer(.cli_args[2L]) else 4L
if (is.na(N_REPLICATES) || N_REPLICATES < 1L)
    stop("N_REPLICATES must be a positive integer; got '", .cli_args[1L], "'")
if (is.na(N_CORES) || N_CORES < 1L)
    stop("N_CORES must be a positive integer; got '", .cli_args[2L], "'")
cat(sprintf("[setup] N_REPLICATES = %d, N_CORES = %d\n", N_REPLICATES, N_CORES))
```

Use `options(mc.cores = N_CORES)` instead of a hard-coded value, and
remove the hard-coded `N_REPLICATES <- 100L` from the MAIN section
(the value now flows from the CLI parsing block).

## Reference example

When generating a new sim1 script, locate any existing graduated
`sim1_<id>.R` in the project root that targets a similar model class
(non-label-switching hierarchical Gaussian regression is the simplest
structural template). Copy it, edit the five per-model regions (A)–(E)
above, and verify by running the smoke test before launching the
100-replicate batch.

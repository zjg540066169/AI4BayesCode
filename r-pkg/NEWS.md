# AI4BayesCode 1.1.3

## The models of a run

* Every model class has `clear_history()` (from `kernel_control_mixin`). It
  empties the stored draws and leaves the current state, the random number
  streams and the tuning unchanged, so the next `step()` records the first
  draw of a new history; until then `get_history()` and `predict_at()` return
  the current draw as one row. A model class that records per-sweep values in
  members of its own defines a public `clear_extra_history_()` that empties
  them; a class that declares its own no-argument `clear_history()` is bound
  to that one.
* `ai4bayescode_run_chains()` takes `n_cores` (`mc.cores` is still accepted)
  and `keep_models = TRUE`. Each chain runs `n_burn` sweeps, `clear_history()`
  when the model has it, and `n_keep` sweeps, so the model's own
  `get_history()` and `predict_at()` cover the kept draws only. The result has
  class `"ai4bayescode_run"` with `histories`, `seeds`, `wall`, `models`,
  `n_burn`, `n_keep` and `history_cleared`, in this order, and prints a
  summary. After a parallel run, `run$models[[i]]` is a handle (class
  `"ai4bayescode_chain_model"`): `run$models[[i]]$f(...)` calls `f` on the
  model of chain `i` in the worker process that holds it, from the R process
  that ran the chains only. After a sequential run it is the model itself.
  With `keep_models = FALSE` no model is kept, and the result is the plain
  list of 1.1.2 (`histories`, `seeds`, `wall`) followed by `n_burn`, `n_keep`
  and `history_cleared`.
* `ai4bayescode_predict(run, newdata = list(), pool = TRUE)`, also
  `predict(run, newdata)`, calls `predict_at(newdata)` on the model of every
  chain and stacks the chains along the first axis of each output, which keeps
  its number of dimensions; `pool = FALSE` returns one list per chain.
  `newdata = list()` (or `NULL`) gives the posterior predictive at the
  training data; a matrix or a data frame as `newdata` raises an error that
  shows the call `predict(run, list(X = X_new))`. The first `n_burn` rows of
  an output with `n_burn + n_keep` rows are dropped (a model without
  `clear_history()`, or one that stores per-sweep values outside the buffers
  `clear_history()` empties, which also gives a warning); with
  `pool = TRUE`, an output whose first dimension is neither `n_keep` nor 1
  gives a warning, because that axis is then not one row per stored draw.
  `ai4bayescode_close_models(run)` terminates the worker processes of a run.
* The worker processes of a kept run are also terminated when the run and
  every handle taken from it are garbage-collected, when R exits, and to make
  room for a later run: at most `getOption("ai4bayescode.max_workers")`, else
  the environment variable `AI4BAYESCODE_MAX_WORKERS`, else
  `max(8, parallel::detectCores())` worker processes of kept runs exist
  at a time. Before a run starts its worker processes, those of the least
  recently used runs are terminated until the new ones fit and
  `n_workers + 2` R connections are free, with one warning, and a call on a
  handle of such a run raises an error that names the reason; when that is
  not possible, the chains run in the session. A worker process that exits
  ends the models of its own chains only. An interrupt (Ctrl+C) during a call
  on a handle leaves the run usable.
* An R error in a chain stops `ai4bayescode_run_chains()` with
  "chain i (seed s): <message>". Only a worker process that exits during the
  run makes every chain run again sequentially; in 1.1.2 a failed parallel
  chain did, and the error did not name the chain.
* `ai4bayescode_run_chains()` calls `set.seed(seeds[i])` immediately before
  `model_ctor(seeds[i])` in every path and restores the caller's
  `.Random.seed` (or its absence) afterwards. The draws of models that use
  R's generator (`bart_block`, `genbart_block` and `softbart_block` under
  Rcpp, and constructor code that calls it) are therefore determined by
  `seeds`, one stream per chain, and differ from 1.1.2; other models give the
  same draws. With `keep_models = FALSE` a single chain runs sequentially.
* Python: `run_chains()` takes `n_cores` (`n_jobs` is still accepted) and
  `keep_models=True`, and returns a `ChainsResult`: the list of per-chain
  dicts with the keys of 1.1.2 (`"seed"`, `"wall"`, `"hist"`; `"error"` and
  `"error_type"` for a failed chain), which prints a summary. The models are
  in `chains.models`: a `ChainModel` handle per chain after a parallel run,
  the model object after a sequential run, `None` for a failed chain, and
  `None` as a whole with `keep_models=False`.
  `AI4BayesCode.predict(chains, newdata)` (`chains.predict()`) and
  `AI4BayesCode.close_models(chains)` (`chains.close()`) correspond to the R
  functions and take the result itself, not a slice of it; `predict()`
  passes `newdata` to compiled models as given (their `predict_at()` reads a
  matrix column-major and checks its number of columns) and flattens a 2-D
  array column-major for models written in Python and for models compiled
  before 1.1.3. The worker processes are not daemonic, so a model can start
  processes of its own; they are terminated by `close_models()`, when the
  result and its handles are garbage-collected, at interpreter exit, and,
  least recently used runs first, to keep at most `AI4BAYESCODE_MAX_WORKERS`
  (default `max(8, os.cpu_count())`) of them within the soft limit on
  open files. The output and the warnings of the chains and of calls on
  handles are shown in the calling process, and an error raised by the model
  of chain `i` has `"chain i: "` in front of its message.
* Python: `run_chains()` calls `np.random.seed(seeds[i] % 2**32)` and
  `random.seed(seeds[i])` before `factory(seeds[i])` on every path (in 1.1.2
  every forked chain drew the same numbers from numpy's global generator) and
  restores the caller's states after a run in the calling process.
  `n_cores <= 1` runs the chains sequentially, and `n_chains` together with
  `seeds` of another length raises `ValueError` (1.1.2 ignored `n_chains`).
* Compatibility. Code written for 1.1.2 runs unchanged: `run$histories`,
  `run$seeds`, `run$wall` and `run[[1]]` to `run[[3]]` in R, and the
  per-chain dicts and `n_jobs=` in Python, are as before. With the default
  `keep_models = TRUE` the R result has a class and prints a summary, and the
  worker processes of a parallel run are terminated only as described above;
  `keep_models = FALSE` (Python: `keep_models=False`) gives the 1.1.2 value. A
  sampler generated with 1.1.2 is compiled again on its first use (see the
  Armadillo item below) and then has `clear_history()` from
  `kernel_control_mixin`. If it records per-sweep values in members of its
  own, the history entries that `clear_history()` leaves with more than
  `n_keep` rows are cut in the returned histories and named in a warning (in R
  when the models are kept) until the class defines `clear_extra_history_()`.
  A model without `clear_history()` (a module compiled against older headers,
  or a custom object) runs as in 1.1.2, and `predict()` drops its warmup
  draws.

## Prediction in the shipped examples

* With `keep_history = TRUE`, `predict_at()` of every shipped example (except
  the prior-only GMRFPrior and IsingPrior, whose output is empty) returns one
  row per stored draw, computed from that draw's own values, so `predict(run)`
  gives `n_chains * n_keep` rows; `predict_at(new_data, last_draw_only = TRUE)`
  (Python: `last_draw_only=True`) returns one row from the current state, as
  with `keep_history = FALSE`. New in this release:
  - `BSplineRegression`, `HSGPRegression` (`f`, `mu`, `y_rep`), `PehSurvival`
    and `PehSharedFrailty` (`S`); their single-draw outputs are 1 x N rows
    instead of N x 1 columns. `PehSharedFrailty` raises an error unless `g` is
    one integer with 0 <= g < G (it read `w[g]` out of range).
  - `OrderMCMCBN`: `sampled_DAG`, the DAG stored for each sweep (it returned
    the current DAG only); `get_history()` keeps its 1.1.2 keys. In a held
    sweep of the frozen block the stored DAG is the held DAG.
  - `CategoricalIsingChainVI`, `StructuredPottsVI`: `z_samples`, the stored
    per-sweep q-draws (`new_data` must be empty). The single-draw path returns
    `n_draws` new samples from the current q, as before.
  - `IsingHiddenPotts`, `ICARSpatialGMRF`: `y_rep` from the observation
    equation (it was an empty list).
  - `GPTimeSeries`, at the training times (`list()`) and at new times
    (`list(t = ...)`): `f_mean`, `f_sd`, `f_star` (one joint draw of f over the
    requested times from its posterior given y and the draw's amp, tau and
    sigma) and `y_rep = f_star + sigma * e`, in O(N + N_new) time per draw
    from the state-space form of the exponential kernel. In 1.1.2, `list()`
    returned `y + sigma * e` and f was drawn independently at each new time,
    so `f_star` and `y_rep` differ; `f_mean` and `f_sd` agree up to rounding,
    and the sampler's draws are unchanged. The declared data input is `t` (it
    was `t_new_flag` and `y`); another key or a non-finite time raises an
    error.
  - `ODE_SIR`: new observation times `t_obs` and initial state `y0`, with
    `I_traj` and `y_rep` per draw.
  - `VCBart`: `mu` per draw, at new `(X, Z)` with the new constructor argument
    `keep_tree = TRUE`, and the new output `y_rep` (`mu` plus the draw's sigma
    times N(0, 1) noise); `mu` is still the first element. The single-draw
    prediction at the training data multiplied `beta_j` by the wrong column of
    `X`.
  - `GBartLogistic`, `GBartPoisson`, `GBartMultinomial`,
    `GBartHeteroscedastic`: prediction at new X from the forest stored at each
    sweep (it stopped with "not yet implemented").
  - `ARDLasso`: `predict_at(new_data, last_draw_only)`, and `m$predict_at()`
    in R.
* BART-family models predict at new X from every stored draw only when built
  with `keep_tree = TRUE`, so `predict(run, list(X = X_new))` needs it in
  `model_ctor`. Without it, and with more than one stored draw, BartNoise,
  SoftBartNoise and the GBart models raise an error that names
  `keep_tree = TRUE` and `last_draw_only = TRUE` (BartNoise and SoftBartNoise
  stopped with "inconsistent history sizes"); VCBart returns one row from
  the current forests, as in 1.1.2, with a warning.
* `HierarchicalLM_joint`, `HierarchicalLM_MultivariateRE`: `predict_at()`
  checks `g_idx` / `group_idx` (one integer per row, >= 0, resp. >= 1). A
  group without training rows gets its random effect drawn once per group and
  draw from N(0, tau_d^2), resp. N(0, diag(tau_d) L_d L_d' diag(tau_d)); it was
  an error.
* `bart_block::predict_history()` adds to each stored forest the mean of the
  working response of the sweep that drew it; predictions at new X were off
  by the change of that mean between sweeps (e.g. with VCBart's partial
  residuals). `softbart_block` stores the level `m` of each forest likewise.
  `bart_block`, `genbart_block`, `softbart_block` and `hmm_block` gain
  `record_held_history()`: inside a nested composite frozen within another
  composite, their histories have one entry per outer sweep. A VI block held
  through a frozen composite before its first stored row appends one row per
  sweep (it kept one row).
* `predict_at()` of a model built on `composite_block` computes each node
  after every parent that the same call recomputes; before, a node could use
  the current-state value of such a parent.
* `set_current()` with an `X` of another row count in `LogisticRegression`,
  `LinearRegJointMixed` and `SpikeSlabRJMCMC` empties the stored draws through
  `clear_history()` also when one draw is stored (that draw was kept). The
  "clearing history" warning of `LogisticRegression` and `SpikeSlabRJMCMC`
  comes after `X`, `N`, `y` and the derived keys are written, so a warning
  turned into an error leaves a consistent model. The `set_current` template
  in `codegen_cpp.md` is changed the same way.

## Draws that change

* `softbart_block` with `center_Y = TRUE` (the default, and SoftBartNoise's
  setting) adds back the mean `m` of the working response that it subtracts
  before the fit: `f`, `current()` and the predictions are
  `m + y_scale * f_Z` on the scale of `y`. Before, SoftBartNoise's `sigma`
  and `y_rep` left out the level of `y` (with `y + 3`, `sigma` was 3.28
  instead of 0.51). SoftBartNoise draws change; `center_Y = FALSE` draws do
  not.
* Python: at its first `step()`, each `bart_block`, `genbart_block` and
  `softbart_block` seeds the BART kernel from a copy of the chain's
  random-number generator (the chain's own stream does not advance), and
  `BartNoise`, `GBartLogistic`, `GBartMultinomial`, `GBartHeteroscedastic`
  and `VCBart` also seed it from a non-zero `rng_seed` in the constructor. So
  every BART-family sampler, including one generated before 1.1.3, gives
  chains that differ, are reproduced from the seeds, and are the same in
  parallel and sequential `run_chains()` (before, forked chains could share
  the kernel stream). The kernel stream is one per thread and compiled module: a
  chain is reproduced when no other BART-family model of the same module is
  built or stepped in that thread in between, as in `run_chains()`. The
  Python draws of BART-family models change; R draws do not.
* On macOS, every model compiled by `ai4bayescode_source()`,
  `ai4bayescode_source_checkout()`, the Rcpp plugin
  (`// [[Rcpp::depends(AI4BayesCode)]]` with `Rcpp::sourceCpp()`) or
  `AI4BayesCode.source()` allocates Armadillo's heap arrays on 256-byte
  boundaries (other systems keep Armadillo's default allocation, as in
  1.1.2). With macOS Accelerate, the results of BLAS and LAPACK routines
  (dot products, triangular and LU / Cholesky solves) depend on the operand
  addresses modulo 256 bytes, so one model built several times with one seed
  gave different draws (`GPRegression` in R); such builds now give
  bit-identical draws. R draws of `GPRegression` differ from 1.1.2, and draws
  of other models that call these routines can differ in the last bits.
  `ai4bayescode_source()` and `ai4bayescode_source_checkout()` build in a
  subfolder of the Rcpp cache folder named after a fingerprint of the
  compiler flags, the header and the package version, and the Python build signature covers the
  header, so modules built by earlier versions are compiled again. A module
  built through the plugin and kept in a lasting `rcpp.cache.dir` folder is
  loaded again for an unchanged file unless `rebuild = TRUE`.

## Other changes

* Python: a `state_map` argument (`predict_at()`, `set_current()`) accepts a
  Mapping whose values are numbers, 0-d, 1-D or 2-D arrays, or nested
  sequences; a 2-D value is flattened column-major, as R flattens a matrix,
  so `m.predict_at({"X": X_new})` takes an `N_new x p` array. The 13 examples
  whose R `predict_at()` checks the column count of `X` check it in Python
  too.
* Python: under `warnings.simplefilter("error")`, `freeze()` and `unfreeze()`
  raise the `UserWarning` instead of `SystemError`.
* The binding macros call `clear_history()` and `readapt_NUTS()` through a
  dispatch that uses the wrapper's own member with the arguments it takes (a
  wrapper with only the 4-argument `readapt_NUTS()` compiles as with 1.1.2,
  and the 3-argument call passes `target_accept = -1`); the other members
  are selected with `static_cast`, so a wrapper that redeclares one of them
  with another signature stops at compile time (before, the call passed
  arguments the function does not take).
* `ai4bayescode_doc()` / `AI4BayesCode.doc()`: the Methods list has
  `clear_history()`, `freeze()`, `unfreeze()`, `get_frozen()` and
  `readapt_NUTS()` also when the binding macros register them, and
  `get_history()` is described as the draws since construction or the last
  `clear_history()` (it said "warmup included").
* Checkout mode: `R/AI4BayesCode_helpers.R` defines every function of the
  package identically, so its `ai4bayescode_diagnose()` warns about
  non-finite draws, `ai4bayescode_rhat_summary()` checks `drop_burn`,
  `ai4bayescode_new_frozen()` rejects dot-path names and
  `ai4bayescode_perf_hint()` is deprecated; sourced into an environment other
  than the global one, it runs the chains in worker processes and registers
  its S3 methods. The deprecation message of `ai4bayescode_perf_hint()` shows
  `sum(run$wall)` instead of the non-existent `run$chains`.
* Documentation: the READMEs and skills describe the models of a run,
  `n_cores`, the worker limit, `keep_models = FALSE` and `keep_tree = TRUE`
  for BART prediction at new X; `?ai4bayescode_diagnose` says to pass
  `n_burn` only for a history that contains the warmup draws; the
  `@example:python` blocks pass `n_cores=1`; the package description and
  README count 43 bundled examples.

# AI4BayesCode 1.1.2

* `bart_block` has a collapsed CHANGE move, which runs in every sweep and has
  no option. After birth/death, one scan per tree updates the split variable of
  every internal node by Metropolis-Hastings from its marginal, with the
  cutpoint and the leaf values integrated out, and draws its cutpoint from its
  full conditional. The posterior is unchanged and mixing over split rules
  improves. The tree update takes about 1.1 to 2.5 times as long for n of 300
  or more, about 3 times at n = 200 and about 4 times at n = 80, and the draws
  for a given seed differ from earlier versions. `bart_block` stops with an
  error if `X` has a missing or infinite value, or if `set_data()` receives an
  `X` with another number of columns.
* `bart_block` no longer copies its stored history in every sweep, so with
  `keep_tree` the time per sweep no longer grows over the chain; its variable
  counts stay correct after `set_tree()`.
* For a model with a `bart_block` or `genbart_block`, the skills forbid `sin`,
  `cos` and `tan` in the simulated data of the examples and the validation
  harness. The BartNoise, GBartPoisson, GBartHeteroscedastic and VCBart
  examples use piecewise-linear terms instead.
* Generated samplers are validated by a separate validator subagent, which
  runs the compile (L1), static (L2) and runtime (L3) checks strictly in that
  order; an edit to the sampler restarts the checks at L1. New checks confirm
  that the model DAG contains every variable that the model equations make
  predictable and that `predict_at()` returns each of them.
* Windows is supported natively. Samplers compile with the Rtools of the
  installed R version, also from a library or temporary folder whose path
  has spaces or non-ASCII characters. On Windows, samplers take `exp()`,
  `log()` and `pow()` from the Universal C Runtime, which is several times
  faster than the versions Rtools links by default.
* `ai4bayescode_source()` compiles a file whose name contains a space on
  every system, from a copy named `.ai4b_<name>` beside it.
* On Windows, the per-user block library is now in the profile folder, shared
  with the Python package. Blocks that earlier versions installed under
  `Documents` are moved there.
* `bart_block`: the sigma prior is calibrated as in `BART::wbart`. `sigest`
  is `sqrt(RSS / (n - rank))` of the least-squares fit of `y` on an
  intercept and `X`, with the rank counted as `lm()` counts it; it was
  `RSS / (n - p)`. `sigest` is `sd(y)` when `X` has `n` or more
  non-constant columns. Models that use this prior (e.g. BartNoise, VCBart)
  draw differently. The fit is a Householder QR in plain C++, so `sigest`
  does not depend on the BLAS library or the number of threads. `lambda`
  can differ between the R build and the Python / standalone builds in the
  last bits (1 ulp at the default `nu = 3`, up to 20 ulp for integer `nu` up
  to 100), because `qchisq` comes from R in the R build. New
  `prior_sigest()` and `prior_lambda()`.
* `softbart_block` normalises its inputs as the SoftBart package does: each
  column of `X` is mapped to `[0, 1]` by its ranks, and the forest fits `Y`
  divided by `y_scale` (new; default the range of `y_init`). `f`, `sigma`
  and predictions are on the scale of `y`. The default `sigma_hat` is the
  root mean squared residual of the least-squares fit of `y` on `X` (it was
  `sd(y)`). Before, the part of a column of `X` outside `[0, 1]` received
  almost no splits, and the leaf prior was narrower than SoftBart's by the
  range of `y`: on the SoftBartNoise example data the 95% intervals of `f`
  covered 81% of the true values (now 99.8%; `SoftBart::softbart` 99.3%).
  Every model with a `softbart_block` draws differently.
* DPGammaMixture lists `shape` and `rate` among the dependencies of
  `cluster_params`, so its update starts from the atoms as relabelled by
  `stick_label_swap_block`. Before, the atoms and `z` disagreed after each
  relabelling and the chain targeted a wrong posterior (no component near
  the mean-3 group of the demo data). The cards give the same wiring for
  every block that keeps a key another block writes: `cluster_atom_block`
  (atom keys), `split_merge_block` (`z`, with `cluster_counts`
  invalidated), and `categorical_gibbs_block` / `binary_gibbs_block` with
  their own key routed. Such a block starts each sweep from the shared
  value of the key, so a restore writes the shared key too (`data().set`,
  `composite_block::set_current` or `sync_child`).
* In the bundled examples, `get_current()`, `get_history()` and
  `predict_at()` use values computed from the current state before the
  first sweep and after `set_current()` or `set_tree()`; 28 examples used
  values from construction or from the last sweep. `set_current()` checks
  every input before it writes anything, so a call that stops with an
  error leaves the model unchanged, and it loads keys it ignored (e.g. `mu`
  and `lambda` of FiniteGaussianMixture, all parameters of
  HierarchicalLM_MultivariateRE). Labels loaded with `set_current()` are
  reported as the sampler holds them (rounded to the nearest label).
  GBartMultinomial recomputes the class probabilities in every sweep;
  `predict_at(list())` drew from equal probabilities. Draws of `step()` are
  unchanged unless `set_current()` or `set_tree()` is called.
* SpikeSlabRJMCMC: `eta = X beta`, used by `predict_at()`, is computed at
  construction and by `set_current()`; it was 0 until the first sweep and
  kept its old value after `set_current()`. `step()` stops with an error
  when `X` and `y` have different numbers of rows, also when `sigma` and
  `gamma_beta_rj` are frozen, where it updated only `pi` and `tau`.
  `get_dag()` lists `sse` in place of `X` among the inputs of `sigma` and
  adds `sse` to the keys refreshed after `gamma_beta_rj`.
* OrderMCMCBN: `get_current()` returns `order_party` (the element sizes of
  the partition), and `set_current()` with the `order` and `order_party`
  values of `get_current()` loads the partition exactly; a second model
  with the same seed then continues the chain identically.
  `get_history()$order` is 1-based, as `get_current()$order` is; it was
  0-based. New `order_mcmc_block::set_partition()` and
  `score_cache::cached_scorer()`.
* NUTS: `get_adaptation()` returns `warmup_done`, and `set_adaptation()` with
  `warmup_done = TRUE` does not run the first-call warmup again. A block
  given `set_current()` with its `current()` value, `set_adaptation()` with
  a snapshot taken after `step()`, and the same RNG state continues the
  saved chain draw for draw when the parameter has no constraint
  transform; with one, the unconstrained value is recomputed from the
  natural one and can differ in the last bit. A missing or NA `warmup_done`
  (Python: None, NaN, pandas NA) counts as FALSE. `set_adaptation()` checks
  the snapshot before it changes anything, and `set_adaptation()`,
  `set_precond_matrix()` and the constructors stop with an error for a
  non-empty `precond_mat` that is not d x d (d: unconstrained dimension).
  Before, a non-square one with d^2 elements made the next `step()` stop
  with an error unless it was replaced first, and one with another number
  of elements was ignored.
* VI blocks: loading values that differ from the current ones restarts the
  averaging epoch; `set_current()` takes the natural-scale value of length
  `dim()` and stops with an error for another length. In exact mode the log
  density of each state is tabulated while the dependencies are unchanged,
  so `log_density` must depend only on `(z, ctx)`.
* `gmrf_precision_block` and `gmrf_whitened_ess_block` redo the symbolic
  analysis of `Q` when its sparsity pattern changes. Before, a `Q` with
  another pattern gave an error, or a wrong factor and draws from a wrong
  distribution.
* `lda_collapsed_gibbs_block` reloads `w` and `doc` when a later context
  holds different token ids; before, the sweeps kept using the first ones.
* `piecewise_exponential_gibbs_block`, `frailty_gamma_gibbs_block` and
  `interval_censored_survival_augmentation_block` stop with an error when
  `edges[0]` is not 0, as their model requires; with `edges[0] > 0` the
  three blocks did not treat a time below `edges[0]` alike.
* `softbart_block::set_tree()` restores the sampler state exactly, so the
  draws after `set_tree()` differ from earlier versions.
* `composite_block` passes each child its context as read-only views of the
  stored values instead of a fresh copy in every update (on Apple arm64, a
  copy for contexts of 36 to 129 KiB, where Accelerate's BLAS runs faster on
  a copy). New `composite_block::sync_child(i)` writes a child's outputs
  into the shared data after a loader was called on that child. New
  `shared_data_t::declare_refreshed_after_set_current()`: the `freeze()`
  warning leaves out derived keys that the model recomputes in
  `set_current()`.
* Faster sweeps with unchanged draws: BART, SoftBART and genBART keep
  per-tree fits or leaf partitions between sweeps (the BART and SoftBART
  stores hold at most 2^22 doubles, 32 MiB, set with
  `AI4BAYESCODE_BART_FIT_STORE_MAX_DOUBLES` /
  `AI4BAYESCODE_SOFTBART_FIT_STORE_MAX_DOUBLES`, defined alike in every
  translation unit), and the NUTS metric setup, the HMM forward filter,
  Ising bond probabilities, the rjmcmc log joint, survival data summaries,
  ESS with an identity factor, split-merge proposal terms and order-MCMC
  scores are reused while their inputs are unchanged.
* Skills: the truncated stick-breaking compositions include
  `stick_label_swap_block` with its wiring, `split_merge_block` is named
  where split-merge was described as not shipped, the VI and validator
  texts describe `set_current()` with natural-scale values, and the
  SoftBart runner recipe no longer passes `sigma_hat` in SoftBartNoise's
  `tau_rate` position.

# AI4BayesCode 1.0.0

* `ai4bayescode_generate("model description")` -- natural-language-to-code
  generation now ships. Drives an LLM (Claude / OpenAI) to write **and**
  validate a sampler from a prose model description, with compile +
  2-chain rank-R-hat + repair-on-failure to convergence. Interactive mode
  prompts for model, thinking level, backend, and per-parameter priors.
  Companion helpers: `ai4bayescode_models()`, `ai4bayescode_set_key()`,
  `ai4bayescode_key_status()`.
* Expanded to 43 bundled reference examples and the full AI codegen skill
  corpus under `inst/skills/`.
* `ai4bayescode_source(code)` -- canonical, self-contained sourcer/loader.
  Resolves the full include tree (AI4BayesCode, mcmclib, BaseMatrixOps,
  Eigen, celerite, libgp_kernels) + the three preprocessor defines +
  platform BLAS/LAPACK from the **installed package** via a temporary
  `R_MAKEVARS_USER` Makevars, so `install.packages("AI4BayesCode")` +
  `library(AI4BayesCode)` is all a user needs to compile any generated
  sampler -- **no AI4BayesCode checkout / source location required**.
  Accepts a `.cpp` **file path or a source string**. Any personal
  `~/.R/Makevars` is preserved (our flags are *prepended*, not clobbered).
  This closes the gap where the previous `ai4bayescode_sourceCpp()` /
  Rcpp-plugin path forwarded only the first `-I` and dropped all `-D`, so
  examples needing eigen / celerite / libgp could not compile.
* `ai4bayescode_sourceCpp()` is now a thin back-compatibility alias for
  `ai4bayescode_source()`.

# AI4BayesCode 0.9.0

First public release.

## What's in v0.9.0

* Bundles the AI4BayesCode header-only C++ library: 28 block headers
  (NUTS, joint NUTS, BART, genBART, SoftBART, Gibbs family, RJMCMC,
  HMM, slice, stick-breaking, VI, etc.).
* Bundles vendored dependencies (mcmclib, Eigen 3.4, autodiff, libgp,
  celerite, BART kernels) so that `library(AI4BayesCode)` works on a
  clean R install — users do NOT need a local `AI4BayesCode/` folder.
* Bundles the 12-skill AI codegen workflow (`start.md`, `codegen.md`,
  `validator.md`, etc.) under `inst/skills/`; an AI coding agent can
  load these via `ai4bayescode_skills_path()`.
* Bundles 34 reference example samplers under `inst/examples/`.
* `ai4bayescode_sourceCpp(file)` -- drop-in `Rcpp::sourceCpp()`
  replacement that auto-injects the `Rcpp::depends` directive and the
  preprocessor defines that AI4BayesCode-generated samplers need.
  Works without any path argument.
* `ai4bayescode_example(name)` -- compile + load a built-in example by
  name (e.g. `"GaussianLocationScale"`).
* Helpers: `ai4bayescode_include_path()`, `ai4bayescode_skills_path()`,
  `ai4bayescode_examples_path()`, `ai4bayescode_list_examples()`,
  `ai4bayescode_list_skills()`, `ai4bayescode_version()`.

## Not yet in v0.9.0 (planned)

* `ai4bayescode_generate("model description")` -- AI-driven code
  generation from natural-language model descriptions. Planned for
  v0.10.
* `m$sample(data = ..., chains = ...)` CmdStanR-style high-level
  wrapper. Planned for v0.10.
* Python package mirror. Planned for v0.11.
* Trimmed Eigen subset to reduce installed size from ~10.7 MB to ~5 MB.
  Planned for v0.9.1.

For full architectural details, validator-check definitions, and
codegen skill spec, see the canonical source repository.

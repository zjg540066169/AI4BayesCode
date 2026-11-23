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

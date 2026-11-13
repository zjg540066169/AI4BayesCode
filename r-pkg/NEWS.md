# AI4BayesCode (development version)

* Windows is supported natively. Samplers compile with the Rtools of the
  installed R version, also from a library or temporary folder whose path
  has spaces or non-ASCII characters. On Windows, samplers take `exp()`,
  `log()` and `pow()` from the Universal C Runtime, which is several times
  faster than the versions Rtools links by default.
* `ai4bayescode_install_skill()` installs the AI4BayesCode agent skill for
  Claude Code and Codex without git, the same way on Windows, macOS and
  Linux. Run it again to update.
* `ai4bayescode_source()` compiles a file whose name contains a space on
  every system, from a copy named `.ai4b_<name>` beside it.
* On Windows, the per-user block library is now in the profile folder, shared
  with the Python package. Blocks that earlier versions installed under
  `Documents` are moved there.

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

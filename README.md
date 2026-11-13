# AI4BayesCode

AI4BayesCode is an AI-enhanced statistical software system for generating
trustworthy Bayesian samplers from natural-language model descriptions.
Supports R (Rcpp), Python (pybind11), and standalone C++ backends from the
same source.

**Full documentation, tutorials, and examples: [https://ai4bayescode.com/](https://ai4bayescode.com/)**

## Install

**Recommended:** new here? Install both -- generate your sampler with a coding
agent, then run it with the R or Python package. See
[https://ai4bayescode.com/install.html](https://ai4bayescode.com/install.html)
for full details.

### As an agent skill (Claude Code / Codex / Cursor / ...)

The repository root is the skill package (`SKILL.md` + `start.md` + the C++
library under `include/`). One command puts it into your agent's skills folder,
and the same command updates it to a new version. It is the same on macOS,
Linux and Windows. On Windows, run it in PowerShell. It does not work in Command
Prompt. If your agent runs inside WSL, run it inside WSL.

**Claude Code** -- install into `~/.claude/skills`, restart, then type `/AI4BayesCode`:
```bash
git init -q "$HOME/.claude/skills/AI4BayesCode"; git -C "$HOME/.claude/skills/AI4BayesCode" pull -q --ff-only https://github.com/zjg540066169/AI4BayesCode.git main
```

**Codex** -- install into `~/.agents/skills`, restart, then type `$AI4BayesCode` or
pick it from `/skills` (the Codex app also lists it among the slash commands):
```bash
git init -q "$HOME/.agents/skills/AI4BayesCode"; git -C "$HOME/.agents/skills/AI4BayesCode" pull -q --ff-only https://github.com/zjg540066169/AI4BayesCode.git main
```

If the command reports an error, delete the folder and run it again.

If you installed with older instructions, delete the old `AI4BayesCode` folder
before you run it. For Codex, also delete the old copy in `~/.codex/skills`, or
Codex lists the skill twice.

For Cursor, the AI-assisted install, or letting your agent install it for you,
see `AI_AGENT_INSTALL.md`.

### R  (needs a C++17 compiler and `Rcpp`, `RcppArmadillo`)
```r
remotes::install_github("zjg540066169/AI4BayesCode", subdir = "r-pkg")
```
On Windows the compiler is [Rtools](https://cran.r-project.org/bin/windows/Rtools/)
for your R version (Rtools44 for R 4.4, Rtools45 for R 4.5 and later).

### Python  (>= 3.11, a C++17 compiler and Armadillo)
```bash
pip install "git+https://github.com/zjg540066169/AI4BayesCode.git#subdirectory=python"
```
Without git, install the same code from GitHub's zip archive:
```bash
pip install "https://github.com/zjg540066169/AI4BayesCode/archive/refs/heads/main.zip#subdirectory=python"
```
Armadillo: `brew install armadillo` on macOS, `apt-get install libarmadillo-dev` on
Debian/Ubuntu. On Windows, Rtools44 or newer (above) already holds the compiler, Armadillo
and OpenBLAS. Without it, use MSYS2 instead, as `python/README.md` describes.

## Quick start (R)

```r
library(AI4BayesCode)

# 0. Set your LLM provider key once per session (never written to disk).
#    Without it, step 1 writes ./generated/PROMPT.txt for a coding agent
#    instead of generating, and step 2 has no file to compile.
ai4bayescode_set_key("sk-ant-api03-...", "anthropic")

# 1. Generate a sampler from a plain-language model description.
ai4bayescode_generate(
  "linear regression, y ~ N(X beta, sigma^2), normal prior on beta, half-normal on sigma",
  classname = "BayesLinReg", output_path = "./generated")

# 2. Compile the generated sampler.
ai4bayescode_source("./generated/BayesLinReg.cpp")

# 3. Run several chains on your data (y, X) and check convergence.
run <- ai4bayescode_run_chains(
  # R matches constructor arguments BY POSITION -- argument names are
  # ignored. Order: data args, then rng_seed, then keep_history.
  function(s) new(BayesLinReg, y, X, as.integer(s), TRUE),
  n_chains = 4, n_burn = 5000, n_keep = 5000)
ai4bayescode_rhat_summary(run)
```

See the [API reference](https://ai4bayescode.com/api.html) for `get_history`,
`predict_at` (posterior-predictive), diagnostics, and the Python equivalents.

## License

Licensed as a whole under GPL-3.0-or-later; see `LICENSE` and
`THIRD_PARTY_LICENSES.md` for the full per-component attributions.

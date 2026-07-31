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
library under `include/`). Clone it into your agent's skills directory, then
launch it.

**Claude Code** -- clone into `~/.claude/skills`, reload, then type `/AI4BayesCode`:
```bash
git clone https://github.com/zjg540066169/AI4BayesCode.git ~/.claude/skills/AI4BayesCode
```

**Codex** -- clone into `~/.codex/skills`, restart, then type `/AI4BayesCode`
(or, if your build has no slash commands, tell it to read `start.md`):
```bash
git clone https://github.com/zjg540066169/AI4BayesCode.git ~/.codex/skills/AI4BayesCode
```

For Cursor, the AI-assisted install, or letting your agent install it for you,
see `AI_AGENT_INSTALL.md`.

### R  (needs a C++17 compiler and `Rcpp`, `RcppArmadillo`)
```r
remotes::install_github("zjg540066169/AI4BayesCode", subdir = "r-pkg")
```

### Python  (>= 3.11; a C++ compiler; `brew install armadillo`)
```bash
pip install "git+https://github.com/zjg540066169/AI4BayesCode.git#subdirectory=python"
```

## Quick start (R)

```r
library(AI4BayesCode)

# 1. Generate a sampler from a plain-language model description.
ai4bayescode_generate(
  "linear regression, y ~ N(X beta, sigma^2), normal prior on beta, half-normal on sigma",
  classname = "BayesLinReg", output_path = "./generated")

# 2. Compile the generated sampler.
ai4bayescode_source("./generated/BayesLinReg.cpp")

# 3. Run several chains on your data (y, X) and check convergence.
run <- ai4bayescode_run_chains(
  function(s) new(BayesLinReg, y, X, seed = s, keep_history = TRUE),
  n_chains = 4, n_burn = 5000, n_keep = 5000)
ai4bayescode_rhat_summary(run)
```

See the [API reference](https://ai4bayescode.com/api.html) for `get_history`,
`predict_at` (posterior-predictive), diagnostics, and the Python equivalents.

## License

Licensed as a whole under GPL-3.0-or-later; see `LICENSE` and
`THIRD_PARTY_LICENSES.md` for the full per-component attributions.

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

The repository root IS the skill package (`SKILL.md` + the C++ library under
`include/`). Clone it, then point your agent at it; after reloading, type
`/AI4BayesCode` to launch (agents without slash commands read `start.md` first).

```bash
git clone https://github.com/zjg540066169/AI4BayesCode.git AI4BayesCode
```

See `AI_AGENT_INSTALL.md` for per-agent install steps (Claude Code, Codex,
Cursor, AI-assisted).

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
library(AI4BayesCode)   # headers ship inside the package -- no checkout path needed
ai4bayescode_sourceCpp(file.path(ai4bayescode_examples_path(), "GaussianLocationScale.cpp"))

set.seed(1)
y <- rnorm(100, 2.0, 1.5)

m <- new(GaussianLocationScale, y, seed = 42L, keep_history = TRUE)
m$step(4000L)   # warmup
m$step(4000L)   # sampling

h  <- m$get_history()
pp <- m$predict_at(list())   # posterior-predictive y_rep

cat("mu:    mean =", mean(h$mu),    " sd =", sd(h$mu),    "\n")
cat("sigma: mean =", mean(h$sigma), " sd =", sd(h$sigma), "\n")
cat("y_rep shape:", paste(dim(pp$y_rep), collapse = " x "), "\n")
```

## License

Licensed as a whole under GPL-3.0-or-later; see `LICENSE` and
`THIRD_PARTY_LICENSES.md` for the full per-component attributions.

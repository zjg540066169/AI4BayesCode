# AI4BayesCode (R package)

R package for the AI4BayesCode block-composite MCMC + variational
inference library. After install, all C++ headers, vendored
dependencies (`mcmclib`, `eigen`, `autodiff`, `libgp`, `celerite`,
`bart`), the AI codegen skill workflow, and 39 reference examples
ship as package data — **no local `AI4BayesCode/` folder needed**.

## Install (from GitHub)

```r
# install.packages("remotes")
remotes::install_github("zjg540066169/AI4BayesCode", subdir = "r-pkg")
```

Alternatively, from r-universe:

```r
options(repos = c(
    jz3183 = "https://jz3183.r-universe.dev",
    getOption("repos")
))
install.packages("AI4BayesCode")
```

Or from a local source tree (from the repo root):

```r
remotes::install_local("AI4BayesCode/r-pkg")
```

```bash
# equivalently, from a shell:
cd AI4BayesCode/r-pkg && R CMD INSTALL .
```

## Quick start

```r
library(AI4BayesCode)

# Run a built-in example — no path setup
ai4bayescode_example("GaussianLocationScale")
m <- new(GaussianLocationScale,
         y = rnorm(100, mean = 2, sd = 1.5),
         seed = 1L,
         keep_history = TRUE)
m$step(4000L)   # warmup
m$step(4000L)   # sampling
h <- m$get_history()
summary(h$mu)
summary(h$sigma)

# Compile a .cpp you generated elsewhere -- no AI4BayesCode checkout needed,
# the headers ship inside the installed package.
ai4bayescode_source("./my_model.cpp")          # file path ...
ai4bayescode_source(my_generated_cpp_string)   # ... or a source string
m <- new(MyModel, ...)
```

`ai4bayescode_source()` is the canonical self-contained compiler;
`ai4bayescode_sourceCpp()` remains as a back-compatibility alias.

## Generate a sampler from natural language

`ai4bayescode_generate()` drives an LLM to write **and validate** a sampler
from a prose model description — compile → 2-chain rank-R-hat →
repair-on-failure, to convergence. It is **LLM-agnostic** (Claude Opus 4.8 /
Sonnet / Haiku and OpenAI GPT-5.5 / Codex), and you pass the API key, the
model, and the thinking level **in the call**; nothing is saved.

```r
res <- ai4bayescode_generate(
  "y ~ N(Xbeta, sigma^2), p(sigma^2) propto 1/sigma^2  (linear regression)",
  API_key = "sk-YOUR-KEY-HERE",            # passed here, never saved
  LLM     = "claude-opus-4-8",   # or gpt-5.5, claude-sonnet-4-6, ...
  effort  = "high",              # thinking level, validated against this model's levels
  backend = "R", output_path = "./generated", max_attempts = 2L)
res$validated; res$cpp_path
```

Or **interactive** — run with no arguments and answer the prompts (menus for
the model + thinking level, then backend, output folder, class name, key, and a
prior for each parameter):

```r
ai4bayescode_generate()
```

`ai4bayescode_models()` lists the selectable models and each model's valid
thinking levels. Requires `httr2` + `jsonlite`; with no key set, `generate()`
just writes the assembled prompt to `PROMPT.txt`.

Prefer not to pass the key every call? Set it **once per session** (session-only,
never written to disk):

```r
ai4bayescode_set_key("sk-ant-api-...", "anthropic")   # or "openai" / "google"
ai4bayescode_key_status()                              # shows what's set (masked)
ai4bayescode_generate("Linear regression.", LLM = "gpt-5.5")  # key picked up
```

`API_key=` still works per call and **overrides** the session key for that one call —
e.g. `ai4bayescode_generate("...", API_key = "sk-other")` — without changing what
`set_key()` configured. Omit it to use the session default.

## What ships in this package (v1.0.0)

| Asset | Count | How to access |
|---|---|---|
| Block headers (`AI4BayesCode/*_block.hpp`) | 34 | `ai4bayescode_include_path()` |
| Vendored deps (mcmclib, eigen, autodiff, libgp, celerite, bart) | 6 | bundled, auto-linked |
| Reference examples (`.cpp`) | 39 | `ai4bayescode_list_examples()` |
| AI codegen skills (markdown) | 18 | `ai4bayescode_list_skills()` |

## Planned for later versions

The package ships the runtime, the bundled assets, **and** the NL→code
generator (`ai4bayescode_generate()`, above). Still planned:

- `m$sample(data = ..., chains = 4, ...)` — a CmdStanR-style high-level wrapper
- Command-line interface (`ai4bayescode` CLI)
- A Python package mirror — **already available** as the `ai4bayescode` wheel

Besides `ai4bayescode_generate()`, AI codegen can also run via your preferred
chat tool (Claude Code / Cursor / etc.) — point the agent at the bundled skills:

```r
ai4bayescode_skills_path("start.md")
#> "/Library/.../R/library/AI4BayesCode/skills/start.md"
```

…then tell the agent "follow the workflow at that path, then write code
for my model". When it produces a `.cpp`, run
`ai4bayescode_source(file)` to compile + load it.

## License

GPL (>= 3). See top-level `LICENSE` in the AI4BayesCode source tree.
Vendored dependencies retain their upstream licenses; see
`THIRD_PARTY_LICENSES.md` in the parent source tree for the full
combination analysis.

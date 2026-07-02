# AI4BayesCode — Python helper

Python-side companion to the AI4BayesCode MCMC library. Provides the
Python equivalent of the R-side `AI4BayesCode_helpers.R`:

| R function                      | Python function                  |
|---------------------------------|----------------------------------|
| `ai4bayescode_source(...)`      | `AI4BayesCode.source(...)`        |
| `ai4bayescode_generate(...)`    | `AI4BayesCode.generate(...)`      |
| `ai4bayescode_models()`         | `AI4BayesCode.models()`           |
| `ai4bayescode_doc(...)`         | `AI4BayesCode.doc(...)`           |
| `ai4bayescode_plot_dag(model)`               | `AI4BayesCode.plot_dag(model)`    |
| `ai4bayescode_run_chains(...)`  | `AI4BayesCode.run_chains(...)`    |
| `posterior::rhat(x)`            | `AI4BayesCode.rhat(x)`            |
| `posterior::ess_bulk(x)`        | `AI4BayesCode.ess_bulk(x)`        |

## Install

From this directory:

```bash
pip install .
# or for dev (editable):
pip install -e .
# with viz + diagnostic extras:
pip install ".[viz,diagnostics]"
```

Requires Python ≥ 3.11, a C++17 compiler (clang++ or g++), and BLAS/LAPACK.

## Usage

```python
import AI4BayesCode

# Compile a user `.cpp` file (pybind11 module) against the AI4BayesCode headers.
# The .cpp must contain `PYBIND11_MODULE(<ModuleName>, m) { ... }`.
# `source()` is self-contained -- the headers ship inside the installed package.
mod = AI4BayesCode.source("MyModel.cpp")            # file path, or a source string

# The module exposes the class defined in the .cpp:
m = mod.MyModel(y, X, seed=1, keep_history=True)

# Standard MCMC workflow:
m.step(5000)          # warmup
m.step(5000)          # keep
hist = m.get_history()

# Convergence diagnostics:
import numpy as np
beta_samples = np.array(hist["beta"])
print(AI4BayesCode.posterior_summary(beta_samples))

# Visualize the model DAG:
AI4BayesCode.plot_dag(m, out_path="dag.png")

# Run 4 parallel chains from different seeds:
def build(seed):
    return mod.MyModel(y, X, seed=seed, keep_history=True)

chains = AI4BayesCode.run_chains(
    factory=build,
    seeds=[101, 202, 303, 404],
    n_burn=5000,
    n_keep=5000,
    n_jobs=4,
)
```

## Generate a sampler from natural language

`AI4BayesCode.generate()` drives an LLM to write **and validate** a sampler from
a prose model description (compile → 2-chain rank-R-hat → repair-on-failure). It
is **LLM-agnostic** — Claude (Opus 4.8 / Sonnet / Haiku) and OpenAI (GPT-5.5 /
Codex) — and you pass the API key, the model, and the thinking level **in the
call**; nothing is saved.

```python
res = AI4BayesCode.generate(
    "y ~ N(Xbeta, sigma^2), p(sigma^2) propto 1/sigma^2  (linear regression)",
    API_key="sk-YOUR-KEY-HERE",            # passed here, never saved
    LLM="gpt-5.5",               # or claude-opus-4-8, claude-sonnet-4-6, ...
    effort="high",               # thinking level, validated against this model's levels
    backend="Python", output_path="./generated", max_attempts=2)
print(res["validated"], res["cpp_path"])
```

Or **interactive** (menus for the model + thinking level, then prompts for
backend, output folder, class name, key, and a prior per parameter):

```python
AI4BayesCode.generate(interactive=True)
```

`AI4BayesCode.models()` lists the selectable models and their valid thinking
levels. The OpenAI path uses stdlib `urllib` (no extra SDK); the Anthropic path
uses the `anthropic` SDK, which is a core dependency (installed automatically).
With no key set, it writes the prompt to `PROMPT.txt`.

Prefer not to pass the key every call? Set it **once per session** (session-only,
never written to disk):

```python
AI4BayesCode.set_key("sk-YOUR-KEY-HERE", "openai")     # or "anthropic" / "google"
AI4BayesCode.key_status()                    # shows what's set (masked)
AI4BayesCode.generate("Linear regression.", LLM="gpt-5.5")  # key picked up
```

`API_key=` still works per call and **overrides** the session key for that one call —
e.g. `AI4BayesCode.generate("...", API_key="sk-other")` — without changing what
`set_key()` configured. Omit it to use the session default.

## .cpp requirements

AI4BayesCode examples ship with **dual** module declarations — one for R
(RCPP_MODULE) and one for Python (PYBIND11_MODULE) — guarded by
`#ifdef`. The Python helper compiles with `-DAI4BAYESCODE_PYBIND_MODULE`,
activating only the pybind11 block:

```cpp
#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(MyModel) {
    Rcpp::class_<MyModel>("MyModel")
        .constructor<...>()
        .method("step", &MyModel::step)
        ...
        ;
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
PYBIND11_MODULE(MyModel, m) {
    py::class_<MyModel>(m, "MyModel")
        .def(py::init<...>())
        .def("step", &MyModel::step)
        ...
        ;
}
#endif
```

See the bundled `AI4BayesCode/_examples/` directory (installed with the package)
for reference dual-module samplers.

## System requirements

Before running `pip install`, install the **Armadillo** C++ linear-algebra
library. It is a mandatory system dependency because `source.py` invokes the
compiler against the vendored AI4BayesCode headers, which require standalone
Armadillo headers at compile time.

| Platform | Command |
|----------|---------|
| macOS (Homebrew) | `brew install armadillo` |
| Debian / Ubuntu | `apt-get install libarmadillo-dev` |
| conda (any OS) | `conda install -c conda-forge armadillo` |

## Files

```
python/
├── AI4BayesCode/
│   ├── __init__.py         # public API
│   ├── source.py           # compile-on-demand pybind11 wrapper
│   ├── sourceCpp.py        # legacy alias for source.py
│   ├── generate.py         # LLM-driven sampler generation
│   ├── plot_dag.py          # networkx + matplotlib DAG viz
│   ├── run_chains.py       # multiprocessing parallel chains
│   ├── diagnose.py         # convergence diagnostics
│   ├── doc.py              # block documentation lookup
│   ├── utils.py            # rhat / ess / posterior summary
│   ├── _blocks.py          # block registry helpers
│   ├── start.md            # quick-start guide (shipped in wheel)
│   ├── _examples/          # reference dual-module samplers
│   ├── _skills/            # LLM skill corpus for generate()
│   └── _vendored_include/  # vendored AI4BayesCode C++ headers
├── setup.py
├── pyproject.toml
├── MANIFEST.in
├── LICENSE
├── THIRD_PARTY_LICENSES.md
└── README.md               # this file
```

## License

GPL-3.0-or-later. The combined work (AI4BayesCode plus its vendored
dependencies, including the Apache-2.0 mcmclib NUTS backend) must be
distributed under GPL-3 or later. See LICENSE and THIRD_PARTY_LICENSES.md
for details.

"""AI4BayesCode — Python helper for the AI4BayesCode MCMC library.

Public API mirrors the R helper (AI4BayesCode_helpers.R):

    sourceCpp(cpp_file, ai4bayescode_path)   # compile a .cpp with pybind11
    plot_dag(model, ...)                 # visualize the model DAG
    run_chains(model_ctor, ...)          # parallel multi-chain MCMC
    rhat(samples), ess_bulk(samples)     # convergence diagnostics

Typical workflow (self-contained -- no AI4BayesCode checkout needed, the
headers ship inside the installed package):

    import AI4BayesCode
    mod = AI4BayesCode.source("MyModel.cpp")   # file path ...
    mod = AI4BayesCode.source(cpp_source_str)  # ... or a string
    m = mod.MyModel(y, X, seed=1, keep_history=True)
    m.step(5000); m.step(5000)
    hist = m.get_history()
    AI4BayesCode.plot_dag(m)

The `.cpp` file must contain a `PYBIND11_MODULE(ModuleName, m) { ... }`
block. See `examples/ODE_SIR.cpp` for a dual-module reference.

`sourceCpp(...)` is retained as a back-compatibility alias (pass
`ai4bayescode_path=` only if you want the legacy checkout-based build).
"""

from .source import source, vendored_include_path
from .doc import doc
from .generate import prompt, generate, models, skills_path, set_key, key_status, stream_check
from .sourceCpp import sourceCpp
from .install_block import (
    install_block, available_blocks, installed_blocks, remove_block, blocks_path)
from .plot_dag import plot_dag
from .run_chains import run_chains
from .perf_hint import perf_hint
from .diagnose import diagnose
from .utils import rhat, ess_bulk, ess_tail, posterior_summary
from ._blocks import blocks

__version__ = "1.0.0"
__all__ = [
    "source",
    "doc",
    "blocks",
    "prompt",
    "generate",
    "models",
    "skills_path",
    "set_key",
    "key_status",
    "stream_check",
    "vendored_include_path",
    "sourceCpp",
    "install_block",
    "available_blocks",
    "installed_blocks",
    "remove_block",
    "blocks_path",
    "plot_dag",
    "run_chains",
    "perf_hint",
    "diagnose",
    "rhat",
    "ess_bulk",
    "ess_tail",
    "posterior_summary",
]

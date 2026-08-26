"""AI4BayesCode — Python helper for the AI4BayesCode MCMC library.

Public API mirrors the R package's exported helpers:

    plot_dag(model, ...)                 # visualize the model DAG
    run_chains(model_ctor, ...)          # parallel multi-chain MCMC
    rhat(samples), ess_bulk(samples)     # convergence diagnostics

Typical workflow (self-contained -- no AI4BayesCode checkout needed, the
headers ship inside the installed package):

    import AI4BayesCode
    mod = AI4BayesCode.source("MyModel.cpp")   # file path ...
    mod = AI4BayesCode.source(cpp_source_str)  # ... or a string
    m = mod.MyModel(y, X, rng_seed=1, keep_history=True)
    m.step(5000); m.step(5000)
    hist = m.get_history()
    AI4BayesCode.plot_dag(m)

The `.cpp` file must contain a `PYBIND11_MODULE(ModuleName, m) { ... }`
block. For a reference model, see AI4BayesCode.examples_path("ODE_SIR").
"""

# ---------------------------------------------------------------------------
# Pin the native BLAS to one thread, BEFORE anything can call into it.
#
# run_chains() parallelises with fork(). A multithreaded BLAS -- macOS
# Accelerate/vecLib above all, but also OpenBLAS with pthreads and MKL --
# keeps a worker-thread pool; fork() copies only the calling thread, so the
# child inherits a pool whose threads do not exist and the first heavy BLAS
# call there dies. On macOS the crash is inside libdispatch
# (`_dispatch_root_queue_push` under a LAPACK entry point, reported as
# "crashed on child side of fork pre-exec") and it takes every worker with it.
#
# The cap has to be in place BEFORE the first BLAS call, because that is when
# Accelerate decides whether to use its dispatch queues -- setting it later,
# for instance inside run_chains(), has no effect at all. Hence: here, at
# import, above every other import in this package.
#
# setdefault, not assignment: an explicit setting from the caller's
# environment wins.
#
# The cap costs nothing in practice. The chains are separate processes
# already, so n_jobs workers x k BLAS threads each would only oversubscribe
# the machine; measured on the GP examples, per-chain wall time is the same
# or slightly better at one thread.
import os as _os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
           "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
del _v, _os
# ---------------------------------------------------------------------------

from .source import source, vendored_include_path
from .example import example, list_examples, examples_path
from .doc import doc
from .generate import prompt, generate, models, skills_path, set_key, key_status, stream_check
from .install_block import (
    install_block, available_blocks, installed_blocks, remove_block, blocks_path)
from .plot_dag import plot_dag
from .run_chains import run_chains
from .perf_hint import perf_hint
from .diagnose import diagnose
from .rhat_summary import rhat_summary
from .utils import rhat, ess_bulk, ess_tail, posterior_summary
from .meta import version, include_path, list_skills
from ._blocks import blocks
from .new_frozen import new_frozen

__version__ = "1.0.0"
__all__ = [
    "source",
    "example",
    "list_examples",
    "examples_path",
    "list_skills",
    "include_path",
    "version",
    "rhat_summary",
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
    "new_frozen",
]

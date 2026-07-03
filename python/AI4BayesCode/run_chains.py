"""Parallel multi-chain MCMC orchestration.

Python equivalent of the R helper's `ai4bayescode_run_chains()`:

    chains = AI4BayesCode.run_chains(
        factory = lambda seed: MyModel(y, seed=seed, keep_history=True),
        seeds   = [101, 202, 303, 404],
        n_burn  = 5000,
        n_keep  = 5000,
        n_jobs  = 4,           # parallel worker processes
    )
    # chains is a list of dicts: {"hist": {...}, "wall": seconds, "seed": ...}

Why multiprocessing (and not threading): the pybind11 modules hold
Armadillo state per-instance and Rcpp/pybind11 instances are not
trivially picklable across threads without careful GIL handling.
Multiprocessing runs each chain in its own Python interpreter, which
side-steps all that. The downside is startup cost — each worker
imports the module fresh — but for any reasonable chain length that's
negligible.

On Unix (macOS / Linux) the chains FORK, so the factory is inherited
through memory and a lambda / local closure works directly -- exactly
like R's `mclapply(function(s) new(Mod, ..., s))`. On Windows (no fork)
the workers spawn a fresh interpreter, so there the factory must be a
PICKLABLE callable (a top-level function or plain class, not a lambda);
if it can't be pickled the run falls back to sequential automatically.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import warnings
from typing import Any, Callable, Iterable, Optional


def _chain_worker(factory: Callable[[int], Any], seed: int,
                  n_burn: int, n_keep: int,
                  history_keys: Optional[list[str]] = None) -> dict:
    """Run one chain. This is called in a fresh subprocess."""
    t0 = time.time()
    try:
        m = factory(seed)
        # Some models have step(n) that returns None; we just call and discard.
        if n_burn > 0:
            m.step(int(n_burn))
        if n_keep > 0:
            m.step(int(n_keep))
        wall = time.time() - t0
        hist = m.get_history() if hasattr(m, "get_history") else None
        if history_keys is not None and hist is not None:
            hist = {k: hist[k] for k in history_keys if k in hist}
        return {"seed": seed, "wall": wall, "hist": hist}
    except Exception as e:  # noqa: BLE001
        # NOTE: unlike R's ai4bayescode_run_chains (which stop()s the whole run
        # on a genuine chain failure), the Python driver degrades gracefully: a
        # failed chain warns loudly and returns a marked-failed result
        # (hist=None + "error") so the surviving chains stay usable.
        warnings.warn(f"run_chains: chain seed={seed} failed: {e}")
        return {"seed": seed, "wall": time.time() - t0, "hist": None,
                "error": str(e)}


# The active factory for the fork path. Set in the parent immediately before the
# workers fork; each forked child inherits it through memory, so it never has to
# be pickled (which is what lets a lambda / closure run in parallel -- mirroring
# R's mclapply). Only touched on the fork path and cleared right after.
_ACTIVE_FACTORY: Optional[Callable[[int], Any]] = None


def _chain_worker_fork(seed: int, n_burn: int, n_keep: int,
                       history_keys: Optional[list[str]] = None) -> dict:
    """Fork-path worker: read the inherited factory from the module global.

    Only the (picklable) seed/counts travel through the task queue; the factory
    comes from `_ACTIVE_FACTORY`, inherited from the parent at fork time.
    """
    if _ACTIVE_FACTORY is None:  # pragma: no cover - defensive
        raise RuntimeError("run_chains: no active factory in forked worker")
    return _chain_worker(_ACTIVE_FACTORY, seed, n_burn, n_keep, history_keys)


def run_chains(
    factory: Callable[[int], Any],
    *,
    seeds: Iterable[int] = (101, 202, 303, 404),
    n_burn: int = 2000,
    n_keep: int = 10000,
    n_jobs: Optional[int] = None,
    history_keys: Optional[list[str]] = None,
    verbose: bool = True,
) -> list[dict]:
    """Run `len(seeds)` chains in parallel (or sequentially if n_jobs=1).

    Parameters
    ----------
    factory : Callable[[int], model]
        A callable that builds a fresh model given a seed. On Unix
        (macOS / Linux) the chains fork, so a lambda or local closure
        works directly -- just like R's ``function(s) new(Mod, ..., s)``::

            chains = AI4BayesCode.run_chains(
                lambda s: MyModel(y, X, seed=s, keep_history=True),
                seeds=[1, 2, 3, 4])

        A top-level (picklable) function also works and is the only
        parallel-safe form on Windows (spawn). If the parallel pool
        cannot start, the chains fall back to running sequentially.

    seeds : iterable of int
        One seed per chain. Chain count = len(seeds).
    n_burn, n_keep : int
        Warmup and sampled iterations per chain.
    n_jobs : int, optional
        Number of parallel processes. Defaults to min(len(seeds),
        cpu_count()). Set to 1 for sequential execution.
    history_keys : list of str, optional
        Subset of history keys to return (saves memory for big chains).
        If None, return all keys from `get_history()`.
    verbose : bool, default True
        Print per-chain status.

    Returns
    -------
    list of dicts, one per chain: `{"seed", "wall", "hist"}`.
    """
    seeds = list(seeds)
    if not seeds:
        return []
    if n_burn < 0 or n_keep < 1:   # R parity: stopifnot(n_burn>=0, n_keep>=1)
        raise ValueError("n_burn must be >= 0 and n_keep >= 1")
    if n_jobs is None:
        n_jobs = min(len(seeds), os.cpu_count() or 1)

    args = [(factory, int(s), n_burn, n_keep, history_keys) for s in seeds]

    def _sequential() -> list[dict]:
        out = []
        for a in args:
            if verbose:
                print(f"[AI4BayesCode.run_chains] chain seed={a[1]} ...")
            out.append(_chain_worker(*a))
        return out

    if n_jobs == 1:
        return _sequential()

    use_fork = sys.platform != "win32" and "fork" in mp.get_all_start_methods()
    results: Optional[list[dict]] = None
    if use_fork:
        # Unix: FORK. The factory is handed to the workers through inherited
        # memory (the `_ACTIVE_FACTORY` global set just before the pool forks),
        # NOT through the task queue -- so only the picklable seed/counts are
        # serialized and a lambda / closure runs in true parallel, exactly like
        # R's mclapply. ProcessPoolExecutor (not multiprocessing.Pool) is used so
        # a crashed worker -- e.g. a fork-unsafe native BLAS like macOS
        # Accelerate -- raises BrokenProcessPool instead of hanging; we then fall
        # back to sequential, the same graceful degradation R does.
        from concurrent.futures import ProcessPoolExecutor
        global _ACTIVE_FACTORY
        _ACTIVE_FACTORY = factory
        try:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
                futs = [ex.submit(_chain_worker_fork, int(s),
                                  n_burn, n_keep, history_keys) for s in seeds]
                results = [f.result() for f in futs]
        except Exception as e:  # noqa: BLE001  (BrokenProcessPool, ...)
            warnings.warn(
                f"run_chains: parallel (fork) execution failed "
                f"({type(e).__name__}: {e}); running the {len(seeds)} chains "
                f"sequentially instead.")
            results = None
        finally:
            _ACTIVE_FACTORY = None
    else:
        # Windows (no fork): SPAWN a fresh interpreter, which requires the factory
        # to be picklable. A lambda / local closure raises PicklingError -- caught
        # below and run sequentially so the call still succeeds.
        try:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_jobs) as pool:
                results = pool.starmap(_chain_worker, args)
        except Exception as e:  # noqa: BLE001  (PicklingError, ...)
            warnings.warn(
                f"run_chains: parallel execution failed "
                f"({type(e).__name__}: {e}); running the {len(seeds)} chains "
                f"sequentially instead. On Windows pass a top-level picklable "
                f"factory (not a lambda / local closure), or use n_jobs=1.")
            results = None

    if results is None:
        results = _sequential()
    if verbose:
        walls = [r["wall"] for r in results]
        print(f"[AI4BayesCode.run_chains] walls (s): "
              + ", ".join(f"{w:.1f}" for w in walls))
    return results

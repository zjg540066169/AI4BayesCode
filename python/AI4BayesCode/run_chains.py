"""Parallel multi-chain MCMC orchestration.

Python equivalent of the R helper's `AI4BayesCode_run_chains()`:

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

Note: the model factory must be a PICKLABLE CALLABLE — a top-level
function or a plain class, NOT a lambda (unless using Python 3.12's
improved lambda pickling). If your factory can't be pickled, use
`n_jobs=1` for sequential execution.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Any, Callable, Iterable, Optional


def _chain_worker(factory: Callable[[int], Any], seed: int,
                  n_burn: int, n_keep: int,
                  history_keys: Optional[list[str]] = None) -> dict:
    """Run one chain. This is called in a fresh subprocess."""
    t0 = time.time()
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


def run_chains(
    factory: Callable[[int], Any],
    *,
    seeds: Iterable[int] = (101, 202, 303, 404),
    n_burn: int = 1000,
    n_keep: int = 1000,
    n_jobs: Optional[int] = None,
    history_keys: Optional[list[str]] = None,
    verbose: bool = True,
) -> list[dict]:
    """Run `len(seeds)` chains in parallel (or sequentially if n_jobs=1).

    Parameters
    ----------
    factory : Callable[[int], model]
        A pickle-friendly callable that builds a fresh model given a
        seed. Example::

            def build(seed):
                return MyModel(y, X, seed=seed, keep_history=True)
            chains = AI4BayesCode.run_chains(build, seeds=[1, 2, 3, 4])

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
    if n_jobs is None:
        n_jobs = min(len(seeds), os.cpu_count() or 1)

    args = [(factory, int(s), n_burn, n_keep, history_keys) for s in seeds]

    if n_jobs == 1:
        results = []
        for a in args:
            if verbose:
                print(f"[AI4BayesCode.run_chains] chain seed={a[1]} ...")
            results.append(_chain_worker(*a))
        return results

    ctx = mp.get_context("spawn")  # "spawn" = fresh interpreter, safest with pybind11
    with ctx.Pool(processes=n_jobs) as pool:
        results = pool.starmap(_chain_worker, args)
    if verbose:
        walls = [r["wall"] for r in results]
        print(f"[AI4BayesCode.run_chains] walls (s): "
              + ", ".join(f"{w:.1f}" for w in walls))
    return results

"""Parallel multi-chain MCMC orchestration.

Python counterpart of the R helper's `ai4bayescode_run_chains()`. The two are
NOT argument-compatible: R takes `n_chains` and returns an `ai4bayescode_run`
object (`run$histories`, `run$models`, ...); this takes `seeds` and returns a
`ChainsResult`, a list of per-chain dicts whose attribute `models` holds the
models. `n_chains=` is accepted as an alias that derives the seeds:

    chains = AI4BayesCode.run_chains(
        factory = lambda seed: MyModel(y, rng_seed=seed, keep_history=True),
        seeds   = [101, 202, 303, 404],
        n_burn  = 5000,
        n_keep  = 5000,
        n_cores = 4,           # parallel worker processes
    )
    # chains is a list of dicts: {"hist": {...}, "wall": seconds, "seed": ...};
    # chains.models[i] is the model of chain i
    pred = AI4BayesCode.predict(chains)   # predict_at() of every chain, pooled
    AI4BayesCode.close_models(chains)     # stop the worker processes

Each chain runs `step(n_burn)`, `clear_history()` and `step(n_keep)`, so the
model's own `get_history()` and `predict_at()` cover the n_keep kept draws; a
model without clear_history() (compiled against headers older than 1.1.3)
runs `step(n_burn)` and `step(n_keep)` as in 1.1.2.
With `keep_models=True` (the default) and parallel chains, the worker
processes are not stopped at the end of the run: `chains.models[i]` is a
`ChainModel` handle, and each method call on it runs in the worker process
that holds chain i, until `close_models(chains)`, until the result and its
handles are garbage-collected, or until the interpreter exits. At most
``AI4BAYESCODE_MAX_WORKERS`` such worker processes are kept over all runs of
the session (see :func:`run_chains`).

Why multiprocessing (and not threading): the pybind11 modules hold
Armadillo state per-instance and Rcpp/pybind11 instances are not
trivially picklable across threads without careful GIL handling.
Multiprocessing runs each chain in its own Python interpreter, which
side-steps all that. The downside is startup cost — each worker
imports the module fresh — but for any reasonable chain length that's
negligible.

On Unix (macOS / Linux) the chains FORK, so the factory is inherited
through memory and a lambda / local closure works directly -- exactly
like R's `mclapply(function(s) new(Mod, ..., s))`. Windows has no fork,
so there the chains run one after another by default, as in R. Pass
``n_cores > 1`` to spawn workers instead: each re-imports the calling
script, so the factory has to be a top-level function, everything it
uses has to be defined at module top level, and the call has to sit under
``if __name__ == "__main__":``. So does all other work the script does: a
worker runs whatever is outside the guard before its chain. When that cannot
work (a lambda, a factory from ``python -c`` or a notebook) the chains run
sequentially, and the models are objects in this process.
"""

from __future__ import annotations

import contextlib
import gc
import io
import itertools
import multiprocessing as mp
import multiprocessing.process as _mp_process
import numbers
import os
import pickle
import random
import signal
import sys
import threading
import time
import traceback
import uuid
import warnings
import weakref
from collections.abc import Mapping
from multiprocessing import util as _mp_util
from multiprocessing.connection import wait as _mp_wait
from typing import Any, Callable, Iterable, Optional

import numpy as np
from ._console import safe_print as print  # noqa: A001 (see _console.py)


# ---------------------------------------------------------------------------
# Warnings and console output
# ---------------------------------------------------------------------------
_PKG_DIR = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))


def _outside_stacklevel() -> int:
    """The ``stacklevel`` that makes ``warnings.warn()``, called in the
    function that calls this one, attribute the warning to the first frame
    outside this package (the caller's own code)."""
    level, f = 1, sys._getframe(1)
    while f is not None:
        fn = os.path.normcase(os.path.abspath(f.f_code.co_filename))
        if not fn.startswith(_PKG_DIR + os.sep):
            return level
        f = f.f_back
        level += 1
    return level


def _warn(message: str, category=UserWarning) -> None:
    """warnings.warn(), located at the caller's code outside this package."""
    warnings.warn(message, category, stacklevel=_outside_stacklevel())


def _write(stream, text: str) -> None:
    """Write ``text`` to ``stream``; characters the stream cannot encode are
    written as '?'."""
    if not text or stream is None:
        return
    try:
        stream.write(text)
    except UnicodeEncodeError:
        enc = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
    try:
        stream.flush()
    except Exception:  # noqa: BLE001
        pass


def _warning_records(records) -> list:
    """``(category, text)`` for each recorded warning. The category is the
    class itself when it pickles, else its qualified name."""
    out = []
    for w in records:
        cat = w.category
        try:
            pickle.dumps(cat)
            ref = cat
        except Exception:  # noqa: BLE001
            ref = (f"{getattr(cat, '__module__', '?')}."
                   f"{getattr(cat, '__qualname__', getattr(cat, '__name__', '?'))}")
        try:
            text = str(w.message)
        except Exception:  # noqa: BLE001
            text = repr(w.message)
        out.append((ref, text))
    return out


def _captured(fn):
    """Run ``fn()`` in a worker process with its printed output and warnings
    recorded. Returns ``(value, exception, cap)``: the exception is None when
    ``fn`` returned; ``cap`` is ``(stdout text, stderr text, [(category,
    text), ...])``, which `_relay` shows in the calling process.

    ``simplefilter("always")`` makes the record independent of the warning
    filters that the worker process inherited at fork time ("error" from
    ``-W error``, "ignore", or a recorder that was active in the caller).
    """
    out, err = io.StringIO(), io.StringIO()
    value = exc = None
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                value = fn()
        except Exception as e:  # noqa: BLE001
            exc = e
    try:
        return value, exc, (out.getvalue(), err.getvalue(), _warning_records(rec))
    finally:
        # The traceback of `exc` holds this frame: clear the local so that the
        # exception and this frame do not form a reference cycle.
        exc = None


def _warning_for(cat, text) -> Warning:
    if isinstance(cat, type) and issubclass(cat, Warning):
        try:
            return cat(text)
        except Exception:  # noqa: BLE001
            cat = getattr(cat, "__name__", str(cat))
    return UserWarning(f"{cat}: {text}")


def _relay(cap) -> None:
    """Show in this process the output and the warnings that `_captured`
    recorded in a worker process: the text on sys.stdout and sys.stderr, and
    each warning through ``warnings.warn``, so the warning filters of this
    process apply (an "error" filter raises it here)."""
    if not cap:
        return
    out, err, warns = cap
    _write(sys.stdout, out)
    _write(sys.stderr, err)
    for cat, text in warns:
        warnings.warn(_warning_for(cat, text), stacklevel=_outside_stacklevel())


def _name_chain(exc: BaseException, chain: int) -> BaseException:
    """Put "chain <chain>: " in front of the message of ``exc``, in place, so
    its type, ``__cause__`` and traceback are kept. An exception whose message
    is not its single string argument (KeyError, OSError, ...) gets a note
    instead. Returns ``exc``."""
    prefix = f"chain {chain}: "
    args = getattr(exc, "args", ())
    if (type(exc).__str__ is BaseException.__str__ and len(args) == 1
            and isinstance(args[0], str)):
        if not args[0].startswith(prefix):
            exc.args = (prefix + args[0],)
    else:
        try:
            exc.add_note(f"{prefix}raised by the model of chain {chain}")
        except Exception:  # noqa: BLE001
            pass
    return exc


# ---------------------------------------------------------------------------
# One chain
# ---------------------------------------------------------------------------
def _strip_burn(hist, n_burn, keys=None):
    """Drop the first ``n_burn`` draws along axis 0 of every history entry
    (only of the entries named in ``keys``, when ``keys`` is given).

    Entries may be 1-D ``(n,)``, 2-D ``(n, d)``, or higher-dimensional
    ``(n, r, c, ...)`` arrays; the slice runs along the FIRST axis only and
    keeps every other axis intact. An entry with ``<= n_burn`` rows is
    returned unchanged: with ``keep_history`` disabled ``get_history()``
    holds a single, already post-warmup, current draw -- emptying it would
    be worse than keeping it.
    """
    if hist is None or n_burn <= 0:
        return hist
    out = {}
    for k, v in hist.items():
        a = np.asarray(v)
        if (keys is not None and k not in keys) or a.ndim == 0 or a.shape[0] <= n_burn:
            out[k] = v
        else:
            out[k] = a[int(n_burn):]
    return out


def _seed_globals(seed: int) -> None:
    """Seed numpy's global generator (``np.random.*``) and the ``random``
    module from the chain seed, immediately before the chain's model is
    built, in every code path, as R's run_chains calls set.seed(seed)."""
    np.random.seed(int(seed) % 4294967296)
    random.seed(int(seed))


class _CallerRandomState:
    """Save the states of numpy's global generator and of the ``random``
    module on entry and restore them on exit, so that chains run in this
    process leave the caller's random-number streams unchanged."""

    def __enter__(self):
        self._np = np.random.get_state()
        self._py = random.getstate()
        return self

    def __exit__(self, *exc):
        np.random.set_state(self._np)
        random.setstate(self._py)
        return False


def _failed_result(seed, e, t0) -> dict:
    # Unlike R's ai4bayescode_run_chains, which stops the whole run on a
    # chain failure, a failed chain gives a warning and a result marked as
    # failed (hist=None, "error"), and the results of the other chains are
    # returned.
    _warn(f"run_chains: chain seed={seed} failed: {e}")
    return {"seed": seed, "wall": time.time() - t0, "hist": None,
            "error": str(e), "error_type": type(e).__name__}


def _run_chain(factory: Callable[[int], Any], seed: int,
               n_burn: int, n_keep: int,
               history_keys: Optional[list[str]] = None) -> tuple[dict, Any]:
    """Run one chain; return ``(result, model)``, with model None on failure.

    Besides "seed", "wall" and "hist", a successful result carries the
    private fields "_cleared", "_class", "_unclear" and "_no_clear", which
    `_finish` removes in the calling process after reporting on them once
    for the whole run.
    """
    t0 = time.time()
    try:
        _seed_globals(seed)
        m = factory(seed)
        # Some models have step(n) that returns None; we just call and discard.
        if n_burn > 0:
            m.step(int(n_burn))
        # clear_history() (AI4BayesCode >= 1.1.3) discards the stored draws
        # and leaves the sampler state unchanged, so after it the model's own
        # get_history() and predict_at() cover the n_keep kept sweeps only. It
        # is called only when the model has it: a model compiled against older
        # headers runs exactly as in 1.1.2.
        has_clear = callable(getattr(m, "clear_history", None))
        cleared = n_burn > 0 and has_clear
        if cleared:
            m.clear_history()
        if n_keep > 0:
            m.step(int(n_keep))
        wall = time.time() - t0
        hist = m.get_history() if hasattr(m, "get_history") else None
        unclear = []
        if cleared and hist is not None:
            # After the clear an entry holds n_keep rows (1 row when the model
            # stores a single draw). An entry with more rows was not emptied
            # by clear_history(), so the model still holds its warmup rows.
            unclear = [k for k, v in hist.items()
                       if np.ndim(v) > 0 and np.shape(v)[0] > n_keep]
        if history_keys is not None and hist is not None:
            hist = {k: hist[k] for k in history_keys if k in hist}
        # Without clear_history(), get_history() spans EVERY stepped iteration
        # (warmup + keep); return keep-only draws, matching this function's
        # documentation. After the clear, only the entries it left are cut.
        hist = _strip_burn(hist, int(n_burn), keys=unclear if cleared else None)
        no_clear = (n_burn > 0 and not has_clear
                    and (hasattr(m, "get_history") or hasattr(m, "predict_at")))
        return ({"seed": seed, "wall": wall, "hist": hist,
                 "_cleared": cleared, "_class": type(m).__name__,
                 "_unclear": unclear, "_no_clear": no_clear}, m)
    except Exception as e:  # noqa: BLE001
        return (_failed_result(seed, e, t0), None)


def _chain_task(factory: Callable[[int], Any], seed: int,
                n_burn: int, n_keep: int,
                history_keys: Optional[list[str]] = None) -> dict:
    """Run one chain in a worker process of the keep_models=False paths.

    The result carries the chain's printed output and warnings in the
    private field "_cap", which the calling process shows (`_relay`)."""
    value, exc, cap = _captured(
        lambda: _run_chain(factory, seed, n_burn, n_keep, history_keys)[0])
    if exc is not None:      # _run_chain returns a failed result instead
        value = {"seed": seed, "wall": 0.0, "hist": None, "error": str(exc),
                 "error_type": type(exc).__name__}
    exc = None
    value["_cap"] = cap
    return value


# The active factory for the fork path. Set in the parent immediately before the
# workers fork; each forked child inherits it through memory, so it never has to
# be pickled (which is what lets a lambda / closure run in parallel -- mirroring
# R's mclapply). Only touched on the fork path and cleared right after.
_ACTIVE_FACTORY: Optional[Callable[[int], Any]] = None


# Errors a spawned worker raises when it cannot rebuild the factory.
_REBUILD_ERRORS = ("NameError", "AttributeError", "ImportError", "ModuleNotFoundError")

_BLAS_THREAD_VARS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "OMP_NUM_THREADS", "MKL_NUM_THREADS")

#: Added to the warning of a run whose worker process ended during the run.
_BLAS_HINT = (
    " A worker process that ends during the run is often terminated by a "
    "multithreaded native BLAS, which does not work after fork() (macOS "
    "Accelerate in particular): set VECLIB_MAXIMUM_THREADS, "
    "OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and MKL_NUM_THREADS to \"1\" before "
    "starting Python to keep the chains parallel, or pass n_cores=1.")


def _warn_if_blas_unpinned() -> None:
    """Warn when the fork path is about to run with a multithreaded BLAS.

    `AI4BayesCode/__init__.py` pins these to one thread at import, which is
    the only place it can work: Accelerate determines whether its dispatch
    queues are used at the FIRST BLAS call, so a cap applied later -- here,
    for instance -- changes nothing. If the variable is not "1" by now, the
    caller overrode it, and a fork with a BLAS-heavy model can terminate every
    worker process.
    """
    unpinned = [v for v in _BLAS_THREAD_VARS
                if os.environ.get(v) not in (None, "1")]
    if not unpinned:
        return
    _warn(
        "run_chains: " + ", ".join(f"{v}={os.environ[v]}" for v in unpinned) +
        " is set, so the native BLAS may be multithreaded. fork() and a "
        "multithreaded BLAS do not mix -- on macOS Accelerate every worker "
        "process is terminated inside libdispatch and the run falls back to "
        "sequential. Set these to \"1\" before starting Python, or pass n_cores=1.")


def _chain_worker_fork(seed: int, n_burn: int, n_keep: int,
                       history_keys: Optional[list[str]] = None) -> dict:
    """Fork-path worker: read the inherited factory from the module global.

    Only the (picklable) seed/counts travel through the task queue; the factory
    comes from `_ACTIVE_FACTORY`, inherited from the parent at fork time.
    """
    if _ACTIVE_FACTORY is None:  # pragma: no cover - defensive
        raise RuntimeError("run_chains: no active factory in forked worker")
    return _chain_task(_ACTIVE_FACTORY, seed, n_burn, n_keep, history_keys)


#: Sentinel default for `seeds`. Compared BY IDENTITY, so a caller who passes
#: these very values explicitly alongside n_chains keeps their seeds -- an
#: equality test made that depend on both the values and the container type
#: (a tuple lost, the same values in a list won).
_DEFAULT_SEEDS = (101, 202, 303, 404)


def _spawn_obstacle(factory) -> str:
    """Why spawned workers could not rebuild ``factory``, or "" if they can."""
    try:
        pickle.dumps(factory)
    except Exception as e:  # noqa: BLE001
        return (f"the factory cannot be sent to worker processes ({type(e).__name__}); "
                "on Windows use a top-level function, not a lambda or closure")
    if getattr(factory, "__module__", None) == "__main__":
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        if not main_file or not os.path.isfile(main_file):
            return ("the factory is defined in a session that has no script file "
                    "(python -c, standard input or a notebook), which worker "
                    "processes cannot import")
    return ""


# ---------------------------------------------------------------------------
# Persistent worker processes (keep_models=True with parallel chains).
#
# A worker process runs its chains, sends one message per chain, and then
# serves method calls on the models it holds, so predict_at() and every other
# model method remain available after the run. The messages are pickled
# tuples on one duplex Pipe per worker:
#
#   worker -> parent, once per chain:  ("chain", i, result, cap)
#   parent -> worker:                  ("call", i, method, args, kwargs)
#                                      or ("close",)
#   worker -> parent, once per call:   ("ok", value, cap)
#                                      or ("err", pickled exception or None,
#                                          (type name, message), traceback
#                                          text, cap)
#
# `cap` holds the printed output and the warnings of the chain run or of the
# call (see `_captured`); the parent shows them before it returns the value or
# raises the error (`_relay`).
#
# A worker handles one call at a time. The parent sends a worker its next call
# only after it has read the reply to the previous one, under that worker's
# lock; calls to different workers run concurrently.
#
# The worker processes are not daemonic, so a model may start processes of
# its own. A multiprocessing.util.Finalize with an exit priority stops them:
# when close_models() runs, when the pool is garbage-collected, and at
# interpreter exit, where multiprocessing's exit function runs it before it
# joins the remaining child processes.
# ---------------------------------------------------------------------------

_CLOSED_MSG = ("the worker processes holding the models of this run were closed "
               "(close_models)")

_DETACHED_PICKLED = (
    "this model handle was restored from a pickle in a process other than the one "
    "that ran run_chains(), or after the worker processes of its run were "
    "garbage-collected, so the model cannot be called here (the histories in the "
    "result are intact)")

#: Pools whose worker processes may still run. A forked worker closes its
#: inherited copies of their connection ends (see `_worker_main`).
_LIVE_POOLS: "weakref.WeakSet[_WorkerPool]" = weakref.WeakSet()

#: The pools of this process by token, so that a handle unpickled in the
#: process that ran the chains refers to the same pool (`_restore_chain_model`).
_POOLS_BY_TOKEN: "weakref.WeakValueDictionary[str, _WorkerPool]" = \
    weakref.WeakValueDictionary()

#: Order of use of the pools: creation, and every call (least recent use first
#: when the worker processes of earlier runs are closed, see `_make_room`).
_USE_COUNTER = itertools.count()


def _drop_workers_in_child() -> None:
    """At-fork hook (child side). A child made by os.fork() inherits the
    Process objects of the worker processes in multiprocessing's set of child
    processes; at the end of the child, multiprocessing's exit function would
    send them SIGTERM (as daemonic processes) or join them (an
    AssertionError). After they are removed from the child's set, only the
    parent process stops the worker processes."""
    try:
        children = _mp_process._children
        for pool in list(_LIVE_POOLS):
            for w in pool.workers:
                children.discard(w.process)
    except Exception:  # noqa: BLE001
        pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_drop_workers_in_child)


def _dumps(obj) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


_CLOSE_REQUEST = _dumps(("close",))


def _exiting_daemon() -> bool:
    """True in a daemon thread during interpreter exit (the main thread has
    ended), when the atexit callbacks stop the worker processes."""
    return (threading.current_thread().daemon
            and not threading.main_thread().is_alive())


def _block_forever() -> None:
    """Block the calling daemon thread until the process exits.

    During interpreter exit, a call that fails because the worker processes
    were stopped would return to the daemon thread's own code, and the
    traceback that the thread then writes to stderr while the main thread
    finalizes the interpreter can abort CPython ("could not acquire lock for
    <_io.BufferedWriter name='<stderr>'> at interpreter shutdown"). Blocking
    the thread here leaves it as it would be without the workers' shutdown.
    """
    threading.Event().wait()


class _WorkerLost(Exception):
    """A worker process exited while a message from it was still due."""


class _RemoteTraceback(Exception):
    """``__cause__`` of an exception re-raised from a worker process; its text
    is the traceback of the exception in that process."""

    def __init__(self, tb: str):
        super().__init__(tb)
        self.tb = tb

    def __str__(self) -> str:
        return self.tb


class _Fields(dict):
    """The fields of a pybind11 record (a DagInfo, an AdaptationInfo) returned
    by a ChainModel call. The record does not pickle, so the worker process
    sends its fields together with the module and the name of its type; this
    process rebuilds the record from them (`_rebuild_record`). When that is
    not possible (the module is not imported here) the value is this dict,
    whose item access and attribute access both return a field, so
    ``get_dag()["predict_edges"]``, ``get_dag().predict_edges`` and
    ``plot_dag()`` work as on the record."""

    def __getattr__(self, name):
        if name.startswith("__") or name == "_record":
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __reduce__(self):
        rec = self.__dict__.get("_record")
        if rec is not None:
            return (_rebuild_record, (rec[0], rec[1], dict(self)))
        return (_Fields, (dict(self),))


def _rebuild_record(module, qualname, fields):
    """The pybind11 record of type ``module.qualname`` with ``fields``, or a
    `_Fields` dict of them when the type cannot be found or built here."""
    try:
        obj = sys.modules[module]
        for part in qualname.split("."):
            obj = getattr(obj, part)
        rec = obj()
        for k, v in fields.items():
            setattr(rec, k, v)
        return rec
    except Exception:  # noqa: BLE001
        out = _Fields(fields)
        out._record = (module, qualname)
        return out


def _is_record(value) -> bool:
    """True for a pybind11 object whose public attributes are data fields
    (plus a keys() method), such as DagInfo and AdaptationInfo."""
    if type(type(value)).__name__ != "pybind11_type":
        return False
    names = [a for a in dir(value) if not a.startswith("_")]
    return bool(names) and all(a == "keys" or not callable(getattr(value, a, None))
                               for a in names)


def _with_record_fields(value, depth=0):
    """``value`` with every pybind11 record in it (also inside dicts, lists
    and tuples) replaced by a `_Fields` of its fields that records the type."""
    if depth > 50:
        return value
    if isinstance(value, dict):
        return {k: _with_record_fields(v, depth + 1) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_with_record_fields(v, depth + 1) for v in value)
    if _is_record(value):
        keys = getattr(value, "keys", None)
        names = (list(keys()) if callable(keys)
                 else [a for a in dir(value) if not a.startswith("_")])
        out = _Fields({a: _with_record_fields(getattr(value, a), depth + 1)
                       for a in names})
        out._record = (type(value).__module__, type(value).__qualname__)
        return out
    return value


class _DeferSigint:
    """Hold back SIGINT (KeyboardInterrupt) until the block ends.

    Used around writing or reading one message on a worker connection, so an
    interrupt never leaves a message half sent or half received. Python raises
    KeyboardInterrupt only in the main thread, so in any other thread this
    does nothing.
    """

    def __enter__(self):
        self._old = self._hit = None
        if threading.current_thread() is threading.main_thread():
            try:
                old = signal.getsignal(signal.SIGINT)
                if callable(old):
                    signal.signal(signal.SIGINT, self._record)
                    self._old = old
            except (ValueError, OSError, AttributeError):  # pragma: no cover
                pass
        return self

    def _record(self, signum, frame):
        self._hit = (signum, frame)

    def __exit__(self, *exc):
        if self._old is not None:
            signal.signal(signal.SIGINT, self._old)
            if self._hit is not None:
                self._old(*self._hit)
        return False


def _worker_main(conn, parent_end, chain_ids, seeds, n_burn, n_keep,
                 history_keys, factory, factory_blob) -> None:
    """Body of a persistent worker process.

    Runs the chains ``chain_ids`` in order and sends ("chain", i, result,
    cap) for each; then receives ("call", i, method, args, kwargs) requests,
    calls the method on chain i's model and sends the reply, until it
    receives a ("close",) request or the connection reaches end-of-file.
    Under fork, ``factory`` is the factory itself (inherited through memory);
    under spawn it is None and ``factory_blob`` holds the pickled factory.
    """
    # Ctrl+C in a terminal sends SIGINT to the whole process group. This
    # process holds models that the parent still calls, so it ignores SIGINT;
    # the parent stops its workers when an interrupt aborts the run.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):  # pragma: no cover
        pass
    if parent_end is not None:
        # Forked: this process holds copies of the parent's end of its own pipe
        # and of the pipes of the workers of every other open pool. Closing
        # them makes recv_bytes() in each worker raise EOFError once the parent
        # process has exited, with or without a "close" request.
        parent_end.close()
        for pool in list(_LIVE_POOLS):
            for w in pool.workers:
                try:
                    w.conn.close()
                except Exception:  # noqa: BLE001
                    pass
    build_error = None
    if factory is None:
        try:
            factory = pickle.loads(factory_blob)
        except Exception as e:  # noqa: BLE001
            build_error = e
    models = {}
    for i, seed in zip(chain_ids, seeds):
        if build_error is not None:
            value, exc, cap = _captured(
                lambda: (_failed_result(seed, build_error, time.time()), None))
        else:
            value, exc, cap = _captured(
                lambda: _run_chain(factory, seed, n_burn, n_keep, history_keys))
        if exc is not None:      # _run_chain returns a failed result instead
            value = ({"seed": seed, "wall": 0.0, "hist": None, "error": str(exc),
                      "error_type": type(exc).__name__}, None)
        exc = None
        res, model = value
        value = None
        if model is not None:
            res["_methods"] = [a for a in dir(model) if not a.startswith("_")]
        try:
            buf = _dumps(("chain", i, res, cap))
        except Exception as e:  # noqa: BLE001
            model = None
            buf = _dumps(("chain", i, {
                "seed": seed, "wall": res.get("wall", 0.0), "hist": None,
                "error": ("the result of this chain cannot be sent from the worker "
                          f"process ({type(e).__name__}: {e})"),
                "error_type": type(e).__name__}, cap))
        if model is not None:
            models[i] = model
        model = res = None
        try:
            conn.send_bytes(buf)
        except OSError:
            return
        buf = None
    factory = None
    while True:
        try:
            buf = conn.recv_bytes()
        except (EOFError, OSError):
            break
        name = "the method"
        cap = None
        try:
            req = pickle.loads(buf)
            if req[0] == "close":
                break
            _, i, name, args, kwargs = req

            def call(i=i, name=name, args=args, kwargs=kwargs):
                if i not in models:
                    raise RuntimeError(
                        f"chain {i} has no model in worker process {os.getpid()}")
                return getattr(models[i], name)(*args, **kwargs)

            value, exc, cap = _captured(call)
        except Exception as e:  # noqa: BLE001  (a request that does not unpickle)
            value, exc = None, e
        if exc is None:
            reply = ("ok", value, cap)
        else:
            try:
                blob = _dumps(exc)
            except Exception:  # noqa: BLE001
                blob = None
            reply = ("err", blob, (type(exc).__name__, str(exc)),
                     "".join(traceback.format_exception(exc)), cap)
        exc = req = None
        try:
            out = _dumps(reply)
        except Exception as e:  # noqa: BLE001
            out = None
            if reply[0] == "ok":
                # A pybind11 record (get_dag() returns a DagInfo) does not
                # pickle; its fields and its type name do.
                try:
                    out = _dumps(("ok", _with_record_fields(reply[1]), cap))
                except Exception:  # noqa: BLE001
                    out = None
            if out is None:
                out = _dumps(("err", None, (type(e).__name__,
                              f"the value returned by {name}() cannot be sent to the "
                              f"calling process ({e})"), "", cap))
        value = reply = None
        try:
            conn.send_bytes(out)
        except OSError:
            break
        out = buf = None
    conn.close()


def _reply_value(buf, name, chain) -> tuple[bool, Any, Any]:
    """``(True, value, cap)`` or ``(False, exception, cap)`` from one reply
    message of chain ``chain``; the exception's message starts with
    "chain <chain>: "."""
    try:
        reply = pickle.loads(buf)
    except Exception as e:  # noqa: BLE001
        return (False, _name_chain(RuntimeError(
            f"the value returned by {name}() cannot be unpickled in this process "
            f"({type(e).__name__}: {e})"), chain), None)
    if reply[0] == "ok":
        return (True, reply[1], reply[2])
    _, blob, (tname, msg), tb, cap = reply
    exc = None
    if blob is not None:
        try:
            exc = pickle.loads(blob)
        except Exception:  # noqa: BLE001
            exc = None
    if not isinstance(exc, BaseException):
        exc = RuntimeError(f"{tname}: {msg}")
    if tb:
        exc.__cause__ = _RemoteTraceback(tb)
    return (False, _name_chain(exc, chain), cap)


def _closed_msg(state) -> str:
    return state.get("reason") or _CLOSED_MSG


class _Worker:
    """Parent-side record of one persistent worker process."""

    def __init__(self, index, process, conn, chains, state):
        self.index = index
        self.process = process
        self.pid = process.pid
        self.conn = conn
        self.chains = list(chains)
        self.state = state          # shared with the pool: {"closed", "reason"}
        self.lock = threading.Lock()
        self.unread = 0             # sent calls whose replies are not read yet
        self.lost = None            # error message after an unexpected exit

    def gone(self) -> str:
        """The error message for a call that cannot reach this worker."""
        if self.state["closed"]:
            return _closed_msg(self.state)
        if self.lost is None:
            code = None
            try:
                self.process.join(0.5)
                code = self.process.exitcode
            except Exception:  # noqa: BLE001
                pass
            self.lost = (f"the worker process {self.pid} holding chain(s) "
                         f"{', '.join(map(str, self.chains))} exited unexpectedly"
                         + ("" if code is None else f" (exit code {code})")
                         + "; the models of these chains are lost")
        return self.lost

    # `unread` changes inside the _DeferSigint block, together with the message
    # it counts: a KeyboardInterrupt raised when the block ends then leaves it
    # equal to the number of replies still in the connection.
    def send(self, payload) -> None:
        try:
            with _DeferSigint():
                self.conn.send_bytes(payload)
                self.unread += 1
        except (OSError, EOFError):
            self.unread = 0
            raise _WorkerLost(self.gone()) from None

    def receive(self) -> bytes:
        try:
            with _DeferSigint():
                buf = self.conn.recv_bytes()
                self.unread -= 1
        except (OSError, EOFError):
            self.unread = 0
            raise _WorkerLost(self.gone()) from None
        return buf

    def drain(self) -> None:
        """Read and discard the replies to calls whose caller was interrupted
        (KeyboardInterrupt) before it read the reply."""
        while self.unread > 0:
            try:
                self.conn.poll(None)
            except (OSError, EOFError):
                self.unread = 0
                raise _WorkerLost(self.gone()) from None
            self.receive()


def _shutdown_workers(workers, state, owner_pid) -> None:
    """Stop the worker processes of a pool. Run once, through the pool's
    multiprocessing.util.Finalize: by close(), when the pool is
    garbage-collected, or at interpreter exit."""
    if os.getpid() != owner_pid:
        return          # a forked copy of the pool: the workers belong to the parent
    state["closed"] = True
    # A worker that receives the "close" request leaves its request loop and
    # exits. A worker that is computing a call (its lock is still held after
    # one second in all, or a reply has not been read yet) is terminated
    # without the request.
    asked = []
    lock_deadline = time.monotonic() + 1.0
    for w in workers:
        got = w.lock.acquire(timeout=max(0.0, lock_deadline - time.monotonic()))
        sent = False
        try:
            if got and w.lost is None and w.unread == 0 and not w.conn.closed:
                with _DeferSigint():
                    w.conn.send_bytes(_CLOSE_REQUEST)
                sent = True
        except Exception:  # noqa: BLE001
            pass
        finally:
            if got:
                w.lock.release()
        asked.append(sent)
    deadline = time.monotonic() + 2.0
    for w, sent in zip(workers, asked):
        try:
            if sent:
                w.process.join(max(0.0, deadline - time.monotonic()))
            if w.process.exitcode is None:
                w.process.terminate()
        except Exception:  # noqa: BLE001
            pass
    for w in workers:
        p = w.process
        try:
            if p.exitcode is None:
                p.join(1.0)
            if p.exitcode is None:
                p.kill()
                p.join(1.0)
        except Exception:  # noqa: BLE001
            pass
    for w in workers:
        p = w.process
        # The parent's end is closed only under the worker's lock: a call in
        # another thread that is blocked in poll() on this descriptor returns
        # (end-of-file of the exited worker) and releases the lock first;
        # closing the descriptor under it would leave that poll() blocked.
        if w.lock.acquire(timeout=5.0):
            try:
                w.conn.close()
            except Exception:  # noqa: BLE001
                pass
            finally:
                w.lock.release()
        # Process.close() releases the two pipe descriptors that
        # multiprocessing keeps for each started process, and removes it from
        # multiprocessing's set of child processes. It raises ValueError for a
        # process that is still running.
        try:
            p.close()
        except Exception:  # noqa: BLE001
            pass


class _WorkerPool:
    """The persistent worker processes of one run_chains() call."""

    def __init__(self):
        self.owner_pid = os.getpid()
        self.token = uuid.uuid4().hex
        self.workers: list[_Worker] = []
        self.by_chain: dict[int, _Worker] = {}
        self.state = {"closed": False, "reason": None}
        self.last_used = next(_USE_COUNTER)
        # The finalizer holds the workers and the state, not the pool, so the
        # pool is garbage-collected once the result and every handle are gone.
        self._finalizer = _mp_util.Finalize(
            self, _shutdown_workers, args=(self.workers, self.state, self.owner_pid),
            exitpriority=10)
        _LIVE_POOLS.add(self)
        _POOLS_BY_TOKEN[self.token] = self

    @property
    def closed(self) -> bool:
        return self.state["closed"]

    def add(self, process, conn, chains) -> None:
        w = _Worker(len(self.workers), process, conn, chains, self.state)
        self.workers.append(w)
        for i in w.chains:
            self.by_chain[i] = w

    def close(self, reason: Optional[str] = None) -> None:
        if reason is not None and not self.state["closed"]:
            self.state["reason"] = reason
        self._finalizer()

    def collect(self, n) -> tuple[list[dict], list]:
        """Receive the results of the n chains, one message per chain;
        returns the results and the captured output and warnings."""
        results: list = [None] * n
        caps: list = [None] * n
        left = {w.conn: len(w.chains) for w in self.workers if w.chains}
        by_conn = {w.conn: w for w in self.workers}
        while left:
            for conn in _mp_wait(list(left)):
                w = by_conn[conn]
                try:
                    buf = conn.recv_bytes()
                except (EOFError, OSError):
                    code = None
                    try:
                        w.process.join(0.5)
                        code = w.process.exitcode
                    except Exception:  # noqa: BLE001
                        pass
                    raise _WorkerLost(
                        f"worker process {w.pid} exited (exit code {code}) before "
                        "it returned all of its chains") from None
                _, i, res, cap = pickle.loads(buf)
                results[i], caps[i] = res, cap
                left[conn] -= 1
                if not left[conn]:
                    del left[conn]
        return results, caps

    def call(self, chain, name, args, kwargs):
        (ok, value, cap), = self.call_many([(chain, name, args, kwargs)])
        _relay(cap)
        if ok:
            return value
        try:
            raise value
        finally:
            value = None    # no reference cycle through this frame's traceback

    def call_many(self, requests) -> list[tuple[bool, Any, Any]]:
        """Run each ``(chain, method, args, kwargs)`` in the worker holding the
        chain; return ``(True, value, cap)`` or ``(False, exception, cap)``
        per request, in request order. Workers process their requests
        concurrently. An exception raised by a model's method has
        "chain <chain>: " in front of its message."""
        if os.getpid() != self.owner_pid:
            raise RuntimeError(
                f"these model handles belong to process {self.owner_pid}; its worker "
                f"processes cannot be called from process {os.getpid()}")
        try:
            if self.state["closed"]:
                raise RuntimeError(_closed_msg(self.state))
            self.last_used = next(_USE_COUNTER)
            out, cut = self._exchange(requests)
        except BaseException:
            if _exiting_daemon():
                _block_forever()
            raise
        if cut and _exiting_daemon():
            _block_forever()
        return out

    def _exchange(self, requests):
        """The body of `call_many`; also returns whether a request failed
        because its worker process was stopped or exited."""
        out: list = [None] * len(requests)
        cut = False
        queues: dict[_Worker, list] = {}
        for k, (chain, name, args, kwargs) in enumerate(requests):
            w = self.by_chain.get(chain)
            if w is None:
                out[k] = (False, RuntimeError(
                    f"chain {chain} has no model in the worker processes of this run"),
                    None)
                continue
            try:
                payload = _dumps(("call", chain, name, tuple(args), dict(kwargs)))
            except Exception as e:  # noqa: BLE001
                out[k] = (False, TypeError(
                    f"the arguments of {name}() cannot be sent to the worker process "
                    f"({type(e).__name__}: {e})"), None)
                continue
            queues.setdefault(w, []).append((k, chain, name, payload))
        order = sorted(queues, key=lambda w: w.index)
        active: dict = {}

        def send_next(w):
            nonlocal cut
            while queues[w]:
                k, chain, name, payload = queues[w].pop(0)
                try:
                    if w.lost is not None:
                        raise _WorkerLost(w.lost)
                    w.drain()
                    w.send(payload)
                except _WorkerLost as e:
                    out[k] = (False, RuntimeError(str(e)), None)
                    cut = True
                    continue
                active[w.conn] = (w, k, chain, name)
                return

        held = []
        try:
            for w in order:             # one fixed order, so threads cannot deadlock
                w.lock.acquire()
                held.append(w)
            if self.state["closed"]:
                raise RuntimeError(_closed_msg(self.state))
            for w in order:
                send_next(w)
            while active:
                try:
                    ready = _mp_wait(list(active))
                except (OSError, ValueError):
                    # close() from another thread closed the connections.
                    if self.state["closed"]:
                        raise RuntimeError(_closed_msg(self.state)) from None
                    raise
                for conn in ready:
                    w, k, chain, name = active.pop(conn)
                    try:
                        out[k] = _reply_value(w.receive(), name, chain)
                    except _WorkerLost as e:
                        out[k] = (False, RuntimeError(str(e)), None)
                        cut = True
                    send_next(w)
        finally:
            for w in held:
                w.lock.release()
        return out, cut


def _detached_chain_model(chain, cls_name, methods, reason):
    return ChainModel(None, chain, None, cls_name, methods, detached=reason)


def _restore_chain_model(token, owner_pid, chain, pid, cls_name, methods):
    """Unpickle a handle: in the process that ran the chains, while its pool
    exists, a handle on the same model; elsewhere a detached handle."""
    pool = _POOLS_BY_TOKEN.get(token)
    if pool is not None and pool.owner_pid == owner_pid == os.getpid():
        return ChainModel(pool, chain, pid, cls_name, methods)
    return _detached_chain_model(chain, cls_name, methods, _DETACHED_PICKLED)


class ChainModel:
    """Handle on the model of one chain, held by a run_chains() worker process.

    ``handle.name(*args, **kwargs)`` sends the call to the worker process
    holding the model, runs ``model.name(*args, **kwargs)`` there and returns
    the result. The output that the call prints and the warnings it issues are
    shown in this process. An exception raised there is raised here with
    "chain i: " in front of its message (a note for an exception whose message
    is not its single argument, such as KeyError): with its own type when it
    can be pickled, else as RuntimeError carrying its type name and message;
    its ``__cause__`` holds the traceback text from the worker. A returned
    pybind11 record, which does not pickle (``get_dag()`` returns a DagInfo),
    is rebuilt here as the same record type when its module is imported in
    this process, and is otherwise returned as a dict of its fields that also
    allows attribute access. Calls from several threads run one at a time per
    worker process. Names that start with "_" are not forwarded.

    After :func:`close_models` every call raises RuntimeError, and so does a
    call after the worker processes were closed to start the workers of a
    later run (the error states this). A handle unpickled in the process that
    ran the chains refers to the same model while the run's worker processes
    exist; one unpickled in another process is detached: its calls raise
    RuntimeError, because the worker processes belong to the process that
    started them.
    """

    def __init__(self, pool, chain, pid, cls_name, methods, detached=None):
        self._pool = pool
        self._chain = chain
        self._pid = pid
        self._cls = cls_name
        self._methods = None if methods is None else frozenset(methods)
        self._detached = detached

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._methods is not None and name not in self._methods:
            raise AttributeError(
                f"the {self._cls} model of chain {self._chain} has no attribute {name!r}")

        def method(*args, **kwargs):
            if self._detached is not None:
                raise RuntimeError(f"chain {self._chain}: {self._detached}")
            return self._pool.call(self._chain, name, args, kwargs)

        method.__name__ = name
        method.__qualname__ = f"{self._cls}.{name}"
        return method

    def __setattr__(self, name, value):
        # A public attribute set here would shadow the forwarded method of the
        # same name and would not reach the model in the worker process.
        if not name.startswith("_"):
            raise AttributeError(
                f"cannot set {name!r}: the {self._cls} model of chain {self._chain} "
                "is in a worker process, and only its methods can be called")
        object.__setattr__(self, name, value)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._methods or ()))

    def __repr__(self) -> str:
        if self._detached is not None:
            where = "detached"
        else:
            where = f"worker process {self._pid}"
            w = self._pool.by_chain.get(self._chain)
            if self._pool.closed:
                where += " (closed)"
            elif w is not None and w.lost is not None:
                where += " (exited)"
        return f"<ChainModel chain {self._chain} ({self._cls}), {where}>"

    def __reduce__(self):
        methods = None if self._methods is None else sorted(self._methods)
        if self._detached is not None:
            return (_detached_chain_model,
                    (self._chain, self._cls, methods, self._detached))
        return (_restore_chain_model, (self._pool.token, self._pool.owner_pid,
                                       self._chain, self._pid, self._cls, methods))

    # A handle refers to a live model; a copy refers to the same model.
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _portable_model(m, chain):
    """``m`` when it pickles (a handle always does), else a detached handle
    that takes its place in a pickled result."""
    if m is None or isinstance(m, ChainModel):
        return m
    try:
        _dumps(m)
        return m
    except Exception:  # noqa: BLE001
        cls = type(m).__name__
        return ChainModel(
            None, chain, None, cls, [a for a in dir(m) if not a.startswith("_")],
            detached=(f"the {cls} model object cannot be pickled, so the unpickled "
                      "result holds this placeholder in its place (the histories "
                      "are intact)"))


def _pools_of(models) -> list:
    """The worker pools of the handles in ``models``, in chain order."""
    pools = []
    for m in models or ():
        if (isinstance(m, ChainModel) and m._detached is None
                and not any(m._pool is q for q in pools)):
            pools.append(m._pool)
    return pools


def _models_where(models) -> str:
    """Where the models of a run are, for the summary that repr() prints."""
    if models is None:
        return "not kept (keep_models=False)"
    kept = [m for m in models if m is not None]
    if not kept:
        return "none (every chain failed)"
    handles = [m for m in kept if isinstance(m, ChainModel)]
    if not handles:
        return "in this Python session (chains.models[i])"
    pools = _pools_of(handles)
    if not pools:
        return "detached (restored from a pickle; method calls raise RuntimeError)"
    if all(p.closed for p in pools):
        reasons = {p.state.get("reason") for p in pools}
        if reasons == {None}:
            return "closed (close_models)"
        return ("closed by a later run_chains() call to start its own worker "
                "processes (see the warning of that call)")
    if any(p.closed for p in pools):
        return "partly closed (close_models, or a later run_chains() call)"
    pids = [w.pid for p in pools for w in p.workers]
    return (f"in {len(pids)} worker process{'' if len(pids) == 1 else 'es'} of this "
            f"Python session (pid {', '.join(map(str, pids))}); "
            "chains.models[i].<method>(...) is evaluated there, and "
            "close_models(chains) terminates them")


def _rebuild_chains_result(chains, n_burn, n_keep, history_cleared, models=None):
    return ChainsResult(chains, n_burn=n_burn, n_keep=n_keep,
                        history_cleared=history_cleared, models=models)


class ChainsResult(list):
    """The value of :func:`run_chains`: a list with one dict per chain.

    Each dict has the keys of AI4BayesCode 1.1.2: "seed", "wall" (seconds)
    and "hist" (the chain's draws after the warmup; None for a failed chain,
    which also has "error" and "error_type").

    Attributes:

    ``models``
        With ``keep_models=True``, a list with one entry per chain: a
        :class:`ChainModel` handle on the model held by a worker process
        (parallel run), the model object (sequential run), or None for a
        failed chain. None with ``keep_models=False``. It is the Python
        counterpart of R's ``run$models``.
    ``n_burn``, ``n_keep``
        The warmup and kept sweeps of each chain.
    ``history_cleared``
        One bool per chain, True when clear_history() was called after the
        warmup, so the model's own get_history() and predict_at() cover the
        n_keep kept draws only.

    Methods :meth:`predict` and :meth:`close`. ``repr()`` (and ``print()``)
    shows a summary instead of the draws; the dicts are ``chains[i]``. A
    slice or ``list(chains)`` is a plain list of the dicts, without the
    models.

    A pickled result keeps its histories. A model handle unpickled in the
    process that ran the chains refers to the same model; one unpickled in
    another process is detached (its calls raise RuntimeError), and a model
    object that cannot be pickled is replaced by such a detached handle.
    """

    def __init__(self, chains=(), *, n_burn=0, n_keep=0, history_cleared=None,
                 models=None):
        super().__init__(chains)
        self.n_burn = int(n_burn)
        self.n_keep = int(n_keep)
        self.history_cleared = (list(history_cleared) if history_cleared is not None
                                else [False] * len(self))
        self.models = None if models is None else list(models)

    def predict(self, newdata=None, pool=True):
        """``predict(self, newdata, pool)``; see :func:`predict`."""
        return predict(self, newdata=newdata, pool=pool)

    def close(self) -> None:
        """``close_models(self)``; see :func:`close_models`."""
        close_models(self)

    def __repr__(self) -> str:
        n = len(self)
        lines = [f"AI4BayesCode run: {n} chain{'' if n == 1 else 's'}, n_burn = "
                 f"{self.n_burn} warmup + n_keep = {self.n_keep} kept sweeps per chain"]
        hist = next((c["hist"] for c in self
                     if isinstance(c, dict) and isinstance(c.get("hist"), dict)), None)
        if hist:
            names = list(hist)
            show = names[:10]
            width = max(len(str(k)) for k in show)
            lines.append("History entries (draws along the first axis):")
            for k in show:
                shape = np.shape(hist[k])
                lines.append(f"  {str(k):<{width}}  "
                             + (" x ".join(map(str, shape)) if shape else "1"))
            if len(names) > 10:
                lines.append(f"  ... and {len(names) - 10} more")
        walls = [float(c["wall"]) for c in self
                 if isinstance(c, dict) and isinstance(c.get("wall"), numbers.Real)]
        if walls:
            lines.append(f"Wall time per chain: {min(walls):.3g} to {max(walls):.3g} s "
                         f"(sum {sum(walls):.3g} s)")
        failed = [f"{i} ({c.get('error_type', 'error')})" for i, c in enumerate(self)
                  if isinstance(c, dict) and "error" in c]
        if failed:
            lines.append("Failed chains: " + ", ".join(failed))
        lines.append("Models: " + _models_where(self.models))
        if (self.models is not None and self.n_burn > 0
                and not all(self.history_cleared)):
            lines.append("  (models without clear_history(): their get_history() and "
                         "predict_at() include the warmup draws; predict() drops them)")
        return "\n".join(lines)

    def _repr_pretty_(self, p, cycle) -> None:
        p.text(repr(self))

    def __reduce__(self):
        models = (None if self.models is None
                  else [_portable_model(m, i) for i, m in enumerate(self.models)])
        return (_rebuild_chains_result,
                (list(self), self.n_burn, self.n_keep, list(self.history_cleared),
                 models))

    def __copy__(self):
        return ChainsResult(self, n_burn=self.n_burn, n_keep=self.n_keep,
                            history_cleared=self.history_cleared, models=self.models)

    def __deepcopy__(self, memo):
        import copy
        # The models are shared, not copied: a handle refers to a live model.
        out = ChainsResult(n_burn=self.n_burn, n_keep=self.n_keep,
                           history_cleared=self.history_cleared, models=self.models)
        memo[id(self)] = out
        for c in self:
            out.append(copy.deepcopy(c, memo))
        return out


def _finish(results, models, n_burn, n_keep, *, walls=False,
            verbose=True) -> ChainsResult:
    """Wrap the per-chain results in a ChainsResult.

    Removes the private fields of every result and reports on them once for
    the whole run: a message when the models have no clear_history() (their
    own histories include the warmup draws), and a warning when
    clear_history() left warmup rows in some history entries. Prints the
    per-chain wall times when ``walls`` and ``verbose``.
    """
    cleared, unclear, no_clear = [], {}, []
    for r in results:
        cleared.append(bool(r.pop("_cleared", False)))
        cls = r.pop("_class", None)
        for k in r.pop("_unclear", ()):
            ks = unclear.setdefault(cls, [])
            if k not in ks:
                ks.append(k)
        if r.pop("_no_clear", False) and cls not in no_clear:
            no_clear.append(cls)
        r.pop("_methods", None)
        r.pop("_cap", None)
    if walls and verbose:
        print(f"[AI4BayesCode.run_chains] walls (s): "
              + ", ".join(f"{r['wall']:.1f}" for r in results))
    if no_clear:
        print(f"run_chains: {', '.join(no_clear)} has no clear_history() method, so the "
              f"model's own get_history() and predict_at() include the n_burn = {n_burn} "
              "warmup draws (the histories in the result exclude them, and predict() "
              "drops them). Recompiling the model with AI4BayesCode >= 1.1.3 adds "
              "clear_history().")
    if unclear:
        _warn(
            "run_chains: after clear_history(), "
            + "; ".join(f"the history entries {', '.join(map(repr, ks))} of {cls}"
                        for cls, ks in unclear.items())
            + f" still hold more than n_keep = {n_keep} rows. Their first n_burn = "
            f"{n_burn} rows are dropped in the returned histories, but the model's own "
            "get_history() and predict_at() still contain the warmup draws for them: "
            "the class has no clear_extra_history_() hook that empties these entries.")
    return ChainsResult(results, n_burn=n_burn, n_keep=n_keep, history_cleared=cleared,
                        models=models)


# ---------------------------------------------------------------------------
# The bound on the persistent worker processes and on open files
# ---------------------------------------------------------------------------
def _max_workers() -> int:
    """The environment variable AI4BAYESCODE_MAX_WORKERS, else
    max(8, 4 * os.cpu_count())."""
    v = os.environ.get("AI4BAYESCODE_MAX_WORKERS", "").strip()
    if v:
        try:
            n = int(v)
        except ValueError:
            n = 0
        if n < 1:
            raise ValueError("run_chains: the environment variable "
                             "AI4BAYESCODE_MAX_WORKERS must be a positive integer; "
                             f"it is {v!r}")
        return n
    return max(8, 4 * (os.cpu_count() or 1))


def _fd_usage() -> Optional[tuple[int, int]]:
    """``(open file descriptors, soft RLIMIT_NOFILE)`` of this process, or
    None where they cannot be read (Windows) or the limit is infinite."""
    try:
        import resource
        soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except Exception:  # noqa: BLE001
        return None
    if soft == resource.RLIM_INFINITY or soft <= 0:
        return None
    for d in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(d)), int(soft)
        except OSError:
            continue
    return None


def _open_pools() -> list:
    """The open worker pools of this process, least recently used first."""
    me = os.getpid()
    pools = [p for p in list(_LIVE_POOLS) if p.owner_pid == me and not p.closed]
    pools.sort(key=lambda p: p.last_used)
    return pools


def _make_room(n_new: int, persistent: bool) -> str:
    """Make room for ``n_new`` new worker processes.

    The workers do not fit when (persistent workers only) the live workers of
    earlier runs plus ``n_new`` exceed :func:`_max_workers`, or when the open
    file descriptors plus 3 per new worker plus 16 exceed the soft
    RLIMIT_NOFILE. Then the workers of earlier runs are closed, least recently
    used first, until the new ones fit, with one warning. Returns "" when the
    new workers can be started, else the reason why they cannot (closing
    every earlier run would not be enough; then nothing is closed).
    """
    max_w = _max_workers() if persistent else None

    def causes(live, fd):
        why = []
        if persistent and live + n_new > max_w:
            why.append("workers")
        if fd is not None and fd[0] + 3 * n_new + 16 > fd[1]:
            why.append("files")
        return why

    def measure():
        pools = _open_pools()
        return pools, sum(len(p.workers) for p in pools), _fd_usage()

    pools, live, fd = measure()
    if not causes(live, fd):
        return ""
    # Pools of results that are no longer referenced are closed by their
    # finalizers; collect them before closing any run that is still referenced.
    gc.collect()
    pools, live, fd = measure()
    why = causes(live, fd)
    if not why:
        return ""
    after_all = None if fd is None else (max(0, fd[0] - 3 * live), fd[1])
    if causes(0, after_all):
        parts = []
        if persistent and n_new > max_w:
            parts.append(f"this run needs {n_new} worker processes and at most {max_w} "
                         "persistent worker processes are kept (the environment "
                         "variable AI4BAYESCODE_MAX_WORKERS)")
        if after_all is not None and after_all[0] + 3 * n_new + 16 > after_all[1]:
            parts.append(f"this process has {fd[0]} open files, and {n_new} worker "
                         f"processes need about {3 * n_new} more, above its limit of "
                         f"{fd[1]} open files (RLIMIT_NOFILE, ulimit -n)")
        return "; ".join(parts) + ", also after closing every earlier run"
    whats, advices = [], []
    if "workers" in why:
        whats.append(f"at most {max_w} persistent worker processes are kept (the "
                     "environment variable AI4BAYESCODE_MAX_WORKERS)")
        advices.append("raise AI4BAYESCODE_MAX_WORKERS")
    if "files" in why:
        whats.append(f"the open files of this process would exceed its limit of "
                     f"{fd[1]} (RLIMIT_NOFILE, ulimit -n)")
        advices.append("raise the open-file limit (ulimit -n)")
    what, advice = "; ".join(whats), "or " + " or ".join(advices)
    reason = ("the worker processes holding the models of this run were closed by a "
              f"later run_chains() call to start its own worker processes: {what}. "
              "Close runs that are no longer needed with close_models(chains), "
              f"{advice}")
    n_runs = n_workers = 0
    for p in pools:
        if not causes(live, fd):
            break
        k = len(p.workers)
        p.close(reason=reason)
        n_runs += 1
        n_workers += k
        live -= k
        fd = _fd_usage()
    _warn(f"run_chains: closed the worker processes of {n_runs} earlier "
          f"run{'' if n_runs == 1 else 's'} ({n_workers} worker "
          f"process{'' if n_workers == 1 else 'es'}, least recently used first) to "
          f"start the {n_new} worker processes of this run: {what}. Calls on the model "
          "handles of the closed runs raise an error; their histories are intact. To "
          "avoid this, close runs that are no longer needed with close_models(chains), "
          f"pass keep_models=False when the models are not needed, {advice}.")
    return ""


def _run_in_workers(kind, factory, seeds, n_burn, n_keep, history_keys, n_workers):
    """Run the chains in ``n_workers`` persistent worker processes.

    Chain i runs in worker ``i % n_workers``, and its model is held in that
    worker process. Returns ``(results, pool, caps)``. Returns None, after a
    warning, when a worker process cannot be started or exits during the run;
    the caller then runs the chains in this process.
    """
    ctx = mp.get_context(kind)
    pool = _WorkerPool()
    n = len(seeds)
    try:
        blob = _dumps(factory) if kind == "spawn" else None
        for w in range(n_workers):
            ids = list(range(w, n, n_workers))
            parent_end, child_end = ctx.Pipe(duplex=True)
            try:
                # Under fork the Process arguments are inherited through
                # memory, not pickled, so the factory may be a lambda or a
                # closure; a spawned worker unpickles `blob` instead.
                proc = ctx.Process(
                    target=_worker_main, name=f"AI4BayesCode-chains-{w}", daemon=False,
                    args=(child_end, parent_end if kind == "fork" else None, ids,
                          [seeds[i] for i in ids], n_burn, n_keep, history_keys,
                          factory if kind == "fork" else None, blob))
                proc.start()
            except BaseException:
                parent_end.close()
                raise
            finally:
                child_end.close()
            pool.add(proc, parent_end, ids)
        results, caps = pool.collect(n)
    except Exception as e:  # noqa: BLE001
        pool.close()
        if _exiting_daemon():
            _block_forever()
        lost = isinstance(e, _WorkerLost)
        detail = str(e) if lost else f"{type(e).__name__}: {e}"
        _warn(f"run_chains: parallel ({kind}) execution failed ({detail}); running "
              f"the {n} chains sequentially instead." + (_BLAS_HINT if lost else ""))
        return None
    except BaseException:
        pool.close()
        raise
    return results, pool, caps


def _in_process(args, keep, verbose):
    """Run the chains of ``args`` one after another in this process; returns
    the results and the models (None when ``keep`` is False). The caller's
    numpy and random global states are restored afterwards."""
    results, models = [], []
    with _CallerRandomState():
        for a in args:
            if verbose:
                print(f"[AI4BayesCode.run_chains] chain seed={a[1]} ...")
            res, m = _run_chain(*a)
            results.append(res)
            models.append(m if keep else None)
            m = None
    return results, (models if keep else None)


# ---------------------------------------------------------------------------
# predict() and close_models()
# ---------------------------------------------------------------------------
def _run_models(chains, fn, need_models=True):
    """``(models, history_cleared)`` of a ChainsResult, checked."""
    if not isinstance(chains, ChainsResult):
        raise TypeError(
            f"{fn}() takes the result of run_chains() (a ChainsResult); got "
            f"{type(chains).__name__}. A slice, list(chains) or a list comprehension "
            "over the result is a plain list of the per-chain dicts without the "
            f"models: pass the result itself, as in {fn}(chains).")
    models = chains.models
    if models is None:
        if need_models:
            raise ValueError(
                f"{fn}() calls predict_at() on the models of the chains, and this "
                "run_chains() result does not hold them (chains.models is None); "
                "rerun run_chains() with keep_models=True (the default)")
        return None, None
    if not need_models:
        return list(models), None
    cleared = list(chains.history_cleared)
    if len(cleared) != len(models):
        raise ValueError(
            f"{fn}(): chains.models has {len(models)} entries and "
            f"chains.history_cleared {len(cleared)}; they must have one entry per "
            "chain of the run")
    return list(models), cleared


def _as_input(value):
    """A value of ``newdata`` as predict_at() takes it: a numeric matrix (or
    higher-dimensional array) is flattened column-major, the layout of an R
    matrix that the compiled models read, and a number becomes a vector of
    length 1. Lists and array-like objects of numbers become arrays; other
    values are passed unchanged."""
    if isinstance(value, np.ndarray):
        a = value
    elif (isinstance(value, (list, tuple, numbers.Number, np.generic))
          or hasattr(value, "__array__")):
        try:
            a = np.asarray(value)
        except Exception:  # noqa: BLE001
            return value
        if a.dtype.kind not in "biuf":
            return value
    else:
        return value
    if a.ndim == 0:
        return a.reshape(1)
    if a.ndim >= 2:
        return np.ravel(a, order="F")
    return a


def _prepare_newdata(newdata) -> dict:
    if newdata is None:
        return {}
    if not isinstance(newdata, Mapping):
        shape = getattr(newdata, "shape", None)
        raise TypeError(
            "predict(): newdata must be a dict that maps each input name of the "
            "model's predict_at() to its values, for example "
            "predict(chains, {\"X\": X_new}) with X_new a matrix of n_new rows; got "
            f"{type(newdata).__name__}" + ("" if shape is None else f" of shape {shape}"))
    return {k: _as_input(v) for k, v in newdata.items()}


def predict(chains, newdata=None, pool=True):
    """Posterior prediction from every chain of a :func:`run_chains` result.

    Calls ``predict_at(newdata)`` on the model of each chain
    (``chains.models[i]``); the worker processes holding models compute their
    chains concurrently. Each chain gives one row per stored draw, computed by
    the model from that draw's own values. ``newdata=None`` means ``{}``: the
    posterior predictive at the training data. For a chain whose model had no
    clear_history() (``chains.history_cleared[i]`` is False), the first
    ``n_burn`` rows of every entry with ``n_burn + n_keep`` rows are dropped.
    A chain that failed in run_chains() is left out, with a warning. The
    output and the warnings of each chain's predict_at() are shown here, in
    chain order; an exception raised by the model of chain i is raised with
    "chain i: " in front of its message.

    Parameters
    ----------
    chains : ChainsResult
        The value of :func:`run_chains` with ``keep_models=True``. A plain
        list (a slice of the result, for example) raises TypeError.
    newdata : dict, optional
        Maps each input name of the models' ``predict_at()`` to its values.
        A matrix value (n_new rows, one column per covariate) is flattened
        column-major, the layout the compiled models read; a number becomes a
        vector of length 1. Another type than a dict raises TypeError.
    pool : bool, default True
        True: ``{key: array}`` with the rows of all chains stacked along the
        first axis in chain order (``np.vstack`` for 2-D entries); an error
        when the predict_at() outputs of the chains have different keys or
        different shapes after the first axis.
        False: a list with the ``predict_at()`` dict of each chain (None for a
        failed chain).
    """
    models, cleared = _run_models(chains, "predict")
    newdata = _prepare_newdata(newdata)
    n_chains = len(models)
    if not n_chains:
        raise ValueError("predict(): the run_chains() result holds no chains")
    use = [i for i, m in enumerate(models) if m is not None]
    if not use:
        raise ValueError("predict(): every chain failed in run_chains(), so no chain "
                         "has a model")
    if len(use) < n_chains:
        _warn("predict: chain(s) "
              + ", ".join(str(i) for i in range(n_chains) if i not in use)
              + " failed in run_chains() and have no model; the prediction uses the "
                "other chains")
    # One request per chain to the worker processes, sent before any reply is
    # read, so that the workers compute concurrently.
    replies: dict = {}
    groups: dict = {}
    for i in use:
        m = models[i]
        if isinstance(m, ChainModel) and m._detached is None:
            groups.setdefault(id(m._pool), (m._pool, []))[1].append(i)
    for wpool, idx in groups.values():
        got = wpool.call_many([(models[i]._chain, "predict_at", (newdata,), {})
                               for i in idx])
        replies.update(zip(idx, got))
        got = None
    outs = {}
    for i in use:
        if i in replies:
            ok, value, cap = replies.pop(i)
            _relay(cap)
            if not ok:
                # Clear every local that refers to the exception, so that it
                # and this frame (in its traceback) form no reference cycle.
                exc, value, replies = value, None, None
                try:
                    raise exc
                finally:
                    exc = None
        else:
            try:
                value = models[i].predict_at(newdata)
            except Exception as e:  # noqa: BLE001
                _name_chain(e, i)
                raise
        outs[i] = value
        value = None

    n_burn, n_keep = chains.n_burn, chains.n_keep
    if n_burn > 0:
        for i in use:
            if not cleared[i] and isinstance(outs[i], Mapping):
                outs[i] = {k: (np.asarray(v)[n_burn:]
                               if np.ndim(v) > 0 and np.shape(v)[0] == n_burn + n_keep
                               else v)
                           for k, v in outs[i].items()}
    if not pool:
        return [outs.get(i) for i in range(n_chains)]

    for i in use:
        if not isinstance(outs[i], Mapping):
            raise TypeError(f"predict(): predict_at() of chain {i} returned "
                            f"{type(outs[i]).__name__}, not a dict of arrays")
    keys = list(outs[use[0]])
    for i in use[1:]:
        if set(outs[i]) != set(keys):
            raise ValueError(
                f"predict(): the predict_at() outputs of chains {use[0]} and {i} have "
                f"different keys: {sorted(keys)} and {sorted(outs[i])}; use pool=False")
    pooled = {}
    for k in keys:
        arrs = [np.atleast_1d(np.asarray(outs[i][k])) for i in use]
        if len({a.shape[1:] for a in arrs}) > 1:
            raise ValueError(
                f"predict(): the predict_at() outputs of the chains have different "
                f"shapes after the first axis for {k!r}: shapes "
                + ", ".join(str(a.shape) for a in arrs) + "; use pool=False")
        pooled[k] = np.concatenate(arrs, axis=0)
    return pooled


def close_models(chains) -> None:
    """Stop the worker processes that hold the models of a :func:`run_chains`
    result.

    Afterwards every call on one of its model handles raises RuntimeError
    ("the worker processes holding the models of this run were closed").
    Calling it again does nothing, and so does calling it on a result whose
    models are objects in this process or that was made with
    ``keep_models=False``. The worker processes are also stopped when the
    result and all of its model handles are garbage-collected, at interpreter
    exit, and when a later run needs room for its own worker processes (see
    :func:`run_chains`). A plain list (a slice of the result) raises TypeError.
    """
    models, _ = _run_models(chains, "close_models", need_models=False)
    for p in _pools_of(models):
        p.close()
    return None


# ---------------------------------------------------------------------------
# run_chains()
# ---------------------------------------------------------------------------
def run_chains(
    factory: Callable[[int], Any],
    *,
    n_chains: int | None = None,
    seeds: Iterable[int] = _DEFAULT_SEEDS,
    n_burn: int = 2000,
    n_keep: int = 10000,
    n_jobs: Optional[int] = None,
    history_keys: Optional[list[str]] = None,
    verbose: bool = True,
    n_cores: Optional[int] = None,
    keep_models: bool = True,
) -> ChainsResult:
    """Run `len(seeds)` chains in parallel (or sequentially if n_cores <= 1).

    Each chain builds its model with ``factory(seed)`` and runs
    ``step(n_burn)``, ``clear_history()`` (when ``n_burn > 0`` and the model
    has it) and ``step(n_keep)``. A model without clear_history() (compiled
    against AI4BayesCode older than 1.1.3) runs ``step(n_burn)`` and
    ``step(n_keep)`` as in 1.1.2.

    Parameters
    ----------
    factory : Callable[[int], model]
        A callable that builds a fresh model given a seed. On Unix
        (macOS / Linux) the chains fork, so a lambda or local closure
        works directly -- just like R's ``function(s) new(Mod, ..., s)``::

            chains = AI4BayesCode.run_chains(
                lambda s: MyModel(y, X, rng_seed=s, keep_history=True),
                seeds=[1, 2, 3, 4])

        On Windows see the module docstring: sequential by default, and
        parallel only for a top-level function called under the
        ``__main__`` guard. If the parallel pool cannot start, the chains
        fall back to running sequentially.

    n_chains : int, optional
        Number of chains when ``seeds`` is not given: the seeds are then
        101, 202, 303, ... A ValueError is raised when ``seeds`` is given
        with another length.
    seeds : iterable of int
        One seed per chain. Chain count = len(seeds). For chain i,
        ``np.random.seed(seeds[i] % 2**32)`` and ``random.seed(seeds[i])``
        are called immediately before ``factory(seeds[i])``, in every code
        path, so a model that draws from numpy's global generator or from the
        ``random`` module gives the same draws in a parallel and a
        sequential run. A run in this process restores the caller's numpy
        and ``random`` states afterwards.
    n_burn, n_keep : int
        Warmup and sampled iterations per chain.
    n_jobs : int, optional
        Alias of ``n_cores`` (a ValueError when both are given and differ).
    history_keys : list of str, optional
        Subset of history keys to return (saves memory for big chains).
        If None, return all keys from `get_history()`.
    verbose : bool, default True
        Print per-chain status.
    n_cores : int, optional
        Number of parallel worker processes. Defaults to min(len(seeds),
        cpu_count()) where the platform can fork, and to 1 (sequential)
        on Windows. A value <= 1 runs the chains sequentially in this
        process.
    keep_models : bool, default True
        Return the models in ``chains.models``. When the chains run in
        parallel (fork available, or a picklable factory on Windows,
        ``n_cores > 1`` and more than one chain), the worker processes are
        not stopped at the end of the run and ``chains.models[i]`` is a
        :class:`ChainModel` handle whose method calls run in the worker
        process holding chain i; the printed output and the warnings of the
        chain runs and of these calls are shown in this process. The worker
        processes are stopped by :func:`close_models`, when the result and
        its handles are garbage-collected, or at interpreter exit. At most
        ``AI4BAYESCODE_MAX_WORKERS`` (an environment variable; default
        ``max(8, 4 * os.cpu_count())``) such worker processes are kept over
        all runs of the session: before a run starts its workers, the
        workers of earlier runs are closed, least recently used first, when
        the limit, or the soft limit on open files (each worker uses 3 file
        descriptors), would be exceeded, with a warning; calls on the
        handles of those runs then raise an error that says so. When even
        closing every earlier run would not be enough, the chains run in
        this process, with a warning. Otherwise ``chains.models[i]`` is the
        model object itself. False: the models are discarded and
        ``chains.models`` is None.

    Returns
    -------
    ChainsResult: a list of dicts, one per chain: `{"seed", "wall", "hist"}`
    (and "error", "error_type" for a failed chain), as in 1.1.2. Each
    ``"hist"`` is KEEP-ONLY: every history entry (1-D, 2-D, and higher-D
    alike) holds the ``n_keep`` post-warmup draws along its first axis (each
    chain calls clear_history() after the warmup; for a model without it the
    first ``n_burn`` draws are stripped here), so downstream ``rhat_summary``
    / ``diagnose`` need no ``drop_burn`` / ``n_burn`` bookkeeping. An entry
    with a single stored draw (``keep_history=False``) is returned unchanged.
    A model without clear_history() keeps its warmup draws in its own history
    (a message says so); an entry that clear_history() leaves with more than
    n_keep rows is cut in "hist" and named in a warning. Attributes
    ``models``, ``n_burn``, ``n_keep`` and ``history_cleared``; methods
    ``predict()`` and ``close()`` (see :func:`predict` and
    :func:`close_models`). Its repr() is a summary of the run.
    """
    # R's ai4bayescode_run_chains takes n_chains; accept it so the same call
    # reads the same way in both frontends. Given together with seeds of
    # another length, it is an error, as in R.
    seeds_given = seeds is not _DEFAULT_SEEDS
    if n_chains is not None:
        if int(n_chains) < 1:
            raise ValueError(f"n_chains must be >= 1; got {n_chains}")
        if not seeds_given:
            seeds = tuple(101 * (i + 1) for i in range(int(n_chains)))
    seeds = list(seeds)
    if n_chains is not None and seeds_given and len(seeds) != int(n_chains):
        raise ValueError(f"run_chains: len(seeds) = {len(seeds)} must equal "
                         f"n_chains = {n_chains}")
    if not seeds:
        return ChainsResult([], n_burn=n_burn, n_keep=n_keep, history_cleared=[],
                            models=[] if keep_models else None)
    if n_burn < 0 or n_keep < 1:   # R parity: stopifnot(n_burn>=0, n_keep>=1)
        raise ValueError("n_burn must be >= 0 and n_keep >= 1")
    # n_cores is the documented name of the worker count; n_jobs is its alias.
    if n_cores is not None and n_jobs is not None and n_cores != n_jobs:
        raise ValueError(
            f"run_chains: n_cores={n_cores!r} and n_jobs={n_jobs!r} differ; n_jobs is "
            "an alias of n_cores, so pass only one of them")
    if n_cores is None:
        n_cores = n_jobs
    can_fork = sys.platform != "win32" and "fork" in mp.get_all_start_methods()
    if n_cores is None:
        # Without fork every worker re-imports the caller's script, which only
        # works for a guarded script with a top-level factory; so sequential
        # is the default there, as it is in R.
        n_cores = min(len(seeds), os.cpu_count() or 1) if can_fork else 1
    n_cores = int(n_cores)
    if getattr(mp.current_process(), "_inheriting", False):
        raise RuntimeError(
            "run_chains() was called while a worker process was importing the main "
            "script. Put the call under  if __name__ == '__main__':  or pass n_cores=1.")

    args = [(factory, int(s), n_burn, n_keep, history_keys) for s in seeds]

    if n_cores <= 1:
        results, models = _in_process(args, keep_models, verbose)
        return _finish(results, models, n_burn, n_keep, verbose=verbose)

    if keep_models:
        # The models are kept after the run: in persistent worker processes
        # when the chains run in parallel, else as objects in this process.
        results, models, in_workers = None, None, False
        if len(seeds) > 1:
            seed_list = [a[1] for a in args]
            kind = "fork" if can_fork else "spawn"
            # On Windows connection.wait() takes at most 64 handles
            # (WaitForMultipleObjects); 61 is also the pool limit there.
            n_workers = min(n_cores, len(seeds), 10**9 if can_fork else 61)
            reason = "" if can_fork else _spawn_obstacle(factory)
            if reason:
                _warn(f"run_chains: {reason}; running the {len(seeds)} chains "
                      "sequentially.")
            else:
                cause = _make_room(n_workers, persistent=True)
                if cause:
                    _warn(f"run_chains: {cause}; running the {len(seeds)} chains in this "
                          "process instead (the models are kept as objects).")
                else:
                    if can_fork:
                        _warn_if_blas_unpinned()
                    got = _run_in_workers(kind, factory, seed_list, n_burn, n_keep,
                                          history_keys, n_workers)
                    if got is not None:
                        results, pool, caps = got
                        got = None
                        models = [None if "error" in r else
                                  ChainModel(pool, i, pool.by_chain[i].pid,
                                             r.get("_class"), r.get("_methods"))
                                  for i, r in enumerate(results)]
                        if all(m is None for m in models):
                            pool.close()
                        in_workers = True
                        if (not can_fork and all(m is None for m in models)
                                and all(r.get("error_type") in _REBUILD_ERRORS
                                        for r in results)):
                            # Every worker failed the way a worker that cannot
                            # rebuild the factory fails. Run the first chain
                            # here: if it works, run the rest here as well; if
                            # it fails too, the results stand as they are.
                            first = _probe_first(args, verbose)
                            if first is not None:
                                _warn("run_chains: every parallel chain failed in its "
                                      "worker (on Windows the factory and everything "
                                      "it uses must be defined at module top level; "
                                      f"chain 0: {results[0].get('error_type')}: "
                                      f"{results[0].get('error')}); running the chains "
                                      "sequentially instead.")
                                rest, rest_models = _in_process(args[1:], True, verbose)
                                results = [first[0]] + rest
                                models = [first[1]] + rest_models
                                caps, in_workers = None, False
                        if caps is not None:
                            try:
                                for cap in caps:
                                    _relay(cap)
                            except BaseException:
                                pool.close()
                                raise
                        pool = None
        if results is None:
            results, models = _in_process(args, True, verbose)
        return _finish(results, models, n_burn, n_keep, walls=in_workers,
                       verbose=verbose)

    results: Optional[list[dict]] = None
    cause = _make_room(min(n_cores, len(seeds)), persistent=False)
    if cause:
        _warn(f"run_chains: {cause}; running the {len(seeds)} chains sequentially in "
              "this process instead.")
    elif can_fork:
        # Unix: FORK. The factory is handed to the workers through inherited
        # memory (the `_ACTIVE_FACTORY` global set just before the pool forks),
        # NOT through the task queue -- so only the picklable seed/counts are
        # serialized and a lambda / closure runs in true parallel, exactly like
        # R's mclapply. ProcessPoolExecutor (not multiprocessing.Pool) is used so
        # a crashed worker raises BrokenProcessPool instead of hanging, and we
        # then fall back to sequential. That fallback is a second line of
        # defence only: the fork-unsafe-BLAS case it used to catch is
        # PREVENTED by the one-thread pin in this package's __init__, because
        # degrading to sequential silently costs the user the whole speedup.
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        global _ACTIVE_FACTORY
        _ACTIVE_FACTORY = factory
        try:
            _warn_if_blas_unpinned()
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_cores, mp_context=ctx) as ex:
                futs = [ex.submit(_chain_worker_fork, int(s),
                                  n_burn, n_keep, history_keys) for s in seeds]
                results = [f.result() for f in futs]
        except Exception as e:  # noqa: BLE001  (BrokenProcessPool, ...)
            _warn(f"run_chains: parallel (fork) execution failed "
                  f"({type(e).__name__}: {e}); running the {len(seeds)} chains "
                  "sequentially instead."
                  + (_BLAS_HINT if isinstance(e, BrokenProcessPool) else ""))
            results = None
        finally:
            _ACTIVE_FACTORY = None
    else:
        # Windows (no fork): SPAWN fresh interpreters. Each one unpickles the
        # factory by importing the module that defines it, so check first that
        # this can work at all; otherwise run sequentially without spawning.
        reason = _spawn_obstacle(factory)
        if reason:
            _warn(f"run_chains: {reason}; running the {len(seeds)} chains "
                  "sequentially.")
        else:
            from concurrent.futures import ProcessPoolExecutor
            from concurrent.futures.process import BrokenProcessPool
            try:
                ctx = mp.get_context("spawn")
                # Windows allows at most 61 worker processes per pool.
                with ProcessPoolExecutor(max_workers=min(n_cores, 61), mp_context=ctx) as ex:
                    results = list(ex.map(_chain_task, *zip(*args)))
            except Exception as e:  # noqa: BLE001  (BrokenProcessPool, ...)
                _warn(f"run_chains: parallel execution failed "
                      f"({type(e).__name__}: {e}); running the {len(seeds)} chains "
                      "sequentially instead."
                      + (_BLAS_HINT if isinstance(e, BrokenProcessPool) else ""))
                results = None
            if results is not None and all(r.get("error_type") in _REBUILD_ERRORS
                                           for r in results):
                # Every worker failed the way a worker that cannot rebuild the
                # factory fails (a name defined only under the __main__ guard,
                # say). Run the first chain here: if it works, run the rest here
                # as well; if it fails too, the model itself fails and the
                # results stand as they are.
                first = _probe_first(args, verbose)
                if first is not None:
                    _warn("run_chains: every parallel chain failed in its worker (on "
                          "Windows the factory and everything it uses must be defined "
                          f"at module top level; chain 0: {results[0].get('error_type')}: "
                          f"{results[0].get('error')}); running the chains "
                          "sequentially instead.")
                    rest, _ = _in_process(args[1:], False, verbose)
                    return _finish([first[0]] + rest, None, n_burn, n_keep,
                                   verbose=verbose)

    if results is None:
        results, _ = _in_process(args, False, verbose)
        return _finish(results, None, n_burn, n_keep, verbose=verbose)
    for r in results:
        _relay(r.pop("_cap", None))
    return _finish(results, None, n_burn, n_keep, walls=True, verbose=verbose)


def _probe_first(args, verbose):
    """Run the first chain of ``args`` in this process, with its warnings
    ignored; ``(result, model)`` when it succeeds, else None."""
    if verbose:
        print(f"[AI4BayesCode.run_chains] chain seed={args[0][1]} ...")
    with _CallerRandomState(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, m = _run_chain(*args[0])
    return None if "error" in res else (res, m)

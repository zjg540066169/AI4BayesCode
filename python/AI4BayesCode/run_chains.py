"""Parallel multi-chain MCMC orchestration.

Python counterpart of the R helper's `ai4bayescode_run_chains()`. The two are
NOT argument-compatible: R takes `n_chains` and returns an `ai4bayescode_run`
object (`run$histories`, `run$models`, ...); this takes `seeds` and returns a
`ChainsResult`, a list of per-chain dicts. `n_chains=` is accepted as an alias
that derives the seeds:

    chains = AI4BayesCode.run_chains(
        factory = lambda seed: MyModel(y, seed=seed, keep_history=True),
        seeds   = [101, 202, 303, 404],
        n_burn  = 5000,
        n_keep  = 5000,
        n_cores = 4,           # parallel worker processes
    )
    # chains is a list of dicts: {"hist": {...}, "wall": seconds, "seed": ...,
    #                             "model": the chain's model}
    pred = AI4BayesCode.predict(chains)   # predict_at() of every chain, pooled
    AI4BayesCode.close_models(chains)     # stop the worker processes

Each chain runs `step(n_burn)`, `clear_history()` and `step(n_keep)`, so the
model's own `get_history()` and `predict_at()` cover the n_keep kept draws.
With `keep_models=True` (the default) and parallel chains, the worker
processes are not stopped when run_chains() returns: `chains[i]["model"]`
is a `ChainModel` handle, and each method call on it runs in the worker
process that holds chain i, until `close_models(chains)` or until the result
and its handles are garbage-collected.

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

import multiprocessing as mp
import os
import pickle
import signal
import sys
import threading
import time
import traceback
import warnings
import weakref
from collections.abc import Mapping
from multiprocessing.connection import wait as _mp_wait
from typing import Any, Callable, Iterable, Optional

import numpy as np
from ._console import safe_print as print  # noqa: A001 (see _console.py)


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


def _run_chain(factory: Callable[[int], Any], seed: int,
               n_burn: int, n_keep: int,
               history_keys: Optional[list[str]] = None) -> tuple[dict, Any]:
    """Run one chain; return ``(result, model)``, with model None on failure.

    Besides "seed", "wall" and "hist", a successful result contains the
    private fields "_cleared", "_class", "_unclear" and "_no_clear", which
    `_finish` removes in the calling process after reporting on them once
    for the whole run.
    """
    t0 = time.time()
    try:
        m = factory(seed)
        # Some models have step(n) that returns None; we just call and discard.
        if n_burn > 0:
            m.step(int(n_burn))
        # clear_history() (AI4BayesCode >= 1.1.3) discards the stored draws
        # and leaves the sampler state unchanged, so after it the model's own
        # get_history() and predict_at() cover the n_keep kept sweeps only.
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
        # NOTE: unlike R's ai4bayescode_run_chains (which stop()s the whole run
        # on a genuine chain failure), the Python driver degrades gracefully: a
        # failed chain warns loudly and returns a marked-failed result
        # (hist=None + "error") so the surviving chains stay usable.
        warnings.warn(f"run_chains: chain seed={seed} failed: {e}")
        return ({"seed": seed, "wall": time.time() - t0, "hist": None,
                 "error": str(e), "error_type": type(e).__name__}, None)


def _chain_worker(factory: Callable[[int], Any], seed: int,
                  n_burn: int, n_keep: int,
                  history_keys: Optional[list[str]] = None) -> dict:
    """Run one chain. This is called in a fresh subprocess."""
    return _run_chain(factory, seed, n_burn, n_keep, history_keys)[0]


# The active factory for the fork path. Set in the parent immediately before the
# workers fork; each forked child inherits it through memory, so it never has to
# be pickled (which is what lets a lambda / closure run in parallel -- mirroring
# R's mclapply). Only touched on the fork path and cleared right after.
_ACTIVE_FACTORY: Optional[Callable[[int], Any]] = None


# Errors a spawned worker raises when it cannot rebuild the factory.
_REBUILD_ERRORS = ("NameError", "AttributeError", "ImportError", "ModuleNotFoundError")

_BLAS_THREAD_VARS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _warn_if_blas_unpinned() -> None:
    """Warn when the fork path is about to run with a multithreaded BLAS.

    `AI4BayesCode/__init__.py` pins these to one thread at import, which is
    the only place it can work: Accelerate decides whether to use its dispatch
    queues at the FIRST BLAS call, so a cap applied later -- here, for
    instance -- changes nothing. If the variable is not "1" by now, the caller
    overrode it, and a fork with a BLAS-heavy model may kill every worker.
    """
    unpinned = [v for v in _BLAS_THREAD_VARS
                if os.environ.get(v) not in (None, "1")]
    if not unpinned:
        return
    warnings.warn(
        "run_chains: " + ", ".join(f"{v}={os.environ[v]}" for v in unpinned) +
        " is set, so the native BLAS may be multithreaded. fork() and a "
        "multithreaded BLAS do not mix -- on macOS Accelerate every worker "
        "dies inside libdispatch and the run falls back to sequential. Set "
        "these to \"1\" before starting Python, or pass n_cores=1.")


def _chain_worker_fork(seed: int, n_burn: int, n_keep: int,
                       history_keys: Optional[list[str]] = None) -> dict:
    """Fork-path worker: read the inherited factory from the module global.

    Only the (picklable) seed/counts travel through the task queue; the factory
    comes from `_ACTIVE_FACTORY`, inherited from the parent at fork time.
    """
    if _ACTIVE_FACTORY is None:  # pragma: no cover - defensive
        raise RuntimeError("run_chains: no active factory in forked worker")
    return _chain_worker(_ACTIVE_FACTORY, seed, n_burn, n_keep, history_keys)


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
#   worker -> parent, once per chain:  ("chain", i, result)
#   parent -> worker:                  ("call", i, method, args, kwargs)
#                                      or ("close",)
#   worker -> parent, once per call:   ("ok", value)
#                                      or ("err", pickled exception or None,
#                                          (type name, message), traceback text)
#
# A worker handles one call at a time. The parent sends a worker its next call
# only after receiving the reply to the previous one, under that worker's
# lock; calls to different workers run concurrently.
# ---------------------------------------------------------------------------

_CLOSED_MSG = ("the worker processes holding the models of this run were closed "
               "(close_models)")

_DETACHED_PICKLED = (
    "this model handle was restored from a pickle: the worker process holding the "
    "model belongs to another Python process or session, or was closed, so the "
    "model cannot be called here (the histories in the result are intact)")

#: Pools whose worker processes may still run. A forked worker closes its
#: inherited copies of their connection ends (see `_worker_main`).
_LIVE_POOLS: "weakref.WeakSet[_WorkerPool]" = weakref.WeakSet()


def _dumps(obj) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


_CLOSE_REQUEST = _dumps(("close",))


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
    """The fields of a pybind11 record (a DagInfo, an AdaptationInfo) that a
    ChainModel call returns: the record does not pickle, so the worker process
    sends its fields instead. Item access and attribute access both return a
    field, so ``get_dag()["predict_edges"]``, ``get_dag().predict_edges`` and
    ``plot_dag()`` work as on the record."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __reduce__(self):
        return (_Fields, (dict(self),))


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
    and tuples) replaced by a _Fields of its fields."""
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
        return _Fields({a: _with_record_fields(getattr(value, a), depth + 1)
                        for a in names})
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

    Runs the chains ``chain_ids`` in order and sends ("chain", i, result) for
    each; then receives ("call", i, method, args, kwargs) requests, calls the
    method on chain i's model and sends the reply, until a ("close",) request
    arrives or the connection reaches end-of-file. Under fork, ``factory`` is
    the factory itself (inherited through memory); under spawn it is None and
    ``factory_blob`` holds the pickled factory.
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
        # and of the pipes of every other live worker. Closing them makes
        # recv_bytes() in each worker raise EOFError once the parent process
        # has exited, with or without a "close" request.
        parent_end.close()
        for live in list(_LIVE_POOLS):
            for w in live.workers:
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
            warnings.warn(f"run_chains: chain seed={seed} failed: {build_error}")
            res, model = ({"seed": seed, "wall": 0.0, "hist": None,
                           "error": str(build_error),
                           "error_type": type(build_error).__name__}, None)
        else:
            res, model = _run_chain(factory, seed, n_burn, n_keep, history_keys)
        if model is not None:
            res["_methods"] = [a for a in dir(model) if not a.startswith("_")]
        try:
            buf = _dumps(("chain", i, res))
        except Exception as e:  # noqa: BLE001
            model = None
            buf = _dumps(("chain", i, {
                "seed": seed, "wall": res.get("wall", 0.0), "hist": None,
                "error": ("the result of this chain cannot be sent from the worker "
                          f"process ({type(e).__name__}: {e})"),
                "error_type": type(e).__name__}))
        if model is not None:
            models[i] = model
        try:
            conn.send_bytes(buf)
        except OSError:
            return
    factory = None
    while True:
        try:
            buf = conn.recv_bytes()
        except (EOFError, OSError):
            break
        name = "the method"
        try:
            req = pickle.loads(buf)
            if req[0] == "close":
                break
            _, i, name, args, kwargs = req
            if i not in models:
                raise RuntimeError(f"chain {i} has no model in worker process {os.getpid()}")
            reply = ("ok", getattr(models[i], name)(*args, **kwargs))
        except Exception as e:  # noqa: BLE001
            try:
                blob = _dumps(e)
            except Exception:  # noqa: BLE001
                blob = None
            reply = ("err", blob, (type(e).__name__, str(e)), traceback.format_exc())
        try:
            out = _dumps(reply)
        except Exception as e:  # noqa: BLE001
            out = None
            if reply[0] == "ok":
                # A pybind11 record (get_dag() returns a DagInfo) does not
                # pickle; its fields do.
                try:
                    out = _dumps(("ok", _with_record_fields(reply[1])))
                except Exception:  # noqa: BLE001
                    out = None
            if out is None:
                out = _dumps(("err", None, (type(e).__name__,
                              f"the value returned by {name}() cannot be sent to the "
                              f"calling process ({e})"), ""))
        try:
            conn.send_bytes(out)
        except OSError:
            break
    conn.close()


def _reply_value(buf, name) -> tuple[bool, Any]:
    """``(True, value)`` or ``(False, exception)`` from one reply message."""
    try:
        reply = pickle.loads(buf)
    except Exception as e:  # noqa: BLE001
        return (False, RuntimeError(
            f"the value returned by {name}() cannot be unpickled in this process "
            f"({type(e).__name__}: {e})"))
    if reply[0] == "ok":
        return (True, reply[1])
    _, blob, (tname, msg), tb = reply
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
    return (False, exc)


class _Worker:
    """Parent-side record of one persistent worker process."""

    def __init__(self, index, process, conn, chains, state):
        self.index = index
        self.process = process
        self.pid = process.pid
        self.conn = conn
        self.chains = list(chains)
        self.state = state          # shared with the pool: {"closed": bool}
        self.lock = threading.Lock()
        self.owed = 0               # sent calls whose replies are not read yet
        self.lost = None            # error message after an unexpected exit

    def gone(self) -> str:
        """The error message for a call that cannot reach this worker."""
        if self.state["closed"]:
            return _CLOSED_MSG
        if self.lost is None:
            try:
                self.process.join(0.5)
            except Exception:  # noqa: BLE001
                pass
            code = self.process.exitcode
            self.lost = (f"the worker process {self.pid} holding chain(s) "
                         f"{', '.join(map(str, self.chains))} exited unexpectedly"
                         + ("" if code is None else f" (exit code {code})")
                         + "; the models of these chains are lost")
        return self.lost

    # `owed` changes inside the _DeferSigint block, together with the message
    # it counts: a KeyboardInterrupt raised when the block ends then leaves it
    # equal to the number of replies still in the connection.
    def send(self, payload) -> None:
        try:
            with _DeferSigint():
                self.conn.send_bytes(payload)
                self.owed += 1
        except (OSError, EOFError):
            self.owed = 0
            raise _WorkerLost(self.gone()) from None

    def receive(self) -> bytes:
        try:
            with _DeferSigint():
                buf = self.conn.recv_bytes()
                self.owed -= 1
        except (OSError, EOFError):
            self.owed = 0
            raise _WorkerLost(self.gone()) from None
        return buf

    def drain(self) -> None:
        """Read and discard the replies to calls whose caller was interrupted
        (KeyboardInterrupt) before the reply arrived."""
        while self.owed > 0:
            try:
                self.conn.poll(None)
            except (OSError, EOFError):
                self.owed = 0
                raise _WorkerLost(self.gone()) from None
            self.receive()


def _shutdown_workers(workers, state, owner_pid) -> None:
    """Stop the worker processes of a pool. Run once, through weakref.finalize:
    by close(), when the pool is garbage-collected, or at interpreter exit."""
    if os.getpid() != owner_pid:
        return          # a forked copy of the pool: the workers belong to the parent
    state["closed"] = True
    # A worker that receives the "close" request exits on its own. A worker
    # that is computing a call (its lock is held, or a reply is still owed)
    # is terminated without the request.
    asked = []
    for w in workers:
        got = w.lock.acquire(timeout=1.0)
        sent = False
        try:
            if got and w.lost is None and w.owed == 0 and not w.conn.closed:
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
        p = w.process
        try:
            if sent:
                p.join(max(0.0, deadline - time.monotonic()))
            if p.exitcode is None:
                p.terminate()
                p.join(1.0)
            if p.exitcode is None:
                p.kill()
                p.join(1.0)
        except Exception:  # noqa: BLE001
            pass
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


class _WorkerPool:
    """The persistent worker processes of one run_chains() call."""

    def __init__(self):
        self.owner_pid = os.getpid()
        self.workers: list[_Worker] = []
        self.by_chain: dict[int, _Worker] = {}
        self.state = {"closed": False}
        # The finalizer holds the workers and the state, not the pool, so the
        # pool is garbage-collected once the result and every handle are gone.
        self._finalizer = weakref.finalize(self, _shutdown_workers, self.workers,
                                           self.state, self.owner_pid)
        _LIVE_POOLS.add(self)

    @property
    def closed(self) -> bool:
        return self.state["closed"]

    def add(self, process, conn, chains) -> None:
        w = _Worker(len(self.workers), process, conn, chains, self.state)
        self.workers.append(w)
        for i in w.chains:
            self.by_chain[i] = w

    def close(self) -> None:
        self._finalizer()

    def collect(self, n) -> list[dict]:
        """Receive the results of the n chains, one message per chain."""
        results: list = [None] * n
        left = {w.conn: len(w.chains) for w in self.workers if w.chains}
        by_conn = {w.conn: w for w in self.workers}
        while left:
            for conn in _mp_wait(list(left)):
                w = by_conn[conn]
                try:
                    buf = conn.recv_bytes()
                except (EOFError, OSError):
                    w.process.join(0.5)
                    raise _WorkerLost(
                        f"worker process {w.pid} exited (exit code "
                        f"{w.process.exitcode}) before it returned all of its "
                        "chains") from None
                _, i, res = pickle.loads(buf)
                results[i] = res
                left[conn] -= 1
                if not left[conn]:
                    del left[conn]
        return results

    def call(self, chain, name, args, kwargs):
        ok, value = self.call_many([(chain, name, args, kwargs)])[0]
        if ok:
            return value
        raise value

    def call_many(self, requests) -> list[tuple[bool, Any]]:
        """Run each ``(chain, method, args, kwargs)`` in the worker holding the
        chain; return ``(True, value)`` or ``(False, exception)`` per request,
        in request order. Workers process their requests concurrently."""
        if os.getpid() != self.owner_pid:
            raise RuntimeError(
                f"these model handles belong to process {self.owner_pid}; its worker "
                f"processes cannot be called from process {os.getpid()}")
        if self.state["closed"]:
            raise RuntimeError(_CLOSED_MSG)
        out: list = [None] * len(requests)
        queues: dict[_Worker, list] = {}
        for k, (chain, name, args, kwargs) in enumerate(requests):
            w = self.by_chain.get(chain)
            if w is None:
                out[k] = (False, RuntimeError(
                    f"chain {chain} has no model in the worker processes of this run"))
                continue
            try:
                payload = _dumps(("call", chain, name, tuple(args), dict(kwargs)))
            except Exception as e:  # noqa: BLE001
                out[k] = (False, TypeError(
                    f"the arguments of {name}() cannot be sent to the worker process "
                    f"({type(e).__name__}: {e})"))
                continue
            queues.setdefault(w, []).append((k, name, payload))
        order = sorted(queues, key=lambda w: w.index)
        active: dict = {}

        def send_next(w):
            while queues[w]:
                k, name, payload = queues[w].pop(0)
                try:
                    if w.lost is not None:
                        raise _WorkerLost(w.lost)
                    w.drain()
                    w.send(payload)
                except _WorkerLost as e:
                    out[k] = (False, RuntimeError(str(e)))
                    continue
                active[w.conn] = (w, k, name)
                return

        held = []
        try:
            for w in order:             # one fixed order, so threads cannot deadlock
                w.lock.acquire()
                held.append(w)
            if self.state["closed"]:
                raise RuntimeError(_CLOSED_MSG)
            for w in order:
                send_next(w)
            while active:
                try:
                    ready = _mp_wait(list(active))
                except (OSError, ValueError):
                    # close() from another thread closed the connections.
                    if self.state["closed"]:
                        raise RuntimeError(_CLOSED_MSG) from None
                    raise
                for conn in ready:
                    w, k, name = active.pop(conn)
                    try:
                        out[k] = _reply_value(w.receive(), name)
                    except _WorkerLost as e:
                        out[k] = (False, RuntimeError(str(e)))
                    send_next(w)
        finally:
            for w in held:
                w.lock.release()
        return out


def _detached_chain_model(chain, cls_name, methods, reason):
    return ChainModel(None, chain, None, cls_name, methods, detached=reason)


class ChainModel:
    """Handle on the model of one chain, held by a run_chains() worker process.

    ``handle.name(*args, **kwargs)`` sends the call to the worker process
    holding the model, runs ``model.name(*args, **kwargs)`` there and returns
    the result. An exception raised there is raised here: with its own type
    when it can be pickled, else as RuntimeError whose message gives its type
    name and message; its ``__cause__`` holds the traceback text from the worker. A
    returned pybind11 record, which does not pickle (``get_dag()`` returns a
    DagInfo), is converted in the worker into a dict of its fields that also
    allows attribute access. Calls from several threads run one at a time per
    worker process.
    Names that start with "_" are not forwarded.

    After :func:`close_models` every call raises RuntimeError. A handle
    restored from a pickle is detached: its calls raise RuntimeError, because
    the worker processes belong to the process that started them.
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
        reason = self._detached if self._detached is not None else _DETACHED_PICKLED
        methods = None if self._methods is None else sorted(self._methods)
        return (_detached_chain_model, (self._chain, self._cls, methods, reason))

    # A handle refers to a live model; a copy refers to the same model.
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _rebuild_chains_result(chains, n_burn, n_keep, history_cleared):
    return ChainsResult(chains, n_burn=n_burn, n_keep=n_keep,
                        history_cleared=history_cleared)


class ChainsResult(list):
    """The value of :func:`run_chains`: a list with one dict per chain.

    Each dict has "seed", "wall" (seconds) and "hist" (the chain's draws after
    the warmup; None for a failed chain, which also has "error" and
    "error_type"). With ``keep_models=True`` it also has "model": the chain's
    model object, a :class:`ChainModel` handle on the model held by a worker
    process, or None for a failed chain.

    Attributes ``n_burn`` and ``n_keep``, and ``history_cleared``: one bool
    per chain, True when clear_history() was called after the warmup, so the
    model's own get_history() and predict_at() cover the n_keep kept draws
    only. Methods :meth:`predict` and :meth:`close`.

    A pickled result keeps its histories; a model handle unpickles as a
    detached handle whose calls raise RuntimeError, and a model object that
    cannot be pickled is replaced by such a handle.
    """

    def __init__(self, chains=(), *, n_burn=0, n_keep=0, history_cleared=None):
        super().__init__(chains)
        self.n_burn = int(n_burn)
        self.n_keep = int(n_keep)
        self.history_cleared = (list(history_cleared) if history_cleared is not None
                                else [False] * len(self))
        self._pool = None

    def predict(self, newdata=None, pool=True):
        """``predict(self, newdata, pool)``; see :func:`predict`."""
        return predict(self, newdata=newdata, pool=pool)

    def close(self) -> None:
        """``close_models(self)``; see :func:`close_models`."""
        close_models(self)

    def __reduce__(self):
        items = []
        for i, c in enumerate(self):
            m = c.get("model") if isinstance(c, dict) else None
            if m is not None and not isinstance(m, ChainModel):
                try:
                    _dumps(m)
                except Exception:  # noqa: BLE001
                    cls = type(m).__name__
                    c = dict(c)
                    c["model"] = ChainModel(
                        None, i, None, cls, [a for a in dir(m) if not a.startswith("_")],
                        detached=(f"the {cls} model object cannot be pickled, so the "
                                  "unpickled result holds this placeholder in its place "
                                  "(the histories are intact)"))
            items.append(c)
        return (_rebuild_chains_result,
                (items, self.n_burn, self.n_keep, list(self.history_cleared)))

    def __copy__(self):
        out = ChainsResult(self, n_burn=self.n_burn, n_keep=self.n_keep,
                           history_cleared=self.history_cleared)
        out._pool = self._pool
        return out

    def __deepcopy__(self, memo):
        import copy
        out = ChainsResult(n_burn=self.n_burn, n_keep=self.n_keep,
                           history_cleared=self.history_cleared)
        memo[id(self)] = out
        for c in self:
            if isinstance(c, dict):     # the models are shared, not copied
                c = {k: (v if k == "model" else copy.deepcopy(v, memo))
                     for k, v in c.items()}
            else:
                c = copy.deepcopy(c, memo)
            out.append(c)
        out._pool = self._pool
        return out


def _finish(results, n_burn, n_keep, *, pool=None, walls=False,
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
    if walls and verbose:
        print(f"[AI4BayesCode.run_chains] walls (s): "
              + ", ".join(f"{r['wall']:.1f}" for r in results))
    if no_clear:
        print(f"run_chains: {', '.join(no_clear)} has no clear_history(), so the "
              f"model's own get_history() and predict_at() include the {n_burn} "
              "warmup draws (the histories returned here do not). Recompiling the "
              "model with AI4BayesCode >= 1.1.3 adds clear_history() (a hand-written "
              "PYBIND11_MODULE binds it with AI4BAYESCODE_PYBIND_KERNEL_CONTROL).")
    if unclear:
        warnings.warn(
            "run_chains: after clear_history(), "
            + "; ".join(f"the history entries {', '.join(map(repr, ks))} of {cls}"
                        for cls, ks in unclear.items())
            + f" still hold more than n_keep = {n_keep} rows. The returned histories "
            f"drop their first {n_burn} rows, but the model's own get_history() and "
            "predict_at() still contain the warmup draws for them: clear_history() "
            "did not empty them. clear_history() calls the clear_extra_history_() "
            "hook declared in the class named in kernel_control_mixin<...>; a class "
            "derived from a wrapper declares clear_history() itself.")
    out = ChainsResult(results, n_burn=n_burn, n_keep=n_keep, history_cleared=cleared)
    out._pool = pool
    return out


def _run_in_workers(kind, factory, seeds, n_burn, n_keep, history_keys, n_workers):
    """Run the chains in ``n_workers`` persistent worker processes.

    Chain i runs in worker ``i % n_workers`` and its model is kept in that
    worker. Returns
    ``(results, pool)``, each result with "model": a ChainModel handle, or None
    for a failed chain (``pool`` is None when no chain has a model). Returns
    None, after a warning, when a worker process cannot be started or exits
    during the run; the caller then runs the chains in this process.
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
                    target=_worker_main, name=f"AI4BayesCode-chains-{w}", daemon=True,
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
        results = pool.collect(n)
    except Exception as e:  # noqa: BLE001
        pool.close()
        detail = str(e) if isinstance(e, _WorkerLost) else f"{type(e).__name__}: {e}"
        warnings.warn(f"run_chains: parallel ({kind}) execution failed ({detail}); "
                      f"running the {n} chains sequentially instead.")
        return None
    except BaseException:
        pool.close()
        raise
    for i, r in enumerate(results):
        r["model"] = (None if "error" in r else
                      ChainModel(pool, i, pool.by_chain[i].pid, r.get("_class"),
                                 r.get("_methods")))
    if all(r["model"] is None for r in results):
        pool.close()
        pool = None
    return results, pool


def predict(chains, newdata=None, pool=True):
    """Posterior prediction from every chain of a :func:`run_chains` result.

    Calls ``predict_at(newdata)`` on the model of each chain; the worker
    processes holding models compute their chains concurrently. ``newdata=None``
    means ``{}``: the posterior predictive at the training data. For a chain
    whose model had no clear_history() (``chains.history_cleared[i]`` is
    False), the first ``n_burn`` rows of every entry with ``n_burn + n_keep``
    rows are dropped. A chain that failed in run_chains() is left out, with a
    warning.

    Parameters
    ----------
    chains : ChainsResult
        The value of :func:`run_chains` with ``keep_models=True``.
    newdata : dict, optional
        Passed to each model's ``predict_at()``.
    pool : bool, default True
        True: ``{key: array}`` with the rows of all chains stacked along the
        first axis in chain order (``np.vstack`` for 2-D entries); an error
        when the chains disagree on the keys or on the other dimensions.
        False: a list with the ``predict_at()`` dict of each chain (None for a
        failed chain).
    """
    if newdata is None:
        newdata = {}
    if not isinstance(chains, list):
        raise TypeError("predict() takes the result of run_chains(); got "
                        f"{type(chains).__name__}")
    if not chains:
        raise ValueError("predict(): the run_chains() result holds no chains")
    if not all(isinstance(c, dict) and "model" in c for c in chains):
        raise ValueError(
            "predict() calls predict_at() on the models of the chains, and this "
            "run_chains() result does not hold them; rerun run_chains() with "
            "keep_models=True (the default)")
    use = [i for i, c in enumerate(chains) if c["model"] is not None]
    if not use:
        raise ValueError("predict(): every chain failed in run_chains(), so no chain "
                         "has a model")
    if len(use) < len(chains):
        warnings.warn("predict: chain(s) "
                      + ", ".join(str(i) for i in range(len(chains)) if i not in use)
                      + " failed in run_chains() and have no model; the prediction "
                        "uses the other chains")
    outs, err = {}, None
    groups: dict = {}
    for i in use:
        m = chains[i]["model"]
        if isinstance(m, ChainModel) and m._detached is None:
            groups.setdefault(id(m._pool), (m._pool, []))[1].append(i)
    for wpool, idx in groups.values():
        replies = wpool.call_many([(chains[i]["model"]._chain, "predict_at",
                                    (newdata,), {}) for i in idx])
        for i, (ok, value) in zip(idx, replies):
            if ok:
                outs[i] = value
            elif err is None:
                err = value
    if err is not None:
        raise err
    for i in use:
        if i not in outs:
            outs[i] = chains[i]["model"].predict_at(newdata)

    n_burn = int(getattr(chains, "n_burn", 0) or 0)
    n_keep = int(getattr(chains, "n_keep", 0) or 0)
    cleared = getattr(chains, "history_cleared", None)
    if n_burn > 0 and cleared is not None and len(cleared) == len(chains):
        for i in use:
            if not cleared[i] and isinstance(outs[i], Mapping):
                outs[i] = {k: (np.asarray(v)[n_burn:]
                               if np.ndim(v) > 0 and np.shape(v)[0] == n_burn + n_keep
                               else v)
                           for k, v in outs[i].items()}
    if not pool:
        return [outs.get(i) for i in range(len(chains))]

    for i in use:
        if not isinstance(outs[i], Mapping):
            raise TypeError(f"predict(): predict_at() of chain {i} returned "
                            f"{type(outs[i]).__name__}, not a dict of arrays")
    keys = list(outs[use[0]])
    for i in use[1:]:
        if set(outs[i]) != set(keys):
            raise ValueError(
                f"predict(): chains {use[0]} and {i} return different keys from "
                f"predict_at(): {sorted(keys)} and {sorted(outs[i])}")
    pooled = {}
    for k in keys:
        arrs = [np.atleast_1d(np.asarray(outs[i][k])) for i in use]
        if len({a.shape[1:] for a in arrs}) > 1:
            raise ValueError(
                f"predict(): the chains disagree on the columns of {k!r}: shapes "
                + ", ".join(str(a.shape) for a in arrs))
        pooled[k] = np.concatenate(arrs, axis=0)
    return pooled


def close_models(chains) -> None:
    """Stop the worker processes that hold the models of a :func:`run_chains`
    result.

    Afterwards every call on one of its model handles raises RuntimeError
    ("the worker processes holding the models of this run were closed").
    Calling it again does nothing, and so does calling it on a result whose
    models are objects in this process. The worker processes also stop when
    the result and all of its model handles are garbage-collected, and at
    interpreter exit.
    """
    pools = []
    p = getattr(chains, "_pool", None)
    if p is not None:
        pools.append(p)
    if isinstance(chains, list):
        for c in chains:
            m = c.get("model") if isinstance(c, dict) else None
            if (isinstance(m, ChainModel) and m._pool is not None
                    and not any(m._pool is q for q in pools)):
                pools.append(m._pool)
    for q in pools:
        q.close()
    return None


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
    """Run `len(seeds)` chains in parallel (or sequentially if n_cores=1).

    Each chain builds its model with ``factory(seed)`` and runs
    ``step(n_burn)``, ``clear_history()`` (when ``n_burn > 0`` and the model
    has it) and ``step(n_keep)``.

    Parameters
    ----------
    factory : Callable[[int], model]
        A callable that builds a fresh model given a seed. On Unix
        (macOS / Linux) the chains fork, so a lambda or local closure
        works directly -- just like R's ``function(s) new(Mod, ..., s)``::

            chains = AI4BayesCode.run_chains(
                lambda s: MyModel(y, X, seed=s, keep_history=True),
                seeds=[1, 2, 3, 4])

        On Windows see the module docstring: sequential by default, and
        parallel only for a top-level function called under the
        ``__main__`` guard. If the parallel pool cannot start, the chains
        fall back to running sequentially.

    seeds : iterable of int
        One seed per chain. Chain count = len(seeds).
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
        on Windows. Set to 1 for sequential execution.
    keep_models : bool, default True
        Return the models in ``chains[i]["model"]``. When the chains run in
        parallel (fork available, or a picklable factory on Windows,
        ``n_cores > 1`` and more than one chain), the worker processes are
        not stopped when run_chains() returns, and ``chains[i]["model"]`` is a
        :class:`ChainModel` handle whose method calls run in the worker
        process holding chain i; they stop at :func:`close_models`, when the
        result and its handles are garbage-collected, or at interpreter exit.
        Otherwise it is the model object itself. False: the models are
        discarded and there is no "model" key.

    Returns
    -------
    ChainsResult: a list of dicts, one per chain: `{"seed", "wall", "hist"}`,
    plus "model" with ``keep_models=True``. Each
    ``"hist"`` is KEEP-ONLY: the first ``n_burn`` draws of every history
    entry are stripped along its first axis (1-D, 2-D, and higher-D
    entries alike), so downstream ``rhat_summary`` / ``diagnose`` need no
    ``drop_burn`` / ``n_burn`` bookkeeping. An entry with a single stored
    draw (``keep_history=False``) is returned unchanged. A model without
    clear_history() keeps its warmup draws in its own history (a message
    says so); an entry that clear_history() leaves with more than n_keep
    rows is cut in "hist" and named in a warning. Attributes ``n_burn``,
    ``n_keep`` and ``history_cleared``; methods ``predict()`` and
    ``close()`` (see :func:`predict` and :func:`close_models`).
    """
    # R's ai4bayescode_run_chains takes n_chains; accept it so the same call
    # reads the same way in both frontends. Explicit seeds win.
    if n_chains is not None:
        if int(n_chains) < 1:
            raise ValueError(f"n_chains must be >= 1; got {n_chains}")
        if seeds is _DEFAULT_SEEDS:
            seeds = tuple(101 * (i + 1) for i in range(int(n_chains)))
    seeds = list(seeds)
    if not seeds:
        return ChainsResult([], n_burn=n_burn, n_keep=n_keep, history_cleared=[])
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
    n_jobs = n_cores
    if getattr(mp.current_process(), "_inheriting", False):
        raise RuntimeError(
            "run_chains() was called while a worker process was importing the main "
            "script. Put the call under  if __name__ == '__main__':  or pass n_cores=1.")

    args = [(factory, int(s), n_burn, n_keep, history_keys) for s in seeds]

    def _sequential() -> list[dict]:
        out = []
        for a in args:
            if verbose:
                print(f"[AI4BayesCode.run_chains] chain seed={a[1]} ...")
            res, m = _run_chain(*a)
            if keep_models:
                res["model"] = m
            out.append(res)
        return out

    if n_jobs == 1:
        return _finish(_sequential(), n_burn, n_keep, verbose=verbose)

    if keep_models:
        # The models are kept after the run: in persistent worker processes
        # when the chains run in parallel, else as objects in this process.
        results, pool = None, None
        if n_jobs > 1 and len(seeds) > 1:
            seed_list = [a[1] for a in args]
            got = None
            if can_fork:
                _warn_if_blas_unpinned()
                got = _run_in_workers("fork", factory, seed_list, n_burn, n_keep,
                                      history_keys, min(n_jobs, len(seeds)))
            else:
                reason = _spawn_obstacle(factory)
                if reason:
                    warnings.warn(f"run_chains: {reason}; running the {len(seeds)} "
                                  "chains sequentially.")
                else:
                    # On Windows connection.wait() takes at most 64 handles
                    # (WaitForMultipleObjects); 61 is also the pool limit below.
                    got = _run_in_workers("spawn", factory, seed_list, n_burn, n_keep,
                                          history_keys, min(n_jobs, len(seeds), 61))
            if got is not None:
                results, pool = got
                if (not can_fork and pool is None
                        and all(r.get("error_type") in _REBUILD_ERRORS for r in results)):
                    # As on the spawn path below: every worker failed the way a
                    # worker that cannot rebuild the factory fails. Run the first
                    # chain here: if it works, run the rest here as well; if it
                    # fails too, the results stand as they are.
                    if verbose:
                        print(f"[AI4BayesCode.run_chains] chain seed={args[0][1]} ...")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        first, first_model = _run_chain(*args[0])
                    if "error" not in first:
                        warnings.warn(
                            "run_chains: every parallel chain failed in its worker (on "
                            "Windows the factory and everything it uses must be defined at "
                            "module top level); running the chains sequentially instead.")
                        first["model"] = first_model
                        results = [first]
                        for a in args[1:]:
                            if verbose:
                                print(f"[AI4BayesCode.run_chains] chain seed={a[1]} ...")
                            res, m = _run_chain(*a)
                            res["model"] = m
                            results.append(res)
        if results is None:
            results = _sequential()
        return _finish(results, n_burn, n_keep, pool=pool, walls=True, verbose=verbose)

    results: Optional[list[dict]] = None
    if can_fork:
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
        global _ACTIVE_FACTORY
        _ACTIVE_FACTORY = factory
        try:
            _warn_if_blas_unpinned()
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
        # Windows (no fork): SPAWN fresh interpreters. Each one unpickles the
        # factory by importing the module that defines it, so check first that
        # this can work at all; otherwise run sequentially without spawning.
        reason = _spawn_obstacle(factory)
        if reason:
            warnings.warn(f"run_chains: {reason}; running the {len(seeds)} chains "
                          "sequentially.")
        else:
            from concurrent.futures import ProcessPoolExecutor
            try:
                ctx = mp.get_context("spawn")
                # Windows allows at most 61 worker processes per pool.
                with ProcessPoolExecutor(max_workers=min(n_jobs, 61), mp_context=ctx) as ex:
                    results = list(ex.map(_chain_worker, *zip(*args)))
            except Exception as e:  # noqa: BLE001  (BrokenProcessPool, ...)
                warnings.warn(
                    f"run_chains: parallel execution failed "
                    f"({type(e).__name__}: {e}); running the {len(seeds)} chains "
                    "sequentially instead.")
                results = None
            if results is not None and all(r.get("error_type") in _REBUILD_ERRORS
                                           for r in results):
                # Every worker failed the way a worker that cannot rebuild the
                # factory fails (a name defined only under the __main__ guard,
                # say). Run the first chain here: if it works, run the rest here
                # as well; if it fails too, the model itself fails and the
                # results stand as they are.
                if verbose:
                    print(f"[AI4BayesCode.run_chains] chain seed={args[0][1]} ...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    first = _chain_worker(*args[0])
                if "error" not in first:
                    warnings.warn(
                        "run_chains: every parallel chain failed in its worker (on "
                        "Windows the factory and everything it uses must be defined at "
                        "module top level); running the chains sequentially instead.")
                    results = [first]
                    for a in args[1:]:
                        if verbose:
                            print(f"[AI4BayesCode.run_chains] chain seed={a[1]} ...")
                        results.append(_chain_worker(*a))

    if results is None:
        results = _sequential()
    return _finish(results, n_burn, n_keep, walls=True, verbose=verbose)

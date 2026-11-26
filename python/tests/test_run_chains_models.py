"""run_chains() keeps the models of the chains: keep_models, predict, close_models.

Each chain runs step(n_burn), clear_history(), step(n_keep). The per-chain
dicts chains[i] keep the keys of AI4BayesCode 1.1.2 ("seed", "wall", "hist",
and "error", "error_type" for a failed chain); the models are in
chains.models. With parallel chains the worker processes are not stopped at
the end of the run and chains.models[i] is a ChainModel handle whose calls run
in the worker holding chain i; otherwise it is the model object.

Most tests use a pure-Python model (fast, no compiler). The end-to-end tests
compile shipped examples and are marked ``compile`` (deselect with
``-m "not compile"``). Tests of the fork code path are skipped where fork is
unavailable; the spawn code path is forced on any OS. Tests of what happens
when a process ends run that process as a subprocess with a time limit.

Run: python -m pytest python/tests/test_run_chains_models.py
"""
from __future__ import annotations

import copy
import gc
import importlib
import json
import multiprocessing as mp
import os
import pickle
import random
import signal
import subprocess
import sys
import textwrap
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import AI4BayesCode

rc = importlib.import_module("AI4BayesCode.run_chains")
ChainModel = rc.ChainModel
ChainsResult = rc.ChainsResult

HAS_FORK = sys.platform != "win32" and "fork" in mp.get_all_start_methods()
needs_fork = pytest.mark.skipif(not HAS_FORK, reason="the fork start method is unavailable")
needs_fd_dir = pytest.mark.skipif(not (os.path.isdir("/dev/fd") or os.path.isdir("/proc/self/fd")),
                                  reason="no descriptor directory to count open files")

# After test_generate.py has run in the same pytest process, the process has
# extra native threads, and Python >= 3.12 emits this DeprecationWarning at
# every fork (the fork path of AI4BayesCode 1.1.2 emits it there too). Run on
# its own, this module forks with a single thread and passes with
# -W error::DeprecationWarning.
pytestmark = pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork:DeprecationWarning")

N_BURN, N_KEEP = 40, 25
PKG_ROOT = str(Path(rc.__file__).resolve().parents[1])


class OldModel:
    """A pure-Python model without clear_history(): an AR(1) draw mu per sweep,
    beta = mu * (1, 2, -1), a history of both, and predict_at() with a
    deterministic entry ("mean") and a random one ("y_rep", drawn from a
    generator of its own, as the compiled models do)."""

    def __init__(self, seed, keep_history=True):
        self.rng = np.random.default_rng(seed)
        self.prng = np.random.default_rng(seed + 7919)
        self.keep = keep_history
        self.mu = 0.0
        self.h_mu, self.h_beta = [], []

    def step(self, n=1):
        for _ in range(int(n)):
            self.mu = 0.5 * self.mu + self.rng.normal()
            beta = self.mu * np.array([1.0, 2.0, -1.0])
            if self.keep:
                self.h_mu.append(self.mu)
                self.h_beta.append(beta)
            else:
                self.h_mu, self.h_beta = [self.mu], [beta]

    def get_history(self):
        return {"mu": np.array(self.h_mu).reshape(-1, 1),
                "beta": np.array(self.h_beta).reshape(-1, 3)}

    def get_current(self):
        return {"mu": np.array([self.mu])}

    def predict_at(self, newdata):
        x = np.asarray(newdata.get("x", [1.0, 2.0]), dtype=float)
        mean = np.array(self.h_mu)[:, None] * x[None, :]
        return {"mean": mean, "y_rep": mean + self.prng.normal(size=mean.shape)}

    # Helpers that the tests call through the handles.
    def pid(self):
        return os.getpid()

    def add(self, a, b=0):
        return a + b

    def nap(self, seconds):
        time.sleep(seconds)
        return seconds

    def fail(self, msg):
        raise ValueError(msg)

    def fail_key(self):
        raise KeyError("missing")

    def fail_unpicklable(self):
        raise TwoArgError("a", "b")

    def unpicklable_value(self):
        return threading.Lock()

    def echo(self, value):
        return value


class Model(OldModel):
    def clear_history(self):
        self.h_mu, self.h_beta = [], []


class LeakyModel(Model):
    """Records a per-sweep "trace" entry that clear_history() leaves in place,
    as a class without its clear_extra_history_() hook does."""

    def __init__(self, seed):
        super().__init__(seed)
        self.trace = []

    def step(self, n=1):
        for _ in range(int(n)):
            super().step(1)
            self.trace.append(self.mu)

    def get_history(self):
        h = super().get_history()
        h["trace"] = np.array(self.trace)
        return h


class UnpicklableModel(Model):
    def __init__(self, seed):
        super().__init__(seed)
        self.lock = threading.Lock()


class NoisyModel(Model):
    """Prints to stdout and stderr and warns in step(), predict_at() and
    hello()."""

    def __init__(self, seed):
        super().__init__(seed)
        self.seed = seed

    def step(self, n=1):
        super().step(n)
        print(f"OUT step seed={self.seed}")
        print(f"ERR step seed={self.seed}", file=sys.stderr)
        warnings.warn(f"WARN step seed={self.seed}", RuntimeWarning)

    def predict_at(self, newdata):
        warnings.warn(f"WARN predict seed={self.seed}")
        return super().predict_at(newdata)

    def hello(self):
        print(f"OUT hello seed={self.seed}")
        warnings.warn(f"WARN hello seed={self.seed}")
        return self.seed


class GlobalRngModel(Model):
    """Draws from numpy's global generator and from the random module, in the
    constructor and in every sweep."""

    def __init__(self, seed):
        super().__init__(seed)
        self.init = np.random.uniform()
        self.h_g = []

    def step(self, n=1):
        for _ in range(int(n)):
            self.h_g.append(np.random.normal() + random.gauss(0.0, 1.0))
        super().step(n)

    def clear_history(self):
        super().clear_history()
        self.h_g = []

    def get_history(self):
        h = super().get_history()
        h["g"] = np.array(self.h_g)
        h["init"] = np.full(len(self.h_g), self.init)
        return h


class PoolUser(Model):
    """Starts processes of its own: in the constructor and in a method."""

    def __init__(self, seed):
        super().__init__(seed)
        with ProcessPoolExecutor(1, mp_context=mp.get_context("fork")) as ex:
            self.child = ex.submit(os.getpid).result()

    def pool_sum(self):
        with mp.get_context("fork").Pool(2) as p:
            return sum(p.map(abs, [-1, -2, -3]))


class Nesting(Model):
    """Runs an inner run_chains() with two worker processes in its
    constructor and keeps the inner result."""

    def __init__(self, seed):
        super().__init__(seed)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            self.inner = AI4BayesCode.run_chains(make_model, seeds=[seed, seed + 1],
                                                 n_burn=2, n_keep=3, n_cores=2,
                                                 verbose=False)
        self.inner_warnings = [str(w.message) for w in rec]

    def inner_pids(self):
        return [m.pid() for m in self.inner.models]

    def inner_info(self):
        return (self.inner_pids(), self.inner_warnings, os.getpid())


class TwoArgError(Exception):
    """Pickles, but does not unpickle: __init__ takes two arguments."""

    def __init__(self, a, b):
        super().__init__(f"{a}+{b}")


class RecordingModel(Model):
    """predict_at() returns the newdata it received."""

    def predict_at(self, newdata):
        out = super().predict_at({})
        out["got"] = {k: (type(v).__name__, np.shape(v), np.asarray(v).tolist())
                      for k, v in newdata.items()}
        return out


def make_model(seed):
    return Model(seed)


def make_old(seed):
    return OldModel(seed)


def make_leaky(seed):
    return LeakyModel(seed)


def make_noisy(seed):
    return NoisyModel(seed)


def make_global_rng(seed):
    return GlobalRngModel(seed)


def make_fail_202(seed):
    if seed == 202:
        raise RuntimeError("no model for seed 202")
    return Model(seed)


def main_only_factory(seed):
    # In a spawned worker the module is imported fresh, so a name that only
    # the test sets in this process does not exist there.
    return MAIN_ONLY(seed)   # noqa: F821


def run(factory=make_model, seeds=(101, 202), **kw):
    kw.setdefault("n_burn", N_BURN)
    kw.setdefault("n_keep", N_KEEP)
    kw.setdefault("verbose", False)
    return AI4BayesCode.run_chains(factory, seeds=list(seeds), **kw)


def assert_hist_equal(a, b):
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))


def gone(pid, timeout=10.0):
    """True once no process with this pid exists; polls for up to `timeout` s."""
    end = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() > end:
            return False
        time.sleep(0.05)


def kill_all(pids):
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def open_fds():
    d = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
    return len(os.listdir(d))


def run_script(tmp_path, source, *args, timeout=120, env_extra=None):
    """Run ``source`` as a script in a fresh interpreter; returns the
    CompletedProcess (text output)."""
    script = tmp_path / f"script_{abs(hash(source)) % 10**8}.py"
    script.write_text(textwrap.dedent(source))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [PKG_ROOT] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(script), *map(str, args)], env=env,
                          timeout=timeout, capture_output=True, text=True)


@pytest.fixture()
def windows_path(monkeypatch):
    """The code path of a platform without fork (spawned workers)."""
    class _Sys:
        platform = "win32"

        def __getattr__(self, name):
            return getattr(sys, name)
    monkeypatch.setattr(rc, "sys", _Sys())


# --------------------------------------------------------------------------
# (1) handles on keep-only models, same draws on every code path
# --------------------------------------------------------------------------
@needs_fork
def test_parallel_models_are_handles_on_keep_only_models():
    chains = run(n_cores=2)
    try:
        assert isinstance(chains, ChainsResult) and isinstance(chains, list)
        assert (chains.n_burn, chains.n_keep) == (N_BURN, N_KEEP)
        assert chains.history_cleared == [True, True]
        assert all(isinstance(m, ChainModel) for m in chains.models)
        # The per-chain dicts have exactly the keys of 1.1.2.
        assert [sorted(c) for c in chains] == [["hist", "seed", "wall"]] * 2
        assert all(type(c) is dict for c in chains)
        pids = [m.pid() for m in chains.models]
        assert len(set(pids)) == 2 and os.getpid() not in pids
        assert all(f"worker process {p}" in repr(m) for p, m in zip(pids, chains.models))
        for c, m in zip(chains, chains.models):
            h = m.get_history()
            assert h["mu"].shape == (N_KEEP, 1) and h["beta"].shape == (N_KEEP, 3)
            assert_hist_equal(h, c["hist"])
            assert m.predict_at({})["mean"].shape == (N_KEEP, 2)
        # The same seeds give the same draws without kept models (the
        # ProcessPoolExecutor path) and sequentially in this process.
        plain = run(n_cores=2, keep_models=False)
        seq = run(n_cores=1)
        assert plain.models is None
        assert plain.history_cleared == [True, True]
        for a, b, s in zip(chains, plain, seq):
            assert_hist_equal(a["hist"], b["hist"])
            assert_hist_equal(a["hist"], s["hist"])
    finally:
        chains.close()


# --------------------------------------------------------------------------
# (2) predict over all chains
# --------------------------------------------------------------------------
@needs_fork
def test_predict_stacks_the_chains_in_chain_order():
    chains = run(seeds=(101, 202, 303), n_cores=2)    # chains 0 and 2 share a worker
    try:
        nd = {"x": [1.0, 2.0, 3.0]}
        per = [m.predict_at(nd) for m in chains.models]
        pooled = AI4BayesCode.predict(chains, nd)
        assert set(pooled) == {"mean", "y_rep"}
        assert pooled["mean"].shape == pooled["y_rep"].shape == (3 * N_KEEP, 3)
        np.testing.assert_array_equal(pooled["mean"], np.vstack([p["mean"] for p in per]))
        np.testing.assert_array_equal(chains.predict(nd)["mean"], pooled["mean"])
        sep = AI4BayesCode.predict(chains, nd, pool=False)
        assert isinstance(sep, list) and len(sep) == 3
        for s, p in zip(sep, per):
            np.testing.assert_array_equal(s["mean"], p["mean"])
        # newdata=None means {}: predict_at() at the training data.
        assert AI4BayesCode.predict(chains)["mean"].shape == (3 * N_KEEP, 2)
    finally:
        chains.close()


@needs_fork
def test_predict_gives_the_values_of_the_sequential_models():
    # predict_at() draws y_rep from a generator seeded by the chain seed, so the
    # first predict_at() of each parallel model equals that of the same model
    # run sequentially.
    par = run(seeds=(101, 202, 303), n_cores=2)
    seq = run(seeds=(101, 202, 303), n_cores=1)
    try:
        got = par.predict()["y_rep"]
        np.testing.assert_array_equal(
            got, np.vstack([m.predict_at({})["y_rep"] for m in seq.models]))
    finally:
        par.close()


def test_predict_without_models_raises():
    plain = run(n_cores=1, keep_models=False)
    with pytest.raises(ValueError, match="keep_models=True"):
        AI4BayesCode.predict(plain)
    plain.close()                       # nothing to stop


@pytest.mark.parametrize("n_cores", [1, pytest.param(2, marks=needs_fork)])
def test_predict_flattens_matrices_column_major(n_cores):
    chains = run(lambda s: RecordingModel(s), n_cores=n_cores)
    try:
        X = np.arange(12.0).reshape(4, 3)          # 4 new rows, 3 covariates
        got = AI4BayesCode.predict(chains, {"X": X, "w": 2.5, "v": [1, 2],
                                            "rows": [[1.0, 2.0], [3.0, 4.0]],
                                            "name": "abc"}, pool=False)
        for g in got:
            seen = g["got"]
            assert seen["X"] == ("ndarray", (12,), X.ravel(order="F").tolist())
            assert seen["w"] == ("ndarray", (1,), [2.5])
            assert seen["v"] == ("ndarray", (2,), [1, 2])
            assert seen["rows"] == ("ndarray", (4,), [1.0, 3.0, 2.0, 4.0])
            assert seen["name"][0] == "str"
        with pytest.raises(TypeError, match=r'predict\(chains, \{"X": X_new\}\).*'
                                            r'got ndarray of shape \(4, 3\)'):
            AI4BayesCode.predict(chains, X)
    finally:
        chains.close()


# --------------------------------------------------------------------------
# (3) stateful calls
# --------------------------------------------------------------------------
@needs_fork
def test_calls_through_a_handle_change_the_model_in_its_worker():
    chains = run(n_cores=2)
    try:
        m0, m1 = chains.models
        m0.step(5)
        h0 = m0.get_history()["mu"]
        assert h0.shape == (N_KEEP + 5, 1)
        np.testing.assert_array_equal(h0[:N_KEEP], chains[0]["hist"]["mu"])
        assert m1.get_history()["mu"].shape == (N_KEEP, 1)
        assert m0.add(2, b=3) == 5
    finally:
        chains.close()


# --------------------------------------------------------------------------
# (4) lifecycle: close, garbage collection, end of the parent process
# --------------------------------------------------------------------------
@needs_fork
def test_close_stops_the_workers_and_later_calls_raise():
    chains = run(n_cores=2)
    pids = [m.pid() for m in chains.models]
    AI4BayesCode.close_models(chains)
    assert all(gone(p) for p in pids)
    for m in chains.models:
        with pytest.raises(RuntimeError, match="were closed"):
            m.get_history()
        assert "(closed)" in repr(m)
    with pytest.raises(RuntimeError, match="were closed"):
        AI4BayesCode.predict(chains)
    assert AI4BayesCode.close_models(chains) is None   # a second close does nothing
    chains.close()
    assert chains[0]["hist"]["mu"].shape == (N_KEEP, 1)
    with pytest.raises(TypeError, match="takes the result of run_chains"):
        AI4BayesCode.close_models(list(chains))


@needs_fork
@needs_fd_dir
def test_close_releases_every_descriptor_of_a_kept_result():
    base = open_fds()
    kept = []
    for _ in range(3):
        chains = run(n_cores=2)
        assert open_fds() >= base + 6                # 3 per worker process
        chains.close()
        kept.append(chains)                          # the closed results stay referenced
    assert open_fds() == base
    assert all(c[0]["hist"]["mu"].shape == (N_KEEP, 1) for c in kept)


@needs_fork
def test_workers_stop_when_the_result_and_its_handles_are_collected():
    chains = run(n_cores=2)
    pids = [m.pid() for m in chains.models]
    handle = chains.models[1]
    del chains
    gc.collect()
    assert handle.add(1, 1) == 2               # a handle keeps the workers running
    assert not gone(pids[0], timeout=0.3)
    del handle
    gc.collect()
    assert all(gone(p) for p in pids)


@needs_fork
def test_failed_calls_leave_no_reference_cycle():
    def boom_predict(seed):
        m = Model(seed)
        if seed == 202:
            m.predict_at = lambda nd: (_ for _ in ()).throw(ValueError("no prediction"))
        return m

    gc.disable()
    try:
        for factory, how in [(make_model, "method"), (make_model, "argument"),
                             (boom_predict, "predict")]:
            chains = run(factory, n_cores=2)
            h = chains.models[0]
            pids = [m.pid() for m in chains.models]
            if how == "method":
                with pytest.raises(ValueError, match="chain 0: boom"):
                    h.fail("boom")
            elif how == "argument":
                with pytest.raises(TypeError, match="cannot be sent"):
                    h.add(threading.Lock())
            else:
                with pytest.raises(ValueError, match="chain 1: no prediction"):
                    chains.predict()
            del chains, h
            # Reference counting alone stops the workers: no gc.collect().
            assert all(gone(p, timeout=10) for p in pids), how
    finally:
        gc.enable()


PARENT_SCRIPT = '''
    import os, signal, sys
    import AI4BayesCode

    class M:
        def step(self, n):
            pass
        def get_history(self):
            return {}
        def pid(self):
            return os.getpid()

    chains = AI4BayesCode.run_chains(lambda s: M(), seeds=[1, 2], n_burn=1,
                                     n_keep=1, n_cores=2, verbose=False)
    with open(sys.argv[1], "w") as fh:
        fh.write(" ".join(str(m.pid()) for m in chains.models))
    if sys.argv[2] == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
'''


@needs_fork
@pytest.mark.parametrize("how", ["exit", "kill"])
def test_workers_stop_when_the_parent_process_ends(tmp_path, how):
    pid_file = tmp_path / "pids.txt"
    run_script(tmp_path, PARENT_SCRIPT, pid_file, how)
    pids = [int(p) for p in pid_file.read_text().split()]
    assert len(pids) == 2
    try:
        assert all(gone(p, timeout=15) for p in pids)
    finally:
        kill_all(pids)


EXIT_WITH_A_FINALIZER_FIRST = '''
    import os, sys, tempfile, time
    keep = tempfile.TemporaryDirectory()      # a weakref.finalize created first
    import AI4BayesCode

    class M:
        def step(self, n):
            pass
        def get_history(self):
            return {}
        def pid(self):
            return os.getpid()

    chains = AI4BayesCode.run_chains(lambda s: M(), seeds=[1, 2], n_burn=1,
                                     n_keep=1, n_cores=2, verbose=False)
    with open(sys.argv[1], "w") as fh:
        fh.write(" ".join(str(m.pid()) for m in chains.models))
    print("END", time.monotonic(), flush=True)
'''


@needs_fork
def test_exit_without_close_is_prompt_with_non_daemonic_workers(tmp_path):
    pid_file = tmp_path / "pids.txt"
    t0 = time.monotonic()
    out = run_script(tmp_path, EXIT_WITH_A_FINALIZER_FIRST, pid_file, timeout=60)
    assert out.returncode == 0, out.stderr
    assert time.monotonic() - t0 < 20
    pids = [int(p) for p in pid_file.read_text().split()]
    try:
        assert all(gone(p, timeout=5) for p in pids)
    finally:
        kill_all(pids)


FORK_CHILD_SCRIPT = '''
    import os, sys
    import numpy as np
    import AI4BayesCode

    class M:
        def __init__(self, s):
            self.h = []
        def step(self, n):
            self.h += [1.0] * n
        def clear_history(self):
            self.h = []
        def get_history(self):
            return {"y": np.array(self.h)}
        def predict_at(self, nd):
            return {"y": np.array(self.h).reshape(-1, 1)}

    chains = AI4BayesCode.run_chains(lambda s: M(s), seeds=[1, 2], n_burn=2,
                                     n_keep=3, n_cores=2, verbose=False)
    pid = os.fork()
    if pid == 0:
        # The child ends through the normal interpreter shutdown.
        if sys.argv[1] == "exit":
            sys.exit(0)
        raise ValueError("the child ends with an uncaught exception")
    os.waitpid(pid, 0)
    pool = chains.models[0]._pool
    print("CODES", [w.process.exitcode for w in pool.workers])
    print("ROWS", AI4BayesCode.predict(chains)["y"].shape[0])
    chains.close()
'''


@needs_fork
@pytest.mark.parametrize("how", ["exit", "uncaught"])
def test_a_forked_child_that_exits_leaves_the_workers_alone(tmp_path, how):
    out = run_script(tmp_path, FORK_CHILD_SCRIPT, how, timeout=60)
    assert "CODES [None, None]" in out.stdout, out.stdout + out.stderr
    assert "ROWS 6" in out.stdout
    assert "can only join a child process" not in out.stderr


# --------------------------------------------------------------------------
# (5) sequential runs return the model objects
# --------------------------------------------------------------------------
def test_sequential_runs_return_the_model_objects():
    seq = run(n_cores=1)
    assert [type(m) for m in seq.models] == [Model, Model]
    assert seq.history_cleared == [True, True]
    assert seq.models[0].get_history()["mu"].shape == (N_KEEP, 1)
    assert seq.predict({"x": [1.0]})["mean"].shape == (2 * N_KEEP, 1)
    seq.close()                                  # nothing to stop
    assert seq.models[0].add(1) == 1
    one = run(seeds=(7,), n_cores=4)             # a single chain runs here
    assert type(one.models[0]) is Model


def test_no_seeds_gives_an_empty_result():
    empty = AI4BayesCode.run_chains(make_model, seeds=[], verbose=False)
    assert isinstance(empty, ChainsResult) and empty == []


@pytest.mark.parametrize("n_cores", [0, -1])
@pytest.mark.parametrize("keep_models", [True, False])
def test_n_cores_below_one_runs_sequentially(capsys, recwarn, n_cores, keep_models):
    chains = run(n_cores=n_cores, keep_models=keep_models, verbose=True)
    out = capsys.readouterr().out
    assert "walls" not in out and out.count("chain seed=") == 2
    assert not [w for w in recwarn if "run_chains" in str(w.message)]
    if keep_models:
        assert [m.pid() for m in chains.models] == [os.getpid()] * 2
    else:
        assert chains.models is None


# --------------------------------------------------------------------------
# (6) a model without clear_history(), and the row check after the clear
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_cores", [1, pytest.param(2, marks=needs_fork)])
def test_a_model_without_clear_history_falls_back(capsys, n_cores):
    chains = run(make_old, n_cores=n_cores)
    try:
        out = capsys.readouterr().out
        assert out.count("has no clear_history() method") == 1 and "OldModel" in out
        assert "n_burn = 40 warmup draws" in out and "predict() drops them" in out
        assert chains.history_cleared == [False, False]
        for c, m in zip(chains, chains.models):
            assert c["hist"]["mu"].shape == (N_KEEP, 1)
            own = m.get_history()["mu"]
            assert own.shape == (N_BURN + N_KEEP, 1)
            np.testing.assert_array_equal(own[N_BURN:], c["hist"]["mu"])
        # predict drops the warmup rows of these chains
        pooled = AI4BayesCode.predict(chains)
        assert pooled["mean"].shape == (2 * N_KEEP, 2)
        full = [m.predict_at({})["mean"] for m in chains.models]
        np.testing.assert_array_equal(pooled["mean"],
                                      np.vstack([f[N_BURN:] for f in full]))
        assert "without clear_history()" in repr(chains)
    finally:
        chains.close()


def test_predict_keeps_the_warmup_bookkeeping_per_chain():
    chains = run(make_old, seeds=(101, 202, 303), n_cores=1)
    # A slice, a copy made with list() and a comprehension are plain lists
    # without the models and the bookkeeping: predict() refuses them.
    for derived in (chains[:2], list(chains), chains.copy(),
                    [c for c in chains if c["seed"] != 202]):
        with pytest.raises(TypeError, match=r"takes the result of run_chains\(\)"):
            AI4BayesCode.predict(derived)
    # Removing a dict leaves the models: predict() uses chains.models.
    del chains[1]
    assert AI4BayesCode.predict(chains)["mean"].shape == (3 * N_KEEP, 2)
    chains.history_cleared = chains.history_cleared[:2]
    with pytest.raises(ValueError, match="chains.models has 3 entries"):
        AI4BayesCode.predict(chains)


def test_fallback_message_without_kept_models_and_without_warmup(capsys):
    plain = run(make_old, n_cores=1, keep_models=False)
    assert capsys.readouterr().out.count("has no clear_history()") == 1
    assert plain[0]["hist"]["mu"].shape == (N_KEEP, 1)
    nowarm = run(make_old, n_burn=0, n_cores=1)
    assert "clear_history" not in capsys.readouterr().out
    assert nowarm.history_cleared == [False, False]
    assert nowarm.models[0].get_history()["mu"].shape == (N_KEEP, 1)


@pytest.mark.parametrize("n_cores", [1, pytest.param(2, marks=needs_fork)])
def test_entries_left_by_clear_history_are_cut_and_named_once(n_cores):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        chains = run(make_leaky, n_cores=n_cores)
    try:
        msgs = [str(w.message) for w in rec if "clear_extra_history_" in str(w.message)]
        assert len(msgs) == 1
        assert "LeakyModel" in msgs[0] and "'trace'" in msgs[0] and "'mu'" not in msgs[0]
        assert chains.history_cleared == [True, True]
        for c, m in zip(chains, chains.models):
            assert c["hist"]["mu"].shape == (N_KEEP, 1)
            assert c["hist"]["trace"].shape == (N_KEEP,)
            np.testing.assert_array_equal(c["hist"]["trace"], c["hist"]["mu"][:, 0])
            assert m.get_history()["trace"].shape == (N_BURN + N_KEEP,)
    finally:
        chains.close()


def test_history_keys_and_single_draw_models():
    sub = run(n_cores=1, history_keys=["beta"])
    assert list(sub[0]["hist"]) == ["beta"] and sub[0]["hist"]["beta"].shape == (N_KEEP, 3)
    single = run(lambda s: Model(s, keep_history=False), n_cores=1)
    assert single[0]["hist"]["mu"].shape == (1, 1)
    assert single.history_cleared == [True, True]


# --------------------------------------------------------------------------
# (7) n_cores / n_jobs / n_chains, and code written for 1.1.2
# --------------------------------------------------------------------------
def test_n_jobs_is_an_alias_of_n_cores():
    with pytest.raises(ValueError, match="n_jobs is an alias of n_cores"):
        run(n_cores=2, n_jobs=3)
    assert type(run(n_jobs=1).models[0]) is Model
    assert type(run(n_cores=1, n_jobs=1).models[0]) is Model


@needs_fork
def test_n_jobs_alone_runs_parallel_workers():
    chains = run(n_jobs=2)
    try:
        assert all(isinstance(m, ChainModel) for m in chains.models)
    finally:
        chains.close()


def test_n_chains_and_seeds_of_another_length_raise():
    with pytest.raises(ValueError, match=r"len\(seeds\) = 3 must equal n_chains = 2"):
        AI4BayesCode.run_chains(make_model, n_chains=2, seeds=[1, 2, 3], n_jobs=1,
                                verbose=False)
    with pytest.raises(ValueError, match=r"len\(seeds\) = 0 must equal n_chains = 2"):
        AI4BayesCode.run_chains(make_model, n_chains=2, seeds=[], verbose=False)
    ok = AI4BayesCode.run_chains(make_model, n_chains=2, seeds=[1, 2], n_burn=2,
                                 n_keep=3, n_jobs=1, verbose=False)
    assert [c["seed"] for c in ok] == [1, 2]
    derived = AI4BayesCode.run_chains(make_model, n_chains=3, n_burn=2, n_keep=3,
                                      n_jobs=1, verbose=False)
    assert [c["seed"] for c in derived] == [101, 202, 303]


@pytest.mark.parametrize("n_jobs", [1, pytest.param(2, marks=needs_fork)])
def test_a_script_written_for_1_1_2(n_jobs):
    # The calls, the loops and the serialisation of a 1.1.2 script.
    chains = AI4BayesCode.run_chains(make_fail_202, n_chains=3, n_burn=N_BURN,
                                     n_keep=N_KEEP, n_jobs=n_jobs, verbose=False)
    try:
        assert len(chains) == 3
        ok = []
        for c in chains:
            if "error" in c:
                assert set(c) == {"seed", "wall", "hist", "error", "error_type"}
                assert c["hist"] is None and c["seed"] == 202
                continue
            assert set(c) == {"seed", "wall", "hist"}
            json.dumps({"seed": c["seed"], "wall": c["wall"],
                        "hist": {k: v.tolist() for k, v in c["hist"].items()}})
            ok.append(c)
        s = AI4BayesCode.rhat_summary(chains)
        assert set(s) == {"mu", "beta"}
        summ, _ = AI4BayesCode.diagnose(ok[0], plot=False)
        assert len(summ) == 4
        # n_jobs=1 and n_jobs=2 give the same draws.
        ref = AI4BayesCode.run_chains(make_fail_202, n_chains=3, n_burn=N_BURN,
                                      n_keep=N_KEEP, n_jobs=1, verbose=False)
        for a, b in zip(chains, ref):
            if "error" not in a:
                assert_hist_equal(a["hist"], b["hist"])
    finally:
        chains.close()


# --------------------------------------------------------------------------
# (8) pickling and copying
# --------------------------------------------------------------------------
@needs_fork
def test_a_result_unpickled_in_this_process_refers_to_the_same_models():
    chains = run(n_cores=2)
    try:
        back = pickle.loads(pickle.dumps(chains))
        assert isinstance(back, ChainsResult)
        assert (back.n_burn, back.n_keep, back.history_cleared) == (N_BURN, N_KEEP, [True, True])
        for a, b, m in zip(chains, back, back.models):
            assert (a["seed"], a["wall"]) == (b["seed"], b["wall"])
            assert_hist_equal(a["hist"], b["hist"])
            assert isinstance(m, ChainModel) and "worker process" in repr(m)
        assert back.models[0].pid() == chains.models[0].pid()
        back.models[0].step(2)                       # the same model
        assert chains.models[0].get_history()["mu"].shape == (N_KEEP + 2, 1)
        single = pickle.loads(pickle.dumps(chains.models[1]))
        assert single.add(1, 2) == 3
        back.close()                                 # closes the workers of both
        with pytest.raises(RuntimeError, match="were closed"):
            chains.models[1].get_history()
    finally:
        chains.close()


DETACH_SCRIPT = '''
    import pickle, sys
    import AI4BayesCode
    back = pickle.loads(open(sys.argv[1], "rb").read())
    print("REPR", repr(back.models[0]))
    try:
        back.models[0].get_history()
    except RuntimeError as e:
        print("ERR", e)
    back.close()
    print("ROWS", back[0]["hist"]["mu"].shape[0])
'''


@needs_fork
def test_a_result_unpickled_in_another_process_is_detached(tmp_path):
    chains = run(make_model, n_cores=2)
    try:
        blob = tmp_path / "chains.pkl"
        blob.write_bytes(pickle.dumps(chains))
        out = run_script(tmp_path, DETACH_SCRIPT, blob, timeout=60)
        assert "detached" in out.stdout, out.stdout + out.stderr
        assert "ERR chain 0: this model handle was restored from a pickle" in out.stdout
        assert f"ROWS {N_KEEP}" in out.stdout
        assert chains.models[0].add(1, 1) == 2       # the original is unaffected
    finally:
        chains.close()


def test_a_pickled_result_with_model_objects():
    seq = run(n_cores=1)
    back = pickle.loads(pickle.dumps(seq))
    assert type(back.models[0]) is Model             # a picklable model is kept
    assert_hist_equal(back.models[0].get_history(), seq[0]["hist"])
    held = run(lambda s: UnpicklableModel(s), n_cores=1)
    back = pickle.loads(pickle.dumps(held))
    assert isinstance(back.models[0], ChainModel)
    with pytest.raises(RuntimeError, match="cannot be pickled"):
        back.models[0].get_history()
    assert_hist_equal(back[0]["hist"], held[0]["hist"])


def _hist_rows(c):
    return c["hist"]["mu"].shape[0]


def test_the_per_chain_dicts_pickle_and_copy_like_in_1_1_2():
    held = run(lambda s: UnpicklableModel(s), n_cores=1)
    for c in held:
        assert_hist_equal(pickle.loads(pickle.dumps(c))["hist"], c["hist"])
        assert_hist_equal(copy.deepcopy(c)["hist"], c["hist"])
    pickle.dumps(held[:1])
    pickle.dumps(list(held))
    with mp.get_context("spawn").Pool(2) as p:
        assert p.map(_hist_rows, held) == [N_KEEP, N_KEEP]


@needs_fork
def test_copies_share_the_models():
    chains = run(n_cores=2)
    try:
        deep = copy.deepcopy(chains)
        assert isinstance(deep, ChainsResult)
        assert deep.models[0] is chains.models[0]
        assert deep[0]["hist"]["mu"] is not chains[0]["hist"]["mu"]
        assert deep.predict()["mean"].shape == (2 * N_KEEP, 2)
        shallow = copy.copy(chains)
        assert isinstance(shallow, ChainsResult) and shallow[0] is chains[0]
        assert shallow.models[1] is chains.models[1]
    finally:
        chains.close()


# --------------------------------------------------------------------------
# (9) the printed summary
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_cores", [1, pytest.param(2, marks=needs_fork)])
def test_the_result_prints_a_summary(n_cores):
    chains = run(make_fail_202, seeds=(101, 202, 303), n_cores=n_cores, n_keep=400)
    try:
        text = repr(chains)
        lines = text.splitlines()
        assert len(lines) <= 8 and str(chains) == text
        assert "3 chains, n_burn = 40 warmup + n_keep = 400 kept sweeps" in lines[0]
        assert any(l.split() == ["mu", "400", "x", "1"] for l in lines)
        assert "Failed chains: 1 (RuntimeError)" in text
        if n_cores > 1:
            assert "Models: in 2 worker processes" in text
            chains.close()
            assert "Models: closed (close_models)" in repr(chains)
        else:
            assert "Models: in this Python session" in text
        assert "not kept" in repr(run(n_cores=1, keep_models=False))
    finally:
        chains.close()


# --------------------------------------------------------------------------
# (10) the diagnostics take a ChainsResult
# --------------------------------------------------------------------------
@needs_fork
def test_rhat_summary_and_diagnose_take_the_result():
    chains = run(seeds=(101, 202, 303, 404), n_cores=2, n_keep=300)
    try:
        s = AI4BayesCode.rhat_summary(chains)
        assert set(s) == {"mu", "beta"}
        assert all(np.isfinite(v["max_rhat"]) for v in s.values())
        for arg in (chains[0]["hist"], chains[0], chains.models[0]):
            summ, _ = AI4BayesCode.diagnose(arg, plot=False)
            assert len(summ) == 4                   # mu, beta[0], beta[1], beta[2]
        with pytest.raises(TypeError, match="list of 4 chains"):
            AI4BayesCode.diagnose(chains)
    finally:
        chains.close()


# --------------------------------------------------------------------------
# (11) random-number streams of numpy's global generator and of random
# --------------------------------------------------------------------------
@pytest.mark.parametrize("keep_models", [True, pytest.param(False, marks=needs_fork)])
def test_global_generators_are_seeded_per_chain(keep_models):
    np.random.seed(7)
    random.seed(7)
    before = (np.random.get_state()[1].copy(), random.getstate())
    seq = run(make_global_rng, seeds=(11, 22, 33, 44), n_cores=1, keep_models=keep_models)
    # A run in this process leaves the caller's streams unchanged.
    assert np.array_equal(np.random.get_state()[1], before[0])
    assert random.getstate() == before[1]
    g = [c["hist"]["g"] for c in seq]
    assert len({float(x[0]) for x in g}) == 4            # every chain has its own stream
    assert len({float(c["hist"]["init"][0]) for c in seq}) == 4
    if HAS_FORK:
        for n_cores in (2, 4):
            par = run(make_global_rng, seeds=(11, 22, 33, 44), n_cores=n_cores,
                      keep_models=keep_models)
            try:
                for a, b in zip(seq, par):
                    assert_hist_equal(a["hist"], b["hist"])
            finally:
                par.close()


# --------------------------------------------------------------------------
# Errors, failed chains, a lost worker, threads, interrupts
# --------------------------------------------------------------------------
@needs_fork
def test_exceptions_in_the_worker_are_raised_here():
    chains = run(n_cores=2)
    m = chains.models[0]
    try:
        with pytest.raises(ValueError, match="^chain 0: boom$") as ei:
            m.fail("boom")
        assert "in fail" in str(ei.value.__cause__)    # the traceback in the worker
        assert m.add(1, 2) == 3                         # the handle can still be called
        with pytest.raises(KeyError) as ek:
            chains.models[1].fail_key()
        assert ek.value.args == ("missing",)            # the key itself is kept ...
        assert "chain 1: raised by the model of chain 1" in ek.value.__notes__
        with pytest.raises(RuntimeError, match=r"^chain 0: TwoArgError: a\+b"):
            m.fail_unpicklable()
        with pytest.raises(RuntimeError, match="cannot be sent to the calling process"):
            m.unpicklable_value()
        with pytest.raises(TypeError, match="cannot be sent to the worker process"):
            m.add(threading.Lock())
        assert m.add(2, b=2) == 4
        with pytest.raises(AttributeError):
            m._private
        with pytest.raises(AttributeError, match="no attribute 'no_such_method'"):
            m.no_such_method
        assert not hasattr(m, "no_such_method") and hasattr(m, "clear_history")
        assert "predict_at" in dir(m)
        with pytest.raises(AttributeError, match="cannot set 'step'"):
            m.step = None
        assert m.add(3) == 3
    finally:
        chains.close()


def failing_predict(seed):
    m = Model(seed)
    if seed == 303:
        def predict_at(newdata):
            raise RuntimeError("chol(): decomposition failed")
        m.predict_at = predict_at
    return m


@pytest.mark.parametrize("n_cores", [1, pytest.param(2, marks=needs_fork)])
def test_predict_names_the_chain_of_an_error(n_cores):
    chains = run(failing_predict, seeds=(101, 202, 303), n_cores=n_cores)
    try:
        for pool in (True, False):
            with pytest.raises(RuntimeError,
                               match=r"^chain 2: chol\(\): decomposition failed$") as ei:
                AI4BayesCode.predict(chains, pool=pool)
            if n_cores > 1:
                assert "decomposition failed" in str(ei.value.__cause__)
        if n_cores > 1:
            with pytest.raises(RuntimeError, match="^chain 2: chol"):
                chains.models[2].predict_at({})
    finally:
        chains.close()


@needs_fork
def test_a_failed_chain_has_no_model_and_predict_leaves_it_out():
    chains = run(make_fail_202, n_cores=2)
    try:
        assert "no model for seed 202" in chains[1]["error"]
        assert chains[1]["hist"] is None and chains.models[1] is None
        assert chains.history_cleared == [True, False]
        assert chains.models[0].add(1, 1) == 2
        with pytest.warns(UserWarning, match=r"chain\(s\) 1 failed"):
            pooled = chains.predict()
        assert pooled["mean"].shape == (N_KEEP, 2)
        with pytest.warns(UserWarning):
            sep = chains.predict(pool=False)
        assert sep[1] is None and sep[0]["mean"].shape == (N_KEEP, 2)
    finally:
        chains.close()


@needs_fork
def test_no_worker_stays_when_every_chain_fails():
    chains = run(lambda s: 1 / 0, n_cores=2)
    assert chains.models == [None, None]
    assert all(c["error_type"] == "ZeroDivisionError" for c in chains)
    assert not [p for p in rc._open_pools()]
    with pytest.raises(ValueError, match="every chain failed"):
        chains.predict()


@needs_fork
def test_a_killed_worker_loses_only_its_chains():
    chains = run(n_cores=2)
    try:
        m0, m1 = chains.models
        os.kill(m0.pid(), signal.SIGKILL)
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            m0.get_history()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            m0.add(1)
        assert "(exited)" in repr(m0)
        assert m1.add(1, 2) == 3
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            chains.predict()
    finally:
        chains.close()


PARENT_PID = os.getpid()


def dies_in_a_worker(seed):
    if os.getpid() != PARENT_PID and seed == 202:
        os._exit(3)            # a worker process exits in the middle of the run
    return Model(seed)


def chain_workers():
    return {p.pid for p in mp.active_children() if p.name.startswith("AI4BayesCode-chains")}


@needs_fork
def test_a_worker_lost_during_the_run_falls_back_to_this_process():
    before = chain_workers()
    with pytest.warns(UserWarning, match=r"parallel \(fork\) execution failed .*exit code "
                                         r"3.*VECLIB_MAXIMUM_THREADS"):
        chains = run(dies_in_a_worker, n_cores=2)
    assert chain_workers() == before               # the other worker was stopped
    assert [type(m) for m in chains.models] == [Model, Model]
    assert chains.history_cleared == [True, True]
    for a, b in zip(chains, run(n_cores=1)):
        assert_hist_equal(a["hist"], b["hist"])


@needs_fork
def test_an_interrupt_during_the_run_stops_the_workers(monkeypatch):
    before = chain_workers()

    def interrupted_wait(objs, timeout=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(rc, "_mp_wait", interrupted_wait)
    with pytest.raises(KeyboardInterrupt):
        run(n_cores=2)
    assert chain_workers() == before


@needs_fork
def test_close_from_another_thread_during_a_call():
    chains = run(n_cores=2)
    m = chains.models[0]
    closer = threading.Timer(0.3, chains.close)
    closer.start()
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="were closed"):
        m.nap(30.0)
    closer.join()
    assert time.monotonic() - t0 < 15.0            # the busy worker was terminated
    with pytest.raises(RuntimeError, match="were closed"):
        m.add(1)


class OddModel(Model):
    """predict_at() of seed 202 has another key ("extra") or another width."""

    def __init__(self, seed, mode):
        super().__init__(seed)
        self.seed, self.mode = seed, mode

    def predict_at(self, newdata):
        out = super().predict_at(newdata)
        if self.seed == 202 and self.mode == "key":
            out["extra"] = out["mean"]
        if self.seed == 202 and self.mode == "width":
            out = {k: np.hstack([v, v]) for k, v in out.items()}
        return out


def test_predict_rejects_chains_with_different_outputs():
    by_key = run(lambda s: OddModel(s, "key"), n_cores=1)
    with pytest.raises(ValueError, match="different keys"):
        by_key.predict()
    assert [sorted(p) for p in by_key.predict(pool=False)] == [
        ["mean", "y_rep"], ["extra", "mean", "y_rep"]]
    by_width = run(lambda s: OddModel(s, "width"), n_cores=1)
    with pytest.raises(ValueError, match="different shapes after the first axis for 'mean'"):
        by_width.predict()


@needs_fork
def test_calls_from_threads_do_not_interleave():
    chains = run(seeds=(101, 202, 303), n_cores=2)
    errors = []

    def work(t):
        try:
            for j in range(30):
                assert chains.models[(t + j) % 3].add(t, b=j) == t + j
                if j % 10 == 0:
                    assert chains.predict()["mean"].shape == (3 * N_KEEP, 2)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    try:
        threads = [threading.Thread(target=work, args=(t,)) for t in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        assert not errors
    finally:
        chains.close()


@needs_fork
def test_an_interrupted_call_leaves_the_handle_usable(monkeypatch):
    chains = run(n_cores=2)
    m = chains.models[0]
    real_wait = rc._mp_wait
    hit = []

    def interrupted_wait(objs, timeout=None):
        if not hit:
            hit.append(1)
            raise KeyboardInterrupt
        return real_wait(objs, timeout)

    try:
        monkeypatch.setattr(rc, "_mp_wait", interrupted_wait)
        with pytest.raises(KeyboardInterrupt):
            m.nap(0.3)              # its reply is still due when the call ends
        monkeypatch.setattr(rc, "_mp_wait", real_wait)
        assert m.add(40, b=2) == 42       # the late reply of nap() is discarded first
        assert_hist_equal(m.get_history(), chains[0]["hist"])
    finally:
        chains.close()


@pytest.fixture()
def default_sigint():
    """Python's own SIGINT handler (raises KeyboardInterrupt) for the test: a
    process started in the background by a shell inherits SIGINT ignored."""
    old = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, old)


@needs_fork
def test_ctrl_c_during_a_call_and_sigint_to_a_worker(default_sigint):
    chains = run(n_cores=2)
    m = chains.models[0]
    inside = threading.Event()

    def interrupt():
        if inside.wait(10):
            time.sleep(0.3)
            if inside.is_set():
                signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)

    sender = threading.Thread(target=interrupt)
    try:
        sender.start()
        t0 = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            inside.set()
            try:
                m.nap(3.0)
            finally:
                inside.clear()
        assert time.monotonic() - t0 < 2.0    # the interrupt ended the blocked call
        sender.join()
        assert m.add(1, b=1) == 2         # read after the reply of nap(3.0)
        # Ctrl+C in a terminal also reaches the worker processes, which ignore it.
        os.kill(m.pid(), signal.SIGINT)
        time.sleep(0.2)
        assert m.add(2, b=2) == 4
    finally:
        chains.close()


EXIT_DURING_A_CALL = '''
    import os, sys, threading, time
    import AI4BayesCode

    class M:
        def __init__(self, s):
            self.s = s
        def step(self, n):
            if self.s == 99:
                time.sleep(30)
        def clear_history(self):
            pass
        def get_history(self):
            return {}
        def pid(self):
            return os.getpid()
        def nap(self, s):
            time.sleep(s)
            return s
        def predict_at(self, nd):
            time.sleep(30)
            return {}

    chains = AI4BayesCode.run_chains(lambda s: M(s), seeds=[1, 2], n_burn=1,
                                     n_keep=1, n_cores=2, verbose=False)
    with open(sys.argv[1], "w") as fh:
        fh.write(" ".join(str(m.pid()) for m in chains.models))
    mode = sys.argv[2]

    def body():
        if mode == "call":
            chains.models[0].nap(30.0)
        elif mode == "predict":
            chains.predict()
        else:                   # inside run_chains() itself
            AI4BayesCode.run_chains(lambda s: M(99), seeds=[3, 4], n_burn=1,
                                    n_keep=1, n_cores=2, verbose=False)
        print("RETURNED", flush=True)

    threading.Thread(target=body, daemon=True).start()
    time.sleep(0.5)
'''


@needs_fork
@pytest.mark.parametrize("mode,repeat", [("call", 8), ("predict", 3), ("run", 3)])
def test_exit_while_a_daemon_thread_is_in_a_call(tmp_path, mode, repeat):
    # The interpreter exits while a daemon thread waits for a worker process.
    # The exit stops the workers; the call must not return into the thread,
    # whose traceback output during interpreter finalization could abort
    # CPython. The loop stops at the first abort.
    for k in range(repeat):
        pid_file = tmp_path / f"pids_{k}.txt"
        t0 = time.monotonic()
        out = run_script(tmp_path, EXIT_DURING_A_CALL, pid_file, mode, timeout=60)
        assert out.returncode == 0 and "Fatal Python error" not in out.stderr, (
            f"run {k}: exit code {out.returncode}\n{out.stderr}")
        assert "RETURNED" not in out.stdout and "Traceback" not in out.stderr, out.stderr
        assert time.monotonic() - t0 < 30
        pids = [int(p) for p in pid_file.read_text().split()]
        try:
            assert all(gone(p, timeout=5) for p in pids)
        finally:
            kill_all(pids)


# --------------------------------------------------------------------------
# Output and warnings of the workers, shown in this process
# --------------------------------------------------------------------------
def _relayed(capsys, rec):
    out = capsys.readouterr()
    return (sorted(l for l in out.out.splitlines() if l.startswith("OUT")),
            sorted(l for l in out.err.splitlines() if l.startswith("ERR")),
            sorted(str(w.message) for w in rec if str(w.message).startswith("WARN")))


@pytest.mark.parametrize("keep_models", [True, False])
def test_output_and_warnings_of_the_workers_are_shown_here(capsys, keep_models):
    got = {}
    for n_cores in [1] + ([2] if HAS_FORK else []):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            chains = run(make_noisy, n_cores=n_cores, keep_models=keep_models)
            run_part = _relayed(capsys, rec)
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                if keep_models:
                    chains.predict()
                    assert chains.models[1].hello() == 202
            call_part = _relayed(capsys, rec)
        finally:
            chains.close()
        got[n_cores] = (run_part, call_part)
    if HAS_FORK:
        assert got[2] == got[1]
    run_part, call_part = got[1]
    assert run_part[0] == sorted([f"OUT step seed={s}" for s in (101, 202)] * 2)
    assert len(run_part[2]) == 4
    if keep_models:
        assert call_part[0] == ["OUT hello seed=202"]
        assert call_part[2] == sorted(["WARN predict seed=101", "WARN predict seed=202",
                                       "WARN hello seed=202"])


@needs_fork
def test_warnings_of_a_handle_call_reach_pytest_warns_and_filters():
    # run_chains() inside a recorder: the worker processes do not keep it.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        chains = run(make_noisy, n_cores=2)
    try:
        with pytest.warns(UserWarning, match="WARN hello seed=101") as rec:
            chains.models[0].hello()
        assert rec[0].filename == __file__            # located at the calling line
        with pytest.warns(UserWarning, match="WARN predict"):
            chains.predict()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(UserWarning, match="WARN hello seed=202"):
                chains.models[1].hello()
    finally:
        chains.close()


@pytest.mark.parametrize("keep_models", [True, pytest.param(False, marks=needs_fork)])
@pytest.mark.parametrize("n_cores", [1, pytest.param(2, marks=needs_fork)])
def test_the_failed_chain_warning_is_issued_here(keep_models, n_cores):
    with pytest.warns(UserWarning, match="chain seed=202 failed: no model for seed 202"):
        chains = run(make_fail_202, n_cores=n_cores, keep_models=keep_models)
    chains.close()
    # Under an "error" filter the chain's own warning is raised here, and the
    # worker processes of the run are stopped.
    before = chain_workers()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="chain seed=202 failed"):
            run(make_fail_202, n_cores=n_cores, keep_models=keep_models)
    assert chain_workers() <= before


# --------------------------------------------------------------------------
# Processes started by a model in a worker process
# --------------------------------------------------------------------------
@needs_fork
def test_a_model_can_start_processes_in_its_worker():
    chains = run(lambda s: PoolUser(s), n_cores=2)
    try:
        assert all(m is not None for m in chains.models), [c.get("error") for c in chains]
        assert chains.models[0].pool_sum() == 6
    finally:
        chains.close()


@needs_fork
def test_a_nested_run_runs_in_parallel_and_is_closed_with_its_worker():
    chains = run(lambda s: Nesting(s), n_cores=2)
    try:
        inner, inner_warnings, worker = chains.models[0].inner_info()
        assert len(set(inner)) == 2 and worker not in inner and os.getpid() not in inner
        assert not [w for w in inner_warnings if "execution failed" in w]
        t0 = time.monotonic()
        chains.close()
        assert time.monotonic() - t0 < 1.5
        assert all(gone(p, timeout=5) for p in inner)
    finally:
        chains.close()


# --------------------------------------------------------------------------
# The bound on the worker processes kept by earlier runs
# --------------------------------------------------------------------------
@needs_fork
def test_forty_kept_runs_stay_within_the_bound():
    bound = max(8, 4 * (os.cpu_count() or 1))
    for p in rc._open_pools():          # runs that earlier tests left open
        p.close()
    kept, evictions = [], 0
    try:
        for k in range(40):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                chains = run(seeds=(1, 2, 3, 4), n_cores=4, n_burn=2, n_keep=3)
            msgs = [str(w.message) for w in rec if "closed the worker processes" in str(w.message)]
            assert len(msgs) <= 1
            evictions += len(msgs)
            kept.append(chains)
            live = sum(len(p.workers) for p in rc._open_pools())
            assert live <= bound
            assert len(chain_workers()) <= bound
        assert evictions == 40 - bound // 4
        # The least recently used runs were closed; their handles say why.
        with pytest.raises(RuntimeError, match="closed by a later run_chains"):
            kept[0].models[0].get_history()
        assert "closed by a later run_chains() call" in repr(kept[0])
        assert kept[-1].models[3].add(1, 1) == 2
        assert all(c[0]["hist"]["mu"].shape == (3, 1) for c in kept)
    finally:
        for c in kept:
            c.close()


@needs_fork
def test_the_bound_follows_the_environment_variable(monkeypatch):
    for p in rc._open_pools():          # runs that earlier tests left open
        p.close()
    monkeypatch.setenv("AI4BAYESCODE_MAX_WORKERS", "4")
    a, b = run(n_cores=2), run(n_cores=2)
    try:
        a.models[0].add(1)                       # a is now more recently used than b
        with pytest.warns(UserWarning, match=r"closed the worker processes of 1 earlier "
                                             r"run \(2 worker processes.*close_models"):
            c = run(n_cores=2)
        with pytest.raises(RuntimeError, match="AI4BAYESCODE_MAX_WORKERS"):
            b.models[0].add(1)
        assert a.models[0].add(1, 1) == 2 and c.models[1].add(1, 1) == 2
        # A run that needs more workers than the bound runs here, and closes
        # nothing.
        monkeypatch.setenv("AI4BAYESCODE_MAX_WORKERS", "1")
        with pytest.warns(UserWarning, match="also after closing every earlier run"):
            d = run(n_cores=2)
        assert [type(m) for m in d.models] == [Model, Model]
        assert a.models[0].add(1, 1) == 2
        monkeypatch.setenv("AI4BAYESCODE_MAX_WORKERS", "many")
        with pytest.raises(ValueError, match="AI4BAYESCODE_MAX_WORKERS"):
            run(n_cores=2)
    finally:
        for r in (a, b):
            r.close()
        if "c" in locals():
            c.close()


FD_LIMIT_SCRIPT = '''
    import resource, sys, warnings
    import AI4BayesCode

    class M:
        def step(self, n):
            pass
        def clear_history(self):
            pass
        def get_history(self):
            return {}
        def one(self):
            return 1

    resource.setrlimit(resource.RLIMIT_NOFILE, (int(sys.argv[1]), resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    kept, msgs = [], []
    for k in range(25):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            kept.append(AI4BayesCode.run_chains(lambda s: M(), seeds=[1, 2], n_burn=1,
                                                n_keep=1, n_cores=2, verbose=False))
        msgs += [str(w.message) for w in rec]
    print("HANDLES", all(type(m).__name__ == "ChainModel" for c in kept for m in c.models))
    print("LAST", kept[-1].models[0].one())
    print("EMFILE", any("Too many open files" in m for m in msgs))
    print("FILES", sum("open files of this process" in m for m in msgs))
'''


@needs_fork
@needs_fd_dir
def test_the_open_file_limit_closes_earlier_runs_first(tmp_path):
    out = run_script(tmp_path, FD_LIMIT_SCRIPT, 64, timeout=120)
    assert "HANDLES True" in out.stdout, out.stdout + out.stderr
    assert "LAST 1" in out.stdout and "EMFILE False" in out.stdout
    assert "FILES 0" not in out.stdout


# --------------------------------------------------------------------------
# The spawn code path (Windows), forced on any OS
# --------------------------------------------------------------------------
def test_spawned_workers_hold_the_models(windows_path):
    chains = run(n_cores=2)
    try:
        assert all(isinstance(m, ChainModel) for m in chains.models)
        pids = [m.pid() for m in chains.models]
        assert len(set(pids)) == 2 and os.getpid() not in pids
        for c, m in zip(chains, chains.models):
            assert_hist_equal(m.get_history(), c["hist"])
        assert chains.predict()["mean"].shape == (2 * N_KEEP, 2)
        for a, b in zip(chains, run(n_cores=1)):
            assert_hist_equal(a["hist"], b["hist"])
    finally:
        chains.close()
    assert all(gone(p) for p in pids)


def test_without_a_picklable_factory_the_models_stay_here(windows_path):
    with pytest.warns(UserWarning, match="cannot be sent to worker processes"):
        chains = run(lambda s: Model(s), n_cores=2)
    assert [type(m) for m in chains.models] == [Model, Model]


@pytest.mark.parametrize("keep_models", [True, False])
def test_spawn_workers_that_cannot_rebuild_the_factory(windows_path, monkeypatch,
                                                       keep_models):
    monkeypatch.setitem(globals(), "MAIN_ONLY", Model)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        chains = run(main_only_factory, n_cores=2, keep_models=keep_models)
    assert [c.get("error") for c in chains] == [None, None]
    assert any("running the chains sequentially" in str(w.message)
               and "NameError" in str(w.message) for w in rec)
    assert chains.history_cleared == [True, True]
    if keep_models:
        assert [type(m) for m in chains.models] == [Model, Model]
    else:
        assert chains.models is None


# --------------------------------------------------------------------------
# End to end with compiled models
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gls():
    from AI4BayesCode.source import _find_armadillo
    if _find_armadillo() is None:
        pytest.skip("no Armadillo headers to compile the example with")
    return AI4BayesCode.example("GaussianLocationScale", quiet=True)


@pytest.fixture(scope="module")
def logreg():
    from AI4BayesCode.source import _find_armadillo
    if _find_armadillo() is None:
        pytest.skip("no Armadillo headers to compile the example with")
    return AI4BayesCode.example("LogisticRegression", quiet=True)


@pytest.mark.compile
@needs_fork
def test_compiled_model_end_to_end(gls):
    y = np.random.default_rng(42).normal(3.0, 2.0, 60)

    def factory(s):
        return gls.GaussianLocationScale(y, s, True)

    nb, nk, seeds = 300, 200, [101, 202]
    chains = AI4BayesCode.run_chains(factory, seeds=seeds, n_burn=nb, n_keep=nk,
                                     n_cores=2, verbose=False)
    try:
        assert chains.history_cleared == [True, True]
        assert all(isinstance(m, ChainModel) for m in chains.models)
        seq = AI4BayesCode.run_chains(factory, seeds=seeds, n_burn=nb, n_keep=nk,
                                      n_cores=1, verbose=False)
        plain = AI4BayesCode.run_chains(factory, seeds=seeds, n_burn=nb, n_keep=nk,
                                        n_cores=2, keep_models=False, verbose=False)
        assert [type(m).__name__ for m in seq.models] == ["GaussianLocationScale"] * 2
        # First predict_at() of every model: y_rep comes from a generator seeded
        # by the chain seed, so the pooled values equal the sequential ones.
        pooled = AI4BayesCode.predict(chains)
        assert pooled["y_rep"].shape == (2 * nk, y.size)
        np.testing.assert_array_equal(
            pooled["y_rep"], np.vstack([m.predict_at({})["y_rep"] for m in seq.models]))
        for c, m, s, p in zip(chains, chains.models, seq, plain):
            h = m.get_history()
            assert {k: v.shape for k, v in h.items()} == {"mu": (nk, 1), "sigma": (nk, 1)}
            assert_hist_equal(h, c["hist"])
            assert_hist_equal(c["hist"], s["hist"])
            assert_hist_equal(c["hist"], p["hist"])
            assert m.predict_at({})["y_rep"].shape == (nk, y.size)
        sep = AI4BayesCode.predict(chains, pool=False)
        assert [s["y_rep"].shape for s in sep] == [(nk, y.size)] * 2
        # get_dag() returns a DagInfo, which does not pickle: through a handle
        # it is rebuilt here as a DagInfo.
        from AI4BayesCode.plot_dag import _dag_to_edges
        dag, ref = chains.models[0].get_dag(), seq.models[0].get_dag()
        assert type(dag) is type(ref)
        assert all(getattr(dag, k) == getattr(ref, k) for k in ref.keys())
        assert dag.predict_edges == ref.predict_edges
        assert _dag_to_edges(dag) == _dag_to_edges(ref)
        m0 = chains.models[0]
        m0.freeze(["mu"])
        assert m0.get_frozen() == seq.models[0].get_frozen() + ["mu_sigma_joint.mu"]
        with pytest.warns(UserWarning, match="already frozen"):
            m0.freeze(["mu"])                         # a C++ warning, shown here
        m0.unfreeze()
        assert m0.get_frozen() == []
        assert m0.predict_at({}, last_draw_only=True)["y_rep"].shape == (1, y.size)
        with pytest.raises(RuntimeError, match="^chain 0: .*no covariate inputs"):
            m0.predict_at({"x": np.ones(3)})       # a C++ exception in the worker
        chains.models[0].step(5)
        assert chains.models[0].get_history()["mu"].shape == (nk + 5, 1)
        assert chains.models[1].get_history()["mu"].shape == (nk, 1)
        assert set(AI4BayesCode.rhat_summary(chains)) == {"mu", "sigma"}
        AI4BayesCode.diagnose(chains[0]["hist"], plot=False)
        pids = [m._pid for m in chains.models]      # the worker process ids
    finally:
        chains.close()
    assert all(gone(p) for p in pids)
    with pytest.raises(RuntimeError, match="were closed"):
        chains.models[0].get_history()


@pytest.mark.compile
@needs_fork
def test_compiled_predict_takes_a_matrix(logreg):
    rng = np.random.default_rng(2026)
    N, p = 200, 3
    X = np.column_stack([np.ones(N), rng.standard_normal((N, p - 1))])
    y = (rng.random(N) < 1 / (1 + np.exp(-(X @ np.array([-0.5, 1.2, -0.8]))))).astype(float)
    X_new = np.column_stack([np.ones(7), rng.standard_normal((7, p - 1))])

    def factory(s):
        return logreg.LogisticRegression(X, y, 10.0, s, True)

    for n_cores in (1, 2):
        chains = AI4BayesCode.run_chains(factory, seeds=[11, 22], n_burn=100, n_keep=150,
                                         n_cores=n_cores, verbose=False)
        try:
            by_matrix = AI4BayesCode.predict(chains, {"X": X_new})
            by_vector = AI4BayesCode.predict(chains, {"X": X_new.ravel(order="F")})
            assert set(by_matrix) == set(by_vector)
            for k in by_matrix:
                assert by_matrix[k].shape[0] == 300
                if k not in ("y_rep",):
                    np.testing.assert_array_equal(by_matrix[k], by_vector[k])
        finally:
            chains.close()

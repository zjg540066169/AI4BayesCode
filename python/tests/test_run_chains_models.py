"""run_chains() keeps the models of the chains: keep_models, predict, close_models.

Each chain runs step(n_burn), clear_history(), step(n_keep). With parallel
chains the worker processes are not stopped when run_chains() returns, and
chains[i]["model"] is a ChainModel handle whose calls run in the worker
holding chain i; otherwise it is the model object.

Most tests use a pure-Python model (fast, no compiler). The end-to-end test
compiles GaussianLocationScale and is marked ``compile`` (deselect with
``-m "not compile"``). Tests of the fork code path are skipped where fork is
unavailable; the spawn code path is forced on any OS.

Run: python -m pytest python/tests/test_run_chains_models.py
"""
from __future__ import annotations

import copy
import gc
import importlib
import multiprocessing as mp
import os
import pickle
import signal
import subprocess
import sys
import textwrap
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

import AI4BayesCode

rc = importlib.import_module("AI4BayesCode.run_chains")
ChainModel = rc.ChainModel
ChainsResult = rc.ChainsResult

HAS_FORK = sys.platform != "win32" and "fork" in mp.get_all_start_methods()
needs_fork = pytest.mark.skipif(not HAS_FORK, reason="the fork start method is unavailable")

# After test_generate.py has run in the same pytest process, the process has
# extra native threads, and Python >= 3.12 emits this DeprecationWarning at
# every fork (the fork path of AI4BayesCode 1.1.2 emits it there too). Run on
# its own, this module forks with a single thread and passes with
# -W error::DeprecationWarning.
pytestmark = pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork:DeprecationWarning")

N_BURN, N_KEEP = 40, 25


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

    def fail_unpicklable(self):
        raise TwoArgError("a", "b")

    def unpicklable_value(self):
        return threading.Lock()


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


class TwoArgError(Exception):
    """Pickles, but does not unpickle: __init__ takes two arguments."""

    def __init__(self, a, b):
        super().__init__(f"{a}+{b}")


def make_model(seed):
    return Model(seed)


def make_old(seed):
    return OldModel(seed)


def make_leaky(seed):
    return LeakyModel(seed)


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
        assert all(isinstance(c["model"], ChainModel) for c in chains)
        assert [sorted(c) for c in chains] == [["hist", "model", "seed", "wall"]] * 2
        pids = [c["model"].pid() for c in chains]
        assert len(set(pids)) == 2 and os.getpid() not in pids
        assert all(f"worker process {p}" in repr(c["model"]) for p, c in zip(pids, chains))
        for c in chains:
            h = c["model"].get_history()
            assert h["mu"].shape == (N_KEEP, 1) and h["beta"].shape == (N_KEEP, 3)
            assert_hist_equal(h, c["hist"])
            assert c["model"].predict_at({})["mean"].shape == (N_KEEP, 2)
        # The same seeds give the same draws without kept models (the
        # ProcessPoolExecutor path) and sequentially in this process.
        plain = run(n_cores=2, keep_models=False)
        seq = run(n_cores=1)
        assert all("model" not in c for c in plain)
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
        per = [c["model"].predict_at(nd) for c in chains]
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
            got, np.vstack([c["model"].predict_at({})["y_rep"] for c in seq]))
    finally:
        par.close()


def test_predict_without_models_raises():
    plain = run(n_cores=1, keep_models=False)
    with pytest.raises(ValueError, match="keep_models=True"):
        AI4BayesCode.predict(plain)
    plain.close()                       # nothing to stop


# --------------------------------------------------------------------------
# (3) stateful calls
# --------------------------------------------------------------------------
@needs_fork
def test_calls_through_a_handle_change_the_model_in_its_worker():
    chains = run(n_cores=2)
    try:
        m0, m1 = chains[0]["model"], chains[1]["model"]
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
    pids = [c["model"].pid() for c in chains]
    AI4BayesCode.close_models(chains)
    assert all(gone(p) for p in pids)
    for c in chains:
        with pytest.raises(RuntimeError, match="were closed"):
            c["model"].get_history()
        assert "(closed)" in repr(c["model"])
    with pytest.raises(RuntimeError, match="were closed"):
        AI4BayesCode.predict(chains)
    assert AI4BayesCode.close_models(chains) is None   # a second close does nothing
    chains.close()
    assert chains[0]["hist"]["mu"].shape == (N_KEEP, 1)


@needs_fork
def test_workers_stop_when_the_result_and_its_handles_are_collected():
    chains = run(n_cores=2)
    pids = [c["model"].pid() for c in chains]
    handle = chains[1]["model"]
    del chains
    gc.collect()
    assert handle.add(1, 1) == 2               # the handle references the pool, so the workers are not stopped
    assert not gone(pids[0], timeout=0.3)
    del handle
    gc.collect()
    assert all(gone(p) for p in pids)


PARENT_SCRIPT = textwrap.dedent('''
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
        fh.write(" ".join(str(c["model"].pid()) for c in chains))
    if sys.argv[2] == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
''')


@needs_fork
@pytest.mark.parametrize("how", ["exit", "kill"])
def test_workers_stop_when_the_parent_process_ends(tmp_path, how):
    script = tmp_path / "parent.py"
    script.write_text(PARENT_SCRIPT)
    pid_file = tmp_path / "pids.txt"
    pkg_root = str(Path(rc.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [pkg_root] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    subprocess.run([sys.executable, str(script), str(pid_file), how], env=env,
                   timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pids = [int(p) for p in pid_file.read_text().split()]
    assert len(pids) == 2
    try:
        assert all(gone(p, timeout=15) for p in pids)
    finally:
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass


# --------------------------------------------------------------------------
# (5) sequential runs return the model objects
# --------------------------------------------------------------------------
def test_sequential_runs_return_the_model_objects():
    seq = run(n_cores=1)
    assert [type(c["model"]) for c in seq] == [Model, Model]
    assert seq.history_cleared == [True, True]
    assert seq[0]["model"].get_history()["mu"].shape == (N_KEEP, 1)
    assert seq.predict({"x": [1.0]})["mean"].shape == (2 * N_KEEP, 1)
    seq.close()                                  # nothing to stop
    assert seq[0]["model"].add(1) == 1
    one = run(seeds=(7,), n_cores=4)             # a single chain runs here
    assert type(one[0]["model"]) is Model


def test_no_seeds_gives_an_empty_result():
    empty = AI4BayesCode.run_chains(make_model, seeds=[], verbose=False)
    assert isinstance(empty, ChainsResult) and empty == []


# --------------------------------------------------------------------------
# (6) a model without clear_history(), and the row check after the clear
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_cores", [1, pytest.param(2, marks=needs_fork)])
def test_a_model_without_clear_history_falls_back(capsys, n_cores):
    chains = run(make_old, n_cores=n_cores)
    try:
        out = capsys.readouterr().out
        assert out.count("has no clear_history()") == 1 and "OldModel" in out
        assert chains.history_cleared == [False, False]
        for c in chains:
            assert c["hist"]["mu"].shape == (N_KEEP, 1)
            own = c["model"].get_history()["mu"]
            assert own.shape == (N_BURN + N_KEEP, 1)
            np.testing.assert_array_equal(own[N_BURN:], c["hist"]["mu"])
        # predict drops the warmup rows of these chains
        pooled = AI4BayesCode.predict(chains)
        assert pooled["mean"].shape == (2 * N_KEEP, 2)
        full = [c["model"].predict_at({})["mean"] for c in chains]
        np.testing.assert_array_equal(pooled["mean"],
                                      np.vstack([f[N_BURN:] for f in full]))
    finally:
        chains.close()


def test_fallback_message_without_kept_models_and_without_warmup(capsys):
    plain = run(make_old, n_cores=1, keep_models=False)
    assert capsys.readouterr().out.count("has no clear_history()") == 1
    assert plain[0]["hist"]["mu"].shape == (N_KEEP, 1)
    nowarm = run(make_old, n_burn=0, n_cores=1)
    assert "clear_history" not in capsys.readouterr().out
    assert nowarm.history_cleared == [False, False]
    assert nowarm[0]["model"].get_history()["mu"].shape == (N_KEEP, 1)


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
        for c in chains:
            assert c["hist"]["mu"].shape == (N_KEEP, 1)
            assert c["hist"]["trace"].shape == (N_KEEP,)
            np.testing.assert_array_equal(c["hist"]["trace"], c["hist"]["mu"][:, 0])
            assert c["model"].get_history()["trace"].shape == (N_BURN + N_KEEP,)
    finally:
        chains.close()


def test_history_keys_and_single_draw_models():
    sub = run(n_cores=1, history_keys=["beta"])
    assert list(sub[0]["hist"]) == ["beta"] and sub[0]["hist"]["beta"].shape == (N_KEEP, 3)
    single = run(lambda s: Model(s, keep_history=False), n_cores=1)
    assert single[0]["hist"]["mu"].shape == (1, 1)
    assert single.history_cleared == [True, True]


# --------------------------------------------------------------------------
# (7) n_cores / n_jobs
# --------------------------------------------------------------------------
def test_n_jobs_is_an_alias_of_n_cores():
    with pytest.raises(ValueError, match="n_jobs is an alias of n_cores"):
        run(n_cores=2, n_jobs=3)
    assert type(run(n_jobs=1)[0]["model"]) is Model
    assert type(run(n_cores=1, n_jobs=1)[0]["model"]) is Model


@needs_fork
def test_n_jobs_alone_runs_parallel_workers():
    chains = run(n_jobs=2)
    try:
        assert all(isinstance(c["model"], ChainModel) for c in chains)
    finally:
        chains.close()


# --------------------------------------------------------------------------
# (8) pickling and copying
# --------------------------------------------------------------------------
@needs_fork
def test_a_pickled_result_keeps_histories_and_detaches_handles():
    chains = run(n_cores=2)
    try:
        back = pickle.loads(pickle.dumps(chains))
        assert isinstance(back, ChainsResult)
        assert (back.n_burn, back.n_keep, back.history_cleared) == (N_BURN, N_KEEP, [True, True])
        for a, b in zip(chains, back):
            assert (a["seed"], a["wall"]) == (b["seed"], b["wall"])
            assert_hist_equal(a["hist"], b["hist"])
            assert isinstance(b["model"], ChainModel) and "detached" in repr(b["model"])
            with pytest.raises(RuntimeError, match="another Python process or session"):
                b["model"].get_history()
        with pytest.raises(RuntimeError, match="another Python process or session"):
            back.predict()
        back.close()                                # nothing to stop
        # Pickling leaves the live result as it was.
        assert_hist_equal(chains[0]["model"].get_history(), chains[0]["hist"])
    finally:
        chains.close()


def test_a_pickled_result_with_model_objects():
    seq = run(n_cores=1)
    back = pickle.loads(pickle.dumps(seq))
    assert type(back[0]["model"]) is Model           # a picklable model is kept
    assert_hist_equal(back[0]["model"].get_history(), seq[0]["hist"])
    held = run(lambda s: UnpicklableModel(s), n_cores=1)
    back = pickle.loads(pickle.dumps(held))
    assert isinstance(back[0]["model"], ChainModel)
    with pytest.raises(RuntimeError, match="cannot be pickled"):
        back[0]["model"].get_history()
    assert_hist_equal(back[0]["hist"], held[0]["hist"])


@needs_fork
def test_copies_share_the_models():
    chains = run(n_cores=2)
    try:
        deep = copy.deepcopy(chains)
        assert isinstance(deep, ChainsResult)
        assert deep[0]["model"] is chains[0]["model"]
        assert deep[0]["hist"]["mu"] is not chains[0]["hist"]["mu"]
        assert deep.predict()["mean"].shape == (2 * N_KEEP, 2)
        shallow = copy.copy(chains)
        assert isinstance(shallow, ChainsResult) and shallow[0] is chains[0]
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
        for arg in (chains[0]["hist"], chains[0], chains[0]["model"]):
            summ, _ = AI4BayesCode.diagnose(arg, plot=False)
            assert len(summ) == 4                   # mu, beta[0], beta[1], beta[2]
        with pytest.raises(TypeError, match="list of 4 chains"):
            AI4BayesCode.diagnose(chains)
    finally:
        chains.close()


# --------------------------------------------------------------------------
# Errors, failed chains, a lost worker, threads, interrupts
# --------------------------------------------------------------------------
@needs_fork
def test_exceptions_in_the_worker_are_raised_here():
    chains = run(n_cores=2)
    m = chains[0]["model"]
    try:
        with pytest.raises(ValueError, match="boom") as ei:
            m.fail("boom")
        assert "in fail" in str(ei.value.__cause__)    # the traceback in the worker
        assert m.add(1, 2) == 3                         # a call after the exception returns its result
        with pytest.raises(RuntimeError, match=r"TwoArgError: a\+b"):
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


@needs_fork
def test_a_failed_chain_has_no_model_and_predict_leaves_it_out():
    chains = run(make_fail_202, n_cores=2)
    try:
        assert "no model for seed 202" in chains[1]["error"]
        assert chains[1]["hist"] is None and chains[1]["model"] is None
        assert chains.history_cleared == [True, False]
        assert chains[0]["model"].add(1, 1) == 2
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
    assert [c["model"] for c in chains] == [None, None]
    assert all(c["error_type"] == "ZeroDivisionError" for c in chains)
    assert chains._pool is None
    with pytest.raises(ValueError, match="every chain failed"):
        chains.predict()


@needs_fork
def test_a_killed_worker_loses_only_its_chains():
    chains = run(n_cores=2)
    try:
        m0, m1 = chains[0]["model"], chains[1]["model"]
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
    with pytest.warns(UserWarning, match=r"parallel \(fork\) execution failed .*exit code 3"):
        chains = run(dies_in_a_worker, n_cores=2)
    assert chain_workers() == before               # the other worker was stopped
    assert [type(c["model"]) for c in chains] == [Model, Model]
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
    m = chains[0]["model"]
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


def test_predict_rejects_chains_that_disagree():
    by_key = run(lambda s: OddModel(s, "key"), n_cores=1)
    with pytest.raises(ValueError, match="different keys"):
        by_key.predict()
    assert [sorted(p) for p in by_key.predict(pool=False)] == [
        ["mean", "y_rep"], ["extra", "mean", "y_rep"]]
    by_width = run(lambda s: OddModel(s, "width"), n_cores=1)
    with pytest.raises(ValueError, match="disagree on the columns of 'mean'"):
        by_width.predict()


@needs_fork
def test_calls_from_threads_do_not_interleave():
    chains = run(seeds=(101, 202, 303), n_cores=2)
    errors = []

    def work(t):
        try:
            for j in range(30):
                assert chains[(t + j) % 3]["model"].add(t, b=j) == t + j
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
    m = chains[0]["model"]
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
    m = chains[0]["model"]
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


# --------------------------------------------------------------------------
# The spawn code path (Windows), forced on any OS
# --------------------------------------------------------------------------
def test_spawned_workers_hold_the_models(windows_path):
    chains = run(n_cores=2)
    try:
        assert all(isinstance(c["model"], ChainModel) for c in chains)
        pids = [c["model"].pid() for c in chains]
        assert len(set(pids)) == 2 and os.getpid() not in pids
        for c in chains:
            assert_hist_equal(c["model"].get_history(), c["hist"])
        assert chains.predict()["mean"].shape == (2 * N_KEEP, 2)
        for a, b in zip(chains, run(n_cores=1)):
            assert_hist_equal(a["hist"], b["hist"])
    finally:
        chains.close()
    assert all(gone(p) for p in pids)


def test_without_a_picklable_factory_the_models_stay_here(windows_path):
    with pytest.warns(UserWarning, match="cannot be sent to worker processes"):
        chains = run(lambda s: Model(s), n_cores=2)
    assert [type(c["model"]) for c in chains] == [Model, Model]


@pytest.mark.parametrize("keep_models", [True, False])
def test_spawn_workers_that_cannot_rebuild_the_factory(windows_path, monkeypatch,
                                                       keep_models):
    monkeypatch.setitem(globals(), "MAIN_ONLY", Model)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        chains = run(main_only_factory, n_cores=2, keep_models=keep_models)
    assert [c.get("error") for c in chains] == [None, None]
    assert any("running the chains sequentially" in str(w.message) for w in rec)
    assert chains.history_cleared == [True, True]
    if keep_models:
        assert [type(c["model"]) for c in chains] == [Model, Model]
    else:
        assert all("model" not in c for c in chains)


# --------------------------------------------------------------------------
# End to end with a compiled model
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gls():
    from AI4BayesCode.source import _find_armadillo
    if _find_armadillo() is None:
        pytest.skip("no Armadillo headers to compile the example with")
    return AI4BayesCode.example("GaussianLocationScale", quiet=True)


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
        assert all(isinstance(c["model"], ChainModel) for c in chains)
        seq = AI4BayesCode.run_chains(factory, seeds=seeds, n_burn=nb, n_keep=nk,
                                      n_cores=1, verbose=False)
        plain = AI4BayesCode.run_chains(factory, seeds=seeds, n_burn=nb, n_keep=nk,
                                        n_cores=2, keep_models=False, verbose=False)
        assert [type(c["model"]).__name__ for c in seq] == ["GaussianLocationScale"] * 2
        # First predict_at() of every model: y_rep comes from a generator seeded
        # by the chain seed, so the pooled values equal the sequential ones.
        pooled = AI4BayesCode.predict(chains)
        assert pooled["y_rep"].shape == (2 * nk, y.size)
        np.testing.assert_array_equal(
            pooled["y_rep"], np.vstack([c["model"].predict_at({})["y_rep"] for c in seq]))
        for c, s, p in zip(chains, seq, plain):
            h = c["model"].get_history()
            assert {k: v.shape for k, v in h.items()} == {"mu": (nk, 1), "sigma": (nk, 1)}
            assert_hist_equal(h, c["hist"])
            assert_hist_equal(c["hist"], s["hist"])
            assert_hist_equal(c["hist"], p["hist"])
            assert c["model"].predict_at({})["y_rep"].shape == (nk, y.size)
        sep = AI4BayesCode.predict(chains, pool=False)
        assert [s["y_rep"].shape for s in sep] == [(nk, y.size)] * 2
        # get_dag() returns a DagInfo, which does not pickle: through a handle
        # the worker returns it as a dict of its fields, with item and attribute access.
        from AI4BayesCode.plot_dag import _dag_to_edges
        dag, ref = chains[0]["model"].get_dag(), seq[0]["model"].get_dag()
        assert sorted(dag) == sorted(ref.keys())
        assert all(dag[k] == ref[k] for k in ref.keys())
        assert dag.predict_edges == ref.predict_edges
        assert _dag_to_edges(dag) == _dag_to_edges(ref)
        m0 = chains[0]["model"]
        m0.freeze(["mu"])
        assert m0.get_frozen() == seq[0]["model"].get_frozen() + ["mu_sigma_joint.mu"]
        m0.unfreeze()
        assert m0.get_frozen() == []
        assert m0.predict_at({}, last_draw_only=True)["y_rep"].shape == (1, y.size)
        with pytest.raises(RuntimeError, match="no covariate inputs"):
            m0.predict_at({"x": np.ones(3)})       # a C++ exception in the worker
        chains[0]["model"].step(5)
        assert chains[0]["model"].get_history()["mu"].shape == (nk + 5, 1)
        assert chains[1]["model"].get_history()["mu"].shape == (nk, 1)
        assert set(AI4BayesCode.rhat_summary(chains)) == {"mu", "sigma"}
        AI4BayesCode.diagnose(chains[0]["hist"], plot=False)
        pids = [c["model"]._pid for c in chains]      # the worker process ids
    finally:
        chains.close()
    assert all(gone(p) for p in pids)
    with pytest.raises(RuntimeError, match="were closed"):
        chains[0]["model"].get_history()

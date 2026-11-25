"""Invariant check of clear_history() on every shipped example, in Python.

Usage (AI4BayesCode importable, e.g. PYTHONPATH=<repo>/python):
    python tools/ci/check_clear_history.py [--examples A,B,...] [--jobs 6]
        [--timeout 900] [--log-dir DIR] [--nb 40] [--nk 30] [--seed 11]

One Python process per example, at most --jobs at a time; a crash or a run
past --timeout seconds is a FAIL, and after a crash the example runs again
without the check that crashed. Constructor and data come from the example's
``// @example:python`` block, executed with AI4BayesCode.run_chains replaced by
a stub that captures the factory and ends the block (a block without that
call: its first Mod.<Class>(...), with rng_seed set to --seed); keep_history
is set to True. With every model built and stepped in a fresh thread, after
Mod.set_kernel_seed(seed) when the module has it (the BART-family kernels draw
from a thread-local engine that some constructors do not seed),
    A = ctor(seed); A.step(nb); A.clear_history(); A.step(nk)
    B = ctor(seed); B.step(nb); B.step(nk)   (and a control run B2 like B)
(a) every get_history() key of A has nk rows (1 when B's has 1) and equals the
    last rows of B's key bit for bit, shape and dtype included; a B2 that
    differs from B (sampler not reproducible) is reported;
(b) predict_at({}): every key of A has nk rows, and a key whose two outputs on
    B are identical equals the last nk rows of B's bit for bit;
(c) as (b) at new data: get_dag().data_inputs set to the constructor
    arguments of the same name (table NEWDATA otherwise), the first
    min(10, N) rows when a new N is accepted, matrices flattened column-major;
    keep_tree=True where the constructor has it, with (a) and (b) repeated on
    those models;
(d) after ctor(seed); step(nb); clear_history(), get_history() and
    predict_at({}) return without an error (row counts recorded).
A failing (b)/(c) key whose B output lacks nb + nk rows is marked as not one
row per stored draw. Output: summary.txt, summary.csv and per-example logs in
--log-dir; exit status 1 when an example fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# predict_at keys that differ from the declared data inputs or from the
# constructor argument names. Each entry maps the constructor arguments
# (v(name) returns the value passed for that C++ parameter) to new data.
NEWDATA = {
    # predict_at takes new times "t"; the declared inputs t_new_flag and y are
    # not prediction covariates.
    "GPTimeSeries": lambda v: {"t": v("t")},
    # Declared inputs y0 = (S0, I0, R0) and t_obs.
    "ODE_SIR": lambda v: {"y0": [v("S0"), v("I0"), v("R0")], "t_obs": v("t_obs")},
    # predict_at takes a 0-based g_idx; the constructor takes g_idx_1based.
    "HierarchicalLM_joint": lambda v: {"X": v("X"),
                                       "g_idx": np.asarray(v("g_idx_1based")) - 1},
    "HierarchicalLM_MultivariateRE": lambda v: {"X": v("X"), "group_idx": v("group")},
    # No declared data inputs; predict_at evaluates S(t) at query times t
    # (and frailty group g, 0-based).
    "PehSurvival": lambda v: {"t": v("t")},
    "PehSharedFrailty": lambda v: {"t": v("t"), "g": 0.0},
}
# Examples whose (c) is skipped although data inputs are declared.
NEWDATA_SKIP = {
    "HSGPRegression": "data input 'phi' is the basis matrix the constructor "
                      "computes from x; no constructor argument holds it",
}

ALL_CHECKS = ("build", "a", "b", "c", "d", "a_kt", "b_kt", "d_kt")


# ---- helpers ----------------------------------------------------------------

def nrows(x) -> int:
    a = np.asarray(x)
    return 1 if a.ndim == 0 else int(a.shape[0])


def shape_of(x) -> str:
    a = np.asarray(x)
    return "x".join(str(d) for d in a.shape) if a.ndim else "1"


def last_rows(x, n):
    """The last n entries along the first axis, or None when x has fewer."""
    a = np.asarray(x)
    if a.ndim == 0:
        return a if n == 1 else None
    if n > a.shape[0]:
        return None
    return a[a.shape[0] - n:]


def same_bits(a, b) -> bool:
    """Bit-for-bit equality, shape and dtype included."""
    if a is None or b is None:
        return False
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def same_map_bits(m1, m2) -> bool:
    return set(m1) == set(m2) and all(same_bits(m1[k], m2[k]) for k in m1)


def err_msg(e: BaseException) -> str:
    return re.sub(r"\s+", " ", f"{type(e).__name__}: {e}").strip()


def isolated(fn):
    """fn() in a new thread (a fresh thread-local kernel RNG engine)."""
    box = {}

    def run():
        try:
            box["v"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if "e" in box:
        raise box["e"]
    return box["v"]


def parse_default(d):
    if d is None:
        return None
    d = d.strip().strip("{}").strip()
    d = re.sub(r"[uUlLfF]+$", "", d)
    if d == "true":
        return True
    if d == "false":
        return False
    try:
        return float(d) if re.search(r"[.eE]", d) else int(d)
    except ValueError:
        return None


def ctor_sigs(src: str, cls_name: str):
    """Every constructor declaration of the class: [(name, default), ...]."""
    from AI4BayesCode.doc import _args_after
    s = re.sub(r"//[^\n]*", "", src)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    out = []
    for m in re.finditer(r"(?m)^[ \t]*" + re.escape(cls_name) + r"[ \t]*\(", s):
        a = _args_after(s, m.end() - 1)
        if a:
            out.append([(x["name"], x["default"]) for x in a])
    return out


def pick_sig(sigs, args, kwargs):
    """The declaration a call matches: positional arguments fill parameters in
    order, keywords name them, every parameter without a default is given."""
    for sig in sigs:
        names = [n for n, _ in sig]
        if any(k not in names for k in kwargs) or len(args) > len(names):
            continue
        covered = set(names[:len(args)]) | set(kwargs)
        if any(d is None and n not in covered for n, d in sig):
            continue
        return sig
    return None


def arg_value(args, kwargs, sig, p):
    """(set, value) of parameter p; set False means the C++ default applies."""
    names = [n for n, _ in sig]
    if p in kwargs:
        return True, kwargs[p]
    if p in names:
        k = names.index(p)
        if k < len(args):
            return True, args[k]
        return False, parse_default(sig[k][1])
    return False, None


def set_arg(args, kwargs, sig, p, value):
    args, kwargs = list(args), dict(kwargs)
    names = [n for n, _ in sig]
    if p in kwargs:
        kwargs[p] = value
    elif p in names and names.index(p) < len(args):
        args[names.index(p)] = value
    else:
        kwargs[p] = value
    return tuple(args), kwargs


def compare_maps(mA, mB, mB2, nb, nk, single):
    """Compare a map of A (history or prediction) with the same map of B.
    mB2: a second prediction of B, to tell deterministic keys; None for
    history. single: accept 1 row in A when B also has 1 row."""
    keys = sorted(set(mA) | set(mB))
    out, bad, not_indexed = {}, [], []
    for k in keys:
        if k not in mA or k not in mB:
            line = f"{k}: returned by {'B' if k not in mA else 'A'} only"
            out[k] = {"ok": False, "line": line}
            bad.append(line)
            continue
        a, b = np.asarray(mA[k]), np.asarray(mB[k])
        ra, rb = nrows(a), nrows(b)
        rows_ok = ra == nk or (single and ra == 1 and rb == 1)
        det = True if mB2 is None else (k in mB2 and same_bits(b, np.asarray(mB2[k])))
        tail_b = last_rows(b, ra)
        val = None if not det else (False if tail_b is None else same_bits(a, tail_b))
        ok = rows_ok and val is not False
        line = (f"{k}: A {shape_of(a)}, B {shape_of(b)}"
                + ("" if det else ", differs between two B calls (rows checked only)")
                + (", values = last rows of B" if val is True
                   else ", values != last rows of B" if val is False else ""))
        if not ok and rb != nb + nk and not (single and rb == 1):
            not_indexed.append(k)
            line += (f"; B rows != nb + nk = {nb + nk}, so the output is not one "
                     "row per stored draw")
        out[k] = {"rows_A": ra, "rows_B": rb, "deterministic": det, "equal": val,
                  "ok": ok, "line": line}
        if not ok:
            bad.append(line)
    r = {"status": "SKIP" if not keys else "FAIL" if bad else "PASS",
         "keys": out, "failing": bad, "not_indexed": not_indexed}
    if not keys:
        r["reason"] = "no keys returned"
    return r


def safe(f):
    try:
        return True, f()
    except Exception as e:  # noqa: BLE001
        return False, err_msg(e)


def check_history(A, B, B2, nb, nk):
    okA, hA = safe(A.get_history)
    okB, hB = safe(B.get_history)
    okB2, hB2 = safe(B2.get_history)
    if not (okA and okB):
        return {"status": "FAIL", "failing":
                ([f"A.get_history() error: {hA}"] if not okA else [])
                + ([f"B.get_history() error: {hB}"] if not okB else [])}
    r = compare_maps(hA, hB, None, nb, nk, single=True)
    r["reproducible"] = bool(okB2 and same_map_bits(hB, hB2))
    if not r["reproducible"]:
        r["reason"] = ("control: B2, built and stepped like B, gives a different "
                       "history, so the sampler is not reproducible for a fixed seed "
                       "and a value difference does not isolate clear_history()")
        if r["status"] == "PASS":
            r["status"] = "FAIL"
    return r


def check_predict(A, B, nd, nb, nk, pB1=None):
    if pB1 is None:
        pB1 = safe(lambda: B.predict_at(nd))
    pB2 = safe(lambda: B.predict_at(nd))
    pA = safe(lambda: A.predict_at(nd))
    if not (pA[0] and pB1[0] and pB2[0]):
        return {"status": "FAIL", "failing":
                ([f"A.predict_at() error: {pA[1]}"] if not pA[0] else [])
                + ([f"B.predict_at() error: {pB1[1]}"] if not pB1[0] else [])
                + ([f"second B.predict_at() error: {pB2[1]}"]
                   if pB1[0] and not pB2[0] else [])}
    return compare_maps(pA[1], pB1[1], pB2[1], nb, nk, single=False)


def check_zero_draws(build0):
    A2 = build0()
    okh, h = safe(A2.get_history)
    okp, p = safe(lambda: A2.predict_at({}))
    bad = ([f"get_history() error: {h}"] if not okh else []) \
        + ([f"predict_at({{}}) error: {p}"] if not okp else [])
    return {"status": "FAIL" if bad else "PASS", "failing": bad,
            "history_rows": {k: nrows(v) for k, v in sorted(h.items())} if okh else {},
            "predict_rows": {k: nrows(v) for k, v in sorted(p.items())} if okp else {}}


def format_result(res) -> list[str]:
    L = [f"== {res.get('example')} [python]: {res.get('status', '?')}"]
    if res.get("ctor_args"):
        L.append(f"   constructor: {res['ctor_args']}")
    L += [f"   note: {n}" for n in res.get("notes", [])]
    if res.get("fatal"):
        L.append(f"   {res['fatal']}")
    labels = {"a_kt": "a, keep_tree = True", "b_kt": "b, keep_tree = True",
              "d_kt": "d, keep_tree = True"}
    for s in ALL_CHECKS:
        r = res.get(s)
        if r is None or (s == "build" and r.get("status") == "PASS"):
            continue
        L.append(f"   ({labels.get(s, s)}) {r['status']}"
                 + (f": {r['reason']}" if r.get("reason") else ""))
        if r.get("inputs"):
            L.append(f"       inputs: {r['inputs']}")
        L += [f"       {x}" for x in r.get("failing", [])]
        if s in ("d", "d_kt") and "history_rows" in r:
            hr = ", ".join(f"{k}={v}" for k, v in r["history_rows"].items())
            pr = ", ".join(f"{k}={v}" for k, v in r["predict_rows"].items())
            L.append(f"       rows after clear_history(): history {{{hr}}}; "
                     f"predict_at {{{pr}}}")
    return L


def overall_status(res) -> str:
    if res.get("fatal"):
        return "FAIL"
    return "FAIL" if any((res.get(k) or {}).get("status") == "FAIL"
                         for k in ALL_CHECKS) else "PASS"


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---- one example (child process) --------------------------------------------

class _Captured(BaseException):
    """Ends the example block (BaseException: an `except Exception` in the
    block does not catch it)."""


def capture_ctor(code: str, name: str):
    import AI4BayesCode
    real_rc, real_ex = AI4BayesCode.run_chains, AI4BayesCode.example
    rc_mode = "run_chains(" in code
    calls, mods = [], []

    class Recorder:
        def __init__(self, cls):
            self.cls = cls

        def __call__(self, *args, **kwargs):
            calls.append((self.cls, args, kwargs))
            if not rc_mode:
                raise _Captured("new")
            return self.cls(*args, **kwargs)

    class ModProxy:
        def __init__(self, mod):
            self.__dict__["_mod"] = mod

        def __getattr__(self, k):
            v = getattr(self._mod, k)
            return Recorder(v) if isinstance(v, type) and hasattr(v, "step") else v

    def stub_run_chains(factory=None, *a, **kw):
        raise _Captured("run_chains", factory)

    def stub_example(*a, **kw):
        m = real_ex(*a, **kw)
        mods.append(m)
        return ModProxy(m)

    AI4BayesCode.run_chains, AI4BayesCode.example = stub_run_chains, stub_example
    ns = {"__name__": "__ai4b_example_block__"}
    try:
        exec(compile(code, f"<{name} @example:python>", "exec"), ns)
        got = {"kind": "none"}
    except _Captured as c:
        got = {"kind": c.args[0], "factory": c.args[1] if len(c.args) > 1 else None}
    except Exception as e:  # noqa: BLE001
        got = {"kind": "error", "msg": err_msg(e)}
    finally:
        AI4BayesCode.run_chains, AI4BayesCode.example = real_rc, real_ex
    got["calls"], got["mods"] = calls, mods
    return got


class _Missing(Exception):
    pass


def new_data_for(name, A, args, kwargs, sig):
    """(nd, how) for check (c), or (None, reason)."""
    if name in NEWDATA_SKIP:
        return None, NEWDATA_SKIP[name]

    def v(p):
        is_set, val = arg_value(args, kwargs, sig, p)
        if not is_set:
            raise _Missing(p)
        return val
    try:
        if name in NEWDATA:
            nd, how = NEWDATA[name](v), "predict_at keys from the NEWDATA table"
        else:
            inputs = list(A.get_dag().data_inputs)
            if not inputs:
                return None, "no declared data inputs (get_dag().data_inputs is empty)"
            nd, how = {k: v(k) for k in inputs}, "get_dag().data_inputs"
    except _Missing as e:
        return None, f"no constructor argument for new-data key {e.args[0]}"
    return {k: np.asarray(x, dtype=float) for k, x in nd.items()}, how


def child_main(a) -> int:
    import faulthandler
    faulthandler.enable()
    name, out, nb, nk, seed = a.child, Path(a.out), a.nb, a.nk, a.seed
    skip = set(filter(None, a.skip.split(",")))
    res = {"example": name, "lang": "python", "nb": nb, "nk": nk, "seed": seed,
           "running": None, "done": [], "notes": []}

    def put():
        out.write_text(json.dumps(res, default=_jsonable), encoding="utf-8")

    def stage(s, f):
        """Runs one check: records it as running (so a crash is attributed to
        it), stores the result, turns an exception into a FAIL, and replaces a
        check that crashed the process in an earlier run by a FAIL."""
        if s in skip:
            res[s] = {"status": "FAIL",
                      "reason": "crashed the process in an earlier run (see the log)"}
        else:
            res["running"] = s
            put()
            print(f"-- stage {s}", flush=True)
            try:
                res[s] = f()
            except Exception as e:  # noqa: BLE001
                res[s] = {"status": "FAIL", "failing": [f"error: {err_msg(e)}"]}
        res["done"].append(s)
        res["running"] = None
        put()

    def finish():
        res["status"] = overall_status(res)
        put()
        print("\n".join(format_result(res)), flush=True)
        return 0

    put()
    import AI4BayesCode
    from AI4BayesCode.doc import _parse_example
    print(f"AI4BayesCode {AI4BayesCode.__version__} from "
          f"{Path(AI4BayesCode.__file__).parent}", flush=True)
    src = Path(AI4BayesCode.examples_path(name)).read_text(encoding="utf-8")
    lines = _parse_example(src, "python")

    # ---- capture: the constructor of the example block
    res["running"] = "capture"
    put()
    print("-- stage capture", flush=True)
    if not lines:
        res["fatal"] = "no // @example:python block"
        return finish()
    got = capture_ctor(textwrap.dedent("\n".join(lines)), name)
    if got["kind"] in ("none", "error"):
        res["fatal"] = (f"example block failed: {got['msg']}" if got["kind"] == "error"
                        else "example block has neither run_chains() nor Mod.<Class>()")
        return finish()
    factory, probe = got.get("factory"), None
    if got["kind"] == "run_chains":
        try:
            probe = factory(seed)
        except Exception as e:  # noqa: BLE001
            res["fatal"] = f"constructor failed: {err_msg(e)}"
            return finish()
        res["ctor_source"] = "factory of run_chains() in the example block"
    else:
        res["ctor_source"] = ("first Mod.<Class>(...) of the example block "
                              "(it has no run_chains call)")
    if not got["calls"]:
        res["fatal"] = "the factory constructs no class of the example module"
        return finish()
    cls, args, kwargs = got["calls"][-1]
    cls_name = cls.__name__
    sig = pick_sig(ctor_sigs(src, cls_name), args, kwargs)
    modified = got["kind"] == "new"
    if got["kind"] == "new":
        if sig is None:
            res["fatal"] = "constructor signature not recovered"
            return finish()
        old = arg_value(args, kwargs, sig, "rng_seed")[1]
        args, kwargs = set_arg(args, kwargs, sig, "rng_seed", seed)
        res["notes"].append(f"rng_seed {old} of the block set to {seed}")
    if sig is not None:
        if arg_value(args, kwargs, sig, "keep_history")[1] is not True:
            args, kwargs = set_arg(args, kwargs, sig, "keep_history", True)
            modified = True
            res["notes"].append("keep_history set to True (the block constructs with False)")
    else:
        res["notes"].append("constructor arguments not recovered: (c) and keep_tree unavailable")
    mod = got["mods"][-1] if got["mods"] else None
    seed_hook = getattr(mod, "set_kernel_seed", None)
    if seed_hook is not None:
        res["notes"].append(f"set_kernel_seed({seed}) before every construction")

    def make():
        return factory(seed) if not modified else cls(*args, **kwargs)

    def desc(x):
        if isinstance(x, (bool, int, float, str, np.number)):
            return repr(x)
        return f"<{type(x).__name__} {shape_of(x)}>" if hasattr(x, "shape") \
            else f"<{type(x).__name__}>"
    if sig is not None:
        res["ctor_args"] = f"{cls_name}(" + ", ".join(
            [desc(x) for x in args] + [f"{k}={desc(x)}" for k, x in kwargs.items()]) + ")"
    print(f"constructor: {res.get('ctor_args', '?')}", flush=True)
    res["class"] = cls_name
    res["done"].append("capture")
    res["running"] = None
    put()

    # Each model is built and stepped in a fresh thread, after the module's
    # kernel seed hook when it has one, so the kernels draw the same numbers.
    def built(mk, steps):
        def f():
            if seed_hook is not None:
                seed_hook(seed)
            m = mk()
            steps(m)
            return m
        return isolated(f)

    def steps_A(m):
        m.step(nb)
        m.clear_history()
        m.step(nk)

    def steps_B(m):
        m.step(nb)
        m.step(nk)

    def steps_0(m):
        m.step(nb)
        m.clear_history()

    models = {}
    state = {"no_clear": False, "make_kt": None}

    # ---- build: A (clear_history() after the warmup) and B (no call)
    def do_build():
        m0 = probe if probe is not None else built(make, lambda m: None)
        if not hasattr(m0, "clear_history"):
            state["no_clear"] = True
            res["fatal"] = "the class has no clear_history() method"
            return {"status": "FAIL", "reason": res["fatal"]}
        models["A"] = built(make, steps_A)
        models["B"] = built(make, steps_B)
        models["B2"] = built(make, steps_B)
        return {"status": "PASS"}
    stage("build", do_build)
    have = "A" in models and "B" in models
    not_run = {"status": "FAIL", "reason": "not run: A and B were not built"}

    # ---- (a) history
    stage("a", lambda: check_history(models["A"], models["B"], models["B2"], nb, nk)
          if have else dict(not_run))
    if (res.get("a") or {}).get("reproducible") is False:
        res["notes"].append("B2 differs from B: value comparisons in (a), (b) and (c) "
                            "do not isolate clear_history() for this sampler")

    # ---- (b) prediction at the training data
    stage("b", lambda: check_predict(models["A"], models["B"], {}, nb, nk)
          if have else dict(not_run))

    # ---- (c) prediction at new data
    def do_c():
        if not have:
            return dict(not_run)
        if sig is None:
            return {"status": "SKIP", "reason": "constructor arguments not recovered"}
        A, B = models["A"], models["B"]
        nd, how = new_data_for(name, A, args, kwargs, sig)
        if nd is None:
            return {"status": "SKIP", "reason": how}
        N = max(nrows(x) for x in nd.values())
        n_sub = min(10, N)
        sub = {k: (x[:n_sub] if x.ndim and x.shape[0] == N else x) for k, x in nd.items()}
        flag = "keep_tree" if "keep_tree" in [n for n, _ in sig] else None
        if flag:
            a_kt, k_kt = set_arg(args, kwargs, sig, "keep_tree", True)
            make_kt = lambda: cls(*a_kt, **k_kt)  # noqa: E731
            state["make_kt"] = make_kt
            A = built(make_kt, steps_A)
            B = built(make_kt, steps_B)
            res["a_kt"] = check_history(A, B, built(make_kt, steps_B), nb, nk)
            res["b_kt"] = check_predict(A, B, {}, nb, nk)
            put()

        def flat(d):
            return {k: (np.atleast_1d(x) if x.ndim < 2 else x.ravel(order="F"))
                    for k, x in d.items()}
        variants = ([("subset", sub)] if n_sub < N else []) + [("all", nd)]
        pB1, used, errs = None, None, []
        for tn, d in variants:
            ok, r = safe(lambda: B.predict_at(flat(d)))
            if ok:
                pB1, used = (ok, r), tn
                break
            errs.append(f"{tn} rows: {r}")
        shown = sub if used == "subset" else nd
        inputs = (", ".join(f"{k} {shape_of(x)}" for k, x in shown.items())
                  + f" from {how}"
                  + ("" if used is None else
                     f" (first {n_sub} of N = {N} rows)" if used == "subset"
                     else f" (all N = {N} rows)")
                  + ("; keep_tree = True" if flag else ""))
        if pB1 is None:
            return {"status": "SKIP", "inputs": inputs, "reason":
                    "predict_at(new data) fails on B, which never called "
                    "clear_history(): " + " | ".join(errs)}
        r = check_predict(A, B, flat(dict(variants)[used]), nb, nk, pB1=pB1)
        r["inputs"] = inputs
        return r
    stage("c", do_c)

    # ---- (d) zero stored draws
    def do_d():
        if state["no_clear"]:
            return {"status": "FAIL",
                    "reason": "not run: the class has no clear_history() method"}
        r = check_zero_draws(lambda: built(make, steps_0))
        res["d"] = r
        put()
        if state["make_kt"] is not None:
            try:
                res["d_kt"] = check_zero_draws(lambda: built(state["make_kt"], steps_0))
            except Exception as e:  # noqa: BLE001
                res["d_kt"] = {"status": "FAIL", "failing": [f"error: {err_msg(e)}"]}
        return r
    stage("d", do_d)
    return finish()


# ---- driver -----------------------------------------------------------------

def crash_excerpt(text: str) -> list[str]:
    """The lines of a crashed run's log from the faulthandler report on (the
    crashing thread first), or its last 8."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    at = next((i for i, ln in enumerate(lines) if ln.startswith("Fatal Python error")), None)
    return lines[at:at + 8] if at is not None else lines[-8:]


def run_example(name, log_dir: Path, a) -> dict:
    out, log = log_dir / f"{name}.json", log_dir / f"{name}.log"
    for f in (out, log):
        f.unlink(missing_ok=True)
    skip, crashes, t0, res = [], [], time.time(), None
    while True:
        left = a.timeout - (time.time() - t0)
        if left < 1:
            crashes.append("time budget used up by earlier runs")
            break
        out.unlink(missing_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", name,
               "--out", str(out), "--nb", str(a.nb), "--nk", str(a.nk),
               "--seed", str(a.seed)] + (["--skip", ",".join(skip)] if skip else [])
        # A new session, so a timeout also ends the compiler processes.
        posix = os.name == "posix"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=posix)
        try:
            text, _ = p.communicate(timeout=left)
            rc, timed_out = p.returncode, False
        except subprocess.TimeoutExpired:
            if posix:
                os.killpg(p.pid, signal.SIGKILL)
            else:
                p.kill()
            text, _ = p.communicate()
            rc, timed_out = p.returncode, True
        text = text.decode("utf-8", errors="replace")
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"######## run {len(skip) + 1} (skip: {','.join(skip) or 'none'}), "
                     f"exit status {rc}{' (timeout)' if timed_out else ''}\n{text}\n")
        res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        if rc == 0 and not timed_out and res is not None:
            break
        stage = (res or {}).get("running") or "start"
        what = (f"timed out ({a.timeout:g} s budget)" if timed_out
                else f"process exited with status {rc}")
        crashes.append(f"{what} during stage '{stage}'; log: "
                       + " | ".join(crash_excerpt(text)))
        if timed_out or stage not in ("build", "a", "b", "c", "d") or stage in skip:
            break
        skip.append(stage)
    res = res or {"example": name, "lang": "python"}
    if crashes:
        res["crashes"] = crashes
        res["fatal"] = "; ".join(([res["fatal"]] if res.get("fatal") else []) + crashes)
    res["status"] = overall_status(res)
    res["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, default=_jsonable), encoding="utf-8")
    return res


def driver_main(a) -> int:
    import AI4BayesCode
    log_dir = Path(a.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    all_ex = AI4BayesCode.list_examples()
    names = [s for s in a.examples.split(",") if s] if a.examples else all_ex
    bad = [n for n in names if n not in all_ex]
    if bad:
        raise SystemExit(f"unknown example(s): {', '.join(bad)}")
    print(f"Python {platform.python_version()} | AI4BayesCode {AI4BayesCode.__version__} "
          f"from {Path(AI4BayesCode.__file__).parent}\n{len(names)} examples | "
          f"nb = {a.nb}, nk = {a.nk}, seed = {a.seed} | jobs = {a.jobs}, "
          f"timeout = {a.timeout:g} s\nlogs: {log_dir}\n", flush=True)
    lock = threading.Lock()

    def one(n):
        try:
            r = run_example(n, log_dir, a)
        except Exception as e:  # noqa: BLE001
            r = {"example": n, "lang": "python", "status": "FAIL",
                 "fatal": f"driver error: {err_msg(e)}"}
        with lock:
            print(f"{n:<32} {r['status']:<5} {r.get('seconds', float('nan')):7.1f}s",
                  flush=True)
        return r
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        results = list(ex.map(one, names))

    st = lambda r, k: (r.get(k) or {}).get("status", "")  # noqa: E731
    with open(log_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["example", "status", "a", "b", "c", "d", "a_keep_tree",
                    "b_keep_tree", "d_keep_tree", "c_inputs", "c_reason", "seconds",
                    "fatal"])
        for r in results:
            w.writerow([r["example"], r["status"], st(r, "a"), st(r, "b"), st(r, "c"),
                        st(r, "d"), st(r, "a_kt"), st(r, "b_kt"), st(r, "d_kt"),
                        (r.get("c") or {}).get("inputs", ""),
                        (r.get("c") or {}).get("reason", ""), r.get("seconds", ""),
                        r.get("fatal", "")])
    txt = [f"clear_history() invariant check, Python {platform.python_version()}, "
           f"AI4BayesCode {AI4BayesCode.__version__}, nb = {a.nb}, nk = {a.nk}, "
           f"seed = {a.seed}", ""]
    for r in results:
        txt += format_result(r) + [""]
    (log_dir / "summary.txt").write_text("\n".join(txt), encoding="utf-8")
    n_bad = sum(r["status"] != "PASS" for r in results)
    print(f"\n{len(results) - n_bad} of {len(results)} examples passed; details in "
          f"{log_dir / 'summary.txt'}")
    return 1 if n_bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--examples", default="", help="comma-separated subset")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--log-dir", default="clear-history-python")
    ap.add_argument("--nb", type=int, default=40)
    ap.add_argument("--nk", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--skip", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        return child_main(a)
    return driver_main(a)


if __name__ == "__main__":
    sys.exit(main())

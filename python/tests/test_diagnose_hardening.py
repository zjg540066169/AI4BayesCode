"""diagnose() / rhat_summary() robustness to run_chains()-shaped input.

Regression for the natural mistakes: diagnose(runs[0]) (a run RECORD, not its
hist), diagnose on a FAILED chain (hist=None), diagnose on a chain LIST, and the
n_burn / drop_burn param-name aliases. All synthetic -- no compilation.
"""
import importlib

import numpy as np
import pytest

diag = importlib.import_module("AI4BayesCode.diagnose")
rs = importlib.import_module("AI4BayesCode.rhat_summary")


def _hist(seed=0, n=400):
    rng = np.random.default_rng(seed)
    return {"mu": rng.normal(size=n), "beta": rng.normal(size=(n, 3))}


def _record(seed=0, ok=True):
    return {"seed": seed, "wall": 0.1,
            "hist": _hist(seed) if ok else None,
            **({} if ok else {"error": "__init__(): incompatible constructor arguments (rng_seed=)"})}


def test_diagnose_accepts_run_record():
    # diagnose(runs[0]) auto-unwraps the record's ["hist"].
    summ, _ = diag.diagnose(_record(1, ok=True), n_burn=100, plot=False)
    assert any(k.startswith("mu") for k in summ)


def test_diagnose_failed_chain_clear_error():
    with pytest.raises(ValueError, match="FAILED to run"):
        diag.diagnose(_record(2, ok=False))


def test_diagnose_rejects_chain_list():
    runs = [_record(1), _record(2)]
    with pytest.raises(TypeError, match="list of 2 chains"):
        diag.diagnose(runs)


def test_diagnose_drop_burn_alias():
    a, _ = diag.diagnose(_hist(3), n_burn=100, plot=False)
    b, _ = diag.diagnose(_hist(3), drop_burn=100, plot=False)
    assert a["mu"]["mean"] == b["mu"]["mean"]


def test_rhat_summary_n_burn_alias():
    runs = [_record(1), _record(2)]
    a = rs.rhat_summary(runs, drop_burn=100)["mu"]["rhat"]
    b = rs.rhat_summary(runs, n_burn=100)["mu"]["rhat"]
    assert a == b


def test_rhat_summary_skips_failed_chain():
    # A failed chain (hist=None) is dropped, surviving chains still summarised.
    runs = [_record(1, ok=True), _record(2, ok=False)]
    out = rs.rhat_summary(runs)
    assert "mu" in out

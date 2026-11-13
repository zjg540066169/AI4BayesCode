"""run_chains() on the Windows code path (spawned workers), forced on any OS.

When every worker reports an error, the chains are run again in this process
only if the workers could not rebuild the factory. A model whose own chains
fail keeps its errors (and is not run twice), and a model without
get_history() is not mistaken for a failure."""
from __future__ import annotations

import importlib
import sys
import warnings

import pytest

rc = importlib.import_module("AI4BayesCode.run_chains")


class _NoHistory:
    def step(self, n):
        pass


class _Broken:
    def step(self, n):
        raise ValueError("this model always fails")


def no_history(seed):
    return _NoHistory()


def broken(seed):
    return _Broken()


@pytest.fixture()
def windows_path(monkeypatch):
    class _Sys:
        platform = "win32"

        def __getattr__(self, name):
            return getattr(sys, name)
    monkeypatch.setattr(rc, "sys", _Sys())


def _fallback_warnings(record):
    return [w for w in record if "running the chains sequentially" in str(w.message)]


def test_a_model_without_history_is_not_a_failure(windows_path):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        res = rc.run_chains(no_history, seeds=[1, 2], n_burn=1, n_keep=1, n_jobs=2,
                            verbose=False)
    assert [r.get("error") for r in res] == [None, None]
    assert not _fallback_warnings(record)


def test_a_failing_model_keeps_its_errors(windows_path):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        res = rc.run_chains(broken, seeds=[1, 2], n_burn=1, n_keep=1, n_jobs=2,
                            verbose=False)
    assert all("always fails" in r["error"] for r in res)
    assert not _fallback_warnings(record)

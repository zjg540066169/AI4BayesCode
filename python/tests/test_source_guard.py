"""source() returns the module already loaded in this process when it is the
same build, however cache_dir is spelled, and refuses a different build.

pybind11 registers a module's classes once per interpreter, so a second build
of a loaded module cannot be imported. The check must compare builds by their
signature: module.__file__ is always absolute, while cache_dir may be relative
(a notebook cell re-run with cache_dir="build" hit this). The toolchain is
faked, so nothing is compiled."""
from __future__ import annotations

import importlib
import sys
import types

import pytest

# The package re-exports source() under the submodule's name.
src = importlib.import_module("AI4BayesCode.source")

CODE = "#include <pybind11/pybind11.h>\nPYBIND11_MODULE(guardmod, m) {}\n"


@pytest.fixture()
def loaded(monkeypatch, tmp_path):
    monkeypatch.setattr(src, "_compiler", lambda: "fake-c++")
    monkeypatch.setattr(src, "_compile_flags", lambda inc, cxx: ["-Ifake"])
    monkeypatch.setattr(src, "_link_flags", lambda cxx=None: [])
    monkeypatch.setattr(src, "_find_armadillo", lambda cxx=None: None)
    monkeypatch.setattr(src, "_compiler_id", lambda cxx: "fake")
    monkeypatch.chdir(tmp_path)
    sig = src._signature(CODE, ["-Ifake"], [], "fake|")
    mod = types.ModuleType("guardmod")
    mod.__file__ = str(tmp_path.resolve() / "build" / f"guardmod_{sig}{src._ext_suffix()}")
    monkeypatch.setitem(sys.modules, "guardmod", mod)
    return mod


def test_same_build_with_a_relative_cache_dir(loaded):
    assert src.source(CODE, cache_dir="build", quiet=True) is loaded
    assert src.source(CODE, cache_dir="./build/", quiet=True) is loaded


def test_same_build_from_another_cache_dir(loaded, tmp_path):
    assert src.source(CODE, cache_dir=tmp_path / "elsewhere", quiet=True) is loaded


def test_a_different_build_is_refused(loaded):
    with pytest.raises(RuntimeError, match="different build"):
        src.source(CODE + "// changed\n", cache_dir="build", quiet=True)

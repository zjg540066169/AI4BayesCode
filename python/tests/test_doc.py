"""Tests for AI4BayesCode.doc @example parsing (no compilation needed)."""
import importlib

docmod = importlib.import_module("AI4BayesCode.doc")

_WITH = """\
// ============================================================
//  Fix.cpp -- toy model
//  y ~ Normal(mu, sigma).
// @example:R
//   library(AI4BayesCode)
//   y <- rnorm(50)
//   m <- new(Fix, y, 1L)
// @example:python
//   import numpy as np, AI4BayesCode
//   m = mod.Fix(np.random.randn(50), 1)
//   m.step(100)
// @example:end
// ============================================================
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
class Fix { public: Fix(const arma::vec& y, int rng_seed){} void step(int n){} };
PYBIND11_MODULE(Fix_module, m){ pybind11::class_<Fix>(m, "Fix"); }
"""

_WITHOUT = """\
//  G.cpp -- m
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
class G { public: G(const arma::vec& y){} };
PYBIND11_MODULE(G_module, m){ pybind11::class_<G>(m, "G"); }
"""


def test_parse_example_python_and_r_distinct():
    exP = docmod._parse_example(_WITH, "python")
    exR = docmod._parse_example(_WITH, "R")
    assert exP and any("np.random" in l for l in exP)
    assert exR and any("rnorm" in l for l in exR)
    assert not any("rnorm" in l for l in exP)
    assert not any("np.random" in l for l in exR)


def test_header_excludes_example_block():
    d = docmod._parse_header(_WITH)
    assert not any("@example" in l for l in d)
    assert not any("np.random" in l for l in d)


def test_parse_example_absent_returns_none():
    assert docmod._parse_example(_WITHOUT, "python") is None


def test_doc_surfaces_python_block(tmp_path, capsys):
    p = tmp_path / "Fix.cpp"
    p.write_text(_WITH, encoding="utf-8")
    r = docmod.doc(str(p))
    assert r is None                       # doc() is a display fn: prints the card, returns None (no IPython echo)
    out = capsys.readouterr().out
    ex = docmod._parse_example(_WITH, "python")
    assert ex is not None
    assert any(ln.strip() and ln.strip() in out for ln in ex)   # python @example surfaced in the printed card


def test_doc_falls_back_when_absent(tmp_path):
    p = tmp_path / "G.cpp"
    p.write_text(_WITHOUT, encoding="utf-8")
    assert docmod.doc(str(p)) is None      # display fn returns None (prints the card, no echo)
    assert docmod._parse_example(_WITHOUT, "python") is None   # no @example block to surface


_MACROS = """\
//  M.cpp -- m
#include <RcppArmadillo.h>
class M { public: M(const arma::vec& y){} void step(int n){} };
PYBIND11_MODULE(M_module, m){ pybind11::class_<M>(m, "M")
    .def("step", &M::step)
    .def("get_history", &M::get_history)
    AI4BAYESCODE_PYBIND_READAPT_NUTS(M)
    AI4BAYESCODE_PYBIND_KERNEL_CONTROL(M); }
"""

_MACRO_IN_COMMENT = """\
//  N.cpp -- m
#include <RcppArmadillo.h>
class N { public: N(const arma::vec& y){} void step(int n){} };
PYBIND11_MODULE(N_module, m){ pybind11::class_<N>(m, "N")
    .def("step", &N::step)
    // AI4BAYESCODE_PYBIND_KERNEL_CONTROL(N)
    ; }
"""

_MACRO_METHOD_NAMES = ("clear_history", "freeze", "unfreeze", "get_frozen", "readapt_NUTS")


def _card_methods(path, capsys):
    capsys.readouterr()
    docmod.doc(str(path))
    out = capsys.readouterr().out
    return out.split("\nMethods:\n", 1)[1].split("\nExample:", 1)[0]


def test_methods_list_covers_macro_bound_methods(tmp_path, capsys):
    """The binding macros register their methods although no .def("...")
    names them in the source; a macro name in a comment registers nothing."""
    p = tmp_path / "M.cpp"
    p.write_text(_MACROS, encoding="utf-8")
    meth = _card_methods(p, capsys)
    for nm in _MACRO_METHOD_NAMES:
        assert f"  .{nm}(" in meth, nm
    assert "warmup included" not in meth

    q = tmp_path / "N.cpp"
    q.write_text(_MACRO_IN_COMMENT, encoding="utf-8")
    meth = _card_methods(q, capsys)
    for nm in _MACRO_METHOD_NAMES:
        assert f"  .{nm}(" not in meth, nm


def test_every_bundled_example_lists_clear_history(capsys):
    import glob
    import os
    exdir = os.path.join(os.path.dirname(docmod.__file__), "_examples")
    files = sorted(glob.glob(os.path.join(exdir, "*.cpp")))
    if not files:
        import pytest
        pytest.skip("no bundled examples")
    without = [os.path.basename(f) for f in files
               if "  .clear_history(" not in _card_methods(f, capsys)]
    assert without == []

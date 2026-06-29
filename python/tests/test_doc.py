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
    p.write_text(_WITH)
    r = docmod.doc(str(p))
    assert r is not None
    assert r["example"] == docmod._parse_example(_WITH, "python")


def test_doc_falls_back_when_absent(tmp_path):
    p = tmp_path / "G.cpp"
    p.write_text(_WITHOUT)
    r = docmod.doc(str(p))
    assert r is not None
    assert r["example"] is None

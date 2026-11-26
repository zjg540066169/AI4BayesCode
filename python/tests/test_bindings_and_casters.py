"""Binding macros and pybind11 casters of the compiled models.

* clear_history is bound through kcm_detail::clear_history_binding. A wrapper
  whose own clear_history takes another parameter list, or is private, gets
  the mixin's (the stored draws are emptied and its own function is not
  called); a wrapper whose own clear_history takes no argument gets that one;
  a class derived from a wrapper that declares clear_history() gets that one.
* freeze / unfreeze under warnings-as-errors raise UserWarning (not
  SystemError) at the first warning.
* A state_map argument accepts numbers, 0-d / 1-D / 2-D arrays and
  sequences; a 2-D value is flattened column-major, as Rcpp flattens an R
  matrix. A returned state_map is a dict of 1-D float64 arrays, as before.
  The caster compiles with and without pybind11/stl.h included first, and
  the arma::vec caster of other arguments is unchanged.
* predict_at of the shipped examples checks the column count of a 2-D X;
  ARDLasso takes last_draw_only.

Every test compiles C++ and is marked ``compile``.
"""
from __future__ import annotations

import collections.abc
import types
import warnings

import numpy as np
import pytest

import AI4BayesCode

pytestmark = pytest.mark.compile

_PRELUDE = r"""
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
"""

# Wrappers on kernel_control_mixin with two gamma_gibbs blocks "s" and "t"
# (keep_history on). own_calls counts the calls of a wrapper's own
# clear_history.
_TOYS = _PRELUDE + r"""
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/gamma_gibbs_block.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"
#include "AI4BayesCode/pybind_casters.hpp"
#include <memory>
#include <random>
#include <string>
#include <vector>

#define KC_TOY_BODY(CLS)                                                        \
public:                                                                         \
    CLS() : rng_(7), impl_(std::make_unique<AI4BayesCode::composite_block>(#CLS)) { \
        for (const char* nm : {"s", "t"}) {                                     \
            AI4BayesCode::gamma_gibbs_block_config c;                           \
            c.name = nm;                                                        \
            c.initial_value = 1.0;                                              \
            c.params_fn = [](const AI4BayesCode::block_context&) {             \
                return AI4BayesCode::gamma_params{3.0, 2.0};                    \
            };                                                                  \
            impl_->add_child(                                                   \
                std::make_unique<AI4BayesCode::gamma_gibbs_block>(std::move(c))); \
        }                                                                       \
        impl_->set_keep_history(true);                                          \
    }                                                                           \
    void step(int n) { for (int i = 0; i < n; ++i) impl_->step(rng_); }       \
    int n_rows() const { return static_cast<int>(impl_->get_history().at("s").n_rows); } \
    int calls() const { return own_calls; }                                     \
    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map&) const { \
        return {};                                                              \
    }                                                                           \
    int own_calls = 0;                                                          \
    std::vector<double> readapt_args;                                           \
    void readapt_NUTS(int n, bool reset, int depth, double target) {            \
        readapt_args = {double(n), double(reset), double(depth), target};       \
    }                                                                           \
    void readapt_NUTS(int n, bool reset = false, int depth = -1) {             \
        readapt_NUTS(n, reset, depth, -2.0);                                    \
    }                                                                           \
    std::vector<double> last_readapt() const { return readapt_args; }          \
    std::mt19937_64 rng_;                                                       \
    std::unique_ptr<AI4BayesCode::composite_block> impl_;

class KcPlain : public AI4BayesCode::kernel_control_mixin<KcPlain> {
    KC_TOY_BODY(KcPlain)
};

class KcBoolClear : public AI4BayesCode::kernel_control_mixin<KcBoolClear> {
    KC_TOY_BODY(KcBoolClear)
    void clear_history(bool) { ++own_calls; }
};

class KcStringClear : public AI4BayesCode::kernel_control_mixin<KcStringClear> {
    KC_TOY_BODY(KcStringClear)
    std::string clear_history() { ++own_calls; return "own"; }
};

class KcPrivateClear : public AI4BayesCode::kernel_control_mixin<KcPrivateClear> {
    KC_TOY_BODY(KcPrivateClear)
private:
    void clear_history() { ++own_calls; }
};

class KcSub : public KcPlain {
public:
    std::vector<double> own_buf;
    void step(int n) {
        KcPlain::step(n);
        for (int i = 0; i < n; ++i) own_buf.push_back(1.0);
    }
    int own_rows() const { return static_cast<int>(own_buf.size()); }
    void clear_history() { KcPlain::clear_history(); own_buf.clear(); ++own_calls; }
};


// readapt_NUTS of other shapes, bound by AI4BAYESCODE_BIND_READAPT_NUTS /
// AI4BAYESCODE_PYBIND_READAPT_NUTS. last() is (n, reset, depth, target) of
// the last call; target -9 marks a readapt_NUTS without target_accept and -2
// the 3-argument overload of a class with both.
#define RX_BODY                                                                 \
public:                                                                         \
    mutable std::vector<double> args;                                           \
    std::vector<double> last() const { return args; }
class Rx4d {     // 1.1.2 compiled this one; 1.1.3 with static_cast did not
    RX_BODY
    void readapt_NUTS(int n, bool r, int d, double t = -1.0) { args = {double(n), double(r), double(d), t}; }
};
class Rx4 {
    RX_BODY
    void readapt_NUTS(int n, bool r, int d, double t) { args = {double(n), double(r), double(d), t}; }
};
class Rx3 {
    RX_BODY
    void readapt_NUTS(int n, bool r = false, int d = -1) { args = {double(n), double(r), double(d), -9.0}; }
};
class RxBothDef {  // the 3-argument call would be ambiguous
    RX_BODY
    void readapt_NUTS(int n, bool r, int d, double t = -1.0) { args = {double(n), double(r), double(d), t}; }
    void readapt_NUTS(int n, bool r = false, int d = -1) { args = {double(n), double(r), double(d), -2.0}; }
};
class RxConst {
    RX_BODY
    void readapt_NUTS(int n, bool r, int d, double t) const { args = {double(n), double(r), double(d), t}; }
};
class RxSizeT {
    RX_BODY
    void readapt_NUTS(std::size_t n, bool r, int d, double t) { args = {double(n), double(r), double(d), t}; }
};
class RxNone {   // takes neither argument list: compiles, every call raises
    RX_BODY
    void readapt_NUTS(int n) { args = {double(n)}; }
};

#define KC_BIND(CLS)                                                            \
    pybind11::class_<CLS>(m, #CLS)                                              \
        .def(pybind11::init<>())                                                \
        .def("step", &CLS::step)                                                \
        .def("n_rows", &CLS::n_rows)                                            \
        .def("calls", &CLS::calls)                                              \
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(CLS)

PYBIND11_MODULE(KcToysPy, m) {
    KC_BIND(KcPlain)
        .def("last_readapt", &KcPlain::last_readapt)
        AI4BAYESCODE_PYBIND_READAPT_NUTS(KcPlain);
    KC_BIND(KcBoolClear);
    KC_BIND(KcStringClear);
    KC_BIND(KcPrivateClear);
    KC_BIND(KcSub).def("own_rows", &KcSub::own_rows);
#define RX_BIND(CLS)                                                            \
    pybind11::class_<CLS>(m, #CLS)                                              \
        .def(pybind11::init<>())                                                \
        .def("last", &CLS::last)                                                \
        AI4BAYESCODE_PYBIND_READAPT_NUTS(CLS)
    RX_BIND(Rx4d); RX_BIND(Rx4); RX_BIND(Rx3); RX_BIND(RxBothDef);
    RX_BIND(RxConst); RX_BIND(RxSizeT); RX_BIND(RxNone);
}
"""


def _probe_source(name: str, stl_first: bool) -> str:
    first = "#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n" if stl_first else ""
    return _PRELUDE + first + r"""
#include "AI4BayesCode/pybind_casters.hpp"
#include "AI4BayesCode/rcpp_predict_guard.hpp"
#include <cstddef>

AI4BayesCode::state_map echo(const AI4BayesCode::state_map& s) { return s; }

PYBIND11_MODULE(""" + name + r""", m) {
    m.def("echo", &echo);
    m.def("ncol", [](const AI4BayesCode::py_new_data& d) { return d.ncol; });
    m.def("guard", [](const AI4BayesCode::py_new_data& d, std::size_t p) {
        return ai4b::predict_input(d, "X", p);
    });
    m.def("vec", [](const arma::vec& v) { return v; });
}
"""


@pytest.fixture(scope="module")
def toys(tmp_path_factory):
    return AI4BayesCode.source(_TOYS, cache_dir=tmp_path_factory.mktemp("kc"), quiet=True)


@pytest.fixture(scope="module", params=[True, False], ids=["stl_first", "casters_only"])
def probe(request, tmp_path_factory):
    name = "CasterProbeStl" if request.param else "CasterProbeOnly"
    return AI4BayesCode.source(_probe_source(name, request.param),
                               cache_dir=tmp_path_factory.mktemp(name), quiet=True)


# ---- clear_history binding -------------------------------------------------

def test_clear_history_binding_dispatch(toys):
    for cls, own_calls, cleared in [("KcPlain", 0, True),
                                    ("KcBoolClear", 0, True),     # mixin's
                                    ("KcPrivateClear", 0, True),  # mixin's
                                    ("KcStringClear", 1, False),  # its own
                                    ("KcSub", 1, True)]:          # its own
        m = getattr(toys, cls)()
        m.step(5)
        assert m.n_rows() == 5
        assert m.clear_history() is None
        m.step(3)
        assert m.calls() == own_calls, cls
        assert m.n_rows() == (3 if cleared else 8), cls
    sub = toys.KcSub()
    sub.step(4)
    sub.clear_history()
    sub.step(2)
    assert (sub.n_rows(), sub.own_rows()) == (2, 2)


def test_readapt_binding_selects_the_overload(toys):
    m = toys.KcPlain()
    m.readapt_NUTS(5)
    assert m.last_readapt() == [5.0, 0.0, -1.0, -1.0]
    m.readapt_NUTS(6, True, 2, 0.7)
    assert m.last_readapt() == [6.0, 1.0, 2.0, 0.7]


def test_readapt_binding_with_one_of_the_two_forms(toys):
    def run(cls, *args, **kw):
        m = getattr(toys, cls)()
        m.readapt_NUTS(*args, **kw)
        return m.last()
    # only the 4-argument form, or both: target_accept = -1 by default
    for cls in ("Rx4d", "Rx4", "RxConst", "RxSizeT", "RxBothDef"):
        assert run(cls, 5) == [5.0, 0.0, -1.0, -1.0], cls
        assert run(cls, 6, True, 2, 0.7) == [6.0, 1.0, 2.0, 0.7], cls
    # only the 3-argument form: target_accept that keeps the target calls it
    assert run("Rx3", 5) == [5.0, 0.0, -1.0, -9.0]
    assert run("Rx3", 6, True, 2, -1.0) == [6.0, 1.0, 2.0, -9.0]
    with pytest.raises(ValueError, match="takes no target_accept"):
        run("Rx3", 5, target_accept=0.7)
    with pytest.raises(RuntimeError, match="takes neither"):
        run("RxNone", 5)


# ---- freeze / unfreeze warnings (PyErr_WarnEx return value) ----------------

def test_freeze_warning_as_error_raises_userwarning(toys):
    m = toys.KcPlain()
    m.freeze(["s"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for names in (["s"], ["s", "s"]):
            with pytest.raises(UserWarning) as ei:
                m.freeze(names)
            assert type(ei.value) is UserWarning
            assert "already frozen" in str(ei.value)
        with pytest.raises(UserWarning) as ei:
            m.unfreeze(["t"])
        assert type(ei.value) is UserWarning
    assert m.get_frozen() == ["s"]
    with pytest.warns(UserWarning) as rec:
        m.freeze(["s", "s"])
    assert len(rec) == 2
    m.step(2)
    assert m.get_frozen() == ["s"]


# ---- state_map caster ------------------------------------------------------

def test_state_map_values(probe):
    out = probe.echo({"a": 1.5, "b": np.float64(2.0), "c": np.array(3.0),
                      "d": [1, 2, 3], "e": (4.0, 5.0), "f": np.arange(3, dtype=np.int32),
                      "g": True})
    assert set(out) == set("abcdefg")
    for v in out.values():
        assert isinstance(v, np.ndarray) and v.dtype == np.float64 and v.ndim == 1
    np.testing.assert_array_equal(out["a"], [1.5])
    np.testing.assert_array_equal(out["b"], [2.0])
    np.testing.assert_array_equal(out["c"], [3.0])
    np.testing.assert_array_equal(out["d"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(out["e"], [4.0, 5.0])
    np.testing.assert_array_equal(out["f"], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(out["g"], [1.0])


def test_state_map_matrix_is_column_major(probe):
    X = np.arange(12.0).reshape(4, 3)
    want = X.ravel(order="F")
    for val in (X, np.asfortranarray(X), X.tolist(), np.arange(24.0).reshape(4, 6)[:, ::2]):
        got = probe.echo({"X": val})["X"]
        np.testing.assert_array_equal(got, np.asarray(val, dtype=float).ravel(order="F"))
    np.testing.assert_array_equal(probe.echo({"X": X})["X"], want)
    assert probe.ncol({"X": X, "y": np.zeros(4), "z": 1.0}) == {"X": 3}


def test_state_map_mappings(probe):
    class M(collections.abc.Mapping):
        def __init__(self, d):
            self._d = d

        def __getitem__(self, k):
            return self._d[k]

        def __iter__(self):
            return iter(self._d)

        def __len__(self):
            return len(self._d)

    for mp in (types.MappingProxyType({"a": [1.0]}), M({"a": [1.0]})):
        np.testing.assert_array_equal(probe.echo(mp)["a"], [1.0])
    assert probe.echo({}) == {}


@pytest.mark.parametrize("bad", [{"a": None}, {"a": "1.5"}, {"a": b"x"},
                                 {"a": np.zeros((2, 2, 2))}, {1: [1.0]},
                                 {"a": [[1.0, 2.0], [3.0]]}, [("a", [1.0])]])
def test_state_map_rejects(probe, bad):
    with pytest.raises(TypeError):
        probe.echo(bad)


def test_arma_vec_caster_unchanged(probe):
    np.testing.assert_array_equal(probe.vec(np.array([1.0, 2.0])), [1.0, 2.0])
    for bad in (np.zeros((2, 2)), [1.0, 2.0], 1.0):
        with pytest.raises(TypeError):
            probe.vec(bad)


def test_predict_input_guard(probe):
    X = np.arange(12.0).reshape(4, 3)
    np.testing.assert_array_equal(probe.guard({"X": X}, 3)["X"], X.ravel(order="F"))
    np.testing.assert_array_equal(probe.guard({"X": X.ravel(order="F")}, 3)["X"],
                                  X.ravel(order="F"))
    np.testing.assert_array_equal(probe.guard({"X": X[:, :2]}, 0)["X"], X[:, :2].ravel(order="F"))
    with pytest.raises(RuntimeError, match="'X' has 2 columns but the model was fitted with p = 3"):
        probe.guard({"X": X[:, :2]}, 3)
    # a 2-D value with 0 columns: recorded, and rejected as R rejects a
    # matrix with 0 columns
    assert probe.ncol({"X": np.zeros((4, 0)), "y": []}) == {"X": 0}
    with pytest.raises(RuntimeError, match="'X' has 0 columns but the model was fitted with p = 3"):
        probe.guard({"X": np.zeros((4, 0))}, 3)


# ---- shipped examples ------------------------------------------------------

def test_probit_predict_with_matrix_x():
    mod = AI4BayesCode.example("ProbitRegression", quiet=True)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    y = (X @ np.array([1.0, -1.0, 0.5]) + rng.normal(size=40) > 0).astype(float)
    Xn = rng.normal(size=(5, 3))
    m = mod.ProbitRegression(X, y, 10.0, 7, True)
    m.step(20)
    flat = m.predict_at({"X": Xn.ravel(order="F")})["mu_lin"]
    assert flat.shape == (20, 5)
    np.testing.assert_array_equal(m.predict_at({"X": Xn})["mu_lin"], flat)
    np.testing.assert_array_equal(m.predict_at({"X": Xn.tolist()})["mu_lin"], flat)
    np.testing.assert_allclose(flat, m.get_history()["beta"] @ Xn.T, rtol=0, atol=1e-12)
    last = m.predict_at({"X": Xn}, last_draw_only=True)["mu_lin"]
    np.testing.assert_array_equal(last, flat[-1:])
    with pytest.raises(RuntimeError, match="'X' has 2 columns but the model was fitted with p = 3"):
        m.predict_at({"X": Xn[:, :2]})
    chains = AI4BayesCode.run_chains(lambda s: mod.ProbitRegression(X, y, 10.0, s, True),
                                     seeds=[11, 12], n_burn=10, n_keep=15, n_cores=1,
                                     verbose=False)
    pooled = AI4BayesCode.predict(chains, {"X": Xn})["mu_lin"]
    np.testing.assert_allclose(pooled, np.vstack([c["hist"]["beta"] @ Xn.T for c in chains]),
                               rtol=0, atol=1e-12)


def test_ardlasso_last_draw_only():
    mod = AI4BayesCode.example("ARDLasso", quiet=True)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(50, 4))
    Y = 1.0 + X @ np.array([2.0, 0.0, -1.0, 0.0]) + 0.5 * rng.normal(size=50)
    Xn = rng.normal(size=(6, 4))
    m = mod.ARDLasso(X, Y, 9, True)
    m.step(12)
    hist = m.get_history()
    full = m.predict_at({"X": Xn})
    assert full["y_hat"].shape == (12, 6) and full["y_rep"].shape == (12, 6)
    np.testing.assert_allclose(full["y_hat"], hist["beta"] @ Xn.T + hist["alpha"],
                               rtol=0, atol=1e-12)
    last = m.predict_at({"X": Xn}, last_draw_only=True)
    assert last["y_hat"].shape == (1, 6) and last["y_rep"].shape == (1, 6)
    np.testing.assert_allclose(last["y_hat"], full["y_hat"][-1:], rtol=0, atol=1e-12)
    assert m.predict_at({"X": Xn.ravel(order="F")}, False)["y_hat"].shape == (12, 6)
    assert m.predict_at({})["y_rep"].shape == (12, 50)
    with pytest.raises(RuntimeError, match="'X' has 3 columns but the model was fitted with p = 4"):
        m.predict_at({"X": Xn[:, :3]})

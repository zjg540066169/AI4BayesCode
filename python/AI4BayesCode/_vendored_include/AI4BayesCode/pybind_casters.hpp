/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later).
 *================================================================================
 *
 *  pybind_casters.hpp  --  pybind11 type casters for the neutral types
 *                          declared in types.hpp, plus arma::vec /
 *                          arma::mat <-> numpy conversions.
 *
 *  Include this header in any .cpp that emits a PYBIND11_MODULE and
 *  returns AI4BayesCode::state_map, history_map, dag_info, or
 *  adaptation_info to Python. Include BEFORE pybind11.h in the
 *  including .cpp — this header pulls in pybind11 itself and also
 *  registers the casters.
 *
 *  Casters implemented:
 *    - arma::vec          <-> py::array_t<double>  (1-D, writable copy)
 *    - arma::mat          <-> py::array_t<double>  (2-D, column-major numpy
 *                                                   copied to row-major Python)
 *    - state_map          <-> dict[str, np.ndarray]  (input: any Mapping; each
 *                                                   value converted as Rcpp
 *                                                   converts an R value, see
 *                                                   state_value_from_python)
 *    - py_new_data        <-  the same Mapping, plus the column count of
 *                             each 2-D value (predict_at's covariate check)
 *    - history_map        <-> dict[str, np.ndarray]  (2-D)
 *    - dag_info           -> dict[str, Any]  (via py::class_ binding in module)
 *    - adaptation_info    -> dict[str, Any]  (via py::class_ binding in module)
 *
 *  Gotcha: Armadillo matrices are COLUMN-MAJOR. numpy defaults to
 *  ROW-MAJOR. The arma::mat caster copies into a numpy array with the
 *  same element order the user would expect (row i, column j), which
 *  means one transposition on each crossing of the boundary. If you
 *  care about zero-copy transfers for large matrices, use the
 *  read_as_fortran variant (not implemented here — premature
 *  optimization for our MCMC chains, whose histories are at most
 *  tens of MB).
 *================================================================================*/

#ifndef AI4BAYESCODE_PYBIND_CASTERS_HPP
#define AI4BAYESCODE_PYBIND_CASTERS_HPP

#include "types.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <armadillo>

#include <cstddef>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace AI4BayesCode {

/**
 * predict_at's new_data as it came from Python: the values, converted as for
 * a state_map argument, and the number of columns of every value that was a
 * 2-D array or a nested sequence. ai4b::predict_input (rcpp_predict_guard.hpp)
 * compares the covariate's column count with the model's p before the column
 * count is lost, as it does for an R matrix.
 */
struct py_new_data {
    state_map values;
    std::unordered_map<std::string, std::size_t> ncol;
};

namespace detail {

/**
 * One value of a state_map argument, converted as Rcpp converts an R value
 * to arma::vec: a number or 0-d array gives one element, a 1-D array or
 * sequence its elements, and a 2-D array or nested sequence its elements in
 * column-major order (element (i, j) of an n x p value at index i + j*n, the
 * order of as.vector() of an R matrix). For a 2-D value *is_2d is set to
 * true and *ncol to p (0 included); otherwise *is_2d is set to false and
 * *ncol to 0. Returns false, and converts nothing, for None, str, bytes, a
 * value numpy cannot convert to float64, and an array of more than two
 * dimensions.
 */
inline bool state_value_from_python(pybind11::handle src, arma::vec& out,
                                    std::size_t* ncol, bool* is_2d = nullptr) {
    if (ncol) *ncol = 0;
    if (is_2d) *is_2d = false;
    if (!src || src.is_none()) return false;
    PyObject* p = src.ptr();
    if (PyUnicode_Check(p) || PyBytes_Check(p) || PyByteArray_Check(p))
        return false;
    // ensure() returns a null handle, and clears the Python error, when numpy
    // cannot convert the value.
    auto arr = pybind11::array_t<double, pybind11::array::forcecast>::ensure(src);
    if (!arr) return false;
    if (arr.ndim() == 0) {
        out.set_size(1);
        out[0] = *arr.data();
    } else if (arr.ndim() == 1) {
        const auto r = arr.unchecked<1>();
        out.set_size(static_cast<arma::uword>(r.shape(0)));
        for (pybind11::ssize_t i = 0; i < r.shape(0); ++i)
            out[static_cast<arma::uword>(i)] = r(i);
    } else if (arr.ndim() == 2) {
        const auto r = arr.unchecked<2>();
        const auto n_row = r.shape(0), n_col = r.shape(1);
        out.set_size(static_cast<arma::uword>(n_row * n_col));
        for (pybind11::ssize_t j = 0; j < n_col; ++j)
            for (pybind11::ssize_t i = 0; i < n_row; ++i)
                out[static_cast<arma::uword>(i + j * n_row)] = r(i, j);
        if (ncol) *ncol = static_cast<std::size_t>(n_col);
        if (is_2d) *is_2d = true;
    } else {
        return false;
    }
    return true;
}

/**
 * The (key, value) pairs of a Python mapping converted into `values` (and
 * the column counts of the 2-D values into `ncol` when it is not null).
 * Accepts what pybind11's own map caster accepts: a dict, an object with
 * __getitem__ and a callable items(), and, in convert mode, an instance of
 * collections.abc.Mapping. Returns false for anything else, a key that is
 * not a string, or a value state_value_from_python rejects.
 */
inline bool state_map_from_python(
        pybind11::handle src, bool convert, state_map& values,
        std::unordered_map<std::string, std::size_t>* ncol) {
    if (!src) return false;
    PyObject* p = src.ptr();
    bool is_mapping = PyDict_Check(p) != 0;
    if (!is_mapping && PyMapping_Check(p) != 0) {
        PyObject* items = PyObject_GetAttrString(p, "items");
        if (items != nullptr) {
            is_mapping = PyCallable_Check(items) != 0;
            Py_DECREF(items);
        } else {
            PyErr_Clear();
        }
    }
    if (!is_mapping && convert) {
        is_mapping = pybind11::isinstance(
            src, pybind11::module_::import("collections.abc").attr("Mapping"));
    }
    if (!is_mapping) return false;

    auto items = pybind11::reinterpret_steal<pybind11::object>(PyMapping_Items(p));
    if (!items) throw pybind11::error_already_set();
    values.clear();
    if (ncol) ncol->clear();
    for (pybind11::handle item : items) {
        if (PySequence_Check(item.ptr()) == 0 || PySequence_Size(item.ptr()) != 2) {
            PyErr_Clear();
            return false;
        }
        const auto kv = pybind11::reinterpret_borrow<pybind11::sequence>(item);
        const pybind11::object k = kv[0];
        const pybind11::object v = kv[1];
        pybind11::detail::make_caster<std::string> key_conv;
        if (!key_conv.load(k, convert)) return false;
        std::string key = pybind11::detail::cast_op<std::string&&>(std::move(key_conv));
        arma::vec value;
        std::size_t n_col = 0;
        bool is_2d = false;
        if (!state_value_from_python(v, value, &n_col, &is_2d)) return false;
        // A 2-D value with 0 columns is recorded as well, so predict_input
        // rejects it as R does a matrix with 0 columns.
        if (ncol && is_2d) ncol->emplace(key, n_col);
        values.emplace(std::move(key), std::move(value));
    }
    return true;
}

} // namespace detail
} // namespace AI4BayesCode

namespace pybind11 { namespace detail {

// ---- arma::vec <-> numpy 1D ---------------------------------------------
template<> struct type_caster<arma::vec> {
public:
    PYBIND11_TYPE_CASTER(arma::vec, _("arma::vec"));

    bool load(handle src, bool /*convert*/) {
        if (!py::isinstance<py::array>(src)) return false;
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(src);
        if (!arr) return false;
        if (arr.ndim() != 1) return false;
        value.set_size(static_cast<arma::uword>(arr.size()));
        std::copy(arr.data(), arr.data() + arr.size(), value.memptr());
        return true;
    }

    static handle cast(const arma::vec& v, return_value_policy, handle) {
        py::array_t<double> out({static_cast<py::ssize_t>(v.n_elem)});
        std::copy(v.memptr(), v.memptr() + v.n_elem, out.mutable_data());
        return out.release();
    }
};

// ---- arma::mat <-> numpy 2D ---------------------------------------------
template<> struct type_caster<arma::mat> {
public:
    PYBIND11_TYPE_CASTER(arma::mat, _("arma::mat"));

    bool load(handle src, bool /*convert*/) {
        if (!py::isinstance<py::array>(src)) return false;
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(src);
        if (!arr) return false;
        if (arr.ndim() != 2) return false;
        const auto r = static_cast<arma::uword>(arr.shape(0));
        const auto c = static_cast<arma::uword>(arr.shape(1));
        value.set_size(r, c);
        // numpy is row-major C-contig; Armadillo is column-major. Copy
        // element by element.
        for (arma::uword i = 0; i < r; ++i) {
            for (arma::uword j = 0; j < c; ++j) {
                value(i, j) = *arr.data(i, j);
            }
        }
        return true;
    }

    static handle cast(const arma::mat& m, return_value_policy, handle) {
        py::array_t<double> out(
            { static_cast<py::ssize_t>(m.n_rows),
              static_cast<py::ssize_t>(m.n_cols) });
        auto r = out.mutable_unchecked<2>();
        for (arma::uword i = 0; i < m.n_rows; ++i) {
            for (arma::uword j = 0; j < m.n_cols; ++j) {
                r(static_cast<py::ssize_t>(i),
                  static_cast<py::ssize_t>(j)) = m(i, j);
            }
        }
        return out.release();
    }
};

// ---- arma::uvec <-> numpy 1D uint ---------------------------------------
template<> struct type_caster<arma::uvec> {
public:
    PYBIND11_TYPE_CASTER(arma::uvec, _("arma::uvec"));

    bool load(handle src, bool /*convert*/) {
        if (!py::isinstance<py::array>(src)) return false;
        auto arr = py::array_t<arma::uword,
                               py::array::c_style | py::array::forcecast>::ensure(src);
        if (!arr) return false;
        if (arr.ndim() != 1) return false;
        value.set_size(static_cast<arma::uword>(arr.size()));
        std::copy(arr.data(), arr.data() + arr.size(), value.memptr());
        return true;
    }

    static handle cast(const arma::uvec& v, return_value_policy, handle) {
        py::array_t<arma::uword> out({static_cast<py::ssize_t>(v.n_elem)});
        std::copy(v.memptr(), v.memptr() + v.n_elem, out.mutable_data());
        return out.release();
    }
};

// ---- state_map <-> dict[str, numpy 1D] -----------------------------------
// A partial specialization for unordered_map<std::string, arma::vec, ...>.
// It is more specialized than the unordered_map caster of pybind11/stl.h, so
// it is the one used whether or not stl.h was included first. Input: a
// Mapping whose values are converted by state_value_from_python (numbers,
// 0-d / 1-D / 2-D arrays, sequences; a 2-D value column-major, as Rcpp
// flattens an R matrix). The arma::vec caster above, which constructor and
// other arguments use, is unchanged. Output: a dict of 1-D arrays, the same
// as the stl.h caster with the arma::vec caster gave.
template <typename Hash, typename Equal, typename Alloc>
struct type_caster<std::unordered_map<std::string, arma::Col<double>, Hash, Equal, Alloc>> {
    using map_type = std::unordered_map<std::string, arma::Col<double>, Hash, Equal, Alloc>;

    bool load(handle src, bool convert) {
        AI4BayesCode::state_map tmp;
        if (!AI4BayesCode::detail::state_map_from_python(src, convert, tmp, nullptr))
            return false;
        value.clear();
        for (auto& kv : tmp) value.emplace(kv.first, std::move(kv.second));
        return true;
    }

    template <typename T>
    static handle cast(T&& src, return_value_policy policy, handle parent) {
        dict d;
        for (auto&& kv : src) {
            auto key = reinterpret_steal<object>(
                make_caster<std::string>::cast(kv.first, policy, parent));
            auto val = reinterpret_steal<object>(
                make_caster<arma::vec>::cast(kv.second, policy, parent));
            if (!key || !val) return handle();
            d[std::move(key)] = std::move(val);
        }
        return d.release();
    }

#if defined(PYBIND11_VERSION_MAJOR) && PYBIND11_VERSION_MAJOR >= 3
    PYBIND11_TYPE_CASTER(map_type,
                         io_name("collections.abc.Mapping[str, numpy.typing.ArrayLike]",
                                 "dict[str, numpy.ndarray]"));
#else
    PYBIND11_TYPE_CASTER(map_type, const_name("dict[str, numpy.ndarray]"));
#endif
};

// ---- py_new_data <- Mapping (predict_at input with column counts) --------
template <> struct type_caster<AI4BayesCode::py_new_data> {
    bool load(handle src, bool convert) {
        return AI4BayesCode::detail::state_map_from_python(src, convert, value.values,
                                                           &value.ncol);
    }

    static handle cast(const AI4BayesCode::py_new_data& src, return_value_policy policy,
                       handle parent) {
        return make_caster<AI4BayesCode::state_map>::cast(src.values, policy, parent);
    }

#if defined(PYBIND11_VERSION_MAJOR) && PYBIND11_VERSION_MAJOR >= 3
    PYBIND11_TYPE_CASTER(AI4BayesCode::py_new_data,
                         io_name("collections.abc.Mapping[str, numpy.typing.ArrayLike]",
                                 "dict[str, numpy.ndarray]"));
#else
    PYBIND11_TYPE_CASTER(AI4BayesCode::py_new_data, const_name("dict[str, numpy.ndarray]"));
#endif
};

} } // namespace pybind11::detail

// ---- register_ai4bayescode_types: call this from each PYBIND11_MODULE ------
namespace AI4BayesCode {

/// Value of adaptation_info::warmup_done from a Python object, by the rule of
/// the R list conversion in rcpp_wrap.hpp: None, NaN, pandas.NA and
/// pandas.NaT give false (as R's NULL and NA do, and as the default for a
/// snapshot without the value), where NaN is a number that compares unequal
/// to itself (float, numpy, complex and quiet decimal NaN, numpy NaT) or a
/// signaling decimal NaN; any other value gives its truth value when its type
/// defines __bool__ (True and False, numbers, numpy booleans) and raises
/// TypeError when it does not (str, list, tuple, dict, bytes) or when
/// __bool__ raises; a numpy array with exactly one element counts as that
/// element unless the element is itself an array (TypeError) or is masked
/// in a numpy masked array (false), and any other array raises TypeError.
inline bool warmup_done_from_python(const py::handle& v) {
    if (v.ptr() == Py_True) return true;
    if (v.ptr() == Py_False || v.is_none()) return false;
    // pandas.NA, pandas.NaT and numpy masked arrays exist only when pandas or
    // numpy.ma is in sys.modules, so the modules are looked up there and are
    // never imported here.
    const py::dict modules =
        py::reinterpret_borrow<py::dict>(PyImport_GetModuleDict());
    if (modules.contains("pandas")) {
        const py::object pd = modules["pandas"];
        for (const char* name : {"NA", "NaT"}) {
            if (!py::hasattr(pd, name)) continue;
            const py::object missing = pd.attr(name);
            if (v.is(missing)) return false;
        }
    }
    if (py::isinstance<py::array>(v)) {
        const auto n = py::reinterpret_borrow<py::array>(v).size();
        if (n != 1) {
            throw py::type_error("AdaptationInfo.warmup_done must be a single "
                                 "value, got an array of " + std::to_string(n) +
                                 " elements");
        }
        if (modules.contains("numpy.ma")) {
            const py::object ma = modules["numpy.ma"];
            const py::object masked_array = ma.attr("MaskedArray");
            if (py::isinstance(v, masked_array) &&
                ma.attr("is_masked")(v).cast<bool>()) return false;
        }
        // One level only: an element that is itself an array is rejected.
        const py::object x = v.attr("item")();
        if (py::isinstance<py::array>(x)) {
            throw py::type_error("AdaptationInfo.warmup_done must be a single "
                                 "value, got an array whose element is an "
                                 "array");
        }
        return warmup_done_from_python(x);
    }
    // A NaN compares unequal to itself; a signaling decimal NaN raises on
    // the comparison, and its is_nan() is true.
    if (PyNumber_Check(v.ptr())) {
        PyObject* ne = PyObject_RichCompare(v.ptr(), v.ptr(), Py_NE);
        int is_nan = ne ? PyObject_IsTrue(ne) : -1;
        Py_XDECREF(ne);
        if (is_nan < 0) {
            PyErr_Clear();
            const py::object decimal =
                py::module_::import("decimal").attr("Decimal");
            is_nan = (py::isinstance(v, decimal) &&
                      v.attr("is_nan")().cast<bool>()) ? 1 : 0;
        }
        if (is_nan == 1) return false;
    }
    pybind11::detail::make_caster<bool> conv;
    if (!conv.load(v, /*convert=*/true)) {
        throw py::type_error("AdaptationInfo.warmup_done must be a bool, a "
                             "number or None");
    }
    return pybind11::detail::cast_op<bool>(conv);
}

/**
 * Register py::class_ bindings for dag_info and adaptation_info.
 * Each PYBIND11_MODULE should call this once at module init so Python
 * users can introspect the returned structs via attribute access AND
 * via dict-like access (both styles wired below).
 *
 *     PYBIND11_MODULE(MyModel, m) {
 *         AI4BayesCode::register_ai4bayescode_types(m);
 *         py::class_<MyModel>(m, "MyModel")
 *             .def(...)
 *             ;
 *     }
 */
inline void register_ai4bayescode_types(py::module& m) {
    // Use py::module_local() so each .so can register its own copy without
    // clashing with other AI4BayesCode modules loaded in the same interpreter.
    // pybind11 treats module-local classes as distinct types per module.
    py::class_<dag_info>(m, "DagInfo", py::module_local())
        .def(py::init<>())
        .def_readwrite("gibbs_reads", &dag_info::gibbs_reads)
        .def_readwrite("gibbs_invalidates", &dag_info::gibbs_invalidates)
        .def_readwrite("predict_edges", &dag_info::predict_edges)
        .def_readwrite("context_edges", &dag_info::context_edges)
        .def_readwrite("data_inputs", &dag_info::data_inputs)
        .def_readwrite("data_input_groups", &dag_info::data_input_groups)
        .def("__getitem__", [](const dag_info& d, const std::string& k) -> py::object {
            if (k == "gibbs_reads")       return py::cast(d.gibbs_reads);
            if (k == "gibbs_invalidates") return py::cast(d.gibbs_invalidates);
            if (k == "predict_edges")     return py::cast(d.predict_edges);
            if (k == "context_edges")     return py::cast(d.context_edges);
            if (k == "data_inputs")       return py::cast(d.data_inputs);
            if (k == "data_input_groups") return py::cast(d.data_input_groups);
            throw py::key_error("no such dag field: " + k);
        })
        .def("keys", [](const dag_info&) {
            return std::vector<std::string>{
                "gibbs_reads", "gibbs_invalidates", "predict_edges",
                "context_edges", "data_inputs", "data_input_groups"};
        })
        .def("__contains__", [](const dag_info&, const std::string& k) {
            static const std::vector<std::string> keys{
                "gibbs_reads", "gibbs_invalidates", "predict_edges",
                "context_edges", "data_inputs", "data_input_groups"};
            return std::find(keys.begin(), keys.end(), k) != keys.end();
        });

    py::class_<adaptation_info>(m, "AdaptationInfo", py::module_local())
        .def(py::init<>())
        .def_readwrite("step_size", &adaptation_info::step_size)
        .def_readwrite("epsilon_bar", &adaptation_info::epsilon_bar)
        .def_readwrite("h_val", &adaptation_info::h_val)
        .def_readwrite("mu_val", &adaptation_info::mu_val)
        .def_readwrite("adapt_iter", &adaptation_info::adapt_iter)
        .def_readwrite("precond_mat", &adaptation_info::precond_mat)
        .def_readwrite("precond_diag", &adaptation_info::precond_diag)
        .def_readwrite("metric_kind", &adaptation_info::metric_kind)
        .def_property("warmup_done",
            [](const adaptation_info& a) { return a.warmup_done; },
            [](adaptation_info& a, const py::object& v) {
                a.warmup_done = warmup_done_from_python(v);
            });
}

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_PYBIND_CASTERS_HPP

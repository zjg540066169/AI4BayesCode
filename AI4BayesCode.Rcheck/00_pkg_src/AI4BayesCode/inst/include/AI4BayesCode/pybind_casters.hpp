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
 *    - state_map          <-> dict[str, np.ndarray]
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

namespace py = pybind11;

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

} } // namespace pybind11::detail

// ---- register_ai4bayescode_types: call this from each PYBIND11_MODULE ------
namespace AI4BayesCode {

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
        .def("__getitem__", [](const dag_info& d, const std::string& k) -> py::object {
            if (k == "gibbs_reads")       return py::cast(d.gibbs_reads);
            if (k == "gibbs_invalidates") return py::cast(d.gibbs_invalidates);
            if (k == "predict_edges")     return py::cast(d.predict_edges);
            if (k == "context_edges")     return py::cast(d.context_edges);
            if (k == "data_inputs")       return py::cast(d.data_inputs);
            throw py::key_error("no such dag field: " + k);
        })
        .def("keys", [](const dag_info&) {
            return std::vector<std::string>{
                "gibbs_reads", "gibbs_invalidates", "predict_edges",
                "context_edges", "data_inputs"};
        })
        .def("__contains__", [](const dag_info&, const std::string& k) {
            static const std::vector<std::string> keys{
                "gibbs_reads", "gibbs_invalidates", "predict_edges",
                "context_edges", "data_inputs"};
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
        .def_readwrite("metric_kind", &adaptation_info::metric_kind);
}

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_PYBIND_CASTERS_HPP

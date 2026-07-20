/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  kernel_control_mixin.hpp  --  CRTP mixin + Rcpp/pybind macros for the
 *                                kernel-control category (freeze / unfreeze /
 *                                get_frozen) per interface.md Sec.1.
 *
 *  DESIGN
 *  ======
 *  Every user-facing wrapper class inherits kernel_control_mixin<Derived>
 *  to automatically expose the three kernel-control methods via the
 *  RCPP_MODULE / PYBIND11_MODULE binding macros. The mixin forwards R/Py
 *  argument types to composite_block's C++ freeze/unfreeze/get_frozen and
 *  converts std::runtime_error / std::invalid_argument into Rcpp::stop
 *  (auto-handled by Rcpp) and advisory std::string warnings into
 *  Rcpp::warning / Python UserWarning.
 *
 *  REQUIREMENTS on Derived
 *  -----------------------
 *  The mixin uses CRTP static_cast<Derived*>(this) to reach the wrapper's
 *  composite_block member. Derived MUST have a member named exactly
 *  `impl_` of type std::unique_ptr<AI4BayesCode::composite_block>. This is
 *  the standard shape codegen_cpp.md Sec.8 already emits.
 *
 *  See DESIGN_NOTES_FREEZE_UNFREEZE_2026-07-19.md Sec.10 for the full
 *  contract and rationale.
 *================================================================================*/

#ifndef AI4BAYESCODE_KERNEL_CONTROL_MIXIN_HPP
#define AI4BAYESCODE_KERNEL_CONTROL_MIXIN_HPP

#include "composite_block.hpp"

#include <string>
#include <vector>

// Rcpp / pybind11 are optional. When neither is defined, the mixin still
// compiles for pure-C++ int-main use; the R/Py forwarder methods are
// simply unavailable outside those build modes.
#ifdef AI4BAYESCODE_RCPP_MODULE
# include <RcppArmadillo.h>
#endif
#ifdef AI4BAYESCODE_PYBIND_MODULE
# include <pybind11/pybind11.h>
# include <pybind11/stl.h>
#endif

namespace AI4BayesCode {

/**
 * @brief CRTP mixin that adds kernel-control forwarders to a wrapper
 *        class. See file header for requirements on Derived.
 *
 * Usage:
 *   class MyWrapper : public AI4BayesCode::kernel_control_mixin<MyWrapper> {
 *       std::unique_ptr<AI4BayesCode::composite_block> impl_;
 *       // ...
 *   };
 *   RCPP_MODULE(MyWrapper_module) {
 *       Rcpp::class_<MyWrapper>("MyWrapper")
 *         // ... core-6 methods + optional readapt_NUTS ...
 *         AI4BAYESCODE_BIND_KERNEL_CONTROL(MyWrapper);
 *   }
 */
template <typename Derived>
class kernel_control_mixin {
public:

#ifdef AI4BAYESCODE_RCPP_MODULE
    // ---- R-layer forwarders (Rcpp-typed) -------------------------------

    void freeze_names(const Rcpp::CharacterVector& names) {
        freeze_names_impl_(names, false);
    }

    void freeze_names_quiet(const Rcpp::CharacterVector& names, bool quiet) {
        freeze_names_impl_(names, quiet);
    }

    void unfreeze_names(Rcpp::Nullable<Rcpp::CharacterVector> names) {
        auto self = static_cast<Derived*>(this);
        if (names.isNull()) {
            self->impl_->unfreeze_all();
            return;
        }
        auto v = Rcpp::as<std::vector<std::string>>(names.get());
        if (v.empty()) {
            Rcpp::stop(
                "unfreeze(character(0)) not allowed; call unfreeze() with no "
                "argument to release all");
        }
        auto warnings = self->impl_->unfreeze(v);
        for (const auto& w : warnings) Rcpp::warning(w);
    }

    Rcpp::CharacterVector get_frozen_names() const {
        auto self = static_cast<const Derived*>(this);
        auto v = self->impl_->get_frozen();   // DFS pre-order
        return Rcpp::wrap(v);
    }

private:
    void freeze_names_impl_(const Rcpp::CharacterVector& names, bool quiet) {
        auto self = static_cast<Derived*>(this);
        auto v = Rcpp::as<std::vector<std::string>>(names);
        if (v.empty()) {
            Rcpp::stop("freeze() requires non-empty CharacterVector");
        }
        auto warnings = self->impl_->freeze(v, quiet);
        for (const auto& w : warnings) Rcpp::warning(w);
    }
public:
#endif // AI4BAYESCODE_RCPP_MODULE

#ifdef AI4BAYESCODE_PYBIND_MODULE
    // ---- Python-layer forwarders --------------------------------------

    void py_freeze(const std::vector<std::string>& names, bool quiet) {
        auto self = static_cast<Derived*>(this);
        if (names.empty()) {
            throw std::invalid_argument(
                "freeze() requires a non-empty list of names");
        }
        auto warnings = self->impl_->freeze(names, quiet);
        for (const auto& w : warnings) {
            PyErr_WarnEx(PyExc_UserWarning, w.c_str(), 1);
        }
    }

    void py_unfreeze(pybind11::object names) {
        auto self = static_cast<Derived*>(this);
        if (names.is_none()) {
            self->impl_->unfreeze_all();
            return;
        }
        auto v = names.cast<std::vector<std::string>>();
        if (v.empty()) {
            throw std::invalid_argument(
                "unfreeze([]) not allowed; call unfreeze() with no argument "
                "(or None) to release all");
        }
        auto warnings = self->impl_->unfreeze(v);
        for (const auto& w : warnings) {
            PyErr_WarnEx(PyExc_UserWarning, w.c_str(), 1);
        }
    }

    std::vector<std::string> py_get_frozen() const {
        auto self = static_cast<const Derived*>(this);
        return self->impl_->get_frozen();
    }
#endif // AI4BAYESCODE_PYBIND_MODULE
};

} // namespace AI4BayesCode

// ---- Rcpp module-binding macro ------------------------------------------
//
// Emits two overloads of `freeze` (1-arg and 2-arg) because Rcpp modules
// IGNORE C++ default arguments (same pattern as step() in codegen_cpp.md).
// Both `m$freeze(names)` and `m$freeze(names, quiet=TRUE)` then work
// from R.

#ifdef AI4BAYESCODE_RCPP_MODULE
#define AI4BAYESCODE_BIND_KERNEL_CONTROL(CLASSNAME)                       \
    .method("freeze",                                                     \
            (void (CLASSNAME::*)(const Rcpp::CharacterVector&))           \
                &CLASSNAME::freeze_names,                                 \
            "Fix sub-kernel(s) at current value (kernel-control category)") \
    .method("freeze",                                                     \
            (void (CLASSNAME::*)(const Rcpp::CharacterVector&, bool))     \
                &CLASSNAME::freeze_names_quiet,                           \
            "Fix sub-kernel(s); quiet=TRUE suppresses redundant-refreeze warning") \
    .method("unfreeze",                                                   \
            &CLASSNAME::unfreeze_names,                                   \
            "Release sub-kernel(s); no arg / NULL = all")                 \
    .method("get_frozen",                                                 \
            &CLASSNAME::get_frozen_names,                                 \
            "List currently-frozen block names (DFS pre-order)")
#endif // AI4BAYESCODE_RCPP_MODULE

// ---- pybind11 module-binding macro --------------------------------------
//
// pybind11 supports C++ default arguments natively, so the single-def
// covers both call shapes (`m.freeze(names)` and
// `m.freeze(names, quiet=True)`).

#ifdef AI4BAYESCODE_PYBIND_MODULE
#define AI4BAYESCODE_PYBIND_KERNEL_CONTROL(M, CLASSNAME)                  \
    M.def("freeze",                                                       \
          &CLASSNAME::py_freeze,                                          \
          pybind11::arg("names"),                                         \
          pybind11::arg("quiet") = false,                                 \
          "Fix sub-kernel(s) at current value; quiet=True suppresses "    \
          "redundant-refreeze warning")                                   \
     .def("unfreeze",                                                     \
          &CLASSNAME::py_unfreeze,                                        \
          pybind11::arg("names") = pybind11::none(),                      \
          "Release sub-kernel(s); no arg / None = all")                   \
     .def("get_frozen",                                                   \
          &CLASSNAME::py_get_frozen,                                      \
          "List currently-frozen block names (DFS pre-order)")
#endif // AI4BAYESCODE_PYBIND_MODULE

#endif // AI4BAYESCODE_KERNEL_CONTROL_MIXIN_HPP

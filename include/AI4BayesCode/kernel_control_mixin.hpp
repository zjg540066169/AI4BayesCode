/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  kernel_control_mixin.hpp  --  CRTP mixin + Rcpp/pybind macros for the
 *                                kernel-control category (freeze / unfreeze /
 *                                get_frozen / clear_history; readapt_NUTS has
 *                                its own macro below) per interface.md Sec.1.
 *
 *  DESIGN
 *  ======
 *  Every user-facing wrapper class inherits kernel_control_mixin<Derived>
 *  to automatically expose the four kernel-control methods via the
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
 *  See skills/system_design_skills/interface.md Sec.1 for the full
 *  contract and rationale.
 *
 *  HISTORY RESET
 *  -------------
 *  clear_history(), the fourth kernel-control method, is backend-neutral and
 *  bound by both macros below. It empties the stored draws of every block in
 *  impl_ and then calls the wrapper's clear_extra_history_() hook. A wrapper
 *  that records per-sweep values in members of its own, outside impl_,
 *  defines that hook PUBLIC and empties those members in it; every other
 *  wrapper inherits the no-op default. Only the hook of the class named in
 *  kernel_control_mixin<...> is called (see clear_history() below).
 *================================================================================*/

#ifndef AI4BAYESCODE_KERNEL_CONTROL_MIXIN_HPP
#define AI4BAYESCODE_KERNEL_CONTROL_MIXIN_HPP

#include "composite_block.hpp"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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

    // ---- history reset (backend-neutral) --------------------------------
    // Discards every stored draw. The current state, the RNG streams and the
    // tuning quantities (NUTS step size and metric, slice widths, ...) are
    // left unchanged, so each sweep after the call records the same draw as
    // the same sweep of a run without the call. The next step() records the
    // first draw of the new history. ai4bayescode_run_chains() / run_chains()
    // call this after the warmup sweeps, so get_history() and predict_at()
    // cover the post-warmup draws only.
    //
    // clear_extra_history_() is a hook: a wrapper that records per-sweep
    // values in members of its own (outside impl_) defines a PUBLIC
    //     void clear_extra_history_() { <those members>.clear(); }
    // which hides the no-op default below. The call goes through
    // static_cast<Derived*>, so it calls the hook declared in Derived (the
    // class named in kernel_control_mixin<Derived>) or the default. If
    // Derived declares it private, the call below either does not compile
    // (name lookup finds the private member) or, in a wrapper that declares
    // this mixin a friend (as every shipped wrapper built on it does), calls
    // it; in neither case is Derived's hook skipped. A hook declared in a
    // class derived from Derived (class Sub : public Derived), public or
    // private, is NOT called. Such a class declares instead
    //     void clear_history() { Derived::clear_history(); <own members>.clear(); }
    // and the binding macros below call that function. The other CRTP
    // forwards of the mixin resolve in Derived the same way: the no-argument
    // predict_at calls Derived::predict_at.
    void clear_history() {
        auto self = static_cast<Derived*>(this);
        self->impl_->clear_history();
        self->clear_extra_history_();
    }

    /// Default hook: the wrapper records no per-sweep values outside impl_.
    void clear_extra_history_() {}

#ifdef AI4BAYESCODE_RCPP_MODULE
    // ---- R-layer forwarders (Rcpp-typed) -------------------------------

    // ---- list-wrapping misuse guards --------------------------------
    // The single most common R-side calling mistake is passing named
    // arguments where ONE list is expected:
    //     m$set_current(X = ..., y = ...)     instead of
    //     m$set_current(list(X = ..., y = ...))
    // Rcpp dispatches on arity, so the 2- and 3-argument calls land here
    // and get an actionable message instead of Rcpp's
    // "could not find valid method". Bound by
    // AI4BAYESCODE_BIND_KERNEL_CONTROL, so every wrapper inherits them.
    void set_current_misuse_(SEXP, SEXP) {
        Rcpp::stop("set_current takes ONE list argument -- wrap the values: "
                   "set_current(list(<name> = <value>, ...))");
    }
    void set_current_misuse3_(SEXP, SEXP, SEXP) {
        Rcpp::stop("set_current takes ONE list argument -- wrap the values: "
                   "set_current(list(<name> = <value>, ...))");
    }
    void predict_at_misuse_(SEXP, SEXP) {
        Rcpp::stop("predict_at takes ONE list argument -- wrap the values: "
                   "predict_at(list(<name> = <value>, ...))");
    }

    // ---- no-argument predict_at ---------------------------------------
    // m$predict_at() is equivalent to m$predict_at(list()) -- the
    // posterior-predictive at the training data. CRTP-forwards to the
    // wrapper's own predict_at(state_map). Bound by
    // AI4BAYESCODE_BIND_KERNEL_CONTROL.
    AI4BayesCode::history_map predict_at_noarg_() const {
        return static_cast<const Derived*>(this)->predict_at(
            AI4BayesCode::state_map{});
    }


    void freeze_names(const Rcpp::CharacterVector& names) {
        freeze_names_impl_(names, false);
    }

    void freeze_names_quiet(const Rcpp::CharacterVector& names, bool quiet) {
        freeze_names_impl_(names, quiet);
    }

    /// 1-arg form: unfreeze specific names OR NULL = unfreeze all.
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

    /// 0-arg form: unfreeze all. Rcpp modules IGNORE C++ default args, so
    /// the no-arg case needs its own explicit method; both overloads are
    /// bound as "unfreeze" via the RCPP macro.
    void unfreeze_names_all() {
        auto self = static_cast<Derived*>(this);
        self->impl_->unfreeze_all();
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
            // PyErr_WarnEx returns -1 when the warning filters turn the
            // warning into an exception (warnings.simplefilter("error"),
            // python -W error); the exception is then raised to the caller
            // and the remaining warnings are not issued.
            if (PyErr_WarnEx(PyExc_UserWarning, w.c_str(), 1) < 0)
                throw pybind11::error_already_set();
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
            if (PyErr_WarnEx(PyExc_UserWarning, w.c_str(), 1) < 0)
                throw pybind11::error_already_set();   // as in py_freeze
        }
    }

    std::vector<std::string> py_get_frozen() const {
        auto self = static_cast<const Derived*>(this);
        return self->impl_->get_frozen();
    }
#endif // AI4BAYESCODE_PYBIND_MODULE
};

namespace kcm_detail {

template <typename...> struct void_type_ { using type = void; };

/// True when w.clear_history() is a valid call on a W lvalue outside W: W
/// declares a public clear_history() callable with no argument, or W
/// declares none and inherits the mixin's.
template <typename W, typename = void>
struct no_arg_clear_history : std::false_type {};
template <typename W>
struct no_arg_clear_history<
    W, typename void_type_<decltype(std::declval<W&>().clear_history())>::type>
    : std::true_type {};

/// The kernel_control_mixin<D> base of a wrapper; D is deduced from W's
/// public base classes (D = W for a wrapper, D = Base for a class derived
/// from a wrapper Base).
template <typename D>
kernel_control_mixin<D>* mixin_base(kernel_control_mixin<D>* p) { return p; }

template <typename W, typename = void>
struct has_mixin_base : std::false_type {};
template <typename W>
struct has_mixin_base<
    W, typename void_type_<decltype(kcm_detail::mixin_base(std::declval<W*>()))>::type>
    : std::true_type {};

/**
 * clear_history() as bound by AI4BAYESCODE_BIND_KERNEL_CONTROL and
 * AI4BAYESCODE_PYBIND_KERNEL_CONTROL, a function of W* (Rcpp and pybind11
 * bind such a function as a method).
 *
 * call() calls W's own clear_history() when it is callable with no argument
 * (in a wrapper that declares none, that is the mixin's). When W declares a
 * clear_history() that is not (another parameter list, private), call()
 * calls the mixin's. Both calls name the function directly, so the binding
 * compiles for every wrapper and never calls a function through a member
 * pointer of another type. A class with neither (it does not derive from
 * kernel_control_mixin) is bound under name() = "clear_history_unavailable_"
 * instead: clear_history is then not a method of the class, and the R and
 * Python run_chains do not call it.
 */
template <typename W>
struct clear_history_binding {
    static const char* name() {
        return (no_arg_clear_history<W>::value || has_mixin_base<W>::value)
                   ? "clear_history" : "clear_history_unavailable_";
    }
    static void call(W* self) {
        dispatch_(self,
                  std::integral_constant<bool, no_arg_clear_history<W>::value>{},
                  std::integral_constant<bool, has_mixin_base<W>::value>{});
    }

private:
    template <typename HasMixin>
    static void dispatch_(W* self, std::true_type, HasMixin) {
        static_cast<void>(self->clear_history());
    }
    static void dispatch_(W* self, std::false_type, std::true_type) {
        kcm_detail::mixin_base(self)->clear_history();
    }
    static void dispatch_(W*, std::false_type, std::false_type) {
        throw std::logic_error(
            "clear_history() is not available: the class neither declares a "
            "clear_history() callable with no argument nor derives from "
            "AI4BayesCode::kernel_control_mixin");
    }
};

} // namespace kcm_detail

} // namespace AI4BayesCode

// ---- Rcpp module-binding macro ------------------------------------------
//
// Emits two overloads of `freeze` (1-arg and 2-arg) because Rcpp modules
// IGNORE C++ default arguments (same pattern as step() in codegen_cpp.md).
// Both `m$freeze(names)` and `m$freeze(names, quiet=TRUE)` then work
// from R.

#ifdef AI4BAYESCODE_RCPP_MODULE
// Every member-pointer entry converts the pointer with static_cast to a
// member-function type of CLASSNAME. Required because the mixin forwarders
// are inherited from the `kernel_control_mixin<CLASSNAME>` base; Rcpp's
// class_<T>::method template can't resolve a pointer-to-base-member against
// class_<Derived>. static_cast performs that base-to-derived conversion and
// selects the named overload; when CLASSNAME declares a member of the same
// name with another signature, the build stops with an error (a C-style cast
// would reinterpret the pointer, and the call would pass arguments the member
// does not take). clear_history is bound through
// kcm_detail::clear_history_binding (see there).
#define AI4BAYESCODE_BIND_KERNEL_CONTROL(CLASSNAME)                       \
    .method("freeze",                                                     \
            static_cast<void (CLASSNAME::*)(const Rcpp::CharacterVector&)>( \
                &CLASSNAME::freeze_names),                                \
            "Fix sub-kernel(s) at current value (kernel-control category)") \
    .method("freeze",                                                     \
            static_cast<void (CLASSNAME::*)(const Rcpp::CharacterVector&, bool)>( \
                &CLASSNAME::freeze_names_quiet),                          \
            "Fix sub-kernel(s); quiet=TRUE suppresses redundant-refreeze warning") \
    .method("unfreeze",                                                   \
            static_cast<void (CLASSNAME::*)()>(                           \
                &CLASSNAME::unfreeze_names_all),                          \
            "Release ALL frozen sub-kernels (no-arg form)")               \
    .method("unfreeze",                                                   \
            static_cast<void (CLASSNAME::*)(Rcpp::Nullable<Rcpp::CharacterVector>)>( \
                &CLASSNAME::unfreeze_names),                              \
            "Release specific sub-kernel(s); NULL = all")                 \
    .method("get_frozen",                                                 \
            static_cast<Rcpp::CharacterVector (CLASSNAME::*)() const>(    \
                &CLASSNAME::get_frozen_names),                            \
            "List currently-frozen block names (DFS pre-order)")          \
    .method(AI4BayesCode::kcm_detail::clear_history_binding<CLASSNAME>::name(), \
            &AI4BayesCode::kcm_detail::clear_history_binding<CLASSNAME>::call,   \
            "Discard the stored draws; the current state and tuning are " \
            "kept, and the next step() records the first draw of a new "  \
            "history")                                                    \
    .method("set_current",                                                \
            static_cast<void (CLASSNAME::*)(SEXP, SEXP)>(                 \
                &CLASSNAME::set_current_misuse_),                         \
            "Misuse guard: wrap the values in ONE list()")                \
    .method("set_current",                                                \
            static_cast<void (CLASSNAME::*)(SEXP, SEXP, SEXP)>(           \
                &CLASSNAME::set_current_misuse3_),                        \
            "Misuse guard: wrap the values in ONE list()")                \
    .method("predict_at",                                                 \
            static_cast<void (CLASSNAME::*)(SEXP, SEXP)>(                 \
                &CLASSNAME::predict_at_misuse_),                          \
            "Misuse guard: wrap the values in ONE list()")                \
    .method("predict_at",                                                 \
            static_cast<AI4BayesCode::history_map (CLASSNAME::*)() const>( \
                &CLASSNAME::predict_at_noarg_),                           \
            "No-argument form: predict_at() == predict_at(list())")
#endif // AI4BAYESCODE_RCPP_MODULE

// ---- pybind11 module-binding macro --------------------------------------
//
// pybind11 supports C++ default arguments natively, so the single-def
// covers both call shapes (`m.freeze(names)` and
// `m.freeze(names, quiet=True)`).
//
// USAGE: this macro emits `.def(...)` clauses INSIDE the pybind11::class_
// fluent chain (mirrors the RCPP macro). Callsite:
//
//     pybind11::class_<CLASSNAME>(m, "CLASSNAME")
//         .def(...)                             // existing methods
//         ...
//         .def("get_history", &CLASSNAME::get_history)   // no ; here
//         AI4BAYESCODE_PYBIND_KERNEL_CONTROL(CLASSNAME); // macro + ;
//
// The macro takes ONLY the class name (not the module var) because
// member-function pointers are bound via class_<T>::def(), not module::def().
// (A previous version incorrectly used `M.def(...)` on the module, which
// pybind11 rejects for pointer-to-member functions.)

#ifdef AI4BAYESCODE_PYBIND_MODULE
#define AI4BAYESCODE_PYBIND_KERNEL_CONTROL(CLASSNAME)                     \
    .def("freeze",                                                        \
         &CLASSNAME::py_freeze,                                           \
         pybind11::arg("names"),                                          \
         pybind11::arg("quiet") = false,                                  \
         "Fix sub-kernel(s) at current value; quiet=True suppresses "     \
         "redundant-refreeze warning")                                    \
    .def("unfreeze",                                                      \
         &CLASSNAME::py_unfreeze,                                         \
         pybind11::arg("names") = pybind11::none(),                       \
         "Release sub-kernel(s); no arg / None = all")                    \
    .def("get_frozen",                                                    \
         &CLASSNAME::py_get_frozen,                                       \
         "List currently-frozen block names (DFS pre-order)")             \
    .def(AI4BayesCode::kcm_detail::clear_history_binding<CLASSNAME>::name(), \
         &AI4BayesCode::kcm_detail::clear_history_binding<CLASSNAME>::call,   \
         "Discard the stored draws; the current state and tuning are "    \
         "kept, and the next step() records the first draw of a new "     \
         "history")
#endif // AI4BAYESCODE_PYBIND_MODULE

// ----------------------------------------------------------------------------
//  AI4BAYESCODE_BIND_READAPT_NUTS(CLASSNAME) -- bind readapt_NUTS, both arities
//
//  EVERY model with a nuts_block or joint_nuts_block child exposes
//  readapt_NUTS. No exception: a NUTS chain that cannot be re-adapted cannot be
//  used online or sequentially, and the caller has no way to discover that
//  except by the method being absent.
//
//  This is deliberately NOT folded into AI4BAYESCODE_BIND_KERNEL_CONTROL. That
//  macro is unconditional -- every wrapper inherits freeze / unfreeze /
//  get_frozen / clear_history from kernel_control_mixin -- whereas
//  readapt_NUTS exists only on models that have a NUTS kernel (16 of the 43
//  shipped examples), so folding it in would fail to compile for the other 27.
//
//  BOTH arities are bound, and that is the point of having a macro at all:
//  Rcpp modules dispatch on ARITY and ignore C++ default arguments, so binding
//  only (int, bool, int) leaves the documented 4-argument call unreachable,
//  and binding only the 4-argument form breaks every 3-argument call site.
//  Written by hand in each of the 16 examples, either arity could be left
//  out.
//
//  Usage, inside RCPP_MODULE, on a class defining both overloads:
//
//      .method("step", &MyModel::step)
//      AI4BAYESCODE_BIND_READAPT_NUTS(MyModel)
//      AI4BAYESCODE_BIND_KERNEL_CONTROL(MyModel)
// ----------------------------------------------------------------------------
#ifdef AI4BAYESCODE_RCPP_MODULE
//  Both entries select the overload with static_cast (see the comment above
//  AI4BAYESCODE_BIND_KERNEL_CONTROL): a class without an overload of exactly
//  that signature stops the build instead of binding another function.
#define AI4BAYESCODE_BIND_READAPT_NUTS(CLASSNAME)                          \
    .method("readapt_NUTS",                                                \
            static_cast<void (CLASSNAME::*)(int, bool, int)>(              \
                &CLASSNAME::readapt_NUTS),                                  \
            "Re-tune NUTS dual-averaging: readapt_NUTS(n, reset, "          \
            "max_tree_depth). Pass all three -- Rcpp dispatches on arity "  \
            "and does not honour the C++ default arguments.")               \
    .method("readapt_NUTS",                                                \
            static_cast<void (CLASSNAME::*)(int, bool, int, double)>(      \
                &CLASSNAME::readapt_NUTS),                                  \
            "Re-tune NUTS; 4th argument target_accept in (0, 1] overrides " \
            "the block's target (default 0.55) for this re-adaptation.")
#endif // AI4BAYESCODE_RCPP_MODULE

#ifdef AI4BAYESCODE_PYBIND_MODULE
#define AI4BAYESCODE_PYBIND_READAPT_NUTS(CLASSNAME)                        \
    .def("readapt_NUTS",                                                   \
         static_cast<void (CLASSNAME::*)(int, bool, int, double)>(         \
             &CLASSNAME::readapt_NUTS),                                     \
         pybind11::arg("n"), pybind11::arg("reset") = false,               \
         pybind11::arg("max_tree_depth") = -1,                             \
         pybind11::arg("target_accept") = -1.0,                            \
         "Re-tune NUTS dual-averaging. pybind11 honours the defaults, so "  \
         "readapt_NUTS(n) works here even though R requires all three.")
#endif // AI4BAYESCODE_PYBIND_MODULE


#endif // AI4BAYESCODE_KERNEL_CONTROL_MIXIN_HPP

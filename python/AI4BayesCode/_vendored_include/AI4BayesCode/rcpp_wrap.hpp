/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later).
 *================================================================================
 *
 *  rcpp_wrap.hpp  --  Rcpp::wrap specializations for the neutral types
 *                     declared in types.hpp.
 *
 *  When RCPP_MODULE compiles a class method that returns one of our
 *  neutral types (state_map, history_map, dag_info, adaptation_info),
 *  Rcpp needs to know how to convert it to an R SEXP. The built-in
 *  wrappers handle std::unordered_map<std::string, X> if X has a
 *  wrapper, and RcppArmadillo handles arma::vec/mat, so state_map
 *  and history_map auto-convert out of the box. The custom types
 *  (dag_info, adaptation_info) need specializations here.
 *
 *  Include this header from any .cpp that uses RCPP_MODULE and returns
 *  these types. Safe to include in a Python build — the specializations
 *  are behind #ifdef RcppArmadillo__RcppArmadillo__h, so if Rcpp isn't
 *  in the translation unit, nothing is emitted.
 *================================================================================*/

#ifndef AI4BAYESCODE_RCPP_WRAP_HPP
#define AI4BAYESCODE_RCPP_WRAP_HPP

#include "types.hpp"

#if defined(Rcpp_hpp) || defined(RcppArmadillo__RcppArmadillo__h) || defined(AI4BAYESCODE_RCPP_MODULE)

#include <RcppArmadillo.h>

namespace Rcpp {

// Explicit wrap() for state_map — name -> arma::vec. Produces an R
// named list of numeric vectors.
template<>
inline SEXP wrap(const AI4BayesCode::state_map& m) {
    Rcpp::List out;
    for (const auto& kv : m) out[kv.first] = Rcpp::wrap(kv.second);
    return out;
}

// Explicit wrap() for history_map — name -> arma::mat. Produces an R
// named list of numeric matrices. Scalar quantities (dim=1) are
// materialized as 1-column matrices; users who want them as vectors
// can call as.numeric(L$key) on the R side.
template<>
inline SEXP wrap(const AI4BayesCode::history_map& m) {
    Rcpp::List out;
    for (const auto& kv : m) out[kv.first] = Rcpp::wrap(kv.second);
    return out;
}

// as() for state_map from an R named list. Empty list is allowed (returns
// empty map); non-empty list without names errors.
template<>
inline AI4BayesCode::state_map as(SEXP x) {
    Rcpp::List L(x);
    AI4BayesCode::state_map m;
    if (L.size() == 0) return m;
    if (Rf_isNull(L.names())) {
        Rcpp::stop("state_map requires a NAMED list (got unnamed).");
    }
    Rcpp::CharacterVector nms = L.names();
    for (R_xlen_t i = 0; i < L.size(); ++i) {
        std::string k = Rcpp::as<std::string>(nms[i]);
        m[k] = Rcpp::as<arma::vec>(L[i]);
    }
    return m;
}

template<>
inline SEXP wrap(const AI4BayesCode::dag_info& d) {
    Rcpp::List gibbs_reads;
    for (const auto& [k, v] : d.gibbs_reads) {
        gibbs_reads[k] = Rcpp::wrap(v);
    }
    Rcpp::List gibbs_invalidates;
    for (const auto& [k, v] : d.gibbs_invalidates) {
        gibbs_invalidates[k] = Rcpp::wrap(v);
    }
    Rcpp::List predict_edges;
    for (const auto& [k, v] : d.predict_edges) {
        predict_edges[k] = Rcpp::wrap(v);
    }
    Rcpp::List context_edges;
    for (const auto& [k, v] : d.context_edges) {
        context_edges[k] = Rcpp::wrap(v);
    }
    return Rcpp::List::create(
        Rcpp::Named("gibbs_reads")       = gibbs_reads,
        Rcpp::Named("gibbs_invalidates") = gibbs_invalidates,
        Rcpp::Named("predict_edges")     = predict_edges,
        Rcpp::Named("context_edges")     = context_edges,
        Rcpp::Named("data_inputs")       = Rcpp::wrap(d.data_inputs)
    );
}

template<>
inline SEXP wrap(const AI4BayesCode::adaptation_info& a) {
    Rcpp::List out;
    out["step_size"]    = a.step_size;
    out["epsilon_bar"]  = a.epsilon_bar;
    out["h_val"]        = a.h_val;
    out["mu_val"]       = a.mu_val;
    out["adapt_iter"]   = (double)a.adapt_iter;
    out["metric_kind"]  = a.metric_kind;
    if (!a.precond_mat.is_empty()) {
        out["precond_mat"] = Rcpp::wrap(a.precond_mat);
    }
    if (!a.precond_diag.empty()) {
        out["precond_diag"] = Rcpp::wrap(a.precond_diag);
    }
    return out;
}

// as() specialization for adaptation_info (set_adaptation consumes this)
template<>
inline AI4BayesCode::adaptation_info as(SEXP x) {
    Rcpp::List L(x);
    AI4BayesCode::adaptation_info a;
    if (L.containsElementNamed("step_size"))    a.step_size   = Rcpp::as<double>(L["step_size"]);
    if (L.containsElementNamed("epsilon_bar"))  a.epsilon_bar = Rcpp::as<double>(L["epsilon_bar"]);
    if (L.containsElementNamed("h_val"))        a.h_val       = Rcpp::as<double>(L["h_val"]);
    if (L.containsElementNamed("mu_val"))       a.mu_val      = Rcpp::as<double>(L["mu_val"]);
    if (L.containsElementNamed("adapt_iter"))   a.adapt_iter  = (std::size_t)Rcpp::as<double>(L["adapt_iter"]);
    if (L.containsElementNamed("metric_kind"))  a.metric_kind = Rcpp::as<std::string>(L["metric_kind"]);
    if (L.containsElementNamed("precond_mat")) {
        a.precond_mat = Rcpp::as<arma::mat>(L["precond_mat"]);
    }
    if (L.containsElementNamed("precond_diag")) {
        a.precond_diag = Rcpp::as<std::vector<double>>(L["precond_diag"]);
    }
    return a;
}

} // namespace Rcpp

#endif // Rcpp is available

#endif // AI4BAYESCODE_RCPP_WRAP_HPP

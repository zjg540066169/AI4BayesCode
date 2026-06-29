/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later).
 *================================================================================
 *
 *  types.hpp  --  backend-neutral result types.
 *
 *  To support both R (Rcpp) and Python (pybind11), every user-facing
 *  accessor on block_sampler / composite_block returns one of these
 *  neutral types. Rcpp converts them to Rcpp::List via auto-wrap
 *  (std::unordered_map<std::string, X> -> named R list; arma::vec ->
 *  NumericVector via RcppArmadillo). pybind11 converts them via the
 *  casters in pybind_casters.hpp.
 *
 *  Design principle: NO backend-specific types appear in public
 *  method signatures below the example-layer. If you find yourself
 *  wanting Rcpp::List or py::dict in a library header, stop and
 *  express it using one of the types here first.
 *================================================================================*/

#ifndef AI4BAYESCODE_TYPES_HPP
#define AI4BAYESCODE_TYPES_HPP

#include <string>
#include <unordered_map>
#include <vector>

#ifndef MCMC_USE_RCPP_ARMADILLO
# include <armadillo>
#else
# include <RcppArmadillo.h>
#endif

namespace AI4BayesCode {

/**
 * Named-parameter state snapshot. Keys are parameter names, values are
 * column vectors; scalars are length-1. Used by:
 *   - block_context (input to set_context)
 *   - current_named_outputs (output of each block per step)
 *   - predict_at(new_data) input + output (see below)
 */
using state_map = std::unordered_map<std::string, arma::vec>;

/**
 * Named-matrix bundle. Keys are parameter or derived-key names; each
 * value is an arma::mat (n_draws x dim). Scalar quantities are stored
 * as n_draws x 1. Used by:
 *   - get_history() — one matrix per stored block
 *   - predict_at() outputs that are naturally matrix-shaped
 *     (e.g. y_rep, f_mean, K_matrix)
 *
 * Predict outputs that are scalars-per-observation still use this type
 * with n_cols=1 for consistency.
 */
using history_map = std::unordered_map<std::string, arma::mat>;

/**
 * Full DAG metadata returned by composite_block::get_dag(). Four
 * independent edge-layers + the set of replaceable data input keys.
 *
 *   gibbs_reads[block_name]        = keys the block reads when sampling
 *                                    (Gibbs DAG, who-reads-whom)
 *   gibbs_invalidates[block_name]  = derived keys the block refreshes
 *                                    after sampling (refresh DAG)
 *   predict_edges[producer_key]    = keys produced downstream (generative)
 *   data_inputs                    = keys that represent replaceable
 *                                    data inputs at predict_at(list(key=x))
 */
struct dag_info {
    std::unordered_map<std::string, std::vector<std::string>> gibbs_reads;
    std::unordered_map<std::string, std::vector<std::string>> gibbs_invalidates;
    std::unordered_map<std::string, std::vector<std::string>> predict_edges;
    std::vector<std::string> data_inputs;
    // VIZ-ONLY prior / hyperprior parents. Rendered faded by plot_dag
    // as the generative context around the (solid) predict sub-DAG;
    // never traversed by predict_at. See shared_data::declare_context_edges.
    std::unordered_map<std::string, std::vector<std::string>> context_edges;
};

/**
 * NUTS adaptation snapshot. Returned by nuts_block::get_adaptation()
 * and consumed by set_adaptation(). Opaque bundle that lets a user
 * save + restore mass-matrix + step-size + dual-averaging state
 * across compile sessions.
 *
 * Implementation note: keep this a POD so both Rcpp and pybind11 can
 * wrap it trivially. Matrix field is an arma::mat (possibly empty).
 */
struct adaptation_info {
    double step_size         = 0.0;
    double epsilon_bar       = 0.0;
    double h_val             = 0.0;
    double mu_val            = 0.0;
    std::size_t adapt_iter   = 0;
    arma::mat precond_mat;                    // n x n dense or diag
    std::vector<double> precond_diag;         // alternate diag-only form
    std::string metric_kind  = "identity";    // "identity" | "diagonal" | "dense"
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_TYPES_HPP

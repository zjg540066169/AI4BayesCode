#pragma once
/*================================================================================
 *  rcpp_predict_guard.hpp -- keep an R matrix's shape from being lost silently
 *
 *  predict_at takes a `state_map` = unordered_map<string, arma::vec>. Rcpp
 *  converts an R matrix to arma::vec by flattening it column-major, so by the
 *  time C++ sees `list(X = X_new)` the column count is GONE. All the wrapper
 *  can then check is that the length divides the training p, and that is a
 *  divisibility test, not a shape test:
 *
 *      p = 5, X_new is 100 x 3  ->  300 elements, 300 %% 5 == 0  ->  accepted,
 *      reshaped to 60 x 5, and 60 predictions come back from columns that have
 *      been re-cut across observation boundaries. No error, no warning.
 *
 *  It slips through whenever nrow*ncol happens to divide p, which is most of
 *  the time for the small, round p values models actually use.
 *
 *  This header runs BEFORE the conversion, while the SEXP still carries its
 *  dim attribute, so the column count can be compared to the model's p. Once
 *  the caller has flattened by hand there is nothing left to check -- any
 *  length divisible by p is a legitimate n_new -- and that stays the caller's
 *  responsibility.
 *
 *  Rcpp only. pybind11 refuses a 2-D array for an arma::vec argument outright
 *  (TypeError), so the Python side cannot reach this failure by accident.
 *
 *  Copyright (C) 2026 AI4BayesCode
 *  SPDX-License-Identifier: Apache-2.0
 *================================================================================*/

#ifdef AI4BAYESCODE_RCPP_MODULE

#include <Rcpp.h>

#include <cstddef>
#include <string>

#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/types.hpp"

namespace ai4b {

/**
 * Convert an R list into a state_map, checking the shape of the covariate
 * entry first.
 *
 * @param new_data       the R list the user passed to predict_at()
 * @param covariate_key  the key whose column count must match, e.g. "X"
 * @param p_train        columns the model was fitted with; 0 disables the check
 *
 * Only an entry that arrived as a MATRIX is checked -- a plain vector carries
 * no shape to verify, and rejecting it would break the documented
 * `list(X = as.vector(X_new))` form.
 */
inline AI4BayesCode::state_map predict_input(const Rcpp::List& new_data,
                                             const char*       covariate_key,
                                             std::size_t       p_train) {
    AI4BayesCode::state_map out;
    if (new_data.size() == 0) return out;

    const Rcpp::CharacterVector names = new_data.names();
    for (R_xlen_t i = 0; i < new_data.size(); ++i) {
        const std::string key = Rcpp::as<std::string>(names[i]);
        SEXP value            = new_data[i];

        if (p_train > 0 && key == covariate_key && Rf_isMatrix(value)) {
            const std::size_t n_col =
                static_cast<std::size_t>(Rf_ncols(value));
            if (n_col != p_train) {
                ai4b::stop(
                    "predict_at: '%s' has %d columns but the model was fitted "
                    "with p = %d. Pass the same predictors, in the same order, "
                    "that the model was built on.",
                    key.c_str(), static_cast<int>(n_col),
                    static_cast<int>(p_train));
            }
        }
        out[key] = Rcpp::as<arma::vec>(value);
    }
    return out;
}

}  // namespace ai4b

#endif  // AI4BAYESCODE_RCPP_MODULE

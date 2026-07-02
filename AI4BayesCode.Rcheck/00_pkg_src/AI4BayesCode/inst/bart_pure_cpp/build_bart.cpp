// build_bart.cpp -- Rcpp entry for standard BART (Chipman et al. 2010).
//
// Exports:
//   bart_MCMC(X, Y, ..., nburn, npost) -> List
//
// All other plug-in API methods (set_X, set_Y, update_step, predict, set_s,
// get_s, current_sigma, current_var_probs, current_var_counts, startdart) are
// available on the bart::bart_model class for direct C++ use.

// [[Rcpp::depends(RcppArmadillo, pg)]]
#include <RcppArmadillo.h>

#include "src/bart_model.h"

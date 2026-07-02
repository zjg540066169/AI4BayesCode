// build_softbart.cpp -- Rcpp entry for Soft BART (Linero & Yang 2018).
//
// Wraps the vendored SoftBart pkg sources in src/SOFTBART_VENDOR/.
// Exports:
//   softbart_MCMC(X, Y, X_test, ..., nburn, npost) -> List
//
// The plug-in C++ class softbart::softbart_model exposes set_X, set_Y,
// set_data, set_offset, set_s, get_s, update_step, predict, current_sigma,
// current_sigma_mu, current_var_probs, current_var_counts, startdart for
// direct C++ use.

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

#include "src/SOFTBART_VENDOR/soft_bart_impl.h"
#include "src/softbart_model.h"

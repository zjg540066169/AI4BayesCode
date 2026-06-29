// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_beta_gibbs_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::beta_gibbs_block (**Option A
// Check #15** from skills/codegen.md).
//
// Purpose
// -------
// Verify that beta_gibbs_block's sampling mechanism correctly draws from
// Beta(alpha, beta) when the user's params_fn returns fixed {alpha, beta}
// hyperparameters. This test covers the MECHANISM, not any specific
// params_fn derivation.
//
// Any example using beta_gibbs_block with a textbook params_fn (e.g.,
// {a_prior + sum(gamma), b_prior + p - sum(gamma)} for Beta-Bernoulli)
// inherits correctness from this test.
//
// Test design
// -----------
// 1. Build a beta_gibbs_block with params_fn returning fixed {5, 10}
//    regardless of context.
// 2. Draw 10,000 samples by repeatedly calling step().
// 3. Compare the sample mean/variance to the analytic Beta(5,10):
//      mean      = alpha / (alpha + beta)      = 5/15  ≈ 0.33333
//      variance  = ab / ((a+b)^2 (a+b+1))      = 50/(225 * 16) ≈ 0.01389
// 4. PASS iff |rel_err(mean)| < 5% AND |rel_err(var)| < 10% (Check #15
//    tolerance from codegen.md).
//
// Returned Rcpp::List so the caller (Rcpp::sourceCpp from an R audit
// script) can programmatically check all_pass.
// ============================================================================

// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <RcppArmadillo.h>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/beta_gibbs_block.hpp"

#include <cmath>
#include <random>
#include <vector>

using AI4BayesCode::beta_gibbs_block;
using AI4BayesCode::beta_gibbs_block_config;
using AI4BayesCode::beta_dist_params;
using AI4BayesCode::block_context;

// [[Rcpp::export]]
Rcpp::List test_beta_gibbs_block() {
    const double alpha_true = 5.0;
    const double beta_true  = 10.0;
    const std::size_t N_DRAWS = 10000;

    // ---- Build the block with fixed hyperparameters ---------------------
    beta_gibbs_block_config cfg;
    cfg.name          = "pi_test";
    cfg.initial_value = 0.5;
    cfg.params_fn     = [alpha_true, beta_true](const block_context& /*ctx*/) {
        return beta_dist_params{alpha_true, beta_true};
    };
    beta_gibbs_block blk(std::move(cfg));

    // Minimal empty context (params_fn ignores it).
    block_context ctx;
    blk.set_context(ctx);

    // ---- Draw samples ---------------------------------------------------
    std::mt19937_64 rng(12345);
    std::vector<double> samples;
    samples.reserve(N_DRAWS);
    for (std::size_t i = 0; i < N_DRAWS; ++i) {
        blk.step(rng);
        samples.push_back(blk.current()[0]);
    }

    // ---- Sample mean + variance ----------------------------------------
    double mean = 0.0;
    for (double v : samples) mean += v;
    mean /= static_cast<double>(N_DRAWS);

    double var = 0.0;
    for (double v : samples) {
        const double d = v - mean;
        var += d * d;
    }
    var /= static_cast<double>(N_DRAWS - 1);

    // ---- Analytic Beta(5, 10) ------------------------------------------
    const double exp_mean = alpha_true / (alpha_true + beta_true);
    const double exp_var  = alpha_true * beta_true /
        ((alpha_true + beta_true) * (alpha_true + beta_true) *
         (alpha_true + beta_true + 1.0));

    const double mean_rel_err = std::abs(mean - exp_mean) / exp_mean;
    const double var_rel_err  = std::abs(var  - exp_var)  / exp_var;

    // ---- Check #15 tolerances ------------------------------------------
    const double tol_mean = 0.05;   // 5%
    const double tol_var  = 0.10;   // 10%

    const bool pass_mean = (mean_rel_err < tol_mean);
    const bool pass_var  = (var_rel_err  < tol_var);
    const bool all_pass  = pass_mean && pass_var;

    // ---- Range check (sanity: 0 < samples < 1) -------------------------
    bool in_range = true;
    for (double v : samples) {
        if (!(v > 0.0 && v < 1.0)) { in_range = false; break; }
    }

    return Rcpp::List::create(
        Rcpp::Named("all_pass")     = all_pass && in_range,
        Rcpp::Named("pass_mean")    = pass_mean,
        Rcpp::Named("pass_var")     = pass_var,
        Rcpp::Named("in_range")     = in_range,
        Rcpp::Named("sample_mean")  = mean,
        Rcpp::Named("exp_mean")     = exp_mean,
        Rcpp::Named("mean_rel_err") = mean_rel_err,
        Rcpp::Named("sample_var")   = var,
        Rcpp::Named("exp_var")      = exp_var,
        Rcpp::Named("var_rel_err")  = var_rel_err,
        Rcpp::Named("n_draws")      = static_cast<int>(N_DRAWS),
        Rcpp::Named("tol_mean")     = tol_mean,
        Rcpp::Named("tol_var")      = tol_var);
}

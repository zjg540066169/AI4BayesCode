// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_piecewise_exponential_gibbs_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::piecewise_exponential_gibbs_block.
// Verifies BOTH the E_k / R_k accounting AND the Gamma(a0+E_k, b0+R_k) sample
// distribution in one shot, on a fixture with hand-computed sufficient stats.
//
// Fixture (n=4 subjects, K=3 bins on edges = {0, 1, 2, 5}):
//   Subject 1: t=0.5, delta=1        (event in bin 1)
//   Subject 2: t=1.5, delta=0        (censored in bin 2)
//   Subject 3: t=3.0, delta=1        (event in bin 3)
//   Subject 4: t=4.5, delta=0        (censored in bin 3)
//   No offset (baseline-only) => exp(f_i) = 1 for all i.
//   No entry-time (v_i = 0 for all i).
//
// Hand-computed sufficient stats (Delta_k(t_i, v_i=0) = max(0, min(t_i, e_k) - e_{k-1})):
//   E_1 = 1  (subject 1),        R_1 = 0.5 + 1.0 + 1.0 + 1.0 = 3.5
//   E_2 = 0,                     R_2 = 0.0 + 0.5 + 1.0 + 1.0 = 2.5
//   E_3 = 1  (subject 3),        R_3 = 0.0 + 0.0 + 1.0 + 2.5 = 3.5
//
// With prior a0=1, b0=1 => full conditionals:
//   lambda_1 ~ Gamma(2, 4.5),  E = 2/4.5    = 0.44444,  Var = 2/4.5^2   = 0.09876
//   lambda_2 ~ Gamma(1, 3.5),  E = 1/3.5    = 0.28571,  Var = 1/3.5^2   = 0.08163
//   lambda_3 ~ Gamma(2, 4.5),  E = 2/4.5    = 0.44444,  Var = 2/4.5^2   = 0.09876
//
// Draw N=20000 samples, compare per-bin mean & variance to analytic values.
// PASS iff |rel_err(mean)| < 5% AND |rel_err(var)| < 15% for every bin.
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
#include "AI4BayesCode/piecewise_exponential_gibbs_block.hpp"

#include <cmath>
#include <random>
#include <vector>

using AI4BayesCode::piecewise_exponential_gibbs_block;
using AI4BayesCode::piecewise_exponential_gibbs_block_config;
using AI4BayesCode::block_context;

// [[Rcpp::export]]
Rcpp::List test_piecewise_exponential_gibbs_block() {
    const std::size_t N_DRAWS = 20000;
    const double a0 = 1.0;
    const double b0 = 1.0;

    // ---- Build the block on the hand-computed fixture -------------------
    piecewise_exponential_gibbs_block_config cfg;
    cfg.name  = "lambda";
    cfg.edges = arma::vec({0.0, 1.0, 2.0, 5.0});
    cfg.a0    = a0;
    cfg.b0    = b0;
    // time_key="t", event_key="delta", no offset, no entry.
    piecewise_exponential_gibbs_block blk(std::move(cfg));

    // ---- Inject fixture data into the block context ---------------------
    block_context ctx;
    ctx["t"]     = arma::vec({0.5, 1.5, 3.0, 4.5});
    ctx["delta"] = arma::vec({1.0, 0.0, 1.0, 0.0});
    blk.set_context(ctx);

    // ---- Draw N_DRAWS samples of (lambda_1, lambda_2, lambda_3) ---------
    std::mt19937_64 rng(20260710);
    const std::size_t K = 3;
    std::vector<std::vector<double>> samples(K);
    for (std::size_t k = 0; k < K; ++k) samples[k].reserve(N_DRAWS);
    for (std::size_t it = 0; it < N_DRAWS; ++it) {
        blk.step(rng);
        const arma::vec& v = blk.current();
        for (std::size_t k = 0; k < K; ++k) samples[k].push_back(v[k]);
    }

    // ---- Analytic Gamma(a0 + E_k, b0 + R_k) mean + variance -------------
    // Ek, Rk computed by hand (see header comment).
    const double E[K] = {1.0, 0.0, 1.0};
    const double R[K] = {3.5, 2.5, 3.5};
    double exp_mean[K], exp_var[K], samp_mean[K], samp_var[K];
    double mean_rel_err[K], var_rel_err[K];
    bool pass_mean[K], pass_var[K];
    bool in_range = true;

    const double tol_mean = 0.05;   // 5%
    const double tol_var  = 0.15;   // 15% (variance estimator is noisier)

    bool all_pass = true;
    for (std::size_t k = 0; k < K; ++k) {
        const double shape_post = a0 + E[k];
        const double rate_post  = b0 + R[k];
        exp_mean[k] = shape_post / rate_post;
        exp_var[k]  = shape_post / (rate_post * rate_post);

        double m = 0.0;
        for (double v : samples[k]) m += v;
        m /= static_cast<double>(N_DRAWS);

        double s2 = 0.0;
        for (double v : samples[k]) {
            const double d = v - m;
            s2 += d * d;
        }
        s2 /= static_cast<double>(N_DRAWS - 1);

        samp_mean[k]    = m;
        samp_var[k]     = s2;
        mean_rel_err[k] = std::abs(m  - exp_mean[k]) / exp_mean[k];
        var_rel_err[k]  = std::abs(s2 - exp_var[k])  / exp_var[k];
        pass_mean[k]    = (mean_rel_err[k] < tol_mean);
        pass_var[k]     = (var_rel_err[k]  < tol_var);
        if (!pass_mean[k] || !pass_var[k]) all_pass = false;

        // Range sanity: lambda > 0 always.
        for (double v : samples[k]) {
            if (!(v > 0.0) || !std::isfinite(v)) { in_range = false; break; }
        }
        if (!in_range) break;
    }

    return Rcpp::List::create(
        Rcpp::Named("all_pass")     = all_pass && in_range,
        Rcpp::Named("pass_mean_k")  = Rcpp::LogicalVector::create(
                                          pass_mean[0], pass_mean[1], pass_mean[2]),
        Rcpp::Named("pass_var_k")   = Rcpp::LogicalVector::create(
                                          pass_var[0], pass_var[1], pass_var[2]),
        Rcpp::Named("in_range")     = in_range,
        Rcpp::Named("sample_mean")  = Rcpp::NumericVector::create(
                                          samp_mean[0], samp_mean[1], samp_mean[2]),
        Rcpp::Named("exp_mean")     = Rcpp::NumericVector::create(
                                          exp_mean[0], exp_mean[1], exp_mean[2]),
        Rcpp::Named("mean_rel_err") = Rcpp::NumericVector::create(
                                          mean_rel_err[0], mean_rel_err[1], mean_rel_err[2]),
        Rcpp::Named("sample_var")   = Rcpp::NumericVector::create(
                                          samp_var[0], samp_var[1], samp_var[2]),
        Rcpp::Named("exp_var")      = Rcpp::NumericVector::create(
                                          exp_var[0], exp_var[1], exp_var[2]),
        Rcpp::Named("var_rel_err")  = Rcpp::NumericVector::create(
                                          var_rel_err[0], var_rel_err[1], var_rel_err[2]),
        Rcpp::Named("n_draws")      = static_cast<int>(N_DRAWS),
        Rcpp::Named("tol_mean")     = tol_mean,
        Rcpp::Named("tol_var")      = tol_var);
}

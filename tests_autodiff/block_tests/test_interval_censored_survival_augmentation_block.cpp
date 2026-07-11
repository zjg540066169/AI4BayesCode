// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_interval_censored_survival_augmentation_block.cpp
//
// LIBRARY-LEVEL correctness test for
//   AI4BayesCode::interval_censored_survival_augmentation_block.
//
// Fixture: constant baseline hazard lambda_k = 1 for all k on edges
// (0, 1, 2, 3, 5), so H_0(t) = t and S(t) = exp(-t). Two subjects:
//   Subject 1: L=1, U=3          (INTERVAL-CENSORED, closed-form analytic mean/var)
//   Subject 2: L=1, U=+infinity  (RIGHT-CENSORED, T-1 ~ Exp(1))
// No covariates (offset = 1).
//
// Analytic truncated moments for subject 1 (T | 1 < T <= 3, T ~ Exp(1)):
//   E[T]  = ( 2*exp(-1) - 4*exp(-3) ) / ( exp(-1) - exp(-3) )   = 1.68690
//   Var[T] = ( 5*exp(-1) - 17*exp(-3) ) / ( exp(-1) - exp(-3) ) - E[T]^2
//          = 3.12167 - 2.84564                                   = 0.27603
// Analytic truncated moments for subject 2 (T | T > 1, T ~ Exp(1)):
//   E[T]  = 1 + 1 = 2         (memoryless: T-1 ~ Exp(1))
//   Var[T] = 1
//
// Draw N=30000 samples from a single call to block.step() repeatedly. Verify:
//   (1) Every draw satisfies L < T <= U for both subjects.
//   (2) Sample mean and variance match analytical to 5% (mean) / 15% (var).
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
#include "AI4BayesCode/interval_censored_survival_augmentation_block.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

using AI4BayesCode::interval_censored_survival_augmentation_block;
using AI4BayesCode::interval_censored_survival_augmentation_block_config;
using AI4BayesCode::block_context;

// [[Rcpp::export]]
Rcpp::List test_interval_censored_survival_augmentation_block() {
    const std::size_t N_DRAWS = 30000;
    const double inf = std::numeric_limits<double>::infinity();

    // ---- Build block ----------------------------------------------------
    interval_censored_survival_augmentation_block_config cfg;
    cfg.name          = "t_latent";
    // K = 4 bins; last edge = 30 keeps right-censored (T-1)~Exp(1) tail
    // clipping to well below 1e-12 (probability), so the sampler test
    // isolates the inverse-CDF mechanism from tail clipping.
    cfg.edges         = arma::vec({0.0, 1.0, 2.0, 3.0, 30.0});
    cfg.initial_times = arma::vec({2.0, 2.0});                    // in (1, 3] and (1, inf)
    interval_censored_survival_augmentation_block blk(std::move(cfg));

    // ---- Context: L, U, lambda ------------------------------------------
    block_context ctx;
    ctx["L"]      = arma::vec({1.0, 1.0});
    ctx["U"]      = arma::vec({3.0, inf});
    ctx["lambda"] = arma::vec({1.0, 1.0, 1.0, 1.0});   // constant hazard
    blk.set_context(ctx);

    // ---- Sample -----------------------------------------------------------
    std::mt19937_64 rng(20260710);
    std::vector<double> s1, s2;
    s1.reserve(N_DRAWS); s2.reserve(N_DRAWS);
    bool in_range = true;
    for (std::size_t it = 0; it < N_DRAWS; ++it) {
        blk.step(rng);
        const arma::vec& v = blk.current();
        s1.push_back(v[0]); s2.push_back(v[1]);
        if (!(v[0] > 1.0 && v[0] <= 3.0)) in_range = false;
        if (!(v[1] > 1.0) || !std::isfinite(v[1])) in_range = false;
    }

    // ---- Analytic ---------------------------------------------------------
    const double e1 = std::exp(-1.0), e3 = std::exp(-3.0);
    const double exp_mean_1 = (2.0*e1 - 4.0*e3) / (e1 - e3);
    const double exp_ET2_1  = (5.0*e1 - 17.0*e3) / (e1 - e3);
    const double exp_var_1  = exp_ET2_1 - exp_mean_1 * exp_mean_1;
    const double exp_mean_2 = 2.0;
    const double exp_var_2  = 1.0;

    auto mean_var = [](const std::vector<double>& s) {
        double m = 0.0; for (double v : s) m += v; m /= static_cast<double>(s.size());
        double v2 = 0.0; for (double v : s) { const double d = v - m; v2 += d*d; }
        v2 /= static_cast<double>(s.size() - 1);
        return std::pair<double,double>{m, v2};
    };
    const auto [samp_mean_1, samp_var_1] = mean_var(s1);
    const auto [samp_mean_2, samp_var_2] = mean_var(s2);

    const double m_re_1 = std::abs(samp_mean_1 - exp_mean_1) / exp_mean_1;
    const double v_re_1 = std::abs(samp_var_1  - exp_var_1)  / exp_var_1;
    const double m_re_2 = std::abs(samp_mean_2 - exp_mean_2) / exp_mean_2;
    const double v_re_2 = std::abs(samp_var_2  - exp_var_2)  / exp_var_2;

    const double tol_mean = 0.05, tol_var = 0.15;
    const bool pass_1 = (m_re_1 < tol_mean) && (v_re_1 < tol_var);
    const bool pass_2 = (m_re_2 < tol_mean) && (v_re_2 < tol_var);
    const bool all_pass = pass_1 && pass_2 && in_range;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")    = all_pass,
        Rcpp::Named("in_range")    = in_range,
        Rcpp::Named("pass_1")      = pass_1,
        Rcpp::Named("pass_2")      = pass_2,
        Rcpp::Named("samp_mean_1") = samp_mean_1,
        Rcpp::Named("exp_mean_1")  = exp_mean_1,
        Rcpp::Named("samp_var_1")  = samp_var_1,
        Rcpp::Named("exp_var_1")   = exp_var_1,
        Rcpp::Named("samp_mean_2") = samp_mean_2,
        Rcpp::Named("exp_mean_2")  = exp_mean_2,
        Rcpp::Named("samp_var_2")  = samp_var_2,
        Rcpp::Named("exp_var_2")   = exp_var_2,
        Rcpp::Named("n_draws")     = static_cast<int>(N_DRAWS),
        Rcpp::Named("tol_mean")    = tol_mean,
        Rcpp::Named("tol_var")     = tol_var);
}

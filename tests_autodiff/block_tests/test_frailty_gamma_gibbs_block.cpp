// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_frailty_gamma_gibbs_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::frailty_gamma_gibbs_block.
// Fixture: G=2 groups, n=4 subjects (2 per group), edges=(0,1,2,5), K=3,
// lambda=(1,1,1), theta=2, no offset, no entry-time.
//
// Subject 1 (group 0): t=0.5, delta=1   => H_0(0.5) = 0.5
// Subject 2 (group 0): t=1.5, delta=0   => H_0(1.5) = 1.5
// Subject 3 (group 1): t=3.0, delta=1   => H_0(3.0) = 3.0
// Subject 4 (group 1): t=4.5, delta=0   => H_0(4.5) = 4.5
//
// Hand-computed:
//   D_0 = 1,  H_0_grp = 0.5 + 1.5 = 2.0   => Gamma(3.0, 4.0)  E=0.75    Var=0.1875
//   D_1 = 1,  H_1_grp = 3.0 + 4.5 = 7.5   => Gamma(3.0, 9.5)  E=0.31579 Var=0.03324
//
// Draw N=20000 samples, check per-group mean & var match analytic within
// 5% (mean) / 15% (var).
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
#include "AI4BayesCode/frailty_gamma_gibbs_block.hpp"

#include <cmath>
#include <random>
#include <vector>

using AI4BayesCode::frailty_gamma_gibbs_block;
using AI4BayesCode::frailty_gamma_gibbs_block_config;
using AI4BayesCode::block_context;

// [[Rcpp::export]]
Rcpp::List test_frailty_gamma_gibbs_block() {
    const std::size_t N_DRAWS = 20000;
    const double theta = 2.0;

    frailty_gamma_gibbs_block_config cfg;
    cfg.name              = "w";
    cfg.G                 = 2;
    cfg.edges             = arma::vec({0.0, 1.0, 2.0, 5.0});   // K = 3
    cfg.theta             = theta;
    cfg.initial_frailties = arma::vec({1.0, 1.0});
    frailty_gamma_gibbs_block blk(std::move(cfg));

    // ---- Fixture context -------------------------------------------------
    block_context ctx;
    ctx["t"]      = arma::vec({0.5, 1.5, 3.0, 4.5});
    ctx["delta"]  = arma::vec({1.0, 0.0, 1.0, 0.0});
    ctx["z"]      = arma::vec({0.0, 0.0, 1.0, 1.0});
    ctx["lambda"] = arma::vec({1.0, 1.0, 1.0});
    blk.set_context(ctx);

    // ---- Draw samples ----------------------------------------------------
    std::mt19937_64 rng(20260710);
    const std::size_t G = 2;
    std::vector<std::vector<double>> s(G);
    for (std::size_t g = 0; g < G; ++g) s[g].reserve(N_DRAWS);
    for (std::size_t it = 0; it < N_DRAWS; ++it) {
        blk.step(rng);
        const arma::vec& v = blk.current();
        for (std::size_t g = 0; g < G; ++g) s[g].push_back(v[g]);
    }

    // ---- Analytic --------------------------------------------------------
    // D_g, H_g_grp hand-computed above.
    const double D[G]     = {1.0, 1.0};
    const double H_grp[G] = {2.0, 7.5};
    double exp_mean[G], exp_var[G], samp_mean[G], samp_var[G], mre[G], vre[G];
    bool pass_m[G], pass_v[G], in_range = true;
    const double tol_m = 0.05, tol_v = 0.15;
    bool all_pass = true;
    for (std::size_t g = 0; g < G; ++g) {
        const double shape_post = theta + D[g];
        const double rate_post  = theta + H_grp[g];
        exp_mean[g] = shape_post / rate_post;
        exp_var[g]  = shape_post / (rate_post * rate_post);

        double m = 0.0;
        for (double v : s[g]) m += v; m /= static_cast<double>(N_DRAWS);
        double v2 = 0.0;
        for (double v : s[g]) { const double d = v - m; v2 += d*d; }
        v2 /= static_cast<double>(N_DRAWS - 1);

        samp_mean[g] = m; samp_var[g] = v2;
        mre[g] = std::abs(m  - exp_mean[g]) / exp_mean[g];
        vre[g] = std::abs(v2 - exp_var[g])  / exp_var[g];
        pass_m[g] = (mre[g] < tol_m);
        pass_v[g] = (vre[g] < tol_v);
        if (!pass_m[g] || !pass_v[g]) all_pass = false;
        for (double v : s[g]) if (!(v > 0.0) || !std::isfinite(v)) { in_range = false; break; }
        if (!in_range) break;
    }

    return Rcpp::List::create(
        Rcpp::Named("all_pass")    = all_pass && in_range,
        Rcpp::Named("in_range")    = in_range,
        Rcpp::Named("pass_m")      = Rcpp::LogicalVector::create(pass_m[0], pass_m[1]),
        Rcpp::Named("pass_v")      = Rcpp::LogicalVector::create(pass_v[0], pass_v[1]),
        Rcpp::Named("samp_mean")   = Rcpp::NumericVector::create(samp_mean[0], samp_mean[1]),
        Rcpp::Named("exp_mean")    = Rcpp::NumericVector::create(exp_mean[0],  exp_mean[1]),
        Rcpp::Named("samp_var")    = Rcpp::NumericVector::create(samp_var[0],  samp_var[1]),
        Rcpp::Named("exp_var")     = Rcpp::NumericVector::create(exp_var[0],   exp_var[1]),
        Rcpp::Named("mean_relerr") = Rcpp::NumericVector::create(mre[0], mre[1]),
        Rcpp::Named("var_relerr")  = Rcpp::NumericVector::create(vre[0], vre[1]),
        Rcpp::Named("n_draws")     = static_cast<int>(N_DRAWS));
}

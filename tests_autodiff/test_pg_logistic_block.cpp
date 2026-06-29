// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_pg_logistic_block.cpp — T12 tests.
//
// (a) PG(1, z) truncated-series sampler matches analytical mean
//     E[ω] = (1/(2z)) tanh(z/2) across z ∈ {0.1, 1, 5} with 10k draws.
// (b) pg_logistic_block recovers synthetic logistic regression coefficients
//     on N=500, p=5 problem after 2000 sweeps.
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
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/pg_logistic_block.hpp"

#include <cmath>
#include <random>

using AI4BayesCode::pg_logistic_block;
using AI4BayesCode::pg_logistic_block_config;
using AI4BayesCode::block_context;
using AI4BayesCode::sample_pg_1_z;
using AI4BayesCode::pg_1_z_mean;

// [[Rcpp::export]]
Rcpp::List test_pg_logistic_block() {
    std::vector<std::string> lines;
    bool all_ok = true;
    auto record = [&](const std::string& n, bool ok, const std::string& m) {
        lines.push_back(n + ": " + (ok ? "PASS" : "FAIL") + " — " + m);
        if (!ok) all_ok = false;
    };

    // =======================================================================
    // (a) PG(1, z) sampler vs analytical mean.
    // =======================================================================
    std::mt19937_64 rng(42);
    const int N_DRAWS = 20000;
    for (double z : {0.1, 1.0, 5.0}) {
        double sum = 0.0;
        for (int i = 0; i < N_DRAWS; ++i) {
            sum += sample_pg_1_z(rng, z, 128);
        }
        const double sample_mean = sum / N_DRAWS;
        const double analytic = pg_1_z_mean(z);
        const double rel_err = std::abs(sample_mean - analytic) / analytic;
        // MC sd scales ~ analytic_var/sqrt(N) which is small; 2% tol.
        bool ok = rel_err < 0.03;
        record("PG(1, z=" + std::to_string(z) + ") sampler mean", ok,
               "sample=" + std::to_string(sample_mean) +
               " analytic=" + std::to_string(analytic) +
               " rel_err=" + std::to_string(rel_err));
    }

    // =======================================================================
    // (b) pg_logistic_block recovers synthetic logistic regression coefs.
    // =======================================================================
    const std::size_t N = 500, p = 5;
    arma::vec beta_true = {1.5, -0.8, 0.5, 0.0, -1.2};

    // Generate X ~ N(0, 1), y ~ Bernoulli(sigmoid(X beta_true)).
    std::mt19937_64 data_rng(20260420);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    arma::mat X(N, p);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < p; ++j)
            X(i, j) = nd(data_rng);
    for (std::size_t i = 0; i < N; ++i) {
        double logit = arma::dot(X.row(i).t(), beta_true);
        double prob = 1.0 / (1.0 + std::exp(-logit));
        y[i] = (ud(data_rng) < prob) ? 1.0 : 0.0;
    }

    pg_logistic_block_config cfg;
    cfg.name = "beta";
    cfg.p = p;
    cfg.y_key = "y";
    cfg.X_key = "X";
    cfg.prior_mean = arma::vec(p, arma::fill::zeros);
    cfg.prior_cov = 100.0 * arma::eye<arma::mat>(p, p);   // weakly informative
    cfg.initial_beta = arma::vec(p, arma::fill::zeros);
    cfg.n_pg_terms = 128;

    pg_logistic_block blk(std::move(cfg));
    block_context ctx;
    ctx["y"] = y;
    ctx["X"] = arma::vectorise(X);
    blk.set_context(ctx);

    // Run 2000 sweeps (first 500 as burn-in).
    std::mt19937_64 mcmc_rng(123);
    arma::mat beta_samples(1500, p);
    for (std::size_t s = 0; s < 500; ++s) blk.step(mcmc_rng);
    for (std::size_t s = 0; s < 1500; ++s) {
        blk.step(mcmc_rng);
        beta_samples.row(s) = blk.current().t();
    }
    arma::vec beta_pm = arma::mean(beta_samples, 0).t();
    arma::vec beta_sd = arma::stddev(beta_samples, 0).t();

    bool ok_recovery = true;
    for (std::size_t j = 0; j < p; ++j) {
        // Recovery: posterior mean within 3 * posterior_sd of truth.
        bool within = std::abs(beta_pm[j] - beta_true[j]) < 3.0 * beta_sd[j];
        if (!within) ok_recovery = false;
    }
    std::string msg = "beta_pm=[";
    for (std::size_t j = 0; j < p; ++j) {
        msg += std::to_string(beta_pm[j]);
        if (j + 1 < p) msg += ",";
    }
    msg += "] true=[";
    for (std::size_t j = 0; j < p; ++j) {
        msg += std::to_string(beta_true[j]);
        if (j + 1 < p) msg += ",";
    }
    msg += "]";
    record("pg_logistic_block beta recovery (N=500, p=5)", ok_recovery, msg);

    Rcpp::CharacterVector details(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) details[i] = lines[i];
    return Rcpp::List::create(
        Rcpp::Named("all_pass") = all_ok,
        Rcpp::Named("details")  = details);
}

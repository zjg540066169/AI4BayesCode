// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_hmm_block.cpp — T10 tests.
//
// (a) Forward filter on a minimal K=2 HMM: sample marginal p(z_t | y) from
//     10k FFBS draws, compare to analytical Baum-Welch forward-backward
//     smoothing probabilities computed directly. Relative error on per-t,
//     per-k marginal should be within MC noise (2%).
//
// (b) End-to-end: with fixed A / pi / emission params, chain stationary
//     distribution on (z_1, z_T) matches smoothing marginals.
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
#include "AI4BayesCode/hmm_block.hpp"

#include <cmath>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::hmm_block;
using AI4BayesCode::hmm_block_config;

namespace {

// Fixture: K=2 HMM.
//   pi     = [0.6, 0.4]
//   A      = [[0.8, 0.2], [0.3, 0.7]]       (row-major, row i = from state i)
//   emission: z=0 produces y ~ N(0, 1); z=1 produces y ~ N(2, 1)
// T=5 fixed y sequence chosen by seed.
struct HMMFixture {
    std::size_t T = 5;
    std::size_t K = 2;
    arma::vec A_flat;
    arma::vec pi;
    arma::vec y;
    arma::vec mu{0.0, 2.0};
    double sigma = 1.0;
    HMMFixture() {
        A_flat = {0.8, 0.2, 0.3, 0.7};   // row-major [[0.8,0.2],[0.3,0.7]]
        pi     = {0.6, 0.4};
        // Synthesize y using true z sequence for repeatability.
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        std::normal_distribution<double> nd(0.0, sigma);
        std::vector<std::size_t> z_true(T);
        // Draw z_true
        z_true[0] = (ud(rng) < pi[0]) ? 0 : 1;
        for (std::size_t t = 1; t < T; ++t) {
            const std::size_t prev = z_true[t - 1];
            z_true[t] = (ud(rng) < A_flat[prev * K]) ? 0 : 1;
        }
        y.set_size(T);
        for (std::size_t t = 0; t < T; ++t) {
            y[t] = mu[z_true[t]] + nd(rng);
        }
    }
    double emission_logp(std::size_t t, std::size_t k) const {
        const double r = y[t] - mu[k];
        return -0.5 * std::log(2.0 * M_PI) - std::log(sigma)
               - 0.5 * (r * r) / (sigma * sigma);
    }
};

HMMFixture fx;

double emission_fn(std::size_t t, std::size_t k,
                   const block_context& /*ctx*/) {
    return fx.emission_logp(t, k);
}

// Analytical Baum-Welch forward-backward smoothing. Returns
// smoothing_marginals[t, k] = p(z_t = k | y_1:T).
arma::mat analytical_smoothing_marginals() {
    const std::size_t T = fx.T, K = fx.K;
    arma::mat alpha(T, K); alpha.zeros();
    arma::mat beta(T, K);  beta.zeros();

    // Forward in log-space (stable).
    auto lse2 = [](double a, double b) {
        if (!std::isfinite(a)) return b;
        if (!std::isfinite(b)) return a;
        double m = std::max(a, b);
        return m + std::log(std::exp(a - m) + std::exp(b - m));
    };

    arma::mat log_alpha(T, K);
    // t = 0
    for (std::size_t k = 0; k < K; ++k)
        log_alpha(0, k) = std::log(fx.pi[k]) + fx.emission_logp(0, k);
    // t = 1..T-1
    for (std::size_t t = 1; t < T; ++t) {
        for (std::size_t k = 0; k < K; ++k) {
            double s = -std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < K; ++j) {
                s = lse2(s, log_alpha(t - 1, j) +
                           std::log(fx.A_flat[j * K + k]));
            }
            log_alpha(t, k) = s + fx.emission_logp(t, k);
        }
    }

    // Backward in log-space.
    arma::mat log_beta(T, K);
    for (std::size_t k = 0; k < K; ++k) log_beta(T - 1, k) = 0.0;
    for (std::size_t t_rev = 1; t_rev < T; ++t_rev) {
        const std::size_t t = T - 1 - t_rev;
        for (std::size_t k = 0; k < K; ++k) {
            double s = -std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < K; ++j) {
                s = lse2(s, std::log(fx.A_flat[k * K + j]) +
                           fx.emission_logp(t + 1, j) +
                           log_beta(t + 1, j));
            }
            log_beta(t, k) = s;
        }
    }

    // Smoothing marginals p(z_t = k | y_1:T) ∝ alpha(t,k) * beta(t,k)
    arma::mat smoothing(T, K);
    for (std::size_t t = 0; t < T; ++t) {
        double norm = -std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < K; ++k) {
            const double lw = log_alpha(t, k) + log_beta(t, k);
            norm = lse2(norm, lw);
        }
        for (std::size_t k = 0; k < K; ++k) {
            smoothing(t, k) = std::exp(
                log_alpha(t, k) + log_beta(t, k) - norm);
        }
    }
    return smoothing;
}

} // anonymous namespace

// [[Rcpp::export]]
Rcpp::List test_hmm_block() {
    std::vector<std::string> lines;
    bool all_ok = true;
    auto record = [&](const std::string& n, bool ok, const std::string& m) {
        lines.push_back(n + ": " + (ok ? "PASS" : "FAIL") + " — " + m);
        if (!ok) all_ok = false;
    };

    // Analytical smoothing marginals (ground truth).
    arma::mat true_marg = analytical_smoothing_marginals();

    // Build block.
    hmm_block_config cfg;
    cfg.name = "z";
    cfg.T = fx.T;
    cfg.K = fx.K;
    cfg.A_key = "A";
    cfg.pi_key = "pi";
    cfg.emission_logp = &emission_fn;
    cfg.initial_z = arma::vec(fx.T, arma::fill::zeros);
    hmm_block blk(std::move(cfg));

    block_context ctx;
    ctx["A"]  = fx.A_flat;
    ctx["pi"] = fx.pi;
    blk.set_context(ctx);

    // Run 10k sweeps, count state occupancy per t.
    std::mt19937_64 rng(42);
    const int N_SWEEPS = 10000;
    arma::mat count(fx.T, fx.K, arma::fill::zeros);
    for (int s = 0; s < N_SWEEPS; ++s) {
        blk.step(rng);
        const arma::vec& z = blk.current();
        for (std::size_t t = 0; t < fx.T; ++t) {
            const std::size_t k = static_cast<std::size_t>(z[t]);
            count(t, k) += 1.0;
        }
    }
    arma::mat emp_marg = count / static_cast<double>(N_SWEEPS);

    // Per (t, k) compare empirical to analytical.
    double max_abs_err = 0.0;
    for (std::size_t t = 0; t < fx.T; ++t) {
        for (std::size_t k = 0; k < fx.K; ++k) {
            const double e = std::abs(emp_marg(t, k) - true_marg(t, k));
            if (e > max_abs_err) max_abs_err = e;
        }
    }
    // MC sd on a proportion estimate from 10k draws is ~ 0.005, so 2%
    // tolerance covers MC noise comfortably.
    bool ok = max_abs_err < 0.02;
    std::string details_str = "max_abs_err=" + std::to_string(max_abs_err);
    details_str += " emp=[";
    for (std::size_t t = 0; t < fx.T; ++t) {
        details_str += "[";
        for (std::size_t k = 0; k < fx.K; ++k) {
            details_str += std::to_string(emp_marg(t, k));
            if (k + 1 < fx.K) details_str += ",";
        }
        details_str += "]";
    }
    details_str += "]";
    record("FFBS marginals vs Baum-Welch analytical", ok, details_str);

    // Sanity: all z values in {0, 1}.
    bool all_valid = true;
    blk.step(rng);
    const arma::vec& z = blk.current();
    for (std::size_t t = 0; t < fx.T && all_valid; ++t) {
        if (!(z[t] == 0.0 || z[t] == 1.0)) all_valid = false;
    }
    record("z values in {0, 1}", all_valid, "sanity");

    Rcpp::CharacterVector details(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) details[i] = lines[i];
    return Rcpp::List::create(
        Rcpp::Named("all_pass") = all_ok,
        Rcpp::Named("details")  = details);
}

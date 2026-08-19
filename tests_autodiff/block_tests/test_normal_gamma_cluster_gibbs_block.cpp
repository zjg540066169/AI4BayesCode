// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_normal_gamma_cluster_gibbs_block.cpp
//
// LIBRARY-LEVEL parity test for
// AI4BayesCode::normal_gamma_cluster_gibbs_block.
//
// WHAT IS VERIFIED
// ----------------
// The block's conditional IS available in closed form, so this test uses the
// strongest option available: it compares the sampled distribution against
// the exact analytic conditional. No invariance argument, no exact
// enumeration and no detailed-balance identity is needed here -- those are
// fallbacks for kernels whose target has no closed form (collapsed samplers,
// split-merge moves), which is not the case for a conjugate Normal-Gamma
// leaf.
//
// Given a FIXED context (cluster labels z and data y), one call to step()
// draws, independently for every cluster k and every dimension j,
//
//     lambda_kj ~ Gamma(shape = a_n, rate = b_n)
//     mu_kj | lambda_kj ~ Normal(mu_n, 1 / (kappa_n * lambda_kj))
//
// i.e. a NormalGamma(mu_n, kappa_n, a_n, b_n) draw. Because the context is
// held fixed, successive step() calls are i.i.d. draws from that law, which
// makes every tolerance below a plain Monte Carlo standard error.
//
// Three groups of checks, all against quantities written down in advance
// from Murphy 2007 "Conjugate Bayesian analysis of the Gaussian
// distribution" (equations 4-9), NOT from what the code emits:
//
//   A. lambda marginal.  lambda ~ Gamma(a_n, rate b_n):
//        E[lambda]   = a_n / b_n
//        Var[lambda] = a_n / b_n^2
//
//   B. mu marginal.  Integrating lambda out gives a Student-t with
//      nu = 2 * a_n degrees of freedom centred at mu_n:
//        E[mu]   = mu_n                          (needs a_n > 1/2)
//        Var[mu] = b_n / (kappa_n * (a_n - 1))   (needs a_n > 1)
//      Plus the JOINT coupling, which the two marginals alone cannot see:
//      the standardized residual
//        s_kj = (mu_kj - mu_n) * sqrt(kappa_n * lambda_kj)
//      must be exactly Normal(0, 1). This catches a block that pairs mu
//      with the wrong lambda (wrong index, or the previous sweep's value)
//      while still producing correct marginals.
//
//   C. Conditional with lambda held fixed.  Freezing the lambda sub-key
//      (interface.md Sec.1) turns the mu update into an exactly Gaussian
//      conditional, Normal(mu_n, 1 / (kappa_n * lambda_fixed)), with
//      lambda_fixed the value the block was initialized at. This is a
//      second closed-form target that pins down the scale factor
//      1 / sqrt(kappa_n * lambda) directly, and simultaneously checks the
//      freeze contract (lambda must not move).
//
// FIXTURE
// -------
// K_trunc = 3, d = 2, N = 6, cluster-major flat index idx = k * d + j.
//   prior: mu_0 = (1, -1), kappa_0 = 2, a_lambda_0 = 5, b_lambda_0 = 2
//   z     = (1, 1, 1, 1, 2, 2)   -> n_1 = 4, n_2 = 2, n_3 = 0
//   y_i   = (1,-1), (2,0), (3,2), (6,7), (10,-5), (12,-7)
//
// Cluster 3 is deliberately EMPTY so the prior-draw branch of step() is
// exercised alongside the two data-driven branches. Hand-computed targets
// (the test recomputes these from the raw y/z, by a different numerical
// route than the block's sum(y^2) - n*ybar^2 shortcut):
//
//   k=1: kappa_n = 6, a_n = 7
//        j=0: ybar =  3, s2 = 14, b_n = 35/3    , mu_n =  7/3
//        j=1: ybar =  2, s2 = 38, b_n = 27      , mu_n =  1
//   k=2: kappa_n = 4, a_n = 6
//        j=0: ybar = 11, s2 =  2, b_n = 53      , mu_n =  6
//        j=1: ybar = -6, s2 =  2, b_n = 15.5    , mu_n = -3.5
//   k=3: kappa_n = 2, a_n = 5, b_n = 2, mu_n = mu_0_j   (prior branch)
//
// All twelve (k, j, parameter) targets are distinct, so an ordering or
// index bug in the K*d flat layout cannot pass.
//
// TOLERANCES
// ----------
// Every threshold is Z_TOL analytic standard errors of the estimator, with
// N_DRAWS = 40000 i.i.d. draws:
//   mean of X:      SE = sqrt(Var[X] / N)
//   variance of X:  SE = Var[X] * sqrt((kurt[X] - 1) / N), the delta-method
//                   standard error of the sample variance, where
//                   kurt = E[(X-EX)^4] / Var[X]^2.
//     Gamma(a):     kurt - 1 = 2 + 6 / a
//     Student-t(nu):kurt - 1 = 3 * (nu - 2) / (nu - 4) - 1   (needs nu > 4;
//                   the smallest here is nu = 2 * 5 = 10)
//     Normal:       kurt - 1 = 2
// Z_TOL = 4.5 is chosen for the 48 checks this file makes: a two-sided
// 4.5-sigma event has probability 6.8e-6, so the family-wise false-alarm
// rate is about 48 * 6.8e-6 = 3.3e-4. The thresholds were fixed before the
// test was run and were not adjusted afterwards.
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
#include "AI4BayesCode/normal_gamma_cluster_gibbs_block.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

using AI4BayesCode::normal_gamma_cluster_gibbs_block;
using AI4BayesCode::normal_gamma_cluster_gibbs_block_config;
using AI4BayesCode::block_context;

namespace {

// Sample mean of a vector.
double vec_mean(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m += x;
    return m / static_cast<double>(v.size());
}

// Unbiased sample variance about the sample mean.
double vec_var(const std::vector<double>& v, double m) {
    double s = 0.0;
    for (double x : v) { const double e = x - m; s += e * e; }
    return s / static_cast<double>(v.size() - 1);
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List test_normal_gamma_cluster_gibbs_block() {
    const std::size_t K       = 3;
    const std::size_t d       = 2;
    const std::size_t N       = 6;
    const std::size_t KD      = K * d;
    const std::size_t N_DRAWS = 40000;
    const double      Z_TOL   = 4.5;   // see TOLERANCES in the header comment

    // ---- Prior hyperparameters ------------------------------------------
    const double    kappa_0 = 2.0;
    const double    a_0     = 5.0;   // Gamma SHAPE on lambda
    const double    b_0     = 2.0;   // Gamma RATE  on lambda
    const arma::vec mu_0    = arma::vec({1.0, -1.0});

    // ---- Fixture data (y stored row-major, y[i * d + j]) -----------------
    const arma::vec z = arma::vec({1.0, 1.0, 1.0, 1.0, 2.0, 2.0});
    const arma::vec y = arma::vec({  1.0, -1.0,
                                     2.0,  0.0,
                                     3.0,  2.0,
                                     6.0,  7.0,
                                    10.0, -5.0,
                                    12.0, -7.0 });

    // ---- Analytic Normal-Gamma posterior, recomputed from raw y/z --------
    // s2 is accumulated as sum of squared deviations about ybar, which is a
    // different numerical route than the block's sum(y^2) - n * ybar^2.
    std::vector<double> mu_n(KD), kap_n(KD), a_n(KD), b_n(KD);
    for (std::size_t k = 0; k < K; ++k) {
        std::vector<std::size_t> members;
        for (std::size_t i = 0; i < N; ++i) {
            if (static_cast<std::size_t>(std::llround(z[i])) == k + 1)
                members.push_back(i);
        }
        const double nk = static_cast<double>(members.size());
        for (std::size_t j = 0; j < d; ++j) {
            const std::size_t idx = k * d + j;
            if (members.empty()) {
                // Empty cluster -> prior.
                mu_n[idx]  = mu_0[j];
                kap_n[idx] = kappa_0;
                a_n[idx]   = a_0;
                b_n[idx]   = b_0;
            } else {
                double ybar = 0.0;
                for (std::size_t i : members) ybar += y[i * d + j];
                ybar /= nk;
                double s2 = 0.0;
                for (std::size_t i : members) {
                    const double e = y[i * d + j] - ybar;
                    s2 += e * e;
                }
                const double dev = ybar - mu_0[j];
                kap_n[idx] = kappa_0 + nk;
                a_n[idx]   = a_0 + 0.5 * nk;
                b_n[idx]   = b_0 + 0.5 * s2
                           + 0.5 * kappa_0 * nk / kap_n[idx] * dev * dev;
                mu_n[idx]  = (kappa_0 * mu_0[j] + nk * ybar) / kap_n[idx];
            }
        }
    }

    // ---- Block under test -------------------------------------------------
    // initial_lambda doubles as the held value for the group-C frozen run,
    // so give every flat index a distinct positive value.
    const arma::vec init_mu     = arma::vec({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    const arma::vec init_lambda = arma::vec({0.5, 1.0, 1.5, 2.0, 2.5, 3.0});

    auto make_cfg = [&]() {
        normal_gamma_cluster_gibbs_block_config cfg;
        cfg.name           = "emit";
        cfg.K_trunc        = K;
        cfg.d              = d;
        cfg.N              = N;
        cfg.z_key          = "z";
        cfg.y_key          = "y";
        cfg.mu_name        = "mu";
        cfg.lambda_name    = "lambda";
        cfg.mu_0           = mu_0;
        cfg.kappa_0        = kappa_0;
        cfg.a_lambda_0     = a_0;
        cfg.b_lambda_0     = b_0;
        cfg.initial_mu     = init_mu;
        cfg.initial_lambda = init_lambda;
        return cfg;
    };

    block_context ctx;
    ctx["z"] = z;
    ctx["y"] = y;

    normal_gamma_cluster_gibbs_block blk(make_cfg());
    blk.set_context(ctx);

    // Structural contract: current() is [mu; lambda] of length 2 * K * d,
    // and the two named outputs are the two halves.
    bool layout_ok = (blk.dim() == 2 * KD) && (blk.current().n_elem == 2 * KD);
    {
        auto named = blk.current_named_outputs();
        layout_ok = layout_ok && (named.count("mu") == 1)
                              && (named.count("lambda") == 1)
                              && (named.at("mu").n_elem == KD)
                              && (named.at("lambda").n_elem == KD);
        if (layout_ok) {
            for (std::size_t i = 0; i < KD; ++i) {
                if (blk.current()[i] != named.at("mu")[i] ||
                    blk.current()[KD + i] != named.at("lambda")[i]) {
                    layout_ok = false;
                    break;
                }
            }
        }
    }

    // ---- Group A/B: unconstrained draws ----------------------------------
    std::mt19937_64 rng(20260818);
    std::vector<std::vector<double>> mu_s(KD), lam_s(KD);
    for (std::size_t i = 0; i < KD; ++i) {
        mu_s[i].reserve(N_DRAWS);
        lam_s[i].reserve(N_DRAWS);
    }
    for (std::size_t t = 0; t < N_DRAWS; ++t) {
        blk.step(rng);
        const arma::vec& cur = blk.current();
        for (std::size_t i = 0; i < KD; ++i) {
            mu_s[i].push_back(cur[i]);
            lam_s[i].push_back(cur[KD + i]);
        }
    }

    bool in_range = true;
    for (std::size_t i = 0; i < KD && in_range; ++i) {
        for (std::size_t t = 0; t < N_DRAWS; ++t) {
            if (!(lam_s[i][t] > 0.0) || !std::isfinite(lam_s[i][t]) ||
                !std::isfinite(mu_s[i][t])) { in_range = false; break; }
        }
    }

    const double nd = static_cast<double>(N_DRAWS);
    Rcpp::NumericVector z_lam_mean(KD), z_lam_var(KD);
    Rcpp::NumericVector z_mu_mean(KD),  z_mu_var(KD);
    Rcpp::NumericVector z_std_mean(KD), z_std_var(KD);
    Rcpp::NumericVector emp_lam_mean(KD), exp_lam_mean(KD);
    Rcpp::NumericVector emp_mu_mean(KD),  exp_mu_mean(KD);
    Rcpp::NumericVector emp_lam_var(KD),  exp_lam_var(KD);
    Rcpp::NumericVector emp_mu_var(KD),   exp_mu_var(KD);

    double worst_abs_z = 0.0;
    for (std::size_t i = 0; i < KD; ++i) {
        // --- A. lambda ~ Gamma(a_n, rate b_n) ---
        const double t_lam_mean = a_n[i] / b_n[i];
        const double t_lam_var  = a_n[i] / (b_n[i] * b_n[i]);
        // Gamma(a) excess kurtosis 6/a  ->  kurt - 1 = 2 + 6/a.
        const double lam_k1     = 2.0 + 6.0 / a_n[i];

        const double m_lam = vec_mean(lam_s[i]);
        const double v_lam = vec_var(lam_s[i], m_lam);
        z_lam_mean[i] = (m_lam - t_lam_mean) / std::sqrt(t_lam_var / nd);
        z_lam_var[i]  = (v_lam - t_lam_var) /
                        (t_lam_var * std::sqrt(lam_k1 / nd));

        // --- B. mu marginal ~ t_{2 a_n}(mu_n, b_n / (a_n kappa_n)) ---
        const double t_mu_mean = mu_n[i];
        const double t_mu_var  = b_n[i] / (kap_n[i] * (a_n[i] - 1.0));
        const double nu        = 2.0 * a_n[i];
        // Student-t(nu) kurtosis 3 (nu-2)/(nu-4)  ->  kurt - 1 below.
        const double mu_k1     = 3.0 * (nu - 2.0) / (nu - 4.0) - 1.0;

        const double m_mu = vec_mean(mu_s[i]);
        const double v_mu = vec_var(mu_s[i], m_mu);
        z_mu_mean[i] = (m_mu - t_mu_mean) / std::sqrt(t_mu_var / nd);
        z_mu_var[i]  = (v_mu - t_mu_var) /
                       (t_mu_var * std::sqrt(mu_k1 / nd));

        // --- B (joint coupling). (mu - mu_n) sqrt(kappa_n lambda) ~ N(0,1) ---
        std::vector<double> s(N_DRAWS);
        for (std::size_t t = 0; t < N_DRAWS; ++t) {
            s[t] = (mu_s[i][t] - mu_n[i]) *
                   std::sqrt(kap_n[i] * lam_s[i][t]);
        }
        const double m_s = vec_mean(s);
        const double v_s = vec_var(s, m_s);
        z_std_mean[i] = (m_s - 0.0) / std::sqrt(1.0 / nd);
        z_std_var[i]  = (v_s - 1.0) / std::sqrt(2.0 / nd);  // Normal: kurt-1 = 2

        emp_lam_mean[i] = m_lam; exp_lam_mean[i] = t_lam_mean;
        emp_lam_var[i]  = v_lam; exp_lam_var[i]  = t_lam_var;
        emp_mu_mean[i]  = m_mu;  exp_mu_mean[i]  = t_mu_mean;
        emp_mu_var[i]   = v_mu;  exp_mu_var[i]   = t_mu_var;

        const double zs[6] = { z_lam_mean[i], z_lam_var[i],
                               z_mu_mean[i],  z_mu_var[i],
                               z_std_mean[i], z_std_var[i] };
        for (double zv : zs) {
            if (std::abs(zv) > worst_abs_z) worst_abs_z = std::abs(zv);
        }
    }

    // ---- Group C: lambda frozen -> mu is exactly Gaussian ----------------
    normal_gamma_cluster_gibbs_block blk_f(make_cfg());
    blk_f.set_context(ctx);
    blk_f.freeze_sub("lambda");

    std::mt19937_64 rng_f(987654321);
    std::vector<std::vector<double>> mu_f(KD);
    for (std::size_t i = 0; i < KD; ++i) mu_f[i].reserve(N_DRAWS);
    bool lambda_held = true;
    for (std::size_t t = 0; t < N_DRAWS; ++t) {
        blk_f.step(rng_f);
        const arma::vec& cur = blk_f.current();
        for (std::size_t i = 0; i < KD; ++i) {
            mu_f[i].push_back(cur[i]);
            if (cur[KD + i] != init_lambda[i]) lambda_held = false;
        }
    }

    Rcpp::NumericVector z_fmu_mean(KD), z_fmu_var(KD);
    Rcpp::NumericVector emp_fmu_var(KD), exp_fmu_var(KD);
    for (std::size_t i = 0; i < KD; ++i) {
        // mu_kj | lambda_kj = lambda_fixed  ~  N(mu_n, 1/(kappa_n lambda_fixed))
        const double t_var = 1.0 / (kap_n[i] * init_lambda[i]);
        const double m = vec_mean(mu_f[i]);
        const double v = vec_var(mu_f[i], m);
        z_fmu_mean[i] = (m - mu_n[i]) / std::sqrt(t_var / nd);
        z_fmu_var[i]  = (v - t_var) / (t_var * std::sqrt(2.0 / nd));
        emp_fmu_var[i] = v; exp_fmu_var[i] = t_var;
        if (std::abs(z_fmu_mean[i]) > worst_abs_z)
            worst_abs_z = std::abs(z_fmu_mean[i]);
        if (std::abs(z_fmu_var[i]) > worst_abs_z)
            worst_abs_z = std::abs(z_fmu_var[i]);
    }

    const bool pass_moments = (worst_abs_z < Z_TOL);
    const bool all_pass = pass_moments && in_range && layout_ok && lambda_held;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")     = all_pass,
        Rcpp::Named("pass_moments") = pass_moments,
        Rcpp::Named("in_range")     = in_range,
        Rcpp::Named("layout_ok")    = layout_ok,
        Rcpp::Named("lambda_held")  = lambda_held,
        Rcpp::Named("worst_abs_z")  = worst_abs_z,
        Rcpp::Named("z_tol")        = Z_TOL,
        Rcpp::Named("n_draws")      = static_cast<int>(N_DRAWS),
        Rcpp::Named("z_lam_mean")   = z_lam_mean,
        Rcpp::Named("z_lam_var")    = z_lam_var,
        Rcpp::Named("z_mu_mean")    = z_mu_mean,
        Rcpp::Named("z_mu_var")     = z_mu_var,
        Rcpp::Named("z_std_mean")   = z_std_mean,
        Rcpp::Named("z_std_var")    = z_std_var,
        Rcpp::Named("z_fmu_mean")   = z_fmu_mean,
        Rcpp::Named("z_fmu_var")    = z_fmu_var,
        // Rcpp::List::create tops out at 20 arguments; the raw empirical and
        // analytic values are nested so all of them still come back.
        Rcpp::Named("empirical")    = Rcpp::List::create(
            Rcpp::Named("lam_mean") = emp_lam_mean,
            Rcpp::Named("lam_var")  = emp_lam_var,
            Rcpp::Named("mu_mean")  = emp_mu_mean,
            Rcpp::Named("mu_var")   = emp_mu_var,
            Rcpp::Named("fmu_var")  = emp_fmu_var),
        Rcpp::Named("analytic")     = Rcpp::List::create(
            Rcpp::Named("lam_mean") = exp_lam_mean,
            Rcpp::Named("lam_var")  = exp_lam_var,
            Rcpp::Named("mu_mean")  = exp_mu_mean,
            Rcpp::Named("mu_var")   = exp_mu_var,
            Rcpp::Named("fmu_var")  = exp_fmu_var));
}

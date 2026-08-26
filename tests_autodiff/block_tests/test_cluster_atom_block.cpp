// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_cluster_atom_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::cluster_atom_block
// (**Check #15** from skills/codegen_priors.md Sec.2c + skills/validator.md).
//
// Purpose
// -------
// cluster_atom_block exists for mixture components whose per-component prior
// has NO conjugate closed form, so it samples each component's atom with a
// univariate slice kernel (Neal 2003 stepping-out + shrinkage, delegated to
// univariate_slice_sampling_block). That makes its correctness impossible to
// check by inspection -- hence this test.
//
// The trick that makes a parity test possible at all: FEED IT A CONJUGATE
// TARGET. Given fixed allocations z, a Normal-Gamma prior
//     lambda_k ~ Gamma(a_0, rate b_0),   mu_k | lambda_k ~ N(m_0, 1/(kappa_0 lambda_k))
// with a Gaussian likelihood has the closed-form conditional
//     lambda_k ~ Gamma(a_n, rate b_n),   mu_k | lambda_k ~ N(m_n, 1/(kappa_n lambda_k))
// (Murphy 2007 "Conjugate Bayesian analysis of the Gaussian distribution"
// Sec.4). The block never learns that -- it is handed the same log-density it
// would get for a non-conjugate model -- so agreement with the analytic
// moments tests the SAMPLING MECHANISM, not the algebra.
//
// The EMPTY-component regime is tested too, and it is not decoration: with
// n_k = 0 the likelihood terms vanish and the conditional IS the prior. That
// is Ishwaran & James (2001) blocked Gibbs step (a) first half ("simulate
// Z_k ~iid H for each unoccupied k"), which the block must reproduce with no
// special-casing. A block that quietly pulled empty components toward the data
// would still look healthy on R-hat; only this comparison catches it.
//
// TOLERANCES
// ----------
// 40000 draws after 4000 burn-in, so every tolerance below is a plain Monte
// Carlo standard error: |empirical - analytic| < Z_TOL * MCSE for the means,
// and a 10% relative band on the variances (Check #15's stated 5% / 10% bar is
// looser than the z-score gate used here for the means).
// ============================================================================
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <cmath>
#include <random>
#include <vector>

#include "AI4BayesCode/cluster_atom_block.hpp"

using AI4BayesCode::block_context;
using AI4BayesCode::cluster_atom_block;
using AI4BayesCode::cluster_atom_block_config;
using AI4BayesCode::joint_constraint;
using AI4BayesCode::joint_nuts_sub_param;

namespace {

double vec_mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}
double vec_var(const std::vector<double>& v, double m) {
    double s = 0.0;
    for (double x : v) { const double e = x - m; s += e * e; }
    return s / static_cast<double>(v.size() - 1);
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List test_cluster_atom_block() {
    const std::size_t K       = 4;      // components 3 and 4 stay EMPTY
    const std::size_t N       = 40;
    const std::size_t N_BURN  = 4000;
    const std::size_t N_DRAWS = 40000;
    const double      Z_TOL   = 4.5;

    const double m_0 = 0.5, kappa_0 = 2.0, a_0 = 5.0, b_0 = 3.0;

    // ---- data: only components 1 and 2 receive observations --------------
    std::mt19937_64 gen(20260826u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::vec y(N), z(N);
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t k = i % 2;                    // 0 or 1
        z[i] = static_cast<double>(k + 1);
        y[i] = (k == 0 ? -2.0 : 3.0) + 0.7 * nd(gen);
    }
    arma::vec counts(K, arma::fill::zeros);
    for (std::size_t i = 0; i < N; ++i)
        counts[static_cast<std::size_t>(std::llround(z[i])) - 1] += 1.0;

    block_context ctx;
    ctx["y"] = y; ctx["z"] = z; ctx["cluster_counts"] = counts;

    // ---- the block is handed a plain natural-scale conditional -----------
    cluster_atom_block_config cfg;
    cfg.name    = "atoms";
    cfg.K_trunc = K;
    cfg.sub_params.push_back(joint_nuts_sub_param{"mu",     1u, joint_constraint::REAL});
    cfg.sub_params.push_back(joint_nuts_sub_param{"lambda", 1u, joint_constraint::POSITIVE});
    cfg.initial_cat = arma::vec(2 * K);
    for (std::size_t k = 0; k < K; ++k) {
        cfg.initial_cat[2 * k]     = 0.0;
        cfg.initial_cat[2 * k + 1] = 1.0;
    }
    cfg.log_density = [=](const arma::vec& th, std::size_t k,
                          const block_context& c) -> double {
        const double mu = th[0], lam = th[1];
        if (!(lam > 0.0) || !std::isfinite(mu))
            return -std::numeric_limits<double>::infinity();
        const arma::vec& yy = c.at("y");
        const arma::vec& zz = c.at("z");
        double n = 0.0, s1 = 0.0, s2 = 0.0;
        for (std::size_t i = 0; i < zz.n_elem; ++i) {
            if (static_cast<std::size_t>(std::llround(zz[i])) != k + 1) continue;
            n += 1.0; s1 += yy[i]; s2 += yy[i] * yy[i];
        }
        const double sse = s2 - 2.0 * mu * s1 + n * mu * mu;
        const double dm  = mu - m_0;
        return (a_0 - 1.0) * std::log(lam) - b_0 * lam
             + 0.5 * std::log(lam) - 0.5 * kappa_0 * lam * dm * dm
             + 0.5 * n * std::log(lam) - 0.5 * lam * sse;
    };
    cluster_atom_block blk(std::move(cfg));
    blk.set_context(ctx);

    std::mt19937_64 rng(4242u);
    for (std::size_t t = 0; t < N_BURN; ++t) blk.step(rng);

    std::vector<std::vector<double>> mu_s(K), lam_s(K);
    for (std::size_t k = 0; k < K; ++k) {
        mu_s[k].reserve(N_DRAWS);
        lam_s[k].reserve(N_DRAWS);
    }
    for (std::size_t t = 0; t < N_DRAWS; ++t) {
        blk.step(rng);
        for (std::size_t k = 0; k < K; ++k) {
            mu_s[k].push_back(blk.current()[2 * k]);
            lam_s[k].push_back(blk.current()[2 * k + 1]);
        }
    }

    // ---- analytic Normal-Gamma conditional per component -----------------
    Rcpp::NumericVector z_mu(K), z_lam(K), rel_var_mu(K), rel_var_lam(K);
    Rcpp::LogicalVector is_empty(K);
    bool all_pass = true;
    const double nd_draws = static_cast<double>(N_DRAWS);
    for (std::size_t k = 0; k < K; ++k) {
        double n = 0.0, s1 = 0.0, s2 = 0.0;
        for (std::size_t i = 0; i < N; ++i)
            if (static_cast<std::size_t>(std::llround(z[i])) == k + 1) {
                n += 1.0; s1 += y[i]; s2 += y[i] * y[i];
            }
        is_empty[k] = (n == 0.0);
        const double ybar  = (n > 0.0) ? s1 / n : 0.0;
        const double kap_n = kappa_0 + n;
        const double m_n   = (kappa_0 * m_0 + s1) / kap_n;
        const double a_n   = a_0 + 0.5 * n;
        const double sse   = (n > 0.0) ? (s2 - n * ybar * ybar) : 0.0;
        const double b_n   = b_0 + 0.5 * sse
                           + 0.5 * kappa_0 * n * (ybar - m_0) * (ybar - m_0) / kap_n;
        // Marginals of NormalGamma(m_n, kap_n, a_n, b_n):
        //   lambda ~ Gamma(a_n, rate b_n)   -> E = a_n/b_n,  Var = a_n/b_n^2
        //   mu     ~ t_{2 a_n}(m_n, b_n/(a_n kap_n)) -> E = m_n,
        //                                    Var = b_n/((a_n - 1) kap_n)
        const double E_mu = m_n,          V_mu  = b_n / ((a_n - 1.0) * kap_n);
        const double E_l  = a_n / b_n,    V_l   = a_n / (b_n * b_n);

        const double em = vec_mean(mu_s[k]),  vm = vec_var(mu_s[k], em);
        const double el = vec_mean(lam_s[k]), vl = vec_var(lam_s[k], el);

        z_mu[k]  = (em - E_mu) / std::sqrt(vm / nd_draws);
        z_lam[k] = (el - E_l)  / std::sqrt(vl / nd_draws);
        rel_var_mu[k]  = std::fabs(vm - V_mu) / V_mu;
        rel_var_lam[k] = std::fabs(vl - V_l)  / V_l;

        if (std::fabs(z_mu[k])  > Z_TOL) all_pass = false;
        if (std::fabs(z_lam[k]) > Z_TOL) all_pass = false;
        if (rel_var_mu[k]  > 0.10)       all_pass = false;
        if (rel_var_lam[k] > 0.10)       all_pass = false;
    }

    return Rcpp::List::create(
        Rcpp::Named("all_pass")    = all_pass,
        Rcpp::Named("z_mu")        = z_mu,
        Rcpp::Named("z_lambda")    = z_lam,
        Rcpp::Named("rel_var_mu")  = rel_var_mu,
        Rcpp::Named("rel_var_lam") = rel_var_lam,
        Rcpp::Named("is_empty")    = is_empty,
        Rcpp::Named("z_tol")       = Z_TOL,
        Rcpp::Named("n_draws")     = static_cast<int>(N_DRAWS));
}

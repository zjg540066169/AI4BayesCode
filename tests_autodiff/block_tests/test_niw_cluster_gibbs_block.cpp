// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_niw_cluster_gibbs_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::niw_cluster_gibbs_block.
//
// WHAT IS VERIFIED
// ----------------
// niw_cluster_gibbs_block draws (mu_k, Sigma_k), for every cluster k, from
// the conjugate Normal-Inverse-Wishart full conditional implied by the
// cluster labels z and the data y. That conditional is available in CLOSED
// FORM, so this test takes the strongest option on the list: compare the
// sampled distribution directly against the analytic NIW moments. The
// weaker fallbacks (invariance of a kernel started at its target, exact
// enumeration, a detailed-balance identity) exist for kernels whose target
// cannot be written down; they are not needed here.
//
// step() recomputes the per-cluster sufficient statistics from the context
// and then draws afresh -- it never reads the previous draw. The draws are
// therefore i.i.d., which is what licenses plain 1/sqrt(M) standard errors
// below instead of ESS-corrected Monte Carlo errors.
//
// ANALYTIC TARGET
// ---------------
// For a cluster whose posterior NIW parameters are (mu_n, kappa_n, Psi_n,
// nu_n) in dimension p:
//
//   Sigma ~ IW(Psi_n, nu_n)
//     E[Sigma_ij]   = Psi_n(i,j) / (nu_n - p - 1)                (nu_n > p+1)
//     Var[Sigma_ij] = [ (nu_n - p + 1) Psi_n(i,j)^2
//                     + (nu_n - p - 1) Psi_n(i,i) Psi_n(j,j) ]
//                     / [ (nu_n - p) (nu_n - p - 1)^2 (nu_n - p - 3) ]
//                                                                (nu_n > p+3)
//   mu | Sigma ~ N(mu_n, Sigma / kappa_n), hence marginally
//     E[mu]   = mu_n
//     Cov[mu] = E[Sigma] / kappa_n = Psi_n / (kappa_n (nu_n - p - 1))
//
// (Standard Inverse-Wishart element moments, Press 1982; posterior update
//  Murphy 2007 "Conjugate Bayesian analysis of the Gaussian distribution"
//  sec. 4. The p = 1 case of the variance formula reduces to the
//  Inverse-Gamma variance 2 psi^2 / ((nu-2)^2 (nu-4)), as it must.)
//
// The four families of checks are chosen so that each posterior quantity is
// pinned by at least one of them:
//   * E[Sigma_k]  -- pins Psi_n up to the factor (nu_n - p - 1).
//   * Var[Sigma_k]-- pins nu_n separately, so a scale/df trade-off that
//                    leaves E[Sigma] intact cannot slip through.
//   * E[mu_k]     -- pins mu_n.
//   * Cov[mu_k]   -- the ONLY check that sees kappa_n; every Sigma check is
//                    blind to it.
//
// FIXTURE (d = 2, K_trunc = 3, N = 7)
// -----------------------------------
// Cluster 1 gets 4 observations, cluster 2 gets 3, cluster 3 gets NONE.
//   * Two non-empty clusters with DIFFERENT posteriors make the check
//     sensitive to mis-routing of observations by z: a routing bug moves
//     both targets, not neither.
//   * Cluster 3 is the only exercise of the empty-cluster branch, which is
//     specified to fall back to the prior (mu_0, kappa_0, Psi_0, nu_0).
//   * Cluster 2's posterior scale matrix has a NEGATIVE off-diagonal while
//     cluster 1's is strongly positive, so the sign of the off-diagonal --
//     the whole reason this block exists next to the diagonal
//     normal_gamma_cluster sibling -- is exercised in both directions.
// The posterior parameters used as the target are recomputed here from the
// raw data by a DIFFERENT route than the block uses: this file forms
// S_k = sum_i (y_i - bar_y)(y_i - bar_y)^T directly, whereas the block forms
// sum_i y_i y_i^T - n_k bar_y bar_y^T.
//
// Prior: mu_0 = (0,0), kappa_0 = 2, Psi_0 = [[2, 0.5], [0.5, 1]], nu_0 = 15.
// nu_0 = 15 is chosen so that the smallest degrees of freedom anywhere in
// the fixture (the empty cluster, nu = 15) still has finite FOURTH moments:
// an IW element has a finite moment of order m iff nu > p - 1 + 2m, i.e.
// nu > 9 for m = 4. The variance checks below need that.
//
// TOLERANCES
// ----------
// M = 50000 i.i.d. draws.
//   * Means (E[Sigma_ij], E[mu_j]): band = 4 * ANALYTIC standard error,
//     sqrt(Var/M), with Var taken from the closed forms above. z = 4 is a
//     two-sided p of 6.3e-5 per check; there are 15 mean checks, so the
//     family-wise false-alarm probability is about 1e-3 at the fixed seed.
//   * Variances (Var[Sigma_ij]) and covariances (Cov[mu_ab]): band =
//     5 * MEASURED standard error, sqrt((m4_hat - s2_hat^2)/M) for a
//     variance and sd(product of centred pairs)/sqrt(M) for a covariance.
//     The fourth moment is finite here (see above) but its estimate is
//     itself noisy for heavy-tailed IW elements, so the band is 5 SE rather
//     than 4. It remains far tighter than any realistic defect: an
//     off-by-one in nu_n changes Var[Sigma] by roughly 30%, which lands at
//     16 measured SE at this M -- three times the band.
//   * Symmetry of every drawn Sigma_k: 1e-12 absolute. The block
//     symmetrizes explicitly, so the only slack is float round-off on 2x2
//     matrices, orders of magnitude below 1e-12.
//   * Positive definiteness of every drawn Sigma_k: Cholesky must succeed.
//     Pass/fail, no tolerance.
//
// No threshold here was widened to make the test pass.
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
#include "AI4BayesCode/niw_cluster_gibbs_block.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

using AI4BayesCode::niw_cluster_gibbs_block;
using AI4BayesCode::niw_cluster_gibbs_block_config;
using AI4BayesCode::block_context;

// [[Rcpp::export]]
Rcpp::List test_niw_cluster_gibbs_block() {
    const std::size_t d = 2;
    const std::size_t K = 3;
    const std::size_t N = 7;
    const std::size_t M = 50000;          // i.i.d. draws
    const double Z_MEAN = 4.0;            // analytic-SE band for first moments
    const double Z_VAR  = 5.0;            // measured-SE band for second moments
    const double SYM_TOL = 1e-12;

    // ---- Prior --------------------------------------------------------
    arma::vec mu_0(d, arma::fill::zeros);
    const double kappa_0 = 2.0;
    const double nu_0    = 15.0;
    arma::mat Psi_0(d, d);
    Psi_0(0, 0) = 2.0;  Psi_0(0, 1) = 0.5;
    Psi_0(1, 0) = 0.5;  Psi_0(1, 1) = 1.0;

    // ---- Fixture data (N x d), and labels z in {1, ..., K} -------------
    // Cluster 1: strongly positively correlated. Cluster 2: negatively
    // tilted. Cluster 3: empty.
    arma::mat Y(N, d);
    Y(0, 0) =  1.0;  Y(0, 1) =  2.0;
    Y(1, 0) =  2.0;  Y(1, 1) =  3.0;
    Y(2, 0) =  3.0;  Y(2, 1) =  5.0;
    Y(3, 0) =  4.0;  Y(3, 1) =  6.0;
    Y(4, 0) = -2.0;  Y(4, 1) =  1.0;
    Y(5, 0) = -1.0;  Y(5, 1) = -1.0;
    Y(6, 0) =  0.0;  Y(6, 1) =  0.0;
    arma::vec z({1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0});

    arma::vec y_flat(N * d);              // row-major, as the block expects
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < d; ++j)
            y_flat[i * d + j] = Y(i, j);

    // ---- Independent recomputation of the NIW posterior per cluster ----
    // S_k is formed here as sum (y - bar_y)(y - bar_y)^T, a different route
    // than the block's sum y y^T - n bar_y bar_y^T.
    std::vector<arma::vec> mu_n(K);
    std::vector<arma::mat> Psi_n(K);
    std::vector<double>    kappa_n(K), nu_n(K);
    for (std::size_t k = 0; k < K; ++k) {
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < N; ++i)
            if (static_cast<std::size_t>(std::llround(z[i])) == k + 1)
                idx.push_back(i);
        const double nk = static_cast<double>(idx.size());
        if (idx.empty()) {                // empty cluster -> prior
            mu_n[k]    = mu_0;
            kappa_n[k] = kappa_0;
            nu_n[k]    = nu_0;
            Psi_n[k]   = Psi_0;
            continue;
        }
        arma::vec bar(d, arma::fill::zeros);
        for (std::size_t i : idx) bar += Y.row(i).t();
        bar /= nk;
        arma::mat S(d, d, arma::fill::zeros);
        for (std::size_t i : idx) {
            const arma::vec dv = Y.row(i).t() - bar;
            S += dv * dv.t();
        }
        kappa_n[k] = kappa_0 + nk;
        nu_n[k]    = nu_0    + nk;
        mu_n[k]    = (kappa_0 * mu_0 + nk * bar) / kappa_n[k];
        const arma::vec dev = bar - mu_0;
        Psi_n[k]   = Psi_0 + S +
                     (kappa_0 * nk / kappa_n[k]) * (dev * dev.t());
    }

    // ---- Build the block ------------------------------------------------
    niw_cluster_gibbs_block_config cfg;
    cfg.name    = "niw_test";
    cfg.K_trunc = K;
    cfg.d       = d;
    cfg.N       = N;
    cfg.z_key   = "z";
    cfg.y_key   = "y";
    cfg.mu_0    = mu_0;
    cfg.kappa_0 = kappa_0;
    cfg.nu_0    = nu_0;
    cfg.Psi_0_flat.set_size(d * d);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j)
            cfg.Psi_0_flat[i * d + j] = Psi_0(i, j);
    cfg.initial_mu.zeros(K * d);
    cfg.initial_sigma.zeros(K * d * d);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t i = 0; i < d; ++i)
            cfg.initial_sigma[k * d * d + i * d + i] = 1.0;
    niw_cluster_gibbs_block blk(std::move(cfg));

    block_context ctx;
    ctx["z"] = z;
    ctx["y"] = y_flat;
    blk.set_context(ctx);

    // ---- Draw -----------------------------------------------------------
    std::vector<arma::mat> mu_draws(K, arma::mat(M, d));
    std::vector<arma::mat> sg_draws(K, arma::mat(M, d * d));
    bool pass_symmetric = true;
    bool pass_pd        = true;
    double worst_asym   = 0.0;

    std::mt19937_64 rng(20260818);
    for (std::size_t t = 0; t < M; ++t) {
        blk.step(rng);
        const arma::vec& cur = blk.current();
        const std::size_t off_s = K * d;
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t j = 0; j < d; ++j)
                mu_draws[k](t, j) = cur[k * d + j];
            arma::mat Sk(d, d);
            for (std::size_t i = 0; i < d; ++i)
                for (std::size_t j = 0; j < d; ++j) {
                    const double v = cur[off_s + k * d * d + i * d + j];
                    Sk(i, j) = v;
                    sg_draws[k](t, i * d + j) = v;
                }
            for (std::size_t i = 0; i < d; ++i)
                for (std::size_t j = i + 1; j < d; ++j) {
                    const double a = std::abs(Sk(i, j) - Sk(j, i));
                    if (a > worst_asym) worst_asym = a;
                }
            arma::mat L_chk;
            if (!arma::chol(L_chk, Sk, "lower")) pass_pd = false;
        }
    }
    pass_symmetric = (worst_asym <= SYM_TOL);

    // ---- Compare against the analytic NIW moments -----------------------
    const double Md = static_cast<double>(M);
    std::vector<double> sm_emp, sm_ana, sm_z;   // E[Sigma_ij]
    std::vector<double> sv_emp, sv_ana, sv_z;   // Var[Sigma_ij]
    std::vector<double> mm_emp, mm_ana, mm_z;   // E[mu_j]
    std::vector<double> mc_emp, mc_ana, mc_z;   // Cov[mu_ab]
    bool pass_sigma_mean = true, pass_sigma_var = true;
    bool pass_mu_mean    = true, pass_mu_cov    = true;

    for (std::size_t k = 0; k < K; ++k) {
        const double nu  = nu_n[k];
        const double p   = static_cast<double>(d);
        const double den = nu - p - 1.0;                    // > 0 by design
        const arma::mat E_Sig  = Psi_n[k] / den;
        const arma::mat Cov_mu = Psi_n[k] / (kappa_n[k] * den);
        // Element variance of an IW draw (see header).
        const double vden = (nu - p) * (nu - p - 1.0) * (nu - p - 1.0) *
                            (nu - p - 3.0);

        for (std::size_t i = 0; i < d; ++i) {
            for (std::size_t j = i; j < d; ++j) {
                const arma::vec x = sg_draws[k].col(i * d + j);
                const double m  = arma::mean(x);
                const arma::vec c = x - m;
                const double s2 = arma::dot(c, c) / (Md - 1.0);
                double m4 = 0.0;
                for (std::size_t t = 0; t < M; ++t) {
                    const double c2 = c[t] * c[t];
                    m4 += c2 * c2;
                }
                m4 /= Md;

                const double var_ana =
                    ((nu - p + 1.0) * Psi_n[k](i, j) * Psi_n[k](i, j) +
                     (nu - p - 1.0) * Psi_n[k](i, i) * Psi_n[k](j, j)) / vden;

                // Mean: analytic SE.
                const double se_m = std::sqrt(var_ana / Md);
                const double zm   = std::abs(m - E_Sig(i, j)) / se_m;
                sm_emp.push_back(m);
                sm_ana.push_back(E_Sig(i, j));
                sm_z.push_back(zm);
                if (!(zm < Z_MEAN)) pass_sigma_mean = false;

                // Variance: measured SE from the sample 4th central moment.
                const double se_v =
                    std::sqrt(std::max(m4 - s2 * s2, 0.0) / Md);
                const double zv = std::abs(s2 - var_ana) / se_v;
                sv_emp.push_back(s2);
                sv_ana.push_back(var_ana);
                sv_z.push_back(zv);
                if (!(zv < Z_VAR)) pass_sigma_var = false;
            }
        }

        std::vector<arma::vec> mc_centered(d);
        for (std::size_t j = 0; j < d; ++j) {
            const arma::vec x = mu_draws[k].col(j);
            const double m = arma::mean(x);
            mc_centered[j] = x - m;
            const double se = std::sqrt(Cov_mu(j, j) / Md);   // analytic SE
            const double zz = std::abs(m - mu_n[k][j]) / se;
            mm_emp.push_back(m);
            mm_ana.push_back(mu_n[k][j]);
            mm_z.push_back(zz);
            if (!(zz < Z_MEAN)) pass_mu_mean = false;
        }
        for (std::size_t a = 0; a < d; ++a) {
            for (std::size_t b = a; b < d; ++b) {
                const arma::vec pr = mc_centered[a] % mc_centered[b];
                const double c_hat = arma::accu(pr) / (Md - 1.0);
                const double pm = arma::mean(pr);
                const arma::vec pc = pr - pm;
                // Measured SE of the sample covariance.
                const double se = std::sqrt(arma::dot(pc, pc) /
                                            ((Md - 1.0) * Md));
                const double zz = std::abs(c_hat - Cov_mu(a, b)) / se;
                mc_emp.push_back(c_hat);
                mc_ana.push_back(Cov_mu(a, b));
                mc_z.push_back(zz);
                if (!(zz < Z_VAR)) pass_mu_cov = false;
            }
        }
    }

    auto vmax = [](const std::vector<double>& v) {
        double m = 0.0;
        for (double x : v) if (x > m) m = x;
        return m;
    };
    // One row per check: (empirical, analytic, |z| in SE units).
    auto tab = [](const std::vector<double>& e, const std::vector<double>& a,
                  const std::vector<double>& zz) {
        Rcpp::NumericMatrix out(static_cast<int>(e.size()), 3);
        for (std::size_t i = 0; i < e.size(); ++i) {
            out(static_cast<int>(i), 0) = e[i];
            out(static_cast<int>(i), 1) = a[i];
            out(static_cast<int>(i), 2) = zz[i];
        }
        Rcpp::colnames(out) = Rcpp::CharacterVector::create(
            "empirical", "analytic", "abs_z");
        return out;
    };

    const bool all_pass = pass_sigma_mean && pass_sigma_var &&
                          pass_mu_mean && pass_mu_cov &&
                          pass_symmetric && pass_pd;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")         = all_pass,
        Rcpp::Named("pass_sigma_mean")  = pass_sigma_mean,
        Rcpp::Named("pass_sigma_var")   = pass_sigma_var,
        Rcpp::Named("pass_mu_mean")     = pass_mu_mean,
        Rcpp::Named("pass_mu_cov")      = pass_mu_cov,
        Rcpp::Named("pass_symmetric")   = pass_symmetric,
        Rcpp::Named("pass_pd")          = pass_pd,
        Rcpp::Named("worst_asym")       = worst_asym,
        Rcpp::Named("max_z_sigma_mean") = vmax(sm_z),
        Rcpp::Named("max_z_sigma_var")  = vmax(sv_z),
        Rcpp::Named("max_z_mu_mean")    = vmax(mm_z),
        Rcpp::Named("max_z_mu_cov")     = vmax(mc_z),
        Rcpp::Named("sigma_mean")       = tab(sm_emp, sm_ana, sm_z),
        Rcpp::Named("sigma_var")        = tab(sv_emp, sv_ana, sv_z),
        Rcpp::Named("mu_mean")          = tab(mm_emp, mm_ana, mm_z),
        Rcpp::Named("mu_cov")           = tab(mc_emp, mc_ana, mc_z),
        Rcpp::Named("n_draws")          = static_cast<int>(M),
        Rcpp::Named("z_mean_band")      = Z_MEAN,
        Rcpp::Named("z_var_band")       = Z_VAR);
}

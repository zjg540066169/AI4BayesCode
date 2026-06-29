// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  FiniteGaussianMixture.cpp
//
//  REFERENCE TEMPLATE for FINITE-K Gaussian mixture with diagonal
//  Normal-Gamma cluster prior (Bishop PRML §10.2, Murphy 2007 §11.4).
//  K is FIXED at construction; cluster mixing weights π are sampled
//  exactly from a symmetric Dirichlet conditional via the SHIPPED
//  `dirichlet_gibbs_block` (this is the first reference example using
//  that block — it complements the BNP truncated-SBP path in
//  `DPGaussianMixture.cpp`).
//
//  Model
//  -----
//      y_i      ~ N(μ_{z_i}, diag(1 / λ_{z_i}))      i = 1..N
//      z_i      ~ Categorical(π)
//      π        ~ Dirichlet(α/K, ..., α/K)           (symmetric)
//      (μ_k, λ_k) ~ NormalGamma(μ_0, κ_0, a_λ_0, b_λ_0)
//      K        : FIXED at construction (user-specified)
//      α        : FIXED at construction (v0; future v0.5 NUTS on log α)
//
//  Block decomposition (Gibbs sweep order)
//  ---------------------------------------
//      child(0) z              categorical_gibbs_block
//      child(1) cluster_params normal_gamma_cluster_gibbs_block
//      child(2) pi             dirichlet_gibbs_block
//                              (alpha_post_fn = α/K + cluster_counts)
//
//  Refreshers
//  ----------
//      cluster_counts   register_refresher (deterministic)
//                       counts_from_z(z, K) — invalidated by z
//
//  JUSTIFICATION (Check #16):
//  - z is DISCRETE → categorical_gibbs_block (Exception 1 from
//    codegen_priors.md §2b). Conditional independence holds because
//    π is sampled separately.
//  - π conditional is EXACTLY Dirichlet (Dirichlet-Categorical
//    conjugate) → dirichlet_gibbs_block (Exception 1 from
//    codegen_priors.md §2b).
//  - (μ, λ): normal_gamma_cluster_gibbs_block (Tier-B block,
//    shipped 2026-05-02 with Check #15 parity test).
//
//  No hand-written log-density: all 3 children are conjugate Gibbs.
//
//  ----------------------------------------------------------------------------
//  FRONTEND-INDEPENDENT STANDALONE VERSION
//  ----------------------------------------------------------------------------
//  This file has been rewritten as a self-contained C++ program with no
//  Rcpp / pybind binding layer. The original Tier-A wrapper class returned
//  Rcpp::List and took Rcpp::NumericMatrix; rather than neutralize those
//  types, main() drives the composite_block DIRECTLY (it replicates the
//  constructor's wiring: data().set(...), declare_dependencies, the
//  cluster_counts refresher, and add_child for the three Gibbs blocks).
//  The MODEL is preserved exactly (priors, conditionals, block config,
//  hyperparameters). The predict-time y_rep stochastic refresher is omitted
//  because the recovery demo does not exercise predict_at.
// ============================================================================

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include "AI4BayesCode/categorical_gibbs_block.hpp"
#include "AI4BayesCode/dirichlet_gibbs_block.hpp"
#include "AI4BayesCode/normal_gamma_cluster_gibbs_block.hpp"
#include "AI4BayesCode/bnp_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::categorical_gibbs_block;
using AI4BayesCode::categorical_gibbs_block_config;
using AI4BayesCode::dirichlet_gibbs_block;
using AI4BayesCode::dirichlet_gibbs_block_config;
using AI4BayesCode::normal_gamma_cluster_gibbs_block;
using AI4BayesCode::normal_gamma_cluster_gibbs_block_config;
namespace bnp = AI4BayesCode::bnp;

// ============================================================================
//  Free helpers
// ============================================================================

namespace {

inline double diag_normal_log_density(const double* y, const double* mu,
                                       const double* lambda,
                                       std::size_t d) {
    double lp = 0.0;
    constexpr double kLog2Pi = 1.83787706640934548356065947281;
    for (std::size_t j = 0; j < d; ++j) {
        const double dev = y[j] - mu[j];
        lp += 0.5 * std::log(lambda[j])
            - 0.5 * kLog2Pi
            - 0.5 * lambda[j] * dev * dev;
    }
    return lp;
}

// Data-driven weakly-informative Normal-Gamma hypers (matches the original
// constructor's dd_mu0_ / dd_blambda_): mu_0 = column means, b_lambda_0 =
// mean column variance.
arma::vec dd_mu0(const arma::mat& y) {
    const std::size_t d = y.n_cols;
    arma::vec m(d, arma::fill::zeros);
    for (std::size_t j = 0; j < d; ++j) m[j] = arma::mean(y.col(j));
    return m;
}
double dd_blambda(const arma::mat& y) {
    const std::size_t n = y.n_rows, d = y.n_cols;
    double acc = 0.0; std::size_t used = 0;
    for (std::size_t j = 0; j < d; ++j) {
        const double mean = arma::mean(y.col(j));
        double ss = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double dv = y(i, j) - mean; ss += dv * dv;
        }
        const double v = (n > 1) ? ss / (n - 1) : 1.0;
        if (std::isfinite(v) && v > 0.0) { acc += v; ++used; }
    }
    const double b = (used > 0) ? acc / used : 1.0;
    return (std::isfinite(b) && b > 0.0) ? b : 1.0;
}

}  // anonymous namespace

// ============================================================================
//  Standalone demo
//
//  Simulate a 2-cluster, 2-D Gaussian mixture from a known truth, fit the
//  finite-K (K=2) Gibbs sampler, and check that the posterior-mean cluster
//  centers recover the two truth centers (matched up to label switching) and
//  that the fitted mixture beats a single-Gaussian (K=1) baseline in
//  held-out-style mean log-likelihood on the training data.
// ============================================================================

int main() {
    // ---- 1. Known truth -----------------------------------------------------
    const std::size_t d = 2;
    const std::size_t K = 2;
    const std::size_t n_per = 150;
    const std::size_t N = K * n_per;

    // True cluster centers (well separated) and per-dim precision (var = 1).
    const double true_mu[2][2] = {{-3.0, -3.0}, {3.0, 3.0}};
    const double true_sd = 1.0;
    const double true_pi[2] = {0.5, 0.5};

    std::mt19937_64 sim_rng(20260621ULL);
    std::normal_distribution<double> snorm(0.0, 1.0);

    arma::mat y(N, d);
    std::vector<std::size_t> true_z(N);
    {
        std::size_t i = 0;
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t r = 0; r < n_per; ++r, ++i) {
                true_z[i] = k;
                for (std::size_t j = 0; j < d; ++j)
                    y(i, j) = true_mu[k][j] + true_sd * snorm(sim_rng);
            }
        }
    }

    // ---- 2. Data-driven hyperparameters (match original ctor) ---------------
    const arma::vec mu0_arma   = dd_mu0(y);
    const double    kappa_0    = 0.01;
    const double    a_lambda_0 = 2.0;
    const double    b_lambda_0 = dd_blambda(y);
    const double    alpha_dir  = 1.0;

    // ---- 3. Build composite block directly (replicates the ctor wiring) -----
    std::mt19937_64 rng(12345ULL);
    auto comp = std::make_unique<composite_block>("FiniteGaussianMixture");

    // Data + priors.
    arma::vec y_flat(N * d);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < d; ++j)
            y_flat[i * d + j] = y(i, j);
    comp->data().set("y", y_flat);

    comp->data().set("mu_0", mu0_arma);
    comp->data().set("kappa_0",    arma::vec{kappa_0});
    comp->data().set("a_lambda_0", arma::vec{a_lambda_0});
    comp->data().set("b_lambda_0", arma::vec{b_lambda_0});
    comp->data().set("alpha_dir",  arma::vec{alpha_dir});

    // Initial sampler state: z spread (i mod K) + 1.
    arma::vec z_init(N);
    for (std::size_t i = 0; i < N; ++i)
        z_init[i] = static_cast<double>((i % K) + 1);
    comp->data().set("z", z_init);

    // π: uniform 1/K.
    arma::vec pi_init(K, arma::fill::value(1.0 / static_cast<double>(K)));
    comp->data().set("pi", pi_init);

    // μ: data-driven anchor spread; λ: prior mean.
    arma::vec mu_init(K * d, arma::fill::zeros);
    arma::vec lambda_init(K * d, arma::fill::value(a_lambda_0 / b_lambda_0));
    for (std::size_t k = 0; k < K; ++k) {
        const std::size_t i_anchor = (k * N) / K;
        for (std::size_t j = 0; j < d; ++j)
            mu_init[k * d + j] = y(i_anchor, j);
    }
    comp->data().set("mu", mu_init);
    comp->data().set("lambda", lambda_init);

    // cluster_counts: derived (refresher).
    comp->data().set("cluster_counts", arma::vec(K, arma::fill::zeros));
    comp->data().register_refresher("cluster_counts",
        [K](const AI4BayesCode::shared_data_t& dat) -> arma::vec {
            const arma::vec& z = dat.get("z");
            return bnp::counts_from_z(z, K);
        });

    // Gibbs DAG dependencies / invalidations.
    comp->data().declare_dependencies("z",  {"y", "pi", "mu", "lambda"});
    comp->data().declare_dependencies("cluster_params", {"z", "y"});
    comp->data().declare_dependencies("pi", {"cluster_counts", "alpha_dir"});
    comp->data().declare_invalidates("z", {"cluster_counts"});

    comp->data().refresh_all();

    // child(0) z (categorical_gibbs_block)
    {
        categorical_gibbs_block_config cfg;
        cfg.name           = "z";
        cfg.n_obs          = N;
        cfg.n_categories   = K;
        cfg.initial_labels = z_init;
        const std::size_t d_capture = d;
        const std::size_t N_capture = N;
        const std::size_t K_capture = K;
        cfg.log_probs_fn = [d_capture, N_capture, K_capture]
            (const block_context& ctx) -> arma::mat {
            const arma::vec& y_flat_c = ctx.at("y");
            const arma::vec& pi       = ctx.at("pi");
            const arma::vec& mu       = ctx.at("mu");
            const arma::vec& lam      = ctx.at("lambda");
            arma::mat lp(N_capture, K_capture);
            for (std::size_t i = 0; i < N_capture; ++i) {
                for (std::size_t k = 0; k < K_capture; ++k) {
                    const double log_pi_k = std::log(pi[k] + 1e-300);
                    lp(i, k) = log_pi_k
                             + diag_normal_log_density(
                                 y_flat_c.memptr() + i * d_capture,
                                 mu.memptr()       + k * d_capture,
                                 lam.memptr()      + k * d_capture,
                                 d_capture);
                }
            }
            return lp;
        };
        comp->add_child(
            std::make_unique<categorical_gibbs_block>(std::move(cfg)));
    }

    // child(1) cluster_params (normal_gamma_cluster_gibbs_block)
    {
        normal_gamma_cluster_gibbs_block_config cfg;
        cfg.name        = "cluster_params";
        cfg.K_trunc     = K;
        cfg.d           = d;
        cfg.N           = N;
        cfg.z_key       = "z";
        cfg.y_key       = "y";
        cfg.mu_name     = "mu";
        cfg.lambda_name = "lambda";
        cfg.mu_0        = mu0_arma;
        cfg.kappa_0     = kappa_0;
        cfg.a_lambda_0  = a_lambda_0;
        cfg.b_lambda_0  = b_lambda_0;
        cfg.initial_mu     = mu_init;
        cfg.initial_lambda = lambda_init;
        comp->add_child(
            std::make_unique<normal_gamma_cluster_gibbs_block>(
                std::move(cfg)));
    }

    // child(2) pi (dirichlet_gibbs_block)
    {
        dirichlet_gibbs_block_config cfg;
        cfg.name           = "pi";
        cfg.n_categories   = K;
        cfg.initial_values = pi_init;
        const std::size_t K_capture = K;
        cfg.alpha_post_fn = [K_capture]
            (const block_context& ctx) -> arma::vec {
            const arma::vec& counts = ctx.at("cluster_counts");
            const double alpha      = ctx.at("alpha_dir")[0];
            const double per_k = alpha / static_cast<double>(K_capture);
            arma::vec out(K_capture);
            for (std::size_t k = 0; k < K_capture; ++k)
                out[k] = per_k + counts[k];
            return out;
        };
        comp->add_child(
            std::make_unique<dirichlet_gibbs_block>(std::move(cfg)));
    }

    // ---- 4. Warmup + sampling, accumulate posterior means -------------------
    const int n_warmup = 500;
    const int n_keep   = 1500;

    for (int s = 0; s < n_warmup; ++s) comp->step(rng);

    arma::vec mu_sum(K * d, arma::fill::zeros);
    arma::vec lam_sum(K * d, arma::fill::zeros);
    arma::vec pi_sum(K, arma::fill::zeros);
    for (int s = 0; s < n_keep; ++s) {
        comp->step(rng);
        mu_sum  += comp->data().get("mu");
        lam_sum += comp->data().get("lambda");
        pi_sum  += comp->data().get("pi");
    }
    const arma::vec mu_hat  = mu_sum  / static_cast<double>(n_keep);
    const arma::vec lam_hat = lam_sum / static_cast<double>(n_keep);
    const arma::vec pi_hat  = pi_sum  / static_cast<double>(n_keep);

    // ---- 5. Match recovered clusters to truth (label-switch invariant) ------
    // For K=2: try both label assignments, pick the one with smaller total
    // center error, then report that error.
    auto center_err = [&](int map0, int map1) {
        const int map[2] = {map0, map1};
        double e = 0.0;
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t j = 0; j < d; ++j) {
                const double dv = mu_hat[k * d + j] - true_mu[map[k]][j];
                e += dv * dv;
            }
        }
        return std::sqrt(e / (K * d));
    };
    const double err_identity = center_err(0, 1);
    const double err_swapped  = center_err(1, 0);
    const double rmse_centers = std::min(err_identity, err_swapped);

    // ---- 6. Mixture vs single-Gaussian baseline mean log-likelihood ---------
    // Mixture log p(y_i) = logsumexp_k [ log pi_k + N(y_i | mu_k, 1/lam_k) ].
    double mix_ll = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double comps[2];
        double maxc = -std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < K; ++k) {
            const double c = std::log(pi_hat[k] + 1e-300)
                + diag_normal_log_density(y_flat.memptr() + i * d,
                                          mu_hat.memptr() + k * d,
                                          lam_hat.memptr() + k * d, d);
            comps[k] = c;
            if (c > maxc) maxc = c;
        }
        double se = 0.0;
        for (std::size_t k = 0; k < K; ++k) se += std::exp(comps[k] - maxc);
        mix_ll += maxc + std::log(se);
    }
    mix_ll /= static_cast<double>(N);

    // Single-Gaussian baseline: MLE mean + per-dim precision.
    arma::vec base_mu(d, arma::fill::zeros), base_lam(d);
    for (std::size_t j = 0; j < d; ++j) base_mu[j] = arma::mean(y.col(j));
    for (std::size_t j = 0; j < d; ++j) {
        double ss = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double dv = y(i, j) - base_mu[j]; ss += dv * dv;
        }
        const double var = ss / static_cast<double>(N);
        base_lam[j] = 1.0 / (var > 0.0 ? var : 1.0);
    }
    double base_ll = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        base_ll += diag_normal_log_density(y_flat.memptr() + i * d,
                                           base_mu.memptr(),
                                           base_lam.memptr(), d);
    base_ll /= static_cast<double>(N);

    // ---- 7. Report ----------------------------------------------------------
    std::printf("FiniteGaussianMixture standalone demo\n");
    std::printf("  N=%zu  d=%zu  K=%zu  (warmup=%d, keep=%d)\n",
                N, d, K, n_warmup, n_keep);
    std::printf("  true centers : (%.2f,%.2f) (%.2f,%.2f)\n",
                true_mu[0][0], true_mu[0][1], true_mu[1][0], true_mu[1][1]);
    const int s = (err_swapped < err_identity) ? 1 : 0;  // matched labels
    std::printf("  fit centers  : (%.2f,%.2f) (%.2f,%.2f)  [matched to truth]\n",
                mu_hat[(s == 0 ? 0 : 1) * d + 0], mu_hat[(s == 0 ? 0 : 1) * d + 1],
                mu_hat[(s == 0 ? 1 : 0) * d + 0], mu_hat[(s == 0 ? 1 : 0) * d + 1]);
    std::printf("  pi_hat       : %.3f %.3f   (true %.3f %.3f)\n",
                pi_hat[0], pi_hat[1], true_pi[0], true_pi[1]);
    std::printf("  RMSE(centers vs truth) = %.4f\n", rmse_centers);
    std::printf("  mean loglik : mixture=%.4f  single-Gaussian=%.4f\n",
                mix_ll, base_ll);

    // PASS criteria (derived from real comparisons, not hard-coded):
    //  (a) recovered centers within 0.3 RMSE of truth (well-separated, easy);
    //  (b) mixture clearly beats the single-Gaussian baseline.
    const bool centers_ok  = rmse_centers < 0.30;
    const bool beats_base  = mix_ll > base_ll + 0.5;
    const bool ok = centers_ok && beats_base;

    std::printf("  [check] centers_ok=%d  beats_baseline=%d\n",
                centers_ok ? 1 : 0, beats_base ? 1 : 0);
    std::printf(ok ? "[demo PASS]\n" : "[demo FAIL]\n");
    return ok ? 0 : 1;
}

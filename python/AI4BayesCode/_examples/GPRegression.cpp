// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// GPRegression.cpp
//
// REFERENCE TEMPLATE for generic Gaussian Process regression with
// Elliptical Slice Sampling on latent f + NUTS on kernel hyperparameters
// + NUTS on noise sigma. Uses the vendored libgp kernel subsystem for
// kernel evaluation (stateless-per-pair, stateful-per-hyperparam).
//
// MODEL
// -----
//   y_i | f(x_i), sigma  ~ Normal(f(x_i), sigma^2),   i = 1..N
//   f                     ~ GP(0, K_amplitude_lengthscale(X, X))
//   amplitude             ~ half-Normal(0, sd(y))     (weakly informative)
//   lengthscale           ~ InverseGamma(5, s_med)    (s_med = median pairwise dist)
//   sigma (noise)         ~ Jeffreys  p(sigma) oc 1/sigma
//
// The GP prior K is built via libgp's CovSEiso (can be swapped for any
// other libgp kernel: SE-ARD, Matern32/52, Periodic, etc.). libgp
// kernel object is held as a member of this wrapper and configured
// (via set_loghyper) inside the K_matrix shared_data refresher.
//
// BLOCKS (Gibbs sweep order: hyperparams FIRST, f LAST)
// ------
//   log_amplitude    : nuts_block with positive constraint + half-Normal prior
//   log_lengthscale  : nuts_block with positive constraint + InverseGamma prior
//   log_sigma        : nuts_block with positive constraint + Jeffreys prior
//   f                : elliptical_slice_sampling_block (reads L_prior from
//                      ctx -- the GP PRIOR covariance Cholesky chol(K),
//                      NOT the noisy chol(K+sigma^2 I))
//
// SHARED_DATA DAG
// ---------------
//   y (fixed data), X (fixed data, flat column-major)
//   amplitude, lengthscale, sigma: sampled parameters
//   K_matrix  (derived, N^2 flat): refresher reads X + amplitude + lengthscale
//                                   + libgp cf; invalidated by amp / ell updates
//   L_prior   (derived, N^2 flat): refresher reads K_matrix; chol(K + jitter)
//                                   = GP PRIOR covariance Cholesky used by the
//                                   ESS block; invalidated by amp / ell only
//                                   (NOT sigma -- decouples f / sigma blocks)
//   L_chol    (derived, N^2 flat): refresher reads K_matrix + sigma;
//                                   chol(K + sigma^2 I) used by predict_at;
//                                   invalidated by amp / ell / sigma updates
//   f         : sampled by ESS block
//   prob ... (not applicable for Gaussian-likelihood regression)
//   y_rep     : stochastic refresher at predict time
//
// PREDICT DAG
// -----------
//   X     -> K_matrix
//   K_matrix + sigma -> L_chol
//   L_chol + f -> (no simple refresher; predict_at uses the BartNoise
//                   pattern: wrapper calls kernel routines directly to
//                   build cross-kernel K_star_X and compute posterior
//                   mean/cov at X_new; injects f_star + sigma into a
//                   scratch context, then y_rep stochastic refresher
//                   fires)
//
// LICENSE WARNING
// ---------------
// libgp_kernels is BSD-3; GPL-compatible. AI4BayesCode as a whole is
// GPL-2.0-or-later; the combined work stays GPL-2.0-or-later. The
// BSD-3 attribution is preserved in AI4BayesCode/libgp_kernels/COPYING and
// listed in THIRD_PARTY_LICENSES.md.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("GPRegression")
//   set.seed(2024); N <- 120                       # dense 1-D grid: many points per lengthscale
//   x <- seq(-3, 3, length.out = N)                # covariate grid on [-3, 3]
//   f_true <- sin(3 * x) + 0.5 * x                 # known smooth latent function
//   y <- f_true + rnorm(N, 0, 0.30)               # Gaussian noise, sigma_true = 0.30
//   X <- matrix(x, ncol = 1)                       # X is N x 1
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(GPRegression, X, y, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(GPRegression, X, y, 11L, TRUE)        # X, y, seed=11, keep_history=TRUE
//   m$step(2500); str(m$get_current())             # single chain; f / amplitude / lengthscale / sigma
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2024); N = 120     # dense 1-D grid: many points per lengthscale
//   x = np.linspace(-3.0, 3.0, N)                  # covariate grid on [-3, 3]
//   f_true = np.sin(3.0 * x) + 0.5 * x             # known smooth latent function
//   y = f_true + rng.normal(0.0, 0.30, N)          # Gaussian noise, sigma_true = 0.30
//   X = x.reshape(N, 1)                            # X is N x 1
//   Mod = AI4BayesCode.example("GPRegression")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.GPRegression(X, y, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.GPRegression(X, y, 11, True)           # X, y, seed=11, keep_history=True
//   m.step(2500); print(m.get_current())           # dict: f, amplitude, lengthscale, sigma
// @example:end

// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#ifdef AI4BAYESCODE_RCPP_MODULE
#  include <RcppArmadillo.h>
#else
#  include <armadillo>
#endif

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/elliptical_slice_sampling_block.hpp"

// Vendored libgp kernel subsystem (BSD-3). Unity header includes both
// headers and .cc sources so Rcpp::sourceCpp picks everything up.
#include "libgp_kernels_unity.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
using AI4BayesCode::elliptical_slice_sampling_block;
using AI4BayesCode::elliptical_slice_sampling_block_config;
namespace constraints = AI4BayesCode::constraints;

// ============================================================================
// Natural-scale log-densities for the three hyperparameter blocks.
//
// Per codegen.md Jacobian discipline: user lambdas on natural scale
// contain ONLY the posterior's natural-scale log-density. The wrap
// function in constraints::positive::wrap automatically adds the
// log|dsigma/dlog sigma| = log sigma Jacobian term. So NO log sigma
// in these lambdas.
// ============================================================================

namespace {

// ------- amplitude: half-Normal(0, s) prior --------
// Natural scale sigma_f > 0.
//   log p(sigma_f) = -sigma_f^2 / (2 s^2)   (const dropped)
//   d/dsigma_f     = -sigma_f / s^2
double amp_natural_log_density(const arma::vec& amp_nat,
                               const block_context& ctx,
                               arma::vec* grad_nat) {
    const double a = amp_nat[0];
    const double s = ctx.at("amp_prior_sd")[0];
    const double s2 = s * s;
    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = -a / s2;
    }
    return -0.5 * a * a / s2;
}

// ------- lengthscale: InverseGamma(shape, scale) prior --------
// Natural scale ell > 0.
//   log p(ell) = -(shape + 1) log(ell) - scale / ell
//   d/dell     = -(shape + 1) / ell + scale / ell^2
double ell_natural_log_density(const arma::vec& ell_nat,
                               const block_context& ctx,
                               arma::vec* grad_nat) {
    const double ell = ell_nat[0];
    const double shape = ctx.at("ell_prior_shape")[0];
    const double scale = ctx.at("ell_prior_scale")[0];
    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = -(shape + 1.0) / ell + scale / (ell * ell);
    }
    return -(shape + 1.0) * std::log(ell) - scale / ell;
}

// ------- sigma (noise): Gaussian likelihood + Jeffreys prior --------
// Natural scale sigma > 0.
// Joint:
//   log p(y|f,sigma) + log p(sigma)
//     = -N log(sigma) - 0.5 * SSE / sigma^2  +  (-log sigma)
//     = -(N+1) log(sigma) - 0.5 * SSE / sigma^2
//   d/dsigma = -(N+1)/sigma + SSE / sigma^3
double sigma_natural_log_density(const arma::vec& sigma_nat,
                                 const block_context& ctx,
                                 arma::vec* grad_nat) {
    const double sig = sigma_nat[0];
    const arma::vec& y = ctx.at("y");
    const arma::vec& f = ctx.at("f");
    const double N = static_cast<double>(y.n_elem);
    double sse = 0.0;
    for (std::size_t i = 0; i < y.n_elem; ++i) {
        double d = y[i] - f[i];
        sse += d * d;
    }
    const double s2 = sig * sig;
    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = -(N + 1.0) / sig + sse / (sig * s2);
    }
    return -(N + 1.0) * std::log(sig) - 0.5 * sse / s2;
}

// Pairwise distance helper for InvGamma scale heuristic.
double median_pairwise_distance(const arma::mat& X) {
    const std::size_t N = X.n_rows;
    if (N < 2) return 1.0;
    std::vector<double> dists;
    dists.reserve(N * (N - 1) / 2);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            const double d = arma::norm(X.row(i) - X.row(j), 2);
            dists.push_back(d);
        }
    }
    std::sort(dists.begin(), dists.end());
    const std::size_t m = dists.size() / 2;
    return (dists.size() % 2 == 1) ? dists[m]
                                   : 0.5 * (dists[m - 1] + dists[m]);
}

}  // anonymous namespace

// ============================================================================
// User-facing class exposed to R.
// ============================================================================

class GPRegression {
public:
    GPRegression(const arma::mat& X,
                 const arma::vec& y,
                 int rng_seed,
                 bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          readapt_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0xBF58476D1CE4E5B9ULL}),
          impl_(std::make_unique<composite_block>("GPRegression")),
          keep_history_(keep_history)
    {
        if (X.n_rows != y.n_elem) {
            ai4b::stop("GPRegression: X and y must have matching row counts");
        }
        if (X.n_rows < 2) {
            ai4b::stop("GPRegression: N must be >= 2");
        }
        const std::size_t N = static_cast<std::size_t>(X.n_rows);
        const std::size_t p = static_cast<std::size_t>(X.n_cols);
        N_ = N;
        p_ = p;

        // ---- Install fixed data -----------------------------------------
        arma::vec y_arma(N);
        for (std::size_t i = 0; i < N; ++i) y_arma[i] = y[i];
        impl_->data().set("y", y_arma);

        // Store X as arma::mat (for heuristics + predict) AND cache per-row
        // as Eigen::VectorXd for libgp kernel evaluation.
        X_arma_ = arma::mat(N, p);
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < p; ++j) X_arma_(i, j) = X(i, j);
        }
        X_rows_.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            Eigen::VectorXd row(p);
            for (std::size_t j = 0; j < p; ++j) row[j] = X(i, j);
            X_rows_.push_back(row);
        }
        // X flat for shared_data (column-major)
        arma::vec x_flat = arma::vectorise(X_arma_);
        impl_->data().set("X", x_flat);

        // ---- Prior hyperparameters -------------------------------------
        // amplitude half-Normal(0, sd(y))
        const double sd_y = arma::stddev(y_arma);
        impl_->data().set("amp_prior_sd",
            arma::vec{std::max(sd_y, 0.1)});

        // lengthscale InverseGamma(5, median_pair_dist)
        const double ell_scale = median_pairwise_distance(X_arma_);
        impl_->data().set("ell_prior_shape", arma::vec{5.0});
        impl_->data().set("ell_prior_scale", arma::vec{ell_scale});

        // Initial hyperparam values (sensible data-driven defaults)
        const double amp_init   = std::max(sd_y, 0.1);
        const double ell_init   = ell_scale / 5.0;  // InvGamma posterior-ish mean
        const double sigma_init = 0.3 * sd_y;

        impl_->data().set("amplitude",   arma::vec{amp_init});
        impl_->data().set("lengthscale", arma::vec{ell_init});
        impl_->data().set("sigma",       arma::vec{sigma_init});

        // ---- libgp kernel object ----------------------------------------
        // CovSEiso: params are [log(ell), log(sf)] in that order.
        cf_ = std::make_unique<libgp::CovSEiso>();
        cf_->init(static_cast<int>(p));

        // Initial f seed = zeros
        impl_->data().set("f", arma::vec(N, arma::fill::zeros));

        // Seed K_matrix + L_chol slots (refreshers will populate)
        impl_->data().set("K_matrix",
            arma::vec(N * N, arma::fill::zeros));
        impl_->data().set("L_chol",
            arma::vec(N * N, arma::fill::zeros));
        // L_prior = chol(K + jitter): the GP PRIOR-covariance Cholesky
        // consumed by the elliptical slice sampling block. ESS draws its
        // prior proposal nu = L_prior z ~ N(0, K). The noise sigma^2 must
        // NOT be folded in here (it enters only via the ESS likelihood);
        // doing so over-shrinks f and biases the conditional sigma low.
        // (L_chol = chol(K + sigma^2 I) is kept separately for predict_at.)
        impl_->data().set("L_prior",
            arma::vec(N * N, arma::fill::zeros));

        // ---- Declare Gibbs dependencies -----------------------------
        impl_->data().declare_dependencies("amplitude",
            {"amp_prior_sd"});
        impl_->data().declare_dependencies("lengthscale",
            {"ell_prior_shape", "ell_prior_scale"});
        impl_->data().declare_dependencies("sigma",
            {"y", "f"});
        impl_->data().declare_dependencies("f",
            {"L_prior", "y", "sigma"});

        // ---- Invalidation chain -----------------------------------------
        // amplitude / lengthscale update -> rebuild K_matrix, then both
        // the prior Cholesky L_prior (ESS) and the noisy L_chol (predict).
        impl_->data().declare_invalidates("amplitude",
            {"K_matrix", "L_prior", "L_chol"});
        impl_->data().declare_invalidates("lengthscale",
            {"K_matrix", "L_prior", "L_chol"});
        // sigma update -> only L_chol (noise on diagonal). L_prior = chol(K)
        // does NOT depend on sigma, so it is deliberately NOT invalidated
        // here -- this also decouples the f/sigma Gibbs blocks.
        impl_->data().declare_invalidates("sigma", {"L_chol"});

        // ---- K_matrix refresher (uses libgp) -----------------------
        const libgp::CovSEiso* cf_ptr = cf_.get();
        const std::vector<Eigen::VectorXd>* X_rows_ptr = &X_rows_;
        impl_->data().register_refresher("K_matrix",
            [cf_ptr, X_rows_ptr, N](
                const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const double amp = d.get("amplitude")[0];
                const double ell = d.get("lengthscale")[0];
                // libgp CovSEiso expects [log(ell), log(sf)].
                Eigen::VectorXd hyper(2);
                hyper[0] = std::log(std::max(ell, 1e-10));
                hyper[1] = std::log(std::max(amp, 1e-10));
                auto* cf_mut = const_cast<libgp::CovSEiso*>(cf_ptr);
                cf_mut->set_loghyper(hyper);

                arma::mat K(N, N);
                for (std::size_t i = 0; i < N; ++i) {
                    K(i, i) = cf_mut->get((*X_rows_ptr)[i],
                                           (*X_rows_ptr)[i]);
                    for (std::size_t j = i + 1; j < N; ++j) {
                        double k = cf_mut->get((*X_rows_ptr)[i],
                                                (*X_rows_ptr)[j]);
                        K(i, j) = k;
                        K(j, i) = k;
                    }
                }
                return arma::vectorise(K);
            });

        // ---- L_chol refresher: chol(K + sigma^2 I + jitter I) --------
        const double jitter = 1e-6;
        impl_->data().register_refresher("L_chol",
            [N, jitter](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& K_flat = d.get("K_matrix");
                const double sig = d.get("sigma")[0];
                arma::mat K(const_cast<double*>(K_flat.memptr()), N, N,
                             /*copy_aux_mem=*/false, /*strict=*/true);
                arma::mat M = K;
                M.diag() += sig * sig + jitter;
                arma::mat L;
                if (!arma::chol(L, M, "lower")) {
                    // Should be extremely rare with positive jitter;
                    // retry with larger jitter.
                    M.diag() += 1e-3;
                    arma::chol(L, M, "lower");
                }
                return arma::vectorise(L);
            });

        // ---- L_prior refresher: chol(K + jitter I) -- the GP PRIOR
        //      covariance Cholesky (NO sigma^2). Consumed by the ESS block
        //      as its prior-proposal covariance nu = L_prior z ~ N(0, K).
        impl_->data().register_refresher("L_prior",
            [N, jitter](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& K_flat = d.get("K_matrix");
                arma::mat K(const_cast<double*>(K_flat.memptr()), N, N,
                             /*copy_aux_mem=*/false, /*strict=*/true);
                arma::mat M = K;
                M.diag() += jitter;
                arma::mat L;
                if (!arma::chol(L, M, "lower")) {
                    // Smooth SE kernel on a dense grid can be near-singular;
                    // bump jitter (matches the L_chol fallback policy).
                    M.diag() += 1e-3;
                    arma::chol(L, M, "lower");
                }
                return arma::vectorise(L);
            });

        // Prime K and L so the first step sees consistent state
        impl_->data().refresh_all();

        // ---- Predict DAG (for predict_at) -----------------------------
        impl_->data().declare_data_input("X");
        impl_->data().declare_predict_edges("X", {"K_matrix"});
        impl_->data().declare_predict_edges("K_matrix", {"L_chol"});
        impl_->data().declare_predict_edges("f", {"y_rep"});
        impl_->data().declare_predict_edges("sigma", {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). The GP kernel K is built from the
        //      sampled hyperparameters (amplitude, lengthscale) — they
        //      are generative parents of K_matrix, analogous to the BART
        //      forest -> f_bart edge. Their priors: amplitude ~
        //      half-Normal(0, amp_prior_sd); lengthscale ~
        //      InverseGamma(ell_prior_shape, ell_prior_scale). f ~
        //      GP(0,K) so L_chol is f's generative parent. sigma ~
        //      Jeffreys (no slot). Drawn faded by ai4bayescode_plot_dag.
        impl_->data().declare_context_edges("amp_prior_sd",   {"amplitude"});
        impl_->data().declare_context_edges("ell_prior_shape",{"lengthscale"});
        impl_->data().declare_context_edges("ell_prior_scale",{"lengthscale"});
        impl_->data().declare_context_edges("amplitude",      {"K_matrix"});
        impl_->data().declare_context_edges("lengthscale",    {"K_matrix"});
        impl_->data().declare_context_edges("L_chol",         {"f"});

        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));
        impl_->data().register_stochastic_refresher("y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                const arma::vec& f = d.get("f");
                const double sig = d.get("sigma")[0];
                std::normal_distribution<double> nd(0.0, 1.0);
                arma::vec out(f.n_elem);
                for (std::size_t i = 0; i < f.n_elem; ++i) {
                    out[i] = f[i] + sig * nd(rng);
                }
                return out;
            });

        // ---- Add blocks in Gibbs order ---------------------------------
        // Order: amplitude -> lengthscale -> sigma -> f
        // Reason: hyperparams update first, invalidates K/L, then f
        // samples against fresh K/L.

        {
            nuts_block_config cfg;
            cfg.name        = "amplitude";
            cfg.initial_unc = arma::vec{std::log(amp_init)};
            cfg.constrain   = constraints::positive::constrain;
            cfg.unconstrain = constraints::positive::unconstrain;
            cfg.log_density_grad =
                [](const arma::vec& t_unc, const block_context& ctx,
                   arma::vec* grad) {
                    return constraints::positive::wrap(t_unc, grad,
                        [&](const arma::vec& t_nat, arma::vec* g_nat) {
                            return amp_natural_log_density(t_nat, ctx, g_nat);
                        });
                };
            impl_->add_child(
                std::make_unique<nuts_block>(std::move(cfg)));
        }
        {
            nuts_block_config cfg;
            cfg.name        = "lengthscale";
            cfg.initial_unc = arma::vec{std::log(ell_init)};
            cfg.constrain   = constraints::positive::constrain;
            cfg.unconstrain = constraints::positive::unconstrain;
            cfg.log_density_grad =
                [](const arma::vec& t_unc, const block_context& ctx,
                   arma::vec* grad) {
                    return constraints::positive::wrap(t_unc, grad,
                        [&](const arma::vec& t_nat, arma::vec* g_nat) {
                            return ell_natural_log_density(t_nat, ctx, g_nat);
                        });
                };
            impl_->add_child(
                std::make_unique<nuts_block>(std::move(cfg)));
        }
        {
            nuts_block_config cfg;
            cfg.name        = "sigma";
            cfg.initial_unc = arma::vec{std::log(sigma_init)};
            cfg.constrain   = constraints::positive::constrain;
            cfg.unconstrain = constraints::positive::unconstrain;
            cfg.log_density_grad =
                [](const arma::vec& t_unc, const block_context& ctx,
                   arma::vec* grad) {
                    return constraints::positive::wrap(t_unc, grad,
                        [&](const arma::vec& t_nat, arma::vec* g_nat) {
                            return sigma_natural_log_density(t_nat, ctx, g_nat);
                        });
                };
            impl_->add_child(
                std::make_unique<nuts_block>(std::move(cfg)));
        }
        {
            elliptical_slice_sampling_block_config cfg;
            cfg.name        = "f";
            cfg.N           = N;
            // ESS prior covariance MUST be the GP prior K (not K+sigma^2 I).
            cfg.L_chol_key  = "L_prior";
            cfg.initial_f   = arma::vec(N, arma::fill::zeros);
            cfg.log_lik =
                [](const arma::vec& f, const block_context& ctx) -> double {
                    const arma::vec& y = ctx.at("y");
                    const double sig = ctx.at("sigma")[0];
                    const double s2 = sig * sig;
                    double sse = 0.0;
                    for (std::size_t i = 0; i < y.n_elem; ++i) {
                        double d = y[i] - f[i];
                        sse += d * d;
                    }
                    return -0.5 * sse / s2;
                };
            impl_->add_child(
                std::make_unique<elliptical_slice_sampling_block>(
                    std::move(cfg)));
        }

        if (keep_history_) {
            impl_->set_keep_history(true);
        }
    }

    void step() { step(1); }              // no-arg convenience: one sweep

    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        // Each output is an arma::vec; the frontend converts state_map ->
        // R list / Python dict. f has length N; amplitude / lengthscale /
        // sigma are length-1.
        AI4BayesCode::state_map out;
        out["f"]           = impl_->data().get("f");            // length N
        out["amplitude"]   = impl_->data().get("amplitude");    // length 1
        out["lengthscale"] = impl_->data().get("lengthscale");  // length 1
        out["sigma"]       = impl_->data().get("sigma");        // length 1
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        // Each value in params is an arma::vec (frontend already converted
        // list/dict). Read via .find().
        auto it_f = params.find("f");
        if (it_f != params.end()) {
            const arma::vec& f_new = it_f->second;
            if (static_cast<std::size_t>(f_new.n_elem) != N_) {
                ai4b::stop("GPRegression::set_current: f length mismatch");
            }
            impl_->data().set("f", f_new);
            dynamic_cast<elliptical_slice_sampling_block&>(
                impl_->child(3)).set_current(f_new);
        }
        auto it_amp = params.find("amplitude");
        if (it_amp != params.end()) {
            const double a = it_amp->second[0];
            if (!(a > 0.0)) ai4b::stop("amplitude must be positive");
            dynamic_cast<nuts_block&>(impl_->child(0))
                .set_current(arma::vec{a});
            impl_->data().set("amplitude", arma::vec{a});
            impl_->data().refresh_derived_for("amplitude");
        }
        auto it_ell = params.find("lengthscale");
        if (it_ell != params.end()) {
            const double l = it_ell->second[0];
            if (!(l > 0.0)) ai4b::stop("lengthscale must be positive");
            dynamic_cast<nuts_block&>(impl_->child(1))
                .set_current(arma::vec{l});
            impl_->data().set("lengthscale", arma::vec{l});
            impl_->data().refresh_derived_for("lengthscale");
        }
        auto it_sig = params.find("sigma");
        if (it_sig != params.end()) {
            const double s = it_sig->second[0];
            if (!(s > 0.0)) ai4b::stop("sigma must be positive");
            dynamic_cast<nuts_block&>(impl_->child(2))
                .set_current(arma::vec{s});
            impl_->data().set("sigma", arma::vec{s});
            impl_->data().refresh_derived_for("sigma");
        }
    }

    // predict_at: state_map{"X" = X_new} -> GP posterior mean + variance at
    // X_new given current (f, amplitude, lengthscale, sigma) state, then
    // samples f_star + y_rep. Empty map -> posterior predictive at training X
    // using current f + sigma. Uses predict_rng_ for reproducibility.
    //
    // Backend-neutral I/O (frontend converts state_map <-> R list / Python
    // dict and history_map <-> R list-of-matrices / Python dict-of-arrays):
    //   * INPUT  : new_data["X"] is a VECTORISED N_new*p arma::vec, column
    //              major (element (i,j) at index i + j*N_new), matching how X
    //              is stored in data() (arma::vectorise of the N x p design).
    //              R/Python callers pass as.vector(X_new) / X_new flattened
    //              column-major.
    //   * OUTPUT : every key is an arma::mat. keep_history = FALSE returns
    //              1-row matrices (single predict at the current draw);
    //              keep_history = TRUE returns n_draws-row matrices (posterior
    //              predictive over all draws).
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        // ---- Parse optional X (vectorised N_new*p, column-major) ----------
        bool has_X = false;
        arma::vec x_flat;
        std::size_t N_new = 0;
        for (const auto& kv : new_data) {
            if (kv.first != "X")
                ai4b::stop("GPRegression::predict_at: unknown key '%s'",
                           kv.first.c_str());
        }
        auto it_X = new_data.find("X");
        if (it_X != new_data.end()) {
            x_flat = it_X->second;
            if (p_ == 0 || x_flat.n_elem % p_ != 0)
                ai4b::stop("GPRegression::predict_at: X must be vectorised "
                           "N_new*p (column-major)");
            N_new = x_flat.n_elem / p_;
            has_X = true;
        }

        AI4BayesCode::history_map out;

        if (!has_X) {
            if (keep_history_) {
                // History mode at training X: per-draw y_rep_d = f_d +
                // sigma_d * N(0,1). f and sigma are block names so we
                // could go through impl_->predict_at, but it's cheaper
                // to just sample directly per draw.
                AI4BayesCode::history_map hist = impl_->get_history();
                const arma::mat& f_hist     = hist.at("f");      // n_draws x N
                const arma::mat& sigma_hist = hist.at("sigma");  // n_draws x 1
                const std::size_t n_draws = f_hist.n_rows;
                const std::size_t N_local = f_hist.n_cols;
                arma::mat yrep_mat(n_draws, N_local);
                arma::mat f_mean_mat(n_draws, N_local);
                std::normal_distribution<double> nd(0.0, 1.0);
                for (std::size_t d = 0; d < n_draws; ++d) {
                    const double sigma_d = sigma_hist(d, 0);
                    for (std::size_t i = 0; i < N_local; ++i) {
                        f_mean_mat(d, i) = f_hist(d, i);
                        yrep_mat(d, i)   = f_hist(d, i) + sigma_d * nd(predict_rng_);
                    }
                }
                out.emplace("f_mean", std::move(f_mean_mat));
                out.emplace("y_rep",  std::move(yrep_mat));
                return out;
            }
            // Stateful: return posterior-predictive y_rep at training X
            // using current f + sigma.
            const arma::vec& f = impl_->data().get("f");
            const double sig   = impl_->data().get("sigma")[0];
            std::normal_distribution<double> nd(0.0, 1.0);
            arma::mat f_mean_mat(1, f.n_elem);
            arma::mat yrep_mat(1, f.n_elem);
            for (std::size_t i = 0; i < f.n_elem; ++i) {
                f_mean_mat(0, i) = f[i];
                yrep_mat(0, i)   = f[i] + sig * nd(predict_rng_);
            }
            out.emplace("f_mean", std::move(f_mean_mat));
            out.emplace("y_rep",  std::move(yrep_mat));
            return out;
        }

        // ---- Reconstruct per-row Eigen vectors from vectorised X ----------
        // x_flat is column-major N_new x p: X_new(i,j) = x_flat[i + j*N_new].
        std::vector<Eigen::VectorXd> X_new_rows(N_new);
        for (std::size_t i = 0; i < N_new; ++i) {
            Eigen::VectorXd row(p_);
            for (std::size_t j = 0; j < p_; ++j)
                row[j] = x_flat[i + j * N_new];
            X_new_rows[i] = row;
        }

        if (keep_history_) {
            // new-X + history: per-draw GP conditioning at X_new using
            // (amp_d, ell_d, sigma_d, f_d) from history. Rebuild K_train,
            // L = chol(K + sigma^2 I), alpha = L^-T L^-1 f_d, then
            // mu_star_d = K_star_X * alpha, sample y_rep_d.
            AI4BayesCode::history_map hist = impl_->get_history();
            const arma::mat& amp_hist   = hist.at("amplitude");
            const arma::mat& ell_hist   = hist.at("lengthscale");
            const arma::mat& sigma_hist = hist.at("sigma");
            const arma::mat& f_hist     = hist.at("f");
            const std::size_t n_draws = amp_hist.n_rows;

            auto* cf_mut = const_cast<libgp::CovSEiso*>(cf_.get());

            arma::mat mu_star_mat (n_draws, N_new);
            arma::mat f_star_mat  (n_draws, N_new);
            arma::mat yrep_mat    (n_draws, N_new);
            std::normal_distribution<double> nd(0.0, 1.0);

            for (std::size_t d = 0; d < n_draws; ++d) {
                const double amp_d = amp_hist(d, 0);
                const double ell_d = ell_hist(d, 0);
                const double sig_d = sigma_hist(d, 0);
                Eigen::VectorXd hyper(2);
                hyper[0] = std::log(std::max(ell_d, 1e-10));
                hyper[1] = std::log(std::max(amp_d, 1e-10));
                cf_mut->set_loghyper(hyper);

                // Rebuild K_train_d and L_d = chol(K + sigma_d^2 I)
                arma::mat K_train_d(N_, N_);
                for (std::size_t i = 0; i < N_; ++i) {
                    for (std::size_t j = i; j < N_; ++j) {
                        double k = cf_mut->get(X_rows_[i], X_rows_[j]);
                        K_train_d(i, j) = k;
                        K_train_d(j, i) = k;
                    }
                }
                arma::mat KsigI = K_train_d + (sig_d * sig_d) * arma::eye(N_, N_);
                arma::mat L_d;
                if (!arma::chol(L_d, KsigI, "lower")) {
                    KsigI.diag() += 1e-8;
                    arma::chol(L_d, KsigI, "lower");
                }

                arma::mat K_star_X(N_new, N_);
                arma::mat K_star_star(N_new, N_new);
                for (std::size_t i = 0; i < N_new; ++i) {
                    for (std::size_t j = 0; j < N_; ++j)
                        K_star_X(i, j) = cf_mut->get(X_new_rows[i], X_rows_[j]);
                    K_star_star(i, i) = cf_mut->get(X_new_rows[i], X_new_rows[i]);
                    for (std::size_t j = i + 1; j < N_new; ++j) {
                        double k = cf_mut->get(X_new_rows[i], X_new_rows[j]);
                        K_star_star(i, j) = k;
                        K_star_star(j, i) = k;
                    }
                }

                arma::vec f_d = f_hist.row(d).t();
                arma::vec alpha = arma::solve(arma::trimatu(L_d.t()),
                                    arma::solve(arma::trimatl(L_d), f_d));
                arma::vec mu_star = K_star_X * alpha;
                arma::mat V = arma::solve(arma::trimatl(L_d), K_star_X.t());
                arma::mat Sigma_star = K_star_star - V.t() * V;
                Sigma_star.diag() += 1e-9;

                arma::mat L_star;
                arma::vec f_star(N_new);
                if (arma::chol(L_star, Sigma_star, "lower")) {
                    arma::vec z(N_new);
                    for (std::size_t i = 0; i < N_new; ++i) z[i] = nd(predict_rng_);
                    f_star = mu_star + L_star * z;
                } else {
                    for (std::size_t i = 0; i < N_new; ++i) {
                        const double sd = std::sqrt(std::max(Sigma_star(i, i), 0.0));
                        f_star[i] = mu_star[i] + sd * nd(predict_rng_);
                    }
                }
                for (std::size_t i = 0; i < N_new; ++i) {
                    mu_star_mat(d, i) = mu_star[i];
                    f_star_mat (d, i) = f_star[i];
                    yrep_mat   (d, i) = f_star[i] + sig_d * nd(predict_rng_);
                }
            }
            out.emplace("f_mean", std::move(mu_star_mat));
            out.emplace("f_star", std::move(f_star_mat));
            out.emplace("y_rep",  std::move(yrep_mat));
            return out;
        }

        // ---- new-X + stateful (current draw) ------------------------------
        const arma::vec& f_cur = impl_->data().get("f");
        const arma::vec& L_flat = impl_->data().get("L_chol");
        const double sig = impl_->data().get("sigma")[0];

        arma::mat L(const_cast<double*>(L_flat.memptr()), N_, N_,
                     /*copy_aux_mem=*/false, /*strict=*/true);

        // Configure libgp cf with current hyperparams
        const double amp = impl_->data().get("amplitude")[0];
        const double ell = impl_->data().get("lengthscale")[0];
        Eigen::VectorXd hyper(2);
        hyper[0] = std::log(std::max(ell, 1e-10));
        hyper[1] = std::log(std::max(amp, 1e-10));
        auto* cf_mut = const_cast<libgp::CovSEiso*>(cf_.get());
        cf_mut->set_loghyper(hyper);

        // Build K_star_X (N_new x N_) and K_star_star (N_new x N_new)
        arma::mat K_star_X(N_new, N_);
        arma::mat K_star_star(N_new, N_new);
        for (std::size_t i = 0; i < N_new; ++i) {
            for (std::size_t j = 0; j < N_; ++j) {
                K_star_X(i, j) = cf_mut->get(X_new_rows[i], X_rows_[j]);
            }
            K_star_star(i, i) = cf_mut->get(X_new_rows[i], X_new_rows[i]);
            for (std::size_t j = i + 1; j < N_new; ++j) {
                double k = cf_mut->get(X_new_rows[i], X_new_rows[j]);
                K_star_star(i, j) = k;
                K_star_star(j, i) = k;
            }
        }

        // Posterior mean: mu_star = K_star_X @ (L L^T)^{-1} @ f
        // = K_star_X @ solve(L^T, solve(L, f))
        arma::vec alpha = arma::solve(arma::trimatu(L.t()),
                            arma::solve(arma::trimatl(L), f_cur));
        arma::vec mu_star = K_star_X * alpha;

        // Posterior variance: Sigma_star = K_star_star - K_star_X @ (L L^T)^{-1} @ K_star_X^T
        arma::mat V = arma::solve(arma::trimatl(L), K_star_X.t());
        arma::mat Sigma_star = K_star_star - V.t() * V;

        // Posterior variance diagonal (for returning SD per test point)
        arma::vec f_star_sd(N_new);
        for (std::size_t i = 0; i < N_new; ++i)
            f_star_sd[i] = std::sqrt(std::max(Sigma_star(i, i), 0.0));

        // Sample f_star ~ N(mu_star, Sigma_star) via Cholesky
        arma::mat Sigma_star_reg = Sigma_star;
        Sigma_star_reg.diag() += 1e-9;  // jitter for numerical stability
        arma::mat L_star;
        bool chol_ok = arma::chol(L_star, Sigma_star_reg, "lower");
        arma::vec f_star(N_new);
        std::normal_distribution<double> nd(0.0, 1.0);
        if (chol_ok) {
            arma::vec z(N_new);
            for (std::size_t i = 0; i < N_new; ++i) z[i] = nd(predict_rng_);
            f_star = mu_star + L_star * z;
        } else {
            // Fallback: sample diagonal-only (independent)
            for (std::size_t i = 0; i < N_new; ++i)
                f_star[i] = mu_star[i] + f_star_sd[i] * nd(predict_rng_);
        }

        // y_rep = f_star + sigma * N(0,1)
        arma::vec y_rep(N_new);
        for (std::size_t i = 0; i < N_new; ++i) {
            y_rep[i] = f_star[i] + sig * nd(predict_rng_);
        }

        // Pack each output as a 1-row matrix (single predict at current draw).
        arma::mat f_mean_mat (1, N_new);
        arma::mat f_sd_mat   (1, N_new);
        arma::mat f_star_mat (1, N_new);
        arma::mat yrep_mat   (1, N_new);
        for (std::size_t i = 0; i < N_new; ++i) {
            f_mean_mat (0, i) = mu_star[i];
            f_sd_mat   (0, i) = f_star_sd[i];
            f_star_mat (0, i) = f_star[i];
            yrep_mat   (0, i) = y_rep[i];
        }
        out.emplace("f_mean", std::move(f_mean_mat));
        out.emplace("f_sd",   std::move(f_sd_mat));
        out.emplace("f_star", std::move(f_star_mat));
        out.emplace("y_rep",  std::move(yrep_mat));
        return out;
    }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    /// 7th R-level method: re-tune NUTS metric (mass matrix + step size +
    /// dual averaging) without advancing chain state. Available because
    /// the composite contains NUTS-family children. See system_design.md
    /// §13 NUTS-family + validator.md §24.
    void readapt_NUTS(int n, bool reset = false) {
        if (n < 0) {
            ai4b::stop("readapt_NUTS: n must be non-negative");
        }
        impl_->readapt_NUTS(static_cast<std::size_t>(n),
                            reset, readapt_rng_);
    }


private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_; // readapt_NUTS() advances it (7th method)
    std::unique_ptr<composite_block> impl_;
    std::size_t                      N_ = 0;
    std::size_t                      p_ = 0;
    bool                             keep_history_ = false;

    // libgp kernel object (long-lived; configured by K-matrix refresher)
    std::unique_ptr<libgp::CovSEiso> cf_;
    arma::mat                        X_arma_;
    std::vector<Eigen::VectorXd>     X_rows_;
};

// ============================================================================
// Rcpp module
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(GPRegression_module) {
    Rcpp::class_<GPRegression>("GPRegression")
        .constructor<arma::mat, arma::vec, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::mat, arma::vec, int, bool>(
            "Construct GP regression with X (N x p), y (length N), seed, "
            "keep_history. Gibbs sweep = amplitude, lengthscale, sigma, f.")
        .method("step", (void (GPRegression::*)())    &GPRegression::step, "Run one sweep.")
        .method("step", (void (GPRegression::*)(int)) &GPRegression::step, "Run n sweeps.")
        .method("get_current", &GPRegression::get_current)
        .method("set_current", &GPRegression::set_current)
        .method("predict_at",  &GPRegression::predict_at)
        .method("get_dag",     &GPRegression::get_dag)
        .method("get_history", &GPRegression::get_history)
        .method("readapt_NUTS", &GPRegression::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(GPRegression, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<GPRegression>(m, "GPRegression")
        .def(pybind11::init<arma::mat, arma::vec, int, bool>(),
             pybind11::arg("X"),
             pybind11::arg("y"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step", (void (GPRegression::*)())    &GPRegression::step, "Run one sweep.")
        .def("step", (void (GPRegression::*)(int)) &GPRegression::step, pybind11::arg("n_steps"))
        .def("get_current",  &GPRegression::get_current)
        .def("set_current",  &GPRegression::set_current, pybind11::arg("params"))
        .def("predict_at",   &GPRegression::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &GPRegression::get_dag)
        .def("get_history",  &GPRegression::get_history)
        .def("readapt_NUTS", &GPRegression::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false);
}
#endif

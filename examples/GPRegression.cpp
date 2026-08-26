// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// GPRegression.cpp
//
// REFERENCE TEMPLATE for generic Gaussian Process regression with a
// GAUSSIAN observation likelihood, sampled through the MARGINAL
// likelihood: the latent f is integrated out analytically and only the
// three covariance hyperparameters are sampled, jointly, by one
// joint_nuts_block. Uses the vendored libgp kernel subsystem for kernel
// evaluation at predict time.
//
// MODEL
// -----
//   y_i | f(x_i), sigma  ~ Normal(f(x_i), sigma^2),   i = 1..N
//   f                     ~ GP(0, K_amplitude_lengthscale(X, X))
//   amplitude             ~ half-Normal(0, sd(y))     (weakly informative)
//   lengthscale           ~ InverseGamma(5, s_med)    (s_med = median pairwise dist)
//   sigma (noise)         ~ Jeffreys  p(sigma) oc 1/sigma
//
// A Gaussian likelihood is the one case where f has a closed-form
// integral, so the sampler never represents it:
//
//   p(y | amplitude, lengthscale, sigma) = N(y | 0, K + sigma^2 I)
//
// This is the architecture every reference GP library uses for Gaussian
// observations (Stan, libgp, GaussianProcesses, GPy, GPflow). It removes
// the f <-> (amplitude, lengthscale) coupling that makes the centred
// latent parameterisation mix badly, and it removes an entire class of
// silent bug: with an explicit f there is a log p(f | amplitude,
// lengthscale) term that no block owns by default, and a hyperparameter
// whose log-density omits it is sampled from its PRIOR while the fit and
// every R-hat still look healthy. See
// include/AI4BayesCode/elliptical_slice_sampling_block.hpp.
//
// For a NON-Gaussian likelihood f cannot be integrated out; use the
// whitened ESS route in examples/GPClassification.cpp instead.
//
// BLOCKS
// ------
//   hyper : ONE joint_nuts_block, sub_params
//           [{ "amplitude", 1, POSITIVE },
//            { "lengthscale", 1, POSITIVE },
//            { "sigma", 1, POSITIVE }].
//           Its joint log-density is the FULL natural-scale log
//           p(amplitude, lengthscale, sigma | y) -- the marginal
//           likelihood plus the three priors, each term appearing exactly
//           once. joint_nuts_block adds the three POSITIVE-slice
//           Jacobians internally; the oracle must NOT add them.
//           The three are sampled jointly because amplitude, lengthscale
//           and sigma trade off against each other along a ridge in the
//           marginal likelihood (a long lengthscale with large amplitude
//           and small sigma fits nearly the same data as the reverse).
//
// SHARED_DATA DAG
// ---------------
//   y (fixed data), X (fixed data, flat column-major)
//   X_sqdist  (fixed, N^2 flat): pairwise squared distances ||x_i - x_j||^2,
//                                 built once; the log-density rebuilds K at
//                                 each proposed (amplitude, lengthscale) from
//                                 this, so the O(N^2 p) distance work does
//                                 not repeat inside a NUTS trajectory
//   amplitude, lengthscale, sigma: sampled parameters (sub-outputs of `hyper`)
//   K_matrix  (derived, N^2 flat): refresher reads X + amplitude + lengthscale
//                                   + libgp cf; invalidated by amp / ell updates
//   L_chol    (derived, N^2 flat): refresher reads K_matrix + sigma;
//                                   chol(K + sigma^2 I); invalidated by
//                                   amp / ell / sigma updates
//   f_mean    (derived, N):        posterior mean of the latent f at the
//                                   training X, K (K + sigma^2 I)^-1 y
//   y_rep     : stochastic refresher at predict time
//
// PREDICT DAG
// -----------
//   X     -> K_matrix
//   K_matrix + sigma -> L_chol
//   L_chol -> f_mean -> y_rep
//   For a NEW X, predict_at builds the cross-kernel K_star_X directly with
//   libgp (the BartNoise pattern) and applies the GP predictive equations
//   (Rasmussen & Williams Eq. 2.23-2.24) rather than going through a
//   refresher.
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
//   # ---- Parallel chains + convergence diagnosis (default) ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(GPRegression, X, y, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   print(ai4bayescode_rhat_summary(run))          # CROSS-chain R-hat / ESS
//   ai4bayescode_diagnose(run$histories[[1]])      # chain 1: summary + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(GPRegression, X, y, 11L, TRUE)        # X, y, seed=11, keep_history=TRUE
//   m$step(2500); str(m$get_current())             # amplitude / lengthscale / sigma
//   # The latent f is integrated out, so it is recovered at predict time:
//   fit <- m$predict_at(list())                    # f_mean / f_sd / f_star / y_rep at training X
//   plot(x, y); lines(x, fit$f_mean[1, ], col = "red")
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2024); N = 120     # dense 1-D grid: many points per lengthscale
//   x = np.linspace(-3.0, 3.0, N)                  # covariate grid on [-3, 3]
//   f_true = np.sin(3.0 * x) + 0.5 * x             # known smooth latent function
//   y = f_true + rng.normal(0.0, 0.30, N)          # Gaussian noise, sigma_true = 0.30
//   X = x.reshape(N, 1)                            # X is N x 1
//   Mod = AI4BayesCode.example("GPRegression")
//   # ---- Parallel chains + diagnosis (default) ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.GPRegression(X, y, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   print(AI4BayesCode.rhat_summary(chains))   # CROSS-chain R-hat / ESS
//   AI4BayesCode.diagnose(chains[0]["hist"])   # chain 1: summary + plots
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.GPRegression(X, y, 11, True)           # X, y, seed=11, keep_history=True
//   m.step(2500); print(m.get_current())           # dict: amplitude, lengthscale, sigma
//   fit = m.predict_at({})                         # f_mean / f_sd / f_star / y_rep at training X
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
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/rcpp_predict_guard.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

// Vendored libgp kernel subsystem (BSD-3). Unity header includes both
// headers and .cc sources so Rcpp::sourceCpp picks everything up.
#include "libgp_kernels_unity.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;
namespace constraints = AI4BayesCode::constraints;

namespace {

// Diagonal jitter added to every covariance before Cholesky. A smooth SE
// kernel on a dense grid is numerically singular without it.
constexpr double GP_JITTER = 1e-8;

// ============================================================================
// Joint natural-scale log-density for (amplitude, lengthscale, sigma).
//
// theta_cat = [amplitude, lengthscale, sigma], all on the NATURAL scale.
// joint_nuts_block applies the three POSITIVE transforms and adds the three
// log|Jacobian| terms itself (constraints.hpp), so this oracle contains ONLY
// the natural-scale model terms and returns d lp / d(natural).
//
//   Ky      = K(amplitude, lengthscale) + sigma^2 I
//   K_ij    = amplitude^2 exp(-r_ij^2 / (2 lengthscale^2)),  r_ij = ||x_i - x_j||
//             (identical to libgp CovSEiso: sf2 * exp(-0.5 * ||dx/ell||^2))
//
//   log p(y | theta) = -0.5 y' Ky^-1 y - 0.5 log|Ky|          [const dropped]
//   log p(amplitude)   = -amplitude^2 / (2 s^2)               half-Normal(0, s)
//   log p(lengthscale) = -(a+1) log(ell) - b / ell            InverseGamma(a, b)
//   log p(sigma)       = -log(sigma)                          Jeffreys
//
// Gradient of the marginal likelihood -- Rasmussen & Williams Sec.5.5 Eq. (5.9):
//
//   d/dtheta = 0.5 tr((alpha alpha' - Ky^-1) dKy/dtheta),   alpha = Ky^-1 y
//
// with
//   dKy/d amplitude   = 2 K / amplitude
//   dKy/d lengthscale = K .* r^2 / lengthscale^3
//   dKy/d sigma       = 2 sigma I      ->  0.5 tr(W * 2 sigma I) = sigma tr(W)
//
// K is rebuilt HERE, at the PROPOSED hyperparameters: the K_matrix refresher
// only fires at block boundaries, never inside a NUTS trajectory. It is built
// analytically from the cached squared distances rather than through the libgp
// object, because libgp's CovSEiso is stateful (set_loghyper) and mutating it
// mid-trajectory would desynchronise it from the K_matrix refresher.
// ============================================================================
double gp_marginal_joint_log_density(const arma::vec& theta_cat,
                                     const block_context& ctx,
                                     arma::vec* grad_nat) {
    if (theta_cat.n_elem != 3) {
        ai4b::stop("GPRegression: joint log-density expects dim 3, got %d",
                   static_cast<int>(theta_cat.n_elem));
    }
    const double amp = theta_cat[0];
    const double ell = theta_cat[1];
    const double sig = theta_cat[2];

    const double neg_inf = -std::numeric_limits<double>::infinity();
    if (!(amp > 0.0) || !(ell > 0.0) || !(sig > 0.0) ||
        !std::isfinite(amp) || !std::isfinite(ell) || !std::isfinite(sig)) {
        return neg_inf;                     // valid NUTS reject; grad unused
    }

    const arma::vec& y       = ctx.at("y");
    const arma::vec& d2_flat = ctx.at("X_sqdist");
    const std::size_t N = y.n_elem;

    const arma::mat R2(const_cast<double*>(d2_flat.memptr()), N, N,
                       /*copy_aux_mem=*/false, /*strict=*/true);

    // ---- Ky = K + sigma^2 I ------------------------------------------------
    const arma::mat K  = (amp * amp) * arma::exp(R2 / (-2.0 * ell * ell));
    arma::mat       Ky = K;
    Ky.diag() += sig * sig + GP_JITTER;

    arma::mat L;
    if (!arma::chol(L, Ky, "lower")) {
        return neg_inf;                     // reject rather than fudge the target
    }
    const arma::vec alpha = arma::solve(arma::trimatu(L.t()),
                              arma::solve(arma::trimatl(L), y));
    double log_det = 0.0;
    for (std::size_t i = 0; i < N; ++i) log_det += std::log(L(i, i));
    log_det *= 2.0;

    // ---- Priors ------------------------------------------------------------
    const double s_amp   = ctx.at("amp_prior_sd")[0];
    const double s_amp2  = s_amp * s_amp;
    const double ig_a    = ctx.at("ell_prior_shape")[0];
    const double ig_b    = ctx.at("ell_prior_scale")[0];

    const double lp = -0.5 * arma::dot(y, alpha) - 0.5 * log_det
                    + (-0.5 * amp * amp / s_amp2)
                    + (-(ig_a + 1.0) * std::log(ell) - ig_b / ell)
                    + (-std::log(sig));

    if (grad_nat) {
        grad_nat->set_size(3);

        // W = alpha alpha' - Ky^-1, built from L (no second Cholesky).
        const arma::mat Linv = arma::inv(arma::trimatl(L));
        const arma::mat W    = alpha * alpha.t() - Linv.t() * Linv;

        // Both W and dKy/dtheta are symmetric, so tr(W dK) = accu(W % dK).
        const arma::mat dK_damp = (2.0 / amp) * K;
        const arma::mat dK_dell = (K % R2) / (ell * ell * ell);

        (*grad_nat)[0] = 0.5 * arma::accu(W % dK_damp) - amp / s_amp2;
        (*grad_nat)[1] = 0.5 * arma::accu(W % dK_dell)
                       + (-(ig_a + 1.0) / ell + ig_b / (ell * ell));
        (*grad_nat)[2] = sig * arma::trace(W) - 1.0 / sig;
    }
    return lp;
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

class GPRegression : public AI4BayesCode::kernel_control_mixin<GPRegression> {
    friend class AI4BayesCode::kernel_control_mixin<GPRegression>;
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

        // Pairwise squared distances, built once. The joint log-density
        // rebuilds K from these at every proposed (amplitude, lengthscale),
        // so this O(N^2 p) loop must not sit inside a NUTS trajectory.
        arma::mat R2(N, N, arma::fill::zeros);
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                const double d2 =
                    arma::accu(arma::square(X_arma_.row(i) - X_arma_.row(j)));
                R2(i, j) = d2;
                R2(j, i) = d2;
            }
        }
        impl_->data().set("X_sqdist", arma::vectorise(R2));

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

        // Seed derived slots (refreshers will populate)
        impl_->data().set("K_matrix", arma::vec(N * N, arma::fill::zeros));
        impl_->data().set("L_chol",   arma::vec(N * N, arma::fill::zeros));
        impl_->data().set("f_mean",   arma::vec(N,     arma::fill::zeros));

        // ---- Declare Gibbs dependencies -----------------------------
        // ONE block, so this is the whole Gibbs graph. `hyper` is the block
        // name; amplitude / lengthscale / sigma are its sub-outputs.
        impl_->data().declare_dependencies("hyper",
            {"y", "X_sqdist", "amp_prior_sd",
             "ell_prior_shape", "ell_prior_scale"});

        // ---- Invalidation chain -----------------------------------------
        // Keyed on the BLOCK name: amplitude / lengthscale / sigma are
        // sub-outputs of `hyper`, not children of the composite, and the
        // composite's Gibbs-DAG guard rejects non-child keys.
        impl_->data().declare_invalidates("hyper",
            {"K_matrix", "L_chol", "f_mean"});

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
        impl_->data().register_refresher("L_chol",
            [N](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& K_flat = d.get("K_matrix");
                const double sig = d.get("sigma")[0];
                arma::mat K(const_cast<double*>(K_flat.memptr()), N, N,
                             /*copy_aux_mem=*/false, /*strict=*/true);
                arma::mat M = K;
                M.diag() += sig * sig + GP_JITTER;
                arma::mat L;
                if (!arma::chol(L, M, "lower")) {
                    M.diag() += 1e-3;
                    arma::chol(L, M, "lower");
                }
                return arma::vectorise(L);
            });

        // ---- f_mean refresher: E[f | y, theta] = K (K + sigma^2 I)^-1 y
        //      The latent f is integrated out of the sampler, so this is
        //      where the smooth fit comes back (Rasmussen & Williams
        //      Eq. 2.23 evaluated at the training inputs).
        impl_->data().register_refresher("f_mean",
            [N](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& K_flat = d.get("K_matrix");
                const arma::vec& L_flat = d.get("L_chol");
                const arma::vec& y_vec  = d.get("y");
                arma::mat K(const_cast<double*>(K_flat.memptr()), N, N,
                             false, true);
                arma::mat L(const_cast<double*>(L_flat.memptr()), N, N,
                             false, true);
                const arma::vec alpha = arma::solve(arma::trimatu(L.t()),
                                          arma::solve(arma::trimatl(L), y_vec));
                return K * alpha;
            });

        // Prime the derived slots so the first step sees consistent state
        impl_->data().refresh_all();

        // ---- Predict DAG (for predict_at) -----------------------------
        impl_->data().declare_data_input("X");
        impl_->data().declare_predict_edges("X",        {"K_matrix"});
        impl_->data().declare_predict_edges("K_matrix", {"L_chol"});
        impl_->data().declare_predict_edges("L_chol",   {"f_mean"});
        impl_->data().declare_predict_edges("f_mean",   {"y_rep"});
        impl_->data().declare_predict_edges("sigma",    {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). The GP kernel K is built from the
        //      sampled hyperparameters (amplitude, lengthscale) -- they
        //      are generative parents of K_matrix, analogous to the BART
        //      forest -> f_bart edge. Their priors: amplitude ~
        //      half-Normal(0, amp_prior_sd); lengthscale ~
        //      InverseGamma(ell_prior_shape, ell_prior_scale); sigma ~
        //      Jeffreys (no slot). Drawn faded by ai4bayescode_plot_dag.
        impl_->data().declare_context_edges("amp_prior_sd",   {"amplitude"});
        impl_->data().declare_context_edges("ell_prior_shape",{"lengthscale"});
        impl_->data().declare_context_edges("ell_prior_scale",{"lengthscale"});
        impl_->data().declare_context_edges("amplitude",      {"K_matrix"});
        impl_->data().declare_context_edges("lengthscale",    {"K_matrix"});
        impl_->data().declare_context_edges("K_matrix",       {"f_mean"});

        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));
        // y_rep at the training inputs: draw the latent f from its Gaussian
        // posterior N(f_mean, K - K Ky^-1 K), then add observation noise.
        // Using f_mean alone would understate the predictive spread.
        impl_->data().register_stochastic_refresher("y_rep",
            [N](const AI4BayesCode::shared_data_t& d,
                std::mt19937_64& rng) {
                const arma::vec& mu     = d.get("f_mean");
                const arma::vec& K_flat = d.get("K_matrix");
                const arma::vec& L_flat = d.get("L_chol");
                const double sig = d.get("sigma")[0];
                arma::mat K(const_cast<double*>(K_flat.memptr()), N, N,
                             false, true);
                arma::mat L(const_cast<double*>(L_flat.memptr()), N, N,
                             false, true);
                const arma::mat V = arma::solve(arma::trimatl(L), K);
                arma::mat Sigma = K - V.t() * V;
                Sigma.diag() += sig * sig + GP_JITTER;
                std::normal_distribution<double> nd(0.0, 1.0);
                arma::vec z(N);
                for (std::size_t i = 0; i < N; ++i) z[i] = nd(rng);
                arma::mat Ls;
                if (arma::chol(Ls, Sigma, "lower")) return arma::vec(mu + Ls * z);
                arma::vec out(N);
                for (std::size_t i = 0; i < N; ++i)
                    out[i] = mu[i] + std::sqrt(std::max(Sigma(i, i), 0.0)) * z[i];
                return out;
            });

        // ---- ONE joint_nuts_block over (amplitude, lengthscale, sigma) ----
        {
            joint_nuts_block_config cfg;
            cfg.name = "hyper";
            cfg.sub_params.push_back(joint_nuts_sub_param{
                "amplitude",   1, joint_constraint::POSITIVE });
            cfg.sub_params.push_back(joint_nuts_sub_param{
                "lengthscale", 1, joint_constraint::POSITIVE });
            cfg.sub_params.push_back(joint_nuts_sub_param{
                "sigma",       1, joint_constraint::POSITIVE });
            // initial_cat is NATURAL-scale.
            cfg.initial_cat = arma::vec{ amp_init, ell_init, sigma_init };
            cfg.log_density_grad = &gp_marginal_joint_log_density;
            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
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
        // The latent f is integrated out, so the sampled state is the three
        // covariance hyperparameters. Recover f via predict_at().
        AI4BayesCode::state_map out;
        out["amplitude"]   = impl_->data().get("amplitude");    // length 1
        out["lengthscale"] = impl_->data().get("lengthscale");  // length 1
        out["sigma"]       = impl_->data().get("sigma");        // length 1
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto& jblk = dynamic_cast<joint_nuts_block&>(impl_->child(0));
        arma::vec cat_new = jblk.current();   // [amplitude, lengthscale, sigma]
        bool touched = false;

        auto assign = [&](const char* key, std::size_t idx) {
            auto it = params.find(key);
            if (it == params.end()) return;
            const double v = it->second[0];
            if (!(v > 0.0)) ai4b::stop("GPRegression: %s must be positive", key);
            cat_new[idx] = v;
            touched = true;
        };
        assign("amplitude",   0);
        assign("lengthscale", 1);
        assign("sigma",       2);

        for (const auto& kv : params) {
            if (kv.first != "amplitude" && kv.first != "lengthscale" &&
                kv.first != "sigma") {
                ai4b::stop("GPRegression::set_current: unknown key '%s' "
                           "(the latent f is integrated out)",
                           kv.first.c_str());
            }
        }

        if (touched) {
            jblk.set_current(cat_new);
            impl_->data().set("amplitude",   arma::vec{cat_new[0]});
            impl_->data().set("lengthscale", arma::vec{cat_new[1]});
            impl_->data().set("sigma",       arma::vec{cat_new[2]});
            impl_->data().refresh_derived_for("amplitude");
            impl_->data().refresh_derived_for("lengthscale");
            impl_->data().refresh_derived_for("sigma");
        }
    }

    // predict_at: state_map{"X" = X_new} -> GP posterior mean + variance at
    // X_new given the current (amplitude, lengthscale, sigma) draw, then
    // samples f_star + y_rep. Empty map -> the same quantities at the
    // training X. Uses predict_rng_ for reproducibility.
    //
    // Because f is integrated out, both branches condition on y directly
    // through Rasmussen & Williams Eq. 2.23-2.24
    //     mu_star    = K_star_X (K + sigma^2 I)^-1 y
    //     Sigma_star = K_star_star - K_star_X (K + sigma^2 I)^-1 K_star_X'
    // rather than on a sampled latent vector.
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
        for (const auto& kv : new_data) {
            if (kv.first != "X")
                ai4b::stop("GPRegression::predict_at: unknown key '%s'",
                           kv.first.c_str());
        }
        std::vector<Eigen::VectorXd> X_new_rows;
        bool has_X = false;
        auto it_X = new_data.find("X");
        if (it_X != new_data.end()) {
            const arma::vec& x_flat = it_X->second;
            if (p_ == 0 || x_flat.n_elem % p_ != 0)
                ai4b::stop("GPRegression::predict_at: X must be vectorised "
                           "N_new*p (column-major)");
            const std::size_t N_new = x_flat.n_elem / p_;
            X_new_rows.resize(N_new);
            for (std::size_t i = 0; i < N_new; ++i) {
                Eigen::VectorXd row(p_);
                for (std::size_t j = 0; j < p_; ++j)
                    row[j] = x_flat[i + j * N_new];
                X_new_rows[i] = row;
            }
            has_X = true;
        }
        // No X -> predict at the training inputs.
        const std::vector<Eigen::VectorXd>& X_star =
            has_X ? X_new_rows : X_rows_;
        const std::size_t N_new = X_star.size();

        AI4BayesCode::history_map out;
        arma::mat mu_mat, sd_mat, f_star_mat, yrep_mat;

        if (keep_history_) {
            AI4BayesCode::history_map hist = impl_->get_history();
            const arma::mat& amp_hist   = hist.at("amplitude");
            const arma::mat& ell_hist   = hist.at("lengthscale");
            const arma::mat& sigma_hist = hist.at("sigma");
            const std::size_t n_draws = amp_hist.n_rows;

            mu_mat    .set_size(n_draws, N_new);
            sd_mat    .set_size(n_draws, N_new);
            f_star_mat.set_size(n_draws, N_new);
            yrep_mat  .set_size(n_draws, N_new);
            for (std::size_t d = 0; d < n_draws; ++d) {
                predict_one_draw_(amp_hist(d, 0), ell_hist(d, 0),
                                  sigma_hist(d, 0), X_star,
                                  mu_mat, sd_mat, f_star_mat, yrep_mat, d);
            }
        } else {
            mu_mat    .set_size(1, N_new);
            sd_mat    .set_size(1, N_new);
            f_star_mat.set_size(1, N_new);
            yrep_mat  .set_size(1, N_new);
            predict_one_draw_(impl_->data().get("amplitude")[0],
                              impl_->data().get("lengthscale")[0],
                              impl_->data().get("sigma")[0], X_star,
                              mu_mat, sd_mat, f_star_mat, yrep_mat, 0);
        }

        out.emplace("f_mean", std::move(mu_mat));
        out.emplace("f_sd",   std::move(sd_mat));
        out.emplace("f_star", std::move(f_star_mat));
        out.emplace("y_rep",  std::move(yrep_mat));
        return out;
    }
#ifdef AI4BAYESCODE_RCPP_MODULE
    // R-facing predict_at. An R matrix loses its dim crossing into state_map
    // (Rcpp flattens it column-major), so the column count is checked HERE,
    // while the SEXP still has it. See rcpp_predict_guard.hpp.
    AI4BayesCode::history_map predict_at_r(Rcpp::List new_data) const {
        return predict_at(ai4b::predict_input(new_data, "X", p_));
    }
#endif


    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    /// 7th R-level method: re-tune NUTS metric (mass matrix + step size +
    /// dual averaging) without advancing chain state. Available because
    /// the composite contains NUTS-family children. See system_design.md
    /// Sec.13 NUTS-family + validator.md Sec.24.
    // FORK MARKER (2026-07-26 restore) [target_accept API expose, default=0.55]
    // 4-arg CORE + 3-arg backward-compat forwarder. Rcpp modules ignore
    // C++ default args so both arities are also exposed as separate
    // bindings below; from C++, this forwarder keeps pre-existing
    // readapt_NUTS(n, reset, mtd) call sites working.
    void readapt_NUTS(int n, bool reset, int max_tree_depth, double target_accept) {
        if (n < 0) {
            ai4b::stop("readapt_NUTS: n must be non-negative");
        }
        impl_->readapt_NUTS(static_cast<std::size_t>(n), reset, readapt_rng_,
                                max_tree_depth < 0 ? std::size_t(0) : static_cast<std::size_t>(max_tree_depth),
                                target_accept);
    }

    /// 3-arg backward-compat overload; target_accept defaults to -1.0
    /// (sentinel = leave the block's target unchanged).
    void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1) {
        readapt_NUTS(n, reset, max_tree_depth, -1.0);
    }


private:
    // GP predictive equations at ONE posterior draw of the hyperparameters.
    // Writes row `row` of each output matrix. Rasmussen & Williams Eq. 2.23-2.24.
    void predict_one_draw_(double amp, double ell, double sig,
                           const std::vector<Eigen::VectorXd>& X_star,
                           arma::mat& mu_mat, arma::mat& sd_mat,
                           arma::mat& f_star_mat, arma::mat& yrep_mat,
                           std::size_t row) const {
        const std::size_t N_new = X_star.size();
        auto* cf_mut = const_cast<libgp::CovSEiso*>(cf_.get());
        Eigen::VectorXd hyper(2);
        hyper[0] = std::log(std::max(ell, 1e-10));
        hyper[1] = std::log(std::max(amp, 1e-10));
        cf_mut->set_loghyper(hyper);

        // K_train + sigma^2 I, and its Cholesky
        arma::mat Ky(N_, N_);
        for (std::size_t i = 0; i < N_; ++i) {
            for (std::size_t j = i; j < N_; ++j) {
                const double k = cf_mut->get(X_rows_[i], X_rows_[j]);
                Ky(i, j) = k;
                Ky(j, i) = k;
            }
        }
        Ky.diag() += sig * sig + GP_JITTER;
        arma::mat L;
        if (!arma::chol(L, Ky, "lower")) {
            Ky.diag() += 1e-3;
            arma::chol(L, Ky, "lower");
        }

        arma::mat K_star_X(N_new, N_);
        arma::mat K_star_star(N_new, N_new);
        for (std::size_t i = 0; i < N_new; ++i) {
            for (std::size_t j = 0; j < N_; ++j)
                K_star_X(i, j) = cf_mut->get(X_star[i], X_rows_[j]);
            K_star_star(i, i) = cf_mut->get(X_star[i], X_star[i]);
            for (std::size_t j = i + 1; j < N_new; ++j) {
                const double k = cf_mut->get(X_star[i], X_star[j]);
                K_star_star(i, j) = k;
                K_star_star(j, i) = k;
            }
        }

        const arma::vec& y_vec = impl_->data().get("y");
        const arma::vec alpha = arma::solve(arma::trimatu(L.t()),
                                  arma::solve(arma::trimatl(L), y_vec));
        const arma::vec mu_star = K_star_X * alpha;
        const arma::mat V = arma::solve(arma::trimatl(L), K_star_X.t());
        arma::mat Sigma_star = K_star_star - V.t() * V;

        arma::vec f_sd(N_new);
        for (std::size_t i = 0; i < N_new; ++i)
            f_sd[i] = std::sqrt(std::max(Sigma_star(i, i), 0.0));

        Sigma_star.diag() += GP_JITTER;
        std::normal_distribution<double> nd(0.0, 1.0);
        arma::vec z(N_new);
        for (std::size_t i = 0; i < N_new; ++i) z[i] = nd(predict_rng_);
        arma::mat L_star;
        arma::vec f_star(N_new);
        if (arma::chol(L_star, Sigma_star, "lower")) {
            f_star = mu_star + L_star * z;
        } else {
            for (std::size_t i = 0; i < N_new; ++i)
                f_star[i] = mu_star[i] + f_sd[i] * z[i];
        }

        for (std::size_t i = 0; i < N_new; ++i) {
            mu_mat    (row, i) = mu_star[i];
            sd_mat    (row, i) = f_sd[i];
            f_star_mat(row, i) = f_star[i];
            yrep_mat  (row, i) = f_star[i] + sig * nd(predict_rng_);
        }
    }

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
            "keep_history. The latent f is integrated out; one joint NUTS "
            "block samples amplitude, lengthscale and sigma.")
        .method("step", (void (GPRegression::*)())    &GPRegression::step, "Run one sweep.")
        .method("step", (void (GPRegression::*)(int)) &GPRegression::step, "Run n sweeps.")
        .method("get_current", &GPRegression::get_current)
        .method("set_current", &GPRegression::set_current)
        .method("predict_at",  &GPRegression::predict_at_r)
        .method("get_dag",     &GPRegression::get_dag)
        .method("get_history", &GPRegression::get_history)
        AI4BAYESCODE_BIND_READAPT_NUTS(GPRegression)
        AI4BAYESCODE_BIND_KERNEL_CONTROL(GPRegression);
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
        .def("step", (void (GPRegression::*)(int)) &GPRegression::step,  pybind11::arg("n_steps"))
        .def("get_current",  &GPRegression::get_current)
        .def("set_current",  &GPRegression::set_current, pybind11::arg("params"))
        .def("predict_at",   &GPRegression::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &GPRegression::get_dag)
        .def("get_history",  &GPRegression::get_history)
        .def("readapt_NUTS", (void (GPRegression::*)(int, bool, int, double)) &GPRegression::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false,
             pybind11::arg("max_tree_depth") = -1,
             pybind11::arg("target_accept") = -1.0)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(GPRegression);
}
#endif

// ============================================================================
// Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
// Simulates 1-D GP-regression data from a KNOWN smooth latent function
//     f_true(x) = sin(3x) + 0.5 x
// observed with Gaussian noise   y_i = f_true(x_i) + eps_i,  eps ~ N(0, s^2).
// Fits the marginal model and checks three things:
//
//   (1) the posterior-mean latent f recovers f_true much better than the
//       naive "y is the signal" baseline (RMSE(f_hat, f_true) <
//       RMSE(y, f_true), i.e. the GP smooths out noise);
//   (2) the noise-sigma posterior mean is in the right ballpark of truth;
//   (3) the covariance hyperparameters are actually LEARNED -- their
//       posterior spread is far tighter than their prior spread.
//
// Check (3) is the one that discriminates. A GP sampler whose
// hyperparameter log-density omits the data term still passes (1) and (2)
// -- the latent fit and the noise scale come out fine -- while amplitude
// and lengthscale are silently drawn from their priors. Comparing the
// posterior SD against the analytic prior SD is what catches that:
//     amplitude   ~ half-Normal(0, s)     -> prior SD = s sqrt(1 - 2/pi)
//     lengthscale ~ InverseGamma(a, b)    -> prior SD = b / ((a-1) sqrt(a-2))
// ============================================================================
#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>

int main() {
    const std::size_t N          = 40;
    const double      sigma_true = 0.30;
    const double      x_lo = -3.0, x_hi = 3.0;

    // ---- Simulate from a known smooth truth -----------------------------
    std::mt19937_64 sim_rng(2024);
    std::normal_distribution<double> noise(0.0, sigma_true);

    arma::mat X(N, 1);
    arma::vec f_true(N);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double x = x_lo + (x_hi - x_lo) * static_cast<double>(i)
                                / static_cast<double>(N - 1);
        X(i, 0)   = x;
        f_true[i] = std::sin(3.0 * x) + 0.5 * x;
        y[i]      = f_true[i] + noise(sim_rng);
    }

    // ---- Fit -------------------------------------------------------------
    GPRegression model(X, y, /*rng_seed=*/11, /*keep_history=*/false);
    model.step(800);   // warmup

    const int    M = 1500;
    arma::vec    f_sum(N, arma::fill::zeros);
    double sigma_bar = 0.0, amp_bar = 0.0, ell_bar = 0.0;
    double amp_sq = 0.0, ell_sq = 0.0;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto gc = model.get_current();          // copy (avoids dangling ref)
        const double amp_cur = gc.at("amplitude")[0];
        const double ell_cur = gc.at("lengthscale")[0];
        sigma_bar += gc.at("sigma")[0];
        amp_bar   += amp_cur;   amp_sq += amp_cur * amp_cur;
        ell_bar   += ell_cur;   ell_sq += ell_cur * ell_cur;
        // f is integrated out of the sampler; recover it at this draw.
        const auto fit = model.predict_at(AI4BayesCode::state_map{});
        const arma::mat& fm = fit.at("f_mean");       // 1 x N
        for (std::size_t i = 0; i < N; ++i) f_sum[i] += fm(0, i);
    }
    const double Md = static_cast<double>(M);
    const arma::vec f_hat = f_sum / Md;
    sigma_bar /= Md;  amp_bar /= Md;  ell_bar /= Md;
    const double amp_sd_post = std::sqrt(std::max(amp_sq / Md - amp_bar * amp_bar, 0.0));
    const double ell_sd_post = std::sqrt(std::max(ell_sq / Md - ell_bar * ell_bar, 0.0));

    // ---- Analytic PRIOR moments (the discriminating baseline) ------------
    const double s_amp = std::max(arma::stddev(y), 0.1);   // half-Normal scale
    const double amp_sd_prior = s_amp * std::sqrt(1.0 - 2.0 / M_PI);
    double ell_scale = 0.0;                                // median pairwise dist
    {
        std::vector<double> d;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = i + 1; j < N; ++j)
                d.push_back(std::abs(X(i, 0) - X(j, 0)));
        std::sort(d.begin(), d.end());
        ell_scale = (d.size() % 2 == 1) ? d[d.size() / 2]
                                        : 0.5 * (d[d.size() / 2 - 1] + d[d.size() / 2]);
    }
    const double ig_a = 5.0, ig_b = ell_scale;
    const double ell_sd_prior = ig_b / ((ig_a - 1.0) * std::sqrt(ig_a - 2.0));
    const double amp_mean_prior = s_amp * std::sqrt(2.0 / M_PI);
    const double ell_mean_prior = ig_b / (ig_a - 1.0);

    // ---- Recovery metrics ------------------------------------------------
    double sse_fit = 0.0, sse_naive = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double df = f_hat[i] - f_true[i];
        const double dn = y[i]     - f_true[i];   // naive: observed == signal
        sse_fit   += df * df;
        sse_naive += dn * dn;
    }
    const double rmse_fit   = std::sqrt(sse_fit   / static_cast<double>(N));
    const double rmse_naive = std::sqrt(sse_naive / static_cast<double>(N));

    std::printf("GPRegression demo (N=%zu, marginal likelihood):\n", N);
    std::printf("  posterior-mean f RMSE vs truth = %.4f\n", rmse_fit);
    std::printf("  naive (y as signal)  RMSE       = %.4f\n", rmse_naive);
    std::printf("  sigma_hat = %.4f  (truth %.2f)\n", sigma_bar, sigma_true);
    std::printf("  amplitude   post %.4f +- %.4f   prior %.4f +- %.4f  (SD ratio %.3f)\n",
                amp_bar, amp_sd_post, amp_mean_prior, amp_sd_prior,
                amp_sd_post / amp_sd_prior);
    std::printf("  lengthscale post %.4f +- %.4f   prior %.4f +- %.4f  (SD ratio %.3f)\n",
                ell_bar, ell_sd_post, ell_mean_prior, ell_sd_prior,
                ell_sd_post / ell_sd_prior);

    // The GP must denoise, the noise scale must land near truth, and the
    // covariance hyperparameters must be informed by the data rather than
    // echoed back from their priors.
    const bool denoises   = rmse_fit < rmse_naive;
    const bool sigma_sane = sigma_bar > 0.5 * sigma_true &&
                            sigma_bar < 2.0 * sigma_true;
    // The data must CONCENTRATE each hyperparameter relative to its prior.
    // A log-density that drops the marginal-likelihood term returns the prior
    // exactly, so this ratio sits at 1.0 (Monte Carlo error on an SD from
    // 1500 draws is about 2%, so 0.7 is a ~15-sigma separation). The posterior
    // MEAN is deliberately not asserted on: both priors are weakly informative
    // and already centred near the truth, so a small mean shift is what a
    // well-chosen prior looks like, not evidence of a bug.
    const bool amp_learned = amp_sd_post < 0.7 * amp_sd_prior;
    const bool ell_learned = ell_sd_post < 0.7 * ell_sd_prior;
    const bool ok = denoises && sigma_sane && amp_learned && ell_learned;

    if (!denoises)    std::printf("  [FAIL] GP did not beat the naive baseline\n");
    if (!sigma_sane)  std::printf("  [FAIL] noise sigma off\n");
    if (!amp_learned) std::printf("  [FAIL] amplitude posterior indistinguishable from prior\n");
    if (!ell_learned) std::printf("  [FAIL] lengthscale posterior indistinguishable from prior\n");

    std::printf("%s\n",
        ok ? "[demo PASS] GP recovers latent f, noise sigma, and LEARNED hyperparameters"
           : "[demo FAIL] GP did not recover the latent function / noise / hyperparameters");
    return ok ? 0 : 1;
}
#endif

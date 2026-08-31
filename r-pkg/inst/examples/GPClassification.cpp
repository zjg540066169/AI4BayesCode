// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// GPClassification.cpp
//
// Gaussian-process classification with a logit link, in the WHITENED
// parameterisation (Murray & Adams 2010), with the two covariance
// hyperparameters and the whole latent sampled TOGETHER by one
// joint_nuts_block.
//
// MODEL
// ------------------------------------------
//   y_i | f(x_i)  ~ Bernoulli(sigmoid(f(x_i))),   y_i in {0, 1}
//   f             = L z,  L L' = K_amplitude_lengthscale(X, X),  z ~ N(0, I)
//   amplitude     ~ half-Normal(0, 1)
//   lengthscale   ~ InverseGamma(5, median pairwise distance)
//
//   K_ij = amplitude^2 * (exp(-r_ij^2 / (2 lengthscale^2)) + 1e-8 * 1(i == j))
//
// The jitter is MULTIPLICATIVE, inside the amplitude^2 factor, so that
// L = amplitude * chol(R + eps I) holds EXACTLY. That makes df/d amplitude =
// f / amplitude exact rather than approximate, which is what lets the
// amplitude gradient below be O(N) instead of another O(N^3) Cholesky
// derivative.
//
// WHY WHITENED
// ------------
// The centred latent (sample f directly with prior covariance K) has two
// problems. It funnels: p(f | amplitude) shrinks with amplitude, so the
// chain collapses toward (amplitude ~= 0, f ~= 0). And log p(f | amplitude,
// lengthscale) is a term that belongs to no block by default, so a
// hyperparameter block that omits it samples from its PRIOR on every dataset
// while the latent fit and every R-hat still look healthy. Whitening removes
// the term rather than relying on someone to remember it: z's prior is
// N(0, I), free of the hyperparameters, so the only place the data enters is
// the Bernoulli likelihood at f = L z.
//
// WHY ONE BLOCK AND NOT GIBBS
// ---------------------------
// Whitening alone is not enough. Splitting the sweep into "hyperparameters
// given z" and "z given hyperparameters" -- for instance with an
// elliptical_slice_sampling_block on z -- leaves the two strongly coupled:
// with z held fixed, f = L(theta) z is a deterministic function of theta, so
// p(theta | z, y) is far sharper than the marginal p(theta | y) and the
// alternation random-walks. Putting (amplitude, lengthscale, z) in ONE NUTS
// trajectory removes the alternation entirely.
//
// Measured on the @example:R dataset below, 4 chains x (1000 burn + 2000
// keep), cross-chain rank R-hat / bulk ESS on the worst component:
//
//                              amplitude        lengthscale      f
//   ESS-Gibbs, diagonal metric  1.623 / 7        1.179 / 41      1.279 / 11
//   ESS-Gibbs, dense metric     1.102 / 28       1.159 / 17      1.067 / 49
//   ONE joint block (this)      1.001 / 1937     1.002 / 1539    1.003 / 3074
//
// against a library bar of R-hat < 1.01. The Gibbs alternation is not merely
// slower; it does not converge at this budget.
//
// BLOCK
// -----
//   amp_ell_joint : ONE joint_nuts_block, sub_params
//                   [{amplitude, 1, POSITIVE},
//                    {lengthscale, 1, POSITIVE},
//                    {z, N, REAL}].
//   f             : DERIVED, refresher f = L_chol * z.
//
// JOINT natural-scale log-density (each term appearing exactly once;
// joint_nuts_block adds the two POSITIVE-slice Jacobians internally, so the
// oracle must NOT include them):
//
//   log p(amplitude, lengthscale, z | y)
//       = sum_i [ y_i f_i - softplus(f_i) ]          f = L z, Bernoulli-logit
//       - 0.5 z'z                                    whitened prior
//       - 0.5 amplitude^2 / amp_prior_sd^2           half-Normal
//       - (a+1) log(lengthscale) - b / lengthscale   InverseGamma(a, b)
//
//   The GP prior's -0.5 log|K| cancels against the whitening Jacobian; only
//   z's unit-Normal survives.
//
//   Gradients, with r = y - sigmoid(f) = d loglik / d f:
//       d/d z           = L' r - z
//       d/d amplitude   = (r . f) / amplitude - amplitude / amp_prior_sd^2
//                         (exact: L = amplitude * L_R, so df/d amplitude = f / amplitude)
//       d/d lengthscale = r' L Phi(L^-1 (dK/d lengthscale) L^-T) z + prior'
//                         Phi(A) = tril(A) with the diagonal halved -- the
//                         Cholesky derivative dL = L Phi(L^-1 dK L^-T)
//                         (Murray 2016).
//       dK/d lengthscale = amplitude^2 * R .* r_ij^2 / lengthscale^3
//
// Class name: GPClassification. Module: GPClassification_module.
//
// LICENSE: libgp_kernels is BSD-3 (GPL-compatible). AI4BayesCode itself is
// GPL-2.0-or-later; combined work remains GPL-2.0-or-later.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("GPClassification")
//   set.seed(2026)
//   N <- 120                                     # >> #hyperparams: rich signal
//   x <- runif(N, -3, 3)                         # 1D inputs on [-3, 3]
//   f_true <- 1.6 * sin(1.3 * x)                 # smooth latent GP-like signal
//   p_true <- 1 / (1 + exp(-f_true))             # Bernoulli success prob
//   y <- as.numeric(runif(N) < p_true)           # 0/1 labels
//   X <- matrix(x, ncol = 1)                     # N x 1 design matrix
//   # ---- Parallel chains + convergence diagnosis (default) ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(GPClassification, X, y, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   print(ai4bayescode_rhat_summary(run))          # CROSS-chain R-hat / ESS
//   ai4bayescode_diagnose(run$histories[[1]])      # chain 1: summary + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(GPClassification, X, y, 7L, TRUE)   # X(Nx1), y(0/1), seed=7L, keep_history=TRUE
//   m$step(2500); str(m$get_current())           # amplitude / lengthscale / z / f
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2026)
//   N = 120
//   x = rng.uniform(-3.0, 3.0, N)                 # 1D inputs on [-3, 3]
//   f_true = 1.6 * np.sin(1.3 * x)                 # smooth latent GP-like signal
//   p_true = 1.0 / (1.0 + np.exp(-f_true))         # Bernoulli success prob
//   y = (rng.uniform(size=N) < p_true).astype(float)   # 0/1 labels
//   X = x.reshape(N, 1)                            # N x 1 design matrix
//   Mod = AI4BayesCode.example("GPClassification")
//   # ---- Parallel chains + diagnosis (default) ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.GPClassification(X, y, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   print(AI4BayesCode.rhat_summary(chains))   # CROSS-chain R-hat / ESS
//   AI4BayesCode.diagnose(chains[0]["hist"])   # chain 1: summary + plots
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.GPClassification(X, y, 7, True)        # X(Nx1), y(0/1), seed=7, keep_history=True
//   m.step(2500); print(m.get_current())          # dict: amplitude, lengthscale, z, f
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

// Vendored libgp kernel subsystem (BSD-3).
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

// Numerically stable softplus: log(1 + exp(x))
inline double stable_softplus(double x) {
    if (x > 0.0)  return x + std::log1p(std::exp(-x));
    else          return std::log1p(std::exp(x));
}

// Numerically stable sigmoid: 1 / (1 + exp(-x))
inline double stable_sigmoid(double x) {
    if (x >= 0.0) {
        const double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    } else {
        const double e = std::exp(x);
        return e / (1.0 + e);
    }
}

// Pairwise-distance helper for InvGamma scale heuristic.
double median_pairwise_distance(const arma::mat& X) {
    const std::size_t N = X.n_rows;
    if (N < 2) return 1.0;
    std::vector<double> dists;
    dists.reserve(N * (N - 1) / 2);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            dists.push_back(arma::norm(X.row(i) - X.row(j), 2));
        }
    }
    std::sort(dists.begin(), dists.end());
    const std::size_t m = dists.size() / 2;
    return (dists.size() % 2 == 1) ? dists[m]
                                   : 0.5 * (dists[m - 1] + dists[m]);
}

// ----------------------------------------------------------------------------
//  Full natural-scale joint log-posterior over (amplitude, lengthscale, z).
//
//  Layout of theta_cat (must match sub_params order in the config):
//      theta_cat[0]         = amplitude    (POSITIVE)
//      theta_cat[1]         = lengthscale  (POSITIVE)
//      theta_cat[2 : 2+N-1] = z            (REAL x N)
//
//  Reads from ctx: "y", "X_sqdist", "amp_prior_sd", "ell_prior_shape",
//  "ell_prior_scale".
//
//  K is rebuilt HERE, at the PROPOSED hyperparameters: the K_matrix refresher
//  fires only at block boundaries, never inside a NUTS trajectory. It is built
//  analytically from the cached squared distances rather than through the libgp
//  object, whose CovSEiso is stateful (set_loghyper) and would desynchronise
//  from the refresher if mutated mid-trajectory. The two agree by construction:
//  CovSEiso is sf2 * exp(-0.5 ||dx/ell||^2) with sf2 = amplitude^2.
//
//  See the file header for the derivation of every term and gradient.
// ----------------------------------------------------------------------------

// Relative jitter, applied INSIDE the amplitude^2 factor so that
// L = amplitude * chol(R + GP_REL_JITTER * I) is exact. R has a unit diagonal,
// so this is a well-scaled perturbation whatever the amplitude.
constexpr double GP_REL_JITTER = 1e-8;

// Phi(A) = lower triangle of A with the diagonal halved (Cholesky derivative).
inline void phi_in_place(arma::mat& A) {
    const std::size_t n = A.n_rows;
    for (std::size_t j = 0; j < n; ++j) {
        A(j, j) *= 0.5;
        for (std::size_t i = 0; i < j; ++i) A(i, j) = 0.0;
    }
}

double amp_ell_joint_log_density(const arma::vec& theta_cat,
                                 const block_context& ctx,
                                 arma::vec* grad_nat) {
    const double neg_inf = -std::numeric_limits<double>::infinity();

    const arma::vec& y       = ctx.at("y");
    const arma::vec& d2_flat = ctx.at("X_sqdist");
    const std::size_t N = y.n_elem;
    if (theta_cat.n_elem != 2 + N || d2_flat.n_elem != N * N) return neg_inf;

    const double amp = theta_cat[0];
    const double ell = theta_cat[1];
    if (!(amp > 0.0) || !(ell > 0.0) ||
        !std::isfinite(amp) || !std::isfinite(ell)) return neg_inf;

    const arma::vec z = theta_cat.subvec(2, 2 + N - 1);
    const arma::mat R2(const_cast<double*>(d2_flat.memptr()), N, N,
                       /*copy_aux_mem=*/false, /*strict=*/true);

    // R = correlation matrix; K = amplitude^2 (R + eps I).
    arma::mat Reps = arma::exp(R2 / (-2.0 * ell * ell));
    const arma::mat R_bare = Reps;                 // keep the un-jittered copy
    Reps.diag() += GP_REL_JITTER;

    arma::mat L_R;
    if (!arma::chol(L_R, Reps, "lower")) return neg_inf;
    if (!std::isfinite(L_R.diag().min()) || L_R.diag().min() < 1e-8) {
        return neg_inf;   // too ill-conditioned to trust; a valid NUTS reject
    }
    const arma::mat L = amp * L_R;                 // exact: K = amp^2 (R + eps I)

    const arma::vec f = L * z;
    double llik = 0.0;
    for (std::size_t i = 0; i < N; ++i) llik += y[i] * f[i] - stable_softplus(f[i]);

    const double s     = ctx.at("amp_prior_sd")[0];
    const double s2    = s * s;
    const double shape = ctx.at("ell_prior_shape")[0];
    const double scale = ctx.at("ell_prior_scale")[0];

    const double lp = llik - 0.5 * arma::dot(z, z)
        + (-0.5 * amp * amp / s2)
        + (-(shape + 1.0) * std::log(ell) - scale / ell);
    if (!std::isfinite(lp)) return neg_inf;

    if (grad_nat) {
        grad_nat->set_size(2 + N);

        arma::vec r(N);
        for (std::size_t i = 0; i < N; ++i) r[i] = y[i] - stable_sigmoid(f[i]);

        // amplitude: L = amp * L_R, so df/d amp = f / amp exactly.
        (*grad_nat)[0] = arma::dot(r, f) / amp - amp / s2;

        // lengthscale: Cholesky derivative dL = L Phi(L^-1 dK L^-T).
        // The eps on R's diagonal does not depend on the lengthscale, so
        // dK/d ell = amplitude^2 * R .* R2 / ell^3 uses the UN-jittered R.
        const arma::mat dK_dell = (amp * amp) * (R_bare % R2) / (ell * ell * ell);
        arma::mat A = arma::solve(arma::trimatl(L), dK_dell);
        A = arma::solve(arma::trimatl(L), A.t()).t();      // L^-1 dK L^-T
        phi_in_place(A);
        (*grad_nat)[1] = arma::dot(r, L * (A * z))
                       + (-(shape + 1.0) / ell + scale / (ell * ell));

        // z: likelihood + whitened unit-Normal prior.
        grad_nat->subvec(2, 2 + N - 1) = L.t() * r - z;
    }
    return lp;
}

}  // anonymous namespace

// ============================================================================
// User-facing class
// ============================================================================

class GPClassification : public AI4BayesCode::kernel_control_mixin<GPClassification> {
    friend class AI4BayesCode::kernel_control_mixin<GPClassification>;
public:
    GPClassification(const arma::mat& X,
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
          impl_(std::make_unique<composite_block>("GPClassification")),
          keep_history_(keep_history)
    {
        if (X.n_rows != y.n_elem)
            ai4b::stop("GPClassification: X and y row/length mismatch");
        if (X.n_rows < 2)
            ai4b::stop("GPClassification: N must be >= 2");

        const std::size_t N = static_cast<std::size_t>(X.n_rows);
        const std::size_t p = static_cast<std::size_t>(X.n_cols);
        N_ = N;
        p_ = p;

        // Validate y in {0, 1}
        arma::vec y_arma(N);
        for (std::size_t i = 0; i < N; ++i) {
            const double yi = y[i];
            if (yi != 0.0 && yi != 1.0)
                ai4b::stop("GPClassification: y must contain only 0/1 values");
            y_arma[i] = yi;
        }
        impl_->data().set("y", y_arma);

        // Cache X as arma::mat + per-row Eigen::VectorXd for libgp.
        X_arma_ = arma::mat(N, p);
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < p; ++j)
                X_arma_(i, j) = X(i, j);
        X_rows_.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            Eigen::VectorXd row(p);
            for (std::size_t j = 0; j < p; ++j) row[j] = X(i, j);
            X_rows_.push_back(row);
        }
        impl_->data().set("X", arma::vectorise(X_arma_));

        // Pairwise squared distances, built once. The joint log-density
        // rebuilds K from these at every proposed (amplitude, lengthscale),
        // so this O(N^2 p) loop must not sit inside a NUTS trajectory.
        {
            arma::mat R2(N, N, arma::fill::zeros);
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = i + 1; j < N; ++j) {
                    const double d2 = arma::accu(
                        arma::square(X_arma_.row(i) - X_arma_.row(j)));
                    R2(i, j) = d2;
                    R2(j, i) = d2;
                }
            impl_->data().set("X_sqdist", arma::vectorise(R2));
        }

        // Prior hyperparameters:
        //   amplitude: half-Normal(0, 1)   -- weakly-informative default
        //   lengthscale: InverseGamma(5, median_pair_dist)
        impl_->data().set("amp_prior_sd", arma::vec{1.0});
        const double ell_scale = median_pairwise_distance(X_arma_);
        impl_->data().set("ell_prior_shape", arma::vec{5.0});
        impl_->data().set("ell_prior_scale", arma::vec{ell_scale});

        // Initial hyperparameter values
        const double amp_init = 1.0;
        const double ell_init = ell_scale / 5.0;  // InvGamma mean ~ scale/(shape-1)
        impl_->data().set("amplitude",   arma::vec{amp_init});
        impl_->data().set("lengthscale", arma::vec{ell_init});

        // libgp kernel
        cf_ = std::make_unique<libgp::CovSEiso>();
        cf_->init(static_cast<int>(p));

        // Whitened latent z, started from a DRAW of its own prior N(0, I).
        // Starting at zero would make f = L z identically zero, so the
        // Bernoulli likelihood -- and with it the hyperparameters' share of
        // the gradient -- would vanish at the initial point, and the joint
        // block would begin its warmup on the prior geometry alone.
        arma::vec z_init(N);
        {
            std::mt19937_64 init_rng(
                rng_seed == 0 ? std::random_device{}()
                              : static_cast<std::uint64_t>(rng_seed)
                                ^ 0x94D049BB133111EBULL);
            std::normal_distribution<double> z_nd(0.0, 1.0);
            for (std::size_t i = 0; i < N; ++i) z_init[i] = z_nd(init_rng);
        }
        impl_->data().set("z", z_init);
        // f is DERIVED: f = L_chol * z.
        impl_->data().set("f", arma::vec(N, arma::fill::zeros));

        // Seed K_matrix + L_chol buffers
        impl_->data().set("K_matrix", arma::vec(N * N, arma::fill::zeros));
        impl_->data().set("L_chol",   arma::vec(N * N, arma::fill::zeros));

        // ---- Gibbs dependencies -----------------------------------------
        // The joint block's dependencies are keyed under the JOINT BLOCK NAME
        // ("amp_ell_joint") and list only the true external data() inputs
        // (the union of what amplitude and lengthscale each read from data(),
        // minus the now-internal cross-reads of each other).
        impl_->data().declare_dependencies("amp_ell_joint",
            {"y", "X_sqdist",
             "amp_prior_sd", "ell_prior_shape", "ell_prior_scale"});
        
        // ---- Invalidation chain -----------------------------------------
        // Keyed under the JOINT BLOCK NAME (composite_block calls
        // refresh_derived_for(block_name) after the joint block steps); keying
        // under sub-param names would never fire -> stale K_matrix/L_chol.
        // One block writes amplitude, lengthscale AND z, so a single
        // invalidation under the block name covers the whole chain.
        impl_->data().declare_invalidates("amp_ell_joint",
            {"K_matrix", "L_chol", "f"});

        // ---- K_matrix refresher (libgp CovSEiso) ------------------------
        const libgp::CovSEiso* cf_ptr = cf_.get();
        const std::vector<Eigen::VectorXd>* X_rows_ptr = &X_rows_;
        impl_->data().register_refresher("K_matrix",
            [cf_ptr, X_rows_ptr, N](
                const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const double amp = d.get("amplitude")[0];
                const double ell = d.get("lengthscale")[0];
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

        // ---- L_chol refresher: chol(amplitude^2 (R + eps I)) -----------
        //      The jitter is MULTIPLICATIVE, exactly as in the joint
        //      log-density above, so the f published here is bit-comparable
        //      with the f the sampler actually scored.
        impl_->data().register_refresher("L_chol",
            [N](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& K_flat = d.get("K_matrix");
                const double amp = d.get("amplitude")[0];
                arma::mat K(const_cast<double*>(K_flat.memptr()), N, N,
                             /*copy_aux_mem=*/false, /*strict=*/true);
                arma::mat M = K;
                M.diag() += amp * amp * GP_REL_JITTER;
                arma::mat L;
                if (!arma::chol(L, M, "lower")) {
                    M.diag() += amp * amp * 1e-4;
                    arma::chol(L, M, "lower");
                }
                return arma::vectorise(L);
            });

        // ---- f refresher: the whitened latent, f = L z ------------------
        impl_->data().register_refresher("f",
            [N](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& L_flat = d.get("L_chol");
                const arma::vec& z_vec  = d.get("z");
                arma::mat L(const_cast<double*>(L_flat.memptr()), N, N,
                             /*copy_aux_mem=*/false, /*strict=*/true);
                return L * z_vec;
            });

        impl_->data().refresh_all();

        // ---- Predict DAG + y_rep stochastic refresher -------------------
        impl_->data().declare_data_input("X");
        impl_->data().declare_predict_edges("X",        {"K_matrix"});
        impl_->data().declare_predict_edges("K_matrix", {"L_chol"});
        impl_->data().declare_predict_edges("L_chol",   {"f"});
        impl_->data().declare_predict_edges("f",        {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). declare_predict_edges + context_edges
        //      stay keyed by SUB-PARAM name (unchanged from original).
        impl_->data().declare_context_edges("amp_prior_sd",   {"amplitude"});
        impl_->data().declare_context_edges("ell_prior_shape",{"lengthscale"});
        impl_->data().declare_context_edges("ell_prior_scale",{"lengthscale"});
        impl_->data().declare_context_edges("amplitude",      {"K_matrix"});
        impl_->data().declare_context_edges("lengthscale",    {"K_matrix"});
        impl_->data().declare_context_edges("L_chol",         {"f"});
        impl_->data().declare_context_edges("z",              {"f"});

        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));
        impl_->data().register_stochastic_refresher("y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                const arma::vec& f = d.get("f");
                std::uniform_real_distribution<double> ud(0.0, 1.0);
                arma::vec out(f.n_elem);
                for (std::size_t i = 0; i < f.n_elem; ++i) {
                    const double pi = stable_sigmoid(f[i]);
                    out[i] = (ud(rng) < pi) ? 1.0 : 0.0;
                }
                return out;
            });

        // ---- Child blocks in Gibbs order --------------------------------
        //   child(0) amp_ell_joint -- the ONLY block: one joint_nuts_block
        //            over (amplitude, lengthscale, z[N]).
        {
            joint_nuts_block_config cfg;
            cfg.name = "amp_ell_joint";
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "amplitude",   1,
                                      joint_constraint::POSITIVE });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "lengthscale", 1,
                                      joint_constraint::POSITIVE });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "z", N, joint_constraint::REAL });
            // initial_cat is NATURAL-scale: [amp_init, ell_init, z_init].
            cfg.initial_cat = arma::vec(2 + N);
            cfg.initial_cat[0] = amp_init;
            cfg.initial_cat[1] = ell_init;
            cfg.initial_cat.subvec(2, 2 + N - 1) = z_init;
            cfg.log_density_grad = &amp_ell_joint_log_density;
            // amplitude and lengthscale operate on very different scales
            // (amp ~ O(1), ell ~ O(median pairwise dist)); diagonal metric
            // is required for faithful sampling of heterogeneous scales.
            // Diagonal metric: (amplitude, lengthscale, z[N]) is N+2
            // dimensional, so a dense metric would be an (N+2)^2 covariance
            // to estimate and invert. n_warmup_first_call is left at its
            // default because a single-pilot metric block floors the
            // first-call warmup at 2500 regardless (the step-size dual
            // averaging needs that many iterations to settle against a
            // freshly installed metric), so a smaller number here would only
            // mislead the reader.
            cfg.use_diagonal_metric = true;
            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
        }
        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        // sub-params are written back to data() under sub-param names by
        // joint_nuts_block; read directly from data(). Each output is an
        // arma::vec; the frontend converts state_map -> R list / Python dict.
        // z is the sampled latent; f = L z is derived but is the interpretable
        // one, so both are reported. set_current rejects being given both at
        // once, and round-tripping set_current(get_current()) restores the
        // chain exactly from z (f is recomputed from it).
        AI4BayesCode::state_map out;
        out["amplitude"]   = impl_->data().get("amplitude");         // length 1
        out["lengthscale"] = impl_->data().get("lengthscale");       // length 1
        out["z"]           = impl_->data().get("z");                 // length N
        out["f"]           = impl_->data().get("f");                 // length N, derived
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        // For the joint block: read its current concatenated vector, overwrite
        // the relevant slice(s), set_current, mirror to data(). Each value in
        // params is an arma::vec (frontend already converted list/dict).
        auto& jblk = dynamic_cast<joint_nuts_block&>(impl_->child(0));
        arma::vec cat_new = jblk.current();   // [amplitude, lengthscale, z(N)]
        bool touched = false;

        auto assign_pos = [&](const char* key, std::size_t idx) {
            auto it = params.find(key);
            if (it == params.end()) return;
            const double v = it->second[0];
            if (!(v > 0.0))
                ai4b::stop("GPClassification::set_current: %s must be positive", key);
            cat_new[idx] = v;
            touched = true;
        };
        assign_pos("amplitude",   0);
        assign_pos("lengthscale", 1);

        auto it_z = params.find("z");
        if (it_z != params.end()) {
            if (static_cast<std::size_t>(it_z->second.n_elem) != N_)
                ai4b::stop("GPClassification::set_current: z length mismatch");
            cat_new.subvec(2, 2 + N_ - 1) = it_z->second;
            touched = true;
        }

        // f is derived from the sampled z through f = L z, so setting f means
        // solving z = L^-1 f, at the (amplitude, lengthscale) supplied in the
        // SAME call -- which is why L is rebuilt here from cat_new rather than
        // read from the not-yet-refreshed L_chol.
        auto it_f = params.find("f");
        if (it_f != params.end()) {
            const arma::vec& f_new = it_f->second;
            if (static_cast<std::size_t>(f_new.n_elem) != N_)
                ai4b::stop("GPClassification::set_current: f length mismatch");
            const arma::mat L = cholesky_at_(cat_new[0], cat_new[1]);
            if (it_z != params.end()) {
                // Both supplied -- the round-trip case, since get_current()
                // reports z AND f. z is the sampled state and wins; f is
                // checked for consistency rather than silently dropped.
                const arma::vec f_from_z =
                    arma::trimatl(L) * arma::vec(cat_new.subvec(2, 2 + N_ - 1));
                const double tol = 1e-6 * std::max(1.0, arma::norm(f_from_z, "inf"));
                if (arma::norm(f_from_z - f_new, "inf") > tol)
                    ai4b::stop("GPClassification::set_current: f and z disagree "
                               "(f must equal L z); pass one or the other");
            } else {
                cat_new.subvec(2, 2 + N_ - 1) =
                    arma::solve(arma::trimatl(L), f_new);
                touched = true;
            }
        }

        for (const auto& kv : params) {
            if (kv.first != "amplitude" && kv.first != "lengthscale" &&
                kv.first != "z" && kv.first != "f") {
                ai4b::stop("GPClassification::set_current: unknown key '%s'",
                           kv.first.c_str());
            }
        }

        if (touched) {
            jblk.set_current(cat_new);
            impl_->data().set("amplitude",   arma::vec{cat_new[0]});
            impl_->data().set("lengthscale", arma::vec{cat_new[1]});
            impl_->data().set("z", arma::vec(cat_new.subvec(2, 2 + N_ - 1)));
            impl_->data().refresh_derived_for("amp_ell_joint");
        }
    }

    // predict_at: state_map{"X" = X_new} -> posterior p(f_star | X_new),
    // samples f_star, returns prob_star = sigmoid(f_star) and y_rep ~
    // Bernoulli(prob_star). Empty map -> posterior predictive at training X
    // using current f.
    //
    // Backend-neutral I/O (frontend converts state_map <-> R list / Python
    // dict and history_map <-> R list-of-matrices / Python dict-of-arrays):
    //   * INPUT  : new_data["X"] is a VECTORISED N_new*p arma::vec, column
    //              major (so element (i,j) is at index i + j*N_new), matching
    //              how X is stored in data() (arma::vectorise of the N x p
    //              design). R/Python callers pass as.vector(X_new) / X_new
    //              flattened column-major.
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
                ai4b::stop("GPClassification::predict_at: unknown key '%s'",
                           kv.first.c_str());
        }
        auto it_X = new_data.find("X");
        if (it_X != new_data.end()) {
            x_flat = it_X->second;
            if (p_ == 0 || x_flat.n_elem % p_ != 0)
                ai4b::stop("GPClassification::predict_at: X must be vectorised "
                           "N_new*p (column-major)");
            N_new = x_flat.n_elem / p_;
            has_X = true;
        }

        AI4BayesCode::history_map out;

        if (!has_X) {
            if (keep_history_) {
                // History mode at training X: per-draw Bernoulli y_rep from
                // sigmoid(f_d). amplitude and lengthscale are sub-outputs of
                // the joint block (keyed by sub-param name in get_history()),
                // so access via hist.at("amplitude") / hist.at("lengthscale").
                // f is keyed by block name "f" (ESS block).
                // this->get_history(), not impl_->get_history(): f is derived
                // (f = L z) and is added by the wrapper, not by any child block.
                AI4BayesCode::history_map hist = get_history();
                const arma::mat& f_hist = hist.at("f");  // n_draws x N
                const std::size_t n_draws = f_hist.n_rows;
                const std::size_t N_local = f_hist.n_cols;
                arma::mat f_mean_mat(n_draws, N_local);
                arma::mat prob_mat(n_draws, N_local);
                arma::mat yrep_mat(n_draws, N_local);
                std::uniform_real_distribution<double> ud(0.0, 1.0);
                for (std::size_t d = 0; d < n_draws; ++d) {
                    for (std::size_t i = 0; i < N_local; ++i) {
                        const double fi = f_hist(d, i);
                        const double pi = stable_sigmoid(fi);
                        f_mean_mat(d, i) = fi;
                        prob_mat(d, i)   = pi;
                        yrep_mat(d, i)   = (ud(predict_rng_) < pi) ? 1.0 : 0.0;
                    }
                }
                out.emplace("f_mean", std::move(f_mean_mat));
                out.emplace("prob",   std::move(prob_mat));
                out.emplace("y_rep",  std::move(yrep_mat));
                return out;
            }
            const arma::vec& f = impl_->data().get("f");
            std::uniform_real_distribution<double> ud(0.0, 1.0);
            arma::mat f_mean_mat(1, f.n_elem);
            arma::mat prob_mat(1, f.n_elem);
            arma::mat yrep_mat(1, f.n_elem);
            for (std::size_t i = 0; i < f.n_elem; ++i) {
                const double pi = stable_sigmoid(f[i]);
                f_mean_mat(0, i) = f[i];
                prob_mat(0, i)   = pi;
                yrep_mat(0, i)   = (ud(predict_rng_) < pi) ? 1.0 : 0.0;
            }
            out.emplace("f_mean", std::move(f_mean_mat));
            out.emplace("prob",   std::move(prob_mat));
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
            // new-X + history: per-draw GP classification at X_new using
            // (amp_d, ell_d, f_d) from history. amplitude and lengthscale are
            // sub-outputs of the joint block, keyed by sub-param name.
            AI4BayesCode::history_map hist = get_history();   // adds derived f
            const arma::mat& amp_hist = hist.at("amplitude");
            const arma::mat& ell_hist = hist.at("lengthscale");
            const arma::mat& f_hist   = hist.at("f");
            const std::size_t n_draws = amp_hist.n_rows;
            auto* cf_mut = const_cast<libgp::CovSEiso*>(cf_.get());

            arma::mat f_mean_mat(n_draws, N_new);
            arma::mat prob_mat  (n_draws, N_new);
            arma::mat yrep_mat  (n_draws, N_new);
            std::normal_distribution<double> nd(0.0, 1.0);
            std::uniform_real_distribution<double> ud(0.0, 1.0);

            for (std::size_t d = 0; d < n_draws; ++d) {
                const double amp_d = amp_hist(d, 0);
                const double ell_d = ell_hist(d, 0);
                Eigen::VectorXd hyper(2);
                hyper[0] = std::log(std::max(ell_d, 1e-10));
                hyper[1] = std::log(std::max(amp_d, 1e-10));
                cf_mut->set_loghyper(hyper);

                arma::mat K_train_d(N_, N_);
                for (std::size_t i = 0; i < N_; ++i) {
                    for (std::size_t j = i; j < N_; ++j) {
                        double k = cf_mut->get(X_rows_[i], X_rows_[j]);
                        K_train_d(i, j) = k;
                        K_train_d(j, i) = k;
                    }
                }
                // Same MULTIPLICATIVE jitter as the log-density, so the
                // K this conditions on is the one f was drawn under.
                K_train_d.diag() += amp_d * amp_d * GP_REL_JITTER;
                arma::mat L_d;
                arma::chol(L_d, K_train_d, "lower");

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
                Sigma_star.diag() += 1e-8;

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
                    f_mean_mat(d, i) = mu_star[i];
                    const double p_di = stable_sigmoid(f_star[i]);
                    prob_mat(d, i) = p_di;
                    yrep_mat(d, i) = (ud(predict_rng_) < p_di) ? 1.0 : 0.0;
                }
            }
            out.emplace("f_mean", std::move(f_mean_mat));
            out.emplace("prob",   std::move(prob_mat));
            out.emplace("y_rep",  std::move(yrep_mat));
            return out;
        }

        // ---- new-X + stateful (current draw) ------------------------------
        const arma::vec& f_cur  = impl_->data().get("f");
        const arma::vec& L_flat = impl_->data().get("L_chol");
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

        // Build K_star_X (N_new x N_) and K_star_star
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

        // mu_star = K_star_X @ (L L^T)^{-1} @ f_cur
        arma::vec alpha = arma::solve(arma::trimatu(L.t()),
                            arma::solve(arma::trimatl(L), f_cur));
        arma::vec mu_star = K_star_X * alpha;

        // Sigma_star = K_star_star - K_star_X @ (L L^T)^{-1} @ K_star_X^T
        arma::mat V = arma::solve(arma::trimatl(L), K_star_X.t());
        arma::mat Sigma_star = K_star_star - V.t() * V;

        arma::vec f_star_sd(N_new);
        for (std::size_t i = 0; i < N_new; ++i)
            f_star_sd[i] = std::sqrt(std::max(Sigma_star(i, i), 0.0));

        // Sample f_star ~ N(mu_star, Sigma_star) via Cholesky with jitter.
        arma::mat Sigma_star_reg = Sigma_star;
        Sigma_star_reg.diag() += 1e-8;
        arma::mat L_star;
        bool chol_ok = arma::chol(L_star, Sigma_star_reg, "lower");
        arma::vec f_star(N_new);
        std::normal_distribution<double> nd(0.0, 1.0);
        if (chol_ok) {
            arma::vec z(N_new);
            for (std::size_t i = 0; i < N_new; ++i) z[i] = nd(predict_rng_);
            f_star = mu_star + L_star * z;
        } else {
            for (std::size_t i = 0; i < N_new; ++i)
                f_star[i] = mu_star[i] + f_star_sd[i] * nd(predict_rng_);
        }

        // prob_star = sigmoid(f_star); y_rep ~ Bern(prob_star)
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        arma::vec prob_star(N_new), y_rep(N_new);
        for (std::size_t i = 0; i < N_new; ++i) {
            prob_star[i] = stable_sigmoid(f_star[i]);
            y_rep[i]     = (ud(predict_rng_) < prob_star[i]) ? 1.0 : 0.0;
        }

        // Pack each output as a 1-row matrix (single predict at current draw).
        arma::mat f_mean_mat (1, N_new);
        arma::mat f_sd_mat   (1, N_new);
        arma::mat f_star_mat (1, N_new);
        arma::mat prob_mat   (1, N_new);
        arma::mat yrep_mat   (1, N_new);
        for (std::size_t i = 0; i < N_new; ++i) {
            f_mean_mat (0, i) = mu_star[i];
            f_sd_mat   (0, i) = f_star_sd[i];
            f_star_mat (0, i) = f_star[i];
            prob_mat   (0, i) = prob_star[i];
            yrep_mat   (0, i) = y_rep[i];
        }
        out.emplace("f_mean", std::move(f_mean_mat));
        out.emplace("f_sd",   std::move(f_sd_mat));
        out.emplace("f_star", std::move(f_star_mat));
        out.emplace("prob",   std::move(prob_mat));
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

    /// History of the sampled state (z, amplitude, lengthscale) plus the
    /// derived latent f, rebuilt per draw as f_d = L(amplitude_d,
    /// lengthscale_d) z_d. f is what the model is about, so diagnostics and
    /// predict_at read it rather than the whitened z.
    AI4BayesCode::history_map get_history() const {
        AI4BayesCode::history_map hist = impl_->get_history();
        auto it_z = hist.find("z");
        auto it_a = hist.find("amplitude");
        auto it_e = hist.find("lengthscale");
        if (it_z == hist.end() || it_a == hist.end() || it_e == hist.end())
            return hist;

        const arma::mat& z_hist = it_z->second;      // n_draws x N
        const arma::mat& a_hist = it_a->second;
        const arma::mat& e_hist = it_e->second;
        const std::size_t n_draws = z_hist.n_rows;

        arma::mat f_hist(n_draws, N_);
        for (std::size_t d = 0; d < n_draws; ++d) {
            const arma::mat L = cholesky_at_(a_hist(d, 0), e_hist(d, 0));
            f_hist.row(d) = (arma::trimatl(L) * z_hist.row(d).t()).t();
        }
        hist.emplace("f", std::move(f_hist));
        return hist;
    }

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
    /// chol(amplitude^2 (R + GP_REL_JITTER I)), the SAME covariance the joint
    /// log-density scores. Every place that maps between z and f goes through
    /// here so the two can never drift apart.
    arma::mat cholesky_at_(double amp, double ell) const {
        const arma::vec& d2_flat = impl_->data().get("X_sqdist");
        const arma::mat R2(const_cast<double*>(d2_flat.memptr()), N_, N_,
                           /*copy_aux_mem=*/false, /*strict=*/true);
        arma::mat Reps = arma::exp(R2 / (-2.0 * ell * ell));
        Reps.diag() += GP_REL_JITTER;
        arma::mat L_R;
        if (!arma::chol(L_R, Reps, "lower")) {
            Reps.diag() += 1e-4;
            arma::chol(L_R, Reps, "lower");
        }
        return amp * L_R;
    }

    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    std::size_t                      N_ = 0;
    std::size_t                      p_ = 0;
    bool                             keep_history_ = false;

    // libgp kernel object
    std::unique_ptr<libgp::CovSEiso> cf_;
    arma::mat                        X_arma_;
    std::vector<Eigen::VectorXd>     X_rows_;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(GPClassification_module) {
    Rcpp::class_<GPClassification>("GPClassification")
        .constructor<arma::mat, arma::vec, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::mat, arma::vec, int, bool>(
            "Whitened GP classification: (amplitude, lengthscale, z) in ONE "
            "joint_nuts_block, f = L z, Bernoulli-logit likelihood. "
            "Inputs: X (N x p), y in {0,1} length N, rng_seed, keep_history.")
        .method("step", (void (GPClassification::*)())    &GPClassification::step, "Run one sweep.")
        .method("step", (void (GPClassification::*)(int)) &GPClassification::step, "Run n sweeps.")
        .method("get_current",  &GPClassification::get_current)
        .method("set_current",  &GPClassification::set_current)
        .method("predict_at",   &GPClassification::predict_at_r)
        .method("get_dag",      &GPClassification::get_dag)
        .method("get_history",  &GPClassification::get_history)
        AI4BAYESCODE_BIND_READAPT_NUTS(GPClassification)
        AI4BAYESCODE_BIND_KERNEL_CONTROL(GPClassification);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(GPClassification, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<GPClassification>(m, "GPClassification")
        .def(pybind11::init<arma::mat, arma::vec, int, bool>(),
             pybind11::arg("X"),
             pybind11::arg("y"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step", (void (GPClassification::*)())    &GPClassification::step, "Run one sweep.")
        .def("step", (void (GPClassification::*)(int)) &GPClassification::step,  pybind11::arg("n_steps"))
        .def("get_current",  &GPClassification::get_current)
        .def("set_current",  &GPClassification::set_current, pybind11::arg("params"))
        .def("predict_at",   &GPClassification::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &GPClassification::get_dag)
        .def("get_history",  &GPClassification::get_history)
        .def("readapt_NUTS", (void (GPClassification::*)(int, bool, int, double)) &GPClassification::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false,
             pybind11::arg("max_tree_depth") = -1,
             pybind11::arg("target_accept") = -1.0)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(GPClassification);
}
#endif

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulates a 1D GP-classification dataset from a KNOWN latent function:
//      x_i      ~ Uniform(-3, 3)
//      f_true_i = 1.6 * sin(1.3 * x_i)          (a smooth latent signal)
//      p_true_i = sigmoid(f_true_i)
//      y_i      ~ Bernoulli(p_true_i)
//
//  Fits the model (ONE joint NUTS block over amplitude, lengthscale and the
//  whitened latent z, Bernoulli-logit likelihood), averages the posterior latent f -> posterior
//  class probability at each training point, and checks that:
//    (a) the posterior probabilities track the TRUE probabilities better than
//        a naive constant-rate baseline (mean(y)) -- i.e. lower mean abs error;
//    (b) the recovered probabilities are accurate in absolute terms;
//    (c) the covariance hyperparameters are actually INFORMED BY THE DATA.
//
//  Check (c) is the one that discriminates. A GP sampler whose hyperparameter
//  log-density omits the likelihood at the proposed (amplitude, lengthscale)
//  still passes (a) and (b) -- the latent fit still tracks the labels, just
//  worse -- while the hyperparameters are silently drawn from their priors.
//
//  Measured on this exact dataset, priors-only against correct:
//      MAE(post prob vs true)      0.1340   vs   0.0733
//      amplitude posterior SD      0.5883   vs   0.2645   (prior SD 0.6028)
//      lengthscale mean displaced  0.05     vs   0.26     prior SDs
//  so the first two separate cleanly and the third does not; only the first
//  two are asserted on. The lengthscale is left unasserted because binary
//  observations carry much less information about a covariance function than
//  Gaussian ones, and because these priors are centred close to the truth for
//  this design -- returning the prior already looks about right.
//
//  The amplitude check is a BAND, not a one-sided bound: a chain that locks
//  up reports a posterior SD of ZERO and would otherwise sail through a
//  "posterior is tighter than the prior" test. The lower edge rejects that.
//
//  State is read via the FULL contract get_current() (keys: f, amplitude,
//  lengthscale), matching the dual-module get_current() implementation.
//==============================================================================
#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>
int main() {
    const std::size_t N = 120;

    std::mt19937_64 sim_rng(20260621ULL);
    std::uniform_real_distribution<double> xdist(-3.0, 3.0);
    std::uniform_real_distribution<double> udist(0.0, 1.0);

    // Stable sigmoid for the simulation side.
    auto sig = [](double z) {
        if (z >= 0.0) { const double e = std::exp(-z); return 1.0 / (1.0 + e); }
        const double e = std::exp(z); return e / (1.0 + e);
    };

    arma::mat X(N, 1);
    arma::vec y(N);
    arma::vec p_true(N);
    double y_mean = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double x = xdist(sim_rng);
        X(i, 0) = x;
        const double f_true = 1.6 * std::sin(1.3 * x);
        const double pi = sig(f_true);
        p_true[i] = pi;
        y[i] = (udist(sim_rng) < pi) ? 1.0 : 0.0;
        y_mean += y[i];
    }
    y_mean /= static_cast<double>(N);

    GPClassification model(X, y, /*rng_seed=*/11, /*keep_history=*/false);
    model.step(300);   // warmup (joint block also self-warms n_warmup_first_call)

    // Posterior mean of sigmoid(f) at each training point, and the first two
    // moments of the two covariance hyperparameters.
    arma::vec prob_bar(N, arma::fill::zeros);
    const int M = 1500;
    double amp_bar = 0.0, ell_bar = 0.0, amp_sq = 0.0, ell_sq = 0.0;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto gc = model.get_current();        // copy (avoids dangling ref)
        const arma::vec& f = gc.at("f");
        for (std::size_t i = 0; i < N; ++i) {
            const double fi = f[i];
            const double pi = (fi >= 0.0)
                ? 1.0 / (1.0 + std::exp(-fi))
                : std::exp(fi) / (1.0 + std::exp(fi));
            prob_bar[i] += pi;
        }
        const double a = gc.at("amplitude")[0], e = gc.at("lengthscale")[0];
        amp_bar += a; amp_sq += a * a;
        ell_bar += e; ell_sq += e * e;
    }
    prob_bar /= static_cast<double>(M);
    const double Md = static_cast<double>(M);
    amp_bar /= Md; ell_bar /= Md;
    const double amp_sd_post = std::sqrt(std::max(amp_sq / Md - amp_bar * amp_bar, 0.0));
    const double ell_sd_post = std::sqrt(std::max(ell_sq / Md - ell_bar * ell_bar, 0.0));

    // ---- Analytic PRIOR moments (the discriminating baseline) ------------
    const double amp_sd_prior = 1.0 * std::sqrt(1.0 - 2.0 / M_PI);  // half-Normal(0,1)
    double ell_scale = 0.0;                                          // median pairwise dist
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
    const double ell_sd_prior   = ig_b / ((ig_a - 1.0) * std::sqrt(ig_a - 2.0));
    const double ell_mean_prior = ig_b / (ig_a - 1.0);
    const double amp_mean_prior = 1.0 * std::sqrt(2.0 / M_PI);   // half-Normal(0,1)

    // Mean absolute error of posterior prob vs TRUE prob, and of the naive
    // constant-rate baseline mean(y) vs TRUE prob.
    double mae_model = 0.0, mae_naive = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        mae_model += std::abs(prob_bar[i] - p_true[i]);
        mae_naive += std::abs(y_mean     - p_true[i]);
    }
    mae_model /= static_cast<double>(N);
    mae_naive /= static_cast<double>(N);

    const double ell_shift    = std::abs(ell_bar - ell_mean_prior) / ell_sd_prior;
    const double amp_sd_ratio = amp_sd_post / amp_sd_prior;

    std::printf("GPClassification demo (N=%zu, 1D GP-logit, whitened joint NUTS):\n", N);
    std::printf("  MAE(post prob vs true) = %.4f\n", mae_model);
    std::printf("  MAE(naive mean(y)=%.3f vs true) = %.4f\n", y_mean, mae_naive);
    std::printf("  amplitude   post %.4f +- %.4f   prior %.4f +- %.4f  (SD ratio %.3f)\n",
                amp_bar, amp_sd_post, amp_mean_prior, amp_sd_prior, amp_sd_ratio);
    std::printf("  lengthscale post %.4f +- %.4f   prior %.4f +- %.4f"
                "  (mean displaced %.2f prior SD)\n",
                ell_bar, ell_sd_post, ell_mean_prior, ell_sd_prior, ell_shift);

    const bool recovers = (mae_model < mae_naive) && (mae_model < 0.11);
    const bool amp_informed = amp_sd_ratio > 0.05 && amp_sd_ratio < 0.80;
    const bool ok = recovers && amp_informed;

    if (!recovers)
        std::printf("  [FAIL] posterior probabilities did not recover the latent surface\n");
    if (amp_sd_ratio >= 0.80)
        std::printf("  [FAIL] amplitude posterior is as wide as its prior\n");
    if (amp_sd_ratio <= 0.05)
        std::printf("  [FAIL] amplitude chain barely moved\n");

    std::printf("%s\n", ok
        ? "[demo PASS] GP-logit recovers latent class probabilities; "
          "hyperparameters informed by the data"
        : "[demo FAIL] latent surface / hyperparameters not recovered");
    return ok ? 0 : 1;
}
#endif

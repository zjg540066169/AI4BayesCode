// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// GPClassification_joint.cpp
//
// JOINT-NUTS rewrite of GPClassification.cpp. Same model, same priors,
// same posterior — but amplitude and lengthscale are sampled by ONE
// joint_nuts_block instead of two separate single nuts_blocks alternated
// Gibbs-style. The elliptical_slice_sampling_block for f stays separate.
//
// MODEL (identical to GPClassification.cpp)
// ------------------------------------------
//   y_i | f(x_i)  ~ Bernoulli(sigmoid(f(x_i))),   y_i in {0, 1}
//   f             ~ GP(0, K_amplitude_lengthscale(X, X))
//   amplitude     ~ half-Normal(0, 1)
//   lengthscale   ~ InverseGamma(5, median pairwise distance)
//
// JOINT natural-scale log-density over (amplitude, lengthscale):
//
//   log p(amplitude, lengthscale) =
//       -0.5 * amplitude^2 / amp_prior_sd^2                (half-Normal prior)
//     - (shape+1)*log(lengthscale) - scale/lengthscale      (InvGamma prior)
//
//   Each prior term appears exactly ONCE. The ESS block for f absorbs the
//   Bernoulli log-likelihood + GP prior on f; the NUTS hyperparameter
//   blocks sample only from the marginal hyperprior. No shared terms exist
//   between the two priors, so no double-counting is possible.
//
//   grad_{amplitude}   = -amplitude / amp_prior_sd^2
//   grad_{lengthscale} = -(shape+1)/lengthscale + scale/lengthscale^2
//
// BLOCKS (Gibbs sweep order: hyperparams FIRST, f LAST)
// -------------------------------------------------------
//   (amplitude, lengthscale) : ONE joint_nuts_block, sub_params
//                              [{amplitude,1,POSITIVE},{lengthscale,1,POSITIVE}]
//   f                        : elliptical_slice_sampling_block (unchanged)
//
// This file coexists with GPClassification.cpp for cross-validation.
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
//   m <- new(GPClassification, X, y, 7L, TRUE)   # X(Nx1), y(0/1), seed=7L, keep_history=TRUE
//   m$step(2500); str(m$get_current())           # single chain; amplitude, lengthscale, f
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2026)
//   N = 120
//   x = rng.uniform(-3.0, 3.0, N)                 # 1D inputs on [-3, 3]
//   f_true = 1.6 * np.sin(1.3 * x)                 # smooth latent GP-like signal
//   p_true = 1.0 / (1.0 + np.exp(-f_true))         # Bernoulli success prob
//   y = (rng.uniform(size=N) < p_true).astype(float)   # 0/1 labels
//   X = x.reshape(N, 1)                            # N x 1 design matrix
//   Mod = AI4BayesCode.source("GPClassification.cpp")
//   m = Mod.GPClassification(X, y, 7, True)        # X(Nx1), y(0/1), seed=7, keep_history=True
//   m.step(2500); print(m.get_current())          # dict: f, amplitude, lengthscale
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
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/elliptical_slice_sampling_block.hpp"

// Vendored libgp kernel subsystem (BSD-3).
#include "libgp_kernels_unity.h"

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
using AI4BayesCode::elliptical_slice_sampling_block;
using AI4BayesCode::elliptical_slice_sampling_block_config;
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
//  Joint natural-scale log-density over (amplitude, lengthscale).
//
//  theta_cat = [ amplitude, lengthscale ]  (NATURAL scale; both > 0,
//  guaranteed by the POSITIVE sub-param transforms inside joint_nuts_block).
//
//  log p(amplitude, lengthscale) =
//      -0.5 * amplitude^2 / amp_prior_sd^2        (half-Normal)
//    + (-(shape+1)*log(lengthscale) - scale/ell)   (InvGamma)
//
//  d/d amplitude   = -amplitude / amp_prior_sd^2
//  d/d lengthscale = -(shape+1)/lengthscale + scale/lengthscale^2
//
//  The ESS block absorbs the Bernoulli likelihood and GP prior on f; these
//  NUTS hyperparameter steps sample only from the marginal hyperprior.
//  joint_nuts_block adds POSITIVE-slice Jacobians (+log amplitude, +log ell)
//  internally; this function must NOT include them.
// ----------------------------------------------------------------------------
double amp_ell_joint_log_density(const arma::vec& theta_cat,
                                 const block_context& ctx,
                                 arma::vec* grad_nat) {
    const double amp   = theta_cat[0];
    const double ell   = theta_cat[1];

    if (amp <= 0.0 || ell <= 0.0) {
        if (grad_nat) grad_nat->set_size(2);
        return -std::numeric_limits<double>::infinity();
    }

    const double s        = ctx.at("amp_prior_sd")[0];
    const double s2       = s * s;
    const double shape    = ctx.at("ell_prior_shape")[0];
    const double scale    = ctx.at("ell_prior_scale")[0];

    // half-Normal prior on amplitude: -0.5 * amp^2 / s^2
    // InvGamma prior on lengthscale: -(shape+1)*log(ell) - scale/ell
    const double lp =
          -0.5 * amp * amp / s2
        + (-(shape + 1.0) * std::log(ell) - scale / ell);

    if (grad_nat) {
        grad_nat->set_size(2);
        (*grad_nat)[0] = -amp / s2;                           // d/d amplitude
        (*grad_nat)[1] = -(shape + 1.0) / ell + scale / (ell * ell); // d/d lengthscale
    }
    return lp;
}

}  // anonymous namespace

// ============================================================================
// User-facing class
// ============================================================================

class GPClassification {
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

        // Initial latent f = zeros (sigmoid(0) = 0.5; neutral prior start)
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
            {"amp_prior_sd", "ell_prior_shape", "ell_prior_scale"});
        impl_->data().declare_dependencies("f", {"L_chol", "y"});

        // ---- Invalidation chain -----------------------------------------
        // Keyed under the JOINT BLOCK NAME (composite_block calls
        // refresh_derived_for(block_name) after the joint block steps); keying
        // under sub-param names would never fire -> stale K_matrix/L_chol.
        impl_->data().declare_invalidates("amp_ell_joint",
            {"K_matrix", "L_chol"});

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

        // ---- L_chol refresher: chol(K + jitter I). NO sigma^2 here. ----
        const double jitter = 1e-5;
        impl_->data().register_refresher("L_chol",
            [N, jitter](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& K_flat = d.get("K_matrix");
                arma::mat K(const_cast<double*>(K_flat.memptr()), N, N,
                             /*copy_aux_mem=*/false, /*strict=*/true);
                arma::mat M = K;
                M.diag() += jitter;
                arma::mat L;
                if (!arma::chol(L, M, "lower")) {
                    M.diag() += 1e-3;
                    arma::chol(L, M, "lower");
                }
                return arma::vectorise(L);
            });

        impl_->data().refresh_all();

        // ---- Predict DAG + y_rep stochastic refresher -------------------
        impl_->data().declare_data_input("X");
        impl_->data().declare_predict_edges("X",        {"K_matrix"});
        impl_->data().declare_predict_edges("K_matrix", {"L_chol"});
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
        //   child(0) amp_ell_joint (joint_nuts_block for amplitude+lengthscale)
        //   child(1) f             (elliptical_slice_sampling_block)
        {
            joint_nuts_block_config cfg;
            cfg.name = "amp_ell_joint";
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "amplitude",   1,
                                      joint_constraint::POSITIVE });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "lengthscale", 1,
                                      joint_constraint::POSITIVE });
            // initial_cat is NATURAL-scale: [amp_init, ell_init].
            cfg.initial_cat = arma::vec{ amp_init, ell_init };
            cfg.log_density_grad = &amp_ell_joint_log_density;
            // amplitude and lengthscale operate on very different scales
            // (amp ~ O(1), ell ~ O(median pairwise dist)); diagonal metric
            // is required for faithful sampling of heterogeneous scales.
            cfg.use_diagonal_metric = true;
            // Give the joint block warmup runway.
            cfg.n_warmup_first_call = 800;
            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
        }
        {
            elliptical_slice_sampling_block_config cfg;
            cfg.name       = "f";
            cfg.N          = N;
            cfg.L_chol_key = "L_chol";
            cfg.initial_f  = arma::vec(N, arma::fill::zeros);
            // Bernoulli-logit log-likelihood: sum y_i f_i - softplus(f_i).
            cfg.log_lik =
                [](const arma::vec& f, const block_context& ctx) -> double {
                    const arma::vec& y = ctx.at("y");
                    double lp = 0.0;
                    for (std::size_t i = 0; i < y.n_elem; ++i) {
                        lp += y[i] * f[i] - stable_softplus(f[i]);
                    }
                    return lp;
                };
            impl_->add_child(
                std::make_unique<elliptical_slice_sampling_block>(
                    std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        // sub-params are written back to data() under sub-param names by
        // joint_nuts_block; read directly from data(). Each output is an
        // arma::vec; the frontend converts state_map -> R list / Python dict.
        AI4BayesCode::state_map out;
        out["f"]           = impl_->data().get("f");                 // length N
        out["amplitude"]   = impl_->data().get("amplitude");         // length 1
        out["lengthscale"] = impl_->data().get("lengthscale");       // length 1
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        // For the joint block: read its current concatenated vector, overwrite
        // the relevant slice(s), set_current, mirror to data(). Each value in
        // params is an arma::vec (frontend already converted list/dict).
        auto& jblk = dynamic_cast<joint_nuts_block&>(impl_->child(0));
        arma::vec cat_new = jblk.current();   // [amplitude, lengthscale]
        bool touched_joint = false;

        auto it_amp = params.find("amplitude");
        if (it_amp != params.end()) {
            const double a = it_amp->second[0];
            if (!(a > 0.0)) ai4b::stop("amplitude must be positive");
            cat_new[0] = a;
            touched_joint = true;
        }
        auto it_ell = params.find("lengthscale");
        if (it_ell != params.end()) {
            const double l = it_ell->second[0];
            if (!(l > 0.0)) ai4b::stop("lengthscale must be positive");
            cat_new[1] = l;
            touched_joint = true;
        }
        if (touched_joint) {
            jblk.set_current(cat_new);
            impl_->data().set("amplitude",   arma::vec{cat_new[0]});
            impl_->data().set("lengthscale", arma::vec{cat_new[1]});
            impl_->data().refresh_derived_for("amplitude");
            impl_->data().refresh_derived_for("lengthscale");
        }

        auto it_f = params.find("f");
        if (it_f != params.end()) {
            const arma::vec& f_new = it_f->second;
            if (static_cast<std::size_t>(f_new.n_elem) != N_)
                ai4b::stop("GPClassification::set_current: f length mismatch");
            impl_->data().set("f", f_new);
            dynamic_cast<elliptical_slice_sampling_block&>(
                impl_->child(1)).set_current(f_new);
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
                AI4BayesCode::history_map hist = impl_->get_history();
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
            AI4BayesCode::history_map hist = impl_->get_history();
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
                K_train_d.diag() += 1e-8;
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

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

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
            "Joint-NUTS GP classification: (amplitude, lengthscale) in one "
            "joint_nuts_block + ESS on latent f + Bernoulli-logit likelihood. "
            "Inputs: X (N x p), y in {0,1} length N, rng_seed, keep_history.")
        .method("step",         &GPClassification::step)
        .method("get_current",  &GPClassification::get_current)
        .method("set_current",  &GPClassification::set_current)
        .method("predict_at",   &GPClassification::predict_at)
        .method("get_dag",      &GPClassification::get_dag)
        .method("get_history",  &GPClassification::get_history)
        .method("readapt_NUTS", &GPClassification::readapt_NUTS);
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
        .def("step",         &GPClassification::step,  pybind11::arg("n_steps"))
        .def("get_current",  &GPClassification::get_current)
        .def("set_current",  &GPClassification::set_current, pybind11::arg("params"))
        .def("predict_at",   &GPClassification::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &GPClassification::get_dag)
        .def("get_history",  &GPClassification::get_history)
        .def("readapt_NUTS", &GPClassification::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false);
}
#endif

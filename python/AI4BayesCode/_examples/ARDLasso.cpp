// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  ARDLasso.cpp
//
//  Automatic Relevance Determination (ARD) LASSO for D=1.
//
//  **LEGACY self-contained example** (pre-dates AI4BayesCode's block system) —
//  all full conditionals are Jeffreys-based conjugate closed forms drawn
//  inline, and the class manages its own state, history, and DAG without
//  `composite_block` / `nuts_block` / `declare_dependencies` / `refresher`.
//  This is the ONE documented exception to the "compose typed blocks"
//  invariant and to Check #17 (no hand-rolled distribution samplers in
//  examples) — both `std::gamma_distribution` (for InvGamma σ² and Gamma
//  ψ² draws) and `std::normal_distribution` (for α and β conditionals) are
//  used inline. These are all textbook conjugate conditionals and derivations
//  have been formally verified against Park & Casella 2008 (Bayesian LASSO).
//  When generating new samplers, DO NOT copy this pattern — use the block
//  system (nuts_block + rjmcmc_block + ... as per `skills/codegen.md §2b`).
//  This file remains here as a reference implementation of the pre-block
//  ARD Gibbs pattern; modernizing it is tracked in `todo/todo.md`.
//
//  DUAL-MODULE: this legacy example is exposed to BOTH R (RCPP_MODULE) and
//  Python (PYBIND11_MODULE). The class I/O methods use the backend-neutral
//  AI4BayesCode::state_map / history_map / dag_info API (NOT raw Rcpp::List),
//  so the same class binds verbatim in both frontends.
//
//  Model (Park & Casella 2008, Bayesian LASSO, Eq. 2.3-2.4 with D=1)
//  ------------------------------------------------------------------
//      Y | beta, alpha, sigma  ~ Normal(alpha*1 + X*beta, sigma^2 * I)
//      beta_j | sigma, psi2_j  ~ N(0, sigma^2 / psi2_j)      (slab prior)
//      psi2_j                  ~ Gamma(D/2, rate = S_j / (2 sigma^2))
//                                 (Jeffreys on psi^2, D=1 here)
//      alpha                   ~ improper flat prior (full conditional is
//                                 N(ybar_residual, sigma^2/N))
//      sigma^2                 ~ Jeffreys p(sigma^2) ∝ 1/sigma^2 (full
//                                 conditional is InvGamma((N+p)/2,
//                                 (SSE+SSE_beta)/2))
//
//  Gibbs sweep order: alpha -> sigma2 -> psi2 -> beta
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("ARDLasso")
//   set.seed(20260621)
//   N <- 200L; p <- 6L                              # N >> p: well-identified
//   beta_true <- c(2.5, 0, -1.5, 0, 0.8, 0)         # 3 active, 3 exactly-zero
//   alpha_true <- 1.0; sigma_true <- 0.5
//   X <- matrix(rnorm(N * p), N, p)                 # NO intercept column (alpha is built in)
//   Y <- as.numeric(alpha_true + X %*% beta_true + sigma_true * rnorm(N))
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- AI4BayesCode_run_chains(
//       function(seed) new(ARDLasso, X, Y, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4b_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(ARDLasso, X, Y, 20260621L, TRUE)       # X (Nxp), Y (len N), seed, keep_history
//   m$step(2500); str(m$get_current())
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(20260621)
//   N, p = 200, 6                                    # N >> p: well-identified
//   beta_true = np.array([2.5, 0.0, -1.5, 0.0, 0.8, 0.0])  # 3 active, 3 zero
//   alpha_true, sigma_true = 1.0, 0.5
//   X = rng.normal(size=(N, p))                      # NO intercept column (alpha is built in)
//   Y = alpha_true + X @ beta_true + sigma_true * rng.normal(size=N)
//   Mod = AI4BayesCode.source("ARDLasso.cpp")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.ARDLasso(X, Y, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.ai4b_diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.ARDLasso(X, Y, 20260621, True)           # (X (Nxp), Y (len N), seed, keep_history)
//   m.step(2500); print(m.get_current())
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

#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"   // also pulls in types.hpp (state_map, ...)

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

class ARDLasso {
public:
    ARDLasso(const arma::mat& X, const arma::vec& Y, int rng_seed,
             bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          X_(X), Y_(Y),
          N_(X.n_rows), p_(X.n_cols),
          keep_history_(keep_history)
    {
        if (Y_.n_elem != N_) ai4b::stop("X rows must match Y length");

        // Precompute
        XtX_ = X_.t() * X_;
        XtY_ = X_.t() * Y_;
        YtY_ = arma::dot(Y_, Y_);
        Ysum_ = arma::sum(Y_);
        Xt1_ = X_.t() * arma::ones<arma::vec>(N_);  // colSums(X)

        // OLS initialization
        if (p_ < N_) {
            beta_ = arma::solve(XtX_, XtY_, arma::solve_opts::likely_sympd);
            arma::vec resid = Y_ - X_ * beta_;
            alpha_ = arma::mean(resid);
            double SS = arma::dot(resid - alpha_, resid - alpha_);
            sigma2_ = std::max(SS / (N_ - p_), 1e-6);
        } else {
            beta_ = arma::vec(p_, arma::fill::zeros);
            alpha_ = arma::mean(Y_);
            sigma2_ = arma::var(Y_);
        }
        psi2_ = arma::vec(p_, arma::fill::ones);
    }

    void step(int n_steps) {
        std::normal_distribution<double> norm01(0.0, 1.0);

        for (int iter = 0; iter < n_steps; ++iter) {
            // ---- 1. Update alpha ~ N(mean(Y - X*beta), sigma2/N) ----
            double Xbeta_sum = arma::dot(Xt1_, beta_);
            double mu_alpha = (Ysum_ - Xbeta_sum) / N_;
            double sd_alpha = std::sqrt(sigma2_ / N_);
            alpha_ = mu_alpha + sd_alpha * norm01(rng_);

            // ---- 2. Update sigma2 ~ InvGamma((N+p)/2, (SSE+SSE_beta)/2) ----
            double SSE = YtY_
                - 2.0 * alpha_ * Ysum_
                + N_ * alpha_ * alpha_
                - 2.0 * arma::dot(beta_, XtY_)
                + 2.0 * alpha_ * arma::dot(Xt1_, beta_)
                + arma::dot(beta_, XtX_ * beta_);
            double SSE_beta = arma::sum(psi2_ % beta_ % beta_);
            double ig_shape = (N_ + p_) / 2.0;
            double ig_scale = (SSE + SSE_beta) / 2.0;
            if (ig_scale < 1e-10) ig_scale = 1e-10;
            // InvGamma: draw g ~ Gamma(shape, rate=scale), return 1/g
            std::gamma_distribution<double> gam_sig(ig_shape, 1.0 / ig_scale);
            double g = gam_sig(rng_);
            if (g < 1e-300) g = 1e-300;
            sigma2_ = 1.0 / g;

            // ---- 3. Update psi2_j ~ Gamma(D/2, scale=2*sigma2/S_j) ----
            // D=1: shape = 0.5
            for (std::size_t j = 0; j < p_; ++j) {
                double S_j = beta_[j] * beta_[j];
                if (S_j < 1e-20) S_j = 1e-20;
                double scale_j = 2.0 * sigma2_ / S_j;
                std::gamma_distribution<double> gam_psi(0.5, scale_j);
                psi2_[j] = gam_psi(rng_);
                // Clamp to prevent overflow
                if (psi2_[j] > 1e6) psi2_[j] = 1e6;
                if (psi2_[j] < 1e-300) psi2_[j] = 1e-300;
            }

            // ---- 4. Update beta ~ MVN(mu, sigma2 * V) ----
            // V = (XtX + diag(psi2))^{-1}
            // mu = V * (XtY - alpha * Xt1)
            arma::mat V = arma::inv_sympd(XtX_ + arma::diagmat(psi2_));
            arma::vec mu_beta = V * (XtY_ - alpha_ * Xt1_);
            arma::mat L;
            arma::chol(L, sigma2_ * V, "lower");
            arma::vec z(p_);
            for (std::size_t j = 0; j < p_; ++j) z[j] = norm01(rng_);
            beta_ = mu_beta + L * z;

            // ---- Append to history if enabled -------------------------
            if (keep_history_) {
                beta_hist_.push_back(beta_);
                alpha_hist_.push_back(alpha_);
                sigma2_hist_.push_back(sigma2_);
                psi2_hist_.push_back(psi2_);
            }
        }
    }

    // Backend-neutral snapshot: keys -> arma::vec (scalars are length-1).
    // Frontend converts state_map -> R named list / Python dict.
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["beta"]   = beta_;
        out["alpha"]  = arma::vec{alpha_};
        out["sigma2"] = arma::vec{sigma2_};
        out["psi2"]   = psi2_;
        return out;
    }

    // Backend-neutral overwrite: each value is an arma::vec (frontend already
    // converted the R list / Python dict). Scalars arrive as length-1 vectors.
    void set_current(const AI4BayesCode::state_map& params) {
        auto it_beta = params.find("beta");
        if (it_beta != params.end()) {
            if (it_beta->second.n_elem != p_)
                ai4b::stop("set_current: beta has wrong length");
            beta_ = it_beta->second;
        }
        auto it_alpha = params.find("alpha");
        if (it_alpha != params.end()) alpha_ = it_alpha->second[0];
        auto it_sigma2 = params.find("sigma2");
        if (it_sigma2 != params.end()) {
            const double s2 = it_sigma2->second[0];
            if (s2 <= 0.0) ai4b::stop("set_current: sigma2 must be > 0");
            sigma2_ = s2;
        }
        auto it_psi2 = params.find("psi2");
        if (it_psi2 != params.end()) {
            if (it_psi2->second.n_elem != p_)
                ai4b::stop("set_current: psi2 has wrong length");
            psi2_ = it_psi2->second;
        }
    }

    // Full model DAG, hand-coded because ARDLasso is a self-contained
    // Gibbs sampler (no composite_block). Structure mirrors the Gibbs
    // sweep order alpha -> sigma2 -> psi2 -> beta and the generative
    // relationship X -> y_hat used by predict_at.
    AI4BayesCode::dag_info get_dag() const {
        AI4BayesCode::dag_info d;
        d.gibbs_reads["alpha"]  = {"Y", "X", "beta", "sigma2"};
        d.gibbs_reads["sigma2"] = {"Y", "X", "beta", "alpha", "psi2"};
        d.gibbs_reads["psi2"]   = {"beta", "sigma2"};
        d.gibbs_reads["beta"]   = {"Y", "X", "alpha", "sigma2", "psi2"};
        // no refreshers -> gibbs_invalidates stays empty
        d.predict_edges["X"]      = {"y_hat"};
        d.predict_edges["beta"]   = {"y_hat"};
        d.predict_edges["alpha"]  = {"y_hat"};
        d.predict_edges["y_hat"]  = {"y_rep"};
        d.predict_edges["sigma2"] = {"y_rep"};
        d.data_inputs = {"X"};
        return d;
    }

    // predict_at (backend-neutral):
    //   INPUT  new_data["X"] : vectorised N_new*p arma::vec, COLUMN-MAJOR
    //          (element (i,j) at index i + j*N_new), matching arma::vectorise
    //          of an N_new x p design. R/Python callers pass as.vector(X_new) /
    //          X_new flattened column-major. Empty map -> posterior predictive
    //          at the TRAINING X.
    //   OUTPUT every key is an arma::mat. keep_history = FALSE returns 1-row
    //          matrices (single predict at the current draw); keep_history =
    //          TRUE returns n_draws-row matrices (over all stored draws).
    //            $y_hat (only when X is supplied) = X_use * beta + alpha
    //            $y_rep                           = y_hat + sigma * Normal(0,1)
    //
    // Uses predict_rng_ for reproducible posterior-predictive sampling.
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        // ---- Parse optional X (vectorised N_new*p, column-major) ----------
        for (const auto& kv : new_data) {
            if (kv.first != "X")
                ai4b::stop("predict_at: unknown key '%s' (only 'X' is accepted)",
                           kv.first.c_str());
        }
        const bool has_X = new_data.count("X") > 0;
        arma::mat X_use;
        if (has_X) {
            const arma::vec& x_flat = new_data.at("X");
            if (p_ == 0 || x_flat.n_elem % p_ != 0)
                ai4b::stop("predict_at: X must be a vectorised N_new*p column-"
                           "major vector (n_elem must be a multiple of p=%d)",
                           static_cast<int>(p_));
            const std::size_t N_new = x_flat.n_elem / p_;
            X_use.set_size(N_new, p_);
            for (std::size_t i = 0; i < N_new; ++i)
                for (std::size_t j = 0; j < p_; ++j)
                    X_use(i, j) = x_flat[i + j * N_new];
        } else {
            X_use = X_;   // training X
        }

        std::normal_distribution<double> norm01(0.0, 1.0);
        AI4BayesCode::history_map out;

        if (keep_history_ && !beta_hist_.empty()) {
            const std::size_t n = beta_hist_.size();
            const std::size_t m = X_use.n_rows;
            arma::mat Y_hat(n, m);
            arma::mat Y_rep(n, m);
            for (std::size_t i = 0; i < n; ++i) {
                arma::vec yh = X_use * beta_hist_[i] + alpha_hist_[i];
                const double sig_d = std::sqrt(sigma2_hist_[i]);
                for (std::size_t j = 0; j < m; ++j) {
                    Y_hat(i, j) = yh[j];
                    Y_rep(i, j) = yh[j] + sig_d * norm01(predict_rng_);
                }
            }
            if (has_X) out.emplace("y_hat", std::move(Y_hat));  // mean only meaningful at X_new
            out.emplace("y_rep", std::move(Y_rep));
            return out;
        }

        // Stateful mode (single current draw -> 1-row matrices)
        arma::vec y_hat = X_use * beta_ + alpha_;
        const double sig_cur = std::sqrt(sigma2_);
        const std::size_t m = y_hat.n_elem;
        arma::mat y_hat_out(1, m);
        arma::mat y_rep_out(1, m);
        for (std::size_t i = 0; i < m; ++i) {
            y_hat_out(0, i) = y_hat[i];
            y_rep_out(0, i) = y_hat[i] + sig_cur * norm01(predict_rng_);
        }
        if (has_X) out.emplace("y_hat", std::move(y_hat_out));
        out.emplace("y_rep", std::move(y_rep_out));
        return out;
    }

    // History returned as a history_map (name -> arma::mat). ARDLasso doesn't
    // use a composite, so we build it by hand — matches the structure
    // composite_block's merged bundle would produce for the same parameters.
    //   $beta   : n_draws x p matrix
    //   $alpha  : n_draws x 1 matrix
    //   $sigma2 : n_draws x 1 matrix
    //   $psi2   : n_draws x p matrix
    // If no draws are stored yet, returns a 1-row fallback with the
    // current values (matches the composite/leaf fallback convention).
    AI4BayesCode::history_map get_history() const {
        const std::size_t n_rec = beta_hist_.size();
        const std::size_t n     = n_rec == 0 ? 1 : n_rec;

        arma::mat beta_mat(n, p_);
        arma::mat psi2_mat(n, p_);
        arma::mat alpha_mat(n, 1);
        arma::mat sigma2_mat(n, 1);

        if (n_rec == 0) {
            for (std::size_t j = 0; j < p_; ++j) {
                beta_mat(0, j) = beta_[j];
                psi2_mat(0, j) = psi2_[j];
            }
            alpha_mat(0, 0)  = alpha_;
            sigma2_mat(0, 0) = sigma2_;
        } else {
            for (std::size_t i = 0; i < n_rec; ++i) {
                for (std::size_t j = 0; j < p_; ++j) {
                    beta_mat(i, j) = beta_hist_[i][j];
                    psi2_mat(i, j) = psi2_hist_[i][j];
                }
                alpha_mat(i, 0)  = alpha_hist_[i];
                sigma2_mat(i, 0) = sigma2_hist_[i];
            }
        }
        AI4BayesCode::history_map out;
        out.emplace("beta",   std::move(beta_mat));
        out.emplace("alpha",  std::move(alpha_mat));
        out.emplace("sigma2", std::move(sigma2_mat));
        out.emplace("psi2",   std::move(psi2_mat));
        return out;
    }

private:
    std::mt19937_64 rng_;
    mutable std::mt19937_64 predict_rng_;
    arma::mat X_;
    arma::vec Y_;
    std::size_t N_, p_;
    bool keep_history_ = false;

    // Precomputed
    arma::mat XtX_;
    arma::vec XtY_;
    arma::vec Xt1_;
    double YtY_, Ysum_;

    // State
    arma::vec beta_;
    double alpha_;
    double sigma2_;
    arma::vec psi2_;

    // History buffers (populated only when keep_history_ == true)
    std::vector<arma::vec> beta_hist_;
    std::vector<double>    alpha_hist_;
    std::vector<double>    sigma2_hist_;
    std::vector<arma::vec> psi2_hist_;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(ARDLasso_module) {
    Rcpp::class_<ARDLasso>("ARDLasso")
        // Backward-compatible constructor (keep_history defaults to FALSE).
        .constructor<arma::mat, arma::vec, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::mat, arma::vec, int, bool>(
            "Construct with: X (N x p), Y (length N), seed, "
            "keep_history (record every draw; default FALSE).")
        .method("step",        &ARDLasso::step,
                "Run n Gibbs sweeps.")
        .method("get_current", &ARDLasso::get_current,
                "Return list(beta, alpha, sigma2, psi2).")
        .method("set_current", &ARDLasso::set_current,
                "Overwrite parameters from a named list.")
        .method("get_dag",     &ARDLasso::get_dag,
                "Full model DAG (gibbs_reads / gibbs_invalidates / "
                "predict_edges / data_inputs).")
        .method("predict_at",  &ARDLasso::predict_at,
                "Predict: list(X = as.vector(X_new)) -> list(y_hat, y_rep). "
                "Returns 1-row matrices in stateful mode, n_draws-row matrices "
                "in history mode. Empty list -> posterior predictive at "
                "training X.")
        .method("get_history", &ARDLasso::get_history,
                "History as a named list: beta (n_draws x p), alpha "
                "(n_draws x 1), sigma2 (n_draws x 1), psi2 (n_draws x p).");
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(ARDLasso, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<ARDLasso>(m, "ARDLasso")
        .def(pybind11::init<arma::mat, arma::vec, int, bool>(),
             pybind11::arg("X"),
             pybind11::arg("Y"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step",        &ARDLasso::step,        pybind11::arg("n_steps"))
        .def("get_current", &ARDLasso::get_current)
        .def("set_current", &ARDLasso::set_current, pybind11::arg("params"))
        .def("get_dag",     &ARDLasso::get_dag)
        .def("predict_at",  &ARDLasso::predict_at,  pybind11::arg("new_data"))
        .def("get_history", &ARDLasso::get_history);
}
#endif

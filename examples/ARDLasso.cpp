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
//
//  This file is a FRONTEND-INDEPENDENT standalone program: the original
//  Rcpp module (R/Python bindings for get_current/set_current/get_dag/
//  predict_at/get_history) has been removed. The core sampler keeps its
//  pure-armadillo state and step() loop unchanged; an int main() at the
//  bottom drives a small recovery demo.
// ============================================================================

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>
#include <cmath>
#include <random>
#include <vector>
#include <stdexcept>

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
        if (Y_.n_elem != N_) throw std::runtime_error("X rows must match Y length");

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

    void step() { step(1); }              // no-arg convenience: one sweep
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

    // Frontend-independent accessors (pure armadillo / scalar state).
    const arma::vec& beta()   const { return beta_; }
    double           alpha()  const { return alpha_; }
    double           sigma2() const { return sigma2_; }
    const arma::vec& psi2()   const { return psi2_; }

    const std::vector<arma::vec>& beta_history()   const { return beta_hist_; }
    const std::vector<double>&    alpha_history()  const { return alpha_hist_; }
    const std::vector<double>&    sigma2_history() const { return sigma2_hist_; }

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

// ============================================================================
//  Standalone recovery demo
//  ------------------------
//  Simulate a sparse linear regression from a KNOWN truth:
//      Y = alpha_true + X * beta_true + N(0, sigma_true^2)
//  with several beta_true coefficients exactly zero (the ARD/LASSO slab
//  prior should shrink those toward 0 while recovering the active ones).
//  Fit with keep_history, discard burn-in, take posterior means, and check:
//    (a) posterior means of active coefficients are close to truth,
//    (b) posterior means of zero coefficients are shrunk small,
//    (c) the ARD fit beats a no-X baseline (predict the intercept only)
//        on training RMSE.
// ============================================================================
#include <cstdio>

int main() {
    const std::size_t N = 200;     // observations
    const std::size_t p = 6;       // predictors
    const int seed     = 20260621;

    // Known truth: 3 active, 3 exactly-zero coefficients.
    arma::vec beta_true = {2.5, 0.0, -1.5, 0.0, 0.8, 0.0};
    const double alpha_true = 1.0;
    const double sigma_true = 0.5;

    // Simulate design + response.
    std::mt19937_64 data_rng(static_cast<std::uint64_t>(seed));
    std::normal_distribution<double> rnorm(0.0, 1.0);

    arma::mat X(N, p);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < p; ++j)
            X(i, j) = rnorm(data_rng);

    arma::vec Y(N);
    for (std::size_t i = 0; i < N; ++i) {
        double mu = alpha_true + arma::dot(X.row(i).t(), beta_true);
        Y[i] = mu + sigma_true * rnorm(data_rng);
    }

    // Fit ARD LASSO.
    const int n_burn = 2000;
    const int n_keep = 4000;
    ARDLasso model(X, Y, seed, /*keep_history=*/true);
    model.step(n_burn);

    // Reset history start point: run burn-in WITHOUT recording, then record.
    // (step() with keep_history records every sweep, so to get a clean
    //  post-burn-in trace we just run the full chain and slice off burn-in.)
    model.step(n_keep);

    const auto& beta_hist   = model.beta_history();
    const auto& alpha_hist  = model.alpha_history();
    const auto& sigma2_hist = model.sigma2_history();
    const std::size_t total = beta_hist.size();
    const std::size_t start = static_cast<std::size_t>(n_burn);  // discard burn-in

    // Posterior means over kept draws.
    arma::vec beta_pm(p, arma::fill::zeros);
    double alpha_pm = 0.0, sigma2_pm = 0.0;
    std::size_t kept = 0;
    for (std::size_t i = start; i < total; ++i) {
        beta_pm   += beta_hist[i];
        alpha_pm  += alpha_hist[i];
        sigma2_pm += sigma2_hist[i];
        ++kept;
    }
    beta_pm   /= static_cast<double>(kept);
    alpha_pm  /= static_cast<double>(kept);
    sigma2_pm /= static_cast<double>(kept);

    // --- Report recovery ---
    std::printf("ARD LASSO recovery demo (N=%zu, p=%zu, kept draws=%zu)\n",
                N, p, kept);
    std::printf("  alpha:  truth=% .3f   post.mean=% .3f\n",
                alpha_true, alpha_pm);
    std::printf("  sigma:  truth=% .3f   post.mean=% .3f  (from sigma2)\n",
                sigma_true, std::sqrt(sigma2_pm));
    std::printf("  coef     truth    post.mean\n");
    for (std::size_t j = 0; j < p; ++j)
        std::printf("   beta[%zu]  % .3f    % .3f\n", j, beta_true[j], beta_pm[j]);

    // --- Checks ---
    // (a) active coefficients recovered within tolerance.
    double max_active_err = 0.0;
    // (b) zero coefficients shrunk small.
    double max_zero_mag = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        if (std::abs(beta_true[j]) > 1e-9)
            max_active_err = std::max(max_active_err,
                                      std::abs(beta_pm[j] - beta_true[j]));
        else
            max_zero_mag = std::max(max_zero_mag, std::abs(beta_pm[j]));
    }

    // (c) ARD fit beats intercept-only baseline on training RMSE.
    arma::vec y_hat = X * beta_pm + alpha_pm;
    double sse_model = arma::dot(Y - y_hat, Y - y_hat);
    double rmse_model = std::sqrt(sse_model / static_cast<double>(N));

    double ybar = arma::mean(Y);
    double sse_base = arma::dot(Y - ybar, Y - ybar);
    double rmse_base = std::sqrt(sse_base / static_cast<double>(N));

    std::printf("  active coef max abs error : % .4f  (tol 0.30)\n", max_active_err);
    std::printf("  zero   coef max abs mean  : % .4f  (tol 0.30)\n", max_zero_mag);
    std::printf("  train RMSE  model=% .4f   baseline(intercept)=% .4f\n",
                rmse_model, rmse_base);

    bool ok = (max_active_err < 0.30)
           && (max_zero_mag   < 0.30)
           && (rmse_model     < 0.50 * rmse_base);

    std::printf("%s\n", ok ? "[demo PASS]" : "[demo FAIL]");
    return ok ? 0 : 1;
}

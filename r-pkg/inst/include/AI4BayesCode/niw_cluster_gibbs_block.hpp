/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  niw_cluster_gibbs_block.hpp -- Full-covariance NIW companion to
 *      normal_gamma_cluster_gibbs_block. Samples per-cluster
 *      (mu_k, Sigma_k) jointly across K_trunc clusters under a
 *      conjugate Normal-Inverse-Wishart prior, with empty clusters
 *      drawing from the prior (Ishwaran & James 2001 §3.2 convention).
 *
 *  WHY THIS BLOCK EXISTS (vs the diagonal Normal-Gamma sibling)
 *  ============================================================
 *  The shipped `normal_gamma_cluster_gibbs_block` assumes diagonal
 *  cluster covariance — fine when the cluster's d dimensions are
 *  approximately independent. For data with off-diagonal correlation
 *  inside clusters (e.g., correlated body measurements, joint x-y
 *  position with a tilt, multi-channel spectra), full NIW gives the
 *  textbook result and recovers the cluster shape, not just its
 *  axis-aligned envelope.
 *
 *  PRIOR (per cluster, identical across clusters)
 *  ----------------------------------------------
 *      Sigma_k ~ Inverse-Wishart(Psi_0, nu_0)
 *      mu_k | Sigma_k ~ N(mu_0, Sigma_k / kappa_0)
 *
 *  Equivalently (mu_k, Sigma_k) ~ NIW(mu_0, kappa_0, Psi_0, nu_0).
 *  Requires: Psi_0 d×d positive definite, nu_0 > d - 1, kappa_0 > 0.
 *
 *  POSTERIOR (per cluster k, with n_k observations y^k = {y_i : z_i = k+1})
 *  ----------------------------------------------------------------------
 *  bar_y_k = sum(y^k) / n_k                       (when n_k > 0)
 *  S_k     = sum_i (y_i^k - bar_y_k)(y_i^k - bar_y_k)^T
 *
 *  kappa_n = kappa_0 + n_k
 *  nu_n    = nu_0 + n_k
 *  mu_n    = (kappa_0 * mu_0 + n_k * bar_y_k) / kappa_n
 *  Psi_n   = Psi_0 + S_k
 *          + (kappa_0 * n_k / kappa_n) * (bar_y_k - mu_0)(bar_y_k - mu_0)^T
 *
 *  Sigma_k | data ~ IW(Psi_n, nu_n)
 *  mu_k | Sigma_k, data ~ N(mu_n, Sigma_k / kappa_n)
 *
 *  When n_k == 0 (empty cluster), draw directly from the prior
 *  (Sigma_k ~ IW(Psi_0, nu_0); mu_k | Sigma_k ~ N(mu_0, Sigma_k/kappa_0)).
 *
 *  Reference: Murphy 2007, "Conjugate Bayesian analysis of the Gaussian
 *  distribution", §4 (equations 4-11). Same scheme as Bishop PRML §10.2.
 *
 *  IW SAMPLING via BARTLETT DECOMPOSITION (Bartlett 1933, Anderson 2003)
 *  --------------------------------------------------------------------
 *  Given X ~ Wishart(I_d, df), one can sample
 *      X = A A^T
 *  where A is lower-triangular with
 *      A_ii ~ sqrt(chi-squared(df - i + 1))      (i = 0, ..., d-1)
 *      A_ij ~ N(0, 1)                             (i > j)
 *      A_ij = 0                                   (i < j)
 *
 *  To sample W ~ Wishart(V, df) where V = L L^T (Cholesky of V):
 *      W = (L A) (L A)^T
 *
 *  To sample Sigma ~ IW(Psi, df) given Psi: use the inverse relation
 *      Sigma^{-1} ~ Wishart(Psi^{-1}, df)
 *  i.e.,
 *      L_inv = chol(Psi^{-1})    (could derive via solve(chol(Psi), I))
 *      sample A as above
 *      Sigma^{-1} = (L_inv A) (L_inv A)^T
 *      Sigma = inv((L_inv A) (L_inv A)^T)
 *
 *  Numerical stability: we form
 *      M = L_psi_inv * A           (lower-triangular product)
 *      Sigma = inv(M M^T)
 *  via two triangular solves on M. No explicit matrix inversion on
 *  the IW path itself.
 *
 *  STORAGE / OUTPUT
 *  ----------------
 *  Two named outputs (current_named_outputs):
 *    - cfg.mu_name:    length K_trunc * d, cluster-major row order
 *                      (mu_1[0], ..., mu_1[d-1], mu_2[0], ..., mu_K[d-1]).
 *    - cfg.sigma_name: length K_trunc * d * d, cluster-major then row-
 *                      major within each d×d block:
 *                      sigma_flat[k * d * d + i * d + j] = Sigma_k(i, j).
 *                      Symmetric. Holds COVARIANCE, NOT precision.
 *
 *  current() returns the concatenated [mu; sigma] of length K*d + K*d*d.
 *  set_current(theta) accepts the same concatenated vector.
 *
 *  COMMENT-PARAMETERIZATION DISCIPLINE (rcpp_api.md §11)
 *  -----------------------------------------------------
 *  - std::gamma_distribution<double>(shape, scale) is shape-SCALE,
 *    E[X] = shape * scale.
 *  - To sample chi-squared(df): use std::gamma_distribution<double>(df/2, 2.0)
 *    so E[X] = (df/2) * 2 = df. ✓
 *  - To sample sqrt(chi-squared(df)) for Bartlett: sqrt of the above.
 *  - std::normal_distribution<double>(mean, stddev) — stddev, NOT variance.
 *
 *  Each call site carries a comment confirming the parameterisation.
 *
 *  RELATIONSHIP TO normal_gamma_cluster_gibbs_block
 *  -----------------------------------------------
 *  At d == 1, NIW reduces to NormalGamma (with Sigma a 1×1 = variance).
 *  This block could be used at d == 1 but normal_gamma_cluster is
 *  faster there (it stores precisions directly without 1×1 matrix
 *  bookkeeping). Use this block for d ≥ 2 with off-diagonal structure.
 *================================================================================*/

#ifndef AI4BAYESCODE_NIW_CLUSTER_GIBBS_BLOCK_HPP
#define AI4BAYESCODE_NIW_CLUSTER_GIBBS_BLOCK_HPP

#include "block_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

struct niw_cluster_gibbs_block_config {
    /// Block label (used by composite for declare_dependencies /
    /// declare_invalidates). NOT used as a shared_data key.
    std::string name;

    /// Truncation level (number of clusters).
    std::size_t K_trunc = 0;

    /// Observation dimension.
    std::size_t d = 0;

    /// Number of observations.
    std::size_t N = 0;

    /// Shared_data key for cluster assignments z (length N, values in
    /// {1, ..., K_trunc}).
    std::string z_key = "z";

    /// Shared_data key for the observed data y (length N * d, row-major).
    std::string y_key = "y";

    /// Output key for the K_trunc * d flat mu vector.
    std::string mu_name = "mu";

    /// Output key for the K_trunc * d * d flat Sigma vector
    /// (cluster-major, row-major within each d×d block).
    /// Holds COVARIANCE matrices, not precisions.
    std::string sigma_name = "sigma";

    /// Prior hyperparameters (shared across all clusters).
    arma::vec mu_0;             // length d: prior mean of mu
    double    kappa_0   = 0.0;  // prior precision multiplier on mu
    arma::vec Psi_0_flat;       // length d*d, row-major; PSD scale matrix
    double    nu_0      = 0.0;  // IW degrees of freedom; require nu_0 > d - 1

    /// Initial values, cluster-major. mu: K*d, sigma: K*d*d.
    arma::vec initial_mu;
    arma::vec initial_sigma;
};

class niw_cluster_gibbs_block : public block_sampler {
public:
    explicit niw_cluster_gibbs_block(niw_cluster_gibbs_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.name.empty())
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: name must be non-empty");
        if (cfg_.K_trunc < 1)
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: K_trunc must be >= 1");
        if (cfg_.d == 0)
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: d must be > 0");
        if (cfg_.N == 0)
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: N must be > 0");
        if (cfg_.mu_0.n_elem != cfg_.d)
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: mu_0 length must equal d");
        if (!(cfg_.kappa_0 > 0.0))
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: kappa_0 must be > 0");
        if (!(cfg_.nu_0 > static_cast<double>(cfg_.d) - 1.0))
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: nu_0 must be > d - 1");
        if (cfg_.Psi_0_flat.n_elem != cfg_.d * cfg_.d)
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: Psi_0_flat length must equal d * d");
        // Symmetry + positive-definiteness check on Psi_0.
        Psi_0_mat_.set_size(cfg_.d, cfg_.d);
        for (std::size_t i = 0; i < cfg_.d; ++i)
            for (std::size_t j = 0; j < cfg_.d; ++j)
                Psi_0_mat_(i, j) = cfg_.Psi_0_flat[i * cfg_.d + j];
        for (std::size_t i = 0; i < cfg_.d; ++i)
            for (std::size_t j = i + 1; j < cfg_.d; ++j)
                if (std::abs(Psi_0_mat_(i, j) - Psi_0_mat_(j, i)) > 1e-10)
                    throw std::invalid_argument(
                        "niw_cluster_gibbs_block: Psi_0 must be symmetric");
        arma::mat L_Psi_test;
        if (!arma::chol(L_Psi_test, Psi_0_mat_, "lower"))
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: Psi_0 must be positive definite");
        // Cache chol(Psi_0^{-1}) for empty-cluster prior draws and as
        // a building block for posterior draws.
        arma::mat Psi_0_inv;
        if (!arma::inv_sympd(Psi_0_inv, Psi_0_mat_))
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: Psi_0 not invertible (PSD-but-singular?)");
        if (!arma::chol(L_Psi0_inv_, Psi_0_inv, "lower"))
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: chol(Psi_0^{-1}) failed");

        const std::size_t flat_mu    = cfg_.K_trunc * cfg_.d;
        const std::size_t flat_sigma = cfg_.K_trunc * cfg_.d * cfg_.d;
        if (cfg_.initial_mu.n_elem != flat_mu)
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: initial_mu length must equal K * d");
        if (cfg_.initial_sigma.n_elem != flat_sigma)
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: initial_sigma length must equal K * d * d");
        if (cfg_.mu_name.empty() || cfg_.sigma_name.empty() ||
            cfg_.mu_name == cfg_.sigma_name)
            throw std::invalid_argument(
                "niw_cluster_gibbs_block: mu_name / sigma_name must be non-empty and differ");

        mu_     = cfg_.initial_mu;
        sigma_  = cfg_.initial_sigma;
        rebuild_current_();
    }

    // ---- block_sampler interface ---------------------------------------

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        const std::size_t K = cfg_.K_trunc;
        const std::size_t d = cfg_.d;
        const std::size_t N = cfg_.N;

        auto it_z = context_.find(cfg_.z_key);
        if (it_z == context_.end())
            throw std::runtime_error(
                "niw_cluster_gibbs_block '" + cfg_.name +
                "': z_key '" + cfg_.z_key + "' not in context");
        const arma::vec& z = it_z->second;
        if (z.n_elem != N)
            throw std::runtime_error(
                "niw_cluster_gibbs_block '" + cfg_.name +
                "': z length mismatch");
        auto it_y = context_.find(cfg_.y_key);
        if (it_y == context_.end())
            throw std::runtime_error(
                "niw_cluster_gibbs_block '" + cfg_.name +
                "': y_key '" + cfg_.y_key + "' not in context");
        const arma::vec& y_flat = it_y->second;
        if (y_flat.n_elem != N * d)
            throw std::runtime_error(
                "niw_cluster_gibbs_block '" + cfg_.name +
                "': y length mismatch");

        // Per-cluster sufficient statistics: n_k, sum_y_k (d-vec),
        // S_k = sum (y - bar_y)(y - bar_y)^T (d×d).
        // We accumulate sum_y and outer-product sum_y_yT, then derive
        // S_k = sum_y_yT - n_k * bar_y * bar_y^T.
        std::vector<std::size_t> n_k(K, 0);
        std::vector<arma::vec>   sum_y(K, arma::vec(d, arma::fill::zeros));
        std::vector<arma::mat>   sum_yyT(K, arma::mat(d, d, arma::fill::zeros));
        for (std::size_t i = 0; i < N; ++i) {
            const long lab = static_cast<long>(std::llround(z[i]));
            if (lab < 1 || static_cast<std::size_t>(lab) > K)
                throw std::runtime_error(
                    "niw_cluster_gibbs_block '" + cfg_.name +
                    "': z[" + std::to_string(i) + "] out of {1, ..., K}");
            const std::size_t k = static_cast<std::size_t>(lab) - 1;
            arma::vec y_i(d);
            for (std::size_t j = 0; j < d; ++j)
                y_i[j] = y_flat[i * d + j];
            n_k[k] += 1;
            sum_y[k]   += y_i;
            sum_yyT[k] += y_i * y_i.t();
        }

        std::normal_distribution<double> stdnorm(0.0, 1.0);

        for (std::size_t k = 0; k < K; ++k) {
            // Compute posterior NIW parameters.
            arma::vec mu_n(d);
            arma::mat Psi_n(d, d);
            double kappa_n, nu_n;
            const std::size_t nk = n_k[k];
            if (nk == 0) {
                // Empty: use prior parameters directly.
                kappa_n = cfg_.kappa_0;
                nu_n    = cfg_.nu_0;
                mu_n    = cfg_.mu_0;
                Psi_n   = Psi_0_mat_;
            } else {
                const arma::vec bar_y = sum_y[k] / static_cast<double>(nk);
                // S_k = sum y y^T - n_k * bar_y * bar_y^T (covariance-of-data
                // matrix, NOT divided by n).
                const arma::mat S_k = sum_yyT[k]
                    - static_cast<double>(nk) * (bar_y * bar_y.t());
                kappa_n = cfg_.kappa_0 + static_cast<double>(nk);
                nu_n    = cfg_.nu_0    + static_cast<double>(nk);
                mu_n    = (cfg_.kappa_0 * cfg_.mu_0 +
                           static_cast<double>(nk) * bar_y) / kappa_n;
                const arma::vec dev = bar_y - cfg_.mu_0;
                const double mult = cfg_.kappa_0 * static_cast<double>(nk) /
                                    kappa_n;
                Psi_n = Psi_0_mat_ + S_k + mult * (dev * dev.t());
                // Symmetrize (numerical hygiene).
                Psi_n = 0.5 * (Psi_n + Psi_n.t());
            }

            // Sample Sigma_k ~ IW(Psi_n, nu_n).
            //
            // Plan:
            //   1. Compute L_psi_inv = chol(Psi_n^{-1}, "lower").
            //   2. Build A: lower-triangular, A_ii = sqrt(chi^2(nu_n - i)),
            //      A_ij ~ N(0,1) for i > j.
            //   3. M = L_psi_inv * A   => Sigma^{-1} = M M^T => Sigma = (M^{-T})(M^{-1}).
            arma::mat Psi_n_inv;
            if (!arma::inv_sympd(Psi_n_inv, Psi_n))
                throw std::runtime_error(
                    "niw_cluster_gibbs_block '" + cfg_.name +
                    "': Psi_n not positive definite at cluster " +
                    std::to_string(k));
            arma::mat L_psi_inv;
            if (!arma::chol(L_psi_inv, Psi_n_inv, "lower"))
                throw std::runtime_error(
                    "niw_cluster_gibbs_block '" + cfg_.name +
                    "': chol(Psi_n^{-1}) failed at cluster " +
                    std::to_string(k));

            arma::mat A(d, d, arma::fill::zeros);
            for (std::size_t i = 0; i < d; ++i) {
                // Diagonal: A_ii ~ sqrt(chi^2(nu_n - i)).
                // Sample chi^2(df) via Gamma(df/2, scale=2).
                // E[chi^2(df)] = df  ✓
                const double df = nu_n - static_cast<double>(i);
                if (!(df > 0.0))
                    throw std::runtime_error(
                        "niw_cluster_gibbs_block '" + cfg_.name +
                        "': IW degrees of freedom too small "
                        "(nu_n - i = " + std::to_string(df) + ")");
                std::gamma_distribution<double> gam(df / 2.0, 2.0);
                const double chi2 = gam(rng);
                A(i, i) = std::sqrt(std::max(chi2, 1e-300));
                for (std::size_t j = 0; j < i; ++j) {
                    A(i, j) = stdnorm(rng);
                }
            }

            // M = L_psi_inv * A   (both lower-triangular)
            arma::mat M = L_psi_inv * A;
            // Sigma = (M M^T)^{-1}.  Compute via inv_sympd((M M^T)).
            arma::mat MMt = M * M.t();
            // Symmetrize for numerical hygiene.
            MMt = 0.5 * (MMt + MMt.t());
            arma::mat Sigma_k;
            if (!arma::inv_sympd(Sigma_k, MMt)) {
                throw std::runtime_error(
                    "niw_cluster_gibbs_block '" + cfg_.name +
                    "': IW draw produced non-PSD precision at cluster " +
                    std::to_string(k));
            }
            Sigma_k = 0.5 * (Sigma_k + Sigma_k.t());

            // Sample mu_k | Sigma_k ~ N(mu_n, Sigma_k / kappa_n).
            // SD-mat = chol(Sigma_k / kappa_n) = chol(Sigma_k) / sqrt(kappa_n).
            arma::mat L_sig;
            if (!arma::chol(L_sig, Sigma_k, "lower")) {
                Sigma_k.diag() += 1e-8;  // tiny jitter
                if (!arma::chol(L_sig, Sigma_k, "lower"))
                    throw std::runtime_error(
                        "niw_cluster_gibbs_block '" + cfg_.name +
                        "': chol(Sigma_k) failed at cluster " +
                        std::to_string(k));
            }
            arma::vec eps(d);
            for (std::size_t j = 0; j < d; ++j) eps[j] = stdnorm(rng);
            const double scale = 1.0 / std::sqrt(kappa_n);
            arma::vec mu_k = mu_n + scale * (L_sig * eps);

            // Write back into flat storage.
            for (std::size_t j = 0; j < d; ++j) {
                mu_[k * d + j] = mu_k[j];
            }
            for (std::size_t i = 0; i < d; ++i) {
                for (std::size_t j = 0; j < d; ++j) {
                    sigma_[k * d * d + i * d + j] = Sigma_k(i, j);
                }
            }
        }

        rebuild_current_();
        if (keep_history_) {
            mu_history_buf_.push_back(mu_);
            sigma_history_buf_.push_back(sigma_);
        }
    }

    const arma::vec& current() const override { return current_; }

    void set_current(const arma::vec& theta) override {
        const std::size_t flat_mu    = cfg_.K_trunc * cfg_.d;
        const std::size_t flat_sigma = cfg_.K_trunc * cfg_.d * cfg_.d;
        if (theta.n_elem != flat_mu + flat_sigma)
            throw std::invalid_argument(
                "niw_cluster_gibbs_block::set_current: expected length " +
                std::to_string(flat_mu + flat_sigma));
        for (std::size_t i = 0; i < flat_mu; ++i)    mu_[i]    = theta[i];
        for (std::size_t i = 0; i < flat_sigma; ++i) sigma_[i] = theta[flat_mu + i];
        rebuild_current_();
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override {
        return cfg_.K_trunc * cfg_.d * (1 + cfg_.d);
    }

    state_map current_named_outputs() const override {
        state_map out;
        out.emplace(cfg_.mu_name,    mu_);
        out.emplace(cfg_.sigma_name, sigma_);
        return out;
    }

    history_map get_history() const override {
        history_map out =
            detail::make_history_map(cfg_.mu_name, mu_history_buf_, mu_);
        history_map sig =
            detail::make_history_map(cfg_.sigma_name,
                                     sigma_history_buf_, sigma_);
        for (auto& kv : sig) out.emplace(kv.first, std::move(kv.second));
        return out;
    }

    std::size_t history_size() const noexcept override {
        return mu_history_buf_.empty() ? 1 : mu_history_buf_.size();
    }

    void clear_history() override {
        mu_history_buf_.clear();
        sigma_history_buf_.clear();
    }

private:
    void rebuild_current_() {
        const std::size_t flat_mu    = cfg_.K_trunc * cfg_.d;
        const std::size_t flat_sigma = cfg_.K_trunc * cfg_.d * cfg_.d;
        current_.set_size(flat_mu + flat_sigma);
        for (std::size_t i = 0; i < flat_mu; ++i)    current_[i]            = mu_[i];
        for (std::size_t i = 0; i < flat_sigma; ++i) current_[flat_mu + i]  = sigma_[i];
    }

    niw_cluster_gibbs_block_config cfg_;
    arma::mat                      Psi_0_mat_;
    arma::mat                      L_Psi0_inv_;  // chol(Psi_0^{-1}, lower)
    arma::vec                      mu_;
    arma::vec                      sigma_;
    arma::vec                      current_;
    block_context                  context_;
    std::vector<arma::vec>         mu_history_buf_;
    std::vector<arma::vec>         sigma_history_buf_;
};

}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_NIW_CLUSTER_GIBBS_BLOCK_HPP

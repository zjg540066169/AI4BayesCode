/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  full_rank_gaussian_vi_block.hpp  --  v1 opt-in full-rank Gaussian VI.
 *
 *  Maintains q(η) = N(η; μ, LL^T) on the UNCONSTRAINED scale, where L is
 *  a lower-triangular Cholesky factor with positive diagonal. The block
 *  captures the FULL posterior correlation matrix among the K
 *  coordinates — the standard remedy for the "mean-field underestimates
 *  variance when posterior is correlated" failure mode (Bishop §10.1.2
 *  caveat 1).
 *
 *  When to use vs mean_field_gaussian_vi_block:
 *    - mean-field is fine when the posterior factorizes (independent
 *      coordinates). Use the mean-field block — far cheaper, K instead
 *      of K(K+1)/2 var params for the Cholesky.
 *    - full-rank is needed when posterior coordinates are strongly
 *      correlated (regression with collinear predictors, BNN
 *      output-layer weights conditional on hidden representations,
 *      etc.). Mean-field returns a variance close to the CONDITIONAL
 *      variance, not the marginal; full-rank captures both.
 *    - For mixed cases (some coords correlated, some independent),
 *      put the correlated coords in a full_rank block and the
 *      independent ones in a separate mean-field block; let
 *      composite_block compose them. Each block sees only its own
 *      slice of parameters.
 *
 *  Parameter packing for avgAdam:
 *    λ = (μ ∈ R^K, log L_diag ∈ R^K, L_offdiag ∈ R^{K(K-1)/2})
 *    Total dim D = K(K+3)/2.
 *
 *  Reparameterization gradient (Kucukelbir 2017 §4):
 *    η = μ + L ε,  ε ~ N(0, I)
 *    log q(η) = -K/2 log(2π) - sum_i log L_ii - 0.5 ε^T ε   (uses log|L L^T| = 2 sum log L_ii)
 *
 *    ∂(-ELBO)/∂μ_k         = -E_ε[ ∂log p̃/∂η_k ]
 *    ∂(-ELBO)/∂L_ij (i>j)  = -E_ε[ (∂log p̃/∂η_i) ε_j ]
 *    ∂(-ELBO)/∂(log L_jj)  = -E_ε[ (∂log p̃/∂η_j) ε_j · L_jj ] - 1   (entropy term)
 *
 *  Caps (suggested):
 *    K ≤ 50    auto-suggest
 *    K ≤ 200   warn
 *    K > 500   reject (consider mean-field over the bulk + full-rank
 *              over a small subset)
 *
 *  See system_design.md §18.5 for the architectural backing.
 *================================================================================*/

#ifndef AI4BAYESCODE_FULL_RANK_GAUSSIAN_VI_BLOCK_HPP
#define AI4BAYESCODE_FULL_RANK_GAUSSIAN_VI_BLOCK_HPP

#include "vi_block.hpp"
#include "vi_optimizer.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MCMC_USE_RCPP_ARMADILLO
# include <armadillo>
#else
# include <RcppArmadillo.h>
#endif

namespace AI4BayesCode {

/// Same signature as mean_field_gaussian_vi_block / nuts_block.
using fr_vi_log_density_gradient_fn =
    std::function<double(const arma::vec& eta_unc,
                         const block_context& ctx,
                         arma::vec* grad_out)>;
using fr_vi_transform_fn = std::function<arma::vec(const arma::vec&)>;

struct full_rank_gaussian_vi_block_config {
    std::string name;
    arma::vec   initial_unc;        // K-dim
    /// Initial Cholesky factor L (lower triangular, K x K). If empty,
    /// defaults to 0.1 * I.
    arma::mat   initial_L;
    fr_vi_log_density_gradient_fn log_density_grad;
    fr_vi_transform_fn constrain;
    fr_vi_transform_fn unconstrain;
    std::vector<std::string> dependencies;
    vi_optimizer::raabbvi_config optimizer;
    std::size_t n_mc_per_step = 1;
};

/**
 * @brief Full-rank Gaussian variational block — q(η) = N(μ, LL^T).
 *
 * State invariants:
 *   - mu_unc_         : variational mean (K)
 *   - L_              : lower-triangular Cholesky factor (K x K, upper = 0)
 *   - opt_state_      : avgAdam state, dim D = K(K+3)/2 packing (μ, log_diag, off-diag)
 *   - epoch_lambda_avg_history_ : λ̄_k snapshots, for SKL termination
 *
 * The block writes a q-sample to shared_data after each composite step;
 * the wrapper's R-side get_current returns the q-mean (constrain(μ)).
 */
class full_rank_gaussian_vi_block : public vi_block {
public:
    explicit full_rank_gaussian_vi_block(
        full_rank_gaussian_vi_block_config cfg)
        : cfg_(std::move(cfg)) {

        if (cfg_.name.empty())
            throw std::invalid_argument("full_rank_gaussian_vi_block: cfg.name required");
        if (cfg_.initial_unc.is_empty())
            throw std::invalid_argument("full_rank_gaussian_vi_block: cfg.initial_unc required");
        if (!cfg_.log_density_grad)
            throw std::invalid_argument("full_rank_gaussian_vi_block: cfg.log_density_grad required");

        K_ = cfg_.initial_unc.n_elem;
        D_ = K_ * (K_ + 3) / 2;     // μ + log_diag + off-diag

        if (K_ > 500)
            throw std::invalid_argument(
                "full_rank_gaussian_vi_block: K > 500 is impractical; "
                "consider splitting parameters into mean-field + small "
                "full-rank subset");

        mu_unc_ = cfg_.initial_unc;

        if (cfg_.initial_L.is_empty()) {
            L_ = 0.1 * arma::eye(K_, K_);
        } else {
            if (cfg_.initial_L.n_rows != K_ || cfg_.initial_L.n_cols != K_)
                throw std::invalid_argument(
                    "full_rank_gaussian_vi_block: cfg.initial_L must be K x K");
            L_ = arma::trimatl(cfg_.initial_L);   // enforce lower-triangular
            for (std::size_t j = 0; j < K_; ++j)
                if (L_(j, j) <= 0.0)
                    throw std::invalid_argument(
                        "full_rank_gaussian_vi_block: initial_L must have positive diagonal");
        }

        if (!cfg_.constrain)   cfg_.constrain   = [](const arma::vec& v){ return v; };
        if (!cfg_.unconstrain) cfg_.unconstrain = [](const arma::vec& v){ return v; };

        epoch_lambda_avg_history_.push_back(pack_lambda(mu_unc_, L_));
        opt_state_.reset_epoch(D_);
        gamma_current_  = cfg_.optimizer.gamma_0;
        epoch_index_    = 0;
        steps_in_epoch_ = 0;
        converged_      = false;
        last_elbo_      = std::numeric_limits<double>::quiet_NaN();
        refresh_cur_natural_();
    }

    // ---- block_sampler interface ----------------------------------------

    void set_context(const block_context& ctx) override {
        ctx_.clear();
        for (const auto& key : cfg_.dependencies) {
            auto it = ctx.find(key);
            if (it != ctx.end()) ctx_.emplace(key, it->second);
        }
    }

    void step(std::mt19937_64& rng) override {
        if (converged_) return;
        arma::vec g_lambda;
        const double elbo = compute_elbo_and_grad_(rng, &g_lambda);
        last_elbo_ = elbo;

        arma::vec lambda = pack_lambda(mu_unc_, L_);
        vi_optimizer::avgAdam_step(lambda, opt_state_, g_lambda,
                                    gamma_current_, cfg_.optimizer);
        unpack_lambda(lambda, mu_unc_, L_);
        refresh_cur_natural_();

        steps_in_epoch_++;

        if (keep_history()) {
            history_.elbo.push_back(elbo);
            history_.mu.push_back(mu_unc_);
            history_.log_sd.push_back(get_log_sd());   // packed packing of L
            history_.gamma.push_back(gamma_current_);
            history_.epoch.push_back(static_cast<int>(epoch_index_));
        }

        if (steps_in_epoch_ >= cfg_.optimizer.inner_iter_per_epoch) {
            close_epoch_(rng);
        }
    }

    const arma::vec& current() const override { return cur_natural_; }

    void set_current(const arma::vec& theta_natural) override {
        if (theta_natural.n_elem != K_)
            throw std::invalid_argument(
                "full_rank_gaussian_vi_block::set_current: dim mismatch");
        mu_unc_ = cfg_.unconstrain(theta_natural);
        refresh_cur_natural_();
    }

    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim() const noexcept override { return K_; }

    history_map get_history() const override {
        history_map out;
        const std::size_t n = history_.elbo.size();
        if (n == 0) {
            arma::mat m(1, K_);
            for (std::size_t j = 0; j < K_; ++j) m(0, j) = cur_natural_[j];
            out.emplace(cfg_.name, std::move(m));
            return out;
        }
        // Pack: rows = step; cols = elbo, gamma, epoch, mu (K), log_sd-flat,
        //       final_khat. log_sd is K + K(K-1)/2 long (log_diag + off-diag),
        //       different from mean-field's K.
        const std::size_t lsd_dim = history_.log_sd[0].n_elem;
        const std::size_t cols = 3 + K_ + lsd_dim + 1;
        arma::mat m(n, cols);
        for (std::size_t i = 0; i < n; ++i) {
            m(i, 0) = history_.elbo[i];
            m(i, 1) = history_.gamma[i];
            m(i, 2) = static_cast<double>(history_.epoch[i]);
            for (std::size_t j = 0; j < K_; ++j)
                m(i, 3 + j) = history_.mu[i][j];
            for (std::size_t j = 0; j < lsd_dim; ++j)
                m(i, 3 + K_ + j) = history_.log_sd[i][j];
            m(i, cols - 1) = history_.final_khat;
        }
        out.emplace(cfg_.name + "__vi_history", std::move(m));
        return out;
    }

    std::size_t history_size() const noexcept override {
        return history_.elbo.size();
    }
    void clear_history() override { history_ = vi_history_t{}; }

    // ---- vi_block interface ---------------------------------------------

    arma::vec current_sample(std::mt19937_64& rng) const override {
        std::normal_distribution<double> N01(0.0, 1.0);
        arma::vec eps(K_);
        for (std::size_t i = 0; i < K_; ++i) eps[i] = N01(rng);
        const arma::vec eta = mu_unc_ + L_ * eps;
        return cfg_.constrain(eta);
    }

    /// Returns log_diag (K) followed by off-diag entries (K(K-1)/2), row-major
    /// (i.e., L_{1,0}, L_{2,0}, L_{2,1}, L_{3,0}, ..., L_{K-1, K-2}).
    arma::vec get_log_sd() const override {
        arma::vec out(K_ + K_ * (K_ - 1) / 2);
        for (std::size_t j = 0; j < K_; ++j) out[j] = std::log(L_(j, j));
        std::size_t idx = K_;
        for (std::size_t i = 1; i < K_; ++i)
            for (std::size_t j = 0; j < i; ++j)
                out[idx++] = L_(i, j);
        return out;
    }

    void set_variational_state(const arma::vec& mu,
                               const arma::vec& log_sd) override {
        const std::size_t expected = K_ + K_ * (K_ - 1) / 2;
        if (mu.n_elem != K_ || log_sd.n_elem != expected)
            throw std::invalid_argument(
                "full_rank_gaussian_vi_block::set_variational_state: dim mismatch");
        mu_unc_ = mu;
        L_.zeros();
        for (std::size_t j = 0; j < K_; ++j) L_(j, j) = std::exp(log_sd[j]);
        std::size_t idx = K_;
        for (std::size_t i = 1; i < K_; ++i)
            for (std::size_t j = 0; j < i; ++j) L_(i, j) = log_sd[idx++];
        refresh_cur_natural_();
    }

    double current_elbo() const override { return last_elbo_; }
    const vi_history_t& vi_history() const override { return history_; }

    // ---- Diagnostics ----------------------------------------------------
    bool converged() const noexcept { return converged_; }
    std::size_t epoch() const noexcept { return epoch_index_; }
    double gamma() const noexcept { return gamma_current_; }
    const arma::mat& L() const noexcept { return L_; }

private:
    // ---- Helpers --------------------------------------------------------

    /// Pack: λ_flat = (μ (K), log L_diag (K), L_offdiag (K(K-1)/2)).
    arma::vec pack_lambda(const arma::vec& mu, const arma::mat& L) const {
        arma::vec out(D_);
        for (std::size_t j = 0; j < K_; ++j) out[j] = mu[j];
        for (std::size_t j = 0; j < K_; ++j) out[K_ + j] = std::log(L(j, j));
        std::size_t idx = 2 * K_;
        for (std::size_t i = 1; i < K_; ++i)
            for (std::size_t j = 0; j < i; ++j) out[idx++] = L(i, j);
        return out;
    }

    void unpack_lambda(const arma::vec& lambda, arma::vec& mu, arma::mat& L) const {
        for (std::size_t j = 0; j < K_; ++j) mu[j] = lambda[j];
        L.zeros();
        for (std::size_t j = 0; j < K_; ++j) L(j, j) = std::exp(lambda[K_ + j]);
        std::size_t idx = 2 * K_;
        for (std::size_t i = 1; i < K_; ++i)
            for (std::size_t j = 0; j < i; ++j) L(i, j) = lambda[idx++];
    }

    void refresh_cur_natural_() { cur_natural_ = cfg_.constrain(mu_unc_); }

    /// Per-MC-sample reparam ELBO and gradient of -ELBO.
    /// Returns ELBO estimate; fills *g_lambda (size D) with -∇ELBO.
    ///
    /// ε ~ N(0, I_K),  η = μ + L ε
    /// log q(η) = -K/2 log 2π - sum_i log L_ii - 0.5 ε^T ε
    /// ELBO per sample = log p̃(η) - log q(η)
    ///
    /// ∂(-ELBO)/∂μ_k    = -∂log p̃/∂η_k
    /// ∂(-ELBO)/∂L_ij (i>j) = -(∂log p̃/∂η_i) ε_j
    /// ∂(-ELBO)/∂(log L_jj) = -(∂log p̃/∂η_j) ε_j L_jj - 1
    double compute_elbo_and_grad_(std::mt19937_64& rng,
                                   arma::vec* g_lambda) const {
        const std::size_t M = cfg_.n_mc_per_step;
        std::normal_distribution<double> N01(0.0, 1.0);
        const double log_2pi = std::log(2.0 * M_PI);

        arma::vec g_mu_sum(K_, arma::fill::zeros);
        arma::vec g_logdiag_sum(K_, arma::fill::zeros);
        std::vector<double> g_offdiag_sum(K_ * (K_ - 1) / 2, 0.0);
        double elbo_sum = 0.0;

        for (std::size_t m = 0; m < M; ++m) {
            arma::vec eps(K_);
            for (std::size_t i = 0; i < K_; ++i) eps[i] = N01(rng);
            const arma::vec eta = mu_unc_ + L_ * eps;

            arma::vec grad_lp(K_, arma::fill::zeros);
            const double log_p = cfg_.log_density_grad(eta, ctx_, &grad_lp);

            double sum_log_diag = 0.0;
            for (std::size_t j = 0; j < K_; ++j) sum_log_diag += std::log(L_(j, j));
            const double log_q =
                -0.5 * static_cast<double>(K_) * log_2pi
                - sum_log_diag
                - 0.5 * arma::sum(arma::square(eps));
            elbo_sum += (log_p - log_q);

            // ∂(-ELBO)/∂μ = -∇log p̃
            g_mu_sum -= grad_lp;

            // ∂(-ELBO)/∂(log L_jj) = -(∇log p̃)_j · ε_j · L_jj - 1
            for (std::size_t j = 0; j < K_; ++j)
                g_logdiag_sum[j] += -grad_lp[j] * eps[j] * L_(j, j) - 1.0;

            // ∂(-ELBO)/∂L_ij (i > j) = -(∇log p̃)_i · ε_j
            std::size_t idx = 0;
            for (std::size_t i = 1; i < K_; ++i)
                for (std::size_t j = 0; j < i; ++j)
                    g_offdiag_sum[idx++] += -grad_lp[i] * eps[j];
        }

        const double inv_M = 1.0 / static_cast<double>(M);
        g_mu_sum *= inv_M;
        g_logdiag_sum *= inv_M;
        for (auto& v : g_offdiag_sum) v *= inv_M;
        const double elbo_mean = elbo_sum * inv_M;

        if (g_lambda) {
            g_lambda->set_size(D_);
            for (std::size_t j = 0; j < K_; ++j) (*g_lambda)[j] = g_mu_sum[j];
            for (std::size_t j = 0; j < K_; ++j) (*g_lambda)[K_ + j] = g_logdiag_sum[j];
            for (std::size_t i = 0; i < g_offdiag_sum.size(); ++i)
                (*g_lambda)[2 * K_ + i] = g_offdiag_sum[i];
        }
        return elbo_mean;
    }

    void close_epoch_(std::mt19937_64& rng) {
        arma::vec mu_bar(K_); arma::mat L_bar(K_, K_);
        unpack_lambda(opt_state_.lambda_bar, mu_bar, L_bar);
        epoch_lambda_avg_history_.push_back(pack_lambda(mu_bar, L_bar));
        const std::size_t H = epoch_lambda_avg_history_.size();

        if (H >= 3) {
            // For full-rank, SKL between Gaussians is more involved (needs
            // |Σ| and Σ^{-1}). Use a fallback metric: ||λ̄_k - λ̄_{k-1}|| /
            // ||λ̄_k - λ̄_0||. Works as an approximate SKL ratio for our
            // termination purposes.
            const arma::vec& cur  = epoch_lambda_avg_history_[H - 1];
            const arma::vec& prev = epoch_lambda_avg_history_[H - 2];
            const arma::vec& init = epoch_lambda_avg_history_[0];
            const double r_consec  = arma::norm(cur - prev);
            const double r_initial = arma::norm(cur - init);
            if (r_initial > 0 && (r_consec / r_initial) < cfg_.optimizer.tau) {
                terminate_(mu_bar, L_bar, rng);
                return;
            }
        }
        if (epoch_index_ + 1 >= cfg_.optimizer.max_epochs) {
            terminate_(mu_bar, L_bar, rng);
            return;
        }
        mu_unc_ = mu_bar;
        L_      = L_bar;
        refresh_cur_natural_();
        gamma_current_ *= cfg_.optimizer.rho;
        ++epoch_index_;
        steps_in_epoch_ = 0;
        opt_state_.reset_epoch(D_);
    }

    void terminate_(const arma::vec& mu_final, const arma::mat& L_final,
                    std::mt19937_64& rng) {
        mu_unc_ = mu_final;
        L_      = L_final;
        refresh_cur_natural_();
        converged_ = true;

        // Compute joint PSIS-k̂ over S samples from converged q.
        const std::size_t S = cfg_.optimizer.S_khat;
        const double log_2pi = std::log(2.0 * M_PI);
        std::normal_distribution<double> N01(0.0, 1.0);

        double sum_log_diag = 0.0;
        for (std::size_t j = 0; j < K_; ++j) sum_log_diag += std::log(L_(j, j));

        arma::vec log_weights(S);
        for (std::size_t s = 0; s < S; ++s) {
            arma::vec eps(K_);
            for (std::size_t i = 0; i < K_; ++i) eps[i] = N01(rng);
            const arma::vec eta = mu_unc_ + L_ * eps;
            arma::vec grad_lp(K_, arma::fill::zeros);
            const double log_p = cfg_.log_density_grad(eta, ctx_, &grad_lp);
            const double log_q =
                -0.5 * static_cast<double>(K_) * log_2pi
                - sum_log_diag
                - 0.5 * arma::sum(arma::square(eps));
            log_weights[s] = log_p - log_q;
        }
        history_.final_khat = vi_optimizer::psis_khat(log_weights);
    }

    // ---- State ----------------------------------------------------------
    full_rank_gaussian_vi_block_config cfg_;
    std::size_t K_ = 0, D_ = 0;
    arma::vec mu_unc_;
    arma::mat L_;
    arma::vec cur_natural_;
    block_context ctx_;
    vi_optimizer::avgAdam_state opt_state_;
    std::vector<arma::vec> epoch_lambda_avg_history_;
    double      gamma_current_   = 0.1;
    std::size_t epoch_index_     = 0;
    std::size_t steps_in_epoch_  = 0;
    bool        converged_       = false;
    double      last_elbo_       = std::numeric_limits<double>::quiet_NaN();
    vi_history_t history_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_FULL_RANK_GAUSSIAN_VI_BLOCK_HPP

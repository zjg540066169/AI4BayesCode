/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  mean_field_gaussian_vi_block.hpp  --  primary v1 VI block.
 *
 *  Maintains q(η) = ∏_i N(η_i; μ_i, σ_i²) on the UNCONSTRAINED scale
 *  η = constraints::unconstrain(θ). Optimizes -ELBO via the RAABBVI-
 *  lite procedure (avgAdam + Polyak-Ruppert iterate averaging +
 *  geometric γ decay + SKL-only termination) from
 *  `vi_optimizer.hpp`. The user's natural-scale log p̃ + ∇_η log p̃
 *  is provided via the same log_density_gradient_fn signature that
 *  `nuts_block` accepts — zero new infrastructure on the user side.
 *
 *  See system_design.md §18 for the full architectural backing.
 *
 *  USAGE
 *  -----
 *  ```cpp
 *  mean_field_gaussian_vi_block_config cfg;
 *  cfg.name           = "beta";
 *  cfg.initial_unc    = arma::zeros(K);
 *  cfg.initial_log_sd = arma::vec(K, arma::fill::value(-2.0));
 *  cfg.constrain      = constraints::real::constrain;       // identity for real
 *  cfg.unconstrain    = constraints::real::unconstrain;
 *  cfg.log_density_grad = [&](const arma::vec& eta,
 *                              const block_context& ctx,
 *                              arma::vec* grad) {
 *      // natural-scale log p̃ at eta (here = log p since real-scale)
 *      // fill *grad with ∇_eta log p̃
 *      ...
 *  };
 *  cfg.dependencies = {"y", "X"};
 *  auto blk = std::make_shared<mean_field_gaussian_vi_block>(cfg);
 *  ```
 *================================================================================*/

#ifndef AI4BAYESCODE_MEAN_FIELD_GAUSSIAN_VI_BLOCK_HPP
#define AI4BAYESCODE_MEAN_FIELD_GAUSSIAN_VI_BLOCK_HPP

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

/// log_density_gradient_fn alias matching nuts_block's signature so the
/// codegen agent can reuse the same wrapped lambda. Returns log p̃ on
/// the UNCONSTRAINED scale (i.e., includes the constraint Jacobian);
/// fills grad_out with the unconstrained-scale gradient.
using vi_log_density_gradient_fn =
    std::function<double(const arma::vec& eta_unc,
                         const block_context& ctx,
                         arma::vec* grad_out)>;

/// Optional transform between unconstrained and natural scales.
/// Identity used when the parameter is already unconstrained.
using vi_transform_fn = std::function<arma::vec(const arma::vec&)>;

/**
 * @brief Configuration bundle for mean_field_gaussian_vi_block.
 *
 * Symmetric with nuts_block_config wherever possible to ease codegen
 * reuse. Optimizer hyperparams live under `optimizer` (POD struct).
 */
struct mean_field_gaussian_vi_block_config {
    /// Unique name for this block within its composite. Becomes the
    /// shared_data_t key that holds the per-step q-sample.
    std::string name;

    /// Initial variational mean μ on the UNCONSTRAINED scale.
    /// Non-empty; dim() is inferred from this.
    arma::vec initial_unc;

    /// Initial variational log σ (length K, same as initial_unc).
    /// If left empty, defaults to a vector of -2.0 (σ ≈ 0.135) —
    /// a weakly informative starting scale appropriate for most
    /// standardised problems.
    arma::vec initial_log_sd;

    /// log-density + gradient oracle, on UNCONSTRAINED scale (must
    /// include the constraint Jacobian — codegen agent emits this
    /// via the same constraints::*::wrap that nuts_block uses).
    vi_log_density_gradient_fn log_density_grad;

    /// Map unconstrained → natural. Default identity if empty.
    vi_transform_fn constrain;

    /// Map natural → unconstrained. Needed only if set_current with
    /// a natural-scale vector will be called.
    vi_transform_fn unconstrain;

    /// shared_data keys this block reads from context during step().
    /// Mirrors nuts_block_config::dependencies — the composite
    /// projects only these into block_context.
    std::vector<std::string> dependencies;

    /// Optimizer hyperparams (γ_0, ρ, τ, inner_iter_per_epoch, ...).
    /// Defaults are Welandawe 2022 §5; see vi_optimizer.hpp.
    vi_optimizer::raabbvi_config optimizer;

    /// Number of Monte Carlo samples per ELBO/grad estimate. Default 1
    /// (single-sample reparam estimator, ADVI-style). Larger reduces
    /// gradient variance at proportional cost.
    std::size_t n_mc_per_step = 1;
};

/**
 * @brief Primary v1 VI block: mean-field Gaussian on R^K.
 *
 * State invariants:
 *   - mu_unc_       : current variational mean (length K, unconstrained)
 *   - log_sd_       : current variational log σ (length K)
 *   - epoch_index_  : 0-based γ-epoch index; γ = γ_0 · ρ^epoch_index_
 *   - epoch_lambda_avg_history_ : λ̄_k for SKL termination (one entry
 *                                  per completed epoch, plus λ̄_0 = initial
 *                                  for the denominator)
 *   - opt_state_    : avgAdam state for the current epoch
 *   - converged_    : flag set on SKL termination; once true, step() is
 *                     a no-op (but current_sample still draws fresh
 *                     samples from the frozen q)
 *   - last_elbo_    : ELBO at the most recent step() (NaN before first)
 *   - history_      : vi_history_t (populated when keep_history=true)
 *   - cur_natural_  : cached q-mean on the natural scale, updated lazily
 *                     by current() for reference returns.
 */
class mean_field_gaussian_vi_block : public vi_block {
public:
    explicit mean_field_gaussian_vi_block(
        mean_field_gaussian_vi_block_config cfg)
        : cfg_(std::move(cfg)) {

        // Validate cfg.
        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "mean_field_gaussian_vi_block: cfg.name required");
        }
        if (cfg_.initial_unc.is_empty()) {
            throw std::invalid_argument(
                "mean_field_gaussian_vi_block: cfg.initial_unc required");
        }
        if (!cfg_.log_density_grad) {
            throw std::invalid_argument(
                "mean_field_gaussian_vi_block: cfg.log_density_grad required");
        }

        K_ = cfg_.initial_unc.n_elem;
        D_ = 2 * K_;  // (μ, log σ) packed into one λ-vector for avgAdam

        mu_unc_ = cfg_.initial_unc;

        if (cfg_.initial_log_sd.is_empty()) {
            log_sd_ = arma::vec(K_, arma::fill::value(-2.0));
        } else {
            if (cfg_.initial_log_sd.n_elem != K_) {
                throw std::invalid_argument(
                    "mean_field_gaussian_vi_block: cfg.initial_log_sd "
                    "must have same length as cfg.initial_unc");
            }
            log_sd_ = cfg_.initial_log_sd;
        }

        // Default constrain / unconstrain to identity.
        if (!cfg_.constrain)   cfg_.constrain   = [](const arma::vec& v) { return v; };
        if (!cfg_.unconstrain) cfg_.unconstrain = [](const arma::vec& v) { return v; };

        // Initial λ̄_0 = current state (used as SKL termination denominator).
        epoch_lambda_avg_history_.push_back(pack_lambda(mu_unc_, log_sd_));

        // avgAdam state for epoch 0.
        opt_state_.reset_epoch(D_);

        gamma_current_ = cfg_.optimizer.gamma_0;
        epoch_index_   = 0;
        steps_in_epoch_ = 0;
        converged_     = false;
        last_elbo_     = std::numeric_limits<double>::quiet_NaN();

        refresh_cur_natural_();
    }

    // ---- block_sampler interface ----------------------------------------

    void set_context(const block_context& ctx) override {
        // Copy only the keys we depend on (mirrors nuts_block).
        ctx_.clear();
        for (const auto& key : cfg_.dependencies) {
            auto it = ctx.find(key);
            if (it != ctx.end()) {
                ctx_.emplace(key, it->second);
            }
        }
    }

    void step(std::mt19937_64& rng) override {
        if (converged_) {
            // Post-convergence: no-op. current_sample still draws fresh
            // samples from the frozen q.
            return;
        }

        // ----- One avgAdam update step -----
        arma::vec g_lambda;
        const double elbo_sample = compute_elbo_and_grad_(rng, &g_lambda);
        last_elbo_ = elbo_sample;

        arma::vec lambda = pack_lambda(mu_unc_, log_sd_);
        vi_optimizer::avgAdam_step(lambda, opt_state_, g_lambda,
                                   gamma_current_, cfg_.optimizer);
        unpack_lambda(lambda, mu_unc_, log_sd_);
        refresh_cur_natural_();

        steps_in_epoch_ += 1;

        if (keep_history()) {
            history_.elbo.push_back(elbo_sample);
            history_.mu.push_back(mu_unc_);
            history_.log_sd.push_back(log_sd_);
            history_.gamma.push_back(gamma_current_);
            history_.epoch.push_back(static_cast<int>(epoch_index_));
        }

        // ----- Epoch boundary check -----
        if (steps_in_epoch_ >= cfg_.optimizer.inner_iter_per_epoch) {
            close_epoch_and_maybe_terminate_(rng);
        }
    }

    const arma::vec& current() const override {
        return cur_natural_;
    }

    void set_current(const arma::vec& theta_natural) override {
        if (theta_natural.n_elem != K_) {
            throw std::invalid_argument(
                "mean_field_gaussian_vi_block::set_current: "
                "size mismatch (expected K)");
        }
        // theta is on the natural scale → unconstrain → mu_unc_
        mu_unc_ = cfg_.unconstrain(theta_natural);
        refresh_cur_natural_();
        // log_sd left as-is per the §18.3 contract.
    }

    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim()   const noexcept override { return K_; }

    history_map get_history() const override {
        // Tier A wrapper's get_history() builds a List per child; for a VI
        // child we expose (μ-trajectory + ELBO + log_sd) in a single
        // history_map by packing them as columns of an (n_steps × ...)
        // matrix. The R-side helper in AI4BayesCode_helpers.R splits this
        // into the documented list(elbo, mu, log_sd, gamma, epoch,
        // final_khat).
        //
        // For a leaf VI block, key it under "<name>__vi_history" to
        // distinguish from MCMC-style history maps; the wrapper merges.
        history_map out;
        const std::size_t n = history_.elbo.size();
        if (n == 0) {
            // Pre-step or keep_history=false: return current q-mean as the
            // fallback row, matching the block_sampler default.
            arma::mat m(1, K_);
            for (std::size_t j = 0; j < K_; ++j) m(0, j) = cur_natural_[j];
            out.emplace(cfg_.name, std::move(m));
            return out;
        }
        // Pack: row = step, columns = [elbo, gamma, epoch, mu (K), log_sd (K), final_khat]
        const std::size_t cols = 3 + 2 * K_ + 1;
        arma::mat m(n, cols);
        for (std::size_t i = 0; i < n; ++i) {
            m(i, 0) = history_.elbo[i];
            m(i, 1) = history_.gamma[i];
            m(i, 2) = static_cast<double>(history_.epoch[i]);
            for (std::size_t j = 0; j < K_; ++j) {
                m(i, 3 + j)       = history_.mu[i][j];
                m(i, 3 + K_ + j)  = history_.log_sd[i][j];
            }
            m(i, cols - 1) = history_.final_khat;  // same value all rows
        }
        out.emplace(cfg_.name + "__vi_history", std::move(m));
        return out;
    }

    std::size_t history_size() const noexcept override {
        return history_.elbo.size();
    }

    void clear_history() override {
        history_ = vi_history_t{};
    }

    // ---- vi_block interface ---------------------------------------------

    arma::vec current_sample(std::mt19937_64& rng) const override {
        // Draw eps ~ N(0, I), eta = mu + sigma * eps, theta = constrain(eta).
        std::normal_distribution<double> N01(0.0, 1.0);
        arma::vec eta(K_);
        const arma::vec sigma = arma::exp(log_sd_);
        for (std::size_t i = 0; i < K_; ++i) {
            eta[i] = mu_unc_[i] + sigma[i] * N01(rng);
        }
        return cfg_.constrain(eta);
    }

    arma::vec get_log_sd() const override { return log_sd_; }

    void set_variational_state(const arma::vec& mu,
                               const arma::vec& log_sd) override {
        if (mu.n_elem != K_ || log_sd.n_elem != K_) {
            throw std::invalid_argument(
                "mean_field_gaussian_vi_block::set_variational_state: "
                "size mismatch");
        }
        mu_unc_ = mu;
        log_sd_ = log_sd;
        refresh_cur_natural_();
    }

    double current_elbo() const override { return last_elbo_; }

    const vi_history_t& vi_history() const override { return history_; }

    // ---- Public diagnostics ---------------------------------------------

    bool converged() const noexcept { return converged_; }
    std::size_t epoch() const noexcept { return epoch_index_; }
    double gamma() const noexcept { return gamma_current_; }

private:
    // ---- Helpers --------------------------------------------------------

    /// Pack (μ, log σ) into a single D=2K λ vector for avgAdam.
    arma::vec pack_lambda(const arma::vec& mu, const arma::vec& lsd) const {
        arma::vec out(D_);
        for (std::size_t i = 0; i < K_; ++i) {
            out[i]       = mu[i];
            out[K_ + i]  = lsd[i];
        }
        return out;
    }

    /// Inverse of pack_lambda.
    void unpack_lambda(const arma::vec& lambda,
                       arma::vec& mu, arma::vec& lsd) const {
        for (std::size_t i = 0; i < K_; ++i) {
            mu[i]  = lambda[i];
            lsd[i] = lambda[K_ + i];
        }
    }

    /// Refresh the cached natural-scale q-mean (= constrain(mu_unc_)).
    /// E_q[θ] is not exactly constrain(μ) when the transform is
    /// nonlinear; for the canonical use cases (real, positive via log,
    /// interval via logit) constrain(μ) is a usable point estimate and
    /// matches the standard ADVI "point estimate" convention. Users
    /// who want the exact E_q[θ] should average over current_sample.
    void refresh_cur_natural_() {
        cur_natural_ = cfg_.constrain(mu_unc_);
    }

    /// Compute one Monte Carlo estimate of -ELBO and its gradient with
    /// respect to (μ, log σ) via the reparameterization trick.
    ///
    /// Reparam: η = μ + σ ⊙ ε, ε ~ N(0, I).
    /// Per-sample gradient (n_mc_per_step = 1 default):
    ///   g_μ_i   = -∂log p̃/∂η_i
    ///   g_ω_i   = -σ_i · ε_i · ∂log p̃/∂η_i - 1   (entropy term)
    /// Per-sample ELBO estimate:
    ///   ELBO = log p̃(η) - log q(η; μ, σ)
    ///        = log p̃(η) - [-K/2·log 2π - sum log σ - 0.5 sum ε²]
    ///        = log p̃(η) + K/2·log 2π + sum log σ + 0.5 sum ε²
    ///
    /// If n_mc_per_step > 1, average over multiple ε samples.
    ///
    /// @param rng       random source for ε
    /// @param g_lambda  output: packed gradient (length D=2K) of -ELBO
    /// @return ELBO estimate (NOT -ELBO; ELBO is what we maximize)
    double compute_elbo_and_grad_(std::mt19937_64& rng,
                                   arma::vec* g_lambda) const {
        const std::size_t M = cfg_.n_mc_per_step;
        std::normal_distribution<double> N01(0.0, 1.0);

        const arma::vec sigma = arma::exp(log_sd_);
        const double log_2pi = std::log(2.0 * M_PI);

        arma::vec g_mu_sum(K_, arma::fill::zeros);
        arma::vec g_om_sum(K_, arma::fill::zeros);
        double elbo_sum = 0.0;

        for (std::size_t m = 0; m < M; ++m) {
            arma::vec eps(K_);
            for (std::size_t i = 0; i < K_; ++i) eps[i] = N01(rng);
            const arma::vec eta = mu_unc_ + sigma % eps;

            arma::vec grad_lp(K_, arma::fill::zeros);
            const double log_p = cfg_.log_density_grad(eta, ctx_, &grad_lp);

            // log q(η) = -K/2 log 2π - sum log σ - 0.5 sum ε²
            const double log_q =
                -0.5 * static_cast<double>(K_) * log_2pi
                - arma::sum(log_sd_)
                - 0.5 * arma::sum(arma::square(eps));

            elbo_sum += (log_p - log_q);

            // Per-sample reparam gradient of -ELBO:
            //   g_μ = -∇_η log p̃
            //   g_ω = -σ ⊙ ε ⊙ ∇_η log p̃ - 1
            g_mu_sum -= grad_lp;
            g_om_sum -= (sigma % eps) % grad_lp;
            g_om_sum -= arma::ones(K_);
        }

        const double inv_M = 1.0 / static_cast<double>(M);
        g_mu_sum *= inv_M;
        g_om_sum *= inv_M;
        const double elbo_mean = elbo_sum * inv_M;

        if (g_lambda != nullptr) {
            g_lambda->set_size(D_);
            for (std::size_t i = 0; i < K_; ++i) {
                (*g_lambda)[i]       = g_mu_sum[i];
                (*g_lambda)[K_ + i]  = g_om_sum[i];
            }
        }
        return elbo_mean;
    }

    /// Called at the end of each γ-epoch: caches λ̄ for the SKL ratio,
    /// checks termination, shrinks γ or sets converged_.
    void close_epoch_and_maybe_terminate_(std::mt19937_64& rng) {
        // Decompose λ̄ into (μ̄, log σ̄).
        arma::vec mu_bar(K_), lsd_bar(K_);
        unpack_lambda(opt_state_.lambda_bar, mu_bar, lsd_bar);

        // Snapshot for SKL termination on the NEXT epoch close.
        epoch_lambda_avg_history_.push_back(pack_lambda(mu_bar, lsd_bar));
        const std::size_t H = epoch_lambda_avg_history_.size();

        // Need at least 3 entries (λ̄_0 initial + λ̄_1 + λ̄_2) for the
        // SKL ratio. epoch_index_ counts EPOCHS, so we have entries
        // 0..epoch_index_+1 after this snapshot. The ratio uses entry
        // epoch_index_+1 (latest) vs entry epoch_index_ (previous) over
        // entry 0 (initial).
        if (H >= 3) {
            arma::vec mu_prev(K_), lsd_prev(K_), mu_init(K_), lsd_init(K_);
            unpack_lambda(epoch_lambda_avg_history_[H - 2], mu_prev, lsd_prev);
            unpack_lambda(epoch_lambda_avg_history_[0],     mu_init, lsd_init);

            const double skl_consec =
                vi_optimizer::skl_mean_field_gaussian(mu_bar, lsd_bar,
                                                       mu_prev, lsd_prev);
            const double skl_initial =
                vi_optimizer::skl_mean_field_gaussian(mu_bar, lsd_bar,
                                                       mu_init, lsd_init);
            if (vi_optimizer::skl_terminate(skl_consec, skl_initial,
                                             cfg_.optimizer.tau)) {
                terminate_(mu_bar, lsd_bar, rng);
                return;
            }
        }

        // Not converged → shrink γ, restart inner epoch.
        if (epoch_index_ + 1 >= cfg_.optimizer.max_epochs) {
            // Cap hit: terminate with whatever we have.
            terminate_(mu_bar, lsd_bar, rng);
            return;
        }

        // Set μ, log σ to the epoch average (improves stability).
        mu_unc_  = mu_bar;
        log_sd_  = lsd_bar;
        refresh_cur_natural_();

        gamma_current_ *= cfg_.optimizer.rho;
        epoch_index_   += 1;
        steps_in_epoch_ = 0;
        opt_state_.reset_epoch(D_);
    }

    /// Set converged_, lock μ and log σ to the final epoch average, and
    /// compute the joint PSIS-k̂ over S samples drawn from the
    /// converged q (the once-only diagnostic stored in history_.final_khat).
    void terminate_(const arma::vec& mu_final, const arma::vec& lsd_final,
                    std::mt19937_64& rng) {
        mu_unc_ = mu_final;
        log_sd_ = lsd_final;
        refresh_cur_natural_();
        converged_ = true;

        // Compute joint k̂.
        const std::size_t S = cfg_.optimizer.S_khat;
        const double log_2pi = std::log(2.0 * M_PI);
        const arma::vec sigma = arma::exp(log_sd_);

        arma::vec log_weights(S);
        std::normal_distribution<double> N01(0.0, 1.0);
        for (std::size_t s = 0; s < S; ++s) {
            arma::vec eps(K_);
            for (std::size_t i = 0; i < K_; ++i) eps[i] = N01(rng);
            const arma::vec eta = mu_unc_ + sigma % eps;
            arma::vec grad_lp(K_, arma::fill::zeros);
            const double log_p = cfg_.log_density_grad(eta, ctx_, &grad_lp);
            const double log_q =
                -0.5 * static_cast<double>(K_) * log_2pi
                - arma::sum(log_sd_)
                - 0.5 * arma::sum(arma::square(eps));
            log_weights[s] = log_p - log_q;
        }
        history_.final_khat = vi_optimizer::psis_khat(log_weights);
    }

    // ---- State ----------------------------------------------------------

    mean_field_gaussian_vi_block_config cfg_;
    std::size_t K_ = 0;
    std::size_t D_ = 0;

    arma::vec mu_unc_;     // current variational mean (unconstrained)
    arma::vec log_sd_;     // current variational log σ
    arma::vec cur_natural_;// cached q-mean on natural scale

    block_context ctx_;    // copy of relevant shared_data keys

    vi_optimizer::avgAdam_state opt_state_;
    std::vector<arma::vec> epoch_lambda_avg_history_;  // λ̄_k packed (H entries)
    double      gamma_current_   = 0.1;
    std::size_t epoch_index_     = 0;
    std::size_t steps_in_epoch_  = 0;
    bool        converged_       = false;
    double      last_elbo_       = std::numeric_limits<double>::quiet_NaN();

    vi_history_t history_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_MEAN_FIELD_GAUSSIAN_VI_BLOCK_HPP

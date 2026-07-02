/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  mean_field_categorical_vi_block.hpp -- mean-field VI for discrete
 *                                         (categorical) latent variables.
 *
 *  VARIATIONAL FAMILY
 *  ==================
 *      q(z_1,...,z_n) = prod_i Categorical(z_i ; phi_i),
 *      phi_i in Simplex^{K_i},   z_i in {0, ..., K_i - 1}
 *
 *  Internally we parameterise each phi_i via an UNCONSTRAINED vector
 *  eta_i in R^{K_i - 1} using a softmax anchored at state 0:
 *      eta_i = (eta_{i,1}, ..., eta_{i,K_i-1}),  eta_{i,0} := 0
 *      phi_{i,k} = exp(eta_{i,k}) / Z_i,  Z_i = 1 + sum_{l>=1} exp(eta_{i,l})
 *  This gives sum_i (K_i - 1) free parameters total.
 *
 *  GRADIENT — ANALYTICAL SUM-OVER-FOCAL-STATE + MONTE-CARLO MARGINALISATION
 *  =======================================================================
 *  Mean-field ELBO:
 *      ELBO(eta) = E_q[log p~(z)] + sum_i H(Cat(phi_i))
 *
 *  For each variable i and state k, define
 *      G_{i,k} := E_{q without i}[ log p~( z_{-i}, z_i = k ) ]
 *
 *  Then
 *      dELBO/dphi_{i,k} = G_{i,k} - log phi_{i,k} - 1
 *  and, chain-ruling through the anchored softmax (eta_{i,0}=0),
 *      dELBO/deta_{i,l} = phi_{i,l} * [
 *           (G_{i,l} - log phi_{i,l})
 *         - sum_k phi_{i,k} (G_{i,k} - log phi_{i,k})
 *      ]                                  for l = 1, ..., K_i - 1
 *
 *  The expectation in G_{i,k} is computed by a SHARED Monte-Carlo set:
 *  one draws z_s ~ q for s = 1..S, and for each (i, k) substitutes
 *  z_s[i] := k before evaluating log p~. The same S samples are reused
 *  for every (i, k) pair, so the gradient cost is
 *      S * sum_i K_i  log-density evaluations per step.
 *
 *  This is the categorical analogue of Bishop PRML eq. 10.9 (CAVI) made
 *  generic for any user-supplied log-density: when log p~ is exponential
 *  family, the inner expectation has a closed form and S=1 (no Monte
 *  Carlo) suffices; for generic non-factorising log p~, S > 1 trades
 *  variance for cost. We use S = 16 by default (configurable).
 *
 *  EXACT MODE
 *  ==========
 *  If cfg.exact_enumeration = true, the block enumerates the full
 *  joint state space prod_i K_i (capped at 4096 states for safety) and
 *  computes G_{i,k} exactly. Useful for unit tests / 4-node Ising
 *  validation (3^4 = 81 states).
 *
 *  ELBO ESTIMATE
 *  =============
 *  We Rao-Blackwellise via the focal-variable sum:
 *      E_q[log p~] ~= (1/n) * sum_i sum_k phi_{i,k} * G_{i,k}
 *  This is exact when G_{i,k} are exact, and lower variance than naive
 *  log p~(z_s) averaging when MC. Plus the closed-form entropy
 *      H_total = sum_i [ - sum_k phi_{i,k} log phi_{i,k} ].
 *
 *  TARGET-SHAPE CATEGORY (system_design.md §11.2(b))
 *  =================================================
 *  Discrete latents with potentially strong local dependence. Mean-field
 *  VI is a deterministic deterministic approximation that converges
 *  cleanly even for strongly coupled targets (Bishop §10.1.2), at the
 *  cost of underestimating posterior variance. Cross-check vs exact
 *  enumeration on the 4-node K=3 Ising chain is shipped in
 *  examples/CategoricalIsingChainVI.cpp + tests/audit_CategoricalIsingChainVI.R.
 *
 *  ENGINE FAMILY
 *  =============
 *  engine_kind() = VI (inherited from vi_block).
 *  Composite_block writes a q-SAMPLE (NOT q-mean) to shared_data after
 *  each step(), per §18.4 — value is the per-variable integer index
 *  vector (length n), so MCMC siblings see a fresh discrete draw each
 *  outer iteration.
 *
 *  STATE EXPOSED VIA current()
 *  ===========================
 *  current() returns the concatenated probability vector phi
 *  (length total_K = sum_i K_i) as the q-mean point estimate. R-level
 *  get_current() consumers receive this and can reshape per variable.
 *
 *  HISTORY
 *  =======
 *  When keep_history=true:
 *      vi_history_t.elbo    : per-step ELBO
 *      vi_history_t.mu      : per-step eta (length total_D = sum_i (K_i-1))
 *      vi_history_t.log_sd  : per-step phi (length total_K = sum_i K_i)
 *                             — reused field for "diagnostic vector"
 *      vi_history_t.gamma   : per-step learning rate
 *      vi_history_t.epoch   : per-step epoch index
 *      vi_history_t.final_khat : joint PSIS-k̂ at SKL termination
 *
 *  VALIDATOR
 *  =========
 *  - Check #15 (parity): finite-difference gradient match in
 *      tests/test_mean_field_categorical_vi_block.cpp T2.
 *  - Check #21 (vi_block contract): step writes q-sample to shared_data
 *      (inherited from vi_block); current_named_outputs(rng) inherited.
 *  - Check #22 (optimizer = RAABBVI): uses vi_optimizer::avgAdam_step,
 *      vi_optimizer::skl_terminate.
 *  - Check #23 (Layer-3 PSIS-k̂ < 0.7): computed at SKL termination
 *      using vi_optimizer::psis_khat over S samples from converged q.
 *================================================================================*/

#ifndef AI4BAYESCODE_MEAN_FIELD_CATEGORICAL_VI_BLOCK_HPP
#define AI4BAYESCODE_MEAN_FIELD_CATEGORICAL_VI_BLOCK_HPP

#include "vi_block.hpp"
#include "vi_optimizer.hpp"

#include <cmath>
#include <cstdint>
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

/// User-supplied log p~ evaluator: returns log p~(z, ctx) for an
/// arbitrary discrete assignment z. The block evaluates this many times
/// per step (S * sum_i K_i times) so the callback should be reasonably
/// fast and SHOULD NOT have side effects (it is called from inside
/// step() in a tight loop). Reading from ctx is fine.
using categorical_log_density_fn =
    std::function<double(const arma::uvec& z, const block_context& ctx)>;

/**
 * @brief Configuration for mean_field_categorical_vi_block.
 *
 * Required fields: name, cardinalities, log_density. Everything else
 * has sensible defaults.
 */
struct mean_field_categorical_vi_block_config {
    /// Unique block name in its composite. Becomes the shared_data key
    /// that holds the per-step q-sample (integer indices, length n).
    std::string name;

    /// Cardinalities K_i of each latent variable. Length n. Each K_i
    /// must be >= 2. Block dim = n; total free params = sum_i (K_i - 1).
    arma::uvec cardinalities;

    /// log p~(z, ctx): joint log-density (up to normalisation). The
    /// block evaluates this on integer-encoded z. Required.
    categorical_log_density_fn log_density;

    /// shared_data keys this block reads via ctx during step().
    std::vector<std::string> dependencies;

    /// Number of MC samples per gradient evaluation. Default 16 is a
    /// good variance/cost balance for moderate problems. Increase for
    /// noisier gradients; decrease if log_density is expensive.
    std::size_t n_mc_samples = 16;

    /// If true (and joint state space prod_i K_i <= exact_state_cap),
    /// computes G_{i,k} by full enumeration — gradient is then EXACT.
    /// Useful for small models / unit tests. Forces single-shot, no MC.
    bool exact_enumeration = false;

    /// Cap on total joint state count for exact_enumeration. Default
    /// 4096: prod K_i larger than this falls back to MC even when
    /// exact_enumeration was requested (with a warning written via
    /// the warning_fn callback if set).
    std::size_t exact_state_cap = 4096;

    /// Optional callback for non-fatal warnings (e.g., exact_enumeration
    /// requested but joint state too large). Default = nullptr (silent).
    std::function<void(const std::string& msg)> warning_fn;

    /// If non-zero, perturb eta initialisation by N(0, init_random_eps).
    /// Breaks symmetry for genuinely coupled posteriors that would
    /// otherwise stay at uniform. Default 0 (pure uniform start).
    double init_random_eps = 0.0;

    /// RAABBVI optimizer hyperparams. Defaults work for most problems.
    vi_optimizer::raabbvi_config optimizer;

    /// Seed for the deterministic init perturbation. Independent of the
    /// step() rng. Default 0.
    std::uint64_t init_rng_seed = 0;
};

/**
 * @brief Mean-field VI block for discrete categorical latents.
 *
 * See file header for the math, gradient, and ELBO derivation. The block
 * inherits vi_block, so engine_kind() == VI and the q-sample-write
 * convention to shared_data is automatic.
 *
 * Invariants:
 *   - cfg_.cardinalities is non-empty; every K_i >= 2.
 *   - eta_ has length total_D_ = sum_i (K_i - 1).
 *   - phi_ has length total_K_ = sum_i K_i. phi_ slice for var i sums
 *     to exactly 1 by construction (anchored softmax + normalise).
 *   - The shared_data write key is cfg_.name; value is a length-n
 *     arma::vec of doubles where entry i is the sampled state z_i in
 *     {0, 1, ..., K_i - 1}.
 */
class mean_field_categorical_vi_block : public vi_block {
public:
    explicit mean_field_categorical_vi_block(
        mean_field_categorical_vi_block_config cfg)
        : cfg_(std::move(cfg)) {

        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "mean_field_categorical_vi_block: cfg.name required");
        }
        if (cfg_.cardinalities.is_empty()) {
            throw std::invalid_argument(
                "mean_field_categorical_vi_block: cfg.cardinalities required");
        }
        if (!cfg_.log_density) {
            throw std::invalid_argument(
                "mean_field_categorical_vi_block: cfg.log_density required");
        }
        for (std::size_t i = 0; i < cfg_.cardinalities.n_elem; ++i) {
            if (cfg_.cardinalities[i] < 2u) {
                throw std::invalid_argument(
                    "mean_field_categorical_vi_block: every cardinality "
                    "must be >= 2");
            }
        }

        n_ = cfg_.cardinalities.n_elem;

        // Build offsets and totals.
        offsets_eta_.assign(n_ + 1, 0);
        offsets_phi_.assign(n_ + 1, 0);
        for (std::size_t i = 0; i < n_; ++i) {
            const std::size_t Ki = cfg_.cardinalities[i];
            offsets_eta_[i + 1] = offsets_eta_[i] + (Ki - 1u);
            offsets_phi_[i + 1] = offsets_phi_[i] + Ki;
        }
        total_D_ = offsets_eta_[n_];
        total_K_ = offsets_phi_[n_];

        // Check exact-mode tractability.
        if (cfg_.exact_enumeration) {
            std::size_t total_states = 1;
            for (std::size_t i = 0; i < n_; ++i) {
                const std::size_t prev = total_states;
                total_states *= cfg_.cardinalities[i];
                if (total_states < prev) {  // overflow
                    total_states = std::numeric_limits<std::size_t>::max();
                    break;
                }
            }
            if (total_states > cfg_.exact_state_cap) {
                if (cfg_.warning_fn) {
                    cfg_.warning_fn(
                        "mean_field_categorical_vi_block: joint state count "
                        + std::to_string(total_states)
                        + " exceeds exact_state_cap "
                        + std::to_string(cfg_.exact_state_cap)
                        + "; falling back to Monte Carlo gradient.");
                }
                exact_enumeration_active_ = false;
            } else {
                exact_enumeration_active_ = true;
                exact_total_states_ = total_states;
            }
        }

        // Initialise eta = 0 (=> phi uniform). Optionally perturb.
        eta_.zeros(total_D_);
        if (cfg_.init_random_eps > 0.0) {
            std::mt19937_64 init_rng(cfg_.init_rng_seed
                                     ^ 0xD1B54A32D192ED03ULL);
            std::normal_distribution<double> N(0.0, cfg_.init_random_eps);
            for (std::size_t i = 0; i < total_D_; ++i) {
                eta_[i] = N(init_rng);
            }
        }

        phi_.zeros(total_K_);
        update_phi_from_eta_();

        // Initial epoch state.
        epoch_eta_avg_history_.push_back(eta_);
        opt_state_.reset_epoch(total_D_);
        gamma_current_ = cfg_.optimizer.gamma_0;
        epoch_index_   = 0;
        steps_in_epoch_ = 0;
        converged_     = false;
        last_elbo_     = std::numeric_limits<double>::quiet_NaN();
    }

    // ---- block_sampler interface ----------------------------------------

    void set_context(const block_context& ctx) override {
        ctx_.clear();
        for (const auto& key : cfg_.dependencies) {
            auto it = ctx.find(key);
            if (it != ctx.end()) {
                ctx_.emplace(key, it->second);
            }
        }
    }

    void step(std::mt19937_64& rng) override {
        if (converged_) return;

        // Compute gradient + ELBO at current eta.
        arma::vec grad_eta(total_D_, arma::fill::zeros);
        const double elbo = compute_elbo_and_grad_(rng, &grad_eta);
        last_elbo_ = elbo;

        // avgAdam step on eta. The gradient computed above is dELBO/deta;
        // RAABBVI expects gradient of -ELBO, so negate.
        arma::vec neg_grad = -grad_eta;
        vi_optimizer::avgAdam_step(eta_, opt_state_, neg_grad,
                                   gamma_current_, cfg_.optimizer);
        update_phi_from_eta_();

        steps_in_epoch_ += 1;

        if (keep_history()) {
            history_.elbo.push_back(elbo);
            history_.mu.push_back(eta_);     // store eta as "mu"
            history_.log_sd.push_back(phi_); // store phi as "log_sd" (diag vec)
            history_.gamma.push_back(gamma_current_);
            history_.epoch.push_back(static_cast<int>(epoch_index_));
        }

        if (steps_in_epoch_ >= cfg_.optimizer.inner_iter_per_epoch) {
            close_epoch_and_maybe_terminate_(rng);
        }
    }

    const arma::vec& current() const override {
        // q-mean (point estimate) = phi concatenated, length total_K_.
        return phi_;
    }

    void set_current(const arma::vec& phi_new) override {
        if (phi_new.n_elem != total_K_) {
            throw std::invalid_argument(
                "mean_field_categorical_vi_block::set_current: expected length "
                + std::to_string(total_K_) + ", got "
                + std::to_string(phi_new.n_elem));
        }
        // Validate simplex per variable and recover eta.
        for (std::size_t i = 0; i < n_; ++i) {
            const std::size_t Ki = cfg_.cardinalities[i];
            double sum = 0.0;
            for (std::size_t k = 0; k < Ki; ++k) {
                const double v = phi_new[offsets_phi_[i] + k];
                if (!(v >= 0.0) || !std::isfinite(v)) {
                    throw std::invalid_argument(
                        "mean_field_categorical_vi_block::set_current: phi "
                        "must be non-negative and finite");
                }
                sum += v;
            }
            if (std::abs(sum - 1.0) > 1e-6) {
                throw std::invalid_argument(
                    "mean_field_categorical_vi_block::set_current: phi "
                    "for variable " + std::to_string(i)
                    + " must sum to 1 (got " + std::to_string(sum) + ")");
            }
        }
        // Recover eta from phi via log(phi_k / phi_0). Clamp phi to avoid
        // -inf eta when phi_0 = 0 (use eps).
        const double eps = 1e-12;
        for (std::size_t i = 0; i < n_; ++i) {
            const std::size_t Ki = cfg_.cardinalities[i];
            const double phi_0 = std::max(phi_new[offsets_phi_[i]], eps);
            for (std::size_t l = 1; l < Ki; ++l) {
                const double phi_l = std::max(phi_new[offsets_phi_[i] + l], eps);
                eta_[offsets_eta_[i] + (l - 1u)] =
                    std::log(phi_l) - std::log(phi_0);
            }
        }
        update_phi_from_eta_();
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    /// dim() = n (number of latent variables). Matches the shared_data
    /// write shape (length-n vector of sampled integer indices).
    std::size_t dim() const noexcept override { return n_; }

    history_map get_history() const override {
        history_map out;
        const std::size_t T = history_.elbo.size();
        if (T == 0) {
            // Fall back to current phi as a single row.
            arma::mat m(1, total_K_);
            for (std::size_t j = 0; j < total_K_; ++j) m(0, j) = phi_[j];
            out.emplace(cfg_.name, std::move(m));
            return out;
        }
        // Layout: row = step, columns = [elbo, gamma, epoch, eta(total_D_),
        //                                phi(total_K_), final_khat]
        const std::size_t cols = 3 + total_D_ + total_K_ + 1;
        arma::mat m(T, cols);
        for (std::size_t t = 0; t < T; ++t) {
            m(t, 0) = history_.elbo[t];
            m(t, 1) = history_.gamma[t];
            m(t, 2) = static_cast<double>(history_.epoch[t]);
            for (std::size_t j = 0; j < total_D_; ++j) {
                m(t, 3 + j) = history_.mu[t][j];
            }
            for (std::size_t j = 0; j < total_K_; ++j) {
                m(t, 3 + total_D_ + j) = history_.log_sd[t][j];
            }
            m(t, cols - 1) = history_.final_khat;
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

    /// Draw z ~ q (one integer per variable). Returns length-n arma::vec
    /// where entry i is the sampled state in {0, ..., K_i - 1}.
    /// const-correct via the rng-aware contract.
    arma::vec current_sample(std::mt19937_64& rng) const override {
        arma::vec z(n_);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (std::size_t i = 0; i < n_; ++i) {
            const std::size_t Ki = cfg_.cardinalities[i];
            const double u = U(rng);
            double acc = 0.0;
            std::size_t pick = Ki - 1;
            for (std::size_t k = 0; k < Ki; ++k) {
                acc += phi_[offsets_phi_[i] + k];
                if (u <= acc) { pick = k; break; }
            }
            z[i] = static_cast<double>(pick);
        }
        return z;
    }

    /// Returns eta (the unconstrained variational params), length total_D_.
    /// Reuses the get_log_sd slot for the "second variational parameter"
    /// vector — for Categorical there is no log sd; eta IS the parameter.
    arma::vec get_log_sd() const override { return eta_; }

    /// For Categorical, mu argument is interpreted as phi (the q-mean
    /// natural-scale point estimate), log_sd argument is eta. Both can
    /// be set independently; we honour both. If both supplied, eta
    /// takes precedence (more direct).
    void set_variational_state(const arma::vec& mu_or_phi,
                               const arma::vec& log_sd_or_eta) override {
        if (log_sd_or_eta.n_elem == total_D_) {
            eta_ = log_sd_or_eta;
            update_phi_from_eta_();
        } else if (mu_or_phi.n_elem == total_K_) {
            set_current(mu_or_phi);
        } else {
            throw std::invalid_argument(
                "mean_field_categorical_vi_block::set_variational_state: "
                "neither mu (length total_K_) nor log_sd (length total_D_) "
                "matches expected dimensions");
        }
    }

    double current_elbo() const override { return last_elbo_; }

    const vi_history_t& vi_history() const override { return history_; }

    // ---- Public diagnostics ---------------------------------------------

    bool converged() const noexcept { return converged_; }
    std::size_t epoch() const noexcept { return epoch_index_; }
    double gamma() const noexcept { return gamma_current_; }
    std::size_t total_K() const noexcept { return total_K_; }
    std::size_t total_D() const noexcept { return total_D_; }
    bool exact_active() const noexcept { return exact_enumeration_active_; }

private:
    // ---- Helpers --------------------------------------------------------

    /// Apply anchored softmax slice-by-slice to refresh phi from eta.
    void update_phi_from_eta_() {
        for (std::size_t i = 0; i < n_; ++i) {
            const std::size_t Ki = cfg_.cardinalities[i];
            const std::size_t eb = offsets_eta_[i];
            const std::size_t pb = offsets_phi_[i];
            // Anchored: eta_{i,0} = 0 implicitly; eta_{i, l} stored at
            // eta_[eb + l - 1] for l = 1..Ki-1.
            double max_eta = 0.0;  // eta_{i,0} = 0
            for (std::size_t l = 1; l < Ki; ++l) {
                const double e = eta_[eb + l - 1];
                if (e > max_eta) max_eta = e;
            }
            double Z = std::exp(0.0 - max_eta);  // term for state 0
            for (std::size_t l = 1; l < Ki; ++l) {
                Z += std::exp(eta_[eb + l - 1] - max_eta);
            }
            phi_[pb + 0] = std::exp(0.0 - max_eta) / Z;
            for (std::size_t l = 1; l < Ki; ++l) {
                phi_[pb + l] = std::exp(eta_[eb + l - 1] - max_eta) / Z;
            }
        }
    }

    /// Compute ELBO and gradient w.r.t. eta in one pass.
    ///
    /// Returns the ELBO estimate. Fills *g_eta with dELBO/deta.
    ///
    /// Two implementations selected by exact_enumeration_active_:
    ///   (a) exact: enumerate all joint z; compute G_{i,k} exactly
    ///   (b) MC: sample z_s ~ q; for each (i,k), replace z_s[i]:=k and
    ///       average log p~ over s; gives unbiased G_{i,k} estimate
    double compute_elbo_and_grad_(std::mt19937_64& rng,
                                   arma::vec* g_eta) const {
        // G_{i,k} stored linearly using phi offsets.
        arma::vec G(total_K_, arma::fill::zeros);

        if (exact_enumeration_active_) {
            compute_G_exact_(G);
        } else {
            compute_G_mc_(rng, G, cfg_.n_mc_samples);
        }

        // dELBO/dphi_{i,k} = G_{i,k} - log phi_{i,k} - 1
        // (we leave -1 implicit; its effect cancels in the softmax chain
        //  rule due to sum_k phi_{i,k} (delta_{kl} - phi_{i,l}) = 0)
        arma::vec g_phi(total_K_);
        for (std::size_t i = 0; i < n_; ++i) {
            const std::size_t Ki = cfg_.cardinalities[i];
            for (std::size_t k = 0; k < Ki; ++k) {
                const double phi_ik = phi_[offsets_phi_[i] + k];
                const double log_phi = (phi_ik > 0.0)
                                       ? std::log(phi_ik)
                                       : -std::numeric_limits<double>::infinity();
                g_phi[offsets_phi_[i] + k] = G[offsets_phi_[i] + k] - log_phi;
            }
        }

        // Chain rule through anchored softmax to get dELBO/deta:
        //   dELBO/deta_{i,l} = phi_{i,l} * [
        //       (G_{i,l} - log phi_{i,l})
        //     - sum_k phi_{i,k} (G_{i,k} - log phi_{i,k})
        //   ]   for l = 1, ..., K_i - 1
        for (std::size_t i = 0; i < n_; ++i) {
            const std::size_t Ki = cfg_.cardinalities[i];
            // Compute centring constant: sum_k phi_{i,k} * g_phi_{i,k}
            double centre = 0.0;
            for (std::size_t k = 0; k < Ki; ++k) {
                centre += phi_[offsets_phi_[i] + k]
                        * g_phi[offsets_phi_[i] + k];
            }
            for (std::size_t l = 1; l < Ki; ++l) {
                const double phi_l = phi_[offsets_phi_[i] + l];
                (*g_eta)[offsets_eta_[i] + (l - 1u)] =
                    phi_l * (g_phi[offsets_phi_[i] + l] - centre);
            }
        }

        // ELBO estimate. Rao-Blackwellise via i=0:
        //   E_q[log p~] ~= sum_k phi_{0,k} * G_{0,k}
        // (any i works; the average over i would be lower variance but
        //  costs nothing extra to use just i=0)
        double e_log_p = 0.0;
        const std::size_t K0 = cfg_.cardinalities[0];
        for (std::size_t k = 0; k < K0; ++k) {
            e_log_p += phi_[k] * G[k];
        }
        // Entropy: H_total = sum_i [- sum_k phi_{i,k} log phi_{i,k}]
        double H_total = 0.0;
        for (std::size_t i = 0; i < n_; ++i) {
            const std::size_t Ki = cfg_.cardinalities[i];
            for (std::size_t k = 0; k < Ki; ++k) {
                const double p = phi_[offsets_phi_[i] + k];
                if (p > 0.0) H_total -= p * std::log(p);
            }
        }
        return e_log_p + H_total;
    }

    /// Exact enumeration over all joint z. Cost = prod K_i log_density
    /// evaluations. Used only when exact_state_cap is not exceeded.
    void compute_G_exact_(arma::vec& G_out) const {
        // Joint state count.
        std::size_t total = 1;
        for (std::size_t i = 0; i < n_; ++i) {
            total *= cfg_.cardinalities[i];
        }
        arma::uvec z(n_, arma::fill::zeros);

        // Need sum over z of [prod_{j != i} phi_{j, z_j}] * log_density(z)
        // for each (i, k) where z_i = k. Equivalently:
        //   G_{i,k} = sum_{z: z_i = k} [prod_{j != i} phi_{j, z_j}] * lp(z)
        //
        // Implementation: enumerate every z, compute weight w_full =
        // prod_j phi_{j, z_j}; for each i, the marginalised weight at
        // z_i = z[i] is w_full / phi_{i, z[i]} (handling zero); accumulate.
        for (std::size_t s = 0; s < total; ++s) {
            // Decode s into z.
            std::size_t r = s;
            for (std::size_t i = 0; i < n_; ++i) {
                const std::size_t Ki = cfg_.cardinalities[i];
                z[i] = r % Ki;
                r /= Ki;
            }
            const double lp = cfg_.log_density(z, ctx_);

            double w_full = 1.0;
            for (std::size_t i = 0; i < n_; ++i) {
                w_full *= phi_[offsets_phi_[i] + z[i]];
            }
            // Skip if w_full is effectively 0 (avoids div-by-zero below).
            if (!(w_full > 0.0)) continue;

            for (std::size_t i = 0; i < n_; ++i) {
                const double phi_iz = phi_[offsets_phi_[i] + z[i]];
                if (!(phi_iz > 0.0)) continue;
                const double w_marg = w_full / phi_iz;
                G_out[offsets_phi_[i] + z[i]] += w_marg * lp;
            }
        }
    }

    /// Monte Carlo G_{i,k} estimator.
    /// Sample z_s ~ q for s = 1..S; for each (i, k), replace z_s[i] := k
    /// and evaluate log_density; G_{i,k} = (1/S) * sum_s log_density(z_s
    /// with z_s[i] = k).
    void compute_G_mc_(std::mt19937_64& rng, arma::vec& G_out,
                       std::size_t S) const {
        if (S == 0) S = 1;
        arma::uvec z(n_);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        const double inv_S = 1.0 / static_cast<double>(S);

        for (std::size_t s = 0; s < S; ++s) {
            // Sample z ~ q.
            for (std::size_t i = 0; i < n_; ++i) {
                const std::size_t Ki = cfg_.cardinalities[i];
                const double u = U(rng);
                double acc = 0.0;
                std::size_t pick = Ki - 1;
                for (std::size_t k = 0; k < Ki; ++k) {
                    acc += phi_[offsets_phi_[i] + k];
                    if (u <= acc) { pick = k; break; }
                }
                z[i] = pick;
            }

            // For each (i, k), replace z[i] := k, evaluate log p~.
            for (std::size_t i = 0; i < n_; ++i) {
                const std::size_t Ki = cfg_.cardinalities[i];
                const std::size_t saved = z[i];
                for (std::size_t k = 0; k < Ki; ++k) {
                    z[i] = k;
                    const double lp = cfg_.log_density(z, ctx_);
                    G_out[offsets_phi_[i] + k] += inv_S * lp;
                }
                z[i] = saved;
            }
        }
    }

    /// SKL termination check + epoch shrink (categorical version).
    void close_epoch_and_maybe_terminate_(std::mt19937_64& rng) {
        // The epoch-averaged eta is opt_state_.lambda_bar.
        const arma::vec eta_bar = opt_state_.lambda_bar;
        epoch_eta_avg_history_.push_back(eta_bar);
        const std::size_t H = epoch_eta_avg_history_.size();

        // Need at least 3 entries (eta_0 init + eta_1 + eta_2).
        if (H >= 3) {
            const arma::vec& eta_prev = epoch_eta_avg_history_[H - 2];
            const arma::vec& eta_init = epoch_eta_avg_history_[0];

            const double skl_consec  = skl_eta_(eta_bar, eta_prev);
            const double skl_initial = skl_eta_(eta_bar, eta_init);

            if (vi_optimizer::skl_terminate(skl_consec, skl_initial,
                                             cfg_.optimizer.tau)) {
                terminate_(eta_bar, rng);
                return;
            }
        }

        // Not converged: shrink γ, restart inner epoch.
        if (epoch_index_ + 1 >= cfg_.optimizer.max_epochs) {
            terminate_(eta_bar, rng);
            return;
        }

        eta_ = eta_bar;
        update_phi_from_eta_();
        gamma_current_ *= cfg_.optimizer.rho;
        epoch_index_  += 1;
        steps_in_epoch_ = 0;
        opt_state_.reset_epoch(total_D_);
    }

    /// Compute SKL(Cat_a, Cat_b) for two eta vectors. Uses the standard
    /// closed form for categorical:
    ///   KL(Cat(p) || Cat(q)) = sum_k p_k * log(p_k / q_k)
    ///   SKL = KL(p||q) + KL(q||p)
    /// We sum over all variables.
    double skl_eta_(const arma::vec& eta_a, const arma::vec& eta_b) const {
        if (eta_a.n_elem != total_D_ || eta_b.n_elem != total_D_) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // Materialise phi_a, phi_b slice by slice and sum.
        double skl_sum = 0.0;
        const double eps = 1e-30;
        for (std::size_t i = 0; i < n_; ++i) {
            const std::size_t Ki = cfg_.cardinalities[i];
            const std::size_t eb = offsets_eta_[i];

            // Softmax of eta_a slice.
            double max_a = 0.0, max_b = 0.0;
            for (std::size_t l = 1; l < Ki; ++l) {
                const double ea = eta_a[eb + (l - 1u)];
                const double eb_ = eta_b[eb + (l - 1u)];
                if (ea > max_a) max_a = ea;
                if (eb_ > max_b) max_b = eb_;
            }
            std::vector<double> pa(Ki), pb(Ki);
            double Za = std::exp(-max_a), Zb = std::exp(-max_b);
            for (std::size_t l = 1; l < Ki; ++l) {
                Za += std::exp(eta_a[eb + (l - 1u)] - max_a);
                Zb += std::exp(eta_b[eb + (l - 1u)] - max_b);
            }
            pa[0] = std::exp(-max_a) / Za;
            pb[0] = std::exp(-max_b) / Zb;
            for (std::size_t l = 1; l < Ki; ++l) {
                pa[l] = std::exp(eta_a[eb + (l - 1u)] - max_a) / Za;
                pb[l] = std::exp(eta_b[eb + (l - 1u)] - max_b) / Zb;
            }
            for (std::size_t k = 0; k < Ki; ++k) {
                const double a = std::max(pa[k], eps);
                const double b = std::max(pb[k], eps);
                skl_sum += pa[k] * (std::log(a) - std::log(b))
                         + pb[k] * (std::log(b) - std::log(a));
            }
        }
        return skl_sum;
    }

    /// Final termination: lock eta, compute joint PSIS-k̂ over S samples
    /// drawn from the converged q.
    void terminate_(const arma::vec& eta_final, std::mt19937_64& rng) {
        eta_ = eta_final;
        update_phi_from_eta_();
        converged_ = true;

        const std::size_t S = cfg_.optimizer.S_khat;
        arma::vec log_weights(S);
        arma::uvec z(n_);
        std::uniform_real_distribution<double> U(0.0, 1.0);

        for (std::size_t s = 0; s < S; ++s) {
            // Sample z ~ q AND record log q(z).
            double log_q = 0.0;
            for (std::size_t i = 0; i < n_; ++i) {
                const std::size_t Ki = cfg_.cardinalities[i];
                const double u = U(rng);
                double acc = 0.0;
                std::size_t pick = Ki - 1;
                for (std::size_t k = 0; k < Ki; ++k) {
                    acc += phi_[offsets_phi_[i] + k];
                    if (u <= acc) { pick = k; break; }
                }
                z[i] = pick;
                const double phi_pick = std::max(phi_[offsets_phi_[i] + pick],
                                                  1e-300);
                log_q += std::log(phi_pick);
            }
            const double log_p = cfg_.log_density(z, ctx_);
            log_weights[s] = log_p - log_q;
        }
        history_.final_khat = vi_optimizer::psis_khat(log_weights);
    }

    // ---- State ----------------------------------------------------------

    mean_field_categorical_vi_block_config cfg_;
    std::size_t n_       = 0;     // number of latent variables
    std::size_t total_K_ = 0;     // sum of K_i
    std::size_t total_D_ = 0;     // sum of (K_i - 1)
    std::vector<std::size_t> offsets_eta_;  // length n+1
    std::vector<std::size_t> offsets_phi_;  // length n+1

    arma::vec eta_;     // length total_D_
    arma::vec phi_;     // length total_K_

    block_context ctx_;

    vi_optimizer::avgAdam_state opt_state_;
    std::vector<arma::vec> epoch_eta_avg_history_;
    double gamma_current_  = 0.1;
    std::size_t epoch_index_     = 0;
    std::size_t steps_in_epoch_  = 0;
    bool converged_        = false;
    double last_elbo_      = std::numeric_limits<double>::quiet_NaN();

    bool exact_enumeration_active_ = false;
    std::size_t exact_total_states_ = 0;

    vi_history_t history_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_MEAN_FIELD_CATEGORICAL_VI_BLOCK_HPP

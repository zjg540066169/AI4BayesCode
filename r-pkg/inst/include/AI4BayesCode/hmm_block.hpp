/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  hmm_block.hpp  --  Exact forward-filter backward-sample block for
 *                     Hidden Markov Models with finite state space
 *                     (T10, Fruhwirth-Schnatter 2006 Ch. 11).
 *
 *  MODEL
 *  =====
 *      z_1          ~ Categorical(pi)                         initial state
 *      z_t | z_{t-1}=k  ~ Categorical(A_{k,:})               t = 2..T
 *      y_t | z_t = k    ~ p(y_t | theta_k)                    emission density
 *
 *  The block samples the LATENT state sequence z_1:T exactly from its
 *  full conditional p(z_1:T | y_1:T, A, pi, theta) using the standard
 *  forward-filter backward-sample (FFBS) algorithm. K is fixed.
 *
 *  The block is a `*_gibbs_block` family member — EXCEPTION 1 of
 *  codegen.md §2b (discrete parameters; NUTS cannot target discrete).
 *  The emission densities and transition / initial distributions are
 *  everything ELSE in the model and are sampled by other blocks
 *  (typically `dirichlet_gibbs_block` for A, `beta_gibbs_block` for pi,
 *  `nuts_block` / `beta_gibbs_block` for emission parameters). This
 *  block reads A, pi, and emission log-density from context each sweep.
 *
 *  WHAT THE USER SUPPLIES
 *  ======================
 *  Config fields (see struct below for full docs):
 *    - T, K              : sequence length and state count
 *    - A_key, pi_key     : shared_data keys for transition + initial
 *    - emission_logp_fn  : per-(t, k) log-density of y_t | z_t=k
 *    - initial_z         : optional starting z_1:T vector
 *
 *  The emission_logp_fn reads any parameters it needs (e.g. cluster
 *  means / variances for a Gaussian mixture HMM) from ctx; these are
 *  managed by sibling blocks.
 *
 *  FFBS ALGORITHM
 *  ==============
 *  Forward filter (log-space):
 *    alpha_1(k)  = log pi_k + log p(y_1 | z_1=k)
 *    alpha_t(k)  = logsumexp_{j=1..K} [ alpha_{t-1}(j) + log A[j, k] ]
 *                  + log p(y_t | z_t=k)
 *
 *  Backward sample:
 *    z_T         ~ Categorical( softmax(alpha_T) )
 *    z_{t-1}     ~ Categorical( softmax(alpha_{t-1} + log A[:, z_t]) )
 *                  for t = T, T-1, ..., 2
 *
 *  Complexity per sweep: O(T * K^2). Exact conditional draw.
 *
 *  VALIDATOR
 *  =========
 *  Check #15 (library parity test):
 *    tests_autodiff/test_hmm_block.cpp — verifies FFBS on a minimal
 *    K=2 HMM against Baum-Welch forward-algorithm marginals.
 *================================================================================*/

#ifndef AI4BAYESCODE_HMM_BLOCK_HPP
#define AI4BAYESCODE_HMM_BLOCK_HPP

#include "block_sampler.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

// Numerically-stable log-sum-exp for a length-K array.
inline double logsumexp(const double* x, std::size_t K) {
    double m = x[0];
    for (std::size_t k = 1; k < K; ++k) if (x[k] > m) m = x[k];
    if (!std::isfinite(m)) return m;   // all -Inf → -Inf
    double s = 0.0;
    for (std::size_t k = 0; k < K; ++k) s += std::exp(x[k] - m);
    return m + std::log(s);
}

struct hmm_block_config {
    std::string name = "z";

    /// Sequence length (number of time points).
    std::size_t T = 0;

    /// State count (size of discrete state space).
    std::size_t K = 0;

    /// shared_data key: length-(K*K) flattened transition matrix. We use
    /// **row-major** storage: A[row j * K + col k] = P(z_{t+1} = k | z_t = j).
    /// Each row must sum to 1.
    std::string A_key = "A";

    /// shared_data key: length-K initial distribution over states.
    /// Must sum to 1.
    std::string pi_key = "pi";

    /// Emission log-density evaluator: log p(y_t | z_t = k, theta).
    /// The function receives the timestep t and state k, plus the full
    /// block_context (where theta / y are stored by sibling blocks).
    /// Must return a finite log-density for legal (t, k).
    std::function<double(std::size_t t, std::size_t k,
                         const block_context& ctx)> emission_logp;

    /// Optional initial z_1:T vector (length T, entries in {0..K-1} as
    /// doubles for arma::vec storage). If empty, z is initialized by
    /// drawing uniformly over the K states.
    arma::vec initial_z;
};

class hmm_block : public block_sampler {
public:
    explicit hmm_block(hmm_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.T == 0) {
            throw std::invalid_argument(
                "hmm_block: T (sequence length) must be > 0");
        }
        if (cfg_.K < 2) {
            throw std::invalid_argument(
                "hmm_block: K (state count) must be >= 2");
        }
        if (!cfg_.emission_logp) {
            throw std::invalid_argument(
                "hmm_block: emission_logp function is required");
        }

        z_.set_size(cfg_.T);
        if (cfg_.initial_z.n_elem == cfg_.T) {
            for (std::size_t t = 0; t < cfg_.T; ++t) {
                const int s = static_cast<int>(std::round(cfg_.initial_z[t]));
                if (s < 0 || s >= static_cast<int>(cfg_.K)) {
                    throw std::invalid_argument(
                        "hmm_block: initial_z entries must be in {0..K-1}");
                }
                z_[t] = static_cast<double>(s);
            }
        } else if (cfg_.initial_z.n_elem == 0) {
            for (std::size_t t = 0; t < cfg_.T; ++t) {
                z_[t] = static_cast<double>(t % cfg_.K);  // deterministic init
            }
        } else {
            throw std::invalid_argument(
                "hmm_block: initial_z length must be T or 0");
        }
    }

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        const std::size_t T = cfg_.T;
        const std::size_t K = cfg_.K;

        const arma::vec& A_flat  = context_.at(cfg_.A_key);
        const arma::vec& pi_flat = context_.at(cfg_.pi_key);
        if (A_flat.n_elem != K * K) {
            throw std::runtime_error(
                "hmm_block '" + cfg_.name +
                "': transition matrix length mismatch (got " +
                std::to_string(A_flat.n_elem) + ", expected " +
                std::to_string(K * K) + ")");
        }
        if (pi_flat.n_elem != K) {
            throw std::runtime_error(
                "hmm_block '" + cfg_.name +
                "': initial distribution length mismatch");
        }

        // Precompute log A[j, k] and log pi for stability.
        std::vector<double> logA(K * K);
        for (std::size_t i = 0; i < K * K; ++i) {
            logA[i] = (A_flat[i] > 0.0) ? std::log(A_flat[i])
                                        : -std::numeric_limits<double>::infinity();
        }
        std::vector<double> logpi(K);
        for (std::size_t k = 0; k < K; ++k) {
            logpi[k] = (pi_flat[k] > 0.0) ? std::log(pi_flat[k])
                                          : -std::numeric_limits<double>::infinity();
        }

        // Forward filter: alpha[t, k] in log space.
        std::vector<double> alpha(T * K);

        // t = 0
        for (std::size_t k = 0; k < K; ++k) {
            const double loglik =
                cfg_.emission_logp(0, k, context_);
            alpha[0 * K + k] = logpi[k] + loglik;
        }
        // t = 1 .. T-1
        std::vector<double> tmp(K);
        for (std::size_t t = 1; t < T; ++t) {
            for (std::size_t k = 0; k < K; ++k) {
                // alpha[t, k] = logsumexp_{j} [ alpha[t-1, j] + logA[j, k] ]
                //               + emission
                for (std::size_t j = 0; j < K; ++j) {
                    tmp[j] = alpha[(t - 1) * K + j] + logA[j * K + k];
                }
                const double lse = logsumexp(tmp.data(), K);
                const double loglik =
                    cfg_.emission_logp(t, k, context_);
                alpha[t * K + k] = lse + loglik;
            }
        }

        // Backward sample.
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        auto sample_cat = [&](const double* log_probs) -> std::size_t {
            // Exponentiate + normalize + sample.
            const double lse = logsumexp(log_probs, K);
            double u = unif(rng);
            double cdf = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                cdf += std::exp(log_probs[k] - lse);
                if (u <= cdf) return k;
            }
            return K - 1;  // safety
        };

        // z_T ~ softmax(alpha[T-1, :])
        z_[T - 1] = static_cast<double>(
            sample_cat(&alpha[(T - 1) * K]));

        // t = T-2 down to 0:
        //   z_t ~ softmax( alpha[t, :] + logA[:, z_{t+1}] )
        std::vector<double> dist(K);
        for (std::size_t t_rev = 0; t_rev + 1 < T; ++t_rev) {
            const std::size_t t = T - 2 - t_rev;
            const std::size_t z_next = static_cast<std::size_t>(z_[t + 1]);
            for (std::size_t k = 0; k < K; ++k) {
                dist[k] = alpha[t * K + k] + logA[k * K + z_next];
            }
            z_[t] = static_cast<double>(sample_cat(dist.data()));
        }

        if (keep_history_) history_buf_.push_back(z_);
    }

    const arma::vec& current() const override { return z_; }

    void set_current(const arma::vec& z_new) override {
        if (z_new.n_elem != cfg_.T) {
            throw std::invalid_argument(
                "hmm_block '" + cfg_.name +
                "': set_current length must equal T");
        }
        for (std::size_t t = 0; t < cfg_.T; ++t) {
            const int s = static_cast<int>(std::round(z_new[t]));
            if (s < 0 || s >= static_cast<int>(cfg_.K)) {
                throw std::invalid_argument(
                    "hmm_block '" + cfg_.name +
                    "': z entries must be in {0..K-1}");
            }
            z_[t] = static_cast<double>(s);
        }
    }

    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim() const noexcept override { return cfg_.T; }

    // Kernel-control freeze BLACKLIST (DESIGN_NOTES Sec.6): latent state
    // sequence z frozen while emission parameters sample yields mismatched
    // conditioning (Baum-Welch forward pass depends on emissions).
    bool supports_freeze() const noexcept override { return false; }
    std::string freeze_not_supported_reason() const override {
        return "freezing hmm_block not supported "
               "(latent state sequence conditioning breaks when emission "
               "parameters sample); model the HMM inside a two-composite "
               "pattern with the emission-freeze at the outer level";
    }

    std::unordered_map<std::string, arma::vec>
    current_named_outputs() const override {
        std::unordered_map<std::string, arma::vec> out;
        out[cfg_.name] = z_;
        return out;
    }

    history_map get_history() const override {
        const std::size_t T = cfg_.T;
        history_map out;
        if (history_buf_.empty()) {
            arma::mat row(1, T);
            for (std::size_t t = 0; t < T; ++t) row(0, t) = z_[t];
            out.emplace(cfg_.name, std::move(row));
            return out;
        }
        const std::size_t n = history_buf_.size();
        arma::mat hist(n, T);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t t = 0; t < T; ++t)
                hist(i, t) = history_buf_[i][t];
        out.emplace(cfg_.name, std::move(hist));
        return out;
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }
    void clear_history() override { history_buf_.clear(); }

private:
    hmm_block_config       cfg_;
    arma::vec              z_;         // length T, entries in {0..K-1}
    block_context          context_;
    std::vector<arma::vec> history_buf_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_HMM_BLOCK_HPP

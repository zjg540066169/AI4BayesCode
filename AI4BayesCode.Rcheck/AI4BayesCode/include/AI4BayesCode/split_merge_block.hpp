/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  split_merge_block.hpp -- Jain-Neal 2004 split-merge MH proposal on a
 *      cluster-allocation vector z, in the TRUNCATED stick-breaking
 *      regime (NOT CRP-marginal).
 *
 *  WHY THIS BLOCK EXISTS
 *  =====================
 *  Per-observation Gibbs (`categorical_gibbs_block`) is correct but
 *  slow at escaping local modes when two clusters could plausibly be
 *  merged or one cluster split into two. Single-i flips can't bridge
 *  to the merged / split state in a single sweep — they take O(n)
 *  flips, each with low individual-flip probability, to migrate a
 *  cluster's mass.
 *
 *  Jain & Neal 2004 propose a split-merge MH step that toggles the
 *  whole partition between merged and split versions in ONE proposal.
 *  Acceptance is computed via Metropolis-Hastings with a "restricted
 *  Gibbs" proposal density (sweep T iterations among the affected
 *  observations to construct the proposed z*). Empirical mixing in
 *  the original paper improves 5-50× on hard fixtures.
 *
 *  TRUNCATED SBP REGIME (this block)
 *  ---------------------------------
 *  Our DPGaussianMixture / PYGaussianMixture / FiniteGaussianMixture
 *  examples keep K_trunc fixed and sample π via stick_breaking_block
 *  or dirichlet_gibbs_block. So:
 *    - K_active varies in {1, ..., K_trunc}.
 *    - π is held fixed during the split-merge MH ratio (split-merge is
 *      a CONDITIONAL update on z given (π, μ, Σ); the next sweep then
 *      updates π, μ, Σ).
 *    - The prior on z under truncated SBP is product-of-categorical
 *      Cat(π) — much simpler than CRP-EPPF.
 *
 *  This contrasts with the CRP-marginal Neal Alg 2 / Jain-Neal context
 *  where π is integrated out and the prior is the CRP/EPPF. Our
 *  derivation handles the SBP form explicitly (see "MH RATIO" below).
 *
 *  ALGORITHM
 *  ---------
 *  Each step():
 *    1. Sample (i, j) uniformly without replacement from {0, ..., N-1}.
 *    2. Let s_i = z[i], s_j = z[j].
 *
 *    SPLIT case (s_i == s_j):
 *       3a. Build S = {k : z[k] == s_i, k != i, k != j} (other obs in
 *           the same cluster).
 *       3b. Pick c_new uniformly from currently EMPTY slots in
 *           {1..K_trunc}. If none, reject the proposal (no available
 *           split target).
 *       3c. Initialise z* = z, then z*[j] = c_new. (z*[i] stays at s_i.)
 *       3d. For k in S: random initial assignment to {s_i, c_new}
 *           (uniform).
 *       3e. Run T iterations of restricted Gibbs: for each k in S,
 *           sample z*[k] from {s_i, c_new} weighted by
 *               π_{z*[k]} * lik(y_k | μ_{z*[k]}, Σ_{z*[k]}).
 *       3f. Compute proposal density q(z* | z) = ∏_k p(z*[k] | ...) at
 *           the FINAL T-th scan (Jain-Neal §3.1).
 *
 *    MERGE case (s_i != s_j):
 *       3a. Build S = {k : z[k] ∈ {s_i, s_j}, k != i, k != j}.
 *       3b. Set c_merge = s_i (target). Set z*[k] = s_i for k ∈ {j} ∪ S.
 *       3c. To compute q(z | z*) for the REVERSE move (split z* back
 *           to z), we need to construct a "launch state" by random
 *           assignment, run T restricted Gibbs iterations, and
 *           accumulate the proposal density at each step. The final
 *           density q(z | z*) is the product of p(z[k] | restricted)
 *           at the T-th iteration (Jain-Neal §3.2).
 *
 *  MH RATIO (truncated SBP, condition on current π / μ / Σ)
 *  --------------------------------------------------------
 *      A_split  =  q(z   | z*) / q(z* | z)
 *                × prior(z*) / prior(z)
 *                × likelihood(y | z*) / likelihood(y | z)
 *
 *      A_merge  =  q(z*  | z)  / q(z  | z*)
 *                × prior(z*) / prior(z)
 *                × likelihood(y | z*) / likelihood(y | z)
 *
 *  prior(z) = ∏_i π_{z[i]}, so
 *      log prior(z*) − log prior(z) = Σ_i [log π_{z*[i]} − log π_{z[i]}]
 *
 *  lik(y | z) = ∏_i N(y_i | μ_{z[i]}, Σ_{z[i]}), so
 *      log lik(z*) − log lik(z) = Σ_i [log N(y_i; μ_{z*[i]}, Σ_{z*[i]})
 *                                       − log N(y_i; μ_{z[i]}, Σ_{z[i]})]
 *
 *  q-density: see Jain-Neal §3.1-3.2 for the precise restricted-Gibbs
 *  form. We accumulate log q at each restricted-Gibbs step.
 *
 *  COVARIANCE FLAVOR
 *  -----------------
 *  At v0 we support TWO emission shapes:
 *    - DIAGONAL Normal-Gamma: cfg.lambda_key non-empty, cfg.sigma_key
 *      empty. y_i | z, μ, λ ~ N(μ_{z_i}, diag(1/λ_{z_i})).
 *    - FULL covariance NIW: cfg.sigma_key non-empty, cfg.lambda_key
 *      empty. y_i | z, μ, Σ ~ N(μ_{z_i}, Σ_{z_i}).
 *
 *  Exactly one of (lambda_key, sigma_key) must be set; constructor
 *  validates.
 *
 *  CHECK #16 INLINE JUSTIFICATION
 *  ------------------------------
 *  This block is a Metropolis-Hastings proposal — NOT a conjugate
 *  Gibbs draw — so Check #15 (parity to a textbook closed form) does
 *  NOT directly apply. Instead we ship a smoke + integration test that
 *  verifies (a) the MH ratio is correctly accepting at near-1 when the
 *  proposed state is identical (no-op proposal), (b) detailed-balance-
 *  style empirical equilibrium when paired with categorical_gibbs (the
 *  composite preserves the joint π/μ/Σ posterior). Reference for the
 *  algorithm: Jain & Neal 2004, "A Split-Merge Markov Chain Monte
 *  Carlo Procedure for the Dirichlet Process Mixture Model".
 *
 *  The block is internal Tier-B; std::*_distribution usage is
 *  whitelisted (Check #17). No hand-written Jacobian (Check #5):
 *  this is a discrete-allocation MH, no continuous transform.
 *================================================================================*/

#ifndef AI4BAYESCODE_SPLIT_MERGE_BLOCK_HPP
#define AI4BAYESCODE_SPLIT_MERGE_BLOCK_HPP

#include "block_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

struct split_merge_block_config {
    /// Block label (used for declare_dependencies / declare_invalidates
    /// in the composite). NOT a shared_data key.
    std::string name;

    /// Number of observations.
    std::size_t N = 0;

    /// Truncation level (number of cluster slots).
    std::size_t K_trunc = 0;

    /// Observation dimension.
    std::size_t d = 0;

    /// Output key — the block writes back into shared_data under this
    /// name. Default "z" matches the convention used by
    /// categorical_gibbs_block.
    std::string z_name = "z";

    /// Input keys.
    std::string y_key = "y";
    std::string pi_key = "pi";
    std::string mu_key = "mu";          // K * d, cluster-major

    /// Cov flavor — exactly one of these must be non-empty:
    std::string lambda_key;             // K * d (precisions, diagonal)
    std::string sigma_key;              // K * d * d (covariance, full)

    /// Restricted-Gibbs scan iterations (Jain-Neal §3.1). Default 5.
    std::size_t n_restricted_gibbs_iters = 5;

    /// Initial z (used only for `current()` before first step).
    arma::vec initial_z;
};

class split_merge_block : public block_sampler {
public:
    explicit split_merge_block(split_merge_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.name.empty())
            throw std::invalid_argument(
                "split_merge_block: name must be non-empty");
        if (cfg_.N < 2)
            throw std::invalid_argument(
                "split_merge_block: N must be >= 2");
        if (cfg_.K_trunc < 2)
            throw std::invalid_argument(
                "split_merge_block: K_trunc must be >= 2");
        if (cfg_.d == 0)
            throw std::invalid_argument(
                "split_merge_block: d must be > 0");
        if (cfg_.z_name.empty() || cfg_.y_key.empty() ||
            cfg_.pi_key.empty() || cfg_.mu_key.empty())
            throw std::invalid_argument(
                "split_merge_block: z_name / y_key / pi_key / mu_key "
                "must all be non-empty");
        const bool has_lambda = !cfg_.lambda_key.empty();
        const bool has_sigma  = !cfg_.sigma_key.empty();
        if (has_lambda == has_sigma) {
            throw std::invalid_argument(
                "split_merge_block: exactly one of lambda_key (diagonal) "
                "or sigma_key (full covariance) must be set");
        }
        is_diagonal_ = has_lambda;
        if (cfg_.initial_z.n_elem != cfg_.N)
            throw std::invalid_argument(
                "split_merge_block: initial_z length must equal N");
        if (cfg_.n_restricted_gibbs_iters < 1)
            cfg_.n_restricted_gibbs_iters = 1;

        z_.set_size(cfg_.N);
        for (std::size_t i = 0; i < cfg_.N; ++i) {
            const long lab = static_cast<long>(std::llround(cfg_.initial_z[i]));
            if (lab < 1 || static_cast<std::size_t>(lab) > cfg_.K_trunc)
                throw std::invalid_argument(
                    "split_merge_block: initial_z[i] out of {1, ..., K_trunc}");
            z_[i] = static_cast<double>(lab);
        }

        n_proposals_ = 0;
        n_accepted_ = 0;
        n_split_proposals_ = 0;
        n_split_accepted_  = 0;
        n_merge_proposals_ = 0;
        n_merge_accepted_  = 0;
    }

    // ---- block_sampler interface ---------------------------------------

    void set_context(const block_context& ctx) override {
        // COPY (per design contract).
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // Read state from context.
        const arma::vec& y_flat = context_.at(cfg_.y_key);
        const arma::vec& pi     = context_.at(cfg_.pi_key);
        const arma::vec& mu     = context_.at(cfg_.mu_key);
        const std::size_t N = cfg_.N;
        const std::size_t K = cfg_.K_trunc;
        const std::size_t d = cfg_.d;

        if (y_flat.n_elem != N * d)
            throw std::runtime_error(
                "split_merge_block: y length mismatch");
        if (pi.n_elem != K)
            throw std::runtime_error(
                "split_merge_block: pi length mismatch");
        if (mu.n_elem != K * d)
            throw std::runtime_error(
                "split_merge_block: mu length mismatch");

        const arma::vec* lam_ptr = nullptr;
        const arma::vec* sig_ptr = nullptr;
        if (is_diagonal_) {
            lam_ptr = &context_.at(cfg_.lambda_key);
            if (lam_ptr->n_elem != K * d)
                throw std::runtime_error(
                    "split_merge_block: lambda length mismatch");
        } else {
            sig_ptr = &context_.at(cfg_.sigma_key);
            if (sig_ptr->n_elem != K * d * d)
                throw std::runtime_error(
                    "split_merge_block: sigma length mismatch");
        }

        // Read current z (mostly from cached z_; could also re-read from ctx).
        // We trust z_ since it's the canonical block state.

        // Sample (i, j) uniformly without replacement.
        std::uniform_int_distribution<std::size_t> uid(0, N - 1);
        std::size_t i = uid(rng);
        std::size_t j;
        do { j = uid(rng); } while (j == i);

        const long lab_i = static_cast<long>(std::llround(z_[i]));
        const long lab_j = static_cast<long>(std::llround(z_[j]));
        const std::size_t s_i = static_cast<std::size_t>(lab_i) - 1;
        const std::size_t s_j = static_cast<std::size_t>(lab_j) - 1;

        n_proposals_ += 1;

        if (s_i == s_j) {
            // ===== SPLIT =====
            n_split_proposals_ += 1;
            // Build S.
            std::vector<std::size_t> S;
            S.reserve(N);
            for (std::size_t k = 0; k < N; ++k) {
                if (k == i || k == j) continue;
                if (static_cast<std::size_t>(std::llround(z_[k])) - 1 == s_i)
                    S.push_back(k);
            }

            // Pick c_new uniformly from currently EMPTY slots.
            std::vector<std::size_t> empty_slots;
            std::vector<std::size_t> n_per_cluster(K, 0);
            for (std::size_t k = 0; k < N; ++k) {
                ++n_per_cluster[
                    static_cast<std::size_t>(std::llround(z_[k])) - 1];
            }
            for (std::size_t k = 0; k < K; ++k) {
                if (n_per_cluster[k] == 0) empty_slots.push_back(k);
            }
            if (empty_slots.empty()) {
                // No room to split; reject.
                if (keep_history_) history_buf_.push_back(z_);
                return;
            }
            std::uniform_int_distribution<std::size_t> ues(
                0, empty_slots.size() - 1);
            const std::size_t c_new = empty_slots[ues(rng)];

            // Build proposed z* and accumulate log q(z* | z) over T
            // restricted-Gibbs iterations.
            arma::vec z_star = z_;
            z_star[j] = static_cast<double>(c_new + 1);
            // Initialise S members uniformly at random in {s_i, c_new}.
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            for (std::size_t kk : S) {
                z_star[kk] = (uniform(rng) < 0.5)
                    ? static_cast<double>(s_i + 1)
                    : static_cast<double>(c_new + 1);
            }

            double log_q_forward = 0.0;
            for (std::size_t t = 0; t < cfg_.n_restricted_gibbs_iters; ++t) {
                const bool accumulate = (t + 1 == cfg_.n_restricted_gibbs_iters);
                double log_q_this = 0.0;
                for (std::size_t kk : S) {
                    const double y_loglik_si = y_log_lik_at_(
                        y_flat, kk, s_i, mu, lam_ptr, sig_ptr);
                    const double y_loglik_cn = y_log_lik_at_(
                        y_flat, kk, c_new, mu, lam_ptr, sig_ptr);
                    const double a = std::log(pi[s_i]   + 1e-300) + y_loglik_si;
                    const double b = std::log(pi[c_new] + 1e-300) + y_loglik_cn;
                    const double m = std::max(a, b);
                    const double e_a = std::exp(a - m);
                    const double e_b = std::exp(b - m);
                    const double p_si = e_a / (e_a + e_b);
                    const double u    = uniform(rng);
                    const std::size_t pick =
                        (u < p_si) ? s_i : c_new;
                    z_star[kk] = static_cast<double>(pick + 1);
                    if (accumulate) {
                        log_q_this += (pick == s_i) ?
                            std::log(p_si + 1e-300) :
                            std::log(1.0 - p_si + 1e-300);
                    }
                }
                if (accumulate) log_q_forward = log_q_this;
            }

            // For the REVERSE move (merge z* → z), q(z | z*) probability
            // is 1: there is exactly one merged state given (i, j, S),
            // namely z. So log q(z | z*) = 0.
            const double log_q_reverse = 0.0;

            // Compute log prior + log lik ratios.
            double log_prior_ratio = 0.0;
            double log_lik_ratio = 0.0;
            for (std::size_t kk = 0; kk < N; ++kk) {
                const std::size_t z_old =
                    static_cast<std::size_t>(std::llround(z_[kk])) - 1;
                const std::size_t z_new =
                    static_cast<std::size_t>(std::llround(z_star[kk])) - 1;
                if (z_old == z_new) continue;
                log_prior_ratio +=
                    std::log(pi[z_new] + 1e-300)
                    - std::log(pi[z_old] + 1e-300);
                log_lik_ratio +=
                    y_log_lik_at_(y_flat, kk, z_new, mu, lam_ptr, sig_ptr)
                    - y_log_lik_at_(y_flat, kk, z_old, mu, lam_ptr, sig_ptr);
            }

            // Acceptance: A = q(z|z*) / q(z*|z) * prior_ratio * lik_ratio
            //   In log: log A = log_q_reverse - log_q_forward
            //                   + log_prior_ratio + log_lik_ratio
            const double log_A = log_q_reverse - log_q_forward
                                + log_prior_ratio + log_lik_ratio;
            const double u_acc = uniform(rng);
            if (std::log(u_acc + 1e-300) < log_A) {
                z_ = z_star;
                n_accepted_ += 1;
                n_split_accepted_ += 1;
            }
        } else {
            // ===== MERGE =====
            n_merge_proposals_ += 1;
            // Build S.
            std::vector<std::size_t> S;
            S.reserve(N);
            for (std::size_t k = 0; k < N; ++k) {
                if (k == i || k == j) continue;
                const std::size_t z_k =
                    static_cast<std::size_t>(std::llround(z_[k])) - 1;
                if (z_k == s_i || z_k == s_j) S.push_back(k);
            }

            // Build z*: merge to s_i.
            arma::vec z_star = z_;
            z_star[j] = static_cast<double>(s_i + 1);
            for (std::size_t kk : S)
                z_star[kk] = static_cast<double>(s_i + 1);

            // For the FORWARD move (merge z → z*), q(z* | z) = 1.
            const double log_q_forward = 0.0;

            // For the REVERSE move (split z* back to z), we construct
            // the same restricted-Gibbs scan as the SPLIT case would
            // produce, BUT with the FINAL state forced to match z (the
            // current state). Accumulate log q at the T-th step.
            //
            // Per Jain-Neal §3.2, this requires a LAUNCH state (random
            // initial assignment of S to {s_i, s_j}) and T restricted-
            // Gibbs iterations. We accumulate q(z | z*) at the final
            // iteration: it is the conditional probability of selecting
            // each k's CURRENT z[k] under the restricted distribution.
            //
            // Note we are computing the proposal probability of the
            // REVERSE move at the FINAL state matching z, not actually
            // sampling from the restricted Gibbs (the destination is
            // fixed to z).
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            arma::vec z_aux = z_star;  // start from merged state
            // Random initial assignment to {s_i, s_j} for S members.
            for (std::size_t kk : S) {
                z_aux[kk] = (uniform(rng) < 0.5)
                    ? static_cast<double>(s_i + 1)
                    : static_cast<double>(s_j + 1);
            }
            // Run T-1 restricted Gibbs scans (no accumulation), then
            // compute q at the T-th scan with destination = z[kk].
            for (std::size_t t = 0; t < cfg_.n_restricted_gibbs_iters; ++t) {
                const bool accumulate = (t + 1 == cfg_.n_restricted_gibbs_iters);
                double log_q_this = 0.0;
                for (std::size_t kk : S) {
                    const double y_loglik_si = y_log_lik_at_(
                        y_flat, kk, s_i, mu, lam_ptr, sig_ptr);
                    const double y_loglik_sj = y_log_lik_at_(
                        y_flat, kk, s_j, mu, lam_ptr, sig_ptr);
                    const double a = std::log(pi[s_i] + 1e-300) + y_loglik_si;
                    const double b = std::log(pi[s_j] + 1e-300) + y_loglik_sj;
                    const double m = std::max(a, b);
                    const double e_a = std::exp(a - m);
                    const double e_b = std::exp(b - m);
                    const double p_si = e_a / (e_a + e_b);
                    if (accumulate) {
                        // q(z | z*): probability of selecting z[kk] at this step.
                        const std::size_t target =
                            static_cast<std::size_t>(std::llround(z_[kk])) - 1;
                        if (target == s_i)
                            log_q_this += std::log(p_si + 1e-300);
                        else
                            log_q_this += std::log(1.0 - p_si + 1e-300);
                        z_aux[kk] = z_[kk];  // pin to z for q computation
                    } else {
                        // Sample for the next iteration's conditioning.
                        const double u = uniform(rng);
                        z_aux[kk] = (u < p_si)
                            ? static_cast<double>(s_i + 1)
                            : static_cast<double>(s_j + 1);
                    }
                }
                if (accumulate) {
                    // Track total reverse log q.
                    // (We finalize log_q_reverse below; this sets it.)
                    log_q_reverse_ = log_q_this;
                }
            }
            const double log_q_reverse = log_q_reverse_;

            // Compute log prior + log lik ratios.
            double log_prior_ratio = 0.0;
            double log_lik_ratio = 0.0;
            for (std::size_t kk = 0; kk < N; ++kk) {
                const std::size_t z_old =
                    static_cast<std::size_t>(std::llround(z_[kk])) - 1;
                const std::size_t z_new =
                    static_cast<std::size_t>(std::llround(z_star[kk])) - 1;
                if (z_old == z_new) continue;
                log_prior_ratio +=
                    std::log(pi[z_new] + 1e-300)
                    - std::log(pi[z_old] + 1e-300);
                log_lik_ratio +=
                    y_log_lik_at_(y_flat, kk, z_new, mu, lam_ptr, sig_ptr)
                    - y_log_lik_at_(y_flat, kk, z_old, mu, lam_ptr, sig_ptr);
            }

            const double log_A = log_q_reverse - log_q_forward
                                + log_prior_ratio + log_lik_ratio;
            const double u_acc = uniform(rng);
            if (std::log(u_acc + 1e-300) < log_A) {
                z_ = z_star;
                n_accepted_ += 1;
                n_merge_accepted_ += 1;
            }
        }

        if (keep_history_) history_buf_.push_back(z_);
    }

    const arma::vec& current() const override { return z_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != cfg_.N)
            throw std::invalid_argument(
                "split_merge_block::set_current: length mismatch");
        for (std::size_t i = 0; i < cfg_.N; ++i) {
            const long lab = static_cast<long>(std::llround(theta[i]));
            if (lab < 1 || static_cast<std::size_t>(lab) > cfg_.K_trunc)
                throw std::invalid_argument(
                    "split_merge_block::set_current: z[i] out of {1, ..., K_trunc}");
            z_[i] = static_cast<double>(lab);
        }
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return cfg_.N; }

    state_map current_named_outputs() const override {
        return {{cfg_.z_name, z_}};
    }

    history_map get_history() const override {
        return detail::make_history_map(cfg_.z_name, history_buf_, z_);
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override { history_buf_.clear(); }

    // Public introspection of acceptance rates.
    std::size_t n_proposals()       const noexcept { return n_proposals_; }
    std::size_t n_accepted()        const noexcept { return n_accepted_; }
    std::size_t n_split_proposals() const noexcept { return n_split_proposals_; }
    std::size_t n_split_accepted()  const noexcept { return n_split_accepted_; }
    std::size_t n_merge_proposals() const noexcept { return n_merge_proposals_; }
    std::size_t n_merge_accepted()  const noexcept { return n_merge_accepted_; }

private:
    /// Log-density log N(y_i | mu_k, Sigma_k).  For diagonal: sum of
    /// independent Normals; for full: standard multivariate Normal via
    /// chol(Sigma_k).
    double y_log_lik_at_(const arma::vec& y_flat,
                          std::size_t i, std::size_t k,
                          const arma::vec& mu,
                          const arma::vec* lambda,
                          const arma::vec* sigma) const {
        constexpr double kLog2Pi = 1.83787706640934548356065947281;
        const std::size_t d = cfg_.d;
        if (lambda) {
            // Diagonal Normal-Gamma form: lambda is precision per dim.
            double lp = 0.0;
            for (std::size_t j = 0; j < d; ++j) {
                const double dev = y_flat[i * d + j] - mu[k * d + j];
                const double lam_kj = (*lambda)[k * d + j];
                lp += 0.5 * std::log(lam_kj)
                    - 0.5 * kLog2Pi
                    - 0.5 * lam_kj * dev * dev;
            }
            return lp;
        } else {
            // Full covariance — load Sigma_k into d×d matrix, chol, evaluate.
            arma::mat S(d, d);
            for (std::size_t a = 0; a < d; ++a)
                for (std::size_t b = 0; b < d; ++b)
                    S(a, b) = (*sigma)[k * d * d + a * d + b];
            arma::mat L;
            if (!arma::chol(L, S, "lower")) {
                S.diag() += 1e-8;
                if (!arma::chol(L, S, "lower")) {
                    return -std::numeric_limits<double>::infinity();
                }
            }
            arma::vec dev(d);
            for (std::size_t j = 0; j < d; ++j)
                dev[j] = y_flat[i * d + j] - mu[k * d + j];
            // log det(Sigma) = 2 sum log(L_ii)
            double log_det = 0.0;
            for (std::size_t a = 0; a < d; ++a)
                log_det += 2.0 * std::log(L(a, a));
            // M dist = dev' Sigma^-1 dev = ||u||^2 where u = solve(L, dev)
            arma::vec u = arma::solve(arma::trimatl(L), dev);
            const double maha = arma::dot(u, u);
            return -0.5 * static_cast<double>(d) * kLog2Pi
                 - 0.5 * log_det - 0.5 * maha;
        }
    }

    split_merge_block_config cfg_;
    bool                     is_diagonal_ = true;
    arma::vec                z_;
    block_context            context_;
    std::vector<arma::vec>   history_buf_;

    // Acceptance bookkeeping.
    std::size_t n_proposals_       = 0;
    std::size_t n_accepted_        = 0;
    std::size_t n_split_proposals_ = 0;
    std::size_t n_split_accepted_  = 0;
    std::size_t n_merge_proposals_ = 0;
    std::size_t n_merge_accepted_  = 0;

    // Cache for merge log_q (avoids cross-function variable sharing).
    double log_q_reverse_ = 0.0;
};

}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_SPLIT_MERGE_BLOCK_HPP

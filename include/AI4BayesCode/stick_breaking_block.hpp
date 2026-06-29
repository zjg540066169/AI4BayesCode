/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  stick_breaking_block.hpp -- a closed-form Gibbs leaf that samples a
 *                              truncated stick-breaking-process (SBP)
 *                              simplex pi of length K_trunc by drawing
 *                              independent Beta sticks V_k.
 *
 *  WHY THIS BLOCK EXISTS
 *  =====================
 *  Bayesian-nonparametric mixture models (Dirichlet Process, Pitman-Yor,
 *  hierarchical-mass-parameter variants) all express the mixing measure
 *  as a stick-breaking sequence of Beta sticks. Under the Ishwaran &
 *  James (2001) truncation
 *
 *      V_1, ..., V_{K_trunc-1}  ~  independent Beta(a_k, b_k)
 *      V_{K_trunc}              =  1                       (forced)
 *      pi_k = V_k * prod_{j<k} (1 - V_j),    k = 1, ..., K_trunc
 *
 *  the conditional p(pi | counts, alpha, discount, ...) is exactly a
 *  product of Beta densities given the stick parameters a_k, b_k.
 *
 *  For the Dirichlet Process (Sethuraman 1994):
 *      a_k = 1 + n_k
 *      b_k = alpha + sum_{j>k} n_j
 *
 *  For the Pitman-Yor process (Pitman & Yor 1997):
 *      a_k = 1 + n_k - discount
 *      b_k = alpha + k * discount + sum_{j>k} n_j
 *
 *  The user supplies these formulas as functor closures; the block
 *  knows nothing about which process it's implementing. This is the
 *  same flexibility pattern as `dirichlet_gibbs_block::alpha_post_fn`
 *  and `categorical_gibbs_block::log_probs_fn` -- the library does the
 *  sampling mechanics, the user supplies the conditional.
 *
 *  RELATIONSHIP TO dirichlet_gibbs_block
 *  -------------------------------------
 *  Use `dirichlet_gibbs_block` when the conditional on the simplex is
 *  EXACTLY a single Dirichlet distribution (Dirichlet-Categorical /
 *  Dirichlet-Multinomial / LDA proportions). Use this block when the
 *  conditional is a sequence of independent Betas with a stick-breaking
 *  product (truncated DP / PY / HDP / kernel-stick-breaking, ...).
 *
 *  The two are NOT interchangeable: a stick-breaking conditional with
 *  DP weights does NOT equal a Dirichlet(1+n_1, ..., 1+n_K) — the
 *  off-diagonal correlation structure differs.
 *
 *  STORAGE / OUTPUT
 *  ----------------
 *  The block writes ONE shared_data key (cfg.name) holding the K_trunc-
 *  length simplex pi. It also writes the K_trunc-length stick fractions
 *  V (cfg.v_name, optional; if empty the V vector is not exposed).
 *
 *  pi[k] is strictly between (1e-300, 1) and sum(pi) == 1 up to
 *  floating-point. Tiny tail entries are EXPECTED for K_trunc much
 *  larger than the populated number of clusters; do NOT treat them as
 *  an R-hat artifact unless your downstream code is sensitive.
 *
 *  NUMERICAL NOTES
 *  ---------------
 *  Beta draws are produced via the standard gamma-trick:
 *      X ~ Gamma(a_k, 1), Y ~ Gamma(b_k, 1)   ==>   V_k = X / (X + Y) ~ Beta(a_k, b_k)
 *  This is numerically stable for any a_k, b_k > 0; the implementation
 *  detects underflow (X = Y = 0) and resamples up to a small bound, then
 *  throws if it persists.
 *
 *  COMMENT-PARAMETERIZATION DISCIPLINE (rcpp_api.md §11)
 *  -----------------------------------------------------
 *  std::gamma_distribution<double>(shape, scale) uses shape-SCALE, where
 *  E[X] = shape * scale. We construct it with scale = 1 throughout, so
 *  X ~ Gamma(shape=a_k, scale=1)  =>  E[X] = a_k.  This matches the
 *  conventional rate=1 parameterisation used in the gamma-trick formula
 *  (R's rgamma(n, shape, rate=1) is the same draw).
 *================================================================================*/

#ifndef AI4BAYESCODE_STICK_BREAKING_BLOCK_HPP
#define AI4BAYESCODE_STICK_BREAKING_BLOCK_HPP

#include "block_sampler.hpp"

#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/**
 * @brief Signature of the per-stick Beta-parameter oracle consumed by
 *        stick_breaking_block.
 *
 * Given the 0-indexed stick number k in {0, ..., K_trunc-2}, the
 * complete cluster_counts vector (length K_trunc) and the current
 * block_context, return the Beta(a_k, b_k) parameter for stick V_k.
 *
 * The block calls each function K_trunc-1 times per `step()` (the last
 * stick V_{K_trunc-1} is forced to 1 per the Ishwaran-James truncation).
 * Both functions must be pure and return strictly positive values.
 */
using stick_breaking_param_fn =
    std::function<double(std::size_t k, const arma::vec& counts,
                         const block_context& ctx)>;

/// Configuration bundle for stick_breaking_block construction.
struct stick_breaking_block_config {
    /// Unique name for this block within its composite; also the key
    /// under which shared_data_t stores the length-K_trunc simplex pi.
    std::string name;

    /// Truncation level. The block samples K_trunc-1 sticks via the
    /// supplied a_fn / b_fn; the last stick is forced to 1, producing
    /// a length-K_trunc simplex.
    std::size_t K_trunc = 0;

    /// Shared_data key under which cluster counts are stored. The
    /// block reads ctx.at(counts_key) on every step. Counts must be a
    /// length-K_trunc vector of non-negative reals (stored as
    /// arma::vec for shared_data compatibility; integral values
    /// expected). Typically this key is registered as a deterministic
    /// refresher of the cluster-assignment block (z) via
    /// `bnp::counts_from_z`.
    std::string counts_key = "cluster_counts";

    /// Optional: shared_data key under which the length-K_trunc
    /// stick-fraction vector V is exposed. If non-empty, the block
    /// declares this as an additional named output. If empty, V is
    /// kept internal.
    std::string v_name;

    /// Beta-parameter functors (see @ref stick_breaking_param_fn).
    /// Required.
    stick_breaking_param_fn a_fn;
    stick_breaking_param_fn b_fn;

    /// Initial value for pi. Must have length K_trunc, sum to 1, and
    /// have all entries > 0 (checked in the constructor; tolerance 1e-8
    /// on the sum).
    arma::vec initial_pi;
};

/**
 * @brief Closed-form Gibbs leaf that samples pi from a truncated
 *        stick-breaking conditional. See the file header for motivation
 *        and usage.
 */
class stick_breaking_block : public block_sampler {
public:
    explicit stick_breaking_block(stick_breaking_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "stick_breaking_block: name must be non-empty");
        }
        if (cfg_.K_trunc < 2) {
            throw std::invalid_argument(
                "stick_breaking_block: K_trunc must be >= 2");
        }
        if (cfg_.counts_key.empty()) {
            throw std::invalid_argument(
                "stick_breaking_block: counts_key must be non-empty");
        }
        if (!cfg_.a_fn || !cfg_.b_fn) {
            throw std::invalid_argument(
                "stick_breaking_block: a_fn and b_fn are both required");
        }
        if (cfg_.initial_pi.n_elem != cfg_.K_trunc) {
            throw std::invalid_argument(
                "stick_breaking_block: initial_pi length must equal K_trunc");
        }
        const double init_sum = arma::sum(cfg_.initial_pi);
        if (std::abs(init_sum - 1.0) > 1e-8) {
            throw std::invalid_argument(
                "stick_breaking_block: initial_pi must sum to 1");
        }
        for (std::size_t k = 0; k < cfg_.K_trunc; ++k) {
            if (!(cfg_.initial_pi[k] > 0.0)) {
                throw std::invalid_argument(
                    "stick_breaking_block: initial_pi entries must be > 0");
            }
        }
        pi_ = cfg_.initial_pi;
        // Initial V values: derive from pi via inverse stick-breaking.
        // V_0 = pi_0
        // V_k = pi_k / prod_{j<k}(1 - V_j)   for k = 1, ..., K_trunc-2
        // V_{K_trunc-1} = 1   (forced)
        v_.set_size(cfg_.K_trunc);
        double remainder = 1.0;
        for (std::size_t k = 0; k + 1 < cfg_.K_trunc; ++k) {
            if (remainder <= 0.0) {
                v_[k] = 0.0;
            } else {
                v_[k] = pi_[k] / remainder;
                if (v_[k] > 1.0) v_[k] = 1.0;
                if (v_[k] < 0.0) v_[k] = 0.0;
            }
            remainder *= (1.0 - v_[k]);
        }
        v_[cfg_.K_trunc - 1] = 1.0;
    }

    // ---- block_sampler interface ---------------------------------------

    void set_context(const block_context& ctx) override {
        // COPY (per design contract).
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // Read counts from context.
        auto it = context_.find(cfg_.counts_key);
        if (it == context_.end()) {
            throw std::runtime_error(
                "stick_breaking_block '" + cfg_.name +
                "': counts_key '" + cfg_.counts_key + "' not in context");
        }
        const arma::vec& counts = it->second;
        if (counts.n_elem != cfg_.K_trunc) {
            throw std::runtime_error(
                "stick_breaking_block '" + cfg_.name +
                "': counts vector length " +
                std::to_string(counts.n_elem) +
                " != K_trunc " +
                std::to_string(cfg_.K_trunc));
        }
        for (std::size_t k = 0; k < cfg_.K_trunc; ++k) {
            if (!(counts[k] >= 0.0) || !std::isfinite(counts[k])) {
                throw std::runtime_error(
                    "stick_breaking_block '" + cfg_.name +
                    "': counts[" + std::to_string(k) +
                    "] is negative or non-finite");
            }
        }

        // Sample V_0, V_1, ..., V_{K_trunc-2}; force V_{K_trunc-1} = 1.
        for (std::size_t k = 0; k + 1 < cfg_.K_trunc; ++k) {
            const double a_k = cfg_.a_fn(k, counts, context_);
            const double b_k = cfg_.b_fn(k, counts, context_);
            if (!(a_k > 0.0) || !(b_k > 0.0) ||
                !std::isfinite(a_k) || !std::isfinite(b_k)) {
                throw std::runtime_error(
                    "stick_breaking_block '" + cfg_.name +
                    "': a_fn / b_fn returned non-positive or non-finite "
                    "value at stick k=" + std::to_string(k));
            }
            v_[k] = sample_beta_(a_k, b_k, rng);
        }
        v_[cfg_.K_trunc - 1] = 1.0;

        // Compute pi_k = V_k * prod_{j<k} (1 - V_j).
        double remainder = 1.0;
        for (std::size_t k = 0; k < cfg_.K_trunc; ++k) {
            pi_[k] = v_[k] * remainder;
            remainder *= (1.0 - v_[k]);
            if (remainder < 0.0) remainder = 0.0;  // floating-point safety
        }
        // Renormalise to guard against accumulated rounding error
        // (typically < 1e-12 across K_trunc=50; renorm makes it exact).
        const double s = arma::sum(pi_);
        if (!(s > 0.0) || !std::isfinite(s)) {
            throw std::runtime_error(
                "stick_breaking_block '" + cfg_.name +
                "': pi sum is zero or non-finite after sampling");
        }
        pi_ /= s;

        // Append to history if enabled.
        if (keep_history_) {
            history_buf_.push_back(pi_);
            if (!cfg_.v_name.empty()) {
                v_history_buf_.push_back(v_);
            }
        }
    }

    const arma::vec& current() const override { return pi_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != cfg_.K_trunc) {
            throw std::invalid_argument(
                "stick_breaking_block::set_current: wrong length");
        }
        const double s = arma::sum(theta);
        if (std::abs(s - 1.0) > 1e-8) {
            throw std::invalid_argument(
                "stick_breaking_block::set_current: pi must sum to 1");
        }
        for (std::size_t k = 0; k < cfg_.K_trunc; ++k) {
            if (!(theta[k] > 0.0)) {
                throw std::invalid_argument(
                    "stick_breaking_block::set_current: pi entries must be > 0");
            }
        }
        pi_ = theta;
        // Recompute V from pi (inverse stick-breaking).
        double remainder = 1.0;
        for (std::size_t k = 0; k + 1 < cfg_.K_trunc; ++k) {
            if (remainder <= 0.0) {
                v_[k] = 0.0;
            } else {
                v_[k] = pi_[k] / remainder;
                if (v_[k] > 1.0) v_[k] = 1.0;
                if (v_[k] < 0.0) v_[k] = 0.0;
            }
            remainder *= (1.0 - v_[k]);
        }
        v_[cfg_.K_trunc - 1] = 1.0;
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return cfg_.K_trunc; }

    /// Expose pi under cfg.name; if cfg.v_name is non-empty also expose V.
    state_map current_named_outputs() const override {
        state_map out;
        out.emplace(cfg_.name, pi_);
        if (!cfg_.v_name.empty()) {
            out.emplace(cfg_.v_name, v_);
        }
        return out;
    }

    // ---- History overrides -----------------------------------------------

    history_map get_history() const override {
        history_map out =
            detail::make_history_map(cfg_.name, history_buf_, pi_);
        if (!cfg_.v_name.empty()) {
            history_map vmap =
                detail::make_history_map(cfg_.v_name, v_history_buf_, v_);
            for (auto& kv : vmap) {
                out.emplace(kv.first, std::move(kv.second));
            }
        }
        return out;
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override {
        history_buf_.clear();
        v_history_buf_.clear();
    }

private:
    /**
     * Beta(a, b) sample via gamma trick.
     *
     * COMMENTED PARAMETERIZATION (rcpp_api.md §11):
     *   X ~ Gamma(shape=a, scale=1) => E[X] = a, Var[X] = a.
     *   Y ~ Gamma(shape=b, scale=1) => E[Y] = b, Var[Y] = b.
     *   V = X / (X + Y) ~ Beta(a, b).
     *
     * On underflow (a or b extremely small => X or Y exactly 0), retry
     * a few times. Throws on persistent failure.
     */
    double sample_beta_(double a, double b, std::mt19937_64& rng) const {
        std::gamma_distribution<double> gam_a(a, 1.0);
        std::gamma_distribution<double> gam_b(b, 1.0);
        for (int retry = 0; retry < 5; ++retry) {
            const double x = gam_a(rng);
            const double y = gam_b(rng);
            const double s = x + y;
            if (s > 0.0 && std::isfinite(s)) {
                double v = x / s;
                // Clamp away from 0 / 1 to avoid log(0) issues downstream.
                if (v < 1e-300) v = 1e-300;
                if (v > 1.0 - 1e-15) v = 1.0 - 1e-15;
                return v;
            }
        }
        throw std::runtime_error(
            "stick_breaking_block: Beta sample underflowed after 5 retries "
            "(try larger prior concentration on the underflowing stick)");
    }

    stick_breaking_block_config cfg_;
    arma::vec                   pi_;
    arma::vec                   v_;
    block_context               context_;
    std::vector<arma::vec>      history_buf_;
    std::vector<arma::vec>      v_history_buf_;
};

}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_STICK_BREAKING_BLOCK_HPP

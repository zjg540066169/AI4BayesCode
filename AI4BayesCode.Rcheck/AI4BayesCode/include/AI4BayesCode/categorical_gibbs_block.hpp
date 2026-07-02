/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  categorical_gibbs_block.hpp -- a closed-form Gibbs leaf for
 *                                 per-observation categorical latents
 *                                 with K > 2 categories. This is the
 *                                 natural K-ary generalization of
 *                                 binary_gibbs_block.
 *
 *  WHY THIS BLOCK EXISTS
 *  =====================
 *  Stan cannot sample discrete parameters directly, so latent
 *  categorical variables (topic assignments in LDA, class indicators
 *  in latent class / regime-switching models, latent treatment states
 *  in epidemiology, discrete-choice latent preferences, ...) either
 *  have to be analytically marginalised (tractable only for very
 *  simple models) or bolted on with a Gibbs step outside the main
 *  HMC sampler.
 *
 *  In block_mcmc this is trivial: pair categorical_gibbs_block with
 *  whatever blocks are sampling the continuous side of the model
 *  (nuts_block, dirichlet_gibbs_block, binary_gibbs_block, etc.)
 *  inside a composite_block, register the right dependencies, and
 *  every Gibbs sweep closes in finite time without any warmup.
 *
 *  DESIGN
 *  ------
 *  The user provides a functor
 *
 *      log_probs_fn(ctx) -> arma::mat of shape (n_obs x n_categories)
 *
 *  whose (i, k) entry is
 *
 *      log [ p(z_i = k | everything else in ctx) ]
 *
 *  up to an additive constant per row (the block normalises via a
 *  stable softmax). Each `step()` draws a new independent categorical
 *  per observation.
 *
 *  LABEL SWITCHING DISCLAIMER
 *  --------------------------
 *  If the user's model has exchangeable components (e.g. a generic
 *  Gaussian mixture where cluster means are unordered), the chain
 *  will suffer from label switching in the classical sense, and
 *  per-component posterior summaries will be unreliable unless the
 *  user imposes an identifiability constraint (ordered mu, mass
 *  constraint on pi, anchoring point, etc.). That is a MODEL-level
 *  problem, not a block-level problem: categorical_gibbs_block
 *  samples correctly from the conditional that its log_probs_fn
 *  describes. If you are using it inside a mixture model, make sure
 *  your model is identifiable.
 *
 *  STORAGE
 *  -------
 *  Labels are stored as doubles in {1, 2, ..., K} (1-indexed, to
 *  match R and Stan conventions). The dim() of this block equals
 *  n_obs -- one label per observation.
 *================================================================================*/

#ifndef AI4BAYESCODE_CATEGORICAL_GIBBS_BLOCK_HPP
#define AI4BAYESCODE_CATEGORICAL_GIBBS_BLOCK_HPP

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
 * @brief Signature of the log-probability oracle consumed by
 *        categorical_gibbs_block.
 *
 * Given the current block_context, return an (n_obs x n_categories)
 * matrix where entry (i, k) is the unnormalised log-probability that
 * observation i belongs to category k, given everything in ctx.
 *
 * The block normalises each row via a numerically-stable softmax, so
 * additive per-row constants are harmless.
 */
using categorical_log_probs_fn =
    std::function<arma::mat(const block_context& ctx)>;

/// Configuration bundle for categorical_gibbs_block construction.
struct categorical_gibbs_block_config {
    /// Unique name for this block within its composite; also the key
    /// under which shared_data_t stores this block's label vector.
    std::string name;

    /// Number of observations (length of the label vector produced
    /// by `current()`).
    std::size_t n_obs = 0;

    /// Number of categories; labels live in {1, ..., n_categories}.
    std::size_t n_categories = 0;

    /// Initial labels. Must have length n_obs and every entry must
    /// round to an integer in {1, ..., n_categories}; non-integer
    /// values are snapped to the nearest valid label.
    arma::vec initial_labels;

    /// The log-probabilities oracle (see @ref categorical_log_probs_fn).
    /// Required.
    categorical_log_probs_fn log_probs_fn;
};

/**
 * @brief Closed-form Gibbs leaf block for per-observation categorical
 *        latents. See the file header for motivation and usage.
 */
class categorical_gibbs_block : public block_sampler {
public:
    explicit categorical_gibbs_block(categorical_gibbs_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.n_obs == 0) {
            throw std::invalid_argument(
                "categorical_gibbs_block: n_obs must be > 0");
        }
        if (cfg_.n_categories < 2) {
            throw std::invalid_argument(
                "categorical_gibbs_block: n_categories must be >= 2 "
                "(use binary_gibbs_block for K = 2 if you prefer the "
                "log-odds interface)");
        }
        if (cfg_.initial_labels.n_elem != cfg_.n_obs) {
            throw std::invalid_argument(
                "categorical_gibbs_block: initial_labels length must "
                "equal n_obs");
        }
        if (!cfg_.log_probs_fn) {
            throw std::invalid_argument(
                "categorical_gibbs_block: log_probs_fn is required");
        }
        values_.set_size(cfg_.n_obs);
        for (std::size_t i = 0; i < cfg_.n_obs; ++i) {
            values_[i] = snap_label_(cfg_.initial_labels[i]);
        }
    }

    // ---- block_sampler interface ---------------------------------------

    void set_context(const block_context& ctx) override {
        // COPY. See the design contract in block_sampler.hpp.
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // SEQUENTIAL update: sample z_i, write back to context, then
        // recompute log_probs for z_{i+1} from updated context.
        // This is correct component-wise Gibbs. Parallel update (compute
        // all log_probs once, then sample all z simultaneously) converges
        // to a DIFFERENT stationary distribution when z_i's are not
        // conditionally independent given the rest (e.g. mixture labels
        // with component-count-dependent priors).
        std::vector<double> probs(cfg_.n_categories);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        for (std::size_t i = 0; i < cfg_.n_obs; ++i) {
            // Recompute log_probs from current context (reflects all
            // previously sampled z's in this sweep).
            const arma::mat log_probs = cfg_.log_probs_fn(context_);
            if (log_probs.n_rows != cfg_.n_obs ||
                log_probs.n_cols != cfg_.n_categories) {
                throw std::runtime_error(
                    "categorical_gibbs_block '" + cfg_.name
                    + "': log_probs_fn returned wrong shape");
            }

            // Numerically stable softmax for row i.
            double max_log = -std::numeric_limits<double>::infinity();
            for (std::size_t k = 0; k < cfg_.n_categories; ++k) {
                if (log_probs(i, k) > max_log) max_log = log_probs(i, k);
            }
            if (!std::isfinite(max_log)) {
                throw std::runtime_error(
                    "categorical_gibbs_block '" + cfg_.name
                    + "': row " + std::to_string(i)
                    + " of log_probs is all -inf or NaN");
            }

            double total = 0.0;
            for (std::size_t k = 0; k < cfg_.n_categories; ++k) {
                const double w = std::exp(log_probs(i, k) - max_log);
                probs[k] = w;
                total   += w;
            }
            if (!(total > 0.0) || !std::isfinite(total)) {
                throw std::runtime_error(
                    "categorical_gibbs_block '" + cfg_.name
                    + "': row " + std::to_string(i)
                    + " softmax produced zero / non-finite total");
            }

            // Inverse CDF draw.
            const double u = uniform(rng) * total;
            double cumul = 0.0;
            std::size_t chosen = cfg_.n_categories - 1;
            for (std::size_t k = 0; k < cfg_.n_categories; ++k) {
                cumul += probs[k];
                if (u < cumul) { chosen = k; break; }
            }
            values_[i] = static_cast<double>(chosen + 1); // 1-indexed

            // Write updated label back into context so the next
            // observation sees the freshly sampled value.
            context_[cfg_.name] = values_;
        }

        // Append to history if enabled
        if (keep_history_) {
            history_buf_.push_back(values_);
        }
    }

    const arma::vec& current() const override { return values_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != cfg_.n_obs) {
            throw std::invalid_argument(
                "categorical_gibbs_block::set_current: wrong length");
        }
        for (std::size_t i = 0; i < cfg_.n_obs; ++i) {
            values_[i] = snap_label_(theta[i]);
        }
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return cfg_.n_obs; }

    // ---- History overrides -----------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_buf_, values_);
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override {
        history_buf_.clear();
    }

private:
    double snap_label_(double raw) const {
        // Round to nearest integer, clamp into {1, ..., K}.
        long v = static_cast<long>(std::llround(raw));
        if (v < 1) v = 1;
        if (static_cast<std::size_t>(v) > cfg_.n_categories) {
            v = static_cast<long>(cfg_.n_categories);
        }
        return static_cast<double>(v);
    }

    categorical_gibbs_block_config cfg_;
    arma::vec                      values_;
    block_context                  context_;
    std::vector<arma::vec>         history_buf_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_CATEGORICAL_GIBBS_BLOCK_HPP

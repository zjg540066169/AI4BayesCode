/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  binary_gibbs_block.hpp -- a closed-form Gibbs leaf for vector-valued
 *                            binary (0/1) parameters, typically the
 *                            inclusion indicators of a spike-and-slab
 *                            model or any other latent Bernoulli.
 *
 *  WHY THIS BLOCK EXISTS
 *  =====================
 *  Stan cannot sample discrete parameters directly; the recommended
 *  workflow for spike-and-slab is either:
 *    (a) analytic marginalisation of the indicators (only tractable for
 *        simple, conjugate likelihoods), or
 *    (b) continuous relaxation (horseshoe, regularised horseshoe, smooth
 *        spike) which changes the model.
 *  Neither option gives you the classical point-mass spike-and-slab
 *  sampler that most Bayesian variable-selection papers actually use.
 *
 *  binary_gibbs_block fills that gap. The conditional distribution of a
 *  binary vector z given everything else is a product of Bernoullis
 *  whose log-odds are a deterministic function of the other blocks'
 *  current values; a single `step()` draws the entire vector in closed
 *  form without any MCMC machinery.
 *
 *  DESIGN
 *  ======
 *  - No gradient, no NUTS, no mcmclib: this block doesn't need any of
 *    them. It sits alongside nuts_block inside a composite_block and
 *    satisfies the same four-method block_sampler interface.
 *  - The per-binary log-odds are computed by a user-provided functor
 *    `log_odds_fn(ctx)` that reads whatever it needs from the context
 *    installed by `set_context`. This is the only piece of
 *    mathematical content the AI has to write, and it is a pure
 *    element-wise function from "current state" to "log p(z_i = 1 |
 *    everything else)" for i = 0..n_binary-1.
 *  - Values are stored as doubles 0.0 / 1.0 so they fit the block_sampler
 *    interface (arma::vec) without introducing a separate integer type.
 *
 *  NUMERICAL STABILITY
 *  -------------------
 *  We never compute exp(log_odds) explicitly. Instead, we sample
 *  directly from Bernoulli with p = 1 / (1 + exp(-log_odds)), which
 *  saturates cleanly to 0 or 1 for |log_odds| >~ 40 without overflow.
 *
 *  WHAT THE AI WRITES IN A SPIKE-AND-SLAB CONTEXT
 *  ----------------------------------------------
 *  For SSVS with continuous beta_j, pi, tau, and spike_sd:
 *
 *      log_odds[j] = log(pi / (1 - pi))
 *                  + log(spike_sd / tau)
 *                  + 0.5 * beta_j^2 * (1/spike_sd^2 - 1/tau^2)
 *
 *  The lambda reads beta, pi, tau, spike_sd from ctx and returns these
 *  values as an arma::vec. That is the whole model-specific content.
 *================================================================================*/

#ifndef AI4BAYESCODE_BINARY_GIBBS_BLOCK_HPP
#define AI4BAYESCODE_BINARY_GIBBS_BLOCK_HPP

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
 * @brief Signature of the log-odds oracle consumed by binary_gibbs_block.
 *
 * Given the current block_context, return a vector of log-odds
 *
 *     log_odds[i] = log [ P(z_i = 1 | others) / P(z_i = 0 | others) ]
 *
 * The returned vector must have length `n_binary` (step() checks this
 * on every call). The function must be pure (no hidden global state).
 * If the context installed by set_context has no entry under the block's
 * name, it is called once per `step()`; otherwise once per component,
 * each time after the previous component has been written into that
 * entry.
 *
 * An entry under the block's name is the starting state of the step: its
 * values, snapped to 0 / 1 as in set_current, replace the block's values
 * before the first oracle call, so a value written there by another block
 * that writes the same key is where the sweep starts. An entry whose length
 * is not n_binary makes step() throw without changing the block's state.
 */
using binary_log_odds_fn =
    std::function<arma::vec(const block_context& ctx)>;

/**
 * @brief Configuration bundle for binary_gibbs_block construction.
 */
struct binary_gibbs_block_config {
    /// Unique name for this block within its composite; also the key
    /// under which shared_data_t stores this block's current vector.
    std::string name;

    /// Number of binary variables to sample at each step().
    std::size_t n_binary = 0;

    /// Initial values. Must have length `n_binary`. Each entry should
    /// be exactly 0.0 or 1.0; anything else is rounded to the nearest.
    arma::vec initial_values;

    /// The log-odds oracle (see @ref binary_log_odds_fn). Required.
    binary_log_odds_fn log_odds_fn;
};

/**
 * @brief Closed-form Gibbs leaf block for vector-valued binary
 *        parameters. See the file header for motivation and usage.
 */
class binary_gibbs_block : public block_sampler {
public:
    explicit binary_gibbs_block(binary_gibbs_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.n_binary == 0) {
            throw std::invalid_argument(
                "binary_gibbs_block: n_binary must be > 0");
        }
        if (cfg_.initial_values.n_elem != cfg_.n_binary) {
            throw std::invalid_argument(
                "binary_gibbs_block: initial_values length must equal "
                "n_binary");
        }
        if (!cfg_.log_odds_fn) {
            throw std::invalid_argument(
                "binary_gibbs_block: log_odds_fn is required");
        }
        // Snap any non-0/1 initial values.
        values_.set_size(cfg_.n_binary);
        for (std::size_t i = 0; i < cfg_.n_binary; ++i) {
            values_[i] = (cfg_.initial_values[i] >= 0.5) ? 1.0 : 0.0;
        }
    }

    // ---- block_sampler interface ---------------------------------------

    void set_context(const block_context& ctx) override {
        // COPY. See the design contract in block_sampler.hpp.
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // SEQUENTIAL update: flip gamma_j, write back to context, then
        // compute the next log-odds from the updated context. This is
        // correct component-wise Gibbs. Parallel update (computing all
        // log-odds at once, then flipping all at once) converges to a
        // DIFFERENT stationary distribution when gamma_j's are not
        // conditionally independent given the rest.
        //
        // If the context has no entry under this block's name, the oracle
        // input is the same for every component, so the log-odds are
        // computed once and each component is drawn from them with the
        // same uniforms, in the same order, as in the sequential scan.
        //
        // If it has one, that entry is the shared value of these indicators,
        // which another block writing the same key may have changed since
        // this block's last step, and the sweep starts from it.
        const auto own_it = context_.find(cfg_.name);
        const bool sequential = own_it != context_.end();
        if (sequential) load_own_entry_(own_it->second);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        arma::vec log_odds;
        for (std::size_t i = 0; i < cfg_.n_binary; ++i) {
            // Recompute log-odds from current context (which reflects
            // all previously flipped gammas in this sweep).
            if (i == 0 || sequential) {
                log_odds = cfg_.log_odds_fn(context_);
                if (log_odds.n_elem != cfg_.n_binary) {
                    throw std::runtime_error(
                        "binary_gibbs_block '" + cfg_.name
                        + "': log_odds_fn returned wrong length");
                }
            }

            // Numerically stable sigmoid.
            const double lo = log_odds[i];
            double p1;
            if (lo >= 0.0) {
                const double e = std::exp(-lo);
                p1 = 1.0 / (1.0 + e);
            } else {
                const double e = std::exp(lo);
                p1 = e / (1.0 + e);
            }
            values_[i] = (uniform(rng) < p1) ? 1.0 : 0.0;

            // Write updated gamma back into context, so that the log-odds
            // of the next component are computed from it. The entry equals
            // values_ at every other component since the load above.
            if (sequential) own_it->second[i] = values_[i];
        }

        // Append to history if enabled
        if (keep_history_) {
            history_buf_.push_back(values_);
        }
    }

    const arma::vec& current() const override { return values_; }


    // FORK MARKER (2026-07-27, JZ) [freeze + predict_at history alignment]:
    // record the HELD value when this block is whole-block frozen, so its
    // history stays the same length as unfrozen siblings (composite_block
    // calls this on a frozen child). See block_sampler::record_held_history.
    void record_held_history() override {
        if (keep_history_) history_buf_.push_back(current());
    }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != cfg_.n_binary) {
            throw std::invalid_argument(
                "binary_gibbs_block::set_current: wrong length");
        }
        for (std::size_t i = 0; i < cfg_.n_binary; ++i) {
            values_[i] = (theta[i] >= 0.5) ? 1.0 : 0.0;
        }
        // A context entry under this block's name is the oracle input for
        // these values at the next step(), so it is set to the loaded ones.
        auto it = context_.find(cfg_.name);
        if (it != context_.end()) it->second = values_;
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return cfg_.n_binary; }

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
    /// Starting state of a step whose context holds the block's own key:
    /// the values in that entry, snapped to 0 / 1 as in set_current, replace
    /// values_, and the snapped values are written back into the entry. A
    /// length other than n_binary throws before anything is written.
    void load_own_entry_(arma::vec& own) {
        if (own.n_elem != cfg_.n_binary) {
            throw std::runtime_error(
                "binary_gibbs_block '" + cfg_.name + "': context key '"
                + cfg_.name + "' (the block's own values) has length "
                + std::to_string(own.n_elem) + ", expected n_binary = "
                + std::to_string(cfg_.n_binary));
        }
        for (std::size_t i = 0; i < cfg_.n_binary; ++i) {
            values_[i] = (own[i] >= 0.5) ? 1.0 : 0.0;
            own[i] = values_[i];
        }
    }

    binary_gibbs_block_config cfg_;
    arma::vec                 values_;
    block_context             context_;
    std::vector<arma::vec>    history_buf_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_BINARY_GIBBS_BLOCK_HPP

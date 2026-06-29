/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  dirichlet_gibbs_block.hpp -- a closed-form Gibbs leaf that draws its
 *                               K-vector theta exactly from a Dirichlet
 *                               distribution whose concentration is
 *                               computed from the context at each step.
 *
 *  WHY THIS BLOCK EXISTS
 *  =====================
 *  `simplex_nuts_block` works for any target on the simplex, including
 *  non-conjugate cases where theta couples into a complicated likelihood.
 *  But for the common case where theta's full conditional is exactly
 *
 *      theta | everything_else  ~  Dirichlet(alpha_post)
 *
 *  (which holds in any Dirichlet-Categorical, Dirichlet-Multinomial,
 *  LDA-style topic model, Bayesian naive Bayes, etc.), running NUTS on
 *  a stick-breaking reparameterization is 2+ orders of magnitude slower
 *  than just drawing directly from the Dirichlet. This block does the
 *  latter via the standard gamma-normalization trick:
 *
 *      g_k  ~  Gamma(alpha_post[k], 1)
 *      theta[k] = g[k] / sum(g)
 *
 *  Each `step()` is O(K) and produces an exact iid draw -- no warmup,
 *  no adaptation, no gradient, no mcmclib. Composes with nuts_block and
 *  other leaves inside a composite_block exactly the same way.
 *
 *  DESIGN
 *  ------
 *  User supplies a `alpha_post_fn(ctx)` that returns the length-K
 *  posterior concentration vector. The function reads whatever it needs
 *  from the context (typically category counts from a categorical block,
 *  plus a prior alpha). Like all other blocks, this function is the
 *  only mathematical content the AI has to write; the remaining code is
 *  a small amount of idiomatic glue.
 *
 *  RELATIONSHIP TO simplex_nuts_block
 *  ----------------------------------
 *  Use this block when the conditional is EXACTLY Dirichlet. Use
 *  simplex_nuts_block when the conditional has additional non-
 *  Dirichlet factors (e.g. a logistic link, a smooth prior on the
 *  log-ratios, a penalty on the simplex location).
 *
 *  NUMERICAL NOTES
 *  ---------------
 *  std::gamma_distribution can return exactly 0 for very small shape
 *  parameters (below ~0.01) due to underflow. In that case the sum of
 *  draws can also be zero, producing NaN on division. We detect this
 *  and throw a descriptive error: the caller should either bound the
 *  prior shape away from zero, or switch to simplex_nuts_block whose
 *  stick-breaking parameterization handles tiny shapes gracefully.
 *================================================================================*/

#ifndef AI4BAYESCODE_DIRICHLET_GIBBS_BLOCK_HPP
#define AI4BAYESCODE_DIRICHLET_GIBBS_BLOCK_HPP

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
 * @brief Signature of the posterior-concentration oracle consumed by
 *        dirichlet_gibbs_block.
 *
 * Given the current block_context, return the length-K vector of
 * Dirichlet concentration parameters for the full conditional
 *
 *     theta | ctx  ~  Dirichlet(alpha_post[0..K-1])
 *
 * All entries must be strictly positive. The function is called once
 * per `step()` and must be pure.
 */
using dirichlet_alpha_fn =
    std::function<arma::vec(const block_context& ctx)>;

/// Configuration bundle for dirichlet_gibbs_block construction.
struct dirichlet_gibbs_block_config {
    /// Unique name for this block within its composite; also the key
    /// under which shared_data_t stores this block's current simplex.
    std::string name;

    /// Number of categories (dimension of theta).
    std::size_t n_categories = 0;

    /// Initial value on the simplex. Must have length n_categories and
    /// sum to 1 (checked in the constructor, tolerance 1e-8).
    arma::vec initial_values;

    /// The posterior-concentration oracle (see @ref dirichlet_alpha_fn).
    /// Required.
    dirichlet_alpha_fn alpha_post_fn;
};

/**
 * @brief Closed-form Gibbs leaf that samples theta exactly from a
 *        Dirichlet conditional via gamma-normalization. See the file
 *        header for motivation and usage.
 */
class dirichlet_gibbs_block : public block_sampler {
public:
    explicit dirichlet_gibbs_block(dirichlet_gibbs_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.n_categories < 2) {
            throw std::invalid_argument(
                "dirichlet_gibbs_block: n_categories must be >= 2");
        }
        if (cfg_.initial_values.n_elem != cfg_.n_categories) {
            throw std::invalid_argument(
                "dirichlet_gibbs_block: initial_values length must "
                "equal n_categories");
        }
        const double init_sum = arma::sum(cfg_.initial_values);
        if (std::abs(init_sum - 1.0) > 1e-8) {
            throw std::invalid_argument(
                "dirichlet_gibbs_block: initial_values must sum to 1");
        }
        for (std::size_t k = 0; k < cfg_.n_categories; ++k) {
            if (cfg_.initial_values[k] <= 0.0) {
                throw std::invalid_argument(
                    "dirichlet_gibbs_block: initial_values entries must "
                    "be strictly positive");
            }
        }
        if (!cfg_.alpha_post_fn) {
            throw std::invalid_argument(
                "dirichlet_gibbs_block: alpha_post_fn is required");
        }
        values_ = cfg_.initial_values;
    }

    // ---- block_sampler interface ---------------------------------------

    void set_context(const block_context& ctx) override {
        // COPY. See the design contract in block_sampler.hpp.
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        const arma::vec alpha_post = cfg_.alpha_post_fn(context_);
        if (alpha_post.n_elem != cfg_.n_categories) {
            throw std::runtime_error(
                "dirichlet_gibbs_block '" + cfg_.name
                + "': alpha_post_fn returned wrong length");
        }
        // Gamma-normalization: g_k ~ Gamma(alpha_post[k], 1) independently,
        // theta[k] = g[k] / sum(g).
        arma::vec g(cfg_.n_categories);
        double total = 0.0;
        for (std::size_t k = 0; k < cfg_.n_categories; ++k) {
            if (!(alpha_post[k] > 0.0)) {
                throw std::runtime_error(
                    "dirichlet_gibbs_block '" + cfg_.name
                    + "': alpha_post contains a non-positive entry");
            }
            std::gamma_distribution<double> gam(alpha_post[k], 1.0);
            g[k]   = gam(rng);
            total += g[k];
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::runtime_error(
                "dirichlet_gibbs_block '" + cfg_.name
                + "': gamma draws underflowed to zero (try raising the "
                "prior concentration or switching to simplex_nuts_block)");
        }
        for (std::size_t k = 0; k < cfg_.n_categories; ++k) {
            values_[k] = g[k] / total;
        }

        // Append to history if enabled
        if (keep_history_) {
            history_buf_.push_back(values_);
        }
    }

    const arma::vec& current() const override { return values_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != cfg_.n_categories) {
            throw std::invalid_argument(
                "dirichlet_gibbs_block::set_current: wrong length");
        }
        const double s = arma::sum(theta);
        if (std::abs(s - 1.0) > 1e-8) {
            throw std::invalid_argument(
                "dirichlet_gibbs_block::set_current: values must sum to 1");
        }
        for (std::size_t k = 0; k < cfg_.n_categories; ++k) {
            if (theta[k] <= 0.0) {
                throw std::invalid_argument(
                    "dirichlet_gibbs_block::set_current: entries must be "
                    "strictly positive");
            }
        }
        values_ = theta;
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override {
        return cfg_.n_categories;
    }

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
    dirichlet_gibbs_block_config cfg_;
    arma::vec                    values_;
    block_context                context_;
    std::vector<arma::vec>       history_buf_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_DIRICHLET_GIBBS_BLOCK_HPP

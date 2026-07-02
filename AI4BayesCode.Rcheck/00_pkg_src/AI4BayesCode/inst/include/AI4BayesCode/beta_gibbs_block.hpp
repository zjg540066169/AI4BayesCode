/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  beta_gibbs_block.hpp -- closed-form Gibbs leaf that draws a scalar
 *                           parameter exactly from a Beta distribution
 *                           whose shape parameters are computed from context.
 *
 *  WHY THIS BLOCK EXISTS
 *  =====================
 *  In spike-and-slab models, the mixing proportion pi has a Beta-Binomial
 *  conjugate posterior:
 *
 *      gamma_j ~ Bernoulli(pi),  j = 1..p
 *      pi ~ Beta(a, b)
 *      => pi | gamma ~ Beta(a + sum(gamma), b + p - sum(gamma))
 *
 *  Using NUTS for pi is extremely inefficient: the full conditional only
 *  depends on a single sufficient statistic sum(gamma), and the Beta
 *  posterior can be drawn exactly with two Gamma draws. This block does
 *  that in O(1) per step.
 *
 *  DESIGN
 *  ------
 *  User supplies a `params_fn(ctx)` that returns the two Beta shape
 *  parameters (alpha, beta). The block draws from Beta(alpha, beta)
 *  via the standard Gamma trick: x ~ Gamma(a, 1), y ~ Gamma(b, 1),
 *  result = x / (x + y).
 *================================================================================*/

#ifndef AI4BAYESCODE_BETA_GIBBS_BLOCK_HPP
#define AI4BAYESCODE_BETA_GIBBS_BLOCK_HPP

#include "block_sampler.hpp"

#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

struct beta_dist_params {
    double alpha;   // > 0
    double beta;    // > 0
};

using beta_params_fn =
    std::function<beta_dist_params(const block_context& ctx)>;

struct beta_gibbs_block_config {
    std::string name;
    double initial_value = 0.5;   // in (0, 1)
    beta_params_fn params_fn;
};

class beta_gibbs_block : public block_sampler {
public:
    explicit beta_gibbs_block(beta_gibbs_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.initial_value <= 0.0 || cfg_.initial_value >= 1.0) {
            throw std::invalid_argument(
                "beta_gibbs_block: initial_value must be in (0, 1)");
        }
        if (!cfg_.params_fn) {
            throw std::invalid_argument(
                "beta_gibbs_block: params_fn is required");
        }
        value_ = arma::vec{cfg_.initial_value};
    }

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        const beta_dist_params par = cfg_.params_fn(context_);
        if (!(par.alpha > 0.0) || !(par.beta > 0.0)) {
            throw std::runtime_error(
                "beta_gibbs_block '" + cfg_.name
                + "': alpha and beta must be > 0 (got alpha="
                + std::to_string(par.alpha) + ", beta="
                + std::to_string(par.beta) + ")");
        }

        // Beta(a, b) via Gamma trick:
        // x ~ Gamma(a, 1), y ~ Gamma(b, 1), result = x / (x + y)
        // std::gamma_distribution uses shape-SCALE form, E[X] = shape * scale
        std::gamma_distribution<double> gam_a(par.alpha, 1.0);
        std::gamma_distribution<double> gam_b(par.beta,  1.0);
        double x = gam_a(rng);
        double y = gam_b(rng);

        // Clamp to avoid exact 0 or 1 from underflow
        if (x < 1e-300) x = 1e-300;
        if (y < 1e-300) y = 1e-300;

        value_[0] = x / (x + y);

        // Append to history if enabled
        if (keep_history_) {
            history_buf_.push_back(value_);
        }
    }

    const arma::vec& current() const override { return value_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != 1) {
            throw std::invalid_argument(
                "beta_gibbs_block::set_current: must be length 1");
        }
        if (theta[0] <= 0.0 || theta[0] >= 1.0) {
            throw std::invalid_argument(
                "beta_gibbs_block::set_current: must be in (0, 1)");
        }
        value_ = theta;
    }

    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim() const noexcept override { return 1; }

    // ---- History overrides -----------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_buf_, value_);
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override {
        history_buf_.clear();
    }

private:
    beta_gibbs_block_config cfg_;
    arma::vec               value_;
    block_context           context_;
    std::vector<arma::vec>  history_buf_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_BETA_GIBBS_BLOCK_HPP

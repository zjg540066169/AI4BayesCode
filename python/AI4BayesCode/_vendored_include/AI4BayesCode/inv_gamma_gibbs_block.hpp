/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  inv_gamma_gibbs_block.hpp -- Closed-form Gibbs leaf block for a SCALAR
 *                                positive parameter whose full conditional
 *                                is an Inverse-Gamma distribution.
 *
 *  WHY
 *  ===
 *  Inverse-Gamma posteriors are ubiquitous in conjugate Bayesian models:
 *    - sigma^2 in Gaussian regression (Normal-InvGamma conjugate)
 *    - tau^2 in a Gaussian slab with InvGamma hyperprior
 *    - theta in Dirichlet-concentration-with-Gamma-prior (inverse form)
 *    - any scale parameter with InvGamma conditional
 *
 *  For these cases, a nuts_block is OVERKILL:
 *    - Wastes a warmup pass to discover a distribution we know in closed form
 *    - Can wedge if the very first warmup happens on a bad initial state
 *      (e.g., when sibling blocks are still at their init values — the
 *      InvGamma shape/rate may be at an extreme regime during that
 *      initial call, and the step size adapts poorly)
 *
 *  This block exposes the same block_sampler interface as the other
 *  Gibbs leaves (binary_gibbs_block, beta_gibbs_block, etc.). User
 *  supplies a lambda that returns the (shape, rate) of the conditional
 *  given the current context; the block draws an exact iid InvGamma.
 *
 *  PARAMETERISATION
 *  ================
 *  We use shape-RATE convention: for x ~ InvGamma(shape, rate),
 *     pdf(x) ∝ x^(-shape - 1) * exp(-rate / x),     x > 0
 *     E[x] = rate / (shape - 1),   shape > 1
 *     Mode = rate / (shape + 1)
 *     Var  = rate^2 / [(shape - 1)^2 (shape - 2)],  shape > 2
 *
 *  To sample x ~ InvGamma(shape, rate):
 *     g ~ Gamma(shape, rate=rate)      # shape-rate convention
 *     x = 1 / g
 *  (std::gamma_distribution uses shape-SCALE, so pass scale = 1/rate.)
 *
 *  USAGE (sigma^2 in Gaussian regression)
 *  --------------------------------------
 *     inv_gamma_gibbs_block_config cfg;
 *     cfg.name          = "sigma2";
 *     cfg.initial_value = sd(y)^2;
 *     cfg.params_fn     = [N, p](const block_context& ctx) {
 *         const arma::vec& y    = ctx.at("y");
 *         const arma::vec& Xb   = ctx.at("Xbeta_cache");
 *         const double    a_pr  = ctx.at("a_sigma_prior")[0];
 *         const double    b_pr  = ctx.at("b_sigma_prior")[0];
 *         double sse = arma::accu(arma::square(y - Xb));
 *         return inv_gamma_params{ a_pr + N/2.0, b_pr + sse/2.0 };
 *     };
 *     impl_->add_child(std::make_unique<inv_gamma_gibbs_block>(cfg));
 *
 *  AI RULE COMPLIANCE
 *  ==================
 *  - No hand-written Jacobian (doesn't apply — pure Gibbs leaf, no transform)
 *  - No gradient (not needed for exact Gibbs)
 *  - Standard block_sampler contract
 *  - Values stored as length-1 arma::vec (scalar posterior)
 *================================================================================*/

#ifndef AI4BAYESCODE_INV_GAMMA_GIBBS_BLOCK_HPP
#define AI4BAYESCODE_INV_GAMMA_GIBBS_BLOCK_HPP

#include "block_sampler.hpp"

#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/// Posterior InvGamma shape / rate bundle returned by the user's lambda.
struct inv_gamma_params {
    double shape;
    double rate;
};

/// Signature of the InvGamma-parameter oracle.
using inv_gamma_params_fn =
    std::function<inv_gamma_params(const block_context& ctx)>;

struct inv_gamma_gibbs_block_config {
    /// Unique name within the composite; shared_data key for this scalar.
    std::string name;

    /// Initial value. Must be strictly positive.
    double initial_value = 1.0;

    /// Params oracle: returns the conditional InvGamma(shape, rate) each sweep.
    inv_gamma_params_fn params_fn;
};

class inv_gamma_gibbs_block : public block_sampler {
public:
    explicit inv_gamma_gibbs_block(inv_gamma_gibbs_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (!(cfg_.initial_value > 0.0)) {
            throw std::invalid_argument(
                "inv_gamma_gibbs_block: initial_value must be > 0");
        }
        if (!cfg_.params_fn) {
            throw std::invalid_argument(
                "inv_gamma_gibbs_block: params_fn is required");
        }
        value_.set_size(1);
        value_[0] = cfg_.initial_value;
    }

    // ---- block_sampler interface --------------------------------------

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        const inv_gamma_params p = cfg_.params_fn(context_);
        if (!(p.shape > 0.0) || !(p.rate > 0.0)) {
            throw std::runtime_error(
                "inv_gamma_gibbs_block '" + cfg_.name
                + "': params_fn returned non-positive shape or rate (shape="
                + std::to_string(p.shape) + ", rate=" + std::to_string(p.rate)
                + ")");
        }
        // Draw Gamma(shape, scale = 1/rate), invert.
        // std::gamma_distribution uses shape-SCALE, so scale = 1/rate.
        std::gamma_distribution<double> gam(p.shape, 1.0 / p.rate);
        const double g = gam(rng);
        if (!(g > 0.0)) {
            // Numerical underflow — inherit previous value (conservative).
            // This is a rare pathology in extreme shape/rate regimes.
            return;
        }
        value_[0] = 1.0 / g;

        if (keep_history_) {
            history_buf_.push_back(value_);
        }
    }

    const arma::vec& current() const override { return value_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != 1) {
            throw std::invalid_argument(
                "inv_gamma_gibbs_block::set_current: length must be 1");
        }
        if (!(theta[0] > 0.0)) {
            throw std::invalid_argument(
                "inv_gamma_gibbs_block::set_current: value must be > 0");
        }
        value_[0] = theta[0];
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return 1; }

    // ---- History overrides ---------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_buf_, value_);
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override { history_buf_.clear(); }

private:
    inv_gamma_gibbs_block_config cfg_;
    arma::vec                    value_;
    block_context                context_;
    std::vector<arma::vec>       history_buf_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_INV_GAMMA_GIBBS_BLOCK_HPP

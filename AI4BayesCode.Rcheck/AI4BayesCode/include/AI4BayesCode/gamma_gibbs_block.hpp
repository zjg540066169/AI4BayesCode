/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  gamma_gibbs_block.hpp -- Closed-form Gibbs leaf block for a SCALAR
 *                           positive parameter whose full conditional
 *                           is a Gamma distribution.
 *
 *  This is the companion / dual of inv_gamma_gibbs_block. Use this when
 *  the conditional posterior on a positive scalar is exactly Gamma
 *  (shape, rate). Common cases:
 *
 *    - DP concentration α with α ~ Gamma(a, b) prior, under the
 *      TRUNCATED stick-breaking representation:
 *        p(α | V_1, ..., V_{T-1}) ~ Gamma(a + T - 1,
 *                                         b - Σ_{k=1}^{T-1} log(1 - V_k))
 *      (closed-form alternative to NUTS on log α; see
 *      examples/DPGaussianMixture.cpp for the NUTS path that this
 *      block can replace.)
 *
 *    - Precision λ in Normal-Gamma cluster prior (already handled
 *      vectorised across K clusters by normal_gamma_cluster_gibbs_block).
 *
 *    - Any scalar positive parameter with Gamma posterior conjugate.
 *
 *  PARAMETERISATION
 *  ================
 *  shape-RATE convention (matches inv_gamma_gibbs_block, R's
 *  rgamma(n, shape, rate=...), and JAGS / NIMBLE / Stan's `gamma`).
 *
 *    x ~ Gamma(shape, rate):
 *      pdf(x) = rate^shape / Gamma(shape) * x^(shape-1) * exp(-rate * x)
 *      E[x]   = shape / rate
 *      Var[x] = shape / rate^2
 *      Mode   = (shape - 1) / rate    (when shape > 1)
 *
 *  Sampling:
 *    std::gamma_distribution<double> uses shape-SCALE, where
 *    scale = 1 / rate. We construct it with scale = 1.0 / p.rate so
 *    the resulting draw is Gamma(shape, rate) in our convention.
 *    E[X] = shape * scale = shape / rate. ✓
 *
 *  CONVENTION NOTE (rcpp_api.md §11): every distribution call site
 *  carries a comment reaffirming the parameterisation explicitly.
 *  This is the most common silent-bug source in conjugate-Gibbs code.
 *
 *  USAGE (DP concentration α under truncated stick-breaking)
 *  ---------------------------------------------------------
 *     gamma_gibbs_block_config cfg;
 *     cfg.name          = "alpha";
 *     cfg.initial_value = a_prior / b_prior;   // prior mean
 *     cfg.params_fn     = [](const block_context& ctx) -> gamma_params {
 *         const arma::vec& V = ctx.at("stick_V");
 *         const double a_pr  = ctx.at("a_alpha")[0];
 *         const double b_pr  = ctx.at("b_alpha")[0];
 *         const std::size_t T = V.n_elem;          // = K_trunc
 *         double Tsum = 0.0;
 *         for (std::size_t k = 0; k + 1 < T; ++k) {
 *             const double w = 1.0 - V[k];
 *             if (!(w > 0.0) || !std::isfinite(w))
 *                 // Conservative: pin to prior on a bad partition state.
 *                 return gamma_params{ a_pr, b_pr };
 *             Tsum += std::log(w);
 *         }
 *         // Posterior under truncated SBP:
 *         //   shape_n = a_prior + (T - 1)
 *         //   rate_n  = b_prior - Σ log(1 - V_k)        (note: -Σ log(1-V) > 0)
 *         return gamma_params{ a_pr + static_cast<double>(T - 1),
 *                              b_pr - Tsum };
 *     };
 *     impl_->add_child(std::make_unique<gamma_gibbs_block>(std::move(cfg)));
 *
 *  This is a STATISTICALLY EQUIVALENT replacement for the
 *  alpha_natural_log_density + nuts_block path used in DPGaussianMixture.cpp.
 *  Speed difference is small (1-D NUTS is fast already), but the Gibbs
 *  draw is exact iid (no warmup / no autocorrelation on α).
 *
 *  AI RULE COMPLIANCE
 *  ==================
 *  - No hand-written Jacobian (pure Gibbs leaf, no transform)
 *  - No gradient (not needed for exact Gibbs)
 *  - Standard block_sampler contract
 *  - Values stored as length-1 arma::vec
 *  - Underflow guard: if std::gamma_distribution returns zero (extreme
 *    shape regime), the block keeps its previous value rather than
 *    setting value to zero.
 *================================================================================*/

#ifndef AI4BAYESCODE_GAMMA_GIBBS_BLOCK_HPP
#define AI4BAYESCODE_GAMMA_GIBBS_BLOCK_HPP

#include "block_sampler.hpp"

#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/// Posterior Gamma shape-RATE bundle returned by the user's lambda.
struct gamma_params {
    double shape;
    double rate;
};

/// Signature of the Gamma-parameter oracle.
using gamma_params_fn =
    std::function<gamma_params(const block_context& ctx)>;

struct gamma_gibbs_block_config {
    /// Unique name within the composite; shared_data key for this scalar.
    std::string name;

    /// Initial value. Must be strictly positive.
    double initial_value = 1.0;

    /// Params oracle: returns the conditional Gamma(shape, rate) each
    /// sweep. shape-RATE convention (NOT shape-scale).
    gamma_params_fn params_fn;
};

class gamma_gibbs_block : public block_sampler {
public:
    explicit gamma_gibbs_block(gamma_gibbs_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "gamma_gibbs_block: name must be non-empty");
        }
        if (!(cfg_.initial_value > 0.0)) {
            throw std::invalid_argument(
                "gamma_gibbs_block: initial_value must be > 0");
        }
        if (!cfg_.params_fn) {
            throw std::invalid_argument(
                "gamma_gibbs_block: params_fn is required");
        }
        value_.set_size(1);
        value_[0] = cfg_.initial_value;
    }

    // ---- block_sampler interface --------------------------------------

    void set_context(const block_context& ctx) override {
        // COPY (per design contract).
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        const gamma_params p = cfg_.params_fn(context_);
        if (!(p.shape > 0.0) || !(p.rate > 0.0) ||
            !std::isfinite(p.shape) || !std::isfinite(p.rate)) {
            throw std::runtime_error(
                "gamma_gibbs_block '" + cfg_.name +
                "': params_fn returned non-positive or non-finite "
                "shape / rate (shape=" + std::to_string(p.shape) +
                ", rate=" + std::to_string(p.rate) + ")");
        }
        // COMMENTED PARAMETERISATION (rcpp_api.md §11):
        //   We want a draw from Gamma(shape, RATE = p.rate).
        //   std::gamma_distribution<double>(shape, scale) is shape-SCALE,
        //   E[X] = shape * scale.  To get Gamma(shape, rate), pass
        //   scale = 1 / rate, so E[X] = shape / rate. ✓
        std::gamma_distribution<double> gam(p.shape, 1.0 / p.rate);
        const double g = gam(rng);
        if (g > 0.0 && std::isfinite(g)) {
            value_[0] = g;
        }
        // Else: keep previous value (rare extreme-shape underflow).

        if (keep_history_) {
            history_buf_.push_back(value_);
        }
    }

    const arma::vec& current() const override { return value_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != 1) {
            throw std::invalid_argument(
                "gamma_gibbs_block::set_current: length must be 1");
        }
        if (!(theta[0] > 0.0)) {
            throw std::invalid_argument(
                "gamma_gibbs_block::set_current: value must be > 0");
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
    gamma_gibbs_block_config cfg_;
    arma::vec                value_;
    block_context            context_;
    std::vector<arma::vec>   history_buf_;
};

}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_GAMMA_GIBBS_BLOCK_HPP

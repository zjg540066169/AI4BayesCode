/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v3.0 or later
 *  (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  frailty_gamma_gibbs_block.hpp -- exact Gamma-conjugate Gibbs update for a
 *      length-G vector of per-group multiplicative Gamma frailties w_g in a
 *      shared-frailty proportional-hazards survival model, driven by an
 *      externally-supplied piecewise-exponential (PEH) baseline hazard lambda[K]
 *      and covariate offset exp(f_i) per subject.
 *
 *  MODEL (Clayton 1991 JRSS-B 53(1):45-73; Ibrahim, Chen & Sinha 2001 Sec.4.3;
 *  Vaupel-Manton-Stallard 1979 Demography 16(3):439-454; Aslanidou-Dey-Sinha
 *  1998 CanJStat 26(1):33-48)
 *  =============================================================================
 *  Subject i belongs to group g(i) in {0..G-1}. Per-subject hazard:
 *      h_i(t) = w_{g(i)} * lambda_{k(t)} * exp(f_i)
 *  Group frailties w_g are independent Gamma(theta, theta) a priori (so
 *  E[w_g] = 1, Var[w_g] = 1/theta -- a scalar concentration where large theta
 *  means little unobserved between-group variation, small theta = strong).
 *  theta is a SCALAR hyperparameter -- fixed via cfg.theta OR supplied every
 *  sweep via cfg.theta_key (a sibling NUTS / slice block); see USAGE.
 *
 *  FULL CONDITIONAL (independent Gamma-Gamma conjugate step across groups)
 *  ======================================================================
 *  Given lambda[K], f, delta, t, v (entry time), and theta, the joint
 *  log-likelihood contribution to w_g reduces to the standard Poisson-kernel
 *  form:
 *      sum_{i in g} [ delta_i * log w_g  -  w_g * exp(f_i) * H_0(t_i, v_i) ]
 *      + (theta - 1) log w_g  -  theta * w_g
 *
 *      = (theta + D_g - 1) * log w_g  -  (theta + H_g) * w_g
 *
 *  where
 *      D_g = sum_{i in g} delta_i                                    (events)
 *      H_g = sum_{i in g} exp(f_i) * H_0(t_i, v_i)                   (weighted at-risk)
 *      H_0(t, v) = sum_j lambda_j * max(0, min(t, e_j) - max(v, e_{j-1}))
 *
 *  So each w_g's full conditional is EXACTLY Gamma:
 *      w_g | rest  ~  Gamma( theta + D_g,  theta + H_g )
 *
 *  One exact Gibbs step draws all G frailties. O(n + G) per step.
 *
 *  ROLE IN THE COMPOSITE
 *  =====================
 *  This block owns w[G]. The user's model composition combines
 *      full_offset[i]  =  w[g(i)]  *  exp(x_i^T beta)   (regression sibling)
 *  in a shared_data refresher, which is then read by the sibling
 *  piecewise_exponential_gibbs_block as its `offset_key`. See the shipped
 *  reference example for the full wiring.
 *
 *  theta HYPERPARAMETER
 *  ====================
 *  theta is NOT Gamma-conjugate under the frailty-with-Gamma prior
 *  (Clayton 1991 gives an Antoniak-style CRP marginal that is intractable
 *  in closed form). Two supported patterns:
 *   1) FIXED theta: cfg.theta > 0, cfg.theta_key = "". Sensible default when
 *      the user has domain knowledge or is doing sensitivity analysis.
 *   2) INFERRED theta: cfg.theta_key = "theta". A sibling nuts_block /
 *      univariate_slice_sampling_block samples theta each sweep and writes
 *      it to shared_data under that key. This block reads it fresh each step.
 *
 *  USAGE (shared-frailty PEH: subjects clustered in G groups)
 *  ---------------------------------------------------------
 *      // Shared data provides "t", "delta", "entry_time" (optional), group
 *      // labels "z" (length n, integer 0..G-1), current "lambda" (length K).
 *      frailty_gamma_gibbs_block_config cfg;
 *      cfg.name       = "w";                              // writes length-G w vector
 *      cfg.G          = G;                                // #groups
 *      cfg.edges      = edges;                            // same as sibling PEH block
 *      cfg.theta      = 2.0;                              // FIXED theta option;
 *                                                         //   OR cfg.theta_key = "theta"
 *      // time_key="t", event_key="delta", lambda_key="lambda", group_key="z" (defaults).
 *      // offset_key: OPTIONAL exp(x_i^T beta) from a sibling regression block.
 *      cfg.initial_frailties = arma::vec(G, arma::fill::ones);
 *      composite->add_child(std::make_unique<frailty_gamma_gibbs_block>(std::move(cfg)));
 *
 *  AI RULE COMPLIANCE
 *  ==================
 *  - Closed-form conjugate Gibbs -> no gradient, no autodiff.
 *  - Standard block_sampler contract; writes arma::vec of length G under `name`.
 *================================================================================*/

#ifndef AI4BAYESCODE_FRAILTY_GAMMA_GIBBS_BLOCK_HPP
#define AI4BAYESCODE_FRAILTY_GAMMA_GIBBS_BLOCK_HPP

#include "block_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/// Configuration for frailty_gamma_gibbs_block.
struct frailty_gamma_gibbs_block_config {
    /// Unique name + shared_data + history key for the length-G frailty vector.
    std::string name;

    /// Number of groups. Group labels in shared_data (`group_key`) must be
    /// integers in {0..G-1}.
    std::size_t G = 0;

    /// Bin edges, length K+1, strictly increasing, edges[0] >= 0.
    /// MUST match the sibling piecewise_exponential_gibbs_block's edges.
    arma::vec edges;

    /// FIXED concentration theta > 0 for the Gamma(theta, theta) frailty prior
    /// (E[w_g] = 1, Var[w_g] = 1/theta). Ignored when theta_key is non-empty.
    double theta = 1.0;

    /// OPTIONAL: shared_data key for a SCALAR theta sampled by another block
    /// each sweep. Non-empty => this block reads theta from context every step
    /// (must exist and be a length-1 vec); empty => fall back to cfg.theta.
    std::string theta_key = "";

    /// shared_data key for length-n group labels z_i in {0..G-1}.
    std::string group_key = "z";

    /// shared_data key for length-n event / censoring times t_i > 0.
    std::string time_key = "t";

    /// shared_data key for length-n event indicators delta_i in {0,1}.
    std::string event_key = "delta";

    /// shared_data key for the length-K current baseline hazard rates.
    std::string lambda_key = "lambda";

    /// shared_data key for length-n covariate offset exp(f_i).
    /// OPTIONAL: empty or missing => all exp(f_i) = 1.
    std::string offset_key = "";

    /// shared_data key for length-n left-truncation entry times v_i in [0, t_i].
    /// OPTIONAL: empty or missing => all v_i = 0.
    std::string entry_time_key = "";

    /// Initial frailties w_1..w_G, length G, each > 0.
    /// Defaults to all-ones (prior mean).
    arma::vec initial_frailties;
};

class frailty_gamma_gibbs_block : public block_sampler {
public:
    explicit frailty_gamma_gibbs_block(frailty_gamma_gibbs_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "frailty_gamma_gibbs_block: name must be non-empty");
        }
        if (cfg_.G < 1) {
            throw std::invalid_argument(
                "frailty_gamma_gibbs_block: G must be >= 1");
        }
        if (cfg_.edges.n_elem < 2) {
            throw std::invalid_argument(
                "frailty_gamma_gibbs_block: edges must have length >= 2");
        }
        K_ = cfg_.edges.n_elem - 1;
        if (!(cfg_.edges[0] >= 0.0)) {
            throw std::invalid_argument(
                "frailty_gamma_gibbs_block: edges[0] must be >= 0");
        }
        for (std::size_t k = 0; k < K_; ++k) {
            if (!(cfg_.edges[k + 1] > cfg_.edges[k])) {
                throw std::invalid_argument(
                    "frailty_gamma_gibbs_block: edges must be strictly increasing");
            }
        }
        if (cfg_.theta_key.empty()) {
            if (!(cfg_.theta > 0.0) || !std::isfinite(cfg_.theta)) {
                throw std::invalid_argument(
                    "frailty_gamma_gibbs_block: theta must be positive finite when theta_key is empty");
            }
        }
        if (cfg_.initial_frailties.n_elem == 0) {
            values_.set_size(cfg_.G); values_.fill(1.0);
        } else if (cfg_.initial_frailties.n_elem == cfg_.G) {
            for (std::size_t g = 0; g < cfg_.G; ++g) {
                if (!(cfg_.initial_frailties[g] > 0.0) || !std::isfinite(cfg_.initial_frailties[g])) {
                    throw std::invalid_argument(
                        "frailty_gamma_gibbs_block: initial_frailties entries must be > 0");
                }
            }
            values_ = cfg_.initial_frailties;
        } else {
            throw std::invalid_argument(
                "frailty_gamma_gibbs_block: initial_frailties length must equal G");
        }
        edges_.assign(cfg_.edges.begin(), cfg_.edges.end());
    }

    // ---- block_sampler interface ----------------------------------------

    void set_context(const block_context& ctx) override { context_ = ctx; }

    void step(std::mt19937_64& rng) override {
        // ---- fetch data ----
        auto it_t = context_.find(cfg_.time_key);
        auto it_d = context_.find(cfg_.event_key);
        auto it_l = context_.find(cfg_.lambda_key);
        auto it_z = context_.find(cfg_.group_key);
        if (it_t == context_.end()) {
            throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                + "': key '" + cfg_.time_key + "' (time) missing from context");
        }
        if (it_d == context_.end()) {
            throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                + "': key '" + cfg_.event_key + "' (event) missing");
        }
        if (it_l == context_.end()) {
            throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                + "': key '" + cfg_.lambda_key + "' (lambda) missing");
        }
        if (it_z == context_.end()) {
            throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                + "': key '" + cfg_.group_key + "' (group labels) missing");
        }
        const arma::vec& t      = it_t->second;
        const arma::vec& delta  = it_d->second;
        const arma::vec& lambda = it_l->second;
        const arma::vec& z      = it_z->second;
        const std::size_t n = t.n_elem;
        if (delta.n_elem != n || z.n_elem != n) {
            throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                + "': t, delta, z lengths inconsistent");
        }
        if (lambda.n_elem != K_) {
            throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                + "': lambda length does not match K");
        }

        // ---- optional keys ----
        const arma::vec* offset_ptr = nullptr;
        const arma::vec* entry_ptr  = nullptr;
        if (!cfg_.offset_key.empty()) {
            auto it_o = context_.find(cfg_.offset_key);
            if (it_o != context_.end()) {
                if (it_o->second.n_elem != n) {
                    throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                        + "': offset length mismatch");
                }
                offset_ptr = &it_o->second;
            }
        }
        if (!cfg_.entry_time_key.empty()) {
            auto it_v = context_.find(cfg_.entry_time_key);
            if (it_v != context_.end()) {
                if (it_v->second.n_elem != n) {
                    throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                        + "': entry_time length mismatch");
                }
                entry_ptr = &it_v->second;
            }
        }

        // ---- resolve theta (fixed or from shared_data) ----
        double theta = cfg_.theta;
        if (!cfg_.theta_key.empty()) {
            auto it_th = context_.find(cfg_.theta_key);
            if (it_th == context_.end() || it_th->second.n_elem < 1) {
                throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                    + "': theta_key '" + cfg_.theta_key + "' missing / empty in context");
            }
            theta = it_th->second[0];
            if (!(theta > 0.0) || !std::isfinite(theta)) {
                throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                    + "': theta from context must be positive finite");
            }
        }

        // ---- accumulate D_g (events per group) and H_g (weighted at-risk) ----
        const double eK = edges_[K_];
        std::vector<double> D(cfg_.G, 0.0);
        std::vector<double> H(cfg_.G, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const double ti = t[i];
            const double vi = entry_ptr ? (*entry_ptr)[i] : 0.0;
            const double di = delta[i];
            const double wi = offset_ptr ? (*offset_ptr)[i] : 1.0;
            const std::size_t gi = static_cast<std::size_t>(z[i]);

            if (!(ti > 0.0) || !std::isfinite(ti) || ti > eK) {
                throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                    + "': t[" + std::to_string(i) + "] out of (0, edges[K]]");
            }
            if (vi < 0.0 || vi > ti) {
                throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                    + "': entry_time[" + std::to_string(i) + "] must satisfy 0 <= v <= t");
            }
            if (gi >= cfg_.G) {
                throw std::runtime_error("frailty_gamma_gibbs_block '" + cfg_.name
                    + "': group label z[" + std::to_string(i) + "] = "
                    + std::to_string(gi) + " out of range 0.." + std::to_string(cfg_.G - 1));
            }
            if (!(wi >= 0.0) || !std::isfinite(wi)) continue;

            if (di == 1.0) D[gi] += 1.0;

            // H_0(t_i, v_i) contribution per bin (same integral as PEH block).
            for (std::size_t k = 0; k < K_; ++k) {
                const double lo = std::max(vi, edges_[k]);
                const double hi = std::min(ti, edges_[k + 1]);
                const double dur = hi - lo;
                if (dur > 0.0) H[gi] += wi * lambda[k] * dur;
            }
        }

        // ---- sample w_g ~ Gamma(theta + D_g, theta + H_g) ----
        for (std::size_t g = 0; g < cfg_.G; ++g) {
            const double shape_post = theta + D[g];
            const double rate_post  = theta + H[g];
            // COMMENTED PARAMETERISATION: Gamma(shape, RATE=rate) via
            //   std::gamma_distribution(shape, SCALE=1/rate). E[X] = shape/rate.
            std::gamma_distribution<double> gam(shape_post, 1.0 / rate_post);
            const double w = gam(rng);
            if (w > 0.0 && std::isfinite(w)) {
                values_[g] = w;
            }
            // else keep previous value
        }

        if (keep_history_) history_buf_.push_back(values_);
    }

    const arma::vec& current() const override { return values_; }

    void set_current(const arma::vec& theta_new) override {
        if (theta_new.n_elem != cfg_.G) {
            throw std::invalid_argument(
                "frailty_gamma_gibbs_block::set_current: length must be G");
        }
        for (std::size_t g = 0; g < cfg_.G; ++g) {
            if (!(theta_new[g] > 0.0) || !std::isfinite(theta_new[g])) {
                throw std::invalid_argument(
                    "frailty_gamma_gibbs_block::set_current: entries must be positive finite");
            }
        }
        values_ = theta_new;
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return cfg_.G; }

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_buf_, values_);
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override { history_buf_.clear(); }

private:
    frailty_gamma_gibbs_block_config cfg_;
    std::size_t                      K_ = 0;
    std::vector<double>              edges_;
    arma::vec                        values_;
    block_context                    context_;
    std::vector<arma::vec>           history_buf_;
};

}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_FRAILTY_GAMMA_GIBBS_BLOCK_HPP

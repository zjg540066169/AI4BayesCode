/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v3.0 or later
 *  (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  piecewise_exponential_gibbs_block.hpp -- exact Gamma-conjugate Gibbs update
 *      for the K piecewise-constant baseline hazard rates of a piecewise-
 *      exponential (PEH) survival model with an EXTERNALLY-supplied per-subject
 *      log-relative-hazard f_i (Cox-style proportional-hazards structure).
 *
 *  MODEL (Ibrahim, Chen & Sinha 2001 "Bayesian Survival Analysis", Sec.3.2;
 *  Kalbfleisch 1978 "Non-parametric Bayesian analysis of survival time data",
 *  JRSS-B 40(2):214-221; Prentice & Gloeckler 1978 Biometrics 34:57-67)
 *  ====================================================================
 *  Time axis is partitioned into K bins by user-supplied edges
 *      0 = e_0 < e_1 < ... < e_K   (edges.n_elem == K+1)
 *  Bin k covers (e_{k-1}, e_k], k = 1..K. Baseline hazard is piecewise constant:
 *      h_0(t) = lambda_k     for t in (e_{k-1}, e_k].
 *  Per-subject hazard multiplies by an externally-supplied log-relative-hazard
 *  f_i = f_i(covariates) (log-linear x_i^T beta, BART, GP, ... -- this block is
 *  AGNOSTIC to how f_i is produced):
 *      h_i(t) = lambda_{k(t)} * exp(f_i).
 *  Observed: t_i (event OR right-censoring time), delta_i in {0,1} (event
 *  indicator). Optional left-truncation entry time v_i >= 0 (subject is at
 *  risk only on [v_i, t_i]).
 *
 *  FULL LIKELIHOOD (rearranged into a per-bin Poisson kernel):
 *      log L(lambda | f, data)
 *          = sum_i { delta_i * [log lambda_{k(t_i)} + f_i]
 *                    - exp(f_i) * sum_k lambda_k * Delta_k(t_i, v_i) }
 *          = sum_k [ E_k * log lambda_k - lambda_k * R_k ]  +  const_in_lambda
 *      E_k = sum_i delta_i * I(t_i in bin k)                (events in k)
 *      R_k = sum_i exp(f_i) * Delta_k(t_i, v_i)             (weighted at-risk time in k)
 *      Delta_k(t_i, v_i) = max(0, min(t_i, e_k) - max(v_i, e_{k-1}))
 *
 *  So lambda_k enters as a Poisson kernel independently across k, and with
 *  the conjugate prior lambda_k ~ Gamma(a0, b0) (shape-RATE) the full
 *  conditional is:
 *
 *      lambda_k | rest ~ Gamma(a0 + E_k,  b0 + R_k)    (independent across k)
 *
 *  This block draws all K rates in one exact Gibbs step -- no NUTS, no
 *  gradient, no warmup. K=10..500 is comfortable at O(n + K) per step.
 *
 *  WHY THIS BLOCK EXISTS
 *  =====================
 *  A joint_nuts_block on log lambda[K] is slow at K=50-200 because the
 *  Gamma-tailed posterior on lambda_k mixes poorly under any diagonal metric,
 *  and a full-dense metric is O(K^2) per sample -- both are 50-100x slower than
 *  the exact Gibbs update this block does. Additionally the E_k / R_k / Delta_k
 *  accounting (with left-truncation) is error-prone to re-derive from scratch
 *  in every user model. Encapsulating it here is the primitive value; the
 *  Gamma draw itself is the trivial part.
 *
 *  CONVENTIONS
 *  ===========
 *  - Gamma shape-RATE (matches gamma_gibbs_block / inv_gamma_gibbs_block).
 *    lambda ~ Gamma(shape, rate) has E[lambda] = shape/rate,
 *    pdf(lambda) = rate^shape/Gamma(shape) * lambda^(shape-1) * exp(-rate * lambda).
 *    std::gamma_distribution takes shape-SCALE = 1/rate; construct with
 *    (shape, 1.0 / rate).
 *  - Bins are (e_{k-1}, e_k], k=1..K. A time t_i = e_k lands in bin k (not k+1).
 *    Times exactly at t=0 (measure-zero in practice) are NOT in any bin --
 *    treat them as sitting at a small positive time in bin 1.
 *  - t_i must satisfy 0 <= v_i <= t_i <= e_K. A t_i > e_K throws (the caller
 *    must ensure the last edge is >= max observed follow-up time).
 *
 *  USAGE (baseline-only PEH: no covariates)
 *  ----------------------------------------
 *      // Shared data must carry "t" (length n, positive) and "delta" (length n, 0/1).
 *      arma::vec edges = { 0.0, 1.0, 2.0, 5.0, 10.0 };   // K = 4 bins
 *      piecewise_exponential_gibbs_block_config cfg;
 *      cfg.name  = "lambda";     // shared_data key + get_history key
 *      cfg.edges = edges;
 *      cfg.a0    = 0.01;
 *      cfg.b0    = 0.01;
 *      // time_key="t", event_key="delta", offset_key="" (no covariates), entry_time_key="" (no truncation)
 *      composite->add_child(std::make_unique<piecewise_exponential_gibbs_block>(std::move(cfg)));
 *
 *  USAGE (PEH + Cox linear log-hazard-ratio beta on x_i)
 *  -----------------------------------------------------
 *      // A sibling block samples beta and writes exp(x_i^T beta) each sweep to
 *      // shared_data key "exp_lp" (length n). Then set cfg.offset_key = "exp_lp".
 *      // See examples/PehCoxRegression.cpp for the wired-up template.
 *
 *  AI RULE COMPLIANCE
 *  ==================
 *  - Closed-form conjugate Gibbs -> no gradient, no Jacobian, no autodiff.
 *  - Standard block_sampler contract.
 *  - Output stored as arma::vec of length K under `name` in shared_data,
 *    logged to history via detail::make_history_map.
 *  - Optional refreshable-context keys (offset, entry_time) look up via
 *    context_.count(key) so absent keys silently degrade to the base model.
 *================================================================================*/

#ifndef AI4BAYESCODE_PIECEWISE_EXPONENTIAL_GIBBS_BLOCK_HPP
#define AI4BAYESCODE_PIECEWISE_EXPONENTIAL_GIBBS_BLOCK_HPP

#include "block_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/// Configuration for piecewise_exponential_gibbs_block.
struct piecewise_exponential_gibbs_block_config {
    /// Unique name within the composite; also the shared_data + history
    /// key under which this block writes its length-K lambda vector.
    std::string name;

    /// Bin edges, length K+1, strictly increasing, edges[0] >= 0.
    /// Bin k covers (edges[k-1], edges[k]] for k = 1..K.
    /// edges[K] must exceed max observed follow-up time (t_i, entry_time_i).
    arma::vec edges;

    /// Gamma(shape=a0, rate=b0) prior on each lambda_k. Independent across k.
    /// Default (0.01, 0.01) is weakly informative (prior mean = 1, prior sd = 10).
    double a0 = 0.01;
    double b0 = 0.01;

    /// shared_data key for length-n event / censoring times t_i > 0.
    /// REQUIRED (throws in step() if missing or wrong length).
    std::string time_key = "t";

    /// shared_data key for length-n event indicators delta_i in {0,1}.
    /// REQUIRED (throws in step() if missing or wrong length).
    std::string event_key = "delta";

    /// shared_data key for length-n log-relative-hazard offset exp(f_i).
    /// OPTIONAL: empty string OR key missing from context => all exp(f_i) = 1
    /// (baseline-only PEH -- no covariates).
    std::string offset_key = "";

    /// shared_data key for length-n left-truncation entry times v_i in [0, t_i].
    /// OPTIONAL: empty string OR key missing from context => all v_i = 0
    /// (no left-truncation; subjects enter at time 0).
    std::string entry_time_key = "";

    /// Initial lambda_1..lambda_K, length K. Each entry > 0.
    /// Defaults to the Gamma-prior mean (a0/b0) for every bin.
    arma::vec initial_lambda;
};

class piecewise_exponential_gibbs_block : public block_sampler {
public:
    explicit piecewise_exponential_gibbs_block(
        piecewise_exponential_gibbs_block_config cfg)
        : cfg_(std::move(cfg))
    {
        // ---- validation of construction-time invariants -----------------
        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "piecewise_exponential_gibbs_block: name must be non-empty");
        }
        if (cfg_.edges.n_elem < 2) {
            throw std::invalid_argument(
                "piecewise_exponential_gibbs_block: edges must have length >= 2 (K >= 1)");
        }
        K_ = cfg_.edges.n_elem - 1;
        if (!(cfg_.edges[0] >= 0.0)) {
            throw std::invalid_argument(
                "piecewise_exponential_gibbs_block: edges[0] must be >= 0");
        }
        for (std::size_t k = 0; k < K_; ++k) {
            if (!(cfg_.edges[k + 1] > cfg_.edges[k])) {
                throw std::invalid_argument(
                    "piecewise_exponential_gibbs_block: edges must be strictly increasing");
            }
        }
        if (!(cfg_.a0 > 0.0) || !(cfg_.b0 > 0.0)
            || !std::isfinite(cfg_.a0) || !std::isfinite(cfg_.b0)) {
            throw std::invalid_argument(
                "piecewise_exponential_gibbs_block: a0 and b0 must be positive finite");
        }
        if (cfg_.time_key.empty()) {
            throw std::invalid_argument(
                "piecewise_exponential_gibbs_block: time_key must be non-empty");
        }
        if (cfg_.event_key.empty()) {
            throw std::invalid_argument(
                "piecewise_exponential_gibbs_block: event_key must be non-empty");
        }

        // Initial lambda: default to prior mean if not supplied.
        if (cfg_.initial_lambda.n_elem == 0) {
            values_.set_size(K_);
            values_.fill(cfg_.a0 / cfg_.b0);
        } else if (cfg_.initial_lambda.n_elem == K_) {
            for (std::size_t k = 0; k < K_; ++k) {
                if (!(cfg_.initial_lambda[k] > 0.0)
                    || !std::isfinite(cfg_.initial_lambda[k])) {
                    throw std::invalid_argument(
                        "piecewise_exponential_gibbs_block: initial_lambda entries must be > 0");
                }
            }
            values_ = cfg_.initial_lambda;
        } else {
            throw std::invalid_argument(
                "piecewise_exponential_gibbs_block: initial_lambda length must be K = edges.n_elem - 1");
        }

        // Cache the edges as a plain std::vector for cache-friendly binary search.
        edges_.assign(cfg_.edges.begin(), cfg_.edges.end());
    }

    // ---- block_sampler interface ----------------------------------------

    void set_context(const block_context& ctx) override {
        // COPY (per design contract).
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // ---- fetch required data ----
        auto it_t = context_.find(cfg_.time_key);
        auto it_d = context_.find(cfg_.event_key);
        if (it_t == context_.end()) {
            throw std::runtime_error(
                "piecewise_exponential_gibbs_block '" + cfg_.name
                + "': shared_data key '" + cfg_.time_key + "' (time) not found in context");
        }
        if (it_d == context_.end()) {
            throw std::runtime_error(
                "piecewise_exponential_gibbs_block '" + cfg_.name
                + "': shared_data key '" + cfg_.event_key + "' (event) not found in context");
        }
        const arma::vec& t     = it_t->second;
        const arma::vec& delta = it_d->second;
        const std::size_t n = t.n_elem;
        if (delta.n_elem != n) {
            throw std::runtime_error(
                "piecewise_exponential_gibbs_block '" + cfg_.name
                + "': time and event vectors have inconsistent lengths");
        }
        if (n == 0) {
            throw std::runtime_error(
                "piecewise_exponential_gibbs_block '" + cfg_.name
                + "': empty data (n = 0)");
        }

        // ---- fetch optional keys (default: no offset, no truncation) ----
        const arma::vec* offset_ptr = nullptr;
        const arma::vec* entry_ptr  = nullptr;
        if (!cfg_.offset_key.empty()) {
            auto it_o = context_.find(cfg_.offset_key);
            if (it_o != context_.end()) {
                if (it_o->second.n_elem != n) {
                    throw std::runtime_error(
                        "piecewise_exponential_gibbs_block '" + cfg_.name
                        + "': offset '" + cfg_.offset_key
                        + "' has length " + std::to_string(it_o->second.n_elem)
                        + ", expected " + std::to_string(n));
                }
                offset_ptr = &it_o->second;
            }
        }
        if (!cfg_.entry_time_key.empty()) {
            auto it_v = context_.find(cfg_.entry_time_key);
            if (it_v != context_.end()) {
                if (it_v->second.n_elem != n) {
                    throw std::runtime_error(
                        "piecewise_exponential_gibbs_block '" + cfg_.name
                        + "': entry_time '" + cfg_.entry_time_key
                        + "' has length " + std::to_string(it_v->second.n_elem)
                        + ", expected " + std::to_string(n));
                }
                entry_ptr = &it_v->second;
            }
        }

        // ---- accumulate E_k (events in bin k) and R_k (weighted at-risk time) ----
        std::vector<double> E(K_, 0.0);
        std::vector<double> R(K_, 0.0);

        const double eK = edges_[K_];   // right edge of last bin
        for (std::size_t i = 0; i < n; ++i) {
            const double ti = t[i];
            const double vi = entry_ptr ? (*entry_ptr)[i] : 0.0;
            const double di = delta[i];
            const double wi = offset_ptr ? (*offset_ptr)[i] : 1.0;

            if (!(ti > 0.0) || !std::isfinite(ti)) {
                throw std::runtime_error(
                    "piecewise_exponential_gibbs_block '" + cfg_.name
                    + "': t[" + std::to_string(i) + "] must be > 0 (got " + std::to_string(ti) + ")");
            }
            if (ti > eK) {
                throw std::runtime_error(
                    "piecewise_exponential_gibbs_block '" + cfg_.name
                    + "': t[" + std::to_string(i) + "] = " + std::to_string(ti)
                    + " exceeds last edge edges[K] = " + std::to_string(eK)
                    + " (raise edges[K] to cover the full follow-up time)");
            }
            if (vi < 0.0 || vi > ti) {
                throw std::runtime_error(
                    "piecewise_exponential_gibbs_block '" + cfg_.name
                    + "': entry_time[" + std::to_string(i) + "] must satisfy 0 <= v <= t");
            }
            if (!(wi >= 0.0) || !std::isfinite(wi)) {
                throw std::runtime_error(
                    "piecewise_exponential_gibbs_block '" + cfg_.name
                    + "': offset[" + std::to_string(i) + "] must be finite and >= 0");
            }
            if (!(di == 0.0 || di == 1.0)) {
                throw std::runtime_error(
                    "piecewise_exponential_gibbs_block '" + cfg_.name
                    + "': delta[" + std::to_string(i) + "] must be 0 or 1 (got "
                    + std::to_string(di) + ")");
            }

            // Bin index k_i containing t_i (1-based math, 0-based storage).
            // Bins are (e_{k-1}, e_k]; find smallest k such that edges[k] >= t_i and edges[k-1] < t_i.
            // std::upper_bound returns first index with edges[index] > t_i, so
            // k_i (0-based) = upper_bound(...) - begin() - 1. For t_i = edges[k] exactly
            // (measure zero), t_i lands in bin k (0-based k-1).
            const auto ub = std::upper_bound(edges_.begin(), edges_.end(), ti);
            std::size_t k_i;
            if (ub == edges_.begin()) {
                // ti <= edges[0] -- since we required ti > 0 and edges[0] >= 0,
                // this can only happen if edges[0] > 0 and ti in (0, edges[0]].
                // We treat that as bin 0 (the first bin, implicitly extending to 0).
                k_i = 0;
            } else if (ub == edges_.end()) {
                // ti == edges[K] exactly (we already threw on ti > edges[K]).
                k_i = K_ - 1;
            } else {
                k_i = static_cast<std::size_t>(ub - edges_.begin()) - 1;
            }

            // Contribution to E_k
            if (di == 1.0) E[k_i] += 1.0;

            // Contribution to R_k: for each bin k, subject at-risk time is
            //   Delta_k = max(0, min(ti, edges[k+1]) - max(vi, edges[k])).
            // Since intervals are (edges[k], edges[k+1]], we allow subject to
            // accumulate risk time on this interval only where [vi, ti] overlaps it.
            for (std::size_t k = 0; k <= k_i; ++k) {
                const double lo = std::max(vi, edges_[k]);
                const double hi = std::min(ti, edges_[k + 1]);
                const double dur = hi - lo;
                if (dur > 0.0) R[k] += wi * dur;
            }
        }

        // ---- sample lambda_k ~ Gamma(a0 + E_k, b0 + R_k) ----
        for (std::size_t k = 0; k < K_; ++k) {
            const double shape_post = cfg_.a0 + E[k];
            const double rate_post  = cfg_.b0 + R[k];
            // COMMENTED PARAMETERISATION (gamma_gibbs_block style):
            //   Gamma(shape, RATE=rate). std::gamma_distribution takes (shape, SCALE),
            //   so pass scale = 1/rate. E[X] = shape / rate.
            std::gamma_distribution<double> gam(shape_post, 1.0 / rate_post);
            const double g = gam(rng);
            if (g > 0.0 && std::isfinite(g)) {
                values_[k] = g;
            }
            // Else: keep previous value (rare extreme-shape underflow).
        }

        if (keep_history_) {
            history_buf_.push_back(values_);
        }
    }

    const arma::vec& current() const override { return values_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != K_) {
            throw std::invalid_argument(
                "piecewise_exponential_gibbs_block::set_current: length must be K");
        }
        for (std::size_t k = 0; k < K_; ++k) {
            if (!(theta[k] > 0.0) || !std::isfinite(theta[k])) {
                throw std::invalid_argument(
                    "piecewise_exponential_gibbs_block::set_current: entries must be positive finite");
            }
        }
        values_ = theta;
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return K_; }

    // ---- History overrides ---------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_buf_, values_);
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override { history_buf_.clear(); }

private:
    piecewise_exponential_gibbs_block_config cfg_;
    std::size_t                              K_ = 0;
    std::vector<double>                      edges_;
    arma::vec                                values_;
    block_context                            context_;
    std::vector<arma::vec>                   history_buf_;
};

}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_PIECEWISE_EXPONENTIAL_GIBBS_BLOCK_HPP

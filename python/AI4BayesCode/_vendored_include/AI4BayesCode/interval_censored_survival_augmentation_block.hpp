/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v3.0 or later
 *  (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  interval_censored_survival_augmentation_block.hpp -- Tanner-Wong (1987) data
 *      augmentation for interval-censored survival data under a piecewise-
 *      exponential (PEH) hazard.  Samples each subject's latent event time
 *      T_i from the truncated PEH conditional
 *          T_i  ~  f_PEH( . | lambda, f_i )  restricted to  ( L_i, U_i ]
 *      given the CURRENT baseline hazard lambda[K] (sampled by a sibling
 *      piecewise_exponential_gibbs_block) and per-subject log-relative-hazard
 *      f_i (fixed or produced by another sibling block).
 *
 *  MOTIVATION (Ibrahim, Chen & Sinha 2001 "Bayesian Survival Analysis" Sec.3.5;
 *  Tanner & Wong 1987 JASA 82:528-540; Sinha, Chen & Ghosh 1999 Biometrics
 *  55(2):585; Lin & Wang 2010 Stat. Med. 29(29):3059-3072)
 *  =========================================================================
 *  When subjects are only known to have their event in a window [L_i, U_i]
 *  (typical of periodic-visit clinical trials, mouse-tumour experiments,
 *  epidemiological surveys), the observed-data likelihood involves an
 *  integral   int_{L_i}^{U_i} f(t) dt = S(L_i) - S(U_i)   which BREAKS the
 *  Gamma-conjugacy that makes PEH Gibbs cheap.  Data augmentation restores
 *  it: introduce the LATENT event time T_i as a parameter, and this block
 *  samples T_i from its truncated PEH conditional given the current lambda.
 *  Then the sibling `piecewise_exponential_gibbs_block` treats T_i as
 *  observed (delta_i = 1) and re-samples lambda under the usual conjugate
 *  Gamma-Poisson step.  The combined 2-block sweep is an exact
 *  MCMC on the joint (T_latent, lambda) posterior; the marginal on lambda
 *  is the interval-censored posterior.
 *
 *  Right-censoring (U_i = +Inf) is a SUPPORTED SPECIAL CASE -- the truncated
 *  conditional degenerates to  T_i > L_i  and the block samples from the
 *  right tail  S(L_i) * (1 - u).  Exactly-observed times must be handled by
 *  the caller EXCLUDING them from this block (the sibling PEH block reads
 *  them directly).
 *
 *  ALGORITHM (inverse-CDF; exact, no rejection)
 *  =============================================
 *  Let w_i = exp(f_i) (per-subject frailty; 1 if no covariates supplied).
 *  Baseline cumulative hazard under piecewise-constant lambda:
 *      H_0(t) = sum_{j: e_j <= t} lambda_j * (e_j - e_{j-1})
 *             + lambda_{k(t)} * (t - e_{k(t)-1})
 *  Survival:
 *      S_i(t) = exp( -w_i * H_0(t) )
 *  Cumulative hazards at edges are precomputed once per step():
 *      cumH[k] = sum_{j=1..k} lambda_j * (e_j - e_{j-1}) for k = 0..K.
 *
 *  Truncated CDF F_i(t | L < T <= U) = [ S_i(L) - S_i(t) ] / [ S_i(L) - S_i(U) ].
 *  Given u ~ Uniform(0,1):
 *      S_i(t) = S_i(L) - u * ( S_i(L) - S_i(U) )
 *  =>  w_i * H_0(t) = -log(S_i(t))
 *  =>  find bin k where cumH[k-1] * w_i <= w_i * H_0(t) <= cumH[k] * w_i,
 *      then invert LINEARLY within bin:
 *          t = e_{k-1} + ( H_0(t) - cumH[k-1] ) / lambda_k
 *
 *  Right-censored (U = +Inf) case:
 *      S_i(U) = 0  =>  S_i(t) = S_i(L) * (1 - u)
 *      Same inversion.
 *
 *  Numerical guards:
 *   - If S_i(L) - S_i(U) < eps (interval too narrow to sample sensibly, e.g.
 *     because H_0 hasn't moved), keep the previous T_i.
 *   - If lambda_k is exactly 0 (extreme prior draw), inversion within-bin is
 *     ill-defined; keep the previous T_i.
 *
 *  USAGE (piecewise-exponential model with interval-censored data)
 *  ---------------------------------------------------------------
 *      // Shared data provides "L", "U" (length n; U_i = +Inf for right-censored),
 *      // initial "t" values in (L_i, U_i], "lambda" (length K, sampled by sibling PEH),
 *      // and optionally "exp_lp" (length n, sampled by sibling regression block).
 *      interval_censored_survival_augmentation_block_config cfg;
 *      cfg.name    = "t";                      // writes updated latent times
 *      cfg.edges   = edges;                    // same as sibling PEH block
 *      // L_key="L", U_key="U", lambda_key="lambda" (defaults); offset_key="" (baseline-only)
 *      cfg.initial_times = t0;                 // t0[i] in (L[i], U[i]]
 *      composite->add_child(std::make_unique<interval_censored_survival_augmentation_block>(std::move(cfg)));
 *
 *  AI RULE COMPLIANCE
 *  ==================
 *  - Exact inverse-CDF augmentation -> no gradient, no autodiff, no accept/reject.
 *  - Standard block_sampler contract.
 *  - Output stored as arma::vec of length n (imputed T_i) under `name`.
 *================================================================================*/

#ifndef AI4BAYESCODE_INTERVAL_CENSORED_SURVIVAL_AUGMENTATION_BLOCK_HPP
#define AI4BAYESCODE_INTERVAL_CENSORED_SURVIVAL_AUGMENTATION_BLOCK_HPP

#include "block_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/// Configuration for interval_censored_survival_augmentation_block.
struct interval_censored_survival_augmentation_block_config {
    /// Unique name within the composite; also the shared_data + history key
    /// under which this block writes the length-n imputed T_latent vector.
    std::string name;

    /// Bin edges, length K+1, strictly increasing, edges[0] >= 0.
    /// MUST match the sibling piecewise_exponential_gibbs_block's edges.
    /// edges[K] must exceed all observed / imputable times (throws otherwise).
    arma::vec edges;

    /// shared_data key for length-n lower interval bounds L_i >= 0.
    std::string L_key = "L";

    /// shared_data key for length-n upper interval bounds U_i > L_i.
    /// Use std::numeric_limits<double>::infinity() for right-censored.
    std::string U_key = "U";

    /// shared_data key for the current length-K baseline hazard rates lambda_k.
    /// Sampled each sweep by a sibling piecewise_exponential_gibbs_block.
    std::string lambda_key = "lambda";

    /// shared_data key for length-n log-relative-hazard offset exp(f_i).
    /// OPTIONAL: "" or missing => all exp(f_i) = 1.
    std::string offset_key = "";

    /// Initial imputed times, length n. Each entry MUST satisfy
    /// L_i < initial_times[i] <= U_i (checked once at first step()).
    arma::vec initial_times;
};

class interval_censored_survival_augmentation_block : public block_sampler {
public:
    explicit interval_censored_survival_augmentation_block(
        interval_censored_survival_augmentation_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "interval_censored_survival_augmentation_block: name must be non-empty");
        }
        if (cfg_.edges.n_elem < 2) {
            throw std::invalid_argument(
                "interval_censored_survival_augmentation_block: edges must have length >= 2");
        }
        K_ = cfg_.edges.n_elem - 1;
        if (!(cfg_.edges[0] >= 0.0)) {
            throw std::invalid_argument(
                "interval_censored_survival_augmentation_block: edges[0] must be >= 0");
        }
        for (std::size_t k = 0; k < K_; ++k) {
            if (!(cfg_.edges[k + 1] > cfg_.edges[k])) {
                throw std::invalid_argument(
                    "interval_censored_survival_augmentation_block: edges must be strictly increasing");
            }
        }
        if (cfg_.initial_times.n_elem == 0) {
            throw std::invalid_argument(
                "interval_censored_survival_augmentation_block: initial_times must be provided");
        }
        edges_.assign(cfg_.edges.begin(), cfg_.edges.end());
        values_  = cfg_.initial_times;   // will be validated in first step()
        first_step_ = true;
    }

    // ---- block_sampler interface ----------------------------------------

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // ---- fetch required data ----
        auto it_L = context_.find(cfg_.L_key);
        auto it_U = context_.find(cfg_.U_key);
        auto it_lambda = context_.find(cfg_.lambda_key);
        if (it_L == context_.end()) {
            throw std::runtime_error("interval_censored_survival_augmentation_block '"
                + cfg_.name + "': L key '" + cfg_.L_key + "' not found in context");
        }
        if (it_U == context_.end()) {
            throw std::runtime_error("interval_censored_survival_augmentation_block '"
                + cfg_.name + "': U key '" + cfg_.U_key + "' not found in context");
        }
        if (it_lambda == context_.end()) {
            throw std::runtime_error("interval_censored_survival_augmentation_block '"
                + cfg_.name + "': lambda key '" + cfg_.lambda_key + "' not found in context");
        }

        const arma::vec& L      = it_L->second;
        const arma::vec& U      = it_U->second;
        const arma::vec& lambda = it_lambda->second;
        const std::size_t n = L.n_elem;
        if (U.n_elem != n) {
            throw std::runtime_error("interval_censored_survival_augmentation_block '"
                + cfg_.name + "': L and U have inconsistent lengths");
        }
        if (values_.n_elem != n) {
            throw std::runtime_error("interval_censored_survival_augmentation_block '"
                + cfg_.name + "': initial_times length must equal n");
        }
        if (lambda.n_elem != K_) {
            throw std::runtime_error("interval_censored_survival_augmentation_block '"
                + cfg_.name + "': lambda length " + std::to_string(lambda.n_elem)
                + " does not match K = " + std::to_string(K_));
        }

        const arma::vec* offset_ptr = nullptr;
        if (!cfg_.offset_key.empty()) {
            auto it_o = context_.find(cfg_.offset_key);
            if (it_o != context_.end()) {
                if (it_o->second.n_elem != n) {
                    throw std::runtime_error("interval_censored_survival_augmentation_block '"
                        + cfg_.name + "': offset length mismatch");
                }
                offset_ptr = &it_o->second;
            }
        }

        // ---- one-time sanity: initial_times[i] in (L[i], U[i]] ----
        if (first_step_) {
            first_step_ = false;
            for (std::size_t i = 0; i < n; ++i) {
                if (!(L[i] >= 0.0) || !(U[i] > L[i])) {
                    throw std::runtime_error("interval_censored_survival_augmentation_block '"
                        + cfg_.name + "': subject " + std::to_string(i)
                        + " must satisfy 0 <= L < U (got L=" + std::to_string(L[i])
                        + " U=" + std::to_string(U[i]) + ")");
                }
                if (!(values_[i] > L[i]) || (std::isfinite(U[i]) && !(values_[i] <= U[i]))) {
                    throw std::runtime_error("interval_censored_survival_augmentation_block '"
                        + cfg_.name + "': initial_times[" + std::to_string(i)
                        + "]=" + std::to_string(values_[i]) + " must lie in (L, U]");
                }
                const double u_or_v = std::isfinite(U[i]) ? U[i] : values_[i];
                if (u_or_v > edges_[K_]) {
                    throw std::runtime_error("interval_censored_survival_augmentation_block '"
                        + cfg_.name + "': subject " + std::to_string(i)
                        + " has t or U exceeding edges[K] = " + std::to_string(edges_[K_]));
                }
            }
        }

        // ---- precompute cumH[k] = sum_{j=1..k} lambda_j * (e_j - e_{j-1}) ----
        std::vector<double> cumH(K_ + 1, 0.0);
        for (std::size_t k = 0; k < K_; ++k) {
            if (!(lambda[k] > 0.0) || !std::isfinite(lambda[k])) {
                // Bad lambda -- skip update, keep previous values.
                if (keep_history_) history_buf_.push_back(values_);
                return;
            }
            cumH[k + 1] = cumH[k] + lambda[k] * (edges_[k + 1] - edges_[k]);
        }

        // ---- per-subject truncated PEH sample via inverse CDF ----
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        const double eps = 1e-300;
        for (std::size_t i = 0; i < n; ++i) {
            const double Li = L[i];
            const double Ui = U[i];
            const double wi = offset_ptr ? (*offset_ptr)[i] : 1.0;
            if (!(wi > 0.0) || !std::isfinite(wi)) continue;  // skip subject silently

            // H_0(L) and H_0(U); if U = +inf, H_0(U) = +inf, S(U) = 0.
            const double HL = piecewise_H0(Li, cumH, lambda);
            const bool U_finite = std::isfinite(Ui);
            const double HU = U_finite ? piecewise_H0(Ui, cumH, lambda)
                                       : std::numeric_limits<double>::infinity();

            // S(L) = exp(-wi * HL); S(U) = exp(-wi * HU) or 0.
            // Draw target S(t) = S(L) - u * (S(L) - S(U)).
            // Work in log-space to avoid underflow: use "increments in wi * H".
            // Let dLU = wi * (HU - HL) >= 0. Then
            //   1 - S(U)/S(L) = 1 - exp(-dLU).
            //   S(t) = S(L) * [ 1 - u * (1 - exp(-dLU)) ]
            //   wi * H_0(t) = wi * HL - log( 1 - u * (1 - exp(-dLU)) )
            const double dLU = wi * (HU - HL);
            // dLU is either finite positive or +infinity (right-censored U=+inf case);
            // both are legitimate. Only skip NaN / zero / negative (degenerate).
            if (!(dLU > 0.0) || std::isnan(dLU)) {
                continue;
            }
            const double u   = unif(rng);
            // one_minus_expmdLU: 1 - exp(-dLU); handle U = +inf (dLU = inf) -> 1.
            const double one_minus_e = std::isinf(dLU) ? 1.0 : -std::expm1(-dLU);
            const double arg = 1.0 - u * one_minus_e;
            if (!(arg > eps)) continue;  // guard against log(0) -- keep previous
            const double H_t = HL + (-std::log(arg)) / wi;

            // Invert H_0(t) = H_t by locating the bin.
            // Find the largest k such that cumH[k] <= H_t.
            const auto ub = std::upper_bound(cumH.begin(), cumH.end(), H_t);
            std::size_t k;
            if (ub == cumH.begin()) {
                // H_t <= cumH[0] = 0; not possible if L >= 0 and H_t > HL >= 0 unless H_t == 0.
                k = 0;
            } else if (ub == cumH.end()) {
                // H_t >= cumH[K]; would give t >= edges[K]. Clip to just below edges[K].
                k = K_ - 1;
            } else {
                k = static_cast<std::size_t>(ub - cumH.begin()) - 1;
            }
            if (!(lambda[k] > 0.0)) continue;  // degenerate bin, keep previous
            const double t_new = edges_[k] + (H_t - cumH[k]) / lambda[k];

            // Numerical safety: keep in (L, U].
            double t_final = t_new;
            if (t_final <= Li) t_final = std::nextafter(Li, std::numeric_limits<double>::infinity());
            if (U_finite && t_final > Ui) t_final = Ui;
            if (t_final > edges_[K_]) t_final = edges_[K_];
            if (std::isfinite(t_final) && t_final > 0.0) values_[i] = t_final;
            // else keep previous
        }

        if (keep_history_) history_buf_.push_back(values_);
    }

    const arma::vec& current() const override { return values_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != values_.n_elem) {
            throw std::invalid_argument(
                "interval_censored_survival_augmentation_block::set_current: wrong length");
        }
        values_ = theta;
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return values_.n_elem; }

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_buf_, values_);
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override { history_buf_.clear(); }

private:
    // Piecewise cumulative hazard at time t.
    double piecewise_H0(double t,
                        const std::vector<double>& cumH,
                        const arma::vec& lambda) const
    {
        if (!(t > 0.0)) return 0.0;
        // Bin containing t.
        const auto ub = std::upper_bound(edges_.begin(), edges_.end(), t);
        std::size_t k;
        if (ub == edges_.begin())      k = 0;
        else if (ub == edges_.end())   k = K_ - 1;
        else                            k = static_cast<std::size_t>(ub - edges_.begin()) - 1;
        return cumH[k] + lambda[k] * (t - edges_[k]);
    }

    interval_censored_survival_augmentation_block_config cfg_;
    std::size_t                                          K_ = 0;
    std::vector<double>                                  edges_;
    arma::vec                                            values_;
    block_context                                        context_;
    std::vector<arma::vec>                               history_buf_;
    bool                                                 first_step_ = true;
};

}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_INTERVAL_CENSORED_SURVIVAL_AUGMENTATION_BLOCK_HPP

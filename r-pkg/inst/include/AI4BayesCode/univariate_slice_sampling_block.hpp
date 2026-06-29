/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  univariate_slice_sampling_block.hpp -- Neal 2003 slice sampling,
 *      univariate (1-D scalar) variant with stepping-out + shrinkage.
 *
 *  NAME DISAMBIGUATION
 *  ===================
 *  "Slice sampling" in the MCMC literature (Neal 2003, Annals of Statistics
 *  31(3):705-767) is a family of algorithms operating on an augmented
 *  (x, y) space where y is a "slice height" auxiliary variable. This file
 *  implements ONLY the univariate variant with stepping-out + shrinkage
 *  (Neal 2003 section 4.1). It does NOT cover:
 *    - Multivariate slice sampling (section 5)
 *    - Doubling procedure for interval expansion (section 4.2)
 *    - Overrelaxed slice sampling (section 6)
 *    - Reflective slice sampling
 *    - Elliptical slice sampling (Murray-Adams-MacKay 2010 -- see
 *      elliptical_slice_sampling_block.hpp)
 *  Future variants can be added as separate blocks.
 *
 *  ALGORITHM (Neal 2003 section 4.1, with random step-out split for
 *  reversibility per Neal 2003 eq. 5)
 *  ====================================================================
 *  On the UNCONSTRAINED scale, given current x_cur and log p(x_cur):
 *
 *    (1) Slice height:
 *          log_y = log p(x_cur) + log(U_1),  U_1 ~ Uniform(0, 1)
 *
 *    (2) Initial bracket [L, R] of width w around x_cur:
 *          L = x_cur - U_2 * w,  R = L + w,  U_2 ~ Uniform(0, 1)
 *
 *    (3) Random budget split (m = max_step_out_iter total):
 *          J = floor(U_3 * m),  K = m - 1 - J,  U_3 ~ Uniform(0, 1)
 *
 *    (4) Stepping-out (reversible thanks to the random split):
 *          while J > 0 and log p(L) > log_y:  L -= w, J -= 1
 *          while K > 0 and log p(R) > log_y:  R += w, K -= 1
 *
 *    (5) Shrinkage:
 *          repeat up to max_shrink_iter:
 *            x_prop ~ Uniform(L, R)
 *            if log p(x_prop) > log_y: accept x_prop, return
 *            else shrink: if x_prop < x_cur: L = x_prop
 *                         else:               R = x_prop
 *
 *  Guaranteed to accept within finite shrinkage iterations (bracket
 *  converges to x_cur). Detailed balance proof in Neal 2003 section 4.1.
 *
 *  CONSTRAINT HANDLING
 *  ===================
 *  Mirrors nuts_block API: the user supplies (constrain, unconstrain)
 *  pair, leaves them empty for identity, or uses
 *  constraints::<kind>::constrain etc. Slice sampling runs on the
 *  UNCONSTRAINED scale; the user's log_density lambda is expected to
 *  return the UNCONSTRAINED-scale log-density (natural-scale lp plus
 *  log|J| from the unconstraining transform), identical to the
 *  discipline in nuts_block's log_density_grad. The canonical pattern
 *  uses constraints::<kind>::wrap(theta_unc, nullptr, inner) to obtain
 *  the unconstrained-scale lp without writing any Jacobian by hand.
 *
 *  WHEN TO USE vs nuts_block
 *  =========================
 *  ALWAYS prefer nuts_block when log p is differentiable AND the
 *  gradient is feasible to write (hand-derived or via autodiff). NUTS
 *  mixes better and is the default for smooth continuous targets.
 *
 *  Use univariate_slice_sampling_block only when NUTS cannot:
 *    (a) log p is non-differentiable (piecewise, kink, floor/ceil),
 *    (b) log p is a black-box library call whose gradient would require
 *        re-implementing that library with autodiff::var types (e.g.,
 *        celerite marginal log-likelihood via
 *        celerite_marginal_likelihood.hpp),
 *    (c) the gradient is prohibitively expensive relative to lp eval.
 *  See skills/codegen.md section 2b.1 for the decision tree.
 *
 *  AI-SAFETY PROFILE
 *  =================
 *  Slice sampling is in the SAME safety category as NUTS (NOT Gibbs):
 *  the user writes ONLY a log-density lambda. No conditional-posterior
 *  derivation (which is the Gibbs risk) is required. The algorithmic
 *  machinery (stepping-out + shrinkage) is Neal 2003 textbook,
 *  implemented once here and validated by the Check #15 parity test at
 *    tests_autodiff/block_tests/test_univariate_slice_sampling_block.cpp
 *  (three fixtures: Normal via identity, Gamma via positive, Beta via
 *  interval; 10k draws each, mean/variance within 5%/10% of analytical).
 *
 *  JUSTIFICATION (Check #16): Exception 1 from codegen.md section 2b --
 *  specialized sampler for 1-D continuous parameters whose log-density
 *  lacks an accessible gradient (violating NUTS's prerequisite).
 *================================================================================*/

#ifndef AI4BAYESCODE_UNIVARIATE_SLICE_SAMPLING_BLOCK_HPP
#define AI4BAYESCODE_UNIVARIATE_SLICE_SAMPLING_BLOCK_HPP

#include "block_sampler.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/// Optional unconstrained<->natural transform matching nuts_block API.
/// Identity when left empty in the config.
using slice_transform_fn = std::function<arma::vec(const arma::vec&)>;

/**
 * @brief Configuration bundle for univariate_slice_sampling_block.
 *
 * Parallels nuts_block_config for easy substitution. The user writes
 * an UNCONSTRAINED-SCALE log-density lambda (returns lp only, no
 * gradient); the canonical pattern is:
 *
 *     cfg.log_density = [](const arma::vec& theta_unc,
 *                          const block_context& ctx) -> double {
 *         return constraints::positive::wrap(theta_unc, nullptr,
 *             [&](const arma::vec& theta_nat, arma::vec* /unused/) -> double {
 *                 // natural-scale lp, NO Jacobian (wrap adds it)
 *                 return lp;
 *             });
 *     };
 *
 * which reuses the existing constraints::<kind>::wrap machinery. The
 * unused grad_nat pointer may be ignored.
 *
 * Strict univariate scope: initial_unc must have n_elem == 1. Passing
 * a longer vector throws std::invalid_argument.
 */
struct univariate_slice_sampling_block_config {
    /// Unique name for this block within its composite. Also the key
    /// under which composite_block stores this block's current value.
    std::string name;

    /// Initial value on the UNCONSTRAINED scale. Must be length 1.
    arma::vec initial_unc;

    /// UNCONSTRAINED-scale log-density oracle. Must already include
    /// log|J| of any unconstraining transform (produce via
    /// constraints::<kind>::wrap). Required.
    std::function<double(const arma::vec&, const block_context&)>
        log_density;

    /// Map unconstrained -> natural. Defaults to identity if left empty.
    slice_transform_fn constrain;

    /// Map natural -> unconstrained. Defaults to identity if left empty.
    /// Only needed if set_current will be called with a natural-scale
    /// value.
    slice_transform_fn unconstrain;

    /// Initial slice-bracket width on the UNCONSTRAINED scale. Default
    /// 1.0 is appropriate for most log-scale parameters. Stepping-out
    /// adapts the effective bracket at each sweep; w mainly controls
    /// the per-step computational cost. A too-small w means more
    /// stepping-out iterations; too-large means more shrinkage iters.
    double w = 1.0;

    /// Total stepping-out budget (Neal 2003 "m" parameter). Randomly
    /// partitioned between the two sides at each step for reversibility.
    /// 50 is ample for any reasonable 1-D log-scale posterior.
    std::size_t max_step_out_iter = 50;

    /// Safety cap on shrinkage iterations. 100 is the standard Neal 2003
    /// recommendation; hitting this cap in practice usually means
    /// log_density is returning NaN, not legitimate slow mixing.
    std::size_t max_shrink_iter = 100;
};

/**
 * @brief Univariate slice sampler (Neal 2003 section 4.1) as a block_sampler.
 *
 * See file header for the algorithm description, API contract, and
 * guidance on when to use this over nuts_block.
 */
class univariate_slice_sampling_block : public block_sampler {
public:
    explicit univariate_slice_sampling_block(
        univariate_slice_sampling_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "univariate_slice_sampling_block: name must be non-empty");
        }
        if (cfg_.initial_unc.n_elem != 1) {
            throw std::invalid_argument(
                "univariate_slice_sampling_block: initial_unc must have "
                "n_elem == 1 (this block is strictly univariate; use "
                "nuts_block or another variant for multi-dim parameters)");
        }
        if (!cfg_.log_density) {
            throw std::invalid_argument(
                "univariate_slice_sampling_block: log_density is required");
        }
        if (!cfg_.constrain) {
            cfg_.constrain = [](const arma::vec& x) { return x; };
        }
        if (!cfg_.unconstrain) {
            cfg_.unconstrain = [](const arma::vec& x) { return x; };
        }
        if (!(cfg_.w > 0.0)) {
            throw std::invalid_argument(
                "univariate_slice_sampling_block: w must be positive");
        }
        theta_unc_ = cfg_.initial_unc;
        theta_nat_ = cfg_.constrain(theta_unc_);
    }

    // ---- block_sampler interface --------------------------------------

    void set_context(const block_context& ctx) override {
        // COPY per block_sampler contract (no retained pointers into ctx).
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        std::uniform_real_distribution<double> unif(0.0, 1.0);

        // Evaluate current log-density on unconstrained scale.
        const double lp_cur = cfg_.log_density(theta_unc_, context_);
        if (!std::isfinite(lp_cur)) {
            // Pathological start -- leave state unchanged; caller should
            // inspect initial_unc / log_density.
            record_history_();
            return;
        }

        // Step (1) slice height on log scale:
        //   log_y = log p(x_cur) + log U_1,  U_1 ~ Uniform(0, 1)
        const double log_y = lp_cur + std::log(unif(rng));

        // Step (2) initial bracket [L, R] of width w around x_cur.
        const double x_cur = theta_unc_[0];
        const double u2 = unif(rng);
        double L = x_cur - u2 * cfg_.w;
        double R = L + cfg_.w;

        // Step (3) random split of the step-out budget (Neal 2003 eq 5).
        // Reversibility of the full slice move requires that the budget
        // be randomly partitioned; without this the chain is biased.
        const std::size_t m = cfg_.max_step_out_iter;
        std::size_t j_rem = 0;
        std::size_t k_rem = 0;
        if (m > 0) {
            const double u3 = unif(rng);
            const std::size_t J =
                static_cast<std::size_t>(std::floor(u3 * static_cast<double>(m)));
            j_rem = J;
            k_rem = (m >= 1 && J <= m - 1) ? (m - 1 - J) : 0;
        }

        // Step (4) stepping-out (left and right, each bounded by its
        // randomly-assigned remaining budget).
        {
            arma::vec th(1);
            th[0] = L;
            while (j_rem > 0) {
                const double lp_L = cfg_.log_density(th, context_);
                if (!std::isfinite(lp_L) || lp_L <= log_y) break;
                L -= cfg_.w;
                th[0] = L;
                --j_rem;
            }
            th[0] = R;
            while (k_rem > 0) {
                const double lp_R = cfg_.log_density(th, context_);
                if (!std::isfinite(lp_R) || lp_R <= log_y) break;
                R += cfg_.w;
                th[0] = R;
                --k_rem;
            }
        }

        // Step (5) shrinkage.
        arma::vec th_prop(1);
        for (std::size_t iter = 0; iter < cfg_.max_shrink_iter; ++iter) {
            std::uniform_real_distribution<double> unif_LR(L, R);
            const double x_prop = unif_LR(rng);
            th_prop[0] = x_prop;
            const double lp_prop = cfg_.log_density(th_prop, context_);
            if (std::isfinite(lp_prop) && lp_prop > log_y) {
                // Accept
                theta_unc_[0] = x_prop;
                theta_nat_ = cfg_.constrain(theta_unc_);
                record_history_();
                return;
            }
            // Shrink the bracket side containing x_prop toward x_cur.
            if (x_prop < x_cur) L = x_prop;
            else                 R = x_prop;
            // Numerical safety: if the bracket collapses to machine
            // precision we cannot make further progress. This is
            // extraordinarily rare -- typically only when log_density
            // produces NaN. Accept current state and return.
            if (R - L < 1e-12) {
                record_history_();
                return;
            }
        }
        // Shrinkage cap reached -- leave state unchanged. This should
        // never happen for well-behaved log_density; treat as a soft
        // warning (log_density probably returning NaN or -Inf too
        // aggressively).
        record_history_();
    }

    const arma::vec& current() const override { return theta_nat_; }

    void set_current(const arma::vec& theta_natural) override {
        if (theta_natural.n_elem != 1) {
            throw std::invalid_argument(
                "univariate_slice_sampling_block::set_current: length "
                "must be 1");
        }
        theta_nat_ = theta_natural;
        theta_unc_ = cfg_.unconstrain(theta_natural);
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return 1; }

    /// Unconstrained-scale current value (for diagnostics / tests).
    const arma::vec& current_unconstrained() const noexcept {
        return theta_unc_;
    }

    // ---- History ------------------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_, theta_nat_);
    }

    std::size_t history_size() const noexcept override {
        return history_.empty() ? 1 : history_.size();
    }

    void clear_history() override { history_.clear(); }

private:
    void record_history_() {
        if (keep_history_) history_.push_back(theta_nat_);
    }

    univariate_slice_sampling_block_config cfg_;
    arma::vec                              theta_unc_;
    arma::vec                              theta_nat_;
    block_context                          context_;
    std::vector<arma::vec>                 history_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_UNIVARIATE_SLICE_SAMPLING_BLOCK_HPP

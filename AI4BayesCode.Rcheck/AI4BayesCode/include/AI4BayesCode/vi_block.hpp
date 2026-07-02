/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  vi_block.hpp  --  abstract base class for variational-inference blocks.
 *
 *  See system_design.md §18 for the full VI architectural backing.
 *  Two concrete subclasses ship in v1:
 *    - mean_field_gaussian_vi_block  (primary; q = prod_i N(mu_i, sigma_i^2))
 *    - full_rank_gaussian_vi_block   (opt-in; q = N(mu, LL^T))
 *
 *  KEY CONTRACT differences from a generic block_sampler (§18.3):
 *
 *  - engine_kind() returns engine_kind_t::VI. composite_block uses this
 *    to route post-step writes through current_sample(rng), preserving
 *    the VI block's uncertainty for any MCMC sibling that reads it in
 *    hybrid mode.
 *
 *  - step(rng) runs ONE RAABBVI optimizer step (Welandawe 2022),
 *    NOT a sampling step. The rng is used inside step() for the
 *    reparameterization draw eps ~ N(0, I).
 *
 *  - current() returns the q-mean E_q[theta] = constrain(mu) — a
 *    deterministic point estimate. R-level get_current() returns this
 *    for VI children.
 *
 *  - current_sample(rng) is the NEW method specific to vi_block.
 *    It draws eta ~ q(eta; lambda), then theta = constrain(eta), and
 *    returns theta. composite_block calls this (via the rng overload
 *    of current_named_outputs) after each step() to seed shared_data
 *    with a fresh q-sample.
 *
 *  - set_current(mu) overwrites the variational mean ONLY; log_sd is
 *    left intact. To overwrite both, call
 *    set_variational_state(mu, log_sd) directly.
 *
 *  - get_history() (via the Tier A wrapper) returns per-step
 *    (elbo, mu, log_sd, gamma, epoch) PLUS a scalar final_khat filled
 *    at SKL termination. NOT posterior draws — q is changing during
 *    optimization, so iterates are NOT samples from the same
 *    distribution. See vi_history_t below.
 *
 *  HYBRID-CORRECTNESS INVARIANT
 *  ----------------------------
 *  In hybrid mode (composite has a mix of MCMC and VI children), the
 *  post-step write of a VI child's value into shared_data must be a
 *  q-SAMPLE (current_sample(rng)), NEVER the q-mean (current()).
 *  Writing q-mean silently underestimates the downstream MCMC
 *  sibling's posterior variance — the MCMC block conditions on a
 *  point estimate of the VI'd parameter instead of integrating over
 *  q. See system_design.md §18.4.
 *
 *  This invariant is enforced by overriding current_named_outputs(rng)
 *  here to route through current_sample(rng); composite_block always
 *  calls the rng overload during step(). Subclasses that override
 *  current_named_outputs(rng) themselves MUST preserve the q-sample
 *  semantics — see Check #21(b) in validator.md.
 *================================================================================*/

#ifndef AI4BAYESCODE_VI_BLOCK_HPP
#define AI4BAYESCODE_VI_BLOCK_HPP

#include "block_sampler.hpp"

#include <cmath>     // std::isnan
#include <limits>    // std::numeric_limits
#include <random>
#include <string>
#include <vector>

#ifndef MCMC_USE_RCPP_ARMADILLO
# include <armadillo>
#else
# include <RcppArmadillo.h>
#endif

namespace AI4BayesCode {

/**
 * @brief Per-step VI optimization-history record.
 *
 * One entry per optimizer step when keep_history is enabled.
 * final_khat is filled (only at the SKL-termination step) with the
 * joint PSIS-k̂ computed from S samples drawn from q at the
 * converged lambda. Used by validator Layer-3 R2-VI (Check #23).
 * NaN before termination.
 *
 * Fields:
 *   elbo[t]    -- ELBO at step t (negative free energy)
 *   mu[t]      -- variational mean at step t (length K, unconstrained scale)
 *   log_sd[t]  -- variational log standard deviation at step t
 *                 (length K for mean-field; vec_lower(L) for full-rank
 *                  packed with log-diag at the diagonal positions)
 *   gamma[t]   -- avgAdam learning rate at step t
 *   epoch[t]   -- which fixed-gamma epoch step t belongs to
 *                 (RAABBVI shrinks gamma between epochs; this tracks
 *                  which iterate-averaging window the step falls in)
 *   final_khat -- joint PSIS-k̂ at the SKL-termination step; NaN
 *                 until terminate fires
 *
 * NOTE: this history is NOT a posterior draw. q is changing as the
 * optimizer iterates. Iterates are samples from a SEQUENCE of
 * distributions, not from one fixed distribution; concatenating them
 * is not a valid Monte Carlo sample. For posterior expectations,
 * use predict_at + current_sample at the CONVERGED lambda (and
 * PSIS-reweight per Dhaka 2021).
 */
struct vi_history_t {
    std::vector<double>    elbo;
    std::vector<arma::vec> mu;
    std::vector<arma::vec> log_sd;
    std::vector<double>    gamma;
    std::vector<int>       epoch;
    double                 final_khat = std::numeric_limits<double>::quiet_NaN();
};

/**
 * @brief Abstract base class for variational-inference blocks.
 *
 * v1 ships two concrete subclasses (Phase 3, Phase 4):
 *   - mean_field_gaussian_vi_block  -- primary; q = prod_i N(mu_i, sigma_i^2)
 *   - full_rank_gaussian_vi_block   -- opt-in;  q = N(mu, LL^T)
 *
 * Both maintain variational parameters on the UNCONSTRAINED scale
 * (eta = constraints::unconstrain(theta)) and reconstruct theta via
 * constraints::constrain in current() and current_sample(). The
 * user's log-density lambda is identical to what nuts_block expects:
 * natural-scale log p, natural-scale gradient, with no hand-written
 * Jacobian (constraints::*::wrap handles |J|).
 *
 * Composite_block uses engine_kind() == VI to write q-samples (not
 * q-means) into shared_data after each step. The mechanism is the
 * overridden current_named_outputs(std::mt19937_64&) below, which
 * routes through current_sample(rng). The hybrid-correctness
 * invariant is enforced at this single location.
 */
class vi_block : public block_sampler {
public:
    // ---- engine_kind override --------------------------------------------

    /**
     * VI blocks announce themselves as VI engine. composite_block uses
     * this for dispatch decisions (currently informational —
     * current_named_outputs(rng) is the actual hot-path dispatch).
     */
    engine_kind_t engine_kind() const noexcept override {
        return engine_kind_t::VI;
    }

    // ---- VI-specific public interface ------------------------------------

    /**
     * Draw theta ~ q. composite_block calls this (via the rng overload
     * of current_named_outputs) after each step() to write a fresh
     * q-sample into shared_data.
     *
     * Implementation contract:
     *   - MUST NOT mutate the block's variational parameters or
     *     optimizer state. const-correct on purpose.
     *   - MAY advance @p rng (that is the only side effect).
     *   - Returns the NATURAL-scale theta (post-constrain), the same
     *     scale as current().
     *
     * For mean-field: draw eps ~ N(0, I); eta = mu + sigma * eps;
     *                 theta = constrain(eta).
     * For full-rank:  draw eps ~ N(0, I); eta = mu + L * eps;
     *                 theta = constrain(eta).
     */
    virtual arma::vec current_sample(std::mt19937_64& rng) const = 0;

    /**
     * Current variational log standard deviation (mean-field) or
     * a packed representation of the lower-triangular Cholesky factor
     * (full-rank). Used for diagnostics, R-level get_history() shape,
     * and any PSIS computation that needs to evaluate log q(eta).
     *
     * The exact shape is subclass-specific:
     *   - mean-field: arma::vec of length K (log sigma per coordinate)
     *   - full-rank:  arma::vec of length K(K+1)/2 packing the lower
     *                 triangle of L (log diagonal at the diagonal
     *                 positions; raw entries off-diagonal). The exact
     *                 packing layout is documented in the full-rank
     *                 block's header.
     */
    virtual arma::vec get_log_sd() const = 0;

    /**
     * Overwrite both variational mean AND log_sd at once. Tier A
     * dispatcher uses this when the user passes
     * list(<param>_mean = mu, <param>_log_sd = log_sd) to set_current.
     *
     * Dimensions must match the block's K (mean-field: log_sd is
     * length K; full-rank: log_sd is length K(K+1)/2). Mismatch
     * triggers an exception (subclass-specific message).
     */
    virtual void set_variational_state(const arma::vec& mu,
                                       const arma::vec& log_sd) = 0;

    /**
     * Last computed ELBO value (Welandawe 2022 §3). NaN before the
     * first step() call. Updated EVERY step() so users can monitor
     * convergence via current_elbo() in real time.
     */
    virtual double current_elbo() const = 0;

    /**
     * Full per-step optimization history. Empty when keep_history is
     * false; populated on every step() call when true. The final_khat
     * field is filled (once, at the SKL-termination step) by step()
     * itself; subclasses must compute joint PSIS-k̂ over S samples at
     * the converged lambda. See vi_optimizer.hpp / psis_diagnostic.hpp
     * for the helper utilities.
     */
    virtual const vi_history_t& vi_history() const = 0;

    // ---- override of current_named_outputs(rng) --------------------------

    /**
     * Override of the rng-aware variant. VI blocks write a q-SAMPLE
     * (not the q-mean) into shared_data after each step(). This is THE
     * hybrid-correctness invariant (system_design.md §18.4): an MCMC
     * sibling that reads this block's key during its own sweep must
     * see a fresh draw, not a point estimate.
     *
     * The rng-less variant (inherited default) still returns the
     * q-mean — kept for defensive use by code paths that haven't been
     * updated to call the rng-aware overload.
     */
    state_map current_named_outputs(std::mt19937_64& rng) const override {
        const std::string& nm = name();
        if (nm.empty()) return {};
        state_map out;
        out.emplace(nm, current_sample(rng));
        return out;
    }
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_VI_BLOCK_HPP

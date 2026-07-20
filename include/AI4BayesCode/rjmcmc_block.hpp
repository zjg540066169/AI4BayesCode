/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  rjmcmc_block.hpp  --  Reversible-Jump MCMC block for Dirac
 *                        spike-and-slab style variable selection
 *                        (and any other trans-dimensional model with
 *                        per-coefficient birth/death proposals).
 *
 *  WHY THIS BLOCK EXISTS
 *  =====================
 *  The Dirac spike-and-slab model
 *
 *      gamma_j  ~ Bernoulli(pi)
 *      beta_j | gamma_j = 0  =  0           (point mass / Dirac delta)
 *      beta_j | gamma_j = 1  ~  N(0, tau^2) (continuous slab)
 *
 *  has a *dimension-changing* state space: (gamma_j, beta_j) lives in
 *     { (0, 0) } cup { 1 } x R
 *
 *  — a single point glued to a line. This is not a fixed-dim manifold,
 *  and naive Gibbs on it is NOT irreducible (see skills/system_design.md
 *  §11.2 for the math). NUTS on the gamma-marginalized target is silently
 *  wrong (mixed Lebesgue + atomic measure; NUTS samples the slab only).
 *
 *  Reversible-Jump MCMC (Green 1995) is the correct algorithm: each step
 *  proposes a trans-dimensional move (birth of a new beta_j or death of
 *  an existing one) with a carefully-constructed bijection between the
 *  "off" and "on" state spaces. The MH accept ratio contains a Jacobian
 *  term |det d(x', u') / d(x, u)| in general.
 *
 *  THREE BIRTH-PROPOSAL FAMILIES (USERS NEVER WRITE A JACOBIAN)
 *  ------------------------------------------------------------
 *  (a) Identity-coordinate (default; no transform configured): the
 *      birth proposal draws beta_j_new directly from `propose_sample`,
 *      so the proposed beta_j IS the auxiliary variable u. The
 *      bijection is the identity and |det J| = 1 by construction.
 *      Covers ~80% of practical RJMCMC use cases — variable selection,
 *      change-point insertion with prior-sampled values, simple
 *      mixture-component birth/death. This path clones NIMBLE's
 *      `configureRJ` approach (de-facto best practice among live
 *      RJMCMC packages).
 *  (b) Library-provided 1D transforms (`rjmcmc_transforms.hpp`):
 *      identity / linear / affine transform classes. Each computes
 *      |det J| internally; the user sets `cfg.transform` to one of
 *      `identity_transform_1d`, `diagonal_linear_transform_1d(scale)`,
 *      `diagonal_affine_transform_1d(scale, offset)`.
 *  (c) Custom user-supplied bijection (`rjmcmc_custom_bijection.hpp`):
 *      for non-linear maps the previous two cannot fit, the user
 *      supplies one templated forward + an analytic inverse, and the
 *      framework computes |dbeta/du| via runtime autodiff.
 *
 *  In none of the three families does the user hand-write a Jacobian
 *  formula — see skills/system_design.md §10 for the universal rule.
 *
 *  STATE REPRESENTATION
 *  ====================
 *  Internally stores:
 *    gamma_ : arma::vec of length p, entries in {0.0, 1.0}
 *    beta_  : arma::vec of length p, with beta_[j] = 0 when gamma_[j]=0
 *
 *  The total state is fixed-dim (2p doubles). "Inactive" beta_j's are
 *  held at 0 by construction — get_current() always returns a length-2p
 *  concatenation [gamma; beta], and get_history() stores fixed-shape
 *  matrices. This sidesteps the variable-dim-history problem that would
 *  plague a naive dimension-changing block; the "dimension change"
 *  manifests only in the accept ratio, not in the storage.
 *
 *  current_named_outputs() splits the concatenated state into two keys
 *  (configurable names, default "gamma" and "beta") so downstream blocks
 *  can read them separately through the composite's shared_data.
 *
 *  PER-SWEEP ALGORITHM
 *  ===================
 *  One step() invocation does one full sweep. For j in a random
 *  permutation of {0, ..., p-1}:
 *
 *    (A) CONTINUOUS RW UPDATE (only if gamma_[j] == 1 and rw_scale > 0)
 *        Propose  beta_j_prop = beta_j_cur + N(0, rw_scale^2)
 *        MH ratio = exp( log_joint(gamma, beta_prop) - log_joint(gamma, beta_cur) )
 *        Accept / reject with Metropolis. No Jacobian (RW on same scale).
 *
 *    (B) REVERSIBLE-JUMP FLIP (always)
 *        If gamma_[j] == 0: propose BIRTH
 *           beta_j_new ~ propose_sample(rng, j, ctx)
 *           log_q_new  = propose_logq(beta_j_new, j, ctx)
 *           MH log-ratio = log_joint(gamma_new, beta_new)
 *                        - log_joint(gamma_cur, beta_cur)
 *                        - log_q_new
 *           (identity-coordinate Jacobian = 1, so log|J| = 0)
 *
 *        If gamma_[j] == 1: propose DEATH
 *           beta_j_new = 0 (deterministic)
 *           log_q_old  = propose_logq(beta_j_cur, j, ctx)   [reverse proposal]
 *           MH log-ratio = log_joint(gamma_new, beta_new)
 *                        - log_joint(gamma_cur, beta_cur)
 *                        + log_q_old
 *
 *        Accept / reject. If accepted, update gamma_ and beta_ AND the
 *        context the log_joint reads from on the next j.
 *
 *  The death accept-ratio derivation (Green 1995, detailed balance):
 *    birth_ratio(cur -> new) = target_new / target_cur / q_forward
 *    death_ratio(new -> cur) = target_cur / target_new * q_forward
 *      (q_forward is the birth proposal that WOULD have produced the
 *       current beta, evaluated at the current beta as "u")
 *  These are reciprocals, i.e. detailed balance holds. For the
 *  acceptance ratio expression we use above, death's +log_q_old is
 *  the log of q(u_reverse) where u_reverse = beta_j_cur.
 *
 *  WHAT THE USER PROVIDES
 *  ======================
 *  Three lambdas in rjmcmc_block_config:
 *
 *    log_joint(gamma, beta, ctx) -> double
 *        Joint log-density of (gamma, beta | rest) up to additive const.
 *        MUST include: likelihood, slab prior on active betas (those
 *        with gamma_j=1), Bernoulli prior on gamma.
 *        MUST NOT include: any Jacobian-related +log(|something|)
 *        correction — identity proposals have Jacobian = 1, nothing to
 *        add. See skills/system_design.md §10 for the general rule.
 *        When gamma[j] = 0, beta[j] is 0 by construction; the function
 *        should NOT evaluate a density of beta[j] against the slab
 *        prior at beta[j]=0 for those j (that would be a double count
 *        / wrong normalization — the Dirac spike provides its own
 *        "probability" via the absence of a slab-density term).
 *
 *    propose_sample(rng, j, ctx) -> double
 *        Sample beta_j_new from the proposal q(.; j, ctx).
 *        For optimal mixing, choose q close to the full conditional
 *        p(beta_j | gamma_j=1, y, rest). For linear Gaussian models,
 *        the closed-form conditional is Normal with mean (X_j' r_{-j})
 *        / (X_j' X_j + sigma^2/tau^2) and variance sigma^2 /
 *        (X_j' X_j + sigma^2/tau^2). Weaker proposals (e.g., just
 *        N(0, tau^2) = the slab prior itself) still correct but mix
 *        slower.
 *
 *    propose_logq(beta_new, j, ctx) -> double
 *        Log proposal density at beta_new. Must match propose_sample.
 *
 *  RNG
 *  ===
 *  Uses the mt19937 passed to step(). No R RNG dependency.
 *
 *  COMPOSITION WITH OTHER BLOCKS
 *  =============================
 *  rjmcmc_block updates (gamma, beta). Everything else — sigma^2, tau^2,
 *  pi (Bernoulli inclusion prior), and any other hyperparameters — is
 *  updated by sibling blocks (typically nuts_block on positive constraint
 *  for variances, beta_gibbs_block for pi when conjugate Beta prior).
 *  The composite drives the Gibbs sweep: rjmcmc_block reads these values
 *  from context and only updates what it owns.
 *
 *  CHECK #5 (NO HAND-WRITTEN JACOBIAN) APPLIES
 *  ===========================================
 *  The user's log_joint lambda MUST NOT add any +log(|J|)-like term
 *  for the birth / death move. identity-coordinate proposals have J = 1
 *  by construction; the block never uses a Jacobian term.
 *
 *================================================================================*/

#ifndef AI4BAYESCODE_RJMCMC_BLOCK_HPP
#define AI4BAYESCODE_RJMCMC_BLOCK_HPP

#include "block_sampler.hpp"
#include "rjmcmc_transforms.hpp"   // transform_1d_base + concrete 1D wrappers

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace AI4BayesCode {

// ---------------------------------------------------------------------------
// Configuration bundle for rjmcmc_block.
// ---------------------------------------------------------------------------

struct rjmcmc_block_config {
    /// Unique name within the composite (used as declare_dependencies key).
    std::string name = "gamma_beta_rj";

    /// shared_data key under which the composite writes the gamma vector
    /// via current_named_outputs(). Downstream blocks see this key.
    std::string gamma_key = "gamma";

    /// shared_data key under which the composite writes the beta vector
    /// via current_named_outputs(). Downstream blocks see this key.
    std::string beta_key = "beta";

    /// Number of candidate variables (length of gamma and beta).
    std::size_t p = 0;

    /// Optional Metropolis random-walk scale for the continuous update of
    /// beta_j when gamma_j = 1. If 0 (default), no RW update is performed;
    /// rely on continuous_update (below) or birth/death alone.
    ///
    /// rw_scale is used ONLY when continuous_update is not set. It is the
    /// simplest fallback and does not require a closed-form conditional.
    double rw_scale = 0.0;

    /// OPTIONAL. Direct in-place update of beta[j] when gamma[j] = 1,
    /// called once per j per sweep BEFORE the RJ birth/death attempt.
    /// Returns the new beta[j] value. The caller may implement:
    ///
    ///   - Exact Gibbs: sample beta[j] from its full conditional
    ///     p(beta_j | gamma_j=1, y, rest). Always "accepts" (no
    ///     accept/reject done by the framework). Best mixing when a
    ///     closed-form conditional exists (e.g., linear Gaussian models).
    ///
    ///   - Manual Metropolis: propose + internally accept/reject, then
    ///     return either the proposed or current value. Same effect as
    ///     rw_scale but with user-supplied proposal distribution.
    ///
    /// The framework calls this function and adopts the returned value
    /// as the new beta[j] unconditionally; any accept/reject logic
    /// MUST be encapsulated inside the function. If the returned value
    /// is non-finite the framework ignores it and leaves beta[j] alone.
    ///
    /// If this is set, rw_scale is ignored.
    ///
    /// CRITICAL: the function MUST preserve detailed balance of the
    /// conditional posterior of beta[j] | gamma[j] = 1. Exact Gibbs is
    /// trivially correct. A custom Metropolis scheme must account for
    /// its own accept ratio internally.
    std::function<double(std::mt19937_64& rng,
                         std::size_t j,
                         const block_context& ctx)> continuous_update;

    /// Joint log-density lambda. See header comment for contract.
    std::function<double(const arma::vec& gamma,
                         const arma::vec& beta,
                         const block_context& ctx)> log_joint;

    /// Birth proposal sampler: draws beta_j_new from q(.; j, ctx).
    std::function<double(std::mt19937_64& rng,
                         std::size_t j,
                         const block_context& ctx)> propose_sample;

    /// Log proposal density at beta_new: log q(beta_new; j, ctx).
    std::function<double(double beta_new,
                         std::size_t j,
                         const block_context& ctx)> propose_logq;

    /// Initial gamma vector. Length must equal p, entries 0/1.
    arma::vec gamma_init;

    /// Initial beta vector. Length must equal p. For any j with
    /// gamma_init[j] == 0, beta_init[j] must be exactly 0.
    arma::vec beta_init;

    /// OPTIONAL: 1D transform for the birth proposal.
    ///
    /// When set, the birth pipeline becomes:
    ///   u          ~ propose_sample(rng, j, ctx)          [aux variable]
    ///   log q(u)   = propose_logq(u, j, ctx)              [aux log-density]
    ///   beta_new   = transform->apply_forward(u, &β_new)  [library T(u)]
    ///   |det J|    from the same apply_forward return value  [library Jacobian]
    /// The MH accept ratio for birth is then
    ///   log p(new) - log p(cur) - log q(u) + log|det J_fwd|.
    /// Death uses the inverse direction:
    ///   u_implied  = transform->apply_reverse(beta_cur)
    ///   log q(u_implied) from propose_logq
    ///   |det J_rev| = 1 / |det J_fwd|
    ///   log accept = log p(new) - log p(cur) + log q(u_implied) + log|det J_rev|
    ///
    /// When NOT set (default nullptr), the identity-coordinate path
    /// is used unchanged: propose_sample returns β_new directly, no
    /// Jacobian term enters. All identity-only examples continue to work
    /// unchanged; the transform field is purely additive.
    ///
    /// IMPORTANT semantic change when `transform` IS set: `propose_sample`
    /// and `propose_logq` now operate on the AUXILIARY variable u, NOT on
    /// β_new. The user must match propose_sample/propose_logq to what
    /// their auxiliary distribution produces — NOT the birth β value.
    ///
    /// Check #5 still holds: the user writes NO Jacobian code. The
    /// transform class (library-provided) computes |det J| internally.
    std::shared_ptr<rjmcmc_transforms::transform_1d_base> transform;
};

// ---------------------------------------------------------------------------
// rjmcmc_block
// ---------------------------------------------------------------------------

class rjmcmc_block : public block_sampler {
public:
    explicit rjmcmc_block(rjmcmc_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.p == 0) {
            throw std::invalid_argument(
                "rjmcmc_block: p must be > 0");
        }
        if (cfg_.gamma_init.n_elem != cfg_.p) {
            throw std::invalid_argument(
                "rjmcmc_block: gamma_init length must equal p");
        }
        if (cfg_.beta_init.n_elem != cfg_.p) {
            throw std::invalid_argument(
                "rjmcmc_block: beta_init length must equal p");
        }
        if (!cfg_.log_joint) {
            throw std::invalid_argument(
                "rjmcmc_block: log_joint lambda is required");
        }
        if (!cfg_.propose_sample) {
            throw std::invalid_argument(
                "rjmcmc_block: propose_sample lambda is required");
        }
        if (!cfg_.propose_logq) {
            throw std::invalid_argument(
                "rjmcmc_block: propose_logq lambda is required");
        }
        if (!(cfg_.rw_scale >= 0.0)) {
            throw std::invalid_argument(
                "rjmcmc_block: rw_scale must be non-negative");
        }

        // Snap gamma to 0/1 and validate coupling between gamma and beta.
        gamma_.set_size(cfg_.p);
        beta_.set_size(cfg_.p);
        for (std::size_t j = 0; j < cfg_.p; ++j) {
            gamma_[j] = (cfg_.gamma_init[j] >= 0.5) ? 1.0 : 0.0;
            if (gamma_[j] == 0.0 && std::abs(cfg_.beta_init[j]) > 0.0) {
                throw std::invalid_argument(
                    "rjmcmc_block: beta_init[j] must be 0 when "
                    "gamma_init[j] == 0");
            }
            beta_[j] = cfg_.beta_init[j];
        }

        // Pre-build the concatenated [gamma; beta] view that current()
        // returns. Kept in sync with gamma_ / beta_ via refresh_current_().
        current_cache_.set_size(2 * cfg_.p);
        refresh_current_cache_();
    }

    // ---- block_sampler interface ---------------------------------------

    void set_context(const block_context& ctx) override {
        // Deep copy; this is the binary_gibbs_block pattern. The block
        // mutates context_ in-sweep (for the gamma_key / beta_key slots)
        // so log_joint always sees the current state on the next j.
        context_ = ctx;
        // Ensure our named outputs are present in context (in case the
        // composite has not seeded them yet — e.g., on the very first
        // call).
        context_[cfg_.gamma_key] = gamma_;
        context_[cfg_.beta_key]  = beta_;
    }

    void step(std::mt19937_64& rng) override {
        // Refuse partial sub-key freeze (2026-07-20).
        //
        // The reversible-jump birth/death move modifies (gamma_j, beta_j)
        // ATOMICALLY: a birth writes a new beta AND flips gamma 0->1; a
        // death writes beta=0 AND flips gamma 1->0. Freezing only one of
        // the two sub-keys does NOT preserve pinning under the joint
        // proposal -- e.g. with only "gamma" frozen the (B) trans-dim
        // sweep is skipped but active betas continue to sample via
        // continuous_update (or the (A) RW), and vice versa the paired
        // move still fires around the frozen slot in the other direction
        // depending on which sub-flag was set. In either single-flag
        // configuration the composite of (frozen sub-key + still-active
        // joint move) is a semantically ill-defined kernel that violates
        // the user's stated "pin this coordinate" intent.
        //
        // Enforcement is a step-guard (option (c) in the design notes):
        // freeze_sub() itself does NOT raise, because the composite may
        // call freeze_sub("gamma") and freeze_sub("beta") in sequence
        // when the user passes freeze({"<name>.gamma", "<name>.beta"});
        // instead we require both flags to be equal at step time. If
        // exactly one is set, throw with a clear, actionable message.
        // Whole-block freeze routes through is_frozen_ and the composite
        // skips step() entirely (see composite_block::step), so this
        // check never fires in that case.
        if (gamma_frozen_ != beta_frozen_) {
            throw std::runtime_error(
                "rjmcmc_block '" + cfg_.name +
                "': rjmcmc joint move requires either freezing both "
                "sub-keys (gamma AND beta) simultaneously or freezing "
                "the whole rjmcmc_block; single sub-key freeze does "
                "not preserve pinning under joint proposals "
                "(currently frozen: " +
                (gamma_frozen_ ? std::string("'") + cfg_.gamma_key + "'"
                               : std::string("'") + cfg_.beta_key + "'") +
                "). Fix: pass freeze({\"" + cfg_.name + "." + cfg_.gamma_key +
                "\", \"" + cfg_.name + "." + cfg_.beta_key +
                "\"}) to freeze both sub-keys, or freeze({\"" + cfg_.name +
                "\"}) to freeze the whole block.");
        }

        // Random permutation of indices for this sweep.
        std::vector<std::size_t> order(cfg_.p);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        std::normal_distribution<double>       rw(0.0, 1.0);

        // Counters for diagnostics / debugging. Reset each sweep.
        std::size_t n_birth_try = 0, n_birth_ok = 0;
        std::size_t n_death_try = 0, n_death_ok = 0;
        std::size_t n_rw_try    = 0, n_rw_ok    = 0;

        for (std::size_t j : order) {

            // (A) CONTINUOUS UPDATE of beta_j (only if gamma_j = 1).
            //     Two mutually-exclusive modes. User picks one or neither
            //     at config time.
            //
            // Kernel-control freeze (DESIGN_NOTES Sec.10.d): beta_frozen_
            // suppresses the entire (A) block so active beta values stay
            // pinned at their current values. gamma sweep (B) still fires
            // unless gamma_frozen_ is also set.
            if (!beta_frozen_) {
            if (gamma_[j] == 1.0 && cfg_.continuous_update) {
                // Mode 1: user-supplied direct update (typically Gibbs from
                // closed-form conditional). Framework does NOT accept/reject
                // — the function returns the new value (if it implemented
                // accept/reject, it has already committed).
                const double beta_new =
                    cfg_.continuous_update(rng, j, context_);
                if (std::isfinite(beta_new)) {
                    beta_[j] = beta_new;
                    context_[cfg_.beta_key] = beta_;
                    ++n_rw_try; ++n_rw_ok;  // count as "accepted" update
                } else {
                    ++n_rw_try;
                }
            } else if (gamma_[j] == 1.0 && cfg_.rw_scale > 0.0) {
                // Mode 2: fallback Metropolis RW with symmetric Gaussian.
                ++n_rw_try;
                const double beta_old = beta_[j];
                const double beta_new = beta_old + cfg_.rw_scale * rw(rng);

                // Score both states via log_joint.
                beta_[j] = beta_old;
                context_[cfg_.beta_key] = beta_;
                const double lp_old = cfg_.log_joint(gamma_, beta_, context_);

                beta_[j] = beta_new;
                context_[cfg_.beta_key] = beta_;
                const double lp_new = cfg_.log_joint(gamma_, beta_, context_);

                // Symmetric Gaussian RW: no proposal ratio term.
                const double log_R = lp_new - lp_old;
                if (std::log(uniform(rng)) < log_R) {
                    // Accept. beta_[j] already set to beta_new.
                    ++n_rw_ok;
                } else {
                    // Reject. Restore.
                    beta_[j] = beta_old;
                    context_[cfg_.beta_key] = beta_;
                }
            }
            }  // end !beta_frozen_

            // Kernel-control freeze: gamma_frozen_ suppresses the entire
            // (B) trans-dim sweep. Active-set membership stays constant.
            // Note: if BOTH sub-keys are frozen, the loop body is a no-op
            // for this j; composite_block's whole-block is_frozen_ check
            // is the outer gate that skips step() entirely.
            if (gamma_frozen_) continue;

            // (B) REVERSIBLE-JUMP FLIP on gamma_j (every sweep).
            const double gamma_cur = gamma_[j];
            const double beta_cur  = beta_[j];

            // Score the current state.
            context_[cfg_.gamma_key] = gamma_;
            context_[cfg_.beta_key]  = beta_;
            const double lp_cur = cfg_.log_joint(gamma_, beta_, context_);

            if (gamma_cur == 0.0) {
                // BIRTH attempt: gamma_j: 0 -> 1, propose beta_j_new.
                // With transform:    u ~ propose_sample; beta = T(u); |J|
                // Without transform: u = beta_new (identity behavior)
                ++n_birth_try;
                const double u_sample =
                    cfg_.propose_sample(rng, j, context_);
                const double log_q_u =
                    cfg_.propose_logq(u_sample, j, context_);
                if (!std::isfinite(log_q_u)) {
                    continue;
                }

                double beta_new;
                double log_J_fwd;
                if (cfg_.transform) {
                    double abs_det =
                        cfg_.transform->apply_forward(u_sample, beta_new);
                    if (!std::isfinite(abs_det) || !(abs_det > 0.0)) {
                        // Degenerate transform; reject.
                        continue;
                    }
                    log_J_fwd = std::log(abs_det);
                } else {
                    // Identity: β_new = u. |det J| = 1 -> log|J| = 0.
                    beta_new  = u_sample;
                    log_J_fwd = 0.0;
                }

                // Propose.
                gamma_[j] = 1.0;
                beta_[j]  = beta_new;
                context_[cfg_.gamma_key] = gamma_;
                context_[cfg_.beta_key]  = beta_;
                const double lp_new = cfg_.log_joint(gamma_, beta_, context_);

                // MH log-ratio (Green 1995):
                //   log R = log p(new) - log p(cur) - log q(u) + log|det J_fwd|
                const double log_R = lp_new - lp_cur - log_q_u + log_J_fwd;

                if (std::isfinite(log_R) &&
                    std::log(uniform(rng)) < log_R) {
                    ++n_birth_ok;
                } else {
                    gamma_[j] = gamma_cur;
                    beta_[j]  = beta_cur;
                    context_[cfg_.gamma_key] = gamma_;
                    context_[cfg_.beta_key]  = beta_;
                }

            } else {
                // DEATH attempt: gamma_j: 1 -> 0, beta_j -> 0.
                // With transform:    u_implied = T^{-1}(beta_cur);
                //                    log q(u_implied);  log|det J_rev|
                // Without transform: u_implied = beta_cur (identity)
                ++n_death_try;

                double u_implied;
                double log_J_rev;
                if (cfg_.transform) {
                    double abs_det_rev =
                        cfg_.transform->apply_reverse(beta_cur, u_implied);
                    if (!std::isfinite(abs_det_rev) || !(abs_det_rev > 0.0)) {
                        continue;
                    }
                    log_J_rev = std::log(abs_det_rev);
                } else {
                    u_implied = beta_cur;
                    log_J_rev = 0.0;
                }
                const double log_q_u_implied =
                    cfg_.propose_logq(u_implied, j, context_);
                if (!std::isfinite(log_q_u_implied)) {
                    continue;
                }

                // Propose.
                gamma_[j] = 0.0;
                beta_[j]  = 0.0;
                context_[cfg_.gamma_key] = gamma_;
                context_[cfg_.beta_key]  = beta_;
                const double lp_new = cfg_.log_joint(gamma_, beta_, context_);

                // MH log-ratio for death (Green 1995):
                //   log R = log p(new) - log p(cur) + log q(u_implied) + log|det J_rev|
                // With identity: log|J_rev| = 0 and u_implied = beta_cur,
                // recovering the no-transform formula exactly.
                const double log_R =
                    lp_new - lp_cur + log_q_u_implied + log_J_rev;

                if (std::isfinite(log_R) &&
                    std::log(uniform(rng)) < log_R) {
                    ++n_death_ok;
                } else {
                    gamma_[j] = gamma_cur;
                    beta_[j]  = beta_cur;
                    context_[cfg_.gamma_key] = gamma_;
                    context_[cfg_.beta_key]  = beta_;
                }
            }
        } // end for j

        // Cache concatenated state.
        refresh_current_cache_();

        // Save diagnostics in member state for monitoring.
        last_n_birth_try_ = n_birth_try; last_n_birth_ok_ = n_birth_ok;
        last_n_death_try_ = n_death_try; last_n_death_ok_ = n_death_ok;
        last_n_rw_try_    = n_rw_try;    last_n_rw_ok_    = n_rw_ok;

        // Append to history if enabled.
        if (keep_history_) {
            history_buf_.push_back(current_cache_);
        }
    }

    /// Returns a length-2p vector: [gamma (p entries); beta (p entries)].
    const arma::vec& current() const override {
        return current_cache_;
    }

    /// Accepts a length-2p vector [gamma; beta] and validates coupling.
    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != 2 * cfg_.p) {
            throw std::invalid_argument(
                "rjmcmc_block::set_current: wrong length (expected 2p = "
                + std::to_string(2 * cfg_.p) + ", got "
                + std::to_string(theta.n_elem) + ")");
        }
        for (std::size_t j = 0; j < cfg_.p; ++j) {
            const double g = (theta[j] >= 0.5) ? 1.0 : 0.0;
            const double b = theta[cfg_.p + j];
            if (g == 0.0 && std::abs(b) > 0.0) {
                throw std::invalid_argument(
                    "rjmcmc_block::set_current: beta[" + std::to_string(j)
                    + "] must be 0 when gamma[" + std::to_string(j)
                    + "] == 0");
            }
            gamma_[j] = g;
            beta_[j]  = b;
        }
        refresh_current_cache_();
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    /// Total storage dimension (2p: gamma then beta).
    std::size_t dim() const noexcept override { return 2 * cfg_.p; }

    /// Split the concatenated state into two named outputs so the
    /// composite writes "gamma" and "beta" separately into shared_data.
    std::unordered_map<std::string, arma::vec>
    current_named_outputs() const override {
        std::unordered_map<std::string, arma::vec> out;
        out[cfg_.gamma_key] = gamma_;
        out[cfg_.beta_key]  = beta_;
        return out;
    }

    // ---- History overrides ---------------------------------------------

    /// History is a list with two entries: $gamma (n_draws x p matrix)
    /// and $beta (n_draws x p matrix). Unlike the default one-key
    /// fallback, this block's named outputs demand two separate slots.
    history_map get_history() const override {
        const std::size_t p = cfg_.p;
        history_map out;
        if (history_buf_.empty()) {
            arma::mat gamma_row(1, p);
            arma::mat beta_row (1, p);
            for (std::size_t j = 0; j < p; ++j) {
                gamma_row(0, j) = gamma_[j];
                beta_row (0, j) = beta_[j];
            }
            out.emplace(cfg_.gamma_key, std::move(gamma_row));
            out.emplace(cfg_.beta_key,  std::move(beta_row));
            return out;
        }
        const std::size_t n_draws = history_buf_.size();
        arma::mat gamma_hist(n_draws, p);
        arma::mat beta_hist (n_draws, p);
        for (std::size_t i = 0; i < n_draws; ++i) {
            const arma::vec& snap = history_buf_[i];
            for (std::size_t j = 0; j < p; ++j) {
                gamma_hist(i, j) = snap[j];
                beta_hist (i, j) = snap[p + j];
            }
        }
        out.emplace(cfg_.gamma_key, std::move(gamma_hist));
        out.emplace(cfg_.beta_key,  std::move(beta_hist));
        return out;
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override { history_buf_.clear(); }

    // ---- rjmcmc_block-specific diagnostics ------------------------------

    /// Per-sweep counters from the last step() call.
    /// Useful for tuning rw_scale and diagnosing low-acceptance issues.
    struct step_diagnostics {
        std::size_t n_birth_try, n_birth_ok;
        std::size_t n_death_try, n_death_ok;
        std::size_t n_rw_try,    n_rw_ok;
    };

    step_diagnostics last_step_diagnostics() const {
        return step_diagnostics{
            last_n_birth_try_, last_n_birth_ok_,
            last_n_death_try_, last_n_death_ok_,
            last_n_rw_try_,    last_n_rw_ok_};
    }

    // ---- Kernel-control freeze API (sub-key, DESIGN_NOTES Sec.10.d) -----
    //
    // rjmcmc_block exposes two sub-keys -- gamma_key (trans-dim sweep) and
    // beta_key (continuous_update on active betas). Composite calls
    // freeze_sub(sub) with either name to record the intent; whole-block
    // freeze() (inherited from base) atomically disables both.
    //
    // PARTIAL SUB-KEY FREEZE IS REFUSED (2026-07-20). Because the
    // trans-dim birth/death move updates (gamma, beta) atomically, a
    // single-sub-key freeze does not preserve pinning under the joint
    // proposal (previously documented as an intrinsic hazard). The block
    // now REQUIRES either (a) both sub-keys frozen simultaneously via
    // freeze({"<name>.<gamma_key>", "<name>.<beta_key>"}) or (b) the
    // whole block frozen via freeze({"<name>"}). freeze_sub() records
    // the flag but step() refuses to run when exactly one flag is set
    // (see the step-guard at the top of step() for the enforcement path
    // and the user-facing error message).

    std::vector<std::string> subnames() const override {
        return {cfg_.gamma_key, cfg_.beta_key};
    }

    void freeze_sub(const std::string& sub) override {
        if (sub == cfg_.gamma_key) {
            gamma_frozen_ = true;
        } else if (sub == cfg_.beta_key) {
            beta_frozen_ = true;
        } else {
            throw std::runtime_error(
                "rjmcmc_block::freeze_sub: unknown sub-name '" + sub +
                "'; valid sub-names for this block: '" + cfg_.gamma_key +
                "', '" + cfg_.beta_key + "'");
        }
    }

    void unfreeze_sub(const std::string& sub) override {
        if (sub == cfg_.gamma_key)      gamma_frozen_ = false;
        else if (sub == cfg_.beta_key)  beta_frozen_  = false;
        // Unknown name = no-op (unfreeze is permissive).
    }

    std::vector<std::string> frozen_subnames() const override {
        std::vector<std::string> out;
        if (gamma_frozen_) out.push_back(cfg_.gamma_key);
        if (beta_frozen_)  out.push_back(cfg_.beta_key);
        return out;
    }

    /// Also unfreeze the two sub-flags on whole-block unfreeze, so a user
    /// who does whole-block freeze then unfreeze gets a clean restart even
    /// if they had sub-key-frozen state before.
    void unfreeze() override {
        block_sampler::unfreeze();   // clears is_frozen_
        gamma_frozen_ = false;
        beta_frozen_  = false;
    }

private:
    void refresh_current_cache_() {
        for (std::size_t j = 0; j < cfg_.p; ++j) {
            current_cache_[j]          = gamma_[j];
            current_cache_[cfg_.p + j] = beta_[j];
        }
    }

    rjmcmc_block_config      cfg_;
    arma::vec                gamma_;
    arma::vec                beta_;
    arma::vec                current_cache_;  // [gamma; beta] concat
    block_context            context_;
    std::vector<arma::vec>   history_buf_;

    std::size_t last_n_birth_try_ = 0, last_n_birth_ok_ = 0;
    std::size_t last_n_death_try_ = 0, last_n_death_ok_ = 0;
    std::size_t last_n_rw_try_    = 0, last_n_rw_ok_    = 0;

    // Kernel-control sub-key freeze flags (DESIGN_NOTES Sec.10.d).
    // Whole-block freeze goes via is_frozen_ in the base class; these two
    // flags additionally gate the individual sub-updates inside step().
    bool                     gamma_frozen_ = false;
    bool                     beta_frozen_  = false;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_RJMCMC_BLOCK_HPP

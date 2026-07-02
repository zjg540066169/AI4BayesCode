/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  nuts_block.hpp  --  a block_sampler that runs the (hacked) MCMClib NUTS
 *                      with persistent dual-averaging adaptation, fed by a
 *                      pluggable log-density-and-gradient oracle.
 *
 *  WHO OWNS WHAT
 *  =============
 *  This file is the single point where block_mcmc touches mcmclib::nuts.
 *  Everything it owns falls into one of three buckets:
 *
 *    1. Inference state (mcmclib's responsibility conceptually, but stored
 *       here so we can persist it across calls to step()):
 *         - theta_unc_      -- current draw on the UNCONSTRAINED scale
 *         - nuts_settings_  -- includes epsilon_bar_persist, h_val_persist,
 *                              mu_val_persist, adapt_iter_persist, precond_mat
 *
 *    2. User-facing value cache (what current() returns):
 *         - theta_natural_  -- constrained representation recomputed after
 *                              each step
 *
 *    3. Context the block just received from the outside world:
 *         - context_        -- a copy of the block_context most recently
 *                              passed to set_context. The target callback
 *                              closes over this member and reads it every
 *                              time mcmclib::nuts asks for a log-density.
 *
 *  WHAT THE USER SUPPLIES
 *  ----------------------
 *  nuts_block does NOT hard-code BridgeStan, Stan Math, CppAD, or any
 *  particular autodiff backend. It takes a plain std::function:
 *
 *    log_density_gradient_fn ::= (theta_unc, ctx) -> (log_p, grad)
 *
 *  Anyone can produce one of those:
 *    - a BridgeStan wrapper built around a .so compiled from Stan
 *    - a hand-written C++ lambda that computes the log-density directly
 *    - a CppAD or autodiff.hpp tape
 *    - an AI-generated C++ functor
 *  nuts_block cannot tell them apart, and does not want to.
 *
 *  The constrain / unconstrain transforms are likewise injected as plain
 *  std::functions. For an unconstrained block they are the identity; for
 *  a simplex they are stick-breaking / inverse; for a positive scalar they
 *  are exp / log; and so on. In the BridgeStan path they are simply
 *  bs_param_constrain / bs_param_unconstrain.
 *
 *  PERSISTENT ADAPTATION
 *  ---------------------
 *  The core of this block's value proposition: each call to step() runs
 *  `n_draws_per_step` NUTS transitions with use_persistent_adapt = true,
 *  so the dual-averaging state (epsilon_bar, h, mu, adapt_iter) accumulates
 *  across calls. The outer Gibbs sweep can therefore update other blocks
 *  in between, and NUTS resumes from the exact step size it had reached.
 *  This is what makes "stateful MCMC inside a larger MCMC" work.
 *
 *  The mass matrix (precond_mat) is fixed at construction for v0. Online
 *  mass-matrix adaptation is a v1 feature; users who want it can call
 *  set_precond_matrix() between sweeps with an empirical covariance
 *  estimated from accumulated draws.
 *================================================================================*/

#ifndef AI4BAYESCODE_NUTS_BLOCK_HPP
#define AI4BAYESCODE_NUTS_BLOCK_HPP

#include "block_sampler.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// mcmclib brings in arma + its type aliases (mcmc::ColVec_t, mcmc::Mat_t,
// mcmc::fp_t, mcmc::algo_settings_t, mcmc::nuts_settings_t, mcmc::nuts).
#include "mcmclib/mcmc.hpp"

namespace AI4BayesCode {

/**
 * @brief Signature of the gradient oracle consumed by nuts_block.
 *
 * @param theta_unc  Current parameter value on the UNCONSTRAINED scale.
 * @param ctx        The context most recently installed via set_context.
 *                   The oracle may read any keys it expects to find here;
 *                   missing keys are a user (or generator) error and
 *                   should throw.
 * @param grad_out   Output buffer for the gradient; same length as
 *                   @p theta_unc. Must be written to in full.
 * @return           log p(theta_unc | ctx), INCLUDING the log|Jacobian| of
 *                   the unconstraining transform if the user-facing
 *                   parameter is constrained.
 *
 * The oracle is required to be a pure function of (theta_unc, ctx): no
 * global state, no hidden side channels. nuts_block closes over it and
 * calls it many times per step() without any synchronization.
 */
using log_density_gradient_fn = std::function<
    double(const arma::vec& theta_unc,
           const block_context& ctx,
           arma::vec* grad_out)>;

/// Optional transform between unconstrained and natural scales. The
/// identity is used when the parameter is already unconstrained.
using transform_fn = std::function<arma::vec(const arma::vec&)>;

/**
 * @brief Configuration bundle for nuts_block construction.
 *
 * All fields are validated in nuts_block's constructor; invalid
 * configurations throw std::invalid_argument with a descriptive message.
 */
struct nuts_block_config {
    /// Unique name for this block within its composite; used as the key
    /// under which shared_data_t stores this block's current value.
    std::string name;

    /// Initial value on the UNCONSTRAINED scale. Must be non-empty.
    arma::vec initial_unc;

    /// The log-density oracle (see @ref log_density_gradient_fn). Required.
    log_density_gradient_fn log_density_grad;

    /// Map unconstrained -> natural. Defaults to identity if left empty.
    transform_fn constrain;

    /// Map natural -> unconstrained. Defaults to identity if left empty.
    /// Only needed if set_current will be called with a natural-scale
    /// value; if the user never calls set_current, this can be omitted.
    transform_fn unconstrain;

    /// NUTS settings. The constructor forces
    /// nuts_settings.use_persistent_adapt = true regardless of what is
    /// passed in, because a nuts_block that does not persist adaptation
    /// would contradict the purpose of the whole package.
    mcmc::algo_settings_t nuts_settings;

    /// Optional override for the initial step size.
    ///
    /// If > 0, nuts_block bypasses mcmclib's nuts_find_initial_step_size
    /// by seeding the persistent-adaptation state (epsilon_bar_persist,
    /// h_val_persist, mu_val_persist, adapt_iter_persist) before the
    /// first call to step(). This routes the very first call through
    /// mcmclib's "continue from epsilon_bar_persist" branch, not its
    /// "fresh initial adaptation" branch, so FindReasonableEpsilon is
    /// skipped entirely.
    ///
    /// Use this when the block's conditional posterior is tight enough
    /// that the paper's mu = log(10 * epsilon_1) bias would push the
    /// first warmup iteration into a divergent regime. Symptoms of
    /// needing this knob: step size explodes to NaN in the first few
    /// warmup iterations, chain wedges at the initial value. A small
    /// conservative value like 0.01 - 0.05 is almost always safe; if
    /// it is too small, dual averaging brings it up during warmup.
    ///
    /// If left at 0 (the default), FindReasonableEpsilon is used. That
    /// is appropriate for most well-conditioned problems (real-valued
    /// locations, positive scalars). Simplex / correlation / ordered
    /// constraints often benefit from a seeded value because the
    /// stick-breaking geometry makes the typical set narrower than
    /// FindReasonableEpsilon's 10x bias can tolerate on iter 1.
    double initial_step_size = 0.0;

    /// Number of dual-averaging adaptation iterations performed at the
    /// very first call to step(). Larger gives the initial step-size
    /// search more runway; the default 200 matches the order used by
    /// Stan's "init buffer + window" but compressed into a single
    /// window because persistent adaptation takes over afterwards.
    std::size_t n_warmup_first_call = 200;

    /// Number of dual-averaging adaptation iterations performed on
    /// every call to step() AFTER the first one. **DEFAULT 0 IS
    /// MANDATORY** for code-generated examples — do NOT override
    /// (validator Check #20).
    ///
    /// Why mandatory 0
    /// ---------------
    /// Setting `n_warmup_per_step > 0` re-enables a chain-state corruption
    /// mechanism that was removed by the 2026-04-12 mcmclib NUTS bugfix
    /// ("n_adapt_draws included kept draws, causing ongoing adaptation
    /// during sampling"). With ongoing adaptation, the dual-averaging
    /// state is recomputed against the same draws that are being kept
    /// in the chain, which both violates detailed balance AND can
    /// progressively shrink the step size to a region where the chain
    /// gets locked into one neighborhood — the runtime symptom is
    /// `rhat_max ~ 2.2`, `ess_bulk ~ NA / single digits`, and AI walltime
    /// SHORTER than Stan's because the locked chain does almost no tree
    /// exploration. This is the same "stuck-fast" pattern as a bad
    /// dense-metric pilot (Check #18); the diagnosis is the gap between
    /// L3 single-dataset PASS and sim1 multi-dataset catastrophic FAIL.
    ///
    /// What was previously documented here ("5-8% 1D variance bias")
    /// described the BOUNDED-bias regime that existed BEFORE the
    /// 2026-04-12 fix and is no longer a valid mental model. The fix
    /// closed that path; non-zero values now produce unbounded chain-
    /// state corruption, not a bounded bias.
    ///
    /// "Rejection lock-up" with n_warmup_per_step = 0
    /// -----------------------------------------------
    /// If a code-gen attempt observes "the sigma block keeps rejecting
    /// at n_warmup_per_step = 0", the actual problem is a non-centered
    /// hierarchical funnel between (sigma_*, raw_effect_*) — the scale
    /// and its raw-effect partner are in separate Gibbs blocks but their
    /// conditionals are multiplicatively coupled. The correct fix is
    /// methodological:
    ///   1. `joint_nuts_block_mixed` over (sigma_*, z_*) per
    ///      `skills/codegen_cpp.md §4a` row "scale + raw effect, non-
    ///      centered" — the joint block samples the multiplicative
    ///      coupling exactly.
    ///   2. Longer `n_warmup_first_call` (1500–3000) so the first-call
    ///      adaptation lands in the typical set.
    ///   3. Better init values via OLS / method of moments in the
    ///      wrapper constructor.
    /// Do NOT escape to `n_warmup_per_step > 0`; that produces silently
    /// wrong posteriors that pass L3 single-dataset checks while failing
    /// sim1 cross-dataset checks.
    ///
    /// Override discipline
    /// -------------------
    /// Override is a system-design action, NOT a code-gen action. A
    /// system-design audit must demonstrate (a) a specific non-stationary
    /// use case where ongoing per-step adaptation is provably necessary
    /// AND (b) the chain-corruption mechanism is provably absent
    /// (paired against full warmup at the same budget on the same
    /// dataset, with R-hat parity at sim1 cross-dataset scale, NOT
    /// just L3 single-dataset). No production example currently
    /// satisfies this bar.
    std::size_t n_warmup_per_step = 0;

    /// Number of NUTS transitions kept per call to step(). In a tight
    /// outer Gibbs loop this is typically 1; larger values give more
    /// thinning per sweep at proportionally higher cost.
    std::size_t n_draws_per_step = 1;
};

/**
 * @brief A stateful NUTS block that participates in a block_mcmc composite.
 *
 * See the file header for the design rationale.
 */
class nuts_block : public block_sampler {
public:
    explicit nuts_block(nuts_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.initial_unc.n_elem == 0) {
            throw std::invalid_argument(
                "nuts_block: initial_unc must be non-empty");
        }
        if (!cfg_.log_density_grad) {
            throw std::invalid_argument(
                "nuts_block: log_density_grad oracle is required");
        }
        if (!cfg_.constrain) {
            cfg_.constrain = [](const arma::vec& x) { return x; };
        }
        if (!cfg_.unconstrain) {
            cfg_.unconstrain = [](const arma::vec& x) { return x; };
        }
        if (cfg_.n_draws_per_step == 0) {
            cfg_.n_draws_per_step = 1;
        }

        auto& ns = cfg_.nuts_settings.nuts_settings;

        // Force persistent adaptation ON. This is non-negotiable: without
        // it there is no reason to use nuts_block over a raw mcmclib call.
        ns.use_persistent_adapt = true;

        if (cfg_.initial_step_size > 0.0) {
            // User-provided seed: skip FindReasonableEpsilon entirely by
            // routing the first call through mcmclib's "continue from
            // epsilon_bar_persist" branch. This is the escape hatch for
            // tight conditionals (e.g. simplex / correlation / ordered)
            // where the paper's mu = log(10 * epsilon_1) bias would send
            // the first dual-averaging iteration into a divergent regime.
            ns.epsilon_bar_persist = cfg_.initial_step_size;
            ns.h_val_persist       = 0.0;
            ns.mu_val_persist      =
                std::log(10.0 * cfg_.initial_step_size);
            ns.adapt_iter_persist  = 1; // nonzero triggers persistent branch
            ns.step_size           = cfg_.initial_step_size;
        }
        // Otherwise leave epsilon_bar_persist / adapt_iter_persist at
        // their defaults (1.0 and 0), so the very first call to step()
        // enters mcmclib's "fresh initial adaptation" branch and runs
        // nuts_find_initial_step_size (paper-faithful; see the
        // modifications header on nuts.ipp).

        theta_unc_     = cfg_.initial_unc;
        theta_natural_ = cfg_.constrain(theta_unc_);
        first_call_    = true;
    }

    // ---- block_sampler interface ---------------------------------------

    void set_context(const block_context& ctx) override {
        // COPY. See the design contract in block_sampler.hpp.
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // MCMClib's NUTS owns its own RNG via nuts_settings.rng_seed_value,
        // so we draw a fresh seed from the caller's rng on every step to
        // get forward progress without collisions between blocks.
        cfg_.nuts_settings.rng_seed_value =
            static_cast<std::uint64_t>(rng());

        // Choose the warmup budget for THIS call. The first call runs
        // the long warmup because mcmclib will do nuts_find_initial_step_size
        // and then a fresh dual-averaging window from scratch; every
        // subsequent call only needs a mini-warmup to refine the step
        // size against the current conditional.
        auto& ns = cfg_.nuts_settings.nuts_settings;
        const std::size_t n_warmup =
            first_call_ ? cfg_.n_warmup_first_call : cfg_.n_warmup_per_step;

        ns.n_burnin_draws = n_warmup;
        ns.n_keep_draws   = cfg_.n_draws_per_step;
        // Adapt ONLY during burnin, not during kept draws. Adapting
        // during sampling violates detailed balance and causes ~7%
        // variance underestimation in 1D.
        ns.n_adapt_draws  = n_warmup;

        // Adapter between our (theta, ctx) -> (lp, grad) signature and
        // mcmclib's (theta, grad*, void*) -> lp signature. Closes over
        // cfg_.log_density_grad and context_ so that mcmclib can call it
        // opaquely without knowing about block_context.
        auto adapter = [this](const mcmc::ColVec_t& theta_in,
                              mcmc::ColVec_t* grad_out,
                              void* /*unused*/) -> mcmc::fp_t {
            const double lp = cfg_.log_density_grad(theta_in,
                                                    context_,
                                                    grad_out);
            return static_cast<mcmc::fp_t>(lp);
        };

        mcmc::Mat_t draws_out;
        const bool ok = mcmc::nuts(
            theta_unc_,
            adapter,
            draws_out,
            /*target_data=*/ nullptr,
            cfg_.nuts_settings);

        if (!ok) {
            throw std::runtime_error(
                "nuts_block '" + cfg_.name + "': mcmclib::nuts returned false");
        }
        if (draws_out.n_rows == 0) {
            throw std::runtime_error(
                "nuts_block '" + cfg_.name + "': nuts produced zero draws");
        }

        // Advance current value to the last kept draw. All prior draws in
        // this step() call are discarded; if the user wants thinning
        // behaviour they must pull draws out of a larger outer loop.
        theta_unc_ = draws_out.row(draws_out.n_rows - 1).t();
        theta_natural_ = cfg_.constrain(theta_unc_);

        // Append to history if enabled
        if (keep_history_) {
            history_buf_.push_back(theta_natural_);
        }

        first_call_ = false;
    }

    const arma::vec& current() const override {
        return theta_natural_;
    }

    void set_current(const arma::vec& theta_natural) override {
        theta_natural_ = theta_natural;
        theta_unc_     = cfg_.unconstrain(theta_natural);
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return theta_natural_.n_elem; }

    // ---- Hooks used by tests and external tuners ----------------------

    /// Current step size after the most recent call to step(). Useful
    /// for monitoring adaptation and for unit tests.
    double current_step_size() const noexcept {
        return cfg_.nuts_settings.nuts_settings.epsilon_bar_persist;
    }

    /// Cumulative number of adaptation iterations performed by this
    /// block so far.
    std::size_t cumulative_adapt_iter() const noexcept {
        return cfg_.nuts_settings.nuts_settings.adapt_iter_persist;
    }

    /// Overwrite the preconditioner (mass matrix) between sweeps. Users
    /// can feed in an empirical covariance to get a v0-style online
    /// mass-matrix update by freezing the old matrix, collecting a
    /// batch of draws, and calling this with the batch covariance.
    void set_precond_matrix(mcmc::Mat_t M) {
        cfg_.nuts_settings.nuts_settings.precond_mat = std::move(M);
    }

    /// T13: snapshot of NUTS dual-averaging adaptation state.
    /// Everything the user would need to checkpoint + resume the chain
    /// exactly: current step size, DA accumulators, persistent iter
    /// counter, and the preconditioner matrix (if non-identity).
    adaptation_info get_adaptation() const {
        const auto& ns = cfg_.nuts_settings.nuts_settings;
        adaptation_info out;
        out.step_size    = ns.step_size;
        out.epsilon_bar  = ns.epsilon_bar_persist;
        out.h_val        = ns.h_val_persist;
        out.mu_val       = ns.mu_val_persist;
        out.adapt_iter   = static_cast<std::size_t>(ns.adapt_iter_persist);
        // Precond matrix: expose only if the user installed one (i.e., its
        // size matches dim × dim). mcmclib's default is an empty Mat_t.
        if (static_cast<std::size_t>(ns.precond_mat.n_elem) ==
            theta_unc_.n_elem * theta_unc_.n_elem) {
            out.precond_mat = ns.precond_mat;
            out.metric_kind = "dense";
        } else {
            out.metric_kind = "identity";
        }
        return out;
    }

    /// T13: restore a previously-captured adaptation snapshot.
    /// Any missing field leaves the corresponding slot unchanged.
    /// Natural complement to get_adaptation() for checkpoint/restart.
    void set_adaptation(const adaptation_info& ad) {
        auto& ns = cfg_.nuts_settings.nuts_settings;
        ns.step_size             = ad.step_size;
        ns.epsilon_bar_persist   = ad.epsilon_bar;
        ns.h_val_persist         = ad.h_val;
        ns.mu_val_persist        = ad.mu_val;
        ns.adapt_iter_persist    = ad.adapt_iter;
        if (!ad.precond_mat.is_empty()) {
            ns.precond_mat = ad.precond_mat;
        }
    }

    /// The unconstrained scale value (for debugging / monitoring).
    const arma::vec& current_unconstrained() const noexcept {
        return theta_unc_;
    }

    // ---- Kernel-tuning interface (readapt) ------------------------------

    /// NUTS-family supports kernel-tuning via readapt(). Composite
    /// dispatch picks this up via supports_readapt() == true.
    bool supports_readapt() const noexcept override { return true; }

    /// Re-tune dual-averaging state by running n adaptation iterations
    /// at the current chain state. Chain state (theta_unc_,
    /// theta_natural_) is restored bitwise after the n iterations.
    /// Mass matrix (precond_mat) is preserved — mcmclib does not adapt
    /// it automatically.
    ///
    /// See system_design.md §13 NUTS-family + validator.md §24.
    void readapt(std::size_t n,
                 bool reset,
                 std::mt19937_64& rng,
                 std::size_t max_tree_depth_override = 0) override {
        if (n == 0) return;

        // 1. Snapshot chain state (block-local).
        const arma::vec snap_theta_unc     = theta_unc_;
        const arma::vec snap_theta_natural = theta_natural_;
        const bool      snap_first_call    = first_call_;

        auto& ns = cfg_.nuts_settings.nuts_settings;

        // 2. If reset, reinitialize dual-averaging state to defaults.
        if (reset) {
            const double init_step =
                (cfg_.initial_step_size > 0.0) ? cfg_.initial_step_size : 1.0;
            ns.epsilon_bar_persist = init_step;
            ns.h_val_persist       = 0.0;
            ns.mu_val_persist      = std::log(10.0 * init_step);
            ns.adapt_iter_persist  = 0;
            // Note: precond_mat preserved (user-installed if any; identity
            // by default — mcmclib does not adapt it automatically).
        }

        // 3. Save settings we'll override for the readapt call.
        const std::size_t saved_n_burnin = ns.n_burnin_draws;
        const std::size_t saved_n_keep   = ns.n_keep_draws;
        const std::size_t saved_n_adapt  = ns.n_adapt_draws;
        const std::uint64_t saved_seed   = cfg_.nuts_settings.rng_seed_value;
        const std::size_t saved_mtd      = ns.max_tree_depth;

        // Configure for n adaptation iterations (mcmclib requires
        // n_keep >= 1; we discard the kept draw).
        ns.n_burnin_draws = n;
        ns.n_keep_draws   = 1;
        ns.n_adapt_draws  = n;
        cfg_.nuts_settings.rng_seed_value =
            static_cast<std::uint64_t>(rng());
        if (max_tree_depth_override > 0)            // per-call tree-depth cap
            ns.max_tree_depth = max_tree_depth_override;

        auto adapter = [this](const mcmc::ColVec_t& theta_in,
                              mcmc::ColVec_t* grad_out,
                              void* /*unused*/) -> mcmc::fp_t {
            const double lp = cfg_.log_density_grad(theta_in,
                                                    context_,
                                                    grad_out);
            return static_cast<mcmc::fp_t>(lp);
        };

        // 4. Run n adaptation iterations. mcmclib::nuts mutates
        //    theta_unc_ during burnin and updates DA persistent state
        //    (epsilon_bar_persist, h_val_persist, etc.).
        mcmc::Mat_t draws_out;
        const bool ok = mcmc::nuts(
            theta_unc_, adapter, draws_out,
            /*target_data=*/ nullptr,
            cfg_.nuts_settings);

        // 5. Restore settings we overrode.
        ns.n_burnin_draws = saved_n_burnin;
        ns.n_keep_draws   = saved_n_keep;
        ns.n_adapt_draws  = saved_n_adapt;
        cfg_.nuts_settings.rng_seed_value = saved_seed;
        ns.max_tree_depth = saved_mtd;

        if (!ok) {
            // Restore state even on failure, then throw.
            theta_unc_     = snap_theta_unc;
            theta_natural_ = snap_theta_natural;
            first_call_    = snap_first_call;
            throw std::runtime_error(
                "nuts_block '" + cfg_.name + "': readapt failed");
        }

        // 6. Restore chain state. Kernel state (DA accumulators +
        //    epsilon_bar) preserved from the n-iteration adaptation.
        theta_unc_     = snap_theta_unc;
        theta_natural_ = snap_theta_natural;
        first_call_    = snap_first_call;
        // History buffer untouched: the n internal iterations did NOT
        // call push_back (no draws_out entries were recorded into
        // history_buf_; see step() for that path).
    }

    // ---- History overrides -----------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_buf_,
                                        theta_natural_);
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override {
        history_buf_.clear();
    }

private:
    nuts_block_config          cfg_;
    arma::vec                  theta_unc_;
    arma::vec                  theta_natural_;
    block_context              context_;
    bool                       first_call_ = true;
    std::vector<arma::vec>     history_buf_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_NUTS_BLOCK_HPP

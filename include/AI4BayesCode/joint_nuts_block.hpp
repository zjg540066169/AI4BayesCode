/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  joint_nuts_block.hpp  --  a single NUTS block that owns MULTIPLE named
 *                            sub-parameters and samples them JOINTLY on a
 *                            single concatenated unconstrained vector.
 *
 *  WHEN TO USE (vs. nuts_block)
 *  ============================
 *  joint_nuts_block is the escape hatch for models where two or more
 *  continuous parameters are TIGHTLY COUPLED in the likelihood and
 *  splitting them into separate Gibbs-wise NUTS blocks would make the
 *  step size collapse (and therefore the NUTS tree max out). Classic
 *  examples:
 *
 *    * IRT / Rasch:      y ~ sigma(theta_i - b_j)
 *                        theta and b share a shift-invariance direction.
 *    * Linear model:     y ~ N(alpha + X beta, sigma^2)
 *                        alpha and beta are correlated through the design
 *                        matrix.
 *    * Hierarchical LM:  y ~ N(X beta + Z u, sigma^2), u ~ N(0, tau^2)
 *                        beta and u both live in the mean structure and
 *                        must be sampled jointly to get a decent step size.
 *
 *  Modular NUTS (one nuts_block per parameter) is still the DEFAULT
 *  in the codegen skill because it produces simpler, auditable code with
 *  lower semantic-bug risk. Only switch to joint_nuts_block when the
 *  generator has identified strong coupling, or when a previously-generated
 *  modular version has validated correctly (R2 + R3) but is too slow to
 *  be practical.
 *
 *  KEY DESIGN CHOICES
 *  ==================
 *  * Per-slice constraints: 15 kinds. DIMENSION-PRESERVING (unconstrained
 *    slice width == natural width): REAL, POSITIVE, LOWER_BOUNDED,
 *    UPPER_BOUNDED, INTERVAL, ORDERED, POSITIVE_ORDERED, UNIT_VECTOR,
 *    OFFSET_MULTIPLIER. DIMENSION-CHANGING (unc width != natural width):
 *    SIMPLEX (K-1), SUM_TO_ZERO (K-1), CHOLESKY_CORR, CHOLESKY_FACTOR_COV,
 *    CORR_MATRIX, COV_MATRIX. Each adds per-slice constrain/unconstrain +
 *    Jacobian bookkeeping via constraints.hpp. Dim-changing kinds use the
 *    dual unc/nat offset scheme (unc_offsets_/unc_dims_ vs offsets_), gated
 *    by has_dim_changing_; matrix kinds auto-enable the diagonal metric.
 *  * The concatenated vector theta_cat = [sub_params[0]; sub_params[1]; ...]
 *    in the order they were added to the config. The user's joint
 *    log-density reads theta_cat and writes a grad of the same length;
 *    it is responsible for its own slicing. The validator's Check #11
 *    audits this file's slicing.
 *  * After each step(), the joint block splits theta_cat back into named
 *    sub-parameter vectors and exposes them via current_named_outputs();
 *    composite_block writes each sub-parameter into shared_data under its
 *    own key. Downstream blocks (e.g. a separate sigma-NUTS) read them as
 *    usual.
 *
 *  VALIDATOR CONTRACT
 *  ==================
 *  Any use of joint_nuts_block in generated code MUST pass validator
 *  Check #11 (skills/validator.md). The constructor asserts that the
 *  sum of sub_param dimensions equals the length of initial_cat; the
 *  log-density adapter asserts on its input dim on the first call. Both
 *  of those are cheap catches for slicing / offset bugs.
 *================================================================================*/

#ifndef AI4BAYESCODE_JOINT_NUTS_BLOCK_HPP
#define AI4BAYESCODE_JOINT_NUTS_BLOCK_HPP

#include "block_sampler.hpp"
// nuts_block.hpp defines log_density_gradient_fn and transform_fn, which
// joint_nuts_block reuses. Pulling it in here also means a user .cpp that
// only includes joint_nuts_block.hpp does not have to include
// nuts_block.hpp separately.
#include "nuts_block.hpp"
// constraints::<kind>::wrap — per-slice transform + log|Jacobian| + gradient
// chain-rule for POSITIVE sub-params (system_design.md §10.1). All Jacobian
// arithmetic lives in constraints.hpp; this block never hand-rolls it.
#include "constraints.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include <limits>

#include "mcmclib/mcmc.hpp"

namespace AI4BayesCode {

/**
 * @brief One sub-parameter inside a joint_nuts_block.
 *
 * The order of sub_params in joint_nuts_block_config defines the layout of
 * the concatenated vector theta_cat: sub_params[0] occupies offsets
 * [0, sub_params[0].dim), sub_params[1] occupies
 * [sub_params[0].dim, sub_params[0].dim + sub_params[1].dim), etc.
 *
 * V0: name and dim only, all real-valued (identity transform). V1 will add
 * per-sub-param constrain / unconstrain / log-Jacobian hooks.
 */
/// Per-slice constraint kind for a joint_nuts_block sub-parameter. All kinds
/// here are DIMENSION-PRESERVING (unconstrained slice width == natural slice
/// width), so they fit the fixed concatenated-slice layout. The per-slice
/// transform + log|Jacobian| + gradient chain-rule all live in constraints.hpp
/// (system_design.md §10.1); this block only dispatches to them.
///   REAL          : identity (unconstrained == natural), no Jacobian.
///   POSITIVE      : natural = exp(unc).                  constraints::positive
///   LOWER_BOUNDED : natural = lower + exp(unc).          uses sub_param.lower
///   UPPER_BOUNDED : natural = upper - exp(unc).          uses sub_param.upper
///   INTERVAL      : natural = lower + (upper-lower)*sigmoid(unc).
///                                                        uses lower AND upper
///   ORDERED       : strictly-increasing K-vector (log-gap encoding). ordered
///   POSITIVE_ORDERED : strictly-increasing K-vector, first element > 0.
///   UNIT_VECTOR   : K-vector on the (K-1)-sphere (||x|| = 1).   unit_vector
///   OFFSET_MULTIPLIER : natural = offset + multiplier*unc (affine non-center);
///                       offset = sub_param.lower, multiplier = sub_param.upper.
/// DIMENSION-CHANGING kinds (unconstrained slice width < natural width;
/// sub_param.dim is the NATURAL width, the unconstrained width is derived via
/// the dual unc/nat offset scheme gated by has_dim_changing_):
///   SIMPLEX        : K natural (sum 1, >0), K-1 unc (stick-breaking).
///   SUM_TO_ZERO    : K natural (sum 0), K-1 unc (isometric basis).
///   CHOLESKY_CORR  : K*K flattened Cholesky-corr factor, K(K-1)/2 unc (LKJ).
///   CHOLESKY_FACTOR_COV : K*K flattened cov factor, K(K+1)/2 unc.
///   CORR_MATRIX    : K*K flattened correlation matrix, K(K-1)/2 unc.
///   COV_MATRIX     : K*K flattened covariance matrix, K(K+1)/2 unc.
/// (stochastic_* column/row-simplex matrices exist in constraints.hpp for the
/// single nuts_block but are not yet wired here — they need a columns count.)
enum class joint_constraint { REAL, POSITIVE, LOWER_BOUNDED, UPPER_BOUNDED,
                              INTERVAL, ORDERED, POSITIVE_ORDERED, UNIT_VECTOR,
                              OFFSET_MULTIPLIER,
                              // DIMENSION-CHANGING (unc width < natural width;
                              // sub_param.dim is the NATURAL width: K for
                              // SIMPLEX/SUM_TO_ZERO, K*K flattened for matrices):
                              SIMPLEX, SUM_TO_ZERO, CHOLESKY_CORR,
                              CHOLESKY_FACTOR_COV, CORR_MATRIX, COV_MATRIX };

struct joint_nuts_sub_param {
    std::string  name;  // key under which this sub-param is stored in
                        // shared_data after write-back. MUST be unique
                        // within the joint block.
    std::size_t  dim;   // length of this sub-param's slice in theta_cat.
    /// Per-slice constraint. Default REAL preserves V0 (all-real) behavior
    /// bit-for-bit; set POSITIVE for sub-params like sigma / tau.
    joint_constraint constraint = joint_constraint::REAL;
    /// Bounds for parameterized constraints. `lower` is read by LOWER_BOUNDED
    /// and INTERVAL; `upper` is read by UPPER_BOUNDED and INTERVAL. Ignored by
    /// REAL / POSITIVE / ORDERED. INTERVAL requires upper > lower (constructor
    /// validates this).
    double lower = 0.0;
    double upper = 0.0;
};

/**
 * @brief Configuration for joint_nuts_block.
 */
struct joint_nuts_block_config {
    /// Unique name for THIS joint block within its composite. Used for
    /// Gibbs DAG bookkeeping (declare_dependencies, declare_invalidates).
    /// Does NOT appear as a key in shared_data; sub-parameter names do.
    std::string name;

    /// Ordered list of sub-parameters. Their dims determine the layout of
    /// theta_cat. Sub-parameter names must all be unique and must be the
    /// same names that downstream blocks read via declare_dependencies.
    std::vector<joint_nuts_sub_param> sub_params;

    /// Joint log-density oracle on the concatenated NATURAL scale. The block
    /// applies each slice's constrain/unconstrain transform and adds log|J|
    /// internally (constraints.hpp); the oracle returns the natural-scale lp
    /// and grad = d lp / d(natural). For an all-REAL block, natural ==
    /// unconstrained, so this is bit-identical to the old V0 contract and no
    /// Jacobian arises. The oracle MUST NOT add any per-slice Jacobian term.
    /// A non-finite (-inf) return is a valid NUTS reject; the block then
    /// ignores the gradient (it need not be sized), so an early
    /// `if (...) return -inf;` guard before sizing grad is safe.
    ///
    /// Validator Check #11 enforces: every sub-parameter prior contributes;
    /// gradient slicing matches sub-param offsets; the oracle does not
    /// hand-roll the per-slice Jacobian.
    log_density_gradient_fn log_density_grad;

    /// Initial value of theta_cat. Must have length == sum(sub_param.dim).
    /// The constructor asserts this.
    arma::vec initial_cat;

    /// Forwarded NUTS settings. See nuts_block.hpp for the full story.
    mcmc::algo_settings_t nuts_settings;

    double      initial_step_size       = 0.0;
    std::size_t n_warmup_first_call     = 500;  // joint blocks tend to
                                                // need more runway than
                                                // modular NUTS because
                                                // the dimension is bigger
    std::size_t n_warmup_per_step       = 0;
    std::size_t n_draws_per_step        = 1;

    /// Max NUTS tree depth: caps each transition at 2^max_tree_depth - 1
    /// leapfrog steps. Forwarded to mcmclib `nuts_settings.max_tree_depth`.
    /// Default 10 (Stan default; preserves prior behaviour). LOWER it
    /// (e.g. 6-8) to cap per-transition cost on high-dim / ill-conditioned
    /// targets where the No-U-Turn criterion otherwise lets trees saturate
    /// at depth 10 (~1023 leapfrogs / transition) — trades trajectory length
    /// for wall-clock speed. Was previously reachable only via the nested
    /// `nuts_settings.nuts_settings.max_tree_depth`; surfaced here as a
    /// first-class knob so codegen / users can tune it.
    std::size_t max_tree_depth          = 10;

    // -------------------------------------------------------------------
    // T11 (v1.1): online dense-metric adaptation via Welford (SHIPPED
    // 2026-04-20).
    //
    // When `use_dense_metric = true`, the FIRST step() call runs a pilot
    // phase:
    //   1. Run NUTS with identity metric for `dense_metric_pilot_iters`
    //      burn-in + `dense_metric_adapt_iters` collected samples.
    //   2. Compute the sample covariance of the collected samples (Welford
    //      algorithm) and install it as the precond matrix for all
    //      subsequent step() calls.
    //   3. Return the LAST pilot sample as this first step's draw.
    //
    // This addresses the known HierLM G=50 failure mode where identity-
    // metric NUTS can't navigate the strongly-correlated posterior even at
    // 10k+10k chain length. Stan-style three-phase warmup is overkill for
    // our typical model dims; this one-shot pilot + dense-metric switch
    // recovers most of the benefit with minimal complexity.
    //
    // DEFAULT: off (use_dense_metric = false) — all existing examples
    // continue to use identity metric. Opt-in for strongly-correlated
    // posteriors.
    // -------------------------------------------------------------------
    bool        use_dense_metric        = false;
    std::size_t dense_metric_pilot_iters  = 200;   // identity-metric burn-in
    std::size_t dense_metric_adapt_iters  = 500;   // samples for covariance

    // -------------------------------------------------------------------
    // v1.2: Stan-style 3-phase windowed warmup. DEFAULT OFF (2026-06-20
    // broad-corpus revision): single-pilot is the recommended default for
    // dense; 3-phase is an EXTREME-COND escalation only. Set
    // use_three_phase_warmup = true to opt in.
    //
    // Triggered when use_dense_metric == true AND use_three_phase_warmup ==
    // true. Replaces the single-phase pilot (adapt_dense_metric_) with a
    // Stan-style 3-phase schedule (Carpenter et al. 2017, Stan reference
    // manual §16.2):
    //   Phase I (tp_phase1_iters): step-size only, identity mass matrix.
    //   Phase II (tp_phase2_windows): expanding windows of mass matrix
    //     adaptation. Each window resets Welford, runs sampler under
    //     previous window's mass matrix, computes Welford cov on all
    //     window iter, installs regularized precision as new mass matrix.
    //   Phase III (tp_phase3_iters): final step-size tune with frozen
    //     mass matrix.
    //
    // Total default budget: 75 + (25+50+100+200+500) + 50 = 1000 iter.
    //
    // After 3-phase completes, first_call_ is set to FALSE so the main
    // step() path skips n_warmup_first_call (the warmup is already done).
    //
    // DEFAULT-OFF RATIONALE (REVISED 2026-06-20): a BROAD 31-model corpus
    // showed single-pilot dense gives 5-23x higher keep-phase ESS/s than
    // 3-phase (3-phase's short Phase III + dual-averaging epsilon_bar lag
    // UNDER-tune the step) and converges across all realistic cond (corpus
    // max 609, 5/5). 3-phase only wins at EXTREME cond (~thousands, where one
    // identity pilot cannot bootstrap the covariance) or high-curvature
    // ridge-trapping (sparse spatial/ICAR). The 3 dense-using shipped examples
    // were re-validated under single-pilot (2026-06-20): HSGP 3-phase was
    // actually BROKEN (R-hat 1.017) and single-pilot FIXED it (1.0015, 6x ESS);
    // BSpline + HierLM both converge on single-pilot. So single-pilot is the
    // default; opt into 3-phase ONLY for a documented extreme-cond bootstrap
    // failure (validation-driven escalation). The earlier "2x wall + 4x ESS"
    // claim held only on the narrow small-spatial-RE fixtures it was measured
    // on, not in general. HARD RULE retained: 3-phase requires dense metric
    // (diagonal + 3-phase is gated off — it under-tunes the step ~38x).
    // -------------------------------------------------------------------
    bool                     use_three_phase_warmup = false;
    std::size_t              tp_phase1_iters        = 75;
    std::vector<std::size_t> tp_phase2_windows      = {25, 50, 100, 200, 500};
    std::size_t              tp_phase3_iters        = 50;

    // -------------------------------------------------------------------
    // v1.3 (2026-06-16): warmup-robustness defaults — prevent the
    // single-first-call-warmup step-size COLLAPSE that froze chains
    // (sd=0) on boundary-singular / overdispersed / ill-conditioned
    // targets. The mcmclib dual-averaging path has no eps floor/cap and,
    // once eps_bar underflows to ~0, use_persistent_adapt locks it for
    // every subsequent draw => infinitesimal leapfrog => identical draws.
    // These knobs are ON by default; set the disabling values below for
    // legacy bit-identical behaviour.
    // -------------------------------------------------------------------
    // (1) Step-size FLOOR/CAP applied to the persisted dual-averaged step
    //     after each warmup. The CAP is the freeze fix: a low-curvature
    //     boundary region (Beta/binomial near 0/1; overdispersed starts)
    //     fools the dual-averaging into a runaway (eps -> 1e+77), so every
    //     leapfrog flies to infinity -> trajectory rejected -> the chain is
    //     frozen (sd=0). Capping eps at pi (the nuts-rs standard) bounds the
    //     runaway; the floor is cheap insurance against the opposite (rare)
    //     collapse. min_step_size = 0 disables the floor; max_step_size = 0
    //     disables the cap.
    double      min_step_size            = 1e-6;
    double      max_step_size            = 3.141592653589793;  // pi (nuts-rs)
    // (2) MODE-REFINEMENT of the initial value: a few backtracking
    //     gradient-ascent steps (Armijo) that move an overdispersed /
    //     boundary-singular start toward the typical set (O(1) gradient)
    //     so warmup does not start in a stiff region. Only activates when
    //     the initial unconstrained gradient is large; a no-op otherwise
    //     (so well-conditioned inits are unchanged).
    bool        refine_initial_mode      = false;  // opt-in (cap already fixes freeze)
    std::size_t refine_max_iters         = 50;
    double      refine_grad_trigger      = 10.0;  // refine only if max|grad|>this
    double      refine_grad_target       = 1.0;   // stop once max|grad|<this
    // (3) Adapted DIAGONAL metric (Stan/NumPyro default) — opt-in; the robust
    //     middle ground between identity and the (fragile) full-dense metric.
    //     Uses the windowed warmup but installs diag(1/Var) — per-axis
    //     rescaling — instead of identity or full-dense. This is the fix for
    //     axis-aligned ill-conditioning (e.g. positive params spanning orders
    //     of magnitude in scale). `max_mass_variance` is an OPTIONAL upper cap
    //     on the per-axis variance: 0 => no cap (full rescaling, the default);
    //     set it > 0 ONLY to bound a funnel's unreliable variance estimate so
    //     the metric cannot amplify the effective step past identity. The eps
    //     cap above is the universal freeze guard, so no variance cap is needed
    //     for robustness.
    bool        use_diagonal_metric      = false;
    double      max_mass_variance        = 0.0;   // 0 => no cap (full rescaling)

    // (4) AUTO-SELECT diagonal vs dense metric (2026-06-19). For COMPLEX / NEW
    //     models where the codegen cannot know a priori whether the posterior is
    //     strongly correlated. When `auto_select_metric = true`, the single-pilot
    //     estimates the sample covariance Sigma, forms the correlation matrix R,
    //     and picks DENSE when the posterior is strongly correlated (condition
    //     number of R > auto_dense_cond_threshold) AND the dimension is modest
    //     (<= auto_dense_max_dim, since dense is fragile/expensive in high dim);
    //     otherwise DIAGONAL (which already covers the well-conditioned and
    //     axis-scaled cases). Off by default => existing behavior unchanged.
    bool        auto_select_metric         = false;
    double      auto_dense_cond_threshold  = 4.0;   // cond(R) above this => dense
    std::size_t auto_dense_max_dim         = 100;   // dense only if unc-dim <= this
};

/**
 * @brief A joint NUTS block that owns multiple named sub-parameters and
 * samples them jointly over the concatenated unconstrained vector.
 *
 * @see file header for design rationale and validator contract.
 */
class joint_nuts_block : public block_sampler {
public:
    explicit joint_nuts_block(joint_nuts_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "joint_nuts_block: name must be non-empty");
        }
        if (cfg_.sub_params.empty()) {
            throw std::invalid_argument(
                "joint_nuts_block: sub_params must be non-empty "
                "(a joint block with 0 sub-params is just a nuts_block)");
        }
        if (!cfg_.log_density_grad) {
            throw std::invalid_argument(
                "joint_nuts_block: log_density_grad is required");
        }

        // Compute total dim, check uniqueness of sub-param names, build
        // offset map for slicing on write-back.
        total_dim_ = 0;
        total_unc_dim_ = 0;
        has_dim_changing_ = false;
        offsets_.reserve(cfg_.sub_params.size());
        unc_offsets_.reserve(cfg_.sub_params.size());
        unc_dims_.reserve(cfg_.sub_params.size());
        std::unordered_map<std::string, int> seen;
        for (const auto& sp : cfg_.sub_params) {
            if (sp.name.empty()) {
                throw std::invalid_argument(
                    "joint_nuts_block: sub_param.name must be non-empty");
            }
            if (sp.dim == 0) {
                throw std::invalid_argument(
                    "joint_nuts_block: sub_param.dim must be > 0 "
                    "(sub_param '" + sp.name + "')");
            }
            if (seen.count(sp.name)) {
                throw std::invalid_argument(
                    "joint_nuts_block: duplicate sub_param name '" +
                    sp.name + "'");
            }
            seen[sp.name] = 1;
            // sp.dim is the NATURAL slice width K; the UNCONSTRAINED width is
            // derived (equal for dim-preserving kinds, K-1 for SIMPLEX).
            const std::size_t ud = unc_dim_for_(sp.constraint, sp.dim);
            if (ud != sp.dim) has_dim_changing_ = true;
            offsets_.push_back(total_dim_);         total_dim_     += sp.dim;
            unc_offsets_.push_back(total_unc_dim_); unc_dims_.push_back(ud);
            total_unc_dim_ += ud;
        }

        if (cfg_.initial_cat.n_elem != total_dim_) {
            throw std::invalid_argument(
                "joint_nuts_block '" + cfg_.name +
                "': initial_cat length (" +
                std::to_string(cfg_.initial_cat.n_elem) +
                ") does not match sum of sub_param dims (" +
                std::to_string(total_dim_) + ")");
        }

        // Every slice REAL? Enables bit-identical V0 fast-paths. Also validate
        // per-slice bound configuration for parameterized constraints.
        all_real_ = true;
        for (const auto& sp : cfg_.sub_params) {
            if (sp.constraint != joint_constraint::REAL) all_real_ = false;
            if (sp.constraint == joint_constraint::INTERVAL &&
                !(sp.upper > sp.lower)) {
                throw std::invalid_argument(
                    std::string("joint_nuts_block '") + cfg_.name +
                    "': INTERVAL sub_param '" + sp.name +
                    "' requires upper > lower (got lower=" +
                    std::to_string(sp.lower) + ", upper=" +
                    std::to_string(sp.upper) + ")");
            }
            if (sp.constraint == joint_constraint::OFFSET_MULTIPLIER &&
                sp.upper == 0.0) {
                throw std::invalid_argument(
                    std::string("joint_nuts_block '") + cfg_.name +
                    "': OFFSET_MULTIPLIER sub_param '" + sp.name +
                    "' requires a nonzero multiplier (set sub_param.upper; "
                    "offset = sub_param.lower)");
            }
        }
        // One-shot-success default: matrix-valued constraints (covariance /
        // correlation matrices and their Cholesky factors) induce sharply
        // ill-conditioned UNCONSTRAINED posteriors — an IDENTITY mass matrix
        // reliably FREEZES (every leapfrog proposal overshoots, accept->0,
        // chains stuck at init: validated COV_MATRIX K=3 identity R-hat=Inf,
        // all four chains sd=0). The diagonal metric adapts per-axis curvature
        // and converges (R-hat~1.00, correct posterior recovery err<0.04). So
        // when such a constraint is present and the user has NOT explicitly
        // chosen a metric, default to the diagonal metric. This makes a
        // generated covariance/correlation model converge on the FIRST run
        // instead of forcing the user to discover use_diagonal_metric by hand.
        // SIMPLEX / SUM_TO_ZERO are well-conditioned and keep the identity
        // default (no behavior change). Fully overridable: set either metric
        // flag explicitly to take control.
        if (!cfg_.use_dense_metric && !cfg_.use_diagonal_metric) {
            for (const auto& sp : cfg_.sub_params) {
                if (sp.constraint == joint_constraint::CHOLESKY_CORR ||
                    sp.constraint == joint_constraint::CHOLESKY_FACTOR_COV ||
                    sp.constraint == joint_constraint::CORR_MATRIX ||
                    sp.constraint == joint_constraint::COV_MATRIX) {
                    cfg_.use_diagonal_metric = true;
                    break;
                }
            }
        }

        // Validate initial_cat lies in-support (it is on the NATURAL scale).
        if (!all_real_) validate_support_(cfg_.initial_cat, "constructor");

        if (cfg_.n_draws_per_step == 0) cfg_.n_draws_per_step = 1;

        auto& ns = cfg_.nuts_settings.nuts_settings;
        ns.use_persistent_adapt = true;
        ns.max_tree_depth       = cfg_.max_tree_depth;   // first-class knob (default 10)

        if (cfg_.initial_step_size > 0.0) {
            ns.epsilon_bar_persist = cfg_.initial_step_size;
            ns.h_val_persist       = 0.0;
            ns.mu_val_persist      =
                std::log(10.0 * cfg_.initial_step_size);
            ns.adapt_iter_persist  = 1;
            ns.step_size           = cfg_.initial_step_size;
        }

        // initial_cat is NATURAL-scale; the sampler integrates UNCONSTRAINED.
        // nat_to_unc_ is identity when all_real_ (=> theta_cat_ == initial_cat).
        theta_cat_  = nat_to_unc_(cfg_.initial_cat);
        first_call_ = true;

        // Seed the named outputs cache so add_child's initial-value seed
        // into shared_data has something to work with.
        rebuild_named_outputs_();
    }

    // ---- block_sampler interface ---------------------------------------

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // T11: dense-metric pilot adaptation (opt-in, first call only).
        // Runs an initial mcmc::nuts with identity metric, collects samples
        // across dense_metric_adapt_iters, computes sample covariance, and
        // installs it as the precond matrix for all subsequent calls.
        //
        // CRITICAL: after pilot adaptation we KEEP first_call_ = true so the
        // next step() call runs a full n_warmup_first_call warmup with the
        // new dense metric. The step size learned under identity metric is
        // inappropriate for the new metric's leapfrog dynamics, and with
        // default n_warmup_per_step = 0, subsequent steps would use stale
        // step size. Re-running the first warmup under dense metric fixes
        // that. This doubles the first-call cost (pilot + re-warmup) but
        // produces a correctly-tuned chain — the cost the user pays for
        // opt-in dense-metric accuracy.
        if ((cfg_.use_dense_metric || cfg_.use_diagonal_metric ||
             cfg_.auto_select_metric) &&
            first_call_ && !dense_metric_adapted_) {
            // 3-phase windowed warmup is gated to the DENSE metric. Although its
            // per-window build_precond_ CAN build a diagonal metric, the diagonal
            // path EMPIRICALLY mis-tunes the STEP: 3-phase gives the step-size
            // dual-averaging only ~1000 iters, but our DA's epsilon_bar averaging
            // lag needs ~2500 to converge, so under a DIAGONAL metric (which
            // leaves the posterior ill-conditioned) the step UNDERSHOOTS and
            // mixing collapses. RE-VERIFIED 2026-06-19 on pl2 2PL-IRT: diag+3phase
            // step 0.103 / ESS-per-s 4.6 vs diag+single-pilot step 0.230 / ESS/s
            // 173.6 (~38x worse). DENSE tolerates the smaller step (its metric
            // makes the geometry isotropic), so 3-phase is dense-only; DIAGONAL
            // uses the single pilot below (which gives the DA the ~2500 iters it
            // needs and is the BEST config for diagonal-sufficient models).
            if (cfg_.use_three_phase_warmup && cfg_.use_dense_metric) {
                // Dense + 3-phase: OPT-IN extreme-cond escalation (default OFF
                // since 2026-06-20; single-pilot is the dense default).
                // Guarded: if 3-phase escapes the typical set on a stiff target,
                // three_phase_guarded_ resets to identity and returns false, and
                // first_call_ stays true so the main step() path re-warms under
                // identity. On success, the full Stan-style schedule is done and
                // the main path proceeds with n_warmup_per_step (default 0).
                dense_metric_adapted_ = true;
                if (three_phase_guarded_(rng)) first_call_ = false;
            } else {
                adapt_dense_metric_(rng);
                dense_metric_adapted_ = true;
                // AUTO-SELECT uses the SINGLE PILOT for whichever metric it picks
                // (diagonal OR dense). REVERTED 2026-06-20 the earlier
                // "auto-dense -> 3-phase upgrade": a broad 31-model corpus showed
                // that upgrade made auto 10-37x SLOWER in keep-phase ESS/s than
                // single-pilot dense (3-phase's short Phase III under-tunes the
                // step via the DA averaging lag; verified timing-independently),
                // while single-pilot dense CONVERGES across all realistic
                // condition numbers (corpus max cond 609, 5/5). 3-phase dense is
                // only needed at EXTREME cond (~thousands, where the single pilot
                // cannot bootstrap the covariance) — rare, and usually a
                // reparameterization smell; for that, set use_dense_metric +
                // use_three_phase_warmup explicitly, or reparameterize. So auto
                // falls through to the single-pilot main warmup below.
                // (Escape / degenerate-pilot guards still protect this path.)
            }
        }

        cfg_.nuts_settings.rng_seed_value =
            static_cast<std::uint64_t>(rng());

        auto& ns = cfg_.nuts_settings.nuts_settings;
        std::size_t n_warmup =
            first_call_ ? cfg_.n_warmup_first_call : cfg_.n_warmup_per_step;

        // Single-pilot metric blocks (diagonal / dense, NOT 3-phase): the
        // step-size dual-averaging converges monotonically but slowly against
        // the freshly-installed metric (epsilon_bar averaging lag — pl2 2PL-IRT
        // needs ~2500 iters to reach the Stan-level step ~0.23; see the
        // find_reasonable_epsilon Algorithm-4 fix in nuts.ipp). This is ONE
        // clean dual-averaging pass (no window/restart noise), so just make it
        // long enough. 3-phase has already run its own schedule (first_call_ ==
        // false here); identity blocks need no metric-step settling.
        if (first_call_ && dense_metric_adapted_ &&
            (cfg_.use_diagonal_metric || cfg_.use_dense_metric)) {
            n_warmup = std::max<std::size_t>(n_warmup, 2500);
        }

        ns.n_burnin_draws = n_warmup;
        ns.n_keep_draws   = cfg_.n_draws_per_step;
        ns.n_adapt_draws  = n_warmup;

        // Snapshot frozen slot values in the unconstrained space (Approach B
        // from DESIGN_NOTES Sec.10.a). When any slot is frozen, the adapter
        // wrap OVERRIDES the theta values on frozen dims with the snapshot
        // and ZEROES the gradient on those dims, so mcmclib sees a target
        // that is CONSTANT in the frozen coordinates. Combined with the
        // block-diagonal mass matrix (built by build_precond_ when frozen
        // dims are non-empty), theta_z's dynamics fully decouple from
        // theta_f, and theta_f's marginal is exactly the conditional
        // posterior given theta_z_snap.
        const bool any_frozen = frozen_unc_idx_.n_elem > 0;
        arma::vec theta_frozen_snap;
        if (any_frozen) {
            theta_frozen_snap = theta_cat_.elem(frozen_unc_idx_);
        }

        // Per-slice transform + Jacobian (identity fast-path when all-real),
        // dim asserts, and the natural-scale oracle call all live in eval_unc_.
        auto adapter = [this, any_frozen, &theta_frozen_snap](
                            const mcmc::ColVec_t& theta_in,
                            mcmc::ColVec_t* grad_out,
                            void* /*unused*/) -> mcmc::fp_t {
            if (!any_frozen) {
                return static_cast<mcmc::fp_t>(eval_unc_(theta_in, grad_out));
            }
            // FROZEN path: override frozen dims with snapshot BEFORE the
            // user's log-density sees them. This makes U_wrap(theta) =
            // U(theta_f, theta_z_snap) -- constant in theta_z per Sec.10.a.
            arma::vec theta_eff(theta_in.n_elem);
            for (arma::uword i = 0; i < theta_in.n_elem; ++i) theta_eff[i] = theta_in[i];
            theta_eff.elem(frozen_unc_idx_) = theta_frozen_snap;
            double lp = eval_unc_(theta_eff, grad_out);
            // Zero gradient on frozen dims so leapfrog leaves them alone
            // even numerically (they wouldn't move anyway under M_z = I +
            // grad_z = 0, but explicit zeroing keeps mcmclib's U-turn and
            // cache invariants exactly on the free-dim manifold).
            if (grad_out != nullptr && grad_out->n_elem == theta_in.n_elem) {
                grad_out->elem(frozen_unc_idx_).zeros();
            }
            return static_cast<mcmc::fp_t>(lp);
        };

        // Snapshot the pre-warmup state for the metric-escape guard below. Only
        // armed on the first-call metric warmup (dense / diagonal): that is the
        // single place a freshly-learned metric can drive the chain out of the
        // typical set during its long settling warmup.
        const bool guard_active = first_call_ &&
            (cfg_.use_dense_metric || cfg_.use_diagonal_metric);
        arma::vec pre_warmup_theta;
        double    pre_warmup_lp = 0.0;
        if (guard_active) {
            pre_warmup_theta = theta_cat_;
            pre_warmup_lp    = eval_unc_(theta_cat_, nullptr);
        }

        mcmc::Mat_t draws_out;
        const bool ok = mcmc::nuts(
            theta_cat_,
            adapter,
            draws_out,
            /*target_data=*/ nullptr,
            cfg_.nuts_settings);

        if (!ok) {
            throw std::runtime_error(
                "joint_nuts_block '" + cfg_.name +
                "': mcmclib::nuts returned false");
        }
        if (draws_out.n_rows == 0) {
            throw std::runtime_error(
                "joint_nuts_block '" + cfg_.name +
                "': nuts produced zero draws");
        }

        // ---- Metric-escape guard (2026-06-19) ------------------------------
        // A freshly-learned metric (dense especially) can, from a DISPERSED
        // init on a STIFF model, drive the settling warmup OUT of the typical
        // set: the step dual-averages up toward the ceiling, a leapfrog overflies
        // into an explosive boundary (e.g. ARMA's non-invertible |theta|>1
        // ridge), and the chain freezes there — the post-warmup log-density is
        // then orders of magnitude worse than where warmup started. The pilot
        // and metric can look perfectly healthy (well-conditioned precond), so
        // this is NOT caught by the pilot-degeneracy guard in adapt_dense_metric_.
        // Detect the escape and RE-RUN the warmup from the (good) pre-warmup
        // state under the IDENTITY metric, which is unconditionally stable.
        if (guard_active) {
            const arma::vec post = draws_out.row(draws_out.n_rows - 1).t();
            const double post_lp = eval_unc_(post, nullptr);
            const bool escaped =
                !std::isfinite(post_lp) ||
                (std::isfinite(pre_warmup_lp) && post_lp < pre_warmup_lp - 1.0e3);
            if (escaped) {
                theta_cat_               = pre_warmup_theta;   // good restart
                cfg_.use_dense_metric    = false;
                cfg_.use_diagonal_metric = false;              // => identity
                auto_selected_dense_     = -1;                 // record: aborted
                auto_selected_cond_      = 0.0;
                ns.precond_mat = mcmc::Mat_t();                // identity metric
                ns.epsilon_bar_persist = 0.0;
                ns.h_val_persist       = 0.0;
                ns.mu_val_persist      = 0.0;
                ns.adapt_iter_persist  = 0;
                ns.n_burnin_draws = n_warmup;
                ns.n_adapt_draws  = n_warmup;
                cfg_.nuts_settings.rng_seed_value =
                    static_cast<std::uint64_t>(rng());
                const bool ok2 = mcmc::nuts(
                    theta_cat_, adapter, draws_out,
                    /*target_data=*/ nullptr, cfg_.nuts_settings);
                if (!ok2 || draws_out.n_rows == 0) {
                    throw std::runtime_error(
                        "joint_nuts_block '" + cfg_.name +
                        "': identity-fallback warmup failed after metric escape");
                }
            }
        }

        // v1.3 freeze guard: clamp the persisted dual-averaged step size into
        // [min_step_size, max_step_size] after every call (covers the default,
        // dense, and 3-phase paths). A low-curvature boundary region fools the
        // dual averaging into a runaway (eps -> 1e+77) so every leapfrog flies
        // to infinity -> trajectory rejected -> the chain freezes to sd = 0;
        // the cap bounds the runaway. No-op when eps is already in range
        // (well-conditioned chains), so existing behaviour is unchanged.
        if (cfg_.min_step_size > 0.0 &&
            !(ns.epsilon_bar_persist >= cfg_.min_step_size)) {
            ns.epsilon_bar_persist = cfg_.min_step_size;
        }
        if (cfg_.max_step_size > 0.0 &&
            ns.epsilon_bar_persist > cfg_.max_step_size) {
            ns.epsilon_bar_persist = cfg_.max_step_size;
        }

        theta_cat_ = draws_out.row(draws_out.n_rows - 1).t();
        // Approach B restore: after mcmc::nuts returns, the frozen dims of
        // theta_cat_ contain wherever the (mass=I, grad=0) Brownian drift
        // took them. This is HARMLESS to the free-dim distribution (proof
        // in Sec.10.a), but we restore the snapshot so downstream
        // rebuild_named_outputs_() / current_named_outputs() / history all
        // see the pinned value.
        if (any_frozen) {
            theta_cat_.elem(frozen_unc_idx_) = theta_frozen_snap;
        }
        rebuild_named_outputs_();

        if (keep_history_) {
            // Record the NATURAL-scale concatenated draw; get_history slices
            // on demand (== theta_cat_ when all_real_).
            history_buf_.push_back(theta_cat_nat_);
        }

        first_call_ = false;
    }

    /// For a joint block, current() exposes the CONCATENATED vector. This
    /// is rarely what downstream code wants directly; the composite calls
    /// current_named_outputs() instead, which returns the sliced sub-params.
    const arma::vec& current() const override {
        return theta_cat_nat_;
    }

    /// Overwrite theta_cat with a new concatenated vector. Length must
    /// match total_dim_. Used for checkpointing / overdispersed starts.
    void set_current(const arma::vec& theta_new) override {
        if (theta_new.n_elem != total_dim_) {
            throw std::invalid_argument(
                "joint_nuts_block '" + cfg_.name +
                "': set_current got length " +
                std::to_string(theta_new.n_elem) + ", expected " +
                std::to_string(total_dim_));
        }
        // theta_new is NATURAL-scale; validate per-slice support and map to the
        // unconstrained sampler state (identity when all_real_).
        if (!all_real_) validate_support_(theta_new, "set_current");
        theta_cat_ = nat_to_unc_(theta_new);
        rebuild_named_outputs_();
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return total_dim_; }

    /// AUTO-SELECT diagnostics (valid after first step() when
    /// cfg.auto_select_metric = true). -1 = auto-select not run,
    /// 0 = picked DIAGONAL, 1 = picked DENSE.
    int    auto_selected_metric() const noexcept { return auto_selected_dense_; }
    /// Condition number of the pilot correlation matrix at the auto-select
    /// decision (0 if auto-select did not run).
    double auto_selected_cond()   const noexcept { return auto_selected_cond_; }

    /// Named outputs, one per sub-parameter. This is how composite_block
    /// writes each sub-parameter into shared_data under its own key.
    std::unordered_map<std::string, arma::vec>
    current_named_outputs() const override {
        return named_outputs_cache_;
    }

    // ---- History (per-sub-param decomposition) --------------------------

    history_map get_history() const override {
        history_map out;
        if (history_buf_.empty()) {
            // Fallback: return the current draw as a single-row history,
            // decomposed per sub-param.
            for (std::size_t s = 0; s < cfg_.sub_params.size(); ++s) {
                const auto& sp = cfg_.sub_params[s];
                const arma::vec slice = theta_cat_nat_.subvec(
                    offsets_[s], offsets_[s] + sp.dim - 1);
                arma::mat m(1, sp.dim);
                for (std::size_t j = 0; j < sp.dim; ++j) m(0, j) = slice[j];
                out.emplace(sp.name, std::move(m));
            }
            return out;
        }

        const std::size_t n = history_buf_.size();
        for (std::size_t s = 0; s < cfg_.sub_params.size(); ++s) {
            const auto& sp = cfg_.sub_params[s];
            arma::mat m(n, sp.dim);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < sp.dim; ++j) {
                    m(i, j) = history_buf_[i][offsets_[s] + j];
                }
            }
            out.emplace(sp.name, std::move(m));
        }
        return out;
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override { history_buf_.clear(); }

    // ---- Introspection --------------------------------------------------

    double current_step_size() const noexcept {
        return cfg_.nuts_settings.nuts_settings.epsilon_bar_persist;
    }

    std::size_t cumulative_adapt_iter() const noexcept {
        return cfg_.nuts_settings.nuts_settings.adapt_iter_persist;
    }

    void set_precond_matrix(mcmc::Mat_t M) {
        cfg_.nuts_settings.nuts_settings.precond_mat = std::move(M);
    }

    /// T13: snapshot of joint NUTS adaptation state (step size + DA
    /// accumulators + preconditioner). Complement get_adaptation() on
    /// nuts_block. For dense-metric enabled joint blocks, precond_mat
    /// is the learned precision matrix Σ⁻¹.
    adaptation_info get_adaptation() const {
        const auto& ns = cfg_.nuts_settings.nuts_settings;
        adaptation_info out;
        out.step_size    = ns.step_size;
        out.epsilon_bar  = ns.epsilon_bar_persist;
        out.h_val        = ns.h_val_persist;
        out.mu_val       = ns.mu_val_persist;
        out.adapt_iter   = static_cast<std::size_t>(ns.adapt_iter_persist);
        if (static_cast<std::size_t>(ns.precond_mat.n_elem) ==
            total_unc_dim_ * total_unc_dim_) {
            out.precond_mat = ns.precond_mat;
            out.metric_kind = dense_metric_adapted_ ? "dense" : "identity";
        } else {
            out.metric_kind = "identity";
        }
        return out;
    }

    /// T13: restore a previously-captured adaptation snapshot for
    /// checkpoint/restart.
    void set_adaptation(const adaptation_info& ad) {
        auto& ns = cfg_.nuts_settings.nuts_settings;
        ns.step_size            = ad.step_size;
        ns.epsilon_bar_persist  = ad.epsilon_bar;
        ns.h_val_persist        = ad.h_val;
        ns.mu_val_persist       = ad.mu_val;
        ns.adapt_iter_persist   = ad.adapt_iter;
        if (!ad.precond_mat.is_empty()) {
            // The metric lives in the UNCONSTRAINED space, so a restored
            // precond_mat must be total_unc_dim_ x total_unc_dim_. A wrong-sized
            // matrix (cross-block restore, or a pre-dual-offset-fix snapshot of a
            // dim-changing block captured at total_dim_ != total_unc_dim_) would
            // be silently discarded by mcmclib::nuts (-> identity metric) while
            // dense_metric_adapted_ wrongly flips true. Reject it loudly instead
            // of degrading to identity behind the user's back. Mirrors the size
            // guard in get_adaptation().
            if (static_cast<std::size_t>(ad.precond_mat.n_elem) !=
                total_unc_dim_ * total_unc_dim_) {
                throw std::runtime_error(
                    "joint_nuts_block '" + cfg_.name +
                    "': set_adaptation precond_mat has " +
                    std::to_string(ad.precond_mat.n_elem) + " elements but the "
                    "metric must be total_unc_dim_ x total_unc_dim_ = " +
                    std::to_string(total_unc_dim_) + "x" +
                    std::to_string(total_unc_dim_));
            }
            ns.precond_mat = ad.precond_mat;
            dense_metric_adapted_ = (ad.metric_kind == "dense");
        }
    }

    // ---- Kernel-control freeze API (slot-level, interface.md Sec.1) -----
    //
    // Slot-level freeze IMPLEMENTED via "Approach B" from DESIGN_NOTES
    // Sec.10.a + subagent-A analysis 2026-07-20: non-invasive to mcmclib.
    //
    // Detailed balance argument (Neal HMC + Girolami RMHMC style):
    //   Let theta = (theta_f, theta_z) where _z is frozen at theta_z_snap.
    //   The step() adapter wraps user's log-density so mcmclib sees
    //     U_wrap(theta_f, theta_z) := U(theta_f, theta_z_snap)
    //   which is CONSTANT in theta_z; the gradient masking zeros grad_z.
    //   The mass matrix is block-diagonalized (see build_precond_) so
    //   M_{f,z} = M_{z,f} = 0 and M_{z,z} = I. Under this metric + potential:
    //     * theta_z's dynamics decouple from theta_f (grad_z = 0, no
    //       off-diagonal in M inverse).
    //     * theta_z drifts freely under identity mass; theta_f's marginal
    //       stationary distribution is exactly p(theta_f | theta_z = snap).
    //     * After mcmc::nuts returns, we restore theta_cat_[frozen] =
    //       theta_z_snap. This is a projection that doesn't perturb the
    //       theta_f marginal (proof: samples of theta_f from any joint
    //       of the form pi(theta_f) x q(theta_z) marginalize to pi(theta_f)).
    // Result: exact conditional posterior on the free slots. No approximation.
    //
    // Sub-name API + storage.
    std::vector<std::string> subnames() const override {
        std::vector<std::string> out;
        out.reserve(cfg_.sub_params.size());
        for (const auto& sp : cfg_.sub_params) out.push_back(sp.name);
        return out;
    }

    void freeze_sub(const std::string& sub) override {
        // Validate: sub must match one of the block's slot names.
        std::size_t hit = static_cast<std::size_t>(-1);
        for (std::size_t s = 0; s < cfg_.sub_params.size(); ++s) {
            if (cfg_.sub_params[s].name == sub) { hit = s; break; }
        }
        if (hit == static_cast<std::size_t>(-1)) {
            std::string valid;
            for (const auto& sp : cfg_.sub_params) {
                valid += (valid.empty() ? "" : ", ") + sp.name;
            }
            throw std::runtime_error(
                "joint_nuts_block '" + cfg_.name +
                "': freeze_sub unknown slot '" + sub +
                "'; valid slot names: " + valid);
        }
        frozen_slots_.insert(sub);
        rebuild_frozen_unc_idx_();
    }

    void unfreeze_sub(const std::string& sub) override {
        auto it = frozen_slots_.find(sub);
        if (it != frozen_slots_.end()) {
            frozen_slots_.erase(it);
            rebuild_frozen_unc_idx_();
        }
    }

    std::vector<std::string> frozen_subnames() const override {
        // Return in slot_specs_ order for stable get_frozen ordering.
        std::vector<std::string> out;
        for (const auto& sp : cfg_.sub_params) {
            if (frozen_slots_.count(sp.name)) out.push_back(sp.name);
        }
        return out;
    }

    /// Whole-block unfreeze also clears slot-frozen state for consistency
    /// (parallels rjmcmc_block's unfreeze override).
    void unfreeze() override {
        block_sampler::unfreeze();
        frozen_slots_.clear();
        frozen_unc_idx_.reset();
    }

    // ---- Kernel-tuning interface (readapt) ------------------------------

    /// joint_nuts_block supports kernel-tuning via readapt().
    bool supports_readapt() const noexcept override { return true; }

    /// Re-tune dual-averaging state by running n adaptation iterations
    /// at the current concatenated state. theta_cat_ is restored
    /// bitwise after the n iterations; dense metric (precond_mat) is
    /// preserved as-is (mcmclib's nuts does not auto-adapt it; T11's
    /// adapt_dense_metric_ is the separate route for that).
    ///
    /// See system_design.md §13 NUTS-family + validator.md §24.
    void readapt(std::size_t n,
                 bool reset,
                 std::mt19937_64& rng,
                 std::size_t max_tree_depth_override = 0) override {
        if (n == 0) return;

        // 1. Snapshot chain state.
        const arma::vec snap_theta_cat  = theta_cat_;
        const bool      snap_first_call = first_call_;

        auto& ns = cfg_.nuts_settings.nuts_settings;

        // 2. If reset, reinitialize dual-averaging persistent state.
        if (reset) {
            const double init_step =
                (cfg_.initial_step_size > 0.0) ? cfg_.initial_step_size : 1.0;
            ns.epsilon_bar_persist = init_step;
            ns.h_val_persist       = 0.0;
            ns.mu_val_persist      = std::log(10.0 * init_step);
            ns.adapt_iter_persist  = 0;
        }

        // 3. Save settings we'll override.
        const std::size_t saved_n_burnin = ns.n_burnin_draws;
        const std::size_t saved_n_keep   = ns.n_keep_draws;
        const std::size_t saved_n_adapt  = ns.n_adapt_draws;
        const std::uint64_t saved_seed   = cfg_.nuts_settings.rng_seed_value;
        const std::size_t saved_mtd      = ns.max_tree_depth;

        ns.n_burnin_draws = n;
        ns.n_keep_draws   = 1;
        ns.n_adapt_draws  = n;
        cfg_.nuts_settings.rng_seed_value =
            static_cast<std::uint64_t>(rng());
        if (max_tree_depth_override > 0)            // per-call tree-depth cap
            ns.max_tree_depth = max_tree_depth_override;

        auto adapter = [this](
                            const mcmc::ColVec_t& theta_in,
                            mcmc::ColVec_t* grad_out,
                            void* /*unused*/) -> mcmc::fp_t {
            return static_cast<mcmc::fp_t>(eval_unc_(theta_in, grad_out));
        };

        // 4. Run n adaptation iterations. theta_cat_ is mutated during
        //    burnin; DA persistent state updated.
        mcmc::Mat_t draws_out;
        const bool ok = mcmc::nuts(
            theta_cat_, adapter, draws_out,
            /*target_data=*/ nullptr,
            cfg_.nuts_settings);

        // 5. Restore settings.
        ns.n_burnin_draws = saved_n_burnin;
        ns.n_keep_draws   = saved_n_keep;
        ns.n_adapt_draws  = saved_n_adapt;
        cfg_.nuts_settings.rng_seed_value = saved_seed;
        ns.max_tree_depth = saved_mtd;

        if (!ok) {
            theta_cat_  = snap_theta_cat;
            first_call_ = snap_first_call;
            rebuild_named_outputs_();
            throw std::runtime_error(
                "joint_nuts_block '" + cfg_.name + "': readapt failed");
        }

        // 6. Restore chain state. Kernel state (DA accumulators)
        //    preserved from the n adaptation iterations.
        theta_cat_  = snap_theta_cat;
        first_call_ = snap_first_call;
        rebuild_named_outputs_();
    }

    /// Expose the offset of a sub-parameter in theta_cat. Used by unit
    /// tests and by Rcpp wrappers that want to slice directly.
    std::size_t sub_param_offset(const std::string& name) const {
        for (std::size_t i = 0; i < cfg_.sub_params.size(); ++i) {
            if (cfg_.sub_params[i].name == name) return offsets_[i];
        }
        throw std::out_of_range(
            "joint_nuts_block '" + cfg_.name +
            "': no sub-parameter named '" + name + "'");
    }

    /// Exposed for TESTING / DIAGNOSTICS only (NOT part of the sampling loop):
    /// evaluate the UNCONSTRAINED-scale log-density + gradient that the NUTS
    /// adapters use — i.e. natural-scale oracle + per-slice log|Jacobian| +
    /// chain-ruled gradient. Lets a test finite-difference the per-slice
    /// constraint wiring directly. theta_unc has length total_unc_dim_ (the
    /// UNCONSTRAINED width; == total_dim_ only for dim-preserving blocks);
    /// grad_unc is filled to total_unc_dim_.
    double eval_log_density_unc(const mcmc::ColVec_t& theta_unc,
                               mcmc::ColVec_t* grad_unc) const {
        return eval_unc_(theta_unc, grad_unc);
    }

    std::size_t sub_param_dim(const std::string& name) const {
        for (std::size_t i = 0; i < cfg_.sub_params.size(); ++i) {
            if (cfg_.sub_params[i].name == name) {
                return cfg_.sub_params[i].dim;
            }
        }
        throw std::out_of_range(
            "joint_nuts_block '" + cfg_.name +
            "': no sub-parameter named '" + name + "'");
    }

private:
    // ---- per-slice constraint transforms (identity when all_real_) -------

    // UNCONSTRAINED slice width for a kind given its NATURAL width. Equal for
    // dimension-preserving kinds; SIMPLEX uses K-1 (stick-breaking).
    static std::size_t unc_dim_for_(joint_constraint k, std::size_t nat_dim) {
        auto Ksqrt = [&]() -> std::size_t {
            const std::size_t K = static_cast<std::size_t>(
                std::llround(std::sqrt(static_cast<double>(nat_dim))));
            if (K * K != nat_dim) throw std::invalid_argument(
                "joint_nuts_block: matrix constraint sub_param.dim must be K*K "
                "(a flattened K x K matrix)");
            return K;
        };
        switch (k) {
            case joint_constraint::SIMPLEX:
            case joint_constraint::SUM_TO_ZERO:
                if (nat_dim < 2) throw std::invalid_argument(
                    "joint_nuts_block: SIMPLEX/SUM_TO_ZERO sub_param.dim (K) "
                    "must be >= 2");
                return nat_dim - 1;                       // K-1
            case joint_constraint::CHOLESKY_CORR:
            case joint_constraint::CORR_MATRIX: {
                const std::size_t K = Ksqrt(); return K * (K - 1) / 2;
            }
            case joint_constraint::CHOLESKY_FACTOR_COV:
            case joint_constraint::COV_MATRIX: {
                const std::size_t K = Ksqrt(); return K * (K + 1) / 2;
            }
            default:
                return nat_dim;  // dimension-preserving
        }
    }

    /// NATURAL -> UNCONSTRAINED. Identity for REAL; log for POSITIVE.
    arma::vec nat_to_unc_(const arma::vec& nat) const {
        if (all_real_) return nat;  // fast-path: bit-identical to V0
        arma::vec unc(total_unc_dim_);
        for (std::size_t s = 0; s < cfg_.sub_params.size(); ++s) {
            const auto& sp = cfg_.sub_params[s];
            const arma::vec slice =
                nat.subvec(offsets_[s], offsets_[s] + sp.dim - 1);
            arma::vec out;
            switch (sp.constraint) {
                case joint_constraint::POSITIVE:
                    out = constraints::positive::unconstrain(slice); break;
                case joint_constraint::LOWER_BOUNDED:
                    out = constraints::lower_bounded::unconstrain(slice, sp.lower); break;
                case joint_constraint::UPPER_BOUNDED:
                    out = constraints::upper_bounded::unconstrain(slice, sp.upper); break;
                case joint_constraint::INTERVAL:
                    out = constraints::interval::unconstrain(slice, sp.lower, sp.upper); break;
                case joint_constraint::ORDERED:
                    out = constraints::ordered::unconstrain(slice); break;
                case joint_constraint::POSITIVE_ORDERED:
                    out = constraints::positive_ordered::unconstrain(slice); break;
                case joint_constraint::UNIT_VECTOR:
                    out = constraints::unit_vector::unconstrain(slice); break;
                case joint_constraint::OFFSET_MULTIPLIER:
                    out = constraints::offset_multiplier::unconstrain(slice, sp.lower, sp.upper); break;
                case joint_constraint::SIMPLEX:
                    out = constraints::simplex::unconstrain(slice); break;  // K -> K-1
                case joint_constraint::SUM_TO_ZERO:
                    out = constraints::sum_to_zero::unconstrain(slice); break;
                case joint_constraint::CHOLESKY_CORR:
                    out = constraints::cholesky_corr::unconstrain(slice); break;
                case joint_constraint::CHOLESKY_FACTOR_COV:
                    out = constraints::cholesky_factor_cov::unconstrain(slice); break;
                case joint_constraint::CORR_MATRIX:
                    out = constraints::corr_matrix::unconstrain(slice); break;
                case joint_constraint::COV_MATRIX:
                    out = constraints::cov_matrix::unconstrain(slice); break;
                case joint_constraint::REAL:
                default:
                    out = slice; break;
            }
            unc.subvec(unc_offsets_[s], unc_offsets_[s] + unc_dims_[s] - 1) = out;
        }
        return unc;
    }

    /// UNCONSTRAINED -> NATURAL. Identity for REAL; exp for POSITIVE.
    arma::vec unc_to_nat_(const arma::vec& unc) const {
        if (all_real_) return unc;  // fast-path: bit-identical to V0
        arma::vec nat(total_dim_);
        for (std::size_t s = 0; s < cfg_.sub_params.size(); ++s) {
            const auto& sp = cfg_.sub_params[s];
            const arma::vec slice =
                unc.subvec(unc_offsets_[s], unc_offsets_[s] + unc_dims_[s] - 1);
            arma::vec out;
            switch (sp.constraint) {
                case joint_constraint::POSITIVE:
                    out = constraints::positive::constrain(slice); break;
                case joint_constraint::LOWER_BOUNDED:
                    out = constraints::lower_bounded::constrain(slice, sp.lower); break;
                case joint_constraint::UPPER_BOUNDED:
                    out = constraints::upper_bounded::constrain(slice, sp.upper); break;
                case joint_constraint::INTERVAL:
                    out = constraints::interval::constrain(slice, sp.lower, sp.upper); break;
                case joint_constraint::ORDERED:
                    out = constraints::ordered::constrain(slice); break;
                case joint_constraint::POSITIVE_ORDERED:
                    out = constraints::positive_ordered::constrain(slice); break;
                case joint_constraint::UNIT_VECTOR:
                    out = constraints::unit_vector::constrain(slice); break;
                case joint_constraint::OFFSET_MULTIPLIER:
                    out = constraints::offset_multiplier::constrain(slice, sp.lower, sp.upper); break;
                case joint_constraint::SIMPLEX:
                    out = constraints::simplex::constrain(slice); break;  // K-1 -> K
                case joint_constraint::SUM_TO_ZERO:
                    out = constraints::sum_to_zero::constrain(slice); break;
                case joint_constraint::CHOLESKY_CORR:
                    out = constraints::cholesky_corr::constrain(slice); break;
                case joint_constraint::CHOLESKY_FACTOR_COV:
                    out = constraints::cholesky_factor_cov::constrain(slice); break;
                case joint_constraint::CORR_MATRIX:
                    out = constraints::corr_matrix::constrain(slice); break;
                case joint_constraint::COV_MATRIX:
                    out = constraints::cov_matrix::constrain(slice); break;
                case joint_constraint::REAL:
                default:
                    out = slice; break;
            }
            nat.subvec(offsets_[s], offsets_[s] + sp.dim - 1) = out;
        }
        return nat;
    }

    /// Throw if any constrained slice of natural-scale `nat` is outside its
    /// support (POSITIVE > 0; LOWER_BOUNDED > lower; UPPER_BOUNDED < upper;
    /// INTERVAL in (lower, upper); ORDERED strictly increasing). Gives a clear
    /// error before nat_to_unc_ would otherwise throw a terser domain_error.
    void validate_support_(const arma::vec& nat, const char* where) const {
        auto fail = [&](const joint_nuts_sub_param& sp, std::size_t k,
                        double v, const char* req) {
            throw std::invalid_argument(
                std::string("joint_nuts_block '") + cfg_.name +
                "': sub_param '" + sp.name + "' value at index " +
                std::to_string(k) + " is " + std::to_string(v) + "; " + req +
                " (" + where + ")");
        };
        for (std::size_t s = 0; s < cfg_.sub_params.size(); ++s) {
            const auto& sp = cfg_.sub_params[s];
            const std::size_t off = offsets_[s];
            switch (sp.constraint) {
                case joint_constraint::POSITIVE:
                    for (std::size_t k = 0; k < sp.dim; ++k) {
                        const double v = nat[off + k];
                        if (!(v > 0.0)) fail(sp, k, v, "must be > 0");
                    }
                    break;
                case joint_constraint::LOWER_BOUNDED:
                    for (std::size_t k = 0; k < sp.dim; ++k) {
                        const double v = nat[off + k];
                        if (!(v > sp.lower)) fail(sp, k, v, "must be > lower bound");
                    }
                    break;
                case joint_constraint::UPPER_BOUNDED:
                    for (std::size_t k = 0; k < sp.dim; ++k) {
                        const double v = nat[off + k];
                        if (!(v < sp.upper)) fail(sp, k, v, "must be < upper bound");
                    }
                    break;
                case joint_constraint::INTERVAL:
                    for (std::size_t k = 0; k < sp.dim; ++k) {
                        const double v = nat[off + k];
                        if (!(v > sp.lower && v < sp.upper))
                            fail(sp, k, v, "must lie strictly inside (lower, upper)");
                    }
                    break;
                case joint_constraint::ORDERED:
                    for (std::size_t k = 1; k < sp.dim; ++k) {
                        const double v = nat[off + k];
                        if (!(v > nat[off + k - 1]))
                            fail(sp, k, v, "must be strictly increasing");
                    }
                    break;
                case joint_constraint::POSITIVE_ORDERED:
                    if (sp.dim > 0 && !(nat[off] > 0.0))
                        fail(sp, 0, nat[off], "first element must be > 0");
                    for (std::size_t k = 1; k < sp.dim; ++k) {
                        const double v = nat[off + k];
                        if (!(v > nat[off + k - 1]))
                            fail(sp, k, v, "must be strictly increasing");
                    }
                    break;
                case joint_constraint::UNIT_VECTOR: {
                    double nrm2 = 0.0;
                    for (std::size_t k = 0; k < sp.dim; ++k)
                        nrm2 += nat[off + k] * nat[off + k];
                    const double nrm = std::sqrt(nrm2);
                    if (!(std::abs(nrm - 1.0) < 1e-6))
                        fail(sp, 0, nrm, "slice must be a unit vector (||x|| = 1)");
                    break;
                }
                case joint_constraint::OFFSET_MULTIPLIER:
                    break;  // full real support; no validation
                case joint_constraint::SIMPLEX: {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < sp.dim; ++k) {
                        const double v = nat[off + k];
                        if (!(v > 0.0)) fail(sp, k, v, "simplex entries must be > 0");
                        sum += v;
                    }
                    if (std::abs(sum - 1.0) > 1e-6)
                        fail(sp, 0, sum, "simplex entries must sum to 1");
                    break;
                }
                case joint_constraint::SUM_TO_ZERO: {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < sp.dim; ++k) sum += nat[off + k];
                    if (std::abs(sum) > 1e-6 * (1.0 + static_cast<double>(sp.dim)))
                        fail(sp, 0, sum, "sum_to_zero entries must sum to 0");
                    break;
                }
                // Matrix kinds: validity (PD / unit-diagonal Cholesky factor) is
                // enforced by the constraint's unconstrain() inside nat_to_unc_;
                // no separate per-element check here.
                case joint_constraint::CHOLESKY_CORR:
                case joint_constraint::CHOLESKY_FACTOR_COV:
                case joint_constraint::CORR_MATRIX:
                case joint_constraint::COV_MATRIX:
                    break;
                case joint_constraint::REAL:
                default:
                    break;
            }
        }
    }

    /// Evaluate log p + grad on the UNCONSTRAINED scale — the single oracle
    /// all four mcmc::nuts adapter sites call.
    ///   all_real_ : natural == unconstrained → call the user oracle directly
    ///               (BIT-IDENTICAL to the V0 adapter).
    ///   otherwise : transform unc → nat, call the natural-scale oracle ONCE,
    ///               then add each slice's log|Jacobian| and chain-rule its
    ///               gradient via constraints::<kind>::wrap (all Jacobian math
    ///               lives in constraints.hpp; system_design §10.1).
    double eval_unc_(const mcmc::ColVec_t& theta_unc,
                     mcmc::ColVec_t* grad_unc) const {
        if (theta_unc.n_elem != total_unc_dim_) {
            throw std::runtime_error(
                "joint_nuts_block '" + cfg_.name +
                "': log_density input dim mismatch (got " +
                std::to_string(theta_unc.n_elem) + ", expected " +
                std::to_string(total_unc_dim_) + ")");
        }

        if (all_real_) {
            const double lp =
                cfg_.log_density_grad(theta_unc, context_, grad_unc);
            // A non-finite lp is a valid NUTS reject (e.g. an oracle that
            // `return -inf`s before sizing the gradient); the grad is unused.
            if (!std::isfinite(lp)) return lp;
            if (grad_unc && grad_unc->n_elem != total_dim_) {
                throw std::runtime_error(
                    "joint_nuts_block '" + cfg_.name +
                    "': log_density wrote gradient of length " +
                    std::to_string(grad_unc->n_elem) + " (expected " +
                    std::to_string(total_dim_) + ")");
            }
            return lp;
        }

        // Mixed path: natural-scale oracle called once on the whole vector.
        const arma::vec theta_nat = unc_to_nat_(theta_unc);
        arma::vec grad_nat;
        const double lp_nat = cfg_.log_density_grad(
            theta_nat, context_, grad_unc ? &grad_nat : nullptr);
        // A non-finite lp is a valid NUTS reject; the (possibly unsized)
        // gradient is irrelevant. This matches the blessed example pattern of
        // `return -inf` before sizing grad (LinearRegJointMixed.cpp), which
        // previously tripped the length check below.
        if (!std::isfinite(lp_nat)) return lp_nat;
        if (grad_unc && grad_nat.n_elem != total_dim_) {
            throw std::runtime_error(
                "joint_nuts_block '" + cfg_.name +
                "': natural-scale gradient wrong length (got " +
                std::to_string(grad_nat.n_elem) + ", expected " +
                std::to_string(total_dim_) + ")");
        }

        double total_lp = lp_nat;
        if (grad_unc) grad_unc->set_size(total_unc_dim_);
        for (std::size_t s = 0; s < cfg_.sub_params.size(); ++s) {
            const auto& sp = cfg_.sub_params[s];
            const std::size_t uoff = unc_offsets_[s];  // unconstrained slice
            const std::size_t udim = unc_dims_[s];
            const std::size_t noff = offsets_[s];       // natural slice
            const arma::vec u_slice =
                theta_unc.subvec(uoff, uoff + udim - 1);
            // inner injects the precomputed natural-scale slice gradient and
            // contributes 0 to lp (lp_nat already counted); wrap returns this
            // slice's log|J| and fills g_slice with the chain-ruled gradient
            // (UNCONSTRAINED width udim; may differ from the natural width).
            arma::vec gn_slice;
            if (grad_unc) gn_slice = grad_nat.subvec(noff, noff + sp.dim - 1);
            auto inner = [&](const arma::vec& /*nat*/,
                             arma::vec* gout) -> double {
                if (gout) *gout = gn_slice;
                return 0.0;
            };
            arma::vec g_slice;
            double log_jac_s = 0.0;
            switch (sp.constraint) {
                case joint_constraint::POSITIVE:
                    log_jac_s = constraints::positive::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, inner);
                    break;
                case joint_constraint::LOWER_BOUNDED:
                    log_jac_s = constraints::lower_bounded::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, sp.lower, inner);
                    break;
                case joint_constraint::UPPER_BOUNDED:
                    log_jac_s = constraints::upper_bounded::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, sp.upper, inner);
                    break;
                case joint_constraint::INTERVAL:
                    log_jac_s = constraints::interval::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr,
                        sp.lower, sp.upper, inner);
                    break;
                case joint_constraint::ORDERED:
                    log_jac_s = constraints::ordered::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, inner);
                    break;
                case joint_constraint::POSITIVE_ORDERED:
                    log_jac_s = constraints::positive_ordered::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, inner);
                    break;
                case joint_constraint::UNIT_VECTOR:
                    log_jac_s = constraints::unit_vector::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, inner);
                    break;
                case joint_constraint::OFFSET_MULTIPLIER:
                    log_jac_s = constraints::offset_multiplier::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr,
                        sp.lower, sp.upper, inner);
                    break;
                case joint_constraint::SIMPLEX:
                    log_jac_s = constraints::simplex::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, inner);
                    break;
                case joint_constraint::SUM_TO_ZERO:
                    log_jac_s = constraints::sum_to_zero::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, inner);
                    break;
                case joint_constraint::CHOLESKY_CORR:
                    log_jac_s = constraints::cholesky_corr::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, inner);
                    break;
                case joint_constraint::CHOLESKY_FACTOR_COV:
                    log_jac_s = constraints::cholesky_factor_cov::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, inner);
                    break;
                case joint_constraint::CORR_MATRIX:
                    log_jac_s = constraints::corr_matrix::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, inner);
                    break;
                case joint_constraint::COV_MATRIX:
                    log_jac_s = constraints::cov_matrix::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, inner);
                    break;
                case joint_constraint::REAL:
                default:
                    log_jac_s = constraints::real::wrap(
                        u_slice, grad_unc ? &g_slice : nullptr, inner);
                    break;
            }
            total_lp += log_jac_s;
            if (grad_unc)
                grad_unc->subvec(uoff, uoff + udim - 1) = g_slice;
        }

        if (!std::isfinite(total_lp)) {
            return -std::numeric_limits<double>::infinity();
        }
        return total_lp;
    }

    void rebuild_named_outputs_() {
        // theta_cat_ is unconstrained; expose the NATURAL-scale view.
        // unc_to_nat_ is identity when all_real_ (=> bit-identical to V0).
        theta_cat_nat_ = unc_to_nat_(theta_cat_);
        named_outputs_cache_.clear();
        for (std::size_t s = 0; s < cfg_.sub_params.size(); ++s) {
            const auto& sp = cfg_.sub_params[s];
            named_outputs_cache_[sp.name] = theta_cat_nat_.subvec(
                offsets_[s], offsets_[s] + sp.dim - 1);
        }
    }

    // Build the mass matrix (precond_mat) from a regularized covariance.
    //   use_diagonal_metric: M = diag(1 / clamp(var, eps, max_mass_variance))
    //     — Stan/NumPyro-style per-axis rescaling; the variance cap lets the
    //     metric only TIGHTEN the effective step (eps*sqrt(var)), never amplify
    //     it (the funnel guard). Robust where full dense is unstable.
    //   else (dense): M = inv_sympd(Sigma_reg), the full precision.
    arma::mat build_precond_(const arma::mat& Sigma_reg) const {
        if (cfg_.use_diagonal_metric) {
            const double cap = cfg_.max_mass_variance > 0.0
                ? cfg_.max_mass_variance : arma::datum::inf;
            arma::vec v = arma::clamp(Sigma_reg.diag(), 1e-12, cap);
            return arma::diagmat(1.0 / v);
        }
        return arma::inv_sympd(Sigma_reg);
    }

    // T11: pilot phase + Welford sample covariance + set precond.
    // Runs identity-metric NUTS for (pilot_iters burn-in + adapt_iters
    // kept), computes sample covariance of the kept draws, and installs
    // it via set_precond_matrix. Leaves theta_cat_ at the last draw.
    // Run the 3-phase windowed warmup WITH the metric-escape guard (2026-06-19).
    // The 3-phase schedule, like the single-pilot path, can drive the warmup out
    // of the typical set on a STIFF model (e.g. ARMA's non-invertible |theta|>1
    // ridge) from a dispersed init — and unlike the single-pilot main-warmup
    // path it had NO escape protection. Here we snapshot the pre-warmup state +
    // log-density, run 3-phase, and if the post-warmup log-density is non-finite
    // or catastrophically worse, reset to the (good) pre-warmup state and fall
    // back to the IDENTITY metric. Returns true if 3-phase succeeded (caller
    // sets first_call_ = false); false if it fell back (caller leaves
    // first_call_ = true so the main step() path re-warms under identity).
    bool three_phase_guarded_(std::mt19937_64& rng) {
        const arma::vec pre    = theta_cat_;
        const double    pre_lp = eval_unc_(pre, nullptr);
        adapt_three_phase_warmup_(rng);
        const double post_lp = eval_unc_(theta_cat_, nullptr);
        auto& ns = cfg_.nuts_settings.nuts_settings;
        if (!std::isfinite(post_lp) ||
            (std::isfinite(pre_lp) && post_lp < pre_lp - 1.0e3)) {
            theta_cat_               = pre;        // good restart
            rebuild_named_outputs_();
            cfg_.use_dense_metric    = false;
            cfg_.use_diagonal_metric = false;      // => identity metric
            auto_selected_dense_     = -1;
            auto_selected_cond_      = 0.0;
            ns.precond_mat           = mcmc::Mat_t();
            ns.epsilon_bar_persist   = 0.0;
            ns.h_val_persist         = 0.0;
            ns.mu_val_persist        = 0.0;
            ns.adapt_iter_persist    = 0;
            return false;
        }
        return true;
    }

    void adapt_dense_metric_(std::mt19937_64& rng) {
        cfg_.nuts_settings.rng_seed_value =
            static_cast<std::uint64_t>(rng());
        auto& ns = cfg_.nuts_settings.nuts_settings;

        // Snapshot the user init + its log-density BEFORE the pilot. Used by the
        // degenerate-pilot guard below: if the pilot wanders into a degenerate /
        // explosive region (near-zero spread => the learned metric would
        // catastrophically overshoot), we reset the chain to this init and fall
        // back to the unconditionally-stable identity metric.
        const arma::vec init_theta_unc = theta_cat_;
        const double    init_lp        = eval_unc_(init_theta_unc, nullptr);

        // Save original per-call budget; restore after adaptation.
        const std::size_t saved_burnin = ns.n_burnin_draws;
        const std::size_t saved_keep   = ns.n_keep_draws;
        const std::size_t saved_adapt  = ns.n_adapt_draws;

        ns.n_burnin_draws = cfg_.dense_metric_pilot_iters;
        ns.n_keep_draws   = cfg_.dense_metric_adapt_iters;
        ns.n_adapt_draws  = cfg_.dense_metric_pilot_iters;

        // Build adapter (same eval_unc_ as main step()).
        auto adapter = [this](
                            const mcmc::ColVec_t& theta_in,
                            mcmc::ColVec_t* grad_out,
                            void* /*unused*/) -> mcmc::fp_t {
            return static_cast<mcmc::fp_t>(eval_unc_(theta_in, grad_out));
        };

        mcmc::Mat_t draws_out;
        const bool ok = mcmc::nuts(
            theta_cat_,
            adapter,
            draws_out,
            /*target_data=*/ nullptr,
            cfg_.nuts_settings);
        if (!ok || draws_out.n_rows == 0) {
            throw std::runtime_error(
                "joint_nuts_block '" + cfg_.name +
                "': dense-metric pilot phase failed (nuts returned no draws)");
        }

        // Welford one-pass mean + covariance on kept draws.
        // draws_out is (n_draws × total_unc_dim_) — the sampler integrates in
        // the UNCONSTRAINED space, so the metric is sized to total_unc_dim_
        // (== total_dim_ only for dim-preserving blocks).
        const std::size_t n = draws_out.n_rows;
        if (n < 10) {
            // Too few kept samples to estimate covariance; fall back to
            // identity metric by leaving precond at its default. Should
            // not happen with dense_metric_adapt_iters >= 100.
            theta_cat_ = draws_out.row(n - 1).t();
            ns.n_burnin_draws = saved_burnin;
            ns.n_keep_draws   = saved_keep;
            ns.n_adapt_draws  = saved_adapt;
            return;
        }

        // ---- Degenerate-pilot guard (2026-06-19) ---------------------------
        // From a dispersed init, a short pilot can wander into a degenerate or
        // explosive region (e.g. the non-invertible |theta|>1 ridge of an ARMA
        // model) and FREEZE there with near-zero spread. Its sample covariance
        // is then ~0 in every direction, so the learned precision (Sigma^{-1})
        // is enormous and the leapfrog overshoots catastrophically — under a
        // DENSE metric the off-diagonal coupling makes this fatal (the chain is
        // thrown far outside the typical set and the step collapses to the
        // floor, freezing the chain). We detect this from the pilot's FINAL
        // log-density: a reliable pilot ends at a log-density no worse (and
        // usually better) than the init; a pilot that escaped ends orders of
        // magnitude worse, or non-finite. On detection, reset the chain to the
        // user init and fall back to the IDENTITY metric (unconditionally
        // stable; the subsequent main warmup re-tunes the step from the init).
        // Only fires on escape => zero effect on well-behaved pilots.
        {
            const arma::vec pilot_end = draws_out.row(n - 1).t();
            const double    end_lp    = eval_unc_(pilot_end, nullptr);
            const bool pilot_escaped =
                !std::isfinite(end_lp) ||
                (std::isfinite(init_lp) && end_lp < init_lp - 1.0e3);
            if (pilot_escaped) {
                theta_cat_               = init_theta_unc;   // back to safe init
                cfg_.use_dense_metric    = false;
                cfg_.use_diagonal_metric = false;            // => identity metric
                auto_selected_dense_     = -1;               // record: aborted
                auto_selected_cond_      = 0.0;
                // Reset DA so the main warmup re-tunes the step from the init.
                ns.epsilon_bar_persist = 0.0;
                ns.h_val_persist       = 0.0;
                ns.mu_val_persist      = 0.0;
                ns.adapt_iter_persist  = 0;
                ns.n_burnin_draws = saved_burnin;
                ns.n_keep_draws   = saved_keep;
                ns.n_adapt_draws  = saved_adapt;
                return;
            }
        }

        // Sample covariance (denominator n - 1).
        arma::mat D(draws_out.memptr(), draws_out.n_rows,
                    draws_out.n_cols, /*copy=*/false);
        arma::vec mu = arma::mean(D, 0).t();
        arma::mat Dc = D.each_row() - mu.t();
        arma::mat Sigma = (Dc.t() * Dc) / static_cast<double>(n - 1);

        // Regularize covariance for numerical stability before inverting.
        // Stan uses a similar shrinkage toward the identity:
        //   Sigma_shrunk = (n / (n + 5)) * Sigma + 1e-3 * ((5 / (n + 5))) * I
        // which both bounds the condition number and regularizes rare
        // near-singular cases early in warmup.
        const double shrink = static_cast<double>(n) / (n + 5.0);
        const double ridge  = 1e-3 * (5.0 / (n + 5.0));
        arma::mat Sigma_reg = shrink * Sigma
            + ridge * arma::eye<arma::mat>(total_unc_dim_, total_unc_dim_);

        // AUTO-SELECT diagonal vs dense (2026-06-19). Decide from the PILOT
        // posterior: a diagonal metric only rescales each axis, so it is
        // optimal exactly when the posterior is (near-)axis-aligned. We measure
        // axis-alignment by the condition number of the CORRELATION matrix R
        // (not the covariance: R strips out the per-axis scales a diagonal
        // metric would absorb anyway, isolating the off-diagonal coupling a
        // diagonal metric CANNOT capture). cond(R)=1 <=> uncorrelated => diagonal
        // is exact; large cond(R) <=> strong coupling => dense pays off. Dense is
        // fragile and O(d^2) in the leapfrog, so only take it in modest dimension.
        if (cfg_.auto_select_metric) {
            arma::vec sd = arma::sqrt(arma::clamp(Sigma.diag(), 1e-12,
                                                  arma::datum::inf));
            arma::mat R  = Sigma / (sd * sd.t());        // correlation matrix
            R = 0.5 * (R + R.t());                        // symmetrize (FP guard)
            double cond_R = 1.0;
            arma::vec ev;
            if (arma::eig_sym(ev, R) && ev.min() > 0.0) {
                cond_R = ev.max() / ev.min();
            } else {
                // eig failed or non-PD correlation => treat as ill-conditioned.
                cond_R = arma::datum::inf;
            }
            // Marchenko-Pastur NOISE FLOOR (2026-06-20). Even a perfectly
            // UNCORRELATED target shows an inflated sample cond(R) from finite,
            // autocorrelated pilot draws: for d dims and ~n effective samples the
            // sample eigenvalues of a true-identity correlation spread to
            // [(1-sqrt(q))^2, (1+sqrt(q))^2], q=d/n, so the *noise* cond is
            // ((1+sqrt q)/(1-sqrt q))^2. Without this floor, high-dim uncorrelated
            // targets cross the fixed cond>4 threshold and get WRONGLY picked
            // dense -> catastrophic O(d^2) cost (broad corpus: diagscaled_d64
            // cond~1 -> auto-dense was ~387x slower than diagonal). Require cond_R
            // to exceed the noise floor (x margin) before picking dense. n_eff is
            // halved as a conservative allowance for pilot autocorrelation. Low
            // dim is unaffected (q tiny => floor ~ base threshold).
            const double n_eff = std::max(1.0, 0.5 * static_cast<double>(n));
            const double q = static_cast<double>(total_unc_dim_) / n_eff;
            double noise_cond = arma::datum::inf;     // q>=1: cannot estimate
            if (q < 1.0) {
                const double sq = std::sqrt(q);
                noise_cond = std::pow((1.0 + sq) / (1.0 - sq), 2.0);
            }
            const double eff_threshold =
                std::max(cfg_.auto_dense_cond_threshold, 1.2 * noise_cond);
            const bool pick_dense =
                (cond_R > eff_threshold) &&
                (total_unc_dim_ <= cfg_.auto_dense_max_dim);
            cfg_.use_dense_metric    = pick_dense;
            cfg_.use_diagonal_metric = !pick_dense;
            auto_selected_dense_     = pick_dense ? 1 : 0;
            auto_selected_cond_      = cond_R;
        }

        // mcmclib's `precond_mat` is the MASS MATRIX M (confirmed by
        // reading mcmc/nuts.hpp: p ~ N(0, precond_mat), q̇ = precond_mat^{-1} p).
        // For isotropic NUTS dynamics on a target with posterior covariance
        // Σ, the optimal mass matrix is M = Σ^{-1}. Pass the PRECISION
        // matrix, not the covariance.
        cfg_.nuts_settings.nuts_settings.precond_mat = build_precond_(Sigma_reg);

        // Reset adapt_iter so dual-averaging step-size retunes against new
        // metric (the step size adapted under identity metric is wrong for
        // the new dense metric).
        ns.epsilon_bar_persist = 0.0;
        ns.h_val_persist       = 0.0;
        ns.mu_val_persist      = 0.0;
        ns.adapt_iter_persist  = 0;

        theta_cat_ = draws_out.row(n - 1).t();

        // Restore per-call budget.
        ns.n_burnin_draws = saved_burnin;
        ns.n_keep_draws   = saved_keep;
        ns.n_adapt_draws  = saved_adapt;
    }

    // v1.2: Stan-style 3-phase windowed warmup. See joint_nuts_block_config
    // comment block (use_three_phase_warmup) for design rationale and
    // schedule. Triggered from step() when use_three_phase_warmup == true.
    //
    // Total iterations executed inside this method:
    //   Phase I:   cfg_.tp_phase1_iters                (default 75)
    //   Phase II:  sum(cfg_.tp_phase2_windows)         (default 875)
    //   Phase III: cfg_.tp_phase3_iters                (default 50)
    //   Default total: 1000 iter (matches Stan default).
    //
    // After completion: precond_mat holds the final regularized precision
    // (inverse covariance estimated across the last + earlier windows),
    // dual-averaging step-size state is finalized for sampling, and
    // theta_cat_ is at the last Phase III draw.
    void adapt_three_phase_warmup_(std::mt19937_64& rng) {
        auto& ns = cfg_.nuts_settings.nuts_settings;

        // Save original per-call budget; restore after warmup.
        const std::size_t saved_burnin = ns.n_burnin_draws;
        const std::size_t saved_keep   = ns.n_keep_draws;
        const std::size_t saved_adapt  = ns.n_adapt_draws;

        // Shared adapter — same eval_unc_ as step() and adapt_dense_metric_.
        auto adapter = [this](
                            const mcmc::ColVec_t& theta_in,
                            mcmc::ColVec_t* grad_out,
                            void* /*unused*/) -> mcmc::fp_t {
            return static_cast<mcmc::fp_t>(eval_unc_(theta_in, grad_out));
        };

        // ----- Phase I: step-size adaptation only, identity mass matrix.
        // Reset DA persistent state to start fresh; no precond_mat (identity).
        {
            cfg_.nuts_settings.rng_seed_value =
                static_cast<std::uint64_t>(rng());
            ns.epsilon_bar_persist = 0.0;
            ns.h_val_persist       = 0.0;
            ns.mu_val_persist      = 0.0;
            ns.adapt_iter_persist  = 0;

            // Ensure identity mass matrix for Phase I. Empty precond_mat
            // → mcmclib uses identity. We don't touch a possibly user-
            // supplied precond_mat here; we override to identity.
            const mcmc::Mat_t saved_precond = ns.precond_mat;
            ns.precond_mat = mcmc::Mat_t();  // empty → identity in mcmclib

            const std::size_t n1 = std::max<std::size_t>(cfg_.tp_phase1_iters, 1);
            ns.n_burnin_draws = n1;
            ns.n_keep_draws   = 1;
            ns.n_adapt_draws  = n1;

            mcmc::Mat_t out1;
            const bool ok1 = mcmc::nuts(
                theta_cat_, adapter, out1, nullptr, cfg_.nuts_settings);
            if (!ok1 || out1.n_rows == 0) {
                // Restore and abort: leave caller in single-phase fallback
                ns.precond_mat = saved_precond;
                ns.n_burnin_draws = saved_burnin;
                ns.n_keep_draws   = saved_keep;
                ns.n_adapt_draws  = saved_adapt;
                throw std::runtime_error(
                    "joint_nuts_block '" + cfg_.name +
                    "': 3-phase warmup Phase I (identity-NUTS) failed");
            }
            theta_cat_ = out1.row(out1.n_rows - 1).t();
            // Phase I complete; precond_mat stays empty (identity) until
            // first Phase II window completes and installs a dense one.
        }

        // ----- Phase II: windowed mass matrix adaptation.
        // Between Phase I and Phase II window 1: reset DA persists once
        // (big metric change from identity to first dense). Within Phase
        // II, DA persists carry across windows (Stan behavior).
        ns.epsilon_bar_persist = 0.0;
        ns.h_val_persist       = 0.0;
        ns.mu_val_persist      = 0.0;
        ns.adapt_iter_persist  = 0;

        // Metric is sized in the UNCONSTRAINED space the sampler integrates
        // (== total_dim_ only for dim-preserving blocks; differs for SIMPLEX,
        // covariance/correlation matrices, sum-to-zero, etc.).
        const std::size_t metric_dim_local = total_unc_dim_;
        for (std::size_t w = 0; w < cfg_.tp_phase2_windows.size(); ++w) {
            const std::size_t win_iter = cfg_.tp_phase2_windows[w];
            if (win_iter == 0) continue;

            cfg_.nuts_settings.rng_seed_value =
                static_cast<std::uint64_t>(rng());
            ns.n_burnin_draws = 0;
            ns.n_keep_draws   = win_iter;
            ns.n_adapt_draws  = win_iter;

            mcmc::Mat_t win_out;
            const bool okw = mcmc::nuts(
                theta_cat_, adapter, win_out, nullptr, cfg_.nuts_settings);
            if (!okw || win_out.n_rows < 10) {
                // Window too short or sampler failed; skip mass-matrix
                // update and proceed to next window. If win_out has
                // some draws, advance theta_cat_.
                if (win_out.n_rows > 0) {
                    theta_cat_ = win_out.row(win_out.n_rows - 1).t();
                }
                continue;
            }
            theta_cat_ = win_out.row(win_out.n_rows - 1).t();

            // Welford covariance on this window.
            const std::size_t n_w = win_out.n_rows;
            arma::mat D(win_out.memptr(), n_w, win_out.n_cols, /*copy=*/false);
            arma::vec mu_w = arma::mean(D, 0).t();
            arma::mat Dc   = D.each_row() - mu_w.t();
            arma::mat Sigma = (Dc.t() * Dc) / static_cast<double>(n_w - 1);

            // Stan-style regularization: shrink toward identity.
            //   Σ_reg = (n/(n+5))·Σ + 1e-3·(5/(n+5))·I
            const double shrink = static_cast<double>(n_w) / (n_w + 5.0);
            const double ridge  = 1e-3 * (5.0 / (n_w + 5.0));
            arma::mat Sigma_reg = shrink * Sigma
                + ridge * arma::eye<arma::mat>(metric_dim_local, metric_dim_local);

            // Install precision = inverse of regularized covariance.
            // If inversion fails (very rare with regularization), keep
            // previous mass matrix.
            try {
                ns.precond_mat = build_precond_(Sigma_reg);
            } catch (const std::runtime_error&) {
                // inv_sympd failed; keep previous mass matrix
            }
            // DA persists carry across windows (no reset between windows).
        }

        // ----- Phase III: final step-size tune with frozen mass matrix.
        // Mass matrix is now whatever Phase II's last successful window
        // installed (or empty/identity if all windows skipped). DA
        // persists continue from Phase II's end.
        {
            cfg_.nuts_settings.rng_seed_value =
                static_cast<std::uint64_t>(rng());
            const std::size_t n3 = std::max<std::size_t>(cfg_.tp_phase3_iters, 1);
            ns.n_burnin_draws = n3;
            ns.n_keep_draws   = 1;
            ns.n_adapt_draws  = n3;

            mcmc::Mat_t out3;
            const bool ok3 = mcmc::nuts(
                theta_cat_, adapter, out3, nullptr, cfg_.nuts_settings);
            if (!ok3 || out3.n_rows == 0) {
                ns.n_burnin_draws = saved_burnin;
                ns.n_keep_draws   = saved_keep;
                ns.n_adapt_draws  = saved_adapt;
                throw std::runtime_error(
                    "joint_nuts_block '" + cfg_.name +
                    "': 3-phase warmup Phase III (final step-size tune) failed");
            }
            theta_cat_ = out3.row(out3.n_rows - 1).t();
        }

        // Restore per-call budget for normal sampling step().
        ns.n_burnin_draws = saved_burnin;
        ns.n_keep_draws   = saved_keep;
        ns.n_adapt_draws  = saved_adapt;
    }

    // Kernel-control slot-freeze state (DESIGN_NOTES Sec.10.a Approach B).
    // frozen_slots_ is the source of truth (set by freeze_sub / cleared by
    // unfreeze_sub); frozen_unc_idx_ is the derived index vector of positions
    // in theta_cat_ that are frozen, rebuilt by rebuild_frozen_unc_idx_.
    std::unordered_set<std::string> frozen_slots_;
    arma::uvec                      frozen_unc_idx_;   // empty when no slot frozen

    void rebuild_frozen_unc_idx_() {
        std::vector<arma::uword> ids;
        for (std::size_t s = 0; s < cfg_.sub_params.size(); ++s) {
            if (!frozen_slots_.count(cfg_.sub_params[s].name)) continue;
            const std::size_t off = unc_offsets_.empty()
                                        ? offsets_[s]
                                        : unc_offsets_[s];
            const std::size_t d   = unc_dims_.empty()
                                        ? cfg_.sub_params[s].dim
                                        : unc_dims_[s];
            for (std::size_t k = 0; k < d; ++k)
                ids.push_back(static_cast<arma::uword>(off + k));
        }
        if (ids.empty()) { frozen_unc_idx_.reset(); return; }
        frozen_unc_idx_.set_size(ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) frozen_unc_idx_[i] = ids[i];
    }

    joint_nuts_block_config cfg_;
    arma::vec               theta_cat_;      // UNCONSTRAINED sampler state
                                             // (what mcmc::nuts integrates).
    arma::vec               theta_cat_nat_;  // NATURAL-scale view, rebuilt in
                                             // rebuild_named_outputs_(); ==
                                             // theta_cat_ when all_real_.
    std::vector<std::size_t> offsets_;   // NATURAL start index per sub_param
    std::size_t             total_dim_ = 0;     // NATURAL total (== initial_cat,
                                                // current(), history length)
    // Dual offset scheme for DIMENSION-CHANGING slices (SIMPLEX etc.). For a
    // dim-preserving block these equal offsets_/total_dim_ exactly, so the
    // sampler state, slicing and asserts are bit-identical to before.
    std::vector<std::size_t> unc_offsets_;  // UNCONSTRAINED start index
    std::vector<std::size_t> unc_dims_;     // UNCONSTRAINED width per sub_param
    std::size_t             total_unc_dim_ = 0;   // == theta_cat_ length
    bool                    has_dim_changing_ = false;
    bool                    all_real_  = true;  // every sub_param REAL?
                                                // enables bit-identical V0
                                                // fast-paths.
    block_context           context_;
    bool                    first_call_ = true;
    bool                    dense_metric_adapted_ = false;  // T11
    int                     auto_selected_dense_ = -1;  // -1=n/a 0=diag 1=dense
    double                  auto_selected_cond_  = 0.0;  // cond(R) at decision
    std::vector<arma::vec>  history_buf_;

    // Cache of (sub_param.name -> arma::vec slice) rebuilt after every
    // step() and set_current(). current_named_outputs() returns this
    // by value so composite_block can write it through.
    std::unordered_map<std::string, arma::vec> named_outputs_cache_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_JOINT_NUTS_BLOCK_HPP

/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  block_sampler.hpp  --  the abstract base class every block in AI4BayesCode
 *                         must satisfy.
 *
 *  DESIGN CONTRACT (v0)
 *  ====================
 *  A block_sampler is a self-contained, stateful MCMC updater for one "block"
 *  of parameters inside a larger Gibbs sampler. Blocks communicate with the
 *  outside world ONLY through the following channels:
 *
 *    set_context(ctx)   <- external inputs required by this step
 *                          (other blocks' current values, fixed data, ...).
 *                          Block MUST copy what it needs out of ctx; it must
 *                          not hold any pointers into ctx after the call
 *                          returns.
 *
 *    step(rng)          -- one MCMC update using the block's own internal
 *                          state + the context most recently set. This must
 *                          be a closed computation: no callbacks, no reads
 *                          from global state, no writes to anything outside
 *                          the block itself.
 *
 *    current()          -> const view of the block's current parameter value
 *                          on the NATURAL (user-facing) scale. The caller
 *                          must copy if it wants to persist the value.
 *
 *    set_current(theta) <- force the block's current parameter value (used
 *                          for warm-starting the outer loop from a prior
 *                          run or from a user-supplied initial value).
 *
 *  HISTORY MODE
 *  ------------
 *  When keep_history is enabled, the block stores every draw produced by
 *  step() into an internal buffer. get_history() returns all stored draws
 *  as an AI4BayesCode::history_map — a named map from block name to an
 *  arma::mat of shape (n_draws x dim). Scalar blocks store 1-col matrices;
 *  vector blocks store (n_draws x dim) matrices. When keep_history is
 *  disabled (the default), get_history() returns the current draw as a
 *  1-row matrix.
 *
 *  The return type is BACKEND-NEUTRAL: Rcpp auto-wraps
 *  std::unordered_map<std::string, arma::mat> to a named R list of
 *  matrices; pybind11 (via pybind_casters.hpp) converts to a Python
 *  dict[str, np.ndarray]. See types.hpp for the full type catalog.
 *
 *  History storage lives in each leaf block, not in the composite. The
 *  composite's get_history() simply merges children's histories.
 *
 *  FORBIDDEN
 *  ---------
 *    - Blocks may not hold references to each other.
 *    - Blocks may not hold references to shared_data_t or composite_block.
 *    - Blocks may not read or write any state outside themselves during step.
 *    - Blocks may not call each other.
 *
 *  The only class in AI4BayesCode that is allowed to know how blocks fit
 *  together is composite_block, and its knowledge is limited to a fixed
 *  Gibbs order + a shared_data_t that holds the current values. composite_block
 *  is itself a block_sampler, so the interface is recursive: a composite of
 *  blocks is indistinguishable from a single block when viewed from outside.
 *
 *  CONTEXT REPRESENTATION
 *  ----------------------
 *  block_context (and state_map more broadly) is a plain
 *  std::unordered_map<std::string, arma::vec>. Scalars are stored as
 *  length-1 vectors. Matrices, when needed, can be stored column-major
 *  flattened with the shape encoded in the key name (e.g. "B.3x20") or
 *  carried via a parallel matrix map; for v0 we keep vectors only,
 *  which is sufficient for the models the first release targets.
 *
 *  The choice of std::unordered_map (as opposed to JSON, variant, or hand
 *  written structs) is deliberate:
 *    - string keys are what an AI code generator will naturally emit;
 *    - arma::vec is the scalar/vector type mcmclib already uses, so no
 *      extra conversion on the hot path;
 *    - no schema: blocks declare the keys they depend on in their own
 *      documentation, and composite_block routes accordingly.
 *================================================================================*/

#ifndef AI4BAYESCODE_BLOCK_SAMPLER_HPP
#define AI4BAYESCODE_BLOCK_SAMPLER_HPP

#include <random>
#include <string>
#include <unordered_map>

#include "types.hpp"

#ifndef MCMC_USE_RCPP_ARMADILLO
# include <armadillo>
#else
# include <RcppArmadillo.h>
#endif

namespace AI4BayesCode {
namespace detail {

/**
 * Build a single-entry history_map for a leaf block.
 *
 * - Scalar blocks (dim==1): stored as arma::mat of shape (n_draws, 1).
 * - Vector blocks (dim>1): stored as arma::mat of shape (n_draws, dim),
 *   each row being one draw's vector.
 *
 * When the buffer is empty (history recording disabled), `fallback` is
 * used to produce a single row — so callers can always call
 * get_history() regardless of mode.
 */
inline history_map make_history_map(const std::string& name,
                                    const std::vector<arma::vec>& buf,
                                    const arma::vec& fallback) {
    const bool empty = buf.empty();
    const std::size_t dim = empty ? fallback.n_elem : buf[0].n_elem;
    const std::size_t n   = empty ? 1 : buf.size();

    arma::mat m(n, dim);
    if (empty) {
        for (std::size_t j = 0; j < dim; ++j) m(0, j) = fallback[j];
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < dim; ++j) m(i, j) = buf[i][j];
        }
    }
    history_map out;
    out.emplace(name, std::move(m));
    return out;
}

} // namespace detail

/**
 * @brief Key-value bundle passed into block_sampler::set_context.
 *
 * Keys are human-readable strings. Values are column vectors; scalars are
 * stored as length-1 vectors. A context is always a complete snapshot of
 * everything a particular block depends on at the moment set_context is
 * called; it is constructed fresh by the composite driver before each step.
 */
using block_context = state_map;

/**
 * @brief Engine kind for a block_sampler.
 *
 * The composite uses this to dispatch how each child's post-step
 * value is written into shared_data_t:
 *   - MCMC blocks: write current() (the freshly drawn sample).
 *   - VI blocks:   write current_sample(rng) (a fresh draw eta ~ q,
 *                  then theta = constrain(eta)). This preserves the
 *                  VI block's posterior uncertainty for any MCMC
 *                  sibling reading the key in hybrid mode.
 *
 * Hybrid composites contain a mix of MCMC and VI children; the
 * dispatch happens per child via composite_block's per-child loop.
 * See system_design.md §18.2 and §18.4 for the full backing.
 *
 * Backward compatibility: every existing block_sampler subclass
 * inherits the default override (MCMC) without code change. New
 * vi_block subclasses override to VI.
 */
enum class engine_kind_t {
    MCMC,   ///< sampling-based update (NUTS / Gibbs / BART / RJMCMC / HMM / SBP / ...)
    VI      ///< optimization-based update (vi_block subclasses)
};

/**
 * @brief Abstract base class for every stateful block in AI4BayesCode.
 *
 * See the file header for the full design contract. Implementations of this
 * interface are the only units that matter: composite_block, nuts_block,
 * bart_block, categorical_gibbs_block, and any user-written block all live
 * behind this signature.
 */
class block_sampler {
public:
    virtual ~block_sampler() = default;

    // ---- The four hot-path methods -------------------------------------

    /**
     * Push the current inputs this block needs from the outside world.
     * The implementation MUST copy whatever it needs out of @p ctx; it may
     * not keep pointers into ctx after the call returns.
     */
    virtual void set_context(const block_context& ctx) = 0;

    /**
     * One MCMC update. Reads only the block's own internal state plus the
     * context most recently installed by set_context. May advance @p rng.
     * Must not touch anything outside the block.
     */
    virtual void step(std::mt19937_64& rng) = 0;

    /**
     * The block's current parameter value on the natural user-facing scale.
     * Returned as a reference into the block's own storage; the caller is
     * responsible for copying if it wants to persist the value across
     * subsequent calls to step or set_current.
     */
    virtual const arma::vec& current() const = 0;

    /**
     * Forcibly overwrite the block's current parameter value. Used for
     * warm-starting the outer loop from a previous run or from a user
     * supplied initial value. Does NOT reset adaptation state.
     */
    virtual void set_current(const arma::vec& theta) = 0;

    // ---- History ----------------------------------------------------------

    virtual void set_keep_history(bool keep) { keep_history_ = keep; }
    virtual bool keep_history() const noexcept { return keep_history_; }

    // ---- Tree snapshot recording (BART-family only) ----------------------
    //
    // keep_tree_ is ORTHOGONAL to keep_history_:
    //   keep_history_  controls numeric / vec recording (sigma, f_bart
    //                  values, refresher outputs, etc.). Cheap O(N) per
    //                  step.
    //   keep_tree_     controls per-step serialisation of BART forest
    //                  snapshots so predict_history(X_new) can iterate
    //                  over historical forests. EXPENSIVE
    //                  O(T * tree_size) per step. Default OFF.
    //
    // Only BART-family blocks (bart_block, softbart_block, genbart_block)
    // override set_keep_tree() to forward to their kernel's history flag.
    // For non-BART blocks the override is a no-op (silently ignored).
    virtual void set_keep_tree(bool keep) { keep_tree_ = keep; }
    virtual bool keep_tree() const noexcept { return keep_tree_; }

    /**
     * Return all stored draws as a history_map: {block_name: arma::mat}.
     *
     * Recursive structure (mirrors the block hierarchy):
     *   - A LEAF block returns a map with (typically) one entry keyed by
     *     name(). The entry is a matrix (n_draws x dim), with dim=1 for
     *     scalar blocks. A block may return multiple entries if it owns
     *     multiple logically separate quantities.
     *   - A COMPOSITE block iterates its children and MERGES the per-child
     *     maps into one flat map. Nested composites therefore flatten
     *     naturally.
     *
     * The default implementation below returns a 1-row fallback for
     * blocks that don't override; this ensures callers can always
     * introspect regardless of history mode.
     */
    virtual history_map get_history() const {
        const std::string& nm = name();
        history_map out;
        if (nm.empty()) return out;
        const arma::vec& cur = current();
        arma::mat m(1, cur.n_elem);
        for (std::size_t j = 0; j < cur.n_elem; ++j) m(0, j) = cur[j];
        out.emplace(nm, std::move(m));
        return out;
    }

    /**
     * Number of draws stored in the history buffer.
     * Returns 1 if history is disabled (current draw only).
     */
    virtual std::size_t history_size() const noexcept { return 1; }

    /**
     * Clear the history buffer (e.g. after burnin).
     */
    virtual void clear_history() {}

    // ---- Introspection --------------------------------------------------

    /**
     * A human-readable name used as the key under which this block's value
     * lives in a shared_data_t. Must be unique within a composite_block.
     */
    virtual const std::string& name() const noexcept {
        static const std::string empty{};
        return empty;
    }

    /**
     * Dimensionality of the block's parameter on the natural scale.
     */
    virtual std::size_t dim() const noexcept = 0;

    /**
     * Named outputs that the block wants the composite to write back into
     * shared_data after step(). The default implementation returns a single
     * entry mapping name() -> current().
     *
     * Blocks that own multiple sub-parameters (e.g. joint_nuts_block over
     * [theta; b]) override this to return one entry per sub-parameter.
     */
    virtual state_map current_named_outputs() const {
        const std::string& nm = name();
        if (nm.empty()) return {};
        return { { nm, current() } };
    }

    /**
     * RNG-aware overload of current_named_outputs. composite_block calls
     * THIS variant during step() so VI children can write a fresh
     * q-sample (current_sample(rng)) instead of their q-mean
     * (current()), preserving uncertainty for MCMC siblings in hybrid
     * mode.
     *
     * The default implementation delegates to the rng-less version
     * (MCMC blocks don't need rng for this — their current() is the
     * fresh sample). vi_block overrides to route through
     * current_sample(rng).
     *
     * See system_design.md §18.4 for the rationale and the hybrid-
     * correctness invariant.
     */
    virtual state_map current_named_outputs(std::mt19937_64& rng) const {
        (void) rng;
        return current_named_outputs();
    }

    // ---- Engine kind ----------------------------------------------------

    /**
     * Engine kind for this block. Default is MCMC; vi_block subclasses
     * override to return VI. composite_block dispatches the post-step
     * shared_data write based on this (see current_named_outputs(rng)).
     *
     * See system_design.md §18.2 for the architectural backing.
     */
    virtual engine_kind_t engine_kind() const noexcept {
        return engine_kind_t::MCMC;
    }

    // ---- Kernel-tuning interface (readapt_NUTS dispatch target) --------

    /**
     * Whether this block supports kernel-tuning via readapt(). Default
     * false: most blocks have no tunable kernel state (Gibbs is
     * closed-form; BART has tree forest, no metric; VI has its own
     * always-on RAABBVI optimizer; RJMCMC has no metric; etc.).
     *
     * NUTS-family blocks override to return true and implement
     * readapt(n, reset, rng) to re-tune their dual-averaging state +
     * (optionally) mass matrix.
     *
     * composite_block::readapt_NUTS uses this flag to decide whether
     * to invoke readapt() on each child or skip it. Non-supporting
     * children are silently skipped. See system_design.md §13 NUTS-
     * family for the full readapt_NUTS contract.
     */
    virtual bool supports_readapt() const noexcept {
        return false;
    }

    /**
     * Re-tune kernel-level adaptation state (mass matrix, step size,
     * dual-averaging accumulators) WITHOUT advancing chain state.
     * Default: no-op (only called when supports_readapt() is true).
     *
     * NUTS-family blocks override to:
     *   1. snapshot theta_unc_ / theta_natural_;
     *   2. if reset, reinitialize dual-averaging persistent state;
     *   3. run n internal adaptation iterations (mcmclib::nuts with
     *      n_burnin=n, n_adapt=n, persistent adapt enabled);
     *   4. restore theta_unc_ / theta_natural_ from snapshot.
     *
     * After readapt(), the block's get_current() value is bitwise
     * identical to before, but its NUTS metric is re-tuned for the
     * current context.
     *
     * \param n      number of internal adaptation iterations to run
     * \param reset  if true, reinitialize dual-averaging state to
     *               defaults before adapting; if false (default),
     *               continue dual-averaging from previous state
     * \param rng    third RNG stream (readapt_rng_), separate from
     *               step()'s rng_ and predict_at's predict_rng_
     */
    virtual void readapt(std::size_t n,
                         bool reset,
                         std::mt19937_64& rng,
                         std::size_t max_tree_depth_override = 0) {
        (void)n; (void)reset; (void)rng; (void)max_tree_depth_override;
        // Default no-op; only called when supports_readapt() == true.
        // max_tree_depth_override: 0 = use the block's configured depth;
        // >0 = temporarily cap NUTS tree depth for these n adaptation iters.
    }

protected:
    bool keep_history_ = false;
    bool keep_tree_    = false;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_BLOCK_SAMPLER_HPP

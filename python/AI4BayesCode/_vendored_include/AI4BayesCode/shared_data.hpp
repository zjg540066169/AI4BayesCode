/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  shared_data.hpp  --  the composite-block-owned bag of everything that
 *                       block_sampler implementations might need to see.
 *
 *  DESIGN NOTES
 *  ============
 *  shared_data_t is the ONLY place in block_mcmc that knows how blocks
 *  depend on each other. A block_sampler never touches shared_data_t
 *  directly; only composite_block does. The flow during a single Gibbs
 *  sweep is:
 *
 *      for each child block B in gibbs_order:
 *          1. ctx = shared.build_context_for(B.name())
 *          2. B.set_context(ctx)
 *          3. B.step(rng)
 *          4. shared.write_back(B.name(), B.current())
 *          5. shared.refresh_derived_for(B.name())
 *
 *  The three per-block tables ({dependencies, invalidates, refreshers})
 *  together encode the DAG of the model. For v0 they are populated by
 *  hand when a composite is assembled; later a code generator will emit
 *  them. They are the minimum metadata needed for a correct Gibbs sweep,
 *  and they are also exactly what an LLM would reason about when
 *  translating a user's prose model description into a block_mcmc setup.
 *
 *  THE DATA MODEL
 *  --------------
 *  All values live in a single string-keyed map of arma::vec. The keys
 *  conventionally fall into three groups:
 *
 *    - fixed data         "y", "X_col_3", "alpha_prior", ...
 *    - block parameters   "beta", "sigma", "theta_simplex", ...
 *    - derived quantities "residual", "linear_predictor", "fitted", ...
 *
 *  block_mcmc itself does not care about these groups; they are just a
 *  naming convention that makes the dependency metadata readable. Only
 *  the refresher closures need to know that "residual" is derived from
 *  "y", "X", and "beta". That knowledge lives inside the closure body.
 *
 *  CONCURRENCY
 *  -----------
 *  shared_data_t is NOT thread-safe. It is owned by exactly one
 *  composite_block and accessed in a single Gibbs-sweep thread. Parallel
 *  chains should each own their own shared_data_t instance.
 *================================================================================*/

#ifndef AI4BAYESCODE_SHARED_DATA_HPP
#define AI4BAYESCODE_SHARED_DATA_HPP

#include "block_sampler.hpp"   // for block_context alias

#include <functional>
#include <iosfwd>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef MCMC_USE_RCPP_ARMADILLO
# include <armadillo>
#endif

namespace AI4BayesCode {

/**
 * @brief Container for all cross-block state in a block_mcmc composite.
 *
 * @see file header for usage patterns.
 */
class shared_data_t {
public:
    /// Signature of a refresher: given the current shared_data_t, compute a
    /// derived arma::vec. Refreshers must be pure functions of the map.
    using refresher_fn = std::function<arma::vec(const shared_data_t&)>;

    /// Signature of a STOCHASTIC refresher: given the current shared_data_t
    /// and a reference to an RNG, return a sampled arma::vec. Used for
    /// observation-layer nodes (y_rep) at predict_at time so posterior-
    /// predictive draws can be produced by the same DAG walk that propagates
    /// deterministic changes. Must NOT mutate the shared_data_t argument
    /// (compiler enforces const); the RNG IS mutated as sampling advances.
    using stochastic_refresher_fn =
        std::function<arma::vec(const shared_data_t&, std::mt19937_64&)>;

    // ---- Construction / bulk setters -----------------------------------

    shared_data_t() = default;

    /// Set or overwrite a value. Typically used during initial setup to
    /// install fixed data and initial parameter values.
    void set(const std::string& key, arma::vec value) {
        values_[key] = std::move(value);
    }

    /// Check whether a key exists.
    bool has(const std::string& key) const {
        return values_.find(key) != values_.end();
    }

    /// Read a value; throws std::out_of_range if missing. Returns a const
    /// reference into the internal storage; do not mutate through it.
    const arma::vec& get(const std::string& key) const {
        auto it = values_.find(key);
        if (it == values_.end()) {
            throw std::out_of_range("shared_data_t::get: missing key '" + key + "'");
        }
        return it->second;
    }

    /// Mutable access used by refreshers when they need to touch a value
    /// in place. Prefer `set` in ordinary code.
    arma::vec& get_mut(const std::string& key) {
        auto it = values_.find(key);
        if (it == values_.end()) {
            throw std::out_of_range("shared_data_t::get_mut: missing key '" + key + "'");
        }
        return it->second;
    }

    // ---- DAG metadata ---------------------------------------------------

    /**
     * Declare the set of shared_data keys a given block reads from when
     * its set_context is called. The composite will package exactly these
     * keys into the block_context passed to set_context.
     *
     * Calling declare_dependencies replaces any previous list for the
     * same block; this is intentional so that block reconfiguration is a
     * single atomic update.
     */
    void declare_dependencies(const std::string& block_name,
                              std::vector<std::string> keys) {
        dependencies_[block_name] = std::move(keys);
    }

    /**
     * Declare the set of derived keys that must be recomputed after the
     * given block updates. The refreshers for those keys must have been
     * registered (see register_refresher) before the composite runs its
     * first sweep.
     */
    void declare_invalidates(const std::string& block_name,
                             std::vector<std::string> derived_keys) {
        invalidates_[block_name] = std::move(derived_keys);
    }

    /**
     * Register the closure that computes a derived key. The closure is
     * pure: it reads from the shared_data_t passed to it and returns the
     * new value; it must not capture anything that could become stale.
     */
    void register_refresher(const std::string& derived_key, refresher_fn fn) {
        refreshers_[derived_key] = std::move(fn);
    }

    /**
     * Immediately recompute every registered derived key. Used once at
     * setup time, after all fixed data and initial parameter values have
     * been installed, to prime the shared_data_t before the first sweep.
     */
    void refresh_all() {
        for (auto& kv : refreshers_) {
            values_[kv.first] = kv.second(*this);
        }
    }

    // ---- Hot-path operations used by composite_block -------------------

    /**
     * Build the block_context that will be handed to a particular block's
     * set_context. This is a simple projection of the dependency list for
     * that block into a fresh map. The context contains COPIES of the
     * values; mutating it does not affect shared_data_t.
     */
    block_context build_context_for(const std::string& block_name) const {
        block_context ctx;
        auto it = dependencies_.find(block_name);
        if (it == dependencies_.end()) {
            // No declared dependencies. This is legal (e.g. a block that
            // samples a prior-only parameter), so return an empty context.
            return ctx;
        }
        for (const auto& key : it->second) {
            auto vit = values_.find(key);
            if (vit == values_.end()) {
                throw std::out_of_range(
                    "shared_data_t::build_context_for: block '" + block_name
                    + "' depends on missing key '" + key + "'");
            }
            ctx[key] = vit->second; // copy
        }
        return ctx;
    }

    /**
     * Write the just-sampled value of a block back into shared storage.
     * Uses the block's name as the shared_data key; if the generator
     * needs to split a block's parameters into multiple shared_data
     * entries, it should do that through the refresher mechanism rather
     * than through this single write-back point.
     */
    void write_back(const std::string& block_name, const arma::vec& new_val) {
        values_[block_name] = new_val; // copy
    }

    /**
     * Recompute every derived key that depends on the given block. The
     * caller (composite_block) invokes this after write_back so that the
     * next child block in the Gibbs order sees the freshly updated
     * derived quantities.
     */
    void refresh_derived_for(const std::string& block_name) {
        auto it = invalidates_.find(block_name);
        if (it == invalidates_.end()) return;
        for (const auto& derived_key : it->second) {
            auto rit = refreshers_.find(derived_key);
            if (rit == refreshers_.end()) {
                throw std::out_of_range(
                    "shared_data_t::refresh_derived_for: no refresher for '"
                    + derived_key + "' (invalidated by block '" + block_name
                    + "')");
            }
            values_[derived_key] = rit->second(*this);
        }
    }

    // ---- Predict-at DAG metadata ----------------------------------------

    /**
     * Mark a key as a replaceable data input for predict_at. Only keys
     * marked here can appear in the replaced_values argument to
     * composite_block::predict_at. Typical data inputs: "X", "v".
     * Do NOT mark block outputs ("sigma", "f_bart") or priors ("alpha").
     */
    void declare_data_input(const std::string& key) {
        data_input_keys_.insert(key);
    }

    /// The set of keys that predict_at accepts. Used for validation.
    const std::unordered_set<std::string>& valid_predict_inputs() const noexcept {
        return data_input_keys_;
    }

    /// Check whether a key has a registered refresher (i.e. it is derived).
    bool has_refresher(const std::string& key) const {
        return refreshers_.find(key) != refreshers_.end();
    }

    /// Run a single refresher, updating values_[key] in place.
    void refresh_key(const std::string& key) {
        auto rit = refreshers_.find(key);
        if (rit == refreshers_.end()) {
            throw std::out_of_range(
                "shared_data_t::refresh_key: no refresher for '" + key + "'");
        }
        values_[key] = rit->second(*this);
    }

    // ---- Stochastic refreshers (for y_rep at predict_at time) ------------

    /// Register a stochastic refresher that samples a node from the current
    /// shared_data_t and an RNG. Used at predict_at time for observation-
    /// layer nodes (e.g. y_rep ~ N(f_bart, sigma^2)).
    void register_stochastic_refresher(const std::string& key,
                                       stochastic_refresher_fn fn) {
        stochastic_refreshers_[key] = std::move(fn);
    }

    /// Check whether a stochastic refresher is registered for this key.
    bool has_stochastic_refresher(const std::string& key) const {
        return stochastic_refreshers_.find(key) !=
               stochastic_refreshers_.end();
    }

    /// Run a stochastic refresher, writing the sampled value into values_.
    /// @param key  key whose stochastic refresher to run
    /// @param rng  RNG state, advanced by the sampling call
    void run_stochastic_refresher(const std::string& key,
                                  std::mt19937_64& rng) {
        auto it = stochastic_refreshers_.find(key);
        if (it == stochastic_refreshers_.end()) {
            throw std::out_of_range(
                "shared_data_t::run_stochastic_refresher: no stochastic "
                "refresher for '" + key + "'");
        }
        values_[key] = it->second(*this, rng);
    }

    /// Keys for which a stochastic refresher has been registered.
    /// Exposed so composite_block::predict_at can iterate and check
    /// reachability without touching the map directly.
    std::vector<std::string> stochastic_refresher_keys() const {
        std::vector<std::string> keys;
        keys.reserve(stochastic_refreshers_.size());
        for (const auto& kv : stochastic_refreshers_) {
            keys.push_back(kv.first);
        }
        return keys;
    }

    // ---- Predict DAG (generative / causal direction) --------------------
    //
    // Separate from the Gibbs DAG (dependencies_ / invalidates_).
    //
    //   Gibbs DAG:   "f reads sigma2 when sampling"  (sigma2 → f)
    //   Predict DAG: "X produces f_bart"             (X → f_bart)
    //
    // The predict DAG stores DIRECT parent→children edges in the
    // generative model. Only direct edges, no skipping levels.
    // predict_at() walks this DAG forward from replaced inputs to
    // find all affected downstream nodes.

    /**
     * Declare direct generative edges: when `from` changes, these
     * `to` nodes are directly affected. Only declare immediate
     * children, not transitive descendants.
     *
     * Example for Y ~ N(f(X), v2), f ~ N(BART(X), sigma2):
     *   declare_predict_edges("X",      {"f_bart"});
     *   declare_predict_edges("f_bart", {"f"});
     *   declare_predict_edges("sigma2", {"f"});
     *   declare_predict_edges("f",      {"Y"});
     *   declare_predict_edges("v2",     {"Y"});
     */
    void declare_predict_edges(const std::string& from,
                               std::vector<std::string> to) {
        predict_edges_[from] = std::move(to);
    }

    /**
     * Declare VISUALIZATION-ONLY context edges: prior / hyperprior
     * parents of a sampled parameter (or the tree prior of a BART
     * forest, etc.). These edges are NEVER traversed by predict_at's
     * BFS — `predict_downstream_of` and `predict_stochastic_sampleable`
     * do not read `context_edges_`. They exist solely so `get_dag()`
     * (and `plot_dag` in R) can render the full *generative* DAG with
     * the prior / hyperprior context shown faded, while the solid
     * sub-DAG remains exactly the set predict_at recomputes.
     *
     * Posterior prediction conditions on the MCMC draws of the
     * parameters, NOT on re-sampling their priors, so a context edge
     * `hyperparam -> param` must NEVER appear in `predict_edges_`
     * (that would wrongly make the BFS think replacing the hyperparam
     * should recompute the parameter). Keep the two maps disjoint.
     *
     * Example (MetaRegBartSpline):
     *   declare_context_edges("sigma_nu",    {"tau"});
     *   declare_context_edges("sigma_lambda",{"tau"});
     *   declare_context_edges("sigma_rw2",   {"beta"});
     *   declare_context_edges("a_smooth",    {"sigma_rw2"});
     *   declare_context_edges("b_smooth",    {"sigma_rw2"});
     *   declare_context_edges("tree_prior",  {"BART"});
     */
    void declare_context_edges(const std::string& from,
                               std::vector<std::string> to) {
        context_edges_[from] = std::move(to);
    }

    /**
     * Walk the predict DAG forward from a set of changed inputs.
     * Returns DETERMINISTIC downstream nodes (excluding those that carry a
     * registered stochastic refresher — those are handled separately in
     * predict_stochastic_sampleable) in topological-safe BFS order.
     *
     * Availability rule for a parent P of node N:
     *   - P is available if P is in `changed` (user-supplied this round or
     *     already recomputed via propagation), OR
     *   - P is a non-data-input key currently in values_ (i.e. P is a
     *     model parameter / derived quantity whose current value we trust,
     *     such as sigma, f_bart, theta).
     *
     * Trigger condition for N to be recomputed:
     *   - SOME parent of N is in `changed` (propagation was triggered), AND
     *   - ALL parents of N are available per the rule above.
     *
     * This yields the meta-analysis example behaviour:
     *   predict_at(X=X_new) walks X → f_bart → f; blocks at f → y because
     *   y's other parent v2 is a data_input and was not supplied, hence
     *   unavailable.
     *
     * @param changed  The set of nodes whose values were replaced.
     * @return  Ordered list of deterministic downstream nodes to refresh.
     */
    std::vector<std::string> predict_downstream_of(
            const std::unordered_set<std::string>& changed) const {
        // Build reverse map: child → set of parents
        std::unordered_map<std::string, std::vector<std::string>> parents;
        for (const auto& kv : predict_edges_) {
            for (const auto& child : kv.second) {
                parents[child].push_back(kv.first);
            }
        }

        // A parent is "available" if it is in the reachable/changed set
        // OR if it is a non-data-input key currently in shared_data.
        // Non-data-input = a model parameter or derived quantity, whose
        // current value at the current MCMC state is what we want.
        auto is_nondata_param = [&](const std::string& k) {
            return values_.count(k) && !data_input_keys_.count(k);
        };

        std::unordered_set<std::string> reachable(changed.begin(),
                                                   changed.end());
        std::vector<std::string> result;
        std::vector<std::string> queue;

        // Seed queue with direct children of changed nodes
        for (const auto& node : changed) {
            auto it = predict_edges_.find(node);
            if (it != predict_edges_.end()) {
                for (const auto& child : it->second) {
                    queue.push_back(child);
                }
            }
        }

        // BFS with fixpoint iteration: nodes may need to wait for a sibling
        // to be resolved before their parent-set is complete.
        std::size_t max_iters = queue.size() * queue.size() + 1;
        std::size_t iter = 0;
        while (!queue.empty() && iter < max_iters) {
            ++iter;
            std::vector<std::string> next_queue;
            bool progress = false;

            for (const auto& node : queue) {
                if (reachable.count(node)) continue;  // already processed
                // Stochastic nodes are handled in Pass 2; exclude here.
                if (stochastic_refreshers_.count(node)) continue;

                bool any_parent_changed = false;
                bool all_parents_avail  = true;
                auto pit = parents.find(node);
                if (pit != parents.end()) {
                    for (const auto& p : pit->second) {
                        if (reachable.count(p)) {
                            any_parent_changed = true;
                        }
                        bool p_avail = reachable.count(p) ||
                                       is_nondata_param(p);
                        if (!p_avail) {
                            all_parents_avail = false;
                            break;
                        }
                    }
                }

                if (any_parent_changed && all_parents_avail) {
                    reachable.insert(node);
                    result.push_back(node);
                    progress = true;
                    auto eit = predict_edges_.find(node);
                    if (eit != predict_edges_.end()) {
                        for (const auto& child : eit->second) {
                            if (!reachable.count(child)) {
                                next_queue.push_back(child);
                            }
                        }
                    }
                } else {
                    next_queue.push_back(node);
                }
            }

            queue = std::move(next_queue);
            if (!progress) break;
        }

        return result;
    }

    /**
     * Returns the list of keys with REGISTERED stochastic refreshers whose
     * predict-DAG parents are all available. Used as Pass 2 of predict_at.
     * The caller should have completed deterministic propagation first and
     * pass the expanded `changed_after_pass1` set (original replaced inputs
     * ∪ deterministically-recomputed keys).
     *
     * Same availability rule as predict_downstream_of: a parent is available
     * if in changed_after_pass1 OR a non-data-input key in shared_data.
     *
     * @param changed_after_pass1  Changed set after Pass 1 completion.
     * @return  Keys of stochastic refreshers to sample (unordered).
     */
    std::vector<std::string> predict_stochastic_sampleable(
            const std::unordered_set<std::string>& changed_after_pass1) const {
        std::unordered_map<std::string, std::vector<std::string>> parents;
        for (const auto& kv : predict_edges_) {
            for (const auto& child : kv.second) {
                parents[child].push_back(kv.first);
            }
        }

        // Availability rule for STOCHASTIC refresher inputs (Pass 2):
        //   A parent P is "available" if it is in changed_after_pass1
        //   (user-supplied this round or recomputed in Pass 1), OR if
        //   it has ANY value in shared_data (i.e. values_.count(P)).
        //
        // The latter covers BOTH:
        //   - non-data-input keys (model parameters / derived quantities
        //     like beta, sigma, f_bart) — same as the old rule, AND
        //   - data_input keys with their default (training) value
        //     installed at construction time — NEW relative to the old
        //     rule.
        //
        // The data_input-with-default branch is what makes
        // declare_predict_edges("X", {"y_rep"}) compatible with
        // predict_at(list()): X has its training value in values_, so
        // it is available; the y_rep refresher reads the scratch X (=
        // training X) and samples posterior predictive at training data.
        // When the user calls predict_at(list(X = X_new)), X is in
        // changed_after_pass1 (via the replaced map), so it is also
        // available — same code path, different X value seen by the
        // refresher.
        //
        // This relaxation matches the natural "training data is the
        // default" semantic and is required for the predict DAG plot to
        // visually include X → y_rep edges for non-intermediate-node
        // models (LogisticRegression, SpikeSlabRJMCMC, SpikeSlabLaplace).
        std::vector<std::string> result;
        for (const auto& kv : stochastic_refreshers_) {
            const std::string& key = kv.first;
            auto pit = parents.find(key);
            bool all_avail = true;
            if (pit != parents.end()) {
                for (const auto& p : pit->second) {
                    bool p_avail = changed_after_pass1.count(p) ||
                                   values_.count(p);
                    if (!p_avail) { all_avail = false; break; }
                }
            }
            // A stochastic node with no predict-DAG parents is considered
            // always-available (e.g. a prior-draw terminal).
            if (all_avail) result.push_back(key);
        }
        return result;
    }

    /// Read-only access to the predict DAG edges for introspection.
    const std::unordered_map<std::string, std::vector<std::string>>&
    predict_edges() const noexcept { return predict_edges_; }

    /// Read-only access to the VIZ-ONLY context (prior/hyperprior)
    /// edges. Consumed by get_dag() / plot_dag only — never by the
    /// predict_at BFS.
    const std::unordered_map<std::string, std::vector<std::string>>&
    context_edges() const noexcept { return context_edges_; }

    // ---- Gibbs DAG downstream (for refreshers) --------------------------

    /**
     * Compute the transitive closure of derived keys that must be
     * recomputed when the given keys are replaced. Walks invalidates_
     * and returns keys in a safe refresh order (parents before children).
     *
     * NOTE: this uses the Gibbs DAG (invalidates_), not the predict DAG.
     * Used internally by composite_block::step() for refresher dispatch.
     */
    std::vector<std::string> downstream_derived_of(
            const std::unordered_set<std::string>& replaced_keys) const {
        std::unordered_set<std::string> affected_blocks;
        for (const auto& dep_kv : dependencies_) {
            for (const auto& dep_key : dep_kv.second) {
                if (replaced_keys.count(dep_key)) {
                    affected_blocks.insert(dep_kv.first);
                    break;
                }
            }
        }

        std::vector<std::string> result;
        std::unordered_set<std::string> visited;
        std::vector<std::string> queue;

        for (const auto& blk : affected_blocks) {
            auto it = invalidates_.find(blk);
            if (it != invalidates_.end()) {
                for (const auto& dk : it->second) {
                    if (!visited.count(dk)) {
                        visited.insert(dk);
                        queue.push_back(dk);
                    }
                }
            }
        }
        for (const auto& rk : replaced_keys) {
            auto it = invalidates_.find(rk);
            if (it != invalidates_.end()) {
                for (const auto& dk : it->second) {
                    if (!visited.count(dk)) {
                        visited.insert(dk);
                        queue.push_back(dk);
                    }
                }
            }
        }

        std::size_t head = 0;
        while (head < queue.size()) {
            const std::string& dk = queue[head++];
            result.push_back(dk);
            auto it = invalidates_.find(dk);
            if (it != invalidates_.end()) {
                for (const auto& child_dk : it->second) {
                    if (!visited.count(child_dk)) {
                        visited.insert(child_dk);
                        queue.push_back(child_dk);
                    }
                }
            }
        }
        return result;
    }

    // ---- Introspection (used by tests and code generators) ------------

    const std::unordered_map<std::string, arma::vec>& values() const noexcept {
        return values_;
    }
    const std::unordered_map<std::string, std::vector<std::string>>&
    dependencies() const noexcept { return dependencies_; }
    const std::unordered_map<std::string, std::vector<std::string>>&
    invalidates() const noexcept { return invalidates_; }

private:
    std::unordered_map<std::string, arma::vec> values_;
    std::unordered_map<std::string, std::vector<std::string>> dependencies_;
    std::unordered_map<std::string, std::vector<std::string>> invalidates_;
    std::unordered_map<std::string, refresher_fn> refreshers_;
    std::unordered_map<std::string, stochastic_refresher_fn>
        stochastic_refreshers_;
    std::unordered_set<std::string> data_input_keys_;
    std::unordered_map<std::string, std::vector<std::string>> predict_edges_;
    // VIZ-ONLY. Never read by predict_downstream_of /
    // predict_stochastic_sampleable. See declare_context_edges().
    std::unordered_map<std::string, std::vector<std::string>> context_edges_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_SHARED_DATA_HPP

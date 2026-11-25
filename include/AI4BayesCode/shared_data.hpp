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
 *          1. ctx = shared.context_view_for(B.name()), or the same keys
 *             and values copied by shared.build_context_for(B.name())
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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
        ++dependencies_version_;
    }

    /**
     * Declare the set of derived keys that must be recomputed after the
     * given block updates. The refreshers for those keys must have been
     * registered (see register_refresher) before the composite runs its
     * first sweep.
     *
     * IMPORTANT: block_name is a child BLOCK name -- the key that
     * composite_block hands to refresh_derived_for, i.e. child->name() --
     * NOT an individual shared_data key. A joint block writes several
     * shared_data entries (one per sub-parameter), but the refresh fires
     * once, under the block's name. So key this on the block name (e.g.
     * "reg_aug"), never on a sub-parameter (e.g. "beta0"): a sub-parameter
     * name here silently never fires. composite_block validates every
     * declared name against its children on the first sweep and throws on
     * any unmatched name (see composite_block::validate_dag_keys_).
     *
     * Repeated calls for the same block ACCUMULATE (set-union) rather than
     * replace: declare_invalidates("b", {"A"}) then ("b", {"B"}) leaves "b"
     * invalidating {"A","B"}. This differs deliberately from
     * declare_dependencies, which replaces -- there the argument is the
     * block's complete read-set, so an atomic replace is the intended
     * reconfiguration semantics.
     */
    void declare_invalidates(const std::string& block_name,
                             std::vector<std::string> derived_keys) {
        auto& existing = invalidates_[block_name];
        for (auto& k : derived_keys) {
            if (std::find(existing.begin(), existing.end(), k)
                    == existing.end()) {
                existing.push_back(std::move(k));
            }
        }
    }

    /**
     * Declare derived keys that the model recomputes from the current values
     * after every set_current call that changes a value they are computed
     * from, before the keys are used again. This holds also while the block
     * that invalidates them is frozen, when refresh_derived_for does not run
     * for that block. composite_block::freeze leaves these keys out of its
     * warning that freezing a block holds its derived keys stale, and emits
     * no such warning when no key remains. Declare a key only if every
     * set_current path recomputes it, or marks it to be recomputed before it
     * is used again. Repeated calls add to the set.
     */
    void declare_refreshed_after_set_current(
            std::vector<std::string> derived_keys) {
        for (auto& k : derived_keys) {
            refreshed_after_set_current_.insert(std::move(k));
        }
    }

    /// The keys declared with declare_refreshed_after_set_current().
    const std::unordered_set<std::string>&
    refreshed_after_set_current() const noexcept {
        return refreshed_after_set_current_;
    }

    /**
     * Register the closure that computes a derived key. The closure is
     * pure: it reads from the shared_data_t passed to it and returns the
     * new value; it must not capture anything that could become stale.
     */
    void register_refresher(const std::string& derived_key, refresher_fn fn) {
        if (refreshers_.find(derived_key) == refreshers_.end()) {
            refresher_order_.push_back(derived_key);
        }
        refreshers_[derived_key] = std::move(fn);
    }

    /**
     * Immediately recompute every registered derived key. Used once at
     * setup time, after all fixed data and initial parameter values have
     * been installed, to prime the shared_data_t before the first sweep.
     */
    void refresh_all() {
        // REGISTRATION order, not hash order. A derived key may read another
        // derived key (GPRegression: L_chol and L_prior both read K_matrix),
        // and the keys are seeded with placeholder zeros before this runs. In
        // hash order the reader can be visited first and get primed from the
        // placeholder -- e.g. an ESS prior Cholesky of sqrt(jitter) * I instead
        // of chol(K), so the first latent draw comes from the wrong prior, with
        // no error and a result that varies with the build. Registration order
        // is the order the model author wrote the chain in, and it is what
        // refresh_derived_for already relies on.
        for (const auto& key : refresher_order_) {
            auto it = refreshers_.find(key);
            if (it != refreshers_.end()) {
                values_[key] = it->second(*this);
            }
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
     * The context build_context_for(block_name) returns, with the same keys,
     * values and iteration order, but without copying the values: each entry
     * is a read-only view of the stored value. The views are made again after
     * a declare_dependencies call or when a stored value has another address
     * or length than when its view was made, so every call refers to the
     * current values. The returned map is valid until this object is next
     * modified; copy from it before that (set_context does).
     */
    const block_context& context_view_for(const std::string& block_name) {
        context_views_t::entry& e = views_.blocks[block_name];
        if (!same_list_(e)) {
            make_views_(e, block_name);
            return e.ctx;
        }
        bool ok = true;
        for (std::size_t i = 0; ok && i < e.keys.size(); ++i) {
            ok = e.addr[i] == context_views_t::address(*e.obj[i])
                 && e.n_elem[i] == e.obj[i]->n_elem;
        }
        if (!ok) remake_views_(e);
        return e.ctx;
    }

    /**
     * The views of context_view_for(block_name), or, when the values in them
     * take between min_bytes and max_bytes bytes (bounds included; the size
     * is the sum of n_elem * sizeof(double) over the entries), a copy with
     * the same keys, values and iteration order: the context
     * build_context_for(block_name) returns. The copy is made in `copy`,
     * which is first replaced by an empty map, from the stored values,
     * without looking the keys up again and without making the views again
     * unless declare_dependencies was called; `copy` is not modified when
     * the views are returned. Returns the views or `copy`.
     */
    const block_context& context_view_or_copy_for(const std::string& block_name,
                                                  std::size_t min_bytes,
                                                  std::size_t max_bytes,
                                                  block_context& copy) {
        context_views_t::entry& e = views_.blocks[block_name];
        if (!same_list_(e)) make_views_(e, block_name);
        // e.obj now points to the stored values of the list.
        bool same_len = true, same_addr = true;
        for (std::size_t i = 0; i < e.keys.size(); ++i) {
            same_len = same_len && e.n_elem[i] == e.obj[i]->n_elem;
            same_addr = same_addr
                        && e.addr[i] == context_views_t::address(*e.obj[i]);
        }
        const std::size_t bytes = same_len ? e.bytes : current_bytes_(e);
        if (bytes < min_bytes || bytes > max_bytes) {
            if (!same_len || !same_addr) remake_views_(e);
            return e.ctx;
        }
        // The copy is made from the stored values, so the views are not made
        // again here when a value has new storage or a new length (a derived
        // key whose refresher returns a new vector); the next call that
        // returns the views makes them again. Inserted in dependency order,
        // as build_context_for inserts them, into a new map.
        block_context().swap(copy);
        for (std::size_t i = 0; i < e.keys.size(); ++i) {
            copy[e.keys[i]] = *e.obj[i];
        }
        return copy;
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

    /**
     * Declare that these data inputs are indexed by the SAME observations, so
     * a prediction at new data has to replace all of them together.
     *
     * WHY THIS CANNOT BE INFERRED. Whether a data input that the caller left
     * out is still usable depends on what it is indexed by, and the graph does
     * not carry that. An initial condition shared by every time point stays
     * valid when the time grid is replaced; a covariate matrix indexed by
     * observation does NOT stay valid when the observations are replaced, even
     * though both look like "a data input feeding a node that also has a
     * replaced parent". Only the model knows which it has, so the model says
     * so here.
     *
     * WHAT IT CHANGES. Replacing a strict subset of a group leaves the other
     * members at their training values, which no longer line up with the
     * replaced ones. Those members, and every node downstream of them, are
     * then NOT PREDICTABLE for that call: predict_at skips them and they are
     * absent from its result, rather than being computed from a mixture of
     * new and training data. Everything reachable WITHOUT them is still
     * computed -- prediction goes as far as the graph allows and stops.
     *
     * Declaring no group (the default) preserves the previous behaviour
     * exactly, so a model that does not declare one is unaffected.
     */
    void declare_data_input_group(std::vector<std::string> keys) {
        for (const auto& k : keys) data_input_keys_.insert(k);
        data_input_groups_.push_back(std::move(keys));
    }

    /// Read-only access to the declared co-indexed data-input groups.
    const std::vector<std::vector<std::string>>&
    data_input_groups() const noexcept { return data_input_groups_; }

    /**
     * The keys that are NOT PREDICTABLE for a call that replaced `replaced`:
     * every co-indexed data input the caller left out, plus everything
     * downstream of one in the predict DAG.
     *
     * A group none of whose members were replaced contributes nothing -- the
     * whole group is at its training values and is mutually consistent, which
     * is what predict_at with no replacements means.
     */
    std::unordered_set<std::string> predict_withheld_cone(
            const std::unordered_set<std::string>& replaced) const {
        std::unordered_set<std::string> cone;
        std::vector<std::string> queue;
        for (const auto& grp : data_input_groups_) {
            bool any_replaced = false;
            for (const auto& k : grp) if (replaced.count(k)) { any_replaced = true; break; }
            if (!any_replaced) continue;
            for (const auto& k : grp)
                if (!replaced.count(k) && cone.insert(k).second) queue.push_back(k);
        }
        while (!queue.empty()) {
            const std::string node = queue.back(); queue.pop_back();
            auto it = predict_edges_.find(node);
            if (it == predict_edges_.end()) continue;
            for (const auto& child : it->second)
                if (cone.insert(child).second) queue.push_back(child);
        }
        return cone;
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
     *
     * Calling this MORE THAN ONCE for the same `from` ADDS to that node's
     * children rather than replacing them, and duplicates are ignored. It
     * used to overwrite, which silently discarded edges wherever a source
     * was declared in a loop (one edge per varying-coefficient forest) or in
     * two separate statements (a scale feeding both a random effect and the
     * likelihood): only the last call survived, so the graph predict_at
     * walked -- and the one get_dag drew -- was missing parents the model
     * plainly had.
     */
    void declare_predict_edges(const std::string& from,
                               std::vector<std::string> to) {
        auto& dst = predict_edges_[from];
        for (auto& child : to) {
            if (std::find(dst.begin(), dst.end(), child) == dst.end()) {
                dst.push_back(std::move(child));
            }
        }
    }

    /**
     * Declare VISUALIZATION-ONLY context edges: prior / hyperprior
     * parents of a sampled parameter (or the tree prior of a BART
     * forest, etc.). These edges are NEVER traversed by predict_at's
     * BFS — `predict_downstream_of` and `predict_stochastic_sampleable`
     * do not read `context_edges_`. They exist solely so `get_dag()`
     * (and `ai4bayescode_plot_dag` in R) can render the full *generative* DAG with
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
     * Order of the returned list: every node comes after each of its
     * parents that is also in the list, so refreshing the list in order
     * computes a node from the recomputed values of its parents. The rule
     * above admits a node whose parent is itself in the list but resolved
     * later (that parent holds a value in values_), so the walk alone does
     * not give this order; it is restored at the end of the function.
     *
     * @param changed  The set of nodes whose values were replaced.
     * @return  Ordered list of deterministic downstream nodes to refresh.
     */
    std::vector<std::string> predict_downstream_of(
            const std::unordered_set<std::string>& changed,
            const std::unordered_set<std::string>& not_predictable =
                std::unordered_set<std::string>()) const {
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
            // A key in the not-predictable cone holds a training value that no
            // longer lines up with the replaced inputs, so it is NOT a usable
            // parent no matter what is sitting in values_.
            if (not_predictable.count(k)) return false;
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
        //
        // Termination comes from `progress`, not from an iteration count:
        // every round that does not resolve at least one node breaks out, and
        // `reachable` only ever grows, so the loop runs at most once per node.
        // An earlier version also capped the rounds at (#seed-children)^2 + 1,
        // which is a width, not a depth -- a single seed child gave a cap of 2,
        // truncating any deterministic chain more than two hops deep. The
        // dropped nodes were then never recomputed, and predict_at returned
        // them (and anything sampled from them) at the TRAINING state with no
        // error. tests/test_predict_dag_depth.cpp pins the depth contract.
        while (!queue.empty()) {
            std::vector<std::string> next_queue;
            bool progress = false;

            for (const auto& node : queue) {
                if (reachable.count(node)) continue;  // already processed
                // Stochastic nodes are handled in Pass 2; exclude here.
                if (stochastic_refreshers_.count(node)) continue;
                // Downstream of a co-indexed input the caller withheld: not
                // predictable for this call, and never a parent for anything
                // further down either.
                if (not_predictable.count(node)) continue;

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

        // Refresh order. A parent that is in `result` counts as available
        // from the start (it holds a value in values_), so the walk can list
        // a child before that parent, with one replaced key as with several:
        //   - the parent resolves in a later round. With X -> Q -> P -> C,
        //     X -> C and the single replaced key X, round 1 resolves Q and C
        //     (parents P and X) and round 2 resolves P;
        //   - the parent resolves in the same round after the child, depending
        //     on the iteration order of `changed`. For the per-draw replaced
        //     set {Intercept, log_sds, log_sigma, z, Bsp} of
        //     examples/BSplineRegression.cpp, round 1 can list f (parents s,
        //     Bsp) before s (parents log_sds, z).
        // composite_block::predict_at refreshes `result` in order, so the
        // child was computed from the parent's value at the current state (C
        // from the P of the current state; f from the s of the current state
        // instead of the replaced log_sds and z). The loop below reorders
        // `result` so that every node follows each of its parents in `result`.
        // At each position it places the earliest node of `result` whose
        // parents in `result` are already placed, so a list that satisfies
        // the condition is returned unchanged. The set of nodes does not
        // change.
        if (result.size() > 1) {
            const std::size_t n = result.size();
            std::unordered_map<std::string, std::size_t> index;
            for (std::size_t i = 0; i < n; ++i) index.emplace(result[i], i);
            // parent_idx[i]: positions in `result` of the parents of result[i].
            std::vector<std::vector<std::size_t>> parent_idx(n);
            for (std::size_t i = 0; i < n; ++i) {
                auto pit = parents.find(result[i]);
                if (pit == parents.end()) continue;
                for (const auto& p : pit->second) {
                    auto it = index.find(p);
                    if (it != index.end() && it->second != i)
                        parent_idx[i].push_back(it->second);
                }
            }
            std::vector<char> placed(n, 0);
            std::vector<std::string> ordered;
            ordered.reserve(n);
            while (ordered.size() < n) {
                std::size_t pick = n;
                for (std::size_t i = 0; i < n && pick == n; ++i) {
                    if (placed[i]) continue;
                    bool ready = true;
                    for (std::size_t j : parent_idx[i]) {
                        if (!placed[j]) { ready = false; break; }
                    }
                    if (ready) pick = i;
                }
                if (pick == n) {
                    // Only a cycle among the predict edges leaves no node
                    // placeable; the remaining nodes keep their order.
                    for (std::size_t i = 0; i < n; ++i)
                        if (!placed[i]) ordered.push_back(result[i]);
                    break;
                }
                placed[pick] = 1;
                ordered.push_back(result[pick]);
            }
            result = std::move(ordered);
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
            const std::unordered_set<std::string>& changed_after_pass1,
            const std::unordered_set<std::string>& not_predictable =
                std::unordered_set<std::string>()) const {
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
            // Downstream of a co-indexed input the caller withheld: sampling it
            // would draw from a mixture of new and training data.
            if (not_predictable.count(key)) continue;
            auto pit = parents.find(key);
            bool all_avail = true;
            if (pit != parents.end()) {
                for (const auto& p : pit->second) {
                    bool p_avail = !not_predictable.count(p) &&
                                   (changed_after_pass1.count(p) ||
                                    values_.count(p));
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
    /// edges. Consumed by get_dag() / ai4bayescode_plot_dag only — never by the
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
     * Provided for inspection and for callers that want the transitive
     * closure. The Gibbs sweep does NOT use it: composite_block::step()
     * calls refresh_derived_for(), which refreshes exactly the keys a block
     * declares and does not descend further.
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
    // Views made by context_view_for() and context_view_or_copy_for(), per
    // block. Copying or moving a shared_data_t does not copy or move them:
    // the target has none and a moved-from object is emptied, so every view
    // refers to values_ of the object that holds it. A view map is only
    // swapped or cleared, never assigned to: assigning to a view would write
    // into values_. obj points to elements of values_; no element of values_
    // is erased while views_ has entries (keys are only added to it, and
    // views_ is cleared before it is assigned), so these pointers remain
    // valid.
    struct context_views_t {
        struct entry {
            std::vector<std::string>      keys;    // dependency list of the views
            std::uint64_t                 version = 0;  // see same_list_()
            std::vector<arma::vec*>       obj;     // the stored value of each key
            std::vector<std::uintptr_t>   addr;    // address of each value's storage
            std::vector<arma::uword>      n_elem;  // length of each value
            std::size_t                   bytes = 0;  // size of the values in ctx
            block_context                 ctx;     // the views
            bool                          valid = false;
            entry() = default;
            entry(const entry&) = delete;
            entry& operator=(const entry&) = delete;
        };
        static std::uintptr_t address(const arma::vec& v) {
            return reinterpret_cast<std::uintptr_t>(v.memptr());
        }
        std::unordered_map<std::string, entry> blocks;

        context_views_t() = default;
        context_views_t(const context_views_t&) {}
        context_views_t(context_views_t&& o) noexcept { o.blocks.clear(); }
        context_views_t& operator=(const context_views_t&) {
            blocks.clear();
            return *this;
        }
        context_views_t& operator=(context_views_t&& o) noexcept {
            blocks.clear();
            o.blocks.clear();
            return *this;
        }
    };

    // Whether entry e was made after the last declare_dependencies call; then
    // e.keys is the block's dependency list and e.obj points to the stored
    // values of its keys.
    bool same_list_(const context_views_t::entry& e) const {
        return e.valid && e.version == dependencies_version_;
    }

    // Makes entry e of block_name for the block's current dependency list:
    // the list, the pointer to each key's stored value, and the views.
    void make_views_(context_views_t::entry& e, const std::string& block_name) {
        e.valid = false;
        const auto dit = dependencies_.find(block_name);
        std::vector<std::string> keys;
        if (dit != dependencies_.end()) keys = dit->second;
        std::vector<arma::vec*> obj(keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i) {
            const auto vit = values_.find(keys[i]);
            if (vit == values_.end()) {
                throw std::out_of_range(
                    "shared_data_t::build_context_for: block '" + block_name
                    + "' depends on missing key '" + keys[i] + "'");
            }
            obj[i] = &vit->second;
        }
        e.keys.swap(keys);
        e.version = dependencies_version_;
        e.obj.swap(obj);
        e.addr.resize(e.keys.size());
        e.n_elem.resize(e.keys.size());
        remake_views_(e);
    }

    // Makes the views of entry e again from e.obj, with the current address
    // and length of each stored value, inserted in dependency order as in
    // build_context_for, so the iteration order is the same. The list and
    // the pointers are not changed.
    static void remake_views_(context_views_t::entry& e) {
        e.valid = false;
        block_context ctx;
        for (std::size_t i = 0; i < e.keys.size(); ++i) {
            arma::vec& v = *e.obj[i];
            ctx.emplace(std::piecewise_construct,
                        std::forward_as_tuple(e.keys[i]),
                        std::forward_as_tuple(v.memptr(), v.n_elem,
                                              /*copy_aux_mem=*/false,
                                              /*strict=*/true));
            e.addr[i] = context_views_t::address(v);
            e.n_elem[i] = v.n_elem;
        }
        std::size_t n = 0;
        for (const auto& kv : ctx) n += kv.second.n_elem;
        e.ctx.swap(ctx);
        e.bytes = n * sizeof(double);
        e.valid = true;
    }

    // The size of the current values of entry e, whose list is the block's
    // current one: the sum of n_elem * sizeof(double) over the keys, a key
    // listed twice counted once (its entries of e.obj are equal).
    static std::size_t current_bytes_(const context_views_t::entry& e) {
        std::size_t n = 0;
        for (std::size_t i = 0; i < e.obj.size(); ++i) {
            bool first = true;
            for (std::size_t j = 0; first && j < i; ++j) {
                first = e.obj[j] != e.obj[i];
            }
            if (first) n += e.obj[i]->n_elem;
        }
        return n * sizeof(double);
    }

    // Declared first: cleared before values_ is assigned, destroyed after it.
    context_views_t views_;
    std::unordered_map<std::string, arma::vec> values_;
    std::unordered_map<std::string, std::vector<std::string>> dependencies_;
    // Number of declare_dependencies calls; a views entry made at another
    // value may hold an older dependency list.
    std::uint64_t dependencies_version_ = 0;
    std::unordered_map<std::string, std::vector<std::string>> invalidates_;
    // Derived keys the model recomputes after set_current; see
    // declare_refreshed_after_set_current().
    std::unordered_set<std::string> refreshed_after_set_current_;
    std::unordered_map<std::string, refresher_fn> refreshers_;
    /// Registration order of refreshers_, so refresh_all() can prime a
    /// derived-reads-derived chain in the order the author declared it.
    std::vector<std::string> refresher_order_;
    std::unordered_map<std::string, stochastic_refresher_fn>
        stochastic_refreshers_;
    std::unordered_set<std::string> data_input_keys_;
    // Data inputs sharing one observation index; see
    // declare_data_input_group().
    std::vector<std::vector<std::string>> data_input_groups_;
    std::unordered_map<std::string, std::vector<std::string>> predict_edges_;
    // VIZ-ONLY. Never read by predict_downstream_of /
    // predict_stochastic_sampleable. See declare_context_edges().
    std::unordered_map<std::string, std::vector<std::string>> context_edges_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_SHARED_DATA_HPP

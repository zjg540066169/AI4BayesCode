/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  score_cache.hpp -- Friedman-Koller 2003 §4.2 family-score cache:
 *                     candidate parents top-C + family cache top-F +
 *                     γ-pruning. Tier C kernel backing order_mcmc_block.
 *
 *  REFERENCE
 *  =========
 *  Friedman, N. & Koller, D. (2003). Machine Learning 50:95-125. §4.2.
 *
 *  THREE-TIER HEURISTIC
 *  ====================
 *  (1) Candidate parents top-C (FK §4.2 ¶2)
 *      For each node X_i, pre-screen the C nodes with the highest
 *      single-edge score score(X_i, {X_j} | D). MCMC restricts
 *      subsequent parent sets to subsets of this fixed C-pool.
 *
 *  (2) Family caching top-F (FK §4.2 ¶3)
 *      Per node, pre-enumerate all families of size <= k drawn from the
 *      C-pool. Compute score, sort, keep the F highest-scoring.
 *
 *  (3) γ-pruning (FK §4.2 ¶3)
 *      Drop families whose score is more than γ below the best family
 *      score in the per-node cache. Approximation error <= 2^(-γ).
 *
 *  PUBLIC API
 *  ==========
 *      score_cache cache(scorer, cfg);
 *
 *      // Compute order-marginal log-score for node i at given order:
 *      double log_p = cache.order_node_score(order, i);
 *
 *      // Sample a parent set U for node i given the order, proportional
 *      // to score(X_i, U | D) over candidate families consistent with ≺:
 *      uint64_t U = cache.sample_parent_set(rng, order, i);
 *
 *      // Total order log-score:
 *      double total = cache.order_log_score(order);
 *
 *  Each top-F per-node family is a (score, parent_mask) pair, sorted
 *  descending by score. Order queries iterate this list, filter for
 *  ≺-consistency (every parent in U must precede i in ≺), and accumulate
 *  via log-sum-exp.
 *================================================================================*/

#ifndef AI4BAYESCODE_SCORE_CACHE_HPP
#define AI4BAYESCODE_SCORE_CACHE_HPP

#include "bde_scorer.hpp"
#include "i_scorer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/// Configuration for score_cache.
struct score_cache_config {
    /// Max parent set size (FK k). Default 5.
    std::size_t max_parents = 5;

    /// Top-C candidate parents per node (FK C). Default 20 (capped to
    /// n-1 internally if smaller).
    std::size_t candidate_top_C = 20;

    /// Top-F families cached per node (FK F). Default 4000.
    std::size_t family_top_F = 4000;

    /// γ-pruning threshold in nats (FK γ in bits ≈ γ_nats × log(2)).
    /// FK §5.2 uses γ_bits = 10, so γ_nats ≈ 6.93. Default 10 (in nats),
    /// matches the spec's per-Q8 default.
    double gamma_prune_nats = 10.0;
};

/// One cached family: parent set encoded as bitmask + its BDe log score.
struct cached_family {
    std::uint64_t parents_mask = 0;
    std::size_t   n_parents    = 0;   // popcount(parents_mask), used by FK
    double        log_score    = 0.0;
};

/**
 * @brief Family-score cache with FK §4.2 candidate top-C, family top-F,
 *        γ-prune. Single-threaded; not thread-safe: the const node-score and
 *        parent-set queries store their last results per node (keyed by the
 *        exact masks), so one object must not be queried from two threads.
 */
class score_cache {
public:
    /// Convenience ctor: wrap a bde_scorer by value (backward-compatible with
    /// the original by-value API). Delegates to the general i_scorer ctor.
    score_cache(bde_scorer scorer, score_cache_config cfg)
        : score_cache(std::make_unique<bde_scorer>(std::move(scorer)), cfg) {}

    /// General ctor: take ownership of any i_scorer implementation
    /// (bde_scorer, bge_scorer, ...).
    score_cache(std::unique_ptr<i_scorer> scorer, score_cache_config cfg)
        : scorer_(std::move(scorer)), cfg_(cfg)
    {
        n_ = scorer_->n();
        if (cfg_.max_parents > 64u) {
            throw std::invalid_argument(
                "score_cache: max_parents > 64 not supported");
        }

        candidate_parents_.assign(n_, {});
        cache_.assign(n_, {});
        max_log_score_.assign(n_, -std::numeric_limits<double>::infinity());

        // Phase 1: candidate parents top-C (per node).
        for (std::size_t i = 0; i < n_; ++i) {
            candidate_parents_[i] = scorer_->top_candidate_parents(
                i, cfg_.candidate_top_C);
        }

        // Phase 2: enumerate all candidate families up to k, compute
        // score, retain top-F per node.
        for (std::size_t i = 0; i < n_; ++i) {
            populate_cache_for_node_(i);
        }

        // Phase 3: γ-prune (drop families < best - γ).
        for (std::size_t i = 0; i < n_; ++i) {
            const double cutoff = max_log_score_[i] - cfg_.gamma_prune_nats;
            auto& v = cache_[i];
            v.erase(std::remove_if(v.begin(), v.end(),
                [cutoff](const cached_family& f) { return f.log_score < cutoff; }),
                v.end());
        }

        // Empty-parent scores, and the cached scores sorted by parent mask
        // for the family_score lookups of cached_scorer().
        empty_score_.resize(n_);
        lookup_.base_ = scorer_.get();
        lookup_.by_mask_.assign(n_, {});
        for (std::size_t i = 0; i < n_; ++i) {
            empty_score_[i] = scorer_->family_score(i, 0ULL);
            auto& t = lookup_.by_mask_[i];
            t.reserve(cache_[i].size());
            for (const auto& f : cache_[i]) t.emplace_back(f.parents_mask, f.log_score);
            std::sort(t.begin(), t.end(),
                      [](const std::pair<std::uint64_t, double>& a,
                         const std::pair<std::uint64_t, double>& b) {
                          return a.first < b.first;
                      });
        }
        order_score_memo_.assign(n_, score_memo{});
        part_score_memo_.assign(n_, score_memo{});
        order_draw_memo_.assign(n_, draw_table{});
        part_draw_memo_.assign(n_, draw_table{});
    }

    std::size_t n() const noexcept { return n_; }

    const std::vector<std::size_t>& candidate_parents(std::size_t i) const {
        return candidate_parents_[i];
    }
    const std::vector<cached_family>& cache_for(std::size_t i) const {
        return cache_[i];
    }

    /**
     * @brief For a given order (a permutation of 0..n-1, position p →
     *        node order[p]), compute log Σ_{U ∈ U_{i,≺}} score(X_i, U | D),
     *        restricted to the cached top-F families for node i.
     *
     * Implementation: iterate cache_[i]; keep families whose parents are
     *                 all in positions < p_i (predecessors of i in ≺);
     *                 log-sum-exp.
     */
    double order_node_score(const std::vector<std::size_t>& order,
                              std::size_t i) const {
        // Build predecessor bitmask: bit j set ⇔ j is a predecessor of i.
        const std::uint64_t pred_mask = predecessors_mask_(order, i);
        return memo_lookup_(order_score_memo_[i], pred_mask, 0ULL, [&] {
            // Iterate cache, filter by parents ⊆ predecessors.
            double m = -std::numeric_limits<double>::infinity();
            // First pass: find max for numerical stability.
            for (const auto& f : cache_[i]) {
                if ((f.parents_mask & ~pred_mask) == 0ULL) {
                    if (f.log_score > m) m = f.log_score;
                }
            }
            if (m == -std::numeric_limits<double>::infinity()) {
                // No candidate family (shouldn't normally happen since empty
                // parent set is always valid). Return the empty-parent score
                // as fallback.
                return empty_score_[i];
            }
            double sum = 0.0;
            for (const auto& f : cache_[i]) {
                if ((f.parents_mask & ~pred_mask) == 0ULL) {
                    sum += std::exp(f.log_score - m);
                }
            }
            return m + std::log(sum);
        });
    }

    /// Total order log-score: Σ_i order_node_score(order, i).
    double order_log_score(const std::vector<std::size_t>& order) const {
        double s = 0.0;
        for (std::size_t i = 0; i < n_; ++i)
            s += order_node_score(order, i);
        return s;
    }

    /**
     * @brief Sample a parent set U for node i given the order,
     *        proportional to exp(score) over ≺-consistent cached
     *        families. Returns the bitmask.
     */
    std::uint64_t sample_parent_set(std::mt19937_64& rng,
                                       const std::vector<std::size_t>& order,
                                       std::size_t i) const {
        const std::uint64_t pred_mask = predecessors_mask_(order, i);
        draw_table& d = order_draw_memo_[i];
        if (!d.used || d.a != pred_mask) {
            // Feasible families in cache order, w = exp(score - max),
            // normalised by Z, and the running sums of the normalised w.
            d.used = true; d.a = pred_mask; d.b = 0ULL;
            d.masks.clear(); d.cum.clear();
            double m = -std::numeric_limits<double>::infinity();
            for (const auto& f : cache_[i]) {
                if ((f.parents_mask & ~pred_mask) == 0ULL) {
                    d.masks.push_back(f.parents_mask);
                    d.cum.push_back(f.log_score);
                    if (f.log_score > m) m = f.log_score;
                }
            }
            double Z = 0.0;
            for (double& w : d.cum) { w = std::exp(w - m); Z += w; }
            double acc = 0.0;
            for (double& w : d.cum) { w /= Z; acc += w; w = acc; }
            d.Z = Z;
        }
        if (d.masks.empty()) return 0ULL;
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        const double u = U01(rng);
        return pick_(d, u);
    }

    /**
     * @brief Sample a DAG given the order, per Friedman-Koller Prop 3.1.
     *        Returns n bitmasks (parent set per node).
     */
    std::vector<std::uint64_t> sample_dag(
        std::mt19937_64& rng,
        const std::vector<std::size_t>& order) const {
        std::vector<std::uint64_t> dag(n_, 0ULL);
        for (std::size_t i = 0; i < n_; ++i) {
            dag[i] = sample_parent_set(rng, order, i);
        }
        return dag;
    }

    // ======================================================================
    //  Partition MCMC (Kuipers-Moffa 2017) support.
    //
    //  A node i in partition element t has permissible parent sets restricted
    //  to: (a) parents ⊆ `allowed_mask` (the nodes in elements STRICTLY to the
    //  right of t, i.e. later elements), AND (b) parents ∩ `required_mask` ≠ ∅
    //  where `required_mask` = the immediately-adjacent right element (t+1).
    //  For the last element there is no right neighbour: required_mask == 0,
    //  the node is a pure source, and its only permissible set is the empty
    //  parent set. These masks are computed by the caller from the partition
    //  state (a node's own + left elements are banned; elements to the right
    //  are allowed; the adjacent right element is required).
    // ======================================================================

    /// Kuipers-Moffa Eq. 3 per-node partition score:
    /// log Σ_{U ⊆ allowed, U ∩ required ≠ ∅} score(X_i, U | D), over the cached
    /// families. required_mask == 0 (last element) ⇒ empty-parent score only.
    double partition_node_score(std::size_t i,
                                std::uint64_t allowed_mask,
                                std::uint64_t required_mask) const {
        if (required_mask == 0ULL) {
            return empty_score_[i];   // last element: no parents
        }
        return memo_lookup_(part_score_memo_[i], allowed_mask, required_mask, [&] {
            double m = -std::numeric_limits<double>::infinity();
            for (const auto& f : cache_[i]) {
                if ((f.parents_mask & ~allowed_mask) == 0ULL &&
                    (f.parents_mask & required_mask) != 0ULL) {
                    if (f.log_score > m) m = f.log_score;
                }
            }
            if (m == -std::numeric_limits<double>::infinity()) {
                return -std::numeric_limits<double>::infinity();  // no permissible family
            }
            double sum = 0.0;
            for (const auto& f : cache_[i]) {
                if ((f.parents_mask & ~allowed_mask) == 0ULL &&
                    (f.parents_mask & required_mask) != 0ULL) {
                    sum += std::exp(f.log_score - m);
                }
            }
            return m + std::log(sum);
        });
    }

    /// Sample a parent set for node i under the partition constraint,
    /// proportional to exp(score) over permissible cached families.
    std::uint64_t partition_sample_parent_set(std::mt19937_64& rng,
                                              std::size_t i,
                                              std::uint64_t allowed_mask,
                                              std::uint64_t required_mask) const {
        if (required_mask == 0ULL) return 0ULL;   // last element: empty parents
        draw_table& d = part_draw_memo_[i];
        if (!d.used || d.a != allowed_mask || d.b != required_mask) {
            // Permissible families in cache order, w = exp(score - max), Z
            // and the running sums of w.
            d.used = true; d.a = allowed_mask; d.b = required_mask;
            d.masks.clear(); d.cum.clear();
            double m = -std::numeric_limits<double>::infinity();
            for (const auto& f : cache_[i]) {
                if ((f.parents_mask & ~allowed_mask) == 0ULL &&
                    (f.parents_mask & required_mask) != 0ULL) {
                    d.masks.push_back(f.parents_mask);
                    d.cum.push_back(f.log_score);
                    if (f.log_score > m) m = f.log_score;
                }
            }
            double Z = 0.0;
            for (double& w : d.cum) { w = std::exp(w - m); Z += w; }
            double acc = 0.0;
            for (double& w : d.cum) { acc += w; w = acc; }
            d.Z = Z;
        }
        if (d.masks.empty()) return 0ULL;
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        const double u = U01(rng) * d.Z;
        return pick_(d, u);
    }

    /// Get the scorer (for callers needing direct family_score access): the
    /// scorer passed to the constructor.
    const i_scorer& scorer() const noexcept { return *scorer_; }

    /// An i_scorer over the cached family scores: family_score of a cached
    /// family returns the cached value (the value scorer() returned when the
    /// cache was built), and family_score of any other mask returns
    /// scorer().family_score. The object is a member of this score_cache.
    const i_scorer& cached_scorer() const noexcept { return lookup_; }

    /// Diagnostic: number of cached families per node.
    std::vector<std::size_t> cache_sizes() const {
        std::vector<std::size_t> out(n_);
        for (std::size_t i = 0; i < n_; ++i) out[i] = cache_[i].size();
        return out;
    }

private:
    void populate_cache_for_node_(std::size_t i) {
        const auto& cands = candidate_parents_[i];
        const std::size_t C = cands.size();
        const std::size_t kmax = std::min(cfg_.max_parents, C);

        std::vector<cached_family> all;
        // Recursive enumeration of all subsets of size <= kmax from cands.
        // Use lexicographic combinations.
        std::vector<std::size_t> chosen;
        chosen.reserve(kmax);
        std::function<void(std::size_t)> recurse;
        recurse = [&](std::size_t start) {
            // Score current chosen set.
            std::uint64_t mask = 0ULL;
            for (auto j : chosen) mask |= (1ULL << cands[j]);
            const double sc = scorer_->family_score(i, mask);
            cached_family f;
            f.parents_mask = mask;
            f.n_parents    = chosen.size();
            f.log_score    = sc;
            all.push_back(f);

            if (chosen.size() >= kmax) return;
            for (std::size_t j = start; j < C; ++j) {
                chosen.push_back(j);
                recurse(j + 1);
                chosen.pop_back();
            }
        };
        recurse(0);

        // Sort descending by log_score.
        std::sort(all.begin(), all.end(),
                  [](const cached_family& a, const cached_family& b) {
                      return a.log_score > b.log_score;
                  });
        // Retain top-F.
        if (all.size() > cfg_.family_top_F) {
            all.resize(cfg_.family_top_F);
        }
        cache_[i] = std::move(all);
        max_log_score_[i] = cache_[i].empty()
                              ? -std::numeric_limits<double>::infinity()
                              : cache_[i].front().log_score;
    }

    /// Bitmask of predecessors of i in the given order.
    std::uint64_t predecessors_mask_(const std::vector<std::size_t>& order,
                                       std::size_t i) const {
        std::uint64_t mask = 0ULL;
        for (std::size_t p = 0; p < order.size(); ++p) {
            const std::size_t j = order[p];
            if (j == i) break;
            mask |= (1ULL << j);
        }
        return mask;
    }

    /// Last two (mask pair -> node score) results of one node; entry 0 is
    /// the most recent. The score depends only on the masks and cache_[i].
    struct score_memo {
        std::uint64_t a[2] = {0ULL, 0ULL};
        std::uint64_t b[2] = {0ULL, 0ULL};
        double        v[2] = {0.0, 0.0};
        int           size = 0;
    };

    template <class Compute>
    static double memo_lookup_(score_memo& memo, std::uint64_t a,
                               std::uint64_t b, Compute&& compute) {
        for (int k = 0; k < memo.size; ++k) {
            if (memo.a[k] == a && memo.b[k] == b) {
                const double v = memo.v[k];
                if (k == 1) {
                    std::swap(memo.a[0], memo.a[1]);
                    std::swap(memo.b[0], memo.b[1]);
                    std::swap(memo.v[0], memo.v[1]);
                }
                return v;
            }
        }
        const double v = compute();
        memo.a[1] = memo.a[0]; memo.b[1] = memo.b[0]; memo.v[1] = memo.v[0];
        memo.a[0] = a; memo.b[0] = b; memo.v[0] = v;
        if (memo.size < 2) ++memo.size;
        return v;
    }

    /// Feasible families of one node for the mask pair (a, b), in cache
    /// order, with the running sums of their weights and the weight total Z.
    struct draw_table {
        bool                       used = false;
        std::uint64_t              a = 0ULL;
        std::uint64_t              b = 0ULL;
        std::vector<std::uint64_t> masks;
        std::vector<double>        cum;
        double                     Z = 0.0;
    };

    /// The first family whose running sum is >= u, else the last family (as
    /// the linear scan; a NaN total makes every comparison false).
    static std::uint64_t pick_(const draw_table& d, double u) {
        if (!std::isnan(d.Z)) {
            const auto it = std::lower_bound(d.cum.begin(), d.cum.end(), u);
            if (it != d.cum.end())
                return d.masks[static_cast<std::size_t>(it - d.cum.begin())];
        }
        return d.masks.back();
    }

    /// i_scorer returned by cached_scorer(): family_score of a cached family
    /// comes from by_mask_, any other mask from the wrapped scorer.
    class family_lookup final : public i_scorer {
    public:
        std::size_t n() const noexcept override { return base_->n(); }
        double family_score(std::size_t i,
                            std::uint64_t parent_mask) const override {
            if (i < by_mask_.size()) {
                const auto& t = by_mask_[i];
                const auto it = std::lower_bound(t.begin(), t.end(), parent_mask,
                    [](const std::pair<std::uint64_t, double>& e, std::uint64_t m) {
                        return e.first < m;
                    });
                if (it != t.end() && it->first == parent_mask) return it->second;
            }
            return base_->family_score(i, parent_mask);
        }
        std::vector<std::size_t> top_candidate_parents(
            std::size_t i, std::size_t C) const override {
            return base_->top_candidate_parents(i, C);
        }

    private:
        friend class score_cache;
        const i_scorer* base_ = nullptr;   // the scorer owned by scorer_
        std::vector<std::vector<std::pair<std::uint64_t, double>>> by_mask_;
    };

private:
    std::unique_ptr<i_scorer>                        scorer_;
    score_cache_config                               cfg_;
    std::size_t                                      n_ = 0;
    std::vector<std::vector<std::size_t>>            candidate_parents_;  // n × C
    std::vector<std::vector<cached_family>>          cache_;              // n × F
    std::vector<double>                              max_log_score_;       // n
    std::vector<double>                              empty_score_;         // n
    family_lookup                                    lookup_;
    mutable std::vector<score_memo>                  order_score_memo_;    // n
    mutable std::vector<score_memo>                  part_score_memo_;     // n
    mutable std::vector<draw_table>                  order_draw_memo_;     // n
    mutable std::vector<draw_table>                  part_draw_memo_;      // n
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_SCORE_CACHE_HPP

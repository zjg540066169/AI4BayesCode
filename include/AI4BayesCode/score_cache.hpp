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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
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
 *        γ-prune. Single-threaded; not thread-safe.
 */
class score_cache {
public:
    score_cache(bde_scorer scorer, score_cache_config cfg)
        : scorer_(std::move(scorer)), cfg_(cfg)
    {
        n_ = scorer_.n();
        if (cfg_.max_parents > 64u) {
            throw std::invalid_argument(
                "score_cache: max_parents > 64 not supported");
        }

        candidate_parents_.assign(n_, {});
        cache_.assign(n_, {});
        max_log_score_.assign(n_, -std::numeric_limits<double>::infinity());

        // Phase 1: candidate parents top-C (per node).
        for (std::size_t i = 0; i < n_; ++i) {
            candidate_parents_[i] = scorer_.top_candidate_parents(
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
        std::uint64_t pred_mask = predecessors_mask_(order, i);

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
            // parent set is always valid). Return scorer's empty-parent
            // score as fallback.
            return scorer_.family_score(i, 0ULL);
        }
        double sum = 0.0;
        for (const auto& f : cache_[i]) {
            if ((f.parents_mask & ~pred_mask) == 0ULL) {
                sum += std::exp(f.log_score - m);
            }
        }
        return m + std::log(sum);
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
        // Gather feasible families.
        std::vector<const cached_family*> feasible;
        feasible.reserve(cache_[i].size());
        double m = -std::numeric_limits<double>::infinity();
        for (const auto& f : cache_[i]) {
            if ((f.parents_mask & ~pred_mask) == 0ULL) {
                feasible.push_back(&f);
                if (f.log_score > m) m = f.log_score;
            }
        }
        if (feasible.empty()) return 0ULL;
        // Convert to probabilities.
        std::vector<double> w(feasible.size());
        double Z = 0.0;
        for (std::size_t k = 0; k < feasible.size(); ++k) {
            w[k] = std::exp(feasible[k]->log_score - m);
            Z += w[k];
        }
        for (auto& v : w) v /= Z;
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        const double u = U01(rng);
        double acc = 0.0;
        for (std::size_t k = 0; k < feasible.size(); ++k) {
            acc += w[k];
            if (u <= acc) return feasible[k]->parents_mask;
        }
        return feasible.back()->parents_mask;
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

    /// Get the bde_scorer (for callers needing direct family_score access).
    const bde_scorer& scorer() const noexcept { return scorer_; }

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
            const double sc = scorer_.family_score(i, mask);
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

private:
    bde_scorer                                       scorer_;
    score_cache_config                               cfg_;
    std::size_t                                      n_ = 0;
    std::vector<std::vector<std::size_t>>            candidate_parents_;  // n × C
    std::vector<std::vector<cached_family>>          cache_;              // n × F
    std::vector<double>                              max_log_score_;       // n
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_SCORE_CACHE_HPP

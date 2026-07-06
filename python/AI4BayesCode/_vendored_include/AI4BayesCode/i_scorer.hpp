/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v3.0 or later
 *  (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  i_scorer.hpp -- abstract decomposable local-score interface for Bayesian-
 *                  network structure MCMC.
 *
 *  score_cache / order_mcmc_block are score-family-agnostic: they only need a
 *  decomposable per-node family score over parent sets. This narrow interface
 *  is that surface. Concrete implementations:
 *      - bde_scorer : discrete data, BDeu score (Heckerman-Geiger-Chickering 1995).
 *      - bge_scorer : continuous Gaussian data, BGe score (Geiger-Heckerman 2002).
 *
 *  family_score(i, U) and top_candidate_parents(i, C) are called only during
 *  score_cache construction (results are cached thereafter), so the virtual-
 *  dispatch cost is one-time and negligible.
 *================================================================================*/

#ifndef AI4BAYESCODE_I_SCORER_HPP
#define AI4BAYESCODE_I_SCORER_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace AI4BayesCode {

/**
 * @brief Abstract decomposable local-score interface for BN structure MCMC.
 *
 * A scorer provides a modular (per-node) family score over parent sets encoded
 * as 64-bit bitmasks (bit j set => node j is a parent). The score MUST be
 * decomposable so that order MCMC (Friedman-Koller) and partition MCMC
 * (Kuipers-Moffa) can marginalise over parent sets.
 */
class i_scorer {
public:
    virtual ~i_scorer() = default;

    /// Number of variables (nodes).
    virtual std::size_t n() const noexcept = 0;

    /// Log family score of node i given parent set U (bitmask). Includes any
    /// structural-prior term. Bit i is ignored (a node is not its own parent).
    virtual double family_score(std::size_t i,
                                std::uint64_t parent_mask) const = 0;

    /// The C highest single-edge-scoring candidate parents of node i, sorted
    /// descending (FK 2003 §4.2 top-C pre-screen). Length <= min(C, n-1).
    virtual std::vector<std::size_t> top_candidate_parents(
        std::size_t i, std::size_t C) const = 0;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_I_SCORER_HPP

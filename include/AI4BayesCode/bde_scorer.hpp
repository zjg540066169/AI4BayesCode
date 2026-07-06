/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  bde_scorer.hpp -- Bayesian Dirichlet (BDe / BDeu) family score for
 *                    Bayesian network structure learning. Tier C kernel
 *                    backing order_mcmc_block (Friedman-Koller 2003).
 *
 *  REFERENCE
 *  =========
 *  Heckerman, D., Geiger, D., Chickering, D.M. (1995).
 *      "Learning Bayesian networks: The combination of knowledge and
 *       statistical data." Machine Learning 20: 197-243.  Eq. 28.
 *  Friedman, N. & Koller, D. (2003). Machine Learning 50:95-125.
 *
 *  BDe FAMILY SCORE (Heckerman Eq. 28)
 *  ====================================
 *      log p(D | X_i, U) =
 *        sum_{j=1..q_i} [ lgamma(N'_ij)  - lgamma(N'_ij  + N_ij)
 *                       + sum_{k=1..r_i} ( lgamma(N'_ijk + N_ijk)
 *                                         - lgamma(N'_ijk) ) ]
 *
 *  where U = parent set of node i, q_i = prod_{j in U} r_j (# parent
 *  configurations), r_i = cardinality of i, N_ijk = data count of
 *  (X_i=k, parents=j), N_ij = sum_k N_ijk, and the prior pseudocounts
 *  are
 *
 *  BDeu (Buntine 1991, default):
 *      N'_ijk = alpha / (r_i * q_i),   N'_ij  = alpha / q_i
 *
 *  alpha is the equivalent sample size. Default alpha = 1.
 *
 *  STRUCTURE PRIOR (FK 2003 Eq. 2, per-family uniform — default)
 *  =============================================================
 *      log rho_{X_i}(U) = -log C(n-1, |U|)   (up to additive const)
 *
 *  The total family score returned by family_score() includes this
 *  structure prior contribution.
 *
 *  USAGE
 *  =====
 *      arma::imat data(N, n);                  // 0..(r_i-1) integer encoding
 *      arma::uvec cards(n);                     // r_i for each variable
 *      bde_scorer_config cfg;
 *      cfg.data = data;
 *      cfg.cardinalities = cards;
 *      cfg.alpha = 1.0;                         // BDeu (default)
 *      bde_scorer scorer(cfg);
 *      double s = scorer.family_score(i, parents_bitmask);
 *
 *  COMPLEXITY
 *  ==========
 *  family_score(i, U) is O(N) for the N count tally + O(q_i * r_i) for
 *  the lgamma sum. Total = O(N + q_i * r_i). For typical BN benchmarks
 *  (r_i in 2..5, |U| <= 5, q_i <= 5^5 = 3125), N dominates.
 *
 *  Validator (system_design.md §11.7) considers Block 3 (order_mcmc_block)
 *  as a specialised non-Gibbs block; bde_scorer is the supporting
 *  closed-form kernel and is unit-tested against the canonical
 *  2-node Heckerman example.
 *================================================================================*/

#ifndef AI4BAYESCODE_BDE_SCORER_HPP
#define AI4BAYESCODE_BDE_SCORER_HPP

#include "i_scorer.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef MCMC_USE_RCPP_ARMADILLO
# include <armadillo>
#else
# include <RcppArmadillo.h>
#endif

namespace AI4BayesCode {

/// Configuration for bde_scorer.
struct bde_scorer_config {
    /// N × n integer-encoded data, values in 0..(cardinalities[j]-1).
    arma::imat data;

    /// Cardinalities r_i for each variable, length n. Each must be >= 2.
    arma::uvec cardinalities;

    /// BDeu equivalent sample size (Buntine 1991 prior). Default 1.0.
    double alpha = 1.0;

    /// If true (default), include the FK 2003 Eq. 2 per-family uniform
    /// structure prior contribution log rho_{X_i}(U) = -log C(n-1, |U|).
    bool use_structure_prior = true;

    /// Hard cap on parent set size (matches max_parents in
    /// order_mcmc_block). Used only for early-rejection guards;
    /// family_score itself respects any |U| up to 64.
    std::size_t max_parents = 5;
};

/**
 * @brief BDe / BDeu family scorer for Bayesian network structure learning.
 *
 * Stateless after construction: data + cardinalities are fixed.
 * Thread-safe for read-only `family_score(i, U)` queries.
 *
 * Parent sets are encoded as 64-bit bitmasks: bit `j` set ⇔ node `j`
 * is a parent. n ≤ 64 is enforced at construction.
 */
class bde_scorer : public i_scorer {
public:
    explicit bde_scorer(bde_scorer_config cfg)
        : cfg_(std::move(cfg))
    {
        N_ = cfg_.data.n_rows;
        n_ = cfg_.data.n_cols;

        if (n_ == 0) {
            throw std::invalid_argument(
                "bde_scorer: data must have at least 1 column");
        }
        if (N_ == 0) {
            throw std::invalid_argument(
                "bde_scorer: data must have at least 1 row");
        }
        if (n_ > 64) {
            throw std::invalid_argument(
                "bde_scorer: n > 64 not supported (bitmask encoding)");
        }
        if (cfg_.cardinalities.n_elem != n_) {
            throw std::invalid_argument(
                "bde_scorer: cardinalities length must equal data ncols");
        }
        for (std::size_t i = 0; i < n_; ++i) {
            if (cfg_.cardinalities[i] < 2u) {
                throw std::invalid_argument(
                    "bde_scorer: every cardinality must be >= 2");
            }
        }
        if (!(cfg_.alpha > 0.0)) {
            throw std::invalid_argument(
                "bde_scorer: alpha must be > 0");
        }
        // Validate data range.
        for (std::size_t i = 0; i < N_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                const int v = cfg_.data(i, j);
                if (v < 0 || v >= static_cast<int>(cfg_.cardinalities[j])) {
                    throw std::invalid_argument(
                        "bde_scorer: data value out of range at row "
                        + std::to_string(i) + " col " + std::to_string(j)
                        + " (got " + std::to_string(v) + ", expected 0.."
                        + std::to_string(cfg_.cardinalities[j] - 1) + ")");
                }
            }
        }

        // Precompute log C(n-1, k) lookup for structure prior.
        log_binom_.assign(n_, 0.0);
        if (cfg_.use_structure_prior && n_ > 1) {
            // log C(n-1, k) = lgamma(n) - lgamma(k+1) - lgamma(n-k)
            const double lg_n = std::lgamma(static_cast<double>(n_));
            for (std::size_t k = 0; k <= n_ - 1; ++k) {
                if (k > n_ - 1) break;
                const double lg_k1     = std::lgamma(static_cast<double>(k + 1));
                const double lg_n_m_k  = std::lgamma(static_cast<double>(n_ - k));
                log_binom_[k] = lg_n - lg_k1 - lg_n_m_k;
            }
        }
    }

    /// Number of variables (columns of data).
    std::size_t n() const noexcept override { return n_; }
    /// Number of observations (rows of data).
    std::size_t N() const noexcept { return N_; }
    /// Variable cardinality.
    std::size_t cardinality(std::size_t j) const { return cfg_.cardinalities[j]; }
    /// BDeu alpha.
    double alpha() const noexcept { return cfg_.alpha; }

    /**
     * @brief Compute BDe family score: log p(D | X_i, U) + log rho(U).
     * @param i             Node index, 0..n-1.
     * @param parent_mask   64-bit bitmask; bit j set ⇔ j ∈ U. Bit i is
     *                      ignored (a node is not its own parent).
     * @return Log family score (natural log).
     */
    double family_score(std::size_t i, std::uint64_t parent_mask) const override {
        if (i >= n_) {
            throw std::invalid_argument(
                "bde_scorer::family_score: node index out of range");
        }
        // Strip bit i if set.
        parent_mask &= ~(1ULL << i);

        // Decode parent indices + product cardinalities (q_i).
        std::vector<std::size_t> parents;
        parents.reserve(8);
        std::size_t q_i = 1;
        for (std::size_t j = 0; j < n_; ++j) {
            if (parent_mask & (1ULL << j)) {
                parents.push_back(j);
                const std::size_t prev = q_i;
                q_i *= cfg_.cardinalities[j];
                if (q_i < prev) {
                    throw std::overflow_error(
                        "bde_scorer::family_score: q_i overflow");
                }
            }
        }
        const std::size_t r_i = cfg_.cardinalities[i];

        // Tally N_ijk counts. N_ijk[j*r_i + k] = count of (X_i = k,
        // parents=j) where j is the linear encoding of parent values
        // (low-order first, mixed radix by cardinalities[parents[0..]]).
        std::vector<std::uint32_t> counts(q_i * r_i, 0u);
        for (std::size_t row = 0; row < N_; ++row) {
            std::size_t j_idx = 0, mult = 1;
            for (std::size_t p = 0; p < parents.size(); ++p) {
                const std::size_t parent_val =
                    static_cast<std::size_t>(cfg_.data(row, parents[p]));
                j_idx += parent_val * mult;
                mult *= cfg_.cardinalities[parents[p]];
            }
            const std::size_t k = static_cast<std::size_t>(cfg_.data(row, i));
            ++counts[j_idx * r_i + k];
        }

        // BDeu pseudocounts.
        const double Np_ijk = cfg_.alpha
                              / (static_cast<double>(r_i)
                                  * static_cast<double>(q_i));
        const double Np_ij  = cfg_.alpha / static_cast<double>(q_i);
        const double lg_Np_ijk = std::lgamma(Np_ijk);
        const double lg_Np_ij  = std::lgamma(Np_ij);

        // BDe sum.
        double score = 0.0;
        for (std::size_t j = 0; j < q_i; ++j) {
            std::uint64_t N_ij = 0;
            for (std::size_t k = 0; k < r_i; ++k) {
                const std::uint32_t N_ijk = counts[j * r_i + k];
                N_ij += N_ijk;
                score += std::lgamma(Np_ijk + static_cast<double>(N_ijk))
                       - lg_Np_ijk;
            }
            score += lg_Np_ij
                   - std::lgamma(Np_ij + static_cast<double>(N_ij));
        }

        // Structure prior (FK Eq. 2): log rho(U) = -log C(n-1, |U|).
        if (cfg_.use_structure_prior) {
            score -= log_binom_[parents.size()];
        }

        return score;
    }

    /**
     * @brief Compute single-edge score score(X_i, {X_j} | D). Used for
     *        the FK §4.2 candidate-parent pre-screening.
     */
    double single_edge_score(std::size_t i, std::size_t j) const {
        if (i == j) return -std::numeric_limits<double>::infinity();
        return family_score(i, 1ULL << j);
    }

    /**
     * @brief Pre-screen candidate parents for node i: rank all
     *        single-edge scores and return the top-C indices.
     * @param i  Node index.
     * @param C  Number of candidates to return.
     * @return   Length-min(C, n-1) vector of node indices (sorted DESC
     *           by single-edge score).
     */
    std::vector<std::size_t> top_candidate_parents(std::size_t i,
                                                       std::size_t C) const override {
        std::vector<std::pair<double, std::size_t>> ranked;
        ranked.reserve(n_ - 1);
        for (std::size_t j = 0; j < n_; ++j) {
            if (j == i) continue;
            ranked.emplace_back(single_edge_score(i, j), j);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const std::pair<double, std::size_t>& a,
                     const std::pair<double, std::size_t>& b) {
                      return a.first > b.first;
                  });
        const std::size_t M = std::min(C, ranked.size());
        std::vector<std::size_t> out(M);
        for (std::size_t k = 0; k < M; ++k) out[k] = ranked[k].second;
        return out;
    }

private:
    bde_scorer_config   cfg_;
    std::size_t         n_ = 0;
    std::size_t         N_ = 0;
    std::vector<double> log_binom_;  // log C(n-1, k), length n
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_BDE_SCORER_HPP

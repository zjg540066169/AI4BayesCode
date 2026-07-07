/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v3.0 or later
 *  (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  bge_scorer.hpp -- BGe (Bayesian Gaussian equivalent) local score for
 *                    continuous-data Bayesian-network structure MCMC.
 *                    Sibling of bde_scorer.hpp; implements i_scorer so it is a
 *                    drop-in for score_cache / order_mcmc_block.
 *
 *  REFERENCE
 *  =========
 *  Geiger, D. & Heckerman, D. (2002). "Parameter priors for directed acyclic
 *  graphical models and the characterization of several probability
 *  distributions." Ann. Statist. 30:1412-1440. (The 1994 "Learning Gaussian
 *  networks" arXiv:1302.6808 subset-prior recipe was corrected by GH2002; the
 *  closed form below is the GH2002 score as implemented in the BiDAG R package,
 *  verified to agree with BiDAG::DAGcorescore to 1e-10.)
 *
 *  MODEL + PRIOR (Normal-Wishart, conjugate)
 *  =========================================
 *      x | m, W ~ N(m, W^{-1})            (W = precision)
 *      m | W    ~ N(mu0, (am * W)^{-1})   (am = prior mean ESS, "am")
 *      W        ~ Wishart_n(aw, T0)       (aw = Wishart d.o.f., "aw")
 *      T0 = t * I_n ,  t = am*(aw - n - 1)/(am + 1)   (isotropic; GH2002 eq 19-20)
 *  Likelihood equivalence (Markov-equivalent DAGs score equal) holds iff T0 is
 *  isotropic, aw > n+1, am > 0.  Defaults (BiDAG/bnlearn): am=1, aw=n+2, mu0=0.
 *
 *  LOCAL SCORE (Schur complement; never invert the full matrix)
 *  ===========================================================
 *  Posterior scatter, formed ONCE for the whole domain:
 *      T_N = T0 + S_N + (am*N/(am+N)) (mu0 - xbar)(mu0 - xbar)^T
 *      S_N = cov(data) * (N-1)  (sample scatter),  awpN = aw + N
 *  For node i with p parents (family {i} u Pa), with A = T_N[i,i],
 *  B = T_N[i,Pa], D = T_N[Pa,Pa], and awpNd2 = (aw + N - n + p + 1)/2:
 *      log p(D_i | Pa) = c(p+1) - awpNd2 * log(A - B^T D^{-1} B) - (1/2) logdet(D)
 *  where B^T D^{-1} B is via a triangular solve on chol(D), and
 *      c(l) = -(N/2) log(pi) + (1/2) log(am/(am+N))
 *             - lgamma(awp/2) + lgamma((awp+N)/2) + ((awp+l-1)/2) log(t),
 *      awp = aw - n + l.
 *  An optional Friedman-Koller structure-prior term -log C(n-1, p) is added when
 *  use_structure_prior = true (identical hook + semantics to bde_scorer).
 *================================================================================*/

#ifndef AI4BAYESCODE_BGE_SCORER_HPP
#define AI4BAYESCODE_BGE_SCORER_HPP

#include "i_scorer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef MCMC_USE_RCPP_ARMADILLO
#  include <armadillo>
#else
#  include <RcppArmadillo.h>
#endif

namespace AI4BayesCode {

/// Configuration for bge_scorer.
struct bge_scorer_config {
    /// Continuous data, N rows (observations) x n cols (variables/nodes).
    arma::mat data;

    /// Prior mean effective sample size (Geiger-Heckerman "am" / alpha_mu).
    /// Must be > 0. Default 1.
    double am = 1.0;

    /// Wishart degrees of freedom (Geiger-Heckerman "aw" / alpha_w). Must be
    /// > n + 1 (for a finite prior covariance + likelihood equivalence).
    /// 0.0 = sentinel "use the default n + am + 1".
    double aw = 0.0;

    /// Prior mean vector mu0 (length n). Empty = default zero vector.
    arma::vec mu0;

    /// Per-edge penalisation factor "edgepf" (BiDAG). 1.0 = no penalty. Enters
    /// the family-size constant as -l*log(edgepf). Kept for exact BiDAG parity.
    double edgepf = 1.0;

    /// FK 2003 Eq 2 structure prior: add -log C(n-1, |Pa_i|) per family so the
    /// induced DAG prior penalises high fan-in. Same hook/semantics as
    /// bde_scorer. Default false (uniform DAG prior).
    bool use_structure_prior = false;

    /// Edge-specific structural prior: n x n matrix of per-edge log-prior
    /// weights; edge_log_prior(i, j) is added to node i's family score when j is
    /// a parent of i. Same semantics as bde_scorer. Empty (default) = none.
    arma::mat edge_log_prior;
};

/**
 * @brief BGe (continuous Gaussian) decomposable local score, implementing the
 *        i_scorer interface. Thread-safe for read-only family_score() queries.
 *
 * Parent sets are 64-bit bitmasks (bit j set <=> node j is a parent);
 * n <= 64 is enforced at construction.
 */
class bge_scorer : public i_scorer {
public:
    explicit bge_scorer(bge_scorer_config cfg)
        : cfg_(std::move(cfg))
    {
        N_ = cfg_.data.n_rows;
        n_ = cfg_.data.n_cols;

        if (n_ == 0)
            throw std::invalid_argument(
                "bge_scorer: data must have at least 1 column");
        if (N_ < 2)
            throw std::invalid_argument(
                "bge_scorer: data must have at least 2 rows (need a scatter)");
        if (n_ > 64)
            throw std::invalid_argument(
                "bge_scorer: n > 64 not supported (bitmask encoding)");
        if (!(cfg_.am > 0.0))
            throw std::invalid_argument("bge_scorer: am must be > 0");

        const double nd = static_cast<double>(n_);
        aw_ = (cfg_.aw == 0.0) ? (nd + cfg_.am + 1.0) : cfg_.aw;
        if (!(aw_ > nd + 1.0))
            throw std::invalid_argument(
                "bge_scorer: aw must be > n + 1 (isotropic prior needs t > 0)");

        arma::vec mu0 = cfg_.mu0.is_empty()
                            ? arma::vec(n_, arma::fill::zeros)
                            : cfg_.mu0;
        if (mu0.n_elem != n_)
            throw std::invalid_argument(
                "bge_scorer: mu0 length must equal number of variables");
        if (!(cfg_.edgepf > 0.0))
            throw std::invalid_argument("bge_scorer: edgepf must be > 0");
        if (!cfg_.edge_log_prior.is_empty() &&
            (cfg_.edge_log_prior.n_rows != n_ || cfg_.edge_log_prior.n_cols != n_))
            throw std::invalid_argument(
                "bge_scorer: edge_log_prior must be empty or n x n");

        // ---- posterior scatter T_N = T0 + S_N + mean-shift ----
        const double Nd = static_cast<double>(N_);
        const arma::rowvec xbar = arma::mean(cfg_.data, 0);      // 1 x n
        const arma::mat    S_N  = arma::cov(cfg_.data) * (Nd - 1.0);  // scatter
        t_ = cfg_.am * (aw_ - nd - 1.0) / (cfg_.am + 1.0);      // T0 scale, > 0
        arma::vec diff = mu0 - xbar.t();                        // n x 1
        T_N_ = t_ * arma::eye(n_, n_) + S_N
             + (cfg_.am * Nd / (cfg_.am + Nd)) * (diff * diff.t());
        awpN_ = aw_ + Nd;

        // ---- family-size constant c(l), l = 1..n ----
        const double constscorefact =
            -(Nd / 2.0) * std::log(M_PI)
            + 0.5 * std::log(cfg_.am / (cfg_.am + Nd));
        const double log_t   = std::log(t_);
        const double log_epf = std::log(cfg_.edgepf);
        c_.assign(n_ + 1, 0.0);   // c_[l] for l = 1..n (c_[0] unused)
        for (std::size_t l = 1; l <= n_; ++l) {
            const double awp = aw_ - nd + static_cast<double>(l);
            c_[l] = constscorefact
                  - std::lgamma(awp / 2.0)
                  + std::lgamma((awp + Nd) / 2.0)
                  + ((awp + static_cast<double>(l) - 1.0) / 2.0) * log_t
                  - static_cast<double>(l) * log_epf;
        }

        // ---- FK structure-prior lookup log C(n-1, k) (same as bde_scorer) ----
        log_binom_.assign(n_, 0.0);
        if (cfg_.use_structure_prior && n_ > 1) {
            const double lg_n = std::lgamma(nd);   // lgamma(n) = log (n-1)!
            for (std::size_t k = 0; k <= n_ - 1; ++k) {
                const double lg_k1    = std::lgamma(static_cast<double>(k + 1));
                const double lg_n_m_k = std::lgamma(static_cast<double>(n_ - k));
                log_binom_[k] = lg_n - lg_k1 - lg_n_m_k;
            }
        }
    }

    // ---- i_scorer interface ------------------------------------------------

    std::size_t n() const noexcept override { return n_; }

    double family_score(std::size_t i,
                        std::uint64_t parent_mask) const override {
        if (i >= n_)
            throw std::invalid_argument(
                "bge_scorer::family_score: node index out of range");
        parent_mask &= ~(1ULL << i);        // a node is not its own parent

        std::vector<arma::uword> pa;
        pa.reserve(static_cast<std::size_t>(n_));
        for (std::size_t j = 0; j < n_; ++j)
            if (parent_mask & (1ULL << j)) pa.push_back(j);
        const std::size_t p = pa.size();

        const double awpNd2 =
            (awpN_ - static_cast<double>(n_) + static_cast<double>(p) + 1.0)
            / 2.0;
        const double A = T_N_(i, i);

        double score;
        if (p == 0) {
            score = c_[1] - awpNd2 * std::log(A);
        } else if (p == 1) {
            const double B = T_N_(i, pa[0]);
            const double D = T_N_(pa[0], pa[0]);
            score = c_[2] - awpNd2 * std::log(A - B * B / D) - 0.5 * std::log(D);
        } else {
            const arma::uvec idx(pa);
            const arma::mat  D = T_N_.submat(idx, idx);       // p x p
            const arma::vec  B = T_N_.submat(arma::uvec{static_cast<arma::uword>(i)},
                                             idx).t();        // p x 1
            arma::mat L;
            if (!arma::chol(L, D, "lower"))
                throw std::runtime_error(
                    "bge_scorer: Cholesky of parent scatter failed "
                    "(collinear parents?)");
            const double logdetD = 2.0 * arma::accu(arma::log(L.diag()));
            const arma::vec q = arma::solve(arma::trimatl(L), B);  // L q = B
            const double schur = A - arma::dot(q, q);              // A - B^T D^-1 B
            score = c_[p + 1] - awpNd2 * std::log(schur) - 0.5 * logdetD;
        }

        if (cfg_.use_structure_prior && n_ > 1)
            score -= log_binom_[p];          // FK Eq 2: 1 / C(n-1, |Pa|)
        if (!cfg_.edge_log_prior.is_empty())
            for (arma::uword j : pa) score += cfg_.edge_log_prior(i, j);

        return score;
    }

    std::vector<std::size_t> top_candidate_parents(
        std::size_t i, std::size_t C) const override {
        std::vector<std::pair<double, std::size_t>> ranked;
        ranked.reserve(n_ > 0 ? n_ - 1 : 0);
        for (std::size_t j = 0; j < n_; ++j) {
            if (j == i) continue;
            ranked.emplace_back(family_score(i, 1ULL << j), j);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        const std::size_t keep = std::min(C, ranked.size());
        std::vector<std::size_t> out;
        out.reserve(keep);
        for (std::size_t k = 0; k < keep; ++k) out.push_back(ranked[k].second);
        return out;
    }

    // ---- diagnostics accessors --------------------------------------------
    std::size_t N() const noexcept { return N_; }
    double am() const noexcept { return cfg_.am; }
    double aw() const noexcept { return aw_; }
    double t0_scale() const noexcept { return t_; }
    const arma::mat& posterior_scatter() const noexcept { return T_N_; }

private:
    bge_scorer_config   cfg_;
    std::size_t         N_ = 0;
    std::size_t         n_ = 0;
    double              aw_ = 0.0;
    double              awpN_ = 0.0;
    double              t_ = 0.0;
    arma::mat           T_N_;          // n x n posterior scatter
    std::vector<double> c_;            // family-size constant, c_[l], l=1..n
    std::vector<double> log_binom_;    // log C(n-1, k), FK structure prior
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_BGE_SCORER_HPP

/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  gmrf_precision_block.hpp -- Direct sampler for Gaussian Markov Random
 *                               Fields specified by a sparse precision Q
 *                               (Rue 2001).
 *
 *  TARGET DISTRIBUTION (Rue 2001 canonical form, eq. (4) / §3.1.2)
 *  ================================================================
 *      pi(x | ...) ∝ exp{ -1/2 x^T Q x + b^T x },   x ∈ R^n
 *
 *  i.e., x ~ N(Q^{-1} b, Q^{-1}) where Q is a sparse symmetric
 *  positive-(semi)definite n×n precision matrix.
 *
 *  ALGORITHM (Rue 2001 §2 + §3.1.2 + §3.1.3 simplified)
 *  ====================================================
 *  Per sweep:
 *    1. Build Q via user functor Q_fn(ctx).
 *    2. Optional ridge regularisation: Q ← Q + ridge_epsilon * I.
 *    3. Sparse Cholesky via Eigen SimplicialLLT + AMD reordering:
 *         P Q P^T = L L^T.
 *       The SYMBOLIC factorisation (sparsity-pattern analysis + AMD
 *       reordering) is computed ONCE on first step() and cached. The
 *       NUMERICAL factorisation is redone every step. Assumes the
 *       sparsity pattern of Q is fixed across steps (only numerical
 *       values vary — typical for hierarchical scale × pattern
 *       decompositions where Q = kappa · R with R fixed).
 *    4. If b_fn supplied: compute mu = Q^{-1} b via the cached
 *       factorisation (solver.solve(b) handles permutation internally).
 *    5. Sample z ~ N(0, I_n).
 *    6. Solve L^T y_perm = z by back-substitution in the permuted basis
 *       (Eigen: solver.matrixU().solve(z)).
 *    7. Apply inverse permutation: x_centered = P^T y_perm.
 *    8. Result: x = mu + x_centered.
 *    9. If sum_to_zero is set, project: x ← x - mean(x).
 *
 *  This is a DIRECT (Gibbs-style) sampler — each step produces an exact
 *  draw from the current full conditional, not a Markov-chain transition
 *  step. The "state" is whatever was drawn last sweep, and only the
 *  draws (not the chain dynamics) carry posterior information.
 *
 *  IGMRF / SUM-TO-ZERO HANDLING (Rue §3.1.3 simplified)
 *  ====================================================
 *  v0 supports a single sum-to-zero constraint via the post-hoc
 *  projection x ← x - mean(x) on a ridge-regularised Q (rank-correction
 *  by epsilon for numerical stability). This approximates Rue's exact
 *  kriging formula
 *      x̃ = x - Q^{-1} A^T (A Q^{-1} A^T)^{-1} (A x - b)
 *  for A = 1^T, b = 0 in the limit ridge_epsilon → 0. The standard
 *  approach used by spam / R-INLA in practice. Exact kriging for
 *  arbitrary A is deferred to v1.2.1.
 *
 *  TARGET-SHAPE CATEGORY (system_design.md §11.1)
 *  ==============================================
 *  Fixed-dimension continuous (multivariate Gaussian with sparse
 *  precision). Specialised efficient alternative to NUTS-on-x for
 *  high-dim Gaussian latents in hierarchical models (spatial smoothing,
 *  RW1 / RW2 splines, ICAR / BYM2, lattice GP approximations, ...).
 *
 *  WHAT THE USER SUPPLIES
 *  ======================
 *  See struct gmrf_precision_block_config below for the full list:
 *    - n                  dimension of x
 *    - Q_fn(ctx)          sparse precision builder, returns arma::sp_mat
 *                         (block converts to Eigen::SparseMatrix
 *                         internally each step)
 *    - b_fn(ctx)          OPTIONAL canonical "b" vector; if unset,
 *                         x ~ N(0, Q^{-1})
 *    - sum_to_zero        bool; if true, enforces sum(x) = 0 via post-
 *                         sampling projection and auto-sets a small
 *                         positive ridge_epsilon if not specified
 *    - ridge_epsilon      diagonal regularisation; default 0.0 for
 *                         proper Q; auto-set to 1e-8 if sum_to_zero is
 *                         true and the user left it at 0.0
 *    - initial_x          optional starting value (default: zero)
 *
 *  COMPLEXITY
 *  ==========
 *  Per step:
 *    - O(nnz)         building Q from Q_fn
 *    - O(n * b_w^2)   sparse Cholesky numerical factorisation, b_w =
 *                     bandwidth after AMD reordering (≈ O(√n) for
 *                     typical 2D lattice GMRFs)
 *    - O(n * b_w)     triangular solves for mean shift + back-sub
 *  Symbolic analysis (AMD + ordering) is amortised across the chain.
 *
 *  VALIDATOR (system_design.md §11.7; Checks #15/16/17)
 *  ===================================================
 *  - Check #15 (parity test): tests/test_gmrf_precision_block.cpp
 *      T0  diagonal Q sanity: empirical Var ≈ 1/κ
 *      T1  AR(1) n=5 tridiagonal: empirical Cov matches dense Q^{-1}
 *      T2  b ≠ 0 mean shift: empirical mean ≈ Q^{-1} b
 *      T3  1D RW IGMRF: sum-to-zero verified empirically
 *      T4  AR(1) n=50 R-hat across 4 chains (direct draws ≈ 1.000)
 *  - Check #16: inline JUSTIFICATION comment at every wrapper call site.
 *  - Check #17: only std::normal_distribution (primitive); no hand-
 *               written conjugate samplers.
 *
 *  ENGINE FAMILY
 *  =============
 *  engine_kind() = MCMC (direct conjugate draw). supports_readapt() =
 *  false (no tunable kernel-level metric).
 *================================================================================*/

#ifndef AI4BAYESCODE_GMRF_PRECISION_BLOCK_HPP
#define AI4BAYESCODE_GMRF_PRECISION_BLOCK_HPP

#include "block_sampler.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>

#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

// ============================================================================
//  arma::sp_mat → Eigen::SparseMatrix<double> conversion helper
// ============================================================================

inline Eigen::SparseMatrix<double>
arma_to_eigen_sparse(const arma::sp_mat& Q) {
    Eigen::SparseMatrix<double> E(static_cast<int>(Q.n_rows),
                                    static_cast<int>(Q.n_cols));
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(Q.n_nonzero);
    for (arma::sp_mat::const_iterator it = Q.begin();
         it != Q.end(); ++it) {
        triplets.emplace_back(static_cast<int>(it.row()),
                              static_cast<int>(it.col()),
                              *it);
    }
    E.setFromTriplets(triplets.begin(), triplets.end());
    return E;
}

// ============================================================================
//  Config
// ============================================================================

struct gmrf_precision_block_config {
    /// Unique block name; also the shared_data key under which the
    /// current sample x is published.
    std::string name = "x";

    /// Dimension of x. Must be > 0.
    std::size_t n = 0;

    /// Required: builder for the precision matrix Q. Called every step()
    /// with the current block_context. Must return a symmetric n×n
    /// positive-(semi)definite sparse matrix. ASSUMES sparsity pattern
    /// of Q is fixed across steps (numerical values may vary); the
    /// symbolic factorisation is computed once on first step and
    /// cached.
    std::function<arma::sp_mat(const block_context&)> Q_fn;

    /// Optional: builder for the canonical "b" vector (mean shift).
    /// Returns a length-n arma::vec. If unset (default), x ~ N(0, Q^{-1});
    /// if set, x ~ N(Q^{-1} b, Q^{-1}) per Rue 2001 §3.1.2.
    std::function<arma::vec(const block_context&)> b_fn;

    /// IGMRF flag: when true, post-sampling projection x ← x - mean(x)
    /// enforces a sum-to-zero constraint; and ridge_epsilon is bumped to
    /// 1e-8 (if left at 0.0) for numerical stability under rank-deficient
    /// Q. See Rue 2001 §3.1.3 (simplified single-constraint case).
    bool sum_to_zero = false;

    /// Diagonal regularisation added to Q each step before
    /// factorisation. Default 0.0 (no regularisation, for strictly PD Q).
    /// When sum_to_zero is true and ridge_epsilon is 0.0, the
    /// constructor auto-bumps to 1e-8. User may set explicitly for
    /// proper Q with near-singular numerics (rare).
    double ridge_epsilon = 0.0;

    /// Optional initial sample (length n). If empty (default), x is
    /// initialised to the zero vector.
    arma::vec initial_x;
};

// ============================================================================
//  Block
// ============================================================================

class gmrf_precision_block : public block_sampler {
public:
    explicit gmrf_precision_block(gmrf_precision_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.n == 0) {
            throw std::invalid_argument(
                "gmrf_precision_block: n must be > 0");
        }
        if (!cfg_.Q_fn) {
            throw std::invalid_argument(
                "gmrf_precision_block '" + cfg_.name +
                "': Q_fn is required");
        }
        if (cfg_.ridge_epsilon < 0.0) {
            throw std::invalid_argument(
                "gmrf_precision_block '" + cfg_.name +
                "': ridge_epsilon must be >= 0");
        }
        if (cfg_.sum_to_zero && cfg_.ridge_epsilon == 0.0) {
            cfg_.ridge_epsilon = 1e-8;
        }

        x_.set_size(cfg_.n);
        if (cfg_.initial_x.n_elem == cfg_.n) {
            x_ = cfg_.initial_x;
        } else if (cfg_.initial_x.n_elem == 0) {
            x_.zeros();
        } else {
            throw std::invalid_argument(
                "gmrf_precision_block '" + cfg_.name +
                "': initial_x length must be n or 0");
        }
        symbolic_analyzed_ = false;
    }

    // ---- block_sampler interface ------------------------------------------

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // (1) Build Q from current context.
        arma::sp_mat Q_arma = cfg_.Q_fn(context_);
        if (Q_arma.n_rows != cfg_.n || Q_arma.n_cols != cfg_.n) {
            throw std::runtime_error(
                "gmrf_precision_block '" + cfg_.name +
                "': Q_fn returned matrix of wrong dimension (got " +
                std::to_string(Q_arma.n_rows) + "x" +
                std::to_string(Q_arma.n_cols) + ", expected " +
                std::to_string(cfg_.n) + "x" + std::to_string(cfg_.n) +
                ")");
        }

        // (2) Optional diagonal ridge.
        if (cfg_.ridge_epsilon > 0.0) {
            Q_arma += cfg_.ridge_epsilon *
                      arma::speye<arma::sp_mat>(cfg_.n, cfg_.n);
        }

        // (3) Convert to Eigen sparse + factorise.
        Eigen::SparseMatrix<double> Q_eigen = arma_to_eigen_sparse(Q_arma);

        if (!symbolic_analyzed_) {
            solver_.analyzePattern(Q_eigen);
            if (solver_.info() != Eigen::Success) {
                throw std::runtime_error(
                    "gmrf_precision_block '" + cfg_.name +
                    "': symbolic analysis (analyzePattern) failed");
            }
            symbolic_analyzed_ = true;
        }
        solver_.factorize(Q_eigen);
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error(
                "gmrf_precision_block '" + cfg_.name +
                "': sparse Cholesky factorisation failed. Q may not be "
                "positive-definite. If this is an IGMRF, set "
                "sum_to_zero=true and ridge_epsilon>0 (e.g. 1e-8) to "
                "regularise.");
        }

        // (4) Mean shift mu = Q^{-1} b, if b_fn provided.
        Eigen::VectorXd mu(static_cast<Eigen::Index>(cfg_.n));
        mu.setZero();
        if (cfg_.b_fn) {
            arma::vec b_arma = cfg_.b_fn(context_);
            if (b_arma.n_elem != cfg_.n) {
                throw std::runtime_error(
                    "gmrf_precision_block '" + cfg_.name +
                    "': b_fn returned vector of wrong length (got " +
                    std::to_string(b_arma.n_elem) + ", expected " +
                    std::to_string(cfg_.n) + ")");
            }
            Eigen::VectorXd b_eigen(static_cast<Eigen::Index>(cfg_.n));
            for (std::size_t i = 0; i < cfg_.n; ++i) b_eigen[i] = b_arma[i];
            mu = solver_.solve(b_eigen);
        }

        // (5) z ~ N(0, I_n).
        std::normal_distribution<double> nd(0.0, 1.0);
        Eigen::VectorXd z(static_cast<Eigen::Index>(cfg_.n));
        for (std::size_t i = 0; i < cfg_.n; ++i) z[i] = nd(rng);

        // (6) Solve L^T y_perm = z (back-substitution) in permuted basis.
        Eigen::VectorXd y_perm = solver_.matrixU().solve(z);

        // (7) Apply inverse permutation: x_centered = P^T y_perm.
        Eigen::VectorXd x_centered = solver_.permutationPinv() * y_perm;

        // (8) Final result.
        Eigen::VectorXd result = mu + x_centered;

        // (9) Optional sum-to-zero projection.
        if (cfg_.sum_to_zero) {
            const double m = result.mean();
            for (Eigen::Index i = 0; i < result.size(); ++i) {
                result[i] -= m;
            }
        }

        // Copy back to arma::vec.
        for (std::size_t i = 0; i < cfg_.n; ++i) x_[i] = result[i];

        if (keep_history_) history_buf_.push_back(x_);
    }

    const arma::vec& current() const override { return x_; }

    void set_current(const arma::vec& x_new) override {
        if (x_new.n_elem != cfg_.n) {
            throw std::invalid_argument(
                "gmrf_precision_block '" + cfg_.name +
                "': set_current length must equal n");
        }
        x_ = x_new;
    }

    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim() const noexcept override { return cfg_.n; }

    state_map current_named_outputs() const override {
        return { { cfg_.name, x_ } };
    }

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_buf_, x_);
    }
    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }
    void clear_history() override { history_buf_.clear(); }

    // engine_kind() defaults to MCMC ✓ (direct conjugate draw is still
    // engine_kind_t::MCMC for hybrid-composite dispatch purposes —
    // shared_data writes the fresh draw, no q-mean / q-sample
    // distinction needed).
    // supports_readapt() defaults to false ✓ (no tunable kernel metric).

private:
    gmrf_precision_block_config cfg_;
    arma::vec                   x_;
    block_context               context_;
    std::vector<arma::vec>      history_buf_;

    // Cached Eigen sparse Cholesky solver.
    // L L^T = P Q P^T where P is the AMD reordering permutation.
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>,
                          Eigen::Lower,
                          Eigen::AMDOrdering<int>> solver_;
    bool symbolic_analyzed_ = false;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_GMRF_PRECISION_BLOCK_HPP

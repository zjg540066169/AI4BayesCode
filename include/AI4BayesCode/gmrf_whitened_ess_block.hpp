// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
// ============================================================================
//  gmrf_whitened_ess_block — Elliptical Slice Sampling for GMRF latents
//                            with non-Gaussian observation likelihood.
//
//  Targets:
//
//      p(x | rest) ∝ exp(-0.5 x^T Q x + b^T x) · L(y | x, ctx)
//
//  where Q is a sparse PSD precision matrix (possibly rank-deficient with
//  optional sum-to-zero constraint, e.g. ICAR), and L(y | x, ctx) is an
//  ARBITRARY (typically non-Gaussian) observation likelihood. This is the
//  natural complement to `gmrf_precision_block` (which assumes the prior
//  IS the full conditional, i.e., Gaussian observation):
//
//      `gmrf_precision_block`            — direct conjugate draw, Gaussian
//                                         conditional (or no observation).
//      `gmrf_whitened_ess_block` (this)  — Elliptical Slice Sampling on
//                                         the implicit GMRF prior, allows
//                                         any non-Gaussian likelihood
//                                         (Poisson, Bernoulli, Student-t…).
//
//  Algorithm — Murray 2010 ESS applied to GMRF prior:
//
//      Per step:
//        1. Build sparse Q = Q_fn(ctx); refactorize sparse Cholesky.
//        2. Sample nu ~ N(0, Q^{-1}) using the same Rue-2001 backsolve as
//           gmrf_precision_block (re-use cached symbolic factorization).
//        3. Compute log_lik_cur = log_lik(x_cur, ctx).
//        4. Slice threshold: log_y = log_lik_cur + log(Uniform(0,1)).
//        5. Sample θ ~ Uniform(0, 2π), bracket [θ-2π, θ].
//        6. Shrink loop:
//              x_prime = x_cur · cos(θ) + nu · sin(θ)
//              if log_lik(x_prime, ctx) > log_y: accept, x_cur ← x_prime.
//              else: shrink bracket containing 0, resample θ from bracket.
//
//  Sum-to-zero invariant: if `cfg.sum_to_zero = true`, both x_cur and the
//  prior sample nu are projected onto Σ x_i = 0. The ESS linear
//  combination x_cur · cos(θ) + nu · sin(θ) is then also zero-mean (sum
//  of two zero-mean vectors), so the constraint is preserved without an
//  extra projection inside the shrink loop.
//
//  Reference for ESS on GMRF: Filippone & Girolami 2014, "Pseudo-marginal
//  Bayesian inference for Gaussian processes". Reference for sparse Q
//  sampling: Rue 2001, "Fast sampling of Gaussian Markov random fields"
//  (we use the same SimplicialLLT + AMD ordering + permuted backsolve as
//  gmrf_precision_block).
//
//  Use case (paper-context BYM2 / Poisson-ICAR):
//      phi ~ ICAR (sparse Q, sum-to-zero)  + y ~ Poisson(exp(α + phi))
//
//      Composite:
//        gmrf_whitened_ess_block(name="phi", Q_fn=κ*R, sum_to_zero=true,
//                                log_lik=poisson(y | exp(α + phi)))
//      + nuts_block("alpha", real)
//      + nuts_block("log_kappa", positive)
//
//      The ESS block samples phi efficiently per Murray 2010 (constant
//      acceptance rate independent of likelihood scaling); the NUTS
//      blocks handle the smooth hyperparameters.
//
//  Stateful semantics — preserved exactly as gmrf_precision_block:
//      • set_context() copies the ctx for use in step().
//      • step(rng) advances x_ by one ESS sweep.
//      • current() / current_named_outputs() expose the latest x_.
//      • set_current(x) overrides x_ (for checkpointing or initialisation).
//      • keep_history mode appends a copy of x_ each step.
//
//  SHIPPED 2026-06-03 as v1.2 architectural addition. Companion to
//  `gmrf_precision_block` (Gaussian-conditional) for the non-Gaussian
//  case.
// ============================================================================

#ifndef AI4BAYESCODE_GMRF_WHITENED_ESS_BLOCK_HPP
#define AI4BAYESCODE_GMRF_WHITENED_ESS_BLOCK_HPP

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace AI4BayesCode {

namespace detail_gmrf_whitened_ess {

// arma::sp_mat → Eigen::SparseMatrix<double>. Same impl as the helper in
// gmrf_precision_block.hpp; placed in a separate detail namespace so the
// two headers can be co-included in a single TU without ODR concerns.
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

}  // namespace detail_gmrf_whitened_ess

// ============================================================================
//  Config
// ============================================================================

struct gmrf_whitened_ess_block_config {
    /// Unique block name; also the shared_data key under which the
    /// current x is published.
    std::string name = "x";

    /// Dimension of the GMRF latent x.
    std::size_t n = 0;

    /// Required: builder for the precision matrix Q. Called every step()
    /// with the current block_context. Must return a symmetric n×n
    /// positive-(semi)definite sparse matrix. ASSUMES sparsity pattern
    /// of Q is fixed across steps (numerical values may vary).
    std::function<arma::sp_mat(const block_context&)> Q_fn;

    /// Required: user-supplied log-likelihood on the natural-scale latent.
    ///   log_lik(x, ctx) returns log p(y | x, ctx).
    /// The block calls this many times per step (inside the slice-shrink
    /// loop); make it cheap. Non-finite return → treated as -∞ (proposal
    /// rejected). The Poisson / Bernoulli / Student-t etc. likelihood
    /// goes here.
    std::function<double(const arma::vec& x, const block_context&)> log_lik;

    /// IGMRF flag: when true, the prior sample nu and the current x are
    /// both projected onto Σ x_i = 0; ridge_epsilon is auto-bumped to
    /// 1e-8 if left at 0.0. See Rue 2001 §3.1.3 (simplified
    /// single-constraint case).
    bool sum_to_zero = false;

    /// Diagonal regularisation added to Q each step before factorisation.
    /// Default 0.0 (no regularisation, for strictly PD Q). When
    /// sum_to_zero is true and ridge_epsilon is 0.0, the constructor
    /// auto-bumps to 1e-8 to make rank-deficient ICAR factorisable.
    double ridge_epsilon = 0.0;

    /// Optional initial sample (length n). If empty (default), x is
    /// initialised to the zero vector (zero-mean iff sum_to_zero).
    arma::vec initial_x;

    /// Safety cap on the slice-shrink inner loop. ESS is guaranteed to
    /// accept eventually under the original (current x); the cap is a
    /// pathological-likelihood safeguard.
    std::size_t max_shrink_iter = 100;
};

// ============================================================================
//  Block
// ============================================================================

class gmrf_whitened_ess_block : public block_sampler {
public:
    explicit gmrf_whitened_ess_block(gmrf_whitened_ess_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.n == 0) {
            throw std::invalid_argument(
                "gmrf_whitened_ess_block: n must be > 0");
        }
        if (!cfg_.Q_fn) {
            throw std::invalid_argument(
                "gmrf_whitened_ess_block '" + cfg_.name +
                "': Q_fn is required");
        }
        if (!cfg_.log_lik) {
            throw std::invalid_argument(
                "gmrf_whitened_ess_block '" + cfg_.name +
                "': log_lik is required");
        }
        if (cfg_.ridge_epsilon < 0.0) {
            throw std::invalid_argument(
                "gmrf_whitened_ess_block '" + cfg_.name +
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
                "gmrf_whitened_ess_block '" + cfg_.name +
                "': initial_x length must be n or 0");
        }
        if (cfg_.sum_to_zero) {
            const double m = arma::mean(x_);
            x_ -= m;
        }
        symbolic_analyzed_ = false;
    }

    // ---- block_sampler interface --------------------------------------

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // (1) Build Q from context, optional ridge.
        arma::sp_mat Q_arma = cfg_.Q_fn(context_);
        if (Q_arma.n_rows != cfg_.n || Q_arma.n_cols != cfg_.n) {
            throw std::runtime_error(
                "gmrf_whitened_ess_block '" + cfg_.name +
                "': Q_fn returned matrix of wrong dimension (got " +
                std::to_string(Q_arma.n_rows) + "x" +
                std::to_string(Q_arma.n_cols) + ", expected " +
                std::to_string(cfg_.n) + "x" + std::to_string(cfg_.n) +
                ")");
        }
        if (cfg_.ridge_epsilon > 0.0) {
            Q_arma += cfg_.ridge_epsilon *
                      arma::speye<arma::sp_mat>(cfg_.n, cfg_.n);
        }

        // (2) Sparse Cholesky factorise (cached symbolic).
        Eigen::SparseMatrix<double> Q_eigen =
            detail_gmrf_whitened_ess::arma_to_eigen_sparse(Q_arma);

        if (!symbolic_analyzed_) {
            solver_.analyzePattern(Q_eigen);
            if (solver_.info() != Eigen::Success) {
                throw std::runtime_error(
                    "gmrf_whitened_ess_block '" + cfg_.name +
                    "': symbolic analysis (analyzePattern) failed");
            }
            symbolic_analyzed_ = true;
        }
        solver_.factorize(Q_eigen);
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error(
                "gmrf_whitened_ess_block '" + cfg_.name +
                "': sparse Cholesky factorisation failed. Q may not be "
                "positive-definite. If this is an IGMRF, set "
                "sum_to_zero=true (auto-bumps ridge_epsilon to 1e-8).");
        }

        // (3) Sample nu ~ N(0, Q^{-1}) using Rue 2001 backsolve.
        std::normal_distribution<double>       std_norm(0.0, 1.0);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        arma::vec nu = sample_prior_(rng, std_norm);

        // (4) ESS: log-slice threshold from current state.
        double log_lik_cur = cfg_.log_lik(x_, context_);
        if (!std::isfinite(log_lik_cur)) {
            // Pathological: current state has -∞ likelihood. Cannot
            // construct a meaningful slice threshold. Record-and-skip
            // (chain unchanged this sweep); next sweep may recover after
            // upstream parameter updates.
            if (keep_history_) history_buf_.push_back(x_);
            return;
        }
        double log_y = log_lik_cur + std::log(unif(rng));

        // (5) Initial bracket + shrink loop.
        const double TWO_PI = 2.0 * M_PI;
        double theta     = unif(rng) * TWO_PI;
        double theta_min = theta - TWO_PI;
        double theta_max = theta;

        arma::vec x_prime(cfg_.n);
        bool accepted = false;
        for (std::size_t iter = 0; iter < cfg_.max_shrink_iter; ++iter) {
            const double c = std::cos(theta);
            const double s = std::sin(theta);
            for (std::size_t i = 0; i < cfg_.n; ++i) {
                x_prime[i] = x_[i] * c + nu[i] * s;
            }
            // Sum-to-zero is automatically preserved: if x_ and nu have
            // mean 0, their linear combination has mean 0. We do NOT
            // re-project here (would waste cycles and break ESS
            // detailed-balance — the linear combination operation on a
            // (n-1)-dim constraint surface is itself the right step).

            const double log_lik_new = cfg_.log_lik(x_prime, context_);
            if (std::isfinite(log_lik_new) && log_lik_new > log_y) {
                x_ = x_prime;
                accepted = true;
                break;
            }

            // Shrink bracket: keep the side that contains 0 (the
            // identity rotation). theta > 0 → upper half; theta < 0 →
            // lower half.
            if (theta < 0.0) {
                theta_min = theta;
            } else {
                theta_max = theta;
            }
            theta = unif(rng) * (theta_max - theta_min) + theta_min;
        }
        // If we exhausted max_shrink_iter without acceptance, x_ stays
        // at its current value (no change). This is degenerate but
        // mathematically valid (identity rotation always acceptable).

        if (keep_history_) history_buf_.push_back(x_);
    }

    const arma::vec& current() const override { return x_; }

    void set_current(const arma::vec& x_new) override {
        if (x_new.n_elem != cfg_.n) {
            throw std::invalid_argument(
                "gmrf_whitened_ess_block '" + cfg_.name +
                "': set_current length must equal n");
        }
        x_ = x_new;
        if (cfg_.sum_to_zero) {
            const double m = arma::mean(x_);
            x_ -= m;
        }
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

private:
    // Sample nu ~ N(0, Q^{-1}) using the cached SimplicialLLT factorisation:
    //   Q = P^T L L^T P  (with permutation P)
    //   Σ = Q^{-1} = P^T L^{-T} L^{-1} P
    //   Cholesky of Σ: P^T L^{-T}  (i.e., the inverse-upper-triangular
    //   solve applied to permuted noise reproduces a draw from N(0, Σ)).
    arma::vec sample_prior_(std::mt19937_64& rng,
                            std::normal_distribution<double>& std_norm) {
        Eigen::VectorXd z(static_cast<Eigen::Index>(cfg_.n));
        for (std::size_t i = 0; i < cfg_.n; ++i) z[i] = std_norm(rng);

        // Solve L^T y_perm = z in permuted basis; apply inverse perm.
        Eigen::VectorXd y_perm = solver_.matrixU().solve(z);
        Eigen::VectorXd nu_eigen = solver_.permutationPinv() * y_perm;

        arma::vec nu(cfg_.n);
        for (std::size_t i = 0; i < cfg_.n; ++i) nu[i] = nu_eigen[i];

        if (cfg_.sum_to_zero) {
            const double m = arma::mean(nu);
            nu -= m;
        }
        return nu;
    }

    gmrf_whitened_ess_block_config cfg_;
    arma::vec      x_;
    block_context  context_;

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>,
                         Eigen::Lower,
                         Eigen::AMDOrdering<int>> solver_;
    bool symbolic_analyzed_ = false;

    std::vector<arma::vec> history_buf_;
};

}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_GMRF_WHITENED_ESS_BLOCK_HPP

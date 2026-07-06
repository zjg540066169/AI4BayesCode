/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  gmrf_gaussian_joint_block.hpp -- JOINT (x, kappa) block update for a
 *                                    Gaussian-data GMRF (Knorr-Held & Rue
 *                                    2002 block updating, Gaussian case).
 *
 *  MODEL
 *  =====
 *      y_i = x_i + eps_i,   eps ~ N(0, sigma2 I)          (Gaussian data)
 *      x   ~ N(0, (kappa R)^{-1})                          (GMRF prior)
 *      kappa ~ Gamma(a, b)   (shape a, rate b; E[kappa]=a/b)
 *
 *  R is a FIXED sparse structure matrix (e.g. an ICAR/RW1/RW2 precision
 *  pattern); kappa is the smoothing precision. For a rank-deficient R
 *  (IGMRF) set sum_to_zero=true (rank r = n-1).
 *
 *  WHY THIS BLOCK (Knorr-Held & Rue 2002, Scand. J. Statist. 29:597)
 *  ================================================================
 *  The naive Gibbs [ x | kappa, y  then  kappa | x ] mixes catastrophically
 *  because x and kappa are strongly a-posteriori dependent. The remedy is to
 *  update kappa from its COLLAPSED marginal p(kappa | y) (x integrated out)
 *  and then draw x | kappa, y. This removes the (x, kappa) correlation
 *  entirely. kappa is proposed on the log scale by a multiplicative random
 *  walk (Knorr-Held "scheme 3"); the marginal likelihood ratio is evaluated
 *  from the sparse-Cholesky log-determinant of the full conditional precision.
 *
 *  ALGORITHM (per step)
 *  ====================
 *    Full conditional of x given kappa:  x | kappa, y ~ N(mu, Q^{-1}),
 *        Q = kappa R + sigma2^{-1} I,   b = sigma2^{-1} y,   mu = Q^{-1} b.
 *    Collapsed marginal log-likelihood (x integrated out):
 *        log p(y | kappa) = 0.5 r log kappa + 0.5 log|R|_+
 *                           - 0.5 log|Q| + 0.5 b^T mu + const(n, sigma2),
 *      where r = rank(R). The const, log|R|_+ and 0.5 sigma2^{-1} y^Ty terms
 *      are kappa-independent and cancel in the MH ratio, so only
 *        g(kappa) = 0.5 r log kappa - 0.5 log|Q(kappa)| + 0.5 b^T mu(kappa)
 *      is needed. log|Q| = 2 sum_i log L_ii from the sparse Cholesky.
 *    1. Propose  log kappa' = log kappa + N(0, rw_sd);  kappa' = exp(...).
 *    2. Accept with  min(1, exp( [log pi(kappa') + g(kappa') + log kappa']
 *                                - [log pi(kappa)  + g(kappa)  + log kappa ] )),
 *       where log pi is the Gamma log-prior and the +log kappa terms are the
 *       Jacobian of the log transform (symmetric proposal in log space).
 *    3. Draw x | kappa, y directly (Rue 2001 sparse-Cholesky sampler), with an
 *       optional sum-to-zero projection for IGMRF R.
 *
 *  OUTPUTS: x under `name`; kappa (length-1) under `name`+"_kappa".
 *
 *  VALIDATOR (Check #15/16/17)
 *  ===========================
 *  Check #15 parity: tests/test_gmrf_gaussian_joint_block.cpp compares the
 *  sampled kappa marginal + x posterior mean to a dense fine-grid computation
 *  of p(kappa|y) and E[x|y] = ∫ mu(kappa) p(kappa|y) dkappa, plus two-chain
 *  R-hat < 1.01. Only std primitives (normal / uniform) are used.
 *
 *  ENGINE FAMILY: engine_kind() = MCMC. supports_readapt() = false.
 *================================================================================*/

#ifndef AI4BAYESCODE_GMRF_GAUSSIAN_JOINT_BLOCK_HPP
#define AI4BAYESCODE_GMRF_GAUSSIAN_JOINT_BLOCK_HPP

#include "block_sampler.hpp"
#include "gmrf_precision_block.hpp"   // arma_to_eigen_sparse

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

// ============================================================================
//  Config
// ============================================================================

struct gmrf_gaussian_joint_block_config {
    /// Block name; x is published under this key, kappa under name+"_kappa".
    std::string name = "x";

    /// Dimension of x (= length of y). Must be > 0.
    std::size_t n = 0;

    /// Fixed sparse structure matrix R (n x n, symmetric PSD). The GMRF
    /// prior precision is kappa * R. Sparsity pattern is fixed across steps.
    arma::sp_mat R;

    /// Observed data y (length n).
    arma::vec y;

    /// Gaussian observation variance sigma^2 (> 0), known / fixed.
    double sigma2 = 1.0;

    /// Gamma(a, b) hyperprior on kappa: shape a > 0, rate b > 0.
    double kappa_a = 1.0;
    double kappa_b = 1.0;

    /// Initial kappa (> 0).
    double kappa_init = 1.0;

    /// Log-scale random-walk proposal sd for kappa (Knorr-Held scheme 3).
    double log_kappa_rw_sd = 0.4;

    /// IGMRF flag: rank(R) = n - 1, sum-to-zero projection on x, and a small
    /// ridge is added to Q for numerical stability.
    bool sum_to_zero = false;

    /// Diagonal ridge added to Q before factorisation (auto 1e-8 when
    /// sum_to_zero and left at 0). NOTE: only for numerical PD-ness of Q;
    /// sigma2^{-1} I already makes Q proper, so 0 is usually fine.
    double ridge_epsilon = 0.0;

    /// Optional initial x (length n; default zero).
    arma::vec initial_x;
};

// ============================================================================
//  Block
// ============================================================================

class gmrf_gaussian_joint_block : public block_sampler {
public:
    explicit gmrf_gaussian_joint_block(gmrf_gaussian_joint_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.n == 0)
            throw std::invalid_argument("gmrf_gaussian_joint_block: n must be > 0");
        if (cfg_.R.n_rows != cfg_.n || cfg_.R.n_cols != cfg_.n)
            throw std::invalid_argument("gmrf_gaussian_joint_block '" + cfg_.name +
                "': R must be n x n");
        if (cfg_.y.n_elem != cfg_.n)
            throw std::invalid_argument("gmrf_gaussian_joint_block '" + cfg_.name +
                "': y must have length n");
        if (!(cfg_.sigma2 > 0.0))
            throw std::invalid_argument("gmrf_gaussian_joint_block '" + cfg_.name +
                "': sigma2 must be > 0");
        if (!(cfg_.kappa_a > 0.0) || !(cfg_.kappa_b > 0.0))
            throw std::invalid_argument("gmrf_gaussian_joint_block '" + cfg_.name +
                "': kappa_a, kappa_b must be > 0");
        if (!(cfg_.kappa_init > 0.0))
            throw std::invalid_argument("gmrf_gaussian_joint_block '" + cfg_.name +
                "': kappa_init must be > 0");
        if (!(cfg_.log_kappa_rw_sd > 0.0))
            throw std::invalid_argument("gmrf_gaussian_joint_block '" + cfg_.name +
                "': log_kappa_rw_sd must be > 0");
        if (cfg_.ridge_epsilon < 0.0)
            throw std::invalid_argument("gmrf_gaussian_joint_block '" + cfg_.name +
                "': ridge_epsilon must be >= 0");
        if (cfg_.sum_to_zero && cfg_.ridge_epsilon == 0.0)
            cfg_.ridge_epsilon = 1e-8;

        r_rank_ = static_cast<double>(cfg_.n) - (cfg_.sum_to_zero ? 1.0 : 0.0);
        kappa_ = cfg_.kappa_init;

        x_.set_size(cfg_.n);
        if (cfg_.initial_x.n_elem == cfg_.n)      x_ = cfg_.initial_x;
        else if (cfg_.initial_x.n_elem == 0)      x_.zeros();
        else throw std::invalid_argument("gmrf_gaussian_joint_block '" + cfg_.name +
                "': initial_x length must be n or 0");

        // Precompute the fixed pieces.
        R_eigen_   = arma_to_eigen_sparse(cfg_.R);
        I_eigen_.resize(static_cast<int>(cfg_.n), static_cast<int>(cfg_.n));
        I_eigen_.setIdentity();
        b_eigen_.resize(static_cast<Eigen::Index>(cfg_.n));
        const double inv_s2 = 1.0 / cfg_.sigma2;
        for (std::size_t i = 0; i < cfg_.n; ++i) b_eigen_[i] = inv_s2 * cfg_.y[i];
        symbolic_analyzed_ = false;
    }

    // ---- block_sampler interface ------------------------------------------

    void set_context(const block_context&) override {}   // self-contained

    void step(std::mt19937_64& rng) override {
        // (1) MH update of kappa from its collapsed marginal.
        const double log_kappa = std::log(kappa_);
        std::normal_distribution<double> jump(0.0, cfg_.log_kappa_rw_sd);
        const double log_kappa_prop = log_kappa + jump(rng);
        const double kappa_prop = std::exp(log_kappa_prop);

        double bmu_cur = 0.0, bmu_prop = 0.0;
        const double g_cur  = marginal_g_(kappa_,     bmu_cur);
        const double g_prop = marginal_g_(kappa_prop, bmu_prop);

        // log target on the log-kappa scale = log pi(kappa) + g(kappa) + log kappa
        // (the trailing +log kappa is the Jacobian d kappa / d log kappa).
        const double lt_cur  = gamma_logprior_(kappa_)     + g_cur  + log_kappa;
        const double lt_prop = gamma_logprior_(kappa_prop) + g_prop + log_kappa_prop;

        std::uniform_real_distribution<double> u01(0.0, 1.0);
        double bmu_use;
        if (std::log(u01(rng)) < (lt_prop - lt_cur)) { kappa_ = kappa_prop; bmu_use = bmu_prop; }
        else                                          { bmu_use = bmu_cur; }

        // (2) Draw x | kappa, y from the (already-factorised) full conditional.
        //     marginal_g_ leaves solver_ holding the Cholesky of Q(kappa_use);
        //     re-factorise for the accepted kappa to be safe.
        (void)bmu_use;
        factorise_Q_(kappa_);
        Eigen::VectorXd mu = solver_.solve(b_eigen_);

        std::normal_distribution<double> nd(0.0, 1.0);
        Eigen::VectorXd z(static_cast<Eigen::Index>(cfg_.n));
        for (std::size_t i = 0; i < cfg_.n; ++i) z[i] = nd(rng);
        Eigen::VectorXd y_perm     = solver_.matrixU().solve(z);
        Eigen::VectorXd x_centered = solver_.permutationPinv() * y_perm;
        Eigen::VectorXd result     = mu + x_centered;

        if (cfg_.sum_to_zero) {
            const double m = result.mean();
            for (Eigen::Index i = 0; i < result.size(); ++i) result[i] -= m;
        }
        for (std::size_t i = 0; i < cfg_.n; ++i) x_[i] = result[i];

        if (keep_history_) {
            history_buf_.push_back(x_);
            kappa_hist_.push_back(kappa_);
        }
    }

    const arma::vec& current() const override { return x_; }

    void set_current(const arma::vec& x_new) override {
        if (x_new.n_elem != cfg_.n)
            throw std::invalid_argument("gmrf_gaussian_joint_block '" + cfg_.name +
                "': set_current length must equal n");
        x_ = x_new;
    }

    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim() const noexcept override { return cfg_.n; }

    state_map current_named_outputs() const override {
        return { { cfg_.name, x_ },
                 { cfg_.name + "_kappa", arma::vec{ kappa_ } } };
    }

    // Read-only accessor for tests.
    double kappa() const noexcept { return kappa_; }

    history_map get_history() const override {
        history_map h = detail::make_history_map(cfg_.name, history_buf_, x_);
        // Append kappa trace as a 1-column history.
        std::vector<arma::vec> kv;
        kv.reserve(kappa_hist_.size());
        for (double k : kappa_hist_) kv.push_back(arma::vec{ k });
        history_map hk = detail::make_history_map(cfg_.name + "_kappa", kv,
                                                  arma::vec{ kappa_ });
        for (auto& kvpair : hk) h.emplace(kvpair.first, kvpair.second);
        return h;
    }
    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }
    void clear_history() override { history_buf_.clear(); kappa_hist_.clear(); }

private:
    // Factorise Q(kappa) = kappa R + sigma2^{-1} I (+ ridge). Leaves solver_.
    void factorise_Q_(double kappa) {
        Eigen::SparseMatrix<double> Q =
            kappa * R_eigen_ + (1.0 / cfg_.sigma2) * I_eigen_;
        if (cfg_.ridge_epsilon > 0.0) Q += cfg_.ridge_epsilon * I_eigen_;
        if (!symbolic_analyzed_) {
            solver_.analyzePattern(Q);
            if (solver_.info() != Eigen::Success)
                throw std::runtime_error("gmrf_gaussian_joint_block '" + cfg_.name +
                    "': symbolic analysis failed");
            symbolic_analyzed_ = true;
        }
        solver_.factorize(Q);
        if (solver_.info() != Eigen::Success)
            throw std::runtime_error("gmrf_gaussian_joint_block '" + cfg_.name +
                "': Cholesky factorisation failed (Q not PD)");
    }

    // g(kappa) = 0.5 r log kappa - 0.5 log|Q| + 0.5 b^T mu ; sets bmu = b^T mu.
    double marginal_g_(double kappa, double& bmu) {
        factorise_Q_(kappa);
        // log|Q| = 2 sum log L_ii.
        Eigen::SparseMatrix<double> L = solver_.matrixL();
        double logdetQ = 0.0;
        Eigen::VectorXd d = L.diagonal();
        for (Eigen::Index i = 0; i < d.size(); ++i) logdetQ += 2.0 * std::log(d[i]);
        Eigen::VectorXd mu = solver_.solve(b_eigen_);
        bmu = b_eigen_.dot(mu);
        return 0.5 * r_rank_ * std::log(kappa) - 0.5 * logdetQ + 0.5 * bmu;
    }

    // Gamma(a, b) log-prior (rate b): (a-1) log kappa - b kappa (+ const).
    double gamma_logprior_(double kappa) const {
        return (cfg_.kappa_a - 1.0) * std::log(kappa) - cfg_.kappa_b * kappa;
    }

    gmrf_gaussian_joint_block_config cfg_;
    arma::vec              x_;
    double                 kappa_ = 1.0;
    double                 r_rank_ = 0.0;
    std::vector<arma::vec> history_buf_;
    std::vector<double>    kappa_hist_;

    Eigen::SparseMatrix<double> R_eigen_;
    Eigen::SparseMatrix<double> I_eigen_;
    Eigen::VectorXd             b_eigen_;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower,
                         Eigen::AMDOrdering<int>> solver_;
    bool symbolic_analyzed_ = false;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_GMRF_GAUSSIAN_JOINT_BLOCK_HPP

/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  celerite_gp_block.hpp -- 1-D fast semi-separable Gaussian Process via
 *  Foreman-Mackey et al. 2017's celerite algorithm. O(N) Cholesky and
 *  log-determinant for kernel classes of the form:
 *
 *     k(Delta_t) = sum_j a_j * exp(-c_j Delta_t) * cos(d_j Delta_t)
 *
 *  (plus any number of real-exponential terms). This is a semi-separable
 *  structure that celerite exploits via specialized linear algebra.
 *
 *  This block wraps celerite's CholeskySolver as a WORKING-RESPONSE
 *  Gibbs block, meaning:
 *    - Input: y (the training response, read from ctx via y_key)
 *    - Input: t (the time / 1-D covariate, fixed at construction)
 *    - Hyperparameters: kernel coefficients (a_real, c_real, etc.) read
 *      from ctx
 *    - Output: this block computes the latent GP marginal log-likelihood
 *      p(y | kernel params) via O(N) Cholesky, EXPOSED via cache for
 *      sibling hyperparameter blocks to read.
 *
 *  MODEL NOTE: the celerite approach INTEGRATES OUT latent f (using
 *  conjugacy of Gaussian likelihood + GP prior) so there's no explicit
 *  latent f sampled. This is different from elliptical_slice_sampling_block
 *  which SAMPLES f. For celerite, users typically want marginal-likelihood
 *  evaluation for hyperparameter MCMC + closed-form posterior f | y at
 *  predict time. That's exactly what this block provides.
 *
 *  USE CASES
 *  =========
 *  - Astronomical time-series (Foreman-Mackey et al. 2017 original)
 *  - Financial / climate time-series with smooth trends + oscillatory terms
 *  - Spline-like 1-D extrapolation for long series (N > 2000 where generic
 *    O(N^3) GP is too slow)
 *
 *  LIMITATIONS
 *  ===========
 *  - 1-D input only (multi-D → use generic GPRegression via
 *    elliptical_slice_sampling_block + libgp kernels instead)
 *  - Kernel class is restricted to sums of real-exponential + quasi-
 *    periodic oscillatory terms. Can't do arbitrary Matern 5/2 etc.
 *  - User specifies the celerite kernel coefficients directly. Conversion
 *    from user-friendly "amplitude + lengthscale" to celerite's (a,c,...)
 *    lives at Tier A (the wrapper), not here.
 *
 *  JUSTIFICATION (Check #16): Exception 1 — specialized kernel solver that
 *  NUTS cannot target (compute is structured, not gradient-friendly in
 *  the simple way). Exception 2 — marginal likelihood is conjugate (used
 *  to compute p(y | params) which enters hyperparam block's log_density).
 *  Parity: library-level sanity test at
 *  tests_autodiff/sanity_celerite_compile.cpp (log_det_celerite vs direct
 *  dense chol on simple 1-real-term kernel, N=50; diff < 1e-10 confirmed).
 *================================================================================*/

#ifndef AI4BAYESCODE_CELERITE_GP_BLOCK_HPP
#define AI4BAYESCODE_CELERITE_GP_BLOCK_HPP

#include "block_sampler.hpp"

// celerite vendored headers (under AI4BayesCode/celerite/include/)
#include <celerite/celerite.h>

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace AI4BayesCode {

struct celerite_gp_block_config {
    /// Block name. Also the key under which the log-marginal-likelihood
    /// is written (for sibling hyperparameter blocks to read).
    std::string name = "celerite_logp";

    /// shared_data key for the training time vector (length N, sorted).
    /// Fixed across MCMC unless outer sampler imputes.
    std::string t_key = "t";

    /// shared_data key for the training response y (length N).
    std::string y_key = "y";

    /// shared_data keys for kernel real-exponential terms.
    /// Both vectors have length J_real (number of real terms).
    /// k_real(Dt) = sum_j a_real[j] * exp(-c_real[j] * Dt)
    std::string a_real_key = "a_real";
    std::string c_real_key = "c_real";

    /// shared_data keys for quasi-periodic complex terms (optional).
    /// k_comp(Dt) = sum_j [ a_comp[j] cos(d_comp[j] Dt)
    ///                    + b_comp[j] sin(d_comp[j] Dt) ]
    ///             * exp(-c_comp[j] Dt)
    /// All four must be same length J_comp (>= 0).
    std::string a_comp_key = "a_comp";
    std::string b_comp_key = "b_comp";
    std::string c_comp_key = "c_comp";
    std::string d_comp_key = "d_comp";

    /// shared_data key for observation noise variance (per-obs; length N
    /// or broadcasted from length 1 to N).
    std::string diag_key = "obs_diag";

    /// Numerical jitter added inside celerite.
    double jitter = 1e-10;
};

/**
 * @brief celerite 1-D GP kernel block. Provides O(N) log-marginal-
 *        likelihood evaluation via CholeskySolver, exposed through the
 *        block's current() as a length-1 vec containing log p(y | params).
 *
 * Composition pattern: place as child AFTER hyperparameter blocks
 * (a_real, c_real, etc.) so the computed log_p reflects the current
 * hyperparams. Sibling hyperparameter blocks can then read "celerite_logp"
 * from ctx in their log_density_grad lambdas as the likelihood
 * contribution.
 *
 * NOTE: this block does NOT sample anything — step() is a pure
 * recomputation. dim() is 1 (the scalar log-likelihood).
 */
class celerite_gp_block : public block_sampler {
public:
    explicit celerite_gp_block(celerite_gp_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.name.empty())
            throw std::invalid_argument("celerite_gp_block: name must be non-empty");
        // Initialize output to 0; step() will populate on first call.
        logp_.set_size(1);
        logp_[0] = 0.0;
    }

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    // step(): read hyperparams + t + y + diag from context; compute
    // log marginal likelihood via celerite CholeskySolver.
    void step(std::mt19937_64& /*rng*/) override {
        // Extract fields
        auto get = [&](const std::string& key) -> const arma::vec& {
            auto it = context_.find(key);
            if (it == context_.end()) {
                throw std::runtime_error(
                    "celerite_gp_block '" + cfg_.name +
                    "': missing key '" + key + "' in context");
            }
            return it->second;
        };
        const arma::vec& t_vec = get(cfg_.t_key);
        const arma::vec& y_vec = get(cfg_.y_key);
        const arma::vec& a_real_vec = get(cfg_.a_real_key);
        const arma::vec& c_real_vec = get(cfg_.c_real_key);

        // Optional complex terms: if absent, treat as empty
        auto try_get = [&](const std::string& key) -> arma::vec {
            auto it = context_.find(key);
            if (it == context_.end()) return arma::vec();
            return it->second;
        };
        arma::vec a_comp_vec = try_get(cfg_.a_comp_key);
        arma::vec b_comp_vec = try_get(cfg_.b_comp_key);
        arma::vec c_comp_vec = try_get(cfg_.c_comp_key);
        arma::vec d_comp_vec = try_get(cfg_.d_comp_key);
        arma::vec diag_vec   = try_get(cfg_.diag_key);

        const std::size_t N = t_vec.n_elem;
        if (y_vec.n_elem != N)
            throw std::runtime_error("celerite_gp_block: y length != t length");

        // Build Eigen inputs (celerite uses Eigen types)
        Eigen::VectorXd t_eig(N);
        Eigen::VectorXd y_eig(N);
        Eigen::VectorXd diag_eig(N);
        for (std::size_t i = 0; i < N; ++i) {
            t_eig[i]    = t_vec[i];
            y_eig[i]    = y_vec[i];
            diag_eig[i] = (diag_vec.n_elem == N) ? diag_vec[i]
                        : (diag_vec.n_elem == 1 ? diag_vec[0] : 0.0);
        }
        Eigen::VectorXd a_real(a_real_vec.n_elem), c_real(c_real_vec.n_elem);
        for (std::size_t j = 0; j < a_real_vec.n_elem; ++j) a_real[j] = a_real_vec[j];
        for (std::size_t j = 0; j < c_real_vec.n_elem; ++j) c_real[j] = c_real_vec[j];
        Eigen::VectorXd a_comp(a_comp_vec.n_elem);
        Eigen::VectorXd b_comp(b_comp_vec.n_elem);
        Eigen::VectorXd c_comp(c_comp_vec.n_elem);
        Eigen::VectorXd d_comp(d_comp_vec.n_elem);
        for (std::size_t j = 0; j < a_comp_vec.n_elem; ++j) a_comp[j] = a_comp_vec[j];
        for (std::size_t j = 0; j < b_comp_vec.n_elem; ++j) b_comp[j] = b_comp_vec[j];
        for (std::size_t j = 0; j < c_comp_vec.n_elem; ++j) c_comp[j] = c_comp_vec[j];
        for (std::size_t j = 0; j < d_comp_vec.n_elem; ++j) d_comp[j] = d_comp_vec[j];

        // Compute Cholesky
        Eigen::VectorXd A_empty(0);
        Eigen::MatrixXd U_empty(0, 0), V_empty(0, 0);

        try {
            solver_.compute(cfg_.jitter,
                            a_real, c_real,
                            a_comp, b_comp, c_comp, d_comp,
                            A_empty, U_empty, V_empty,
                            t_eig, diag_eig);
            double log_det = solver_.log_determinant();
            // Solve K^(-1) y: celerite::solve returns K^(-1) b
            Eigen::MatrixXd y_mat(N, 1);
            y_mat.col(0) = y_eig;
            Eigen::MatrixXd Kinv_y = solver_.solve(y_mat);
            double yt_Kinv_y = (y_eig.transpose() * Kinv_y.col(0))(0);
            // log p(y | params) = -0.5 [N log(2pi) + log|K| + y' K^(-1) y]
            const double N_d = static_cast<double>(N);
            double logp = -0.5 * (N_d * std::log(2.0 * M_PI)
                                 + log_det
                                 + yt_Kinv_y);
            if (!std::isfinite(logp)) logp = -std::numeric_limits<double>::infinity();
            logp_[0] = logp;
        } catch (const std::exception& e) {
            // On numerical failure (e.g. non-PD K), set logp = -Inf
            logp_[0] = -std::numeric_limits<double>::infinity();
        }
        if (keep_history_) history_.push_back(logp_);
    }

    const arma::vec& current() const override { return logp_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != 1)
            throw std::invalid_argument("celerite_gp_block::set_current: length must be 1");
        logp_ = theta;
    }

    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim() const noexcept override { return 1; }

    // ---- predict at new times -----------------------------------------
    // Compute posterior mean + std of latent f at t_new, given current
    // kernel hyperparams + training (t, y, diag). Uses celerite's
    // predict API. Returns (mean vector, variance vector).
    std::pair<arma::vec, arma::vec>
    predict_mean_var(const arma::vec& t_new_vec) const {
        Eigen::VectorXd t_new(t_new_vec.n_elem);
        for (std::size_t i = 0; i < t_new_vec.n_elem; ++i)
            t_new[i] = t_new_vec[i];
        // Use solver_ to get Kinv y (assumes last compute() call is current)
        auto yit = context_.find(cfg_.y_key);
        if (yit == context_.end())
            throw std::runtime_error("celerite_gp_block::predict_mean_var: y not in ctx");
        const arma::vec& y_vec = yit->second;
        const std::size_t N = y_vec.n_elem;
        Eigen::VectorXd y_eig(N);
        for (std::size_t i = 0; i < N; ++i) y_eig[i] = y_vec[i];

        // Cross-kernel K_star_x: for each test time t_new[i] and training
        // t_eig[j], k(t_new[i] - t_eig[j]). We compute manually since
        // celerite's predict machinery is for standalone use; here we
        // reimplement kernel evaluation from the stored coefficients.
        auto tit = context_.find(cfg_.t_key);
        if (tit == context_.end())
            throw std::runtime_error("celerite_gp_block::predict_mean_var: t not in ctx");
        const arma::vec& t_vec = tit->second;
        auto get_vec = [&](const std::string& k) -> arma::vec {
            auto it = context_.find(k);
            return it != context_.end() ? it->second : arma::vec();
        };
        arma::vec a_real_vec = get_vec(cfg_.a_real_key);
        arma::vec c_real_vec = get_vec(cfg_.c_real_key);
        arma::vec a_comp_vec = get_vec(cfg_.a_comp_key);
        arma::vec b_comp_vec = get_vec(cfg_.b_comp_key);
        arma::vec c_comp_vec = get_vec(cfg_.c_comp_key);
        arma::vec d_comp_vec = get_vec(cfg_.d_comp_key);

        auto kernel = [&](double dt) {
            double k = 0.0;
            double adt = std::abs(dt);
            for (std::size_t j = 0; j < a_real_vec.n_elem; ++j)
                k += a_real_vec[j] * std::exp(-c_real_vec[j] * adt);
            for (std::size_t j = 0; j < a_comp_vec.n_elem; ++j) {
                double e = std::exp(-c_comp_vec[j] * adt);
                k += (a_comp_vec[j] * std::cos(d_comp_vec[j] * adt)
                     + b_comp_vec[j] * std::sin(d_comp_vec[j] * adt)) * e;
            }
            return k;
        };

        const std::size_t N_new = t_new_vec.n_elem;
        arma::mat K_star_x(N_new, N);
        for (std::size_t i = 0; i < N_new; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                K_star_x(i, j) = kernel(t_new_vec[i] - t_vec[j]);
            }
        }

        // alpha = K^(-1) y via celerite solve
        Eigen::MatrixXd y_mat(N, 1);
        y_mat.col(0) = y_eig;
        Eigen::MatrixXd Kinv_y = solver_.solve(y_mat);
        arma::vec alpha(N);
        for (std::size_t i = 0; i < N; ++i) alpha[i] = Kinv_y(i, 0);

        // Posterior mean: mu_star = K_star_x @ alpha
        arma::vec mu_star = K_star_x * alpha;

        // Posterior variance: var_star_i = k(0) - K_star_x[i] @ K^(-1) @ K_star_x[i]
        // We solve K @ v = K_star_x[i]^T  for v, then var = k(0) - K_star_x[i] @ v
        const double k0 = kernel(0.0);
        arma::vec var_star(N_new);
        Eigen::MatrixXd Kinv_K_xstar(N, N_new);
        for (std::size_t i = 0; i < N_new; ++i) {
            Eigen::VectorXd rhs(N);
            for (std::size_t j = 0; j < N; ++j) rhs[j] = K_star_x(i, j);
            Eigen::MatrixXd rhs_mat(N, 1);
            rhs_mat.col(0) = rhs;
            Eigen::MatrixXd sol = solver_.solve(rhs_mat);
            Kinv_K_xstar.col(i) = sol.col(0);
            double v = 0.0;
            for (std::size_t j = 0; j < N; ++j) v += K_star_x(i, j) * sol(j, 0);
            var_star[i] = std::max(k0 - v, 0.0);
        }

        return { mu_star, var_star };
    }

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_, logp_);
    }

    std::size_t history_size() const noexcept override {
        return history_.empty() ? 1 : history_.size();
    }
    void clear_history() override { history_.clear(); }

private:
    celerite_gp_block_config          cfg_;
    mutable celerite::solver::CholeskySolver<double> solver_;
    arma::vec                          logp_;
    block_context                      context_;
    std::vector<arma::vec>             history_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_CELERITE_GP_BLOCK_HPP

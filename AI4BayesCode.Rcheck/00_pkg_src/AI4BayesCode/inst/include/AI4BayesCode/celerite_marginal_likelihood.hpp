/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  celerite_marginal_likelihood.hpp -- pure-function wrapper around the
 *      celerite solver, returning the log-marginal-likelihood
 *      log p(y | kernel params) for a 1-D time series with
 *      semi-separable kernel structure (Foreman-Mackey et al. 2017).
 *
 *  WHY A SEPARATE HELPER
 *  =====================
 *  celerite_gp_block is a stateful block_sampler: it holds a
 *  CholeskySolver member and computes log p(y | params) on the
 *  CURRENT shared_data at each step() call. When sibling hyperparameter
 *  blocks (e.g. univariate_slice_sampling_block for amp / tau / sigma)
 *  need to evaluate the log-marginal at a PROPOSED parameter value
 *  within their sampler's step, they cannot simply read
 *  celerite_gp_block's cached logp -- it reflects the current state,
 *  not the proposed one. Options:
 *    (a) mutate celerite_gp_block's solver from inside another block's
 *        lambda (violates block_sampler::step isolation);
 *    (b) recompute via a stateless function that accepts raw inputs
 *        and returns a scalar.
 *
 *  This header provides option (b). It is a thin wrapper: construct a
 *  temporary CholeskySolver<double> internally, feed in (t, y,
 *  a_real, c_real, [a_comp, b_comp, c_comp, d_comp,] obs_diag, jitter),
 *  return log p(y | ...). Solver state is destroyed on return.
 *
 *  COST NOTE
 *  =========
 *  celerite is O(N); a fresh CholeskySolver<double> construction is
 *  trivial (mostly just member-initializer boilerplate). Per-call cost
 *  is dominated by the compute() call, which is O(N) for the kernel
 *  class supported by celerite. Calling this helper from a slice-
 *  sampler lambda (~3-10 times per step per hyperparam) is entirely
 *  feasible for N up to ~10,000.
 *
 *  USE CASES
 *  =========
 *  - Hyperparameter MCMC on celerite-marginalized GP (via slice_block /
 *    MH / etc). See examples/GPTimeSeries.cpp (v0.5).
 *  - Model selection / log-evidence sweeps at different hyperparameter
 *    settings.
 *  - Independent verification of celerite_gp_block's internal logp
 *    (sanity tests).
 *
 *  LIMITATIONS
 *  ===========
 *  1-D input only. Kernel class restricted to sums of real-exponential
 *  + quasi-periodic terms. Caller is responsible for ensuring t is
 *  sorted ascending and kernel parameters (a_real, c_real, c_comp, ...)
 *  are positive.
 *================================================================================*/

#ifndef AI4BAYESCODE_CELERITE_MARGINAL_LIKELIHOOD_HPP
#define AI4BAYESCODE_CELERITE_MARGINAL_LIKELIHOOD_HPP

#ifdef AI4BAYESCODE_RCPP_MODULE
# include <RcppArmadillo.h>
#else
# include <armadillo>
#endif

#include <celerite/celerite.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace AI4BayesCode {

/**
 * @brief Compute log p(y | celerite kernel params) for a 1-D time series.
 *
 * Kernel class:
 *     k(dt) = sum_j a_real[j]   * exp(-c_real[j] * |dt|)
 *           + sum_j (a_comp[j] cos(d_comp[j] |dt|)
 *                  + b_comp[j] sin(d_comp[j] |dt|))
 *                  * exp(-c_comp[j] * |dt|)
 *
 * @param t          Training times, length N, sorted ascending.
 * @param y          Training response, length N.
 * @param a_real     Real-term amplitudes, length J_real (>= 0).
 * @param c_real     Real-term decay rates, length J_real (>= 0).
 * @param a_comp     Complex-term cos amplitudes, length J_comp (>= 0).
 *                   Pass arma::vec() for no complex terms.
 * @param b_comp     Complex-term sin amplitudes, length J_comp (>= 0).
 * @param c_comp     Complex-term decay rates, length J_comp (>= 0).
 * @param d_comp     Complex-term oscillation frequencies, length J_comp (>= 0).
 * @param obs_diag   Observation noise variance per point, length N; or
 *                   length 1 (broadcast to N); or empty (treated as 0).
 * @param jitter     Numerical stabilizer passed to celerite::compute
 *                   (default 1e-10).
 *
 * @return log p(y | params) if Cholesky is positive-definite; otherwise
 *         -std::numeric_limits<double>::infinity().
 *
 * @throws std::runtime_error on shape mismatches between t, y, obs_diag.
 */
inline double celerite_log_marginal(
    const arma::vec& t,
    const arma::vec& y,
    const arma::vec& a_real,
    const arma::vec& c_real,
    const arma::vec& a_comp,
    const arma::vec& b_comp,
    const arma::vec& c_comp,
    const arma::vec& d_comp,
    const arma::vec& obs_diag,
    double           jitter = 1e-10)
{
    const std::size_t N = t.n_elem;
    if (y.n_elem != N) {
        throw std::runtime_error(
            "celerite_log_marginal: y.n_elem != t.n_elem");
    }
    if (N < 2) {
        throw std::runtime_error(
            "celerite_log_marginal: N must be >= 2");
    }

    // Build Eigen inputs
    Eigen::VectorXd t_eig(N), y_eig(N), diag_eig(N);
    for (std::size_t i = 0; i < N; ++i) {
        t_eig[i] = t[i];
        y_eig[i] = y[i];
        if (obs_diag.n_elem == N)      diag_eig[i] = obs_diag[i];
        else if (obs_diag.n_elem == 1) diag_eig[i] = obs_diag[0];
        else                            diag_eig[i] = 0.0;
    }

    Eigen::VectorXd a_real_e(a_real.n_elem), c_real_e(c_real.n_elem);
    for (std::size_t j = 0; j < a_real.n_elem; ++j) a_real_e[j] = a_real[j];
    for (std::size_t j = 0; j < c_real.n_elem; ++j) c_real_e[j] = c_real[j];

    Eigen::VectorXd a_comp_e(a_comp.n_elem);
    Eigen::VectorXd b_comp_e(b_comp.n_elem);
    Eigen::VectorXd c_comp_e(c_comp.n_elem);
    Eigen::VectorXd d_comp_e(d_comp.n_elem);
    for (std::size_t j = 0; j < a_comp.n_elem; ++j) a_comp_e[j] = a_comp[j];
    for (std::size_t j = 0; j < b_comp.n_elem; ++j) b_comp_e[j] = b_comp[j];
    for (std::size_t j = 0; j < c_comp.n_elem; ++j) c_comp_e[j] = c_comp[j];
    for (std::size_t j = 0; j < d_comp.n_elem; ++j) d_comp_e[j] = d_comp[j];

    // Empty "A/U/V" (non-expanded) inputs required by celerite's public API.
    Eigen::VectorXd A_empty(0);
    Eigen::MatrixXd U_empty(0, 0), V_empty(0, 0);

    celerite::solver::CholeskySolver<double> solver;
    try {
        solver.compute(jitter,
                       a_real_e, c_real_e,
                       a_comp_e, b_comp_e, c_comp_e, d_comp_e,
                       A_empty, U_empty, V_empty,
                       t_eig, diag_eig);
        const double log_det = solver.log_determinant();

        Eigen::MatrixXd y_mat(N, 1);
        y_mat.col(0) = y_eig;
        Eigen::MatrixXd Kinv_y = solver.solve(y_mat);
        const double yt_Kinv_y = (y_eig.transpose() * Kinv_y.col(0))(0);

        const double N_d = static_cast<double>(N);
        const double logp = -0.5 * (N_d * std::log(2.0 * M_PI)
                                    + log_det
                                    + yt_Kinv_y);
        if (!std::isfinite(logp)) {
            return -std::numeric_limits<double>::infinity();
        }
        return logp;
    } catch (const std::exception&) {
        // Numerical failure (non-PD kernel, NaN input, etc.)
        return -std::numeric_limits<double>::infinity();
    }
}

/**
 * @brief Convenience overload: real-exponential terms only (no complex
 *        / quasi-periodic terms). Covers the common "OU / Matern 1/2 /
 *        single-scale random walk" use case.
 */
inline double celerite_log_marginal_real(
    const arma::vec& t,
    const arma::vec& y,
    const arma::vec& a_real,
    const arma::vec& c_real,
    const arma::vec& obs_diag,
    double           jitter = 1e-10)
{
    return celerite_log_marginal(t, y,
                                 a_real, c_real,
                                 arma::vec(), arma::vec(),
                                 arma::vec(), arma::vec(),
                                 obs_diag, jitter);
}

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_CELERITE_MARGINAL_LIKELIHOOD_HPP

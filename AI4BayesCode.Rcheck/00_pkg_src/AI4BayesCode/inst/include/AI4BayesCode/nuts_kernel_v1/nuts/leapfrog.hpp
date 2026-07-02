// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
// leapfrog.hpp — Leapfrog integrator with cached gradient
//
// Fixes the mcmclib 2× bug: gradient at the END of step k IS the gradient
// at the START of step k+1. mcmclib re-evaluates this; we cache.
//
// API:
//   LpAndGrad evaluate(theta) -> {lp, grad}             user lambda
//   leapfrog_step(theta_in, p_in, grad_in, eps, M_inv, kind)
//                    -> {theta_new, p_new, grad_new, lp_new}
//
// After N leapfrog steps from (theta_0, p_0, grad_0):
//   - mcmclib: 2N grad calls (1 unused per pair)
//   - ours: N+1 grad calls (initial grad outside loop, then N inside)
//
// Diagonal vs Dense metric — separate code paths:
//   - Diagonal: theta += eps * (M_inv % p)        (elementwise, no BLAS gemv)
//   - Dense:    theta += eps * (M_inv * p)        (BLAS gemv)

#pragma once

#include <armadillo>
#include <utility>

namespace AI4BayesCode {
namespace internal {

// Metric type tag (compile-time dispatch via overload).
struct DiagonalMetric { const arma::vec& inv_diag; };
struct DenseMetric    { const arma::mat& inv_mat; };

// Result of a single leapfrog step: new position, momentum, gradient,
// log-density value (the grad is at theta_new — caller can use it
// as grad_in for the next step).
struct LeapfrogResult {
    arma::vec theta_new;
    arma::vec p_new;
    arma::vec grad_new;
    double   lp_new;
};

// Diagonal metric leapfrog. eps is step size, mass^{-1}_diag is the
// diagonal of the inverse mass matrix.
//
// CRITICAL: grad_in must be ∇log_density(theta_in). We use it for the
// first half-kick; the second half-kick uses the NEW gradient computed
// at the updated theta.
//
// Returned grad_new is at theta_new (NOT theta_in). Use it as grad_in
// for the next leapfrog step.
template <typename LogDensityFn>
LeapfrogResult leapfrog_step(const arma::vec& theta_in,
                              const arma::vec& p_in,
                              const arma::vec& grad_in,
                              double eps,
                              DiagonalMetric M,
                              LogDensityFn& log_density) {
    // First half-kick: p_half = p + (eps/2) * grad(theta_in)
    arma::vec p_half = p_in + (eps * 0.5) * grad_in;

    // Drift: theta_new = theta + eps * M_inv * p_half
    //         (diagonal metric — elementwise multiply, no BLAS gemv)
    arma::vec theta_new = theta_in + eps * (M.inv_diag % p_half);

    // Compute gradient at theta_new — ONE evaluation, becomes grad_in
    // for the next leapfrog step.
    arma::vec grad_new;
    double lp_new = log_density(theta_new, &grad_new);

    // Second half-kick: p_new = p_half + (eps/2) * grad(theta_new)
    arma::vec p_new = p_half + (eps * 0.5) * grad_new;

    return LeapfrogResult{
        std::move(theta_new),
        std::move(p_new),
        std::move(grad_new),
        lp_new
    };
}

// Dense metric leapfrog. M.inv_mat is the dense inverse mass matrix.
template <typename LogDensityFn>
LeapfrogResult leapfrog_step(const arma::vec& theta_in,
                              const arma::vec& p_in,
                              const arma::vec& grad_in,
                              double eps,
                              DenseMetric M,
                              LogDensityFn& log_density) {
    arma::vec p_half = p_in + (eps * 0.5) * grad_in;
    // Drift: BLAS gemv for dense metric.
    arma::vec theta_new = theta_in + eps * (M.inv_mat * p_half);
    arma::vec grad_new;
    double lp_new = log_density(theta_new, &grad_new);
    arma::vec p_new = p_half + (eps * 0.5) * grad_new;
    return LeapfrogResult{
        std::move(theta_new),
        std::move(p_new),
        std::move(grad_new),
        lp_new
    };
}

// Kinetic energy K(p) = 0.5 * p^T M^{-1} p.
// Diagonal: 0.5 * sum(p % M_inv_diag % p).
inline double kinetic_energy(const arma::vec& p, DiagonalMetric M) {
    return 0.5 * arma::accu(p % M.inv_diag % p);
}

// Dense: 0.5 * dot(p, M_inv * p) via BLAS gemv.
inline double kinetic_energy(const arma::vec& p, DenseMetric M) {
    return 0.5 * arma::dot(p, M.inv_mat * p);
}

// Sample momentum from N(0, M). For diagonal metric, p_i ~ N(0, M_diag_i),
// equivalently p_i = sqrt(M_diag_i) * z_i where z ~ N(0, I).
//
// M_diag here is the diagonal of M itself (not M_inv). Caller must
// invert before passing to leapfrog (M_inv = 1.0 / M_diag elementwise).
template <typename Rng>
arma::vec sample_momentum_diagonal(const arma::vec& mass_diag, Rng& rng) {
    arma::vec z(mass_diag.n_elem);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (std::size_t i = 0; i < z.n_elem; ++i) {
        z(i) = normal(rng);
    }
    return arma::sqrt(mass_diag) % z;
}

// Sample momentum from N(0, M) for dense metric.
// L = chol(M); p = L * z where z ~ N(0, I).
template <typename Rng>
arma::vec sample_momentum_dense(const arma::mat& mass_chol_L, Rng& rng) {
    arma::vec z(mass_chol_L.n_rows);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (std::size_t i = 0; i < z.n_elem; ++i) {
        z(i) = normal(rng);
    }
    return mass_chol_L * z;
}

}  // namespace internal
}  // namespace AI4BayesCode

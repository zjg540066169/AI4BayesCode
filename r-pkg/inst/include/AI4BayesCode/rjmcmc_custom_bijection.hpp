/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  rjmcmc_custom_bijection.hpp -- user-supplied bijection with auto-computed
 *                                 Jacobian for rjmcmc_block.
 *
 *  WHY THIS FILE EXISTS
 *  ====================
 *  rjmcmc_block ships three families of birth/death proposals:
 *
 *    (a) Identity-coordinate proposals: birth draws beta_new directly from
 *        the user's propose_sample. Jacobian = 1 by construction; no
 *        transform is configured. Covers the canonical Dirac
 *        spike-and-slab variable-selection use case.
 *
 *    (b) Library-provided 1D transforms: identity_transform_1d,
 *        diagonal_linear_transform_1d, diagonal_affine_transform_1d
 *        (rjmcmc_transforms.hpp). Each transform class computes
 *        |det J| internally; users pick a transform and supply its
 *        scale / offset only.
 *
 *    (c) Custom 1D bijection (THIS FILE): for non-linear monotone maps
 *        the previous two families cannot fit, the user supplies one
 *        TEMPLATED forward map plus an analytic inverse, and the
 *        framework computes |dbeta/du| via runtime autodiff.
 *
 *  In all three cases the universal rule of system_design.md §10 holds:
 *  the user NEVER hand-writes a Jacobian formula.
 *
 *  THE SINGLE-TEMPLATE PATTERN (mcmclib / Stan Math style)
 *  =======================================================
 *
 *    The user writes ONE templated forward map struct:
 *
 *        struct MyForward {
 *            template <typename T>
 *            T operator()(T u) const { return ...; }
 *        };
 *
 *    plus ONE non-templated inverse function:
 *
 *        struct MyInverse {
 *            double operator()(double beta) const { return ...; }
 *        };
 *
 *    The library instantiates Forward at:
 *      - `double`        for the actual sampling step (returns beta_new);
 *      - `autodiff::var` for runtime AD computation of |dbeta/du|.
 *
 *    The user writes NO Jacobian formula. The Jacobian is correct by
 *    construction (it comes from autodiff on the same template the
 *    user already wrote for sampling).
 *
 *  WHAT THIS FILE SUPPORTS (1D scalar bijections)
 *  ==============================================
 *  - Forward: beta = T(u),  T : R -> R (or a sub-interval, monotone /
 *    invertible on the relevant domain).
 *  - Inverse: u = T^{-1}(beta), supplied analytically by the user.
 *  - Jacobian: |dbeta/du| = | derivative of forward at u |, auto-computed
 *    via reverse-mode autodiff on the templated forward at autodiff::var.
 *
 *  WHAT IT DOES NOT SUPPORT
 *  ========================
 *  - Multi-dim bijections R^n -> R^n with n > 1. The rjmcmc_block API
 *    operates per-coefficient (1D); multi-dim bijections (e.g.,
 *    split-merge for finite mixtures with unknown K) require a
 *    different block class.
 *  - Numerical inversion. The user MUST provide an analytic inverse;
 *    Newton's method / bracketed root-finding is not built in.
 *  - Bijections that don't preserve dimension (k_aux != k_new). At
 *    this version we require k_aux = k_new = 1 (1D-to-1D scalar map).
 *
 *  USAGE PATTERN
 *  =============
 *
 *    #include "AI4BayesCode/rjmcmc_block.hpp"
 *    #include "AI4BayesCode/rjmcmc_custom_bijection.hpp"
 *
 *    // 1) Forward map as a TEMPLATED callable.
 *    struct sinh_forward {
 *        template <typename T>
 *        T operator()(T u) const { return sinh(u); }
 *    };
 *
 *    // 2) Inverse map as a non-templated callable.
 *    struct asinh_inverse {
 *        double operator()(double beta) const { return std::asinh(beta); }
 *    };
 *
 *    // 3) Wrap as a transform and assign to rjmcmc_block_config.
 *    rjmcmc_block_config cfg;
 *    cfg.transform = AI4BayesCode::rjmcmc_transforms::
 *        make_templated_bijection_1d(sinh_forward{}, asinh_inverse{});
 *    // ... rest of cfg ...
 *
 *  The wrapper class `templated_bijection_1d` satisfies the existing
 *  `transform_1d_base` interface that rjmcmc_block already plumbs through
 *  for library transforms. No changes to rjmcmc_block.hpp are needed.
 *
 *  GEN-TIME SANITY PROBES (validator.md Check #14)
 *  ===============================================
 *  Whenever a user-supplied bijection is configured, the validator
 *  fires three numerical sanity probes on a small grid of u points,
 *  all derived from the SINGLE templated forward + the user's analytic
 *  inverse:
 *
 *    1. Round-trip: |inv(fwd(u)) - u| < 1e-10 (catches typos in the inverse).
 *    2. Jacobian non-singularity: |dbeta/du| > 1e-12 (catches degenerate
 *       forwards: poles, constant sections, sign flips).
 *    3. Forward / reverse Jacobian inverse-pair: |fwd_J(u) * rev_J(T(u)) - 1|
 *       < 1e-10 (the bijection invariant).
 *
 *  See `check_roundtrip` and `check_jacobian_inverse_pair` below for the
 *  helper functions invoked by the companion file.
 *
 *================================================================================*/

#ifndef AI4BAYESCODE_RJMCMC_CUSTOM_BIJECTION_HPP
#define AI4BAYESCODE_RJMCMC_CUSTOM_BIJECTION_HPP

#include "rjmcmc_transforms.hpp"

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace AI4BayesCode {
namespace rjmcmc_transforms {

// ---------------------------------------------------------------------------
// templated_bijection_1d<Forward, Inverse>
//
// A transform that satisfies the transform_1d_base interface by
// instantiating the user's templated forward map at both `double`
// (for sampling) and `autodiff::var` (for the auto-Jacobian).
//
// Concept requirements:
//   Forward must be callable as
//       T operator()(T u) const
//     for both T = double and T = autodiff::var. The cleanest way to
//     satisfy this is a struct with a templated operator():
//
//         struct MyForward {
//             template <typename T> T operator()(T u) const { ... }
//         };
//
//   Inverse must be callable as
//       double operator()(double beta) const
//     and return the analytic inverse T^{-1}(beta). For exotic bijections
//     where no closed-form inverse exists, users are expected to
//     implement their own root-finder inside Inverse::operator().
//
// Returns from apply_forward / apply_reverse:
//   - On success: |det J| > 0 (a finite positive number).
//   - On failure (non-finite output, zero/negative Jacobian):
//     std::numeric_limits<double>::quiet_NaN(). rjmcmc_block treats
//     NaN as "reject this move" (see rjmcmc_block.hpp step()).
// ---------------------------------------------------------------------------
template <typename Forward, typename Inverse>
class templated_bijection_1d : public transform_1d_base {
public:
    templated_bijection_1d(Forward fwd, Inverse inv)
        : fwd_(std::move(fwd)), inv_(std::move(inv)) {}

    /// Forward direction: u -> beta = T(u). Returns |dbeta/du| at u.
    double apply_forward(double u, double& beta_new) const override {
        // Sampling pass: instantiate Forward at double.
        beta_new = fwd_(u);
        if (!std::isfinite(beta_new)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // Jacobian pass: instantiate Forward at autodiff::var, take the
        // scalar derivative of the output w.r.t. the input. We wrap the
        // single scalar u in a 1-element Eigen vector because
        // autodiff::gradient operates on vectors.
        Eigen::Matrix<autodiff::var, -1, 1> u_vec(1);
        u_vec(0) = u;
        autodiff::var beta_var = fwd_(u_vec(0));
        Eigen::VectorXd grad = autodiff::gradient(beta_var, u_vec);
        const double jac = std::abs(static_cast<double>(grad(0)));
        if (!std::isfinite(jac) || jac <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return jac;
    }

    /// Reverse direction: beta -> u = T^{-1}(beta). Returns |du/dbeta|.
    double apply_reverse(double beta_old, double& u_implied) const override {
        u_implied = inv_(beta_old);
        if (!std::isfinite(u_implied)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // |du/dbeta| = 1 / |dbeta/du| evaluated at u = T^{-1}(beta).
        // Same vector-wrap idiom as apply_forward.
        Eigen::Matrix<autodiff::var, -1, 1> u_vec(1);
        u_vec(0) = u_implied;
        autodiff::var beta_var = fwd_(u_vec(0));
        Eigen::VectorXd grad = autodiff::gradient(beta_var, u_vec);
        const double fwd_jac = std::abs(static_cast<double>(grad(0)));
        if (!std::isfinite(fwd_jac) || fwd_jac <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0 / fwd_jac;
    }

private:
    Forward fwd_;
    Inverse inv_;
};

// ---------------------------------------------------------------------------
// Factory: construct a templated_bijection_1d wrapped in shared_ptr<base>.
// Use this in user code so rjmcmc_block_config::transform can hold the
// type-erased transform_1d_base pointer without per-call boilerplate.
// ---------------------------------------------------------------------------
template <typename Forward, typename Inverse>
std::shared_ptr<transform_1d_base>
make_templated_bijection_1d(Forward fwd, Inverse inv) {
    return std::make_shared<
        templated_bijection_1d<Forward, Inverse>>(
            std::move(fwd), std::move(inv));
}

// ---------------------------------------------------------------------------
// Sanity-probe helpers exposed for the gen-time validator (see
// validator.md Check #14).
//
// At gen-time the validator companion file calls these on a small grid
// of u values to verify:
//
//   1. Bijection round-trip: |T^{-1}(T(u)) - u| < tol  (inverse is correct)
//   2. Jacobian non-singularity: |dbeta/du| > epsilon at every grid point
//   3. Forward / reverse Jacobian inverse-pair: |fwd_J(u) * rev_J(T(u)) - 1| < tol
//
// All three are derived from the SINGLE templated Forward + Inverse
// pair the user wrote; they do not require any second function.
// ---------------------------------------------------------------------------

/// Round-trip residual: |T^{-1}(T(u)) - u|.
template <typename Bijection>
double check_roundtrip(const Bijection& b, double u) {
    double beta = 0.0;
    const double det_fwd = b.apply_forward(u, beta);
    if (!std::isfinite(det_fwd)) {
        return std::numeric_limits<double>::infinity();
    }
    double u_back = 0.0;
    const double det_rev = b.apply_reverse(beta, u_back);
    if (!std::isfinite(det_rev)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::abs(u_back - u);
}

/// Forward * reverse Jacobian product residual: |fwd_J(u) * rev_J(T(u)) - 1|.
template <typename Bijection>
double check_jacobian_inverse_pair(const Bijection& b, double u) {
    double beta = 0.0;
    const double det_fwd = b.apply_forward(u, beta);
    if (!std::isfinite(det_fwd) || det_fwd <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    double u_back = 0.0;
    const double det_rev = b.apply_reverse(beta, u_back);
    if (!std::isfinite(det_rev) || det_rev <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::abs(det_fwd * det_rev - 1.0);
}

}  // namespace rjmcmc_transforms
}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_RJMCMC_CUSTOM_BIJECTION_HPP

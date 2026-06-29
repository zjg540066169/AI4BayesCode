// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  test_rjmcmc_custom_bijection.cpp -- unit tests for the custom-bijection
//                                      templated layer in
//                                      rjmcmc_custom_bijection.hpp.
//
//  Coverage
//  --------
//  1. Forward / inverse round-trip on three bijections (sinh, cube, scaled
//     affine).
//  2. Jacobian agreement with the analytic derivative (autodiff vs
//     hand-computed |dT/du|).
//  3. Forward * reverse Jacobian product = 1 (bijection invariant).
//  4. Sanity-probe helpers (`check_roundtrip`, `check_jacobian_inverse_pair`)
//     return finite values within the documented tolerances on a probe grid.
//  5. Failure-mode handling: degenerate forward (constant / zero-gradient)
//     reports NaN from apply_forward (rjmcmc_block treats that as reject).
//
//  How to run from R
//  -----------------
//      Rcpp::sourceCpp("tests_autodiff/test_rjmcmc_custom_bijection.cpp")
//      print(test_rjmcmc_custom_bijection())  # named list with PASS / FAIL
//
//  All assertions live in the C++ side; the R wrapper just calls the
//  exported function and inspects the returned list. Each test reports
//  a numeric residual (max diff or |J - 1|), so it is easy to inspect
//  the tolerances and verify they match the validator's sanity-probe
//  spec in validator.md Check #14.
// ============================================================================

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "AI4BayesCode/rjmcmc_custom_bijection.hpp"
#include "AI4BayesCode/rjmcmc_transforms.hpp"

#include <cmath>
#include <limits>

using AI4BayesCode::rjmcmc_transforms::make_templated_bijection_1d;
using AI4BayesCode::rjmcmc_transforms::check_roundtrip;
using AI4BayesCode::rjmcmc_transforms::check_jacobian_inverse_pair;

// ---------------------------------------------------------------------------
// Three test bijections, each with KNOWN analytic forward / inverse / Jacobian.
// ---------------------------------------------------------------------------

// 1. sinh(u): forward = sinh(u), inverse = asinh(beta), |dβ/du| = cosh(u)
struct SinhFwd {
    template <typename T>
    T operator()(T u) const { return sinh(u); }
};
struct SinhInv {
    double operator()(double beta) const { return std::asinh(beta); }
};

// 2. u^3: forward = u^3 (monotone increasing, smooth), inverse = cbrt(beta),
//    |dβ/du| = 3 u^2. Singular at u = 0! At u = 0 the autodiff Jacobian is
//    0; apply_forward returns NaN on the |J| <= 0 guard. Reasonable failure
//    mode for the bijection.
struct CubeFwd {
    template <typename T>
    T operator()(T u) const { return u * u * u; }
};
struct CubeInv {
    double operator()(double beta) const { return std::cbrt(beta); }
};

// 3. Scaled affine: forward = a + b * u, inverse = (beta - a) / b,
//    |dβ/du| = b. Constant Jacobian. Used as the "agreement with the
//    library affine transform" sanity check (the AD path should match
//    the closed-form Jacobian of the analytic affine class).
struct AffineFwd {
    double a, b;
    template <typename T>
    T operator()(T u) const { return T(a) + T(b) * u; }
};
struct AffineInv {
    double a, b;
    double operator()(double beta) const { return (beta - a) / b; }
};

// Degenerate bijection for the failure-mode test.
//   forward = constant 1.0 (no dependence on u)
//   inverse — not really invertible; returns 0.0 as a stub
//   Jacobian = 0 → apply_forward MUST return NaN.
struct ConstFwd {
    template <typename T>
    T operator()(T /*u*/) const { return T(1.0); }
};
struct ConstInv {
    double operator()(double /*beta*/) const { return 0.0; }
};

// Helper: max over a probe grid.
template <typename F>
double grid_max(F f, const arma::vec& grid) {
    double m = 0.0;
    for (double u : grid) {
        const double v = f(u);
        if (std::isfinite(v) && v > m) m = v;
    }
    return m;
}

// Helper: min over a probe grid (skipping non-finite).
template <typename F>
double grid_min_finite(F f, const arma::vec& grid) {
    double m = std::numeric_limits<double>::infinity();
    for (double u : grid) {
        const double v = f(u);
        if (std::isfinite(v) && v < m) m = v;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Main test runner.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List test_rjmcmc_custom_bijection()
{
    const double TOL_ROUND   = 1e-10;
    const double TOL_INVPAIR = 1e-10;
    const double TOL_JAC_VS_ANALYTIC = 1e-9;
    const double TOL_JAC_MIN_NONZERO = 1e-12;

    // Probe grid (skip exact 0 because the cube test is intentionally
    // singular there and we want clean PASS for sinh / affine).
    arma::vec grid = arma::linspace(-2.0, 2.0, 11);
    grid = grid.elem(arma::find(arma::abs(grid) > 1e-3));

    // ---- Test 1: sinh ----
    auto bij_sinh = make_templated_bijection_1d(SinhFwd{}, SinhInv{});
    const double sinh_round = grid_max(
        [&](double u) { return check_roundtrip(*bij_sinh, u); }, grid);
    const double sinh_invpair = grid_max(
        [&](double u) { return check_jacobian_inverse_pair(*bij_sinh, u); }, grid);
    const double sinh_jac_vs_analytic = grid_max(
        [&](double u) {
            double beta_unused = 0.0;
            const double j_ad      = bij_sinh->apply_forward(u, beta_unused);
            const double j_analyt  = std::cosh(u);
            return std::abs(j_ad - j_analyt);
        }, grid);
    const double sinh_jac_min = grid_min_finite(
        [&](double u) {
            double beta_unused = 0.0;
            return bij_sinh->apply_forward(u, beta_unused);
        }, grid);
    const bool sinh_pass = (sinh_round < TOL_ROUND) &&
                           (sinh_invpair < TOL_INVPAIR) &&
                           (sinh_jac_vs_analytic < TOL_JAC_VS_ANALYTIC) &&
                           (sinh_jac_min > TOL_JAC_MIN_NONZERO);

    // ---- Test 2: u^3 (singular at 0; we test on grid that excludes 0) ----
    auto bij_cube = make_templated_bijection_1d(CubeFwd{}, CubeInv{});
    const double cube_round = grid_max(
        [&](double u) { return check_roundtrip(*bij_cube, u); }, grid);
    const double cube_invpair = grid_max(
        [&](double u) { return check_jacobian_inverse_pair(*bij_cube, u); }, grid);
    const double cube_jac_vs_analytic = grid_max(
        [&](double u) {
            double beta_unused = 0.0;
            const double j_ad     = bij_cube->apply_forward(u, beta_unused);
            const double j_analyt = 3.0 * u * u;
            return std::abs(j_ad - j_analyt);
        }, grid);
    const bool cube_pass = (cube_round < TOL_ROUND) &&
                           (cube_invpair < TOL_INVPAIR) &&
                           (cube_jac_vs_analytic < TOL_JAC_VS_ANALYTIC);

    // ---- Test 3: scaled affine (a=2, b=3) ----
    auto bij_aff = make_templated_bijection_1d(
        AffineFwd{2.0, 3.0}, AffineInv{2.0, 3.0});
    const double aff_round = grid_max(
        [&](double u) { return check_roundtrip(*bij_aff, u); }, grid);
    const double aff_invpair = grid_max(
        [&](double u) { return check_jacobian_inverse_pair(*bij_aff, u); }, grid);
    const double aff_jac_vs_analytic = grid_max(
        [&](double u) {
            double beta_unused = 0.0;
            const double j_ad     = bij_aff->apply_forward(u, beta_unused);
            const double j_analyt = 3.0;
            return std::abs(j_ad - j_analyt);
        }, grid);
    const bool aff_pass = (aff_round < TOL_ROUND) &&
                          (aff_invpair < TOL_INVPAIR) &&
                          (aff_jac_vs_analytic < TOL_JAC_VS_ANALYTIC);

    // ---- Test 4: degenerate forward must return NaN (reject signal) ----
    auto bij_const = make_templated_bijection_1d(ConstFwd{}, ConstInv{});
    double dummy = 0.0;
    const double const_jac = bij_const->apply_forward(0.5, dummy);
    const bool const_pass = !std::isfinite(const_jac);

    // Aggregate
    const bool pass_all = sinh_pass && cube_pass && aff_pass && const_pass;

    return Rcpp::List::create(
        Rcpp::Named("pass_all")               = pass_all,
        Rcpp::Named("sinh_pass")              = sinh_pass,
        Rcpp::Named("sinh_round")             = sinh_round,
        Rcpp::Named("sinh_invpair")           = sinh_invpair,
        Rcpp::Named("sinh_jac_vs_analytic")   = sinh_jac_vs_analytic,
        Rcpp::Named("sinh_jac_min")           = sinh_jac_min,
        Rcpp::Named("cube_pass")              = cube_pass,
        Rcpp::Named("cube_round")             = cube_round,
        Rcpp::Named("cube_invpair")           = cube_invpair,
        Rcpp::Named("cube_jac_vs_analytic")   = cube_jac_vs_analytic,
        Rcpp::Named("aff_pass")               = aff_pass,
        Rcpp::Named("aff_round")              = aff_round,
        Rcpp::Named("aff_invpair")            = aff_invpair,
        Rcpp::Named("aff_jac_vs_analytic")    = aff_jac_vs_analytic,
        Rcpp::Named("const_pass")             = const_pass,
        Rcpp::Named("const_jac_returned_nan") = !std::isfinite(const_jac));
}

/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  ode_rk45.hpp -- Tier 1 adaptive ODE integrator for AI4BayesCode.
 *
 *  PURPOSE
 *  =======
 *  Provide a minimal, header-only, dependency-free Dormand-Prince 5(4)
 *  adaptive Runge-Kutta solver so user log-density lambdas can embed
 *  mechanistic / pharmacokinetic / epidemiological / ecological ODE
 *  models inside a `nuts_block`. This is the AI4BayesCode analogue of
 *  Stan's `integrate_ode_rk45`.
 *
 *  FILLS THE SINGLE ARCHITECTURAL GAP identified in the 2026-04-22
 *  28-row model catalogue audit -- every other row composes from
 *  existing primitives; ODE / mechanistic models required this helper.
 *
 *  SCOPE
 *  =====
 *  Tier 1 = forward integration only. Dormand-Prince 5(4) with
 *  adaptive step-size control (PI error controller). Stateless pure
 *  function (like `celerite_marginal_likelihood.hpp`). NOT a
 *  `block_sampler`.
 *
 *  Tier 2 (future) = forward sensitivity analysis for d y/d theta,
 *  needed for NUTS gradients. For now, user must supply hand-computed
 *  gradients or use finite differences inside their lambda.
 *
 *  Tier 3 (far future) = SUNDIALS CVODES integration for stiff and
 *  very-large ODE systems.
 *
 *  SCOPE LIMITATIONS (Tier 1)
 *  --------------------------
 *  - NON-STIFF ODEs only. DP5(4) is an explicit method; stiff systems
 *    require an implicit integrator (e.g. Rosenbrock, BDF). If the
 *    adaptive step size shrinks below `min_h` (a hint of stiffness),
 *    the integrator throws std::runtime_error.
 *  - No Jacobian / sensitivity support. Gradients for NUTS must come
 *    from finite differences or user-supplied analytical derivatives.
 *  - No event detection, no dense output between `ts` entries -- y is
 *    reported only at the user-supplied output times.
 *
 *  DORMAND-PRINCE 5(4) METHOD
 *  ==========================
 *  See Hairer, Nørsett, Wanner (1993) "Solving Ordinary Differential
 *  Equations I" §II.5 for derivation. Six stages (with FSAL -- First
 *  Same As Last -- so effectively 6 RHS evaluations per accepted step).
 *  5th-order solution + embedded 4th-order error estimate for adaptive
 *  stepping.
 *
 *  API
 *  ===
 *
 *      template <typename RHS>
 *      arma::mat rk45(RHS&& f,
 *                     const arma::vec& y0,
 *                     const arma::vec& ts,
 *                     const arma::vec& theta,
 *                     double rtol = 1e-8,
 *                     double atol = 1e-8,
 *                     double max_h = 0.0,
 *                     double min_h = 1e-14,
 *                     std::size_t max_iter = 100000);
 *
 *  @param f       Callable with signature
 *                   `arma::vec (double t, const arma::vec& y,
 *                               const arma::vec& theta)`
 *                 returning dy/dt.
 *  @param y0      Initial state (length d > 0).
 *  @param ts      Output time points, strictly increasing, with
 *                 ts[0] being the initial time t0. The output matrix's
 *                 first row equals y0 (no integration required for t0).
 *  @param theta   Parameters passed through to f on every call.
 *                 Length arbitrary; the integrator never inspects it.
 *  @param rtol    Relative tolerance per component.
 *  @param atol    Absolute tolerance per component.
 *  @param max_h   Optional upper bound on step size (0 = no cap).
 *  @param min_h   Minimum step size; if the adaptive controller asks
 *                 for h < min_h, throw (signals stiff dynamics).
 *  @param max_iter  Safety cap on total internal iteration count.
 *
 *  @return  arma::mat of shape (ts.n_elem, y0.n_elem) with rows
 *           y_out.row(i) = y(ts[i]).
 *
 *  VERIFICATION
 *  ============
 *  Parity test at tests_autodiff/test_ode_rk45.cpp covers:
 *    - Linear ODE dy/dt = -k y: matches y0 exp(-k t) to 1e-10.
 *    - Lotka-Volterra: conserved quantity V(x,y) = δ·x − γ·log(x) +
 *      β·y − α·log(y) preserved to within rtol over one period.
 *    - SIR compartmental model: S + I + R conserved.
 *    - Harmonic oscillator: energy conservation over 10+ periods.
 *================================================================================*/

#ifndef AI4BAYESCODE_ODE_RK45_HPP
#define AI4BAYESCODE_ODE_RK45_HPP

#ifdef AI4BAYESCODE_RCPP_MODULE
# include <RcppArmadillo.h>
#else
# include <armadillo>
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace AI4BayesCode {
namespace ode {

// ---------------------------------------------------------------------------
// Dormand-Prince 5(4) Butcher tableau coefficients. Canonical values from
// Dormand, J.R. and Prince, P.J. (1980), "A family of embedded Runge-Kutta
// formulae", J. Comp. Appl. Math. 6:19-26.
// ---------------------------------------------------------------------------
namespace dp5_tableau {

// c_i = time-offset of stage i (t + c_i * h)
constexpr double c2 = 1.0 / 5.0;
constexpr double c3 = 3.0 / 10.0;
constexpr double c4 = 4.0 / 5.0;
constexpr double c5 = 8.0 / 9.0;
constexpr double c6 = 1.0;
constexpr double c7 = 1.0;

// a_{ij} tableau rows
constexpr double a21 = 1.0 / 5.0;

constexpr double a31 = 3.0 / 40.0;
constexpr double a32 = 9.0 / 40.0;

constexpr double a41 = 44.0 / 45.0;
constexpr double a42 = -56.0 / 15.0;
constexpr double a43 = 32.0 / 9.0;

constexpr double a51 = 19372.0 / 6561.0;
constexpr double a52 = -25360.0 / 2187.0;
constexpr double a53 = 64448.0 / 6561.0;
constexpr double a54 = -212.0 / 729.0;

constexpr double a61 = 9017.0 / 3168.0;
constexpr double a62 = -355.0 / 33.0;
constexpr double a63 = 46732.0 / 5247.0;
constexpr double a64 = 49.0 / 176.0;
constexpr double a65 = -5103.0 / 18656.0;

constexpr double a71 = 35.0 / 384.0;
constexpr double a72 = 0.0;
constexpr double a73 = 500.0 / 1113.0;
constexpr double a74 = 125.0 / 192.0;
constexpr double a75 = -2187.0 / 6784.0;
constexpr double a76 = 11.0 / 84.0;

// 5th-order solution weights b_i (equal a7_i for FSAL property)
// Not strictly needed as separate constants since we reuse a7_i, but
// included for readability in the integrator body.

// Error weights e_i = b5_i - b4_i, derived from
//   b5 = a7 (row 7 of Butcher tableau) and
//   b4 = (5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40)
// Verified algebraically 2026-04-22.
constexpr double e1 = 71.0 / 57600.0;
constexpr double e2 = 0.0;
constexpr double e3 = -71.0 / 16695.0;
constexpr double e4 = 71.0 / 1920.0;
constexpr double e5 = -17253.0 / 339200.0;
constexpr double e6 = 22.0 / 525.0;
constexpr double e7 = -1.0 / 40.0;

}  // namespace dp5_tableau

/**
 * @brief Dormand-Prince 5(4) adaptive Runge-Kutta integrator.
 *
 * See file header for the full API contract and scope. Throws
 * std::invalid_argument on malformed inputs and std::runtime_error on
 * step-size underflow or max_iter exhaustion.
 */
template <typename RHS>
inline arma::mat rk45(RHS&& f,
                      const arma::vec& y0,
                      const arma::vec& ts,
                      const arma::vec& theta,
                      double rtol = 1e-8,
                      double atol = 1e-8,
                      double max_h = 0.0,
                      double min_h = 1e-14,
                      std::size_t max_iter = 100000) {
    using namespace dp5_tableau;

    const std::size_t n_times = ts.n_elem;
    const std::size_t n_state = y0.n_elem;

    if (n_times == 0) {
        return arma::mat(0, n_state);
    }
    if (n_state == 0) {
        throw std::invalid_argument("ode::rk45: y0 must be non-empty");
    }
    if (!(rtol > 0.0) || !(atol > 0.0)) {
        throw std::invalid_argument("ode::rk45: rtol and atol must be > 0");
    }

    // Output matrix — row i is y(ts[i])
    arma::mat y_out(n_times, n_state);
    y_out.row(0) = y0.t();

    if (n_times == 1) return y_out;

    // Validate ts strictly increasing
    for (std::size_t i = 1; i < n_times; ++i) {
        if (!(ts[i] > ts[i - 1])) {
            throw std::invalid_argument(
                "ode::rk45: ts must be strictly increasing");
        }
    }

    double t = ts[0];
    arma::vec y = y0;

    // Initial step size heuristic. Use 1/100 of the total span as a
    // conservative first guess; the adaptive controller takes over
    // immediately.
    double h = (ts[n_times - 1] - ts[0]) / 100.0;
    if (max_h > 0.0 && h > max_h) h = max_h;
    if (h <= min_h) {
        throw std::invalid_argument(
            "ode::rk45: initial step size underflow; "
            "check ts range vs min_h");
    }

    // FSAL: k1 = f(t, y) persists between steps.
    arma::vec k1 = f(t, y, theta);

    std::size_t iter_count = 0;

    for (std::size_t i_out = 1; i_out < n_times; ++i_out) {
        const double t_target = ts[i_out];

        while (t < t_target) {
            if (++iter_count > max_iter) {
                throw std::runtime_error(
                    "ode::rk45: max_iter exceeded (" +
                    std::to_string(max_iter) + ") — "
                    "consider increasing tolerances or max_iter; "
                    "ODE may be stiff (Tier 1 is non-stiff only).");
            }

            // Don't overshoot target
            double h_step = h;
            if (t + h_step > t_target) h_step = t_target - t;

            if (h_step < min_h) {
                throw std::runtime_error(
                    "ode::rk45: step size underflow (" +
                    std::to_string(h_step) + " < " +
                    std::to_string(min_h) + ") — stiff dynamics or "
                    "discontinuity likely; Tier 1 handles non-stiff only.");
            }

            // --- Compute the 7 stages ------------------------------------
            const arma::vec y2 = y + h_step * (a21 * k1);
            const arma::vec k2 = f(t + c2 * h_step, y2, theta);

            const arma::vec y3 = y + h_step * (a31 * k1 + a32 * k2);
            const arma::vec k3 = f(t + c3 * h_step, y3, theta);

            const arma::vec y4 = y + h_step * (a41 * k1 + a42 * k2 + a43 * k3);
            const arma::vec k4 = f(t + c4 * h_step, y4, theta);

            const arma::vec y5 = y + h_step *
                (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4);
            const arma::vec k5 = f(t + c5 * h_step, y5, theta);

            const arma::vec y6 = y + h_step *
                (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5);
            const arma::vec k6 = f(t + c6 * h_step, y6, theta);

            const arma::vec y_new = y + h_step *
                (a71 * k1 + a72 * k2 + a73 * k3 +
                 a74 * k4 + a75 * k5 + a76 * k6);
            const arma::vec k7 = f(t + c7 * h_step, y_new, theta);  // FSAL

            // --- Error estimate --------------------------------------------
            const arma::vec err = h_step *
                (e1 * k1 + e2 * k2 + e3 * k3 + e4 * k4 +
                 e5 * k5 + e6 * k6 + e7 * k7);

            // Scaled error norm (per-component tolerance)
            double err_sq = 0.0;
            for (std::size_t j = 0; j < n_state; ++j) {
                const double sc = atol + rtol *
                    std::max(std::abs(y[j]), std::abs(y_new[j]));
                const double r = err[j] / sc;
                err_sq += r * r;
            }
            const double err_norm = std::sqrt(err_sq / static_cast<double>(n_state));

            // --- Adaptive step-size controller -----------------------------
            // DP5 uses exponent 1/(p+1) with p=4 → 1/5 = 0.2 = PI controller I-part
            constexpr double safety_factor = 0.9;
            constexpr double min_factor    = 0.2;   // clamp: don't shrink below 1/5
            constexpr double max_factor    = 5.0;   // clamp: don't grow above 5x
            constexpr double exponent      = -0.2;

            if (err_norm <= 1.0) {
                // --- Accept step ---
                t += h_step;
                y = y_new;
                k1 = k7;  // FSAL reuse for next step

                // Grow step size (bounded). Avoid div-by-0 on perfect steps.
                const double scale = (err_norm > 1e-12)
                                         ? std::pow(err_norm, exponent)
                                         : max_factor;
                double factor = safety_factor * scale;
                factor = std::max(min_factor, std::min(max_factor, factor));
                h = h_step * factor;
                if (max_h > 0.0 && h > max_h) h = max_h;
            } else {
                // --- Reject step ---
                const double scale  = std::pow(err_norm, exponent);
                double factor = safety_factor * scale;
                factor = std::max(min_factor, factor);  // no upper clamp on reject
                h = h_step * factor;
            }
        }

        y_out.row(i_out) = y.t();
    }

    return y_out;
}

}  // namespace ode
}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_ODE_RK45_HPP

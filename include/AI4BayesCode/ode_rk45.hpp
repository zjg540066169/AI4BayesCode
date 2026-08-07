/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v3.0 or later
 *  (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
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
 *  `block_sampler`. `rk45(...)`.
 *
 *  Tier 2 (SHIPPED 2026-07-27) = forward sensitivity analysis for the
 *  parameter-Jacobian S(t) = d y(t)/d theta, chained into the
 *  log-likelihood gradient. Replaces the (2p+1) finite-difference
 *  ODE re-solves per gradient with ONE augmented adaptive solve and
 *  yields ANALYTIC-QUALITY (non-noisy) gradients for NUTS. Public
 *  entry points `rk45_sens(...)` (autodiff RHS Jacobians, preferred),
 *  `rk45_sens_fd(...)` (finite-difference RHS Jacobians, no RHS
 *  templating required), and `sens_chain(...)`. See the TIER 2 API
 *  block below.
 *
 *  Tier 3 (far future) = SUNDIALS CVODES integration for stiff and
 *  very-large ODE systems.
 *
 *  SCOPE LIMITATIONS (Tier 1 AND Tier 2)
 *  -------------------------------------
 *  - NON-STIFF ODEs only. DP5(4) is an explicit method; stiff systems
 *    require an implicit integrator (e.g. Rosenbrock, BDF). If the
 *    adaptive step size shrinks below `min_h` (a hint of stiffness),
 *    the integrator throws std::runtime_error.
 *  - No event detection, no dense output between `ts` entries -- y is
 *    reported only at the user-supplied output times.
 *  - Tier 2 forward sensitivities are appropriate for small-to-moderate
 *    (n_state * n_param); the augmented system has n_state*(1+n_param)
 *    components. For very large systems adjoint sensitivity (Tier 3)
 *    scales better.
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
 *                     double rtol = 1e-6,
 *                     double atol = 1e-6,
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
 *  TIER 2 API -- FORWARD SENSITIVITY ANALYSIS
 *  ==========================================
 *  The forward sensitivity S(t) = d y(t)/d theta (n_state x n_param)
 *  satisfies the variational ODE
 *
 *      dS/dt = J_y(y,theta,t) * S + J_theta(y,theta,t),
 *      S(t0) = d y0/d theta   (0 if y0 does not depend on theta),
 *
 *  where J_y = df/dy (n_state x n_state) and J_theta = df/dtheta
 *  (n_state x n_param). rk45_sens integrates the AUGMENTED system
 *  [y ; vec(S)] in ONE adaptive DP5(4) solve, with the sensitivity
 *  block included in the adaptive error norm (Stan-style coupled error
 *  control), so both y(t) and S(t) are resolved to (rtol, atol).
 *
 *      struct rk45_sens_result {
 *          arma::mat              y;   // n_times x n_state (see NOTE)
 *          std::vector<arma::mat> S;   // S[i] = d y(ts[i])/d theta,
 *                                      //        shape n_state x n_param
 *          arma::uvec         theta_idx;  // theta columns S refers to
 *      };
 *
 *      // (A) PREFERRED: autodiff forward-mode dual Jacobians (machine
 *      //     precision; this is what Stan's coupled sensitivity does).
 *      //     `f` is the ordinary double RHS used for the y-trajectory
 *      //     (identical evaluation path to Tier-1 rk45); `f_ad` is its
 *      //     scalar-type-templated twin used ONLY to build J_y, J_theta:
 *      //         template <typename T>
 *      //         std::vector<T> f_ad(double t, const std::vector<T>& y,
 *      //                             const std::vector<T>& theta);
 *      template <typename RHS, typename RHS_AD>
 *      rk45_sens_result rk45_sens(RHS&& f, RHS_AD&& f_ad,
 *          const arma::vec& y0, const arma::vec& ts, const arma::vec& theta,
 *          const arma::uvec& theta_idx = arma::uvec(),   // empty = all theta
 *          double rtol = 1e-6, double atol = 1e-6,
 *          const arma::mat& S0 = arma::mat(),            // empty = zeros
 *          double max_h = 0.0, double min_h = 1e-14,
 *          std::size_t max_iter = 100000);
 *      // (only declared when <autodiff/forward/dual.hpp> is on the
 *      //  include path; define AI4BAYESCODE_ODE_SENS_NO_AUTODIFF to skip.)
 *
 *      // (B) ROBUST FALLBACK: central finite differences of the CHEAP
 *      //     RHS. No RHS templating. FD OF THE RHS is far more accurate
 *      //     than FD of the whole trajectory (~1e-9 vs the ODE-solve
 *      //     error floor) because there is no ODE-solve-error
 *      //     amplification -- it differences a smooth analytic function.
 *      template <typename RHS>
 *      rk45_sens_result rk45_sens_fd(RHS&& f,
 *          const arma::vec& y0, const arma::vec& ts, const arma::vec& theta,
 *          const arma::uvec& theta_idx = arma::uvec(),
 *          double rtol = 1e-6, double atol = 1e-6,
 *          const arma::mat& S0 = arma::mat(),
 *          double max_h = 0.0, double min_h = 1e-14,
 *          std::size_t max_iter = 100000);
 *
 *      // Chain the sensitivities into a parameter gradient. Given
 *      // dlp_dy (n_times x n_state) = d(log target)/d y at each output
 *      // time, returns d(log target)/d theta (length n_param):
 *      //     grad[j] = sum_i sum_k dlp_dy(i,k) * S[i](k,j).
 *      arma::vec sens_chain(const rk45_sens_result& r,
 *                           const arma::mat& dlp_dy);
 *
 *  JACOBIAN STRATEGY (why autodiff is the default)
 *  -----------------------------------------------
 *  The old pattern -- central FD of the whole TRAJECTORY -- differences
 *  two adaptive ODE solves, each carrying integration error ~rtol; the
 *  difference amplifies that error into a noisy gradient (relative noise
 *  ~ rtol/h), which degrades NUTS mixing and causes divergences. Tier 2
 *  removes this entirely: the Jacobians J_y, J_theta are of the cheap
 *  algebraic RHS, obtained either by autodiff (exact, path A) or by
 *  central FD of the RHS (~1e-9, path B) -- both far below the old noise
 *  floor -- and are propagated through ONE augmented solve. Autodiff is
 *  preferred for exactness and to match Stan; FD-of-RHS is the
 *  dependency-free fallback for RHS forms that are awkward to template.
 *
 *  NOTE on the y trajectory: because the sensitivity block participates
 *  in the adaptive error norm, rk45_sens may take slightly different
 *  steps than a bare Tier-1 rk45 call; y matches Tier-1 to within the
 *  requested tolerance (verified in the parity test), not bit-for-bit.
 *
 *  PERFORMANCE (MEASURED, not assumed -- do NOT expect a universal
 *  speedup). The augmented system has n_state*(1+n_param) components and
 *  every step also builds the RHS Jacobians, so for SMALL n_param the
 *  raw per-gradient wall time can be LARGER than the (2*n_param+1) plain
 *  solves it replaces. Measured on the SIR example (n_state=3,
 *  n_param=2, 15 output times, rtol=atol=1e-6), per gradient eval:
 *      old (2p+1)=5 FD solves : ~34 us
 *      rk45_sens    (autodiff): ~105 us   (0.32x -- a SLOWDOWN)
 *      rk45_sens_fd (FD-RHS)  : ~54 us    (0.63x)
 *  The value of Tier-2 at this size is NOT raw speed but (i) EXACT,
 *  DETERMINISTIC gradients -- no FD step size to tune, no differencing
 *  noise that worsens in hard regions of the posterior -- and (ii) a
 *  single solve regardless of n_param, so the ratio improves as n_param
 *  grows and/or each ODE solve is expensive (on a cheap linear n=p
 *  system the paths reach parity around n_param ~ 32). Rule of thumb:
 *  reach for Tier-2 for gradient QUALITY / robustness and larger p; if a
 *  plain FD-of-trajectory gradient already mixes well and p is small it
 *  may be faster. FD-of-RHS (path B) is ~2x faster than autodiff
 *  (path A) here, with equally clean (~1e-9) gradients.
 *
 *  VERIFICATION
 *  ============
 *  Parity test at tests_autodiff/test_ode_rk45.cpp covers:
 *    - Linear ODE dy/dt = -k y: matches y0 exp(-k t) to 1e-10.
 *    - Lotka-Volterra: conserved quantity V(x,y) = delta*x - gamma*log(x)
 *      + beta*y - alpha*log(y) preserved to within rtol over one period.
 *    - SIR compartmental model: S + I + R conserved.
 *    - Harmonic oscillator: energy conservation over 10+ periods.
 *  Tier-2 sensitivity test at tests_autodiff/test_ode_rk45_sens.cpp:
 *    - Linear ODE dy/dt = -k y: S = d/dk (y0 e^{-k t}) = -t y0 e^{-k t},
 *      matched analytically (both autodiff and FD paths).
 *    - Nonlinear multi-state ODEs (SIR, Lotka-Volterra): S matched to a
 *      tight-tolerance central-FD-of-trajectory reference to < 1e-4 rel.
 *    - autodiff vs FD-of-RHS paths agree; augmented y matches Tier-1 y.
 *================================================================================*/

#ifndef AI4BAYESCODE_ODE_RK45_HPP
#define AI4BAYESCODE_ODE_RK45_HPP

#ifdef AI4BAYESCODE_RCPP_MODULE
# include <RcppArmadillo.h>
#else
# include <armadillo>
#endif

// --- Tier-2 forward sensitivity: optional autodiff dependency ---------------
// The autodiff-based rk45_sens overload builds the RHS Jacobians with
// forward-mode dual numbers (vendored, header-only, under include/autodiff).
// Guard with __has_include so Tier-1 rk45 (and the FD-based rk45_sens_fd)
// stay completely dependency-free for callers that do NOT put autodiff on
// the include path -- the autodiff overload is then simply not declared.
// Define AI4BAYESCODE_ODE_SENS_NO_AUTODIFF to force-disable it.
#if defined(AI4BAYESCODE_ODE_SENS_NO_AUTODIFF)
    // autodiff explicitly disabled by the caller
#elif defined(__has_include)
#  if __has_include(<autodiff/forward/dual.hpp>)
#    include <autodiff/forward/dual.hpp>
#    define AI4BAYESCODE_ODE_HAVE_AUTODIFF 1
#  endif
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

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

// ---------------------------------------------------------------------------
// Hard accuracy floor for EVERY rk45* solve. rtol/atol are clamped so a solve
// can NEVER run LOOSER than this. Looser tolerances trade posterior accuracy
// for speed -- a generated log-density must not do that, and it also keeps
// AI-vs-Stan benchmarks apples-to-apples (Stan's integrate_ode_rk45 default is
// 1e-6). TIGHTER values pass through unchanged, so high-accuracy reference and
// data-simulation solves (e.g. 1e-9) still work. This is a fixed library
// constant -- NOT a per-call argument or codegen-settable knob; changing it is
// a deliberate edit here, not something an agent can loosen.
// ---------------------------------------------------------------------------
static constexpr double RK45_TOL_FLOOR = 1e-6;

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
                      double rtol = 1e-6,
                      double atol = 1e-6,
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
    // Accuracy floor: never integrate looser than RK45_TOL_FLOOR (tighter OK).
    if (rtol > RK45_TOL_FLOOR) rtol = RK45_TOL_FLOOR;
    if (atol > RK45_TOL_FLOOR) atol = RK45_TOL_FLOOR;

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

// ===========================================================================
//  TIER 2 -- forward sensitivity analysis
// ===========================================================================

/**
 * @brief Result of an rk45_sens / rk45_sens_fd augmented solve.
 *
 * `y` is the trajectory (identical layout to Tier-1 rk45: n_times x
 * n_state). `S[i]` is the parameter-Jacobian d y(ts[i])/d theta with
 * shape n_state x n_param. `theta_idx` records which components of the
 * full `theta` vector the columns of every S[i] correspond to (in order).
 */
struct rk45_sens_result {
    arma::mat              y;          // n_times x n_state
    std::vector<arma::mat> S;          // S[i] : n_state x n_param
    arma::uvec             theta_idx;  // tracked theta component indices
};

namespace detail {

// Resolve the effective list of tracked theta indices (empty -> all).
inline arma::uvec resolve_theta_idx(const arma::uvec& theta_idx,
                                    std::size_t n_theta) {
    if (theta_idx.n_elem == 0) {
        arma::uvec idx(n_theta);
        for (std::size_t j = 0; j < n_theta; ++j)
            idx[j] = static_cast<arma::uword>(j);
        return idx;
    }
    for (arma::uword j = 0; j < theta_idx.n_elem; ++j) {
        if (theta_idx[j] >= n_theta)
            throw std::invalid_argument(
                "ode::rk45_sens: theta_idx entry out of range");
    }
    return theta_idx;
}

// -------------------------------------------------------------------------
// Augmented DP5(4) solver shared by both public overloads. `f` supplies the
// double RHS dy = f(t, y, theta) (identical evaluation path to Tier-1 rk45);
// `jac(t, y, Jy, Jth)` fills Jy (n x n) = df/dy and Jth (n x p) = df/dtheta
// restricted to the tracked columns. The sensitivity block vec(S) is carried
// alongside y and INCLUDED in the adaptive error norm (Stan-style coupled
// error control), so both are resolved to (rtol, atol).
// -------------------------------------------------------------------------
template <typename RHS, typename JacFn>
inline rk45_sens_result rk45_sens_core(RHS&& f,
                                       JacFn&& jac,
                                       const arma::vec& y0,
                                       const arma::vec& ts,
                                       const arma::vec& theta,
                                       const arma::uvec& theta_idx_in,
                                       double rtol,
                                       double atol,
                                       const arma::mat& S0_in,
                                       double max_h,
                                       double min_h,
                                       std::size_t max_iter) {
    using namespace dp5_tableau;

    const std::size_t n_times = ts.n_elem;
    const std::size_t n_state = y0.n_elem;

    rk45_sens_result res;
    if (n_state == 0)
        throw std::invalid_argument("ode::rk45_sens: y0 must be non-empty");
    if (!(rtol > 0.0) || !(atol > 0.0))
        throw std::invalid_argument("ode::rk45_sens: rtol and atol must be > 0");
    // Accuracy floor: never integrate looser than RK45_TOL_FLOOR (tighter OK).
    if (rtol > RK45_TOL_FLOOR) rtol = RK45_TOL_FLOOR;
    if (atol > RK45_TOL_FLOOR) atol = RK45_TOL_FLOOR;

    const arma::uvec idx = resolve_theta_idx(theta_idx_in, theta.n_elem);
    const std::size_t p  = idx.n_elem;
    res.theta_idx = idx;

    // Initial sensitivity S(t0) = d y0/d theta (default zeros).
    arma::mat S(n_state, p, arma::fill::zeros);
    if (S0_in.n_elem != 0) {
        if (S0_in.n_rows != n_state || S0_in.n_cols != p)
            throw std::invalid_argument(
                "ode::rk45_sens: S0 must be n_state x n_param");
        S = S0_in;
    }

    if (n_times == 0) { res.y = arma::mat(0, n_state); return res; }

    res.y = arma::mat(n_times, n_state);
    res.y.row(0) = y0.t();
    res.S.assign(n_times, arma::mat(n_state, p));
    res.S[0] = S;
    if (n_times == 1) return res;

    for (std::size_t i = 1; i < n_times; ++i) {
        if (!(ts[i] > ts[i - 1]))
            throw std::invalid_argument(
                "ode::rk45_sens: ts must be strictly increasing");
    }

    double t = ts[0];
    arma::vec y = y0;

    double h = (ts[n_times - 1] - ts[0]) / 100.0;
    if (max_h > 0.0 && h > max_h) h = max_h;
    if (h <= min_h)
        throw std::invalid_argument(
            "ode::rk45_sens: initial step size underflow; "
            "check ts range vs min_h");

    // Augmented RHS evaluation: (kY, kS) at (t_eval, Y, SS).
    auto eval_aug = [&](double te, const arma::vec& Y, const arma::mat& SS,
                        arma::vec& kY, arma::mat& kS) {
        kY = f(te, Y, theta);
        arma::mat Jy, Jth;
        jac(te, Y, Jy, Jth);         // Jy: n x n, Jth: n x p
        kS = Jy * SS + Jth;          // n x p
    };

    // FSAL: k1 = augmented RHS at (t, y, S) persists between steps.
    arma::vec kY1; arma::mat kS1;
    eval_aug(t, y, S, kY1, kS1);

    std::size_t iter_count = 0;

    for (std::size_t i_out = 1; i_out < n_times; ++i_out) {
        const double t_target = ts[i_out];

        while (t < t_target) {
            if (++iter_count > max_iter)
                throw std::runtime_error(
                    "ode::rk45_sens: max_iter exceeded (" +
                    std::to_string(max_iter) + ") -- consider increasing "
                    "tolerances or max_iter; ODE may be stiff (non-stiff only).");

            double h_step = h;
            if (t + h_step > t_target) h_step = t_target - t;

            if (h_step < min_h)
                throw std::runtime_error(
                    "ode::rk45_sens: step size underflow (" +
                    std::to_string(h_step) + " < " + std::to_string(min_h) +
                    ") -- stiff dynamics or discontinuity likely; non-stiff only.");

            // --- 7 DP stages on the augmented state (y, S) -----------------
            arma::vec kY2, kY3, kY4, kY5, kY6, kY7;
            arma::mat kS2, kS3, kS4, kS5, kS6, kS7;

            const arma::vec Y2 = y + h_step * (a21 * kY1);
            const arma::mat S2 = S + h_step * (a21 * kS1);
            eval_aug(t + c2 * h_step, Y2, S2, kY2, kS2);

            const arma::vec Y3 = y + h_step * (a31 * kY1 + a32 * kY2);
            const arma::mat S3 = S + h_step * (a31 * kS1 + a32 * kS2);
            eval_aug(t + c3 * h_step, Y3, S3, kY3, kS3);

            const arma::vec Y4 = y + h_step * (a41 * kY1 + a42 * kY2 + a43 * kY3);
            const arma::mat S4 = S + h_step * (a41 * kS1 + a42 * kS2 + a43 * kS3);
            eval_aug(t + c4 * h_step, Y4, S4, kY4, kS4);

            const arma::vec Y5 = y + h_step *
                (a51 * kY1 + a52 * kY2 + a53 * kY3 + a54 * kY4);
            const arma::mat S5 = S + h_step *
                (a51 * kS1 + a52 * kS2 + a53 * kS3 + a54 * kS4);
            eval_aug(t + c5 * h_step, Y5, S5, kY5, kS5);

            const arma::vec Y6 = y + h_step *
                (a61 * kY1 + a62 * kY2 + a63 * kY3 + a64 * kY4 + a65 * kY5);
            const arma::mat S6 = S + h_step *
                (a61 * kS1 + a62 * kS2 + a63 * kS3 + a64 * kS4 + a65 * kS5);
            eval_aug(t + c6 * h_step, Y6, S6, kY6, kS6);

            const arma::vec y_new = y + h_step *
                (a71 * kY1 + a72 * kY2 + a73 * kY3 +
                 a74 * kY4 + a75 * kY5 + a76 * kY6);
            const arma::mat S_new = S + h_step *
                (a71 * kS1 + a72 * kS2 + a73 * kS3 +
                 a74 * kS4 + a75 * kS5 + a76 * kS6);
            eval_aug(t + c7 * h_step, y_new, S_new, kY7, kS7);  // FSAL

            // --- Embedded 4th-order error estimate -------------------------
            const arma::vec err_y = h_step *
                (e1 * kY1 + e2 * kY2 + e3 * kY3 + e4 * kY4 +
                 e5 * kY5 + e6 * kY6 + e7 * kY7);
            const arma::mat err_S = h_step *
                (e1 * kS1 + e2 * kS2 + e3 * kS3 + e4 * kS4 +
                 e5 * kS5 + e6 * kS6 + e7 * kS7);

            // Scaled error norm over the FULL augmented state.
            double err_sq = 0.0;
            for (std::size_t j = 0; j < n_state; ++j) {
                const double sc = atol + rtol *
                    std::max(std::abs(y[j]), std::abs(y_new[j]));
                const double r = err_y[j] / sc;
                err_sq += r * r;
            }
            for (std::size_t c = 0; c < p; ++c)
                for (std::size_t j = 0; j < n_state; ++j) {
                    const double sc = atol + rtol *
                        std::max(std::abs(S(j, c)), std::abs(S_new(j, c)));
                    const double r = err_S(j, c) / sc;
                    err_sq += r * r;
                }
            const double n_comp = static_cast<double>(n_state * (1 + p));
            const double err_norm = std::sqrt(err_sq / n_comp);

            // --- Adaptive step-size controller (identical to Tier-1) -------
            constexpr double safety_factor = 0.9;
            constexpr double min_factor    = 0.2;
            constexpr double max_factor    = 5.0;
            constexpr double exponent      = -0.2;

            if (err_norm <= 1.0) {
                t += h_step;
                y  = y_new;
                S  = S_new;
                kY1 = kY7;   // FSAL reuse
                kS1 = kS7;

                const double scale = (err_norm > 1e-12)
                                         ? std::pow(err_norm, exponent)
                                         : max_factor;
                double factor = safety_factor * scale;
                factor = std::max(min_factor, std::min(max_factor, factor));
                h = h_step * factor;
                if (max_h > 0.0 && h > max_h) h = max_h;
            } else {
                const double scale = std::pow(err_norm, exponent);
                double factor = safety_factor * scale;
                factor = std::max(min_factor, factor);
                h = h_step * factor;
            }
        }

        res.y.row(i_out) = y.t();
        res.S[i_out]     = S;
    }

    return res;
}

// Central-FD-of-RHS Jacobian builder. Differences the cheap algebraic RHS
// (NOT the trajectory), so there is no ODE-solve-error amplification.
template <typename RHS>
struct fd_rhs_jacobian {
    RHS f;
    const arma::vec& theta;
    arma::uvec idx;
    std::size_t p;

    void operator()(double te, const arma::vec& Y,
                    arma::mat& Jy, arma::mat& Jth) const {
        const std::size_t n = Y.n_elem;
        Jy.set_size(n, n);
        Jth.set_size(n, p);
        arma::vec Yp = Y, Ym = Y;
        for (std::size_t k = 0; k < n; ++k) {
            const double hk = 1e-6 * std::max(1.0, std::abs(Y[k]));
            Yp[k] = Y[k] + hk;
            Ym[k] = Y[k] - hk;
            const arma::vec fp = f(te, Yp, theta);
            const arma::vec fm = f(te, Ym, theta);
            Jy.col(k) = (fp - fm) / (2.0 * hk);
            Yp[k] = Y[k];
            Ym[k] = Y[k];
        }
        arma::vec th = theta;
        for (std::size_t c = 0; c < p; ++c) {
            const arma::uword tk = idx[c];
            const double hk = 1e-6 * std::max(1.0, std::abs(theta[tk]));
            th[tk] = theta[tk] + hk;
            const arma::vec fp = f(te, Y, th);
            th[tk] = theta[tk] - hk;
            const arma::vec fm = f(te, Y, th);
            th[tk] = theta[tk];
            Jth.col(c) = (fp - fm) / (2.0 * hk);
        }
    }
};

}  // namespace detail

/**
 * @brief Forward sensitivities via central FD of the RHS (robust fallback).
 *
 * No RHS templating required -- takes the same double RHS as Tier-1 rk45.
 * See the TIER 2 API block in the file header for the full contract.
 */
template <typename RHS>
inline rk45_sens_result rk45_sens_fd(RHS&& f,
                                     const arma::vec& y0,
                                     const arma::vec& ts,
                                     const arma::vec& theta,
                                     const arma::uvec& theta_idx = arma::uvec(),
                                     double rtol = 1e-6,
                                     double atol = 1e-6,
                                     const arma::mat& S0 = arma::mat(),
                                     double max_h = 0.0,
                                     double min_h = 1e-14,
                                     std::size_t max_iter = 100000) {
    const arma::uvec idx = detail::resolve_theta_idx(theta_idx, theta.n_elem);
    detail::fd_rhs_jacobian<RHS&> jac{f, theta, idx, idx.n_elem};
    return detail::rk45_sens_core(f, jac, y0, ts, theta, theta_idx,
                                  rtol, atol, S0, max_h, min_h, max_iter);
}

#ifdef AI4BAYESCODE_ODE_HAVE_AUTODIFF
/**
 * @brief Forward sensitivities via autodiff forward-mode dual (PREFERRED).
 *
 * `f` is the ordinary double RHS (used for the y-trajectory, identical
 * evaluation path to Tier-1 rk45). `f_ad` is its scalar-type-templated twin
 *     template <typename T>
 *     std::vector<T> f_ad(double t, const std::vector<T>& y,
 *                         const std::vector<T>& theta);
 * used ONLY to build J_y and J_theta at machine precision (one forward
 * dual pass per input direction, n_state + n_param passes per RHS Jacobian).
 * `f` and `f_ad` MUST encode the same dynamics.
 *
 * Only declared when <autodiff/forward/dual.hpp> is on the include path.
 */
template <typename RHS, typename RHS_AD>
inline rk45_sens_result rk45_sens(RHS&& f,
                                  RHS_AD&& f_ad,
                                  const arma::vec& y0,
                                  const arma::vec& ts,
                                  const arma::vec& theta,
                                  const arma::uvec& theta_idx = arma::uvec(),
                                  double rtol = 1e-6,
                                  double atol = 1e-6,
                                  const arma::mat& S0 = arma::mat(),
                                  double max_h = 0.0,
                                  double min_h = 1e-14,
                                  std::size_t max_iter = 100000) {
    using autodiff::dual;
    const arma::uvec idx = detail::resolve_theta_idx(theta_idx, theta.n_elem);
    const std::size_t p  = idx.n_elem;
    const std::size_t n_theta = theta.n_elem;

    auto jac_ad = [&](double te, const arma::vec& Y,
                      arma::mat& Jy, arma::mat& Jth) {
        const std::size_t n = Y.n_elem;
        Jy.set_size(n, n);
        Jth.set_size(n, p);
        std::vector<dual> yv(n), thv(n_theta);
        for (std::size_t i = 0; i < n; ++i)       yv[i]  = Y[i];
        for (std::size_t i = 0; i < n_theta; ++i) thv[i] = theta[i];

        // d f / d y : seed each state component in turn.
        for (std::size_t k = 0; k < n; ++k) {
            yv[k].grad = 1.0;
            std::vector<dual> dy = f_ad(te, yv, thv);
            if (dy.size() != n)
                throw std::runtime_error(
                    "ode::rk45_sens: f_ad returned wrong length");
            for (std::size_t i = 0; i < n; ++i) Jy(i, k) = dy[i].grad;
            yv[k].grad = 0.0;
        }
        // d f / d theta : seed each TRACKED parameter in turn.
        for (std::size_t c = 0; c < p; ++c) {
            const arma::uword tk = idx[c];
            thv[tk].grad = 1.0;
            std::vector<dual> dy = f_ad(te, yv, thv);
            for (std::size_t i = 0; i < n; ++i) Jth(i, c) = dy[i].grad;
            thv[tk].grad = 0.0;
        }
    };

    return detail::rk45_sens_core(f, jac_ad, y0, ts, theta, theta_idx,
                                  rtol, atol, S0, max_h, min_h, max_iter);
}
#endif  // AI4BAYESCODE_ODE_HAVE_AUTODIFF

/**
 * @brief Chain forward sensitivities into a parameter gradient.
 *
 * Given dlp_dy (n_times x n_state) = d(log target)/d y evaluated at each
 * output time, returns d(log target)/d theta (length n_param) via
 *     grad[j] = sum_i sum_k dlp_dy(i,k) * S[i](k,j).
 * Output times with no contribution should have a zero row in dlp_dy.
 */
inline arma::vec sens_chain(const rk45_sens_result& r,
                            const arma::mat& dlp_dy) {
    const std::size_t n_times = r.S.size();
    if (n_times == 0) return arma::vec();
    const std::size_t n_state = r.S[0].n_rows;
    const std::size_t p       = r.S[0].n_cols;
    if (dlp_dy.n_rows != n_times || dlp_dy.n_cols != n_state)
        throw std::invalid_argument(
            "ode::sens_chain: dlp_dy must be n_times x n_state");
    arma::vec grad(p, arma::fill::zeros);
    for (std::size_t i = 0; i < n_times; ++i)
        grad += r.S[i].t() * dlp_dy.row(i).t();
    return grad;
}


// ===========================================================================
//  ALLOCATION-FREE IN-PLACE VARIANTS (Tier-2 fast path)
// ===========================================================================
// rk45_inplace / rk45_sens_fd_inplace are allocation-free (raw-loop) twins of
// ode::rk45 and ode::rk45_sens_fd. The shipped arma rk45_sens_fd allocates
// several arma temporaries per Dormand-Prince stage; on a small system that can
// be SLOWER than plain central-FD of the whole solve (see the timing note in
// the file header). These variants carry the augmented state y + vec(S) in
// fixed-size stack buffers and evaluate an IN-PLACE RHS, removing all per-stage
// heap traffic. They are numerical drop-ins -- same Butcher tableau, same PI
// controller, same FD step (1e-6*max(1,|x|), matching detail::fd_rhs_jacobian),
// and (with coupled_error=true) the same coupled error norm as rk45_sens_core
// -- and reuse rk45_sens_result + sens_chain unchanged.
//
// IN-PLACE RHS signature (no arma dependency):
//     void f(double t, const double* y, const double* theta, double* dy)
//
// FD-of-RHS discipline (jacobian.md Sec.10): the LIBRARY central-differences the
// user's cheap RHS to build J_y = df/dy and J_theta = df/dtheta. Users / codegen
// never hand-write Jacobians; runtime autodiff stays validation-only.

// Compile-time stack-buffer caps. RK45_IP_MAXN bounds the BASE state dim and the
// full theta length; RK45_IP_MAXAUG bounds the AUGMENTED dim n_state*(1+n_param).
// Raise if a model needs more (each is a small stack-array size, not a heap cost).
static constexpr int RK45_IP_MAXN   = 64;
static constexpr int RK45_IP_MAXAUG = 128;

namespace detail {

// ---------------------------------------------------------------------------
// Core allocation-free raw-loop Dormand-Prince 5(4). Integrates ANY state whose
// derivative is supplied by the in-place RHS `f(t, a, theta, da)`. The adaptive
// error norm is taken over the first `n_err` components:
//   * n_err == n              -> plain base solve (identical to ode::rk45).
//   * n_err == n (aug system) -> STATE-ONLY control for an augmented state.
//   * n_err == aug_dim        -> COUPLED control (S in the norm; matches
//                                rk45_sens_core).
// Butcher tableau, initial-step heuristic and PI controller are identical to the
// shipped ode::rk45, so a base solve reproduces it to machine precision.
// ---------------------------------------------------------------------------
template <class RHS_IP>
inline arma::mat rk45_raw(RHS_IP&& f,
                          const arma::vec& a0,
                          const arma::vec& ts,
                          const double* theta,
                          int n_err,
                          double rtol,
                          double atol,
                          double max_h,
                          double min_h,
                          long max_iter) {
    using namespace dp5_tableau;

    const int n  = static_cast<int>(a0.n_elem);
    const int nt = static_cast<int>(ts.n_elem);
    if (n > RK45_IP_MAXAUG)
        throw std::invalid_argument("rk45_raw: augmented state dim exceeds RK45_IP_MAXAUG");
    if (!(rtol > 0.0) || !(atol > 0.0))
        throw std::invalid_argument("rk45_raw: rtol and atol must be > 0");
    // Accuracy floor: never integrate looser than RK45_TOL_FLOOR (tighter OK).
    if (rtol > RK45_TOL_FLOOR) rtol = RK45_TOL_FLOOR;
    if (atol > RK45_TOL_FLOOR) atol = RK45_TOL_FLOOR;

    arma::mat out(nt, n);
    for (int j = 0; j < n; ++j) out(0, j) = a0[j];
    if (nt == 1) return out;

    for (int i = 1; i < nt; ++i)
        if (!(ts[i] > ts[i - 1]))
            throw std::invalid_argument("rk45_raw: ts must be strictly increasing");

    double y[RK45_IP_MAXAUG], yn[RK45_IP_MAXAUG], ys[RK45_IP_MAXAUG];
    double k1[RK45_IP_MAXAUG], k2[RK45_IP_MAXAUG], k3[RK45_IP_MAXAUG], k4[RK45_IP_MAXAUG];
    double k5[RK45_IP_MAXAUG], k6[RK45_IP_MAXAUG], k7[RK45_IP_MAXAUG];
    for (int j = 0; j < n; ++j) y[j] = a0[j];
    if (n_err <= 0 || n_err > n) n_err = n;

    double t = ts[0];
    double h = (ts[nt - 1] - ts[0]) / 100.0;
    if (max_h > 0.0 && h > max_h) h = max_h;
    if (h <= min_h)
        throw std::invalid_argument("rk45_raw: initial step size underflow; check ts range vs min_h");

    long iter = 0;
    f(t, y, theta, k1);  // FSAL: k1 persists between accepted steps
    for (int io = 1; io < nt; ++io) {
        const double tt = ts[io];
        while (t < tt) {
            if (++iter > max_iter)
                throw std::runtime_error("rk45_raw: max_iter exceeded (stiff dynamics; non-stiff only)");
            double hs = h;
            if (t + hs > tt) hs = tt - t;
            if (hs < min_h)
                throw std::runtime_error("rk45_raw: step size underflow (stiff dynamics / discontinuity)");

            for (int j = 0; j < n; ++j) ys[j] = y[j] + hs * (a21 * k1[j]);
            f(t + c2 * hs, ys, theta, k2);
            for (int j = 0; j < n; ++j) ys[j] = y[j] + hs * (a31 * k1[j] + a32 * k2[j]);
            f(t + c3 * hs, ys, theta, k3);
            for (int j = 0; j < n; ++j) ys[j] = y[j] + hs * (a41 * k1[j] + a42 * k2[j] + a43 * k3[j]);
            f(t + c4 * hs, ys, theta, k4);
            for (int j = 0; j < n; ++j) ys[j] = y[j] + hs * (a51 * k1[j] + a52 * k2[j] + a53 * k3[j] + a54 * k4[j]);
            f(t + c5 * hs, ys, theta, k5);
            for (int j = 0; j < n; ++j) ys[j] = y[j] + hs * (a61 * k1[j] + a62 * k2[j] + a63 * k3[j] + a64 * k4[j] + a65 * k5[j]);
            f(t + c6 * hs, ys, theta, k6);
            for (int j = 0; j < n; ++j) yn[j] = y[j] + hs * (a71 * k1[j] + a72 * k2[j] + a73 * k3[j] + a74 * k4[j] + a75 * k5[j] + a76 * k6[j]);
            f(t + c7 * hs, yn, theta, k7);  // FSAL stage

            double es = 0.0;
            for (int j = 0; j < n_err; ++j) {
                const double e  = hs * (e1 * k1[j] + e2 * k2[j] + e3 * k3[j] + e4 * k4[j] + e5 * k5[j] + e6 * k6[j] + e7 * k7[j]);
                const double sc = atol + rtol * std::max(std::fabs(y[j]), std::fabs(yn[j]));
                const double r  = e / sc;
                es += r * r;
            }
            const double en = std::sqrt(es / static_cast<double>(n_err));

            constexpr double SF = 0.9, MN = 0.2, MX = 5.0, EX = -0.2;
            if (en <= 1.0) {
                t += hs;
                for (int j = 0; j < n; ++j) { y[j] = yn[j]; k1[j] = k7[j]; }
                const double sc = (en > 1e-12) ? std::pow(en, EX) : MX;
                h = hs * std::max(MN, std::min(MX, SF * sc));
                if (max_h > 0.0 && h > max_h) h = max_h;
            } else {
                h = hs * std::max(MN, SF * std::pow(en, EX));
            }
        }
        for (int j = 0; j < n; ++j) out(io, j) = y[j];
    }
    return out;
}

// ---------------------------------------------------------------------------
// FD-of-RHS augmented derivative. Over the augmented state
//     a = [ y(n) ; s_0(n) ; s_1(n) ; ... ; s_{p-1}(n) ],  s_c = d y / d theta_{sidx[c]},
// builds J_y = df/dy (central FD of the RHS) once, then for each tracked param c
// forms d(s_c)/dt = J_y * s_c + df/d theta_{sidx[c]} (the theta column also by
// central FD of the RHS). Allocation-free; FD step matches fd_rhs_jacobian.
// ---------------------------------------------------------------------------
template <class RHS_IP>
struct fd_aug_rhs {
    RHS_IP f;
    int n;         // base state dim
    int p;         // number of tracked params
    int n_theta;   // full theta length
    const int* sidx;  // length p: tracked theta component indices

    void operator()(double t, const double* a, const double* theta, double* da) const {
        double f0[RK45_IP_MAXN], yp[RK45_IP_MAXN], ym[RK45_IP_MAXN];
        double fp[RK45_IP_MAXN], fm[RK45_IP_MAXN], th2[RK45_IP_MAXN];
        double Jy[RK45_IP_MAXN][RK45_IP_MAXN];

        // Base derivative dy/dt = f(t, y, theta).
        f(t, a, theta, f0);
        for (int i = 0; i < n; ++i) da[i] = f0[i];

        // J_y = df/dy, central FD (step 1e-6 * max(1, |y_c|)).
        for (int c = 0; c < n; ++c) {
            const double hc = 1e-6 * std::max(1.0, std::fabs(a[c]));
            for (int i = 0; i < n; ++i) { yp[i] = a[i]; ym[i] = a[i]; }
            yp[c] = a[c] + hc; ym[c] = a[c] - hc;
            f(t, yp, theta, fp);
            f(t, ym, theta, fm);
            const double inv = 0.5 / hc;
            for (int i = 0; i < n; ++i) Jy[i][c] = (fp[i] - fm[i]) * inv;
        }

        // For each tracked param: d(s_c)/dt = J_y s_c + df/d theta_{sidx[c]}.
        for (int c = 0; c < n_theta; ++c) th2[c] = theta[c];
        for (int j = 0; j < p; ++j) {
            const int tj = sidx[j];
            const double hj = 1e-6 * std::max(1.0, std::fabs(theta[tj]));
            th2[tj] = theta[tj] + hj; f(t, a, th2, fp);
            th2[tj] = theta[tj] - hj; f(t, a, th2, fm);
            th2[tj] = theta[tj];
            const double inv = 0.5 / hj;
            const double* s = a + n + n * j;   // s_j
            double* ds      = da + n + n * j;  // d(s_j)/dt
            for (int i = 0; i < n; ++i) {
                double v = (fp[i] - fm[i]) * inv;         // (df/d theta_{tj})_i
                for (int k = 0; k < n; ++k) v += Jy[i][k] * s[k];  // (J_y s_j)_i
                ds[i] = v;
            }
        }
    }
};

}  // namespace detail

// ===========================================================================
//  Public: allocation-free base solve (in-place RHS)
// ===========================================================================
/**
 * @brief Allocation-free base ODE solve; numerical drop-in for ode::rk45.
 *
 * `f` is an in-place RHS `void f(double t, const double* y, const double* theta,
 * double* dy)`. Returns the trajectory as an (n_times x n_state) matrix, one row
 * per requested time. Same Butcher tableau / step control as ode::rk45.
 */
template <class RHS_IP>
inline arma::mat rk45_inplace(RHS_IP&& f,
                              const arma::vec& y0,
                              const arma::vec& ts,
                              const arma::vec& theta,
                              double rtol   = 1e-6,
                              double atol   = 1e-6,
                              double max_h  = 0.0,
                              double min_h  = 1e-14,
                              long   max_iter = 100000) {
    return detail::rk45_raw(std::forward<RHS_IP>(f), y0, ts, theta.memptr(),
                            static_cast<int>(y0.n_elem), rtol, atol,
                            max_h, min_h, max_iter);
}

// ===========================================================================
//  Public: allocation-free FD-of-RHS forward sensitivities
// ===========================================================================
/**
 * @brief Forward sensitivities via central FD of an IN-PLACE RHS, allocation-free.
 *
 * Numerical drop-in for ode::rk45_sens_fd (same rk45_sens_result, same
 * sens_chain). `f` is the in-place RHS; the library builds J_y and J_theta by
 * central FD of `f` (users never write Jacobians). The augmented state y + vec(S)
 * is carried in stack buffers; `S[i]` is d y(ts[i])/d theta restricted to the
 * tracked columns.
 *
 * @param coupled_error  true  -> S is in the adaptive error norm (drop-in for
 *                                rk45_sens_fd); false -> STATE-ONLY control.
 */
template <class RHS_IP>
inline rk45_sens_result rk45_sens_fd_inplace(RHS_IP&& f,
                                             const arma::vec& y0,
                                             const arma::vec& ts,
                                             const arma::vec& theta,
                                             const arma::uvec& theta_idx = arma::uvec(),
                                             double rtol = 1e-6,
                                             double atol = 1e-6,
                                             const arma::mat& S0 = arma::mat(),
                                             bool   coupled_error = true,
                                             double max_h = 0.0,
                                             double min_h = 1e-14,
                                             long   max_iter = 100000) {
    const int n = static_cast<int>(y0.n_elem);
    if (n == 0) throw std::invalid_argument("rk45_sens_fd_inplace: y0 must be non-empty");
    if (n > RK45_IP_MAXN)
        throw std::invalid_argument("rk45_sens_fd_inplace: n_state exceeds RK45_IP_MAXN");

    const arma::uvec idx = detail::resolve_theta_idx(theta_idx, theta.n_elem);
    const int p = static_cast<int>(idx.n_elem);
    const int aug = n * (1 + p);
    if (aug > RK45_IP_MAXAUG)
        throw std::invalid_argument("rk45_sens_fd_inplace: n_state*(1+n_param) exceeds RK45_IP_MAXAUG");
    if (static_cast<int>(theta.n_elem) > RK45_IP_MAXN)
        throw std::invalid_argument("rk45_sens_fd_inplace: theta length exceeds RK45_IP_MAXN");

    // Augmented initial state a0 = [ y0 ; vec(S0) ]  (S0 default = 0).
    arma::vec a0(aug, arma::fill::zeros);
    for (int j = 0; j < n; ++j) a0[j] = y0[j];
    if (S0.n_elem != 0) {
        if (static_cast<int>(S0.n_rows) != n || static_cast<int>(S0.n_cols) != p)
            throw std::invalid_argument("rk45_sens_fd_inplace: S0 must be n_state x n_param");
        for (int j = 0; j < p; ++j)
            for (int k = 0; k < n; ++k) a0[n + n * j + k] = S0(k, j);
    }

    std::vector<int> sidx(p);
    for (int j = 0; j < p; ++j) sidx[j] = static_cast<int>(idx[j]);

    detail::fd_aug_rhs<RHS_IP&> af{f, n, p, static_cast<int>(theta.n_elem), sidx.data()};
    const int n_err = coupled_error ? aug : n;
    const arma::mat A = detail::rk45_raw(af, a0, ts, theta.memptr(), n_err,
                                         rtol, atol, max_h, min_h, max_iter);

    // Reshape augmented trajectory into rk45_sens_result.
    const int nt = static_cast<int>(ts.n_elem);
    rk45_sens_result res;
    res.theta_idx = idx;
    res.y = A.cols(0, n - 1);
    res.S.assign(nt, arma::mat(n, p));
    for (int i = 0; i < nt; ++i)
        for (int j = 0; j < p; ++j)
            for (int k = 0; k < n; ++k)
                res.S[i](k, j) = A(i, n + n * j + k);
    return res;
}

}  // namespace ode
}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_ODE_RK45_HPP

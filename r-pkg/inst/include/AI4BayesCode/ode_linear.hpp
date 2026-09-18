/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v3.0 or later
 *  (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  ode_linear.hpp -- EXACT solution of the constant-coefficient linear ODE
 *
 *      y'(t) = A y(t) + b,       A (n x n) and b (n) INDEPENDENT of t,
 *
 *  with ANALYTIC parameter sensitivities d y(t)/d theta. Companion to the
 *  general integrator ode::rk45 (ode_rk45.hpp): same output layout, same
 *  result type for sensitivities, so ode::sens_chain consumes the result of
 *  linear_sens() unchanged.
 *
 *  PUBLIC API (namespace AI4BayesCode::ode)
 *  ========================================
 *    arma::mat        linear(A, b, y0, ts)
 *    rk45_sens_result linear_sens(A, b, y0, ts, dA, db, dy0)
 *    bool             linear_overflows(A, b, y0, t_span)
 *
 *  LAYOUT (identical to ode::rk45): row i of the trajectory is y(ts[i]);
 *  ts[0] is the initial time t0 and row 0 EQUALS y0 exactly. The solution is
 *  evaluated at tau_i = ts[i] - ts[0], never at ts[i] itself. Times need not
 *  be increasing and may be negative; each is evaluated independently from
 *  t0, so nothing accumulates along the grid.
 *
 *  METHOD
 *  ======
 *  n == 2  CLOSED FORM. With m = tr(A)/2 and N = A - mI, N^2 = s^2 I where
 *          s^2 = ((a11-a22)/2)^2 + a12 a21, so the exponential telescopes:
 *              exp(At) = e^{mt} [cosh(st) I + t sinhc(st) N].
 *          Everything is evaluated as a real-analytic function of the REAL
 *          pair p = m t, w = s^2 t^2 (w < 0 = complex eigenvalues, handled by
 *          cos / sinc with no complex arithmetic); s itself is never formed
 *          where it is small, so s -> 0 (a defective A) is a regular point of
 *          the code and of its derivative. The forcing enters by variation of
 *          constants, Psi(t) = int_0^t exp(Au) du = t g0 I + t^2 g1 N, with
 *          g1 = exp[0, x1, x2] the second divided difference of exp at the
 *          scaled eigenvalues x1,2 = p +- sqrt(w), evaluated by three branches
 *          (origin Taylor / confluent Taylor in w / direct). No matrix inverse:
 *          a singular A needs no special case. Sensitivities are forward-mode
 *          dual numbers carried through the same kernels (analytic chain
 *          rule, not finite differences), one sweep per parameter direction.
 *
 *          Two arithmetic points are load-bearing and are tested explicitly:
 *          (a) e^{mt} is NEVER applied as a separate factor. For |w| > 1 the
 *              kernels are formed from e^{x1} and e^{x2}, the exponentials of
 *              the actual eigenvalues, which are in range whenever the answer
 *              is (e^{mt} cosh(st) is 0 * inf on every stiff system).
 *          (b) s^2 and det(A) are formed with error-free transformations
 *              (fma two-product + two-sum), because d^2 + a12 a21 cancels
 *              completely for e.g. A = [[3, 1.7], [-5.2941176470588234, -3]]
 *              -- an O(1) matrix whose exact s^2 is 5.0e-16 but whose rounded
 *              sum is 0 -- and that error grows like t^2 in the trajectory
 *              and its derivative with nothing downstream able to repair it.
 *          Entries of A are pre-scaled by a power of two when they are
 *          extreme (|A| > 2^128 or < 2^-128) so that w = s^2 t^2 does not
 *          leave the double range at the SQUARE ROOT of it; the scaling is
 *          exact and leaves ordinary inputs bit-for-bit untouched.
 *
 *  n != 2  AUGMENTED PADE. The forcing is absorbed exactly by the augmented
 *          matrix Atil = [[A, b], [0, 0]] (size n+1), ytil = [y; 1], and
 *          exp(Atil t) is computed by scaling and squaring with a diagonal
 *          Pade approximant, Higham (2005) "The scaling and squaring method
 *          for the matrix exponential revisited", SIAM J. Matrix Anal. Appl.
 *          26(4), Algorithm 2.3 / Table 2.1: order m in {3,5,7,9,13} chosen
 *          by ||M||_1 against theta_m, and for ||M||_1 > theta_13 (= 4.25, Al-Mohy & Higham 2009) the 13-Pade
 *          on M/2^s followed by s squarings. The augmented column is balanced
 *          by an exact power-of-two similarity so that a forcing much larger
 *          than the dynamics does not drive the squaring count.
 *          Sensitivities are Van Loan (1978) block exponentials: the (1,j+1)
 *          block of exp([[Atil, E_1..E_p],[0, Atil, ..],...]) t is the
 *          Frechet derivative L(Atil t, E_j t), so ONE exponential of size
 *          (p+1)(n+1) gives every direction at a time (the off-diagonal
 *          blocks are nilpotent, N^2 = 0, so the Duhamel series terminates
 *          and this is exact). Above a size cap the per-direction 2(n+1)
 *          block form is used instead; the two are verified against each
 *          other. There is NO uniform-grid fast path: every output time is
 *          its own exponential of Atil (ts[i]-ts[0]), so the documented
 *          contract "row i = y(ts[i])" holds for every grid exactly.
 *
 *  DENSITY / GRADIENT CONSISTENCY
 *  ==============================
 *  The trajectory returned by linear() and the .y of linear_sens() are
 *  computed by the SAME code path on the same inputs and are bitwise
 *  identical (linear_sens takes only the derivative part from the dual /
 *  Frechet evaluation). A codegen may take the log-density from one and the
 *  gradient from the other without any lp/grad inconsistency.
 *
 *  ACCURACY CONTRACT (measured)
 *  ============================
 *  * Relative to |y(t)|_inf, the trajectory is accurate to eps * kappa where
 *    kappa = (largest term in the reconstruction) / |y|_inf; kappa is O(1)
 *    for stable systems (Re lambda <= 0) and the closed form is then within a
 *    few eps (measured 3.6e-14 worst over 528 cases against a 70-digit
 *    reference, at |lambda t| = 237 where exp(x) itself cannot beat |x| eps).
 *  * The general-n path loses roughly s = log2(||A t||_1 / 4.25) bits in the
 *    squaring phase; at ||A t||_1 = 2e6 that is six digits in the state and
 *    the Frechet block (squared alongside) is 2-3 orders worse than the
 *    state at the same norm. On a block whose M^2 lies below the rounding
 *    noise of its own products (eps ||M||^2 > ||M^2||: both off-diagonals
 *    nonzero with a12 a21 ~ -d^2) the squarings amplify that noise instead:
 *    measured 2e-8 at ||A t||_1 = 8e3 and 1e-2 at 8e5 for such a 2x2 block
 *    inside a 3x3. For n == 2 neither arises (closed form, no squaring).
 *  * S carries its own kappa_S = (largest derivative term) / |S|_inf; for a
 *    strongly non-normal N (||N|| >> |lambda|) it is of order ||N|| |y0| /
 *    |S|_inf, e.g. 2.5e-11 at ||N|| = 1e6 with |S| = O(1).
 *  * exp(x) cannot beat |x| eps relative accuracy because x itself is
 *    rounded; nothing here promises otherwise.
 *  * s^2 = ((a11-a22)/2)^2 + a12 a21 and det, and their directional
 *    derivatives, are formed with error-free transformations (two_sum /
 *    two_prod), so a12 a21 ~ -d^2 (a system that can oscillate, or is near
 *    a defective A) costs nothing: measured <= 1.1e-14 componentwise in y
 *    AND in S against a 120-digit reference, flat in t, including a pair
 *    whose a11 - a22 is itself inexact in double, and y0, b or a direction
 *    along the null direction (a12, -d) of A - m I (N v is compensated too).
 *
 *  THE ONE GENUINE LIMITATION
 *  ==========================
 *  SUBDOMINANT-COMPONENT CANCELLATION. y is reconstructed from terms of
 *  size e^{x_big}, so a component whose true value is ~e^{x_small} comes out
 *  by cancellation and carries ABSOLUTE accuracy eps * |y|_inf, i.e.
 *  RELATIVE accuracy ~ eps * e^{2q}, 2q = (lambda_max - lambda_min) t.
 *  Measured on A = diag(-1, -1-g), t = 10 (which isolates it exactly):
 *      2q = g t :     2        5        10       20       30       40
 *      rel.err(y2):   6.9e-16  9.7e-15  7.0e-13  6.1e-9   8.5e-5   1.0
 *      eps e^{2q}:    1.6e-15  3.3e-14  4.8e-12  1.1e-7   2.4e-3   5.2e1
 *  The law is confirmed; accuracy relative to |y|_inf stays <= 1.5e-16 in
 *  every one of those cases, and THAT is the guarantee. It bites when a
 *  likelihood takes log(y_i) or divides by y_i for a component that is
 *  exponentially below |y|_inf: in the lower-triangular cascade
 *  A = [[-k1, 0],[0.45 k1, -1e-3]] the fast component at k1 = 1e4, t = 50 is
 *  returned as 1.19999999999901e-4 against the true 1.2e-4 (8.3e-12 per
 *  component, 5.8e-12 on its sensitivity); at k1 = 50 it is 1.2e-15. The
 *  general-n path has the same property (it forms e^{At} as a matrix and is
 *  accurate relative to the matrix norm). No spectral/projector branch is
 *  provided: for dense A it only moves the cancellation into the projectors
 *  and is singular at s -> 0.
 *
 *  OVERFLOW. Once Re(lambda_max) (t - t0) exceeds ~709.78, or |y0|, |b|
 *  push an intermediate past DBL_MAX, the returned rows contain +-inf or NaN
 *  (no renormalisation is attempted; nothing throws). linear_overflows(A, b,
 *  y0, t_span) reports this in advance by running the single-time
 *  evaluation at t_span and testing finiteness -- it is exact by
 *  construction, from A, b, y0 AND t, and costs one evaluation.
 *
 *  SCOPE. A and b CONSTANT in t. A time-varying A(t) is not this solver's
 *  problem (exp(int A) is wrong unless A(t) commutes with itself at different
 *  times) -- use ode::rk45 for that.
 *================================================================================
 */
#ifndef AI4BAYESCODE_ODE_LINEAR_HPP
#define AI4BAYESCODE_ODE_LINEAR_HPP

#include "AI4BayesCode/ode_rk45.hpp"   // rk45_sens_result, sens_chain, armadillo

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace AI4BayesCode {
namespace ode {

namespace linear_detail {

// ===========================================================================
//  Validation (every public entry point, every n, including n == 1)
// ===========================================================================
inline void check_system(const arma::mat& A, const arma::vec& b,
                         const arma::vec& y0, const arma::vec& ts) {
    const arma::uword n = A.n_rows;
    if (n == 0)             throw std::invalid_argument("ode::linear: empty system (A is 0 x 0)");
    if (A.n_cols != n)      throw std::invalid_argument("ode::linear: A must be square");
    if (b.n_elem != n)      throw std::invalid_argument("ode::linear: b must have length n");
    if (y0.n_elem != n)     throw std::invalid_argument("ode::linear: y0 must have length n");
    if (ts.n_elem == 0)     throw std::invalid_argument("ode::linear: ts must have at least one element (ts[0] is t0)");
    if (!A.is_finite())     throw std::invalid_argument("ode::linear: A has a non-finite entry");
    if (!b.is_finite())     throw std::invalid_argument("ode::linear: b has a non-finite entry");
    if (!y0.is_finite())    throw std::invalid_argument("ode::linear: y0 has a non-finite entry");
    if (!ts.is_finite())    throw std::invalid_argument("ode::linear: ts has a non-finite entry");
}

inline void check_directions(arma::uword n,
                             const std::vector<arma::mat>& dA,
                             const std::vector<arma::vec>& db,
                             const std::vector<arma::vec>& dy0) {
    const std::size_t p = dA.size();
    if (db.size() != p || dy0.size() != p)
        throw std::invalid_argument(
            "ode::linear_sens: dA, db and dy0 must have the same length p "
            "(one entry per parameter; pass zeros for no dependence)");
    for (std::size_t j = 0; j < p; ++j) {
        const std::string J = "[" + std::to_string(j) + "]";
        if (dA[j].n_rows != n || dA[j].n_cols != n)
            throw std::invalid_argument("ode::linear_sens: dA" + J + " must be n x n");
        if (db[j].n_elem != n)
            throw std::invalid_argument("ode::linear_sens: db" + J + " must have length n");
        if (dy0[j].n_elem != n)
            throw std::invalid_argument("ode::linear_sens: dy0" + J + " must have length n");
        if (!dA[j].is_finite())
            throw std::invalid_argument("ode::linear_sens: dA" + J + " has a non-finite entry");
        if (!db[j].is_finite())
            throw std::invalid_argument("ode::linear_sens: db" + J + " has a non-finite entry");
        if (!dy0[j].is_finite())
            throw std::invalid_argument("ode::linear_sens: dy0" + J + " has a non-finite entry");
    }
}

// ===========================================================================
//  n == 2 closed form
// ===========================================================================

// Forward-mode dual number (value, directional derivative). The VALUE part of
// every operation is the same double operation the plain path performs; the
// derivative part is the chain rule.
struct dual {
    double v{0.0};
    double d{0.0};
    dual() = default;
    dual(double x) : v(x), d(0.0) {}          // NOLINT: implicit on purpose
    dual(double x, double dx) : v(x), d(dx) {}
};
inline dual operator+(const dual& a, const dual& b) { return dual(a.v + b.v, a.d + b.d); }
inline dual operator-(const dual& a, const dual& b) { return dual(a.v - b.v, a.d - b.d); }
inline dual operator-(const dual& a) { return dual(-a.v, -a.d); }
inline dual operator*(const dual& a, const dual& b) {
    return dual(a.v * b.v, a.d * b.v + a.v * b.d);
}
inline dual operator/(const dual& a, const dual& b) {
    const double q = a.v / b.v;
    return dual(q, (a.d - q * b.d) / b.v);
}
inline dual exp(const dual& a)   { const double e = std::exp(a.v);   return dual(e, e * a.d); }
inline dual expm1(const dual& a) { const double e = std::exp(a.v);   return dual(std::expm1(a.v), e * a.d); }
inline dual sqrt(const dual& a)  { const double s = std::sqrt(a.v);  return dual(s, a.d / (2.0 * s)); }
inline dual sin(const dual& a)   { return dual(std::sin(a.v),  std::cos(a.v) * a.d); }
inline dual cos(const dual& a)   { return dual(std::cos(a.v), -std::sin(a.v) * a.d); }

inline double value(const dual& a) { return a.v; }
inline double value(double a)      { return a; }
inline double deriv(const dual& a) { return a.d; }
inline double deriv(double)        { return 0.0; }

// E(x) = (e^x - 1)/x, entire, E(0) = 1. The series branch is required for the
// DERIVATIVE: d/dx of expm1(x)/x is (x e^x - e^x + 1)/x^2, whose numerator
// cancels to O(x^2).
template <class T>
inline T Efun(const T& x) {
    using std::expm1;
    if (std::abs(value(x)) < 0.5) {
        T term(1.0), sum(1.0);
        for (int k = 1; k <= 18; ++k) {          // 0.5^18/19! ~ 3e-23
            term = term * x / double(k + 1);
            sum  = sum + term;
        }
        return sum;
    }
    return expm1(x) / x;
}

// Stable pair of scaled eigenvalues x1 = p+q, x2 = p-q (q = sqrt(w) > 0),
// given dt = x1*x2 = det(A) t^2 supplied from the ENTRIES of A. p + q and
// p - q cancel catastrophically whenever one eigenvalue is much smaller in
// magnitude than the other (the stiff case; measured 1.6e-3 relative error on
// lambda_small for diag(-1e-6, -1e8)). Quadratic-formula rule: take the root
// that adds, get the other from the product. Nodes are returned unordered;
// every use below is symmetric in them.
template <class T>
inline void stable_nodes(const T& p, const T& q, const T& dt, T& xb, T& xs) {
    xb = (value(p) >= 0.0) ? T(p + q) : T(p - q);
    xs = dt / xb;                      // |xb| >= q > 0 in every caller
}

// Ec = e^p cosh(sqrt(w)),   Es = e^p sinhc(sqrt(w)).
//   |w| <= 1 : series in w (no sqrt, no cancellation; e^p times a factor in
//              [1, 1.55], so it is finite exactly when the answer is).
//   w  >  1  : (e^{x1} + e^{x2})/2 and (e^{x1} - e^{x2})/(x1 - x2), from the
//              eigenvalue exponentials themselves -- never e^p * cosh(q).
//   w  < -1  : e^p cos(r), e^p sin(r)/r, r = sqrt(-w); e^p IS the modulus.
// `force` selects a branch by hand (1 series, 2 real, 3 complex) so a test can
// evaluate two branches at the SAME point; 0 = automatic.
template <class T>
inline void kern_exp(const T& p, const T& w, const T& dt, T& Ec, T& Es,
                     int force = 0) {
    using std::exp;
    using std::sqrt;
    using std::sin;
    using std::cos;
    const double W = value(w);

    if (force == 1 || (force == 0 && std::abs(W) <= 1.0)) {
        T tc(1.0), tsc(1.0), C(1.0), S(1.0);
        for (int j = 1; j <= 14; ++j) {          // 1/28! ~ 3e-30
            tc  = tc  * w / double((2 * j - 1) * (2 * j));
            tsc = tsc * w / double((2 * j) * (2 * j + 1));
            C = C + tc;
            S = S + tsc;
        }
        const T ep = exp(p);
        Ec = ep * C;
        Es = ep * S;
    } else if (force == 2 || (force == 0 && W > 0.0)) {
        T xb, xs;
        stable_nodes(p, T(sqrt(w)), dt, xb, xs);
        const T eb = exp(xb);
        const T es = exp(xs);
        Ec = (eb + es) * 0.5;                    // symmetric in the two nodes
        Es = (eb - es) / (xb - xs);              // = exp[x1,x2]; |x1-x2| > 2
    } else {
        const T r  = sqrt(-w);
        const T ep = exp(p);
        Ec = ep * cos(r);
        Es = ep * sin(r) / r;
    }
}

// g0(p,w) = int_0^1 e^{pv} cosh(qv) dv,  g1(p,w) = int_0^1 v e^{pv} sinhc(qv) dv,
// q = sqrt(w). Three regimes on R = max(|x1|,|x2|), P = |p|, Q = sqrt(|w|):
//   (B1) R <= 2: Taylor at the origin via the real recursion
//        z_j = 2p z_{j-1} - (x1 x2) z_{j-2}, term count fixed by a rigorous
//        bound on R (a data-dependent break could stop on an h_j that vanishes
//        for complex nodes).
//   (B2) Q <= 0.25 max(1,P) and (p < 0 or Q <= 1): confluent eigenvalues,
//        Taylor in w at fixed p with E^{(k)}(p) = int_0^1 v^k e^{pv} dv from
//        E^{(k)} = (e^p - k E^{(k-1)})/p (|p| > 1.6 here). The k!/|p|^k error
//        growth of the recursion is cancelled by the 1/(2j+1)! weight: the
//        j-th term's error is eps |E| (w/p^2)^j / |p|, a decaying series.
//   (B3) otherwise (well separated): the divided differences directly; for
//        w < 0 the explicit real and imaginary parts of E(p + i r). Reached
//        only when Q > 0.25, so no division by a small q.
template <class T>
inline void kern_inhom(const T& p, const T& w, const T& dt, T& g0, T& g1,
                       int force = 0) {
    using std::exp;
    using std::sqrt;
    using std::sin;
    using std::cos;

    const double P  = std::abs(value(p));
    const double W  = value(w);
    const double AW = std::abs(W);
    const double Q  = std::sqrt(AW);
    const double R  = (W >= 0.0) ? (P + Q) : std::sqrt(P * P + AW);

    if (force == 1 || (force == 0 && R <= 2.0)) {
        // ---- (B1) Taylor at the origin -------------------------------------
        // h_j  = sum_{i=0}^{j} x1^i x2^{j-i}   -> g1 = sum h_j /(j+2)!
        // pw_j = (x1^j + x2^j)/2               -> g0 = sum pw_j/(j+1)!
        const T e1c = p * 2.0;          // x1 + x2
        const T e2c = dt;               // x1 * x2 = det(A) t^2, from the entries
        int J = 2;
        {
            double rp = R * R, f = 6.0;          // R^j, (j+2)! at j = 2
            while (J < 30 && (J + 1) * rp / f > 1e-19) {
                ++J;
                rp *= R;
                f *= double(J + 2);
            }
        }
        T h_prev(1.0), h_cur = e1c;              // h_0, h_1
        T w_prev(1.0), w_cur = p;                // pw_0, pw_1
        T G0 = T(1.0) + w_cur / 2.0;             // j = 0, 1
        T G1 = T(1.0) / 2.0 + h_cur / 6.0;
        double f0 = 2.0, f1 = 6.0;               // (j+1)!, (j+2)! at j = 1
        for (int j = 2; j <= J; ++j) {
            const T h_next = e1c * h_cur - e2c * h_prev;
            const T w_next = e1c * w_cur - e2c * w_prev;
            h_prev = h_cur;  h_cur = h_next;
            w_prev = w_cur;  w_cur = w_next;
            f0 *= double(j + 1);
            f1 *= double(j + 2);
            G0 = G0 + w_cur / f0;
            G1 = G1 + h_cur / f1;
        }
        g0 = G0;
        g1 = G1;
        return;
    }

    if (force == 2 ||
        (force == 0 && Q <= 0.25 * std::max(1.0, P) &&
         (value(p) < 0.0 || Q <= 1.0))) {
        // ---- (B2) confluent eigenvalues: Taylor in w at fixed p -------------
        const int JMAX = 30;
        const T ep = exp(p);
        T Elast = Efun(p);                        // E^{(0)}
        T G0 = Elast;
        Elast = (ep - Elast) / p;                 // E^{(1)}
        T G1 = Elast;
        T u0(1.0), u1(1.0);
        for (int j = 1; j <= JMAX; ++j) {
            const T E2j  = (ep - Elast * double(2 * j)) / p;
            const T E2j1 = (ep - E2j * double(2 * j + 1)) / p;
            Elast = E2j1;
            u0 = u0 * w / double((2 * j - 1) * (2 * j));
            u1 = u1 * w / double((2 * j) * (2 * j + 1));
            const T t0 = E2j * u0;
            const T t1 = E2j1 * u1;
            if (!std::isfinite(value(t0)) || !std::isfinite(value(t1))) break;
            G0 = G0 + t0;
            G1 = G1 + t1;
            if (std::abs(value(t0)) <= 1e-18 * std::abs(value(G0)) &&
                std::abs(value(t1)) <= 1e-18 * std::abs(value(G1))) break;
        }
        g0 = G0;
        g1 = G1;
        return;
    }

    // ---- (B3) well-separated eigenvalues -----------------------------------
    if (W >= 0.0) {
        T xb, xs;
        stable_nodes(p, T(sqrt(w)), dt, xb, xs);
        const T Eb = Efun(xb);
        const T Es_ = Efun(xs);
        g0 = (Eb + Es_) * 0.5;                   // symmetric
        g1 = (Eb - Es_) / (xb - xs);             // = E[x1,x2], symmetric
    } else {
        // x = p + i r;  E(x) = [(e^p cos r - 1) + i e^p sin r](p - i r)/(p^2+r^2)
        // and p^2 + r^2 = |x|^2 = x1 x2 = dt exactly (conjugate pair).
        const T r  = sqrt(-w);
        const T ep = exp(p);
        const T D  = dt;
        const T cr = cos(r), sr = sin(r);
        const T re = ep * cr - 1.0;         // Re(e^x) - 1
        const T im = ep * sr;               // Im(e^x)
        g0 = (re * p + im * r) / D;         // Re E(x)
        g1 = (im * p / r - re) / D;         // Im E(x) / r
    }
}

// ---- error-free transformations ------------------------------------------
inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
inline void two_prod(double a, double b, double& p, double& e) {
    p = a * b;
    e = std::fma(a, b, -p);
}

// s^2 = d^2 + a12 a21 (d = (a11 - a22)/2) and det = a11 a22 - a12 a21, each
// with the rounding errors of the products and of the cancelling sum carried
// explicitly. Both are then correct to a few eps RELATIVE TO THEIR OWN VALUE
// even when d^2 and -a12 a21 agree to all 53 bits (the rounded sum is exactly
// 0 there while the true value is O(eps d^2), and the trajectory inherits the
// difference as an error growing like t^2), down to |s^2| ~ eps^2 d^2, the
// resolution of a two-product expansion.
// Compensated dot product sum_i x_i y_i (Ogita-Rump-Oishi 2005, "Dot2"):
// every product and every partial sum carries its rounding error, so a sum
// whose terms cancel is returned to ~eps relative to the RESULT, not to the
// largest term. Used for the derivatives of s^2 and det, which cancel along
// exactly the directions their values do -- a scale direction dA = A gives
// d(s^2) = 2 s^2 -- and a rounded sum there would put eps * max|term| / |d s^2|
// into the sensitivity while the value stayed exact.
inline double dot_compensated(const double* x, const double* y, int k) {
    double hi = 0.0, lo = 0.0;
    for (int i = 0; i < k; ++i) {
        double ph, pl; two_prod(x[i], y[i], ph, pl);
        double sh, sl; two_sum(hi, ph, sh, sl);
        hi = sh; lo += sl + pl;
    }
    return hi + lo;
}

inline void s2_det_compensated(double a11, double a12, double a21, double a22,
                               double& s2, double& det) {
    double Dh, Dl;
    two_sum(a11, -a22, Dh, Dl);              // a11 - a22 = Dh + Dl exactly
    const double d  = 0.5 * Dh;              // exact
    const double dl = 0.5 * Dl;
    double P1, E1, P2, E2, P3, E3;
    two_prod(d, d, P1, E1);                  // d^2 = P1 + E1
    two_prod(a12, a21, P2, E2);              // a12 a21 = P2 + E2
    two_prod(a11, a22, P3, E3);              // a11 a22 = P3 + E3
    const double cross = 2.0 * d * dl + dl * dl;   // (d + dl)^2 - d^2
    double Sh, Sl;
    two_sum(P1, P2, Sh, Sl);
    s2 = Sh + (Sl + (E1 + E2 + cross));
    double Vh, Vl;
    two_sum(P3, -P2, Vh, Vl);
    det = Vh + (Vl + (E3 - E2));
}

// The 2x2 system after the trace/traceless split, pre-scaled by the exact
// power of two `c` (A/c, b/c, times t*c). T is double or dual.
template <class T>
struct Sys2 {
    T m, d, s2, det;     // m = tr/2, d = (a11-a22)/2, s^2, det -- of A/c
    T a12, a21;          // off-diagonals of A/c
    T b1, b2;            // b/c
    T y1, y2;            // y0
    double c;            // the scale
    double a11s, a22s;   // diagonal of A/c (for the det chain rule)
    double dh, dl;       // d = dh + dl exactly (two_sum halves of (a11 - a22)/2)
    double ddh{0.0}, ddl{0.0};   // dual system only: the halves of (d11 - d22)/2
};

// Choose the power-of-two scale for A: 1 for ordinary inputs (so the
// certified arithmetic is bit-for-bit unchanged), 2^ilogb(max|A|) when the
// entries are extreme, so that s^2 t^2 and det t^2 stay in range at the
// square root of the double range rather than overflow / underflow there.
inline double scale_2x2(const arma::mat& A, const arma::vec& b,
                        const arma::vec& ts) {
    const double amax = arma::abs(A).max();
    if (!(amax > 0.0)) return 1.0;
    if (amax < std::ldexp(1.0, 128) && amax > std::ldexp(1.0, -128)) return 1.0;
    const double c = std::ldexp(1.0, std::ilogb(amax));
    // b/c and t*c must remain finite; if they cannot, forgo the rescaling
    // (the unscaled evaluation is then the best available).
    for (arma::uword i = 0; i < 2; ++i)
        if (!std::isfinite(b(i) / c)) return 1.0;
    for (arma::uword i = 0; i < ts.n_elem; ++i)
        if (!std::isfinite((ts(i) - ts(0)) * c)) return 1.0;
    return c;
}

inline Sys2<double> make_sys2(const arma::mat& A, const arma::vec& b,
                              const arma::vec& y0, double c) {
    const double a11 = A(0, 0) / c, a12 = A(0, 1) / c,
                 a21 = A(1, 0) / c, a22 = A(1, 1) / c;
    Sys2<double> S;
    S.m = 0.5 * (a11 + a22);
    S.d = 0.5 * (a11 - a22);
    {   double Dh, Dl; two_sum(a11, -a22, Dh, Dl); S.dh = 0.5 * Dh; S.dl = 0.5 * Dl; }
    s2_det_compensated(a11, a12, a21, a22, S.s2, S.det);
    S.a12 = a12; S.a21 = a21;
    S.b1 = b(0) / c; S.b2 = b(1) / c;
    S.y1 = y0(0);    S.y2 = y0(1);
    S.c = c;
    S.a11s = a11;    S.a22s = a22;
    return S;
}

// Dual system for one parameter direction: values exactly those of the double
// system, derivatives by the (linear) chain rule through m, d, s^2, det.
inline Sys2<dual> make_sys2_dual(const Sys2<double>& S,
                                 const arma::mat& dA, const arma::vec& db,
                                 const arma::vec& dy0) {
    const double c = S.c;
    const double a11 = S.a11s, a22 = S.a22s;             // diagonal of A/c
    const double d11 = dA(0, 0) / c, d12 = dA(0, 1) / c,
                 d21 = dA(1, 0) / c, d22 = dA(1, 1) / c;
    const double dm = 0.5 * (d11 + d22);
    const double dd = 0.5 * (d11 - d22);
    Sys2<dual> D;
    D.m   = dual(S.m,  dm);
    D.d   = dual(S.d,  dd);
    {   // d(s^2) = 2 d dd + d12 a21 + a12 d21 ; d(det) = d11 a22 + a11 d22
        //                                              - d12 a21 - a12 d21
        // Both cancel along the same directions their values do; keep them to
        // eps relative to the result (see dot_compensated).  d and dd are the
        // rounded halves of a11 - a22 and d11 - d22, and that rounding is the
        // same size as the cancellation residual, so 2 d dd is expanded through
        // the exact two_sum halves: 2 d dd = 2 (dh + dl)(ddh + ddl).
        double DDh, DDl;
        two_sum(d11, -d22, DDh, DDl);
        D.dh = S.dh;  D.dl = S.dl;
        D.ddh = 0.5 * DDh;  D.ddl = 0.5 * DDl;
        const double xs[6] = {2.0 * S.dh, 2.0 * S.dh, 2.0 * S.dl, 2.0 * S.dl, d12,   S.a12};
        const double ys[6] = {D.ddh,      D.ddl,      D.ddh,      D.ddl,      S.a21, d21};
        const double xd[4] = {d11, a11, -d12,  -S.a12};
        const double yd[4] = {a22, d22, S.a21, d21};
        D.s2  = dual(S.s2,  dot_compensated(xs, ys, 6));
        D.det = dual(S.det, dot_compensated(xd, yd, 4));
    }
    D.a12 = dual(S.a12, d12);
    D.a21 = dual(S.a21, d21);
    D.b1  = dual(S.b1, db(0) / c);
    D.b2  = dual(S.b2, db(1) / c);
    D.y1  = dual(S.y1, dy0(0));
    D.y2  = dual(S.y2, dy0(1));
    D.c   = c;
    D.a11s = a11;  D.a22s = a22;
    return D;
}

// N v for N = A - m I = [[d, a12], [a21, -d]], with d = dh + dl (the exact
// two_sum halves) and the sum carried by Dot2. v = (a12, -d) is the null
// direction of N as s -> 0; there the plain products would leave an absolute
// error eps |d| |v| against a true O(s^2), which the term t Es (N v) turns
// into eps |d| t relative to |y|_inf.
inline void nvec(const Sys2<double>& S, double v1, double v2, double& r1, double& r2) {
    const double x1[3] = {S.dh, S.dl, S.a12},   z1[3] = {v1, v1, v2};
    const double x2[3] = {S.a21, -S.dh, -S.dl}, z2[3] = {v1, v2, v2};
    r1 = dot_compensated(x1, z1, 3);
    r2 = dot_compensated(x2, z2, 3);
}
// Dual: d(N v) = dN v + N dv with dN = [[dd, d12], [d21, -dd]], dd = ddh + ddl.
inline void nvec(const Sys2<dual>& S, const dual& v1, const dual& v2, dual& r1, dual& r2) {
    const double x1[3]  = {S.dh, S.dl, S.a12.v},   z1[3]  = {v1.v, v1.v, v2.v};
    const double x2[3]  = {S.a21.v, -S.dh, -S.dl}, z2[3]  = {v1.v, v2.v, v2.v};
    const double x1d[6] = {S.ddh, S.ddl, S.a12.d, S.dh, S.dl, S.a12.v};
    const double z1d[6] = {v1.v,  v1.v,  v2.v,    v1.d, v1.d, v2.d};
    const double x2d[6] = {S.a21.d, -S.ddh, -S.ddl, S.a21.v, -S.dh, -S.dl};
    const double z2d[6] = {v1.v,    v2.v,   v2.v,   v1.d,    v2.d,  v2.d};
    r1 = dual(dot_compensated(x1, z1, 3), dot_compensated(x1d, z1d, 6));
    r2 = dual(dot_compensated(x2, z2, 3), dot_compensated(x2d, z2d, 6));
}

// y(t) = Ec y0 + t Es (N y0) + t g0 b + t^2 g1 (N b), at ONE time offset tau
// (already scaled by c). T = double gives the trajectory; T = dual gives the
// trajectory in the value part and the directional derivative in the
// derivative part.
template <class T>
inline void eval2_at(const Sys2<T>& S, double tau, T& o1, T& o2) {
    const double tau2 = tau * tau;
    const T p  = S.m * tau;
    const T w  = S.s2 * tau2;
    const T dt = S.det * tau2;          // x1 x2, from the ENTRIES (not p^2 - w)

    T Ec, Es, g0, g1;
    kern_exp(p, w, dt, Ec, Es);
    kern_inhom(p, w, dt, g0, g1);

    // N = A - m I = [[d, a12], [a21, -d]], products compensated (see nvec)
    T Ny1, Ny2, Nb1, Nb2;
    nvec(S, S.y1, S.y2, Ny1, Ny2);
    nvec(S, S.b1, S.b2, Nb1, Nb2);

    o1 = Ec * S.y1 + Es * (Ny1 * tau) + g0 * (S.b1 * tau) + g1 * (Nb1 * tau2);
    o2 = Ec * S.y2 + Es * (Ny2 * tau) + g0 * (S.b2 * tau) + g1 * (Nb2 * tau2);
}

// The trajectory (this is THE value path; linear_sens reuses it verbatim).
inline arma::mat linear_2x2(const arma::mat& A, const arma::vec& b,
                            const arma::vec& y0, const arma::vec& ts) {
    const double c = scale_2x2(A, b, ts);
    const Sys2<double> S = make_sys2(A, b, y0, c);
    arma::mat Y(ts.n_elem, 2);
    Y(0, 0) = y0(0);
    Y(0, 1) = y0(1);
    for (arma::uword k = 1; k < ts.n_elem; ++k) {
        double o1, o2;
        eval2_at(S, (ts(k) - ts(0)) * c, o1, o2);
        Y(k, 0) = o1;
        Y(k, 1) = o2;
    }
    return Y;
}

inline rk45_sens_result linear_sens_2x2(const arma::mat& A, const arma::vec& b,
                                        const arma::vec& y0, const arma::vec& ts,
                                        const std::vector<arma::mat>& dA,
                                        const std::vector<arma::vec>& db,
                                        const std::vector<arma::vec>& dy0) {
    const std::size_t p = dA.size();
    const arma::uword n_times = ts.n_elem;
    rk45_sens_result res;
    res.y = linear_2x2(A, b, y0, ts);                 // bitwise the linear() path
    res.S.assign(n_times, arma::mat(2, p, arma::fill::zeros));
    res.theta_idx = (p == 0) ? arma::uvec()
                             : arma::regspace<arma::uvec>(0, static_cast<arma::uword>(p) - 1);

    const double c = scale_2x2(A, b, ts);
    const Sys2<double> S = make_sys2(A, b, y0, c);
    for (std::size_t j = 0; j < p; ++j) {
        const Sys2<dual> D = make_sys2_dual(S, dA[j], db[j], dy0[j]);
        res.S[0](0, j) = dy0[j](0);
        res.S[0](1, j) = dy0[j](1);
        for (arma::uword k = 1; k < n_times; ++k) {
            dual o1, o2;
            eval2_at(D, (ts(k) - ts(0)) * c, o1, o2);
            res.S[k](0, j) = o1.d;
            res.S[k](1, j) = o2.d;
        }
    }
    return res;
}

// ===========================================================================
//  General n: augmented Pade scaling-and-squaring
// ===========================================================================

// Diagonal Pade numerator coefficients normalised to b[0] = 1,
//     p_m(x) = sum_{k=0}^m b_k x^k,   b_k = (2m-k)! m! / ((2m)! k! (m-k)!),
// generated by the exact recurrence b_k = b_{k-1} (m-k+1) / (k (2m-k+1)).
// (The normalised b_k are not all dyadic -- b_2 = 3/25 for m = 13 -- so the
// recurrence carries up to 4e-16 relative rounding per coefficient; measured
// against the exact rationals on 4000 matrices at s = 0 the effect on the
// exponential is <= 5e-15, inside the solve's own rounding.)
inline void pade_coeffs(int m, double* b) {
    b[0] = 1.0;
    for (int k = 1; k <= m; ++k)
        b[k] = b[k - 1] * static_cast<double>(m - k + 1) /
               (static_cast<double>(k) * static_cast<double>(2 * m - k + 1));
}

// r_m(M) = (V - U)^{-1} (V + U) for m in {3,5,7,9}: U odd, V even.
inline arma::mat pade_small(const arma::mat& M, int m) {
    const arma::uword n = M.n_rows;
    double b[10];
    pade_coeffs(m, b);
    const arma::mat I  = arma::eye<arma::mat>(n, n);
    const arma::mat M2 = M * M;
    arma::mat Ueven = b[1] * I;
    arma::mat V     = b[0] * I;
    arma::mat Mpow  = I;
    for (int j = 1; 2 * j <= m; ++j) {
        Mpow  = Mpow * M2;
        V     += b[2 * j] * Mpow;
        if (2 * j + 1 <= m) Ueven += b[2 * j + 1] * Mpow;
    }
    const arma::mat U = M * Ueven;
    arma::mat X;
    if (!arma::solve(X, V - U, V + U))
        throw std::runtime_error("ode::linear: Pade denominator solve failed");
    return X;
}

// Higham (2005) eqns (3.5)-(3.6): the m = 13 approximant through M2, M4, M6.
inline arma::mat pade13(const arma::mat& M) {
    const arma::uword n = M.n_rows;
    double b[14];
    pade_coeffs(13, b);
    const arma::mat I  = arma::eye<arma::mat>(n, n);
    const arma::mat M2 = M * M;
    const arma::mat M4 = M2 * M2;
    const arma::mat M6 = M2 * M4;
    const arma::mat U = M * (M6 * (b[13] * M6 + b[11] * M4 + b[9] * M2)
                             + b[7] * M6 + b[5] * M4 + b[3] * M2 + b[1] * I);
    const arma::mat V = M6 * (b[12] * M6 + b[10] * M4 + b[8] * M2)
                        + b[6] * M6 + b[4] * M4 + b[2] * M2 + b[0] * I;
    arma::mat X;
    if (!arma::solve(X, V - U, V + U))
        throw std::runtime_error("ode::linear: Pade denominator solve failed");
    return X;
}

// exp(M), Higham (2005) Algorithm 2.3. A non-finite M (an overflowed A*tau)
// yields a NaN-filled result rather than an exception, so that overflow is
// signalled the same way on every path: by a non-finite output row.
inline arma::mat expm(const arma::mat& M) {
    const arma::uword n = M.n_rows;
    if (M.n_cols != n)
        throw std::invalid_argument("ode::linear: expm needs a square matrix");
    if (n == 0) return arma::mat();
    if (!M.is_finite()) {
        arma::mat X(n, n);
        X.fill(std::numeric_limits<double>::quiet_NaN());
        return X;
    }
    if (n == 1) { arma::mat X(1, 1); X(0, 0) = std::exp(M(0, 0)); return X; }

    const double nrm = arma::norm(M, 1);
    if (nrm == 0.0) return arma::eye<arma::mat>(n, n);

    // Higham (2005) Table 2.1: largest ||M||_1 with m-Pade backward error <= u.
    static const double theta[4]  = {1.495585217958292e-2,   // m = 3
                                     2.539398330063230e-1,   // m = 5
                                     9.504178996162932e-1,   // m = 7
                                     2.097847961257068e+0};  // m = 9
    static const int    order[4]  = {3, 5, 7, 9};
    // Al-Mohy & Higham 2009 (SIAM J. Matrix Anal. Appl. 31) lower theta_13 from
    // Higham 2005's 5.37 to 4.25: in (4.25, 5.37] the unscaled 13-Pade error
    // was measured at ~100 eps on this header, vs ~1 eps below 2.1.
    static const double theta13   = 4.25e+0;

    for (int i = 0; i < 4; ++i)
        if (nrm <= theta[i]) return pade_small(M, order[i]);

    int s = static_cast<int>(std::ceil(std::log2(nrm / theta13)));
    if (s < 0) s = 0;
    arma::mat X = pade13(M / std::ldexp(1.0, s));
    for (int k = 0; k < s; ++k) X = X * X;
    return X;
}

// Van Loan (1978): exp(M) and the Frechet derivative L(M,E) = d/d eps
// exp(M + eps E)|_0 from one exponential of [[M, E],[0, M]]. E is normalised
// to unit 1-norm (L is linear in E) so the scaling parameter is set by M.
inline void expm_frechet(const arma::mat& M, const arma::mat& E,
                         arma::mat& expM, arma::mat& L) {
    const arma::uword n = M.n_rows;
    const double en = arma::norm(E, 1);
    if (en == 0.0) {
        expM = expm(M);
        L.zeros(n, n);
        return;
    }
    arma::mat C(2 * n, 2 * n, arma::fill::zeros);
    C.submat(0, 0, n - 1, n - 1)         = M;
    C.submat(n, n, 2 * n - 1, 2 * n - 1) = M;
    C.submat(0, n, n - 1, 2 * n - 1)     = E / en;
    const arma::mat X = expm(C);
    expM = X.submat(0, 0, n - 1, n - 1);
    L    = en * X.submat(0, n, n - 1, 2 * n - 1);
}

// BALANCING THE AUGMENTED COLUMN. ||[[A, b],[0,0]]||_1 = max(||A||_1,
// ||b||_1), so a forcing much larger than the dynamics inflates the norm that
// drives scaling and squaring (measured: ||b||/||A|| = 1e12 needs s = 35
// squarings and loses six digits). With D = diag(I_n, sigma),
//     D^{-1} [[A, b],[0,0]] D = [[A, sigma b],[0,0]],
// and y(t) = head_n(exp(C t) [y0 ; 1/sigma]). sigma ~ ||A||_1/||b||_1 rounded
// to a POWER OF TWO, so the similarity is exact in binary.
inline double aug_scale(const arma::mat& A, const arma::vec& b) {
    const double na = arma::norm(A, 1);
    const double nb = arma::norm(b, 1);
    if (!(na > 0.0) || !(nb > 0.0)) return 1.0;
    const double r = na / nb;
    if (!std::isfinite(r) || r <= 0.0) return 1.0;
    double e = std::round(std::log2(r));
    if (e >  500.0) e =  500.0;
    if (e < -500.0) e = -500.0;
    const double sig = std::ldexp(1.0, static_cast<int>(e));
    return (std::isfinite(sig) && sig > 0.0) ? sig : 1.0;
}

// Atil = [[A, b], [0, 0]] of size n+1.
inline arma::mat augment(const arma::mat& A, const arma::vec& b) {
    const arma::uword n = A.n_rows;
    arma::mat At(n + 1, n + 1, arma::fill::zeros);
    At.submat(0, 0, n - 1, n - 1) = A;
    At.submat(0, n, n - 1, n)     = b;
    return At;
}

struct AugSys {
    arma::mat Atil;      // [[A, sig b],[0,0]]
    arma::vec ytil;      // [y0; 1/sig]
    double    sig;
    arma::uword n;
};

inline AugSys make_aug(const arma::mat& A, const arma::vec& b, const arma::vec& y0) {
    AugSys S;
    S.n    = A.n_rows;
    S.sig  = aug_scale(A, b);
    S.Atil = augment(A, S.sig * b);
    S.ytil.set_size(S.n + 1);
    S.ytil.head(S.n) = y0;
    S.ytil(S.n)      = 1.0 / S.sig;
    return S;
}

// y(t0 + tau) -- THE value path of the general-n case; linear_sens reuses it.
inline arma::vec y_general_at(const AugSys& S, double tau) {
    const arma::vec z = expm(S.Atil * tau) * S.ytil;
    return z.head(S.n);
}

inline arma::mat linear_general(const arma::mat& A, const arma::vec& b,
                                const arma::vec& y0, const arma::vec& ts) {
    const AugSys S = make_aug(A, b, y0);
    const arma::uword n = A.n_rows;
    arma::mat out(ts.n_elem, n);
    out.row(0) = y0.t();
    for (arma::uword i = 1; i < ts.n_elem; ++i)
        out.row(i) = y_general_at(S, ts(i) - ts(0)).t();
    return out;
}

// [[Atil, E_1 .. E_p], [0, Atil, ..], ...] * t with the E_j divided by escale.
inline arma::mat stacked_block(const arma::mat& Atil,
                               const std::vector<arma::mat>& Etil,
                               double t, double escale) {
    const arma::uword m = Atil.n_rows;
    const std::size_t p = Etil.size();
    const arma::uword N = m * static_cast<arma::uword>(p + 1);
    arma::mat M(N, N, arma::fill::zeros);
    for (std::size_t j = 0; j <= p; ++j) {
        const arma::uword o = m * static_cast<arma::uword>(j);
        M.submat(o, o, o + m - 1, o + m - 1) = Atil * t;
    }
    for (std::size_t j = 0; j < p; ++j) {
        const arma::uword c = m * static_cast<arma::uword>(j + 1);
        M.submat(0, c, m - 1, c + m - 1) = (Etil[j] * t) / escale;
    }
    return M;
}

inline double direction_scale(const std::vector<arma::mat>& Etil, double t) {
    double e = 0.0;
    for (const auto& E : Etil) e = std::max(e, arma::norm(E, 1));
    e *= std::fabs(t);
    return (e > 0.0) ? e : 1.0;
}

// Size of the stacked block matrix above which the per-direction Van Loan
// form (p exponentials of size 2(n+1)) is cheaper than one of (p+1)(n+1).
constexpr arma::uword kStackedMax = 48;

inline rk45_sens_result linear_sens_general(const arma::mat& A, const arma::vec& b,
                                            const arma::vec& y0, const arma::vec& ts,
                                            const std::vector<arma::mat>& dA,
                                            const std::vector<arma::vec>& db,
                                            const std::vector<arma::vec>& dy0) {
    const arma::uword n = A.n_rows, m = n + 1, n_times = ts.n_elem;
    const std::size_t p = dA.size();

    rk45_sens_result res;
    res.y = linear_general(A, b, y0, ts);              // bitwise the linear() path
    res.S.assign(n_times, arma::mat(n, p, arma::fill::zeros));
    res.theta_idx = (p == 0) ? arma::uvec()
                             : arma::regspace<arma::uvec>(0, static_cast<arma::uword>(p) - 1);
    for (std::size_t j = 0; j < p; ++j) res.S[0].col(j) = dy0[j];
    if (p == 0 || n_times == 1) return res;

    const AugSys S = make_aug(A, b, y0);
    std::vector<arma::mat> Etil(p);
    for (std::size_t j = 0; j < p; ++j) Etil[j] = augment(dA[j], S.sig * db[j]);
    std::vector<arma::vec> dytil0(p, arma::vec(m, arma::fill::zeros));
    for (std::size_t j = 0; j < p; ++j) dytil0[j].head(n) = dy0[j];

    const bool stacked = (m * static_cast<arma::uword>(p + 1) <= kStackedMax);
    for (arma::uword i = 1; i < n_times; ++i) {
        const double tau = ts(i) - ts(0);
        const arma::mat Mt = S.Atil * tau;
        if (stacked) {
            const double esc = direction_scale(Etil, tau);
            const arma::mat X = expm(stacked_block(S.Atil, Etil, tau, esc));
            const arma::mat expAt = X.submat(0, 0, m - 1, m - 1);
            for (std::size_t j = 0; j < p; ++j) {
                const arma::uword c = m * static_cast<arma::uword>(j + 1);
                arma::vec d = esc * (X.submat(0, c, m - 1, c + m - 1) * S.ytil);
                d += expAt * dytil0[j];
                res.S[i].col(j) = d.head(n);
            }
        } else {
            arma::mat expAt, Lj;
            for (std::size_t j = 0; j < p; ++j) {
                expm_frechet(Mt, Etil[j] * tau, expAt, Lj);
                arma::vec d = Lj * S.ytil + expAt * dytil0[j];
                res.S[i].col(j) = d.head(n);
            }
        }
    }
    return res;
}

}  // namespace linear_detail

// ===========================================================================
//  Public API
// ===========================================================================

/**
 * @brief Exact trajectory of y' = A y + b.
 *
 * Row i = y(ts[i]); ts[0] is t0 and row 0 EQUALS y0 -- the layout of
 * ode::rk45. The solution is evaluated at ts[i] - ts[0]. Throws
 * std::invalid_argument on a shape error or a non-finite input (n = 1 too).
 * Overflow is signalled by a non-finite row, never by an exception; see
 * linear_overflows().
 */
inline arma::mat linear(const arma::mat& A, const arma::vec& b,
                        const arma::vec& y0, const arma::vec& ts) {
    linear_detail::check_system(A, b, y0, ts);
    if (A.n_rows == 2) return linear_detail::linear_2x2(A, b, y0, ts);
    return linear_detail::linear_general(A, b, y0, ts);
}

/**
 * @brief Exact trajectory plus analytic sensitivities.
 *
 * dA[j], db[j], dy0[j] are the derivatives of A, b, y0 with respect to
 * theta[j]; the three vectors must have equal length p (a caller with no
 * dependence passes zeros). Returns ode::rk45_sens_result with y
 * (n_times x n, bitwise identical to linear() on the same inputs), S[i]
 * (n x p) = d y(ts[i]) / d theta and theta_idx = 0..p-1, so that
 * ode::sens_chain(result, dlp_dy) consumes it unchanged.
 */
inline rk45_sens_result linear_sens(const arma::mat& A, const arma::vec& b,
                                    const arma::vec& y0, const arma::vec& ts,
                                    const std::vector<arma::mat>& dA,
                                    const std::vector<arma::vec>& db,
                                    const std::vector<arma::vec>& dy0) {
    linear_detail::check_system(A, b, y0, ts);
    linear_detail::check_directions(A.n_rows, dA, db, dy0);
    if (A.n_rows == 2) return linear_detail::linear_sens_2x2(A, b, y0, ts, dA, db, dy0);
    return linear_detail::linear_sens_general(A, b, y0, ts, dA, db, dy0);
}

/**
 * @brief Predicts whether linear(A, b, y0, ts) will produce a non-finite
 *        value at the last time, for t_span = ts[last] - ts[0].
 *
 * Decided from A, b, y0 AND t_span, by evaluating the single-time solution at
 * t_span through the same code path linear() uses and testing finiteness --
 * exact by construction (no false negatives from a large |b| or |y0|, no false
 * positives), at the cost of one evaluation. Same validation as linear().
 */
inline bool linear_overflows(const arma::mat& A, const arma::vec& b,
                             const arma::vec& y0, double t_span) {
    const arma::vec ts{0.0, t_span};
    linear_detail::check_system(A, b, y0, ts);
    if (A.n_rows == 2) {
        const double c = linear_detail::scale_2x2(A, b, ts);
        const linear_detail::Sys2<double> S = linear_detail::make_sys2(A, b, y0, c);
        double o1, o2;
        linear_detail::eval2_at(S, t_span * c, o1, o2);
        return !(std::isfinite(o1) && std::isfinite(o2));
    }
    const linear_detail::AugSys S = linear_detail::make_aug(A, b, y0);
    return !linear_detail::y_general_at(S, t_span).is_finite();
}

}  // namespace ode
}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_ODE_LINEAR_HPP

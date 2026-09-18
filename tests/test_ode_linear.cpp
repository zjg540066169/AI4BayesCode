// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later.
// ============================================================================
//  test_ode_linear.cpp
//
//  Certification ladder for ode::linear / ode::linear_sens / ode::linear_overflows
//  (ode_linear.hpp): the exact solution of y' = A y + b with analytic
//  sensitivities, n == 2 by the closed form and every other n by the augmented
//  Pade exponential.
//
//    T0  sanity     closed form vs the general-n path on the same inputs;
//                   t = 0 returns y0; b = 0, y0 = 0 returns 0; n = 1 vs
//                   std::exp; A = 0 returns y0 + b t; validation; ts[0]
//                   semantics; negative t; y bitwise equal across the two
//                   public entry points.
//    T1  parity     vs ode::rk45 at 1e-12 per eigenvalue family, and the
//                   sensitivities vs Richardson-extrapolated central FD AND vs
//                   the algebraically independent other path, gated by the
//                   conditioning floor eps * ||block exponential|| / ||result||
//                   (the best any double method that forms e^{At} can do),
//                   not by a flat tolerance. A throwing derivative is a
//                   FAILURE. Directions perturb EVERY entry of A, b and y0.
//    T2  recovery   a two-state linear ODE observable: linear() vs rk45, and
//                   Gauss-Newton recovery of the truth through linear_sens.
//    T3  [SKIP]     cross-chain R-hat is exercised end-to-end by a separate
//                   comparison, a separate step.
//    T4  stress     s -> 0 down to s = 0 exactly; derivative continuity
//                   across it; stiff ratios to 1e8 where rk45 is wrong;
//                   overflow prediction vs actual on 20000 random systems
//                   with |b|, |y0| to 1e300; the reviewer's cancellation
//                   matrix against a pinned 120-digit reference; unit
//                   invariance at |A| = 1e-170 and 1e160.
//    AD-twin        Check #12: the hand-written sensitivities against an
//                   autodiff::var (reverse-mode) twin of the general-n path,
//                   driven through autodiff_wrap::wrap_real.
//
//  Every gate is a [PASS]/[FAIL] line; the process exits non-zero on any
//  failure. Numbers are printed so a reader can see the margin, not just the
//  verdict.
// ============================================================================

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "AI4BayesCode/ode_linear.hpp"
#include "AI4BayesCode/ode_rk45.hpp"
#include "AI4BayesCode/autodiff_wrap.hpp"

namespace ode = AI4BayesCode::ode;
namespace ld  = AI4BayesCode::ode::linear_detail;

static int g_pass = 0, g_fail = 0;
static const double EPS = std::numeric_limits<double>::epsilon();

static void check(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  [FAIL] %s\n", what.c_str()); }
}
static void check_tol(const std::string& what, double got, double tol) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "%-62s %10.3e  (tol %8.1e)", what.c_str(), got, tol);
    check(std::isfinite(got) && got <= tol, buf);
}

// ---------------------------------------------------------------------------
//  helpers
// ---------------------------------------------------------------------------
static bool all_finite(const arma::mat& M) { return M.is_finite(); }

// max |X - R| / max|R|  (matrix-scaled relative error), NaN if X non-finite
static double rel_err(const arma::mat& X, const arma::mat& R) {
    if (X.n_rows != R.n_rows || X.n_cols != R.n_cols) return std::nan("");
    if (!all_finite(X)) return std::nan("");
    const double den = arma::abs(R).max();
    const double num = arma::abs(X - R).max();
    return (den > 0.0) ? num / den : num;
}
static bool bitwise_equal(const arma::mat& X, const arma::mat& Y) {
    if (X.n_rows != Y.n_rows || X.n_cols != Y.n_cols) return false;
    for (arma::uword i = 0; i < X.n_elem; ++i)
        if (std::memcmp(&X[i], &Y[i], sizeof(double)) != 0) return false;
    return true;
}

static arma::mat ref_rk45(const arma::mat& A, const arma::vec& b,
                          const arma::vec& y0, const arma::vec& ts,
                          double tol = 1e-12, std::size_t max_iter = 200000000) {
    auto f = [&](double, const arma::vec& y, const arma::vec&) -> arma::vec {
        return A * y + b;
    };
    return ode::rk45(f, y0, ts, arma::vec(), tol, tol, 0.0, 1e-16, max_iter);
}

// A direction in (A, b, y0) space.
struct Dir {
    std::string name;
    arma::mat dA;
    arma::vec db, dy0;
};
static std::vector<Dir> all_directions(arma::uword n) {
    std::vector<Dir> D;
    const arma::mat Z(n, n, arma::fill::zeros);
    const arma::vec z(n, arma::fill::zeros);
    for (arma::uword i = 0; i < n; ++i)
        for (arma::uword j = 0; j < n; ++j) {
            Dir d{"dA(" + std::to_string(i) + "," + std::to_string(j) + ")", Z, z, z};
            d.dA(i, j) = 1.0;
            D.push_back(d);
        }
    if (n >= 2) {   // one rate appearing in two entries (a shared parameter)
        Dir d{"shared-k", Z, z, z};
        d.dA(0, 0) = -1.0; d.dA(1, 0) = 0.35;
        D.push_back(d);
    }
    for (arma::uword i = 0; i < n; ++i) {
        Dir d{"db[" + std::to_string(i) + "]", Z, z, z};
        d.db(i) = 1.0;
        D.push_back(d);
    }
    for (arma::uword i = 0; i < n; ++i) {
        Dir d{"dy0[" + std::to_string(i) + "]", Z, z, z};
        d.dy0(i) = 1.0;
        D.push_back(d);
    }
    {   // a mixed direction touching everything at once
        Dir d{"mixed", Z, z, z};
        for (arma::uword i = 0; i < n * n; ++i) d.dA[i] = 0.3 * std::cos(1.0 + double(i));
        for (arma::uword i = 0; i < n; ++i) { d.db(i) = 0.2 * std::sin(2.0 + double(i));
                                              d.dy0(i) = 0.5 * std::cos(3.0 + double(i)); }
        D.push_back(d);
    }
    return D;
}

// Directional derivative from linear_sens (one direction = one parameter).
static arma::mat dsolve(const arma::mat& A, const arma::vec& b, const arma::vec& y0,
                        const arma::vec& ts, const Dir& d) {
    const auto r = ode::linear_sens(A, b, y0, ts, {d.dA}, {d.db}, {d.dy0});
    arma::mat out(ts.n_elem, A.n_rows);
    for (arma::uword i = 0; i < ts.n_elem; ++i) out.row(i) = r.S[i].col(0).t();
    return out;
}
// Same through the general-n path regardless of n.
static arma::mat dsolve_general(const arma::mat& A, const arma::vec& b, const arma::vec& y0,
                                const arma::vec& ts, const Dir& d) {
    const auto r = ld::linear_sens_general(A, b, y0, ts, {d.dA}, {d.db}, {d.dy0});
    arma::mat out(ts.n_elem, A.n_rows);
    for (arma::uword i = 0; i < ts.n_elem; ++i) out.row(i) = r.S[i].col(0).t();
    return out;
}

// Richardson-extrapolated central difference of linear() in a direction,
// with an estimate of its own uncertainty (the last extrapolation step).
static arma::mat fd_richardson(const arma::mat& A, const arma::vec& b, const arma::vec& y0,
                               const arma::vec& ts, const Dir& d, double& drift,
                               double* noise = nullptr) {
    // The step is NOT scaled by ||A||: on a stiff system a step proportional to
    // the fast rate destroys the slow eigenvalue (measured: h = 89 on
    // lam = (-1, -1e4) gives FD values of 1e15 against a true 0.44). y is
    // linear in b and y0, so those directions may use their own scale.
    double sc = 1.0;
    if (arma::norm(d.db, "inf") > 0.0)  sc = std::max(sc, arma::abs(b).max());
    if (arma::norm(d.dy0, "inf") > 0.0) sc = std::max(sc, arma::abs(y0).max());
    const double dn = std::max({arma::abs(d.dA).max(), arma::abs(d.db).max(),
                                arma::abs(d.dy0).max()});
    const double h = 1e-3 * sc / dn;
    auto cd = [&](double hh) -> arma::mat {
        return (ode::linear(A + hh * d.dA, b + hh * d.db, y0 + hh * d.dy0, ts)
              - ode::linear(A - hh * d.dA, b - hh * d.db, y0 - hh * d.dy0, ts)) / (2.0 * hh);
    };
    const arma::mat f1 = cd(h), f2 = cd(0.5 * h), f3 = cd(0.25 * h);
    const arma::mat r1 = (4.0 * f2 - f1) / 3.0;
    const arma::mat r2 = (4.0 * f3 - f2) / 3.0;
    const arma::mat rr = (16.0 * r2 - r1) / 15.0;
    drift = rel_err(r2, rr);
    // Rounding floor of a central difference, relative to |S|_inf: the two
    // perturbed matrices A +- h E are ROUNDED independently, and an ulp change
    // in an entry moves the solution by up to eps ||A|| t |y| (on the stiff
    // dense family lam = (-1, -1e7) an ulp of a 3e6 entry moves the slow
    // eigenvalue by 2e-9, which / 2h = 2e-3 is the 4e-6 FD error measured
    // there while the two exact methods agree to their floor), plus the plain
    // eps |y| / h term. This is a limit of the FD REFERENCE; the exact-vs-exact
    // comparison certifies those cases.
    if (noise) {
        const double ymax = arma::abs(ode::linear(A, b, y0, ts)).max();
        const double smax = std::max(arma::abs(rr).max(), 1e-300);
        const double tmax = arma::abs(ts - ts(0)).max();
        const double input = (arma::norm(d.dA, "inf") > 0.0) ? arma::norm(A, 1) * tmax : 0.0;
        *noise = EPS * ymax * (1.0 + input) / (0.25 * h * smax);
    }
    return rr;
}

// The conditioning floor the harness derived: once e^{At} is formed as a
// matrix its entries are known to eps of the matrix NORM, so a result of
// magnitude r cannot be more accurate than eps * Xnorm / r. Xnorm is the
// largest entry of the block exponential [[e^{Atil t}, L],[0, e^{Atil t}]]
// over the output times (the (1,1) block alone for the trajectory).
static double cond_floor_traj(const arma::mat& A, const arma::vec& b,
                              const arma::vec& ts, const arma::mat& result) {
    const arma::mat At = ld::augment(A, b);
    double xn = 0.0;
    for (arma::uword i = 1; i < ts.n_elem; ++i)
        xn = std::max(xn, arma::abs(ld::expm(At * (ts(i) - ts(0)))).max());
    const double r = arma::abs(result).max();
    if (!(r > 0.0) || !std::isfinite(xn)) return EPS;
    return std::max(EPS, EPS * xn / r);
}
static double cond_floor_sens(const arma::mat& A, const arma::vec& b,
                              const arma::vec& ts, const Dir& d,
                              const arma::mat& result) {
    const arma::mat At = ld::augment(A, b), Et = ld::augment(d.dA, d.db);
    const arma::uword m = At.n_rows;
    double xn = 0.0;
    for (arma::uword i = 1; i < ts.n_elem; ++i) {
        const double tau = ts(i) - ts(0);
        arma::mat C(2 * m, 2 * m, arma::fill::zeros);
        C.submat(0, 0, m - 1, m - 1) = At * tau;
        C.submat(m, m, 2 * m - 1, 2 * m - 1) = At * tau;
        C.submat(0, m, m - 1, 2 * m - 1) = Et * tau;
        xn = std::max(xn, arma::abs(ld::expm(C)).max());
    }
    const double r = arma::abs(result).max();
    if (!(r > 0.0) || !std::isfinite(xn)) return EPS;
    return std::max(EPS, EPS * xn / r);
}

// Higham's squaring count on the widest output time: the general-n path loses
// about 2^s eps relative to the matrix norm in the squaring phase (documented
// in the header; measured by the reviewers as 0.17 * 2^s * u at stiffness
// 1e6, and reproduced here: S disagreement / (2^s eps) = 0.94 .. 1.06 over
// stiffness 1e3 .. 1e8). A comparison AGAINST the Pade path therefore carries
// this factor; the closed form itself is gated separately against exact and
// 120-digit references (T4) with no such allowance.
static double pade_2s(const arma::mat& A, const arma::vec& b, const arma::vec& ts) {
    const arma::mat At = ld::augment(A, b);
    int smax = 0;
    for (arma::uword i = 1; i < ts.n_elem; ++i) {
        const double nrm = arma::norm(At * (ts(i) - ts(0)), 1);
        const int s = (nrm > 5.371920351148152) ? int(std::ceil(std::log2(nrm / 5.371920351148152))) : 0;
        smax = std::max(smax, s);
    }
    return std::ldexp(1.0, smax);
}

// ---- eigenvalue families (prescribed spectrum under a fixed similarity) ---
static arma::mat V2() { return arma::mat{{1.0, 0.3}, {-0.4, 1.0}}; }
static arma::mat V3() { return arma::mat{{1.0, 0.2, 0.1}, {0.3, 1.0, -0.2}, {-0.1, 0.4, 1.0}}; }
static arma::mat A_from_diag(const arma::mat& V, const arma::vec& lam) {
    return V * arma::diagmat(lam) * arma::inv(V);
}
static arma::mat A_jordan2(const arma::mat& V, double l) {
    return V * arma::mat{{l, 1.0}, {0.0, l}} * arma::inv(V);
}
static arma::mat A_rot2(const arma::mat& V, double al, double be) {
    return V * arma::mat{{al, be}, {-be, al}} * arma::inv(V);
}

struct Case {
    std::string family, label;
    arma::mat A;
    arma::vec b, y0, ts;
    bool rk45_ok = true;     // rk45 @1e-12 is expected to run (not stiff >= 1e6)
};
static std::vector<Case> families() {
    std::vector<Case> C;
    const arma::vec ts1{0.0, 0.05, 0.25, 0.6, 1.0, 2.0};
    const arma::vec y0a{1.0, -2.0}, bA{0.7, -0.3}, b0{0.0, 0.0};
    const arma::mat V = V2();
    C.push_back({"wellsep",       "lam=(-1,-10) b!=0",       A_from_diag(V, {-1.0, -10.0}), bA, y0a, ts1});
    C.push_back({"wellsep",       "lam=(-1,-10) b=0",        A_from_diag(V, {-1.0, -10.0}), b0, y0a, ts1});
    C.push_back({"neareq",        "gap 1e-6",                A_from_diag(V, {-1.0, -1.0 - 1e-6}), bA, y0a, ts1});
    C.push_back({"neareq",        "gap 1e-10",               A_from_diag(V, {-1.0, -1.0 - 1e-10}), bA, y0a, ts1});
    C.push_back({"exactly_equal", "Jordan J(-1)",            A_jordan2(V, -1.0), bA, y0a, ts1});
    C.push_back({"exactly_equal", "semisimple -1 I",         arma::mat{{-1.0, 0.0}, {0.0, -1.0}}, bA, y0a, ts1});
    C.push_back({"complex_pair",  "al=-0.5 be=3",            A_rot2(V, -0.5, 3.0), bA, y0a, ts1});
    C.push_back({"complex_pair",  "al=0 be=5 (undamped)",    A_rot2(V, 0.0, 5.0), bA, y0a, ts1});
    C.push_back({"one_zero",      "lam=(0,-3)",              A_from_diag(V, {0.0, -3.0}), bA, y0a, ts1});
    C.push_back({"one_zero",      "J(0) defective",          A_jordan2(V, 0.0), bA, y0a, ts1});
    C.push_back({"decaying",      "lam=(-0.3,-2.5)",         A_from_diag(V, {-0.3, -2.5}), bA, y0a, ts1});
    C.push_back({"growing",       "lam=(0.5,2)",             A_from_diag(V, {0.5, 2.0}), bA, y0a, ts1});
    C.push_back({"growing",       "lam=(3,3+1e-4)",          A_from_diag(V, {3.0, 3.0 + 1e-4}), bA, y0a, ts1});
    for (int k = 2; k <= 8; ++k) {
        const double fast = -std::pow(10.0, double(k));
        char lab[64];
        std::snprintf(lab, sizeof lab, "lam=(-1,-1e%d)", k);
        arma::vec ts{0.0, 0.5 / std::abs(fast), 5.0 / std::abs(fast), 0.2, 0.5, 1.0};
        Case c{"stiff_1e" + std::to_string(k), lab, A_from_diag(V, {-1.0, fast}), bA, y0a, ts};
        c.rk45_ok = (k <= 5);
        C.push_back(c);
    }
    C.push_back({"n3_general",    "lam=(-0.4,-1.3,-6)",      A_from_diag(V3(), {-0.4, -1.3, -6.0}),
                 arma::vec{0.7, -0.3, 0.2}, arma::vec{1.0, -2.0, 0.5}, ts1});
    C.push_back({"n3_general",    "lam=(0,-1,-1-1e-9)",      A_from_diag(V3(), {0.0, -1.0, -1.0 - 1e-9}),
                 arma::vec{0.7, -0.3, 0.2}, arma::vec{1.0, -2.0, 0.5}, ts1});
    return C;
}

// Exact reference for the upper-triangular A = [[l1, c],[0, l2]], b = (b1,b2):
//   y2 = K e^{l2 t} + M,  K = y02 + b2/l2,  M = -b2/l2
//   y1 = e^{l1 t} y01 + cK (e^{l2 t} - e^{l1 t})/(l2 - l1) + (cM + b1)(e^{l1 t} - 1)/l1
static arma::vec ref_tri(double l1, double l2, double c, double b1, double b2,
                         double y01, double y02, double t) {
    const double K = y02 + b2 / l2, M = -b2 / l2;
    const double e1 = std::exp(l1 * t), e2 = std::exp(l2 * t);
    arma::vec y(2);
    y(1) = K * e2 + M;
    y(0) = e1 * y01 + c * K * (e2 - e1) / (l2 - l1) + (c * M + b1) * std::expm1(l1 * t) / l1;
    return y;
}
// ... and its confluent limit l1 = l2 = l:  y1 = e^{lt} y01 + cK t e^{lt} + (cM+b1)(e^{lt}-1)/l
static arma::vec ref_tri_confluent(double l, double c, double b1, double b2,
                                   double y01, double y02, double t) {
    const double K = y02 + b2 / l, M = -b2 / l;
    const double e = std::exp(l * t);
    arma::vec y(2);
    y(1) = K * e + M;
    y(0) = e * y01 + c * K * t * e + (c * M + b1) * std::expm1(l * t) / l;
    return y;
}

// ===========================================================================
//  T0
// ===========================================================================
static void T0() {
    std::printf("\n[T0] sanity\n");
    std::mt19937_64 rng(20260918ULL);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    // (a) closed form vs general path, 400 random 2x2 systems incl. oscillatory
    {
        double worst_y = 0.0, worst_S = 0.0;
        for (int rep = 0; rep < 400; ++rep) {
            arma::mat A(2, 2);
            for (auto& v : A) v = 3.0 * U(rng);
            arma::vec b{U(rng), U(rng)}, y0{2.0 * U(rng), 2.0 * U(rng)};
            arma::vec ts{0.0, 0.3, 1.0, 2.5};
            const arma::mat Y2 = ode::linear(A, b, y0, ts);
            const arma::mat Yg = ld::linear_general(A, b, y0, ts);
            const double fl = cond_floor_traj(A, b, ts, Yg), p2s = pade_2s(A, b, ts);
            worst_y = std::max(worst_y, rel_err(Y2, Yg) / (p2s * std::max(1e-13, 30.0 * fl)));
            for (const auto& d : all_directions(2)) {
                const arma::mat S2 = dsolve(A, b, y0, ts, d);
                const arma::mat Sg = dsolve_general(A, b, y0, ts, d);
                const double fs = cond_floor_sens(A, b, ts, d, Sg);
                worst_S = std::max(worst_S, rel_err(S2, Sg) / (p2s * std::max(1e-13, 30.0 * fs)));
            }
        }
        check_tol("closed form vs general path, y, 400 random 2x2 (x 2^s floor)", worst_y, 1.0);
        check_tol("closed form vs general path, S, 400 random x 13 dirs (x 2^s floor)", worst_S, 1.0);
    }
    // (b) the same on every 2x2 family (incl. defective, complex, stiff to 1e5)
    {
        double worst_y = 0.0, worst_S = 0.0;
        std::string wy, ws;
        for (const auto& c : families()) {
            if (c.A.n_rows != 2 || c.family.rfind("stiff_1e", 0) == 0) {
                if (c.family != "stiff_1e2" && c.family != "stiff_1e3" && c.family != "stiff_1e4") continue;
            }
            const arma::mat Y2 = ode::linear(c.A, c.b, c.y0, c.ts);
            const arma::mat Yg = ld::linear_general(c.A, c.b, c.y0, c.ts);
            const double fl = cond_floor_traj(c.A, c.b, c.ts, Yg), p2s = pade_2s(c.A, c.b, c.ts);
            const double ey = rel_err(Y2, Yg) / (p2s * std::max(1e-13, 30.0 * fl));
            if (ey > worst_y) { worst_y = ey; wy = c.family + " " + c.label; }
            for (const auto& d : all_directions(2)) {
                const arma::mat S2 = dsolve(c.A, c.b, c.y0, c.ts, d);
                const arma::mat Sg = dsolve_general(c.A, c.b, c.y0, c.ts, d);
                const double fs = cond_floor_sens(c.A, c.b, c.ts, d, Sg);
                const double es = rel_err(S2, Sg) / (p2s * std::max(1e-13, 30.0 * fs));
                if (es > worst_S) { worst_S = es; ws = c.family + " " + c.label + " " + d.name; }
            }
        }
        std::printf("      worst y at [%s], worst S at [%s]\n", wy.c_str(), ws.c_str());
        check_tol("closed form vs general path on the 2x2 families, y (x 2^s floor)", worst_y, 1.0);
        check_tol("closed form vs general path on the 2x2 families, S (x 2^s floor)", worst_S, 1.0);
    }
    // (c) t = 0 returns y0 bitwise; ts[0] != 0 semantics; negative t
    {
        arma::mat A{{-1.0, 2.0}, {0.5, -3.0}};
        arma::vec b{1.0, -2.0}, y0{0.3, -0.7};
        arma::mat Y = ode::linear(A, b, y0, arma::vec{0.0});
        check(Y.n_rows == 1 && bitwise_equal(Y, y0.t()), "t = 0 alone returns y0 bitwise (n=2)");
        arma::mat A3{{-1.0, 0.2, 0.0}, {0.1, -2.0, 0.3}, {0.0, 0.4, -0.5}};
        arma::vec b3{1.0, 0.0, -1.0}, y03{0.3, -0.7, 1.1};
        arma::mat Y3 = ode::linear(A3, b3, y03, arma::vec{2.0});
        check(Y3.n_rows == 1 && bitwise_equal(Y3, y03.t()), "t = 0 alone returns y0 bitwise (n=3)");

        // ts[0] is t0: y(ts[i]) for ts = {1, 1.5, 2, 3} equals the solve on {0, .5, 1, 2}
        const arma::vec tsa{1.0, 1.5, 2.0, 3.0}, tsb{0.0, 0.5, 1.0, 2.0};
        const arma::mat Ya = ode::linear(A, b, y0, tsa), Yb = ode::linear(A, b, y0, tsb);
        check(bitwise_equal(Ya, Yb) && bitwise_equal(Ya.row(0), y0.t()),
              "ts[0] is t0: solve on {1,1.5,2,3} == solve on {0,.5,1,2}, row 0 == y0");
        const arma::mat Ra = ref_rk45(A, b, y0, tsa);
        check_tol("ts[0] = 1: vs ode::rk45 on the same ts (layout parity)", rel_err(Ya, Ra), 1e-10);
        const arma::mat Y3a = ode::linear(A3, b3, y03, tsa), Y3b = ode::linear(A3, b3, y03, tsb);
        check(bitwise_equal(Y3a, Y3b), "ts[0] is t0 on the general path too");

        // negative t: y(-t) must invert y(t); stiff spectrum so e^{mt} alone would be 0*inf
        arma::mat As{{0.0, 1.0}, {0.0, 2000.0}};
        arma::vec ys{1.0, -2.0}, zb{0.0, 0.0};
        const arma::mat Yn = ode::linear(As, zb, ys, arma::vec{0.0, -1.0});
        // exact: y1(-1) = 1 + (-2)(e^{-2000} - 1)/2000, y2(-1) = -2 e^{-2000} = 0
        const double y1x = 1.0 - 2.0 * std::expm1(-2000.0) / 2000.0;
        check_tol("negative t on a stiff spectrum (reviewer's NaN case)",
                  std::abs(Yn(1, 0) - y1x) + std::abs(Yn(1, 1)), 1e-15);
        arma::vec tsn{0.0, -0.7, -1.3};
        const arma::mat Yneg = ode::linear(A, b, y0, tsn);
        const arma::mat Yrt  = ode::linear(A, b, Yneg.row(2).t(), arma::vec{-1.3, -0.7, 0.0});
        check_tol("negative t: forward from y(-1.3) recovers y(-0.7), y(0)",
                  std::max(rel_err(Yrt.row(1), Yneg.row(1)), rel_err(Yrt.row(2), y0.t())), 1e-13);
        const arma::mat Y3n = ode::linear(A3, b3, y03, tsn);
        const arma::mat Y3r = ode::linear(A3, b3, Y3n.row(2).t(), arma::vec{-1.3, -0.7, 0.0});
        check_tol("negative t on the general path: round trip",
                  std::max(rel_err(Y3r.row(1), Y3n.row(1)), rel_err(Y3r.row(2), y03.t())), 1e-13);
    }
    // (d) b = 0, y0 = 0 -> 0 exactly; A = 0 -> y0 + b t; n = 1 vs std::exp
    {
        arma::mat A{{-1.0, 2.0}, {0.5, -3.0}};
        arma::vec z{0.0, 0.0};
        const arma::mat Y = ode::linear(A, z, z, arma::vec{0.0, 1.0, 10.0});
        check(all_finite(Y) && arma::abs(Y).max() == 0.0, "b = 0, y0 = 0 returns exactly 0 (n=2)");
        arma::mat A3{{-1.0, 0.2, 0.0}, {0.1, -2.0, 0.3}, {0.0, 0.4, -0.5}};
        arma::vec z3(3, arma::fill::zeros);
        const arma::mat Y3 = ode::linear(A3, z3, z3, arma::vec{0.0, 1.0, 10.0});
        check(all_finite(Y3) && arma::abs(Y3).max() == 0.0, "b = 0, y0 = 0 returns exactly 0 (n=3)");

        arma::vec b{1.0, -2.0}, y0{0.3, -0.7}, ts{0.0, 1.0, 7.5, -2.0};
        const arma::mat Yz = ode::linear(arma::mat(2, 2, arma::fill::zeros), b, y0, ts);
        double w = 0.0;
        for (arma::uword i = 0; i < ts.n_elem; ++i)
            for (int k = 0; k < 2; ++k)
                w = std::max(w, std::abs(Yz(i, k) - (y0(k) + b(k) * ts(i))) /
                                std::max(1e-300, std::abs(y0(k) + b(k) * ts(i))));
        check_tol("A = 0 returns y0 + b t (n=2)", w, 4.0 * EPS);
        arma::vec b3{1.0, -2.0, 0.5}, y03{0.3, -0.7, 2.0};
        const arma::mat Yz3 = ode::linear(arma::mat(3, 3, arma::fill::zeros), b3, y03, ts);
        w = 0.0;
        for (arma::uword i = 0; i < ts.n_elem; ++i)
            for (int k = 0; k < 3; ++k)
                w = std::max(w, std::abs(Yz3(i, k) - (y03(k) + b3(k) * ts(i))) /
                                std::max(1e-300, std::abs(y03(k) + b3(k) * ts(i))));
        check_tol("A = 0 returns y0 + b t (n=3)", w, 4.0 * EPS);

        // n = 1: y = e^{at} y0 + b (e^{at} - 1)/a
        double worst = 0.0;
        for (double a : {-2.0, -1e-3, 0.7, -50.0}) {
            arma::mat A1(1, 1); A1(0, 0) = a;
            arma::vec b1{0.4}, y1{1.5}, ts1{0.0, 0.1, 1.0, 3.0, -0.5};
            const arma::mat Y1 = ode::linear(A1, b1, y1, ts1);
            const double p2s = pade_2s(A1, b1, ts1);     // n = 1 rides the Pade path: 2^s eps law
            for (arma::uword i = 0; i < ts1.n_elem; ++i) {
                const double ex = std::exp(a * ts1(i)) * y1(0) + b1(0) * std::expm1(a * ts1(i)) / a;
                worst = std::max(worst, std::abs(Y1(i, 0) - ex) / std::abs(ex) / (p2s * 10.0 * EPS));
            }
        }
        check_tol("n = 1 scalar case vs std::exp / expm1 (x 10 * 2^s eps; a t to -150)", worst, 1.0);
        // n = 1 sensitivity vs the hand derivative: d/da [e^{at} y0 + b(e^{at}-1)/a]
        {
            const double a = -0.8, t = 1.7, bb = 0.4, yy = 1.5;
            arma::mat A1(1, 1); A1(0, 0) = a;
            arma::mat dA1(1, 1); dA1(0, 0) = 1.0;
            const auto r = ode::linear_sens(A1, arma::vec{bb}, arma::vec{yy}, arma::vec{0.0, t},
                                            {dA1}, {arma::vec{0.0}}, {arma::vec{0.0}});
            const double ex = t * std::exp(a * t) * yy + bb * (t * std::exp(a * t) / a - std::expm1(a * t) / (a * a));
            check_tol("n = 1 sensitivity d y/d a vs hand derivative",
                      std::abs(r.S[1](0, 0) - ex) / std::abs(ex), 1e-13);
        }
    }
    // (e) validation: every path, library messages, n = 1 included
    {
        auto throws_with = [](auto&& fn, const char* needle) -> bool {
            try { fn(); } catch (const std::invalid_argument& e) {
                return std::string(e.what()).find(needle) != std::string::npos;
            } catch (...) { return false; }
            return false;
        };
        arma::mat A1(1, 1); A1(0, 0) = std::nan("");
        check(throws_with([&] { ode::linear(A1, arma::vec{1.0}, arma::vec{1.0}, arma::vec{0.0, 1.0}); },
                          "non-finite"), "n = 1 with NaN in A throws (same validation as n >= 2)");
        arma::mat Ai(1, 1); Ai(0, 0) = std::numeric_limits<double>::infinity();
        check(throws_with([&] { ode::linear_overflows(Ai, arma::vec{1.0}, arma::vec{1.0}, 1.0); },
                          "non-finite"), "n = 1 with Inf in A throws from linear_overflows too");
        arma::mat A{{-1.0, 2.0}, {0.5, -3.0}};
        arma::vec b{1.0, -2.0}, y0{0.3, -0.7}, ts{0.0, 1.0};
        check(throws_with([&] { ode::linear(A, arma::vec{1.0}, y0, ts); }, "b must have length n"),
              "b of wrong length throws with the library's message");
        check(throws_with([&] { ode::linear(A, b, arma::vec{1.0, 2.0, 3.0}, ts); }, "y0 must have length n"),
              "y0 of wrong length throws with the library's message");
        check(throws_with([&] { ode::linear(arma::mat(2, 3, arma::fill::zeros), b, y0, ts); }, "square"),
              "non-square A throws");
        check(throws_with([&] { ode::linear(A, b, y0, arma::vec{0.0, std::nan("")}); }, "ts has a non-finite"),
              "NaN in ts throws");
        check(throws_with([&] { ode::linear(A, b, y0, arma::vec()); }, "at least one"),
              "empty ts throws");
        // per-direction checks on BOTH paths
        for (arma::uword n : {2u, 3u}) {
            arma::mat An = (n == 2) ? A : arma::mat{{-1.0, 0.2, 0.0}, {0.1, -2.0, 0.3}, {0.0, 0.4, -0.5}};
            arma::vec bn(n, arma::fill::ones), yn(n, arma::fill::ones), zn(n, arma::fill::zeros);
            const std::string tag = " (n=" + std::to_string(n) + ")";
            check(throws_with([&] { ode::linear_sens(An, bn, yn, ts, {arma::mat(n + 1, n + 1, arma::fill::zeros)}, {zn}, {zn}); },
                              "dA[0] must be n x n"), "mis-sized dA[j] throws with the library's message" + tag);
            check(throws_with([&] { ode::linear_sens(An, bn, yn, ts, {arma::mat(n, n, arma::fill::zeros)}, {arma::vec(n + 1, arma::fill::zeros)}, {zn}); },
                              "db[0] must have length n"), "mis-sized db[j] throws with the library's message" + tag);
            check(throws_with([&] { ode::linear_sens(An, bn, yn, ts, {arma::mat(n, n, arma::fill::zeros)}, {zn}, {arma::vec(n - 1, arma::fill::zeros)}); },
                              "dy0[0] must have length n"), "mis-sized dy0[j] throws with the library's message" + tag);
            check(throws_with([&] { ode::linear_sens(An, bn, yn, ts, {arma::mat(n, n, arma::fill::zeros)}, {}, {zn}); },
                              "same length"), "dA/db/dy0 length mismatch throws" + tag);
            arma::mat dAn(n, n, arma::fill::zeros); dAn(0, 0) = std::nan("");
            check(throws_with([&] { ode::linear_sens(An, bn, yn, ts, {dAn}, {zn}, {zn}); },
                              "dA[0] has a non-finite"), "NaN in dA[j] throws" + tag);
        }
        // p = 0 is legal: S[i] are n x 0, theta_idx empty, y equals linear()
        const auto r0 = ode::linear_sens(A, b, y0, ts, {}, {}, {});
        check(r0.S.size() == 2 && r0.S[1].n_rows == 2 && r0.S[1].n_cols == 0 && r0.theta_idx.n_elem == 0
              && bitwise_equal(r0.y, ode::linear(A, b, y0, ts)), "p = 0 directions: legal, empty S, y bitwise");
    }
    // (f) y bitwise identical between linear() and linear_sens().y, both paths;
    //     result type is rk45_sens_result and sens_chain consumes it unchanged
    {
        bool ok2 = true, ok3 = true;
        for (int rep = 0; rep < 200; ++rep) {
            arma::mat A(2, 2);
            for (auto& v : A) v = 3.0 * U(rng);
            arma::vec b{U(rng), U(rng)}, y0{2.0 * U(rng), 2.0 * U(rng)};
            arma::vec ts{0.0, 0.3, 1.0, 2.5, 7.0};
            std::vector<arma::mat> dA(3, arma::mat(2, 2, arma::fill::zeros));
            std::vector<arma::vec> db(3, arma::vec(2, arma::fill::zeros)), dy0(3, arma::vec(2, arma::fill::zeros));
            dA[0](0, 0) = 1.0; dA[1](1, 0) = 0.7; db[1](0) = 1.0; dy0[2](1) = 1.0;
            const auto r = ode::linear_sens(A, b, y0, ts, dA, db, dy0);
            ok2 = ok2 && bitwise_equal(r.y, ode::linear(A, b, y0, ts));
            arma::mat A3(3, 3);
            for (auto& v : A3) v = 2.0 * U(rng);
            arma::vec b3{U(rng), U(rng), U(rng)}, y03{U(rng), U(rng), U(rng)};
            std::vector<arma::mat> dA3(2, arma::mat(3, 3, arma::fill::zeros));
            std::vector<arma::vec> db3(2, arma::vec(3, arma::fill::zeros)), dy03(2, arma::vec(3, arma::fill::zeros));
            dA3[0](0, 1) = 1.0; db3[1](2) = 1.0;
            const auto r3 = ode::linear_sens(A3, b3, y03, ts, dA3, db3, dy03);
            ok3 = ok3 && bitwise_equal(r3.y, ode::linear(A3, b3, y03, ts));
        }
        check(ok2, "linear().y == linear_sens().y BITWISE, 200 random 2x2 (dual path)");
        check(ok3, "linear().y == linear_sens().y BITWISE, 200 random 3x3 (Pade path)");

        arma::mat A{{-1.0, 2.0}, {0.5, -3.0}};
        arma::vec b{1.0, -2.0}, y0{0.3, -0.7}, ts{0.0, 1.0, 2.0};
        std::vector<arma::mat> dA{arma::mat{{1.0, 0.0}, {0.0, 0.0}}, arma::mat{{0.0, 0.0}, {0.0, 1.0}}};
        std::vector<arma::vec> db{arma::vec{0.0, 0.0}, arma::vec{0.0, 0.0}}, dy0 = db;
        const ode::rk45_sens_result r = ode::linear_sens(A, b, y0, ts, dA, db, dy0);
        arma::mat dlp(3, 2, arma::fill::ones);
        const arma::vec g = ode::sens_chain(r, dlp);
        arma::vec g_hand(2, arma::fill::zeros);
        for (arma::uword i = 0; i < 3; ++i) g_hand += r.S[i].t() * dlp.row(i).t();
        check(g.n_elem == 2 && r.theta_idx.n_elem == 2 && r.theta_idx(0) == 0 && r.theta_idx(1) == 1
              && arma::abs(g - g_hand).max() == 0.0,
              "returns ode::rk45_sens_result; ode::sens_chain consumes it unchanged (theta_idx = 0..p-1)");
    }
}

// ===========================================================================
//  T1
// ===========================================================================
static void T1() {
    std::printf("\n[T1] parity vs ode::rk45 @1e-12 and sensitivities per family\n");
    std::printf("      %-14s %-22s %10s %10s %10s %10s %10s %10s\n", "family", "label",
                "y vs rk45", "y floor", "S vs FD", "FD floor", "S vs other", "S floor");
    std::printf("      (S vs FD = Richardson central FD of linear(), gated by max(30 x floor, 10 x FD drift, 10 x FD rounding floor, 1e-9);\n"
                "       S vs other = the algebraically independent path (dual <-> Van Loan), gated by 2^s x max(30 x floor, 1e-13))\n");
    double worst_fd_excess = 0.0, worst_other_excess = 0.0, worst_rk = 0.0;
    std::string wfd, wot, wrk;
    int thrown = 0;
    for (const auto& c : families()) {
        const arma::uword n = c.A.n_rows;
        const arma::mat Y = ode::linear(c.A, c.b, c.y0, c.ts);
        double ey = std::nan("");
        std::string rk_note;
        if (c.rk45_ok) {
            try {
                const arma::mat R = ref_rk45(c.A, c.b, c.y0, c.ts);
                ey = rel_err(Y, R);
                const double gate = (c.family.rfind("stiff_1e", 0) == 0) ? 1e-8 : 1e-9;  // rk45-limited
                if (ey / gate > worst_rk) { worst_rk = ey / gate; wrk = c.family + " " + c.label; }
            } catch (const std::exception& e) { rk_note = "rk45 THREW"; ++thrown; }
        } else {
            rk_note = "rk45 n/a (T4)";
        }
        const double yfl = cond_floor_traj(c.A, c.b, c.ts, Y);

        double fd_worst = 0.0, ot_worst = 0.0, sfl_worst = 0.0, fdfl_worst = 0.0;
        for (const auto& d : all_directions(n)) {
            arma::mat S, So;
            try {
                S  = dsolve(c.A, c.b, c.y0, c.ts, d);
                if (n == 2) So = dsolve_general(c.A, c.b, c.y0, c.ts, d);
            } catch (const std::exception& e) {
                ++thrown;
                std::printf("      DERIVATIVE THREW on %s %s dir %s: %s\n", c.family.c_str(),
                            c.label.c_str(), d.name.c_str(), e.what());
                continue;
            }
            if (!all_finite(S)) { ++thrown; std::printf("      NON-FINITE derivative %s %s\n", c.family.c_str(), d.name.c_str()); continue; }
            const double fl = cond_floor_sens(c.A, c.b, c.ts, d, S);
            sfl_worst = std::max(sfl_worst, fl);
            double drift = 0.0, noise = 0.0;
            const arma::mat F = fd_richardson(c.A, c.b, c.y0, c.ts, d, drift, &noise);
            const double efd = rel_err(S, F);
            const double tol_fd = std::max({1e-9, 30.0 * fl, 10.0 * drift, 10.0 * noise});
            fd_worst = std::max(fd_worst, efd);
            fdfl_worst = std::max(fdfl_worst, noise);
            if (efd / tol_fd > worst_fd_excess) { worst_fd_excess = efd / tol_fd; wfd = c.family + " " + c.label + " " + d.name; }
            if (n == 2) {
                const double eo = rel_err(S, So);
                ot_worst = std::max(ot_worst, eo);
                const double tol_o = pade_2s(c.A, c.b, c.ts) * std::max(1e-13, 30.0 * fl);
                if (eo / tol_o > worst_other_excess) { worst_other_excess = eo / tol_o; wot = c.family + " " + c.label + " " + d.name; }
            }
        }
        char ybuf[32], obuf[32];
        if (!rk_note.empty()) std::snprintf(ybuf, sizeof ybuf, "%s", rk_note.c_str());
        else std::snprintf(ybuf, sizeof ybuf, "%10.2e", ey);
        if (n == 2) std::snprintf(obuf, sizeof obuf, "%10.2e", ot_worst);
        else std::snprintf(obuf, sizeof obuf, "%10s", "-");
        std::printf("      %-14s %-22s %10s %10.2e %10.2e %10.2e %10s %10.2e\n", c.family.c_str(), c.label.c_str(),
                    ybuf, yfl, fd_worst, fdfl_worst, obuf, sfl_worst);
    }
    std::printf("      worst y/rk45 gate at [%s]; worst S/FD gate at [%s]; worst S/other gate at [%s]\n",
                wrk.c_str(), wfd.c_str(), wot.c_str());
    check_tol("y vs ode::rk45 @1e-12, all families where rk45 runs (x rk45-limited gate)", worst_rk, 1.0);
    check_tol("S vs Richardson FD, all families x all directions (x gate)", worst_fd_excess, 1.0);
    check_tol("S closed form vs S Van Loan, 2x2 families x all directions (x 2^s floor gate)", worst_other_excess, 1.0);
    check(thrown == 0, "no derivative threw and none was non-finite (throwing derivative = FAIL); count = " + std::to_string(thrown));

    // n = 3 sensitivities: stacked path vs per-direction Frechet path (the two
    // general-n implementations) on the n3 families and random systems
    {
        std::mt19937_64 rng(77ULL);
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        double worst = 0.0;
        for (int rep = 0; rep < 60; ++rep) {
            const arma::uword n = 3 + (rep % 3);          // 3, 4, 5
            arma::mat A(n, n);
            for (auto& v : A) v = 1.5 * U(rng);
            A -= 0.5 * arma::eye<arma::mat>(n, n);
            arma::vec b(n), y0(n);
            for (auto& v : b) v = U(rng);
            for (auto& v : y0) v = U(rng);
            arma::vec ts{0.0, 0.4, 1.0, 3.0};
            std::vector<arma::mat> dA; std::vector<arma::vec> db, dy0;
            for (const auto& d : all_directions(n)) { dA.push_back(d.dA); db.push_back(d.db); dy0.push_back(d.dy0); }
            // p is large enough that the stacked cap (48) is exceeded for n >= 3
            // -> per-direction Frechet; also run with only 3 directions (stacked)
            const auto rs = ode::linear_sens(A, b, y0, ts, dA, db, dy0);
            for (std::size_t j = 0; j < dA.size(); ++j) {
                const auto r1 = ode::linear_sens(A, b, y0, ts, {dA[j]}, {db[j]}, {dy0[j]});
                for (arma::uword i = 0; i < ts.n_elem; ++i)
                    worst = std::max(worst, rel_err(r1.S[i].col(0), rs.S[i].col(j)));
            }
        }
        check_tol("general n: stacked block vs per-direction Frechet agree (n=3..5, all dirs)", worst, 1e-13);
    }
}

// ===========================================================================
//  T2  a two-state linear ODE model: obs = T - y1 - y2
// ===========================================================================
struct Lin2 {
    double k1, k2, a21, a12, gam, T;
    arma::mat A() const { return arma::mat{{-k1, a12 * k2}, {a21 * k1, -k2}}; }
    arma::vec y0() const { return arma::vec{gam * T, (1.0 - gam) * T}; }
    // theta = (k1, k2, a21, a12, gam)
    void dirs(std::vector<arma::mat>& dA, std::vector<arma::vec>& db, std::vector<arma::vec>& dy0) const {
        dA.assign(5, arma::mat(2, 2, arma::fill::zeros));
        db.assign(5, arma::vec(2, arma::fill::zeros));
        dy0.assign(5, arma::vec(2, arma::fill::zeros));
        dA[0] = arma::mat{{-1.0, 0.0}, {a21, 0.0}};
        dA[1] = arma::mat{{0.0, a12}, {0.0, -1.0}};
        dA[2] = arma::mat{{0.0, 0.0}, {k1, 0.0}};
        dA[3] = arma::mat{{0.0, k2}, {0.0, 0.0}};
        dy0[4] = arma::vec{T, -T};
    }
};
static void T2() {
    std::printf("\n[T2] recovery: two-state linear ODE observable\n");
    const Lin2 truth{0.6, 0.25, 0.3, 0.2, 0.9, 100.0};
    arma::vec ts(21);
    ts(0) = 0.0;
    for (int i = 1; i <= 20; ++i) ts(i) = 0.5 + (10.0 - 0.5) * (i - 1) / 19.0;   // seq(0.5, 10, 20)
    const arma::vec zb(2, arma::fill::zeros);

    const arma::mat Y = ode::linear(truth.A(), zb, truth.y0(), ts);
    const arma::mat R = ref_rk45(truth.A(), zb, truth.y0(), ts);
    arma::vec e_lin(21), e_rk(21);
    for (int i = 0; i <= 20; ++i) { e_lin(i) = truth.T - Y(i, 0) - Y(i, 1); e_rk(i) = truth.T - R(i, 0) - R(i, 1); }
    double worst = 0.0;
    for (int i = 1; i <= 20; ++i) worst = std::max(worst, std::abs(e_lin(i) - e_rk(i)) / std::abs(e_rk(i)));
    std::printf("      obs(t=10) linear = %.12f  rk45 = %.12f\n", e_lin(20), e_rk(20));
    check_tol("observable curve: linear() vs rk45 @1e-12, max relative error over 20 times", worst, 1e-9);

    // Gauss-Newton recovery from the noise-free curve, using the analytic
    // Jacobian d obs/d theta = -1^T S (sens_chain with dlp_dy = -1). The
    // observable determines the rates only through (trace, disc, S), a
    // 3-dimensional function of the five parameters (the model file
    // states this; the 5 x 5 normal matrix is rank 3), so the identifiable
    // problem recovers (k1, k2, gamma) with the off-diagonal rates held
    // at truth. Correct sensitivities converge; a wrong chain rule stalls.
    // (k1 == k2 is a symmetric point where the (k1, k2, gamma) Jacobian has
    // rank 2 -- d trace and d disc are equal for the two rates there -- so the
    // start is taken off that line.)
    Lin2 th{0.45, 0.35, truth.a21, truth.a12, 0.8, 100.0};
    auto residual = [&](const Lin2& q, arma::vec& res, arma::mat* J) {
        std::vector<arma::mat> dA; std::vector<arma::vec> db, dy0;
        q.dirs(dA, db, dy0);
        const auto r = ode::linear_sens(q.A(), zb, q.y0(), ts, dA, db, dy0);
        res.set_size(20);
        if (J) J->set_size(20, 3);
        for (int i = 1; i <= 20; ++i) {
            res(i - 1) = (q.T - r.y(i, 0) - r.y(i, 1)) - e_lin(i);
            if (J) {
                const arma::rowvec g = -(r.S[i].row(0) + r.S[i].row(1));   // d obs_i / d theta
                (*J)(i - 1, 0) = g(0); (*J)(i - 1, 1) = g(1); (*J)(i - 1, 2) = g(4);
            }
        }
    };
    // Levenberg-Marquardt with step acceptance (the plain Gauss-Newton step
    // overshoots on this curved, nearly rank-deficient problem)
    double err = 1.0, mu = 1e-3;
    arma::vec res; arma::mat J;
    residual(th, res, &J);
    for (int it = 0; it < 200 && err > 1e-12; ++it) {
        const arma::mat JtJ = J.t() * J;
        arma::vec step;
        if (!arma::solve(step, JtJ + mu * arma::diagmat(JtJ.diag() + 1e-12), -J.t() * res)) break;
        Lin2 trial = th;
        trial.k1 += step(0); trial.k2 += step(1); trial.gam += step(2);
        trial.gam = std::min(0.999, std::max(0.001, trial.gam));
        trial.k1 = std::max(1e-4, trial.k1); trial.k2 = std::max(1e-4, trial.k2);
        arma::vec res_t; arma::mat J_t;
        residual(trial, res_t, &J_t);
        if (arma::dot(res_t, res_t) < arma::dot(res, res)) { th = trial; res = res_t; J = J_t; mu = std::max(mu * 0.3, 1e-15); }
        else mu *= 10.0;
        err = std::max({std::abs(th.k1 - truth.k1), std::abs(th.k2 - truth.k2), std::abs(th.gam - truth.gam)});
    }
    std::printf("      recovered k1=%.12f k2=%.12f gam=%.12f (truth %.2f %.2f %.2f)\n", th.k1, th.k2, th.gam, truth.k1, truth.k2, truth.gam);
    check_tol("Gauss-Newton from a wrong start recovers (k1, k2, gamma) via linear_sens", err, 1e-8);

    // the model Jacobian itself vs Richardson FD and vs the Van Loan path
    {
        std::vector<arma::mat> dA; std::vector<arma::vec> db, dy0;
        truth.dirs(dA, db, dy0);
        const auto r = ode::linear_sens(truth.A(), zb, truth.y0(), ts, dA, db, dy0);
        const auto rg = ld::linear_sens_general(truth.A(), zb, truth.y0(), ts, dA, db, dy0);
        double wfd = 0.0, wvl = 0.0;
        for (int j = 0; j < 5; ++j) {
            Dir d{"", dA[j], db[j], dy0[j]};
            double drift;
            const arma::mat F = fd_richardson(truth.A(), zb, truth.y0(), ts, d, drift);
            arma::mat S(21, 2), Sg(21, 2);
            for (int i = 0; i <= 20; ++i) { S.row(i) = r.S[i].col(j).t(); Sg.row(i) = rg.S[i].col(j).t(); }
            wfd = std::max(wfd, rel_err(S, F));
            wvl = std::max(wvl, rel_err(S, Sg));
        }
        check_tol("model Jacobian (5 params incl. k2 on a22 and gamma on y0) vs Richardson FD", wfd, 1e-9);
        check_tol("model Jacobian closed form vs Van Loan (x 2^s)", wvl / pade_2s(truth.A(), zb, ts), 1e-13);
    }
}

// ===========================================================================
//  T4  stress
// ===========================================================================
struct CancelRef { double a21, t, pad; double y[2]; double S[3][2]; };
// generated by gen_cancel_ref.py (adv/hpref.py, decimal at 120 digits; Taylor,
// not Pade; nothing shared with the header). A = [[3, 1.7],[a21, -3]],
// b = (0.4, -0.3), y0 = (1.3, -0.8); S columns: dA(0,0), dA(1,1), db[0].
static const CancelRef kCancelRef[] = {
  {-5.2941176470588234, 1.0, 0, {4.58500000000000085e+00, -6.19117647058823550e+00}, {{6.39125000000000121e+00, -6.18750000000000089e+00}, {-2.12125000000000030e+00, 3.49264705882352866e-01}, {2.50000000000000000e+00, -2.64705882352941169e+00}}},
  {-5.2941176470588234, 10.0, 0, {6.52000000000004292e+01, -1.09505882352941853e+02}, {{2.80250000000001410e+03, -4.46029411764708038e+03}, {-2.28550000000001000e+03, 3.58317647058825014e+03}, {1.60000000000000711e+02, -2.64705882352942297e+02}}},
  {-5.2941176470588234, 100.0, 0, {3.74530000000169139e+03, -6.56727058823824063e+03}, {{1.02443300000037085e+07, -1.78491176470652446e+07}, {-9.98680000000356883e+06, 1.73968317647120617e+07}, {1.51000000000063537e+04, -2.64705882353051820e+04}}},
  {-5.2941176470588234, 1000.0, 0, {3.47941300014665991e+05, -6.13606682378788013e+05}, {{8.78384213029637604e+10, -1.54803441181690369e+11}, {-8.76056800029519958e+10, 1.54392925675787262e+11}, {1.50100000006277859e+06, -2.64705882364004990e+06}}},
  {-5.2941176470588234, 10000.0, 0, {3.45294014444441795e+07, -6.09301775254557580e+07}, {{8.64085344904920250e+14, -1.52465329039648625e+15}, {-8.63855070890764375e+14, 1.52424694479821125e+15}, {1.50010000627033055e+08, -2.64705883459322602e+08}}},
  {-5.2941176470588234, 100000.0, 0, {3.45029544352979660e+09, -6.08871607528923702e+09}, {{8.62658791896309350e+18, -1.52231874780093030e+19}, {-8.62635789144758170e+18, 1.52227815491291402e+19}, {1.50001062695793686e+10, -2.64705992991101913e+10}}},
  {-5.2941176470588234, 1000000.0, 0, {3.45017360326025940e+11, -6.08853759363444824e+11}, {{8.62544690892388067e+22, -1.52213566027059937e+23}, {-8.62542390749624921e+22, 1.52213160119716488e+23}, {1.50006369607905273e+12, -2.64716946352139551e+12}}},
  {-5.2941176470588225, 668900000.0, 0, {1.83239974114899461e+27, -3.23364655368533727e+27}, {{3.82607220396954080e+43, -6.75189202010782833e+43}, {-3.82607208548600006e+43, 6.75189181101923081e+43}, {7.96695399294181406e+27, -1.40593303655957931e+28}}},
};
// A NON-MIRROR cancellation matrix: a11 - a22 is inexact in double, so the
// rounded d = (a11 - a22)/2 carries an error the size of the s^2 residual
// (-4.8e-16). Scale direction dA = A, for which d(s^2) = 2 s^2 exactly.
struct DroundRef { double t; double y[2]; double S[2][2]; };
// generated by gen_dround_ref.py (adv/hpref.py, 120 digits)
static const DroundRef kDroundRef[] = {
  {100.0, {5.78213840707314648e+03, -1.01477973932604982e+04}, {{8.29374546245839701e+03, -1.46194591590937343e+04}, {1.71036075178202316e+07, -2.98197658799034953e+07}}},
  {1000.0, {4.52989834006561860e+07, -7.99304481241047829e+07}, {{2.88174360822019458e+08, -5.08523313387792349e+08}, {1.76247924321914766e+13, -3.10765040498458516e+13}}},
  {3000.0, {3.35893968043381914e+13, -5.92852474745235156e+13}, {{6.22382202304621125e+14, -1.09851123471130038e+15}, {1.36976952696459166e+20, -2.41708005309404119e+20}}},
  {10000.0, {5.78209267357799467e+32, -1.02062384015032346e+33}, {{3.55821839769261566e+34, -6.28077888288467107e+34}, {2.80351487393172591e+40, -4.94827856535182260e+40}}},
  {30000.0, {4.59265060196632213e+86, -8.10688332815127830e+86}, {{8.47470624655548717e+88, -1.49594351749852805e+89}, {2.04575220489285694e+95, -3.61105285988661297e+95}}},
};
// y0, b, dy0, db along the null direction v = (a12, -d) of N = A - m I on the
// cancellation matrix. kase 1: y0 = v, b = 0; kase 2: y0 = (1.3, -0.8), b = v.
// Directions: dA(0,0); dy0 = v; db = v.
struct NullRef { int kase; double t; double y[2]; double S[3][2]; };
// generated by gen_nullvec_ref.py (adv/hpref.py, 120 digits)
static const NullRef kNullRef[] = {
  {1, 1.0, {1.70000000000000040e+00, -3.00000000000000044e+00}, {{4.25000000000000000e+00, -4.50000000000000000e+00}, {1.70000000000000040e+00, -3.00000000000000044e+00}, {1.70000000000000018e+00, -3.00000000000000000e+00}}},
  {1, 100.0, {1.70000000000426321e+00, -3.00000000000747313e+00}, {{2.56700000000215987e+04, -4.50000000000376167e+04}, {1.70000000000426321e+00, -3.00000000000747313e+00}, {1.70000000000142109e+02, -3.00000000000248292e+02}}},
  {1, 10000.0, {1.70000004263256432e+00, -3.00000007522892131e+00}, {{2.55017002131912410e+08, -4.50000003761696815e+08}, {1.70000004263256432e+00, -3.00000007522892131e+00}, {1.70000001421085472e+04, -3.00000002507547106e+04}}},
  {1, 1000000.0, {1.70042634346072963e+00, -3.00075237031145114e+00}, {{2.55021486845071240e+12, -4.50037617911732520e+12}, {1.70042634346072963e+00, -3.00075237031145114e+00}, {1.70014211211098963e+06, -3.00025078582742671e+06}}},
  {2, 1.0, {5.54000000000000092e+00, -8.28235294117647136e+00}, {{7.49000000000000110e+00, -7.18235294117647083e+00}, {1.70000000000000040e+00, -3.00000000000000044e+00}, {1.70000000000000018e+00, -3.00000000000000000e+00}}},
  {2, 100.0, {4.25300000000357727e+02, -7.49035294118272077e+02}, {{2.16083000000109756e+06, -3.77558823529602261e+06}, {1.70000000000426321e+00, -3.00000000000747313e+00}, {1.70000000000142109e+02, -3.00000000000248292e+02}}},
  {2, 10000.0, {4.24013003544680396e+04, -7.48243300372339872e+04}, {{2.12040702363646558e+12, -3.74152060700239941e+12}, {1.70000004263256432e+00, -3.00000007522892131e+00}, {1.70000001421085472e+04, -3.00000002507547106e+04}}},
  {2, 1000000.0, {4.24035574464990757e+06, -7.48297923110963032e+06}, {{2.12011040287680973e+18, -3.74136755770285978e+18}, {1.70042634346072963e+00, -3.00075237031145114e+00}, {1.70014211211098963e+06, -3.00025078582742671e+06}}},
};

static void T4() {
    std::printf("\n[T4] stress\n");
    std::mt19937_64 rng(4242ULL);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    // (a) s -> 0 sweep down to s = 0 exactly: A = [[l, c],[0, l + 2^-k]]
    {
        const double l = -1.0, c = 0.7, b1 = 0.4, b2 = -0.3, y01 = 1.3, y02 = -0.8, t = 2.0;
        arma::vec b{b1, b2}, y0{y01, y02}, ts{0.0, t};
        const arma::vec yc = ref_tri_confluent(l, c, b1, b2, y01, y02, t);
        const double den = std::max(std::abs(yc(0)), std::abs(yc(1)));
        double worst_ratio = 0.0, worst_exact = 0.0, worst_dcont = 0.0;
        bool fin = true;
        // derivative reference at s = 0: a shift of both diagonal entries and
        // the k2-type direction dA(1,1), both from the closed form at s == 0
        arma::mat A0{{l, c}, {0.0, l}};
        Dir dshift{"shift", arma::mat{{1.0, 0.0}, {0.0, 1.0}}, arma::vec{0.0, 0.0}, arma::vec{0.0, 0.0}};
        Dir da22{"a22", arma::mat{{0.0, 0.0}, {0.0, 1.0}}, arma::vec{0.0, 0.0}, arma::vec{0.0, 0.0}};
        const arma::mat D0s = dsolve(A0, b, y0, ts, dshift), D0a = dsolve(A0, b, y0, ts, da22);
        std::printf("      %-9s %-16s %-16s %-10s %-10s\n", "s", "y1", "y2", "|y-yconf|", "|dy/da22 - at s=0|");
        for (int k = 0; k <= 60; k += 4) {
            const double delta = std::ldexp(1.0, -k);
            arma::mat A{{l, c}, {0.0, l + delta}};
            const arma::mat Y = ode::linear(A, b, y0, ts);
            const arma::mat Ds = dsolve(A, b, y0, ts, dshift), Da = dsolve(A, b, y0, ts, da22);
            if (!all_finite(Y) || !all_finite(Ds) || !all_finite(Da)) fin = false;
            const double gap = std::max(std::abs(Y(1, 0) - yc(0)), std::abs(Y(1, 1) - yc(1)));
            const double dgap = std::max(rel_err(Ds, D0s), rel_err(Da, D0a));
            if (k % 12 == 0) std::printf("      %-9.1e %-16.12f %-16.12f %-10.2e %-10.2e\n",
                                         delta / 2.0, Y(1, 0), Y(1, 1), gap, dgap);
            // |y(s) - y(0)| = O(s) asymptotically: gap/(2s + 4eps) bounded all
            // the way down (a formula that breaks down shows gap/s exploding)
            if (delta < 1e-9) {
                worst_ratio = std::max(worst_ratio, gap / den / (delta + 4.0 * EPS));
                // dy/dtheta continuous: deviation from the s = 0 value is O(s) too
                worst_dcont = std::max(worst_dcont, dgap / (delta * 4.0 + 1e-14));
            }
        }
        check(fin, "s -> 0 sweep: every value and derivative finite down to s = 2^-61");
        check_tol("s -> 0 sweep, s < 1e-9: |y(s) - y(0)|/|y| <= C s down to the rounding floor (C)", worst_ratio, 2.0);
        check_tol("s -> 0 sweep, s < 1e-9: dy/dtheta continuous through s = 0 (|dS|/s bounded, C)", worst_dcont, 2.0);
        // s == 0 exactly, several times, vs the confluent closed form
        arma::vec tss{0.0, 0.1, 1.0, 2.0, 20.0, -3.0};
        const arma::mat Y0 = ode::linear(A0, b, y0, tss);
        for (arma::uword i = 1; i < tss.n_elem; ++i) {
            const arma::vec r = ref_tri_confluent(l, c, b1, b2, y01, y02, tss(i));
            worst_exact = std::max(worst_exact, std::max(std::abs(Y0(i, 0) - r(0)), std::abs(Y0(i, 1) - r(1))) /
                                                    std::max(std::abs(r(0)), std::abs(r(1))));
        }
        check_tol("s == 0 EXACTLY (defective A) vs the confluent closed form, t in {.1,1,2,20,-3}", worst_exact, 5e-15);
        // derivative at s == 0 vs Richardson FD and vs Van Loan
        {
            double drift, w = 0.0, wv = 0.0;
            for (const auto& d : all_directions(2)) {
                const arma::mat S = dsolve(A0, b, y0, tss, d);
                const arma::mat F = fd_richardson(A0, b, y0, tss, d, drift);
                const arma::mat G = dsolve_general(A0, b, y0, tss, d);
                w = std::max(w, rel_err(S, F));
                wv = std::max(wv, rel_err(S, G));
            }
            check_tol("dy/dtheta at s == 0 exactly vs Richardson FD (all 13 directions)", w, 1e-9);
            check_tol("dy/dtheta at s == 0 exactly vs Van Loan (all 13 directions, x 2^s)", wv / pade_2s(A0, b, tss), 1e-13);
        }
        // complex approach: s^2 = -2^-k
        double wc = 0.0; bool fc = true;
        for (int k = 0; k <= 60; k += 2) {
            arma::mat A{{l, std::ldexp(1.0, -k)}, {-1.0, l}};
            const arma::mat Y = ode::linear(A, b, y0, ts), G = ld::linear_general(A, b, y0, ts);
            if (!all_finite(Y)) fc = false;
            wc = std::max(wc, rel_err(Y, G) / pade_2s(A, b, ts));
        }
        check(fc, "complex s^2 = -2^-k, k = 0..60: all finite");
        check_tol("complex s^2 = -2^-k, k = 0..60: closed form vs Pade (x 2^s)", wc, 1e-13);
    }

    // (b) stiff ratios to 1e8 vs the exact triangular reference; rk45 reported
    {
        std::printf("      stiff: A = [[-1, 0.315],[0, fast]], t in {1e-8 .. 1}\n");
        double worst = 0.0, worst_S = 0.0;
        for (int k = 4; k <= 8; ++k) {
            const double fast = -std::pow(10.0, double(k));
            arma::mat A{{-1.0, 0.315}, {0.0, fast}};
            arma::vec b{0.7, -0.3}, y0{1.0, -2.0};
            arma::vec ts{0.0, 1e-8, 5.0 / std::abs(fast), 0.2, 1.0};
            const arma::mat Y = ode::linear(A, b, y0, ts);
            double w = 0.0;
            for (arma::uword i = 1; i < ts.n_elem; ++i) {
                const arma::vec r = ref_tri(-1.0, fast, 0.315, 0.7, -0.3, 1.0, -2.0, ts(i));
                w = std::max(w, std::max(std::abs(Y(i, 0) - r(0)), std::abs(Y(i, 1) - r(1))) / std::max(std::abs(r(0)), std::abs(r(1))));
            }
            std::string rk;
            try {
                const arma::mat R = ref_rk45(A, b, y0, ts, 1e-12, 4000000);
                char buf[64]; std::snprintf(buf, sizeof buf, "rk45 err %.2e", rel_err(R, Y)); rk = buf;
            } catch (const std::exception&) { rk = "rk45 THREW (max_iter / stiff)"; }
            // sensitivity vs Van Loan (the general path squares 20+ times here,
            // so it is the WEAKER party; report only) and vs FD in the a22
            // direction, which is a pure e^{fast t} derivative: exact ref
            const arma::mat S = dsolve(A, b, y0, ts, Dir{"a22", arma::mat{{0.0, 0.0}, {0.0, 1.0}}, arma::vec{0.0, 0.0}, arma::vec{0.0, 0.0}});
            double ws = 0.0, wsc = 0.0, smax = 0.0;
            arma::vec refs(ts.n_elem, arma::fill::zeros);
            for (arma::uword i = 1; i < ts.n_elem; ++i) {
                // y2 = K e^{lt} + M, K = y02 + b2/l, M = -b2/l  ->
                // dy2/dl = t K e^{lt} - (b2/l^2) e^{lt} + b2/l^2
                const double t = ts(i), e = std::exp(fast * t);
                const double K = -2.0 + (-0.3) / fast;
                refs(i) = t * K * e - (-0.3) / (fast * fast) * e + (-0.3) / (fast * fast);
                smax = std::max(smax, std::abs(refs(i)));
            }
            for (arma::uword i = 1; i < ts.n_elem; ++i) {
                ws  = std::max(ws,  std::abs(S(i, 1) - refs(i)) / smax);                       // rel. to |S|_inf
                wsc = std::max(wsc, std::abs(S(i, 1) - refs(i)) / std::max(std::abs(refs(i)), 1e-300));  // componentwise
            }
            std::printf("      ratio 1e%d: y err %.2e (exact ref), dy2/da22 err %.2e rel |S|_inf / %.2e componentwise (steady-state value b2/l^2 = %.1e), %s\n",
                        k, w, ws, wsc, -0.3 / (fast * fast), rk.c_str());
            worst = std::max(worst, w);
            worst_S = std::max(worst_S, ws);
        }
        check_tol("stiff 1e4..1e8 vs exact triangular reference (rel. to |y|_inf)", worst, 1e-14);
        check_tol("stiff 1e4..1e8: d y2/d a22 vs hand derivative (rel. to |S|_inf; componentwise = the documented limit)", worst_S, 1e-14);
    }

    // (c) overflow prediction vs actual, 20000 random systems, |b|,|y0| to 1e300
    {
        int mism = 0, n_over = 0, total = 0;
        const double scales[] = {1.0, 1e50, 1e100, 1e150, 1e200, 1e250, 1e300};
        for (int rep = 0; rep < 20000; ++rep) {
            const double sc = scales[rep % 7];
            const arma::uword n = (rep % 5 == 0) ? 3 : ((rep % 11 == 0) ? 1 : 2);
            arma::mat A(n, n);
            for (auto& v : A) v = 20.0 * U(rng);
            arma::vec b(n), y0(n);
            for (auto& v : b) v = sc * U(rng);
            for (auto& v : y0) v = sc * U(rng);
            if (rep % 3 == 0) b.zeros();
            const double t = std::pow(10.0, 3.0 * U(rng)) * ((rep % 13 == 0) ? -1.0 : 1.0);
            const bool pred = ode::linear_overflows(A, b, y0, t);
            const arma::mat Y = ode::linear(A, b, y0, arma::vec{0.0, t});
            const bool actual = !Y.row(1).is_finite();
            ++total;
            if (pred != actual) ++mism;
            if (actual) ++n_over;
        }
        std::printf("      %d / %d systems overflow; predictor mismatches: %d\n", n_over, total, mism);
        check(mism == 0 && n_over > 1000, "linear_overflows == actual non-finiteness on 20000 random systems (n=1,2,3; |b|,|y0| to 1e300)");
        // the reviewer's minimal example and the A-only false negative
        check(ode::linear_overflows(arma::mat{{2.0, 0.0}, {0.0, -1.0}}, arma::vec{0.0, 0.0}, arma::vec{1e300, 1.0}, 100.0),
              "A = diag(2,-1), y0 = (1e300, 1), t = 100 is predicted to overflow (A-only predictor said no)");
        check(!ode::linear_overflows(arma::mat{{2.0, 0.0}, {0.0, -1.0}}, arma::vec{0.0, 0.0}, arma::vec{1.0, 1.0}, 354.0),
              "A = diag(2,-1), t = 354 (2t = 708) finite and predicted finite");
        check(ode::linear_overflows(arma::mat{{2.0, 0.0}, {0.0, -1.0}}, arma::vec{0.0, 0.0}, arma::vec{1.0, 1.0}, 355.0),
              "A = diag(2,-1), t = 355 (2t = 710) overflows and predicted so");
    }

    // (d) the reviewer's cancellation matrix vs the pinned 120-digit reference
    {
        std::printf("      A = [[3, 1.7],[a21, -3]] with a12 a21 = -9 + 5.0e-16 (rounded sum is exactly 0):\n");
        std::printf("      %-12s %-12s %-12s %-12s %-12s\n", "t", "y err", "S dA(0,0)", "S dA(1,1)", "S db[0]");
        double wy = 0.0, wS = 0.0;
        for (const auto& r : kCancelRef) {
            arma::mat A{{3.0, 1.7}, {r.a21, -3.0}};
            arma::vec b{0.4, -0.3}, y0{1.3, -0.8}, ts{0.0, r.t};
            const arma::mat Y = ode::linear(A, b, y0, ts);
            const double ey = std::max(std::abs(Y(1, 0) - r.y[0]), std::abs(Y(1, 1) - r.y[1])) / std::max(std::abs(r.y[0]), std::abs(r.y[1]));
            const Dir dd[3] = {{"", arma::mat{{1.0, 0.0}, {0.0, 0.0}}, arma::vec{0.0, 0.0}, arma::vec{0.0, 0.0}},
                               {"", arma::mat{{0.0, 0.0}, {0.0, 1.0}}, arma::vec{0.0, 0.0}, arma::vec{0.0, 0.0}},
                               {"", arma::mat(2, 2, arma::fill::zeros), arma::vec{1.0, 0.0}, arma::vec{0.0, 0.0}}};
            double es[3];
            for (int j = 0; j < 3; ++j) {
                const arma::mat S = dsolve(A, b, y0, ts, dd[j]);
                es[j] = std::max(std::abs(S(1, 0) - r.S[j][0]), std::abs(S(1, 1) - r.S[j][1])) / std::max(std::abs(r.S[j][0]), std::abs(r.S[j][1]));
                wS = std::max(wS, es[j]);
            }
            wy = std::max(wy, ey);
            std::printf("      %-12.4g %-12.2e %-12.2e %-12.2e %-12.2e\n", r.t, ey, es[0], es[1], es[2]);
        }
        check_tol("cancellation matrix: y vs 120-digit reference, t = 1 .. 6.7e8 (flat in t)", wy, 1e-14);
        check_tol("cancellation matrix: S (3 dirs) vs 120-digit reference, t = 1 .. 6.7e8", wS, 1e-13);
        // and s^2, det themselves
        double s2, det;
        ld::s2_det_compensated(3.0, 1.7, -5.2941176470588234, -3.0, s2, det);
        check_tol("compensated s^2 for the cancellation matrix vs exact 5.015595781836e-16",
                  std::abs(s2 - 5.015595781836e-16) / 5.015595781836e-16, 1e-9);
        check_tol("compensated det for the cancellation matrix vs exact -5.015595781836e-16",
                  std::abs(det + 5.015595781836e-16) / 5.015595781836e-16, 1e-9);
        // a11 - a22 = Dh + Dl inexact, a21 = -Dh/2 and a12 = Dh/2 + Dl, so
        // a12 a21 = -(d^2 + 2 d dl) exactly and s^2 = dl^2 = 4.93e-32. That is
        // BELOW the resolution eps^2 d^2 of a two-product expansion (dl <= ulp/4
        // makes dl^2 <= eps^2 d^2 always), so the documented floor is what is
        // checked here, not the value.
        ld::s2_det_compensated(3.007, 3.0080500000000003, -3.00805, -3.0091, s2, det);
        {
            const double dd = 0.5 * (3.007 + 3.0091), eps = std::numeric_limits<double>::epsilon();
            check_tol("compensated s^2 within the two-product resolution 4 eps^2 d^2 of the exact dl^2 (4.93e-32 lies below it)",
                      std::abs(s2 - 4.930380657631324e-32) / (4.0 * eps * eps * dd * dd), 1.0);
        }
    }

    // (d2) the non-mirror cancellation matrix: the rounded-d dual path gave
    //      2.2e-9 in S here (y was 1.1e-14); the two_sum halves make S match y
    {
        std::printf("      A = [[3.007, 1.7],[a21, -2.9947]], a12 a21 = -d^2 - 4.8e-16, a11 - a22 inexact:\n");
        std::printf("      %-12s %-12s %-12s %-12s\n", "t", "y err", "S dA = A", "S dA(0,0)");
        const arma::mat A{{3.007, 1.7}, {-5.297118072058824, -2.9947}};
        const arma::vec b{0.4, -0.3}, y0{1.3, -0.8};
        const Dir dd[2] = {{"", A, arma::vec{0.0, 0.0}, arma::vec{0.0, 0.0}},
                           {"", arma::mat{{1.0, 0.0}, {0.0, 0.0}}, arma::vec{0.0, 0.0}, arma::vec{0.0, 0.0}}};
        double wy = 0.0, wS = 0.0;
        for (const auto& r : kDroundRef) {
            const arma::vec ts{0.0, r.t};
            const arma::mat Y = ode::linear(A, b, y0, ts);
            const double ey = std::max(std::abs(Y(1, 0) - r.y[0]), std::abs(Y(1, 1) - r.y[1])) / std::max(std::abs(r.y[0]), std::abs(r.y[1]));
            double es[2];
            for (int j = 0; j < 2; ++j) {
                const arma::mat S = dsolve(A, b, y0, ts, dd[j]);
                es[j] = std::max(std::abs(S(1, 0) - r.S[j][0]), std::abs(S(1, 1) - r.S[j][1])) / std::max(std::abs(r.S[j][0]), std::abs(r.S[j][1]));
                wS = std::max(wS, es[j]);
            }
            wy = std::max(wy, ey);
            std::printf("      %-12.4g %-12.2e %-12.2e %-12.2e\n", r.t, ey, es[0], es[1]);
        }
        check_tol("non-mirror cancellation matrix: y vs 120-digit reference, t = 1e2 .. 3e4", wy, 1e-13);
        check_tol("non-mirror cancellation matrix: S (dA = A, dA(0,0)) vs 120-digit reference", wS, 1e-13);
    }

    // (d3) y0, b, dy0 and db along the null direction v = (a12, -d) of
    //      N = A - m I: with plain products N v carried an absolute error
    //      eps |d| |v| that the term t Es (N v) turned into eps |d| t (7e-11 in
    //      y and 2e-10 in S at t = 1e6); the compensated N v keeps both flat
    {
        std::printf("      null direction v = (1.7, -3): kase 1 y0 = v, b = 0; kase 2 y0 = (1.3, -0.8), b = v\n");
        std::printf("      %-5s %-10s %-12s %-12s %-12s %-12s\n", "kase", "t", "y err", "S dA(0,0)", "S dy0 = v", "S db = v");
        const arma::mat A{{3.0, 1.7}, {-5.2941176470588234, -3.0}};
        const arma::vec v{1.7, -3.0}, z{0.0, 0.0};
        const Dir dd[3] = {{"", arma::mat{{1.0, 0.0}, {0.0, 0.0}}, z, z},
                           {"", arma::mat(2, 2, arma::fill::zeros), z, v},
                           {"", arma::mat(2, 2, arma::fill::zeros), v, z}};
        double wy = 0.0, wS = 0.0;
        for (const auto& r : kNullRef) {
            const arma::vec b  = (r.kase == 1) ? z : v;
            const arma::vec y0 = (r.kase == 1) ? v : arma::vec{1.3, -0.8};
            const arma::vec ts{0.0, r.t};
            const arma::mat Y = ode::linear(A, b, y0, ts);
            const double ey = std::max(std::abs(Y(1, 0) - r.y[0]), std::abs(Y(1, 1) - r.y[1])) / std::max(std::abs(r.y[0]), std::abs(r.y[1]));
            double es[3];
            for (int j = 0; j < 3; ++j) {
                const arma::mat S = dsolve(A, b, y0, ts, dd[j]);
                es[j] = std::max(std::abs(S(1, 0) - r.S[j][0]), std::abs(S(1, 1) - r.S[j][1])) / std::max(std::abs(r.S[j][0]), std::abs(r.S[j][1]));
                wS = std::max(wS, es[j]);
            }
            wy = std::max(wy, ey);
            std::printf("      %-5d %-10.3g %-12.2e %-12.2e %-12.2e %-12.2e\n", r.kase, r.t, ey, es[0], es[1], es[2]);
        }
        check_tol("null-direction y0 / b: y vs 120-digit reference, t = 1 .. 1e6 (flat)", wy, 1e-14);
        check_tol("null-direction dA(0,0), dy0 = v, db = v: S vs 120-digit reference, t = 1 .. 1e6", wS, 1e-13);
    }

    // (e) unit invariance: A -> sc A, t -> t/sc leaves y unchanged; sc to 1e+-170
    {
        arma::mat A0{{-1.0, 0.0}, {1.0, -2.0}};
        arma::vec b0{0.0, 0.0}, y0{1.0, -2.0};
        const arma::mat Yref = ode::linear(A0, b0, y0, arma::vec{0.0, 1.0});
        double worst = 0.0; bool fin = true;
        for (double sc : {1e-170, 1e-160, 1e-100, 1e100, 1e160, 1e170, 1e300}) {
            const arma::mat Y = ode::linear(sc * A0, b0, y0, arma::vec{0.0, 1.0 / sc});
            if (!all_finite(Y)) fin = false;
            worst = std::max(worst, rel_err(Y, Yref));
        }
        check(fin, "unit invariance |A| = 1e-170 .. 1e300: finite (q = s^2 would have left the range)");
        check_tol("unit invariance |A| = 1e-170 .. 1e300: y unchanged", worst, 1e-14);
        // and with forcing, sensitivities too
        arma::vec b1{0.7, -0.3};
        const auto r0 = ode::linear_sens(A0, b1, y0, arma::vec{0.0, 1.0}, {arma::mat{{1.0, 0.0}, {0.0, 0.0}}}, {arma::vec{1.0, 0.0}}, {arma::vec{0.0, 0.0}});
        double ws = 0.0, wy = 0.0;
        for (double sc : {1e-170, 1e-100, 1e100, 1e170}) {
            // y' = sc A y + sc b on t/sc; d/d(a11) of the scaled system = sc * d/d(sc a11)
            const auto r = ode::linear_sens(sc * A0, sc * b1, y0, arma::vec{0.0, 1.0 / sc},
                                            {arma::mat{{sc, 0.0}, {0.0, 0.0}}}, {arma::vec{sc, 0.0}}, {arma::vec{0.0, 0.0}});
            wy = std::max(wy, rel_err(r.y, r0.y));
            ws = std::max(ws, rel_err(r.S[1], r0.S[1]));
        }
        check_tol("unit invariance with forcing: y", wy, 1e-14);
        check_tol("unit invariance with forcing: S", ws, 1e-13);
    }

    // (f) the documented limitation, measured (report only + the |y|_inf guarantee)
    {
        std::printf("      subdominant component, A = diag(-1,-1-g), t = 10:\n      %-8s %-8s %-14s %-14s %-12s\n",
                    "g", "2q", "rel.err(y2)", "eps e^2q", "rel |y|_inf");
        double worst_norm = 0.0;
        for (double g : {0.2, 0.5, 1.0, 2.0, 3.0, 4.0}) {
            arma::mat A{{-1.0, 0.0}, {0.0, -1.0 - g}};
            const arma::mat Y = ode::linear(A, arma::vec{0.0, 0.0}, arma::vec{1.0, 1.0}, arma::vec{0.0, 10.0});
            const double e1 = std::exp(-10.0), e2 = std::exp(-10.0 * (1.0 + g));
            const double r2 = std::abs(Y(1, 1) - e2) / e2;
            const double rn = std::max(std::abs(Y(1, 0) - e1), std::abs(Y(1, 1) - e2)) / e1;
            worst_norm = std::max(worst_norm, rn);
            std::printf("      %-8.2g %-8.3g %-14.3e %-14.3e %-12.3e\n", g, 10.0 * g, r2, EPS * std::exp(10.0 * g), rn);
        }
        check_tol("subdominant cases: accuracy relative to |y|_inf (the stated guarantee)", worst_norm, 5e-16);
    }
}

// ===========================================================================
//  AD-twin (Check #12): autodiff twins of the general-n path
// ===========================================================================
// exp(M) for an Eigen matrix of an autodiff scalar: scale to ||M||_1 <= thr,
// Taylor to K terms, s squarings. Nothing here is shared with the header
// (Taylor, not Pade; Eigen, not Armadillo; the library's autodiff types, not
// the header's hand-written dual / Van Loan block).
//
// The vendored reverse-mode `var` propagates through the expression tree
// WITHOUT memoisation (BinaryExpr::propagate recurses into both children), so
// a matrix DAG with K chained products and s squarings costs m^K (2m)^s node
// visits. The reverse-mode twin (through autodiff_wrap::wrap_real, as the
// library's Check #12 runs it) is therefore driven on spans with
// ||Atil tau||_1 <= 0.1 and K = 10 (truncation 0.1^11/11! = 2.5e-19, no
// squaring); the forward-mode `dual` twin has no graph and runs the full spans
// including the stiff family, with the 2^s eps squaring allowance.
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

static double dbl(const autodiff::var& x)  { return double(x); }
static double dbl(const autodiff::dual& x) { return x.val; }

template <class S>
static Eigen::Matrix<S, -1, -1> expm_twin(const Eigen::Matrix<S, -1, -1>& M, double thr, int K, int& s_out) {
    using Mat = Eigen::Matrix<S, -1, -1>;
    const int n = int(M.rows());
    double nrm = 0.0;
    for (int j = 0; j < n; ++j) {
        double c = 0.0;
        for (int i = 0; i < n; ++i) c += std::abs(dbl(M(i, j)));
        nrm = std::max(nrm, c);
    }
    int s = 0;
    while (nrm > thr) { nrm *= 0.5; ++s; }
    s_out = s;
    const double sc = std::ldexp(1.0, -s);
    Mat X = M * sc;
    Mat T = Mat::Identity(n, n), term = Mat::Identity(n, n);
    for (int k = 1; k <= K; ++k) {
        term = (term * X) * (1.0 / double(k));
        T = T + term;
    }
    for (int k = 0; k < s; ++k) T = T * T;
    return T;
}

// lp(theta) = sum_i dlp(i,:) . y_i(theta) with theta = (vec A, b, y0), through
// the augmented Taylor twin in scalar type S.
template <class S>
static S lp_twin(const Eigen::Matrix<S, -1, 1>& th, arma::uword n, const arma::vec& ts,
                 const arma::mat& dlp, double thr, int K, int& smax) {
    using Mat = Eigen::Matrix<S, -1, -1>;
    using Vec = Eigen::Matrix<S, -1, 1>;
    Mat At = Mat::Zero(n + 1, n + 1);
    for (arma::uword j = 0; j < n; ++j)
        for (arma::uword i = 0; i < n; ++i) At(i, j) = th[j * n + i];
    for (arma::uword i = 0; i < n; ++i) At(i, n) = th[n * n + i];
    Vec yt(n + 1);
    for (arma::uword i = 0; i < n; ++i) yt[i] = th[n * n + n + i];
    yt[n] = 1.0;
    S lp = 0.0;
    for (arma::uword i = 0; i < n; ++i) lp += dlp(0, i) * yt[i];
    smax = 0;
    for (arma::uword k = 1; k < ts.n_elem; ++k) {
        int s = 0;
        const Mat E = expm_twin<S>(At * (ts(k) - ts(0)), thr, K, s);
        smax = std::max(smax, s);
        const Vec z = E * yt;
        for (arma::uword i = 0; i < n; ++i) lp += dlp(k, i) * z[i];
    }
    return lp;
}

static void AD_twin() {
    std::printf("\n[AD-twin] hand-written sensitivities vs autodiff twins of the general-n path\n");
    std::mt19937_64 rng(1234ULL);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    struct Fam { std::string name; int n; std::function<arma::mat()> make; arma::vec ts; };
    std::vector<Fam> fams;
    const arma::vec ts4{0.0, 0.2, 0.7, 1.5};
    fams.push_back({"random n=1", 1, [&] { arma::mat A(1, 1); A(0, 0) = -1.0 + 2.0 * U(rng); return A; }, ts4});
    fams.push_back({"random n=2", 2, [&] { arma::mat A(2, 2); for (auto& v : A) v = 2.0 * U(rng); return A; }, ts4});
    fams.push_back({"random n=3", 3, [&] { arma::mat A(3, 3); for (auto& v : A) v = 1.5 * U(rng); return A; }, ts4});
    fams.push_back({"random n=4", 4, [&] { arma::mat A(4, 4); for (auto& v : A) v = U(rng); return A; }, ts4});
    fams.push_back({"complex pair n=2", 2, [&] { return A_rot2(V2(), -0.3 + 0.3 * U(rng), 2.0 + 3.0 * U(rng)); }, ts4});
    fams.push_back({"near-equal n=2", 2, [&] { return A_from_diag(V2(), {-1.0, -1.0 - 1e-7 * (1.0 + U(rng))}); }, ts4});
    fams.push_back({"defective n=2", 2, [&] { return A_jordan2(V2(), -0.5 + 0.5 * U(rng)); }, ts4});
    fams.push_back({"stiff 1e3 n=2", 2, [&] { return A_from_diag(V2(), {-1.0, -1e3 * (1.0 + 0.5 * U(rng))}); }, arma::vec{0.0, 1e-3, 0.01, 0.5}});
    fams.push_back({"zero eigenvalue n=3", 3, [&] { return A_from_diag(V3(), {0.0, -1.0, -3.0 * (1.0 + 0.5 * U(rng))}); }, ts4});

    std::printf("      %-22s %12s %12s %12s %6s\n", "family", "rev-var short", "fwd-dual full", "fwd 2^s eps", "s");
    double worst_rev = 0.0, worst_fwd = 0.0;
    for (const auto& f : fams) {
        double wrev = 0.0, wfwd = 0.0, wall = 0.0;
        int smax_all = 0;
        for (int rep = 0; rep < 8; ++rep) {
            const arma::uword n = f.n;
            const arma::mat A = f.make();
            arma::vec b(n), y0(n);
            for (auto& v : b) v = U(rng);
            for (auto& v : y0) v = 1.0 + U(rng);
            arma::mat dlp(f.ts.n_elem, n);
            for (auto& v : dlp) v = U(rng);

            const arma::uword p = n * n + 2 * n;
            arma::vec theta(p);
            theta.head(n * n) = arma::vectorise(A);
            theta.subvec(n * n, n * n + n - 1) = b;
            theta.tail(n) = y0;
            std::vector<arma::mat> dA(p, arma::mat(n, n, arma::fill::zeros));
            std::vector<arma::vec> db(p, arma::vec(n, arma::fill::zeros)), dy0(p, arma::vec(n, arma::fill::zeros));
            for (arma::uword k = 0; k < n * n; ++k) dA[k][k] = 1.0;
            for (arma::uword k = 0; k < n; ++k) { db[n * n + k](k) = 1.0; dy0[n * n + n + k](k) = 1.0; }

            // ---- (a) reverse-mode var through autodiff_wrap::wrap_real, short span
            {
                const double nA = arma::norm(ld::augment(A, b), 1);
                const arma::vec tss = f.ts * (0.05 / (nA * f.ts.max()));
                const arma::mat dlps = dlp;
                const auto r = ode::linear_sens(A, b, y0, tss, dA, db, dy0);
                const arma::vec g_hand = ode::sens_chain(r, dlps);
                const auto rg = ld::linear_sens_general(A, b, y0, tss, dA, db, dy0);
                const arma::vec g_gen = ode::sens_chain(rg, dlps);
                arma::vec g_ad;
                int sm = 0;
                const double lp_ad = AI4BayesCode::autodiff_wrap::wrap_real(theta, &g_ad,
                    [&](const autodiff::VectorXvar& th) -> autodiff::var {
                        return lp_twin<autodiff::var>(th, n, tss, dlps, 0.1, 10, sm);
                    });
                if (sm != 0) { ++g_fail; std::printf("  [FAIL] reverse twin needed squaring\n"); }
                const double lp_hand = arma::accu(dlps % r.y);
                const double den = std::max(arma::abs(g_ad).max(), 1e-300);
                wrev = std::max({wrev, arma::abs(g_hand - g_ad).max() / den, arma::abs(g_gen - g_ad).max() / den,
                                 std::abs(lp_hand - lp_ad) / std::max(std::abs(lp_ad), 1e-300)});
            }
            // ---- (b) forward-mode dual, full span (one seeded pass per parameter)
            {
                const auto r = ode::linear_sens(A, b, y0, f.ts, dA, db, dy0);
                const arma::vec g_hand = ode::sens_chain(r, dlp);
                const auto rg = ld::linear_sens_general(A, b, y0, f.ts, dA, db, dy0);
                const arma::vec g_gen = ode::sens_chain(rg, dlp);
                arma::vec g_ad(p);
                double lp_ad = 0.0;
                int sm = 0;
                for (arma::uword j = 0; j < p; ++j) {
                    Eigen::Matrix<autodiff::dual, -1, 1> th(p);
                    for (arma::uword k = 0; k < p; ++k) { th[k].val = theta(k); th[k].grad = (k == j) ? 1.0 : 0.0; }
                    const autodiff::dual lp = lp_twin<autodiff::dual>(th, n, f.ts, dlp, 1.0 / 32.0, 18, sm);
                    g_ad(j) = lp.grad;
                    lp_ad = lp.val;
                }
                smax_all = std::max(smax_all, sm);
                const double allow = std::max(1e-11, 30.0 * std::ldexp(1.0, sm) * EPS);
                wall = std::max(wall, allow);
                const double lp_hand = arma::accu(dlp % r.y);
                const double den = std::max(arma::abs(g_ad).max(), 1e-300);
                const double e = std::max({arma::abs(g_hand - g_ad).max() / den, arma::abs(g_gen - g_ad).max() / den,
                                           std::abs(lp_hand - lp_ad) / std::max(std::abs(lp_ad), 1e-300)});
                wfwd = std::max(wfwd, e / allow);
            }
        }
        std::printf("      %-22s %12.3e %12.3e %12.3e %6d\n", f.name.c_str(), wrev, wfwd, wall, smax_all);
        worst_rev = std::max(worst_rev, wrev);
        worst_fwd = std::max(worst_fwd, wfwd);
    }
    check_tol("AD-twin (reverse var, autodiff_wrap::wrap_real): grad + lp vs sens_chain(linear_sens), short spans", worst_rev, 1e-12);
    check_tol("AD-twin (forward dual): grad + lp vs sens_chain(linear_sens), full spans incl. stiff (x max(1e-11, 30 2^s eps))", worst_fwd, 1.0);
}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("=== test_ode_linear: exact linear ODE + analytic sensitivities ===\n");
    if (std::getenv("T2ONLY")) { T2(); std::printf("\n%d passed, %d failed\n", g_pass, g_fail); return g_fail == 0 ? 0 : 1; }
    T0();
    T1();
    T2();
    std::printf("\n[T3] [SKIP] cross-chain R-hat is exercised end-to-end by a separate comparison, a separate step\n");
    T4();
    AD_twin();
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

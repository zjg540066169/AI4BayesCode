// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_ode_rk45_sens.cpp
//
// Correctness gate for AI4BayesCode::ode Tier-2 forward sensitivity analysis
// (rk45_sens autodiff path, rk45_sens_fd FD-of-RHS path, sens_chain).
//
// Tests
// -----
//  (A) LINEAR ODE  dy/dt = -k*y :  ANALYTIC sensitivity
//        y(t) = y0*exp(-k t),  S(t) = d y/d k = -t*y0*exp(-k t).
//      Both the autodiff and the FD-of-RHS paths must match the closed form
//      to < 1e-6 relative.
//
//  (B) SIR (n=3, p=2) and (C) LOTKA-VOLTERRA (n=2, p=4) :  no closed form,
//      so the reference is a high-accuracy central-FD-of-TRAJECTORY sensitivity
//      (Tier-1 rk45 re-solved at theta +/- h with tol 1e-11). rk45_sens (both
//      paths) must match this reference to < 1e-4 relative -- THE KEY GATE --
//      and the autodiff and FD-of-RHS paths must agree with each other, and the
//      augmented y must match a bare Tier-1 rk45 solve.
//
//  (C2) theta_idx SUBSET: tracking a subset of parameters returns exactly the
//       corresponding columns of the full sensitivity.
//
//  (D) INITIAL-CONDITION sensitivity via S0 seeding: a parameter that IS an
//      initial condition (RHS ignores it, J_theta column = 0) with S0 = I must
//      give d y/d y0 = exp(-k t) for the linear ODE.
//
//  (E) sens_chain: chains S into a parameter gradient and matches a direct
//      finite-difference of a scalar functional of the trajectory.
//
// Standalone int main; returns 0 iff every gate passes.
// ============================================================================

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#ifdef AI4BAYESCODE_RCPP_MODULE
#  include <RcppArmadillo.h>
#else
#  include <armadillo>
#endif

#include "AI4BayesCode/ode_rk45.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace ode = AI4BayesCode::ode;

// ---------------------------------------------------------------------------
// RHS definitions: a plain double (arma) form and a scalar-type-templated
// std::vector twin for the autodiff path.
// ---------------------------------------------------------------------------

// --- Linear: dy/dt = -k*y (n=1). theta = [k] (plus a dummy param for test D).
static arma::vec lin_rhs(double, const arma::vec& y, const arma::vec& th) {
    return arma::vec{ -th[0] * y[0] };
}
template <typename T>
static std::vector<T> lin_rhs_ad(double, const std::vector<T>& y,
                                 const std::vector<T>& th) {
    return std::vector<T>{ -th[0] * y[0] };
}

// --- SIR (n=3). theta = [beta, gamma].
static arma::vec sir_rhs(double, const arma::vec& y, const arma::vec& th) {
    const double S = y[0], I = y[1], R = y[2], N = y[0] + y[1] + y[2];
    return arma::vec{ -th[0] * S * I / N,
                       th[0] * S * I / N - th[1] * I,
                       th[1] * I };
}
template <typename T>
static std::vector<T> sir_rhs_ad(double, const std::vector<T>& y,
                                 const std::vector<T>& th) {
    T S = y[0], I = y[1], R = y[2], N = y[0] + y[1] + y[2];
    return std::vector<T>{ -th[0] * S * I / N,
                            th[0] * S * I / N - th[1] * I,
                            th[1] * I };
}

// --- Lotka-Volterra (n=2). theta = [alpha, beta, delta, gamma].
static arma::vec lv_rhs(double, const arma::vec& y, const arma::vec& th) {
    return arma::vec{ th[0] * y[0] - th[1] * y[0] * y[1],
                      th[2] * y[0] * y[1] - th[3] * y[1] };
}
template <typename T>
static std::vector<T> lv_rhs_ad(double, const std::vector<T>& y,
                                const std::vector<T>& th) {
    return std::vector<T>{ th[0] * y[0] - th[1] * y[0] * y[1],
                           th[2] * y[0] * y[1] - th[3] * y[1] };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// High-accuracy central-FD-of-trajectory sensitivity reference.
template <typename RHS>
static std::vector<arma::mat> fd_traj_sens(RHS f, const arma::vec& y0,
                                           const arma::vec& ts,
                                           const arma::vec& theta,
                                           const arma::uvec& idx) {
    const std::size_t n = y0.n_elem, p = idx.n_elem, nt = ts.n_elem;
    std::vector<arma::mat> S(nt, arma::mat(n, p, arma::fill::zeros));
    for (std::size_t c = 0; c < p; ++c) {
        const arma::uword tk = idx[c];
        const double h = 1e-6 * std::max(1.0, std::abs(theta[tk]));
        arma::vec tp = theta, tm = theta;
        tp[tk] += h; tm[tk] -= h;
        arma::mat yp = ode::rk45(f, y0, ts, tp, 1e-11, 1e-11);
        arma::mat ym = ode::rk45(f, y0, ts, tm, 1e-11, 1e-11);
        arma::mat d  = (yp - ym) / (2.0 * h);          // nt x n
        for (std::size_t i = 0; i < nt; ++i)
            S[i].col(c) = d.row(i).t();
    }
    return S;
}

// max relative error over two sensitivity stacks (abs floor for tiny refs).
static void sens_err(const std::vector<arma::mat>& A,
                     const std::vector<arma::mat>& B,
                     double& max_rel, double& max_abs_small) {
    max_rel = 0.0; max_abs_small = 0.0;
    for (std::size_t i = 0; i < A.size(); ++i)
        for (arma::uword r = 0; r < A[i].n_rows; ++r)
            for (arma::uword c = 0; c < A[i].n_cols; ++c) {
                const double a = A[i](r, c), b = B[i](r, c);
                const double e = std::abs(a - b);
                if (std::abs(b) > 1e-6) max_rel = std::max(max_rel, e / std::abs(b));
                else                    max_abs_small = std::max(max_abs_small, e);
            }
}

int main() {
    bool all_pass = true;
    std::printf("== Tier-2 forward sensitivity correctness ==\n");
#ifdef AI4BAYESCODE_ODE_HAVE_AUTODIFF
    std::printf("   autodiff path: ENABLED\n");
#else
    std::printf("   autodiff path: DISABLED (only FD-of-RHS tested)\n");
#endif

    // ---- (A) Linear ODE : analytic sensitivity ---------------------------
    {
        const double k = 0.7, y0v = 2.0;
        arma::vec y0{ y0v }, theta{ k };
        arma::vec ts(21);
        for (arma::uword i = 0; i < 21; ++i) ts[i] = i * 0.15;

        auto Sana = [&](std::size_t i) {   // d/dk (y0 e^{-k t}) = -t y0 e^{-k t}
            return -ts[i] * y0v * std::exp(-k * ts[i]);
        };

        double relA_fd = 0.0;
        auto rfd = ode::rk45_sens_fd(lin_rhs, y0, ts, theta, arma::uvec(),
                                     1e-9, 1e-9);
        for (arma::uword i = 0; i < ts.n_elem; ++i) {
            const double ref = Sana(i);
            const double e = std::abs(rfd.S[i](0, 0) - ref);
            relA_fd = std::max(relA_fd, std::abs(ref) > 1e-9 ? e/std::abs(ref) : e);
        }
        bool passA = relA_fd < 1e-6;
#ifdef AI4BAYESCODE_ODE_HAVE_AUTODIFF
        double relA_ad = 0.0;
        auto rad = ode::rk45_sens(lin_rhs, lin_rhs_ad<autodiff::dual>, y0, ts,
                                  theta, arma::uvec(), 1e-9, 1e-9);
        for (arma::uword i = 0; i < ts.n_elem; ++i) {
            const double ref = Sana(i);
            const double e = std::abs(rad.S[i](0, 0) - ref);
            relA_ad = std::max(relA_ad, std::abs(ref) > 1e-9 ? e/std::abs(ref) : e);
        }
        passA = passA && (relA_ad < 1e-6);
        std::printf("(A) linear analytic S: max rel err  FD=%.2e  AD=%.2e  -> %s\n",
                    relA_fd, relA_ad, passA ? "PASS" : "FAIL");
#else
        std::printf("(A) linear analytic S: max rel err  FD=%.2e  -> %s\n",
                    relA_fd, passA ? "PASS" : "FAIL");
#endif
        all_pass = all_pass && passA;
    }

    // ---- (B) SIR : vs FD-of-trajectory reference -------------------------
    {
        arma::vec y0{ 990.0, 10.0, 0.0 }, theta{ 0.6, 0.2 };
        arma::vec ts(15);
        for (arma::uword i = 0; i < 15; ++i) ts[i] = static_cast<double>(i);
        arma::uvec idx;   // all

        auto Sref = fd_traj_sens(sir_rhs, y0, ts, theta, arma::uvec{0,1});
        arma::mat y_t1 = ode::rk45(sir_rhs, y0, ts, theta, 1e-8, 1e-8);

        auto rfd = ode::rk45_sens_fd(sir_rhs, y0, ts, theta, idx, 1e-8, 1e-8);
        double rel_fd, abs_fd; sens_err(rfd.S, Sref, rel_fd, abs_fd);
        double y_rel = arma::abs(rfd.y - y_t1).max() / (arma::abs(y_t1).max());
        bool passB = rel_fd < 1e-4 && y_rel < 1e-5;
#ifdef AI4BAYESCODE_ODE_HAVE_AUTODIFF
        auto rad = ode::rk45_sens(sir_rhs, sir_rhs_ad<autodiff::dual>, y0, ts,
                                  theta, idx, 1e-8, 1e-8);
        double rel_ad, abs_ad; sens_err(rad.S, Sref, rel_ad, abs_ad);
        double rel_ad_fd, abs_ad_fd; sens_err(rad.S, rfd.S, rel_ad_fd, abs_ad_fd);
        passB = passB && rel_ad < 1e-4 && rel_ad_fd < 1e-4;
        std::printf("(B) SIR vs FD-traj ref: FD rel=%.2e  AD rel=%.2e  "
                    "AD-vs-FD rel=%.2e  y(AD vs Tier1) rel=%.2e -> %s\n",
                    rel_fd, rel_ad, rel_ad_fd, y_rel, passB ? "PASS" : "FAIL");
#else
        std::printf("(B) SIR vs FD-traj ref: FD rel=%.2e  y rel=%.2e -> %s\n",
                    rel_fd, y_rel, passB ? "PASS" : "FAIL");
#endif
        all_pass = all_pass && passB;
    }

    // ---- (C) Lotka-Volterra : vs FD-of-trajectory reference --------------
    {
        arma::vec y0{ 1.0, 1.0 }, theta{ 1.0, 0.8, 1.0, 0.9 };
        arma::vec ts(17);
        for (arma::uword i = 0; i < 17; ++i) ts[i] = i * 0.5;   // [0, 8]
        arma::uvec idx;   // all 4

        auto Sref = fd_traj_sens(lv_rhs, y0, ts, theta, arma::uvec{0,1,2,3});
        auto rfd  = ode::rk45_sens_fd(lv_rhs, y0, ts, theta, idx, 1e-8, 1e-8);
        double rel_fd, abs_fd; sens_err(rfd.S, Sref, rel_fd, abs_fd);
        bool passC = rel_fd < 1e-4;
#ifdef AI4BAYESCODE_ODE_HAVE_AUTODIFF
        auto rad = ode::rk45_sens(lv_rhs, lv_rhs_ad<autodiff::dual>, y0, ts,
                                  theta, idx, 1e-8, 1e-8);
        double rel_ad, abs_ad; sens_err(rad.S, Sref, rel_ad, abs_ad);
        passC = passC && rel_ad < 1e-4;
        std::printf("(C) Lotka-Volterra vs FD-traj ref: FD rel=%.2e  AD rel=%.2e -> %s\n",
                    rel_fd, rel_ad, passC ? "PASS" : "FAIL");

        // ---- (C2) theta_idx subset [1,3] == columns 1,3 of full ----------
        auto rsub = ode::rk45_sens(lv_rhs, lv_rhs_ad<autodiff::dual>, y0, ts,
                                   theta, arma::uvec{1,3}, 1e-8, 1e-8);
        double sub_err = 0.0;
        for (std::size_t i = 0; i < ts.n_elem; ++i) {
            sub_err = std::max(sub_err, arma::abs(rsub.S[i].col(0) - rad.S[i].col(1)).max());
            sub_err = std::max(sub_err, arma::abs(rsub.S[i].col(1) - rad.S[i].col(3)).max());
        }
        // The subset and full runs take slightly different adaptive steps
        // (their coupled error norms include 2 vs 4 sensitivity columns), so
        // the shared columns agree only to integration tolerance (~rtol),
        // not bit-for-bit. A real indexing bug would give O(1) differences.
        bool passC2 = sub_err < 1e-6 && rsub.S[0].n_cols == 2;
        std::printf("(C2) theta_idx subset: max col diff=%.2e -> %s\n",
                    sub_err, passC2 ? "PASS" : "FAIL");
        all_pass = all_pass && passC2;
#else
        std::printf("(C) Lotka-Volterra vs FD-traj ref: FD rel=%.2e -> %s\n",
                    rel_fd, passC ? "PASS" : "FAIL");
#endif
        all_pass = all_pass && passC;
    }

    // ---- (D) initial-condition sensitivity via S0 seeding ----------------
    {
        const double k = 0.5, y0v = 3.0;
        arma::vec y0{ y0v };
        arma::vec theta{ k, 0.0 };          // theta[1] = dummy IC param
        arma::vec ts(11);
        for (arma::uword i = 0; i < 11; ++i) ts[i] = i * 0.2;
        arma::uvec idx{ 1 };                 // track the dummy IC param
        arma::mat S0(1, 1); S0(0, 0) = 1.0;  // d y0 / d(param) = 1

        auto rfd = ode::rk45_sens_fd(lin_rhs, y0, ts, theta, idx,
                                     1e-9, 1e-9, S0);
        double relD = 0.0;
        for (arma::uword i = 0; i < ts.n_elem; ++i) {
            const double ref = std::exp(-k * ts[i]);   // d y/d y0 = e^{-k t}
            relD = std::max(relD, std::abs(rfd.S[i](0, 0) - ref) / ref);
        }
        bool passD = relD < 1e-6;
        std::printf("(D) IC sensitivity (S0 seed): max rel err=%.2e -> %s\n",
                    relD, passD ? "PASS" : "FAIL");
        all_pass = all_pass && passD;
    }

    // ---- (E) sens_chain vs direct FD of a scalar functional --------------
    {
        // Functional g(theta) = sum_k w_k * I(t_k) for SIR (I = state 1).
        arma::vec y0{ 990.0, 10.0, 0.0 }, theta{ 0.6, 0.2 };
        arma::vec ts(15);
        for (arma::uword i = 0; i < 15; ++i) ts[i] = static_cast<double>(i);
        arma::vec w(15);
        for (arma::uword i = 0; i < 15; ++i) w[i] = 0.1 + 0.03 * i;

        auto rfd = ode::rk45_sens_fd(sir_rhs, y0, ts, theta, arma::uvec(),
                                     1e-9, 1e-9);
        // dlp_dy: n_times x 3, only the I column (index 1) nonzero = w_k.
        arma::mat dlp_dy(15, 3, arma::fill::zeros);
        dlp_dy.col(1) = w;
        arma::vec grad = ode::sens_chain(rfd, dlp_dy);

        // Direct FD of g wrt theta.
        auto g_of = [&](const arma::vec& th) {
            arma::mat y = ode::rk45(sir_rhs, y0, ts, th, 1e-11, 1e-11);
            return arma::dot(w, y.col(1));
        };
        arma::vec grad_ref(2);
        for (arma::uword j = 0; j < 2; ++j) {
            const double h = 1e-6 * std::max(1.0, std::abs(theta[j]));
            arma::vec tp = theta, tm = theta; tp[j] += h; tm[j] -= h;
            grad_ref[j] = (g_of(tp) - g_of(tm)) / (2.0 * h);
        }
        double relE = arma::abs(grad - grad_ref).max() /
                      std::max(1e-8, arma::abs(grad_ref).max());
        bool passE = relE < 1e-4;
        std::printf("(E) sens_chain vs FD functional: grad=[%.5f,%.5f] "
                    "ref=[%.5f,%.5f] rel=%.2e -> %s\n",
                    grad[0], grad[1], grad_ref[0], grad_ref[1], relE,
                    passE ? "PASS" : "FAIL");
        all_pass = all_pass && passE;
    }

    std::printf("== %s ==\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}

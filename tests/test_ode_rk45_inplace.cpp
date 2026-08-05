// Generic (model-agnostic) validation of the allocation-free in-place ODE
// solvers rk45_inplace / rk45_sens_fd_inplace in ode_rk45.hpp. Two generic ODE
// systems, NO statistical/benchmark model:
//   M1 Lotka-Volterra (nonlinear): dx=a x-b x y, dy=c x y-d y   (n=2,p=4)
//   M2 3-compartment linear chain: dy0=-k1 y0, dy1=k1 y0-k2 y1, dy2=k2 y1-k3 y2 (n=3,p=3)
// Checks: (a) base parity vs ode::rk45; (b) coupled S + sens_chain gradient vs
// shipped ode::rk45_sens_fd; (c) coupled/state-only vs FD-of-solve reference;
// (d) tracked-subset theta_idx parity; plus a per-solve timing report.
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/ode_rk45.hpp"
#include <cstdio>
#include <chrono>
using namespace AI4BayesCode;

typedef void (*RhsIP)(double, const double*, const double*, double*);
typedef arma::vec (*RhsAr)(double, const arma::vec&, const arma::vec&);

// ---- M1 Lotka-Volterra ----
static inline void lv_ip(double, const double* y, const double* th, double* dy) {
    dy[0] = th[0] * y[0] - th[1] * y[0] * y[1];
    dy[1] = th[2] * y[0] * y[1] - th[3] * y[1];
}
static inline arma::vec lv_ar(double, const arma::vec& y, const arma::vec& th) {
    arma::vec dy(2);
    dy[0] = th[0] * y[0] - th[1] * y[0] * y[1];
    dy[1] = th[2] * y[0] * y[1] - th[3] * y[1];
    return dy;
}
// ---- M2 linear 3-compartment chain ----
static inline void lin_ip(double, const double* y, const double* th, double* dy) {
    dy[0] = -th[0] * y[0];
    dy[1] =  th[0] * y[0] - th[1] * y[1];
    dy[2] =  th[1] * y[1] - th[2] * y[2];
}
static inline arma::vec lin_ar(double, const arma::vec& y, const arma::vec& th) {
    arma::vec dy(3);
    dy[0] = -th[0] * y[0];
    dy[1] =  th[0] * y[0] - th[1] * y[1];
    dy[2] =  th[1] * y[1] - th[2] * y[2];
    return dy;
}

static double max_rel_S(const std::vector<arma::mat>& A, const std::vector<arma::mat>& B) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
        num = std::max(num, arma::abs(A[i] - B[i]).max());
        den = std::max(den, arma::abs(A[i]).max());
    }
    return num / std::max(1e-30, den);
}

// central-FD-of-the-solve reference sensitivities at solver tol `stol`.
static std::vector<arma::mat> ref_S(RhsAr fa, const arma::vec& y0, const arma::vec& ts,
                                    const arma::vec& theta, double stol) {
    const int nt = (int)ts.n_elem, n = (int)y0.n_elem, p = (int)theta.n_elem;
    std::vector<arma::mat> S(nt, arma::mat(n, p));
    for (int j = 0; j < p; ++j) {
        arma::vec tp = theta, tm = theta;
        double hj = 1e-6 * std::max(1.0, std::fabs(theta[j]));
        tp[j] += hj; tm[j] -= hj;
        arma::mat Yp = ode::rk45(fa, y0, ts, tp, stol, stol);
        arma::mat Ym = ode::rk45(fa, y0, ts, tm, stol, stol);
        arma::mat D  = (Yp - Ym) / (2.0 * hj);
        for (int i = 0; i < nt; ++i)
            for (int k = 0; k < n; ++k) S[i](k, j) = D(i, k);
    }
    return S;
}

static bool run_model(const char* name, RhsIP fi, RhsAr fa,
                      const arma::vec& y0, const arma::vec& ts, const arma::vec& theta) {
    bool ok = true;
    const int nt = (int)ts.n_elem, n = (int)y0.n_elem, last = nt - 1;
    std::printf("==== %s (n=%d, p=%d) ====\n", name, n, (int)theta.n_elem);

    // (a) base solve parity
    arma::mat Bship = ode::rk45(fa, y0, ts, theta, 1e-8, 1e-8);
    arma::mat Bfast = ode::rk45_inplace(fi, y0, ts, theta, 1e-8, 1e-8);
    double da = arma::abs(Bship - Bfast).max();
    std::printf("(a) base rk45_inplace vs ode::rk45        : max|d|=%.3e  %s\n",
                da, da < 1e-10 ? "OK" : "FAIL");
    ok = ok && da < 1e-10;

    // (b) coupled S vs shipped arma rk45_sens_fd (same tol, same control, same FD step)
    auto Rship = ode::rk45_sens_fd(fa, y0, ts, theta, arma::uvec(), 1e-6, 1e-6);
    auto Rcpl  = ode::rk45_sens_fd_inplace(fi, y0, ts, theta, arma::uvec(), 1e-6, 1e-6, arma::mat(), true);
    double rS = max_rel_S(Rship.S, Rcpl.S);
    std::printf("(b) S inplace(coupled) vs shipped sens_fd : S rel=%.3e  %s\n",
                rS, rS < 1e-6 ? "OK" : "FAIL");
    ok = ok && rS < 1e-6;
    arma::mat dlp(nt, n);
    for (int i = 0; i < nt; ++i) for (int k = 0; k < n; ++k) dlp(i, k) = 0.1 * (i + 1) - 0.05 * (k + 1);
    arma::vec gs = ode::sens_chain(Rship, dlp), gc = ode::sens_chain(Rcpl, dlp);
    double dg = arma::abs(gs - gc).max() / std::max(1e-30, arma::abs(gs).max());
    std::printf("    sens_chain gradient rel diff          : %.3e  %s\n", dg, dg < 1e-6 ? "OK" : "FAIL");
    ok = ok && dg < 1e-6;

    // (c) both variants vs FD-of-solve reference
    std::vector<arma::mat> Sref10 = ref_S(fa, y0, ts, theta, 1e-10);  // near-truth
    std::vector<arma::mat> Sref06 = ref_S(fa, y0, ts, theta, 1e-6);   // matched production tol
    auto Rc8 = ode::rk45_sens_fd_inplace(fi, y0, ts, theta, arma::uvec(), 1e-8, 1e-8, arma::mat(), true);
    auto Rs8 = ode::rk45_sens_fd_inplace(fi, y0, ts, theta, arma::uvec(), 1e-8, 1e-8, arma::mat(), false);
    double rc8 = max_rel_S(Sref10, Rc8.S), rs8 = max_rel_S(Sref10, Rs8.S);
    std::printf("(c) vs FD-of-solve(near-truth) @tol1e-8   : coupled rel=%.3e  state-only rel=%.3e  %s\n",
                rc8, rs8, (rc8 < 1e-6 && rs8 < 1e-6) ? "OK" : "FAIL");
    ok = ok && rc8 < 1e-6 && rs8 < 1e-6;

    auto Rc6 = ode::rk45_sens_fd_inplace(fi, y0, ts, theta, arma::uvec(), 1e-6, 1e-6, arma::mat(), true);
    auto Rs6 = ode::rk45_sens_fd_inplace(fi, y0, ts, theta, arma::uvec(), 1e-6, 1e-6, arma::mat(), false);
    std::printf("    @tol1e-6 vs NEAR-TRUTH(1e-10) ref     : coupled rel=%.3e  state-only rel=%.3e\n",
                max_rel_S(Sref10, Rc6.S), max_rel_S(Sref10, Rs6.S));
    std::printf("    @tol1e-6 vs MATCHED-TOL(1e-6) ref     : coupled rel=%.3e  state-only rel=%.3e\n",
                max_rel_S(Sref06, Rc6.S), max_rel_S(Sref06, Rs6.S));

    // (4) timing
    const long N = 60000; double cs = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (long i = 0; i < N; ++i) { arma::vec th = theta; th[0] += 1e-9 * (i % 7);
        auto R = ode::rk45_sens_fd(fa, y0, ts, th, arma::uvec(), 1e-6, 1e-6); cs += R.S[last](0, 0); }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (long i = 0; i < N; ++i) { arma::vec th = theta; th[0] += 1e-9 * (i % 7);
        auto R = ode::rk45_sens_fd_inplace(fi, y0, ts, th, arma::uvec(), 1e-6, 1e-6, arma::mat(), true); cs += R.S[last](0, 0); }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (long i = 0; i < N; ++i) { arma::vec th = theta; th[0] += 1e-9 * (i % 7);
        auto R = ode::rk45_sens_fd_inplace(fi, y0, ts, th, arma::uvec(), 1e-6, 1e-6, arma::mat(), false); cs += R.S[last](0, 0); }
    auto t3 = std::chrono::high_resolution_clock::now();
    double us_s = 1e6 * std::chrono::duration<double>(t1 - t0).count() / N;
    double us_c = 1e6 * std::chrono::duration<double>(t2 - t1).count() / N;
    double us_o = 1e6 * std::chrono::duration<double>(t3 - t2).count() / N;
    std::printf("(4) us/solve: shipped %.2f | inplace-coupled %.2f (%.2fx) | inplace-state %.2f (%.2fx)  cs=%.3e\n",
                us_s, us_c, us_s / us_c, us_o, us_s / us_o, cs);
    std::printf("%s\n\n", ok ? "-> PASS" : "-> FAIL");
    return ok;
}

int main() {
    bool ok = true;
    ok &= run_model("M1 Lotka-Volterra", lv_ip, lv_ar,
                    arma::vec{10.0, 5.0}, arma::vec{0.0, 1.0, 2.0, 4.0, 7.0, 10.0},
                    arma::vec{0.8, 0.06, 0.02, 0.5});
    ok &= run_model("M2 linear 3-compartment", lin_ip, lin_ar,
                    arma::vec{100.0, 0.0, 0.0}, arma::vec{0.0, 0.5, 1.0, 2.0, 3.5, 5.0},
                    arma::vec{0.7, 0.4, 0.2});

    // (d) tracked-subset theta_idx parity (M1, track only params {0,3})
    {
        arma::vec y0{10.0, 5.0}, ts{0.0, 1.0, 2.0, 4.0, 7.0, 10.0}, theta{0.8, 0.06, 0.02, 0.5};
        arma::uvec idx{0, 3};
        auto Rship = ode::rk45_sens_fd(lv_ar, y0, ts, theta, idx, 1e-6, 1e-6);
        auto Rin   = ode::rk45_sens_fd_inplace(lv_ip, y0, ts, theta, idx, 1e-6, 1e-6, arma::mat(), true);
        bool idx_ok = arma::all(Rship.theta_idx == Rin.theta_idx)
                   && Rin.S[0].n_cols == 2;
        double rS = max_rel_S(Rship.S, Rin.S);
        std::printf("==== (d) subset theta_idx={0,3} ====\n");
        std::printf("(d) subset S inplace vs shipped          : S rel=%.3e  cols=%llu idx=%s  %s\n",
                    rS, (unsigned long long)Rin.S[0].n_cols, idx_ok ? "match" : "MISMATCH",
                    (rS < 1e-6 && idx_ok) ? "OK" : "FAIL");
        ok &= (rS < 1e-6 && idx_ok);
    }

    std::printf("\n%s\n", ok ? "ALL PASS" : "FAILED");
    return ok ? 0 : 1;
}

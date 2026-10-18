// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_stick_label_swap_exact.cpp
//
//  The label-switching moves of stick_label_swap_block must leave the joint
//  posterior of (z, theta, V) invariant. Tiny truncated mixture: K = 4 slots,
//  n = 6 observations, y_i | z, mu ~ N(mu_{z_i}, 1), mu_k ~ N(0, 4). The
//  exact posterior over all 4^6 labelled allocations follows from the
//  closed-form stick-breaking prior of z and the conjugate marginal
//  likelihood. A blocked Gibbs chain (z | pi, mu;  mu | z;  V | z) with the
//  block applied five times per sweep must reproduce every marginal
//  P(z_i = k). Slot positions matter under stick-breaking, so a wrong
//  acceptance ratio shifts these marginals. A chain with no swap moves
//  calibrates the Monte Carlo error.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/stick_label_swap_block.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include "portable_rng.hpp"

using namespace AI4BayesCode;
static constexpr int K = 4, N = 6;
static const double Y[N] = {-2.1, -1.8, -2.4, 1.9, 2.2, 0.1};
static constexpr double S2 = 1.0, T2 = 4.0;
static int g_pass = 0, g_fail = 0;

static double lbeta(double a, double b) { return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b); }

// Exact marginals P(z_i = k) under the truncated stick-breaking prior.
static std::vector<double> exact_marginals(double alpha, double d) {
    std::vector<double> logw(1 << (2 * N)), marg(N * K, 0.0);
    double mx = -1e300;
    for (int code = 0; code < (1 << (2 * N)); ++code) {
        int z[N]; double nk[K] = {0}, S[K] = {0}, Q[K] = {0};
        for (int i = 0; i < N; ++i) { z[i] = (code >> (2 * i)) & 3; nk[z[i]] += 1; S[z[i]] += Y[i]; Q[z[i]] += Y[i] * Y[i]; }
        double lp = 0.0, tail = 0.0;
        for (int k = K - 1; k >= 0; --k) {
            if (k < K - 1) {
                const double a = 1.0 - d, b = alpha + (k + 1) * d;
                lp += lbeta(a + nk[k], b + tail) - lbeta(a, b);
            }
            tail += nk[k];
        }
        for (int k = 0; k < K; ++k) if (nk[k] > 0) {
            const double m = nk[k];
            lp += -0.5 * m * std::log(2 * M_PI * S2) - 0.5 * std::log(1 + m * T2 / S2)
                  - 0.5 / S2 * (Q[k] - T2 * S[k] * S[k] / (S2 + m * T2));
        }
        logw[code] = lp; if (lp > mx) mx = lp;
    }
    double tot = 0.0;
    for (int code = 0; code < (1 << (2 * N)); ++code) {
        const double w = std::exp(logw[code] - mx); tot += w;
        for (int i = 0; i < N; ++i) marg[i * K + ((code >> (2 * i)) & 3)] += w;
    }
    for (auto& m : marg) m /= tot;
    return marg;
}

struct Run { const char* label; double alpha, d; bool ns, hlr; int reps; };

static std::vector<double> chain(const Run& r, long sweeps, double* acc_ns, double* acc_hlr) {
    std::mt19937_64 g(20261018u), rng_block(77u);
    stick_label_swap_block_config cfg;
    cfg.K_trunc = K; cfg.atom_keys = {{"mu", 1}}; cfg.discount = r.d;
    cfg.neighbour_swap = r.ns; cfg.weight_preserving_swap = r.hlr;
    stick_label_swap_block blk(cfg);
    arma::vec z(N), mu(K, arma::fill::zeros), V(K), pi(K);
    for (int i = 0; i < N; ++i) z[i] = (i % K) + 1;
    for (int k = 0; k < K - 1; ++k) V[k] = 0.5; V[K - 1] = 1.0;
    std::vector<double> marg(N * K, 0.0);
    double ans = 0, ahl = 0;
    const long burn = 5000;
    for (long s = 0; s < burn + sweeps; ++s) {
        for (int rep = 0; rep < r.reps; ++rep) {
            double rest = 1.0;
            for (int k = 0; k < K - 1; ++k) { pi[k] = V[k] * rest; rest *= 1 - V[k]; }
            pi[K - 1] = rest;
            blk.set_context({{"z", z}, {"mu", mu}, {"pi", pi}, {"stick_V", V}, {"alpha", arma::vec{r.alpha}}});
            blk.step(rng_block);
            auto out = blk.current_named_outputs();
            z = out.at("z"); mu = out.at("mu"); V = out.at("stick_V");
            if (s >= burn) { ans += blk.current()[0]; ahl += blk.current()[1]; }
        }
        // z | pi, mu
        double rest = 1.0;
        for (int k = 0; k < K - 1; ++k) { pi[k] = V[k] * rest; rest *= 1 - V[k]; }
        pi[K - 1] = rest;
        for (int i = 0; i < N; ++i) {
            double w[K], tot = 0;
            for (int k = 0; k < K; ++k) { w[k] = pi[k] * std::exp(-0.5 * (Y[i] - mu[k]) * (Y[i] - mu[k]) / S2); tot += w[k]; }
            double u = prng::u01(g) * tot; int k = 0;
            while (k < K - 1 && u > w[k]) { u -= w[k]; ++k; }
            z[i] = k + 1;
        }
        // mu | z
        double nk[K] = {0}, Sk[K] = {0};
        for (int i = 0; i < N; ++i) { const int k = (int)z[i] - 1; nk[k] += 1; Sk[k] += Y[i]; }
        for (int k = 0; k < K; ++k) {
            const double prec = 1 / T2 + nk[k] / S2;
            mu[k] = prng::normal(g, (Sk[k] / S2) / prec, std::sqrt(1 / prec));
        }
        // V | z
        double tail = 0.0, tails[K];
        for (int k = K - 1; k >= 0; --k) { tails[k] = tail; tail += nk[k]; }
        for (int k = 0; k < K - 1; ++k) {
            const double a = 1.0 - r.d + nk[k], b = r.alpha + (k + 1) * r.d + tails[k];
            const double x = prng::gamma(g, a, 1.0), y = prng::gamma(g, b, 1.0);
            V[k] = x / (x + y);
        }
        V[K - 1] = 1.0;
        if (s >= burn) for (int i = 0; i < N; ++i) marg[i * K + (int)z[i] - 1] += 1.0 / sweeps;
    }
    *acc_ns  = ans / (double(sweeps) * r.reps * (K - 1));   // K - 2 neighbour pairs + the last pair
    *acc_hlr = ahl / (double(sweeps) * r.reps * (K - 2));
    return marg;
}

int main() {
    const long S = 400000;
    const Run runs[] = {
        {"DP, no swap moves (Monte Carlo error)", 0.5, 0.0, false, false, 1},
        {"DP, neighbour swap",                    0.5, 0.0, true,  false, 5},
        {"DP, weight-preserving swap",            0.5, 0.0, false, true,  5},
        {"DP, both moves",                        0.5, 0.0, true,  true,  5},
        {"PY d = 0.3, neighbour swap",            0.8, 0.3, true,  true,  5},
    };
    for (const auto& r : runs) {
        const auto ex = exact_marginals(r.alpha, r.d);
        double an = 0, ah = 0;
        const auto mc = chain(r, S, &an, &ah);
        double mx = 0;
        for (int j = 0; j < N * K; ++j) mx = std::max(mx, std::fabs(mc[j] - ex[j]));
        const bool ok = mx < 0.012;
        (ok ? g_pass : g_fail)++;
        std::printf("  [%s] %-40s max |P_mcmc(z_i=k) - P_exact| = %.4f  (accept: swap %.3f, weight-preserving %.3f)\n",
                    ok ? "PASS" : "FAIL", r.label, mx, an, ah);
    }
    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_softbart_store_default.cpp -- the default bound on the per-tree
//  fits that the SoftBART forest stores is 4194304 = 2^22 doubles.
//
//  This file is compiled without a definition of
//  AI4BAYESCODE_SOFTBART_FIT_STORE_MAX_DOUBLES (an #error stops the build if
//  one is given), so the bound is the default of soft_bart.h.
//  A. The macro equals 4194304, and Forest::fit_store_allowed(ntrees, N)
//     holds for 200 trees at N = 20971 (4194200 doubles) and for 1 tree at
//     N = 4194304, and does not hold for 200 trees at N = 20972 (4194400
//     doubles) or for 1 tree at N = 4194305; the same with ntrees and N
//     exchanged for the single-row cases. Any other default fails the
//     1-tree and single-row checks.
//  B. After one sweep of a softbart_model with 200 trees, the forest holds
//     no stored fits at N = 20972 and 200 at N = 20971 (a default outside
//     4194200..4194399 fails one of the two). The peak resident set size
//     (getrusage) grows during the sweep at N = 20972 by less than half of
//     the 200 x N doubles a store would take, so no store is filled during
//     that sweep, and during the sweep at N = 20971, which keeps the store,
//     by at least half of it, so the measurement registers a store of that
//     size. The peak is not measured on Windows or in AddressSanitizer
//     builds, where freed blocks are placed in a quarantine and not reused
//     while there, so the peak includes memory freed during the sweep.
// ============================================================================
#ifdef AI4BAYESCODE_SOFTBART_FIT_STORE_MAX_DOUBLES
#error "test_fix_softbart_store_default checks the default bound: build it without defining AI4BAYESCODE_SOFTBART_FIT_STORE_MAX_DOUBLES"
#endif
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/softbart_block.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

// AddressSanitizer: GCC and MSVC define __SANITIZE_ADDRESS__; clang reports it
// through __has_feature(address_sanitizer).
#if defined(__SANITIZE_ADDRESS__)
# define TEST_ASAN_BUILD 1
#elif defined(__has_feature)
# if __has_feature(address_sanitizer)
#  define TEST_ASAN_BUILD 1
# endif
#endif
#ifndef TEST_ASAN_BUILD
# define TEST_ASAN_BUILD 0
#endif

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

// Peak resident set size of the process in bytes, or -1 where getrusage is
// not available and in AddressSanitizer builds.
static long long peak_rss_bytes() {
#if defined(_WIN32) || TEST_ASAN_BUILD
    return -1;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1;
#if defined(__APPLE__)
    return static_cast<long long>(ru.ru_maxrss);          // bytes
#else
    return static_cast<long long>(ru.ru_maxrss) * 1024;   // kilobytes
#endif
#endif
}

static void make_data(int N, int p, std::uint64_t seed, arma::mat& X, arma::vec& y) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> Z(0.0, 1.0);
    X.set_size(N, p); y.set_size(N);
    for (int j = 0; j < p; ++j) for (int i = 0; i < N; ++i) X(i, j) = U(g);
    for (int i = 0; i < N; ++i) y[i] = std::sin(6.0 * X(i, 0)) + X(i, 1) + 0.3 * Z(g);
    y -= arma::mean(y);
}

int main() {
    std::printf("test_fix_softbart_store_default (bound %llu doubles)\n",
                static_cast<unsigned long long>(AI4BAYESCODE_SOFTBART_FIT_STORE_MAX_DOUBLES));

    std::printf("A. fit_store_allowed at the default bound\n");
    check(static_cast<unsigned long long>(AI4BAYESCODE_SOFTBART_FIT_STORE_MAX_DOUBLES) == 4194304ull,
          "AI4BAYESCODE_SOFTBART_FIT_STORE_MAX_DOUBLES is 4194304 (2^22)");
    check(Forest::fit_store_allowed(200, 20971) && !Forest::fit_store_allowed(200, 20972),
          "200 trees: N = 20971 within the bound, N = 20972 above it");
    check(Forest::fit_store_allowed(1, 4194304) && !Forest::fit_store_allowed(1, 4194305),
          "1 tree: N = 4194304 within the bound, N = 4194305 above it");
    check(Forest::fit_store_allowed(4194304, 1) && !Forest::fit_store_allowed(4194305, 1),
          "N = 1: 4194304 trees within the bound, 4194305 above it");

    std::printf("B. stored fits after one sweep with 200 trees\n");
    for (int N : {20972, 20971}) {
        const bool within = N == 20971;
        arma::mat X; arma::vec y; make_data(N, 3, 11, X, y);
        bart_rng::set_seed(3);
        softbart::softbart_model m(X, y, 200);
        const long long rss0 = peak_rss_bytes();
        m.update_step();
        const long long rss1 = peak_rss_bytes();
        const std::size_t stored = m.forest()->tree_fit_store_size();
        char what[128];
        std::snprintf(what, sizeof what, "N = %d: %zu stored fits after a sweep (expected %d)",
                      N, stored, within ? 200 : 0);
        check(stored == (within ? 200u : 0u) && m.train_fit() != nullptr, what);
        if (rss0 < 0 || rss1 < 0) {
            std::printf("  peak resident set size not measured; not checked\n");
            continue;
        }
        const double grow_mib = static_cast<double>(rss1 - rss0) / 1048576.0;
        const double half_store_mib = 0.5 * 200.0 * N * sizeof(double) / 1048576.0;
        std::snprintf(what, sizeof what, "N = %d: peak RSS grows by %.1f MiB in the sweep (%s %.1f MiB)",
                      N, grow_mib, within ? ">=" : "<", half_store_mib);
        check(within ? grow_mib >= half_store_mib : grow_mib < half_store_mib, what);
    }

    if (g_fail) { std::printf("FAILED: %d check(s)\n", g_fail); return 1; }
    std::printf("ALL PASS\n");
    return 0;
}

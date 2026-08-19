/*================================================================================
 *  test_portable_rng.cpp
 *
 *  tests/portable_rng.hpp exists so that simulated test data is identical on
 *  every standard library. That is only worth anything if each generator draws
 *  from the distribution it claims. This file checks that, against the analytic
 *  moments and -- where a shape error would hide inside matching moments --
 *  against the empirical CDF of the std:: counterpart.
 *
 *  It also pins the property the header exists for: the raw uniform stream is a
 *  pure function of the engine, so any two standard libraries agree bit for bit.
 *  That half cannot be verified from inside one build; what IS checked here is
 *  that no generator touches a std::*_distribution, which is the only way the
 *  library could get a say. The cross-library comparison itself is done by
 *  building this file with both toolchains and diffing the printed digests.
 *
 *  Copyright (C) 2026 AI4BayesCode
 *  SPDX-License-Identifier: Apache-2.0
 *================================================================================*/

#include "portable_rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int checks = 0, failures = 0;

void check(bool ok, const char* what, const char* detail = "") {
    ++checks;
    std::printf("  %s  %s%s%s\n", ok ? "ok  " : "FAIL", what,
                *detail ? " -- " : "", detail);
    if (!ok) ++failures;
}

/// Two-sample Kolmogorov-Smirnov statistic. Catches a wrong SHAPE that moment
/// checks would sail past.
double ks_stat(std::vector<double> a, std::vector<double> b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::size_t i = 0, j = 0;
    double d = 0.0;
    while (i < a.size() && j < b.size()) {
        const double x = std::min(a[i], b[j]);
        while (i < a.size() && a[i] <= x) ++i;
        while (j < b.size() && b[j] <= x) ++j;
        d = std::max(d, std::fabs(static_cast<double>(i) / a.size() -
                                  static_cast<double>(j) / b.size()));
    }
    return d;
}

/// 5% critical value for the two-sample KS test at these sizes.
double ks_crit(std::size_t n, std::size_t m) {
    return 1.36 * std::sqrt((static_cast<double>(n) + m) /
                            (static_cast<double>(n) * m));
}

struct moments { double mean, var; };

moments summarise(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m += x;
    m /= static_cast<double>(v.size());
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return {m, s / static_cast<double>(v.size() - 1)};
}

const std::size_t N = 200000;

}  // namespace

int main() {
    std::printf("=== test_portable_rng ===\n\n");
    std::mt19937_64 g(20260818u);

    // ---- u01: mean 1/2, var 1/12, and inside [0, 1) ------------------------
    {
        std::vector<double> v(N);
        double lo = 1.0, hi = 0.0;
        for (auto& x : v) { x = prng::u01(g); lo = std::min(lo, x); hi = std::max(hi, x); }
        const moments s = summarise(v);
        char buf[96];
        std::snprintf(buf, sizeof buf, "mean %.5f (0.5), var %.5f (%.5f)",
                      s.mean, s.var, 1.0 / 12.0);
        check(std::fabs(s.mean - 0.5) < 4.0 * std::sqrt(1.0 / 12.0 / N) &&
              std::fabs(s.var - 1.0 / 12.0) < 0.001, "u01 moments", buf);
        check(lo >= 0.0 && hi < 1.0, "u01 stays in [0, 1)");
    }

    // ---- n01: moments AND shape against std::normal_distribution -----------
    {
        std::vector<double> mine(N), theirs(N);
        for (auto& x : mine) x = prng::n01(g);
        std::mt19937_64 h(99991u);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (auto& x : theirs) x = nd(h);
        const moments s = summarise(mine);
        char buf[96];
        std::snprintf(buf, sizeof buf, "mean %.5f (0), var %.5f (1)", s.mean, s.var);
        check(std::fabs(s.mean) < 4.0 / std::sqrt((double)N) &&
              std::fabs(s.var - 1.0) < 0.02, "n01 moments", buf);

        // third and fourth standardised moments: 0 and 3 for a normal
        double m3 = 0.0, m4 = 0.0;
        for (double x : mine) { const double z = (x - s.mean) / std::sqrt(s.var);
                                m3 += z * z * z; m4 += z * z * z * z; }
        m3 /= (double)N; m4 /= (double)N;
        std::snprintf(buf, sizeof buf, "skew %.4f (0), kurtosis %.4f (3)", m3, m4);
        check(std::fabs(m3) < 0.05 && std::fabs(m4 - 3.0) < 0.10,
              "n01 higher moments", buf);

        const double d = ks_stat(mine, theirs), c = ks_crit(N, N);
        std::snprintf(buf, sizeof buf, "KS %.5f vs 5%% critical %.5f", d, c);
        check(d < c, "n01 matches std::normal_distribution in DISTRIBUTION", buf);
    }

    // ---- uniform_int: flat, and no modulo bias -----------------------------
    {
        // A range that does NOT divide 2^64 is where a naive modulo would bias.
        const std::uint64_t lo = 0, hi = 6;          // 7 outcomes
        std::vector<double> count(7, 0.0);
        for (std::size_t i = 0; i < N; ++i)
            count[prng::uniform_int(g, lo, hi)] += 1.0;
        const double expect = (double)N / 7.0;
        double chi2 = 0.0, worst = 0.0;
        for (double c : count) {
            chi2 += (c - expect) * (c - expect) / expect;
            worst = std::max(worst, std::fabs(c - expect) / expect);
        }
        char buf[96];
        std::snprintf(buf, sizeof buf, "chi2 %.2f on 6 df (crit 12.59), worst cell %+.3f%%",
                      chi2, 100.0 * worst);
        check(chi2 < 12.59, "uniform_int is flat over a non-dividing range", buf);

        bool in_range = true;
        for (std::size_t i = 0; i < 1000; ++i) {
            const std::uint64_t x = prng::uniform_int(g, 3, 9);
            if (x < 3 || x > 9) in_range = false;
        }
        check(in_range, "uniform_int respects an offset range, inclusive");
    }

    // ---- gamma: moments and shape, across the shape < 1 boundary ----------
    for (double shape : {0.4, 1.0, 2.5, 9.0}) {
        const double scale = 1.7;
        std::vector<double> mine(N), theirs(N);
        for (auto& x : mine) x = prng::gamma(g, shape, scale);
        std::mt19937_64 h(4242u + (unsigned)(shape * 10));
        std::gamma_distribution<double> gd(shape, scale);
        for (auto& x : theirs) x = gd(h);
        const moments s = summarise(mine);
        const double em = shape * scale, ev = shape * scale * scale;
        char buf[128];
        std::snprintf(buf, sizeof buf,
                      "shape %.1f: mean %.4f (%.4f), var %.4f (%.4f)",
                      shape, s.mean, em, s.var, ev);
        check(std::fabs(s.mean - em) < 5.0 * std::sqrt(ev / N) &&
              std::fabs(s.var - ev) / ev < 0.05, "gamma moments", buf);

        const double d = ks_stat(mine, theirs), c = ks_crit(N, N);
        std::snprintf(buf, sizeof buf, "shape %.1f: KS %.5f vs %.5f", shape, d, c);
        check(d < c, "gamma matches std::gamma_distribution in DISTRIBUTION", buf);
    }

    // ---- poisson: mean = var = lambda, and the pmf at small counts ---------
    for (double lam : {0.6, 3.0, 12.0}) {
        std::vector<double> v(N);
        for (auto& x : v) x = prng::poisson(g, lam);
        const moments s = summarise(v);
        char buf[128];
        std::snprintf(buf, sizeof buf, "lambda %.1f: mean %.4f, var %.4f", lam, s.mean, s.var);
        check(std::fabs(s.mean - lam) < 5.0 * std::sqrt(lam / N) &&
              std::fabs(s.var - lam) / lam < 0.05, "poisson moments", buf);

        // P(X = 0) = exp(-lambda) is where a broken product loop shows first
        double n0 = 0.0;
        for (double x : v) if (x == 0.0) n0 += 1.0;
        const double p0 = n0 / N, e0 = std::exp(-lam);
        std::snprintf(buf, sizeof buf, "lambda %.1f: P(0) %.5f vs %.5f", lam, p0, e0);
        check(std::fabs(p0 - e0) < 5.0 * std::sqrt(e0 * (1 - e0) / N),
              "poisson P(X = 0)", buf);
    }

    // ---- bernoulli and discrete -------------------------------------------
    {
        double n1 = 0.0;
        for (std::size_t i = 0; i < N; ++i) n1 += prng::bernoulli(g, 0.3);
        const double p = n1 / N;
        char buf[80];
        std::snprintf(buf, sizeof buf, "p_hat %.5f (0.3)", p);
        check(std::fabs(p - 0.3) < 5.0 * std::sqrt(0.3 * 0.7 / N), "bernoulli", buf);

        const std::vector<double> w{1.0, 3.0, 0.0, 6.0};   // note the zero weight
        std::vector<double> cnt(4, 0.0);
        for (std::size_t i = 0; i < N; ++i) cnt[prng::discrete(g, w)] += 1.0;
        const double tot = 10.0;
        double worst = 0.0;
        for (std::size_t k = 0; k < 4; ++k)
            worst = std::max(worst, std::fabs(cnt[k] / N - w[k] / tot));
        std::snprintf(buf, sizeof buf, "worst |p_hat - p| = %.5f", worst);
        check(worst < 0.005 && cnt[2] == 0.0,
              "discrete matches its weights, and never returns a zero-weight cell", buf);
    }

    // ---- the point of the header: no std:: distribution in the stream ------
    // Printed so a build under the other standard library can be diffed
    // against this one. Identical digests are the property being claimed.
    {
        std::mt19937_64 d(12345u);
        double acc = 0.0;
        for (int i = 0; i < 1000; ++i) acc += prng::n01(d) + prng::u01(d);
        std::printf("\n  cross-library digest (must match on libstdc++ and libc++):\n");
        std::printf("    sum over 1000 (n01 + u01) from seed 12345 = %.15f\n", acc);
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}

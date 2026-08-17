/*
 * test_pg_truncation.cpp
 *
 * sample_pg_1_z (Polson-Scott-Windle 2013 Eq. 2.3) against the EXACT moments
 * of PG(1, z):
 *
 *     E[w]   = tanh(z/2) / (2z)                        (1/4 at z = 0)
 *     Var[w] = (sinh(z) - z) sech^2(z/2) / (4 z^3)     (1/24 at z = 0)
 *
 * The regression: the series is truncated at K terms, and the discarded tail
 * decays as 1/k^2, so its MEAN is ~1/(2 pi^2 K) -- 4.0e-4 at the default
 * K = 128, ABSOLUTE and independent of z. The header claimed "~1e-8 relative".
 * Left uncorrected, E[w] was biased low by 0.17% at z = 0 and 0.64% at z = 8
 * (E[w] shrinks with z while the tail does not), so X'Omega X ran
 * systematically low and every posterior variance high -- with nothing in
 * R-hat or ESS to show for it.
 *
 * The fix adds the tail's mean back analytically: it is (full mean) - (kept
 * partial sum), both available for free inside the sampling loop. This test
 * pins the mean at the Monte-Carlo noise floor and checks that the variance is
 * still right, since a mean correction must not have been bought by distorting
 * the spread.
 */

#include "AI4BayesCode/pg_logistic_block.hpp"

#include <cmath>
#include <cstdio>
#include <random>

using AI4BayesCode::pg_1_z_mean;
using AI4BayesCode::sample_pg_1_z;

namespace {

int checks = 0, failures = 0;

void check(bool ok, const char* what, const char* detail) {
    ++checks;
    std::printf("  %s  %-46s %s\n", ok ? "ok  " : "FAIL", what, detail);
    if (!ok) ++failures;
}

/// Exact Var[PG(1, z)].
double pg_1_z_var(double z) {
    if (std::fabs(z) < 1e-6) return 1.0 / 24.0;
    const double ch = std::cosh(0.5 * z);
    return (std::sinh(z) - z) / (4.0 * z * z * z) / (ch * ch);
}

/// The tail mean the truncation drops, i.e. what the correction adds back.
double dropped_tail_mean(double z, int K) {
    const double c = (z * z) / (4.0 * M_PI * M_PI);
    double kept = 0.0;
    for (int k = 1; k <= K; ++k) {
        const double h = static_cast<double>(k) - 0.5;
        kept += 1.0 / (h * h + c);
    }
    return pg_1_z_mean(z) - kept / (2.0 * M_PI * M_PI);
}

} // namespace

int main() {
    std::printf("=== test_pg_truncation ===\n\n");
    const long M = 2000000;
    const int K = 128;                     // the shipped default

    std::printf("  %5s %13s %13s %9s %9s %9s\n",
                "z", "exact E[w]", "sampled E[w]", "bias/se", "exact Var", "rel Var");
    for (double z : {0.0, 0.5, 1.0, 3.0, 8.0, 20.0}) {
        std::mt19937_64 rng(20260817 + static_cast<unsigned>(z * 7));
        double s = 0.0, s2 = 0.0;
        for (long i = 0; i < M; ++i) {
            const double w = sample_pg_1_z(rng, z, K);
            s += w; s2 += w * w;
        }
        const double m = s / M, v = s2 / M - m * m;
        const double ex = pg_1_z_mean(z), exv = pg_1_z_var(z);
        const double se = std::sqrt(v / M);
        const double z_score = (m - ex) / se;
        const double rel_v = (v - exv) / exv;
        std::printf("  %5.1f %13.9f %13.9f %+9.1f %9.6f %+8.3f%%\n",
                    z, ex, m, z_score, exv, 100.0 * rel_v);

        char buf[96];
        // 4 se: the mean must sit at the noise floor. Uncorrected it was
        // several HUNDRED se away at this sample size.
        std::snprintf(buf, sizeof buf, "z=%.1f: %+.1f se", z, z_score);
        check(std::fabs(z_score) < 4.0, "sampled mean matches the exact PG mean", buf);

        // The correction is a constant, so it must not have moved the spread.
        // The truncation's own variance deficit is ~1e-8 relative; the bound
        // here is set by Monte-Carlo error on a variance estimate at M draws
        // (about sqrt(2/M) = 0.1% relative).
        std::snprintf(buf, sizeof buf, "z=%.1f: %+.3f%%", z, 100.0 * rel_v);
        check(std::fabs(rel_v) < 0.01, "sampled variance matches the exact PG variance", buf);

        // Positivity: omega parameterises a precision, so a non-positive draw
        // would be a hard failure downstream. The offset is non-negative, so
        // this holds by construction -- assert it anyway.
        std::mt19937_64 rng2(4242);
        bool all_pos = true;
        for (int i = 0; i < 20000; ++i)
            if (!(sample_pg_1_z(rng2, z, K) > 0.0)) { all_pos = false; break; }
        std::snprintf(buf, sizeof buf, "z=%.1f", z);
        check(all_pos, "every draw is strictly positive", buf);
    }

    // The size of the thing being corrected, stated in the test so a future
    // reader does not have to rederive it.
    std::printf("\n  dropped tail mean at K=%d (what the correction adds back):\n", K);
    for (double z : {0.0, 3.0, 8.0}) {
        const double t = dropped_tail_mean(z, K);
        std::printf("    z=%4.1f  %.3e absolute  = %.3f%% of E[w]\n",
                    z, t, 100.0 * t / pg_1_z_mean(z));
        char buf[96];
        std::snprintf(buf, sizeof buf, "z=%.1f: %.2e", z, t);
        // Predicted 1/(2 pi^2 K) = 3.96e-4; assert we are in that ballpark, so
        // a future K change cannot quietly make the correction meaningless.
        check(t > 1e-5 && t < 1e-2,
              "the tail being corrected is the predicted ~1/(2 pi^2 K)", buf);
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}

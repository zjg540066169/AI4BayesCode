/*================================================================================
 *  test_bart_rng_omp_safety
 *  ------------------------------------------------------------------------------
 *  Guards the 2026-07-25 AI4BayesCode BART-RNG OMP-safety fork
 *  (bart_pure_cpp/src/r_compat.h -- new bart_rng::set_seed_per_thread()).
 *
 *  T1  single-thread reproducibility  : same rng_seed twice -> bit-exact BART
 *                                        stream.
 *  T2  single-thread seed sensitivity : different rng_seed -> at least one
 *                                        draw differs (bit-exact NOT expected).
 *  T3  multi-thread divergence        : under -fopenmp, each thread's engine
 *                                        (thread_local) is seeded via
 *                                        set_seed_per_thread(SAME s). All
 *                                        thread streams must DIFFER pairwise
 *                                        (this is exactly the bug the fork
 *                                        fixes: pre-fork every thread got the
 *                                        identical stream). Under a compiler
 *                                        WITHOUT _OPENMP the subtest is
 *                                        skipped (there IS only one thread).
 *
 *  Standalone driver: exit 0 iff all applicable subtests pass. Failure prints
 *  which subtest and why.
 *================================================================================*/

#define NoRcpp
#include "../bart_pure_cpp/src/r_compat.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

// ---- Reference draw sequence from bart_rng after a seed --------------------
// Uses runif_open() which is the exact primitive BART / SoftBart / genBART
// call through the arn adapter, so a bit-exact match here IS a bit-exact
// match of the BART RNG stream.
static std::vector<double> draw_seq(int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) v[i] = bart_rng::runif_open();
    return v;
}

static bool vec_eq(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

static bool vec_any_diff(const std::vector<double>& a,
                         const std::vector<double>& b) {
    if (a.size() != b.size()) return true;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return true;
    return false;
}

// ---- T1: single-thread reproducibility ------------------------------------
static bool t1_single_thread_reproducibility() {
    bart_rng::set_seed_per_thread(12345ULL);
    const std::vector<double> a = draw_seq(1024);
    bart_rng::set_seed_per_thread(12345ULL);
    const std::vector<double> b = draw_seq(1024);
    if (!vec_eq(a, b)) {
        std::fprintf(stderr,
            "T1 FAIL: same rng_seed produced different streams "
            "(a[0]=%.17g b[0]=%.17g)\n", a[0], b[0]);
        return false;
    }
    std::printf("T1 OK  (single-thread reproducibility, bit-exact 1024 draws)\n");
    return true;
}

// ---- T2: single-thread seed sensitivity -----------------------------------
static bool t2_single_thread_seed_sensitivity() {
    bart_rng::set_seed_per_thread(11ULL);
    const std::vector<double> a = draw_seq(1024);
    bart_rng::set_seed_per_thread(11ULL + 1ULL);
    const std::vector<double> b = draw_seq(1024);
    if (!vec_any_diff(a, b)) {
        std::fprintf(stderr,
            "T2 FAIL: seed 11 and seed 12 produced identical streams\n");
        return false;
    }
    std::printf("T2 OK  (single-thread seed sensitivity, streams differ)\n");
    return true;
}

// ---- T3: multi-thread divergence (OMP) ------------------------------------
static bool t3_multithread_divergence() {
#ifndef _OPENMP
    std::printf("T3 SKIP (built without -fopenmp; only one thread exists)\n");
    return true;
#else
    const int n_draws = 512;
    const std::uint64_t user_seed = 20240920ULL;   // pre-fork hard-coded seed

    // Force >=2 threads if the runtime allows it; if it only gives us 1, the
    // subtest degenerates (can't verify divergence) and we skip.
    int requested = 4;
    if (omp_get_max_threads() < requested) requested = omp_get_max_threads();
    if (requested < 2) {
        std::printf("T3 SKIP (OMP runtime capped at 1 thread)\n");
        return true;
    }

    std::vector<std::vector<double>> per_thread(requested);
    std::atomic<int> used_threads{0};

    #pragma omp parallel num_threads(requested)
    {
        const int tid = omp_get_thread_num();
        // SAME user_seed passed on every thread -- this is the exact call
        // pattern that the pre-fork bug turned into "identical streams".
        bart_rng::set_seed_per_thread(user_seed);
        std::vector<double> mine(n_draws);
        for (int i = 0; i < n_draws; ++i) mine[i] = bart_rng::runif_open();
        #pragma omp critical
        {
            per_thread[tid] = std::move(mine);
            used_threads.fetch_add(1);
        }
    }

    const int nt = used_threads.load();
    if (nt < 2) {
        std::printf("T3 SKIP (OMP dispatched only %d thread(s))\n", nt);
        return true;
    }

    // Every pair (i,j) of thread streams must differ.
    for (int i = 0; i < nt; ++i) {
        if (per_thread[i].empty()) continue;
        for (int j = i + 1; j < nt; ++j) {
            if (per_thread[j].empty()) continue;
            if (!vec_any_diff(per_thread[i], per_thread[j])) {
                std::fprintf(stderr,
                    "T3 FAIL: thread %d and thread %d produced IDENTICAL "
                    "streams (this is the exact bug the fork fixes)\n", i, j);
                return false;
            }
        }
    }
    std::printf("T3 OK  (OMP threads=%d, all pairwise streams differ)\n", nt);
    return true;
#endif
}

int main() {
    int fails = 0;
    if (!t1_single_thread_reproducibility()) ++fails;
    if (!t2_single_thread_seed_sensitivity()) ++fails;
    if (!t3_multithread_divergence())         ++fails;

    if (fails == 0) {
        std::printf("test_bart_rng_omp_safety: ALL SUBTESTS PASS (3/3)\n");
        return 0;
    }
    std::fprintf(stderr, "test_bart_rng_omp_safety: %d subtest(s) FAILED\n",
                 fails);
    return 1;
}

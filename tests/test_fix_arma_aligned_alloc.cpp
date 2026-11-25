// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_arma_aligned_alloc.cpp
//
//  AI4BayesCode/arma_aligned_alloc.hpp, included before <armadillo> as the R
//  and Python packages include it (-include):
//    1. ARMA_ALIEN_MEM_ALLOC_FUNCTION and ARMA_ALIEN_MEM_FREE_FUNCTION are
//       defined, and arma::memory::acquire / release go through the two
//       functions of the header.
//    2. Every heap array of a Mat, Col, Row, Cube and SpMat lies on a 64-byte
//       boundary: after construction, set_size, resize, copy and move, for
//       the results of arithmetic, chol, solve, inv and vectorise, and for
//       Mat objects held in a std::vector. Arrays of at most 16 elements
//       (Mat) or 64 (Cube) are stored inside the object and are not checked.
//    3. dot(y, y), accu(W % S), solve(trimatl(L), y) and
//       solve(trimatu(L.t()), y) (N = 120) of the same values give
//       bit-identical results in 24 repetitions, each on new copies of the
//       operands made after allocations of other sizes, with the arrays of
//       earlier repetitions still allocated. With macOS Accelerate,
//       accu(W % S) and solve(trimatu(L.t()), y) give other bits when their
//       arrays lie 16, 32 or 48 bytes past a 64-byte boundary.
//    4. examples/GPRegression.cpp (joint NUTS over the hyperparameters of the
//       marginal GP likelihood, whose gradient uses accu(W % dK) and
//       triangular solves): four models built with the same seed, each after
//       different heap allocations, give bit-identical get_history() and
//       get_current() after the same number of sweeps.
// ============================================================================
#include "AI4BayesCode/arma_aligned_alloc.hpp"

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define main gp_regression_demo_main
#include "../examples/GPRegression.cpp"
#undef main

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

bool on_boundary(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p) % AI4BayesCode::arma_heap_alignment == 0;
}

bool same_bits(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           (a.n_elem == 0 ||
            std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0);
}

// Heap blocks of sizes drawn from `rng`, kept alive by the caller, so the
// arrays allocated next are placed at other addresses.
void move_heap(std::vector<std::vector<char>>& keep, std::mt19937_64& rng, int n) {
    std::uniform_int_distribution<int> size(1, 3000);
    for (int i = 0; i < n; ++i) keep.emplace_back(static_cast<std::size_t>(size(rng)));
}

// ---- 1. the allocator macros ------------------------------------------------
void allocator_macros() {
    std::printf("1. allocator macros\n");
#if defined(ARMA_ALIEN_MEM_ALLOC_FUNCTION) && defined(ARMA_ALIEN_MEM_FREE_FUNCTION)
    check(true, "ARMA_ALIEN_MEM_ALLOC_FUNCTION and ARMA_ALIEN_MEM_FREE_FUNCTION defined");
#else
    check(false, "ARMA_ALIEN_MEM_ALLOC_FUNCTION and ARMA_ALIEN_MEM_FREE_FUNCTION defined");
#endif
    check(arma::arma_config::extra_code, "arma_config::extra_code reports the alien allocator");
    // all blocks held until the end, so each acquire returns a new block
    std::vector<double*> blocks;
    bool all_on = true;
    for (arma::uword n = 1; n <= 4096; ++n) {
        double* p = arma::memory::acquire<double>(n);
        all_on = all_on && on_boundary(p);
        blocks.push_back(p);
    }
    for (double* p : blocks) arma::memory::release(p);
    check(all_on, "arma::memory::acquire<double>(n) is 64-byte aligned for n = 1..4096");
}

// ---- 2. heap arrays of Armadillo objects -----------------------------------
void heap_arrays() {
    std::printf("2. heap arrays of Armadillo objects\n");
    std::mt19937_64 rng(3);
    std::vector<std::vector<char>> keep;
    std::size_t n_checked = 0, n_on = 0;
    auto rec = [&](const void* p, arma::uword n_elem, arma::uword inline_max) {
        if (n_elem <= inline_max) return;   // stored inside the object
        ++n_checked;
        if (on_boundary(p)) ++n_on;
    };
    const arma::uword M = arma::arma_config::mat_prealloc;   // 16
    const arma::uword C = arma::Cube_prealloc::mem_n_elem;   // 64
    std::vector<arma::mat> held;
    for (arma::uword n = 1; n <= 260; ++n) {
        move_heap(keep, rng, 2);
        arma::vec v(n, arma::fill::randu);              rec(v.memptr(), v.n_elem, M);
        arma::rowvec r(n, arma::fill::randu);           rec(r.memptr(), r.n_elem, M);
        arma::mat A(n, 3, arma::fill::randu);           rec(A.memptr(), A.n_elem, M);
        arma::cube Q(2, 3, n, arma::fill::zeros);       rec(Q.memptr(), Q.n_elem, C);
        arma::vec w = 2.0 * v + 1.0;                    rec(w.memptr(), w.n_elem, M);
        arma::mat AtA = A.t() * A;                      rec(AtA.memptr(), AtA.n_elem, M);
        arma::mat vvt = v * v.t();                      rec(vvt.memptr(), vvt.n_elem, M);
        arma::vec flat = arma::vectorise(A);            rec(flat.memptr(), flat.n_elem, M);
        v.set_size(n + 3);                              rec(v.memptr(), v.n_elem, M);
        v.resize(2 * n + 5);                            rec(v.memptr(), v.n_elem, M);
        arma::vec copy(v);                              rec(copy.memptr(), copy.n_elem, M);
        arma::vec moved(std::move(copy));               rec(moved.memptr(), moved.n_elem, M);
        held.push_back(A);
        held.emplace_back(n + 1, 2, arma::fill::ones);
    }
    for (const auto& h : held) rec(h.memptr(), h.n_elem, M);
    for (arma::uword n = 2; n <= 62; n += 4) {
        move_heap(keep, rng, 3);
        arma::mat B(n, n, arma::fill::randu);
        arma::mat S = B * B.t() + static_cast<double>(n) * arma::eye(n, n);
        rec(S.memptr(), S.n_elem, M);
        arma::mat L = arma::chol(S, "lower");           rec(L.memptr(), L.n_elem, M);
        arma::vec b(n, arma::fill::randu);
        arma::vec x = arma::solve(arma::trimatl(L), b); rec(x.memptr(), x.n_elem, M);
        arma::mat Li = arma::inv(arma::trimatl(L));     rec(Li.memptr(), Li.n_elem, M);
        arma::sp_mat Sp = arma::sprandu<arma::sp_mat>(n, n, 0.4);
        if (Sp.n_nonzero > 0) rec(Sp.values, Sp.n_nonzero, 0);
    }
    check(n_checked > 2000 && n_on == n_checked,
          "every heap array on a 64-byte boundary (" + std::to_string(n_on) + " of " +
          std::to_string(n_checked) + ")");
}

// ---- 3. alignment-dependent BLAS / LAPACK results --------------------------
void blas_results() {
    std::printf("3. dot, accu(W %% S), solve(trimatl(L), y), solve(trimatu(L.t()), y)\n");
    const arma::uword N = 120;
    arma::arma_rng::set_seed(17);
    const arma::mat B0 = arma::randn(N, N);
    const arma::mat S0 = B0 * B0.t() + static_cast<double>(N) * arma::eye(N, N);
    const arma::mat L0 = arma::chol(S0, "lower");
    const arma::mat W0 = arma::randn(N, N);
    const arma::vec y0 = arma::randn(N);
    std::mt19937_64 rng(5);
    std::vector<std::vector<char>> keep;
    std::vector<arma::mat> held;   // no array of one repetition is freed before the next
    arma::mat ref[4];
    bool same[4] = {true, true, true, true};
    for (int rep = 0; rep < 24; ++rep) {
        move_heap(keep, rng, 7 + rep);
        // fresh copies, so every repetition reads arrays at new addresses
        const arma::mat W = W0, S = S0, L = L0;
        const arma::vec y = y0;
        const arma::mat r[4] = {
            arma::mat{arma::dot(y, y)},
            arma::mat{arma::accu(W % S)},
            arma::solve(arma::trimatl(L), y),
            arma::solve(arma::trimatu(L.t()), y)};
        for (int k = 0; k < 4; ++k) {
            if (rep == 0) ref[k] = r[k];
            else same[k] = same[k] && same_bits(r[k], ref[k]);
            held.push_back(r[k]);
        }
        held.push_back(y);
    }
    check(same[0], "dot(y, y): 24 repetitions bit-identical");
    check(same[1], "accu(W % S): 24 repetitions bit-identical");
    check(same[2], "solve(trimatl(L), y): 24 repetitions bit-identical");
    check(same[3], "solve(trimatu(L.t()), y): 24 repetitions bit-identical");
}

// ---- 4. GPRegression, same seed, different heap states ---------------------
void gp_regression_repeats() {
    std::printf("4. examples/GPRegression.cpp built four times with seed 11\n");
    const std::size_t N = 120;
    arma::mat X(N, 1);
    arma::vec y(N);
    std::mt19937_64 sim(2024);
    std::normal_distribution<double> noise(0.0, 0.30);
    for (std::size_t i = 0; i < N; ++i) {
        const double x = -3.0 + 6.0 * static_cast<double>(i) / static_cast<double>(N - 1);
        X(i, 0) = x;
        y[i] = std::sin(3.0 * x) + 0.5 * x + noise(sim);
    }
    std::mt19937_64 rng(9);
    std::vector<std::vector<char>> keep;
    std::vector<std::unique_ptr<GPRegression>> others;
    std::vector<AI4BayesCode::history_map> hist;
    std::vector<AI4BayesCode::state_map> cur;
    for (int b = 0; b < 4; ++b) {
        move_heap(keep, rng, 11 * (b + 1));
        // a model of another size alive alongside, as in an R or Python session
        const std::size_t n_o = 20 + 7 * static_cast<std::size_t>(b);
        others.push_back(std::make_unique<GPRegression>(
            arma::mat(X.rows(0, n_o - 1)), arma::vec(y.subvec(0, n_o - 1)), 100 + b, true));
        others.back()->step(3 * (b + 1));
        GPRegression m(X, y, /*rng_seed=*/11, /*keep_history=*/true);
        m.step(150);
        hist.push_back(m.get_history());
        cur.push_back(m.get_current());
    }
    bool same_hist = true, same_cur = true;
    for (int b = 1; b < 4; ++b) {
        for (const auto& kv : hist[0]) {
            auto it = hist[b].find(kv.first);
            same_hist = same_hist && it != hist[b].end() && same_bits(it->second, kv.second);
        }
        for (const auto& kv : cur[0]) {
            auto it = cur[b].find(kv.first);
            same_cur = same_cur && it != cur[b].end() &&
                       same_bits(arma::mat(it->second), arma::mat(kv.second));
        }
    }
    check(hist[0].at("amplitude").n_rows == 150, "get_history() has 150 rows");
    check(same_hist, "get_history() bit-identical in the four builds");
    check(same_cur,  "get_current() bit-identical in the four builds");
}

}  // namespace

int main() {
    std::printf("=== test_fix_arma_aligned_alloc ===\n\n");
    allocator_macros();
    heap_arrays();
    blas_results();
    gp_regression_repeats();
    std::printf("\ntest_fix_arma_aligned_alloc: %d failed check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}

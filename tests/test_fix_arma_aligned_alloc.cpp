// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_arma_aligned_alloc.cpp
//
//  AI4BayesCode/arma_aligned_alloc.hpp, included before <armadillo> as the R
//  and Python packages include it (-include). A = arma_heap_alignment.
//    1. ARMA_ALIEN_MEM_ALLOC_FUNCTION and ARMA_ALIEN_MEM_FREE_FUNCTION are
//       defined, and arma::memory::acquire / release go through the two
//       functions of the header.
//    2. Every heap array of a Mat, Col, Row, Cube and SpMat lies on an A-byte
//       boundary: after construction, set_size, resize, copy and move, for
//       the results of arithmetic, chol, solve, inv and vectorise, and for
//       Mat objects held in a std::vector. Arrays of at most 16 elements
//       (Mat) or 64 (Cube) are stored inside the object and are not checked.
//    3. dot(y, y), accu(W % S), solve(trimatl(L), y) and
//       solve(trimatu(L.t()), y) of the same values, for N = 110 (vectors of
//       880 bytes) and N = 120, give bit-identical results in 24
//       repetitions, each on new copies of the operands made after
//       allocations of other sizes, with the arrays of earlier repetitions
//       still allocated.
//    4. examples/GPRegression.cpp (joint NUTS over the hyperparameters of the
//       marginal GP likelihood, whose gradient uses accu(W % dK) and
//       triangular solves), N = 110 and N = 120: four models built with the
//       same seed, each after different heap allocations, give bit-identical
//       get_history() and get_current() after the same number of sweeps.
//    5. The objects the header does not cover: a Mat with at most 16
//       elements (arma_config::mat_prealloc) and a Cube with at most 64
//       elements (Cube_prealloc::mem_n_elem) hold their elements inside the
//       object; with one element more the elements are on the heap, on an
//       A-byte boundary.
//    6. ddot (n = 5000, 14400), dtrtrs (n = 110, one right-hand side) and
//       dgetrf + dgetrs (n = 110, one right-hand side), called with every
//       operand at offset k * 64 bytes from a 16384-byte boundary,
//       k = 0, ..., 255, give one result per class of the offset modulo A.
//       On an Apple M2 Max with macOS 27.0.1 these Accelerate routines give
//       results that depend on the offset modulo 256 bytes, so there this
//       check fails for A = 64 or A = 128. Sections 3 and 4 depend on the
//       addresses that malloc returns; this section places the operands
//       itself.
// ============================================================================
#include "AI4BayesCode/arma_aligned_alloc.hpp"

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
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

const std::size_t A_BYTES = AI4BayesCode::arma_heap_alignment;
const std::string A_STR = std::to_string(AI4BayesCode::arma_heap_alignment);

bool on_boundary(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p) % A_BYTES == 0;
}

bool same_bits(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           (a.n_elem == 0 ||
            std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0);
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
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
    check(all_on, "arma::memory::acquire<double>(n) is " + A_STR +
                  "-byte aligned for n = 1..4096");
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
          "every heap array on a " + A_STR + "-byte boundary (" + std::to_string(n_on) +
          " of " + std::to_string(n_checked) + ")");
}

// ---- 3. alignment-dependent BLAS / LAPACK results --------------------------
void blas_results(const arma::uword N) {
    std::printf("3. dot, accu(W %% S), solve(trimatl(L), y), solve(trimatu(L.t()), y), "
                "N = %u\n", static_cast<unsigned>(N));
    arma::arma_rng::set_seed(17);
    const arma::mat B0 = arma::randn(N, N);
    const arma::mat S0 = B0 * B0.t() + static_cast<double>(N) * arma::eye(N, N);
    const arma::mat L0 = arma::chol(S0, "lower");
    const arma::mat W0 = arma::randn(N, N);
    const arma::vec y0 = arma::randn(N);
    std::mt19937_64 rng(5 + N);
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
    const std::string sn = " (N = " + std::to_string(N) + ")";
    check(same[0], "dot(y, y): 24 repetitions bit-identical" + sn);
    check(same[1], "accu(W % S): 24 repetitions bit-identical" + sn);
    check(same[2], "solve(trimatl(L), y): 24 repetitions bit-identical" + sn);
    check(same[3], "solve(trimatu(L.t()), y): 24 repetitions bit-identical" + sn);
}

// ---- 4. GPRegression, same seed, different heap states ---------------------
void gp_regression_repeats(const std::size_t N) {
    std::printf("4. examples/GPRegression.cpp built four times with seed 11, N = %u\n",
                static_cast<unsigned>(N));
    arma::mat X(N, 1);
    arma::vec y(N);
    std::mt19937_64 sim(2024);
    std::normal_distribution<double> noise(0.0, 0.30);
    for (std::size_t i = 0; i < N; ++i) {
        const double x = -3.0 + 6.0 * static_cast<double>(i) / static_cast<double>(N - 1);
        X(i, 0) = x;
        y[i] = std::sin(3.0 * x) + 0.5 * x + noise(sim);
    }
    std::mt19937_64 rng(9 + N);
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
    const std::string sn = " (N = " + std::to_string(N) + ")";
    check(hist[0].at("amplitude").n_rows == 150, "get_history() has 150 rows" + sn);
    check(same_hist, "get_history() bit-identical in the four builds" + sn);
    check(same_cur,  "get_current() bit-identical in the four builds" + sn);
}

// ---- 5. elements stored inside the object ----------------------------------
// true if address p lies within the n_bytes of the object at obj
bool inside(const void* obj, std::size_t n_bytes, const void* p) {
    const std::uintptr_t b = reinterpret_cast<std::uintptr_t>(obj);
    const std::uintptr_t q = reinterpret_cast<std::uintptr_t>(p);
    return q >= b && q < b + n_bytes;
}

void elements_inside_the_object() {
    std::printf("5. elements stored inside the object\n");
    check(arma::arma_config::mat_prealloc == 16 && arma::Cube_prealloc::mem_n_elem == 64,
          "arma_config::mat_prealloc is 16 and Cube_prealloc::mem_n_elem is 64");
    std::mt19937_64 rng(13);
    std::vector<std::vector<char>> keep;
    bool mat16 = true, mat17 = true, cube64 = true, cube65 = true;
    for (int rep = 0; rep < 32; ++rep) {
        move_heap(keep, rng, 3);
        const arma::mat a(4, 4, arma::fill::ones);          // 16 elements
        const arma::mat b(17, 1, arma::fill::ones);         // 17
        const arma::cube q(4, 4, 4, arma::fill::ones);      // 64
        const arma::cube r(5, 13, 1, arma::fill::ones);     // 65
        mat16  = mat16  && inside(&a, sizeof(a), a.memptr());
        mat17  = mat17  && !inside(&b, sizeof(b), b.memptr()) && on_boundary(b.memptr());
        cube64 = cube64 && inside(&q, sizeof(q), q.memptr());
        cube65 = cube65 && !inside(&r, sizeof(r), r.memptr()) && on_boundary(r.memptr());
    }
    check(mat16,  "Mat with 16 elements: elements inside the object");
    check(mat17,  "Mat with 17 elements: elements on the heap, on a " + A_STR + "-byte boundary");
    check(cube64, "Cube with 64 elements: elements inside the object");
    check(cube65, "Cube with 65 elements: elements on the heap, on a " + A_STR + "-byte boundary");
}

// ---- 6. routine results at offsets k * 64 from a 16384-byte boundary --------
// Room for n_bytes at every offset k * 64 bytes, k = 0..255, from a
// 16384-byte boundary inside the buffer.
class OffsetBuffer {
public:
    explicit OffsetBuffer(std::size_t n_bytes) : raw_(n_bytes + 2 * 16384) {
        const std::uintptr_t p = reinterpret_cast<std::uintptr_t>(raw_.data());
        base_ = raw_.data() + (16384 - p % 16384) % 16384;
    }
    template <class T> T* at(std::size_t off) { return reinterpret_cast<T*>(base_ + off); }
private:
    std::vector<unsigned char> raw_;
    unsigned char* base_;
};

// f(off) returns the result of one call with every operand at offset off.
// Returns true if all offsets in one class of the offset modulo A give the
// same bits; n_distinct is the number of different results over all offsets.
template <class F>
bool one_result_per_class(F f, std::size_t& n_distinct) {
    std::map<std::size_t, std::vector<double>> first;   // class -> result
    std::vector<std::vector<double>> distinct;
    bool ok = true;
    for (std::size_t k = 0; k < 256; ++k) {
        const std::size_t off = 64 * k;
        const std::vector<double> r = f(off);
        bool seen = false;
        for (const auto& d : distinct) seen = seen || same_bits(d, r);
        if (!seen) distinct.push_back(r);
        const auto it = first.find(off % A_BYTES);
        if (it == first.end()) first.emplace(off % A_BYTES, r);
        else ok = ok && same_bits(it->second, r);
    }
    n_distinct = distinct.size();
    return ok;
}

std::vector<double> normal_draws(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd;
    std::vector<double> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

void routine_offset_classes() {
    std::printf("6. ddot, dtrtrs and dgetrf + dgetrs with operands at offsets k * 64 bytes\n");
    std::size_t n_distinct = 0;
    for (const int n : {5000, 14400}) {
        const std::size_t nn = static_cast<std::size_t>(n);
        const std::vector<double> x0 = normal_draws(nn, 21), y0 = normal_draws(nn, 22);
        OffsetBuffer bx(nn * 8), by(nn * 8);
        const bool ok = one_result_per_class([&](std::size_t off) {
            double* x = bx.at<double>(off); double* y = by.at<double>(off);
            std::memcpy(x, x0.data(), nn * 8); std::memcpy(y, y0.data(), nn * 8);
            return std::vector<double>{arma::blas::dot(arma::uword(n), x, y)};
        }, n_distinct);
        check(ok, "ddot, n = " + std::to_string(n) + ": one result per class of the offset "
                  "modulo " + A_STR + " (" + std::to_string(n_distinct) +
                  " different results over the 256 offsets)");
    }
    // not const: Armadillo releases before 15.8 declare the LAPACK wrappers
    // with non-const pointer parameters
    arma::blas_int n = 110, n_rhs = 1;
    const std::size_t nn = 110, n2 = nn * nn;
    const std::vector<double> a0 = normal_draws(n2, 23), b0 = normal_draws(nn, 24);
    std::vector<double> L0(n2, 0.0), G0(a0);
    for (std::size_t j = 0; j < nn; ++j) {
        for (std::size_t i = j; i < nn; ++i)
            L0[i + j * nn] = (i == j) ? 5.0 + std::fabs(a0[i + j * nn]) : 0.3 * a0[i + j * nn];
        G0[j + j * nn] += static_cast<double>(nn);
    }
    {
        OffsetBuffer bL(n2 * 8), bb(nn * 8);
        const bool ok = one_result_per_class([&](std::size_t off) {
            double* L = bL.at<double>(off); double* b = bb.at<double>(off);
            std::memcpy(L, L0.data(), n2 * 8); std::memcpy(b, b0.data(), nn * 8);
            char uplo = 'L', trans = 'N', diag = 'N';
            arma::blas_int info = 0;
            arma::lapack::trtrs<double>(&uplo, &trans, &diag, &n, &n_rhs, L, &n, b, &n, &info);
            return std::vector<double>(b, b + nn);
        }, n_distinct);
        check(ok, "dtrtrs (lower, n = 110, one right-hand side): one result per class of the "
                  "offset modulo " + A_STR + " (" + std::to_string(n_distinct) +
                  " different results over the 256 offsets)");
    }
    {
        OffsetBuffer bA(n2 * 8), bb(nn * 8), bp(nn * sizeof(arma::blas_int));
        const bool ok = one_result_per_class([&](std::size_t off) {
            double* G = bA.at<double>(off); double* b = bb.at<double>(off);
            arma::blas_int* ipiv = bp.at<arma::blas_int>(off);
            std::memcpy(G, G0.data(), n2 * 8); std::memcpy(b, b0.data(), nn * 8);
            char trans = 'N';
            arma::blas_int info = 0;
            arma::lapack::getrf<double>(&n, &n, G, &n, ipiv, &info);
            arma::lapack::getrs<double>(&trans, &n, &n_rhs, G, &n, ipiv, b, &n, &info);
            return std::vector<double>(b, b + nn);
        }, n_distinct);
        check(ok, "dgetrf + dgetrs (n = 110, one right-hand side): one result per class of "
                  "the offset modulo " + A_STR + " (" + std::to_string(n_distinct) +
                  " different results over the 256 offsets)");
    }
}

}  // namespace

int main() {
    std::printf("=== test_fix_arma_aligned_alloc (arma_heap_alignment = %s) ===\n\n",
                A_STR.c_str());
    allocator_macros();
    heap_arrays();
    blas_results(110);
    blas_results(120);
    gp_regression_repeats(110);
    gp_regression_repeats(120);
    elements_inside_the_object();
    routine_offset_classes();
    std::printf("\ntest_fix_arma_aligned_alloc: %d failed check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}

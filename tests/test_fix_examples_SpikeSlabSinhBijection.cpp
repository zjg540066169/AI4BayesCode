// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_SpikeSlabSinhBijection.cpp
//
//  examples/SpikeSlabSinhBijection.cpp: set_current of the scalar keys gamma
//  and beta.
//    C1  set_current with an empty value for gamma and for beta throws and
//        leaves get_current() unchanged (bit for bit);
//    C2  set_current of gamma = 1 and beta = 1.2 is reported by
//        get_current() (relative tolerance 1e-12).
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main spike_slab_sinh_demo_main
#include "../examples/SpikeSlabSinhBijection.cpp"
#undef main

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

// set_current(s) throws, and get_current() is unchanged.
bool throws_unchanged(SpikeSlabSinhBijection& m, const AI4BayesCode::state_map& s) {
    const auto before = m.get_current();
    bool threw = false;
    try { m.set_current(s); } catch (const std::exception&) { threw = true; }
    const auto after = m.get_current();
    bool same = threw && before.size() == after.size();
    for (const auto& kv : before)
        same = same && after.count(kv.first) && same_bits(kv.second, after.at(kv.first));
    return same;
}

}  // namespace

int main() {
    // Unbuffered, so the lines printed before a crash reach the log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== test_fix_examples_SpikeSlabSinhBijection ===\n");
    const std::size_t N = 60;
    std::mt19937_64 g(123u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::vec x(N), y(N);
    for (std::size_t i = 0; i < N; ++i) { x[i] = nd(g); y[i] = 1.5 * x[i] + nd(g); }
    SpikeSlabSinhBijection m(y, x, 1.0, 5.0, 0.5, 7, false);
    m.step(2);
    for (const char* key : {"gamma", "beta"}) {
        AI4BayesCode::state_map s1;
        s1[key] = arma::vec();
        check(throws_unchanged(m, s1),
              std::string("C1 set_current(") + key + " = empty) throws and leaves the state unchanged");
    }

    AI4BayesCode::state_map s2;
    s2["gamma"] = arma::vec{1.0};
    s2["beta"] = arma::vec{1.2};
    m.set_current(s2);
    const auto cur = m.get_current();
    bool loaded = true;
    for (const auto& kv : s2) {
        const arma::vec& v = cur.at(kv.first);
        loaded = loaded && v.n_elem == kv.second.n_elem;
        for (std::size_t i = 0; loaded && i < v.n_elem; ++i)
            loaded = std::abs(v[i] - kv.second[i]) <= 1e-12 * std::max(1.0, std::abs(kv.second[i]));
    }
    check(loaded, "C2 the loaded values are reported by get_current()");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

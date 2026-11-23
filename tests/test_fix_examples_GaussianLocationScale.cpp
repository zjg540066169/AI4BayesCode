// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_GaussianLocationScale.cpp
//
//  examples/GaussianLocationScale.cpp: set_current of the scalar keys mu and
//  sigma.
//    C1  set_current with an empty value for mu and for sigma throws and
//        leaves get_current() unchanged (bit for bit);
//    C2  set_current of mu = 2.5 and sigma = 1.5 is reported by
//        get_current() (relative tolerance 1e-12).
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main gaussian_location_scale_demo_main
#include "../examples/GaussianLocationScale.cpp"
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
bool throws_unchanged(GaussianLocationScale& m, const AI4BayesCode::state_map& s) {
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
    std::printf("=== test_fix_examples_GaussianLocationScale ===\n");
    const std::size_t N = 80;
    std::mt19937_64 g(42u);
    std::normal_distribution<double> nd(3.0, 2.0);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) y[i] = nd(g);
    GaussianLocationScale m(y, 7, false);
    m.step(2);
    for (const char* key : {"mu", "sigma"}) {
        AI4BayesCode::state_map s1;
        s1[key] = arma::vec();
        check(throws_unchanged(m, s1),
              std::string("C1 set_current(") + key + " = empty) throws and leaves the state unchanged");
    }

    AI4BayesCode::state_map s2;
    s2["mu"] = arma::vec{2.5};
    s2["sigma"] = arma::vec{1.5};
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

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_IRT1PL_joint.cpp
//
//  examples/IRT1PL_joint.cpp: set_current of the scalar keys sigma_b.
//    C1  set_current with an empty value for sigma_b throws and leaves
//        get_current() unchanged (bit for bit);
//    C2  set_current of sigma_b = 0.8 is reported by get_current()
//        (relative tolerance 1e-12).
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main irt1pl_joint_demo_main
#include "../examples/IRT1PL_joint.cpp"
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
bool throws_unchanged(IRT1PL_joint& m, const AI4BayesCode::state_map& s) {
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
    std::printf("=== test_fix_examples_IRT1PL_joint ===\n");
    const std::size_t N = 40, J = 6;
    std::mt19937_64 g(12345u);
    std::normal_distribution<double> rn(0.0, 1.0);
    std::uniform_real_distribution<double> ru(0.0, 1.0);
    arma::mat Y(N, J);
    for (std::size_t j = 0; j < J; ++j)
        for (std::size_t i = 0; i < N; ++i)
            Y(i, j) = ru(g) < 1.0 / (1.0 + std::exp(-rn(g))) ? 1.0 : 0.0;
    const arma::vec th0(N, arma::fill::zeros), b0(J, arma::fill::zeros);
    IRT1PL_joint m(Y, th0, b0, 1.0, 7, false);
    m.step(2);
    for (const char* key : {"sigma_b"}) {
        AI4BayesCode::state_map s1;
        s1[key] = arma::vec();
        check(throws_unchanged(m, s1),
              std::string("C1 set_current(") + key + " = empty) throws and leaves the state unchanged");
    }

    AI4BayesCode::state_map s2;
    s2["sigma_b"] = arma::vec{0.8};
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

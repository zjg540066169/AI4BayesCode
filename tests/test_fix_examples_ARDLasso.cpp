// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_ARDLasso.cpp
//
//  examples/ARDLasso.cpp: set_current of beta, alpha, sigma2 and psi2 (the
//  members the Gibbs sweep updates).
//    C1  set_current with a valid beta and sigma2 = -1 throws and leaves
//        get_current() unchanged (bit for bit);
//    C2  the same for a valid beta and alpha with a psi2 of wrong length;
//    C3  set_current of all four keys is reported by get_current();
//    C4  set_current with an empty alpha, or an empty sigma2, throws and
//        leaves get_current() unchanged.
// ============================================================================
#include <armadillo>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#define main ard_lasso_demo_main
#include "../examples/ARDLasso.cpp"
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
bool throws_unchanged(ARDLasso& m, const AI4BayesCode::state_map& s) {
    const auto before = m.get_current();
    bool threw = false;
    try { m.set_current(s); } catch (const std::exception&) { threw = true; }
    const auto after = m.get_current();
    bool same = threw && before.size() == after.size();
    for (const auto& kv : before) same = same && same_bits(kv.second, after.at(kv.first));
    return same;
}

}  // namespace

int main() {
    // Unbuffered, so the lines printed before a crash reach the log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== test_fix_examples_ARDLasso ===\n");
    const std::size_t N = 100, p = 4;
    std::mt19937_64 gen(29u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::mat X(N, p);
    arma::vec Y(N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < p; ++j) X(i, j) = nd(gen);
        Y[i] = 1.0 + 2.0 * X(i, 0) - 1.0 * X(i, 2) + 0.5 * nd(gen);
    }
    ARDLasso m(X, Y, 7, false);
    m.step(3);

    // C1
    AI4BayesCode::state_map s1;
    s1["beta"] = arma::vec{1.0, 2.0, 3.0, 4.0};
    s1["sigma2"] = arma::vec{-1.0};
    const bool c1 = throws_unchanged(m, s1);
    check(c1, "C1 a valid beta with sigma2 = -1 throws and changes nothing");

    // C2
    AI4BayesCode::state_map s2;
    s2["beta"] = arma::vec{4.0, 3.0, 2.0, 1.0};
    s2["alpha"] = arma::vec{5.0};
    s2["psi2"] = arma::vec{1.0, 1.0};
    const bool c2 = throws_unchanged(m, s2);
    check(c2, "C2 a valid beta and alpha with a psi2 of wrong length throw and change nothing");

    // C3
    AI4BayesCode::state_map s3;
    s3["beta"] = arma::vec{1.0, 2.0, 3.0, 4.0};
    s3["alpha"] = arma::vec{5.0};
    s3["sigma2"] = arma::vec{0.25};
    s3["psi2"] = arma::vec{1.0, 2.0, 3.0, 4.0};
    m.set_current(s3);
    const auto g3 = m.get_current();
    bool c3 = true;
    for (const auto& kv : s3) c3 = c3 && same_bits(g3.at(kv.first), kv.second);
    check(c3, "C3 the loaded beta, alpha, sigma2 and psi2 are reported");

    for (const char* key : {"alpha", "sigma2"}) {
        AI4BayesCode::state_map s4;
        s4[key] = arma::vec();
        check(throws_unchanged(m, s4),
              std::string("C4 set_current(") + key + " = empty) throws and leaves the state unchanged");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

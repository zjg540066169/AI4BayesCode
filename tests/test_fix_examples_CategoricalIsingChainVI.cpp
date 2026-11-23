// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_CategoricalIsingChainVI.cpp
//
//  examples/CategoricalIsingChainVI.cpp: the VI log-density holds beta and
//  h from construction, and get_current() reports the shared "beta" and
//  "h". set_current must not report a beta or h that the optimisation does
//  not use.
//    C1  set_current with another beta throws and leaves the shared beta
//        and phi unchanged;
//    C2  set_current with another h throws and leaves the shared h
//        unchanged;
//    C3  set_current with a valid phi and another h throws and leaves phi
//        unchanged;
//    C4  set_current(get_current()) is accepted and leaves phi, beta and h
//        unchanged.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main categorical_ising_chain_vi_demo_main
#include "../examples/CategoricalIsingChainVI.cpp"
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

bool throws(CategoricalIsingChainVI& m, const AI4BayesCode::state_map& s) {
    try { m.set_current(s); } catch (const std::exception&) { return true; }
    return false;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_CategoricalIsingChainVI ===\n");
    const int n = 4, K = 3;
    const arma::vec h(n, arma::fill::value(2.0));
    CategoricalIsingChainVI m(n, K, 0.8, h, true, 20260621, false);
    m.step(300);

    // C1
    const auto c0 = m.get_current();
    AI4BayesCode::state_map s1;
    s1["beta"] = arma::vec{1.6};
    const bool t1 = throws(m, s1);
    const auto c1 = m.get_current();
    std::printf("C1 set_current(beta = 1.6): threw = %d, reported beta = %.2f\n",
                t1 ? 1 : 0, c1.at("beta")[0]);
    check(t1 && same_bits(c0.at("beta"), c1.at("beta")) && same_bits(c0.at("phi"), c1.at("phi")),
          "C1 another beta is rejected and the reported beta is the one in use");

    // C2
    AI4BayesCode::state_map s2;
    s2["h"] = arma::vec(n, arma::fill::value(-1.0));
    const bool t2 = throws(m, s2);
    check(t2 && same_bits(c0.at("h"), m.get_current().at("h")),
          "C2 another h is rejected and the reported h is the one in use");

    // C3
    arma::vec phi_new(n * K, arma::fill::value(1.0 / K));
    AI4BayesCode::state_map s3;
    s3["phi"] = phi_new;
    s3["h"]   = arma::vec(n, arma::fill::value(-1.0));
    const bool t3 = throws(m, s3);
    check(t3 && same_bits(c0.at("phi"), m.get_current().at("phi")),
          "C3 a set_current that fails on h leaves phi unchanged");

    // C4
    const bool t4 = throws(m, m.get_current());
    const auto c4 = m.get_current();
    check(!t4 && same_bits(c0.at("phi"), c4.at("phi")) && same_bits(c0.at("beta"), c4.at("beta")) &&
          same_bits(c0.at("h"), c4.at("h")),
          "C4 set_current(get_current()) is accepted and changes nothing");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

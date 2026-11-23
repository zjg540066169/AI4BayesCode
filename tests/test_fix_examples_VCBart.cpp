// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_VCBart.cpp
//
//  examples/VCBart.cpp: sigma is drawn inside the BART kernel of beta_0 (no
//  loader exists for it) and copied to the shared "sigma" after each beta_0
//  update; get_current() reports the shared value.
//    C1  set_current with another sigma throws and leaves the reported sigma
//        unchanged (a value written only to the shared data would be
//        replaced by beta_0's own sigma at the next sweep);
//    C2  set_current with a y of the wrong length throws and leaves y, the
//        working responses and mu unchanged;
//    C3  set_current(get_current()) is accepted and changes nothing;
//    C4  set_current of a valid y recomputes every working response
//        wr_j = (y - sum_{k != j} x_k beta_k) / x_j and mu (relative
//        tolerance 1e-12).
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main vcbart_demo_main
#include "../examples/VCBart.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> VCBart::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &VCBart::impl_>;

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

bool throws(VCBart& m, const AI4BayesCode::state_map& s) {
    try { m.set_current(s); } catch (const std::exception&) { return true; }
    return false;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_VCBart ===\n");
    const std::size_t N = 150;
    std::mt19937_64 g(47u);
    std::uniform_real_distribution<double> uZ(-1.0, 1.0), uX(0.5, 1.5);
    std::normal_distribution<double> eps(0.0, 0.3);
    arma::mat X(N, 1), Z(N, 1);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        Z(i, 0) = uZ(g); X(i, 0) = uX(g);
        y[i] = 2.0 * std::max(Z(i, 0), 0.0) + Z(i, 0) * Z(i, 0) * X(i, 0) + eps(g);
    }
    VCBart m(X, Z, y, 10, 2.0, 2.0, 0.95, 3.0, 100, 1, false);
    m.step(10);
    const auto& d = (m.*member_of(impl_tag{}))->data();

    // C1
    const auto c0 = m.get_current();
    AI4BayesCode::state_map s1;
    s1["sigma"] = arma::vec{c0.at("sigma")[0] * 3.0};
    const bool t1 = throws(m, s1);
    std::printf("C1 set_current(sigma x 3): threw = %d, reported sigma %.4f (before %.4f)\n",
                t1 ? 1 : 0, m.get_current().at("sigma")[0], c0.at("sigma")[0]);
    check(t1 && same_bits(c0.at("sigma"), m.get_current().at("sigma")),
          "C1 another sigma is rejected and the reported sigma is the one in use");

    // C2
    const arma::vec wr0 = d.get("wr_0"), mu0 = d.get("mu");
    AI4BayesCode::state_map s2;
    s2["y"] = arma::vec(N - 1, arma::fill::ones);
    const bool t2 = throws(m, s2);
    check(t2 && same_bits(d.get("y"), y) && same_bits(d.get("wr_0"), wr0) && same_bits(d.get("mu"), mu0),
          "C2 a y of the wrong length is rejected and nothing is written");

    // C3
    const auto before = m.get_current();
    const bool t3 = throws(m, before);
    const auto after = m.get_current();
    check(!t3 && same_bits(before.at("sigma"), after.at("sigma")) &&
          same_bits(before.at("mu"), after.at("mu")),
          "C3 set_current(get_current()) is accepted and changes nothing");

    // C4
    const arma::vec y2 = y * 2.0 + 1.0;
    AI4BayesCode::state_map s4;
    s4["y"] = y2;
    m.set_current(s4);
    const arma::vec& b0 = d.get("beta_0");
    const arma::vec& b1 = d.get("beta_1");
    double e = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double wr0_ref = (y2[i] - X(i, 0) * b1[i]) / 1.0;
        const double wr1_ref = (y2[i] - b0[i]) / X(i, 0);
        const double mu_ref  = b0[i] + X(i, 0) * b1[i];
        e = std::max(e, std::abs(d.get("wr_0")[i] - wr0_ref) / std::max(1.0, std::abs(wr0_ref)));
        e = std::max(e, std::abs(d.get("wr_1")[i] - wr1_ref) / std::max(1.0, std::abs(wr1_ref)));
        e = std::max(e, std::abs(d.get("mu")[i] - mu_ref) / std::max(1.0, std::abs(mu_ref)));
    }
    std::printf("C4 set_current(y): max rel error of wr_0, wr_1, mu = %.3g\n", e);
    check(e <= 1e-12, "C4 set_current(y) recomputes the working responses and mu");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

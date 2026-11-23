// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_PehSharedFrailty.cpp
//
//  examples/PehSharedFrailty.cpp: w (frailties, child 0) and lambda (hazard
//  steps, child 1) are block states; "expanded_w" = w[z] is derived from w
//  and is the offset of lambda's conditional.
//    C1  after set_current of w and lambda, both blocks hold the loaded
//        values (the fallback rows of get_history()), and get_current()
//        reports expanded_w = w[z] of the loaded w;
//    C2  with "w" frozen, after set_current(w) and 5 sweeps, expanded_w
//        equals w[z] of the loaded w;
//    C3  set_current with a valid lambda and a w of the wrong length
//        throws and leaves lambda, w and expanded_w unchanged; so does a w
//        with a non-positive entry;
//    C4  set_current(get_current()) leaves lambda, w and expanded_w
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

#define main peh_shared_frailty_demo_main
#include "../examples/PehSharedFrailty.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> PehSharedFrailty::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &PehSharedFrailty::impl_>;

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

arma::vec expand(const arma::vec& w, const arma::vec& z) {
    arma::vec e(z.n_elem);
    for (std::size_t i = 0; i < z.n_elem; ++i) e[i] = w[static_cast<std::size_t>(z[i])];
    return e;
}

bool throws(PehSharedFrailty& m, const AI4BayesCode::state_map& s) {
    try { m.set_current(s); } catch (const std::exception&) { return true; }
    return false;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_PehSharedFrailty ===\n");
    const std::size_t G = 5, n_per = 40, n = G * n_per;
    const arma::vec edges({0.0, 2.0, 5.0, 30.0});
    std::mt19937_64 rng(53u);
    std::exponential_distribution<double> ex(0.8);
    arma::vec t(n), delta(n), z(n);
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = static_cast<double>(i / n_per);
        t[i] = std::min(ex(rng) * (1.0 + 0.3 * z[i]), 25.0);
        delta[i] = (t[i] < 25.0) ? 1.0 : 0.0;
    }
    const arma::vec w_new{0.5, 1.5, 2.0, 0.8, 1.1};
    const arma::vec lam_new{0.3, 0.6, 0.9};

    // C1
    PehSharedFrailty m(t, delta, z, static_cast<int>(G), edges, 2.0, 0.01, 0.01, 7, false);
    m.step(5);
    AI4BayesCode::state_map s;
    s["w"] = w_new;
    s["lambda"] = lam_new;
    m.set_current(s);
    const auto h = m.get_history();
    const auto c = m.get_current();
    std::printf("C1 set_current(w, lambda): history w[0] %.3f, expanded_w[0] %.3f (loaded w[0] %.3f)\n",
                h.at("w")(0, 0), c.at("expanded_w")[0], w_new[0]);
    check(same_bits(arma::vec(h.at("w").row(0).t()), w_new) &&
          same_bits(arma::vec(h.at("lambda").row(0).t()), lam_new),
          "C1 both blocks hold the loaded values");
    check(same_bits(c.at("expanded_w"), expand(w_new, z)),
          "C1 get_current() reports expanded_w = w[z] of the loaded w");

    // C2
    PehSharedFrailty m2(t, delta, z, static_cast<int>(G), edges, 2.0, 0.01, 0.01, 9, false);
    m2.step(5);
    (m2.*member_of(impl_tag{}))->freeze(std::vector<std::string>{"w"}, true);
    AI4BayesCode::state_map s2;
    s2["w"] = w_new;
    m2.set_current(s2);
    m2.step(5);
    check(same_bits(m2.get_current().at("expanded_w"), expand(w_new, z)),
          "C2 with w frozen, expanded_w equals w[z] of the loaded w");

    // C3
    const auto before = m.get_current();
    AI4BayesCode::state_map bad;
    bad["lambda"] = arma::vec{9.0, 9.0, 9.0};
    bad["w"] = arma::vec{1.0, 1.0};
    const bool t3a = throws(m, bad);
    AI4BayesCode::state_map bad2;
    bad2["w"] = arma::vec{1.0, -1.0, 1.0, 1.0, 1.0};
    const bool t3b = throws(m, bad2);
    const auto after = m.get_current();
    check(t3a && t3b && same_bits(before.at("lambda"), after.at("lambda")) &&
          same_bits(before.at("w"), after.at("w")) &&
          same_bits(before.at("expanded_w"), after.at("expanded_w")),
          "C3 a failing set_current leaves lambda, w and expanded_w unchanged");

    // C4
    m.step(3);
    const auto b4 = m.get_current();
    m.set_current(b4);
    const auto a4 = m.get_current();
    check(same_bits(b4.at("lambda"), a4.at("lambda")) && same_bits(b4.at("w"), a4.at("w")) &&
          same_bits(b4.at("expanded_w"), a4.at("expanded_w")),
          "C4 set_current(get_current()) changes nothing");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

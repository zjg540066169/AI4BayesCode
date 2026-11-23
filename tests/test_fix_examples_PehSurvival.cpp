// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_PehSurvival.cpp
//
//  examples/PehSurvival.cpp: lambda (hazard steps) is the state of the
//  piecewise-exponential block.
//    C1  after set_current(lambda), the block holds the loaded lambda (the
//        fallback row of get_history()) and get_current() reports it;
//    C2  with keep_history and "lambda" frozen, the rows recorded after
//        set_current(lambda) equal the loaded lambda;
//    C3  set_current with a lambda of the wrong length, or with a
//        non-positive entry, throws and leaves lambda unchanged;
//    C4  set_current(get_current()) leaves lambda unchanged.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main peh_survival_demo_main
#include "../examples/PehSurvival.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> PehSurvival::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &PehSurvival::impl_>;

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

bool throws(PehSurvival& m, const AI4BayesCode::state_map& s) {
    try { m.set_current(s); } catch (const std::exception&) { return true; }
    return false;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_PehSurvival ===\n");
    const std::size_t n = 200;
    const arma::vec edges({0.0, 2.0, 5.0, 30.0});
    std::mt19937_64 rng(59u);
    std::exponential_distribution<double> ex(0.8);
    arma::vec t(n), delta(n);
    for (std::size_t i = 0; i < n; ++i) {
        t[i] = std::min(ex(rng), 25.0);
        delta[i] = (t[i] < 25.0) ? 1.0 : 0.0;
    }
    const arma::vec lam_new{0.3, 0.6, 0.9};

    // C1
    PehSurvival m(t, delta, edges, 0.01, 0.01, 7, false);
    m.step(5);
    AI4BayesCode::state_map s;
    s["lambda"] = lam_new;
    m.set_current(s);
    const arma::vec hrow = m.get_history().at("lambda").row(0).t();
    std::printf("C1 set_current(lambda): block lambda[0] %.3f, reported %.3f (loaded %.3f)\n",
                hrow[0], m.get_current().at("lambda")[0], lam_new[0]);
    check(same_bits(hrow, lam_new) && same_bits(m.get_current().at("lambda"), lam_new),
          "C1 the block holds the loaded lambda");

    // C2
    PehSurvival m2(t, delta, edges, 0.01, 0.01, 9, true);
    m2.step(5);
    (m2.*member_of(impl_tag{}))->freeze(std::vector<std::string>{"lambda"}, true);
    m2.set_current(s);
    m2.step(3);
    const arma::mat H = m2.get_history().at("lambda");
    bool rows_ok = H.n_rows == 8;
    for (std::size_t r = 5; rows_ok && r < 8; ++r)
        rows_ok = same_bits(arma::vec(H.row(r).t()), lam_new);
    check(rows_ok, "C2 with lambda frozen, the recorded rows are the loaded lambda");

    // C3
    const arma::vec before = m.get_current().at("lambda");
    AI4BayesCode::state_map bad1, bad2;
    bad1["lambda"] = arma::vec{1.0, 1.0};
    bad2["lambda"] = arma::vec{1.0, 0.0, 1.0};
    const bool t1 = throws(m, bad1), t2 = throws(m, bad2);
    check(t1 && t2 && same_bits(before, m.get_current().at("lambda")) &&
          same_bits(before, arma::vec(m.get_history().at("lambda").row(0).t())),
          "C3 an invalid lambda is rejected and nothing is written");

    // C4
    m.step(2);
    const auto b4 = m.get_current();
    m.set_current(b4);
    check(same_bits(b4.at("lambda"), m.get_current().at("lambda")),
          "C4 set_current(get_current()) changes nothing");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

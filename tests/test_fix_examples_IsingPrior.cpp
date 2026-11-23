// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_IsingPrior.cpp
//
//  examples/IsingPrior.cpp: x (labels) is the state of the Ising cluster
//  block, which rounds a loaded x; get_current() reports the shared "x".
//    C1  after set_current(x = 0.6 everywhere), the shared x equals the
//        block's labels (all 1);
//    C2  set_current with an x whose last entry rounds outside 0..Q-1
//        throws and leaves the block's labels and the shared x unchanged;
//    C3  set_current with a valid x and beta < 0 throws and leaves x
//        unchanged;
//    C4  the error for an x entry outside 0..Q-1 after rounding is
//        std::invalid_argument, the exception of
//        ising_cluster_block::set_current.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main ising_prior_demo_main
#include "../examples/IsingPrior.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> IsingPrior::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &IsingPrior::impl_>;

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

bool throws(IsingPrior& m, const AI4BayesCode::state_map& s) {
    try { m.set_current(s); } catch (const std::exception&) { return true; }
    return false;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_IsingPrior ===\n");
    const int L = 6, Q = 3;
    const std::size_t N = static_cast<std::size_t>(L * L);
    IsingPrior m(L, L, Q, 0.4, false, false, 7, false);
    m.step(3);
    auto& c = *(m.*member_of(impl_tag{}));

    AI4BayesCode::state_map s;
    s["x"] = arma::vec(N, arma::fill::value(0.6));
    m.set_current(s);
    std::printf("C1 set_current(x = 0.6): block x[0] %.1f, reported x[0] %.1f\n",
                c.child(0).current()[0], m.get_current().at("x")[0]);
    check(same_bits(c.child(0).current(), m.get_current().at("x")) &&
          c.child(0).current()[0] == 1.0,
          "C1 the shared x equals the block's labels");

    m.step(2);
    const arma::vec xb = c.child(0).current(), xs = m.get_current().at("x");
    arma::vec bad_x(N, arma::fill::zeros);
    bad_x[N - 1] = 7.0;
    AI4BayesCode::state_map b1, b2;
    b1["x"] = bad_x;
    b2["x"] = arma::vec(N, arma::fill::ones);
    b2["beta"] = arma::vec{-1.0};
    const bool t1 = throws(m, b1);
    const bool c2 = same_bits(xb, c.child(0).current()) && same_bits(xs, m.get_current().at("x"));
    const bool t2 = throws(m, b2);
    const bool c3 = same_bits(xb, c.child(0).current()) && same_bits(xs, m.get_current().at("x"));
    check(t1 && c2, "C2 an x entry outside 0..Q-1 is rejected and nothing is written");
    check(t2 && c3, "C3 a set_current that fails on beta leaves x unchanged");

    {
        bool invalid_argument = false;
        try { m.set_current(b1); }
        catch (const std::invalid_argument&) { invalid_argument = true; }
        catch (const std::exception&) {}
        check(invalid_argument, "C4 an x entry outside 0..Q-1 throws std::invalid_argument");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

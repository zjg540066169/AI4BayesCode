// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_IsingHiddenPotts.cpp
//
//  examples/IsingHiddenPotts.cpp: z (labels) is the state of the Ising
//  cluster block, which rounds a loaded z; get_current() reports the shared
//  "z".
//    C1  after set_current(z = 0.6 everywhere), the shared z equals the
//        block's labels (all 1);
//    C2  set_current with a z whose last entry is -0.6 (rounds to -1)
//        throws and leaves the block's labels and the shared z unchanged;
//    C3  an entry of -1, which passes the example's own check (z + 0.5
//        truncated toward zero is 0 or 1), throws std::invalid_argument, the
//        exception of ising_cluster_block::set_current; an entry of 2 throws
//        std::runtime_error from the example's own check.
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

#define main ising_hidden_potts_demo_main
#include "../examples/IsingHiddenPotts.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> IsingHiddenPotts::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &IsingHiddenPotts::impl_>;

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

}  // namespace

int main() {
    std::printf("=== test_fix_examples_IsingHiddenPotts ===\n");
    IsingHiddenPotts m(8, 0.6, 0.5, -1.0, 1.0, 0.8, 2026, 7, 0, false);
    const std::size_t N = m.n_sites();
    m.step(3);
    auto& c = *(m.*member_of(impl_tag{}));

    AI4BayesCode::state_map s;
    s["z"] = arma::vec(N, arma::fill::value(0.6));
    m.set_current(s);
    std::printf("C1 set_current(z = 0.6): block z[0] %.1f, reported z[0] %.1f\n",
                c.child(0).current()[0], m.get_current().at("z")[0]);
    check(same_bits(c.child(0).current(), m.get_current().at("z")) &&
          c.child(0).current()[0] == 1.0,
          "C1 the shared z equals the block's labels");

    m.step(2);
    const arma::vec zb = c.child(0).current(), zs = m.get_current().at("z");
    arma::vec bad(N, arma::fill::ones);
    bad[N - 1] = -0.6;
    AI4BayesCode::state_map b;
    b["z"] = bad;
    bool threw = false;
    try { m.set_current(b); } catch (const std::exception&) { threw = true; }
    check(threw && same_bits(zb, c.child(0).current()) && same_bits(zs, m.get_current().at("z")),
          "C2 a z entry rounding to -1 is rejected and nothing is written");

    {
        arma::vec z_m1(N, arma::fill::ones), z_2(N, arma::fill::ones);
        z_m1[0] = -1.0;
        z_2[0] = 2.0;
        AI4BayesCode::state_map sm1, s2;
        sm1["z"] = z_m1;
        s2["z"] = z_2;
        bool invalid_argument = false, runtime_error = false;
        try { m.set_current(sm1); }
        catch (const std::invalid_argument&) { invalid_argument = true; }
        catch (const std::exception&) {}
        try { m.set_current(s2); }
        catch (const std::invalid_argument&) {}
        catch (const std::runtime_error&) { runtime_error = true; }
        catch (const std::exception&) {}
        check(invalid_argument && runtime_error,
              "C3 z = -1 throws std::invalid_argument and z = 2 std::runtime_error");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

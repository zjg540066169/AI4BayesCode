// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_StructuredPottsVI.cpp
//
//  examples/StructuredPottsVI.cpp: set_current. clique_phi is the state of
//  the structured VI block (child 0); z is a shared value.
//    C1  set_current with a valid clique_phi (uniform on each clique) and a
//        z entry outside 0..K-1 throws and leaves every shared value and the
//        block's current value unchanged (bit for bit);
//    C2  set_current of clique_phi and z is reported by get_current().
// ============================================================================
#include <armadillo>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main structured_potts_vi_demo_main
#include "../examples/StructuredPottsVI.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> StructuredPottsVI::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &StructuredPottsVI::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const StructuredPottsVI& m) {
    return *(m.*member_of(impl_tag{}));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_StructuredPottsVI ===\n");
    arma::mat edges(kEdges.size(), 2);
    arma::vec strengths(kEdges.size());
    for (std::size_t e = 0; e < kEdges.size(); ++e) {
        edges(e, 0) = static_cast<double>(kEdges[e].u) + 1.0;
        edges(e, 1) = static_cast<double>(kEdges[e].v) + 1.0;
        strengths[e] = kEdges[e].beta;
    }
    arma::mat h(kN, kK);
    for (std::size_t i = 0; i < kN; ++i)
        for (std::size_t k = 0; k < kK; ++k) h(i, k) = kH[i][k];
    const std::vector<std::vector<int>> cliques = {{1, 2, 3}, {4, 5, 6}};
    StructuredPottsVI m(static_cast<int>(kN), static_cast<int>(kK), edges, strengths, h,
                        cliques, true, 7, false);
    m.step(50);

    const std::size_t S = m.get_current().at("clique_phi").n_elem;   // 2^3 + 2^3
    const arma::vec phi_uniform(S, arma::fill::value(1.0 / 8.0));

    // C1
    const std::map<std::string, arma::vec> shared0(impl(m).data().values().begin(),
                                                   impl(m).data().values().end());
    const arma::vec phi0 = impl(m).child(0).current();
    AI4BayesCode::state_map s1;
    s1["clique_phi"] = phi_uniform;
    arma::vec z_bad(kN, arma::fill::zeros);
    z_bad[kN - 1] = 5.0;
    s1["z"] = z_bad;
    bool threw = false;
    try { m.set_current(s1); } catch (const std::exception&) { threw = true; }
    bool same = threw && same_bits(impl(m).child(0).current(), phi0) &&
                impl(m).data().values().size() == shared0.size();
    for (const auto& kv : shared0) same = same && same_bits(impl(m).data().get(kv.first), kv.second);
    std::printf("C1 set_current(uniform clique_phi, z with entry 5): threw %d, clique_phi "
                "unchanged %d\n", threw ? 1 : 0,
                same_bits(impl(m).child(0).current(), phi0) ? 1 : 0);
    check(same, "C1 a valid clique_phi with an invalid z throws and changes nothing");

    // C2
    AI4BayesCode::state_map s2;
    s2["clique_phi"] = phi_uniform;
    const arma::vec z2{0.0, 1.0, 0.0, 1.0, 1.0, 0.0};
    s2["z"] = z2;
    m.set_current(s2);
    const auto g2 = m.get_current();
    check(same_bits(g2.at("clique_phi"), phi_uniform) && same_bits(g2.at("z"), z2),
          "C2 the loaded clique_phi and z are reported");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_ICARSpatialGMRF.cpp
//
//  examples/ICARSpatialGMRF.cpp: set_current. Intercept, tau and sigma are
//  the state of the joint block (child 1), phi the state of the GMRF block
//  (child 0); y is data.
//    C1  set_current with a valid Intercept and tau and a phi of wrong
//        length throws and leaves every shared value and every block's
//        current value unchanged (bit for bit);
//    C2  the same for a valid phi and a y of wrong length;
//    C3  set_current of Intercept, tau, sigma and phi is reported by
//        get_current(), and the blocks hold the loaded values (the joint
//        block up to rounding, rel. 1e-12: it holds tau and sigma through
//        their logarithms).
// ============================================================================
#include <armadillo>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main icar_spatial_gmrf_demo_main
#include "../examples/ICARSpatialGMRF.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> ICARSpatialGMRF::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &ICARSpatialGMRF::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const ICARSpatialGMRF& m) {
    return *(m.*member_of(impl_tag{}));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

// Every shared value and every block's current value.
struct state_t {
    std::map<std::string, arma::vec> shared;
    std::vector<arma::vec> blocks;
};
state_t state_of(const ICARSpatialGMRF& m) {
    state_t s;
    for (const auto& kv : impl(m).data().values()) s.shared.emplace(kv.first, kv.second);
    for (std::size_t i = 0; i < impl(m).n_children(); ++i)
        s.blocks.push_back(impl(m).child(i).current());
    return s;
}
bool same_state(const state_t& a, const state_t& b) {
    if (a.shared.size() != b.shared.size() || a.blocks.size() != b.blocks.size())
        return false;
    for (const auto& kv : a.shared) {
        auto it = b.shared.find(kv.first);
        if (it == b.shared.end() || !same_bits(kv.second, it->second)) return false;
    }
    for (std::size_t i = 0; i < a.blocks.size(); ++i)
        if (!same_bits(a.blocks[i], b.blocks[i])) return false;
    return true;
}
// set_current(s) throws, and the state is unchanged.
bool throws_unchanged(ICARSpatialGMRF& m, const AI4BayesCode::state_map& s) {
    const state_t before = state_of(m);
    bool threw = false;
    try { m.set_current(s); } catch (const std::exception&) { threw = true; }
    return threw && same_state(before, state_of(m));
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_ICARSpatialGMRF ===\n");
    std::vector<int> ei_v, ej_v;
    int n_nodes = 0;
    build_lattice(3, ei_v, ej_v, n_nodes);
    const std::size_t N = static_cast<std::size_t>(n_nodes);
    arma::vec edges_i(ei_v.size()), edges_j(ej_v.size());
    for (std::size_t e = 0; e < ei_v.size(); ++e) {
        edges_i[e] = static_cast<double>(ei_v[e]);
        edges_j[e] = static_cast<double>(ej_v[e]);
    }
    std::mt19937_64 gen(31u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::vec node(N), y(N);
    for (std::size_t t = 0; t < N; ++t) {
        node[t] = static_cast<double>(t) + 1.0;
        y[t] = 4.0 + 0.3 * static_cast<double>(t % 3) + 0.5 * nd(gen);
    }
    ICARSpatialGMRF m(y, node, n_nodes, edges_i, edges_j, 7, false);
    m.step(3);

    // C1
    AI4BayesCode::state_map s1;
    s1["Intercept"] = arma::vec{2.0};
    s1["tau"] = arma::vec{3.0};
    s1["phi"] = arma::vec(N + 1, arma::fill::zeros);
    check(throws_unchanged(m, s1),
          "C1 a valid Intercept and tau with a phi of wrong length throw and change nothing");

    // C2
    AI4BayesCode::state_map s2;
    s2["phi"] = arma::linspace<arma::vec>(-1.0, 1.0, N);
    s2["y"] = arma::vec(N + 2, arma::fill::ones);
    check(throws_unchanged(m, s2),
          "C2 a valid phi with a y of wrong length throws and changes nothing");

    // C3
    AI4BayesCode::state_map s3;
    s3["Intercept"] = arma::vec{2.0};
    s3["tau"] = arma::vec{3.0};
    s3["sigma"] = arma::vec{0.75};
    s3["phi"] = arma::linspace<arma::vec>(-1.0, 1.0, N);
    m.set_current(s3);
    const auto g3 = m.get_current();
    bool c3 = same_bits(impl(m).child(0).current(), s3.at("phi"));
    for (const auto& kv : s3) c3 = c3 && same_bits(g3.at(kv.first), kv.second);
    // The joint block holds tau and sigma through their logarithms, so its
    // natural values equal the loaded ones up to rounding (rel. 1e-12).
    const arma::vec& jb = impl(m).child(1).current();
    const arma::vec want{2.0, 3.0, 0.75};
    c3 = c3 && jb.n_elem == 3 &&
         arma::max(arma::abs(jb - want) / arma::abs(want)) <= 1e-12;
    check(c3, "C3 the loaded values are reported and held by the blocks");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

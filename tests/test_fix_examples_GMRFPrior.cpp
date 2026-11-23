// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_GMRFPrior.cpp
//
//  examples/GMRFPrior.cpp: set_current of x (the state of the GMRF block,
//  child 0, and its shared copy) and kappa (a shared value of the block's
//  context).
//    C1  set_current with a valid x and kappa = -1 throws and leaves every
//        shared value and the block's current value unchanged (bit for
//        bit);
//    C2  the same for a valid x and a kappa of length 2;
//    C3  set_current of x and kappa is reported by get_current(), and the
//        block holds the loaded x.
// ============================================================================
#include <armadillo>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main gmrf_prior_demo_main
#include "../examples/GMRFPrior.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> GMRFPrior::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &GMRFPrior::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const GMRFPrior& m) {
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
state_t state_of(const GMRFPrior& m) {
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
bool throws_unchanged(GMRFPrior& m, const AI4BayesCode::state_map& s) {
    const state_t before = state_of(m);
    bool threw = false;
    try { m.set_current(s); } catch (const std::exception&) { threw = true; }
    return threw && same_state(before, state_of(m));
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_GMRFPrior ===\n");
    const std::size_t N = 16;
    GMRFPrior m(4, 4, 2.0, false, false, 7, false);
    m.step(3);

    // C1
    AI4BayesCode::state_map s1;
    s1["x"] = arma::linspace<arma::vec>(-1.0, 1.0, N);
    s1["kappa"] = arma::vec{-1.0};
    check(throws_unchanged(m, s1), "C1 a valid x with kappa = -1 throws and changes nothing");

    // C2
    AI4BayesCode::state_map s2;
    s2["x"] = arma::linspace<arma::vec>(1.0, -1.0, N);
    s2["kappa"] = arma::vec{1.0, 2.0};
    check(throws_unchanged(m, s2),
          "C2 a valid x with a kappa of length 2 throws and changes nothing");

    // C3
    AI4BayesCode::state_map s3;
    s3["x"] = arma::linspace<arma::vec>(-1.0, 1.0, N);
    s3["kappa"] = arma::vec{3.0};
    m.set_current(s3);
    const auto g3 = m.get_current();
    check(same_bits(g3.at("x"), s3.at("x")) && same_bits(g3.at("kappa"), s3.at("kappa")) &&
          same_bits(impl(m).child(0).current(), s3.at("x")),
          "C3 the loaded x and kappa are reported and the block holds x");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

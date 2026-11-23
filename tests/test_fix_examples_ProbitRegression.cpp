// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_ProbitRegression.cpp
//
//  examples/ProbitRegression.cpp: set_current. z is the state of the
//  augmentation block (child 0), beta the state of the NUTS block (child 1);
//  mu_lin = X beta and prob = Phi(mu_lin) are recomputed from beta and X.
//    C1  set_current with a valid beta and an X of wrong length throws and
//        leaves every shared value (mu_lin and prob included) and every
//        block's current value unchanged (bit for bit);
//    C2  the same for a valid z and a y with an entry other than 0 and 1;
//    C3  set_current of beta and z is reported by get_current(), the blocks
//        hold the loaded values, and mu_lin = X beta (rel. 1e-12).
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main probit_regression_demo_main
#include "../examples/ProbitRegression.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> ProbitRegression::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &ProbitRegression::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const ProbitRegression& m) {
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
state_t state_of(const ProbitRegression& m) {
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
bool throws_unchanged(ProbitRegression& m, const AI4BayesCode::state_map& s) {
    const state_t before = state_of(m);
    bool threw = false;
    try { m.set_current(s); } catch (const std::exception&) { threw = true; }
    return threw && same_state(before, state_of(m));
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_ProbitRegression ===\n");
    const std::size_t N = 120, p = 3;
    std::mt19937_64 gen(37u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::mat X(N, p, arma::fill::ones);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        X(i, 1) = nd(gen); X(i, 2) = nd(gen);
        y[i] = (0.3 + 1.0 * X(i, 1) - 0.8 * X(i, 2) + nd(gen) > 0.0) ? 1.0 : 0.0;
    }
    ProbitRegression m(X, y, 10.0, 7, false);
    m.step(3);

    // C1
    AI4BayesCode::state_map s1;
    s1["beta"] = arma::vec{0.5, -0.5, 1.0};
    s1["X"] = arma::vec(N * p + 1, arma::fill::ones);
    check(throws_unchanged(m, s1),
          "C1 a valid beta with an X of wrong length throws and changes nothing");

    // C2
    AI4BayesCode::state_map s2;
    s2["z"] = arma::linspace<arma::vec>(-1.0, 1.0, N);
    arma::vec y_bad = y;
    y_bad[0] = 2.0;
    s2["y"] = y_bad;
    check(throws_unchanged(m, s2),
          "C2 a valid z with a y entry outside {0, 1} throws and changes nothing");

    // C3
    AI4BayesCode::state_map s3;
    s3["beta"] = arma::vec{0.5, -0.5, 1.0};
    s3["z"] = arma::linspace<arma::vec>(-1.0, 1.0, N);
    m.set_current(s3);
    const auto g3 = m.get_current();
    const arma::vec mu = X * s3.at("beta");
    const arma::vec& mu_lin = impl(m).data().get("mu_lin");
    const double e_mu = arma::max(arma::abs(mu_lin - mu)) / arma::max(arma::abs(mu));
    std::printf("C3 set_current(beta, z): rel error mu_lin %.3g\n", e_mu);
    check(same_bits(g3.at("beta"), s3.at("beta")) && same_bits(g3.at("z"), s3.at("z")) &&
          same_bits(impl(m).child(1).current(), s3.at("beta")) &&
          same_bits(impl(m).child(0).current(), s3.at("z")) && e_mu <= 1e-12,
          "C3 the loaded beta and z are reported and held by the blocks; mu_lin = X beta");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

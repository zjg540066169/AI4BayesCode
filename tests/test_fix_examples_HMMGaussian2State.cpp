// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_HMMGaussian2State.cpp
//
//  examples/HMMGaussian2State.cpp: the hmm block rounds a loaded z to state
//  labels; get_current() reports the shared "z" and predict_at({}) draws
//  y_rep[t] ~ N(mu[z_t], sigma^2) from it.
//    C1  after set_current(z = 0.6 everywhere), the shared z equals the
//        block's labels (all 1), and y_rep from predict_at({}) is centred on
//        mu[1] = 2: the mean of (y_rep - 2) / sigma lies within 5 / sqrt(T)
//        of 0;
//    C2  set_current with a z entry outside {0, 1} after rounding throws and
//        leaves the block's labels and the shared z unchanged; so does a
//        valid z with a y of the wrong length;
//    C3  the error for a z entry outside {0, 1} after rounding is
//        std::invalid_argument, the exception of hmm_block::set_current.
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

#define main hmm_gaussian_2state_demo_main
#include "../examples/HMMGaussian2State.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> HMMGaussian2State::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &HMMGaussian2State::impl_>;

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

bool throws(HMMGaussian2State& m, const AI4BayesCode::state_map& s) {
    try { m.set_current(s); } catch (const std::exception&) { return true; }
    return false;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_HMMGaussian2State ===\n");
    const std::size_t T = 300;
    std::mt19937_64 g(61u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::vec y(T);
    for (std::size_t t = 0; t < T; ++t) y[t] = ((t / 50) % 2 ? 2.0 : -2.0) + nd(g);
    const arma::vec A{0.92, 0.08, 0.08, 0.92}, pi0{0.5, 0.5}, mu{-2.0, 2.0};
    HMMGaussian2State m(y, A, pi0, mu, 1.0, 7, false);
    m.step(3);
    auto& c = *(m.*member_of(impl_tag{}));

    // C1
    AI4BayesCode::state_map s;
    s["z"] = arma::vec(T, arma::fill::value(0.6));
    m.set_current(s);
    const arma::vec block_z = c.child(0).current();
    const arma::vec shared_z = c.data().get("z");
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    double zs = 0.0;
    for (std::size_t t = 0; t < T; ++t) zs += (yr[t] - 2.0);
    zs /= std::sqrt(static_cast<double>(T));
    std::printf("C1 set_current(z = 0.6): block z[0] %.1f, shared z[0] %.1f, y_rep z-score vs mu[1] %.3f\n",
                block_z[0], shared_z[0], zs);
    check(same_bits(block_z, shared_z) && block_z[0] == 1.0,
          "C1 the shared z equals the block's labels");
    check(std::abs(zs) < 5.0, "C1 predict_at({}) y_rep drawn from the block's labels");

    // C2
    const arma::vec z0 = c.child(0).current(), zs0 = c.data().get("z");
    arma::vec zbad(T, arma::fill::zeros);
    zbad[T - 1] = 3.0;
    AI4BayesCode::state_map b1, b2;
    b1["z"] = zbad;
    b2["z"] = arma::vec(T, arma::fill::zeros);
    b2["y"] = arma::vec(T - 1, arma::fill::zeros);
    const bool t1 = throws(m, b1), t2 = throws(m, b2);
    check(t1 && t2 && same_bits(z0, c.child(0).current()) && same_bits(zs0, c.data().get("z")),
          "C2 a failing set_current leaves the labels and the shared z unchanged");

    {
        bool invalid_argument = false;
        try { m.set_current(b1); }
        catch (const std::invalid_argument&) { invalid_argument = true; }
        catch (const std::exception&) {}
        check(invalid_argument, "C3 a z entry outside {0, 1} throws std::invalid_argument");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

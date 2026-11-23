// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_HierarchicalLM_joint.cpp
//
//  The derived keys "mu_fixed" = alpha + X beta and "u" = tau * z_u of
//  examples/HierarchicalLM_joint.cpp, from which predict_at({}) draws
//  y_rep = mu_fixed + u[g] + sigma * N(0, 1).
//    C1  after construction, both equal the values at the initial state;
//    C2  after one sweep and set_current of alpha, beta and tau, both equal
//        the values at the loaded state, and y_rep is centred on them;
//    C3  set_current(get_current()) after 5 more sweeps returns the same
//        get_current() values, with mu_fixed and u exact.
//  The values are recomputed with loops in this test (relative tolerance
//  1e-12). Centred: the mean of (y_rep - mu_fixed - u[g]) / sigma lies
//  within 5 / sqrt(N) of 0.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main hierarchical_lm_joint_demo_main
#include "../examples/HierarchicalLM_joint.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> HierarchicalLM_joint::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &HierarchicalLM_joint::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

const AI4BayesCode::shared_data_t& data_of(const HierarchicalLM_joint& m) {
    return (m.*member_of(impl_tag{}))->data();
}

double rel_err(const arma::vec& a, const arma::vec& b) {
    if (a.n_elem != b.n_elem) return INFINITY;
    double e = 0.0;
    for (std::size_t i = 0; i < a.n_elem; ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::max(1.0, std::abs(b[i])));
    return e;
}

// alpha + X beta from the current alpha, beta and X.
arma::vec mu_ref(const HierarchicalLM_joint& m) {
    const auto& d = data_of(m);
    const double alpha = d.get("alpha")[0];
    const arma::vec& beta = d.get("beta");
    const arma::vec& X = d.get("X");
    const std::size_t p = beta.n_elem, N = X.n_elem / p;
    arma::vec mu(N);
    for (std::size_t n = 0; n < N; ++n) {
        double xb = 0.0;
        for (std::size_t k = 0; k < p; ++k) xb += X[n + k * N] * beta[k];
        mu[n] = alpha + xb;
    }
    return mu;
}

arma::vec u_ref(const HierarchicalLM_joint& m) {
    const auto& d = data_of(m);
    return d.get("tau")[0] * d.get("z_u");
}

double derived_error(const HierarchicalLM_joint& m) {
    const auto& d = data_of(m);
    return std::max(rel_err(d.get("mu_fixed"), mu_ref(m)), rel_err(d.get("u"), u_ref(m)));
}

double yrep_z(const HierarchicalLM_joint& m) {
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    const arma::vec mu = mu_ref(m), u = u_ref(m);
    const arma::vec& g = data_of(m).get("g_idx");
    const double sigma = data_of(m).get("sigma")[0];
    double s = 0.0;
    for (std::size_t n = 0; n < mu.n_elem; ++n)
        s += (yr[n] - mu[n] - u[static_cast<std::size_t>(g[n])]) / sigma;
    return s / std::sqrt(static_cast<double>(mu.n_elem));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_HierarchicalLM_joint ===\n");
    const std::size_t N = 200, p = 2;
    const int G = 8;
    std::mt19937_64 gen(17u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::mat X(N, p);
    arma::vec gidx(N), y(N);
    arma::vec u_true(G);
    for (int g = 0; g < G; ++g) u_true[g] = 0.7 * nd(gen);
    for (std::size_t n = 0; n < N; ++n) {
        X(n, 0) = nd(gen); X(n, 1) = nd(gen);
        gidx[n] = static_cast<double>(1 + n % G);
        y[n] = 1.0 + 0.8 * X(n, 0) - 0.5 * X(n, 1) + u_true[n % G] + 0.4 * nd(gen);
    }

    HierarchicalLM_joint m(y, X, gidx, G, 1.0, 1.0, 7, false);
    std::printf("C1 construction: max rel error of mu_fixed, u = %.3g\n", derived_error(m));
    check(derived_error(m) <= 1e-12, "C1 mu_fixed and u equal the values at the initial state");

    m.step(1);
    AI4BayesCode::state_map s;
    s["alpha"] = arma::vec{-2.0};
    s["beta"]  = arma::vec{3.0, -4.0};
    s["tau"]   = arma::vec{2.5};
    m.set_current(s);
    std::printf("C2 set_current(alpha, beta, tau): max rel error of mu_fixed, u = %.3g, "
                "y_rep z = %.3f\n", derived_error(m), yrep_z(m));
    check(derived_error(m) <= 1e-12, "C2 mu_fixed and u equal the values at the loaded state");
    check(std::abs(yrep_z(m)) < 5.0, "C2 predict_at({}) y_rep centred on the loaded state");

    m.step(5);
    const auto before = m.get_current();
    m.set_current(before);
    const auto after = m.get_current();
    bool same = true;
    for (const char* k : {"alpha", "beta", "z_u", "u", "sigma", "tau"})
        same = same && same_bits(before.at(k), after.at(k));
    check(same && derived_error(m) <= 1e-12,
          "C3 set_current(get_current()) leaves the state unchanged and mu_fixed, u exact");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

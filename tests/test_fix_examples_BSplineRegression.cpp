// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_BSplineRegression.cpp
//
//  examples/BSplineRegression.cpp: the derived keys s = exp(log_sds) z,
//  f = Bsp s and mu = Intercept + f, and the loader set_current.
//    C1  after construction, s, f and mu equal the values at the initial
//        state, and y_rep from predict_at({}) is centred on mu;
//    C2  after one sweep, set_current of Intercept, log_sds and z loads the
//        values into the joint block (its current value and the fallback
//        rows of get_history() equal the loaded values) and s, f and mu equal
//        the values at the loaded state;
//    C3  the next sweep starts from the loaded state: it equals the sweep of
//        a second model with the same seed and history whose block was
//        loaded with the same values through composite_block::set_current;
//    C4  a set_current call with a valid Intercept and a z of the wrong
//        length throws and leaves the state unchanged;
//    C5  set_current(get_current()) returns the same get_current() values,
//        with s, f and mu exact.
//  The derived values are recomputed with loops in this test (relative
//  tolerance 1e-12). Centred: the mean of (y_rep - mu) / sigma lies within
//  5 / sqrt(N) of 0.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main bspline_regression_demo_main
#include "../examples/BSplineRegression.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> BSplineRegression::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &BSplineRegression::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const BSplineRegression& m) {
    return *(m.*member_of(impl_tag{}));
}

double rel_err(const arma::vec& a, const arma::vec& b) {
    if (a.n_elem != b.n_elem) return INFINITY;
    double e = 0.0;
    for (std::size_t i = 0; i < a.n_elem; ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::max(1.0, std::abs(b[i])));
    return e;
}

// mu = Intercept + Bsp (exp(log_sds) z) from the current shared values.
arma::vec mu_ref(const BSplineRegression& m) {
    const auto& d = impl(m).data();
    const arma::vec& B = d.get("Bsp");
    const arma::vec& z = d.get("z");
    const double sds = std::exp(d.get("log_sds")[0]);
    const std::size_t K = z.n_elem, N = B.n_elem / K;
    arma::vec mu(N);
    for (std::size_t n = 0; n < N; ++n) {
        double f = 0.0;
        for (std::size_t k = 0; k < K; ++k) f += sds * z[k] * B[n + k * N];
        mu[n] = d.get("Intercept")[0] + f;
    }
    return mu;
}

double derived_error(const BSplineRegression& m) {
    const auto& d = impl(m).data();
    const arma::vec s = std::exp(d.get("log_sds")[0]) * d.get("z");
    const arma::vec mu = mu_ref(m);
    return std::max(std::max(rel_err(d.get("s"), s), rel_err(d.get("mu"), mu)),
                    rel_err(d.get("f"), mu - d.get("Intercept")[0]));
}

double yrep_z(const BSplineRegression& m) {
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    const arma::vec mu = mu_ref(m);
    const double sigma = std::exp(impl(m).data().get("log_sigma")[0]);
    double s = 0.0;
    for (std::size_t n = 0; n < mu.n_elem; ++n) s += (yr[n] - mu[n]) / sigma;
    return s / std::sqrt(static_cast<double>(mu.n_elem));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_BSplineRegression ===\n");
    const std::size_t N = 150, K = 6;
    std::mt19937_64 gen(29u);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    arma::mat B(N, K);
    arma::vec y(N);
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t k = 0; k < K; ++k) B(n, k) = ud(gen);
        y[n] = 3.0 + std::sin(6.0 * B(n, 0)) + 0.2 * nd(gen);
    }

    BSplineRegression m(y, B, 7, false);
    std::printf("C1 construction: max rel error of s, f, mu = %.3g, y_rep z = %.3f\n",
                derived_error(m), yrep_z(m));
    check(derived_error(m) <= 1e-12, "C1 s, f and mu equal the values at the initial state");
    check(std::abs(yrep_z(m)) < 5.0, "C1 predict_at({}) y_rep centred on mu");

    m.step(1);
    AI4BayesCode::state_map s;
    s["Intercept"] = arma::vec{-4.0};
    s["log_sds"]   = arma::vec{0.3};
    s["z"]         = arma::vec{1.0, -1.0, 0.5, 2.0, -0.5, 0.25};
    m.set_current(s);
    const arma::vec cur = impl(m).child(0).current();
    const auto hist = m.get_history();
    const bool loaded = cur[0] == -4.0 && cur[1] == 0.3 &&
                        same_bits(cur.subvec(3, 3 + K - 1), s.at("z")) &&
                        hist.at("Intercept")(0, 0) == -4.0 &&
                        hist.at("log_sds")(0, 0) == 0.3;
    std::printf("C2 set_current: block Intercept %.3f log_sds %.3f, history row "
                "Intercept %.3f; max rel error of s, f, mu = %.3g\n",
                cur[0], cur[1], hist.at("Intercept")(0, 0), derived_error(m));
    check(loaded, "C2 the joint block and get_history() hold the loaded values");
    check(derived_error(m) <= 1e-12, "C2 s, f and mu equal the values at the loaded state");

    // C3: a twin with the same seed and history, loaded through
    // composite_block::set_current (the per-child loader of the framework).
    BSplineRegression t(y, B, 7, false);
    t.step(1);
    {
        AI4BayesCode::composite_block& c = impl(t);
        arma::vec theta = c.child(0).current();
        theta[0] = -4.0; theta[1] = 0.3;
        theta.subvec(3, 3 + K - 1) = s.at("z");
        c.set_current(theta);
    }
    m.step(1); t.step(1);
    const auto cm = m.get_current(), ct = t.get_current();
    bool same = true;
    for (const char* k : {"Intercept", "log_sds", "log_sigma", "z"})
        same = same && same_bits(cm.at(k), ct.at(k));
    check(same, "C3 the next sweep starts from the loaded state");

    // C4
    const auto before = m.get_current();
    const arma::vec mu0 = impl(m).data().get("mu");
    AI4BayesCode::state_map bad;
    bad["Intercept"] = arma::vec{9.0};
    bad["z"]         = arma::vec{1.0, 2.0};
    bool threw = false;
    try { m.set_current(bad); } catch (const std::exception&) { threw = true; }
    const auto after = m.get_current();
    const bool unchanged = same_bits(before.at("Intercept"), after.at("Intercept")) &&
                           same_bits(before.at("z"), after.at("z")) &&
                           impl(m).child(0).current()[0] == before.at("Intercept")[0] &&
                           same_bits(mu0, impl(m).data().get("mu"));
    std::printf("C4 failing set_current: threw = %d, state unchanged = %d\n",
                threw ? 1 : 0, unchanged ? 1 : 0);
    check(threw && unchanged, "C4 a failing set_current leaves the state unchanged");

    // C5
    m.step(3);
    const auto b5 = m.get_current();
    m.set_current(b5);
    const auto a5 = m.get_current();
    bool same5 = true;
    for (const char* k : {"Intercept", "log_sds", "log_sigma", "z", "sds", "sigma"})
        same5 = same5 && same_bits(b5.at(k), a5.at(k));
    check(same5 && derived_error(m) <= 1e-12,
          "C5 set_current(get_current()) leaves the state unchanged and s, f, mu exact");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

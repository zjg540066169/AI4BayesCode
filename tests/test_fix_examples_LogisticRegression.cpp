// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_LogisticRegression.cpp
//
//  The derived keys "eta" = X beta and "prob" = 1 / (1 + exp(-eta)) of
//  examples/LogisticRegression.cpp, which predict_at({}) uses to draw y_rep.
//    C1  after construction, "eta" and "prob" equal the values computed from
//        the initial beta and X;
//    C2  after one sweep and set_current of beta, they equal the values at
//        the loaded beta,
//        and the mean of y_rep from predict_at({}) matches sum(prob);
//    C3  after set_current of X (and y) with another row count, they have the
//        new length and equal the values at the new X;
//    C4  a set_current call that fails on y (valid beta, y not 0/1) leaves
//        beta, X, "eta" and "prob" unchanged;
//    C5  set_current(get_current()) after 5 more sweeps returns the same
//        beta, with "eta" and "prob" exact.
//  "eta" and "prob" are compared with loops in this test (relative tolerance
//  1e-12). The y_rep check uses z = (sum(y_rep) - sum(prob)) /
//  sqrt(sum(prob (1 - prob))) and requires |z| < 5.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main logistic_regression_demo_main
#include "../examples/LogisticRegression.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> LogisticRegression::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &LogisticRegression::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

const AI4BayesCode::shared_data_t& data_of(const LogisticRegression& m) {
    return (m.*member_of(impl_tag{}))->data();
}

// eta = X beta from the current beta and X (loop); prob = 1 / (1 + exp(-eta)).
arma::vec eta_from_inputs(const LogisticRegression& m) {
    const auto& d = data_of(m);
    const arma::vec& beta = d.get("beta");
    const arma::vec& X = d.get("X");
    const std::size_t p = beta.n_elem, N = X.n_elem / p;
    arma::vec eta(N);
    for (std::size_t i = 0; i < N; ++i) {
        double xb = 0.0;
        for (std::size_t j = 0; j < p; ++j) xb += X[i + j * N] * beta[j];
        eta[i] = xb;
    }
    return eta;
}

arma::vec prob_from_inputs(const LogisticRegression& m) {
    const arma::vec eta = eta_from_inputs(m);
    arma::vec pr(eta.n_elem);
    for (std::size_t i = 0; i < eta.n_elem; ++i) pr[i] = 1.0 / (1.0 + std::exp(-eta[i]));
    return pr;
}

double rel_err(const arma::vec& a, const arma::vec& b) {
    if (a.n_elem != b.n_elem) return INFINITY;
    double e = 0.0;
    for (std::size_t i = 0; i < a.n_elem; ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::max(1.0, std::abs(b[i])));
    return e;
}

// Largest relative deviation of "eta" and "prob" from the values at the
// current beta and X; +inf on a length mismatch.
double prob_error(const LogisticRegression& m) {
    const auto& d = data_of(m);
    return std::max(rel_err(d.get("eta"), eta_from_inputs(m)),
                    rel_err(d.get("prob"), prob_from_inputs(m)));
}

double yrep_z(const LogisticRegression& m) {
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    const arma::vec pr = prob_from_inputs(m);
    if (yr.n_elem != pr.n_elem) return INFINITY;
    double s = 0.0, v = 0.0;
    for (std::size_t i = 0; i < pr.n_elem; ++i) { s += yr[i] - pr[i]; v += pr[i] * (1.0 - pr[i]); }
    return s / std::sqrt(v);
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_LogisticRegression ===\n");
    const std::size_t N = 400, p = 3;
    std::mt19937_64 gen(5u);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    arma::mat X(N, p);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        X(i, 0) = 1.0; X(i, 1) = nd(gen); X(i, 2) = nd(gen);
        const double eta = -0.5 + 1.2 * X(i, 1) - 0.8 * X(i, 2);
        y[i] = (ud(gen) < 1.0 / (1.0 + std::exp(-eta))) ? 1.0 : 0.0;
    }

    LogisticRegression m(X, y, 10.0, 7, false);

    std::printf("C1 construction: max rel error of eta/prob = %.3g\n", prob_error(m));
    check(prob_error(m) <= 1e-12, "C1 eta and prob equal the values at the initial beta");

    m.step(1);   // C2 starts from a state written by a sweep
    AI4BayesCode::state_map s;
    s["beta"] = arma::vec{2.0, -1.5, 3.0};
    m.set_current(s);
    std::printf("C2 set_current(beta): max rel error of eta/prob = %.3g, y_rep z = %.3f\n",
                prob_error(m), yrep_z(m));
    check(prob_error(m) <= 1e-12, "C2 eta and prob equal the values at the loaded beta");
    check(std::abs(yrep_z(m)) < 5.0, "C2 predict_at({}) y_rep drawn from the loaded beta");

    AI4BayesCode::state_map s3;
    s3["X"] = arma::vec(arma::vectorise(arma::mat(X.rows(0, 149))));
    s3["y"] = arma::vec(y.subvec(0, 149));
    m.set_current(s3);
    std::printf("C3 set_current(X with 150 rows): length(prob) = %u, max rel error = %.3g\n",
                static_cast<unsigned>(data_of(m).get("prob").n_elem), prob_error(m));
    check(data_of(m).get("prob").n_elem == 150 && prob_error(m) <= 1e-12,
          "C3 eta and prob have the new length and equal the values at the new X");

    const arma::vec b0 = data_of(m).get("beta"), X0 = data_of(m).get("X");
    const arma::vec e0 = data_of(m).get("eta"), p0 = data_of(m).get("prob");
    AI4BayesCode::state_map bad;
    bad["beta"] = arma::vec{0.1, 0.2, 0.3};
    bad["y"]    = arma::vec(150, arma::fill::value(2.0));
    bool threw = false;
    try { m.set_current(bad); } catch (const std::exception&) { threw = true; }
    const bool unchanged = same_bits(b0, data_of(m).get("beta")) &&
                           same_bits(X0, data_of(m).get("X")) &&
                           same_bits(e0, data_of(m).get("eta")) &&
                           same_bits(p0, data_of(m).get("prob")) &&
                           same_bits(b0, m.get_history().at("beta").row(0).t());
    std::printf("C4 failing set_current: threw = %d, state unchanged = %d\n",
                threw ? 1 : 0, unchanged ? 1 : 0);
    check(threw && unchanged, "C4 a failing set_current leaves the state unchanged");

    m.step(5);
    const auto before = m.get_current();
    m.set_current(before);
    check(same_bits(before.at("beta"), m.get_current().at("beta")) && prob_error(m) <= 1e-12,
          "C5 set_current(get_current()) leaves beta unchanged and eta, prob exact");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

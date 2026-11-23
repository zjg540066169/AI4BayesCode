// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_LinearRegJointMixed.cpp
//
//  The derived key "mu" = alpha + X beta of examples/LinearRegJointMixed.cpp
//  and the posterior predictive y_rep that predict_at({}) draws from it.
//    C1  after construction, "mu" equals alpha + X beta of the initial state,
//        and y_rep from predict_at({}) is centred on it;
//    C2  after one sweep and set_current of alpha and beta, the same holds
//        for the loaded values;
//    C3  after set_current of X with another row count, "mu" has the new
//        length and equals alpha + X beta at the new X;
//    C4  a set_current call that fails on X (valid beta, X length not a
//        multiple of p) leaves alpha, beta, X and "mu" unchanged;
//    C5  set_current(get_current()) after 5 more sweeps returns the same
//        get_current() values, with "mu" exact;
//    C6  set_current with an empty alpha, or an empty sigma, throws and
//        leaves get_current() unchanged.
//  "mu" is compared with a loop in this test (relative tolerance 1e-12).
//  y_rep is centred when the mean of (y_rep - mu) / sigma lies within
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

#define main linreg_joint_mixed_demo_main
#include "../examples/LinearRegJointMixed.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> LinearRegJointMixed::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &LinearRegJointMixed::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

const AI4BayesCode::shared_data_t& data_of(const LinearRegJointMixed& m) {
    return (m.*member_of(impl_tag{}))->data();
}

arma::vec mu_from_inputs(const LinearRegJointMixed& m) {
    const auto& d = data_of(m);
    const double alpha = d.get("alpha")[0];
    const arma::vec& beta = d.get("beta");
    const arma::vec& X = d.get("X");
    const std::size_t p = beta.n_elem, N = X.n_elem / p;
    arma::vec mu(N);
    for (std::size_t i = 0; i < N; ++i) {
        double xb = 0.0;
        for (std::size_t j = 0; j < p; ++j) xb += X[i + j * N] * beta[j];
        mu[i] = alpha + xb;
    }
    return mu;
}

// Largest relative deviation of "mu" from alpha + X beta; +inf on a length
// mismatch.
double mu_error(const LinearRegJointMixed& m) {
    const arma::vec& mu = data_of(m).get("mu");
    const arma::vec ref = mu_from_inputs(m);
    if (mu.n_elem != ref.n_elem) return INFINITY;
    double e = 0.0;
    for (std::size_t i = 0; i < mu.n_elem; ++i)
        e = std::max(e, std::abs(mu[i] - ref[i]) / std::max(1.0, std::abs(ref[i])));
    return e;
}

// Mean of (y_rep - mu_ref) / sigma for one predict_at({}) draw, times sqrt(N).
double yrep_z(const LinearRegJointMixed& m) {
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    const arma::vec ref = mu_from_inputs(m);
    const double sigma = data_of(m).get("sigma")[0];
    if (yr.n_elem != ref.n_elem) return INFINITY;
    double s = 0.0;
    for (std::size_t i = 0; i < ref.n_elem; ++i) s += (yr[i] - ref[i]) / sigma;
    return s / std::sqrt(static_cast<double>(ref.n_elem));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

}  // namespace

int main() {
    // Unbuffered, so the lines printed before a crash reach the log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== test_fix_examples_LinearRegJointMixed ===\n");
    const std::size_t N = 120, p = 3;
    std::mt19937_64 gen(11u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::mat X(N, p);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = nd(gen);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i)
        y[i] = 4.0 + 1.5 * X(i, 0) - 2.0 * X(i, 1) + 0.5 * X(i, 2) + 0.3 * nd(gen);

    LinearRegJointMixed m(y, X, 7, false);

    // C1
    std::printf("C1 construction: max rel |mu - (alpha + X beta)| = %.3g, "
                "y_rep z = %.3f\n", mu_error(m), yrep_z(m));
    check(mu_error(m) <= 1e-12, "C1 mu equals alpha + X beta after construction");
    check(std::abs(yrep_z(m)) < 5.0, "C1 predict_at({}) y_rep centred on alpha + X beta");

    // C2, from a state written by a sweep
    m.step(1);
    AI4BayesCode::state_map s;
    s["alpha"] = arma::vec{-3.0};
    s["beta"]  = arma::vec{0.25, 4.0, -1.0};
    m.set_current(s);
    std::printf("C2 set_current(alpha, beta): max rel |mu - (alpha + X beta)| = %.3g, "
                "y_rep z = %.3f\n", mu_error(m), yrep_z(m));
    check(mu_error(m) <= 1e-12, "C2 mu equals alpha + X beta after set_current");
    check(std::abs(yrep_z(m)) < 5.0, "C2 predict_at({}) y_rep centred after set_current");

    // C3
    arma::mat X2 = X.rows(0, 59) * 2.0;
    AI4BayesCode::state_map s3;
    s3["X"] = arma::vec(arma::vectorise(X2));
    s3["y"] = arma::vec(y.subvec(0, 59));
    m.set_current(s3);
    const arma::vec& mu3 = data_of(m).get("mu");
    std::printf("C3 set_current(X with 60 rows): length(mu) = %u, "
                "max rel |mu - (alpha + X beta)| = %.3g\n",
                static_cast<unsigned>(mu3.n_elem), mu_error(m));
    check(mu3.n_elem == 60 && mu_error(m) <= 1e-12,
          "C3 mu has the new length and equals alpha + X beta at the new X");

    // C4
    const arma::vec a0 = data_of(m).get("alpha"), b0 = data_of(m).get("beta");
    const arma::vec X0 = data_of(m).get("X"), mu0 = data_of(m).get("mu");
    AI4BayesCode::state_map bad;
    bad["beta"] = arma::vec{9.0, 9.0, 9.0};
    bad["X"]    = arma::vec(7, arma::fill::ones);   // 7 is not a multiple of p = 3
    bool threw = false;
    try { m.set_current(bad); } catch (const std::exception&) { threw = true; }
    const bool unchanged = same_bits(a0, data_of(m).get("alpha")) &&
                           same_bits(b0, data_of(m).get("beta")) &&
                           same_bits(X0, data_of(m).get("X")) &&
                           same_bits(mu0, data_of(m).get("mu"));
    std::printf("C4 failing set_current: threw = %d, state unchanged = %d\n",
                threw ? 1 : 0, unchanged ? 1 : 0);
    check(threw && unchanged, "C4 a failing set_current leaves the state unchanged");

    // C5
    m.step(5);
    const auto before = m.get_current();
    m.set_current(before);
    const auto after = m.get_current();
    const bool same = same_bits(before.at("alpha"), after.at("alpha")) &&
                      same_bits(before.at("beta"), after.at("beta")) &&
                      same_bits(before.at("sigma"), after.at("sigma"));
    check(same && mu_error(m) <= 1e-12,
          "C5 set_current(get_current()) leaves the state unchanged and mu exact");

    {
        LinearRegJointMixed me(y, X, 17, false);
        me.step(2);
        bool ok = true;
        for (const char* key : {"alpha", "sigma"}) {
            const auto before = me.get_current();
            AI4BayesCode::state_map se;
            se[key] = arma::vec();
            bool threw = false;
            try { me.set_current(se); } catch (const std::exception&) { threw = true; }
            const auto after = me.get_current();
            ok = ok && threw && after.size() == before.size();
            for (const auto& kv : before)
                ok = ok && after.count(kv.first) && same_bits(kv.second, after.at(kv.first));
        }
        check(ok, "C6 set_current with an empty alpha or sigma throws and leaves get_current() unchanged");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

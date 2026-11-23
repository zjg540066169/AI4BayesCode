// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_GPRegression.cpp
//
//  examples/GPRegression.cpp: the derived keys K_matrix, L_chol and f_mean
//  (invalidated under the block name "hyper") and the K_matrix refresher.
//    C1  after one sweep and set_current of amplitude, lengthscale and
//        sigma, K_matrix equals the squared-exponential kernel
//        amp^2 exp(-|x_i - x_j|^2 / (2 ell^2)) at the loaded values, L_chol
//        is the Cholesky factor of K + (sigma^2 + 1e-8) I and f_mean equals
//        K (K + (sigma^2 + 1e-8) I)^-1 y (relative tolerance 1e-10);
//    C2  after one sweep, set_current(amplitude = 1e200) and
//        set_current(amplitude = +Inf), at which K + (sigma^2 + jitter) I has
//        no Cholesky factor: either the call throws and leaves every shared
//        value and the joint block's current value unchanged (bit for bit),
//        or K_matrix, L_chol and f_mean are the values at the loaded
//        hyperparameters (as in C1);
//    C3  the same for set_current(sigma = 1e200), at which the solve for
//        f_mean fails;
//    C4  a model moved into another object (move construction; the
//        moved-from source is not destroyed) gives the same draws over 3
//        sweeps as a twin with the same seed that is not moved;
//    C5  set_current with an empty amplitude, lengthscale or sigma throws and
//        leaves get_current() unchanged.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main gp_regression_demo_main
#include "../examples/GPRegression.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> GPRegression::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &GPRegression::impl_>;

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

double max_rel(const arma::vec& a, const arma::vec& b) {
    if (a.n_elem != b.n_elem) return INFINITY;
    double e = 0.0;
    for (std::size_t i = 0; i < a.n_elem; ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::max(1.0, std::abs(b[i])));
    return e;
}

// K_matrix, L_chol and f_mean of the model equal the values at (amp, ell,
// sig) within relative tolerance 1e-10; false when a reference value cannot be
// computed.
bool derived_at(const AI4BayesCode::shared_data_t& d, const arma::mat& X,
                const arma::vec& y, double amp, double ell, double sig) {
    const std::size_t N = X.n_rows;
    arma::mat K(N, N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            const double r = X(i, 0) - X(j, 0);
            K(i, j) = amp * amp * std::exp(-0.5 * r * r / (ell * ell));
        }
    arma::mat Ky = K;
    Ky.diag() += sig * sig + 1e-8;
    arma::mat L;
    if (!K.is_finite() || !arma::chol(L, Ky, "lower")) return false;
    arma::vec fm;
    if (!arma::solve(fm, Ky, y)) return false;
    fm = K * fm;
    return max_rel(d.get("K_matrix"), arma::vectorise(K)) <= 1e-10 &&
           max_rel(d.get("L_chol"), arma::vectorise(L)) <= 1e-10 &&
           max_rel(d.get("f_mean"), fm) <= 1e-10;
}

}  // namespace

int main() {
    // Unbuffered, so the lines printed before a crash of a moved model reach
    // the log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== test_fix_examples_GPRegression ===\n");
    const std::size_t N = 40;
    std::mt19937_64 g(73u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::mat X(N, 1);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        X(i, 0) = -3.0 + 6.0 * static_cast<double>(i) / (N - 1.0);
        y[i] = std::sin(X(i, 0)) + 0.2 * nd(g);
    }

    GPRegression m(X, y, 7, false);
    m.step(1);
    AI4BayesCode::state_map s;
    s["amplitude"] = arma::vec{1.7}; s["lengthscale"] = arma::vec{0.6}; s["sigma"] = arma::vec{0.3};
    m.set_current(s);
    const auto& d = (m.*member_of(impl_tag{}))->data();
    arma::mat K(N, N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            const double r = X(i, 0) - X(j, 0);
            K(i, j) = 1.7 * 1.7 * std::exp(-0.5 * r * r / (0.6 * 0.6));
        }
    arma::mat Ky = K;
    Ky.diag() += 0.3 * 0.3 + 1e-8;
    const arma::mat L = arma::chol(Ky, "lower");
    const arma::vec fm = K * arma::solve(Ky, y);
    const double eK = max_rel(d.get("K_matrix"), arma::vectorise(K));
    const double eL = max_rel(d.get("L_chol"), arma::vectorise(L));
    const double eF = max_rel(d.get("f_mean"), fm);
    std::printf("C1 set_current: max rel error K_matrix %.3g, L_chol %.3g, f_mean %.3g\n", eK, eL, eF);
    check(eK <= 1e-10 && eL <= 1e-10 && eF <= 1e-10,
          "C1 K_matrix, L_chol and f_mean follow the loaded hyperparameters");

    struct load_case { const char* label; const char* key; double value; };
    const load_case cases[] = {
        {"C2 set_current(amplitude = 1e200)", "amplitude", 1e200},
        {"C2 set_current(amplitude = Inf)", "amplitude", std::numeric_limits<double>::infinity()},
        {"C3 set_current(sigma = 1e200)", "sigma", 1e200}};
    for (const load_case& lc : cases) {
        GPRegression m2(X, y, 7, false);
        m2.step(1);
        const AI4BayesCode::composite_block& c = *(m2.*member_of(impl_tag{}));
        const auto values0 = c.data().values();
        const arma::vec block0 = c.child(0).current();
        AI4BayesCode::state_map s2;
        s2[lc.key] = arma::vec{lc.value};
        bool threw = false;
        try { m2.set_current(s2); } catch (const std::exception&) { threw = true; }
        bool ok = false;
        if (threw) {
            ok = c.data().values().size() == values0.size() &&
                 same_bits(block0, c.child(0).current());
            for (const auto& kv : values0)
                ok = ok && c.data().has(kv.first) &&
                     same_bits(kv.second, c.data().get(kv.first));
        } else {
            const auto cur = m2.get_current();
            ok = derived_at(c.data(), X, y, cur.at("amplitude")[0],
                            cur.at("lengthscale")[0], cur.at("sigma")[0]);
        }
        std::printf("%s: threw = %d, %s\n", lc.label, threw ? 1 : 0,
                    threw ? (ok ? "state unchanged" : "state changed")
                          : (ok ? "derived keys at the loaded values"
                                : "derived keys not at the loaded values"));
        check(ok, std::string(lc.label) +
                  ": unchanged after the error, or derived keys at the loaded values");
    }

    GPRegression twin(X, y, 9, false);
    GPRegression src(X, y, 9, false);
    GPRegression moved(std::move(src));
    twin.step(3);
    moved.step(3);
    const auto ct = twin.get_current(), cm = moved.get_current();
    check(same_bits(ct.at("amplitude"), cm.at("amplitude")) &&
          same_bits(ct.at("lengthscale"), cm.at("lengthscale")) &&
          same_bits(ct.at("sigma"), cm.at("sigma")) &&
          same_bits((moved.*member_of(impl_tag{}))->data().get("K_matrix"),
                    (twin.*member_of(impl_tag{}))->data().get("K_matrix")),
          "C4 a moved model gives the draws of a twin that is not moved");

    {
        GPRegression me(X, y, 17, false);
        me.step(1);
        bool ok = true;
        for (const char* key : {"amplitude", "lengthscale", "sigma"}) {
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
        check(ok, "C5 set_current with an empty amplitude, lengthscale or sigma throws and leaves get_current() unchanged");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_GBartHeteroscedastic.cpp
//
//  examples/GBartHeteroscedastic.cpp stores the forest fit r at the training X
//  and "mean_r" = exp(r) in the shared data; get_current() reports r and
//  predict_at({}) draws y_rep ~ N(mean_r, phi mean_r).
//    C1  after construction, the shared r is the forest's fit and mean_r
//        equals exp(fit);
//    C2  after 20 sweeps and set_current of a new X (same N): get_current()
//        reports the forest's fit at the new X (= predict_at({X: new X}) of
//        the same forest), the shared mean_r equals exp(fit), and y_rep from
//        predict_at({}) is drawn around exp(fit at the new X): with
//        w = exp(new fit) - exp(old fit), T = sum w (y_rep - exp(new fit))
//        has |T| / sd(T) < 5 (a draw from the old fit gives
//        E[T] = -sum w^2);
//    C3  after set_tree of the forest of a second model on the same training
//        X, r equals that model's r (relative tolerance 1e-12) and mean_r
//        equals exp(fit);
//    C4  set_current with an X whose length is not a multiple of p throws
//        and leaves r, mean_r and X unchanged;
//    C5  with keep_history, the history-mode predict_at({}) draws y_rep of
//        draw d with the phi of sweep d (phi_d from get_current() after
//        that sweep): v_d = mean_i (y_rep - m)^2 / m over the N training
//        points satisfies |v_d / phi_d - 1| < 6 sqrt(2 / N) for every d.
//  The forest's fit is the block's own current value; mean_r is compared
//  with exp(fit) computed in this test (relative tolerance 1e-12).
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main gbart_heteroscedastic_demo_main
#include "../examples/GBartHeteroscedastic.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> GBartHeteroscedastic::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &GBartHeteroscedastic::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const GBartHeteroscedastic& m) {
    return *(m.*member_of(impl_tag{}));
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

arma::vec rate_of(const arma::vec& r) {
    arma::vec p(r.n_elem);
    for (std::size_t i = 0; i < r.n_elem; ++i) p[i] = std::exp(r[i]);
    return p;
}

// The forest's fit at the training X (the block's own value).
arma::vec fit(const GBartHeteroscedastic& m) { return impl(m).child(0).current(); }

// Shared r equals the fit and shared mean_r equals exp(fit).
bool shared_exact(const GBartHeteroscedastic& m) {
    const auto& d = impl(m).data();
    return same_bits(d.get("r"), fit(m)) && max_rel(d.get("mean_r"), rate_of(fit(m))) <= 1e-12;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_GBartHeteroscedastic ===\n");
    bart_rng::set_seed(20261123ULL);
    const std::size_t N = 300, p = 2;
    std::mt19937_64 gen(31u);
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::mat X(N, p), X2(N, p);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < p; ++j) { X(i, j) = unif(gen); X2(i, j) = unif(gen); }
        const double m = std::exp(1.0 + 0.8 * std::max(X(i, 0), 0.0) + 0.6 * X(i, 1));
        y[i] = m + std::sqrt(0.5 * m) * nd(gen);
    }

    GBartHeteroscedastic m(X, y, 20, 1.0, 7, false, false);
    check(shared_exact(m), "C1 r is the forest's fit and mean_r = exp(fit) after construction");

    m.step(20);
    const arma::vec p_old = rate_of(m.get_current().at("r"));
    AI4BayesCode::state_map s;
    s["X"] = arma::vec(arma::vectorise(X2));
    m.set_current(s);
    AI4BayesCode::state_map at_new;
    at_new["X"] = arma::vec(arma::vectorise(X2));
    const arma::vec r_new = arma::vectorise(m.predict_at(at_new).at("r"));
    const arma::vec r_cur = m.get_current().at("r");
    const arma::vec p_new = rate_of(r_new);
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    double T = 0.0, V = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double w = p_new[i] - p_old[i];
        T += w * (yr[i] - p_new[i]);
        V += w * w * m.get_current().at("phi")[0] * p_new[i];
    }
    std::printf("C2 set_current(X): max |get_current r - fit at new X| = %.3g, "
                "y_rep T/sd = %.3f\n", arma::abs(r_cur - r_new).max(), T / std::sqrt(V));
    check(same_bits(r_cur, r_new), "C2 get_current() reports the forest's fit at the new X");
    check(shared_exact(m), "C2 shared r is the fit and mean_r = exp(fit) after set_current(X)");
    check(std::abs(T / std::sqrt(V)) < 5.0, "C2 predict_at({}) y_rep drawn from the new fit");

    // C3: two models on the same training X; the first loads the forest of
    // the second.
    GBartHeteroscedastic m3(X, y, 20, 1.0, 7, false, false), other(X, y, 20, 1.0, 9, false, false);
    m3.step(10);
    other.step(15);
    m3.set_tree(other.get_tree());
    std::printf("C3 set_tree: max rel |r - other r| = %.3g\n",
                max_rel(m3.get_current().at("r"), other.get_current().at("r")));
    check(max_rel(m3.get_current().at("r"), other.get_current().at("r")) <= 1e-12 &&
          shared_exact(m3),
          "C3 set_tree loads the forest's r and mean_r = exp(fit)");

    const auto& d = impl(m).data();
    const arma::vec r0 = d.get("r"), p0 = d.get("mean_r"), X0 = d.get("X");
    AI4BayesCode::state_map bad;
    bad["X"] = arma::vec(N * p + 1, arma::fill::zeros);
    bool threw = false;
    try { m.set_current(bad); } catch (const std::exception&) { threw = true; }
    check(threw && same_bits(r0, d.get("r")) && same_bits(p0, d.get("mean_r")) &&
          same_bits(X0, d.get("X")),
          "C4 a failing set_current leaves r, mean_r and X unchanged");

    // C5: phi starts far from its posterior, so it changes over the first
    // sweeps.
    {
        const std::size_t Nh = 1000;
        arma::mat Xh(Nh, p);
        arma::vec yh(Nh);
        for (std::size_t i = 0; i < Nh; ++i) {
            for (std::size_t j = 0; j < p; ++j) Xh(i, j) = unif(gen);
            const double mm = std::exp(1.0 + 0.8 * std::max(Xh(i, 0), 0.0) + 0.6 * Xh(i, 1));
            yh[i] = mm + std::sqrt(0.5 * mm) * nd(gen);
        }
        GBartHeteroscedastic h(Xh, yh, 20, 20.0, 11, false, true);
        std::vector<double> phi_d;
        for (int t = 0; t < 30; ++t) {
            h.step(1);
            phi_d.push_back(h.get_current().at("phi")[0]);
        }
        const arma::mat R = h.get_history().at("r");
        const arma::mat Y = h.predict_at(AI4BayesCode::state_map{}).at("y_rep");
        double worst = 0.0;
        for (std::size_t dd = 0; dd < phi_d.size(); ++dd) {
            double v = 0.0;
            for (std::size_t i = 0; i < Nh; ++i) {
                const double mm = std::exp(R(dd, i));
                v += (Y(dd, i) - mm) * (Y(dd, i) - mm) / mm;
            }
            v /= static_cast<double>(Nh);
            worst = std::max(worst, std::abs(v / phi_d[dd] - 1.0));
        }
        std::printf("C5 history predict_at: phi_d from %.3f to %.3f, "
                    "max |v_d / phi_d - 1| = %.3f (bound %.3f)\n",
                    phi_d.front(), phi_d.back(), worst, 6.0 * std::sqrt(2.0 / Nh));
        check(worst < 6.0 * std::sqrt(2.0 / Nh),
              "C5 history-mode y_rep of each draw uses the phi of that sweep");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

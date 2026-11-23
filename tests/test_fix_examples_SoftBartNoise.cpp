// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_SoftBartNoise.cpp
//
//  examples/SoftBartNoise.cpp: the shared f_softbart (the forest's fit at
//  the training X, reported by get_current() and used by predict_at({}))
//  and the loader set_current.
//    C1  after construction, the shared f_softbart is the forest's fit;
//    C2  after 10 sweeps and set_current of a new X (same N), get_current()
//        reports the forest's fit at the new X (= predict_at({X: new X}) of
//        the same forest) and y_rep from predict_at({}) is drawn around it:
//        with w = fit(new X) - fit(old X), T = sum w (y_rep - fit(new X))
//        has |T| / sd(T) < 5 (a draw around the old fit gives
//        E[T] = -sum w^2);
//    C3  after set_current of y, the shared "y" and the working response
//        equal the loaded y and f_softbart is the forest's fit;
//    C4  set_current with a valid X and sigma = -1 throws and leaves X, y,
//        f_softbart and sigma unchanged;
//    C5  with keep_history on: 4 sweeps, set_current of X, set_current of
//        y, set_tree(get_tree()) and 2 sweeps give f_softbart and sigma
//        histories of 6 rows each (one per sweep), the last f_softbart row
//        is the forest's fit, and predict_at({}) returns 6 rows of y_rep;
//    C6  set_current with an empty sigma throws and leaves get_current()
//        unchanged.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main softbart_noise_demo_main
#include "../examples/SoftBartNoise.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> SoftBartNoise::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &SoftBartNoise::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const SoftBartNoise& m) {
    return *(m.*member_of(impl_tag{}));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

SoftBartNoise make(const arma::mat& X, const arma::vec& y, int seed) {
    return SoftBartNoise(X, y, 20, 2.0, 10.0, false, seed, 3.0, false, false);
}

}  // namespace

int main() {
    // Unbuffered, so the lines printed before a crash reach the log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== test_fix_examples_SoftBartNoise ===\n");
    const std::size_t N = 300, p = 3;
    std::mt19937_64 gen(41u);
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::normal_distribution<double> nd(0.0, 0.5);
    arma::mat X(N, p), X2(N, p);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < p; ++j) { X(i, j) = unif(gen); X2(i, j) = unif(gen); }
        y[i] = 2.0 * std::max(X(i, 0), 0.0) + 0.5 * X(i, 1) * X(i, 1) - X(i, 2) + nd(gen);
    }

    SoftBartNoise m = make(X, y, 7);
    check(same_bits(impl(m).data().get("f_softbart"), impl(m).child(0).current()),
          "C1 f_softbart is the forest's fit after construction");

    m.step(10);
    const arma::vec f_old = m.get_current().at("f_softbart");
    AI4BayesCode::state_map s;
    s["X"] = arma::vec(arma::vectorise(X2));
    m.set_current(s);
    AI4BayesCode::state_map at_new;
    at_new["X"] = arma::vec(arma::vectorise(X2));
    const arma::vec f_new = arma::vectorise(m.predict_at(at_new).at("f_softbart"));
    const arma::vec f_cur = m.get_current().at("f_softbart");
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    const double sg = m.get_current().at("sigma")[0];
    double T = 0.0, V = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double w = f_new[i] - f_old[i];
        T += w * (yr[i] - f_new[i]);
        V += w * w * sg * sg;
    }
    std::printf("C2 set_current(X): max |get_current f_softbart - fit at new X| = %.3g, "
                "y_rep T/sd = %.3f\n", arma::abs(f_cur - f_new).max(), T / std::sqrt(V));
    check(same_bits(f_cur, f_new) && same_bits(f_cur, impl(m).child(0).current()),
          "C2 get_current() reports the forest's fit at the new X");
    check(std::abs(T / std::sqrt(V)) < 5.0, "C2 predict_at({}) y_rep drawn around the new fit");

    SoftBartNoise m3 = make(X, y, 9);
    m3.step(10);
    const arma::vec y2 = y + 0.25;
    AI4BayesCode::state_map s3;
    s3["y"] = y2;
    m3.set_current(s3);
    const auto& d3 = impl(m3).data();
    check(same_bits(d3.get("y"), y2) && same_bits(d3.get("softbart_target"), y2) &&
          same_bits(d3.get("f_softbart"), impl(m3).child(0).current()),
          "C3 y and the working response hold the loaded y, f_softbart is the forest's fit");

    SoftBartNoise m5 = make(X, y, 11);
    m5.step(3);
    const auto& d5 = impl(m5).data();
    const arma::vec X0 = d5.get("X"), y0 = d5.get("y"), f0 = d5.get("f_softbart"), s0 = d5.get("sigma");
    AI4BayesCode::state_map bad;
    bad["X"] = arma::vec(arma::vectorise(X2));
    bad["sigma"] = arma::vec{-1.0};
    bool threw = false;
    try { m5.set_current(bad); } catch (const std::exception&) { threw = true; }
    check(threw && same_bits(X0, d5.get("X")) && same_bits(y0, d5.get("y")) &&
          same_bits(f0, d5.get("f_softbart")) && same_bits(f0, impl(m5).child(0).current()) &&
          same_bits(s0, d5.get("sigma")),
          "C4 a failing set_current leaves X, y, f_softbart and sigma unchanged");

    // C5
    SoftBartNoise m6(X, y, 20, 2.0, 10.0, false, 13, 3.0, false, true);
    m6.step(4);
    AI4BayesCode::state_map s6x, s6y;
    s6x["X"] = arma::vec(arma::vectorise(X2));
    s6y["y"] = arma::vec(y + 0.25);
    m6.set_current(s6x);
    m6.set_current(s6y);
    m6.set_tree(m6.get_tree());
    m6.step(2);
    const auto h6 = m6.get_history();
    const arma::mat F6 = h6.at("f_softbart");
    const arma::uword rows_s6 = h6.at("sigma").n_rows;
    arma::uword rows_y6 = 0;
    try {
        rows_y6 = m6.predict_at(AI4BayesCode::state_map{}).at("y_rep").n_rows;
    } catch (const std::exception& e) {
        std::printf("C5 predict_at({}) threw: %s\n", e.what());
    }
    std::printf("C5 keep_history: history rows f_softbart %u, sigma %u; y_rep rows %u\n",
                (unsigned)F6.n_rows, (unsigned)rows_s6, (unsigned)rows_y6);
    check(F6.n_rows == 6 && rows_s6 == 6 && rows_y6 == 6 &&
          same_bits(arma::vec(F6.row(F6.n_rows - 1).t()), impl(m6).child(0).current()),
          "C5 the loaders add no history row: one f_softbart row per sweep");

    {
        SoftBartNoise me = make(X, y, 17);
        me.step(2);
        bool ok = true;
        for (const char* key : {"sigma"}) {
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
        check(ok, "C6 set_current(sigma = empty) throws and leaves get_current() unchanged");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

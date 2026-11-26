// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_BartNoise.cpp
//
//  examples/BartNoise.cpp: the shared f_bart (the forest's fit at the
//  training X, reported by get_current() and used by predict_at({})), the
//  shared "y" (sigma's conditional uses y - f_bart) and the loader
//  set_current.
//    C1  after construction, the shared f_bart is the forest's fit;
//    C2  after 10 sweeps and set_current of a new X (same N), get_current()
//        reports the forest's fit at the new X (= predict_at({X: new X}) of
//        the same forest) and y_rep from predict_at({}) is drawn around it:
//        with w = fit(new X) - fit(old X), T = sum w (y_rep - fit(new X))
//        has |T| / sd(T) < 5 (a draw around the old fit gives
//        E[T] = -sum w^2);
//    C3  after set_current of y, the shared "y" and the bart working
//        response equal the loaded y and f_bart is the forest's fit;
//    C4  after set_current(y = y + 10) on a model with sigma near 0.5,
//        sigma is below 2 after each of the next 5 sweeps (sigma's
//        conditional uses the loaded y and the fit to it);
//    C5  set_current with a valid X and sigma = -1 throws and leaves X, y,
//        f_bart and sigma unchanged;
//    C6  with keep_history on: 4 sweeps, set_current of X, set_current of
//        y, set_tree(get_tree()) and 2 sweeps give f_bart and sigma
//        histories of 6 rows each (one per sweep), the last f_bart row is
//        the forest's fit, and predict_at({}) returns 6 rows of y_rep;
//    C7  set_current with an empty sigma throws and leaves get_current()
//        unchanged;
//    C8  with keep_history on and keep_tree off, predict_at at new X returns
//        one row (f_bart from the current forest) after 0 and after 1 sweep,
//        as in 1.1.2; after 5 sweeps it throws with a message that names
//        keep_tree = TRUE and the 5 stored draws, and last_draw_only gives
//        one row; with keep_tree on as well it gives one row per stored draw.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main bart_noise_demo_main
#include "../examples/BartNoise.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> BartNoise::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &BartNoise::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const BartNoise& m) {
    return *(m.*member_of(impl_tag{}));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

BartNoise make(const arma::mat& X, const arma::vec& y, int seed) {
    return BartNoise(X, y, 30, 2.0, 2.0, 0.95, 3.0, 100, false, false, seed, false, false);
}

}  // namespace

int main() {
    // Unbuffered, so the lines printed before a crash reach the log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== test_fix_examples_BartNoise ===\n");
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

    BartNoise m = make(X, y, 7);
    check(same_bits(impl(m).data().get("f_bart"), impl(m).child(0).current()),
          "C1 f_bart is the forest's fit after construction");

    m.step(10);
    const arma::vec f_old = m.get_current().at("f_bart");
    AI4BayesCode::state_map s;
    s["X"] = arma::vec(arma::vectorise(X2));
    m.set_current(s);
    AI4BayesCode::state_map at_new;
    at_new["X"] = arma::vec(arma::vectorise(X2));
    const arma::vec f_new = arma::vectorise(m.predict_at(at_new).at("f_bart"));
    const arma::vec f_cur = m.get_current().at("f_bart");
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    const double sg = m.get_current().at("sigma")[0];
    double T = 0.0, V = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double w = f_new[i] - f_old[i];
        T += w * (yr[i] - f_new[i]);
        V += w * w * sg * sg;
    }
    std::printf("C2 set_current(X): max |get_current f_bart - fit at new X| = %.3g, "
                "y_rep T/sd = %.3f\n", arma::abs(f_cur - f_new).max(), T / std::sqrt(V));
    check(same_bits(f_cur, f_new) && same_bits(f_cur, impl(m).child(0).current()),
          "C2 get_current() reports the forest's fit at the new X");
    check(std::abs(T / std::sqrt(V)) < 5.0, "C2 predict_at({}) y_rep drawn around the new fit");

    BartNoise m3 = make(X, y, 9);
    m3.step(10);
    const arma::vec y2 = y + 10.0;
    AI4BayesCode::state_map s3;
    s3["y"] = y2;
    m3.set_current(s3);
    const auto& d3 = impl(m3).data();
    check(same_bits(d3.get("y"), y2) && same_bits(d3.get("bart_target"), y2) &&
          same_bits(d3.get("f_bart"), impl(m3).child(0).current()),
          "C3 y and the working response hold the loaded y, f_bart is the forest's fit");

    double sigma_max = 0.0;
    for (int t = 0; t < 5; ++t) {
        m3.step(1);
        sigma_max = std::max(sigma_max, m3.get_current().at("sigma")[0]);
    }
    std::printf("C4 set_current(y + 10): largest sigma over the next 5 sweeps = %.3f\n", sigma_max);
    check(sigma_max < 2.0, "C4 sigma's conditional uses the loaded y");

    BartNoise m5 = make(X, y, 11);
    m5.step(3);
    const auto& d5 = impl(m5).data();
    const arma::vec X0 = d5.get("X"), y0 = d5.get("y"), f0 = d5.get("f_bart"), s0 = d5.get("sigma");
    AI4BayesCode::state_map bad;
    bad["X"] = arma::vec(arma::vectorise(X2));
    bad["sigma"] = arma::vec{-1.0};
    bool threw = false;
    try { m5.set_current(bad); } catch (const std::exception&) { threw = true; }
    check(threw && same_bits(X0, d5.get("X")) && same_bits(y0, d5.get("y")) &&
          same_bits(f0, d5.get("f_bart")) && same_bits(f0, impl(m5).child(0).current()) &&
          same_bits(s0, d5.get("sigma")),
          "C5 a failing set_current leaves X, y, f_bart and sigma unchanged");

    // C6
    BartNoise m6(X, y, 30, 2.0, 2.0, 0.95, 3.0, 100, false, false, 13, false, true);
    m6.step(4);
    AI4BayesCode::state_map s6x, s6y;
    s6x["X"] = arma::vec(arma::vectorise(X2));
    s6y["y"] = arma::vec(y + 0.5);
    m6.set_current(s6x);
    m6.set_current(s6y);
    m6.set_tree(m6.get_tree());
    m6.step(2);
    const auto h6 = m6.get_history();
    const arma::mat F6 = h6.at("f_bart");
    const arma::uword rows_s6 = h6.at("sigma").n_rows;
    arma::uword rows_y6 = 0;
    try {
        rows_y6 = m6.predict_at(AI4BayesCode::state_map{}).at("y_rep").n_rows;
    } catch (const std::exception& e) {
        std::printf("C6 predict_at({}) threw: %s\n", e.what());
    }
    std::printf("C6 keep_history: history rows f_bart %u, sigma %u; y_rep rows %u\n",
                (unsigned)F6.n_rows, (unsigned)rows_s6, (unsigned)rows_y6);
    check(F6.n_rows == 6 && rows_s6 == 6 && rows_y6 == 6 &&
          same_bits(arma::vec(F6.row(F6.n_rows - 1).t()), impl(m6).child(0).current()),
          "C6 the loaders add no history row: one f_bart row per sweep");

    {
        BartNoise me = make(X, y, 17);
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
        check(ok, "C7 set_current(sigma = empty) throws and leaves get_current() unchanged");
    }

    {
        AI4BayesCode::state_map at_x;
        at_x["X"] = arma::vec(arma::vectorise(arma::mat(X2.rows(0, 9))));
        BartNoise mh(X, y, 30, 2.0, 2.0, 0.95, 3.0, 100, false, false, 19, false, true);
        bool ok01 = true;
        for (int sweeps = 0; sweeps <= 1; ++sweeps) {
            if (sweeps == 1) mh.step(1);
            AI4BayesCode::history_map r;
            try {
                r = mh.predict_at(at_x);
            } catch (const std::exception& e) {
                std::printf("C8 keep_tree off, %d sweep(s): threw \"%s\"\n", sweeps, e.what());
                ok01 = false;
                continue;
            }
            const arma::mat F1 = r.at("f_bart"), Y1 = r.at("y_rep");
            const arma::mat Fl = mh.predict_at(at_x, true).at("f_bart");
            std::printf("C8 keep_tree off, %d sweep(s): f_bart %u x %u, y_rep %u x %u\n",
                        sweeps, (unsigned)F1.n_rows, (unsigned)F1.n_cols,
                        (unsigned)Y1.n_rows, (unsigned)Y1.n_cols);
            ok01 = ok01 && F1.n_rows == 1 && F1.n_cols == 10 && Y1.n_rows == 1 &&
                   Y1.n_cols == 10 &&
                   same_bits(arma::vec(F1.row(0).t()), arma::vec(Fl.row(0).t()));
        }
        check(ok01, "C8 without keep_tree, 0 and 1 stored draws give one row at new X "
                    "(f_bart from the current forest)");
        mh.step(4);
        std::string msg;
        try { mh.predict_at(at_x); } catch (const std::exception& e) { msg = e.what(); }
        const arma::uword rows_last = mh.predict_at(at_x, true).at("f_bart").n_rows;
        BartNoise mt(X, y, 30, 2.0, 2.0, 0.95, 3.0, 100, false, false, 19, true, true);
        mt.step(5);
        const arma::mat Ft = mt.predict_at(at_x).at("f_bart");
        std::printf("C8 keep_tree off, 5 sweeps: \"%s\"; last_draw_only rows %u; "
                    "keep_tree on: f_bart %u x %u\n", msg.c_str(), (unsigned)rows_last,
                    (unsigned)Ft.n_rows, (unsigned)Ft.n_cols);
        check(msg.find("keep_tree = TRUE") != std::string::npos &&
              msg.find("5 stored draws") != std::string::npos && rows_last == 1 &&
              Ft.n_rows == 5 && Ft.n_cols == 10,
              "C8 without keep_tree, 5 stored draws raise an error that names "
              "keep_tree = TRUE");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

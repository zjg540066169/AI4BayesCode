// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_GBartPoisson.cpp
//
//  examples/GBartPoisson.cpp stores the forest fit r at the training X and
//  "rate" = exp(r) in the shared data; get_current() reports r and
//  predict_at({}) draws y_rep from "rate".
//    C1  after construction, the shared r is the forest's fit and rate
//        equals exp(fit);
//    C2  after 20 sweeps and set_current of a new X (same N): get_current()
//        reports the forest's fit at the new X (= predict_at({X: new X}) of
//        the same forest), the shared rate equals exp(fit), and y_rep from
//        predict_at({}) is drawn from exp(fit at the new X): with
//        w = exp(new fit) - exp(old fit), T = sum w (y_rep - exp(new fit))
//        has |T| / sd(T) < 5 (a draw from the old fit gives
//        E[T] = -sum w^2);
//    C3  after set_tree of the forest of a second model on the same training
//        X, r equals that model's r (relative tolerance 1e-12) and rate
//        equals exp(fit);
//    C4  set_current with an X whose length is not a multiple of p throws
//        and leaves r, rate and X unchanged.
//  predict_at at new X with keep_history = TRUE and keep_tree = TRUE:
//    C5  r, rate and y_rep have one row per stored draw; row d of r and rate
//        equals the single-draw prediction of a second model loaded with
//        get_tree_history()[d] (set_tree, then predict_at(X, TRUE)), bit for
//        bit; y_rep holds non-negative integers and
//        T = sum_{d,i} (y_rep - rate) / sqrt(sum rate) has |T| < 5; the last
//        row equals the last_draw_only prediction bit for bit;
//    C6  with the training X as new data, row d of r equals row d of
//        get_history() r (relative tolerance 1e-12);
//    C7  without keep_tree, the history-mode prediction at new X gives one
//        row (r from the current forest) after 0 and after 1 sweep; after 3
//        sweeps it throws with a message that names keep_tree = TRUE and the
//        3 stored draws, and last_draw_only gives one row;
//    C8  a model with clear_history() after K1 sweeps and K2 more sweeps
//        gives K2 rows of r and rate, equal to the last K2 rows of a model
//        built and stepped with the same seed without the call, bit for bit.
//  The forest's fit is the block's own current value; rate is compared with
//  exp(fit) computed in this test (relative tolerance 1e-12).
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main gbart_poisson_demo_main
#include "../examples/GBartPoisson.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> GBartPoisson::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &GBartPoisson::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const GBartPoisson& m) {
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

bool same_bits(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

double max_rel(const arma::mat& a, const arma::mat& b) {
    if (a.n_rows != b.n_rows || a.n_cols != b.n_cols) return INFINITY;
    double e = 0.0;
    for (arma::uword k = 0; k < a.n_elem; ++k)
        e = std::max(e, std::abs(a[k] - b[k]) / std::max(1.0, std::abs(b[k])));
    return e;
}

arma::vec rate_of(const arma::vec& r) {
    arma::vec p(r.n_elem);
    for (std::size_t i = 0; i < r.n_elem; ++i) p[i] = std::exp(r[i]);
    return p;
}

// The forest's fit at the training X (the block's own value).
arma::vec fit(const GBartPoisson& m) { return impl(m).child(0).current(); }

// Shared r equals the fit and shared rate equals exp(fit).
bool shared_exact(const GBartPoisson& m) {
    const auto& d = impl(m).data();
    return same_bits(d.get("r"), fit(m)) && max_rel(d.get("rate"), rate_of(fit(m))) <= 1e-12;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_GBartPoisson ===\n");
    bart_rng::set_seed(20261123ULL);
    const std::size_t N = 300, p = 3;
    std::mt19937_64 gen(31u);
    std::uniform_real_distribution<double> unif(-1.5, 1.5);
    arma::mat X(N, p), X2(N, p);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < p; ++j) { X(i, j) = unif(gen); X2(i, j) = unif(gen); }
        std::poisson_distribution<int> pois(std::exp(1.0 + 0.8 * X(i, 0) - 0.6 * X(i, 1)));
        y[i] = static_cast<double>(pois(gen));
    }

    GBartPoisson m(X, y, 20, 7, false, false);
    check(shared_exact(m), "C1 r is the forest's fit and rate = exp(fit) after construction");

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
        V += w * w * p_new[i];
    }
    std::printf("C2 set_current(X): max |get_current r - fit at new X| = %.3g, "
                "y_rep T/sd = %.3f\n", arma::abs(r_cur - r_new).max(), T / std::sqrt(V));
    check(same_bits(r_cur, r_new), "C2 get_current() reports the forest's fit at the new X");
    check(shared_exact(m), "C2 shared r is the fit and rate = exp(fit) after set_current(X)");
    check(std::abs(T / std::sqrt(V)) < 5.0, "C2 predict_at({}) y_rep drawn from the new fit");

    // C3: two models on the same training X; the first loads the forest of
    // the second.
    GBartPoisson m3(X, y, 20, 7, false, false), other(X, y, 20, 9, false, false);
    m3.step(10);
    other.step(15);
    m3.set_tree(other.get_tree());
    std::printf("C3 set_tree: max rel |r - other r| = %.3g\n",
                max_rel(m3.get_current().at("r"), other.get_current().at("r")));
    check(max_rel(m3.get_current().at("r"), other.get_current().at("r")) <= 1e-12 &&
          shared_exact(m3),
          "C3 set_tree loads the forest's r and rate = exp(fit)");

    const auto& d = impl(m).data();
    const arma::vec r0 = d.get("r"), p0 = d.get("rate"), X0 = d.get("X");
    AI4BayesCode::state_map bad;
    bad["X"] = arma::vec(N * p + 1, arma::fill::zeros);
    bool threw = false;
    try { m.set_current(bad); } catch (const std::exception&) { threw = true; }
    check(threw && same_bits(r0, d.get("r")) && same_bits(p0, d.get("rate")) &&
          same_bits(X0, d.get("X")),
          "C4 a failing set_current leaves r, rate and X unchanged");

    // ---- predict_at at new X in history mode ------------------------------
    const std::size_t K = 25, K1 = 6, K2 = 9, Nn = 40;
    arma::mat Xn(Nn, p);
    for (std::size_t i = 0; i < Nn; ++i)
        for (std::size_t j = 0; j < p; ++j) Xn(i, j) = unif(gen);
    AI4BayesCode::state_map nd, nd_train;
    nd["X"] = arma::vec(arma::vectorise(Xn));
    nd_train["X"] = arma::vec(arma::vectorise(X));

    // C5
    GBartPoisson mh(X, y, 20, 5, true, true);
    mh.step(static_cast<int>(K));
    const auto pn = mh.predict_at(nd);
    const arma::mat& R5 = pn.at("r");
    const arma::mat& L5 = pn.at("rate");
    const arma::mat& Y5 = pn.at("y_rep");
    const std::vector<std::string> th = mh.get_tree_history();
    GBartPoisson probe(X, y, 20, 99, false, false);
    bool rows_ok = th.size() == K;
    for (std::size_t dd = 0; rows_ok && dd < K; ++dd) {
        probe.set_tree(th[dd]);
        const auto q = probe.predict_at(nd, true);
        rows_ok = same_bits(arma::mat(R5.row(dd)), q.at("r")) &&
                  same_bits(arma::mat(L5.row(dd)), q.at("rate"));
    }
    double T5 = 0.0, V5 = 0.0;
    bool counts = true;
    for (arma::uword k = 0; k < Y5.n_elem; ++k) {
        counts = counts && Y5[k] >= 0.0 && Y5[k] == std::floor(Y5[k]);
        T5 += Y5[k] - L5[k];
        V5 += L5[k];
    }
    const auto pl = mh.predict_at(nd, true);
    const bool last5 = R5.n_rows == K && same_bits(arma::mat(R5.row(K - 1)), pl.at("r")) &&
                       same_bits(arma::mat(L5.row(K - 1)), pl.at("rate"));
    std::printf("C5 history, new X: r %d x %d, rate %d x %d, y_rep %d x %d; rows = stored "
                "forests: %d; y_rep T/sd = %.3f; last row = last_draw_only: %d\n",
                (int)R5.n_rows, (int)R5.n_cols, (int)L5.n_rows, (int)L5.n_cols,
                (int)Y5.n_rows, (int)Y5.n_cols, rows_ok ? 1 : 0, T5 / std::sqrt(V5),
                last5 ? 1 : 0);
    check(R5.n_rows == K && R5.n_cols == Nn && L5.n_rows == K && L5.n_cols == Nn &&
          Y5.n_rows == K && Y5.n_cols == Nn && rows_ok,
          "C5 row d of r and rate comes from the forest stored at sweep d");
    check(counts && std::abs(T5 / std::sqrt(V5)) < 5.0,
          "C5 y_rep of row d is drawn from Poisson(rate of row d)");
    check(last5, "C5 the last row equals the last_draw_only prediction");

    // C6
    const arma::mat R6 = mh.predict_at(nd_train).at("r");
    const double e6 = max_rel(R6, mh.get_history().at("r"));
    std::printf("C6 training X as new data: max rel |r - get_history() r| = %.3g\n", e6);
    check(R6.n_rows == K && e6 <= 1e-12,
          "C6 the forests stored at sweep d reproduce the stored r of draw d");

    // C7
    GBartPoisson mn(X, y, 20, 5, false, true);
    bool ok01 = true;
    for (int sweeps = 0; sweeps <= 1; ++sweeps) {
        if (sweeps == 1) mn.step(1);
        try {
            const auto r01 = mn.predict_at(nd);
            const arma::mat r_l = mn.predict_at(nd, true).at("r");
            ok01 = ok01 && r01.at("r").n_rows == 1 && r01.at("y_rep").n_rows == 1 &&
                   arma::approx_equal(r01.at("r"), r_l, "absdiff", 0.0);
        } catch (const std::exception& e) {
            std::printf("C7 keep_tree off, %d sweep(s): threw \"%s\"\n", sweeps, e.what());
            ok01 = false;
        }
    }
    check(ok01, "C7 without keep_tree, 0 and 1 stored draws give one row at new X "
                "(r from the current forest)");
    mn.step(2);
    std::string msg;
    try { mn.predict_at(nd); } catch (const std::exception& e) { msg = e.what(); }
    std::printf("C7 keep_tree off, 3 sweeps: \"%s\"\n", msg.c_str());
    const arma::mat R7 = mn.predict_at(nd, true).at("r");
    check(msg.find("keep_tree = TRUE") != std::string::npos &&
          msg.find("3 stored draws") != std::string::npos && R7.n_rows == 1 &&
          R7.n_cols == Nn,
          "C7 without keep_tree the history-mode prediction at new X names keep_tree = TRUE");

    // C8: the models are built and stepped one after the other (they share
    // the thread's kernel stream, which each constructor seeds).
    GBartPoisson mA(X, y, 20, 5, true, true);
    mA.step(static_cast<int>(K1));
    mA.clear_history();
    mA.step(static_cast<int>(K2));
    const auto pA = mA.predict_at(nd);
    GBartPoisson mB(X, y, 20, 5, true, true);
    mB.step(static_cast<int>(K1 + K2));
    const auto pB = mB.predict_at(nd);
    const arma::mat& RA = pA.at("r");
    const arma::mat& RB = pB.at("r");
    const bool tail8 = RA.n_rows == K2 && RB.n_rows == K1 + K2 &&
                       same_bits(RA, arma::mat(RB.rows(K1, K1 + K2 - 1))) &&
                       same_bits(pA.at("rate"), arma::mat(pB.at("rate").rows(K1, K1 + K2 - 1))) &&
                       pA.at("y_rep").n_rows == K2;
    std::printf("C8 clear_history(): rows %d (model without the call: %d)\n",
                (int)RA.n_rows, (int)RB.n_rows);
    check(tail8, "C8 after clear_history() the rows are the draws of the later sweeps");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

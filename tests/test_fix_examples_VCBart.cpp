// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_VCBart.cpp
//
//  examples/VCBart.cpp: sigma is drawn inside the BART kernel of beta_0 (no
//  loader exists for it) and copied to the shared "sigma" after each beta_0
//  update; get_current() reports the shared value.
//    C1  set_current with another sigma throws and leaves the reported sigma
//        unchanged (a value written only to the shared data would be
//        replaced by beta_0's own sigma at the next sweep);
//    C2  set_current with a y of the wrong length throws and leaves y, the
//        working responses and mu unchanged;
//    C3  set_current(get_current()) is accepted and changes nothing;
//    C4  set_current of a valid y recomputes every working response
//        wr_j = (y - sum_{k != j} x_k beta_k) / x_j and mu (relative
//        tolerance 1e-12).
//  predict_at (mu = sum_{j=0..p} x_j beta_j(Z), x_0 = 1), p = 2:
//    C5  single draw at the training data: 1 x N, equal to get_current() mu
//        (forests evaluated at the training Z, relative tolerance 1e-12);
//        X column j multiplies beta_{j+1};
//    C6  keep_history, training data: one row per stored draw, row d =
//        sum_j x_j beta_{j,d} from the rows of get_history() (relative
//        tolerance 1e-12); the last row equals get_current() mu;
//    C7  keep_history + keep_tree, new (X, Z): one row per stored draw; with
//        the training (X, Z) as new data, row d equals row d of C6 (stored
//        forest d plus the fmean of sweep d, relative tolerance 1e-12); the
//        last row equals the last_draw_only prediction bit for bit;
//    C8  keep_history without keep_tree: new (X, Z) throws, last_draw_only
//        gives 1 x N_new; after clear_history() and K more sweeps every
//        history-mode prediction has K rows.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main vcbart_demo_main
#include "../examples/VCBart.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> VCBart::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &VCBart::impl_>;

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

bool throws(VCBart& m, const AI4BayesCode::state_map& s) {
    try { m.set_current(s); } catch (const std::exception&) { return true; }
    return false;
}

bool same_bits(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

// max_k |a_k - b_k| / max(1, |b_k|); infinity when the shapes differ.
double max_rel(const arma::mat& a, const arma::mat& b) {
    if (a.n_rows != b.n_rows || a.n_cols != b.n_cols) return INFINITY;
    double e = 0.0;
    for (arma::uword k = 0; k < a.n_elem; ++k)
        e = std::max(e, std::abs(a[k] - b[k]) / std::max(1.0, std::abs(b[k])));
    return e;
}

arma::mat as_row(const arma::vec& v) { return arma::mat(v.t()); }

}  // namespace

int main() {
    std::printf("=== test_fix_examples_VCBart ===\n");
    const std::size_t N = 150;
    std::mt19937_64 g(47u);
    std::uniform_real_distribution<double> uZ(-1.0, 1.0), uX(0.5, 1.5);
    std::normal_distribution<double> eps(0.0, 0.3);
    arma::mat X(N, 1), Z(N, 1);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        Z(i, 0) = uZ(g); X(i, 0) = uX(g);
        y[i] = 2.0 * std::max(Z(i, 0), 0.0) + Z(i, 0) * Z(i, 0) * X(i, 0) + eps(g);
    }
    VCBart m(X, Z, y, 10, 2.0, 2.0, 0.95, 3.0, 100, 1, false);
    m.step(10);
    const auto& d = (m.*member_of(impl_tag{}))->data();

    // C1
    const auto c0 = m.get_current();
    AI4BayesCode::state_map s1;
    s1["sigma"] = arma::vec{c0.at("sigma")[0] * 3.0};
    const bool t1 = throws(m, s1);
    std::printf("C1 set_current(sigma x 3): threw = %d, reported sigma %.4f (before %.4f)\n",
                t1 ? 1 : 0, m.get_current().at("sigma")[0], c0.at("sigma")[0]);
    check(t1 && same_bits(c0.at("sigma"), m.get_current().at("sigma")),
          "C1 another sigma is rejected and the reported sigma is the one in use");

    // C2
    const arma::vec wr0 = d.get("wr_0"), mu0 = d.get("mu");
    AI4BayesCode::state_map s2;
    s2["y"] = arma::vec(N - 1, arma::fill::ones);
    const bool t2 = throws(m, s2);
    check(t2 && same_bits(d.get("y"), y) && same_bits(d.get("wr_0"), wr0) && same_bits(d.get("mu"), mu0),
          "C2 a y of the wrong length is rejected and nothing is written");

    // C3
    const auto before = m.get_current();
    const bool t3 = throws(m, before);
    const auto after = m.get_current();
    check(!t3 && same_bits(before.at("sigma"), after.at("sigma")) &&
          same_bits(before.at("mu"), after.at("mu")),
          "C3 set_current(get_current()) is accepted and changes nothing");

    // C4
    const arma::vec y2 = y * 2.0 + 1.0;
    AI4BayesCode::state_map s4;
    s4["y"] = y2;
    m.set_current(s4);
    const arma::vec& b0 = d.get("beta_0");
    const arma::vec& b1 = d.get("beta_1");
    double e = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double wr0_ref = (y2[i] - X(i, 0) * b1[i]) / 1.0;
        const double wr1_ref = (y2[i] - b0[i]) / X(i, 0);
        const double mu_ref  = b0[i] + X(i, 0) * b1[i];
        e = std::max(e, std::abs(d.get("wr_0")[i] - wr0_ref) / std::max(1.0, std::abs(wr0_ref)));
        e = std::max(e, std::abs(d.get("wr_1")[i] - wr1_ref) / std::max(1.0, std::abs(wr1_ref)));
        e = std::max(e, std::abs(d.get("mu")[i] - mu_ref) / std::max(1.0, std::abs(mu_ref)));
    }
    std::printf("C4 set_current(y): max rel error of wr_0, wr_1, mu = %.3g\n", e);
    check(e <= 1e-12, "C4 set_current(y) recomputes the working responses and mu");

    // ---- predict_at, p = 2 ------------------------------------------------
    const std::size_t N2 = 120, K = 25, K2 = 7, Nn = 9;
    arma::mat X2(N2, 2), Z2(N2, 1);
    arma::vec y3(N2);
    for (std::size_t i = 0; i < N2; ++i) {
        Z2(i, 0) = uZ(g); X2(i, 0) = uX(g); X2(i, 1) = uX(g);
        y3[i] = 2.0 * std::max(Z2(i, 0), 0.0) + Z2(i, 0) * Z2(i, 0) * X2(i, 0)
                - Z2(i, 0) * X2(i, 1) + eps(g);
    }
    arma::mat Zn(Nn, 1), Xn(Nn, 2);
    for (std::size_t i = 0; i < Nn; ++i) {
        Zn(i, 0) = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(Nn - 1);
        Xn(i, 0) = uX(g); Xn(i, 1) = uX(g);
    }
    AI4BayesCode::state_map nd_new, nd_train;
    nd_new["Z"]   = arma::vectorise(Zn); nd_new["X"]   = arma::vectorise(Xn);
    nd_train["Z"] = arma::vectorise(Z2); nd_train["X"] = arma::vectorise(X2);
    const AI4BayesCode::state_map empty;

    // C5
    VCBart m5(X2, Z2, y3, 10, 2.0, 2.0, 0.95, 3.0, 100, 3, false);
    m5.step(15);
    const arma::mat p5 = m5.predict_at(empty).at("mu");
    const double e5 = max_rel(p5, as_row(m5.get_current().at("mu")));
    std::printf("C5 single draw, training data: %d x %d, max rel |mu - get_current mu| = %.3g\n",
                (int)p5.n_rows, (int)p5.n_cols, e5);
    check(p5.n_rows == 1 && p5.n_cols == N2 && e5 <= 1e-12,
          "C5 single-draw mu at the training data equals get_current() mu");

    // C6
    VCBart mh(X2, Z2, y3, 10, 2.0, 2.0, 0.95, 3.0, 100, 5, true, true);
    mh.step(static_cast<int>(K));
    const AI4BayesCode::history_map h = mh.get_history();
    const arma::mat p6 = mh.predict_at(empty).at("mu");
    arma::mat ref6(K, N2);
    for (std::size_t r = 0; r < K; ++r)
        for (std::size_t i = 0; i < N2; ++i)
            ref6(r, i) = h.at("beta_0")(r, i) + X2(i, 0) * h.at("beta_1")(r, i)
                         + X2(i, 1) * h.at("beta_2")(r, i);
    const double e6 = max_rel(p6, ref6);
    const arma::mat mu6 = as_row(mh.get_current().at("mu"));
    const arma::mat last6 = (p6.n_rows == K) ? arma::mat(p6.row(K - 1)) : arma::mat();
    std::printf("C6 history, training data: %d x %d (K = %d), max rel |row d - sum_j x_j beta_j,d| "
                "= %.3g, last row vs get_current mu: max rel %.3g, bit-identical %d\n",
                (int)p6.n_rows, (int)p6.n_cols, (int)K, e6, max_rel(last6, mu6),
                same_bits(last6, mu6) ? 1 : 0);
    check(p6.n_rows == K && p6.n_cols == N2 && e6 <= 1e-12 && max_rel(last6, mu6) <= 1e-12,
          "C6 history-mode mu at the training data: row d from the stored betas of draw d");

    // C7
    const arma::mat p7  = mh.predict_at(nd_train).at("mu");
    const arma::mat p7n = mh.predict_at(nd_new).at("mu");
    const arma::mat p7l = mh.predict_at(nd_new, true).at("mu");
    const double e7 = max_rel(p7, p6);
    const bool last7 = p7n.n_rows == K && same_bits(arma::mat(p7n.row(K - 1)), p7l);
    std::printf("C7 history, new (X, Z): %d x %d; training (X, Z) as new data vs C6: max rel %.3g; "
                "last row == last_draw_only (1 x %d) bit for bit: %d\n",
                (int)p7n.n_rows, (int)p7n.n_cols, e7, (int)p7l.n_cols, last7 ? 1 : 0);
    check(p7.n_rows == K && e7 <= 1e-12 && p7n.n_rows == K && p7n.n_cols == Nn &&
          p7l.n_rows == 1 && last7,
          "C7 history-mode mu at new (X, Z): row d from the forests stored at sweep d");

    // C8
    VCBart mn(X2, Z2, y3, 10, 2.0, 2.0, 0.95, 3.0, 100, 7, true);   // keep_tree = false
    mn.step(5);
    bool threw = false;
    try { mn.predict_at(nd_new); } catch (const std::exception&) { threw = true; }
    const arma::mat p8l = mn.predict_at(nd_new, true).at("mu");
    mh.clear_history();
    mh.step(static_cast<int>(K2));
    const std::size_t r0 = mh.predict_at(empty).at("mu").n_rows;
    const std::size_t r1 = mh.predict_at(nd_new).at("mu").n_rows;
    std::printf("C8 keep_tree = false: new (X, Z) threw = %d, last_draw_only %d x %d; "
                "after clear_history() + %d sweeps: rows %d (training), %d (new)\n",
                threw ? 1 : 0, (int)p8l.n_rows, (int)p8l.n_cols, (int)K2, (int)r0, (int)r1);
    check(threw && p8l.n_rows == 1 && p8l.n_cols == Nn && r0 == K2 && r1 == K2,
          "C8 new (X, Z) in history mode needs keep_tree; row counts follow clear_history()");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

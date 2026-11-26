// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_GBartMultinomial.cpp
//
//  examples/GBartMultinomial.cpp stores the C-1 forest fits r_j at the
//  training X and "probs" = softmax(0, r_1, ..., r_{C-1}) in the shared data;
//  predict_at({}) draws y_rep from "probs".
//    C1  after construction, each shared r_j is its forest's fit and probs
//        equals the softmax of the fits;
//    C2  after 15 sweeps, the same holds, and y_rep from predict_at({}) is
//        drawn from softmax(fits): with w_ic = p_ic - 1/C,
//        T = sum_i sum_c w_ic (1[y_rep_i = c] - p_ic) has |T| / sd(T) < 5
//        (a draw from probs = 1/C gives E[T] = -sum w^2);
//    C3  after set_current of a new X (same N), each r_j is its forest's fit
//        at the new X (= predict_at({X: new X}) of the same forests) and
//        probs equals the softmax of the fits;
//    C4  after set_tree of the forests of a second model on the same
//        training X, r_j equals that model's r_j (relative tolerance 1e-12)
//        and probs equals the softmax of the fits;
//    C5  after 3 more sweeps, set_tree with a valid first forest (of a third
//        model) and an invalid second forest throws and leaves the trees of
//        both forests (get_tree()) unchanged; each shared r_j is its forest's
//        fit (bit for bit), probs their softmax, and r_j equals its value
//        before the call within relative tolerance 1e-12 (the fit of a
//        restored forest is recomputed from its trees);
//    C6  set_current with a valid y and an X with another row count throws
//        and leaves y unchanged;
//    C7  freezing a class forest throws: probs is recomputed after the last
//        class forest only, which gives the softmax of the current fits
//        because every forest updates in each sweep.
//  predict_at at new X with keep_history = TRUE and keep_tree = TRUE:
//    C8  r (K x Nn(C-1)), probs (K x Nn C) and y_rep (K x Nn) have one row
//        per stored draw; row d of r and probs equals the single-draw
//        prediction of a second model loaded with the forests stored at
//        sweep d (get_tree_history() entries d and K + d, joined as in
//        get_tree(); set_tree, then predict_at(X, TRUE)), bit for bit; y_rep
//        is in {0, ..., C-1} and, with w_ic = p_ic - 1/C,
//        T = sum_{d,i,c} w_ic (1[y_rep = c] - p_ic) has |T| / sd(T) < 5; the
//        last row equals the last_draw_only prediction bit for bit;
//    C9  with the training X as new data, row d of r equals rows d of
//        get_history() r_1 and r_2 (relative tolerance 1e-12);
//    C10 without keep_tree, the history-mode prediction at new X throws with
//        a message that names keep_tree = TRUE, and last_draw_only gives one
//        row;
//    C11 a model with clear_history() after K1 sweeps and K2 more sweeps
//        gives K2 rows of r and probs, equal to the last K2 rows of a model
//        built and stepped with the same seed without the call, bit for bit.
//  The fits are the blocks' own current values; probs is compared with a
//  softmax computed in this test (relative tolerance 1e-12).
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

#define main gbart_multinomial_demo_main
#include "../examples/GBartMultinomial.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> GBartMultinomial::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &GBartMultinomial::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

const std::size_t C = 3;

AI4BayesCode::composite_block& impl(const GBartMultinomial& m) {
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

// softmax(0, f_1, ..., f_{C-1}) of the forests' own fits, row-major N x C.
arma::vec softmax_of_fits(const GBartMultinomial& m) {
    const arma::vec f1 = impl(m).child(1).current(), f2 = impl(m).child(2).current();
    arma::vec p(f1.n_elem * C);
    for (std::size_t i = 0; i < f1.n_elem; ++i) {
        const double e0 = 1.0, e1 = std::exp(f1[i]), e2 = std::exp(f2[i]);
        const double z = e0 + e1 + e2;
        p[i * C] = e0 / z; p[i * C + 1] = e1 / z; p[i * C + 2] = e2 / z;
    }
    return p;
}

bool shared_exact(const GBartMultinomial& m) {
    const auto& d = impl(m).data();
    return same_bits(d.get(r_key(1)), impl(m).child(1).current()) &&
           same_bits(d.get(r_key(2)), impl(m).child(2).current()) &&
           max_rel(d.get("probs"), softmax_of_fits(m)) <= 1e-12;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_GBartMultinomial ===\n");
    bart_rng::set_seed(20261123ULL);
    const std::size_t N = 300, p = 3;
    std::mt19937_64 gen(37u);
    std::uniform_real_distribution<double> unif(-1.5, 1.5), u01(0.0, 1.0);
    arma::mat X(N, p), X2(N, p);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < p; ++j) { X(i, j) = unif(gen); X2(i, j) = unif(gen); }
        const double e1 = std::exp(2.0 * X(i, 0)), e2 = std::exp(-2.0 * X(i, 0) + 1.5 * X(i, 2));
        const double z = 1.0 + e1 + e2, u = u01(gen);
        y[i] = (u < 1.0 / z) ? 0.0 : (u < (1.0 + e1) / z ? 1.0 : 2.0);
    }

    GBartMultinomial m(X, y, static_cast<int>(C), 20, 7, false, false);
    check(shared_exact(m), "C1 r_j are the forests' fits and probs their softmax after construction");

    m.step(15);
    const arma::vec pr = softmax_of_fits(m);
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    double T = 0.0, V = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double s1 = 0.0, s2 = 0.0;
        for (std::size_t c = 0; c < C; ++c) {
            const double w = pr[i * C + c] - 1.0 / C;
            T += w * ((yr[i] == static_cast<double>(c) ? 1.0 : 0.0) - pr[i * C + c]);
            s1 += w * w * pr[i * C + c];
            s2 += w * pr[i * C + c];
        }
        V += s1 - s2 * s2;
    }
    std::printf("C2 after 15 sweeps: max rel |probs - softmax(fits)| = %.3g, y_rep T/sd = %.3f\n",
                max_rel(impl(m).data().get("probs"), pr), T / std::sqrt(V));
    check(shared_exact(m), "C2 probs is the softmax of the current fits");
    check(std::abs(T / std::sqrt(V)) < 5.0, "C2 predict_at({}) y_rep drawn from softmax(fits)");

    AI4BayesCode::state_map s;
    s["X"] = arma::vec(arma::vectorise(X2));
    m.set_current(s);
    AI4BayesCode::state_map at_new;
    at_new["X"] = arma::vec(arma::vectorise(X2));
    const arma::vec r_new = arma::vectorise(m.predict_at(at_new).at("r"));
    const arma::vec r_cur = m.get_current().at("r");
    std::printf("C3 set_current(X): max |get_current r - fits at new X| = %.3g\n",
                arma::abs(r_cur - r_new).max());
    check(same_bits(r_cur, r_new) && shared_exact(m),
          "C3 r_j are the forests' fits at the new X and probs their softmax");

    GBartMultinomial m4(X, y, static_cast<int>(C), 20, 7, false, false);
    GBartMultinomial other(X, y, static_cast<int>(C), 20, 9, false, false);
    m4.step(5);
    other.step(8);
    m4.set_tree(other.get_tree());
    std::printf("C4 set_tree: max rel |r - other r| = %.3g\n",
                max_rel(m4.get_current().at("r"), other.get_current().at("r")));
    check(max_rel(m4.get_current().at("r"), other.get_current().at("r")) <= 1e-12 &&
          shared_exact(m4), "C4 set_tree loads the forests' r_j and probs is their softmax");

    m4.step(3);   // the fits now carry the sums of the sweeps' tree updates
    const std::string t_before = m4.get_tree();
    const arma::vec r1_0 = impl(m4).data().get(r_key(1)), pr_0 = impl(m4).data().get("probs");
    const std::string sep = "\n===GBARTCLASS===\n";
    GBartMultinomial third(X, y, static_cast<int>(C), 20, 13, false, false);
    third.step(6);
    const std::string good_first = third.get_tree().substr(0, third.get_tree().find(sep));
    bool threw = false;
    try { m4.set_tree(good_first + sep + "not a forest"); }
    catch (const std::exception&) { threw = true; }
    std::printf("C5 failing set_tree: max rel |r_1 - r_1 before| = %.3g, max rel |probs - "
                "probs before| = %.3g\n", max_rel(impl(m4).data().get(r_key(1)), r1_0),
                max_rel(impl(m4).data().get("probs"), pr_0));
    check(threw && m4.get_tree() == t_before && shared_exact(m4) &&
          max_rel(impl(m4).data().get(r_key(1)), r1_0) <= 1e-12 &&
          max_rel(impl(m4).data().get("probs"), pr_0) <= 1e-12,
          "C5 a failing set_tree keeps the trees; r_j are the forests' fits, as before the call");

    GBartMultinomial m6(X, y, static_cast<int>(C), 20, 7, false, false);
    arma::vec y_alt = y;
    y_alt[0] = (y[0] == 0.0) ? 1.0 : 0.0;
    AI4BayesCode::state_map bad;
    bad["y"] = y_alt;
    bad["X"] = arma::vec(arma::vectorise(arma::mat(X.rows(0, N - 2))));
    threw = false;
    try { m6.set_current(bad); } catch (const std::exception&) { threw = true; }
    check(threw && same_bits(impl(m6).data().get("y"), y),
          "C6 a set_current that fails on X leaves y unchanged");

    threw = false;
    try { impl(m6).freeze(std::vector<std::string>{r_key(1)}, true); }
    catch (const std::exception&) { threw = true; }
    check(threw, "C7 a class forest cannot be frozen");

    // ---- predict_at at new X in history mode ------------------------------
    const std::size_t K = 20, K1 = 5, K2 = 8, Nn = 40;
    arma::mat Xn(Nn, p);
    for (std::size_t i = 0; i < Nn; ++i)
        for (std::size_t j = 0; j < p; ++j) Xn(i, j) = unif(gen);
    AI4BayesCode::state_map nd, nd_train;
    nd["X"] = arma::vec(arma::vectorise(Xn));
    nd_train["X"] = arma::vec(arma::vectorise(X));
    const int Ci = static_cast<int>(C);

    // C8
    GBartMultinomial mh(X, y, Ci, 20, 5, true, true);
    mh.step(static_cast<int>(K));
    const auto pn = mh.predict_at(nd);
    const arma::mat& R8 = pn.at("r");
    const arma::mat& P8 = pn.at("probs");
    const arma::mat& Y8 = pn.at("y_rep");
    const std::vector<std::string> th = mh.get_tree_history();   // class-major
    GBartMultinomial probe(X, y, Ci, 20, 99, false, false);
    bool rows_ok = th.size() == (C - 1) * K;
    for (std::size_t dd = 0; rows_ok && dd < K; ++dd) {
        probe.set_tree(th[dd] + sep + th[K + dd]);
        const auto q = probe.predict_at(nd, true);
        rows_ok = same_bits(arma::mat(R8.row(dd)), q.at("r")) &&
                  same_bits(arma::mat(P8.row(dd)), q.at("probs"));
    }
    double T8 = 0.0, V8 = 0.0;
    bool labels = true;
    for (std::size_t dd = 0; dd < Y8.n_rows; ++dd)
        for (std::size_t i = 0; i < Nn; ++i) {
            const double yv = Y8(dd, i);
            labels = labels && (yv == 0.0 || yv == 1.0 || yv == 2.0);
            double s1 = 0.0, s2 = 0.0;
            for (std::size_t c = 0; c < C; ++c) {
                const double pc = P8(dd, i * C + c), w = pc - 1.0 / C;
                T8 += w * ((yv == static_cast<double>(c) ? 1.0 : 0.0) - pc);
                s1 += w * w * pc;
                s2 += w * pc;
            }
            V8 += s1 - s2 * s2;
        }
    const auto pl = mh.predict_at(nd, true);
    const bool last8 = R8.n_rows == K && same_bits(arma::mat(R8.row(K - 1)), pl.at("r")) &&
                       same_bits(arma::mat(P8.row(K - 1)), pl.at("probs"));
    std::printf("C8 history, new X: r %d x %d, probs %d x %d, y_rep %d x %d; rows = stored "
                "forests: %d; y_rep T/sd = %.3f; last row = last_draw_only: %d\n",
                (int)R8.n_rows, (int)R8.n_cols, (int)P8.n_rows, (int)P8.n_cols,
                (int)Y8.n_rows, (int)Y8.n_cols, rows_ok ? 1 : 0, T8 / std::sqrt(V8),
                last8 ? 1 : 0);
    check(R8.n_rows == K && R8.n_cols == Nn * (C - 1) && P8.n_rows == K &&
          P8.n_cols == Nn * C && Y8.n_rows == K && Y8.n_cols == Nn && rows_ok,
          "C8 row d of r and probs comes from the forests stored at sweep d");
    check(labels && std::abs(T8 / std::sqrt(V8)) < 5.0,
          "C8 y_rep of row d is drawn from Categorical(probs of row d)");
    check(last8, "C8 the last row equals the last_draw_only prediction");

    // C9
    const arma::mat R9 = mh.predict_at(nd_train).at("r");
    const auto h9 = mh.get_history();
    const arma::mat ref9 = arma::join_rows(h9.at(r_key(1)), h9.at(r_key(2)));
    const double e9 = max_rel(R9, ref9);
    std::printf("C9 training X as new data: max rel |r - get_history() r_1, r_2| = %.3g\n", e9);
    check(R9.n_rows == K && e9 <= 1e-12,
          "C9 the forests stored at sweep d reproduce the stored r_j of draw d");

    // C10
    GBartMultinomial mn(X, y, Ci, 20, 5, false, true);
    mn.step(3);
    std::string msg;
    try { mn.predict_at(nd); } catch (const std::exception& e) { msg = e.what(); }
    const arma::mat R10 = mn.predict_at(nd, true).at("r");
    check(msg.find("keep_tree = TRUE") != std::string::npos && R10.n_rows == 1 &&
          R10.n_cols == Nn * (C - 1),
          "C10 without keep_tree the history-mode prediction at new X names keep_tree = TRUE");

    // C11: the models are built and stepped one after the other (they share
    // the thread's kernel stream, which each constructor seeds).
    GBartMultinomial mA(X, y, Ci, 20, 5, true, true);
    mA.step(static_cast<int>(K1));
    mA.clear_history();
    mA.step(static_cast<int>(K2));
    const auto pA = mA.predict_at(nd);
    GBartMultinomial mB(X, y, Ci, 20, 5, true, true);
    mB.step(static_cast<int>(K1 + K2));
    const auto pB = mB.predict_at(nd);
    const arma::mat& RA = pA.at("r");
    const arma::mat& RB = pB.at("r");
    const bool tail11 = RA.n_rows == K2 && RB.n_rows == K1 + K2 &&
                        same_bits(RA, arma::mat(RB.rows(K1, K1 + K2 - 1))) &&
                        same_bits(pA.at("probs"), arma::mat(pB.at("probs").rows(K1, K1 + K2 - 1))) &&
                        pA.at("y_rep").n_rows == K2;
    std::printf("C11 clear_history(): rows %d (model without the call: %d)\n",
                (int)RA.n_rows, (int)RB.n_rows);
    check(tail11, "C11 after clear_history() the rows are the draws of the later sweeps");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

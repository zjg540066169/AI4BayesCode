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

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_PehSharedFrailty.cpp
//
//  examples/PehSharedFrailty.cpp: w (frailties, child 0) and lambda (hazard
//  steps, child 1) are block states; "expanded_w" = w[z] is derived from w
//  and is the offset of lambda's conditional.
//    C1  after set_current of w and lambda, both blocks hold the loaded
//        values (the fallback rows of get_history()), and get_current()
//        reports expanded_w = w[z] of the loaded w;
//    C2  with "w" frozen, after set_current(w) and 5 sweeps, expanded_w
//        equals w[z] of the loaded w;
//    C3  set_current with a valid lambda and a w of the wrong length
//        throws and leaves lambda, w and expanded_w unchanged; so does a w
//        with a non-positive entry;
//    C4  set_current(get_current()) leaves lambda, w and expanded_w
//        unchanged;
//    C5  predict_at({t: query times, g}) returns S with one row per stored
//        draw (keep_history = true); row d equals exp(-w_d[g] H_0(t;
//        lambda_d)) for lambda_d and w_d of stored draw d (relative
//        tolerance 1e-14); predict_at(new_data, true), and a keep_history =
//        false model, return one row (1 x n_q) from the current draw;
//        predict_at({}) returns no keys;
//    C6  g that is not an integer in [0, G) (negative, G, 1.5, NaN, two
//        values, no value) throws an error naming g and G, on both paths;
//        g = 0 and g = G - 1 are accepted;
//    C7  clear_history() with S as a per-draw buffer
//        (clear_history_contract.hpp): after k1 sweeps, clear_history() and
//        k2 sweeps, S has k2 rows and equals the last k2 rows of a run
//        without the call.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main peh_shared_frailty_demo_main
#include "../examples/PehSharedFrailty.cpp"
#undef main

#include "clear_history_contract.hpp"

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> PehSharedFrailty::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &PehSharedFrailty::impl_>;

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

arma::vec expand(const arma::vec& w, const arma::vec& z) {
    arma::vec e(z.n_elem);
    for (std::size_t i = 0; i < z.n_elem; ++i) e[i] = w[static_cast<std::size_t>(z[i])];
    return e;
}

bool throws(PehSharedFrailty& m, const AI4BayesCode::state_map& s) {
    try { m.set_current(s); } catch (const std::exception&) { return true; }
    return false;
}

bool same_mat(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

// S(t_q | g) = exp(-w[g] H_0(t_q)) for each row of lambda_rows and w_rows.
arma::mat S_ref(const arma::mat& lambda_rows, const arma::mat& w_rows, std::size_t g,
                const arma::vec& tq, const arma::vec& edges) {
    arma::mat S(lambda_rows.n_rows, tq.n_elem);
    for (std::size_t d = 0; d < lambda_rows.n_rows; ++d)
        for (std::size_t i = 0; i < tq.n_elem; ++i) {
            double H = 0.0;
            for (std::size_t k = 0; k + 1 < edges.n_elem; ++k) {
                const double dur = std::min(tq[i], edges[k + 1]) - edges[k];
                if (dur > 0.0) H += lambda_rows(d, k) * dur;
            }
            S(d, i) = std::exp(-w_rows(d, g) * H);
        }
    return S;
}

double max_rel(const arma::mat& a, const arma::mat& b) {
    if (a.n_rows != b.n_rows || a.n_cols != b.n_cols) return INFINITY;
    double e = 0.0;
    for (std::size_t i = 0; i < a.n_elem; ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::max(1.0, std::abs(b[i])));
    return e;
}

// The error message of predict_at, or "" when it returns.
std::string predict_error(const PehSharedFrailty& m, const AI4BayesCode::state_map& nd,
                          bool last_draw_only) {
    try { m.predict_at(nd, last_draw_only); } catch (const std::exception& e) { return e.what(); }
    return "";
}

// C7: a keep_history model whose per-draw buffers are get_history() and S.
clear_history_check::model contract_model(const arma::vec& t, const arma::vec& delta,
                                          const arma::vec& z, int G, const arma::vec& edges,
                                          const AI4BayesCode::state_map& nd) {
    auto w = std::make_shared<PehSharedFrailty>(t, delta, z, G, edges, 2.0, 0.01, 0.01, 13, true);
    PehSharedFrailty* p = w.get();
    clear_history_check::model c;
    c.step    = [p](int n) { p->step(n); };
    c.clear   = [p] { p->clear_history(); };
    c.history = [p](clear_history_check::buffers& b) {
        for (const auto& kv : p->get_history()) b.mats[kv.first] = kv.second;
    };
    c.extra   = [p, nd](clear_history_check::buffers& b) {
        for (const auto& kv : p->predict_at(nd)) b.mats["predict_at " + kv.first] = kv.second;
    };
    c.owners.push_back(w);
    return c;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_PehSharedFrailty ===\n");
    const std::size_t G = 5, n_per = 40, n = G * n_per;
    const arma::vec edges({0.0, 2.0, 5.0, 30.0});
    std::mt19937_64 rng(53u);
    std::exponential_distribution<double> ex(0.8);
    arma::vec t(n), delta(n), z(n);
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = static_cast<double>(i / n_per);
        t[i] = std::min(ex(rng) * (1.0 + 0.3 * z[i]), 25.0);
        delta[i] = (t[i] < 25.0) ? 1.0 : 0.0;
    }
    const arma::vec w_new{0.5, 1.5, 2.0, 0.8, 1.1};
    const arma::vec lam_new{0.3, 0.6, 0.9};

    // C1
    PehSharedFrailty m(t, delta, z, static_cast<int>(G), edges, 2.0, 0.01, 0.01, 7, false);
    m.step(5);
    AI4BayesCode::state_map s;
    s["w"] = w_new;
    s["lambda"] = lam_new;
    m.set_current(s);
    const auto h = m.get_history();
    const auto c = m.get_current();
    std::printf("C1 set_current(w, lambda): history w[0] %.3f, expanded_w[0] %.3f (loaded w[0] %.3f)\n",
                h.at("w")(0, 0), c.at("expanded_w")[0], w_new[0]);
    check(same_bits(arma::vec(h.at("w").row(0).t()), w_new) &&
          same_bits(arma::vec(h.at("lambda").row(0).t()), lam_new),
          "C1 both blocks hold the loaded values");
    check(same_bits(c.at("expanded_w"), expand(w_new, z)),
          "C1 get_current() reports expanded_w = w[z] of the loaded w");

    // C2
    PehSharedFrailty m2(t, delta, z, static_cast<int>(G), edges, 2.0, 0.01, 0.01, 9, false);
    m2.step(5);
    (m2.*member_of(impl_tag{}))->freeze(std::vector<std::string>{"w"}, true);
    AI4BayesCode::state_map s2;
    s2["w"] = w_new;
    m2.set_current(s2);
    m2.step(5);
    check(same_bits(m2.get_current().at("expanded_w"), expand(w_new, z)),
          "C2 with w frozen, expanded_w equals w[z] of the loaded w");

    // C3
    const auto before = m.get_current();
    AI4BayesCode::state_map bad;
    bad["lambda"] = arma::vec{9.0, 9.0, 9.0};
    bad["w"] = arma::vec{1.0, 1.0};
    const bool t3a = throws(m, bad);
    AI4BayesCode::state_map bad2;
    bad2["w"] = arma::vec{1.0, -1.0, 1.0, 1.0, 1.0};
    const bool t3b = throws(m, bad2);
    const auto after = m.get_current();
    check(t3a && t3b && same_bits(before.at("lambda"), after.at("lambda")) &&
          same_bits(before.at("w"), after.at("w")) &&
          same_bits(before.at("expanded_w"), after.at("expanded_w")),
          "C3 a failing set_current leaves lambda, w and expanded_w unchanged");

    // C4
    m.step(3);
    const auto b4 = m.get_current();
    m.set_current(b4);
    const auto a4 = m.get_current();
    check(same_bits(b4.at("lambda"), a4.at("lambda")) && same_bits(b4.at("w"), a4.at("w")) &&
          same_bits(b4.at("expanded_w"), a4.at("expanded_w")),
          "C4 set_current(get_current()) changes nothing");

    // C5
    const std::size_t g = 3;
    AI4BayesCode::state_map nd;
    nd["t"] = arma::vec{0.5, 1.0, 2.0, 4.5, 7.0, 31.0};
    nd["g"] = arma::vec{static_cast<double>(g)};
    PehSharedFrailty h5(t, delta, z, static_cast<int>(G), edges, 2.0, 0.01, 0.01, 11, true);
    h5.step(20);
    {
        const auto hist = h5.get_history();
        const auto p = h5.predict_at(nd);
        const auto p_last = h5.predict_at(nd, true);
        const bool shape_ok = p.size() == 1 && p.count("S") && p.at("S").n_rows == 20 &&
                              p.at("S").n_cols == 6;
        const double e = shape_ok
            ? max_rel(p.at("S"), S_ref(hist.at("lambda"), hist.at("w"), g, nd.at("t"), edges))
            : INFINITY;
        std::printf("C5 history mode: S %llux%llu, max rel error vs exp(-w_d[g] H_0(t; lambda_d)) = %.3g\n",
                    static_cast<unsigned long long>(shape_ok ? p.at("S").n_rows : 0),
                    static_cast<unsigned long long>(shape_ok ? p.at("S").n_cols : 0), e);
        check(shape_ok && e <= 1e-14, "C5 one row per stored draw, row d from lambda_d and w_d[g]");
        check(p_last.count("S") && same_mat(p_last.at("S"), arma::mat(p.at("S").row(19))),
              "C5 last_draw_only = true: one row = the last row of the history-mode output");
        check(h5.predict_at(AI4BayesCode::state_map{}).empty(), "C5 predict_at({}) returns no keys");
        const auto p0 = m.predict_at(nd);   // keep_history = false
        const arma::mat lam_cur = m.get_current().at("lambda").t();
        const arma::mat w_cur = m.get_current().at("w").t();
        check(p0.count("S") && p0.at("S").n_rows == 1 &&
              max_rel(p0.at("S"), S_ref(lam_cur, w_cur, g, nd.at("t"), edges)) <= 1e-14,
              "C5 keep_history = false: S is 1 x n_q from the current lambda and w");
    }

    // C6
    {
        bool all_named = true;
        for (const arma::vec& gv : {arma::vec{-1.0}, arma::vec{static_cast<double>(G)},
                                    arma::vec{1.5}, arma::vec{NAN}, arma::vec{0.0, 1.0},
                                    arma::vec()}) {
            AI4BayesCode::state_map bad = nd;
            bad["g"] = gv;
            for (bool last : {false, true}) {
                const std::string e = predict_error(h5, bad, last);
                const bool named = e.find("g") != std::string::npos &&
                                   e.find("G = " + std::to_string(G)) != std::string::npos;
                if (!named) std::printf("  g of length %llu (last_draw_only = %d): '%s'\n",
                                        static_cast<unsigned long long>(gv.n_elem),
                                        last ? 1 : 0, e.c_str());
                all_named = all_named && named;
            }
        }
        AI4BayesCode::state_map ok0 = nd, ok1 = nd, neg = nd;
        ok0["g"] = arma::vec{0.0};
        ok1["g"] = arma::vec{static_cast<double>(G - 1)};
        neg["g"] = arma::vec{-1.0};
        std::printf("C6 g = -1: '%s'\n", predict_error(h5, neg, false).c_str());
        check(all_named, "C6 g outside the integers 0..G-1 throws an error naming g and G");
        check(predict_error(h5, ok0, false).empty() && predict_error(h5, ok1, true).empty(),
              "C6 g = 0 and g = G - 1 are accepted");
    }

    // C7
    clear_history_check::run("C7 clear_history() and predict_at",
                             [&] { return contract_model(t, delta, z, static_cast<int>(G),
                                                         edges, nd); }, 40, 30);

    const int n_fail = g_fail + clear_history_check::failures();
    std::printf("%s (%d failure%s)\n", n_fail ? "FAILED" : "ALL PASSED", n_fail,
                n_fail == 1 ? "" : "s");
    return n_fail ? 1 : 0;
}

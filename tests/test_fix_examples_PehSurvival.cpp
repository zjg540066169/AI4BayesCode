// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_PehSurvival.cpp
//
//  examples/PehSurvival.cpp: lambda (hazard steps) is the state of the
//  piecewise-exponential block.
//    C1  after set_current(lambda), the block holds the loaded lambda (the
//        fallback row of get_history()) and get_current() reports it;
//    C2  with keep_history and "lambda" frozen, the rows recorded after
//        set_current(lambda) equal the loaded lambda;
//    C3  set_current with a lambda of the wrong length, or with a
//        non-positive entry, throws and leaves lambda unchanged;
//    C4  set_current(get_current()) leaves lambda unchanged;
//    C5  predict_at({t: query times}) returns S with one row per stored
//        draw (keep_history = true); row d equals exp(-H_0(t; lambda_d))
//        for lambda_d of stored draw d (relative tolerance 1e-14);
//        predict_at(new_data, true), and a keep_history = false model,
//        return one row (1 x n_q) from the current lambda; predict_at({})
//        returns no keys;
//    C6  clear_history() with S as a per-draw buffer
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

#define main peh_survival_demo_main
#include "../examples/PehSurvival.cpp"
#undef main

#include "clear_history_contract.hpp"

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> PehSurvival::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &PehSurvival::impl_>;

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

bool throws(PehSurvival& m, const AI4BayesCode::state_map& s) {
    try { m.set_current(s); } catch (const std::exception&) { return true; }
    return false;
}

bool same_mat(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

// S(t_q) = exp(-H_0(t_q)) for each row of lambda_rows (one draw per row).
arma::mat S_ref(const arma::mat& lambda_rows, const arma::vec& tq, const arma::vec& edges) {
    arma::mat S(lambda_rows.n_rows, tq.n_elem);
    for (std::size_t d = 0; d < lambda_rows.n_rows; ++d)
        for (std::size_t i = 0; i < tq.n_elem; ++i) {
            double H = 0.0;
            for (std::size_t k = 0; k + 1 < edges.n_elem; ++k) {
                const double dur = std::min(tq[i], edges[k + 1]) - edges[k];
                if (dur > 0.0) H += lambda_rows(d, k) * dur;
            }
            S(d, i) = std::exp(-H);
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

// C6: a keep_history model whose per-draw buffers are get_history() and S.
clear_history_check::model contract_model(const arma::vec& t, const arma::vec& delta,
                                          const arma::vec& edges,
                                          const AI4BayesCode::state_map& nd) {
    auto w = std::make_shared<PehSurvival>(t, delta, edges, 0.01, 0.01, 13, true);
    PehSurvival* p = w.get();
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
    std::printf("=== test_fix_examples_PehSurvival ===\n");
    const std::size_t n = 200;
    const arma::vec edges({0.0, 2.0, 5.0, 30.0});
    std::mt19937_64 rng(59u);
    std::exponential_distribution<double> ex(0.8);
    arma::vec t(n), delta(n);
    for (std::size_t i = 0; i < n; ++i) {
        t[i] = std::min(ex(rng), 25.0);
        delta[i] = (t[i] < 25.0) ? 1.0 : 0.0;
    }
    const arma::vec lam_new{0.3, 0.6, 0.9};

    // C1
    PehSurvival m(t, delta, edges, 0.01, 0.01, 7, false);
    m.step(5);
    AI4BayesCode::state_map s;
    s["lambda"] = lam_new;
    m.set_current(s);
    const arma::vec hrow = m.get_history().at("lambda").row(0).t();
    std::printf("C1 set_current(lambda): block lambda[0] %.3f, reported %.3f (loaded %.3f)\n",
                hrow[0], m.get_current().at("lambda")[0], lam_new[0]);
    check(same_bits(hrow, lam_new) && same_bits(m.get_current().at("lambda"), lam_new),
          "C1 the block holds the loaded lambda");

    // C2
    PehSurvival m2(t, delta, edges, 0.01, 0.01, 9, true);
    m2.step(5);
    (m2.*member_of(impl_tag{}))->freeze(std::vector<std::string>{"lambda"}, true);
    m2.set_current(s);
    m2.step(3);
    const arma::mat H = m2.get_history().at("lambda");
    bool rows_ok = H.n_rows == 8;
    for (std::size_t r = 5; rows_ok && r < 8; ++r)
        rows_ok = same_bits(arma::vec(H.row(r).t()), lam_new);
    check(rows_ok, "C2 with lambda frozen, the recorded rows are the loaded lambda");

    // C3
    const arma::vec before = m.get_current().at("lambda");
    AI4BayesCode::state_map bad1, bad2;
    bad1["lambda"] = arma::vec{1.0, 1.0};
    bad2["lambda"] = arma::vec{1.0, 0.0, 1.0};
    const bool t1 = throws(m, bad1), t2 = throws(m, bad2);
    check(t1 && t2 && same_bits(before, m.get_current().at("lambda")) &&
          same_bits(before, arma::vec(m.get_history().at("lambda").row(0).t())),
          "C3 an invalid lambda is rejected and nothing is written");

    // C4
    m.step(2);
    const auto b4 = m.get_current();
    m.set_current(b4);
    check(same_bits(b4.at("lambda"), m.get_current().at("lambda")),
          "C4 set_current(get_current()) changes nothing");

    // C5
    AI4BayesCode::state_map nd;
    nd["t"] = arma::vec{0.5, 1.0, 2.0, 4.5, 7.0, 31.0};
    {
        PehSurvival h5(t, delta, edges, 0.01, 0.01, 11, true);
        h5.step(20);
        const arma::mat L = h5.get_history().at("lambda");
        const auto p = h5.predict_at(nd);
        const auto p_last = h5.predict_at(nd, true);
        const bool shape_ok = p.size() == 1 && p.count("S") && p.at("S").n_rows == 20 &&
                              p.at("S").n_cols == 6;
        const double e = shape_ok ? max_rel(p.at("S"), S_ref(L, nd.at("t"), edges)) : INFINITY;
        std::printf("C5 history mode: S %llux%llu, max rel error vs exp(-H_0(t; lambda_d)) = %.3g\n",
                    static_cast<unsigned long long>(shape_ok ? p.at("S").n_rows : 0),
                    static_cast<unsigned long long>(shape_ok ? p.at("S").n_cols : 0), e);
        check(shape_ok && e <= 1e-14, "C5 one row per stored draw, row d from lambda_d");
        check(p_last.count("S") && same_mat(p_last.at("S"), arma::mat(p.at("S").row(19))),
              "C5 last_draw_only = true: one row = the last row of the history-mode output");
        check(h5.predict_at(AI4BayesCode::state_map{}).empty(), "C5 predict_at({}) returns no keys");
        const auto p0 = m.predict_at(nd);   // keep_history = false
        const arma::mat cur = m.get_current().at("lambda").t();
        check(p0.count("S") && p0.at("S").n_rows == 1 &&
              max_rel(p0.at("S"), S_ref(cur, nd.at("t"), edges)) <= 1e-14,
              "C5 keep_history = false: S is 1 x n_q from the current lambda");
    }

    // C6
    clear_history_check::run("C6 clear_history() and predict_at",
                             [&] { return contract_model(t, delta, edges, nd); }, 40, 30);

    const int n_fail = g_fail + clear_history_check::failures();
    std::printf("%s (%d failure%s)\n", n_fail ? "FAILED" : "ALL PASSED", n_fail,
                n_fail == 1 ? "" : "s");
    return n_fail ? 1 : 0;
}

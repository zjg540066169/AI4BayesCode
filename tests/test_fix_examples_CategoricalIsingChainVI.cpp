// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_CategoricalIsingChainVI.cpp
//
//  examples/CategoricalIsingChainVI.cpp: the VI log-density holds beta and
//  h from construction, and get_current() reports the shared "beta" and
//  "h". set_current must not report a beta or h that the optimisation does
//  not use.
//    C1  set_current with another beta throws and leaves the shared beta
//        and phi unchanged;
//    C2  set_current with another h throws and leaves the shared h
//        unchanged;
//    C3  set_current with a valid phi and another h throws and leaves phi
//        unchanged;
//    C4  set_current(get_current()) is accepted and leaves phi, beta and h
//        unchanged.
//
//  predict_at. B: keep_history, nb + nk sweeps; A: the same model with
//  clear_history() after nb sweeps; C: keep_history off, nb + nk sweeps.
//    P1  B.predict_at({}) returns only z_samples, (nb + nk) x n, equal bit for
//        bit to get_history()["z"] (the stored q-draws), entries in 0..K-1;
//    P2  a second call and predict_at({}, false) return the same matrix;
//    P3  A.predict_at({}) has 0 rows (n columns) right after clear_history();
//        after nk more sweeps it equals the last nk rows of B's;
//    P4  predict_at({n_draws}) on the history path throws;
//    P5  B.predict_at(nd, true) equals C.predict_at(nd) bit for bit
//        (n_draws = 7, then the default 100 rows): the single-draw path is
//        the keep_history-off path, and the history-path calls before it
//        did not advance predict_rng_;
//    P6  get_history() and get_current() of B are unchanged by the calls.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main categorical_ising_chain_vi_demo_main
#include "../examples/CategoricalIsingChainVI.cpp"
#undef main

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

bool throws(CategoricalIsingChainVI& m, const AI4BayesCode::state_map& s) {
    try { m.set_current(s); } catch (const std::exception&) { return true; }
    return false;
}

bool same_bits(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

bool same_maps(const AI4BayesCode::history_map& a, const AI4BayesCode::history_map& b) {
    if (a.size() != b.size()) return false;
    for (const auto& kv : a) {
        const auto it = b.find(kv.first);
        if (it == b.end() || !same_bits(kv.second, it->second)) return false;
    }
    return true;
}

bool same_maps(const AI4BayesCode::state_map& a, const AI4BayesCode::state_map& b) {
    if (a.size() != b.size()) return false;
    for (const auto& kv : a) {
        const auto it = b.find(kv.first);
        if (it == b.end() || !same_bits(kv.second, it->second)) return false;
    }
    return true;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_CategoricalIsingChainVI ===\n");
    const int n = 4, K = 3;
    const arma::vec h(n, arma::fill::value(2.0));
    CategoricalIsingChainVI m(n, K, 0.8, h, true, 20260621, false);
    m.step(300);

    // C1
    const auto c0 = m.get_current();
    AI4BayesCode::state_map s1;
    s1["beta"] = arma::vec{1.6};
    const bool t1 = throws(m, s1);
    const auto c1 = m.get_current();
    std::printf("C1 set_current(beta = 1.6): threw = %d, reported beta = %.2f\n",
                t1 ? 1 : 0, c1.at("beta")[0]);
    check(t1 && same_bits(c0.at("beta"), c1.at("beta")) && same_bits(c0.at("phi"), c1.at("phi")),
          "C1 another beta is rejected and the reported beta is the one in use");

    // C2
    AI4BayesCode::state_map s2;
    s2["h"] = arma::vec(n, arma::fill::value(-1.0));
    const bool t2 = throws(m, s2);
    check(t2 && same_bits(c0.at("h"), m.get_current().at("h")),
          "C2 another h is rejected and the reported h is the one in use");

    // C3
    arma::vec phi_new(n * K, arma::fill::value(1.0 / K));
    AI4BayesCode::state_map s3;
    s3["phi"] = phi_new;
    s3["h"]   = arma::vec(n, arma::fill::value(-1.0));
    const bool t3 = throws(m, s3);
    check(t3 && same_bits(c0.at("phi"), m.get_current().at("phi")),
          "C3 a set_current that fails on h leaves phi unchanged");

    // C4
    const bool t4 = throws(m, m.get_current());
    const auto c4 = m.get_current();
    check(!t4 && same_bits(c0.at("phi"), c4.at("phi")) && same_bits(c0.at("beta"), c4.at("beta")) &&
          same_bits(c0.at("h"), c4.at("h")),
          "C4 set_current(get_current()) is accepted and changes nothing");

    // ---- predict_at --------------------------------------------------------
    const int nb = 40, nk = 30;
    CategoricalIsingChainVI B(n, K, 0.8, h, true, 20260621, true);
    B.step(nb + nk);
    const auto hist_B0 = B.get_history();
    const auto cur_B0  = B.get_current();

    // P1
    const auto pB = B.predict_at(AI4BayesCode::state_map{});
    const arma::mat zB = pB.count("z_samples") ? pB.at("z_samples") : arma::mat();
    bool in_range = zB.n_elem > 0;
    for (arma::uword j = 0; j < zB.n_elem; ++j)
        in_range = in_range && zB[j] >= 0.0 && zB[j] < K && std::floor(zB[j]) == zB[j];
    std::printf("P1 history path: z_samples %llux%llu, get_history()[\"z\"] %llux%llu\n",
                (unsigned long long)zB.n_rows, (unsigned long long)zB.n_cols,
                (unsigned long long)hist_B0.at("z").n_rows,
                (unsigned long long)hist_B0.at("z").n_cols);
    check(pB.size() == 1 && zB.n_rows == static_cast<arma::uword>(nb + nk) &&
          zB.n_cols == static_cast<arma::uword>(n) &&
          same_bits(zB, hist_B0.at("z")) && in_range,
          "P1 z_samples = the stored q-draws, one row per kept sweep");

    // P2
    check(same_bits(B.predict_at(AI4BayesCode::state_map{}).at("z_samples"), zB) &&
          same_bits(B.predict_at(AI4BayesCode::state_map{}, false).at("z_samples"), zB),
          "P2 a second call and predict_at(nd, false) give the same rows");

    // P3
    CategoricalIsingChainVI A(n, K, 0.8, h, true, 20260621, true);
    A.step(nb);
    A.clear_history();
    const arma::mat zA0 = A.predict_at(AI4BayesCode::state_map{}).at("z_samples");
    A.step(nk);
    const arma::mat zA = A.predict_at(AI4BayesCode::state_map{}).at("z_samples");
    std::printf("P3 after clear_history(): %llux%llu; after %d more sweeps: %llux%llu\n",
                (unsigned long long)zA0.n_rows, (unsigned long long)zA0.n_cols, nk,
                (unsigned long long)zA.n_rows, (unsigned long long)zA.n_cols);
    check(zA0.n_rows == 0 && zA0.n_cols == static_cast<arma::uword>(n) &&
          same_bits(zA, arma::mat(zB.rows(nb, nb + nk - 1))),
          "P3 0 rows after clear_history(), then the last nk rows of B");

    // P4
    AI4BayesCode::state_map nd7;
    nd7["n_draws"] = arma::vec{7.0};
    bool t_p4 = false;
    try { B.predict_at(nd7); } catch (const std::exception&) { t_p4 = true; }
    check(t_p4, "P4 n_draws on the history path throws");

    // P5
    CategoricalIsingChainVI C(n, K, 0.8, h, true, 20260621, false);
    C.step(nb + nk);
    try {
        const arma::mat zB7 = B.predict_at(nd7, true).at("z_samples");
        const arma::mat zC7 = C.predict_at(nd7).at("z_samples");
        const arma::mat zB100 = B.predict_at(AI4BayesCode::state_map{}, true).at("z_samples");
        const arma::mat zC100 = C.predict_at(AI4BayesCode::state_map{}).at("z_samples");
        std::printf("P5 last_draw_only: %llux%llu and %llux%llu\n",
                    (unsigned long long)zB7.n_rows, (unsigned long long)zB7.n_cols,
                    (unsigned long long)zB100.n_rows, (unsigned long long)zB100.n_cols);
        check(zB7.n_rows == 7 && zB100.n_rows == 100 && same_bits(zB7, zC7) &&
              same_bits(zB100, zC100),
              "P5 last_draw_only = true gives the keep_history-off q-samples");
    } catch (const std::exception& e) {
        check(false, std::string("P5 last_draw_only = true threw: ") + e.what());
    }

    // P6
    check(same_maps(B.get_history(), hist_B0) && same_maps(B.get_current(), cur_B0),
          "P6 get_history() and get_current() unchanged by predict_at");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

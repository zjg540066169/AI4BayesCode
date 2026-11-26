// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_StructuredPottsVI.cpp
//
//  examples/StructuredPottsVI.cpp: set_current. clique_phi is the state of
//  the structured VI block (child 0); z is a shared value.
//    C1  set_current with a valid clique_phi (uniform on each clique) and a
//        z entry outside 0..K-1 throws and leaves every shared value and the
//        block's current value unchanged (bit for bit);
//    C2  set_current of clique_phi and z is reported by get_current().
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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main structured_potts_vi_demo_main
#include "../examples/StructuredPottsVI.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> StructuredPottsVI::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &StructuredPottsVI::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const StructuredPottsVI& m) {
    return *(m.*member_of(impl_tag{}));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
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
    std::printf("=== test_fix_examples_StructuredPottsVI ===\n");
    arma::mat edges(kEdges.size(), 2);
    arma::vec strengths(kEdges.size());
    for (std::size_t e = 0; e < kEdges.size(); ++e) {
        edges(e, 0) = static_cast<double>(kEdges[e].u) + 1.0;
        edges(e, 1) = static_cast<double>(kEdges[e].v) + 1.0;
        strengths[e] = kEdges[e].beta;
    }
    arma::mat h(kN, kK);
    for (std::size_t i = 0; i < kN; ++i)
        for (std::size_t k = 0; k < kK; ++k) h(i, k) = kH[i][k];
    const std::vector<std::vector<int>> cliques = {{1, 2, 3}, {4, 5, 6}};
    StructuredPottsVI m(static_cast<int>(kN), static_cast<int>(kK), edges, strengths, h,
                        cliques, true, 7, false);
    m.step(50);

    const std::size_t S = m.get_current().at("clique_phi").n_elem;   // 2^3 + 2^3
    const arma::vec phi_uniform(S, arma::fill::value(1.0 / 8.0));

    // C1
    const std::map<std::string, arma::vec> shared0(impl(m).data().values().begin(),
                                                   impl(m).data().values().end());
    const arma::vec phi0 = impl(m).child(0).current();
    AI4BayesCode::state_map s1;
    s1["clique_phi"] = phi_uniform;
    arma::vec z_bad(kN, arma::fill::zeros);
    z_bad[kN - 1] = 5.0;
    s1["z"] = z_bad;
    bool threw = false;
    try { m.set_current(s1); } catch (const std::exception&) { threw = true; }
    bool same = threw && same_bits(impl(m).child(0).current(), phi0) &&
                impl(m).data().values().size() == shared0.size();
    for (const auto& kv : shared0) same = same && same_bits(impl(m).data().get(kv.first), kv.second);
    std::printf("C1 set_current(uniform clique_phi, z with entry 5): threw %d, clique_phi "
                "unchanged %d\n", threw ? 1 : 0,
                same_bits(impl(m).child(0).current(), phi0) ? 1 : 0);
    check(same, "C1 a valid clique_phi with an invalid z throws and changes nothing");

    // C2
    AI4BayesCode::state_map s2;
    s2["clique_phi"] = phi_uniform;
    const arma::vec z2{0.0, 1.0, 0.0, 1.0, 1.0, 0.0};
    s2["z"] = z2;
    m.set_current(s2);
    const auto g2 = m.get_current();
    check(same_bits(g2.at("clique_phi"), phi_uniform) && same_bits(g2.at("z"), z2),
          "C2 the loaded clique_phi and z are reported");

    // ---- predict_at --------------------------------------------------------
    const int nb = 40, nk = 30;
    const arma::uword n = kN;
    StructuredPottsVI B(static_cast<int>(kN), static_cast<int>(kK), edges, strengths, h,
                        cliques, true, 7, true);
    B.step(nb + nk);
    const auto hist_B0 = B.get_history();
    const auto cur_B0  = B.get_current();

    // P1
    const auto pB = B.predict_at(AI4BayesCode::state_map{});
    const arma::mat zB = pB.count("z_samples") ? pB.at("z_samples") : arma::mat();
    bool in_range = zB.n_elem > 0;
    for (arma::uword j = 0; j < zB.n_elem; ++j)
        in_range = in_range && zB[j] >= 0.0 && zB[j] < kK && std::floor(zB[j]) == zB[j];
    std::printf("P1 history path: z_samples %llux%llu, get_history()[\"z\"] %llux%llu\n",
                (unsigned long long)zB.n_rows, (unsigned long long)zB.n_cols,
                (unsigned long long)hist_B0.at("z").n_rows,
                (unsigned long long)hist_B0.at("z").n_cols);
    check(pB.size() == 1 && zB.n_rows == static_cast<arma::uword>(nb + nk) && zB.n_cols == n &&
          same_bits(zB, hist_B0.at("z")) && in_range,
          "P1 z_samples = the stored q-draws, one row per kept sweep");

    // P2
    check(same_bits(B.predict_at(AI4BayesCode::state_map{}).at("z_samples"), zB) &&
          same_bits(B.predict_at(AI4BayesCode::state_map{}, false).at("z_samples"), zB),
          "P2 a second call and predict_at(nd, false) give the same rows");

    // P3
    StructuredPottsVI A(static_cast<int>(kN), static_cast<int>(kK), edges, strengths, h,
                        cliques, true, 7, true);
    A.step(nb);
    A.clear_history();
    const arma::mat zA0 = A.predict_at(AI4BayesCode::state_map{}).at("z_samples");
    A.step(nk);
    const arma::mat zA = A.predict_at(AI4BayesCode::state_map{}).at("z_samples");
    std::printf("P3 after clear_history(): %llux%llu; after %d more sweeps: %llux%llu\n",
                (unsigned long long)zA0.n_rows, (unsigned long long)zA0.n_cols, nk,
                (unsigned long long)zA.n_rows, (unsigned long long)zA.n_cols);
    check(zA0.n_rows == 0 && zA0.n_cols == n &&
          same_bits(zA, arma::mat(zB.rows(nb, nb + nk - 1))),
          "P3 0 rows after clear_history(), then the last nk rows of B");

    // P4
    AI4BayesCode::state_map nd7;
    nd7["n_draws"] = arma::vec{7.0};
    bool t_p4 = false;
    try { B.predict_at(nd7); } catch (const std::exception&) { t_p4 = true; }
    check(t_p4, "P4 n_draws on the history path throws");

    // P5
    StructuredPottsVI C(static_cast<int>(kN), static_cast<int>(kK), edges, strengths, h,
                        cliques, true, 7, false);
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

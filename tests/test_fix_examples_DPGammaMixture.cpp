// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_DPGammaMixture.cpp
//
//  examples/DPGammaMixture.cpp: set_current of pi. The stick-breaking block
//  (child 2) computes its sticks V from the loaded pi and exposes them as
//  "stick_V"; stick_label_swap_block uses stick_V and rebuilds pi from it.
//    C1  after set_current(pi), the shared stick_V equals the block's V and
//        the shared pi equals the loaded pi;
//    C2  with K_trunc = 3 and "pi" frozen, the sweep after set_current(pi =
//        (0.5, 0.3, 0.2)) ends with pi[2] in {0.2, 0.3} (tolerance 1e-12):
//        from sticks V = (0.5, 0.6, 1) of the loaded pi, the swap of slots 0
//        and 1 leaves pi[2] = 0.2 and puts 0.3 or 0.2 in slot 1, the
//        exchange of the weights of slots 1 and 2 moves that value to slot
//        2, and the weight-preserving swap of slots 0 and 1 leaves slot 2
//        unchanged;
//    C3  after construction, the shared cluster_counts equals the counts of
//        the initial z (with "z" frozen, the first sweep runs cluster_params,
//        pi and alpha on it before label_swap recomputes it), and with "z"
//        frozen before the first sweep it equals the counts of z after 3
//        sweeps;
//    C4  set_current(get_current()), which includes K_trunc, is accepted and
//        leaves get_current() unchanged; another K_trunc is rejected;
//    C5  set_current with a valid z (all 2) and alpha = -1 throws and leaves
//        every shared value and every block's current value unchanged (bit
//        for bit); the same for a valid pi and a shape of +Inf (rejected by
//        cluster_atom_block), and for a valid y and a shape of 0;
//    C6  after set_current(z = 1.6 everywhere), the shared z (get_current)
//        equals the block's labels (all 2) and cluster_counts their counts;
//    C7  set_current(pi) with a pi that does not sum to 1, or with an entry
//        <= 0, throws std::invalid_argument, the exception of
//        stick_breaking_block::set_current;
//    C8  set_current with an empty alpha throws and leaves every shared value
//        and every block's current value unchanged.
// ============================================================================
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main dp_gamma_mixture_demo_main
#include "../examples/DPGammaMixture.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> DPGammaMixture::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &DPGammaMixture::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const DPGammaMixture& m) {
    return *(m.*member_of(impl_tag{}));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

// Every shared value and every block's current value.
struct state_t {
    std::map<std::string, arma::vec> shared;
    std::vector<arma::vec> blocks;
};
state_t state_of(const DPGammaMixture& m) {
    state_t s;
    for (const auto& kv : impl(m).data().values()) s.shared.emplace(kv.first, kv.second);
    for (std::size_t i = 0; i < impl(m).n_children(); ++i)
        s.blocks.push_back(impl(m).child(i).current());
    return s;
}
bool same_state(const state_t& a, const state_t& b) {
    if (a.shared.size() != b.shared.size() || a.blocks.size() != b.blocks.size())
        return false;
    for (const auto& kv : a.shared) {
        auto it = b.shared.find(kv.first);
        if (it == b.shared.end() || !same_bits(kv.second, it->second)) return false;
    }
    for (std::size_t i = 0; i < a.blocks.size(); ++i)
        if (!same_bits(a.blocks[i], b.blocks[i])) return false;
    return true;
}
// set_current(s) throws, and the state is unchanged.
bool throws_unchanged(DPGammaMixture& m, const AI4BayesCode::state_map& s) {
    const state_t before = state_of(m);
    bool threw = false;
    try { m.set_current(s); } catch (const std::exception&) { threw = true; }
    return threw && same_state(before, state_of(m));
}

}  // namespace

int main() {
    // Unbuffered, so the lines printed before a crash reach the log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== test_fix_examples_DPGammaMixture ===\n");
    const std::size_t N = 150;
    std::mt19937_64 gen(43u);
    std::gamma_distribution<double> g1(9.0, 1.0 / 3.0), g2(40.0, 1.0 / 2.0);
    arma::vec Y(N);
    for (std::size_t i = 0; i < N; ++i) Y[i] = (i % 3 == 0) ? g2(gen) : g1(gen);

    // C1
    DPGammaMixture m(Y, 3, 7, false);
    m.step(5);
    AI4BayesCode::state_map s;
    s["pi"] = arma::vec{0.5, 0.3, 0.2};
    m.set_current(s);
    const auto outs = impl(m).child(2).current_named_outputs();
    const auto& d = impl(m).data();
    std::printf("C1 set_current(pi): shared stick_V (%.4f, %.4f), block V (%.4f, %.4f)\n",
                d.get("stick_V")[0], d.get("stick_V")[1],
                outs.at("stick_V")[0], outs.at("stick_V")[1]);
    check(same_bits(d.get("stick_V"), outs.at("stick_V")) &&
          same_bits(d.get("pi"), s.at("pi")),
          "C1 shared stick_V equals the sticks of the loaded pi");

    // C2
    DPGammaMixture m2(Y, 3, 9, false);
    m2.step(5);
    impl(m2).freeze(std::vector<std::string>{"pi"}, true);
    m2.set_current(s);
    m2.step(1);
    const arma::vec pi_after = m2.get_current().at("pi");
    const double e2 = std::min(std::abs(pi_after[2] - 0.2), std::abs(pi_after[2] - 0.3));
    std::printf("C2 pi frozen, one sweep after set_current(pi = (0.5, 0.3, 0.2)): pi = "
                "(%.6f, %.6f, %.6f)\n", pi_after[0], pi_after[1], pi_after[2]);
    check(e2 <= 1e-12, "C2 the label swap of the sweep starts from the sticks of the loaded pi");

    // C3
    DPGammaMixture m3(Y, 4, 13, false);
    auto& c3 = impl(m3);
    const bool at_start = same_bits(c3.data().get("cluster_counts"),
                                    AI4BayesCode::bnp::counts_from_z(c3.data().get("z"), 4));
    c3.freeze(std::vector<std::string>{"z"}, true);
    m3.step(3);
    const bool after = same_bits(c3.data().get("cluster_counts"),
                                 AI4BayesCode::bnp::counts_from_z(c3.data().get("z"), 4));
    std::printf("C3 cluster_counts = counts of z: after construction %d, after 3 sweeps with z frozen %d\n",
                at_start ? 1 : 0, after ? 1 : 0);
    check(at_start && after, "C3 cluster_counts holds the counts of z from construction on");

    // C4
    const auto c4 = m3.get_current();
    bool threw4 = false;
    try { m3.set_current(c4); } catch (const std::exception&) { threw4 = true; }
    const auto a4 = m3.get_current();
    bool same4 = !threw4;
    for (const auto& kv : c4) same4 = same4 && same_bits(kv.second, a4.at(kv.first));
    AI4BayesCode::state_map k5;
    k5["K_trunc"] = arma::vec{5.0};
    bool threw5 = false;
    try { m3.set_current(k5); } catch (const std::exception&) { threw5 = true; }
    check(same4 && threw5, "C4 set_current(get_current()) round-trips; another K_trunc is rejected");

    // C5
    DPGammaMixture m5(Y, 3, 7, false);
    m5.step(3);
    AI4BayesCode::state_map s5a, s5b, s5c;
    s5a["z"] = arma::vec(N, arma::fill::value(2.0));
    s5a["alpha"] = arma::vec{-1.0};
    s5b["pi"] = arma::vec{0.5, 0.3, 0.2};
    s5b["shape"] = arma::vec{1.0, std::numeric_limits<double>::infinity(), 1.0};
    s5c["y"] = Y + 1.0;
    s5c["shape"] = arma::vec{1.0, 0.0, 1.0};
    const bool c5a = throws_unchanged(m5, s5a), c5b = throws_unchanged(m5, s5b),
               c5c = throws_unchanged(m5, s5c);
    std::printf("C5 throws and changes nothing: z + alpha -1 %d, pi + shape Inf %d, y + shape 0 %d\n",
                c5a ? 1 : 0, c5b ? 1 : 0, c5c ? 1 : 0);
    check(c5a && c5b && c5c, "C5 a call with an invalid key throws and changes nothing");

    // C6
    AI4BayesCode::state_map s6;
    s6["z"] = arma::vec(N, arma::fill::value(1.6));
    m5.set_current(s6);
    const arma::vec z6 = m5.get_current().at("z");
    const arma::vec& cc6 = impl(m5).data().get("cluster_counts");
    std::printf("C6 set_current(z = 1.6): get_current z[0] %.3g, block z[0] %.3g, cluster_counts (%g, %g, %g)\n",
                z6[0], impl(m5).child(0).current()[0], cc6[0], cc6[1], cc6[2]);
    check(same_bits(z6, impl(m5).child(0).current()) &&
          same_bits(z6, arma::vec(N, arma::fill::value(2.0))) &&
          cc6[0] == 0.0 && cc6[1] == static_cast<double>(N) && cc6[2] == 0.0,
          "C6 the shared z is the block's rounded labels, cluster_counts their counts");

    {
        DPGammaMixture mp(Y, 3, 23, false);
        mp.step(2);
        int n_invalid_argument = 0;
        for (const arma::vec& pi_bad : {arma::vec{0.5, 0.3, 0.3}, arma::vec{1.2, -0.1, -0.1}}) {
            AI4BayesCode::state_map sp;
            sp["pi"] = pi_bad;
            try { mp.set_current(sp); }
            catch (const std::invalid_argument&) { ++n_invalid_argument; }
            catch (const std::exception&) {}
        }
        check(n_invalid_argument == 2, "C7 an invalid pi throws std::invalid_argument");
    }

    {
        DPGammaMixture m8(Y, 3, 21, false);
        m8.step(3);
        AI4BayesCode::state_map s8;
        s8["alpha"] = arma::vec();
        check(throws_unchanged(m8, s8),
              "C8 set_current(alpha = empty) throws and leaves the state unchanged");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

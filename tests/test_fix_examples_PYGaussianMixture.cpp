// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_PYGaussianMixture.cpp
//
//  examples/PYGaussianMixture.cpp: set_current of pi. The stick-breaking block
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
//    C3  set_current with a discount other than the construction value
//        throws and leaves the shared discount and every other key
//        unchanged; the construction value is accepted (round trip);
//    C4  set_current with a valid z (all 2) and alpha = -1 throws and leaves
//        every shared value and every block's current value unchanged (bit
//        for bit); the same for a valid pi and alpha = -1, and for a valid mu
//        and a lambda with a zero entry;
//    C5  after set_current(z = 1.6 everywhere), the shared z (get_current)
//        equals the block's labels (all 2) and cluster_counts their counts;
//    C6  set_current of mu and lambda (K x d, column-major, as get_current
//        reports them) loads the cluster block, whose current value is
//        [mu; lambda] in row-major order, and get_current() reports the
//        loaded values;
//    C7  set_current(get_current()) leaves get_current() unchanged;
//    C8  set_current(pi) with a pi that does not sum to 1, or with an entry
//        <= 0, throws std::invalid_argument, the exception of
//        stick_breaking_block::set_current;
//    C9  set_current with an empty alpha, or an empty discount, throws and
//        leaves every shared value and every block's current value
//        unchanged.
// ============================================================================
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main py_gaussian_mixture_demo_main
#include "../examples/PYGaussianMixture.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> PYGaussianMixture::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &PYGaussianMixture::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const PYGaussianMixture& m) {
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
state_t state_of(const PYGaussianMixture& m) {
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
bool throws_unchanged(PYGaussianMixture& m, const AI4BayesCode::state_map& s) {
    const state_t before = state_of(m);
    bool threw = false;
    try { m.set_current(s); } catch (const std::exception&) { threw = true; }
    return threw && same_state(before, state_of(m));
}

}  // namespace

int main() {
    // Unbuffered, so the lines printed before a crash reach the log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== test_fix_examples_PYGaussianMixture ===\n");
    const std::size_t N = 120;
    std::mt19937_64 gen(43u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::mat Y(N, 2);
    for (std::size_t i = 0; i < N; ++i) {
        const double c = (i % 3 == 0) ? 4.0 : -2.0;
        Y(i, 0) = c + 0.7 * nd(gen);
        Y(i, 1) = -c + 0.7 * nd(gen);
    }

    // C1
    PYGaussianMixture m(Y, 3, 0.25, 7, false);
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
    PYGaussianMixture m2(Y, 3, 0.25, 9, false);
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
    PYGaussianMixture m3(Y, 3, 0.25, 7, false);
    m3.step(3);
    const auto before = m3.get_current();
    AI4BayesCode::state_map s3;
    s3["alpha"]    = arma::vec{2.0};
    s3["discount"] = arma::vec{0.5};
    bool threw = false;
    try { m3.set_current(s3); } catch (const std::exception&) { threw = true; }
    const auto after = m3.get_current();
    check(threw && after.at("discount")[0] == 0.25 &&
          same_bits(before.at("alpha"), after.at("alpha")),
          "C3 another discount is rejected and nothing is written");
    bool threw_same = false;
    try { m3.set_current(m3.get_current()); } catch (const std::exception&) { threw_same = true; }
    check(!threw_same, "C3 set_current(get_current()) is accepted");

    // C4
    PYGaussianMixture m4(Y, 3, 0.25, 7, false);
    m4.step(3);
    AI4BayesCode::state_map s4a, s4b, s4c;
    s4a["z"] = arma::vec(N, arma::fill::value(2.0));
    s4a["alpha"] = arma::vec{-1.0};
    s4b["pi"] = arma::vec{0.5, 0.3, 0.2};
    s4b["alpha"] = arma::vec{-1.0};
    arma::vec lam_bad = m4.get_current().at("lambda");
    lam_bad[1] = 0.0;
    s4c["mu"] = m4.get_current().at("mu") + 1.0;
    s4c["lambda"] = lam_bad;
    const bool c4a = throws_unchanged(m4, s4a), c4b = throws_unchanged(m4, s4b),
               c4c = throws_unchanged(m4, s4c);
    std::printf("C4 throws and changes nothing: z + alpha -1 %d, pi + alpha -1 %d, mu + lambda 0 %d\n",
                c4a ? 1 : 0, c4b ? 1 : 0, c4c ? 1 : 0);
    check(c4a && c4b && c4c, "C4 a call with an invalid key throws and changes nothing");

    // C5
    AI4BayesCode::state_map s5;
    s5["z"] = arma::vec(N, arma::fill::value(1.6));
    m4.set_current(s5);
    const arma::vec z5 = m4.get_current().at("z");
    const arma::vec& cc5 = impl(m4).data().get("cluster_counts");
    std::printf("C5 set_current(z = 1.6): get_current z[0] %.3g, block z[0] %.3g, cluster_counts (%g, %g, %g)\n",
                z5[0], impl(m4).child(0).current()[0], cc5[0], cc5[1], cc5[2]);
    check(same_bits(z5, impl(m4).child(0).current()) &&
          same_bits(z5, arma::vec(N, arma::fill::value(2.0))) &&
          cc5[0] == 0.0 && cc5[1] == static_cast<double>(N) && cc5[2] == 0.0,
          "C5 the shared z is the block's rounded labels, cluster_counts their counts");

    // C6: K = 3, d = 2; column-major input, row-major block state.
    const std::size_t K6 = 3, d6 = 2;
    const arma::vec mu6 = arma::regspace<arma::vec>(1.0, 6.0);
    const arma::vec lam6 = arma::regspace<arma::vec>(2.0, 7.0) / 4.0;
    AI4BayesCode::state_map s6;
    s6["mu"] = mu6;
    s6["lambda"] = lam6;
    m4.set_current(s6);
    const auto g6 = m4.get_current();
    const arma::vec blk6 = impl(m4).child(1).current();
    bool rows_ok = blk6.n_elem == 2 * K6 * d6;
    for (std::size_t k = 0; rows_ok && k < K6; ++k)
        for (std::size_t j = 0; j < d6; ++j)
            rows_ok = rows_ok && blk6[k * d6 + j] == mu6[j * K6 + k] &&
                      blk6[K6 * d6 + k * d6 + j] == lam6[j * K6 + k];
    std::printf("C6 set_current(mu, lambda): get_current mu[0] %.3g lambda[0] %.3g, block mu[0] %.3g\n",
                g6.at("mu")[0], g6.at("lambda")[0], blk6[0]);
    check(rows_ok && same_bits(g6.at("mu"), mu6) && same_bits(g6.at("lambda"), lam6),
          "C6 mu and lambda are loaded into the cluster block and reported");

    // C7
    m4.step(2);
    const auto c7 = m4.get_current();
    m4.set_current(c7);
    const auto a7 = m4.get_current();
    bool same7 = true;
    for (const auto& kv : c7) same7 = same7 && same_bits(kv.second, a7.at(kv.first));
    check(same7, "C7 set_current(get_current()) leaves get_current() unchanged");

    {
        PYGaussianMixture mp(Y, 3, 0.25, 23, false);
        mp.step(2);
        int n_invalid_argument = 0;
        for (const arma::vec& pi_bad : {arma::vec{0.5, 0.3, 0.3}, arma::vec{1.2, -0.1, -0.1}}) {
            AI4BayesCode::state_map sp;
            sp["pi"] = pi_bad;
            try { mp.set_current(sp); }
            catch (const std::invalid_argument&) { ++n_invalid_argument; }
            catch (const std::exception&) {}
        }
        check(n_invalid_argument == 2, "C8 an invalid pi throws std::invalid_argument");
    }

    {
        PYGaussianMixture m9(Y, 3, 0.25, 21, false);
        m9.step(3);
        AI4BayesCode::state_map se;
        se["alpha"] = arma::vec();
        check(throws_unchanged(m9, se), "C9 set_current(alpha = empty) throws and leaves the state unchanged");
    }

    {
        PYGaussianMixture m9(Y, 3, 0.25, 21, false);
        m9.step(3);
        AI4BayesCode::state_map se;
        se["discount"] = arma::vec();
        check(throws_unchanged(m9, se), "C9 set_current(discount = empty) throws and leaves the state unchanged");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_FiniteGaussianMixture.cpp
//
//  examples/FiniteGaussianMixture.cpp: set_current. z is the state of the
//  categorical block (child 0), which rounds a loaded z; mu and lambda are
//  the state of the cluster block (child 1), [mu; lambda] in row-major
//  order; pi is the state of the Dirichlet block (child 2).
//    C1  set_current with a valid z (all 2) and alpha_dir = -1 throws and
//        leaves every shared value and every block's current value
//        unchanged (bit for bit); the same for a valid pi and a lambda with
//        a zero entry, and for a valid y and a pi that does not sum to 1;
//    C2  after set_current(z = 1.6 everywhere), the shared z (get_current)
//        equals the block's labels (all 2) and cluster_counts their counts;
//    C3  set_current of mu and lambda (K x d, column-major, as get_current
//        reports them) loads the cluster block, whose current value is
//        [mu; lambda] in row-major order, and get_current() reports the
//        loaded values;
//    C4  set_current(get_current()) leaves get_current() unchanged;
//    C5  set_current(pi) with a pi that does not sum to 1, or with an entry
//        <= 0, throws std::invalid_argument, the exception of
//        dirichlet_gibbs_block::set_current;
//    C6  set_current with an empty alpha_dir throws and leaves every shared
//        value and every block's current value unchanged.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main finite_gaussian_mixture_demo_main
#include "../examples/FiniteGaussianMixture.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> FiniteGaussianMixture::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &FiniteGaussianMixture::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const FiniteGaussianMixture& m) {
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
state_t state_of(const FiniteGaussianMixture& m) {
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
bool throws_unchanged(FiniteGaussianMixture& m, const AI4BayesCode::state_map& s) {
    const state_t before = state_of(m);
    bool threw = false;
    try { m.set_current(s); } catch (const std::exception&) { threw = true; }
    return threw && same_state(before, state_of(m));
}

}  // namespace

int main() {
    // Unbuffered, so the lines printed before a crash reach the log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== test_fix_examples_FiniteGaussianMixture ===\n");
    const std::size_t N = 120, K = 3, d = 2;
    std::mt19937_64 gen(43u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::mat Y(N, d);
    for (std::size_t i = 0; i < N; ++i) {
        const double c = (i % 3 == 0) ? 4.0 : -2.0;
        Y(i, 0) = c + 0.7 * nd(gen);
        Y(i, 1) = -c + 0.7 * nd(gen);
    }

    // C1
    FiniteGaussianMixture m(Y, static_cast<int>(K), 7, false);
    m.step(3);
    AI4BayesCode::state_map s1a, s1b, s1c;
    s1a["z"] = arma::vec(N, arma::fill::value(2.0));
    s1a["alpha_dir"] = arma::vec{-1.0};
    s1b["pi"] = arma::vec{0.5, 0.3, 0.2};
    arma::vec lam_bad = m.get_current().at("lambda");
    lam_bad[1] = 0.0;
    s1b["lambda"] = lam_bad;
    s1c["y"] = arma::vectorise(Y) + 1.0;
    s1c["pi"] = arma::vec{0.5, 0.3, 0.3};
    const bool c1a = throws_unchanged(m, s1a), c1b = throws_unchanged(m, s1b),
               c1c = throws_unchanged(m, s1c);
    std::printf("C1 throws and changes nothing: z + alpha_dir -1 %d, pi + lambda 0 %d, "
                "y + pi sum 1.1 %d\n", c1a ? 1 : 0, c1b ? 1 : 0, c1c ? 1 : 0);
    check(c1a && c1b && c1c, "C1 a call with an invalid key throws and changes nothing");

    // C2
    AI4BayesCode::state_map s2;
    s2["z"] = arma::vec(N, arma::fill::value(1.6));
    m.set_current(s2);
    const arma::vec z2 = m.get_current().at("z");
    const arma::vec& cc2 = impl(m).data().get("cluster_counts");
    std::printf("C2 set_current(z = 1.6): get_current z[0] %.3g, block z[0] %.3g, "
                "cluster_counts (%g, %g, %g)\n",
                z2[0], impl(m).child(0).current()[0], cc2[0], cc2[1], cc2[2]);
    check(same_bits(z2, impl(m).child(0).current()) &&
          same_bits(z2, arma::vec(N, arma::fill::value(2.0))) &&
          cc2[0] == 0.0 && cc2[1] == static_cast<double>(N) && cc2[2] == 0.0,
          "C2 the shared z is the block's rounded labels, cluster_counts their counts");

    // C3: column-major input, row-major block state.
    const arma::vec mu3 = arma::regspace<arma::vec>(1.0, 6.0);
    const arma::vec lam3 = arma::regspace<arma::vec>(2.0, 7.0) / 4.0;
    AI4BayesCode::state_map s3;
    s3["mu"] = mu3;
    s3["lambda"] = lam3;
    m.set_current(s3);
    const auto g3 = m.get_current();
    const arma::vec blk3 = impl(m).child(1).current();
    bool rows_ok = blk3.n_elem == 2 * K * d;
    for (std::size_t k = 0; rows_ok && k < K; ++k)
        for (std::size_t j = 0; j < d; ++j)
            rows_ok = rows_ok && blk3[k * d + j] == mu3[j * K + k] &&
                      blk3[K * d + k * d + j] == lam3[j * K + k];
    std::printf("C3 set_current(mu, lambda): get_current mu[0] %.3g lambda[0] %.3g, "
                "block mu[0] %.3g\n", g3.at("mu")[0], g3.at("lambda")[0], blk3[0]);
    check(rows_ok && same_bits(g3.at("mu"), mu3) && same_bits(g3.at("lambda"), lam3),
          "C3 mu and lambda are loaded into the cluster block and reported");

    // C4
    m.step(2);
    const auto c4 = m.get_current();
    m.set_current(c4);
    const auto a4 = m.get_current();
    bool same4 = true;
    for (const auto& kv : c4) same4 = same4 && same_bits(kv.second, a4.at(kv.first));
    check(same4, "C4 set_current(get_current()) leaves get_current() unchanged");

    {
        FiniteGaussianMixture mp(Y, static_cast<int>(K), 23, false);
        mp.step(2);
        int n_invalid_argument = 0;
        for (const arma::vec& pi_bad : {arma::vec{0.5, 0.3, 0.3}, arma::vec{1.2, -0.1, -0.1}}) {
            AI4BayesCode::state_map sp;
            sp["pi"] = pi_bad;
            try { mp.set_current(sp); }
            catch (const std::invalid_argument&) { ++n_invalid_argument; }
            catch (const std::exception&) {}
        }
        check(n_invalid_argument == 2, "C5 an invalid pi throws std::invalid_argument");
    }

    {
        FiniteGaussianMixture m6(Y, static_cast<int>(K), 21, false);
        m6.step(3);
        AI4BayesCode::state_map se;
        se["alpha_dir"] = arma::vec();
        check(throws_unchanged(m6, se), "C6 set_current(alpha_dir = empty) throws and leaves the state unchanged");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

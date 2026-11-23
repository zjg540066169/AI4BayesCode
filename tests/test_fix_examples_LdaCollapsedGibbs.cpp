// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_LdaCollapsedGibbs.cpp
//
//  examples/LdaCollapsedGibbs.cpp before the first sweep. The block sets up
//  z, its count tables and its initial theta and phi from w and doc.
//    C1  after construction, the shared z, theta and phi equal the block's
//        outputs: z holds labels in 1..K, every row of theta (M x K) and of
//        phi (K x V) sums to 1 (tolerance 1e-12);
//    C2  predict_at({}) before the first sweep draws tokens from theta and
//        phi: the share of tokens equal to V lies within 0.1 of the
//        probability of V under theta and phi;
//    C3  set_current(z) before the first sweep is accepted and the block
//        holds the loaded z;
//    C4  the first sweeps of a model on which get_current() and
//        predict_at({}) were called before the first sweep equal those of
//        a twin on which they were not called;
//    C5  after 3 sweeps, set_current with a valid z (all 2) and a w entry
//        outside 1..V (with doc), or with a valid z (all 1) and w without
//        doc, throws and leaves every shared value and the block's z
//        unchanged (bit for bit).
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main lda_collapsed_gibbs_demo_main
#include "../examples/LdaCollapsedGibbs.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> LdaCollapsedGibbs::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &LdaCollapsedGibbs::impl_>;

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

// set_current(s) throws, and every shared value and the block's z are
// unchanged.
bool throws_unchanged(LdaCollapsedGibbs& m, const AI4BayesCode::state_map& s) {
    const auto& c = *(m.*member_of(impl_tag{}));
    const std::map<std::string, arma::vec> shared0(c.data().values().begin(),
                                                   c.data().values().end());
    const arma::vec z0 = c.child(0).current();
    bool threw = false;
    try { m.set_current(s); } catch (const std::exception&) { threw = true; }
    bool same = threw && c.data().values().size() == shared0.size() &&
                same_bits(c.child(0).current(), z0);
    for (const auto& kv : shared0) same = same && same_bits(c.data().get(kv.first), kv.second);
    return same;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_LdaCollapsedGibbs ===\n");
    const std::size_t M = 10, V = 6, K = 2, Ld = 20, N = M * Ld;
    std::mt19937_64 g(67u);
    std::uniform_int_distribution<int> uw(1, static_cast<int>(V));
    arma::vec w(N), doc(N);
    for (std::size_t t = 0; t < N; ++t) {
        w[t] = static_cast<double>(uw(g));
        doc[t] = static_cast<double>(t / Ld + 1);
    }
    const arma::vec alpha(K, arma::fill::value(0.5)), beta(V, arma::fill::value(0.1));

    LdaCollapsedGibbs m(w, doc, M, V, K, alpha, beta, 7, false);
    auto& c = *(m.*member_of(impl_tag{}));
    const auto cur = m.get_current();
    const arma::vec& z = cur.at("z");
    const arma::mat theta(cur.at("theta").memptr(), M, K);
    const arma::mat phi(cur.at("phi").memptr(), K, V);
    bool labels_ok = z.n_elem == N;
    for (std::size_t t = 0; labels_ok && t < N; ++t) labels_ok = z[t] >= 1.0 && z[t] <= K;
    const double e_theta = arma::abs(arma::sum(theta, 1) - 1.0).max();
    const double e_phi = arma::abs(arma::sum(phi, 1) - 1.0).max();
    const auto outs = c.child(0).current_named_outputs();
    std::printf("C1 construction: z labels in 1..K = %d, max |row sum - 1| theta %.3g, phi %.3g\n",
                labels_ok ? 1 : 0, e_theta, e_phi);
    check(labels_ok && e_theta <= 1e-12 && e_phi <= 1e-12 && outs.count("z") &&
          same_bits(c.data().get("z"), outs.at("z")) &&
          same_bits(c.data().get("theta"), outs.at("theta")) &&
          same_bits(c.data().get("phi"), outs.at("phi")),
          "C1 z, theta and phi are the block's initial state");

    // C2: P(token = V) = mean over tokens of sum_k theta[d, k] phi[k, V].
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    double p_v = 0.0, share_v = 0.0;
    for (std::size_t t = 0; t < N; ++t) {
        const std::size_t d = static_cast<std::size_t>(doc[t]) - 1;
        for (std::size_t k = 0; k < K; ++k) p_v += theta(d, k) * phi(k, V - 1);
        share_v += (yr[t] == static_cast<double>(V)) ? 1.0 : 0.0;
    }
    p_v /= N; share_v /= N;
    std::printf("C2 predict_at({}) before the first sweep: share of token V %.3f, probability %.3f\n",
                share_v, p_v);
    check(std::abs(share_v - p_v) < 0.1, "C2 y_rep drawn from the initial theta and phi");

    // C3
    LdaCollapsedGibbs m3(w, doc, M, V, K, alpha, beta, 7, false);
    arma::vec z_new(N);
    for (std::size_t t = 0; t < N; ++t) z_new[t] = static_cast<double>(1 + (t % 3 == 0));
    AI4BayesCode::state_map s;
    s["z"] = z_new;
    bool threw = false;
    try { m3.set_current(s); } catch (const std::exception&) { threw = true; }
    check(!threw && same_bits((m3.*member_of(impl_tag{}))->child(0).current(), z_new),
          "C3 set_current(z) before the first sweep loads z into the block");

    // C4
    LdaCollapsedGibbs a(w, doc, M, V, K, alpha, beta, 11, false), b(w, doc, M, V, K, alpha, beta, 11, false);
    (void)a.get_current();
    (void)a.predict_at(AI4BayesCode::state_map{});
    a.step(5); b.step(5);
    const auto ca = a.get_current(), cb = b.get_current();
    check(same_bits(ca.at("z"), cb.at("z")) && same_bits(ca.at("theta"), cb.at("theta")) &&
          same_bits(ca.at("phi"), cb.at("phi")),
          "C4 get_current() and predict_at({}) before the first sweep leave the sweeps unchanged");

    // C5
    LdaCollapsedGibbs m5(w, doc, M, V, K, alpha, beta, 7, false);
    m5.step(3);
    arma::vec w_bad = w;
    w_bad[0] = static_cast<double>(V + 5);
    AI4BayesCode::state_map s5a, s5b;
    s5a["z"] = arma::vec(N, arma::fill::value(2.0));
    s5a["w"] = w_bad;
    s5a["doc"] = doc;
    s5b["z"] = arma::vec(N, arma::fill::value(1.0));
    s5b["w"] = w;
    const bool c5a = throws_unchanged(m5, s5a), c5b = throws_unchanged(m5, s5b);
    std::printf("C5 throws and changes nothing: z + w out of range %d, z + w without doc %d\n",
                c5a ? 1 : 0, c5b ? 1 : 0);
    check(c5a && c5b, "C5 a call with an invalid w or doc throws and changes nothing");

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

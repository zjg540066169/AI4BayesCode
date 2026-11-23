// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_hmm_forward_cache.cpp
//
//  hmm_block reuses its forward filter while the context is bitwise
//  unchanged and recomputes it after any change. Checks:
//   1. With an unchanged context, emission_logp is evaluated T*K times at the
//      first step and not at all at later steps, for the block driven
//      directly and inside composite_block.
//   2. A change of A, of pi, of an emission parameter by one ulp, of the sign
//      of a zero, an added or a removed key, or of y in shared data makes the
//      next step evaluate emission_logp T*K times again.
//   3. Along context changes, repeated set_context calls, set_current, copy
//      and move construction and assignment, and an exception thrown inside
//      the forward pass, every draw and the RNG position equal those of a
//      newly constructed block given the same context and RNG state.
//   4. With two contexts alternating every sweep, the frequencies of the 2^4
//      state paths match the exact posterior under each context.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/hmm_block.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace AI4BayesCode;

static int g_pass = 0, g_fail = 0;
static void report(bool ok, const std::string& what, const std::string& detail) {
    (ok ? g_pass : g_fail)++;
    std::printf("  [%s] %s  %s\n", ok ? "PASS" : "FAIL", what.c_str(), detail.c_str());
}

// Gaussian emission: K means in ctx["mu"], common sd in ctx["sigma"].
static double em_pure(std::size_t t, std::size_t k, const block_context& ctx) {
    const double two_pi = 6.283185307179586;
    const arma::vec& y  = ctx.at("y");
    const arma::vec& mu = ctx.at("mu");
    const double s      = ctx.at("sigma")[0];
    const double r      = y[t] - mu[k];
    return -0.5 * std::log(two_pi) - std::log(s) - 0.5 * (r * r) / (s * s);
}
// Same value; counts evaluations and throws at time g_throw_t when armed.
static long long g_calls = 0;
static bool g_armed = false;
static std::size_t g_throw_t = 0;
static double em_count(std::size_t t, std::size_t k, const block_context& ctx) {
    ++g_calls;
    if (g_armed && t == g_throw_t) throw std::runtime_error("emission failure");
    return em_pure(t, k, ctx);
}

static hmm_block_config make_cfg(std::size_t T, std::size_t K,
                                 double (*em)(std::size_t, std::size_t, const block_context&)) {
    hmm_block_config cfg;
    cfg.name = "z"; cfg.T = T; cfg.K = K;
    cfg.emission_logp = em;
    return cfg;
}
static arma::vec sticky_A(std::size_t K, double stay) {
    arma::vec A(K * K);
    for (std::size_t j = 0; j < K; ++j)
        for (std::size_t k = 0; k < K; ++k)
            A[j * K + k] = (j == k) ? stay : (1.0 - stay) / double(K - 1);
    return A;
}
static block_context base_ctx(std::size_t T, std::size_t K) {
    std::mt19937_64 r(5);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::vec y(T), mu(K);
    for (std::size_t t = 0; t < T; ++t) y[t] = 2.0 * double((t / 6) % K) - double(K - 1) + nd(r);
    for (std::size_t k = 0; k < K; ++k) mu[k] = 2.0 * double(k) - double(K - 1);
    block_context c;
    c["y"] = y; c["A"] = sticky_A(K, 0.85);
    c["pi"] = arma::vec(K, arma::fill::value(1.0 / double(K)));
    c["mu"] = mu; c["sigma"] = arma::vec{1.0};
    return c;
}
static bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           (a.n_elem == 0 || std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0);
}

static const std::size_t kT = 50, kK = 3;

// ---- 1 and 2: evaluation counts ---------------------------------------------
static void check_counts_direct() {
    const block_context c1 = base_ctx(kT, kK);
    hmm_block b(make_cfg(kT, kK, &em_count));
    std::mt19937_64 rng(1);
    b.set_context(c1);
    g_calls = 0; b.step(rng);
    const long long first = g_calls;
    long long later = 0;
    for (int i = 0; i < 9; ++i) {
        block_context same = c1;              // equal content, separate copy
        b.set_context(same);
        g_calls = 0; b.step(rng); later += g_calls;
    }
    report(first == (long long)(kT * kK) && later == 0,
           "direct: forward filter reused while the context is unchanged",
           "first step " + std::to_string(first) + " evaluations (expect " +
           std::to_string(kT * kK) + "), next 9 steps " + std::to_string(later) + " (expect 0)");

    // Each change below must trigger one new forward pass.
    std::vector<std::pair<std::string, block_context>> changes;
    { block_context c = c1; c["A"] = sticky_A(kK, 0.6);                       changes.push_back({"A", c}); }
    { block_context c = c1; c["pi"] = arma::vec{0.7, 0.2, 0.1};               changes.push_back({"pi", c}); }
    { block_context c = c1; c["mu"][1] = std::nextafter(c["mu"][1], 1e300);   changes.push_back({"mu + 1 ulp", c}); }
    { block_context c = c1; c["mu"][1] = 0.0;                                 changes.push_back({"mu[1] = +0", c}); }
    { block_context c = c1; c["mu"][1] = -0.0;                                changes.push_back({"mu[1] = -0", c}); }
    { block_context c = c1; c["extra"] = arma::vec{1.0};                      changes.push_back({"key added", c}); }
    { block_context c = c1;                                                   changes.push_back({"key removed", c}); }
    { block_context c = c1; c["y"][kT - 1] += 1.0;                            changes.push_back({"y[T-1]", c}); }
    bool ok = true; std::string detail;
    for (const auto& ch : changes) {
        b.set_context(ch.second);
        g_calls = 0; b.step(rng);
        const long long n1 = g_calls;
        if (n1 != (long long)(kT * kK)) { ok = false; detail += ch.first + ":" + std::to_string(n1) + " "; }
    }
    report(ok, "direct: every context change triggers a new forward pass",
           detail.empty() ? std::to_string(changes.size()) + " changes, " +
                            std::to_string(kT * kK) + " evaluations each" : detail);
}

static void check_counts_composite() {
    const block_context c1 = base_ctx(kT, kK);
    composite_block c("hmm");
    for (const auto& kv : c1) c.data().set(kv.first, kv.second);
    c.data().declare_dependencies("z", {"y", "A", "pi", "mu", "sigma"});
    c.add_child(std::make_unique<hmm_block>(make_cfg(kT, kK, &em_count)));
    std::mt19937_64 rng(2);
    g_calls = 0; c.step(rng);
    const long long first = g_calls;
    g_calls = 0; for (int i = 0; i < 20; ++i) c.step(rng);
    const long long later = g_calls;
    report(first == (long long)(kT * kK) && later == 0,
           "composite: forward filter reused while the context is unchanged",
           "first sweep " + std::to_string(first) + " evaluations, next 20 sweeps " +
           std::to_string(later) + " (expect 0)");
    arma::vec y2 = c1.at("y"); y2[3] = -y2[3];
    c.data().set("y", y2);
    g_calls = 0; c.step(rng);
    const long long after = g_calls;
    report(after == (long long)(kT * kK),
           "composite: y replaced in shared data triggers a new forward pass",
           std::to_string(after) + " evaluations (expect " + std::to_string(kT * kK) + ")");
}

// ---- 3: draws equal a fresh recomputation along every path ------------------
struct exact_checker {
    int steps = 0, bad = 0;
    // Steps blk with r and compares with a new block given ctx and a copy of r.
    void step(hmm_block& blk, const block_context& ctx, std::mt19937_64& r) {
        hmm_block ref(make_cfg(kT, kK, &em_pure));
        ref.set_context(ctx);
        std::mt19937_64 rr = r;
        ref.step(rr);
        blk.step(r);
        ++steps;
        if (!same_bits(blk.current(), ref.current()) || !(rr == r)) ++bad;
    }
};

static void check_exact_paths() {
    const block_context c1 = base_ctx(kT, kK);
    block_context c2 = c1; c2["A"] = sticky_A(kK, 0.55);
    block_context c3 = c1; c3["mu"] = arma::vec{1.5, 0.0, -1.5};   // reverses the states
    block_context c4 = c1; c4["pi"] = arma::vec{0.05, 0.05, 0.9}; c4["y"] = -c1.at("y");
    exact_checker x;
    std::mt19937_64 rng(3);
    hmm_block b(make_cfg(kT, kK, &em_count));

    b.set_context(c1); for (int i = 0; i < 3; ++i) x.step(b, c1, rng);
    b.set_context(c2); x.step(b, c2, rng);
    b.set_context(c2); b.set_context(c2); x.step(b, c2, rng);   // repeated call
    b.set_context(c3); b.set_context(c3); x.step(b, c3, rng);   // change, then repeat
    b.set_context(c1); b.set_context(c2); x.step(b, c2, rng);   // two changes
    b.set_context(c2); b.set_context(c1); x.step(b, c1, rng);
    b.set_context(c3); x.step(b, c3, rng); x.step(b, c3, rng);
    b.set_context(c4); x.step(b, c4, rng);
    arma::vec zalt(kT); for (std::size_t t = 0; t < kT; ++t) zalt[t] = double(t % kK);
    b.set_current(zalt);
    const bool set_ok = same_bits(b.current(), zalt);
    x.step(b, c4, rng);

    hmm_block cp(b);                                             // copy construction
    std::mt19937_64 rc = rng;
    x.step(cp, c4, rc); x.step(b, c4, rng);
    cp.set_context(c3); x.step(cp, c3, rc);
    hmm_block ca(make_cfg(kT, kK, &em_count));                   // copy assignment
    { std::mt19937_64 r9(9); ca.set_context(c2); ca.step(r9); }
    ca = b; std::mt19937_64 rca = rng; x.step(ca, c4, rca);
    hmm_block mv(std::move(cp));                                 // move construction
    x.step(mv, c3, rc);
    hmm_block ma(make_cfg(kT, kK, &em_count));                   // move assignment
    { std::mt19937_64 r8(8); ma.set_context(c1); ma.step(r8); }
    ma = std::move(mv); x.step(ma, c3, rc); ma.set_context(c1); x.step(ma, c1, rc);
    cp = b; std::mt19937_64 rcp = rng; x.step(cp, c4, rcp);      // assign into moved-from

    // An exception inside the forward pass leaves nothing behind.
    bool threw = false;
    b.set_context(c2); g_armed = true; g_throw_t = kT / 2;
    try { b.step(rng); } catch (const std::runtime_error&) { threw = true; }
    g_armed = false;
    x.step(b, c2, rng);
    b.set_context(c2); x.step(b, c2, rng);
    b.set_context(c1); x.step(b, c1, rng);

    report(x.bad == 0 && threw && set_ok,
           "draws equal a fresh recomputation along every path",
           std::to_string(x.steps) + " steps, " + std::to_string(x.bad) + " mismatches" +
           (threw ? "" : ", forward-pass exception not raised") +
           (set_ok ? "" : ", set_current not applied"));
}

// ---- 4: enumerated posterior with alternating contexts ----------------------
static std::vector<double> exact_paths(const block_context& c, std::size_t T4) {
    const std::size_t K2 = 2, n = std::size_t(1) << T4;
    std::vector<double> lw(n);
    double mx = -1e300;
    for (std::size_t code = 0; code < n; ++code) {
        std::size_t prev = code & 1u;
        double lp = std::log(c.at("pi")[prev]) + em_pure(0, prev, c);
        for (std::size_t t = 1; t < T4; ++t) {
            const std::size_t s = (code >> t) & 1u;
            lp += std::log(c.at("A")[prev * K2 + s]) + em_pure(t, s, c);
            prev = s;
        }
        lw[code] = lp; if (lp > mx) mx = lp;
    }
    double tot = 0.0;
    for (auto& v : lw) { v = std::exp(v - mx); tot += v; }
    for (auto& v : lw) v /= tot;
    return lw;
}

static void check_enumerated() {
    const std::size_t T4 = 4;
    block_context ca, cb;
    ca["y"] = arma::vec{-0.4, 0.3, 1.2, -0.2}; ca["A"] = arma::vec{0.8, 0.2, 0.3, 0.7};
    ca["pi"] = arma::vec{0.6, 0.4}; ca["mu"] = arma::vec{-0.5, 0.8}; ca["sigma"] = arma::vec{1.0};
    cb = ca; cb["A"] = arma::vec{0.3, 0.7, 0.6, 0.4}; cb["mu"] = arma::vec{0.9, -0.6};
    const auto pa = exact_paths(ca, T4), pb = exact_paths(cb, T4);
    double gap = 0.0;
    for (std::size_t i = 0; i < pa.size(); ++i) gap = std::max(gap, std::fabs(pa[i] - pb[i]));

    hmm_block b(make_cfg(T4, 2, &em_count));
    std::mt19937_64 rng(4);
    const long S = 40000;
    std::vector<double> fa(pa.size(), 0.0), fb(pb.size(), 0.0);
    for (long s = 0; s < 2 * S; ++s) {
        const bool use_a = (s % 2 == 0);
        b.set_context(use_a ? ca : cb);
        b.step(rng);
        std::size_t code = 0;
        for (std::size_t t = 0; t < T4; ++t) code |= std::size_t(b.current()[t]) << t;
        (use_a ? fa : fb)[code] += 1.0 / double(S);
    }
    double ea = 0.0, eb = 0.0;
    for (std::size_t i = 0; i < pa.size(); ++i) {
        ea = std::max(ea, std::fabs(fa[i] - pa[i]));
        eb = std::max(eb, std::fabs(fb[i] - pb[i]));
    }
    char buf[200];
    std::snprintf(buf, sizeof buf,
                  "max |freq - exact|: context A %.4f, context B %.4f (tolerance 0.015; "
                  "the two exact posteriors differ by up to %.3f)", ea, eb, gap);
    report(ea < 0.015 && eb < 0.015, "alternating contexts match the enumerated posterior", buf);
}

int main() {
    std::printf("test_fix_hmm_forward_cache\n");
    check_counts_direct();
    check_counts_composite();
    check_exact_paths();
    check_enumerated();
    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

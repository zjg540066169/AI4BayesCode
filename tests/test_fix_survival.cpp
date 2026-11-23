/*================================================================================
 *  test_fix_survival.cpp
 *
 *  piecewise_exponential_gibbs_block, frailty_gamma_gibbs_block and
 *  interval_censored_survival_augmentation_block store quantities that depend
 *  only on their data (event counts, exposure, bin ranges, bins of L and U)
 *  and reuse them while the data are unchanged. This test drives each block
 *  through changes of every such input -- t, delta, entry_time, z, L, U, also
 *  back to earlier values and on every step -- and compares every step bitwise
 *  with a freshly constructed block that starts from the same state and RNG
 *  state. It also checks that every step throws while the context holds
 *  invalid data, and that copied and moved blocks continue identically.
 *
 *  Copyright (C) 2026 AI4BayesCode
 *  SPDX-License-Identifier: GPL-3.0-or-later
 *================================================================================*/
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
#  define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
#  define ARMA_DONT_USE_WRAPPER
#endif

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/piecewise_exponential_gibbs_block.hpp"
#include "AI4BayesCode/frailty_gamma_gibbs_block.hpp"
#include "AI4BayesCode/interval_censored_survival_augmentation_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace AI4BayesCode;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const std::string& what) {
    if (ok) { ++g_pass; return; }
    ++g_fail;
    std::printf("  FAIL: %s\n", what.c_str());
}

static bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem
        && (a.n_elem == 0 || std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0);
}

// ------------------------------------------------------------------ data
// Left truncation, edges[0] > 0, times on edges, t in (0, edges[0]], v == t,
// entry times on edges.
struct Data { arma::vec t, delta, v, z, edges; std::size_t K = 0, G = 0; };

static Data make_data(std::size_t n, std::size_t K, std::size_t G, std::uint64_t seed) {
    Data d; d.K = K; d.G = G;
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    d.edges.set_size(K + 1); d.edges[0] = 0.3;
    for (std::size_t k = 1; k <= K; ++k) d.edges[k] = d.edges[k - 1] + 0.2 + u(g);
    d.t.set_size(n); d.delta.set_size(n); d.v.set_size(n); d.z.set_size(n);
    for (std::size_t i = 0; i < n; ++i) {
        d.t[i] = 0.01 + u(g) * (d.edges[K] - 0.01);
        if (i % 17 == 0) d.t[i] = d.edges[1 + i % K];
        if (i % 19 == 0) d.t[i] = 0.5 * d.edges[0];
        d.v[i] = (i % 3 == 0) ? 0.0 : u(g) * d.t[i];
        if (i % 23 == 0) d.v[i] = d.t[i];
        if (i % 29 == 1) d.v[i] = std::min(d.edges[(i / 29) % (K + 1)], d.t[i]);
        d.delta[i] = u(g) < 0.6 ? 1.0 : 0.0;
        d.z[i] = static_cast<double>(i % G);
    }
    return d;
}

// The data of step s: every input changes at some step, some return to
// earlier values, and t changes on every step for a while.
static Data data_at(const Data& d0, int s) {
    Data d = d0;
    const std::size_t n = d.t.n_elem;
    if (s >= 40 && s < 60)  d.t = d0.t * 0.9;                        // t changes
    if (s >= 60 && s < 80)  for (std::size_t i = 0; i < n; i += 5) d.delta[i] = 1.0 - d.delta[i];
    if (s >= 80 && s < 100) d.v = d0.v * 0.5;                        // entry_time changes
    if (s >= 100 && s < 120) for (std::size_t i = 0; i < n; ++i) d.z[i] = static_cast<double>((i / 2) % d.G);
    if (s >= 120 && s < 140) d.t = arma::min(d0.t * 1.3, arma::vec(n, arma::fill::value(d0.edges[d0.K])));  // t grows
    if (s >= 140 && s < 170) d.t = d0.t * (1.0 - 1e-4 * (s % 4));    // t changes on most steps
    if (s >= 170 && s < 180) { d.t = d0.t; d.t[7] = std::nextafter(d.t[7], 0.0); }
    d.v = arma::min(d.v, d.t);                                       // keeps 0 <= v <= t
    return d;
}

static piecewise_exponential_gibbs_block_config peh_cfg(const Data& d, bool offset, bool entry) {
    piecewise_exponential_gibbs_block_config c;
    c.name = "lambda"; c.edges = d.edges; c.a0 = 0.5; c.b0 = 0.5;
    c.offset_key = offset ? "off" : ""; c.entry_time_key = entry ? "v" : "";
    return c;
}

static frailty_gamma_gibbs_block_config fr_cfg(const Data& d, bool offset, bool entry) {
    frailty_gamma_gibbs_block_config c;
    c.name = "w"; c.G = d.G; c.edges = d.edges; c.theta = 1.5;
    c.offset_key = offset ? "off" : ""; c.entry_time_key = entry ? "v" : "";
    return c;
}

static arma::vec draw_offset(std::size_t n, std::mt19937_64& g, bool with_invalid) {
    std::gamma_distribution<double> gam(2.0, 0.5);
    arma::vec w(n);
    for (std::size_t i = 0; i < n; ++i) w[i] = gam(g);
    if (with_invalid) {   // skipped by the frailty block
        w[3] = std::numeric_limits<double>::quiet_NaN(); w[4] = -1.0; w[5] = 0.0;
    }
    return w;
}

// ------------------------------------------------------------------ PEH
static void test_peh(bool offset, bool entry) {
    const std::string tag = std::string("PEH offset=") + (offset ? "1" : "0") + " entry=" + (entry ? "1" : "0");
    const Data d0 = make_data(300, 12, 1, 11);
    piecewise_exponential_gibbs_block blk(peh_cfg(d0, offset, entry));
    std::mt19937_64 rng(5), aux(6);
    int mismatches = 0;
    for (int s = 0; s < 200; ++s) {
        const Data d = data_at(d0, s);
        block_context ctx; ctx["t"] = d.t; ctx["delta"] = d.delta;
        if (entry) ctx["v"] = d.v;
        if (offset) ctx["off"] = draw_offset(d.t.n_elem, aux, false);
        piecewise_exponential_gibbs_block_config fc = peh_cfg(d0, offset, entry);
        fc.initial_lambda = blk.current();
        piecewise_exponential_gibbs_block fresh(fc);
        std::mt19937_64 rng_fresh = rng;
        blk.set_context(ctx);   blk.step(rng);
        fresh.set_context(ctx); fresh.step(rng_fresh);
        if (!same_bits(blk.current(), fresh.current()) || !(rng == rng_fresh)) ++mismatches;
    }
    check(mismatches == 0, tag + ": " + std::to_string(mismatches) + " of 200 steps differ from a fresh block");
}

// ------------------------------------------------------------------ frailty
static void test_frailty(bool offset, bool entry) {
    const std::string tag = std::string("frailty offset=") + (offset ? "1" : "0") + " entry=" + (entry ? "1" : "0");
    const Data d0 = make_data(300, 12, 7, 12);
    frailty_gamma_gibbs_block blk(fr_cfg(d0, offset, entry));
    std::mt19937_64 rng(7), aux(8), lam_rng(9);
    std::gamma_distribution<double> gl(2.0, 0.5);
    int mismatches = 0;
    for (int s = 0; s < 200; ++s) {
        const Data d = data_at(d0, s);
        arma::vec lambda(d.K); for (std::size_t k = 0; k < d.K; ++k) lambda[k] = gl(lam_rng);
        block_context ctx; ctx["t"] = d.t; ctx["delta"] = d.delta; ctx["z"] = d.z; ctx["lambda"] = lambda;
        if (entry) ctx["v"] = d.v;
        if (offset) ctx["off"] = draw_offset(d.t.n_elem, aux, s % 3 == 0);
        frailty_gamma_gibbs_block_config fc = fr_cfg(d0, offset, entry);
        fc.initial_frailties = blk.current();
        frailty_gamma_gibbs_block fresh(fc);
        std::mt19937_64 rng_fresh = rng;
        blk.set_context(ctx);   blk.step(rng);
        fresh.set_context(ctx); fresh.step(rng_fresh);
        if (!same_bits(blk.current(), fresh.current()) || !(rng == rng_fresh)) ++mismatches;
    }
    check(mismatches == 0, tag + ": " + std::to_string(mismatches) + " of 200 steps differ from a fresh block");
}

// ------------------------------------------------------------------ interval censoring
struct ICData { arma::vec L, U, t, edges; };

static ICData make_ic(std::size_t n, std::uint64_t seed) {
    ICData d;
    d.edges = arma::vec({0.0, 1.5, 3.0, 4.5, 6.0, 8.0, 10.0, 30.0});
    std::mt19937_64 g(seed);
    std::exponential_distribution<double> ex(0.3);
    const double visits[] = {1, 2, 3, 4, 5, 6, 8, 10};
    d.L.set_size(n); d.U.set_size(n); d.t.set_size(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double T = ex(g);
        double L = 0.0, U = std::numeric_limits<double>::infinity();
        for (double v : visits) { if (v <= T) L = std::max(L, v); else U = std::min(U, v); }
        d.L[i] = L; d.U[i] = U; d.t[i] = std::isfinite(U) ? 0.5 * (L + U) : L + 1.0;
    }
    return d;
}

// L and U of step s: each changes at some step, on every step for a while,
// and both return to their first values.
static void ic_bounds_at(const ICData& d0, int s, arma::vec& L, arma::vec& U) {
    L = d0.L; U = d0.U;
    const std::size_t n = L.n_elem;
    if (s >= 40 && s < 60) L = d0.L * 0.5;
    if (s >= 60 && s < 80) for (std::size_t i = 0; i < n; ++i) if (std::isfinite(U[i])) U[i] += 1.0;
    if (s >= 80 && s < 100) for (std::size_t i = 0; i < n; i += 4) U[i] = std::numeric_limits<double>::infinity();
    if (s >= 100 && s < 130) L = d0.L * (1.0 - 1e-3 * (s % 2));
    if (s >= 130 && s < 140) for (std::size_t i = 0; i < n; i += 3) L[i] = 0.0;
}

// Start times inside (L, U] and below edges[K] = 30.
static arma::vec valid_times(const arma::vec& L, const arma::vec& U) {
    arma::vec t(L.n_elem);
    for (std::size_t i = 0; i < L.n_elem; ++i) t[i] = std::isfinite(U[i]) ? U[i] : L[i] + 1.0;
    return t;
}

static void test_ic(bool offset) {
    const std::string tag = std::string("interval censoring offset=") + (offset ? "1" : "0");
    const ICData d0 = make_ic(300, 13);
    const std::size_t n = d0.L.n_elem;
    interval_censored_survival_augmentation_block_config cfg;
    cfg.name = "t"; cfg.edges = d0.edges; cfg.initial_times = d0.t; cfg.offset_key = offset ? "off" : "";
    interval_censored_survival_augmentation_block blk(cfg);
    std::mt19937_64 rng(10), aux(11), lam_rng(12);
    std::gamma_distribution<double> gl(2.0, 0.25);
    int mismatches = 0;
    for (int s = 0; s < 160; ++s) {
        arma::vec L, U; ic_bounds_at(d0, s, L, U);
        arma::vec lambda(d0.edges.n_elem - 1);
        for (std::size_t k = 0; k < lambda.n_elem; ++k) lambda[k] = gl(lam_rng);
        block_context ctx; ctx["L"] = L; ctx["U"] = U; ctx["lambda"] = lambda;
        if (offset) { arma::vec w = draw_offset(n, aux, s % 2 == 0); ctx["off"] = w; }
        // Fresh block: a throw-away step from start times valid for the current
        // L and U passes the first-step checks; then it takes the state of blk
        // and the same RNG state.
        interval_censored_survival_augmentation_block_config fc = cfg;
        fc.initial_times = valid_times(L, U);
        interval_censored_survival_augmentation_block fresh(fc);
        fresh.set_context(ctx);
        std::mt19937_64 throwaway(999);
        fresh.step(throwaway);
        fresh.set_current(blk.current());
        std::mt19937_64 rng_fresh = rng;
        blk.set_context(ctx); blk.step(rng);
        fresh.step(rng_fresh);
        if (!same_bits(blk.current(), fresh.current()) || !(rng == rng_fresh)) ++mismatches;
    }
    check(mismatches == 0, tag + ": " + std::to_string(mismatches) + " of 160 steps differ from a fresh block");
}

// ------------------------------------------------------------------ composite
// Frailty + PEH wiring of examples/PehSharedFrailty.cpp, with data changed
// through shared_data; every sweep is compared with a fresh composite.
static std::unique_ptr<composite_block> make_frailty_composite(const Data& d) {
    const std::size_t n = d.t.n_elem;
    auto comp = std::make_unique<composite_block>("frailty");
    comp->data().set("t", d.t); comp->data().set("delta", d.delta); comp->data().set("z", d.z);
    arma::vec li(d.K); li.fill(1.0); comp->data().set("lambda", li);
    comp->data().set("w", arma::vec(d.G, arma::fill::ones));
    comp->data().set("expanded_w", arma::vec(n, arma::fill::ones));
    comp->data().register_refresher("expanded_w",
        [n](const shared_data_t& sd) -> arma::vec {
            const arma::vec& w = sd.get("w"); const arma::vec& zz = sd.get("z");
            arma::vec out(n);
            for (std::size_t i = 0; i < n; ++i) out[i] = w[static_cast<std::size_t>(zz[i])];
            return out;
        });
    comp->data().declare_dependencies("w", {"t", "delta", "z", "lambda"});
    comp->data().declare_dependencies("lambda", {"t", "delta", "expanded_w"});
    comp->data().declare_invalidates("w", {"expanded_w"});
    frailty_gamma_gibbs_block_config fc = fr_cfg(d, false, false);
    comp->add_child(std::make_unique<frailty_gamma_gibbs_block>(fc));
    piecewise_exponential_gibbs_block_config pc = peh_cfg(d, false, false);
    pc.offset_key = "expanded_w";
    comp->add_child(std::make_unique<piecewise_exponential_gibbs_block>(pc));
    return comp;
}

static void test_composite() {
    const Data d0 = make_data(240, 10, 6, 14);
    auto comp = make_frailty_composite(d0);
    std::mt19937_64 rng(15);
    int mismatches = 0;
    for (int s = 0; s < 200; ++s) {
        const Data d = data_at(d0, s);
        comp->data().set("t", d.t); comp->data().set("delta", d.delta); comp->data().set("z", d.z);
        auto fresh = make_frailty_composite(d);
        fresh->set_current(comp->current());
        fresh->data().set("expanded_w", comp->data().get("expanded_w"));
        std::mt19937_64 rng_fresh = rng;
        comp->step(rng); fresh->step(rng_fresh);
        for (const char* key : {"w", "lambda", "expanded_w"})
            if (!same_bits(comp->data().get(key), fresh->data().get(key))) { ++mismatches; break; }
    }
    check(mismatches == 0, "frailty + PEH composite: " + std::to_string(mismatches)
                           + " of 200 sweeps differ from a fresh composite");
}

// ------------------------------------------------------------------ invalid data
template <class Block>
static int throws_per_step(Block& blk, const block_context& ctx, std::mt19937_64& rng, int steps) {
    int n_throw = 0;
    for (int s = 0; s < steps; ++s) {
        blk.set_context(ctx);
        try { blk.step(rng); } catch (const std::runtime_error&) { ++n_throw; }
    }
    return n_throw;
}

static void test_invalid_data() {
    const Data d = make_data(200, 8, 4, 16);
    const double eK = d.edges[d.K];
    std::mt19937_64 rng(17);
    {   // PEH: invalid t, entry_time and delta after the data were reused
        piecewise_exponential_gibbs_block blk(peh_cfg(d, false, true));
        block_context ok; ok["t"] = d.t; ok["delta"] = d.delta; ok["v"] = d.v;
        check(throws_per_step(blk, ok, rng, 3) == 0, "PEH: valid data do not throw");
        block_context bad_t = ok;  bad_t["t"][10] = eK * 1.5;
        block_context bad_v = ok;  bad_v["v"][11] = d.t[11] * 2.0;
        block_context bad_d = ok;  bad_d["delta"][12] = 0.5;
        check(throws_per_step(blk, bad_t, rng, 3) == 3, "PEH: t > edges[K] throws on every step");
        check(throws_per_step(blk, bad_v, rng, 3) == 3, "PEH: entry_time > t throws on every step");
        check(throws_per_step(blk, bad_d, rng, 3) == 3, "PEH: delta = 0.5 throws on every step");
        check(throws_per_step(blk, ok, rng, 3) == 0, "PEH: valid data after invalid data do not throw");
    }
    {   // PEH with an offset: an invalid offset throws while the other data are reused
        piecewise_exponential_gibbs_block blk(peh_cfg(d, true, false));
        block_context ok; ok["t"] = d.t; ok["delta"] = d.delta; ok["off"] = arma::vec(d.t.n_elem, arma::fill::ones);
        check(throws_per_step(blk, ok, rng, 3) == 0, "PEH: valid offset does not throw");
        block_context bad_w = ok; bad_w["off"][9] = -1.0;
        check(throws_per_step(blk, bad_w, rng, 3) == 3, "PEH: negative offset throws on every step");
        check(throws_per_step(blk, ok, rng, 2) == 0, "PEH: valid offset after invalid offset does not throw");
    }
    {   // frailty: invalid t, entry_time and z after the data were reused
        frailty_gamma_gibbs_block blk(fr_cfg(d, false, true));
        block_context ok; ok["t"] = d.t; ok["delta"] = d.delta; ok["z"] = d.z; ok["v"] = d.v;
        ok["lambda"] = arma::vec(d.K, arma::fill::ones);
        check(throws_per_step(blk, ok, rng, 3) == 0, "frailty: valid data do not throw");
        block_context bad_t = ok; bad_t["t"][10] = eK * 1.5;
        block_context bad_v = ok; bad_v["v"][11] = -0.5;
        block_context bad_z = ok; bad_z["z"][12] = static_cast<double>(d.G);
        check(throws_per_step(blk, bad_t, rng, 3) == 3, "frailty: t > edges[K] throws on every step");
        check(throws_per_step(blk, bad_v, rng, 3) == 3, "frailty: entry_time < 0 throws on every step");
        check(throws_per_step(blk, bad_z, rng, 3) == 3, "frailty: z = G throws on every step");
        check(throws_per_step(blk, ok, rng, 3) == 0, "frailty: valid data after invalid data do not throw");
    }
}

// ------------------------------------------------------------------ copies
static void test_copy_move() {
    const Data d0 = make_data(250, 10, 5, 18);
    piecewise_exponential_gibbs_block peh(peh_cfg(d0, false, true));
    frailty_gamma_gibbs_block fr(fr_cfg(d0, false, true));
    block_context ctx; ctx["t"] = d0.t; ctx["delta"] = d0.delta; ctx["v"] = d0.v; ctx["z"] = d0.z;
    ctx["lambda"] = arma::vec(d0.K, arma::fill::ones);
    std::mt19937_64 rng(19);
    for (int s = 0; s < 5; ++s) {
        peh.set_context(ctx); peh.step(rng);
        fr.set_context(ctx);  fr.step(rng);
    }
    // Continue the original and a copy with new t: both must equal a fresh block.
    block_context ctx2 = ctx; ctx2["t"] = d0.t * 0.8; ctx2["v"] = arma::min(d0.v, d0.t * 0.8);
    piecewise_exponential_gibbs_block peh_copy(peh);
    piecewise_exponential_gibbs_block peh_moved(std::move(peh_copy));
    frailty_gamma_gibbs_block fr_copy(fr);
    frailty_gamma_gibbs_block fr_assigned(fr_cfg(d0, false, true));
    fr_assigned = fr_copy;
    piecewise_exponential_gibbs_block_config pc = peh_cfg(d0, false, true); pc.initial_lambda = peh.current();
    frailty_gamma_gibbs_block_config fc = fr_cfg(d0, false, true); fc.initial_frailties = fr.current();
    piecewise_exponential_gibbs_block peh_fresh(pc);
    frailty_gamma_gibbs_block fr_fresh(fc);
    for (int s = 0; s < 3; ++s) {
        std::mt19937_64 r1(100 + s), r2(100 + s), r3(100 + s);
        const block_context& c = (s == 1) ? ctx : ctx2;     // new data, then back, then new again
        peh.set_context(c); peh_moved.set_context(c); peh_fresh.set_context(c);
        peh.step(r1); peh_moved.step(r2); peh_fresh.step(r3);
        check(same_bits(peh.current(), peh_fresh.current()) && same_bits(peh_moved.current(), peh_fresh.current()),
              "PEH: original and moved copy equal a fresh block after data change " + std::to_string(s));
        std::mt19937_64 q1(200 + s), q2(200 + s), q3(200 + s);
        fr.set_context(c); fr_assigned.set_context(c); fr_fresh.set_context(c);
        fr.step(q1); fr_assigned.step(q2); fr_fresh.step(q3);
        check(same_bits(fr.current(), fr_fresh.current()) && same_bits(fr_assigned.current(), fr_fresh.current()),
              "frailty: original and assigned copy equal a fresh block after data change " + std::to_string(s));
        // Keep the three blocks in the same state for the next comparison.
        pc.initial_lambda = peh_fresh.current(); peh_fresh = piecewise_exponential_gibbs_block(pc);
        peh.set_current(peh_fresh.current()); peh_moved.set_current(peh_fresh.current());
        fc.initial_frailties = fr_fresh.current(); fr_fresh = frailty_gamma_gibbs_block(fc);
        fr.set_current(fr_fresh.current()); fr_assigned.set_current(fr_fresh.current());
    }
}

int main() {
    std::printf("=== test_fix_survival ===\n");
    for (int o = 0; o < 2; ++o)
        for (int e = 0; e < 2; ++e) { test_peh(o == 1, e == 1); test_frailty(o == 1, e == 1); }
    test_ic(false);
    test_ic(true);
    test_composite();
    test_invalid_data();
    test_copy_move();
    std::printf("test_fix_survival: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

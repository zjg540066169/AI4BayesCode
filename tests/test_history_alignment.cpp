/*================================================================================
 *  test_history_alignment.cpp
 *
 *  Every child of a composite must contribute EXACTLY ONE history entry per
 *  composite sweep. Any block that skips its append on some code path leaves a
 *  short history, and from that sweep on, draw d of that block no longer
 *  corresponds to draw d of its siblings -- so posterior-predictive and
 *  history-mode predict_at either hard-error ("inconsistent history sizes") or
 *  silently pair mismatched draws.
 *
 *  Two paths are covered:
 *
 *    A. NUMERIC FALLBACK. inv_gamma_gibbs_block used to `return` early when the
 *       Gamma draw underflowed to 0, skipping its own append. Its twin
 *       gamma_gibbs_block keeps the previous value and appends unconditionally.
 *       Both must behave the same way.
 *
 *    B. FROZEN CHILDREN. composite_block::step() skips a frozen child's step(),
 *       so the child's normal append is skipped too; the child supplies the
 *       held value via record_held_history(). A frozen NESTED COMPOSITE must
 *       forward that call to its own children, or the whole subtree stalls.
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
#include "AI4BayesCode/gamma_gibbs_block.hpp"
#include "AI4BayesCode/inv_gamma_gibbs_block.hpp"

#include <cstdio>
#include <memory>
#include <random>

#include "portable_rng.hpp"   // portable draws: identical on every stdlib
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::gamma_gibbs_block;
using AI4BayesCode::gamma_gibbs_block_config;
using AI4BayesCode::gamma_params;
using AI4BayesCode::inv_gamma_gibbs_block;
using AI4BayesCode::inv_gamma_gibbs_block_config;
using AI4BayesCode::inv_gamma_params;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  [FAIL] %s\n", what.c_str()); }
}

namespace {

// shape this small makes prng::gamma_distribution underflow to exactly 0 on
// essentially every draw -- the numeric fallback path both blocks must handle.
constexpr double kDegenerateShape = 1e-8;
constexpr double kDegenerateRate  = 1e8;

std::unique_ptr<gamma_gibbs_block> make_gamma(const std::string& nm,
                                              double shape, double rate) {
    gamma_gibbs_block_config c;
    c.name = nm;
    c.params_fn = [shape, rate](const block_context&) {
        gamma_params p;
        p.shape = shape;
        p.rate  = rate;
        return p;
    };
    return std::make_unique<gamma_gibbs_block>(std::move(c));
}

std::unique_ptr<inv_gamma_gibbs_block> make_inv_gamma(const std::string& nm,
                                                      double shape, double rate) {
    inv_gamma_gibbs_block_config c;
    c.name = nm;
    c.params_fn = [shape, rate](const block_context&) {
        inv_gamma_params p;
        p.shape = shape;
        p.rate  = rate;
        return p;
    };
    return std::make_unique<inv_gamma_gibbs_block>(std::move(c));
}

std::size_t rows(const AI4BayesCode::block_sampler& b, const std::string& key) {
    const auto h = b.get_history();
    auto it = h.find(key);
    return it == h.end() ? 0 : it->second.n_rows;
}

}  // namespace

int main() {
    std::printf("=== composite history alignment ===\n\n");
    std::mt19937_64 rng(20260815);

    // ---- A. numeric-fallback path appends anyway ---------------------------
    {
        std::printf("A. underflow fallback still appends (inv_gamma vs gamma)\n");

        auto root = std::make_unique<composite_block>("root");
        root->add_child(make_inv_gamma("ig", kDegenerateShape, kDegenerateRate));
        root->add_child(make_gamma("g", kDegenerateShape, kDegenerateRate));
        root->set_keep_history(true);

        const std::size_t N = 200;
        for (std::size_t i = 0; i < N; ++i) root->step(rng);

        const std::size_t n_ig = rows(root->child(0), "ig");
        const std::size_t n_g  = rows(root->child(1), "g");
        std::printf("        %zu sweeps -> ig rows=%zu, g rows=%zu\n", N, n_ig, n_g);
        check(n_ig == N, "inv_gamma history has one row per sweep");
        check(n_g  == N, "gamma history has one row per sweep");
        check(n_ig == n_g, "the two blocks stay aligned");
        check(root->history_size() == N,
              "composite history_size equals the sweep count");
    }

    // ---- B1. frozen leaf keeps its history growing -------------------------
    {
        std::printf("\nB1. frozen leaf holds its history\n");

        auto root = std::make_unique<composite_block>("root");
        root->add_child(make_inv_gamma("s2", 3.0, 2.0));
        root->add_child(make_gamma("g", 2.0, 1.0));
        root->set_keep_history(true);

        const std::size_t pre = 20, post = 30;
        for (std::size_t i = 0; i < pre; ++i) root->step(rng);
        root->freeze(std::vector<std::string>{"s2"});
        for (std::size_t i = 0; i < post; ++i) root->step(rng);

        const std::size_t n_s = rows(root->child(0), "s2");
        const std::size_t n_g = rows(root->child(1), "g");
        std::printf("        %zu free + %zu frozen sweeps -> s2 rows=%zu, g rows=%zu\n",
                    pre, post, n_s, n_g);
        check(n_s == pre + post, "frozen child still has one row per sweep");
        check(n_s == n_g, "frozen and free children stay aligned");

        const auto h = root->child(0).get_history();
        const arma::mat& m = h.at("s2");
        bool constant = m.n_rows == pre + post;
        if (constant) {
            const double held = m(pre, 0);
            for (arma::uword r = pre; r < m.n_rows; ++r)
                if (m(r, 0) != held) { constant = false; break; }
        }
        check(constant, "held value is constant across the frozen stretch");
    }

    // ---- B2. frozen NESTED COMPOSITE forwards to its grandchildren ---------
    {
        std::printf("\nB2. frozen nested composite forwards the held append\n");

        auto inner = std::make_unique<composite_block>("inner");
        inner->add_child(make_gamma("inner_a", 2.0, 1.0));
        inner->add_child(make_gamma("inner_b", 3.0, 1.0));
        composite_block* inner_raw = inner.get();

        auto outer = std::make_unique<composite_block>("outer");
        outer->add_child(std::move(inner));
        outer->add_child(make_gamma("outer_c", 2.0, 1.0));
        outer->set_keep_history(true);

        const std::size_t pre = 15, post = 25;
        for (std::size_t i = 0; i < pre; ++i) outer->step(rng);
        outer->freeze(std::vector<std::string>{"inner"});
        for (std::size_t i = 0; i < post; ++i) outer->step(rng);

        const std::size_t na = rows(inner_raw->child(0), "inner_a");
        const std::size_t nb = rows(inner_raw->child(1), "inner_b");
        const std::size_t nc = rows(outer->child(1), "outer_c");
        std::printf("        %zu free + %zu frozen sweeps -> inner_a=%zu inner_b=%zu "
                    "outer_c=%zu\n", pre, post, na, nb, nc);
        check(nc == pre + post, "the free outer child has one row per sweep");
        check(na == pre + post,
              "grandchild inner_a keeps growing while its parent is frozen");
        check(nb == pre + post,
              "grandchild inner_b keeps growing while its parent is frozen");
        check(outer->history_size() == pre + post,
              "composite history_size is not dragged down by the frozen subtree");
    }

    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

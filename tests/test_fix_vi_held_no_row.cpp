// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_vi_held_no_row.cpp
//
//  vi_block::record_held_history() for a VI block held through a frozen
//  ancestor composite while it has neither a stored nor a held row (frozen
//  before its first recorded sweep, keep_history off for every earlier
//  sweep, or clear_history() on a fresh model). Layout, for each of the four
//  VI blocks:
//      root { tau: gamma_gibbs, grp: composite { s2: gamma_gibbs, w: VI } }
//  with grp frozen through root.freeze({"grp"}).
//    V1  freeze before any sweep: after every sweep k, root.history_size()
//        is k and every get_history() key has k rows; every row of w is the
//        value stored for w in the composite's shared_data (the value
//        add_child() seeded);
//    V2  as V1, then unfreeze: every key still has one row per sweep after
//        each of the unfrozen sweeps;
//    V3  freeze, 5 sweeps, clear_history(), 4 sweeps: 4 rows per key, equal
//        bit for bit to the last 4 rows of the same run without the call;
//    V4  keep_history off for 5 sweeps, then on, freeze, 4 sweeps: 4 rows
//        per key, every row of w equal to the value stored for w at the
//        freeze (the last published q-draw); the same with clear_history()
//        before keep_history is turned on, bit for bit;
//    V5  clear_history() on a fresh model, freeze, 4 sweeps: 4 rows per key,
//        equal to the first 4 rows of V1;
//    V6  control (rows stored before the freeze): 3 sweeps, freeze, 5
//        sweeps: 8 rows per key, rows 4..8 of w equal to row 3.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>

#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/gamma_gibbs_block.hpp"
#include "AI4BayesCode/mean_field_gaussian_vi_block.hpp"
#include "AI4BayesCode/full_rank_gaussian_vi_block.hpp"
#include "AI4BayesCode/mean_field_categorical_vi_block.hpp"
#include "AI4BayesCode/structured_categorical_vi_block.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

namespace ab = AI4BayesCode;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

bool same_bits(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           (a.n_elem == 0 ||
            std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0);
}

std::unique_ptr<ab::block_sampler> make_gamma(const std::string& nm, double init) {
    ab::gamma_gibbs_block_config c;
    c.name = nm;
    c.initial_value = init;
    c.params_fn = [](const ab::block_context&) { return ab::gamma_params{3.0, 2.0}; };
    return std::make_unique<ab::gamma_gibbs_block>(std::move(c));
}

ab::vi_optimizer::raabbvi_config opt_cfg() {
    ab::vi_optimizer::raabbvi_config o;
    o.inner_iter_per_epoch = 50;
    return o;
}

std::unique_ptr<ab::block_sampler> make_vi(int type) {
    switch (type) {
    case 0: {
        ab::mean_field_gaussian_vi_block_config c;
        c.name = "w";
        c.initial_unc = arma::vec{0.3, -0.2, 0.1};
        c.log_density_grad = [](const arma::vec& e, const ab::block_context&, arma::vec* g) {
            const arma::vec d = e - arma::vec{1.0, 2.0, -1.0};
            if (g) *g = -d;
            return -0.5 * arma::dot(d, d);
        };
        c.optimizer = opt_cfg();
        return std::make_unique<ab::mean_field_gaussian_vi_block>(std::move(c));
    }
    case 1: {
        ab::full_rank_gaussian_vi_block_config c;
        c.name = "w";
        c.initial_unc = arma::vec{0.0, 0.0};
        c.log_density_grad = [](const arma::vec& e, const ab::block_context&, arma::vec* g) {
            const arma::mat P{{1.5, 0.6}, {0.6, 1.0}};
            const arma::vec d = e - arma::vec{0.5, -1.0};
            const arma::vec Pd = P * d;
            if (g) *g = -Pd;
            return -0.5 * arma::dot(d, Pd);
        };
        c.optimizer = opt_cfg();
        return std::make_unique<ab::full_rank_gaussian_vi_block>(std::move(c));
    }
    case 2: {
        ab::mean_field_categorical_vi_block_config c;
        c.name = "w";
        c.cardinalities = arma::uvec{3, 2};
        c.log_density = [](const arma::uvec& z, const ab::block_context&) {
            return 0.7 * (z[0] == 2) - 0.4 * (z[1] == 1) + 0.9 * (z[0] == 0 && z[1] == 0);
        };
        c.optimizer = opt_cfg();
        return std::make_unique<ab::mean_field_categorical_vi_block>(std::move(c));
    }
    default: {
        ab::structured_categorical_vi_block_config c;
        c.name = "w";
        c.cardinalities = arma::uvec{2, 2, 3};
        c.clique_partition = {{0, 1}, {2}};
        c.log_density = [](const arma::uvec& z, const ab::block_context&) {
            return 1.1 * (z[0] == z[1]) + 0.5 * (z[2] == 1) - 0.3 * (z[0] == 1);
        };
        c.optimizer = opt_cfg();
        return std::make_unique<ab::structured_categorical_vi_block>(std::move(c));
    }
    }
}

const char* vi_name(int t) {
    static const char* n[] = {"mean_field_gaussian", "full_rank_gaussian",
                              "mean_field_categorical", "structured_categorical"};
    return n[t];
}

struct model {
    std::unique_ptr<ab::composite_block> root;
    ab::composite_block* grp = nullptr;   // holds w in its shared_data
    std::mt19937_64 rng{20261125ULL};
};

model build(int type) {
    model m;
    m.root = std::make_unique<ab::composite_block>("root");
    m.root->add_child(make_gamma("tau", 1.5));
    auto g = std::make_unique<ab::composite_block>("grp");
    g->add_child(make_gamma("s2", 0.7));
    g->add_child(make_vi(type));
    m.grp = g.get();
    m.root->add_child(std::move(g));
    return m;
}

// True when root.history_size() and every get_history() key equal k rows.
bool aligned(const model& m, std::size_t k) {
    if (m.root->history_size() != k) return false;
    const auto h = m.root->get_history();
    if (!h.count("w") || !h.count("w__vi_history")) return false;
    for (const auto& kv : h)
        if (kv.second.n_rows != k) return false;
    return true;
}

// Steps one sweep at a time; true when every key has `base + i` rows after
// sweep i.
bool step_aligned(model& m, int n, std::size_t base) {
    bool ok = true;
    for (int i = 1; i <= n; ++i) {
        m.root->step(m.rng);
        ok = ok && aligned(m, base + static_cast<std::size_t>(i));
    }
    return ok;
}

// Every row of h["w"] equals v.
bool rows_equal(const arma::mat& w, const arma::vec& v) {
    if (w.n_cols != v.n_elem) return false;
    for (arma::uword r = 0; r < w.n_rows; ++r)
        if (!same_bits(w.row(r), arma::mat(v.t()))) return false;
    return true;
}

bool maps_tail_equal(const ab::history_map& a, const ab::history_map& b, std::size_t n) {
    if (a.size() != b.size()) return false;
    for (const auto& kv : a) {
        const auto it = b.find(kv.first);
        if (it == b.end() || it->second.n_rows < n || kv.second.n_rows != n) return false;
        const arma::mat tail = it->second.rows(it->second.n_rows - n, it->second.n_rows - 1);
        if (!same_bits(kv.second, tail)) return false;
    }
    return true;
}

bool maps_head_equal(const ab::history_map& a, const ab::history_map& b, std::size_t n) {
    if (a.size() != b.size()) return false;
    for (const auto& kv : a) {
        const auto it = b.find(kv.first);
        if (it == b.end() || it->second.n_rows < n || kv.second.n_rows != n) return false;
        if (!same_bits(kv.second, it->second.rows(0, n - 1))) return false;
    }
    return true;
}

}  // namespace

int run() {
    for (int type = 0; type < 4; ++type) {
        const std::string t = vi_name(type);
        std::printf("-- %s\n", t.c_str());

        // V1
        model a = build(type);
        a.root->set_keep_history(true);
        const arma::vec seeded = a.grp->data().get("w");
        a.root->freeze({"grp"});
        const bool v1_aligned = step_aligned(a, 6, 0);
        const auto h1 = a.root->get_history();
        check(v1_aligned, "V1 " + t + ": frozen before the first sweep, one row per sweep in every key");
        check(rows_equal(h1.at("w"), seeded) && same_bits(a.grp->data().get("w"), seeded),
              "V1 " + t + ": every row of w is the value stored for w (seeded by add_child)");

        // V2
        a.root->unfreeze({"grp"});
        check(step_aligned(a, 4, 6), "V2 " + t + ": after unfreeze, one row per sweep in every key");

        // V3
        model b = build(type), c = build(type);
        b.root->set_keep_history(true);
        c.root->set_keep_history(true);
        b.root->freeze({"grp"});
        c.root->freeze({"grp"});
        for (int i = 0; i < 5; ++i) b.root->step(b.rng);
        b.root->clear_history();
        const bool v3_aligned = step_aligned(b, 4, 0);
        for (int i = 0; i < 9; ++i) c.root->step(c.rng);
        check(v3_aligned && maps_tail_equal(b.root->get_history(), c.root->get_history(), 4),
              "V3 " + t + ": clear_history() while frozen, 4 rows = last 4 rows without the call");

        // V4
        model d = build(type), e = build(type);
        for (int i = 0; i < 5; ++i) { d.root->step(d.rng); e.root->step(e.rng); }
        e.root->clear_history();
        d.root->set_keep_history(true);
        e.root->set_keep_history(true);
        const arma::vec published = d.grp->data().get("w");
        d.root->freeze({"grp"});
        e.root->freeze({"grp"});
        const bool d_aligned = step_aligned(d, 4, 0);
        const bool v4_aligned = step_aligned(e, 4, 0) && d_aligned;
        const auto h4 = d.root->get_history();
        check(v4_aligned && rows_equal(h4.at("w"), published),
              "V4 " + t + ": keep_history turned on before the freeze, every row of w = the stored value");
        check(maps_tail_equal(e.root->get_history(), h4, 4),
              "V4 " + t + ": the same with clear_history() first, bit for bit");

        // V5
        model f = build(type);
        f.root->set_keep_history(true);
        f.root->clear_history();
        f.root->freeze({"grp"});
        const bool v5_aligned = step_aligned(f, 4, 0);
        check(v5_aligned && maps_head_equal(f.root->get_history(), h1, 4),
              "V5 " + t + ": clear_history() on a fresh model, then freeze: rows = first 4 rows of V1");

        // V6
        model g = build(type);
        g.root->set_keep_history(true);
        for (int i = 0; i < 3; ++i) g.root->step(g.rng);
        g.root->freeze({"grp"});
        const bool v6_aligned = step_aligned(g, 5, 3);
        const arma::mat w6 = g.root->get_history().at("w");
        bool v6_held = w6.n_rows == 8;
        for (arma::uword r = 3; v6_held && r < 8; ++r)
            v6_held = same_bits(w6.row(r), w6.row(2));
        check(v6_aligned && v6_held, "V6 " + t + ": rows stored before the freeze are repeated as before");
    }
    return 0;
}

int main() {
    std::printf("=== test_fix_vi_held_no_row ===\n");
    try {
        run();
    } catch (const std::exception& e) {
        std::printf("  [FAIL] exception: %s\n", e.what());
        ++g_fail;
    }
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

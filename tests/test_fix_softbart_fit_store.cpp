// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_softbart_fit_store.cpp -- the SoftBART forest stores the fit of
//  every tree at the training X (ntrees x N doubles) only while ntrees * N is
//  within AI4BAYESCODE_SOFTBART_FIT_STORE_MAX_DOUBLES, and the draws with and
//  without the store equal those of Forest::do_gibbs, which computes every
//  fit from the trees.
//
//  The number of stored fits (tree_fit_store_size()) is checked after every
//  sweep. That no store is filled during a sweep above the bound and released
//  at its end is checked only by memory measurement, as the forest has no
//  count of the fits it stores within a sweep: test_fix_softbart_store_default
//  compares the growth of the peak resident set size during one sweep above
//  and one within the default bound.
//
//  Here the macro names the variable g_store_bound, so the bound is set at run
//  time: 4000 doubles unless a part sets it otherwise (20 trees at N = 200
//  use the store, 20 trees at N = 201 do not).
//  A. fit_store_allowed at the boundary; after a sweep the store holds one
//     fit per tree under the bound and none above it.
//  B. softbart_model::update_step (internal sigma, external sigma, DART)
//     against Forest::do_gibbs on a model built from the same seed, under and
//     above the bound: f(X) kept after every sweep, sigma, sigma_mu, the tree
//     string and the RNG stream agree bitwise.
//  C. softbart_block with set_context + step and set_X, set_tree (round trip
//     and an earlier forest) and set_data between sweeps, under and above the
//     bound, against the same block whose model is advanced by
//     Forest::do_gibbs.
//  D. A forest with update_num_tree, with the bound placed at the number of
//     trees before each sweep: a sweep that adds a tree ends without the
//     store, and with the bound one below, a sweep that removes a tree ends
//     with it; do_gibbs_fit against do_gibbs.
//  E. One forest swept at two X with different row counts (one under, one
//     above the bound) in alternating runs; do_gibbs_fit against do_gibbs.
//  F. softbart_block with the bound switched between 4000 and 0 every five
//     sweeps: the store is released and refilled while the kept f(X) stays
//     valid; against the same block advanced by Forest::do_gibbs.
// ============================================================================
static unsigned long long g_store_bound = 4000;
#define AI4BAYESCODE_SOFTBART_FIT_STORE_MAX_DOUBLES g_store_bound
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/softbart_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::softbart_block;
using AI4BayesCode::softbart_block_config;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

static bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           (a.n_elem == 0 || std::memcmp(a.memptr(), b.memptr(), sizeof(double) * a.n_elem) == 0);
}
static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

static void make_data(int N, int p, std::uint64_t seed, arma::mat& X, arma::vec& y) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> Z(0.0, 1.0);
    X.set_size(N, p); y.set_size(N);
    for (int j = 0; j < p; ++j) for (int i = 0; i < N; ++i) X(i, j) = U(g);
    for (int i = 0; i < N; ++i) {
        double f = std::sin(6.0 * X(i, 0));
        if (p > 1) f += X(i, 1);
        y[i] = f + 0.3 * Z(g);
    }
    y -= arma::mean(y);
}

// State after one sweep.
struct Sweep {
    arma::vec fit;
    double sigma = 0.0, sigma_mu = 0.0;
    std::string trees;
    std::mt19937_64 rng;
};
// Index of the first sweep at which two chains differ, or -1.
static int first_diff(const std::vector<Sweep>& a, const std::vector<Sweep>& b) {
    if (a.size() != b.size()) return 0;
    for (std::size_t s = 0; s < a.size(); ++s)
        if (!same_bits(a[s].fit, b[s].fit) || !same_bits(a[s].sigma, b[s].sigma) ||
            !same_bits(a[s].sigma_mu, b[s].sigma_mu) || a[s].trees != b[s].trees ||
            !(a[s].rng == b[s].rng)) return (int)s;
    return -1;
}

// ---------------------------------------------------------------------------
// A. bound
// ---------------------------------------------------------------------------
static void partA() {
    std::printf("A. bound of 4000 doubles\n");
    check(Forest::fit_store_allowed(20, 200) && !Forest::fit_store_allowed(20, 201) &&
          Forest::fit_store_allowed(4000, 1) && !Forest::fit_store_allowed(4001, 1) &&
          Forest::fit_store_allowed(0, 1000000) && Forest::fit_store_allowed(1000000, 0),
          "fit_store_allowed(ntrees, N) is ntrees * N <= 4000");
    for (int N : {200, 201}) {
        arma::mat X; arma::vec y; make_data(N, 3, 1, X, y);
        bart_rng::set_seed(5);
        softbart::softbart_model m(X, y, 20);
        const std::size_t before = m.forest()->tree_fit_store_size();
        m.update_step();
        const std::size_t after = m.forest()->tree_fit_store_size();
        char what[128];
        std::snprintf(what, sizeof what, "N = %d, 20 trees: %zu stored fits after a sweep, f(X) kept",
                      N, after);
        check(before == 0 && after == (N == 200 ? 20u : 0u) && m.train_fit() != nullptr, what);
    }
}

// ---------------------------------------------------------------------------
// B. softbart_model chains
// ---------------------------------------------------------------------------
// Chain of softbart_model::update_step, or with reference = true the same
// model advanced by Forest::do_gibbs (the transition update_step makes, with
// every fit computed from the trees).
static std::vector<Sweep> run_model(const arma::mat& X, const arma::vec& y, bool dart,
                                    bool ext_sigma, bool reference, std::size_t& store) {
    const int M = 50;
    bart_rng::set_seed(1234);
    softbart::softbart_model m(X, y, 20, 2.0, -1.0, 1.0, 2.0, 0.95, 0.1, 1.0, 10.0,
                               -1.0, 0.5, 1.0, dart);
    const arma::mat X0(0, X.n_cols);
    std::vector<Sweep> out;
    store = 0;
    for (int s = 0; s < M; ++s) {
        const double sig = 0.25 + 0.01 * (s % 7);
        Sweep w;
        if (reference) {
            if (ext_sigma) m.forest()->set_sigma(sig);
            m.forest()->do_gibbs(X, y, X0, 1);
            if (ext_sigma) m.forest()->set_sigma(sig);
            w.rng = bart_rng::engine();
            w.fit = m.predict(X);
        } else {
            if (ext_sigma) m.update_step(sig); else m.update_step();
            w.rng = bart_rng::engine();
            const arma::vec* kept = m.train_fit();
            if (kept != nullptr) w.fit = *kept;
            store = std::max(store, m.forest()->tree_fit_store_size());
        }
        w.sigma = m.current_sigma(); w.sigma_mu = m.current_sigma_mu(); w.trees = m.get_tree();
        out.push_back(std::move(w));
    }
    return out;
}

static void partB() {
    std::printf("B. softbart_model::update_step against Forest::do_gibbs\n");
    const char* names[3] = {"internal sigma", "external sigma", "DART, internal sigma"};
    for (int N : {200, 201}) {
        arma::mat X; arma::vec y; make_data(N, 3, 2, X, y);
        for (int v = 0; v < 3; ++v) {
            const bool dart = v == 2, ext = v == 1;
            std::size_t st_ref = 0, st = 0;
            const std::vector<Sweep> ref = run_model(X, y, dart, ext, true, st_ref);
            const std::vector<Sweep> got = run_model(X, y, dart, ext, false, st);
            const int d = first_diff(ref, got);
            char what[160];
            std::snprintf(what, sizeof what, "N = %d (%s), %s: chains equal (first diff %d)",
                          N, st ? "store" : "no store", names[v], d);
            check(d < 0 && st == (N == 200 ? 20u : 0u), what);
        }
    }
}

// ---------------------------------------------------------------------------
// C. softbart_block with setters; F. bound switched between sweeps
// ---------------------------------------------------------------------------
struct BlockRun {
    std::vector<Sweep> sweeps;
    std::size_t store_min = 1000, store_max = 0;
    long refills_with_kept_fit = 0;   // sweeps starting with f(X) kept, store empty, store allowed
};

// switch_bound: the bound is 4000 for five sweeps, then 0 for five, and so on.
static BlockRun run_block(const arma::mat& X, const arma::mat& X2, const arma::vec& y,
                          const arma::vec& y2, bool reference, bool setters, bool switch_bound) {
    const int M = 80;
    bart_rng::set_seed(99);
    softbart_block_config cfg;
    cfg.x_train = X; cfg.y_init = y; cfg.ntrees = 20; cfg.center_Y = false;
    softbart_block b(cfg);
    const arma::mat X0(0, X.n_cols);
    arma::mat Xc = X;
    std::string older;
    BlockRun out;
    for (int s = 0; s < M; ++s) {
        if (setters) {
            if (s == 20) older = b.get_tree();
            if (s == 30) { b.set_X(X2); Xc = X2; }
            if (s == 45) b.set_tree(b.get_tree());
            if (s == 55) b.set_tree(older);
            if (s == 65) { b.set_data(X, y); Xc = X; }
        }
        if (switch_bound) g_store_bound = ((s / 5) % 2 == 0) ? 4000 : 0;
        const arma::vec& yc = (setters && s >= 40) ? y2 : y;
        const double sig = 0.3 + 0.02 * (s % 5);
        block_context ctx;
        ctx["softbart_target"] = yc;
        ctx["sigma"] = arma::vec{sig};
        b.set_context(ctx);
        Sweep w;
        if (reference) {
            Forest* f = b.underlying()->forest();
            f->set_sigma(sig);
            f->do_gibbs(Xc, yc, X0, 1);
            f->set_sigma(sig);
            w.rng = bart_rng::engine();
            w.fit = b.underlying()->predict(Xc);
        } else {
            const Forest* f = b.underlying()->forest();
            if (b.underlying()->train_fit() != nullptr && f->tree_fit_store_size() == 0 &&
                Forest::fit_store_allowed(f->get_trees().size(), Xc.n_rows))
                ++out.refills_with_kept_fit;
            std::mt19937_64 unused(1);
            b.step(unused);
            w.rng = bart_rng::engine();
            w.fit = b.current();
            const std::size_t st = f->tree_fit_store_size();
            out.store_min = std::min(out.store_min, st); out.store_max = std::max(out.store_max, st);
        }
        w.sigma = b.current_sigma(); w.sigma_mu = b.current_sigma_mu(); w.trees = b.get_tree();
        out.sweeps.push_back(std::move(w));
    }
    g_store_bound = 4000;
    return out;
}

static void partC() {
    std::printf("C. softbart_block (set_context + step, setters) against Forest::do_gibbs\n");
    for (int N : {200, 201}) {
        arma::mat X, X2; arma::vec y, y2;
        make_data(N, 3, 3, X, y); make_data(N, 3, 4, X2, y2);
        const BlockRun ref = run_block(X, X2, y, y2, true, true, false);
        const BlockRun got = run_block(X, X2, y, y2, false, true, false);
        const int d = first_diff(ref.sweeps, got.sweeps);
        const std::size_t want = N == 200 ? 20u : 0u;
        char what[160];
        std::snprintf(what, sizeof what, "N = %d (%s): chains equal (first diff %d), stored fits in [%zu, %zu]",
                      N, want ? "store" : "no store", d, got.store_min, got.store_max);
        check(d < 0 && got.store_min == want && got.store_max == want, what);
    }
}

static void partF() {
    std::printf("F. softbart_block, bound switched between 4000 and 0 every five sweeps\n");
    arma::mat X, X2; arma::vec y, y2;
    make_data(200, 3, 8, X, y); make_data(200, 3, 9, X2, y2);
    const BlockRun ref = run_block(X, X2, y, y2, true, false, true);
    const BlockRun got = run_block(X, X2, y, y2, false, false, true);
    const int d = first_diff(ref.sweeps, got.sweeps);
    std::printf("    stored fits in [%zu, %zu]; %ld sweeps refill the store while f(X) is kept\n",
                got.store_min, got.store_max, got.refills_with_kept_fit);
    check(got.store_min == 0 && got.store_max == 20 && got.refills_with_kept_fit >= 5,
          "store released and refilled within the chain");
    check(d < 0, "chain == Forest::do_gibbs chain");
    if (d >= 0) std::printf("    first differing sweep: %d\n", d);
}

// ---------------------------------------------------------------------------
// D. update_num_tree with the bound at the number of trees
// ---------------------------------------------------------------------------
static Forest make_forest(arma::uword p, int ntrees, bool num_tree) {
    Hypers h;
    h.alpha = 1.0; h.beta = 2.0; h.gamma = 0.95; h.sigma = 1.0; h.sigma_hat = 1.0;
    h.sigma_mu = 0.5 / (2.0 * std::sqrt((double)ntrees)); h.shape = 1.0; h.width = 0.1;
    h.num_tree = ntrees; h.tau_rate = 10.0; h.num_tree_prob = 2.0 / ntrees; h.temperature = 1.0;
    h.alpha_scale = (double)p; h.alpha_shape_1 = 0.5; h.alpha_shape_2 = 1.0;
    h.group = arma::regspace<arma::uvec>(0, p - 1);
    h.finalize_from_group();
    Opts o;
    o.update_sigma_mu = true; o.update_s = true; o.update_alpha = true; o.update_beta = false;
    o.update_gamma = false; o.update_tau = true; o.update_tau_mean = false;
    o.update_num_tree = num_tree; o.update_sigma = true; o.num_burn = 0; o.num_thin = 1;
    o.num_save = 1; o.num_print = 1000000; o.cache_trees = false;
    return Forest(h, o);
}

static Sweep forest_sweep(Forest& f, const arma::vec& fit) {
    Sweep w; w.rng = bart_rng::engine(); w.fit = fit;
    w.sigma = f.get_sigma(); w.sigma_mu = f.get_sigma_mu();
    w.trees = softbart::serialize_forest_to_str(f.get_trees());
    return w;
}

static void partD() {
    std::printf("D. Forest with update_num_tree, N = 200, bound at the number of trees\n");
    const int M = 150;
    arma::mat X; arma::vec y; make_data(200, 4, 5, X, y);
    const arma::mat X0(0, X.n_cols);
    const unsigned long long N = X.n_rows;
    std::vector<Sweep> ref;
    {
        bart_rng::set_seed(17);
        Forest f = make_forest(X.n_cols, 20, true);
        for (int s = 0; s < M; ++s) { f.do_gibbs(X, y, X0, 1); ref.push_back(forest_sweep(f, f.do_predict(X))); }
    }
    // offset 0: bound = ntrees * N, so a sweep that adds a tree ends without the
    // store. offset 1: bound = ntrees * N - 1, so a sweep starts without the
    // store and one that removes a tree ends with it.
    for (int offset = 0; offset < 2; ++offset) {
        bart_rng::set_seed(17);
        Forest f = make_forest(X.n_cols, 20, true);
        std::vector<Sweep> got;
        long changed = 0, mismatch = 0;
        std::size_t tmin = 1000, tmax = 0;
        for (int s = 0; s < M; ++s) {
            const std::size_t nt0 = f.get_trees().size();
            g_store_bound = N * nt0 - (unsigned long long)offset;
            const arma::vec& fit = f.do_gibbs_fit(X, y, 1);
            got.push_back(forest_sweep(f, fit));
            const std::size_t nt1 = f.get_trees().size();
            tmin = std::min(tmin, nt1); tmax = std::max(tmax, nt1);
            if (f.tree_fit_store_size() != (Forest::fit_store_allowed(nt1, X.n_rows) ? nt1 : 0)) ++mismatch;
            if (offset == 0 ? nt1 > nt0 : nt1 < nt0) ++changed;
        }
        g_store_bound = 4000;
        const int d = first_diff(ref, got);
        std::printf("    bound ntrees * N%s: trees in [%zu, %zu]; %ld sweeps %s\n",
                    offset == 0 ? "" : " - 1", tmin, tmax, changed,
                    offset == 0 ? "add a tree and end without the store"
                                : "remove a tree and end with the store");
        char what[160];
        std::snprintf(what, sizeof what, "bound ntrees * N%s: store as bounded, chains equal (first diff %d)",
                      offset == 0 ? "" : " - 1", d);
        check(d < 0 && mismatch == 0 && changed > 0, what);
    }
}

// ---------------------------------------------------------------------------
// E. two X of different row counts
// ---------------------------------------------------------------------------
static void partE() {
    std::printf("E. one forest, X with 150 rows (store) and 250 rows (no store) in turn\n");
    const int M = 80;
    arma::mat Xa, Xb; arma::vec ya, yb;
    make_data(150, 3, 6, Xa, ya); make_data(250, 3, 7, Xb, yb);
    const arma::mat X0(0, 3);
    std::vector<Sweep> ref, got;
    long stored_a = 0, stored_b = 0;
    for (int pass = 0; pass < 2; ++pass) {
        bart_rng::set_seed(23);
        Forest f = make_forest(3, 20, false);
        for (int s = 0; s < M; ++s) {
            const bool a = ((s / 4) % 2) == 0;
            const arma::mat& X = a ? Xa : Xb;
            const arma::vec& y = a ? ya : yb;
            if (pass == 0) {
                f.do_gibbs(X, y, X0, 1);
                ref.push_back(forest_sweep(f, f.do_predict(X)));
            } else {
                const arma::vec& fit = f.do_gibbs_fit(X, y, a ? 1 : 2);
                got.push_back(forest_sweep(f, fit));
                if (f.tree_fit_store_size() == 20) (a ? stored_a : stored_b)++;
            }
        }
    }
    const int d = first_diff(ref, got);
    check(stored_a == M / 2 && stored_b == 0, "store used at 150 rows and released at 250 rows");
    check(d < 0, "do_gibbs_fit chain == do_gibbs chain");
    if (d >= 0) std::printf("    first differing sweep: %d\n", d);
}

int main() {
    partA();
    partB();
    partC();
    partD();
    partE();
    partF();
    std::printf("%s\n", g_fail == 0 ? "ALL PASS" : "SOME FAILED");
    return g_fail == 0 ? 0 : 1;
}

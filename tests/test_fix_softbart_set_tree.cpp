// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_softbart_set_tree.cpp -- softbart_block::set_tree restores every
//  node field the SoftBart kernel reads, so that set_tree(get_tree()) leaves
//  the transition kernel unchanged.
//
//  A. Constructed trees, exact values. The cutpoint limits [lower, upper] of
//     every branch equal those of the nearest same-variable ancestor's side
//     (the values BirthLeaves / perturb_decision_rule keep), so the birth range
//     and the cutpoint prior used in the perturb ratio are the kernel's; every
//     leaf has its parent's tau; a stump keeps its tau through get_tree /
//     set_tree, and a stump written without a tau gets the tau of a new root.
//  B. Chains. From the same seed, a chain that calls set_tree(get_tree()) before
//     every sweep is bitwise identical to the uninterrupted chain (fits, sigma,
//     every kernel node field), for the block driven as composite_block drives
//     it and inside a composite with an external sigma block; the restore
//     points include stumps whose tau differs from the initial width and
//     branches with a same-variable ancestor.
//  C. set_tree of an earlier forest restores that forest's node fields
//     exactly, and predict_history reads forest snapshots that contain stumps.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/softbart_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/univariate_slice_sampling_block.hpp"
#include "AI4BayesCode/constraints.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::softbart_block;
using AI4BayesCode::softbart_block_config;
using AI4BayesCode::univariate_slice_sampling_block;
using AI4BayesCode::univariate_slice_sampling_block_config;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

// Kernel-read fields of a node, in pre-order: branch var, val, lower, upper,
// tau; leaf mu, tau. (A leaf's var/val/lower/upper and every current_weight
// below the root are not read before being rewritten.)
static void fields(const Node* n, std::vector<double>& out) {
    out.push_back(n->is_leaf ? 1.0 : 0.0);
    if (n->is_leaf) { out.push_back(n->mu); out.push_back(n->tau); return; }
    out.push_back(n->var); out.push_back(n->val); out.push_back(n->lower);
    out.push_back(n->upper); out.push_back(n->tau);
    fields(n->left, out); fields(n->right, out);
}
static std::vector<double> forest_fields(softbart_block& b) {
    std::vector<double> out;
    for (const Node* t : b.underlying()->forest()->get_trees()) {
        out.push_back(t->current_weight);
        fields(t, out);
    }
    return out;
}
static bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), sizeof(double) * a.size()) == 0);
}
static bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           (a.n_elem == 0 || std::memcmp(a.memptr(), b.memptr(), sizeof(double) * a.n_elem) == 0);
}

// Restore points that exercise the restored fields.
struct Coverage { long stumps_off_width = 0, same_var_branches = 0, branches = 0; };
static void same_var_count(const Node* n, std::vector<int>& path, Coverage& c) {
    if (n->is_leaf) return;
    ++c.branches;
    for (int v : path) if (v == n->var) { ++c.same_var_branches; break; }
    path.push_back(n->var);
    same_var_count(n->left, path, c); same_var_count(n->right, path, c);
    path.pop_back();
}
static void coverage(softbart_block& b, double width, Coverage& c) {
    for (const Node* t : b.underlying()->forest()->get_trees()) {
        if (t->is_leaf && t->tau != width) ++c.stumps_off_width;
        std::vector<int> path; same_var_count(t, path, c);
    }
}

static void make_data(int N, int p, std::uint64_t seed, arma::mat& X, arma::vec& y) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> Z(0.0, 1.0);
    X.set_size(N, p); y.set_size(N);
    for (int j = 0; j < p; ++j) for (int i = 0; i < N; ++i) X(i, j) = U(g);
    for (int i = 0; i < N; ++i) {
        double f = std::sin(6.0 * X(i, 0));
        if (p > 1) f += 2.0 * (X(i, 1) - 0.5) * (X(i, 1) - 0.5);
        y[i] = f + 0.3 * Z(g);
    }
}

// ---------------------------------------------------------------------------
// A. constructed trees
// ---------------------------------------------------------------------------
static bool partA() {
    std::printf("A. constructed trees\n");
    arma::mat X; arma::vec y; make_data(50, 2, 1, X, y);
    softbart_block_config cfg;
    cfg.x_train = X; cfg.y_init = y; cfg.sigma_key = ""; cfg.ntrees = 1;
    softbart_block b(cfg);
    const double width = b.underlying()->forest()->get_hypers().width;

    // root x0 < 0.5, left child x0 < 0.25 (nested split on the same variable).
    b.set_tree("1\nB 0 0.5 0.1\nB 0 0.25 0.1\nL 0.1\nL 0.2\nL 0.3\n---\n");
    Node* r = b.underlying()->forest()->get_trees()[0];
    Node* lc = r->left;
    check(r->lower == 0.0 && r->upper == 1.0, "root limits [0, 1]");
    check(lc->lower == 0.0 && lc->upper == 0.5, "left child x0 < 0.25 under x0 < 0.5: limits [0, 0.5]");
    check(lc->left->tau == 0.1 && lc->right->tau == 0.1 && r->right->tau == 0.1,
          "every leaf has the tree's tau 0.1");
    check(calc_cutpoint_likelihood(r) == 2.0, "cutpoint prior 1/(1-0) * 1/(0.5-0) = 2");
    {   // birth on x0 at the left child's right leaf: range [0.25, 0.5]
        Node* leaf = lc->right; const int v = leaf->var; leaf->var = 0; leaf->GetLimits();
        check(leaf->lower == 0.25 && leaf->upper == 0.5, "birth on x0 at that leaf draws val in [0.25, 0.5]");
        leaf->var = v;
    }

    // root x0 < 0.6; right child x1 < 0.3; its left child x0 < 0.8; its right
    // child x0 < 0.7 (nearest same-variable ancestor is the root, from the right).
    b.set_tree("1\nB 0 0.6 0.2\nL 1\nB 1 0.3 0.2\nB 0 0.8 0.2\nL 2\nL 3\nB 0 0.7 0.2\nL 4\nL 5\n---\n");
    r = b.underlying()->forest()->get_trees()[0];
    Node* rc = r->right;
    check(rc->lower == 0.0 && rc->upper == 1.0, "x1 branch under x0 splits: limits [0, 1]");
    check(rc->left->lower == 0.6 && rc->left->upper == 1.0 &&
          rc->right->lower == 0.6 && rc->right->upper == 1.0,
          "x0 branches right of x0 < 0.6 (through x1): limits [0.6, 1]");
    check(r->left->tau == 0.2 && rc->left->left->tau == 0.2 && rc->right->right->tau == 0.2,
          "leaves of a tree with tau 0.2 have tau 0.2");

    // A stump keeps its tau.
    bool ok = true;
    try { b.set_tree("1\nL 0.05 0.37\n---\n"); } catch (const std::exception& e) { ok = false; std::printf("    (%s)\n", e.what()); }
    check(ok && b.underlying()->forest()->get_trees()[0]->tau == 0.37, "stump written with tau 0.37 is read with tau 0.37");
    ok = true;
    try { b.set_tree(b.get_tree()); } catch (const std::exception& e) { ok = false; std::printf("    (%s)\n", e.what()); }
    check(ok && b.underlying()->forest()->get_trees()[0]->tau == 0.37, "set_tree(get_tree()) keeps a stump's tau 0.37");
    b.set_tree("1\nL 0.05\n---\n");
    check(b.underlying()->forest()->get_trees()[0]->tau == width,
          "stump written without tau gets the tau of a new root (width)");

    // Invalid input leaves the forest unchanged.
    try { b.set_tree("1\nL 0.05 0.37\n---\n"); } catch (const std::exception&) {}
    const std::string before = b.get_tree();
    bool threw = false;
    try { b.set_tree("1\nL 0.05 0\n---\n"); } catch (const std::exception&) { threw = true; }
    check(threw && b.get_tree() == before, "stump tau 0 is rejected and the forest is unchanged");
    return true;
}

// ---------------------------------------------------------------------------
// B. chains: uninterrupted vs set_tree(get_tree()) before every sweep
// ---------------------------------------------------------------------------
static void partB_block() {
    std::printf("B1. softbart_block driven by set_context + step (internal sigma)\n");
    arma::mat X; arma::vec y; make_data(200, 1, 2, X, y);   // p = 1: nested splits share x0
    softbart_block_config cfg;
    cfg.x_train = X; cfg.y_init = y; cfg.sigma_key = ""; cfg.ntrees = 20;
    block_context ctx; ctx["softbart_target"] = y;
    std::mt19937_64 rng(3);
    const int M = 150;

    bart_rng::set_seed(77);
    softbart_block A(cfg);
    std::vector<arma::vec> fA; std::vector<double> sA; std::vector<std::vector<double> > nA;
    for (int s = 0; s < M; ++s) {
        A.set_context(ctx); A.step(rng);
        fA.push_back(A.current()); sA.push_back(A.current_sigma()); nA.push_back(forest_fields(A));
    }

    bart_rng::set_seed(77);
    softbart_block B(cfg);
    const double width = B.underlying()->forest()->get_hypers().width;
    Coverage cov;
    int first_diff = -1, restore_diff = -1;
    for (int s = 0; s < M; ++s) {
        if (s > 0) {
            const std::vector<double> pre = forest_fields(B);
            B.set_tree(B.get_tree());
            if (restore_diff < 0 && !same_bits(pre, forest_fields(B))) restore_diff = s;
            coverage(B, width, cov);
        }
        B.set_context(ctx); B.step(rng);
        if (first_diff < 0 && (!same_bits(B.current(), fA[s]) || B.current_sigma() != sA[s] ||
                               !same_bits(forest_fields(B), nA[s]))) first_diff = s;
    }
    std::printf("    restore points: %ld stumps with tau != width, %ld of %ld branches with a same-variable ancestor\n",
                cov.stumps_off_width, cov.same_var_branches, cov.branches);
    check(cov.stumps_off_width > 0 && cov.same_var_branches > 0, "restore points include both cases");
    check(restore_diff < 0, "set_tree(get_tree()) leaves every kernel node field unchanged");
    check(first_diff < 0, "chain with a restore before every sweep == uninterrupted chain");
    if (restore_diff >= 0) std::printf("    first restore that changed a field: sweep %d\n", restore_diff);
    if (first_diff >= 0) std::printf("    first differing sweep: %d\n", first_diff);
}

struct Model {
    std::unique_ptr<composite_block> c;
    softbart_block* sb = nullptr;
};
static double sigma_log_density(const arma::vec& s, const block_context& ctx, arma::vec* g) {
    const double sigma = s[0];
    const arma::vec& yy = ctx.at("y");
    const arma::vec& f = ctx.at("f_softbart");
    const double ss = arma::accu(arma::square(yy - f)) + 1.0;
    const double n = (double)yy.n_elem;
    if (g) { g->set_size(1); (*g)[0] = -(n + 4.0) / sigma + ss / (sigma * sigma * sigma); }
    return -(n + 4.0) * std::log(sigma) - 0.5 * ss / (sigma * sigma);
}
static Model make_model(const arma::mat& X, const arma::vec& y) {
    Model m;
    m.c = std::make_unique<composite_block>("M");
    m.c->data().set("y", y);
    m.c->data().set("softbart_target", y);
    m.c->data().set("sigma", arma::vec{arma::stddev(y)});
    m.c->data().set("f_softbart", arma::vec(X.n_rows, arma::fill::zeros));
    m.c->data().declare_dependencies("f_softbart", {"softbart_target", "sigma"});
    m.c->data().declare_dependencies("sigma", {"y", "f_softbart"});
    softbart_block_config cfg;
    cfg.x_train = X; cfg.y_init = y; cfg.ntrees = 20;
    auto blk = std::make_unique<softbart_block>(cfg);
    m.sb = blk.get();
    m.c->add_child(std::move(blk));
    univariate_slice_sampling_block_config sg;
    sg.name = "sigma";
    sg.initial_unc = arma::vec{std::log(arma::stddev(y))};
    sg.constrain = AI4BayesCode::constraints::positive::constrain;
    sg.unconstrain = AI4BayesCode::constraints::positive::unconstrain;
    sg.log_density = [](const arma::vec& th, const block_context& ctx) {
        return AI4BayesCode::constraints::positive::wrap(th, nullptr,
            [&](const arma::vec& s, arma::vec* g) { return sigma_log_density(s, ctx, g); });
    };
    m.c->add_child(std::make_unique<univariate_slice_sampling_block>(std::move(sg)));
    return m;
}

static void partB_composite() {
    std::printf("B2. composite_block: softbart_block + slice sigma block\n");
    arma::mat X; arma::vec y; make_data(200, 2, 4, X, y);
    const int M = 150;
    std::vector<arma::vec> fA, gA;
    {
        bart_rng::set_seed(91); std::mt19937_64 rng(5);
        Model m = make_model(X, y);
        for (int s = 0; s < M; ++s) {
            m.c->step(rng);
            fA.push_back(m.c->data().get("f_softbart")); gA.push_back(m.c->data().get("sigma"));
        }
    }
    bart_rng::set_seed(91); std::mt19937_64 rng(5);
    Model m = make_model(X, y);
    int first_diff = -1;
    for (int s = 0; s < M; ++s) {
        if (s > 0) { m.sb->set_tree(m.sb->get_tree()); m.c->data().set("f_softbart", m.sb->current()); }
        m.c->step(rng);
        if (first_diff < 0 && (!same_bits(m.c->data().get("f_softbart"), fA[s]) ||
                               !same_bits(m.c->data().get("sigma"), gA[s]))) first_diff = s;
    }
    check(first_diff < 0, "composite chain with a restore before every sweep == uninterrupted chain");
    if (first_diff >= 0) std::printf("    first differing sweep: %d\n", first_diff);
}

// ---------------------------------------------------------------------------
// C. earlier forest; snapshots with stumps
// ---------------------------------------------------------------------------
// Number of trees in a serialized forest that are stumps (first line "L ...").
static int stumps_in(const std::string& s) {
    int n = 0; bool tree_start = false; std::size_t pos = 0;
    while (pos < s.size()) {
        std::size_t e = s.find('\n', pos); if (e == std::string::npos) e = s.size();
        const std::string line = s.substr(pos, e - pos);
        if (tree_start && !line.empty() && line[0] == 'L') ++n;
        tree_start = (pos == 0) || line == "---";
        pos = e + 1;
    }
    return n;
}

static void partC() {
    std::printf("C. earlier forest and forest snapshots\n");
    arma::mat X; arma::vec y; make_data(200, 1, 6, X, y);
    softbart_block_config cfg;
    cfg.x_train = X; cfg.y_init = y; cfg.sigma_key = ""; cfg.ntrees = 20;
    block_context ctx; ctx["softbart_target"] = y;
    {
        std::mt19937_64 rng(7);
        bart_rng::set_seed(8);
        softbart_block b(cfg);
        for (int s = 0; s < 40; ++s) { b.set_context(ctx); b.step(rng); }
        const std::string saved = b.get_tree();
        const std::vector<double> saved_fields = forest_fields(b);
        const arma::vec saved_fit = b.current();
        for (int s = 0; s < 40; ++s) { b.set_context(ctx); b.step(rng); }
        b.set_tree(saved);
        check(same_bits(forest_fields(b), saved_fields), "set_tree(forest of sweep 40) restores its kernel node fields");
        check(same_bits(b.current(), saved_fit), "and its fit");
    }
    {
        std::mt19937_64 rng(9);
        bart_rng::set_seed(10);
        softbart_block b(cfg);
        b.set_keep_tree(true);
        b.set_keep_history(true);
        const int M = 60;
        for (int s = 0; s < M; ++s) { b.set_context(ctx); b.step(rng); }
        int with_stumps = 0;
        for (const std::string& t : b.get_tree_history()) with_stumps += stumps_in(t) > 0 ? 1 : 0;
        const arma::mat H = b.predict_history(X);
        const arma::mat F = b.get_history().at(cfg.name);   // f(X) after each sweep
        double max_err = 0.0;
        for (arma::uword d = 0; d < H.n_rows && d < F.n_rows; ++d)
            for (arma::uword i = 0; i < X.n_rows; ++i)
                max_err = std::max(max_err, std::fabs(H(d, i) - F(d, i)));
        std::printf("    %d of %d snapshots contain a stump; max |predict_history - fit| = %.3g\n",
                    with_stumps, M, max_err);
        check(H.n_rows == (arma::uword)M && F.n_rows == (arma::uword)M && with_stumps > 0 && max_err < 1e-12,
              "predict_history reads every snapshot (stumps included) as the fitted forest");
    }
}

int main() {
    partA();
    partB_block();
    partB_composite();
    partC();
    std::printf("%s\n", g_fail == 0 ? "ALL PASS" : "SOME FAILED");
    return g_fail == 0 ? 0 : 1;
}

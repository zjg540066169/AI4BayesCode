// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_held_history.cpp
//
//  record_held_history() of the blocks that cannot be frozen themselves:
//  composite_block::record_held_history() calls it on every child of a nested
//  composite that is frozen within another composite, and the free siblings
//  of the nested composite record one entry per sweep of the root. For a root
//  with children [inner = nested composite holding the block, p = beta_gibbs
//  block], inner frozen after k1 sweeps (held_history_check.hpp):
//    1. bart_block (internal sigma, keep_tree): the fit rows; the stored
//       forests, sigma, split counts, split probabilities and fmean of the
//       kernel; predict_history(X_new) -- k1 + k2 entries each, the held
//       entries equal to the last one recorded before the freeze;
//    2. hmm_block: the z rows, likewise.
//  Then the clear_history() check of clear_history_contract.hpp with the same
//  freeze after the call: right after clear_history() no entry is stored, so
//  the held entries are taken from the current state (for the BART kernel the
//  current forest, sigma, split counts, split probabilities and fmean), and
//  they equal the copies of the last stored entry that a run without the call
//  records. genbart_block and softbart_block are checked in
//  test_fix_held_history_genbart.cpp and test_fix_held_history_softbart.cpp.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/bart_block.hpp"
#include "AI4BayesCode/beta_gibbs_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/hmm_block.hpp"

#include "clear_history_contract.hpp"
#include "held_history_check.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

namespace ab   = AI4BayesCode;
namespace chk  = clear_history_check;
namespace held = held_history_check;

constexpr int K1 = 40, K2 = 30;

// root = [inner (holding `block`), p ~ Beta(2, 3)], keep_history on (and
// keep_tree when `trees`); model.between freezes inner.
chk::model nested(std::unique_ptr<ab::composite_block> inner, std::uint64_t seed,
                  bool trees, std::function<void(chk::buffers&)> extra = nullptr) {
    auto root = std::make_shared<ab::composite_block>("root");
    root->add_child(std::move(inner));
    ab::beta_gibbs_block_config c;
    c.name = "p";
    c.params_fn = [](const ab::block_context&) { return ab::beta_dist_params{2.0, 3.0}; };
    root->add_child(std::make_unique<ab::beta_gibbs_block>(std::move(c)));
    root->set_keep_history(true);
    if (trees) root->set_keep_tree(true);
    ab::composite_block* r = root.get();
    chk::model m = chk::composite_model(root, seed, std::move(extra));
    m.between = [r] { r->freeze(std::vector<std::string>{"inner"}); };
    return m;
}

// 1. bart_block "f" with an internal sigma; its kernel draws from the
// thread-local engine bart_rng, seeded before construction.
chk::model bart_case() {
    const std::size_t N = 60, p = 2;
    std::mt19937_64 g(139);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> nd(0.0, 0.3);
    arma::mat X(N, p);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        X(i, 0) = U(g); X(i, 1) = U(g);
        y[i] = (X(i, 0) > 0.5 ? 1.0 : -1.0) + X(i, 1) + nd(g);
    }
    bart_rng::set_seed(149);
    auto inner = std::make_unique<ab::composite_block>("inner");
    inner->data().set("bart_target", y);
    inner->data().declare_dependencies("f", {"bart_target"});
    ab::bart_block_config c;
    c.name = "f"; c.x_train = X; c.y_init = y;
    c.working_response_key = "bart_target"; c.sigma_key = "";
    c.ntrees = 10; c.numcut = 20;
    ab::bart_block* bb = nullptr;
    { auto b = std::make_unique<ab::bart_block>(std::move(c)); bb = b.get(); inner->add_child(std::move(b)); }
    const arma::mat X_new = X.rows(0, 9);
    return nested(std::move(inner), 151, true, [bb, X_new](chk::buffers& b) {
        const stdbart::BartHistory h = bb->get_full_history();
        b.strings["f: kernel tree_history"] = h.tree_history;
        b.mats["f: kernel sigma_history"] = arma::mat(h.sigma_history);
        b.mats["f: kernel var_counts_history"] =
            arma::conv_to<arma::mat>::from(h.var_counts_history);
        b.mats["f: kernel var_probs_history"] = h.var_probs_history;
        b.mats["f: kernel fmean_history"] = arma::mat(h.fmean_history);
        b.mats["f: predict_history(X_new)"] = bb->predict_history(X_new);
    });
}

// 2. hmm_block "z": two states, Gaussian emissions with means -1 and 1, sd 1.
chk::model hmm_case() {
    const std::size_t T = 50, Kst = 2;
    std::mt19937_64 g(157);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::vec y(T);
    for (std::size_t t = 0; t < T; ++t) y[t] = ((t / 10) % 2 == 0 ? -1.0 : 1.0) + nd(g);
    auto inner = std::make_unique<ab::composite_block>("inner");
    inner->data().set("y", y);
    inner->data().set("A", arma::vec{0.9, 0.1, 0.1, 0.9});
    inner->data().set("pi", arma::vec{0.5, 0.5});
    inner->data().declare_dependencies("z", {"y", "A", "pi"});
    ab::hmm_block_config c;
    c.name = "z"; c.T = T; c.K = Kst;
    c.emission_logp = [](std::size_t t, std::size_t k, const ab::block_context& ctx) {
        const double r = ctx.at("y")[t] - (k == 0 ? -1.0 : 1.0);
        return -0.5 * r * r;
    };
    inner->add_child(std::make_unique<ab::hmm_block>(std::move(c)));
    return nested(std::move(inner), 163, false);
}

}  // namespace

int main() {
    std::printf("=== test_fix_held_history ===\n\n");
    held::run("bart_block in a frozen nested composite", bart_case, K1, K2, {"p"});
    held::run("hmm_block in a frozen nested composite", hmm_case, K1, K2, {"p"});
    chk::run("bart_block held by a frozen composite after the call", bart_case, K1, K2);
    chk::run("hmm_block held by a frozen composite after the call", hmm_case, K1, K2);
    return chk::finish("test_fix_held_history");
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_held_history_softbart.cpp
//
//  The checks of test_fix_held_history.cpp for softbart_block (internal
//  sigma, keep_tree), which cannot share a translation unit with bart_block:
//  in a nested composite frozen within another composite, the fit rows; the
//  stored forests, sigma, sigma_mu, split probabilities, split counts and m
//  of the kernel; predict_history(X_new) -- one entry per sweep of the root, the
//  held entries equal to the last one recorded before the freeze
//  (held_history_check.hpp). Then the clear_history() check of
//  clear_history_contract.hpp with the same freeze after the call. The kernel
//  draws from the thread-local engine bart_rng, seeded before construction.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/beta_gibbs_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/softbart_block.hpp"

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

// root = [inner (holding softbart_block "f"), p ~ Beta(2, 3)], keep_history
// and keep_tree on; model.between freezes inner.
chk::model softbart_case() {
    const std::size_t N = 60, p = 3;
    std::mt19937_64 g(181);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> nd(0.0, 0.3);
    arma::mat X(N, p);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < p; ++j) X(i, j) = U(g);
        y[i] = std::sin(4.0 * X(i, 0)) + X(i, 1) + nd(g);
    }
    bart_rng::set_seed(191);
    auto inner = std::make_unique<ab::composite_block>("inner");
    inner->data().set("softbart_target", y);
    inner->data().declare_dependencies("f", {"softbart_target"});
    ab::softbart_block_config c;
    c.name = "f"; c.x_train = X; c.y_init = y;
    c.working_response_key = "softbart_target"; c.sigma_key = "";
    c.ntrees = 8;
    ab::softbart_block* sb = nullptr;
    { auto b = std::make_unique<ab::softbart_block>(std::move(c)); sb = b.get(); inner->add_child(std::move(b)); }
    auto root = std::make_shared<ab::composite_block>("root");
    root->add_child(std::move(inner));
    ab::beta_gibbs_block_config pc;
    pc.name = "p";
    pc.params_fn = [](const ab::block_context&) { return ab::beta_dist_params{2.0, 3.0}; };
    root->add_child(std::make_unique<ab::beta_gibbs_block>(std::move(pc)));
    root->set_keep_history(true);
    root->set_keep_tree(true);
    ab::composite_block* r = root.get();
    const arma::mat X_new = X.rows(0, 9);
    chk::model m = chk::composite_model(root, 193, [sb, X_new](chk::buffers& b) {
        const softbart::SoftbartHistory h = sb->get_full_history();
        b.strings["f: kernel tree_history"] = h.tree_history;
        b.mats["f: kernel sigma_history"] = arma::mat(h.sigma_history);
        b.mats["f: kernel sigma_mu_history"] = arma::mat(h.sigma_mu_history);
        b.mats["f: kernel center_history"] = arma::mat(h.center_history);
        b.mats["f: kernel s_history"] = h.s_history;
        b.mats["f: kernel var_counts_history"] =
            arma::conv_to<arma::mat>::from(h.var_counts_history);
        b.mats["f: predict_history(X_new)"] = sb->predict_history(X_new);
    });
    m.between = [r] { r->freeze(std::vector<std::string>{"inner"}); };
    return m;
}

}  // namespace

int main() {
    std::printf("=== test_fix_held_history_softbart ===\n\n");
    held::run("softbart_block in a frozen nested composite", softbart_case, K1, K2, {"p"});
    chk::run("softbart_block held by a frozen composite after the call", softbart_case, K1, K2);
    return chk::finish("test_fix_held_history_softbart");
}

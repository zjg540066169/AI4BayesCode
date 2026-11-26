// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_clear_history_softbart.cpp
//
//  The clear_history() check of test_fix_clear_history.cpp
//  (clear_history_contract.hpp) for softbart_block, which cannot share a
//  translation unit with bart_block: with an internal sigma and the kernel's
//  history recording on (keep_tree), the fits; the stored forests, sigma,
//  sigma_mu, split probabilities, split counts and m of the kernel; the
//  predictions over the stored forests. Once with DART, whose update of the
//  split probabilities starts after a number of sweeps the kernel counts
//  (clear_history() leaves that count unchanged). The kernel draws from the
//  thread-local engine bart_rng, seeded before each construction.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/softbart_block.hpp"

#include "clear_history_contract.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>

namespace {

namespace ab  = AI4BayesCode;
namespace chk = clear_history_check;

constexpr int K1 = 40, K2 = 30;

chk::model softbart_case(bool dart) {
    const std::size_t N = 60, p = 3;
    std::mt19937_64 g(107);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> nd(0.0, 0.3);
    arma::mat X(N, p);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < p; ++j) X(i, j) = U(g);
        y[i] = std::sin(4.0 * X(i, 0)) + X(i, 1) + nd(g);
    }
    bart_rng::set_seed(109);
    auto root = std::make_shared<ab::composite_block>("root");
    root->data().set("softbart_target", y);
    root->data().declare_dependencies("f", {"softbart_target"});
    ab::softbart_block_config c;
    c.name = "f"; c.x_train = X; c.y_init = y;
    c.working_response_key = "softbart_target"; c.sigma_key = "";
    c.ntrees = 8; c.dart = dart;
    ab::softbart_block* sb = nullptr;
    { auto b = std::make_unique<ab::softbart_block>(std::move(c)); sb = b.get(); root->add_child(std::move(b)); }
    root->set_keep_history(true);
    root->set_keep_tree(true);
    const arma::mat X_new = X.rows(0, 9);
    return chk::composite_model(root, 113, [sb, X_new](chk::buffers& b) {
        const softbart::SoftbartHistory h = sb->get_full_history();
        b.strings["f: kernel tree_history"] = h.tree_history;
        b.strings["f: get_tree_history()"] = sb->get_tree_history();
        b.mats["f: kernel sigma_history"] = arma::mat(h.sigma_history);
        b.mats["f: kernel sigma_mu_history"] = arma::mat(h.sigma_mu_history);
        b.mats["f: kernel center_history"] = arma::mat(h.center_history);
        b.mats["f: kernel s_history"] = h.s_history;
        b.mats["f: kernel var_counts_history"] =
            arma::conv_to<arma::mat>::from(h.var_counts_history);
        b.mats["f: predict_history(X_new)"] = sb->predict_history(X_new);
    });
}

}  // namespace

int main() {
    std::printf("=== test_fix_clear_history_softbart ===\n\n");
    chk::run("softbart_block with keep_tree", [] { return softbart_case(false); }, K1, K2);
    chk::run("softbart_block with keep_tree and DART", [] { return softbart_case(true); }, K1, K2);
    return chk::finish("test_fix_clear_history_softbart");
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_clear_history_genbart.cpp
//
//  The clear_history() check of test_fix_clear_history.cpp
//  (clear_history_contract.hpp) for the genBART family, which cannot share a
//  translation unit with bart_block:
//    1. genbart_block (Poisson likelihood) with the kernel's history
//       recording on (keep_tree): the r fits; the stored forests, nuisance,
//       sigma_mu, split probabilities and split counts of the kernel; the
//       predictions over the stored forests.
//    2. examples/GBartHeteroscedastic.cpp, which stores the phi of each sweep
//       in a member of its own: get_history() and predict_at() at the
//       training X (one y_rep row per stored draw, from r and phi of the
//       same sweep).
//  The kernels draw from the thread-local engine bart_rng, seeded before
//  each construction (by the GBartHeteroscedastic constructor from its
//  rng_seed in case 2).
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/genbart_block.hpp"

#define main gbart_heteroscedastic_demo_main
#include "../examples/GBartHeteroscedastic.cpp"
#undef main

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

void make_x(std::size_t N, std::size_t p, std::uint64_t seed, arma::mat& X) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    X.set_size(N, p);
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t i = 0; i < N; ++i) X(i, j) = U(g);
}

chk::model genbart_case() {
    const std::size_t N = 80, p = 2;
    arma::mat X;
    make_x(N, p, 67, X);
    std::mt19937_64 g(71);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        std::poisson_distribution<int> P(std::exp(0.5 + (X(i, 0) > 0.5 ? 1.0 : 0.0)));
        y[i] = static_cast<double>(P(g));
    }
    auto lik = std::make_shared<genbart::lik::poisson_lik>();
    bart_rng::set_seed(73);
    auto root = std::make_shared<ab::composite_block>("root");
    ab::genbart_block_config c;
    c.name = "r"; c.x_train = X; c.y_init = y; c.lik = lik.get(); c.ntrees = 10;
    ab::genbart_block* gb = nullptr;
    { auto b = std::make_unique<ab::genbart_block>(std::move(c)); gb = b.get(); root->add_child(std::move(b)); }
    root->set_keep_history(true);
    root->set_keep_tree(true);
    const arma::mat X_new = X.rows(0, 9);
    chk::model m = chk::composite_model(root, 79, [gb, X_new](chk::buffers& b) {
        const genbart::GenbartHistory h = gb->get_full_history();
        b.strings["r: kernel tree_history"] = h.tree_history;
        b.strings["r: get_tree_history()"] = gb->get_tree_history();
        b.mats["r: kernel sigma_history"] = arma::mat(h.sigma_history);
        b.mats["r: kernel sigma_mu_history"] = arma::mat(h.sigma_mu_history);
        b.mats["r: kernel s_history"] = h.s_history;
        b.mats["r: kernel var_counts_history"] =
            arma::conv_to<arma::mat>::from(h.var_counts_history);
        b.mats["r: predict_history(X_new)"] = gb->predict_history(X_new);
    });
    m.owners.push_back(lik);
    return m;
}

chk::model gbart_heteroscedastic_case() {
    const std::size_t N = 80, p = 2;
    arma::mat X;
    make_x(N, p, 83, X);
    std::mt19937_64 g(89);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double m = std::exp(1.0 + 0.6 * X(i, 0));
        y[i] = m + std::sqrt(0.5 * m) * nd(g);
    }
    auto w = std::make_shared<GBartHeteroscedastic>(X, y, 10, 1.0, 101, false, true);
    GBartHeteroscedastic* mdl = w.get();
    chk::model out;
    out.step    = [mdl](int n) { mdl->step(n); };
    out.clear   = [mdl] { mdl->clear_history(); };
    out.history = [mdl](chk::buffers& b) {
        for (auto& kv : mdl->get_history()) b.mats[kv.first] = kv.second;
    };
    out.extra   = [mdl](chk::buffers& b) {
        b.rows_only["predict_at({}): y_rep"] =
            mdl->predict_at(AI4BayesCode::state_map{}).at("y_rep");
    };
    out.owners.push_back(w);
    return out;
}

}  // namespace

int main() {
    std::printf("=== test_fix_clear_history_genbart ===\n\n");
    chk::run("genbart_block (Poisson) with keep_tree", genbart_case, K1, K2);
    chk::run("GBartHeteroscedastic", gbart_heteroscedastic_case, K1, K2);
    return chk::finish("test_fix_clear_history_genbart");
}

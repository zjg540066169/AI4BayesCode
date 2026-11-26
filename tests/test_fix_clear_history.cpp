// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_clear_history.cpp
//
//  clear_history() discards the stored draws and changes nothing that later
//  sweeps use: after k1 sweeps, clear_history() and k2 sweeps, every per-draw
//  buffer has k2 rows and equals, bit for bit, the last k2 rows of a run
//  without the call with the same seeds (clear_history_contract.hpp).
//    1. The four VI blocks, terminated within the k1 sweeps: the trajectory
//       '<name>__vi_history' (final_khat in its last column) and the q-draws;
//       final_khat itself is equal in both runs.
//    2. The same VI blocks inside a composite frozen after the call: the held
//       rows repeat the last rows the call removed, as the held rows of a run
//       without the call repeat its last rows.
//    3. A DP Gaussian mixture as in examples/DPGaussianMixture.cpp
//       (categorical z, Normal-Gamma atoms mu and lambda, stick-breaking pi
//       and stick_V, label swap); history_size() is 1 right after the call.
//    4. Blocks with more than one per-draw buffer: gmrf_gaussian_joint_block
//       (x, kappa), lda_collapsed_gibbs_block (z, theta, phi),
//       niw_cluster_gibbs_block (mu, sigma), order_mcmc_block (order,
//       log-score, sampled DAG) and bart_block (fits; stored forests, sigma,
//       split counts, split probabilities and fmean of the kernel;
//       predictions over the stored forests).
//    5. The wrappers with per-sweep members of their own: ARDLasso (no
//       composite_block, so a clear_history() of its own) and VCBart (sigma).
//  genbart_block with GBartHeteroscedastic, and softbart_block, are checked in
//  test_fix_clear_history_genbart.cpp and test_fix_clear_history_softbart.cpp.
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
#include "AI4BayesCode/bnp_utils.hpp"
#include "AI4BayesCode/categorical_gibbs_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/full_rank_gaussian_vi_block.hpp"
#include "AI4BayesCode/gmrf_gaussian_joint_block.hpp"
#include "AI4BayesCode/lda_collapsed_gibbs_block.hpp"
#include "AI4BayesCode/mean_field_categorical_vi_block.hpp"
#include "AI4BayesCode/mean_field_gaussian_vi_block.hpp"
#include "AI4BayesCode/niw_cluster_gibbs_block.hpp"
#include "AI4BayesCode/normal_gamma_cluster_gibbs_block.hpp"
#include "AI4BayesCode/order_mcmc_block.hpp"
#include "AI4BayesCode/stick_breaking_block.hpp"
#include "AI4BayesCode/stick_label_swap_block.hpp"
#include "AI4BayesCode/structured_categorical_vi_block.hpp"

#define main ardlasso_demo_main
#include "../examples/ARDLasso.cpp"
#undef main
#define main vcbart_demo_main
#include "../examples/VCBart.cpp"
#undef main

#include "clear_history_contract.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

namespace ab  = AI4BayesCode;
namespace chk = clear_history_check;

constexpr int K1 = 40, K2 = 30;

// ---- 1, 2. VI blocks --------------------------------------------------------

// Epochs close at sweeps 5, 10, 15 and 20, and the max_epochs cap terminates
// the optimizer at sweep 20 at the latest, within the K1 sweeps.
ab::vi_optimizer::raabbvi_config short_epochs() {
    ab::vi_optimizer::raabbvi_config o;
    o.inner_iter_per_epoch = 5;
    o.max_epochs = 4;
    o.S_khat = 200;
    return o;
}

// Gaussian target with mean (1, -0.5) and precision [[2, -0.8], [-0.8, 1.5]].
double gauss_lp(const arma::vec& t, arma::vec* g) {
    const double a = t[0] - 1.0, b = t[1] + 0.5;
    if (g) {
        g->set_size(2);
        (*g)[0] = -(2.0 * a - 0.8 * b);
        (*g)[1] = -(-0.8 * a + 1.5 * b);
    }
    return -0.5 * (2.0 * a * a - 1.6 * a * b + 1.5 * b * b);
}

// Three categorical latents with 2, 3 and 2 states.
double cat_lp(const arma::uvec& z) {
    return 0.8 * (z[0] == 1) + 0.5 * (z[1] == 2) - 0.3 * (z[1] == 0) +
           1.0 * (z[0] == z[2]);
}

const char* const VI_NAME[] = {"mean_field_gaussian_vi_block",
                               "full_rank_gaussian_vi_block",
                               "mean_field_categorical_vi_block",
                               "structured_categorical_vi_block"};

std::unique_ptr<ab::vi_block> make_vi(int type, const std::string& nm) {
    switch (type) {
    case 0: {
        ab::mean_field_gaussian_vi_block_config c;
        c.name = nm;
        c.initial_unc = arma::vec(2, arma::fill::zeros);
        c.log_density_grad = [](const arma::vec& t, const ab::block_context&,
                                arma::vec* g) { return gauss_lp(t, g); };
        c.optimizer = short_epochs();
        return std::make_unique<ab::mean_field_gaussian_vi_block>(std::move(c));
    }
    case 1: {
        ab::full_rank_gaussian_vi_block_config c;
        c.name = nm;
        c.initial_unc = arma::vec(2, arma::fill::zeros);
        c.log_density_grad = [](const arma::vec& t, const ab::block_context&,
                                arma::vec* g) { return gauss_lp(t, g); };
        c.optimizer = short_epochs();
        return std::make_unique<ab::full_rank_gaussian_vi_block>(std::move(c));
    }
    case 2: {
        ab::mean_field_categorical_vi_block_config c;
        c.name = nm;
        c.cardinalities = arma::uvec{2, 3, 2};
        c.log_density = [](const arma::uvec& z, const ab::block_context&) {
            return cat_lp(z);
        };
        c.optimizer = short_epochs();
        return std::make_unique<ab::mean_field_categorical_vi_block>(std::move(c));
    }
    default: {
        ab::structured_categorical_vi_block_config c;
        c.name = nm;
        c.cardinalities = arma::uvec{2, 3, 2};
        c.clique_partition = {{0, 2}, {1}};
        c.log_density = [](const arma::uvec& z, const ab::block_context&) {
            return cat_lp(z);
        };
        c.optimizer = short_epochs();
        return std::make_unique<ab::structured_categorical_vi_block>(std::move(c));
    }
    }
}

void vi_contract(int type) {
    auto make = [type]() {
        auto root = std::make_shared<ab::composite_block>("root");
        ab::vi_block* vi = nullptr;
        { auto b = make_vi(type, "w"); vi = b.get(); root->add_child(std::move(b)); }
        root->set_keep_history(true);
        chk::model m = chk::composite_model(root, 101);
        m.extra = [vi](chk::buffers& b) {
            b.fixed["w: final_khat"] =
                arma::mat(1, 1, arma::fill::value(vi->vi_history().final_khat));
        };
        return m;
    };
    {
        chk::model m = make();
        m.step(K1);
        const double khat = chk::collect(m).fixed.at("w: final_khat")(0, 0);
        chk::check(std::isfinite(khat),
                   std::string(VI_NAME[type]) + ": the optimizer terminates "
                   "within the k1 sweeps (final_khat " + std::to_string(khat) + ")");
    }
    chk::run(std::string(VI_NAME[type]) + ": trajectory and q-draws", make, K1, K2);
}

// The VI block in an inner composite, a beta_gibbs_block next to it; the
// inner composite is frozen after the k1 sweeps (after clear_history() in A).
void vi_frozen_after_clear(int type) {
    auto make = [type]() {
        auto inner = std::make_unique<ab::composite_block>("inner");
        inner->add_child(make_vi(type, "w"));
        auto root = std::make_shared<ab::composite_block>("root");
        root->add_child(std::move(inner));
        ab::beta_gibbs_block_config c;
        c.name = "p";
        c.params_fn = [](const ab::block_context&) { return ab::beta_dist_params{2.0, 3.0}; };
        root->add_child(std::make_unique<ab::beta_gibbs_block>(std::move(c)));
        root->set_keep_history(true);
        ab::composite_block* r = root.get();
        chk::model m = chk::composite_model(root, 202);
        m.between = [r] { r->freeze(std::vector<std::string>{"inner"}); };
        return m;
    };
    chk::run(std::string(VI_NAME[type]) + " held by a frozen composite after the call",
             make, K1, K2);
}

// ---- 3. DP Gaussian mixture with a label swap --------------------------------

chk::model dp_mixture_case() {
    constexpr std::size_t N = 60, K = 8;
    auto comp = std::make_shared<ab::composite_block>("dpmm");
    auto& D = comp->data();
    std::mt19937_64 g(5u);
    std::normal_distribution<double> nd(0.0, 0.7);
    arma::vec y(N), z0(N), pi0(K, arma::fill::value(1.0 / K)), V0(K);
    for (std::size_t i = 0; i < N; ++i) {
        y[i] = (i % 2 == 0 ? -3.0 : 3.0) + nd(g);
        z0[i] = static_cast<double>(i % 3 + 1);
    }
    double rem = 1.0;
    for (std::size_t k = 0; k + 1 < K; ++k) { V0[k] = pi0[k] / rem; rem *= 1.0 - V0[k]; }
    V0[K - 1] = 1.0;
    arma::vec mu0(K), lam0(K, arma::fill::ones);
    for (std::size_t k = 0; k < K; ++k) mu0[k] = y[k];
    D.set("y", y); D.set("z", z0); D.set("pi", pi0); D.set("stick_V", V0);
    D.set("mu", mu0); D.set("lambda", lam0); D.set("alpha", arma::vec{1.0});
    D.set("cluster_counts", arma::vec(K, arma::fill::zeros));
    D.register_refresher("cluster_counts", [](const ab::shared_data_t& s) -> arma::vec {
        return ab::bnp::counts_from_z(s.get("z"), K);
    });
    D.declare_dependencies("z", {"y", "pi", "mu", "lambda"});
    D.declare_invalidates("z", {"cluster_counts"});
    D.declare_dependencies("cluster_params", {"z", "y"});
    D.declare_dependencies("pi", {"cluster_counts", "alpha"});
    D.declare_dependencies("label_swap", {"z", "mu", "lambda", "pi", "stick_V", "alpha"});
    D.declare_invalidates("label_swap", {"cluster_counts"});
    D.refresh_all();
    {
        ab::categorical_gibbs_block_config c;
        c.name = "z"; c.n_obs = N; c.n_categories = K; c.initial_labels = z0;
        c.log_probs_fn = [](const ab::block_context& ctx) -> arma::mat {
            const arma::vec& yy = ctx.at("y");
            const arma::vec& pi = ctx.at("pi");
            const arma::vec& mu = ctx.at("mu");
            const arma::vec& lam = ctx.at("lambda");
            arma::mat lp(N, K);
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t k = 0; k < K; ++k) {
                    const double r = yy[i] - mu[k];
                    lp(i, k) = std::log(pi[k] + 1e-300) + 0.5 * std::log(lam[k]) -
                               0.5 * lam[k] * r * r;
                }
            return lp;
        };
        comp->add_child(std::make_unique<ab::categorical_gibbs_block>(std::move(c)));
    }
    {
        ab::normal_gamma_cluster_gibbs_block_config c;
        c.name = "cluster_params"; c.K_trunc = K; c.d = 1; c.N = N;
        c.z_key = "z"; c.y_key = "y"; c.mu_name = "mu"; c.lambda_name = "lambda";
        c.mu_0 = arma::vec{0.0}; c.kappa_0 = 0.1; c.a_lambda_0 = 2.0; c.b_lambda_0 = 1.0;
        c.initial_mu = mu0; c.initial_lambda = lam0;
        comp->add_child(std::make_unique<ab::normal_gamma_cluster_gibbs_block>(std::move(c)));
    }
    {
        ab::stick_breaking_block_config c;
        c.name = "pi"; c.K_trunc = K; c.counts_key = "cluster_counts"; c.v_name = "stick_V";
        c.a_fn = [](std::size_t k, const arma::vec& counts, const ab::block_context&) {
            return 1.0 + counts[k];
        };
        c.b_fn = [](std::size_t k, const arma::vec& counts, const ab::block_context& ctx) {
            double t = 0.0;
            for (std::size_t j = k + 1; j < counts.n_elem; ++j) t += counts[j];
            return ctx.at("alpha")[0] + t;
        };
        c.initial_pi = pi0;
        comp->add_child(std::make_unique<ab::stick_breaking_block>(std::move(c)));
    }
    {
        ab::stick_label_swap_block_config c;
        c.name = "label_swap"; c.K_trunc = K; c.atom_keys = {{"mu", 1}, {"lambda", 1}};
        comp->add_child(std::make_unique<ab::stick_label_swap_block>(std::move(c)));
    }
    comp->set_keep_history(true);
    return chk::composite_model(comp, 20261122u);
}

// ---- 4. blocks with more than one per-draw buffer ----------------------------

chk::model gmrf_joint_case() {
    const arma::uword P = 4, n = P * P;
    arma::sp_mat R(n, n);
    for (arma::uword r = 0; r < P; ++r)
        for (arma::uword c = 0; c < P; ++c) {
            const arma::uword k = r * P + c;
            if (c + 1 < P) { R(k, k + 1) = -1; R(k + 1, k) = -1; R(k, k) += 1; R(k + 1, k + 1) += 1; }
            if (r + 1 < P) { R(k, k + P) = -1; R(k + P, k) = -1; R(k, k) += 1; R(k + P, k + P) += 1; }
        }
    ab::gmrf_gaussian_joint_block_config c;
    c.name = "x"; c.n = n; c.R = R; c.sigma2 = 0.25;
    c.y = arma::sin(arma::linspace<arma::vec>(0.0, 9.0, n));
    c.kappa_a = 1.0; c.kappa_b = 0.01; c.kappa_init = 3.0;
    c.log_kappa_rw_sd = 2.0; c.sum_to_zero = true;
    auto root = std::make_shared<ab::composite_block>("root");
    root->add_child(std::make_unique<ab::gmrf_gaussian_joint_block>(std::move(c)));
    root->set_keep_history(true);
    return chk::composite_model(root, 303);
}

chk::model lda_case() {
    const std::size_t M = 6, V = 5, K = 2, L = 20, N = M * L;
    std::mt19937_64 g(17);
    std::uniform_int_distribution<int> word(1, static_cast<int>(V));
    arma::vec w(N), doc(N), z0(N);
    for (std::size_t t = 0; t < N; ++t) {
        w[t] = static_cast<double>(word(g));
        doc[t] = static_cast<double>(t / L + 1);
        z0[t] = static_cast<double>(t % K + 1);
    }
    auto root = std::make_shared<ab::composite_block>("root");
    root->data().set("w", w);
    root->data().set("doc", doc);
    root->data().declare_dependencies("lda", {"w", "doc"});
    ab::lda_collapsed_gibbs_block_config c;
    c.name = "lda"; c.M = M; c.V = V; c.K = K;
    c.alpha = arma::vec(K, arma::fill::value(0.5));
    c.beta = arma::vec(V, arma::fill::value(0.5));
    c.z_init = z0;
    root->add_child(std::make_unique<ab::lda_collapsed_gibbs_block>(std::move(c)));
    root->set_keep_history(true);
    return chk::composite_model(root, 404);
}

chk::model niw_case() {
    const std::size_t N = 40, d = 2, K = 3;
    std::mt19937_64 g(23);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::vec y(N * d), z(N);
    for (std::size_t i = 0; i < N; ++i) {
        z[i] = static_cast<double>(i % K + 1);
        for (std::size_t j = 0; j < d; ++j)
            y[i * d + j] = 2.0 * static_cast<double>(i % K) - 2.0 + nd(g);
    }
    arma::vec mu0(K * d, arma::fill::zeros), sig0(K * d * d, arma::fill::zeros);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < d; ++j) sig0[k * d * d + j * d + j] = 1.0;
    auto root = std::make_shared<ab::composite_block>("root");
    root->data().set("y", y);
    root->data().set("z", z);
    root->data().set("mu", mu0);
    root->data().set("sigma", sig0);
    root->data().declare_dependencies("niw", {"z", "y"});
    ab::niw_cluster_gibbs_block_config c;
    c.name = "niw"; c.K_trunc = K; c.d = d; c.N = N;
    c.mu_0 = arma::vec(d, arma::fill::zeros); c.kappa_0 = 0.1;
    c.Psi_0_flat = arma::vec{1.0, 0.0, 0.0, 1.0}; c.nu_0 = 4.0;
    c.initial_mu = mu0; c.initial_sigma = sig0;
    root->add_child(std::make_unique<ab::niw_cluster_gibbs_block>(std::move(c)));
    root->set_keep_history(true);
    return chk::composite_model(root, 505);
}

chk::model order_mcmc_case() {
    const std::size_t n = 4, N = 100;
    std::mt19937_64 g(29);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    arma::imat data(N, n, arma::fill::zeros);
    for (std::size_t i = 0; i < N; ++i) {
        data(i, 0) = U(g) < 0.5 ? 1 : 0;
        for (std::size_t j = 1; j < n; ++j)
            data(i, j) = U(g) < 0.85 ? data(i, j - 1) : 1 - data(i, j - 1);
    }
    ab::order_mcmc_block_config c;
    c.name = "order";
    c.data = data;
    c.cardinalities = arma::uvec(n, arma::fill::value(2));
    c.structure_prior = ab::dag_prior::UNIFORM;
    c.max_parents = 3;
    c.candidate_top_C = 5;
    c.family_cache_F = 200;
    c.init_rng_seed = 12345;
    auto root = std::make_shared<ab::composite_block>("root");
    auto blk_owner = std::make_unique<ab::order_mcmc_block>(std::move(c));
    const ab::order_mcmc_block* blk = blk_owner.get();
    root->add_child(std::move(blk_owner));
    root->set_keep_history(true);
    // The stored DAGs are not in get_history(): an extra per-draw buffer.
    return chk::composite_model(root, 606, [blk](chk::buffers& b) {
        b.mats["order: sampled_dag_history_matrix()"] = blk->sampled_dag_history_matrix();
    });
}

// bart_block with an internal sigma, the kernel's history recording on
// (keep_tree). Its kernel draws from the thread-local engine bart_rng, which
// is seeded before construction.
chk::model bart_case() {
    const std::size_t N = 60, p = 2;
    std::mt19937_64 g(31);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> nd(0.0, 0.3);
    arma::mat X(N, p);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        X(i, 0) = U(g); X(i, 1) = U(g);
        y[i] = (X(i, 0) > 0.5 ? 1.0 : -1.0) + X(i, 1) + nd(g);
    }
    bart_rng::set_seed(37);
    auto root = std::make_shared<ab::composite_block>("root");
    root->data().set("bart_target", y);
    root->data().declare_dependencies("f", {"bart_target"});
    ab::bart_block_config c;
    c.name = "f"; c.x_train = X; c.y_init = y;
    c.working_response_key = "bart_target"; c.sigma_key = "";
    c.ntrees = 10; c.numcut = 20;
    ab::bart_block* bb = nullptr;
    { auto b = std::make_unique<ab::bart_block>(std::move(c)); bb = b.get(); root->add_child(std::move(b)); }
    root->set_keep_history(true);
    root->set_keep_tree(true);
    const arma::mat X_new = X.rows(0, 9);
    return chk::composite_model(root, 41, [bb, X_new](chk::buffers& b) {
        const stdbart::BartHistory h = bb->get_full_history();
        b.strings["f: kernel tree_history"] = h.tree_history;
        b.strings["f: get_tree_history()"] = bb->get_tree_history();
        b.mats["f: kernel sigma_history"] = arma::mat(h.sigma_history);
        b.mats["f: kernel var_counts_history"] =
            arma::conv_to<arma::mat>::from(h.var_counts_history);
        b.mats["f: kernel var_probs_history"] = h.var_probs_history;
        b.mats["f: kernel fmean_history"] = arma::mat(h.fmean_history);
        b.mats["f: predict_history(X_new)"] = bb->predict_history(X_new);
    });
}

// ---- 5. wrappers with per-sweep members of their own ------------------------

chk::model ardlasso_case() {
    const std::size_t N = 80, p = 4;
    std::mt19937_64 g(43);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::mat X(N, p);
    arma::vec Y(N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < p; ++j) X(i, j) = nd(g);
        Y[i] = 1.0 + 2.0 * X(i, 0) - 1.5 * X(i, 2) + 0.5 * nd(g);
    }
    auto w = std::make_shared<ARDLasso>(X, Y, 47, true);
    ARDLasso* m = w.get();
    const arma::vec x_new = arma::vectorise(X.rows(0, 4));   // column-major
    chk::model out;
    out.step    = [m](int n) { m->step(n); };
    out.clear   = [m] { m->clear_history(); };
    out.history = [m](chk::buffers& b) {
        for (auto& kv : m->get_history()) b.mats[kv.first] = kv.second;
    };
    out.extra   = [m, x_new](chk::buffers& b) {
        AI4BayesCode::state_map nd;
        nd["X"] = x_new;
        const AI4BayesCode::history_map pr = m->predict_at(nd);
        b.mats["predict_at(X_new): y_hat"] = pr.at("y_hat");
        b.rows_only["predict_at(X_new): y_rep"] = pr.at("y_rep");
    };
    out.owners.push_back(w);
    return out;
}

// VCBart's kernels draw from the thread-local engine bart_rng, which the
// constructor seeds with rng_seed.
chk::model vcbart_case() {
    const std::size_t N = 60;
    std::mt19937_64 g(53);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> nd(0.0, 0.3);
    arma::mat X(N, 1), Z(N, 1);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        X(i, 0) = nd(g) / 0.3;
        Z(i, 0) = U(g);
        y[i] = Z(i, 0) + (Z(i, 0) > 0.5 ? 1.0 : -1.0) * X(i, 0) + nd(g);
    }
    auto w = std::make_shared<VCBart>(X, Z, y, 10, 2.0, 2.0, 0.95, 3.0, 20, 61, true);
    VCBart* m = w.get();
    chk::model out;
    out.step    = [m](int n) { m->step(n); };
    out.clear   = [m] { m->clear_history(); };
    out.history = [m](chk::buffers& b) {
        for (auto& kv : m->get_history()) b.mats[kv.first] = kv.second;
    };
    out.owners.push_back(w);
    return out;
}

}  // namespace

int main() {
    std::printf("=== test_fix_clear_history ===\n\n");
    for (int t = 0; t < 4; ++t) vi_contract(t);
    for (int t = 0; t < 4; ++t) vi_frozen_after_clear(t);
    chk::run("DP Gaussian mixture (categorical, normal_gamma_cluster, "
             "stick_breaking, stick_label_swap)", dp_mixture_case, K1, K2);
    chk::run("gmrf_gaussian_joint_block", gmrf_joint_case, K1, K2);
    chk::run("lda_collapsed_gibbs_block", lda_case, K1, K2);
    chk::run("niw_cluster_gibbs_block", niw_case, K1, K2);
    chk::run("order_mcmc_block (partition)", order_mcmc_case, K1, K2);
    chk::run("bart_block with keep_tree", bart_case, K1, K2);
    chk::run("ARDLasso", ardlasso_case, K1, K2);
    chk::run("VCBart", vcbart_case, K1, K2);
    return chk::finish("test_fix_clear_history");
}

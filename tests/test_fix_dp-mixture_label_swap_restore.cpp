// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_dp-mixture_label_swap_restore.cpp
//
//  composite_block::set_current writes every child's named outputs into the
//  shared data. stick_label_swap_block (last child, as in the DP examples)
//  must not write the z, atoms, pi and stick_V of its last step over the
//  values the owning blocks have just restored. DP Gaussian mixture as in
//  examples/DPGaussianMixture.cpp: categorical z, Normal-Gamma atoms,
//  stick-breaking pi, label swap; alpha fixed.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/bnp_utils.hpp"
#include "AI4BayesCode/categorical_gibbs_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/normal_gamma_cluster_gibbs_block.hpp"
#include "AI4BayesCode/stick_breaking_block.hpp"
#include "AI4BayesCode/stick_label_swap_block.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

using namespace AI4BayesCode;

namespace {

constexpr std::size_t N = 60, K = 8;
int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

std::unique_ptr<composite_block> build() {
    auto comp = std::make_unique<composite_block>("dpmm");
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
    D.register_refresher("cluster_counts", [](const shared_data_t& s) -> arma::vec {
        return bnp::counts_from_z(s.get("z"), K);
    });
    D.declare_dependencies("z", {"y", "pi", "mu", "lambda"});
    D.declare_invalidates("z", {"cluster_counts"});
    D.declare_dependencies("cluster_params", {"z", "y"});
    D.declare_dependencies("pi", {"cluster_counts", "alpha"});
    D.declare_dependencies("label_swap", {"z", "mu", "lambda", "pi", "stick_V", "alpha"});
    D.declare_invalidates("label_swap", {"cluster_counts"});
    D.refresh_all();
    {
        categorical_gibbs_block_config c;
        c.name = "z"; c.n_obs = N; c.n_categories = K; c.initial_labels = z0;
        c.log_probs_fn = [](const block_context& ctx) -> arma::mat {
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
        comp->add_child(std::make_unique<categorical_gibbs_block>(std::move(c)));
    }
    {
        normal_gamma_cluster_gibbs_block_config c;
        c.name = "cluster_params"; c.K_trunc = K; c.d = 1; c.N = N;
        c.z_key = "z"; c.y_key = "y"; c.mu_name = "mu"; c.lambda_name = "lambda";
        c.mu_0 = arma::vec{0.0}; c.kappa_0 = 0.1; c.a_lambda_0 = 2.0; c.b_lambda_0 = 1.0;
        c.initial_mu = mu0; c.initial_lambda = lam0;
        comp->add_child(std::make_unique<normal_gamma_cluster_gibbs_block>(std::move(c)));
    }
    {
        stick_breaking_block_config c;
        c.name = "pi"; c.K_trunc = K; c.counts_key = "cluster_counts"; c.v_name = "stick_V";
        c.a_fn = [](std::size_t k, const arma::vec& counts, const block_context&) {
            return 1.0 + counts[k];
        };
        c.b_fn = [](std::size_t k, const arma::vec& counts, const block_context& ctx) {
            double t = 0.0;
            for (std::size_t j = k + 1; j < counts.n_elem; ++j) t += counts[j];
            return ctx.at("alpha")[0] + t;
        };
        c.initial_pi = pi0;
        comp->add_child(std::make_unique<stick_breaking_block>(std::move(c)));
    }
    {
        stick_label_swap_block_config c;
        c.name = "label_swap"; c.K_trunc = K; c.atom_keys = {{"mu", 1}, {"lambda", 1}};
        comp->add_child(std::make_unique<stick_label_swap_block>(std::move(c)));
    }
    return comp;
}

}  // namespace

int main() {
    std::printf("=== test_fix_dp-mixture_label_swap_restore ===\n");
    auto comp = build();
    std::mt19937_64 rng(20261122u);
    for (int s = 0; s < 200; ++s) comp->step(rng);
    const arma::vec theta = comp->current();
    for (int s = 0; s < 30; ++s) comp->step(rng);
    const arma::vec z_before = comp->data().get("z");

    // theta concatenates the children's current(), z first.
    const arma::vec theta_z = theta.subvec(0, N - 1);
    check(!same_bits(z_before, theta_z),
          "the shared z before the restore differs from the checkpoint z");

    comp->set_current(theta);
    const auto& D = comp->data();
    const auto atoms = comp->child(1).current_named_outputs();
    const auto sticks = comp->child(2).current_named_outputs();
    check(same_bits(D.get("z"), theta_z) && same_bits(D.get("z"), comp->child(0).current()),
          "after set_current: shared z is the restored z");
    check(same_bits(D.get("mu"), atoms.at("mu")) &&
              same_bits(D.get("lambda"), atoms.at("lambda")),
          "after set_current: shared mu and lambda are the restored atoms");
    check(same_bits(D.get("pi"), comp->child(2).current()) &&
              same_bits(D.get("stick_V"), sticks.at("stick_V")),
          "after set_current: shared pi and stick_V are the restored weights");
    check(same_bits(D.get("cluster_counts"), bnp::counts_from_z(theta_z, K)),
          "after set_current: cluster_counts are the counts of the restored z");
    check(comp->child(3).current_named_outputs().size() == 1,
          "after set_current: the label swap writes only its acceptance counts");

    // After a new context, the copies from the last step are not written back
    // until the next step.
    auto& swap = comp->child(3);
    swap.set_context(comp->data().build_context_for("label_swap"));
    swap.step(rng);
    const bool full = swap.current_named_outputs().size() == 6;
    swap.set_context(comp->data().build_context_for("label_swap"));
    check(full && swap.current_named_outputs().size() == 1,
          "set_context: the last step's z, atoms, pi, stick_V are not written back");

    for (int s = 0; s < 10; ++s) comp->step(rng);
    check(comp->data().get("z").n_elem == N, "sampling continues after the restore");

    std::printf("\n=== SUMMARY: %d failed ===\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}

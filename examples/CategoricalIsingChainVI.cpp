// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  CategoricalIsingChainVI.cpp
//
//  Tier A demo for v1.2 Block 4 (mean_field_categorical_vi_block).
//
//  Model
//  -----
//  Discrete Potts / Ising chain on n nodes with K states per node and
//  optional per-node external field h_i (length n):
//
//      log p~(z_1,...,z_n) = beta * sum_{(i,i+1)} I[z_i = z_{i+1}]
//                          + sum_i h_i * I[z_i = 1]
//
//  Mean-field VI factorises q(z) = prod_i Categorical(z_i ; phi_i) and
//  optimises the variational marginals phi_i in (K-1)-simplex via
//  RAABBVI on the ELBO. For small n*K (state space prod_i K_i <=
//  exact_state_cap), gradients are EXACT; otherwise Monte Carlo with
//  configurable S.
//
//  This is a PURE-VI demo: no observation model.
//
//  JUSTIFICATION (Check #16): Discrete latents with strong local
//  dependence -- system_design.md S11.2(b). Per-site Gibbs mixes
//  catastrophically near critical coupling; categorical mean-field VI
//  gives a deterministic approximation that converges cleanly to a
//  (biased) joint with correct marginals in many regimes.
//
//  Reference: Bishop PRML S10.1, Jaakkola-Jordan 1999 (QMR-DT), Welandawe
//  et al. 2022 (RAABBVI).
//
//  ----------------------------------------------------------------------------
//  STANDALONE DEMO (this file)
//  ----------------------------------------------------------------------------
//  This file was originally an Rcpp/pybind binding wrapper. It is now a
//  frontend-independent standalone C++ program: the Rcpp module class has
//  been removed and an int main() drives the composite block directly.
//
//  Validation strategy (a prior-only VI sampler has NO likelihood, so we
//  check the variational marginals against the KNOWN exact joint marginals
//  obtained by brute-force enumeration of the small chain):
//
//    (1) Symmetric chain (h = 0): by the state-permutation symmetry of the
//        Potts energy the exact node marginal is uniform (1/K per state).
//        VI must recover phi_i ~ uniform.
//
//    (2) Asymmetric chain (strong positive field h on state 1): we
//        enumerate the true joint p~(z) over all K^n states, normalise,
//        and marginalise to get the exact node marginals. VI marginals
//        must (a) be close to exact in total-variation distance, and
//        (b) be a large improvement over the uniform baseline, i.e. VI
//        actually learned the field-induced concentration on state 1.
// ============================================================================

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/mean_field_categorical_vi_block.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::mean_field_categorical_vi_block;
using AI4BayesCode::mean_field_categorical_vi_block_config;

namespace {

// ----------------------------------------------------------------------------
// Build a CategoricalIsingChainVI composite (replicates the former Rcpp
// constructor's wiring exactly: same priors, log-density, block config,
// hyperparameters -- only the frontend binding is gone).
// ----------------------------------------------------------------------------
std::unique_ptr<composite_block>
build_chain(std::size_t n, std::size_t K, double beta,
            const std::vector<double>& h, bool exact_enumeration,
            std::uint64_t rng_seed) {
    auto impl = std::make_unique<composite_block>("CategoricalIsingChainVI");

    // Uniform cardinalities K across all nodes.
    arma::uvec cards(n);
    for (std::size_t i = 0; i < n; ++i) cards[i] = K;

    // log-density closure: captures beta, h by value.
    const double beta_cap = beta;
    const std::vector<double> h_cap = h;
    auto lp_fn = [beta_cap, h_cap](const arma::uvec& z,
                                   const block_context&) -> double {
        double v = 0.0;
        for (std::size_t i = 0; i + 1 < z.n_elem; ++i) {
            if (z[i] == z[i + 1]) v += beta_cap;
        }
        for (std::size_t i = 0; i < z.n_elem; ++i) {
            if (z[i] == 1u) v += h_cap[i];
        }
        return v;
    };

    // ---- VI block config (identical to original) -----------------------
    mean_field_categorical_vi_block_config cfg;
    cfg.name              = "z";
    cfg.cardinalities     = cards;
    cfg.log_density       = lp_fn;
    cfg.exact_enumeration = exact_enumeration;
    cfg.exact_state_cap   = 4096;
    cfg.n_mc_samples      = 32;
    cfg.optimizer.gamma_0              = 0.1;
    cfg.optimizer.rho                  = 0.5;
    cfg.optimizer.tau                  = 0.01;
    cfg.optimizer.inner_iter_per_epoch = 200;
    cfg.optimizer.max_epochs           = 25;
    cfg.optimizer.S_khat               = 1000;
    cfg.init_random_eps                = 0.1;
    cfg.init_rng_seed                  = rng_seed;

    // ---- shared_data setup ---------------------------------------------
    impl->data().set("beta", arma::vec{beta});
    arma::vec h_arma(n);
    for (std::size_t i = 0; i < n; ++i) h_arma[i] = h[i];
    impl->data().set("h", h_arma);

    // Initial z sample (uniform random).
    std::mt19937_64 init_rng(rng_seed == 0
                                 ? std::random_device{}()
                                 : rng_seed);
    arma::vec z_init(n);
    std::uniform_int_distribution<std::size_t> Ud(0, K - 1);
    for (std::size_t i = 0; i < n; ++i) {
        z_init[i] = static_cast<double>(Ud(init_rng));
    }
    impl->data().set("z", z_init);

    // Gibbs DAG: z reads beta and h.
    impl->data().declare_dependencies("z", {"beta", "h"});

    // Generative-DAG context edges (VIZ-ONLY).
    impl->data().declare_context_edges("beta", {"z"});
    impl->data().declare_context_edges("h",    {"z"});

    impl->add_child(std::make_unique<mean_field_categorical_vi_block>(
        std::move(cfg)));

    return impl;
}

// Extract VI marginals phi as an n x K matrix from a converged composite.
arma::mat get_phi(const composite_block& comp, std::size_t n, std::size_t K) {
    const auto* vi = dynamic_cast<const mean_field_categorical_vi_block*>(
        &comp.child(0));
    if (!vi) throw std::runtime_error("child 0 is not the VI block");
    const arma::vec phi_flat = vi->current();
    arma::mat phi_mat(n, K);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < K; ++k)
            phi_mat(i, k) = phi_flat[i * K + k];
    return phi_mat;
}

// Brute-force EXACT node marginals of the true joint p~(z) by enumerating
// all K^n states. Returns an n x K matrix of marginal probabilities.
arma::mat exact_marginals(std::size_t n, std::size_t K, double beta,
                          const std::vector<double>& h) {
    std::size_t total = 1;
    for (std::size_t i = 0; i < n; ++i) total *= K;

    auto log_density = [&](const std::vector<std::size_t>& z) -> double {
        double v = 0.0;
        for (std::size_t i = 0; i + 1 < n; ++i)
            if (z[i] == z[i + 1]) v += beta;
        for (std::size_t i = 0; i < n; ++i)
            if (z[i] == 1u) v += h[i];
        return v;
    };

    // First pass: max log-density for numerical stability.
    std::vector<std::size_t> z(n, 0);
    double max_lp = -std::numeric_limits<double>::infinity();
    for (std::size_t s = 0; s < total; ++s) {
        std::size_t r = s;
        for (std::size_t i = 0; i < n; ++i) { z[i] = r % K; r /= K; }
        const double lp = log_density(z);
        if (lp > max_lp) max_lp = lp;
    }

    arma::mat marg(n, K, arma::fill::zeros);
    double Z = 0.0;
    for (std::size_t s = 0; s < total; ++s) {
        std::size_t r = s;
        for (std::size_t i = 0; i < n; ++i) { z[i] = r % K; r /= K; }
        const double w = std::exp(log_density(z) - max_lp);
        Z += w;
        for (std::size_t i = 0; i < n; ++i) marg(i, z[i]) += w;
    }
    marg /= Z;
    return marg;
}

// Mean total-variation distance between two n x K marginal matrices.
double mean_tv(const arma::mat& a, const arma::mat& b) {
    const std::size_t n = a.n_rows, K = a.n_cols;
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double tv = 0.0;
        for (std::size_t k = 0; k < K; ++k)
            tv += std::abs(a(i, k) - b(i, k));
        acc += 0.5 * tv;
    }
    return acc / static_cast<double>(n);
}

} // namespace

int main() {
    std::printf("=== CategoricalIsingChainVI standalone demo ===\n");
    std::printf("Mean-field VI on a discrete Potts/Ising chain.\n");
    std::printf("(prior-only sampler: validate VI marginals vs EXACT "
                "enumerated marginals)\n\n");

    const std::size_t n = 4;     // chain length
    const std::size_t K = 3;     // states per node (K^n = 81 states: exact)
    const double      beta = 0.8;
    const std::uint64_t seed = 20260621ULL;

    bool ok = true;

    // ---- Test 1: symmetric chain (h = 0) -> exact marginal is uniform ----
    {
        std::vector<double> h(n, 0.0);
        auto comp = build_chain(n, K, beta, h, /*exact=*/true, seed);
        std::mt19937_64 rng(seed);
        // Run enough steps for RAABBVI to converge (inner_iter_per_epoch=200,
        // up to max_epochs=25). step() is a no-op once converged.
        for (int s = 0; s < 6000; ++s) comp->step(rng);

        const arma::mat phi   = get_phi(*comp, n, K);
        const arma::mat exact = exact_marginals(n, K, beta, h);

        const double tv = mean_tv(phi, exact);
        // Exact marginal here is uniform 1/K; max deviation from uniform:
        double max_dev = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < K; ++k)
                max_dev = std::max(max_dev,
                                   std::abs(phi(i, k) - 1.0 / K));

        const bool t1_ok = (tv < 0.02) && (max_dev < 0.03);
        ok = ok && t1_ok;
        std::printf("[Test 1] symmetric chain  beta=%.2f  h=0\n", beta);
        std::printf("  VI phi[node0]   = [%.4f %.4f %.4f]\n",
                    phi(0, 0), phi(0, 1), phi(0, 2));
        std::printf("  exact (uniform) = [%.4f %.4f %.4f]\n",
                    exact(0, 0), exact(0, 1), exact(0, 2));
        std::printf("  mean TV(VI,exact)=%.5f   max|phi-1/K|=%.5f   -> %s\n\n",
                    tv, max_dev, t1_ok ? "OK" : "FAIL");
    }

    // ---- Test 2: asymmetric chain (strong field on state 1) -------------
    {
        std::vector<double> h(n, 2.0);   // strong positive field on state 1
        auto comp = build_chain(n, K, beta, h, /*exact=*/true, seed);
        std::mt19937_64 rng(seed);
        for (int s = 0; s < 6000; ++s) comp->step(rng);

        const arma::mat phi   = get_phi(*comp, n, K);
        const arma::mat exact = exact_marginals(n, K, beta, h);

        const double tv_vi = mean_tv(phi, exact);

        // Uniform baseline marginals (what you'd report with NO inference).
        arma::mat uniform(n, K);
        uniform.fill(1.0 / K);
        const double tv_base = mean_tv(uniform, exact);

        // VI must (a) track exact closely and (b) beat the uniform baseline,
        // confirming it actually learned the field-induced concentration.
        const bool t2_ok = (tv_vi < 0.05) && (tv_vi < 0.5 * tv_base);
        ok = ok && t2_ok;

        std::printf("[Test 2] asymmetric chain  beta=%.2f  h=2.0 (state 1)\n",
                    beta);
        std::printf("  VI phi[node0]    = [%.4f %.4f %.4f]\n",
                    phi(0, 0), phi(0, 1), phi(0, 2));
        std::printf("  exact[node0]     = [%.4f %.4f %.4f]\n",
                    exact(0, 0), exact(0, 1), exact(0, 2));
        std::printf("  mean TV(VI,   exact)=%.5f\n", tv_vi);
        std::printf("  mean TV(unif, exact)=%.5f  (naive baseline)\n",
                    tv_base);
        std::printf("  VI improvement over baseline: %.1fx   -> %s\n\n",
                    tv_base / std::max(tv_vi, 1e-12),
                    t2_ok ? "OK" : "FAIL");
    }

    std::printf("=== %s ===\n", ok ? "[demo PASS]" : "[demo FAIL]");
    return ok ? 0 : 1;
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  StructuredPottsVI.cpp
//
//  Tier A demo for v1.2 Block 5 (structured_categorical_vi_block).
//
//  Model
//  -----
//  Discrete Potts on an arbitrary undirected graph with user-supplied
//  edges + per-edge coupling strengths + per-node K-dim external field:
//
//      log p~(z) = sum_e beta_e * I[z_u = z_v]
//                + sum_i h_i[k] * I[z_i = k]
//
//  Variational family q(z) = prod_C q_C(z_C) — user-specified clique
//  partition. Each q_C is a JOINT Categorical over the clique's joint
//  state space (size prod_{i in C} K_i). Optimised via RAABBVI on
//  the analytical clique-sum-over-state gradient (Saul-Jordan 1996
//  §2.2 style).
//
//  This is a PURE-VI demo. predict_at(list(n_draws=N)) returns N fresh
//  q-samples (matrix of integer indices, n_draws × n).
//
//  Validation
//  ----------
//    tests/test_structured_categorical_vi_block.cpp — basic unit tests
//      including singleton-clique → Block 4 and grand-clique → exact
//      degeneracies.
//    tests/test_structured_categorical_vi_chain.cpp — head-to-head with
//      Block 4 fully factorised MF: structured ELBO tighter to log Z,
//      pairwise KL ~5 orders of magnitude smaller within clique.
//    tests/test_structured_categorical_vi_cavi_cross.cpp — 4-path
//      cross-check (VI / clique-CAVI / Gibbs / exact) + R-hat
//      convergence diagnostics.
//
//  JUSTIFICATION (Check #16): discrete latents with strong local
//  dependence (system_design.md §11.2(b)). Structured MF refines Block
//  4 by preserving intra-clique correlation exactly while factorising
//  ACROSS cliques (Saul-Jordan 1996). Gives demonstrably tighter
//  approximations than Block 4 when the user can identify
//  strong-coupling clusters; degenerates to Block 4 with singleton
//  cliques, to exact inference with a single all-encompassing clique.
//
//  Reference: Saul-Jordan 1996 (NIPS); Bishop PRML §10.1.
// ============================================================================

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that builds a small Potts MRF
// with a KNOWN ground truth (computed by exact enumeration), fits the
// structured clique-level mean-field VI block, and checks that the recovered
// per-node marginals match the exact marginals (and beat a naive uniform
// baseline). No R / Python binding is built or required.

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
#include "AI4BayesCode/structured_categorical_vi_block.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::structured_categorical_vi_block;
using AI4BayesCode::structured_categorical_vi_block_config;

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  We build a small discrete Potts MRF whose unnormalised log-density is
//
//      log p~(z) = sum_e beta_e * I[z_u = z_v] + sum_i h_i[z_i],
//
//  with K = 2 states per node and 6 nodes arranged as TWO triangles:
//
//        0---1        3---4
//         \ /          \ /
//          2            5
//
//  Strong intra-triangle coupling (beta = 1.5) + a single weak bridge edge
//  (2--3, beta = 0.2) + a small external field that breaks the label
//  symmetry. The clique partition is {0,1,2} and {3,4,5} — exactly the two
//  triangles — so the structured mean field captures all the strong coupling
//  exactly and only mean-fields the weak bridge. This is the Saul-Jordan
//  regime where structured VI is expected to be very accurate.
//
//  GROUND TRUTH is computed in main() by full enumeration of all 2^6 = 64
//  joint states (same log-density), giving exact per-node marginals
//  q*_i(z_i = 1). We then fit the VI block (exact-enumeration gradient mode,
//  since 64 <= exact_state_cap) and check that the recovered marginals match
//  the exact ones, and beat a naive uniform-0.5 baseline.
//==============================================================================
#include <cstdio>

namespace {

// Edge list (0-based) and per-edge coupling strengths.
struct Edge { std::size_t u, v; double beta; };

const std::vector<Edge> kEdges = {
    {0, 1, 1.5}, {1, 2, 1.5}, {0, 2, 1.5},   // triangle A
    {3, 4, 1.5}, {4, 5, 1.5}, {3, 5, 1.5},   // triangle B
    {2, 3, 0.2},                             // weak bridge
};

constexpr std::size_t kN = 6;
constexpr std::size_t kK = 2;

// Per-node external field h_i[k]; a mild field that breaks the global
// label-flip symmetry so the marginals are non-trivially != 0.5.
const double kH[kN][kK] = {
    { 0.0,  0.6}, { 0.0,  0.4}, { 0.0,  0.2},
    { 0.0, -0.3}, { 0.0, -0.5}, { 0.0, -0.7},
};

// Unnormalised log-density log p~(z), shared by the exact-enumeration ground
// truth and the VI block's log_density callback.
double potts_log_density(const arma::uvec& z) {
    double v = 0.0;
    for (const auto& e : kEdges)
        if (z[e.u] == z[e.v]) v += e.beta;
    for (std::size_t i = 0; i < kN; ++i)
        v += kH[i][z[i]];
    return v;
}

}  // namespace

int main() {
    // ---- 1. EXACT per-node marginals by full enumeration of 2^6 states ----
    arma::vec exact_marg1(kN, arma::fill::zeros);   // q*_i(z_i = 1)
    double    Z = 0.0;
    arma::uvec z(kN);
    const std::size_t total = static_cast<std::size_t>(1) << kN;  // 2^6
    for (std::size_t s = 0; s < total; ++s) {
        for (std::size_t i = 0; i < kN; ++i)
            z[i] = (s >> i) & 1u;
        const double w = std::exp(potts_log_density(z));
        Z += w;
        for (std::size_t i = 0; i < kN; ++i)
            if (z[i] == 1u) exact_marg1[i] += w;
    }
    exact_marg1 /= Z;

    // ---- 2. Build the composite + VI block DIRECTLY (no frontend wrapper) --
    auto impl = std::make_unique<composite_block>("StructuredPottsVI");

    std::mt19937_64 rng(7);

    // VI block config — same wiring the old Rcpp constructor performed.
    arma::uvec cards(kN);
    cards.fill(kK);

    auto lp_fn = [](const arma::uvec& zz, const block_context&) -> double {
        return potts_log_density(zz);
    };

    structured_categorical_vi_block_config cfg;
    cfg.name              = "z";
    cfg.cardinalities     = cards;
    cfg.clique_partition  = { {0, 1, 2}, {3, 4, 5} };   // the two triangles
    cfg.log_density       = lp_fn;
    cfg.exact_enumeration = true;     // 2^6 = 64 <= exact_state_cap
    cfg.exact_state_cap   = 4096;
    cfg.n_mc_samples      = 32;
    cfg.optimizer.gamma_0              = 0.1;
    cfg.optimizer.rho                  = 0.5;
    cfg.optimizer.tau                  = 0.005;
    cfg.optimizer.inner_iter_per_epoch = 200;
    cfg.optimizer.max_epochs           = 25;
    cfg.optimizer.S_khat               = 1000;
    cfg.init_random_eps                = 0.1;
    cfg.init_rng_seed                  = 7;

    // shared_data seeding (mirrors the original constructor).
    arma::mat h_mat(kN, kK);
    for (std::size_t i = 0; i < kN; ++i)
        for (std::size_t k = 0; k < kK; ++k)
            h_mat(i, k) = kH[i][k];
    impl->data().set("h", arma::vectorise(h_mat));

    arma::vec z_init(kN);
    std::uniform_int_distribution<std::size_t> Ud(0, kK - 1);
    for (std::size_t i = 0; i < kN; ++i)
        z_init[i] = static_cast<double>(Ud(rng));
    impl->data().set("z", z_init);

    impl->data().declare_dependencies("z", {"h"});
    impl->data().declare_context_edges("h", {"z"});

    impl->add_child(std::make_unique<structured_categorical_vi_block>(
        std::move(cfg)));

    // ---- 3. Run VI to convergence -----------------------------------------
    const auto* vi = dynamic_cast<const structured_categorical_vi_block*>(
        &impl->child(0));
    if (!vi) {
        std::printf("[demo FAIL] child 0 is not a structured VI block\n");
        return 1;
    }

    int steps = 0;
    const int kMaxSteps = 20000;
    while (!vi->converged() && steps < kMaxSteps) {
        impl->step(rng);
        ++steps;
    }

    // ---- 4. Recover per-node VI marginals q_i(z_i = 1) --------------------
    const arma::mat marg = vi->per_node_marginals();   // kN x K_max
    arma::vec vi_marg1(kN);
    for (std::size_t i = 0; i < kN; ++i)
        vi_marg1[i] = marg(i, 1);

    // ---- 5. Compare to exact truth + naive uniform baseline ---------------
    double max_abs_err_vi = 0.0;
    double max_abs_err_naive = 0.0;
    for (std::size_t i = 0; i < kN; ++i) {
        max_abs_err_vi    = std::max(max_abs_err_vi,
                                     std::abs(vi_marg1[i] - exact_marg1[i]));
        max_abs_err_naive = std::max(max_abs_err_naive,
                                     std::abs(0.5 - exact_marg1[i]));
    }

    std::printf("StructuredPottsVI demo  (2 triangles, K=2, exact-grad VI)\n");
    std::printf("  converged=%s after %d steps, epoch=%zu, ELBO=%.4f, "
                "exact_active=%s\n",
                vi->converged() ? "yes" : "no", steps,
                vi->epoch(), vi->current_elbo(),
                vi->exact_active() ? "yes" : "no");
    std::printf("  node   exact q(z=1)   VI q(z=1)   |err|\n");
    for (std::size_t i = 0; i < kN; ++i)
        std::printf("   %2zu     %8.4f     %8.4f   %7.4f\n",
                    i, exact_marg1[i], vi_marg1[i],
                    std::abs(vi_marg1[i] - exact_marg1[i]));
    std::printf("  max |err|: VI=%.4f   naive-uniform=%.4f\n",
                max_abs_err_vi, max_abs_err_naive);

    // VI on the true cliques should track the exact marginals tightly (the
    // only approximation is the single weak bridge edge), and must clearly
    // beat the naive 0.5 baseline.
    const bool ok = vi->converged()
                 && max_abs_err_vi < 0.03
                 && max_abs_err_vi < 0.5 * max_abs_err_naive;
    std::printf("%s\n", ok
        ? "[demo PASS] structured VI recovers Potts marginals (beats naive)"
        : "[demo FAIL] VI marginals did not match exact within tolerance");
    return ok ? 0 : 1;
}


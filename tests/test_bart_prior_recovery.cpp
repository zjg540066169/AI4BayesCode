// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_bart_prior_recovery.cpp
//
//  The BART tree moves are birth and death only, so with a flat likelihood
//  the tree chain must have the tree PRIOR as its stationary distribution.
//  Setup: one tree, one covariate, n = 3000 observations on a 100-point cut
//  grid (each grid interval holds about 30 observations, so the kernel's
//  minimum-leaf-size rule never binds), sigma fixed at 1e6 so the
//  likelihood ratio of every move is 1 to within about 1e-8. The reference
//  is the exact BART prior over tree size -- split probability
//  base / (1 + depth)^power at a node with an available cut, the cut drawn
//  uniformly among the available ones -- simulated directly.
//
//  Pins the birth proposal's reverse-death probability: when the root is the
//  tree's only nog (two leaves), a birth at one of its children leaves the
//  number of nogs unchanged, so P(choose that nog for death) is 1, not 1/2.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/bart_block.hpp"
#include <cmath>
#include <random>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include "portable_rng.hpp"

static constexpr int    kMaxLeaves = 12;
static constexpr double kBase = 0.95, kPower = 2.0;
static constexpr int    kNumcut = 100;

// Leaves of a prior-simulated subtree whose range holds `ncut` available cuts.
static int prior_leaves(int depth, int ncut, std::mt19937_64& g) {
    const double pg = (ncut > 0) ? kBase / std::pow(1.0 + depth, kPower) : 0.0;
    if (prng::u01(g) >= pg) return 1;
    const int c = static_cast<int>(prng::u01(g) * ncut);   // 0..ncut-1
    return prior_leaves(depth + 1, c, g) + prior_leaves(depth + 1, ncut - c - 1, g);
}

static int leaves_of(const std::string& forest) {
    std::istringstream is(forest);
    long ntrees = 0, nnodes = 0;
    is >> ntrees >> nnodes;
    return static_cast<int>((nnodes + 1) / 2);
}

int main() {
    // Reference: the prior, 2e6 direct draws.
    std::vector<double> ref(kMaxLeaves + 2, 0.0);
    {
        std::mt19937_64 g(20261018u);
        const long R = 2000000;
        for (long r = 0; r < R; ++r) {
            int L = prior_leaves(0, kNumcut, g);
            ref[L > kMaxLeaves ? kMaxLeaves + 1 : L] += 1.0 / R;
        }
    }
    // Chain: flat likelihood.
    std::vector<double> got(kMaxLeaves + 2, 0.0);
    {
        bart_rng::set_seed(20261018u);
        std::mt19937_64 g(7u);
        const int n = 3000;
        arma::mat X(n, 1);
        arma::vec y(n);
        for (int i = 0; i < n; ++i) { X(i, 0) = prng::u01(g); y[i] = prng::normal(g, 0.0, 1.0); }
        stdbart::bart_model m(X, y, kNumcut, /*usequants=*/false, /*cont=*/false, /*rm_const=*/false,
                              /*ntrees=*/1, std::numeric_limits<double>::quiet_NaN(),
                              /*k=*/2.0, kPower, kBase, /*nu=*/3.0);
        m.set_history(false);
        const long burn = 5000, S = 400000;
        for (long s = 0; s < burn; ++s) m.update_step(1e6);
        for (long s = 0; s < S; ++s) {
            m.update_step(1e6);
            int L = leaves_of(m.get_tree());
            got[L > kMaxLeaves ? kMaxLeaves + 1 : L] += 1.0 / S;
        }
    }
    double tv = 0.0;
    std::printf("  leaves   prior      chain\n");
    for (int L = 1; L <= kMaxLeaves + 1; ++L) {
        tv += 0.5 * std::fabs(ref[L] - got[L]);
        if (L <= 6) std::printf("  %-6d   %.4f     %.4f\n", L, ref[L], got[L]);
    }
    const bool ok = tv < 0.015;
    std::printf("  total variation distance = %.4f (tol 0.015)\n", tv);
    std::printf("%s BART birth-death chain recovers the tree prior under a flat likelihood\n",
                ok ? "[PASS]" : "[FAIL]");
    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", ok ? 1 : 0, ok ? 0 : 1);
    return ok ? 0 : 1;
}

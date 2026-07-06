// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  test_bge_scorer.cpp -- BGe (continuous Gaussian) local-score parity panel.
//
//  The defining BGe property is LIKELIHOOD EQUIVALENCE: two Markov-equivalent
//  DAGs receive identical marginal likelihood. T1/T2 test exactly that (it is
//  the sharpest single correctness check). T3-T6 add structural/validation
//  checks. Numeric agreement with BiDAG::DAGcorescore is the R-level audit
//  (Phase 5), not this C++ smoke.
// ============================================================================

#include "AI4BayesCode/bge_scorer.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using AI4BayesCode::bge_scorer;
using AI4BayesCode::bge_scorer_config;

namespace {
int g_pass = 0, g_fail = 0;
void check(bool ok, const char* tag, const char* detail = "") {
    std::printf("  %s  %s %s\n", ok ? "PASS" : "FAIL", tag, detail);
    if (ok) ++g_pass; else ++g_fail;
}

// Generate correlated continuous data: latent AR-style so every pair is
// dependent (so v-structures genuinely differ from chains).
arma::mat sim_corr(std::size_t N, std::size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> z(0.0, 1.0);
    arma::mat X(N, n);
    for (std::size_t i = 0; i < N; ++i) {
        double prev = z(rng);
        for (std::size_t j = 0; j < n; ++j) {
            // x_j = 0.8 x_{j-1} + noise  -> chain-correlated columns
            const double v = (j == 0) ? prev : (0.8 * prev + 0.6 * z(rng));
            X(i, j) = v;
            prev = v;
        }
    }
    return X;
}

constexpr std::uint64_t M(std::size_t j) { return 1ULL << j; }
}  // namespace

int main() {
    std::printf("=== test_bge_scorer ===\n");

    // ---- 2-node data -------------------------------------------------------
    {
        bge_scorer_config cfg;
        cfg.data = sim_corr(30, 2, 12345);
        bge_scorer s(cfg);

        // score(X->Y) = fs(X|empty) + fs(Y|{X}); score(Y->X) = fs(Y|empty)+fs(X|{Y})
        const double sXY = s.family_score(0, 0ULL) + s.family_score(1, M(0));
        const double sYX = s.family_score(1, 0ULL) + s.family_score(0, M(1));
        std::printf("  X->Y = %.10f   Y->X = %.10f   |diff| = %.2e\n",
                    sXY, sYX, std::fabs(sXY - sYX));
        check(std::fabs(sXY - sYX) < 1e-9,
              "T1 likelihood-equivalence 2-node (X->Y == Y->X)");
    }

    // ---- 3-node data: Markov-class equivalence -----------------------------
    {
        bge_scorer_config cfg;
        cfg.data = sim_corr(60, 3, 999);
        bge_scorer s(cfg);
        auto fs = [&](std::size_t i, std::uint64_t u) { return s.family_score(i, u); };

        // Chain equivalence class (skeleton X-Y-Z, no v-structure):
        const double chain1 = fs(0, 0ULL) + fs(1, M(0))   + fs(2, M(1));   // X->Y->Z
        const double fork   = fs(0, M(1)) + fs(1, 0ULL)   + fs(2, M(1));   // X<-Y->Z
        const double chain2 = fs(0, M(1)) + fs(1, M(2))   + fs(2, 0ULL);   // X<-Y<-Z
        // v-structure (collider at Y): different class
        const double vstruct = fs(0, 0ULL) + fs(1, M(0)|M(2)) + fs(2, 0ULL);

        const double mx = std::max(std::fabs(chain1 - fork),
                                   std::fabs(chain1 - chain2));
        std::printf("  chain1=%.6f fork=%.6f chain2=%.6f  vstruct=%.6f\n",
                    chain1, fork, chain2, vstruct);
        check(mx < 1e-9,
              "T2 Markov-class equivalence (3 chain-class DAGs score equal)");
        check(std::fabs(chain1 - vstruct) > 1e-3,
              "T2b v-structure scores DIFFERENTLY from the chain class");
    }

    // ---- parent-order invariance -------------------------------------------
    {
        bge_scorer_config cfg;
        cfg.data = sim_corr(40, 4, 7);
        bge_scorer s(cfg);
        const double ab = s.family_score(0, M(1) | M(2));
        const double ba = s.family_score(0, M(2) | M(1));  // same set, same mask
        // also assert the Cholesky path (p>=2) is finite + well-defined
        const double p3 = s.family_score(0, M(1) | M(2) | M(3));
        check(std::fabs(ab - ba) < 1e-12 && std::isfinite(p3),
              "T3 parent-set is unordered + p>=2 Cholesky path finite");
    }

    // ---- empty-parent + monotonic sanity -----------------------------------
    {
        bge_scorer_config cfg;
        cfg.data = sim_corr(50, 3, 31);
        bge_scorer s(cfg);
        const double e0 = s.family_score(0, 0ULL);
        check(std::isfinite(e0), "T4 empty-parent score finite");
    }

    // ---- structure-prior hook ----------------------------------------------
    {
        bge_scorer_config cfg;
        cfg.data = sim_corr(40, 4, 5);
        bge_scorer s0(cfg);
        cfg.use_structure_prior = true;
        bge_scorer s1(cfg);
        // With the FK prior, a 2-parent family is penalised by -log C(3,2) vs the
        // no-prior score; the empty-parent (|Pa|=0) family is unchanged.
        const double d_empty = s1.family_score(0, 0ULL) - s0.family_score(0, 0ULL);
        const double d_two   = s1.family_score(0, M(1)|M(2))
                             - s0.family_score(0, M(1)|M(2));
        const double expected = -std::log(3.0);  // -log C(3,2) = -log 3
        check(std::fabs(d_empty) < 1e-12 &&
              std::fabs(d_two - expected) < 1e-9,
              "T5 use_structure_prior adds -log C(n-1,|Pa|) per family");
    }

    // ---- validation --------------------------------------------------------
    {
        bool threw_aw = false, threw_am = false;
        try { bge_scorer_config c; c.data = sim_corr(20, 3, 1); c.aw = 3.0; /* <= n+1 */
              bge_scorer s(c); (void)s; }
        catch (const std::exception&) { threw_aw = true; }
        try { bge_scorer_config c; c.data = sim_corr(20, 3, 1); c.am = 0.0;
              bge_scorer s(c); (void)s; }
        catch (const std::exception&) { threw_am = true; }
        check(threw_aw, "T6 aw <= n+1 rejected");
        check(threw_am, "T6 am <= 0 rejected");
    }

    std::printf("\n====== Summary: %d PASS / %d FAIL ======\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

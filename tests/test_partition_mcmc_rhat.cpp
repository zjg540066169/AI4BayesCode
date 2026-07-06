// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  test_partition_mcmc_rhat.cpp -- two-chain Gelman-Rubin R-hat convergence gate
//  for the partition sampler on a realistic n=6 network (3.7M DAGs -- far too
//  many to enumerate, so R-hat is the right convergence tool here; exact
//  enumeration covers unbiasedness at n<=4 elsewhere).
//
//  Two OVER-DISPERSED chains (one from the empty-DAG partition, one from a
//  full lower-triangular DAG's partition). R-hat computed on every edge-
//  inclusion marginal AND the partition log-score. Gate: max R-hat < 1.01.
// ============================================================================

#include "AI4BayesCode/bde_scorer.hpp"
#include "AI4BayesCode/score_cache.hpp"
#include "AI4BayesCode/partition_state.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace AI4BayesCode;

namespace {
int g_pass = 0, g_fail = 0;
void check(bool ok, const char* tag, const char* detail = "") {
    std::printf("  %s  %s %s\n", ok ? "PASS" : "FAIL", tag, detail);
    if (ok) ++g_pass; else ++g_fail;
}

// Two-chain Gelman-Rubin R-hat from per-chain (mean, sample-variance, N).
double rhat2(double m1, double v1, double m2, double v2, double N) {
    const double W = 0.5 * (v1 + v2);
    if (!(W > 1e-300)) return 1.0;                 // constant in both chains
    const double gm = 0.5 * (m1 + m2);
    const double B  = N * ((m1 - gm) * (m1 - gm) + (m2 - gm) * (m2 - gm)); // M-1=1
    const double varhat = (N - 1.0) / N * W + B / N;
    return std::sqrt(varhat / W);
}

arma::imat sim_discrete(std::size_t N, std::size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    arma::imat X(N, n);
    for (std::size_t r = 0; r < N; ++r) {
        int prev = U(rng) < 0.5 ? 0 : 1;
        for (std::size_t j = 0; j < n; ++j) {
            double p = (j == 0) ? 0.5 : (prev == 1 ? 0.8 : 0.2);
            int v = U(rng) < p ? 1 : 0; X(r, j) = v; prev = v;
        }
    }
    return X;
}
}  // namespace

int main() {
    std::printf("=== test_partition_mcmc_rhat (two-chain Gelman-Rubin) ===\n");
    const std::size_t n = 6;

    bde_scorer_config bcfg;
    bcfg.data = sim_discrete(200, n, 71u);
    bcfg.cardinalities = arma::uvec(n); bcfg.cardinalities.fill(2u);
    bcfg.alpha = 1.0; bcfg.use_structure_prior = false;
    score_cache_config sc;   // realistic pruning defaults
    sc.max_parents = 4; sc.candidate_top_C = n - 1; sc.family_top_F = 4000;
    sc.gamma_prune_nats = 20.0;
    score_cache cache(bde_scorer(bcfg), sc);

    // over-dispersed inits
    partition_state init_a = trivial_partition(n);              // empty DAG
    std::vector<std::uint64_t> full(n, 0ULL);                   // full lower-tri
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) full[i] |= (1ULL << j);
    partition_state init_b = dag_to_partition(full, n);

    const std::size_t burnin = 30000, N = 300000;
    // per-chain edge-inclusion counts + Welford(log_score)
    std::vector<std::vector<long>> edge_cnt(2, std::vector<long>(n * n, 0));
    double ls_mean[2] = {0, 0}, ls_m2[2] = {0, 0};

    for (int ch = 0; ch < 2; ++ch) {
        std::mt19937_64 rng(2024u + ch);
        partition_chain chain =
            partition_chain_init(cache, ch == 0 ? init_a : init_b);
        long cnt = 0;
        for (std::size_t step = 0; step < burnin + N; ++step) {
            partition_mcmc_step(chain, cache, rng);
            if (step >= burnin) {
                const auto dag = partition_sample_dag(cache, rng, chain.state);
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        if (dag[i] & (1ULL << j)) edge_cnt[ch][i * n + j]++;
                ++cnt;
                const double x = chain.log_score;   // Welford
                const double d = x - ls_mean[ch];
                ls_mean[ch] += d / static_cast<double>(cnt);
                ls_m2[ch]   += d * (x - ls_mean[ch]);
            }
        }
    }

    // edge-marginal R-hat (0/1 series: sample var = p(1-p))
    double max_edge_rhat = 1.0, worst_p1 = 0, worst_p2 = 0; int n_active = 0;
    for (std::size_t e = 0; e < n * n; ++e) {
        const double p1 = static_cast<double>(edge_cnt[0][e]) / N;
        const double p2 = static_cast<double>(edge_cnt[1][e]) / N;
        if (p1 < 1e-6 && p2 < 1e-6) continue;       // edge never present
        ++n_active;
        const double r = rhat2(p1, p1 * (1 - p1), p2, p2 * (1 - p2),
                               static_cast<double>(N));
        if (r > max_edge_rhat) { max_edge_rhat = r; worst_p1 = p1; worst_p2 = p2; }
    }
    std::printf("  worst edge: p1=%.4f p2=%.4f (|diff|=%.4f)\n",
                worst_p1, worst_p2, std::fabs(worst_p1 - worst_p2));
    // SKELETON (undirected) R-hat: edge {a,b} present iff a->b or b->a (mutually
    // exclusive in a DAG, so p_skel = p(a->b)+p(b->a)).
    double max_skel_rhat = 1.0;
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = a + 1; b < n; ++b) {
            const double p1 = (static_cast<double>(edge_cnt[0][b*n+a] + edge_cnt[0][a*n+b])) / N;
            const double p2 = (static_cast<double>(edge_cnt[1][b*n+a] + edge_cnt[1][a*n+b])) / N;
            if (p1 < 1e-6 && p2 < 1e-6) continue;
            max_skel_rhat = std::max(max_skel_rhat,
                rhat2(p1, p1*(1-p1), p2, p2*(1-p2), static_cast<double>(N)));
        }

    const double ls_rhat = rhat2(ls_mean[0], ls_m2[0] / (N - 1),
                                 ls_mean[1], ls_m2[1] / (N - 1),
                                 static_cast<double>(N));
    std::printf("  SKELETON (undirected) max R-hat=%.4f\n", max_skel_rhat);

    std::printf("  n=%zu, 2 chains x %zu draws (over-dispersed inits): "
                "%d active edges; max edge R-hat=%.4f ; log_score R-hat=%.4f\n",
                n, N, n_active, max_edge_rhat, ls_rhat);
    check(ls_rhat < 1.01,       "R-hat: partition log_score two-chain R-hat < 1.01");
    check(max_skel_rhat < 1.01, "R-hat: SKELETON (undirected) two-chain R-hat < 1.01");
    // With the Sec.5 edge-reversal move in the mixture, edge DIRECTIONS mix too
    // (both directions of a reversible edge are sampled), so the directed-edge
    // R-hat now converges as well -- gated, not just reported.
    check(max_edge_rhat < 1.01, "R-hat: DIRECTED edges two-chain R-hat < 1.01 (edge-reversal move)");

    std::printf("\n====== Summary: %d PASS / %d FAIL ======\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

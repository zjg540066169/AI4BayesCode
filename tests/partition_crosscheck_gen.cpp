// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  partition_crosscheck_gen.cpp -- generate a SYNTHETIC discrete Bayesian
//  network dataset + run this library's partition-MCMC sampler on it, writing
//  (1) the data and (2) the sampled DAGs to CSV for the R cross-check against
//  BiDAG::partitionMCMC (audit_PartitionMCMCBN_vs_BiDAG.R). No benchmark data.
//
//  Usage: partition_crosscheck_gen <data.csv> <ai_dags.csv>
//  DAGs are written in the "incidence" convention adj[a][b] = 1 iff a -> b
//  (row-major, byrow), matching BiDAG's traceadd$incidence so the R side scores
//  both chains identically.
// ============================================================================

#include "AI4BayesCode/order_mcmc_block.hpp"

#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

using namespace AI4BayesCode;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "need <data.csv> <ai_dags.csv>\n"); return 2; }
    const std::size_t n = 6, N = 200;
    const int K_MAX = 5;   // = n-1: uncapped, to match BiDAG's full startspace

    // ---- random sparse ground-truth DAG in topo order 0..n-1 ----
    std::mt19937_64 grng(20260706ULL);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<std::vector<std::size_t>> pa(n);
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (U(grng) < 0.30 && pa[i].size() < 2) pa[i].push_back(j);

    // ---- simulate binary data: node = XOR(parents) with 85% fidelity ----
    arma::imat data(N, n);
    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t i = 0; i < n; ++i) {
            int x = 0;
            for (std::size_t p : pa[i]) x ^= data(r, p);
            if (pa[i].empty()) x = (U(grng) < 0.5) ? 1 : 0;
            if (U(grng) > 0.85) x = 1 - x;   // observation noise
            data(r, i) = x;
        }
    { std::ofstream f(argv[1]);
      for (std::size_t r = 0; r < N; ++r) {
          for (std::size_t i = 0; i < n; ++i) { f << data(r, i); if (i + 1 < n) f << ","; }
          f << "\n"; } }

    // ---- run THIS library's partition MCMC (BDeu, uniform DAG prior) ----
    order_mcmc_block_config cfg;
    cfg.data = data;
    cfg.cardinalities = arma::uvec(n); cfg.cardinalities.fill(2u);
    cfg.bdeu_alpha = 1.0; cfg.max_parents = K_MAX;
    cfg.method = order_mcmc_block_config::method_t::partition;
    cfg.use_structure_prior = false;          // uniform DAG prior == BiDAG edgepf=1
    order_mcmc_block blk(cfg);

    std::mt19937_64 srng(12345ULL);
    for (int i = 0; i < 60000; ++i) {                 // burn-in
        blk.step(srng);
        if (i % 15000 == 0)
            std::fprintf(stderr, "  burn %d: partition log_score = %.2f\n",
                         i, blk.current_log_score());
    }

    const int KEEP = 4000;
    std::ofstream fd(argv[2]);
    for (int s = 0; s < KEEP; ++s) {
        blk.step(srng);
        const auto out = blk.current_named_outputs(srng);
        const arma::vec& dag = out.at("order_sampled_DAG");   // dag[i*n+j]=1 iff j -> i
        // emit incidence adj[a][b] = 1 iff a -> b  (a parent of b) = dag[b*n+a]
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = 0; b < n; ++b) {
                const int e = (dag[b * n + a] > 0.5) ? 1 : 0;
                fd << e;
                if (a * n + b + 1 < n * n) fd << ",";
            }
        fd << "\n";
    }
    std::printf("wrote %s (%zux%zu) + %s (%d AI partition-MCMC DAGs, n=%zu)\n",
                argv[1], N, n, argv[2], KEEP, n);
    return 0;
}

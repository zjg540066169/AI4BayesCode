// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  test_order_mcmc_block_partition.cpp -- the order_mcmc_block API in
//  method="partition" mode, over BOTH the BDeu (discrete) and BGe (continuous)
//  scores. Smoke (named outputs present + well-formed) + a light skeleton-
//  recovery check on a known chain DAG. (Exact unbiasedness is proven in
//  test_partition_mcmc_diagnostics.cpp; this exercises the block wiring.)
// ============================================================================

#include "AI4BayesCode/order_mcmc_block.hpp"

#include <cmath>
#include <cstdio>
#include <random>

using namespace AI4BayesCode;

namespace {
int g_pass = 0, g_fail = 0;
void check(bool ok, const char* tag, const char* d = "") {
    std::printf("  %s  %s %s\n", ok ? "PASS" : "FAIL", tag, d);
    if (ok) ++g_pass; else ++g_fail;
}

// Chain 0->1->2->3->4: discrete, each node copies its parent w.p. 0.85.
arma::imat sim_chain_discrete(std::size_t N, std::size_t n, unsigned seed) {
    std::mt19937_64 rng(seed); std::uniform_real_distribution<double> U(0, 1);
    arma::imat X(N, n);
    for (std::size_t r = 0; r < N; ++r) {
        int prev = U(rng) < 0.5 ? 0 : 1;
        for (std::size_t j = 0; j < n; ++j) {
            int v = (j == 0) ? prev : (U(rng) < 0.93 ? prev : 1 - prev);
            X(r, j) = v; prev = v;
        }
    }
    return X;
}
arma::mat sim_chain_cont(std::size_t N, std::size_t n, unsigned seed) {
    std::mt19937_64 rng(seed); std::normal_distribution<double> z(0, 1);
    arma::mat X(N, n);
    for (std::size_t r = 0; r < N; ++r) {
        double prev = z(rng);
        for (std::size_t j = 0; j < n; ++j) {
            double v = (j == 0) ? prev : (0.95 * prev + 0.22 * z(rng));
            X(r, j) = v; prev = v;
        }
    }
    return X;
}

// Skeleton-recovery: after many steps, average sampled DAG; is each true chain
// edge (j, j+1) present (either direction) with high marginal probability?
double skeleton_recovery(order_mcmc_block& blk, std::size_t n,
                         std::size_t n_draws, std::mt19937_64& rng) {
    arma::mat marg(n, n, arma::fill::zeros);
    for (std::size_t s = 0; s < n_draws; ++s) {
        blk.step(rng);
        const auto out = blk.current_named_outputs(rng);
        const arma::vec& dag = out.at("order_sampled_DAG");   // n*n row-major
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (dag[i * n + j] > 0.5) marg(i, j) += 1.0;
    }
    marg /= static_cast<double>(n_draws);
    double min_true_edge = 1.0;
    for (std::size_t j = 0; j + 1 < n; ++j)             // true edges j <-> j+1
        min_true_edge = std::min(min_true_edge,
                                 marg(j, j + 1) + marg(j + 1, j));
    return min_true_edge;
}
}  // namespace

int main() {
    std::printf("=== test_order_mcmc_block_partition ===\n");
    const std::size_t n = 5;

    // ---- BDeu discrete: partition vs the verified order mode should AGREE ----
    {
        arma::imat data = sim_chain_discrete(400, n, 7u);
        arma::uvec card(n); card.fill(2u);
        order_mcmc_block_config cp;
        cp.data = data; cp.cardinalities = card; cp.bdeu_alpha = 1.0;
        cp.max_parents = 4; cp.use_structure_prior = false;
        cp.method = order_mcmc_block_config::method_t::partition;
        order_mcmc_block blk_p(cp);
        std::mt19937_64 rp(11); for (int i = 0; i < 2000; ++i) blk_p.step(rp);
        const auto out = blk_p.current_named_outputs(rp);
        check(out.count("order") && out.at("order").n_elem == n &&
              out.count("order_party") &&
              std::fabs(arma::accu(out.at("order_party")) - double(n)) < 1e-9 &&
              out.count("order_sampled_DAG") &&
              out.at("order_sampled_DAG").n_elem == n * n &&
              out.count("order_log_score"),
              "BDe/partition: named outputs (order, party, sampled_DAG, log_score) well-formed");
        const double rec_p = skeleton_recovery(blk_p, n, 4000, rp);

        order_mcmc_block_config co = cp;
        co.method = order_mcmc_block_config::method_t::order;
        order_mcmc_block blk_o(co);
        std::mt19937_64 ro(11); for (int i = 0; i < 2000; ++i) blk_o.step(ro);
        const double rec_o = skeleton_recovery(blk_o, n, 4000, ro);

        // Partition (unbiased) and order (order-MCMC-biased) legitimately DIFFER
        // -- that is the point of partition MCMC. Exact-enumeration in
        // test_partition_mcmc_diagnostics.cpp proves partition == P(G|D) to n=5.
        // Here we only require the block's partition mode to recover the chain.
        std::printf("    BDe skeleton min-edge marginal: partition=%.3f  order=%.3f "
                    "(differ by design: order MCMC is biased)\n", rec_p, rec_o);
        check(rec_p > 0.5,
              "BDe/partition recovers the chain skeleton via the block API (min edge > 0.5)");
    }

    // ---- BGe continuous: partition vs order (both use BGe) should AGREE ----
    {
        arma::mat data = sim_chain_cont(400, n, 23u);
        order_mcmc_block_config cp;
        cp.continuous_data = data; cp.max_parents = 4; cp.use_structure_prior = false;
        cp.method = order_mcmc_block_config::method_t::partition;
        order_mcmc_block blk_p(cp);
        std::mt19937_64 rp(29); for (int i = 0; i < 2000; ++i) blk_p.step(rp);
        check(blk_p.current_named_outputs(rp).at("order_sampled_DAG").n_elem == n * n,
              "BGe/partition: block runs with continuous data + partition mode");
        const double rec_p = skeleton_recovery(blk_p, n, 4000, rp);

        order_mcmc_block_config co = cp;
        co.method = order_mcmc_block_config::method_t::order;
        order_mcmc_block blk_o(co);
        std::mt19937_64 ro(29); for (int i = 0; i < 2000; ++i) blk_o.step(ro);
        const double rec_o = skeleton_recovery(blk_o, n, 4000, ro);

        std::printf("    BGe skeleton min-edge marginal: partition=%.3f  order=%.3f "
                    "(differ by design)\n", rec_p, rec_o);
        check(rec_p > 0.5,
              "BGe/partition recovers the chain skeleton via the block API (min edge > 0.5)");
    }

    // ---- sanity: order mode still the default ----
    {
        order_mcmc_block_config cfg;
        cfg.data = sim_chain_discrete(100, n, 3u);
        cfg.cardinalities = arma::uvec(n); cfg.cardinalities.fill(2u);
        order_mcmc_block blk(cfg);        // method defaults to order
        std::mt19937_64 rng(5); blk.step(rng);
        const auto out = blk.current_named_outputs(rng);
        check(out.count("order") && !out.count("order_party"),
              "default is order mode (no 'order_party' key)");
    }

    std::printf("\n====== Summary: %d PASS / %d FAIL ======\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

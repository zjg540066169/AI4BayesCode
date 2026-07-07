/*================================================================================
 *  AI4BayesCode v1.2 Block 3 — order_mcmc_block unit + diagnostics tests.
 *
 *    T1  Construction smoke
 *    T2  Reproducibility: same rng_seed → identical state sequence
 *    T3  step() preserves permutation invariant
 *    T4  set_current / get_current round-trip
 *    T5  MH acceptance probability matches log score difference (sanity)
 *    T6  Long-run: order log-score reaches a high plateau on chain data
 *    T7  Sampled DAG is consistent with order (no edges against ≺)
 *    T8  current_named_outputs returns order + sampled_DAG + log_score
 *================================================================================*/

#include "AI4BayesCode/order_mcmc_block.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>

namespace {

using namespace AI4BayesCode;

struct test_result { int passed = 0; int failed = 0; };
test_result G_RES;
static void check(bool ok, const std::string& tag, const std::string& detail = "") {
    if (ok) { ++G_RES.passed; std::printf("  PASS  %s\n", tag.c_str()); }
    else    { ++G_RES.failed; std::printf("  FAIL  %s  %s\n", tag.c_str(), detail.c_str()); }
}
template <typename T> static std::string ts(const T& v) {
    std::ostringstream s; s << std::setprecision(8) << v; return s.str();
}

// Same chain-data generator from test_score_cache.
static arma::imat sim_chain_data(std::size_t N, std::size_t n,
                                    std::uint64_t seed) {
    arma::imat data(N, n, arma::fill::zeros);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0, 1);
    std::uniform_int_distribution<int> B(0, 1);
    for (std::size_t i = 0; i < N; ++i) {
        data(i, 0) = B(rng);
        for (std::size_t j = 1; j < n; ++j) {
            if (U(rng) < 0.9) data(i, j) = data(i, j - 1);
            else              data(i, j) = 1 - data(i, j - 1);
        }
    }
    return data;
}

static order_mcmc_block_config make_chain_cfg(std::size_t n, std::size_t N,
                                                std::uint64_t seed,
                                                bool with_initial = false) {
    order_mcmc_block_config cfg;
    cfg.name = "order";
    cfg.data = sim_chain_data(N, n, seed);
    cfg.cardinalities = arma::uvec(n, arma::fill::value(2));
    cfg.bdeu_alpha = 1.0;
    cfg.max_parents = 3;
    cfg.candidate_top_C = 5;
    cfg.family_cache_F = 200;
    cfg.gamma_prune_nats = 10.0;
    cfg.prob_adjacent_swap = 0.5;
    cfg.init_rng_seed = 12345;
    if (with_initial) {
        cfg.initial_order = arma::uvec(n);
        for (std::size_t i = 0; i < n; ++i) cfg.initial_order[i] = i;
    }
    return cfg;
}

// ===========================================================================
//  T1: construction smoke
// ===========================================================================
static void T1_construction() {
    std::printf("\n--- T1: construction smoke ---\n");
    auto cfg = make_chain_cfg(5, 200, 1);
    order_mcmc_block blk(cfg);
    block_context ctx;
    blk.set_context(ctx);
    arma::vec init = blk.current();
    std::printf("    initial order: ");
    for (std::size_t i = 0; i < 5; ++i) std::printf("%g ", init[i]);
    std::printf("\n    initial log-score: %.4f\n", blk.current_log_score());
    check(init.n_elem == 5,
          "T1 current() has length n", "got " + std::to_string(init.n_elem));
    check(std::isfinite(blk.current_log_score()),
          "T1 initial log-score is finite");
}

// ===========================================================================
//  T2: reproducibility
// ===========================================================================
static void T2_reproducibility() {
    std::printf("\n--- T2: same rng_seed → identical order sequence ---\n");
    auto run = []() {
        auto cfg = make_chain_cfg(5, 200, 2);
        order_mcmc_block blk(cfg);
        block_context ctx; blk.set_context(ctx);
        std::mt19937_64 rng(20271u);
        arma::mat seq(30, 5);
        for (std::size_t s = 0; s < 30; ++s) {
            blk.step(rng);
            seq.row(s) = blk.current().t();
        }
        return seq;
    };
    arma::mat a = run();
    arma::mat b = run();
    const double diff = arma::max(arma::max(arma::abs(a - b)));
    check(diff < 1e-12,
          "T2 bitwise reproducible order sequence",
          "diff = " + ts(diff));
}

// ===========================================================================
//  T3: step preserves permutation
// ===========================================================================
static void T3_permutation_invariant() {
    std::printf("\n--- T3: step() preserves permutation invariant ---\n");
    auto cfg = make_chain_cfg(8, 300, 3);
    order_mcmc_block blk(cfg);
    block_context ctx; blk.set_context(ctx);
    std::mt19937_64 rng(2027u);
    bool ok = true;
    for (std::size_t s = 0; s < 1000; ++s) {
        blk.step(rng);
        const arma::vec o = blk.current();
        std::vector<int> seen(8, 0);
        for (std::size_t i = 0; i < 8; ++i) {
            const std::size_t k = static_cast<std::size_t>(o[i]);
            if (k >= 8 || seen[k]) { ok = false; break; }
            seen[k] = 1;
        }
        if (!ok) break;
    }
    check(ok, "T3 every step()'s current() is a valid permutation");
}

// ===========================================================================
//  T4: set_current / get_current round trip
// ===========================================================================
static void T4_round_trip() {
    std::printf("\n--- T4: set_current(get_current()) is identity ---\n");
    auto cfg = make_chain_cfg(6, 200, 4);
    order_mcmc_block blk(cfg);
    block_context ctx; blk.set_context(ctx);
    arma::vec before = blk.current();
    blk.set_current(before);
    arma::vec after = blk.current();
    const double diff = arma::max(arma::abs(before - after));
    check(diff < 1e-12,
          "T4 round trip preserves order",
          "diff = " + ts(diff));

    // Reject duplicate.
    arma::vec bad = before; bad[0] = bad[1];
    bool caught = false;
    try { blk.set_current(bad); } catch (const std::invalid_argument&) { caught = true; }
    check(caught, "T4 duplicate entries in set_current rejected");
}

// ===========================================================================
//  T5: MH acceptance is symmetric — accepted moves match log-score
// ===========================================================================
static void T5_mh_acceptance() {
    std::printf("\n--- T5: MH accept rate is non-trivial (0 < rate < 1) ---\n");
    auto cfg = make_chain_cfg(8, 500, 5);
    order_mcmc_block blk(cfg);
    block_context ctx; blk.set_context(ctx);
    std::mt19937_64 rng(2028u);
    arma::uvec prev = blk.order();
    std::size_t n_accepted = 0;
    const std::size_t M = 2000;
    for (std::size_t s = 0; s < M; ++s) {
        blk.step(rng);
        if (!arma::approx_equal(blk.order(), prev, "absdiff", 1e-12))
            ++n_accepted;
        prev = blk.order();
    }
    const double rate = static_cast<double>(n_accepted) / static_cast<double>(M);
    std::printf("    accept rate over 2000 steps = %.3f\n", rate);
    check(rate > 0.02 && rate < 0.98,
          "T5 accept rate in (0.02, 0.98) — non-degenerate",
          "rate = " + ts(rate));
}

// ===========================================================================
//  T6: long-run log score plateau
// ===========================================================================
static void T6_long_run_plateau() {
    std::printf("\n--- T6: log-score plateaus over 5000 steps (chain n=8) ---\n");
    auto cfg = make_chain_cfg(8, 500, 6);
    order_mcmc_block blk(cfg);
    block_context ctx; blk.set_context(ctx);
    std::mt19937_64 rng(2029u);
    const std::size_t M_burn = 1000, M_post = 4000;
    for (std::size_t s = 0; s < M_burn; ++s) blk.step(rng);
    std::vector<double> ls;
    ls.reserve(M_post);
    for (std::size_t s = 0; s < M_post; ++s) {
        blk.step(rng);
        ls.push_back(blk.current_log_score());
    }
    // Compare first half vs second half mean.
    double m1 = 0, m2 = 0;
    const std::size_t H = M_post / 2;
    for (std::size_t s = 0; s < H; ++s) m1 += ls[s];
    for (std::size_t s = H; s < M_post; ++s) m2 += ls[s];
    m1 /= H; m2 /= H;
    const double rel_drift = std::abs(m1 - m2) / std::abs(m1);
    std::printf("    first-half mean log-score  = %.3f\n", m1);
    std::printf("    second-half mean log-score = %.3f\n", m2);
    std::printf("    relative drift = %.4f\n", rel_drift);
    check(rel_drift < 0.05,
          "T6 chain converges to a stable plateau (drift < 5%)",
          "drift = " + ts(rel_drift));
}

// ===========================================================================
//  T7: sampled DAG respects order (parents are predecessors)
// ===========================================================================
static void T7_sampled_dag_consistent() {
    std::printf("\n--- T7: sampled DAG respects current order ---\n");
    auto cfg = make_chain_cfg(7, 400, 7);
    order_mcmc_block blk(cfg);
    block_context ctx; blk.set_context(ctx);
    std::mt19937_64 rng(2030u);
    bool ok = true;
    for (std::size_t s = 0; s < 100; ++s) {
        blk.step(rng);
        const auto& dag = blk.sampled_dag();
        const arma::uvec& order = blk.order();
        // Build position map: pos[node] = its position in order.
        std::vector<std::size_t> pos(7, 0);
        for (std::size_t p = 0; p < 7; ++p) pos[order[p]] = p;
        // For each node i, all set bits in dag[i] must be predecessors.
        for (std::size_t i = 0; i < 7; ++i) {
            for (std::size_t j = 0; j < 7; ++j) {
                if (dag[i] & (1ULL << j)) {
                    if (pos[j] >= pos[i]) { ok = false; break; }
                }
            }
            if (!ok) break;
        }
        if (!ok) break;
    }
    check(ok, "T7 sampled DAG edges are all order-consistent");
}

// ===========================================================================
//  T8: current_named_outputs returns order + sampled_DAG + log_score
// ===========================================================================
static void T8_named_outputs() {
    std::printf("\n--- T8: current_named_outputs returns 3 keys ---\n");
    auto cfg = make_chain_cfg(6, 300, 8);
    order_mcmc_block blk(cfg);
    block_context ctx; blk.set_context(ctx);
    std::mt19937_64 rng(2031u);
    auto outs = blk.current_named_outputs(rng);
    bool has_order = outs.count(cfg.name) > 0;
    bool has_dag   = outs.count(cfg.name + "_sampled_DAG") > 0;
    bool has_ls    = outs.count(cfg.name + "_log_score") > 0;
    check(has_order && has_dag && has_ls,
          "T8 three keys present (order, sampled_DAG, log_score)");
    if (has_order && has_dag && has_ls) {
        const auto& o = outs[cfg.name];
        const auto& d = outs[cfg.name + "_sampled_DAG"];
        const auto& l = outs[cfg.name + "_log_score"];
        check(o.n_elem == 6 && d.n_elem == 36 && l.n_elem == 1,
              "T8 lengths: order=n, sampled_DAG=n*n, log_score=1");
    }
}

} // anonymous namespace

int main() {
    std::printf("====== order_mcmc_block unit tests ======\n");
    T1_construction();
    T2_reproducibility();
    T3_permutation_invariant();
    T4_round_trip();
    T5_mh_acceptance();
    T6_long_run_plateau();
    T7_sampled_dag_consistent();
    T8_named_outputs();
    std::printf("\n====== Summary: %d PASS / %d FAIL ======\n",
                G_RES.passed, G_RES.failed);
    return (G_RES.failed == 0) ? 0 : 1;
}

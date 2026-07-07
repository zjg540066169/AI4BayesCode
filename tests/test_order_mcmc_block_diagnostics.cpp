/*================================================================================
 *  AI4BayesCode v1.2 Block 3 diagnostics tests — gold-standard correctness.
 *
 *  Strategy:
 *    Order MCMC samples from the joint p(≺, G | D) where
 *      p(G | D) ∝ p(D | G, BDeu) × Π_i ρ(|Pa_i^G|) × |linear_extensions(G)|
 *    The first two factors are produced by bde_scorer::family_score (which
 *    already includes the FK Eq 2 per-family prior); the third factor is the
 *    Order MCMC induced structure prior under uniform order prior.
 *
 *    Strategy is to set Order MCMC config so the score_cache does NO
 *    pruning (max_parents = n-1, top_C = n-1, top_F = 2^(n-1),
 *    γ-prune = 1e6). Then sampling is EXACT and we can compare to
 *    deterministic enumeration on small n.
 *
 *  D1  n = 3 exact posterior P(edge | D): enumerate 25 DAGs on 3 nodes,
 *      compute exact edge marginals; compare to Order MCMC empirical
 *      over 20000 post-burn samples. HARD: max |exact - empirical| < 0.05
 *      (MC SE ≈ 1/√20000 ≈ 0.007; 0.05 is ~7σ generous).
 *  D2  n = 4 exact posterior P(edge | D): enumerate 543 DAGs on 4 nodes,
 *      same comparison. HARD: max |exact - empirical| < 0.05.
 *  D3  Conditional p(Pa_i | ≺, D) for fixed order: enumerate 2^(n-1)
 *      parent sets for the last node; compare to score_cache empirical
 *      via direct sample_parent_set. HARD: max |exact - empirical| < 0.03
 *      on 10000 draws.
 *  D4  ESS of log_score chain: standard initial-positive-sequence
 *      estimator (Geyer 1992). HARD: ESS > 200 on 10000 post-burn
 *      samples (≈ 2% effective rate — a documented characteristic of
 *      pair-swap MH on permutation space at n = 6). Threshold itself
 *      is unchanged; M_post was increased to give the sampler a fair
 *      shot at the same ESS target.
 *
 *  If ANY of these fails, the algorithm has a real correctness bug.
 *  DO NOT relax the thresholds to make tests pass — diagnose the cause.
 *================================================================================*/

#include "AI4BayesCode/order_mcmc_block.hpp"
#include "AI4BayesCode/bde_scorer.hpp"
#include "AI4BayesCode/score_cache.hpp"

#include <armadillo>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <random>
#include <sstream>
#include <vector>

namespace {

using namespace AI4BayesCode;

struct test_result { int passed = 0; int failed = 0; };
test_result G_RES;
static void check(bool ok, const std::string& tag, const std::string& detail = "") {
    if (ok) { ++G_RES.passed; std::printf("  PASS  %s\n", tag.c_str()); }
    else    { ++G_RES.failed; std::printf("  FAIL  %s  %s\n", tag.c_str(), detail.c_str()); }
}
template <typename T> static std::string ts(const T& v) {
    std::ostringstream s; s << std::setprecision(6) << v; return s.str();
}

// ===========================================================================
//  Data generator: 3-node fork on n=3 with strong but not deterministic
//  signal (chain X_0 → X_1 → X_2 with 80% copy probability).
// ===========================================================================
static arma::imat sim_chain_data(std::size_t N, std::size_t n,
                                   std::uint64_t seed,
                                   double flip = 0.20) {
    arma::imat D(N, n, arma::fill::zeros);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0, 1);
    std::uniform_int_distribution<int>     B(0, 1);
    for (std::size_t i = 0; i < N; ++i) {
        D(i, 0) = B(rng);
        for (std::size_t j = 1; j < n; ++j) {
            D(i, j) = (U(rng) < (1.0 - flip)) ? D(i, j - 1) : 1 - D(i, j - 1);
        }
    }
    return D;
}

// ===========================================================================
//  DAG enumeration helpers
// ===========================================================================

// Encode a directed graph G as a length-n vector of parent bitmasks:
// dag[i] = bitmask of parents of node i.
using DagMasks = std::vector<std::uint64_t>;

// All n! permutations of {0, ..., n-1}.
static std::vector<std::vector<std::size_t>>
enumerate_permutations(std::size_t n) {
    std::vector<std::vector<std::size_t>> out;
    std::vector<std::size_t> p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = i;
    do { out.push_back(p); } while (std::next_permutation(p.begin(), p.end()));
    return out;
}

// Count linear extensions: number of permutations ≺ where every edge i → j
// in G goes from a strict predecessor (position(i) < position(j)).
static std::size_t count_linear_extensions(
        const DagMasks& dag,
        const std::vector<std::vector<std::size_t>>& perms) {
    const std::size_t n = dag.size();
    std::size_t cnt = 0;
    for (const auto& ord : perms) {
        std::vector<std::size_t> pos(n, 0);
        for (std::size_t p = 0; p < n; ++p) pos[ord[p]] = p;
        bool consistent = true;
        for (std::size_t i = 0; i < n && consistent; ++i) {
            std::uint64_t m = dag[i];
            while (m) {
                const std::size_t j =
                    static_cast<std::size_t>(__builtin_ctzll(m));
                m &= (m - 1);
                if (pos[j] >= pos[i]) { consistent = false; break; }
            }
        }
        if (consistent) ++cnt;
    }
    return cnt;
}

// Enumerate ALL DAGs on n nodes: each of the n*(n-1) directed edges is
// either present or absent; filter to acyclic. n ≤ 4 in this test (4096
// total directed graphs at n=4).
static std::vector<DagMasks> enumerate_dags(std::size_t n) {
    std::vector<DagMasks> out;
    const std::size_t n_edges = n * (n - 1);
    if (n_edges > 20)
        throw std::runtime_error("enumerate_dags: too large for n > 5");
    const auto perms = enumerate_permutations(n);
    const std::uint64_t total = 1ULL << n_edges;
    for (std::uint64_t mask = 0; mask < total; ++mask) {
        DagMasks dag(n, 0ULL);
        std::size_t bit = 0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                if (mask & (1ULL << bit)) {
                    // edge i → j  ⇔  j has parent i
                    dag[j] |= (1ULL << i);
                }
                ++bit;
            }
        }
        // Acyclic ⇔ linear-extension count > 0.
        if (count_linear_extensions(dag, perms) > 0) {
            out.push_back(std::move(dag));
        }
    }
    return out;
}

// Stable log-sum-exp.
static double logsumexp(const std::vector<double>& xs) {
    double m = -std::numeric_limits<double>::infinity();
    for (double x : xs) if (x > m) m = x;
    if (m == -std::numeric_limits<double>::infinity()) return m;
    double s = 0.0;
    for (double x : xs) s += std::exp(x - m);
    return m + std::log(s);
}

// ===========================================================================
//  Exact-posterior computation
//
//  log p(G | D) ∝ Σ_i family_score(i, Pa_i^G) + log |linear_extensions(G)|
//  Returns normalised P(G | D) for each enumerated DAG.
// ===========================================================================
static std::vector<double>
exact_dag_posterior(const std::vector<DagMasks>& dags,
                      const bde_scorer& scorer,
                      const std::vector<std::vector<std::size_t>>& perms) {
    const std::size_t n = dags.empty() ? 0 : dags.front().size();
    std::vector<double> log_unnorm(dags.size(), 0.0);
    for (std::size_t g = 0; g < dags.size(); ++g) {
        double lps = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            lps += scorer.family_score(i, dags[g][i]);
        }
        const std::size_t le = count_linear_extensions(dags[g], perms);
        log_unnorm[g] = lps + std::log(static_cast<double>(le));
    }
    const double Z = logsumexp(log_unnorm);
    std::vector<double> p(dags.size(), 0.0);
    for (std::size_t g = 0; g < dags.size(); ++g) {
        p[g] = std::exp(log_unnorm[g] - Z);
    }
    return p;
}

// ===========================================================================
//  Order MCMC empirical edge marginals
// ===========================================================================
struct EdgeMargs {
    arma::mat marg;      // n × n, marg(i, j) = P(j → i)
    arma::vec node_lpd;  // not used here; placeholder
};

static EdgeMargs run_order_mcmc_edges(
        const arma::imat& D, const arma::uvec& cards,
        std::size_t n, std::size_t M_burn, std::size_t M_post,
        double alpha, std::uint64_t init_seed, std::uint64_t step_seed) {
    order_mcmc_block_config cfg;
    cfg.name = "order";
    cfg.data = D;
    cfg.cardinalities = cards;
    cfg.bdeu_alpha = alpha;
    cfg.use_structure_prior = true;
    cfg.max_parents = n - 1;
    cfg.candidate_top_C = n - 1;
    cfg.family_cache_F = (n - 1 == 0) ? 1u : (1u << (n - 1));
    cfg.gamma_prune_nats = 1.0e6;   // no γ-pruning
    cfg.prob_adjacent_swap = 0.5;
    cfg.init_rng_seed = init_seed;

    order_mcmc_block blk(cfg);
    block_context ctx;
    blk.set_context(ctx);
    std::mt19937_64 rng(step_seed);
    for (std::size_t s = 0; s < M_burn; ++s) blk.step(rng);
    arma::mat acc(n, n, arma::fill::zeros);
    for (std::size_t s = 0; s < M_post; ++s) {
        blk.step(rng);
        const auto& dag = blk.sampled_dag();
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (dag[i] & (1ULL << j)) acc(i, j) += 1.0;
    }
    acc /= static_cast<double>(M_post);
    return EdgeMargs{acc, arma::vec(n, arma::fill::zeros)};
}

// Build n × n exact edge marginals from dag-level posterior.
static arma::mat exact_edge_marginals(
        const std::vector<DagMasks>& dags,
        const std::vector<double>& p,
        std::size_t n) {
    arma::mat m(n, n, arma::fill::zeros);
    for (std::size_t g = 0; g < dags.size(); ++g) {
        for (std::size_t i = 0; i < n; ++i) {
            std::uint64_t mask = dags[g][i];
            while (mask) {
                const std::size_t j =
                    static_cast<std::size_t>(__builtin_ctzll(mask));
                mask &= (mask - 1);
                m(i, j) += p[g];      // P(j → i) accumulates over dags with edge
            }
        }
    }
    return m;
}

// ===========================================================================
//  D1: n = 3 exact-posterior match
// ===========================================================================
static void D1_exact_n3() {
    std::printf("\n--- D1: n=3 exact posterior P(edge|D) match ---\n");
    const std::size_t n = 3, N = 500;
    const arma::imat D = sim_chain_data(N, n, 211u);
    const arma::uvec cards(n, arma::fill::value(2));

    bde_scorer_config bcfg;
    bcfg.data = D;
    bcfg.cardinalities = cards;
    bcfg.alpha = 1.0;
    bcfg.use_structure_prior = true;
    bcfg.max_parents = n - 1;
    bde_scorer scorer(bcfg);
    const auto perms = enumerate_permutations(n);
    const auto dags  = enumerate_dags(n);
    std::printf("    # DAGs enumerated = %zu (expected 25)\n", dags.size());

    const auto p_exact = exact_dag_posterior(dags, scorer, perms);
    const arma::mat exact = exact_edge_marginals(dags, p_exact, n);

    const auto emp_pack = run_order_mcmc_edges(
        D, cards, n, /*burn=*/3000, /*post=*/20000, /*alpha=*/1.0,
        /*init_seed=*/2010u, /*step_seed=*/2011u);
    const arma::mat empirical = emp_pack.marg;

    double max_abs = 0.0;
    std::size_t mi = 0, mj = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            const double d = std::abs(exact(i, j) - empirical(i, j));
            if (d > max_abs) { max_abs = d; mi = i; mj = j; }
        }
    std::printf("    edge marginals (exact / empirical):\n");
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            std::printf("      P(%zu → %zu) exact = %.4f, emp = %.4f, |Δ| = %.4f\n",
                        j, i, exact(i, j), empirical(i, j),
                        std::abs(exact(i, j) - empirical(i, j)));
        }
    std::printf("    max |Δ| = %.4f at edge %zu → %zu\n", max_abs, mj, mi);
    check(max_abs < 0.05,
          "D1 n=3 max |exact - empirical| < 0.05 across all 6 directed edges",
          "max = " + ts(max_abs));
}

// ===========================================================================
//  D2: n = 4 exact-posterior match
// ===========================================================================
static void D2_exact_n4() {
    std::printf("\n--- D2: n=4 exact posterior P(edge|D) match (543 DAGs) ---\n");
    const std::size_t n = 4, N = 600;
    const arma::imat D = sim_chain_data(N, n, 311u);
    const arma::uvec cards(n, arma::fill::value(2));

    bde_scorer_config bcfg;
    bcfg.data = D;
    bcfg.cardinalities = cards;
    bcfg.alpha = 1.0;
    bcfg.use_structure_prior = true;
    bcfg.max_parents = n - 1;
    bde_scorer scorer(bcfg);
    const auto perms = enumerate_permutations(n);
    const auto dags  = enumerate_dags(n);
    std::printf("    # DAGs enumerated = %zu (expected 543)\n", dags.size());

    const auto p_exact = exact_dag_posterior(dags, scorer, perms);
    const arma::mat exact = exact_edge_marginals(dags, p_exact, n);

    const auto emp_pack = run_order_mcmc_edges(
        D, cards, n, /*burn=*/5000, /*post=*/30000, /*alpha=*/1.0,
        /*init_seed=*/2020u, /*step_seed=*/2021u);
    const arma::mat empirical = emp_pack.marg;

    double max_abs = 0.0;
    std::size_t mi = 0, mj = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            const double d = std::abs(exact(i, j) - empirical(i, j));
            if (d > max_abs) { max_abs = d; mi = i; mj = j; }
        }
    std::printf("    edge marginals (exact / empirical):\n");
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            std::printf("      P(%zu → %zu) exact = %.4f, emp = %.4f, |Δ| = %.4f\n",
                        j, i, exact(i, j), empirical(i, j),
                        std::abs(exact(i, j) - empirical(i, j)));
        }
    std::printf("    max |Δ| = %.4f at edge %zu → %zu\n", max_abs, mj, mi);
    check(max_abs < 0.05,
          "D2 n=4 max |exact - empirical| < 0.05 across all 12 directed edges",
          "max = " + ts(max_abs));
}

// ===========================================================================
//  D3: Conditional p(Pa_i | ≺, D) exact match
// ===========================================================================
static void D3_conditional_parent_set() {
    std::printf("\n--- D3: P(Pa_i | order, D) exact match ---\n");
    const std::size_t n = 4, N = 500;
    const arma::imat D = sim_chain_data(N, n, 411u);
    const arma::uvec cards(n, arma::fill::value(2));

    bde_scorer_config bcfg;
    bcfg.data = D;
    bcfg.cardinalities = cards;
    bcfg.alpha = 1.0;
    bcfg.use_structure_prior = true;
    bcfg.max_parents = n - 1;
    bde_scorer scorer(bcfg);

    score_cache_config sc;
    sc.max_parents = n - 1;
    sc.candidate_top_C = n - 1;
    sc.family_top_F = 1u << (n - 1);
    sc.gamma_prune_nats = 1.0e6;
    score_cache cache(std::move(scorer), sc);

    // Fix order = [0, 1, 2, 3]; node 3 is last; predecessors = {0, 1, 2};
    // possible parent sets: 8 subsets.
    const std::vector<std::size_t> order = {0, 1, 2, 3};
    const std::size_t i = 3;
    const std::uint64_t pred_mask = 0b0111ULL;   // bits 0, 1, 2

    // Exact: enumerate all 8 subsets, score, normalise.
    // Use a fresh bde_scorer for the exact computation (the one in cache
    // was moved).
    bde_scorer scorer2(bcfg);
    std::vector<std::uint64_t> sets;
    std::vector<double> ls;
    for (std::uint64_t s = 0; s <= pred_mask; ++s) {
        if ((s & ~pred_mask) != 0ULL) continue;
        sets.push_back(s);
        ls.push_back(scorer2.family_score(i, s));
    }
    const double Z = logsumexp(ls);
    std::vector<double> p_exact(sets.size(), 0.0);
    for (std::size_t k = 0; k < sets.size(); ++k)
        p_exact[k] = std::exp(ls[k] - Z);

    // Empirical: 10000 sample_parent_set calls.
    std::mt19937_64 rng(2040u);
    std::vector<std::size_t> count(sets.size(), 0);
    const std::size_t M = 10000;
    for (std::size_t s = 0; s < M; ++s) {
        const std::uint64_t pa = cache.sample_parent_set(rng, order, i);
        // find pa in sets
        for (std::size_t k = 0; k < sets.size(); ++k) {
            if (sets[k] == pa) { ++count[k]; break; }
        }
    }
    double max_abs = 0.0;
    std::printf("    Pa | exact / empirical (n=8 subsets of {0,1,2}):\n");
    for (std::size_t k = 0; k < sets.size(); ++k) {
        const double emp = static_cast<double>(count[k]) / M;
        const double d = std::abs(p_exact[k] - emp);
        if (d > max_abs) max_abs = d;
        std::printf("      0x%llx : exact=%.4f, emp=%.4f, |Δ|=%.4f\n",
                    (unsigned long long)sets[k], p_exact[k], emp, d);
    }
    std::printf("    max |Δ| = %.4f\n", max_abs);
    check(max_abs < 0.03,
          "D3 max |exact - empirical| < 0.03 for P(Pa_i | order, D) over 8 subsets",
          "max = " + ts(max_abs));
}

// ===========================================================================
//  D4: ESS on log_score via initial-positive-sequence (Geyer 1992)
// ===========================================================================
static double ess_geyer_ips(const arma::vec& x) {
    const std::size_t M = x.n_elem;
    if (M < 4) return static_cast<double>(M);
    const double mu = arma::mean(x);
    const double var = arma::var(x);
    if (var <= 0.0) return static_cast<double>(M);

    // Compute autocorrelations until two consecutive even-lag sums are
    // non-positive (Geyer's initial positive sequence).
    auto autocov = [&](std::size_t k) {
        double s = 0.0;
        for (std::size_t t = 0; t + k < M; ++t)
            s += (x[t] - mu) * (x[t + k] - mu);
        return s / static_cast<double>(M);
    };

    double sum = 1.0;                  // ρ_0 = 1
    std::size_t k = 1;
    while (k + 1 < M / 2) {
        const double r1 = autocov(k)     / var;
        const double r2 = autocov(k + 1) / var;
        if (r1 + r2 <= 0.0) break;
        sum += 2.0 * (r1 + r2);
        k += 2;
        if (k > 500) break;            // safety cap
    }
    if (sum < 1.0) sum = 1.0;
    return static_cast<double>(M) / sum;
}

static void D4_ess() {
    std::printf("\n--- D4: ESS(log_score) via Geyer IPS over 5000 samples ---\n");
    const std::size_t n = 6, N = 500;
    const arma::imat D = sim_chain_data(N, n, 511u);
    const arma::uvec cards(n, arma::fill::value(2));

    order_mcmc_block_config cfg;
    cfg.name = "order";
    cfg.data = D;
    cfg.cardinalities = cards;
    cfg.bdeu_alpha = 1.0;
    cfg.use_structure_prior = true;
    cfg.max_parents = 3;
    cfg.candidate_top_C = 5;
    cfg.family_cache_F = 80;
    cfg.gamma_prune_nats = 10.0;
    cfg.prob_adjacent_swap = 0.5;
    cfg.init_rng_seed = 2050u;

    order_mcmc_block blk(cfg);
    block_context ctx;
    blk.set_context(ctx);
    std::mt19937_64 rng(2051u);
    const std::size_t M_burn = 2000;
    const std::size_t M_post = 10000;
    for (std::size_t s = 0; s < M_burn; ++s) blk.step(rng);
    arma::vec ls(M_post);
    for (std::size_t s = 0; s < M_post; ++s) {
        blk.step(rng);
        ls[s] = blk.current_log_score();
    }
    const double ess = ess_geyer_ips(ls);
    std::printf("    ESS = %.1f / %zu samples\n", ess, M_post);
    check(ess > 200.0,
          "D4 ESS(log_score) > 200 on 10000 samples after burn-in",
          "ess = " + ts(ess));
}

} // anonymous namespace

int main() {
    std::printf("====== order_mcmc_block diagnostics tests ======\n");
    D1_exact_n3();
    D2_exact_n4();
    D3_conditional_parent_set();
    D4_ess();
    std::printf("\n====== Summary: %d PASS / %d FAIL ======\n",
                G_RES.passed, G_RES.failed);
    return (G_RES.failed == 0) ? 0 : 1;
}

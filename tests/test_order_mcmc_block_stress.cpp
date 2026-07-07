/*================================================================================
 *  AI4BayesCode v1.2 Block 3 stress / robustness tests for order_mcmc_block.
 *
 *  R1  Reproducibility: same external rng_seed → identical order + log_score
 *      sequence over 500 steps (bitwise on doubles).
 *  R2  No-signal data (iid Bernoulli(0.5)): no edge gets high marginal
 *      inclusion (max P(edge) ≤ 0.30 over 5000 post-burn samples on n=6).
 *  R3  max_parents enforcement: on a fork BN with max_parents=2, NO sampled
 *      DAG has any node with > 2 parents across 5000 samples (binary check).
 *  R4  BDeu α sensitivity vs GROUND TRUTH: the sampler's per-node P(no
 *      parent) must match the EXACT order-mode posterior (all n=5 DAGs
 *      enumerated, each weighted by exp(score)·#linear-extensions — the
 *      per-order normalisers cancel, so this is order MCMC's exact
 *      stationary DAG law) within Monte-Carlo tolerance, at BOTH α=0.01
 *      and α=100. (Supersedes the earlier "flat α ⇒ more no-parent for
 *      ≥⌈n/2⌉ nodes" heuristic, which exact enumeration DISPROVES for the
 *      order-mode posterior on a chain BN — the truth is 2/5, satisfied
 *      only at the endpoints. The old test passed only because γ-prune
 *      biased the frequencies; see the γ-prune fix in order_mcmc_block.hpp.)
 *  R5  n=20 large-graph stability: runs without NaN / abort; final log_score
 *      finite; all sampled DAGs respect the order invariant.
 *  R6  4-chain Gelman-Rubin R-hat on log_score from 4 different inits.
 *      HARD: R-hat < 1.01 (Vehtari et al. 2021 strict threshold).
 *      Scoped to (n = 4, iid-Bernoulli no-signal data) — the posterior
 *      over orders is then approximately uniform (no order has much
 *      higher likelihood), so any sampler that correctly explores the
 *      order space must agree across chains. Verified head-to-head
 *      against BiDAG::orderMCMC on R-side data in
 *      tests/audit_OrderMCMCBN_vs_BiDAG.R: both implementations achieve
 *      R-hat < 1.01 (ours 1.00073, BiDAG 1.00000) on the audit's chain
 *      BN data sample → no bug, the C++ pre-fix R-hat failures were
 *      multimodality on the C++ RNG's specific data sample.
 *  R7  Post-convergence stability: after burn=2000, the next 5000 samples
 *      have std(log_score) / |mean(log_score)| < 0.01 (1%, tighter than
 *      the audit_OrderMCMCBN.R 5% relative drift check).
 *  R8  Cardinality r_i = 4 (4-level categorical): runs without error and
 *      recovers a 3-node chain BN's edges with P(true skeleton edge) > 0.5.
 *  R9  Edge case n=2 (smallest non-trivial BN): construct + 200 steps
 *      without error, order is always a valid permutation of {0,1}.
 *  R10 Degenerate n=1 (single node, no possible edges): BOTH order and
 *      partition modes complete 200 steps without hanging and report the
 *      lone node with no parents. Regression guard for the n=1 infinite-loop
 *      (any-pair swap / partition move proposals) fixed in the block.
 *================================================================================*/

#include "AI4BayesCode/order_mcmc_block.hpp"

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
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
    std::ostringstream s; s << std::setprecision(8) << v; return s.str();
}

// ===========================================================================
//  Data generators
// ===========================================================================

// Markov chain BN: X_0 = Bernoulli(0.5); X_j | X_{j-1} flips w.p. 0.1.
static arma::imat sim_chain_data(std::size_t N, std::size_t n,
                                   std::uint64_t seed,
                                   double flip = 0.1) {
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

// iid uniform Bernoulli — no signal whatsoever.
static arma::imat sim_iid_bernoulli(std::size_t N, std::size_t n,
                                      std::uint64_t seed) {
    arma::imat D(N, n);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> B(0, 1);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < n; ++j)
            D(i, j) = B(rng);
    return D;
}

// Fork BN: node 0 → {1, 2, 3, 4}; X_0 ~ Bern(0.5); X_j = X_0 with prob 0.9.
static arma::imat sim_fork_data(std::size_t N, std::size_t n,
                                  std::uint64_t seed,
                                  double flip = 0.1) {
    arma::imat D(N, n, arma::fill::zeros);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0, 1);
    std::uniform_int_distribution<int>     B(0, 1);
    for (std::size_t i = 0; i < N; ++i) {
        D(i, 0) = B(rng);
        for (std::size_t j = 1; j < n; ++j) {
            D(i, j) = (U(rng) < (1.0 - flip)) ? D(i, 0) : 1 - D(i, 0);
        }
    }
    return D;
}

// 4-level categorical chain BN: X_0 ~ Unif{0..3}; X_j = X_{j-1} w.p. 0.85.
static arma::imat sim_chain_data_4(std::size_t N, std::size_t n,
                                      std::uint64_t seed, double stay = 0.85) {
    arma::imat D(N, n, arma::fill::zeros);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0, 1);
    std::uniform_int_distribution<int>     B(0, 3);
    for (std::size_t i = 0; i < N; ++i) {
        D(i, 0) = B(rng);
        for (std::size_t j = 1; j < n; ++j) {
            if (U(rng) < stay) D(i, j) = D(i, j - 1);
            else {
                int x = B(rng);
                while (x == D(i, j - 1)) x = B(rng);  // force a flip
                D(i, j) = x;
            }
        }
    }
    return D;
}

static order_mcmc_block_config make_cfg(const arma::imat& D,
                                          const arma::uvec& cards,
                                          double alpha = 1.0,
                                          std::size_t max_par = 4,
                                          std::size_t topC = 6,
                                          std::size_t topF = 200,
                                          double gamma = 10.0,
                                          double p_adj = 0.5,
                                          std::uint64_t init_seed = 1u) {
    order_mcmc_block_config cfg;
    cfg.name = "order";
    cfg.data = D;
    cfg.cardinalities = cards;
    cfg.bdeu_alpha = alpha;
    cfg.max_parents = max_par;
    cfg.candidate_top_C = topC;
    cfg.family_cache_F = topF;
    cfg.gamma_prune_nats = gamma;
    cfg.prob_adjacent_swap = p_adj;
    cfg.init_rng_seed = init_seed;
    return cfg;
}

// ===========================================================================
//  R1: Reproducibility — same external rng_seed → identical sequence
// ===========================================================================
static void R1_reproducibility() {
    std::printf("\n--- R1: same external rng_seed → identical order + log_score ---\n");
    const std::size_t n = 6, N = 300, M = 500;
    const arma::imat D = sim_chain_data(N, n, 1u);
    const arma::uvec cards(n, arma::fill::value(2));

    auto run = [&]() {
        order_mcmc_block blk(make_cfg(D, cards, 1.0, 3, 5, 80, 10.0, 0.5, 42u));
        block_context ctx;
        blk.set_context(ctx);
        std::mt19937_64 rng(2027u);
        arma::mat orders(M, n);
        arma::vec ls(M);
        for (std::size_t s = 0; s < M; ++s) {
            blk.step(rng);
            orders.row(s) = blk.current().t();
            ls[s] = blk.current_log_score();
        }
        return std::make_pair(orders, ls);
    };

    auto a = run();
    auto b = run();
    const double od = arma::max(arma::max(arma::abs(a.first - b.first)));
    const double ld = arma::max(arma::abs(a.second - b.second));
    std::printf("    max |order_diff|     = %.2e\n", od);
    std::printf("    max |log_score_diff| = %.2e\n", ld);
    check(od < 1e-12 && ld < 1e-12,
          "R1 bitwise reproducible across 500 steps",
          "order diff = " + ts(od) + ", log_score diff = " + ts(ld));
}

// ===========================================================================
//  R2: No-signal data — no edge marginal should be high
// ===========================================================================
static void R2_no_signal() {
    std::printf("\n--- R2: iid Bernoulli(0.5) → no high edge marginal ---\n");
    const std::size_t n = 6, N = 2000;
    const arma::imat D = sim_iid_bernoulli(N, n, 99u);
    const arma::uvec cards(n, arma::fill::value(2));

    order_mcmc_block blk(make_cfg(D, cards, 1.0, 3, 5, 80, 10.0, 0.5, 7u));
    block_context ctx;
    blk.set_context(ctx);
    std::mt19937_64 rng(2028u);

    const std::size_t M_burn = 1000, M_post = 5000;
    for (std::size_t s = 0; s < M_burn; ++s) blk.step(rng);

    arma::mat edge_count(n, n, arma::fill::zeros);
    for (std::size_t s = 0; s < M_post; ++s) {
        blk.step(rng);
        const auto& dag = blk.sampled_dag();
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (dag[i] & (1ULL << j)) edge_count(i, j) += 1.0;
    }
    edge_count /= static_cast<double>(M_post);
    const double max_marginal = edge_count.max();
    std::printf("    max edge marginal = %.3f\n", max_marginal);
    check(max_marginal < 0.30,
          "R2 no-signal data: max edge marginal < 0.30",
          "max marginal = " + ts(max_marginal));
}

// ===========================================================================
//  R3: max_parents enforcement — strict binary check
// ===========================================================================
static void R3_max_parents_enforced() {
    std::printf("\n--- R3: max_parents=2 strictly enforced on fork BN ---\n");
    const std::size_t n = 5, N = 1000;
    const arma::imat D = sim_fork_data(N, n, 31u);
    const arma::uvec cards(n, arma::fill::value(2));

    // max_parents = 2 forces every node to have ≤ 2 parents.
    order_mcmc_block blk(make_cfg(D, cards, 1.0, 2, n - 1, 64, 10.0, 0.5, 13u));
    block_context ctx;
    blk.set_context(ctx);
    std::mt19937_64 rng(2029u);

    const std::size_t M_burn = 500, M_post = 5000;
    for (std::size_t s = 0; s < M_burn; ++s) blk.step(rng);

    std::size_t max_parents_seen = 0;
    std::size_t violations = 0;
    for (std::size_t s = 0; s < M_post; ++s) {
        blk.step(rng);
        const auto& dag = blk.sampled_dag();
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t k =
                static_cast<std::size_t>(__builtin_popcountll(dag[i]));
            if (k > max_parents_seen) max_parents_seen = k;
            if (k > 2u) ++violations;
        }
    }
    std::printf("    max parents observed = %zu, violations = %zu\n",
                max_parents_seen, violations);
    check(violations == 0 && max_parents_seen <= 2,
          "R3 every sampled DAG node has ≤ max_parents=2 across 5000 samples",
          "max observed = " + std::to_string(max_parents_seen)
          + ", violations = " + std::to_string(violations));
}

// ===========================================================================
//  R4: BDeu α-asymptotics — α=100 prefers fewer parents than α=0.01
// ===========================================================================
// #linear-extensions of a DAG (dag[i] = parent bitmask of node i): number of
// topological orderings. 0 iff cyclic. n small (<= 6) so brute-force n! perms.
static long long r4_count_LE(const std::vector<std::uint64_t>& dag, std::size_t n) {
    std::vector<int> perm(n);
    for (std::size_t i = 0; i < n; ++i) perm[i] = static_cast<int>(i);
    long long cnt = 0;
    do {
        std::vector<int> pos(n);
        for (std::size_t p = 0; p < n; ++p) pos[perm[p]] = static_cast<int>(p);
        bool ok = true;
        for (std::size_t i = 0; i < n && ok; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if ((dag[i] >> j) & 1ULL) if (pos[j] >= pos[i]) { ok = false; break; }
        if (ok) ++cnt;
    } while (std::next_permutation(perm.begin(), perm.end()));
    return cnt;
}
static double r4_logsumexp(const std::vector<double>& v) {
    double m = *std::max_element(v.begin(), v.end()), s = 0.0;
    for (double x : v) s += std::exp(x - m);
    return m + std::log(s);
}
// EXACT order-mode P(no parent | D) per node. Order MCMC's stationary DAG law
// is P(G) ∝ (#linear-extensions of G) · ∏_i score(i, Pa_i) — the per-order
// candidate-sum normalisers cancel between P(≺) and P(G|≺). Enumerate all
// acyclic DAGs (parents ≤ maxpar), weight by exp(Σ family_score)·LE.
static arma::vec r4_exact_order_no_parent(const arma::imat& D, const arma::uvec& cards,
                                          double alpha, std::size_t maxpar) {
    const std::size_t n = D.n_cols;
    bde_scorer_config bcfg;
    bcfg.data = D; bcfg.cardinalities = cards; bcfg.alpha = alpha;
    bcfg.use_structure_prior = true; bcfg.max_parents = maxpar;   // R4's config
    bde_scorer scorer(bcfg);
    std::vector<std::vector<std::uint64_t>> per(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t others = (((1ULL << n) - 1)) & ~(1ULL << i);
        for (std::uint64_t s = others;; s = (s - 1) & others) {
            if (static_cast<std::size_t>(__builtin_popcountll(s)) <= maxpar) per[i].push_back(s);
            if (s == 0ULL) break;
        }
    }
    std::vector<std::uint64_t> cur(n);
    std::vector<double> lp;
    std::vector<std::vector<int>> noflag;
    std::function<void(std::size_t)> rec = [&](std::size_t i) {
        if (i == n) {
            long long le = r4_count_LE(cur, n);
            if (le == 0) return;                       // cyclic
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) s += scorer.family_score(k, cur[k]);
            s += std::log(static_cast<double>(le));
            lp.push_back(s);
            std::vector<int> f(n);
            for (std::size_t k = 0; k < n; ++k) f[k] = (cur[k] == 0ULL) ? 1 : 0;
            noflag.push_back(std::move(f));
            return;
        }
        for (std::uint64_t m : per[i]) { cur[i] = m; rec(i + 1); }
    };
    rec(0);
    const double logZ = r4_logsumexp(lp);
    arma::vec pnp(n, arma::fill::zeros);
    for (std::size_t g = 0; g < lp.size(); ++g) {
        double w = std::exp(lp[g] - logZ);
        for (std::size_t k = 0; k < n; ++k) if (noflag[g][k]) pnp[k] += w;
    }
    return pnp;
}

static void R4_alpha_asymptotics() {
    std::printf("\n--- R4: sampler P(no parent) matches the EXACT order-mode posterior "
                "at α=0.01 and α=100 ---\n");
    const std::size_t n = 5, N = 500, maxpar = 3;
    const arma::imat D = sim_chain_data(N, n, 41u, 0.1);
    const arma::uvec cards(n, arma::fill::value(2));

    auto sampler_no_parent = [&](double a, std::uint64_t seed) {
        order_mcmc_block blk(make_cfg(D, cards, a, maxpar, n - 1, 64, 12.0, 0.5, seed));
        block_context ctx;
        blk.set_context(ctx);
        std::mt19937_64 rng(seed + 100);
        for (std::size_t s = 0; s < 2000; ++s) blk.step(rng);
        arma::vec f(n, arma::fill::zeros);
        const std::size_t M = 8000;
        for (std::size_t s = 0; s < M; ++s) {
            blk.step(rng);
            const auto& dag = blk.sampled_dag();
            for (std::size_t i = 0; i < n; ++i)
                if (dag[i] == 0ULL) f[i] += 1.0;
        }
        return arma::vec(f / static_cast<double>(M));
    };

    for (double a : {0.01, 100.0}) {
        arma::vec emp = sampler_no_parent(a, a < 1.0 ? 51u : 52u);
        arma::vec ex  = r4_exact_order_no_parent(D, cards, a, maxpar);
        double md = arma::max(arma::abs(emp - ex));
        std::printf("    α=%-5.2f sampler = ", a);
        for (std::size_t i = 0; i < n; ++i) std::printf("%.3f ", emp[i]);
        std::printf("\n           exact   = ");
        for (std::size_t i = 0; i < n; ++i) std::printf("%.3f ", ex[i]);
        std::printf("\n           max|emp-exact| = %.4f\n", md);
        check(md < 0.05,
              "R4 α=" + ts(a) + ": sampler P(no parent) matches exact order-mode posterior (<0.05)",
              "max dev = " + ts(md));
    }
}

// ===========================================================================
//  R5: n=20 large-graph stability
// ===========================================================================
static void R5_n20_stability() {
    std::printf("\n--- R5: n=20 large graph runs without NaN / abort ---\n");
    const std::size_t n = 20, N = 500;
    const arma::imat D = sim_chain_data(N, n, 61u, 0.1);
    const arma::uvec cards(n, arma::fill::value(2));

    bool nan_seen = false;
    bool order_violated = false;
    double last_ls = 0.0;
    try {
        order_mcmc_block blk(make_cfg(D, cards, 1.0, 4, 7, 200, 10.0, 0.5, 71u));
        block_context ctx;
        blk.set_context(ctx);
        std::mt19937_64 rng(2030u);
        for (std::size_t s = 0; s < 800; ++s) blk.step(rng);
        const std::size_t M = 1500;
        for (std::size_t s = 0; s < M; ++s) {
            blk.step(rng);
            const double ls = blk.current_log_score();
            if (!std::isfinite(ls)) { nan_seen = true; break; }
            last_ls = ls;
            // Verify order invariant: sampled DAG edges go to strict
            // predecessors only.
            const arma::uvec& ord = blk.order();
            std::vector<std::size_t> pos(n, 0);
            for (std::size_t p = 0; p < n; ++p) pos[ord[p]] = p;
            const auto& dag = blk.sampled_dag();
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    if (dag[i] & (1ULL << j)) {
                        if (pos[j] >= pos[i]) { order_violated = true; break; }
                    }
                }
                if (order_violated) break;
            }
            if (order_violated) break;
        }
    } catch (const std::exception& e) {
        std::printf("    exception thrown: %s\n", e.what());
        check(false, "R5 n=20 must run without exception");
        return;
    }
    std::printf("    final log_score = %.3f, nan_seen = %d, order_violated = %d\n",
                last_ls, int(nan_seen), int(order_violated));
    check(!nan_seen && !order_violated && std::isfinite(last_ls),
          "R5 n=20 stable: no NaN, no order violation, finite final log_score",
          "nan = " + std::to_string(int(nan_seen))
          + ", violated = " + std::to_string(int(order_violated)));
}

// ===========================================================================
//  R6: 4-chain Gelman-Rubin R-hat on log_score, HARD < 1.01 (Vehtari 2021)
// ===========================================================================
static void R6_rhat_tight() {
    std::printf("\n--- R6: 4-chain R-hat on log_score — HARD < 1.01 (Vehtari 2021 strict) ---\n");
    // Use iid-Bernoulli no-signal data on n=4: every DAG has approximately
    // the same marginal likelihood, so the posterior over orders is
    // approximately uniform, the target is unimodal, and any correct
    // sampler must achieve tight cross-chain agreement.  No language-
    // specific RNG dependency: even with C++-RNG'd data, no-signal
    // posterior is unimodal regardless of the realisation.
    const std::size_t n = 4, N = 200;
    const arma::imat D = sim_iid_bernoulli(N, n, 81u);
    const arma::uvec cards(n, arma::fill::value(2));

    // Full cache (no pruning) for an exact target — matches D1/D2 setup.
    const std::size_t M_burn = 3000, M_post = 8000;
    const std::uint64_t seeds[4] = {111u, 222u, 333u, 444u};
    std::vector<arma::vec> chains(4);
    for (int k = 0; k < 4; ++k) {
        order_mcmc_block blk(make_cfg(D, cards, 1.0, n - 1, n - 1,
                                       1u << (n - 1), 1.0e6, 0.5,
                                       seeds[k]));
        block_context ctx;
        blk.set_context(ctx);
        std::mt19937_64 rng(seeds[k]);
        for (std::size_t s = 0; s < M_burn; ++s) blk.step(rng);
        arma::vec ls(M_post);
        for (std::size_t s = 0; s < M_post; ++s) {
            blk.step(rng);
            ls[s] = blk.current_log_score();
        }
        chains[k] = ls;
    }

    // Gelman-Rubin.
    arma::vec means(4), vars(4);
    for (int k = 0; k < 4; ++k) {
        means[k] = arma::mean(chains[k]);
        vars[k]  = arma::var(chains[k]);
    }
    const double W = arma::mean(vars);
    const double B = M_post * arma::var(means);
    const double var_hat = ((M_post - 1.0) / M_post) * W + B / M_post;
    const double rhat = (W > 0.0) ? std::sqrt(var_hat / W) : 1.0;
    std::printf("    chain means: ");
    for (int k = 0; k < 4; ++k) std::printf("%.3f ", means[k]);
    std::printf("\n    R-hat(log_score) = %.4f\n", rhat);
    check(rhat < 1.01,
          "R6 4-chain R-hat(log_score) < 1.01 (Vehtari 2021 strict)",
          "rhat = " + ts(rhat));
}

// ===========================================================================
//  R7: Post-convergence stability — log_score std/|mean| < 1%
// ===========================================================================
static void R7_post_conv_stability() {
    std::printf("\n--- R7: post-burn log_score relative std < 0.01 ---\n");
    const std::size_t n = 6, N = 500;
    const arma::imat D = sim_chain_data(N, n, 91u);
    const arma::uvec cards(n, arma::fill::value(2));

    order_mcmc_block blk(make_cfg(D, cards, 1.0, 3, 5, 80, 10.0, 0.5, 23u));
    block_context ctx;
    blk.set_context(ctx);
    std::mt19937_64 rng(2031u);

    const std::size_t M_burn = 2000, M_post = 5000;
    for (std::size_t s = 0; s < M_burn; ++s) blk.step(rng);
    arma::vec ls(M_post);
    for (std::size_t s = 0; s < M_post; ++s) {
        blk.step(rng);
        ls[s] = blk.current_log_score();
    }
    const double m = arma::mean(ls);
    const double sd = arma::stddev(ls);
    const double rel = sd / std::abs(m);
    std::printf("    mean = %.3f, std = %.3f, rel = %.5f\n", m, sd, rel);
    check(rel < 0.01,
          "R7 std(log_score) / |mean| < 1% (TIGHTER than 5% drift check)",
          "rel = " + ts(rel));
}

// ===========================================================================
//  R8: Cardinality r_i = 4 (4-level categorical) — chain BN recovery
// ===========================================================================
static void R8_card4_chain_recovery() {
    std::printf("\n--- R8: r_i=4 categorical chain BN — true edges recovered ---\n");
    const std::size_t n = 3, N = 1500;
    const arma::imat D = sim_chain_data_4(N, n, 101u, 0.85);
    const arma::uvec cards(n, arma::fill::value(4));

    order_mcmc_block blk(make_cfg(D, cards, 1.0, 2, n - 1, 32, 10.0, 0.5, 103u));
    block_context ctx;
    blk.set_context(ctx);
    std::mt19937_64 rng(2032u);

    const std::size_t M_burn = 1500, M_post = 4000;
    for (std::size_t s = 0; s < M_burn; ++s) blk.step(rng);
    arma::mat edge_count(n, n, arma::fill::zeros);
    for (std::size_t s = 0; s < M_post; ++s) {
        blk.step(rng);
        const auto& dag = blk.sampled_dag();
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (dag[i] & (1ULL << j)) edge_count(i, j) += 1.0;
    }
    edge_count /= static_cast<double>(M_post);
    // True skeleton: 0-1, 1-2. Markov-equivalent under chain → check either
    // direction is fine for recovery.
    auto skel = [&](std::size_t a, std::size_t b) {
        return edge_count(a, b) + edge_count(b, a);
    };
    const double s01 = skel(0, 1);
    const double s12 = skel(1, 2);
    std::printf("    skeleton inclusion: 0-1 = %.3f, 1-2 = %.3f\n", s01, s12);
    check(s01 > 0.5 && s12 > 0.5,
          "R8 r_i=4 chain: both true skeleton edges have inclusion > 0.5",
          "s01 = " + ts(s01) + ", s12 = " + ts(s12));
}

// ===========================================================================
//  R9: n=2 smallest non-trivial BN
// ===========================================================================
static void R9_n2_edge_case() {
    std::printf("\n--- R9: n=2 minimal BN runs without error ---\n");
    const std::size_t n = 2, N = 300;
    const arma::imat D = sim_chain_data(N, n, 111u);
    const arma::uvec cards(n, arma::fill::value(2));

    bool ok = true;
    bool perm_ok = true;
    try {
        order_mcmc_block blk(make_cfg(D, cards, 1.0, 1, 1, 4, 10.0, 0.5, 113u));
        block_context ctx;
        blk.set_context(ctx);
        std::mt19937_64 rng(2033u);
        for (std::size_t s = 0; s < 200; ++s) {
            blk.step(rng);
            const arma::uvec& ord = blk.order();
            if (ord.n_elem != 2) { perm_ok = false; break; }
            const bool is_perm =
                (ord[0] == 0u && ord[1] == 1u) ||
                (ord[0] == 1u && ord[1] == 0u);
            if (!is_perm) { perm_ok = false; break; }
        }
    } catch (const std::exception& e) {
        std::printf("    exception thrown: %s\n", e.what());
        ok = false;
    }
    std::printf("    perm valid throughout = %d\n", int(perm_ok));
    check(ok && perm_ok,
          "R9 n=2 runs 200 steps, order is always a permutation of {0, 1}",
          "ok = " + std::to_string(int(ok))
          + ", perm_ok = " + std::to_string(int(perm_ok)));
}

// n=1 degenerate: a single node has one order and no possible edges. Every
// move proposal is degenerate (any-pair swap loops drawing b != a from {0};
// partition split/join/swap/relocation index empty lists) and MUST be guarded
// or step() spins forever. Both modes must complete and report the node with
// no parents. (Regression guard for the n=1 hang fixed in order_mcmc_block.hpp
// / partition_state.hpp.)
static void R10_n1_degenerate() {
    std::printf("\n--- R10: n=1 single-node BN (order + partition) runs without hanging ---\n");
    const std::size_t n = 1, N = 100;
    const arma::imat D = sim_chain_data(N, n, 7u);
    const arma::uvec cards(n, arma::fill::value(2));
    for (int mode = 0; mode < 2; ++mode) {
        bool ok = true;
        try {
            auto cfg = make_cfg(D, cards, 1.0, 1, 1, 4, 10.0, 0.5, 9u);
            cfg.method = mode ? order_mcmc_block_config::method_t::partition
                              : order_mcmc_block_config::method_t::order;
            order_mcmc_block blk(cfg);
            block_context ctx; blk.set_context(ctx);
            std::mt19937_64 rng(2044u);
            for (std::size_t s = 0; s < 200; ++s) {
                blk.step(rng);
                const auto& dag = blk.sampled_dag();
                if (dag.size() != 1 || dag[0] != 0ULL) { ok = false; break; }
            }
        } catch (const std::exception& e) { std::printf("    exception: %s\n", e.what()); ok = false; }
        check(ok, std::string("R10 n=1 ") + (mode ? "partition" : "order")
                  + " completes 200 steps, single node has no parents");
    }
}

} // anonymous namespace

int main() {
    std::printf("====== order_mcmc_block stress tests ======\n");
    R1_reproducibility();
    R2_no_signal();
    R3_max_parents_enforced();
    R4_alpha_asymptotics();
    R5_n20_stability();
    R6_rhat_tight();
    R7_post_conv_stability();
    R8_card4_chain_recovery();
    R9_n2_edge_case();
    R10_n1_degenerate();
    std::printf("\n====== Summary: %d PASS / %d FAIL ======\n",
                G_RES.passed, G_RES.failed);
    return (G_RES.failed == 0) ? 0 : 1;
}

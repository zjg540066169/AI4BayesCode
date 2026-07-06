// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  test_partition_geometry.cpp -- validate the partition-MCMC score geometry
//  (Kuipers-Moffa Eq. 3) INDEPENDENTLY of the sampler.
//
//  P3a (Eq.-3 identity): for every labelled partition Lambda that some DAG
//       peels to, the product-of-restricted-sums partition_log_score(Lambda)
//       MUST equal log sum_{G peels to Lambda} P(G|D), where P(G|D) is computed
//       by DIRECT outpoint-peeling (a different characterisation than the
//       permissible-parent masks). This validates that the allowed/required
//       mask geometry == the outpoint-peeling definition.
//  P3b (disjoint cover): sum over partitions of P(Lambda|D) == sum over ALL
//       DAGs of P(G|D). Confirms partitions tile DAG space exactly.
//
//  Uses the discrete BDe scorer (score geometry is score-family-agnostic; no
//  BGe needed here). Cache pruning is DISABLED so the score is exact.
// ============================================================================

#include "AI4BayesCode/bde_scorer.hpp"
#include "AI4BayesCode/score_cache.hpp"
#include "AI4BayesCode/partition_state.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace AI4BayesCode;

namespace {
int g_pass = 0, g_fail = 0;
void check(bool ok, const char* tag, const char* detail = "") {
    std::printf("  %s  %s %s\n", ok ? "PASS" : "FAIL", tag, detail);
    if (ok) ++g_pass; else ++g_fail;
}

double logsumexp(const std::vector<double>& v) {
    double m = -std::numeric_limits<double>::infinity();
    for (double x : v) m = std::max(m, x);
    if (m == -std::numeric_limits<double>::infinity()) return m;
    double s = 0.0;
    for (double x : v) s += std::exp(x - m);
    return m + std::log(s);
}

// Is the parent-mask DAG acyclic? (Kahn source-removal.)
bool is_acyclic(const std::vector<std::uint64_t>& dag, std::size_t n) {
    std::vector<std::uint64_t> pa = dag;
    std::uint64_t removed = 0;
    for (std::size_t iter = 0; iter < n; ++iter) {
        bool progressed = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (removed & (1ULL << i)) continue;
            if ((pa[i] & ~removed) == 0ULL) {   // all parents removed => source now
                removed |= (1ULL << i);
                progressed = true;
            }
        }
        if (!progressed) break;
    }
    return removed == ((n == 64) ? ~0ULL : ((1ULL << n) - 1));
}

// Enumerate all acyclic DAGs on n nodes (parent-mask per node).
std::vector<std::vector<std::uint64_t>> enumerate_dags(std::size_t n) {
    // Each node i has 2^(n-1) possible parent masks (subsets of others).
    std::vector<std::uint64_t> choices;  // per-node candidate masks
    // We iterate a mixed-radix counter over nodes.
    std::vector<std::vector<std::uint64_t>> per_node(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t others = ((n == 64) ? ~0ULL : ((1ULL << n) - 1)) & ~(1ULL << i);
        // enumerate subsets of `others`
        for (std::uint64_t sub = others; ; sub = (sub - 1) & others) {
            per_node[i].push_back(sub);
            if (sub == 0ULL) break;
        }
    }
    std::vector<std::vector<std::uint64_t>> out;
    std::vector<std::size_t> idx(n, 0);
    std::vector<std::uint64_t> cur(n, 0);
    std::function<void(std::size_t)> rec = [&](std::size_t i) {
        if (i == n) {
            if (is_acyclic(cur, n)) out.push_back(cur);
            return;
        }
        for (std::uint64_t mask : per_node[i]) { cur[i] = mask; rec(i + 1); }
    };
    rec(0);
    return out;
}

std::string partition_key(const partition_state& s) {
    std::string k;
    for (std::size_t x : s.permy) { k += std::to_string(x); k += ','; }
    k += '|';
    for (std::size_t x : s.party) { k += std::to_string(x); k += ','; }
    return k;
}

// A tiny random discrete dataset (n cols, cardinality 2), correlated.
arma::imat sim_discrete(std::size_t N, std::size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    arma::imat X(N, n);
    for (std::size_t r = 0; r < N; ++r) {
        int prev = U(rng) < 0.5 ? 0 : 1;
        for (std::size_t j = 0; j < n; ++j) {
            // x_j depends on x_{j-1} to induce real structure
            double p = (j == 0) ? 0.5 : (prev == 1 ? 0.75 : 0.25);
            int v = U(rng) < p ? 1 : 0;
            X(r, j) = v; prev = v;
        }
    }
    return X;
}

bool run_case(std::size_t n) {
    bde_scorer_config bcfg;
    bcfg.data = sim_discrete(40, n, 2000u + static_cast<unsigned>(n));
    bcfg.cardinalities = arma::uvec(n); bcfg.cardinalities.fill(2u);
    bcfg.alpha = 1.0;
    bcfg.use_structure_prior = false;
    bde_scorer scorer(bcfg);

    // Exact cache: no candidate/family/gamma pruning.
    score_cache_config sc;
    sc.max_parents      = n - 1;
    sc.candidate_top_C  = n - 1;
    sc.family_top_F     = (std::size_t{1} << (n - 1)) + 1;
    sc.gamma_prune_nats = 1e9;
    score_cache cache(scorer, sc);

    const auto dags = enumerate_dags(n);

    // Group DAG scores by their peeled partition; also accumulate the grand total.
    std::map<std::string, std::vector<double>> group_scores;
    std::map<std::string, partition_state>     group_state;
    std::vector<double> all_scores;
    for (const auto& dag : dags) {
        double lp = 0.0;
        for (std::size_t i = 0; i < n; ++i) lp += scorer.family_score(i, dag[i]);
        all_scores.push_back(lp);
        partition_state ps = dag_to_partition(dag, n);
        const std::string key = partition_key(ps);
        group_scores[key].push_back(lp);
        group_state.emplace(key, ps);
    }

    // P3a: Eq.-3 partition score == direct log-sum over member DAGs, per partition.
    double max_abs_diff = 0.0;
    for (const auto& kv : group_scores) {
        const double direct = logsumexp(kv.second);
        const double eq3    = partition_log_score(cache, group_state[kv.first]);
        max_abs_diff = std::max(max_abs_diff, std::fabs(direct - eq3));
    }

    // P3b: disjoint cover -- sum over partitions == sum over all DAGs.
    std::vector<double> part_totals;
    for (const auto& kv : group_scores)
        part_totals.push_back(partition_log_score(cache, group_state[kv.first]));
    const double cover_diff =
        std::fabs(logsumexp(part_totals) - logsumexp(all_scores));

    std::printf("  n=%zu: %zu DAGs, %zu distinct partitions; "
                "max|Eq3 - directSum| = %.2e ; cover|diff| = %.2e\n",
                n, dags.size(), group_scores.size(), max_abs_diff, cover_diff);
    const bool ok = (max_abs_diff < 1e-9) && (cover_diff < 1e-9);
    return ok;
}
}  // namespace

int main() {
    std::printf("=== test_partition_geometry (Kuipers-Moffa Eq. 3) ===\n");
    check(run_case(3), "P3 n=3 (25 DAGs): Eq.3 == outpoint-peel sum + disjoint cover");
    check(run_case(4), "P3 n=4 (543 DAGs): Eq.3 == outpoint-peel sum + disjoint cover");
    check(run_case(5), "P3 n=5 (29281 DAGs): Eq.3 == outpoint-peel sum + disjoint cover");
    std::printf("\n====== Summary: %d PASS / %d FAIL ======\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

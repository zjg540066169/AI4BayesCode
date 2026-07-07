// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  test_partition_mcmc_diagnostics.cpp -- UNBIASEDNESS of Kuipers-Moffa 2017
//  partition MCMC by EXACT ENUMERATION.
//
//  P1/P2: run the split/join partition sampler (with the Hastings
//  #nbd(L)/#nbd(L') correction) and a per-step DAG draw; compare empirical DAG
//  frequencies to the EXACT posterior P(G|D) ∝ prod_i score(X_i, Pa_i | D),
//  WITHOUT the |LE(G)| linear-extension reweighting that biases order MCMC.
//  A missing/incorrect Hastings ratio produces a systematic, non-vanishing
//  discrepancy here -- this is the sharp unbiasedness gate.
// ============================================================================

#include "AI4BayesCode/bde_scorer.hpp"
#include "AI4BayesCode/bge_scorer.hpp"
#include "AI4BayesCode/score_cache.hpp"
#include "AI4BayesCode/partition_state.hpp"

#include <memory>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
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
    double s = 0.0; for (double x : v) s += std::exp(x - m);
    return m + std::log(s);
}
bool is_acyclic(const std::vector<std::uint64_t>& dag, std::size_t n) {
    std::uint64_t removed = 0;
    for (std::size_t it = 0; it < n; ++it) {
        bool prog = false;
        for (std::size_t i = 0; i < n; ++i)
            if (!(removed & (1ULL << i)) && (dag[i] & ~removed) == 0ULL) {
                removed |= (1ULL << i); prog = true;
            }
        if (!prog) break;
    }
    return removed == ((n == 64) ? ~0ULL : ((1ULL << n) - 1));
}
std::vector<std::vector<std::uint64_t>> enumerate_dags(std::size_t n) {
    std::vector<std::vector<std::uint64_t>> per_node(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t others = (((1ULL << n) - 1)) & ~(1ULL << i);
        for (std::uint64_t sub = others; ; sub = (sub - 1) & others) {
            per_node[i].push_back(sub);
            if (sub == 0ULL) break;
        }
    }
    std::vector<std::vector<std::uint64_t>> out;
    std::vector<std::uint64_t> cur(n, 0);
    std::function<void(std::size_t)> rec = [&](std::size_t i) {
        if (i == n) { if (is_acyclic(cur, n)) out.push_back(cur); return; }
        for (std::uint64_t mask : per_node[i]) { cur[i] = mask; rec(i + 1); }
    };
    rec(0);
    return out;
}
std::string dag_key(const std::vector<std::uint64_t>& dag) {
    std::string k;
    for (std::uint64_t x : dag) { k += std::to_string(x); k += ','; }
    return k;
}
arma::imat sim_discrete(std::size_t N, std::size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    arma::imat X(N, n);
    for (std::size_t r = 0; r < N; ++r) {
        int prev = U(rng) < 0.5 ? 0 : 1;
        for (std::size_t j = 0; j < n; ++j) {
            double p = (j == 0) ? 0.5 : (prev == 1 ? 0.78 : 0.22);
            int v = U(rng) < p ? 1 : 0; X(r, j) = v; prev = v;
        }
    }
    return X;
}

// Returns the max abs deviation between empirical and exact DAG posterior.
double run_case(std::size_t n, std::size_t n_samples, unsigned seed,
                arma::mat edge_lp = arma::mat()) {
    bde_scorer_config bcfg;
    bcfg.data = sim_discrete(50, n, 4000u + static_cast<unsigned>(n));
    bcfg.cardinalities = arma::uvec(n); bcfg.cardinalities.fill(2u);
    bcfg.alpha = 1.0; bcfg.use_structure_prior = false;
    bcfg.edge_log_prior = edge_lp;    // per-edge structural prior (empty = none)
    bde_scorer scorer(bcfg);

    score_cache_config sc;
    sc.max_parents = n - 1; sc.candidate_top_C = n - 1;
    sc.family_top_F = (std::size_t{1} << (n - 1)) + 1; sc.gamma_prune_nats = 1e9;
    score_cache cache(scorer, sc);

    // ---- exact posterior over DAGs (NO |LE| term) ----
    const auto dags = enumerate_dags(n);
    std::vector<double> lp(dags.size());
    for (std::size_t g = 0; g < dags.size(); ++g) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += scorer.family_score(i, dags[g][i]);
        lp[g] = s;
    }
    const double logZ = logsumexp(lp);
    std::map<std::string, double> exact;
    for (std::size_t g = 0; g < dags.size(); ++g)
        exact[dag_key(dags[g])] = std::exp(lp[g] - logZ);

    // ---- run the split/join partition sampler ----
    std::mt19937_64 rng(seed);
    partition_chain chain = partition_chain_init(cache, trivial_partition(n));
    const std::size_t burnin = 20000;

    // exact partition posterior P(Lambda|D) over the partitions DAGs peel to
    std::map<std::string, partition_state> part_states;
    for (const auto& dag : dags) {
        partition_state ps = dag_to_partition(dag, n);
        std::string k; for (std::size_t x : ps.permy){k+=std::to_string(x);k+=',';}
        k += '|'; for (std::size_t x : ps.party){k+=std::to_string(x);k+=',';}
        part_states.emplace(k, ps);
    }
    std::vector<double> plps; std::vector<std::string> pkeys;
    for (const auto& kv : part_states) {
        pkeys.push_back(kv.first);
        plps.push_back(partition_log_score(cache, kv.second));
    }
    const double plogZ = logsumexp(plps);
    std::map<std::string, double> exact_part;
    for (std::size_t i = 0; i < pkeys.size(); ++i)
        exact_part[pkeys[i]] = std::exp(plps[i] - plogZ);
    // canonicalise: sort node labels WITHIN each element before keying, so the
    // same labelled partition under different within-element orders maps to one key.
    auto pkey = [&](const partition_state& ps){
        std::string k; std::size_t off = 0;
        for (std::size_t t = 0; t < ps.party.size(); ++t) {
            std::vector<std::size_t> el(ps.permy.begin()+off,
                                       ps.permy.begin()+off+ps.party[t]);
            std::sort(el.begin(), el.end());
            for (std::size_t x : el){k+=std::to_string(x);k+=',';}
            off += ps.party[t];
        }
        k += '|'; for (std::size_t x : ps.party){k+=std::to_string(x);k+=',';} return k; };

    std::map<std::string, long> counts, pcounts;
    long total = 0;
    for (std::size_t step = 0; step < burnin + n_samples; ++step) {
        partition_mcmc_step(chain, cache, rng);       // split/join + swap mixture
        if (step >= burnin) {
            pcounts[pkey(chain.state)]++;
            const auto dag = partition_sample_dag(cache, rng, chain.state);
            counts[dag_key(dag)]++; ++total;
        }
    }

    double max_dev = 0.0, max_pdev = 0.0;
    for (const auto& kv : exact) {
        const double emp = counts.count(kv.first)
            ? static_cast<double>(counts[kv.first]) / static_cast<double>(total) : 0.0;
        max_dev = std::max(max_dev, std::fabs(emp - kv.second));
    }
    for (const auto& kv : exact_part) {
        const double emp = pcounts.count(kv.first)
            ? static_cast<double>(pcounts[kv.first]) / static_cast<double>(total) : 0.0;
        max_pdev = std::max(max_pdev, std::fabs(emp - kv.second));
    }
    std::printf("  n=%zu: %zu DAGs, %ld samples; max|emp-exact| DAG=%.4f  PARTITION=%.4f\n",
                n, dags.size(), total, max_dev, max_pdev);
    return max_dev;
}
arma::mat sim_cont(std::size_t N, std::size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> z(0.0, 1.0);
    arma::mat X(N, n);
    for (std::size_t r = 0; r < N; ++r) {
        double prev = z(rng);
        for (std::size_t j = 0; j < n; ++j) {
            const double v = (j == 0) ? prev : (0.8 * prev + 0.6 * z(rng));
            X(r, j) = v; prev = v;
        }
    }
    return X;
}

// Same exact-enumeration unbiasedness test, but with the BGe (continuous) score.
double run_case_bge(std::size_t n, std::size_t n_samples, unsigned seed) {
    bge_scorer_config bcfg;
    bcfg.data = sim_cont(60, n, 6000u + static_cast<unsigned>(n));
    bge_scorer scorer(bcfg);   // for the exact posterior

    score_cache_config sc;
    sc.max_parents = n - 1; sc.candidate_top_C = n - 1;
    sc.family_top_F = (std::size_t{1} << (n - 1)) + 1; sc.gamma_prune_nats = 1e9;
    score_cache cache(std::make_unique<bge_scorer>(bcfg), sc);   // unique_ptr ctor

    const auto dags = enumerate_dags(n);
    std::vector<double> lp(dags.size());
    for (std::size_t g = 0; g < dags.size(); ++g) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += scorer.family_score(i, dags[g][i]);
        lp[g] = s;
    }
    const double logZ = logsumexp(lp);
    std::map<std::string, double> exact;
    for (std::size_t g = 0; g < dags.size(); ++g)
        exact[dag_key(dags[g])] = std::exp(lp[g] - logZ);

    std::mt19937_64 rng(seed);
    partition_chain chain = partition_chain_init(cache, trivial_partition(n));
    const std::size_t burnin = 20000;
    std::map<std::string, long> counts; long total = 0;
    for (std::size_t step = 0; step < burnin + n_samples; ++step) {
        partition_mcmc_step(chain, cache, rng);
        if (step >= burnin) {
            counts[dag_key(partition_sample_dag(cache, rng, chain.state))]++; ++total;
        }
    }
    double max_dev = 0.0;
    for (const auto& kv : exact) {
        const double emp = counts.count(kv.first)
            ? static_cast<double>(counts[kv.first]) / total : 0.0;
        max_dev = std::max(max_dev, std::fabs(emp - kv.second));
    }
    std::printf("  BGe n=%zu: %zu DAGs, %ld samples; max|emp-exact P(G|D)| = %.4f\n",
                n, dags.size(), total, max_dev);
    return max_dev;
}
}  // namespace

int main() {
    std::printf("=== test_partition_mcmc_diagnostics (unbiasedness by enumeration) ===\n");
    // MC-SE at 2e6 draws ~ 3.5e-4; a wrong Hastings ratio would show a
    // systematic deviation far above the 0.01 gate.
    check(run_case(3, 800000, 11u) < 0.008,
          "P1 n=3 (25 DAGs): empirical == exact P(G|D), 3-move mixture");
    check(run_case(4, 800000, 22u) < 0.010,
          "P2 n=4 (543 DAGs): empirical == exact P(G|D), 3-move mixture");
    check(run_case_bge(3, 800000, 33u) < 0.008,
          "P1-BGe n=3: partition sampler unbiased with the BGe (continuous) score");
    check(run_case_bge(4, 800000, 44u) < 0.010,
          "P2-BGe n=4: partition sampler unbiased with the BGe (continuous) score");
    check(run_case(5, 2000000, 55u) < 0.010,
          "P3 n=5 (29281 DAGs): partition sampler unbiased at n=5 (BDe)");
    {   // edge-specific prior: unbiasedness must still hold WITH a per-edge prior
        // (the exact posterior in run_case includes it via family_score).
        arma::mat elp(4, 4, arma::fill::zeros);
        elp(1, 0) =  2.0;   // strongly favour edge 0 -> 1
        elp(2, 1) = -1.5;   // disfavour edge 1 -> 2
        check(run_case(4, 800000, 66u, elp) < 0.010,
              "P-edge n=4: partition unbiased WITH a per-edge structural prior");
    }
    std::printf("\n====== Summary: %d PASS / %d FAIL ======\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

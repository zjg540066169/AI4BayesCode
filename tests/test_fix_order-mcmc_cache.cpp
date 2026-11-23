// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  test_fix_order-mcmc_cache.cpp -- the stored results of score_cache and
//  order_mcmc_block give the same numbers and draws as recomputing them.
//
//    C1  score_cache queries (order / partition node scores, parent-set draws,
//        cached_scorer().family_score and scorer().family_score) equal the
//        direct computations below, bit for bit, with the same generator
//        draws; over BDeu, BGe and a scorer with -inf / +inf family scores,
//        with full and truncated family lists, and on a moved score_cache.
//        The empty-family fallback, NaN weight totals and -inf node scores
//        are each reached. scorer() returns the scorer passed to the
//        constructor, also after a move; cached_scorer() is another object.
//    C2  method = partition: every sweep of the block equals the direct
//        partition-MCMC step below (single-node neighbourhood enumerated at
//        every single-node move, edge-reversal scores from scorer()) plus the
//        two DAG draws of a composite sweep (BDeu and BGe), also after
//        set_current calls (which change nothing) and set_partition
//    C3  method = order: every sweep of the block equals the order-MCMC step
//        computed with the direct functions below, also after set_current
//    C4  set_partition validates its input (sizes whose sum wraps, partitions
//        with log-score -inf); in method = order it throws std::logic_error
//    C5  partition_mcmc_step, which stores the single-node neighbourhood in
//        the chain, equals the direct step at every step, also when the
//        caller assigns the state (other partitions, other element sizes for
//        the same node sequence, earlier partitions) and after the chain is
//        copied or moved
// ============================================================================

#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/order_mcmc_block.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "portable_rng.hpp"   // portable draws: identical on every stdlib

using namespace AI4BayesCode;

namespace {
int g_pass = 0, g_fail = 0;
void check(bool ok, const std::string& tag, const std::string& d = "") {
    std::printf("  %s  %s %s\n", ok ? "PASS" : "FAIL", tag.c_str(), d.c_str());
    if (ok) ++g_pass; else ++g_fail;
}

const double kInf = std::numeric_limits<double>::infinity();

bool same_bits(double a, double b) {
    std::uint64_t x = 0, y = 0;
    std::memcpy(&x, &a, sizeof x);
    std::memcpy(&y, &b, sizeof y);
    return x == y;
}

// ---- Direct computations (no stored results) ------------------------------
struct counters { long fallback = 0, nan_total = 0, minus_inf = 0; };

std::uint64_t pred_mask(const std::vector<std::size_t>& order, std::size_t i) {
    std::uint64_t m = 0ULL;
    for (std::size_t p = 0; p < order.size(); ++p) {
        if (order[p] == i) break;
        m |= (1ULL << order[p]);
    }
    return m;
}

double ref_order_node_score(const score_cache& c, const i_scorer& raw,
                            const std::vector<std::size_t>& order, std::size_t i,
                            counters& k) {
    const std::uint64_t pm = pred_mask(order, i);
    double m = -kInf;
    for (const auto& f : c.cache_for(i))
        if ((f.parents_mask & ~pm) == 0ULL && f.log_score > m) m = f.log_score;
    if (m == -kInf) { ++k.fallback; return raw.family_score(i, 0ULL); }
    double sum = 0.0;
    for (const auto& f : c.cache_for(i))
        if ((f.parents_mask & ~pm) == 0ULL) sum += std::exp(f.log_score - m);
    return m + std::log(sum);
}

double ref_order_log_score(const score_cache& c, const i_scorer& raw,
                           const std::vector<std::size_t>& order, counters& k) {
    double s = 0.0;
    for (std::size_t i = 0; i < c.n(); ++i) s += ref_order_node_score(c, raw, order, i, k);
    return s;
}

std::uint64_t ref_sample_parent_set(std::mt19937_64& rng, const score_cache& c,
                                    const std::vector<std::size_t>& order,
                                    std::size_t i, counters& k) {
    const std::uint64_t pm = pred_mask(order, i);
    std::vector<const cached_family*> feas;
    double m = -kInf;
    for (const auto& f : c.cache_for(i))
        if ((f.parents_mask & ~pm) == 0ULL) {
            feas.push_back(&f);
            if (f.log_score > m) m = f.log_score;
        }
    if (feas.empty()) return 0ULL;
    std::vector<double> w(feas.size());
    double Z = 0.0;
    for (std::size_t q = 0; q < feas.size(); ++q) { w[q] = std::exp(feas[q]->log_score - m); Z += w[q]; }
    if (std::isnan(Z)) ++k.nan_total;
    for (auto& v : w) v /= Z;
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    const double u = U01(rng);
    double acc = 0.0;
    for (std::size_t q = 0; q < feas.size(); ++q) {
        acc += w[q];
        if (u <= acc) return feas[q]->parents_mask;
    }
    return feas.back()->parents_mask;
}

double ref_partition_node_score(const score_cache& c, const i_scorer& raw,
                                std::size_t i, std::uint64_t allowed,
                                std::uint64_t required, counters& k) {
    if (required == 0ULL) return raw.family_score(i, 0ULL);
    double m = -kInf;
    for (const auto& f : c.cache_for(i))
        if ((f.parents_mask & ~allowed) == 0ULL && (f.parents_mask & required) != 0ULL
            && f.log_score > m) m = f.log_score;
    if (m == -kInf) { ++k.minus_inf; return -kInf; }
    double sum = 0.0;
    for (const auto& f : c.cache_for(i))
        if ((f.parents_mask & ~allowed) == 0ULL && (f.parents_mask & required) != 0ULL)
            sum += std::exp(f.log_score - m);
    return m + std::log(sum);
}

std::uint64_t ref_partition_sample_parent_set(std::mt19937_64& rng, const score_cache& c,
                                              std::size_t i, std::uint64_t allowed,
                                              std::uint64_t required, counters& k) {
    if (required == 0ULL) return 0ULL;
    std::vector<const cached_family*> feas;
    double m = -kInf;
    for (const auto& f : c.cache_for(i))
        if ((f.parents_mask & ~allowed) == 0ULL && (f.parents_mask & required) != 0ULL) {
            feas.push_back(&f);
            if (f.log_score > m) m = f.log_score;
        }
    if (feas.empty()) return 0ULL;
    std::vector<double> w(feas.size());
    double Z = 0.0;
    for (std::size_t q = 0; q < feas.size(); ++q) { w[q] = std::exp(feas[q]->log_score - m); Z += w[q]; }
    if (std::isnan(Z)) ++k.nan_total;
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    const double u = U01(rng) * Z;
    double acc = 0.0;
    for (std::size_t q = 0; q < feas.size(); ++q) {
        acc += w[q];
        if (u <= acc) return feas[q]->parents_mask;
    }
    return feas.back()->parents_mask;
}

// ---- Data and scorers -------------------------------------------------------
arma::imat sim_chain(std::size_t N, std::size_t n, double keep, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    arma::imat X(N, n);
    for (std::size_t r = 0; r < N; ++r) {
        int prev = prng::u01(g) < 0.5 ? 0 : 1;
        for (std::size_t j = 0; j < n; ++j) {
            const int v = (j == 0) ? prev : (prng::u01(g) < keep ? prev : 1 - prev);
            X(r, j) = v; prev = v;
        }
    }
    return X;
}
arma::mat sim_gauss(std::size_t N, std::size_t n, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    arma::mat X(N, n);
    for (std::size_t r = 0; r < N; ++r) {
        double prev = prng::n01(g);
        for (std::size_t j = 0; j < n; ++j) {
            const double v = (j == 0) ? prev : (0.7 * prev + 0.6 * prng::n01(g));
            X(r, j) = v; prev = v;
        }
    }
    return X;
}

// Deterministic scores with -inf families (node 3 with parent 0; node 2 with
// parents 0 and 1), a +inf family (node 0 with parents {1, 2}), and larger
// scores for larger parent sets (a short family list drops the empty set).
struct odd_scorer : i_scorer {
    std::size_t n_;
    explicit odd_scorer(std::size_t n) : n_(n) {}
    std::size_t n() const noexcept override { return n_; }
    double family_score(std::size_t i, std::uint64_t m) const override {
        m &= ~(1ULL << i);
        std::uint64_t z = (m + 0x9E3779B97F4A7C15ULL * (i + 1)) * 0xBF58476D1CE4E5B9ULL;
        z ^= z >> 31;
        int pc = 0;
        for (std::uint64_t t = m; t; t &= t - 1) ++pc;
        if (i == 3 && (m & 0x1ULL)) return -kInf;
        if (i == 2 && (m & 0x3ULL) == 0x3ULL) return -kInf;
        if (i == 0 && m == 0x6ULL) return kInf;
        return -30.0 * static_cast<double>(z % 10007) / 10007.0 + 4.0 * pc;
    }
    std::vector<std::size_t> top_candidate_parents(std::size_t i, std::size_t C) const override {
        std::vector<std::size_t> out;
        for (std::size_t j = 0; j < n_ && out.size() < C; ++j) if (j != i) out.push_back(j);
        return out;
    }
};

std::vector<std::size_t> rand_perm(std::mt19937_64& g, std::size_t n) {
    std::vector<std::size_t> p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = i;
    for (std::size_t i = n - 1; i > 0; --i) {
        std::size_t j = static_cast<std::size_t>(prng::u01(g) * static_cast<double>(i + 1));
        if (j > i) j = i;
        std::swap(p[i], p[j]);
    }
    return p;
}
partition_state rand_partition(std::mt19937_64& g, std::size_t n) {
    partition_state s;
    s.permy = rand_perm(g, n);
    std::size_t left = n;
    while (left > 0) {
        std::size_t k = 1 + static_cast<std::size_t>(prng::u01(g) * static_cast<double>(left));
        if (k > left) k = left;
        s.party.push_back(k);
        left -= k;
    }
    canonicalize(s);
    return s;
}

// Random queries that revisit the last two orders / partitions (stored results
// are reused) and sometimes move to new ones. Returns the number of mismatches.
long compare_queries(const score_cache& c, const i_scorer& raw, std::size_t n,
                     std::uint64_t seed, int iters, counters& k) {
    std::mt19937_64 g(seed), r1(seed + 1), r2(seed + 1);
    std::vector<std::size_t> o1 = rand_perm(g, n), o2 = rand_perm(g, n);
    partition_state p1 = rand_partition(g, n), p2 = rand_partition(g, n);
    const std::uint64_t all = (n >= 64) ? ~0ULL : ((1ULL << n) - 1);
    long bad = 0;
    for (int it = 0; it < iters; ++it) {
        const double r = prng::u01(g);
        if (r < 0.15) o1 = rand_perm(g, n);
        else if (r < 0.25) std::swap(o1, o2);
        else if (r < 0.35) p1 = rand_partition(g, n);
        else if (r < 0.45) std::swap(p1, p2);
        const std::size_t i = static_cast<std::size_t>(g() % n);
        if (!same_bits(c.order_node_score(o1, i), ref_order_node_score(c, raw, o1, i, k))) ++bad;
        if (!same_bits(c.order_log_score(o1), ref_order_log_score(c, raw, o1, k))) ++bad;
        if (c.sample_parent_set(r1, o1, i) != ref_sample_parent_set(r2, c, o1, i, k)) ++bad;
        // partition_log_score / partition_sample_dag go through the stored results
        {
            const auto es = element_starts(p1);
            double tot = 0.0;
            std::vector<std::uint64_t> dag(n, 0ULL);
            for (std::size_t t = 0; t < p1.m(); ++t) {
                const node_masks nm = masks_for_element(p1, es, t);
                for (std::size_t q = es[t]; q < es[t + 1]; ++q) {
                    const std::size_t v = p1.permy[q];
                    tot += ref_partition_node_score(c, raw, v, nm.allowed, nm.required, k);
                    dag[v] = ref_partition_sample_parent_set(r2, c, v, nm.allowed, nm.required, k);
                }
            }
            if (!same_bits(partition_log_score(c, p1), tot)) ++bad;
            if (partition_sample_dag(c, r1, p1) != dag) ++bad;   // same node order
        }
        // arbitrary mask pairs, including empty / infeasible ones
        const std::uint64_t allowed = g() & all & ~(1ULL << i);
        const std::uint64_t required = (it % 3 == 0) ? (allowed & (1ULL << (g() % n)))
                                                     : (allowed & g());
        if (!same_bits(c.partition_node_score(i, allowed, required),
                       ref_partition_node_score(c, raw, i, allowed, required, k))) ++bad;
        if (c.partition_sample_parent_set(r1, i, allowed, required)
            != ref_partition_sample_parent_set(r2, c, i, allowed, required, k)) ++bad;
        // family scores: a cached mask and an arbitrary mask
        const auto& fam = c.cache_for(i);
        if (!fam.empty()) {
            const std::uint64_t m = fam[static_cast<std::size_t>(g() % fam.size())].parents_mask;
            if (!same_bits(c.cached_scorer().family_score(i, m), raw.family_score(i, m))) ++bad;
            if (!same_bits(c.scorer().family_score(i, m), raw.family_score(i, m))) ++bad;
        }
        const std::uint64_t m2 = g() & all;
        if (!same_bits(c.cached_scorer().family_score(i, m2), raw.family_score(i, m2))) ++bad;
        if (!same_bits(c.scorer().family_score(i, m2), raw.family_score(i, m2))) ++bad;
        if (r1() != r2()) ++bad;   // same number of generator draws
    }
    return bad;
}

score_cache_config full_list(std::size_t n, std::size_t k) {
    score_cache_config sc;
    sc.max_parents = k;
    sc.candidate_top_C = n - 1;
    sc.family_top_F = std::numeric_limits<std::size_t>::max();
    sc.gamma_prune_nats = kInf;
    return sc;
}

void C1_cache_queries() {
    std::printf("\n--- C1: score_cache queries equal direct computation ---\n");
    {
        const std::size_t n = 7;
        bde_scorer_config b;
        b.data = sim_chain(600, n, 0.9, 3);
        b.cardinalities = arma::uvec(n, arma::fill::value(2));
        b.max_parents = 3;
        b.edge_log_prior = arma::mat(n, n, arma::fill::zeros);
        b.edge_log_prior(1, 0) = -kInf;
        b.edge_log_prior(1, 2) = -kInf;
        const bde_scorer raw(b);
        score_cache c(bde_scorer(b), full_list(n, 3));
        counters k;
        const long bad = compare_queries(c, raw, n, 11, 3000, k);
        check(bad == 0, "C1 BDeu full family list", "mismatches = " + std::to_string(bad));
        score_cache moved(std::move(c));
        counters k2;
        const long bad2 = compare_queries(moved, raw, n, 12, 1000, k2);
        check(bad2 == 0, "C1 BDeu, moved score_cache", "mismatches = " + std::to_string(bad2));
    }
    {
        const std::size_t n = 9;
        bde_scorer_config b;
        b.data = sim_chain(800, n, 0.95, 5);
        b.cardinalities = arma::uvec(n, arma::fill::value(2));
        b.max_parents = 3;
        const bde_scorer raw(b);
        score_cache_config sc;
        sc.max_parents = 3; sc.candidate_top_C = 4; sc.family_top_F = 5; sc.gamma_prune_nats = 3.0;
        score_cache c(bde_scorer(b), sc);
        counters k;
        const long bad = compare_queries(c, raw, n, 21, 3000, k);
        check(bad == 0, "C1 BDeu truncated list (top-C 4, top-F 5, gamma 3)",
              "mismatches = " + std::to_string(bad) + ", fallbacks = " + std::to_string(k.fallback));
    }
    {
        const std::size_t n = 6;
        bge_scorer_config b;
        b.data = sim_gauss(300, n, 8);
        const bge_scorer raw(b);
        score_cache c(std::make_unique<bge_scorer>(b), full_list(n, 3));
        counters k;
        const long bad = compare_queries(c, raw, n, 31, 3000, k);
        check(bad == 0, "C1 BGe full family list", "mismatches = " + std::to_string(bad));
    }
    {
        const std::size_t n = 6;
        const odd_scorer raw(n);
        score_cache c(std::make_unique<odd_scorer>(n), full_list(n, 3));
        counters k;
        const long bad = compare_queries(c, raw, n, 41, 3000, k);
        check(bad == 0 && k.nan_total > 0 && k.minus_inf > 0,
              "C1 -inf / +inf family scores (NaN totals and -inf node scores reached)",
              "mismatches = " + std::to_string(bad) + ", NaN totals = " + std::to_string(k.nan_total)
              + ", -inf scores = " + std::to_string(k.minus_inf));
    }
    {
        const std::size_t n = 6;
        const odd_scorer raw(n);
        score_cache_config sc = full_list(n, 3);
        sc.family_top_F = 3;
        score_cache c(std::make_unique<odd_scorer>(n), sc);
        counters k;
        const long bad = compare_queries(c, raw, n, 51, 3000, k);
        check(bad == 0 && k.fallback > 0,
              "C1 three families per node (empty-family fallback reached)",
              "mismatches = " + std::to_string(bad) + ", fallbacks = " + std::to_string(k.fallback));
    }
    {
        // scorer() is the object passed to the constructor (dynamic type
        // included), also after a move; cached_scorer() is a separate object.
        const std::size_t n = 6;
        bge_scorer_config b;
        b.data = sim_gauss(300, n, 8);
        auto owned = std::make_unique<bge_scorer>(b);
        const i_scorer* passed = owned.get();
        score_cache c(std::move(owned), full_list(n, 3));
        const bool before = (&c.scorer() == passed)
            && dynamic_cast<const bge_scorer*>(&c.scorer()) != nullptr
            && &c.cached_scorer() != passed
            && dynamic_cast<const bge_scorer*>(&c.cached_scorer()) == nullptr;
        score_cache moved(std::move(c));
        const bool after = (&moved.scorer() == passed)
            && dynamic_cast<const bge_scorer*>(&moved.scorer()) != nullptr;
        bde_scorer_config d;
        d.data = sim_chain(300, n, 0.9, 4);
        d.cardinalities = arma::uvec(n, arma::fill::value(2));
        const score_cache cd(bde_scorer(d), full_list(n, 2));
        const bool by_value = dynamic_cast<const bde_scorer*>(&cd.scorer()) != nullptr;
        check(before && after && by_value,
              "C1 scorer() returns the scorer passed to the constructor (also after a move); "
              "cached_scorer() is another object");
    }
}

// ---- Direct partition-MCMC step (no stored neighbourhood) ------------------
// The moves, probabilities and generator draws of partition_mcmc_step. The
// single-node neighbourhood is enumerated at every single-node move, and the
// edge reversal scores families with score_cache::scorer().
struct ref_chain {
    partition_state state;
    double          log_score = 0.0;
    double          sj_nbhd   = 0.0;
};

ref_chain make_ref(const score_cache& cache, const partition_state& s) {
    ref_chain r;
    r.state = s;
    r.log_score = partition_log_score(cache, s);
    r.sj_nbhd = splitjoin_nbhd(s);
    return r;
}

void ref_edge_reversal_step(ref_chain& c, const score_cache& cache, std::mt19937_64& rng) {
    const std::size_t n = c.state.n();
    const std::vector<std::uint64_t> G = partition_sample_dag(cache, rng, c.state);
    std::vector<std::pair<std::size_t, std::size_t>> arcs;
    for (std::size_t b = 0; b < n; ++b)
        for (std::size_t a = 0; a < n; ++a)
            if ((G[b] & (1ULL << a)) && is_reversible(G, a, b, n)) arcs.emplace_back(a, b);
    if (arcs.empty()) return;
    std::uniform_int_distribution<std::size_t> pick(0, arcs.size() - 1);
    const auto ab = arcs[pick(rng)];
    const std::size_t a = ab.first, b = ab.second;
    std::vector<std::uint64_t> Gp = G;
    Gp[b] &= ~(1ULL << a);
    Gp[a] |= (1ULL << b);
    const i_scorer& sc = cache.scorer();
    const double delta = (sc.family_score(a, Gp[a]) + sc.family_score(b, Gp[b]))
                       - (sc.family_score(a, G[a]) + sc.family_score(b, G[b]));
    const double log_acc = delta + std::log(static_cast<double>(arcs.size()))
                         - std::log(static_cast<double>(count_reversible(Gp, n)));
    std::uniform_real_distribution<double> U(0.0, 1.0);
    if (std::log(U(rng)) < log_acc) {
        c.state = dag_to_partition(Gp, n);
        c.log_score = partition_log_score(cache, c.state);
        c.sj_nbhd = splitjoin_nbhd(c.state);
    }
}

void ref_partition_mcmc_step(ref_chain& c, const score_cache& cache, std::mt19937_64& rng) {
    const double p_stay = 0.01, p_edgerev = 0.07, p_splitjoin = 0.37, p_swap = 0.27;
    if (c.state.permy.size() <= 1) return;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double u = U(rng);
    if (u < p_stay) return;
    if (u < p_stay + p_edgerev) { ref_edge_reversal_step(c, cache, rng); return; }
    if (u < p_stay + p_edgerev + p_splitjoin) {
        partition_state prop = propose_splitjoin(rng, c.state);
        const double ls_p = partition_log_score(cache, prop);
        const double nbd_p = splitjoin_nbhd(prop);
        if (std::log(U(rng)) < (ls_p - c.log_score) + std::log(c.sj_nbhd) - std::log(nbd_p)) {
            c.state = std::move(prop); c.log_score = ls_p; c.sj_nbhd = nbd_p;
        }
    } else if (u < p_stay + p_edgerev + p_splitjoin + p_swap) {
        partition_state prop = propose_swap_any(rng, c.state);
        const double ls_p = partition_log_score(cache, prop);
        if (std::log(U(rng)) < (ls_p - c.log_score)) {
            c.state = std::move(prop); c.log_score = ls_p; c.sj_nbhd = splitjoin_nbhd(c.state);
        }
    } else {
        const std::vector<partition_state> nbrs = single_node_neighbours(c.state);
        if (nbrs.empty()) return;
        std::uniform_int_distribution<std::size_t> pick(0, nbrs.size() - 1);
        partition_state prop = nbrs[pick(rng)];
        const double ls_p = partition_log_score(cache, prop);
        const std::size_t nbd_p = single_node_neighbours(prop).size();
        if (std::log(U(rng)) < (ls_p - c.log_score)
                + std::log(static_cast<double>(nbrs.size()))
                - std::log(static_cast<double>(nbd_p))) {
            c.state = std::move(prop); c.log_score = ls_p; c.sj_nbhd = splitjoin_nbhd(c.state);
        }
    }
}

// ---- C2: partition-mode block against the direct step -----------------------
order_mcmc_block_config block_cfg(std::size_t n, bool gauss, bool partition, std::size_t k) {
    order_mcmc_block_config cfg;
    cfg.name = "order";
    if (gauss) {
        cfg.continuous_data = sim_gauss(400, n, 100 + n);
    } else {
        cfg.data = sim_chain(1500, n, 0.9, 200 + n);
        cfg.cardinalities = arma::uvec(n, arma::fill::value(2));
    }
    cfg.structure_prior = gauss ? dag_prior::UNIFORM : dag_prior::FK_EQ2;
    cfg.max_parents = k;
    cfg.init_rng_seed = 9;
    cfg.method = partition ? order_mcmc_block_config::method_t::partition
                           : order_mcmc_block_config::method_t::order;
    return cfg;
}

// Composite sweeps of the block vs the direct step with the same generator
// draws (step: one mixture step and one DAG draw; outputs: one DAG draw).
long compare_partition_sweeps(composite_block& comp, const order_mcmc_block& blk,
                              ref_chain& ref, std::mt19937_64& rng, int sweeps) {
    std::mt19937_64 rr = rng;
    long bad = 0;
    for (int s = 0; s < sweeps; ++s) {
        comp.step(rng);
        ref_partition_mcmc_step(ref, blk.cache(), rr);
        const auto d1 = partition_sample_dag(blk.cache(), rr, ref.state);
        const auto d2 = partition_sample_dag(blk.cache(), rr, ref.state);
        const arma::vec& out_dag = comp.data().get("order_sampled_DAG");
        bool dag2_ok = true;
        const std::size_t n = ref.state.n();
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if ((out_dag[i * n + j] != 0.0) != ((d2[i] >> j) & 1ULL)) dag2_ok = false;
        if (blk.partition().permy != ref.state.permy || blk.partition().party != ref.state.party
            || !same_bits(blk.current_log_score(), ref.log_score)
            || blk.sampled_dag() != d1 || !dag2_ok) ++bad;
    }
    if (rng() != rr()) ++bad;
    return bad;
}

void C2_partition_block() {
    std::printf("\n--- C2: partition-mode block equals the direct partition-MCMC step ---\n");
    struct cs { std::size_t n; bool gauss; std::size_t k; int sweeps; };
    const cs cases[] = {{4, false, 3, 3000}, {6, false, 4, 2000}, {8, false, 3, 800},
                        {5, true, 4, 2000}, {7, true, 3, 800}};
    for (const auto& c : cases) {
        composite_block comp("bn");
        comp.add_child(std::make_unique<order_mcmc_block>(block_cfg(c.n, c.gauss, true, c.k)));
        auto* blk = dynamic_cast<order_mcmc_block*>(&comp.child(0));
        std::mt19937_64 rng(1234 + c.n);
        ref_chain ref = make_ref(blk->cache(), blk->partition());
        long bad = compare_partition_sweeps(comp, *blk, ref, rng, c.sweeps);
        // set_current changes nothing: a round trip returns (it throws on the
        // one-element partition) and the reversed sequence throws; the chain
        // continues as ref
        bool threw = false;
        try { blk->set_current(blk->current()); }
        catch (const std::invalid_argument&) { threw = true; }
        if (threw != (blk->partition().m() == 1)) ++bad;
        arma::vec v(c.n);
        for (std::size_t q = 0; q < c.n; ++q) v[q] = blk->current()[c.n - 1 - q];
        threw = false;
        try { blk->set_current(v); } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) ++bad;
        bad += compare_partition_sweeps(comp, *blk, ref, rng, c.sweeps / 4);
        // set_partition: element sizes reversed
        const partition_state cur = blk->partition();
        arma::uvec permy(c.n), party(cur.party.size());
        for (std::size_t q = 0; q < c.n; ++q) permy[q] = cur.permy[q];
        for (std::size_t t = 0; t < cur.party.size(); ++t) party[t] = cur.party[cur.party.size() - 1 - t];
        blk->set_partition(permy, party);
        partition_state expect;
        expect.permy = cur.permy;
        expect.party.assign(party.begin(), party.end());
        canonicalize(expect);
        const bool loaded = (blk->partition().permy == expect.permy
                             && blk->partition().party == expect.party
                             && same_bits(blk->current_log_score(),
                                          partition_log_score(blk->cache(), expect)));
        ref = make_ref(blk->cache(), expect);
        bad += compare_partition_sweeps(comp, *blk, ref, rng, c.sweeps / 4);
        check(bad == 0 && loaded,
              std::string("C2 ") + (c.gauss ? "BGe" : "BDeu") + " n=" + std::to_string(c.n)
              + " k=" + std::to_string(c.k) + ": sweeps, set_current, set_partition",
              "mismatching sweeps = " + std::to_string(bad));
    }
}

// ---- C3: order-mode block against the direct order-MCMC step ------------------
void C3_order_block() {
    std::printf("\n--- C3: order-mode block equals the direct order-MCMC step ---\n");
    struct cs { std::size_t n; bool gauss; std::size_t k; int sweeps; };
    const cs cases[] = {{6, false, 3, 3000}, {10, false, 4, 1000}, {7, true, 3, 1500}};
    for (const auto& c : cases) {
        auto cfg = block_cfg(c.n, c.gauss, false, c.k);
        cfg.prob_adjacent_swap = 0.4;
        order_mcmc_block blk(cfg);
        // The raw scorer for the direct computations.
        std::unique_ptr<i_scorer> raw;
        if (c.gauss) { bge_scorer_config b; b.data = cfg.continuous_data; raw = std::make_unique<bge_scorer>(b); }
        else {
            bde_scorer_config b; b.data = cfg.data; b.cardinalities = cfg.cardinalities;
            b.use_structure_prior = true; b.max_parents = c.k;
            raw = std::make_unique<bde_scorer>(b);
        }
        counters k;
        std::vector<std::size_t> order(blk.order().begin(), blk.order().end());
        double ls = ref_order_log_score(blk.cache(), *raw, order, k);
        long bad = same_bits(ls, blk.current_log_score()) ? 0 : 1;
        std::mt19937_64 rng(77 + c.n), rr(77 + c.n);
        block_context ctx;
        const std::size_t n = c.n;
        for (int s = 0; s < c.sweeps; ++s) {
            if (s == c.sweeps / 2) {   // set_current mid-run
                arma::vec v(n);
                for (std::size_t q = 0; q < n; ++q) v[q] = static_cast<double>(order[(q + 2) % n]);
                blk.set_current(v);
                for (std::size_t q = 0; q < n; ++q) order[q] = static_cast<std::size_t>(v[q]);
                ls = ref_order_log_score(blk.cache(), *raw, order, k);
                if (!same_bits(ls, blk.current_log_score())) ++bad;
            }
            blk.set_context(ctx);
            blk.step(rng);
            // direct step
            std::uniform_real_distribution<double> U01(0.0, 1.0);
            std::size_t a = 0, b = 0;
            if (U01(rr) < cfg.prob_adjacent_swap) {
                std::uniform_int_distribution<std::size_t> Up(0, n - 2);
                a = Up(rr); b = a + 1;
            } else {
                std::uniform_int_distribution<std::size_t> Up(0, n - 1);
                a = Up(rr);
                do { b = Up(rr); } while (b == a);
                if (a > b) std::swap(a, b);
            }
            std::swap(order[a], order[b]);
            const double prop = ref_order_log_score(blk.cache(), *raw, order, k);
            if (std::log(U01(rr)) < prop - ls) ls = prop;
            else std::swap(order[a], order[b]);
            std::vector<std::uint64_t> dag(n);
            for (std::size_t i = 0; i < n; ++i) dag[i] = ref_sample_parent_set(rr, blk.cache(), order, i, k);
            const std::vector<std::size_t> got(blk.order().begin(), blk.order().end());
            if (got != order || !same_bits(blk.current_log_score(), ls) || blk.sampled_dag() != dag) ++bad;
            // current_named_outputs draws a second DAG
            const auto out = blk.current_named_outputs(rng);
            std::vector<std::uint64_t> dag2(n);
            for (std::size_t i = 0; i < n; ++i) dag2[i] = ref_sample_parent_set(rr, blk.cache(), order, i, k);
            const arma::vec& od = out.at("order_sampled_DAG");
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    if ((od[i * n + j] != 0.0) != ((dag2[i] >> j) & 1ULL)) { ++bad; i = n; break; }
        }
        if (rng() != rr()) ++bad;
        check(bad == 0, std::string("C3 ") + (c.gauss ? "BGe" : "BDeu") + " n=" + std::to_string(c.n)
              + ": sweeps and set_current", "mismatches = " + std::to_string(bad));
    }
}

// 1: std::invalid_argument, 2: another std::logic_error, 0: no exception.
template <class F>
int thrown_kind(F&& f) {
    try { f(); }
    catch (const std::invalid_argument&) { return 1; }
    catch (const std::logic_error&) { return 2; }
    return 0;
}

void C4_set_partition_validation() {
    std::printf("\n--- C4: set_partition input checks ---\n");
    order_mcmc_block blk(block_cfg(4, false, true, 3));
    auto rejects = [&](const arma::uvec& p, const arma::uvec& s) {
        return thrown_kind([&] { blk.set_partition(p, s); }) == 1;
    };
    const arma::vec before = blk.current();
    const double ls_before = blk.current_log_score();
    const auto dag_before = blk.sampled_dag();
    const arma::uword big = std::numeric_limits<arma::uword>::max();
    check(rejects(arma::uvec{0, 1, 2}, arma::uvec{3}), "C4 wrong length rejected");
    check(rejects(arma::uvec{0, 1, 1, 3}, arma::uvec{4}), "C4 repeated node rejected");
    check(rejects(arma::uvec{0, 1, 2, 4}, arma::uvec{4}), "C4 node out of range rejected");
    check(rejects(arma::uvec{0, 1, 2, 3}, arma::uvec{2, 0, 2}), "C4 empty element rejected");
    check(rejects(arma::uvec{0, 1, 2, 3}, arma::uvec{2, 1}), "C4 sizes not summing to n rejected");
    check(rejects(arma::uvec{0, 1, 2, 3}, arma::uvec()), "C4 no elements rejected");
    check(rejects(arma::uvec{3, 2, 1, 0}, arma::uvec{big, 5}),
          "C4 sizes whose sum wraps to n rejected (max, 5)");
    check(rejects(arma::uvec{3, 2, 1, 0}, arma::uvec{big, big, 6}),
          "C4 sizes whose sum wraps to n rejected (max, max, 6)");
    check(arma::all(blk.current() == before) && blk.current_log_score() == ls_before
          && blk.sampled_dag() == dag_before,
          "C4 rejected input leaves the state unchanged");
    // max_parents = 0: a node outside the last element has no permitted
    // parent set, so every partition with two or more elements has log-score
    // -inf; the one-element partition is accepted.
    order_mcmc_block b0(block_cfg(4, false, true, 0));
    const double ls0 = b0.current_log_score();
    const bool inf2 = thrown_kind([&] {
        b0.set_partition(arma::uvec{0, 1, 2, 3}, arma::uvec{2, 2}); }) == 1;
    const bool inf4 = thrown_kind([&] {
        b0.set_partition(arma::uvec{3, 1, 2, 0}, arma::uvec{1, 1, 1, 1}); }) == 1;
    check(inf2 && inf4 && b0.current_log_score() == ls0 && b0.partition().m() == 1,
          "C4 partitions with log-score -inf rejected (max_parents = 0), state unchanged");
    check(thrown_kind([&] { b0.set_partition(arma::uvec{3, 1, 2, 0}, arma::uvec{4}); }) == 0
          && b0.partition().m() == 1 && b0.current_log_score() == ls0,
          "C4 the one-element partition accepted (max_parents = 0)");
    check(thrown_kind([&] { blk.set_partition(arma::uvec{0, 1, 2, 3}, arma::uvec{2, 2}); }) == 0
          && blk.partition().m() == 2 && std::isfinite(blk.current_log_score()),
          "C4 the same two-element partition accepted (max_parents = 3)");
    // method = order: std::logic_error that is not std::invalid_argument,
    // also for input that set_partition would reject in method = partition.
    order_mcmc_block ob(block_cfg(4, false, false, 3));
    check(thrown_kind([&] { ob.set_partition(arma::uvec{0, 1, 2, 3}, arma::uvec{4}); }) == 2
          && thrown_kind([&] { ob.set_partition(arma::uvec{0, 1}, arma::uvec{2, 0}); }) == 2,
          "C4 set_partition in method = order throws std::logic_error, not std::invalid_argument");
}

// ---- C5: partition_mcmc_step (stored neighbourhood) against the direct step ----
bool same_chain(const partition_chain& c, const ref_chain& r) {
    return c.state.permy == r.state.permy && c.state.party == r.state.party
        && same_bits(c.log_score, r.log_score) && same_bits(c.sj_nbhd, r.sj_nbhd);
}

// A partition with the same node sequence and other element sizes: all
// singletons (every sequence is sorted within singleton elements), or, when
// s already has singleton elements only, the first adjacent ascending pair
// merged; s itself when the sequence is descending.
partition_state other_sizes(const partition_state& s) {
    partition_state r;
    r.permy = s.permy;
    if (s.m() != s.n()) {
        r.party.assign(s.n(), 1);
        return r;
    }
    for (std::size_t p = 0; p + 1 < s.n(); ++p) {
        if (s.permy[p] < s.permy[p + 1]) {
            r.party.assign(s.n() - 1, 1);
            r.party[p] = 2;
            return r;
        }
    }
    return s;
}

void C5_partition_step_store() {
    std::printf("\n--- C5: partition_mcmc_step equals the direct step (assignments, copies, moves) ---\n");
    struct cs { std::size_t n; bool gauss; std::size_t k; int steps; };
    const cs cases[] = {{6, false, 3, 6000}, {5, false, 1, 6000}, {7, true, 3, 3000}};
    for (const auto& cc : cases) {
        const order_mcmc_block_config cfg = block_cfg(cc.n, cc.gauss, true, cc.k);
        const order_mcmc_block blk(cfg);          // its cache is the partition-mode cache
        const score_cache& cache = blk.cache();
        partition_chain c = partition_chain_init(cache, trivial_partition(cc.n));
        ref_chain r = make_ref(cache, trivial_partition(cc.n));
        std::mt19937_64 g1(4321 + cc.n), g2(4321 + cc.n), choose(99 + cc.n);
        long bad = 0, assigned = 0, resized = 0, revisits = 0, copies = 0, moves = 0;
        partition_state earlier = c.state;
        auto assign = [&](const partition_state& p) {
            const double ls = partition_log_score(cache, p);
            if (!(ls > -kInf)) return false;
            c.state = p; c.log_score = ls; c.sj_nbhd = splitjoin_nbhd(p);
            r = make_ref(cache, p);
            return true;
        };
        for (int s = 0; s < cc.steps; ++s) {
            partition_mcmc_step(c, cache, g1);
            ref_partition_mcmc_step(r, cache, g2);
            if (!same_chain(c, r)) ++bad;
            if (s % 400 == 100) earlier = c.state;
            if (s % 97 == 41 && assign(rand_partition(choose, cc.n))) ++assigned;
            if (s % 89 == 13 && assign(other_sizes(c.state))) ++resized;
            if (s % 211 == 77 && assign(earlier)) ++revisits;
            if (s % 59 == 29) {                    // continue with a copy
                partition_chain copy(c);
                c = partition_chain();
                c = copy;
                ++copies;
            }
            if (s % 131 == 61) {                   // continue with a moved-to chain
                partition_chain moved(std::move(c));
                c = std::move(moved);
                ++moves;
            }
        }
        if (g1() != g2()) ++bad;
        check(bad == 0 && assigned > 0 && resized > 0 && revisits > 0,
              std::string("C5 ") + (cc.gauss ? "BGe" : "BDeu") + " n=" + std::to_string(cc.n)
              + " k=" + std::to_string(cc.k) + ": every step equals the direct step",
              "mismatches = " + std::to_string(bad) + ", assignments = " + std::to_string(assigned)
              + ", other sizes = " + std::to_string(resized) + ", earlier partitions = "
              + std::to_string(revisits) + ", copies = " + std::to_string(copies)
              + ", moves = " + std::to_string(moves));
    }
}
}  // namespace

int main() {
    std::printf("=== test_fix_order-mcmc_cache ===\n");
    C1_cache_queries();
    C2_partition_block();
    C3_order_block();
    C4_set_partition_validation();
    C5_partition_step_store();
    std::printf("\n=== %d PASS / %d FAIL ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

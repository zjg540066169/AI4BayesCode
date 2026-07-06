/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v3.0 or later
 *  (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  partition_state.hpp -- labelled-partition state + geometry for Kuipers-Moffa
 *                         2017 partition MCMC. Used by order_mcmc_block in
 *                         method="partition" mode.
 *
 *  A labelled ordered partition of the n nodes:
 *      permy  : node labels, concatenated element-by-element (element 0 first).
 *      party  : element sizes [k_0, ..., k_{m-1}], a composition of n.
 *
 *  GEOMETRY (edges point strictly RIGHT-to-LEFT in element order)
 *  =============================================================
 *  A node in element t may take parents only from elements STRICTLY to the
 *  right (t+1 .. m-1), and MUST take at least one parent from the immediately-
 *  adjacent right element (t+1). Nodes in the last element (m-1) have no
 *  element to their right => they are pure sources (no parents). This makes the
 *  DAG->partition map a bijection (outpoint peeling): each DAG maps to exactly
 *  ONE labelled partition, which is why partition MCMC is unbiased.
 *
 *  OUTPOINT PEELING (dag_to_partition)
 *  ===================================
 *  With edges j -> i meaning j is a parent of i, define the right-depth
 *      r(i) = 0                         if i has no parents (a source),
 *           = 1 + max_{j in Pa(i)} r(j) otherwise.
 *  Then m = max_i r(i) + 1 elements, and element(i) = (m-1) - r(i). The parent
 *  achieving the max lies in element(i)+1, so the adjacent-right constraint
 *  holds automatically; sources (r=0) land in the last element (m-1).
 *================================================================================*/

#ifndef AI4BAYESCODE_PARTITION_STATE_HPP
#define AI4BAYESCODE_PARTITION_STATE_HPP

#include "score_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace AI4BayesCode {

/// A labelled ordered partition of n nodes.
struct partition_state {
    std::vector<std::size_t> permy;   ///< node labels, element-0 first
    std::vector<std::size_t> party;   ///< element sizes (composition of n)

    std::size_t n() const noexcept { return permy.size(); }
    std::size_t m() const noexcept { return party.size(); }
};

/// Cumulative element-start positions: elem_start[t] = sum_{i<t} party[i];
/// length m+1 with elem_start[m] = n. So element t occupies positions
/// [elem_start[t], elem_start[t+1]).
inline std::vector<std::size_t> element_starts(const partition_state& s) {
    std::vector<std::size_t> es(s.m() + 1, 0);
    for (std::size_t t = 0; t < s.m(); ++t) es[t + 1] = es[t] + s.party[t];
    return es;
}

/// The allowed-parent mask (nodes in elements strictly right of `elem`) and the
/// required mask (nodes in the adjacent right element elem+1) for a node whose
/// partition element is `elem`. required == 0 for the last element.
struct node_masks {
    std::uint64_t allowed  = 0;
    std::uint64_t required = 0;
};

inline node_masks masks_for_element(const partition_state& s,
                                    const std::vector<std::size_t>& es,
                                    std::size_t elem) {
    node_masks nm;
    const std::size_t n = s.n();
    // allowed = all nodes in positions es[elem+1] .. n-1
    for (std::size_t p = es[elem + 1]; p < n; ++p)
        nm.allowed |= (1ULL << s.permy[p]);
    // required = nodes in element elem+1 (positions es[elem+1] .. es[elem+2]-1)
    if (elem + 1 < s.m()) {
        for (std::size_t p = es[elem + 1]; p < es[elem + 2]; ++p)
            nm.required |= (1ULL << s.permy[p]);
    }
    return nm;
}

/// Canonicalise: sort node labels WITHIN each element. Within-element order is
/// irrelevant to a labelled partition (it does not change the consistent-DAG
/// set, the score, or #nbd), so the chain state MUST be canonical — otherwise
/// each partition is represented by Π k_t! permy orderings and its stationary
/// mass is inflated by that multiplicity (a systematic bias toward
/// large-element partitions). Every move canonicalises its result.
inline void canonicalize(partition_state& s) {
    const auto es = element_starts(s);
    for (std::size_t t = 0; t < s.m(); ++t)
        std::sort(s.permy.begin() + es[t], s.permy.begin() + es[t + 1]);
}

/// element index (0-based) of every POSITION in permy.
inline std::vector<std::size_t> element_of_position(const partition_state& s) {
    std::vector<std::size_t> pos_elem(s.n(), 0);
    std::size_t p = 0;
    for (std::size_t t = 0; t < s.m(); ++t)
        for (std::size_t c = 0; c < s.party[t]; ++c) pos_elem[p++] = t;
    return pos_elem;
}

/// Partition log-score, Kuipers-Moffa Eq. 3:
///   log P(Lambda | D) = sum_i log( sum_{U permissible for i} score(X_i, U|D) ).
inline double partition_log_score(const score_cache& cache,
                                  const partition_state& s) {
    const auto es = element_starts(s);
    double total = 0.0;
    for (std::size_t t = 0; t < s.m(); ++t) {
        const node_masks nm = masks_for_element(s, es, t);
        for (std::size_t p = es[t]; p < es[t + 1]; ++p) {
            const std::size_t node = s.permy[p];
            total += cache.partition_node_score(node, nm.allowed, nm.required);
        }
    }
    return total;
}

/// Sample a DAG (parent bitmask per node) consistent with the partition,
/// drawing each node's parent set independently proportional to its score.
inline std::vector<std::uint64_t> partition_sample_dag(
    const score_cache& cache, std::mt19937_64& rng, const partition_state& s) {
    const auto es = element_starts(s);
    std::vector<std::uint64_t> dag(s.n(), 0ULL);
    for (std::size_t t = 0; t < s.m(); ++t) {
        const node_masks nm = masks_for_element(s, es, t);
        for (std::size_t p = es[t]; p < es[t + 1]; ++p) {
            const std::size_t node = s.permy[p];
            dag[node] = cache.partition_sample_parent_set(
                rng, node, nm.allowed, nm.required);
        }
    }
    return dag;
}

/// Outpoint-peel a DAG (parent bitmask per node) into its unique labelled
/// partition. Within each element the node labels are sorted (canonical form).
inline partition_state dag_to_partition(const std::vector<std::uint64_t>& dag,
                                        std::size_t n) {
    // right-depth r(i): longest path length from a source to i.
    std::vector<long> r(n, -1);
    std::function<long(std::size_t)> depth = [&](std::size_t i) -> long {
        if (r[i] >= 0) return r[i];
        long best = 0;
        bool has_parent = false;
        for (std::size_t j = 0; j < n; ++j) {
            if (dag[i] & (1ULL << j)) {
                has_parent = true;
                best = std::max(best, depth(j) + 1);
            }
        }
        r[i] = has_parent ? best : 0;
        return r[i];
    };
    long maxr = 0;
    for (std::size_t i = 0; i < n; ++i) maxr = std::max(maxr, depth(i));
    const std::size_t m = static_cast<std::size_t>(maxr) + 1;

    // element(i) = (m-1) - r(i); group + canonicalise.
    std::vector<std::vector<std::size_t>> buckets(m);
    for (std::size_t i = 0; i < n; ++i)
        buckets[m - 1 - static_cast<std::size_t>(r[i])].push_back(i);

    partition_state s;
    s.party.resize(m);
    for (std::size_t t = 0; t < m; ++t) {
        std::sort(buckets[t].begin(), buckets[t].end());
        s.party[t] = buckets[t].size();
        for (std::size_t node : buckets[t]) s.permy.push_back(node);
    }
    return s;
}

/// Split/join move neighbourhood size (Kuipers-Moffa §4.4 / Alg. 1):
///   #nbd = (m-1) joins + sum_i (2^{k_i} - 2) splits = sum_i 2^{k_i} - m - 1.
inline double splitjoin_nbhd(const partition_state& s) {
    double nb = 0.0;
    for (std::size_t k : s.party) nb += std::exp2(static_cast<double>(k));
    return nb - static_cast<double>(s.m()) - 1.0;
}

/// Propose a split-or-join neighbour (Kuipers-Moffa Algorithm 1). Draw an
/// integer j uniformly in 1..#nbd: j in 1..m-1 JOINS elements (j-1, j); else it
/// SPLITS one element, peeling a proper non-empty subset of its nodes off into
/// a NEW element placed immediately to the LEFT. Split and join are exact
/// inverses, so the pairing is reversible. The Hastings correction
/// #nbd(Lambda)/#nbd(Lambda') is applied by the caller (both are recomputed).
inline partition_state propose_splitjoin(std::mt19937_64& rng,
                                         const partition_state& s) {
    const std::size_t m = s.m();
    const auto es = element_starts(s);
    const double nbd = splitjoin_nbhd(s);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    // j uniform in {1, ..., #nbd}
    std::size_t j = 1 + static_cast<std::size_t>(U01(rng) * nbd);
    if (j > static_cast<std::size_t>(nbd)) j = static_cast<std::size_t>(nbd);

    partition_state r;
    if (j <= m - 1) {
        // JOIN elements b and b+1 (0-indexed b = j-1). permy is unchanged
        // (the two elements are already adjacent); merge their sizes.
        const std::size_t b = j - 1;
        r.permy = s.permy;
        r.party.reserve(m - 1);
        for (std::size_t t = 0; t < m; ++t) {
            if (t == b) { r.party.push_back(s.party[b] + s.party[b + 1]); }
            else if (t == b + 1) { /* absorbed */ }
            else { r.party.push_back(s.party[t]); }
        }
        canonicalize(r);
        return r;
    }

    // SPLIT: locate element t and the jj-th proper non-empty subset of its nodes.
    std::size_t jj = j - (m - 1);   // 1 .. sum_i (2^{k_i}-2)
    std::size_t t = 0;
    for (; t < m; ++t) {
        const std::size_t kt = s.party[t];
        const std::size_t opts = (kt >= 2) ? ((std::size_t{1} << kt) - 2) : 0;
        if (jj <= opts) break;
        jj -= opts;
    }
    // local subset mask over element t's k_t nodes == jj (jj in 1..2^{k_t}-2).
    const std::size_t kt = s.party[t];
    const std::uint64_t local = static_cast<std::uint64_t>(jj);
    std::vector<std::size_t> left, right;   // split-off (new, left) vs remainder
    for (std::size_t c = 0; c < kt; ++c) {
        const std::size_t node = s.permy[es[t] + c];
        if (local & (1ULL << c)) left.push_back(node);
        else                     right.push_back(node);
    }
    // rebuild permy: elements 0..t-1, then [left] (new element), then [right]
    // (old element t shrunk), then elements t+1..m-1.
    r.permy.reserve(s.n());
    for (std::size_t p = 0; p < es[t]; ++p) r.permy.push_back(s.permy[p]);
    for (std::size_t x : left)  r.permy.push_back(x);
    for (std::size_t x : right) r.permy.push_back(x);
    for (std::size_t p = es[t + 1]; p < s.n(); ++p) r.permy.push_back(s.permy[p]);
    r.party.reserve(m + 1);
    for (std::size_t u = 0; u < t; ++u) r.party.push_back(s.party[u]);
    r.party.push_back(left.size());
    r.party.push_back(right.size());
    for (std::size_t u = t + 1; u < m; ++u) r.party.push_back(s.party[u]);
    canonicalize(r);
    return r;
}

/// Symmetric swap move (Kuipers-Moffa §4.7 variant C1): swap the node labels at
/// two distinct, uniformly-chosen positions of permy. The neighbourhood is
/// n(n-1)/2, CONSTANT (independent of the partition), so the proposal is
/// symmetric and needs NO Hastings term. A swap inside one element is a null
/// move (canonicalised away). This move changes which nodes sit in which element
/// at FIXED element sizes -- the complement of split/join (which changes sizes)
/// -- and is what makes the mixture mix well.
inline partition_state propose_swap_any(std::mt19937_64& rng,
                                        const partition_state& s) {
    const std::size_t n = s.n();
    std::uniform_int_distribution<std::size_t> P(0, n - 1);
    std::size_t a = P(rng), b = P(rng);
    while (b == a) b = P(rng);
    partition_state r = s;
    std::swap(r.permy[a], r.permy[b]);
    canonicalize(r);
    return r;
}

/// The single-element partition (empty DAG): all nodes in one element.
inline partition_state trivial_partition(std::size_t n) {
    partition_state s;
    s.permy.resize(n);
    for (std::size_t i = 0; i < n; ++i) s.permy[i] = i;
    s.party = {n};
    return s;
}

/// Canonical string key of a partition (canonicalised element contents +
/// sizes). Used to deduplicate single-node neighbours.
inline std::string canon_key(const partition_state& s) {
    partition_state c = s; canonicalize(c);
    std::string k; std::size_t off = 0;
    for (std::size_t t = 0; t < c.m(); ++t) {
        for (std::size_t x = 0; x < c.party[t]; ++x) {
            k += std::to_string(c.permy[off + x]); k += ',';
        }
        k += '|'; off += c.party[t];
    }
    return k;
}

/// All DISTINCT partitions reachable by relocating exactly ONE node to a
/// different element or into a new singleton element (Kuipers-Moffa §4.6 move,
/// enumerated exactly). The original partition is excluded. Enumerating the
/// neighbourhood (rather than a closed-form count) makes the Hastings ratio
/// #nbd(Lambda)/#nbd(Lambda') trivially correct: propose one uniformly, and
/// #nbd of each state is just the size of its enumerated neighbourhood.
inline std::vector<partition_state>
single_node_neighbours(const partition_state& s) {
    const std::string orig = canon_key(s);
    std::set<std::string> seen;
    std::vector<partition_state> out;
    // element contents of s
    std::vector<std::vector<std::size_t>> S(s.m());
    { std::size_t off = 0;
      for (std::size_t t = 0; t < s.m(); ++t) {
          for (std::size_t c = 0; c < s.party[t]; ++c) S[t].push_back(s.permy[off + c]);
          off += s.party[t]; } }

    for (std::size_t p = 0; p < s.n(); ++p) {
        const std::size_t v = s.permy[p];
        // base = element contents with v removed (drop any now-empty element)
        std::vector<std::vector<std::size_t>> base;
        for (const auto& el : S) {
            std::vector<std::size_t> e2;
            for (std::size_t x : el) if (x != v) e2.push_back(x);
            if (!e2.empty()) base.push_back(std::move(e2));
        }
        const std::size_t mb = base.size();
        auto emit = [&](const std::vector<std::vector<std::size_t>>& els) {
            partition_state r;
            for (const auto& el : els) {
                r.party.push_back(el.size());
                for (std::size_t x : el) r.permy.push_back(x);
            }
            canonicalize(r);
            const std::string k = canon_key(r);
            if (k != orig && !seen.count(k)) { seen.insert(k); out.push_back(r); }
        };
        // (a) insert v into each existing base element
        for (std::size_t d = 0; d < mb; ++d) {
            auto els = base; els[d].push_back(v); emit(els);
        }
        // (b) insert v as a NEW singleton element at each gap 0..mb
        for (std::size_t g = 0; g <= mb; ++g) {
            std::vector<std::vector<std::size_t>> els;
            for (std::size_t t = 0; t <= mb; ++t) {
                if (t == g) els.push_back({v});
                if (t < mb) els.push_back(base[t]);
            }
            emit(els);
        }
    }
    return out;
}

/// A partition-MCMC chain: the current state + cached log-score + split/join
/// neighbourhood size (kept in sync so each step is cheap).
struct partition_chain {
    partition_state state;
    double          log_score = 0.0;
    double          sj_nbhd   = 0.0;
};

inline partition_chain partition_chain_init(const score_cache& cache,
                                            const partition_state& s0) {
    partition_chain c;
    c.state     = s0;
    c.log_score = partition_log_score(cache, s0);
    c.sj_nbhd   = splitjoin_nbhd(s0);
    return c;
}

/// Can node `from` reach node `to` following child edges of the DAG (parent
/// masks: node y is a child of x iff dag[y] has bit x)?
inline bool dag_reachable(const std::vector<std::uint64_t>& dag,
                          std::size_t from, std::size_t to, std::size_t n) {
    std::vector<bool> vis(n, false);
    std::vector<std::size_t> stk{from}; vis[from] = true;
    while (!stk.empty()) {
        const std::size_t x = stk.back(); stk.pop_back();
        for (std::size_t y = 0; y < n; ++y)
            if (!vis[y] && (dag[y] & (1ULL << x))) {
                if (y == to) return true;
                vis[y] = true; stk.push_back(y);
            }
    }
    return false;
}

/// Is arc a->b (a is a parent of b) reversible, i.e. does removing a->b and
/// adding b->a keep the DAG acyclic? Reversible iff, with a->b removed, a can no
/// longer reach b (otherwise b->a would close a cycle).
inline bool is_reversible(const std::vector<std::uint64_t>& dag,
                          std::size_t a, std::size_t b, std::size_t n) {
    std::vector<std::uint64_t> d = dag;
    d[b] &= ~(1ULL << a);
    return !dag_reachable(d, a, b, n);
}

/// Count the reversible arcs of a DAG.
inline std::size_t count_reversible(const std::vector<std::uint64_t>& dag,
                                    std::size_t n) {
    std::size_t c = 0;
    for (std::size_t b = 0; b < n; ++b)
        for (std::size_t a = 0; a < n; ++a)
            if ((dag[b] & (1ULL << a)) && is_reversible(dag, a, b, n)) ++c;
    return c;
}

/// Edge-reversal step (Kuipers-Moffa 2017 Sec.5 wrapper around a single-edge
/// structure-MCMC reversal): sample a DAG G from the current partition BY SCORE,
/// reverse one reversible arc, accept on the DAG-posterior ratio times the
/// reversible-arc-count ratio, and map the accepted DAG back to its partition.
/// Because G is sampled from Lambda by score, the partition-score ratio cancels
/// (Eqs 5-6, p.16) -- the acceptance is the plain DAG reversal ratio, no
/// partition factor. This specifically mixes EDGE DIRECTIONS within a Markov
/// equivalence class, which the split/join/swap/single-node moves do slowly.
inline void partition_edge_reversal_step(partition_chain& c,
                                         const score_cache& cache,
                                         std::mt19937_64& rng) {
    const std::size_t n = c.state.n();
    const std::vector<std::uint64_t> G = partition_sample_dag(cache, rng, c.state);
    // collect reversible arcs a->b
    std::vector<std::pair<std::size_t, std::size_t>> arcs;
    for (std::size_t b = 0; b < n; ++b)
        for (std::size_t a = 0; a < n; ++a)
            if ((G[b] & (1ULL << a)) && is_reversible(G, a, b, n))
                arcs.emplace_back(a, b);
    if (arcs.empty()) return;

    std::uniform_int_distribution<std::size_t> pick(0, arcs.size() - 1);
    const auto [a, b] = arcs[pick(rng)];
    std::vector<std::uint64_t> Gp = G;
    Gp[b] &= ~(1ULL << a);      // remove a->b
    Gp[a] |=  (1ULL << b);      // add    b->a

    const i_scorer& sc = cache.scorer();
    const double delta = (sc.family_score(a, Gp[a]) + sc.family_score(b, Gp[b]))
                       - (sc.family_score(a, G[a])  + sc.family_score(b, G[b]));
    const std::size_t RG  = arcs.size();
    const std::size_t RGp = count_reversible(Gp, n);
    const double log_acc = delta
        + std::log(static_cast<double>(RG)) - std::log(static_cast<double>(RGp));

    std::uniform_real_distribution<double> U(0.0, 1.0);
    if (std::log(U(rng)) < log_acc) {
        c.state     = dag_to_partition(Gp, n);
        c.log_score = partition_log_score(cache, c.state);
        c.sj_nbhd   = splitjoin_nbhd(c.state);
    }
}

/// One partition-MCMC step: a mixture of the split/join move (with the
/// #nbd(L)/#nbd(L') Hastings correction) and the symmetric swap move, plus a
/// small stay-still probability for aperiodicity. All three preserve the target
/// P(Lambda|D) (proven by exact enumeration in
/// tests/test_partition_mcmc_diagnostics.cpp), so the mixture does too.
inline void partition_mcmc_step(partition_chain& c, const score_cache& cache,
                                std::mt19937_64& rng,
                                double p_stay = 0.01,
                                double p_edgerev = 0.07,
                                double p_splitjoin = 0.37,
                                double p_swap = 0.27) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double u = U(rng);
    if (u < p_stay) return;                                   // stay still
    if (u < p_stay + p_edgerev) {                             // edge reversal
        partition_edge_reversal_step(c, cache, rng);
        return;
    }
    if (u < p_stay + p_edgerev + p_splitjoin) {               // split / join
        partition_state prop = propose_splitjoin(rng, c.state);
        const double ls_p  = partition_log_score(cache, prop);
        const double nbd_p = splitjoin_nbhd(prop);
        const double log_acc = (ls_p - c.log_score)
                             + std::log(c.sj_nbhd) - std::log(nbd_p);
        if (std::log(U(rng)) < log_acc) {
            c.state = std::move(prop); c.log_score = ls_p; c.sj_nbhd = nbd_p;
        }
    } else if (u < p_stay + p_edgerev + p_splitjoin + p_swap) {  // symmetric swap
        partition_state prop = propose_swap_any(rng, c.state);
        const double ls_p = partition_log_score(cache, prop);
        if (std::log(U(rng)) < (ls_p - c.log_score)) {        // q-ratio = 1
            c.state = std::move(prop); c.log_score = ls_p;
            c.sj_nbhd = splitjoin_nbhd(c.state);
        }
    } else {                                                  // single-node move
        std::vector<partition_state> nbrs = single_node_neighbours(c.state);
        if (nbrs.empty()) return;
        std::uniform_int_distribution<std::size_t> pick(0, nbrs.size() - 1);
        partition_state prop = nbrs[pick(rng)];
        const double ls_p = partition_log_score(cache, prop);
        const std::size_t nbd_p = single_node_neighbours(prop).size();
        const double log_acc = (ls_p - c.log_score)
            + std::log(static_cast<double>(nbrs.size()))
            - std::log(static_cast<double>(nbd_p));
        if (std::log(U(rng)) < log_acc) {
            c.state = std::move(prop); c.log_score = ls_p;
            c.sj_nbhd = splitjoin_nbhd(c.state);
        }
    }
}

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_PARTITION_STATE_HPP

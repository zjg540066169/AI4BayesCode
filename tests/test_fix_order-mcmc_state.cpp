// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  test_fix_order-mcmc_state.cpp -- order_mcmc_block stores ONE state.
//
//  In method = partition (the default) the chain state is the partition
//  chain. order(), current(), get_history() without a buffer, the rows that
//  record_held_history() appends, current_log_score() and sampled_dag() must
//  all describe that state. A node sequence does not give the element sizes,
//  so set_current() either returns without a change (round trip) or throws,
//  and the constructor rejects initial_order. In method = order,
//  set_current() must also replace the sampled DAG by one consistent with the
//  loaded order.
//
//    S1  order() equals current() at every sweep, and sampled_dag() is a DAG
//        of the partition (order(), party) at every sweep
//    S2  stepped block: set_current(v) with v != current() throws and leaves
//        the state unchanged; the next 200 sweeps equal an untouched twin
//    S3  composite_block::set_current(current()) on a partition with two or
//        more elements leaves the partition log-score and the sampled DAG
//        unchanged
//    S4  frozen block with keep_history: the held rows store the held state,
//        frozen at a node sequence other than 0..n-1 (the sequence of the
//        constructor's one-element partition)
//    S5  keep_history off: get_history() returns current()
//    S6  method = order: after set_current(v), sampled_dag() respects v
//    S7  new block (one-element partition): set_current(v) throws for every
//        v, 0..n-1 included, directly and through composite_block, and leaves
//        the state unchanged
//    S8  checkpoint restore: composite_block::set_current into a new composite
//        of 0..n-1 (the node sequence of every partition that cuts 0..n-1
//        into consecutive elements) and of 40 saved states of a stepped chain
//        throws and leaves the new composite unchanged
//    S9  a non-empty initial_order throws in method = partition; method =
//        order starts from it
// ============================================================================

#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/order_mcmc_block.hpp"

#include <cstdio>
#include <cstring>
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

// Binary chain 0 -> 1 -> ... -> n-1; each node copies its parent w.p. 0.9.
arma::imat sim_chain(std::size_t N, std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    prng::uniform_real_distribution<double> U(0, 1);
    arma::imat X(N, n);
    for (std::size_t r = 0; r < N; ++r) {
        int prev = U(rng) < 0.5 ? 0 : 1;
        for (std::size_t j = 0; j < n; ++j) {
            const int v = (j == 0) ? prev : (U(rng) < 0.9 ? prev : 1 - prev);
            X(r, j) = v; prev = v;
        }
    }
    return X;
}

order_mcmc_block_config make_cfg(std::size_t n, bool partition) {
    order_mcmc_block_config cfg;
    cfg.name = "order";
    cfg.data = sim_chain(2000, n, 20261122u);
    cfg.cardinalities = arma::uvec(n, arma::fill::value(2));
    cfg.structure_prior = dag_prior::FK_EQ2;
    cfg.max_parents = 3;
    cfg.init_rng_seed = 7;
    cfg.method = partition ? order_mcmc_block_config::method_t::partition
                           : order_mcmc_block_config::method_t::order;
    return cfg;
}

std::vector<std::size_t> to_sizes(const arma::vec& v) {
    std::vector<std::size_t> out(v.n_elem);
    for (std::size_t k = 0; k < v.n_elem; ++k)
        out[k] = static_cast<std::size_t>(v[k]);
    return out;
}

// Element sizes of the block's partition, read from its named outputs with a
// scratch generator (the block's own state is not changed).
std::vector<std::size_t> party_of(const order_mcmc_block& blk) {
    std::mt19937_64 scratch(1);
    return to_sizes(blk.current_named_outputs(scratch).at("order_party"));
}

bool same_values(const arma::uvec& a, const arma::vec& b) {
    if (a.n_elem != b.n_elem) return false;
    for (std::size_t k = 0; k < a.n_elem; ++k)
        if (static_cast<double>(a[k]) != b[k]) return false;
    return true;
}

// Is `dag` a DAG of the labelled partition (permy, party)? Every parent of a
// node lies in a later element, and a node outside the last element has a
// parent in the next element.
bool dag_in_partition(const std::vector<std::uint64_t>& dag,
                      const std::vector<std::size_t>& permy,
                      const std::vector<std::size_t>& party) {
    const std::size_t n = permy.size();
    if (dag.size() != n) return false;
    std::vector<std::size_t> elem(n, 0);
    std::size_t p = 0;
    for (std::size_t t = 0; t < party.size(); ++t)
        for (std::size_t c = 0; c < party[t]; ++c) elem[permy[p++]] = t;
    if (p != n) return false;
    for (std::size_t i = 0; i < n; ++i) {
        bool next = false;
        for (std::size_t j = 0; j < n; ++j) {
            if (!(dag[i] & (1ULL << j))) continue;
            if (elem[j] <= elem[i]) return false;
            if (elem[j] == elem[i] + 1) next = true;
        }
        if (elem[i] + 1 < party.size() && !next) return false;
    }
    return true;
}

// Does every parent precede its child in `order`?
bool dag_respects_order(const std::vector<std::uint64_t>& dag,
                        const arma::uvec& order) {
    const std::size_t n = order.n_elem;
    std::vector<std::size_t> pos(n);
    for (std::size_t q = 0; q < n; ++q) pos[order[q]] = q;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if ((dag[i] & (1ULL << j)) && pos[j] >= pos[i]) return false;
    return true;
}

// ---------------------------------------------------------------------------
void S1_order_tracks_chain() {
    std::printf("\n--- S1: order() and sampled_dag() follow the partition chain ---\n");
    order_mcmc_block blk(make_cfg(6, true));
    block_context ctx;
    std::mt19937_64 rng(99);
    int bad_order = same_values(blk.order(), blk.current()) ? 0 : 1;
    int bad_dag = 0;
    for (int s = 0; s < 400; ++s) {
        blk.set_context(ctx);
        blk.step(rng);
        if (!same_values(blk.order(), blk.current())) ++bad_order;
        const std::vector<std::size_t> permy(blk.order().begin(), blk.order().end());
        if (!dag_in_partition(blk.sampled_dag(), permy, party_of(blk))) ++bad_dag;
    }
    check(bad_order == 0, "S1 order() == current() at construction and in 400 sweeps",
          "mismatches = " + std::to_string(bad_order));
    check(bad_dag == 0, "S1 sampled_dag() is a DAG of the partition (order(), party)",
          "violations = " + std::to_string(bad_dag));
}

// Same node sequence, element sizes, log-score bits and sampled DAG.
bool same_state(const order_mcmc_block& a, const order_mcmc_block& b) {
    const double la = a.current_log_score(), lb = b.current_log_score();
    return arma::all(a.current() == b.current())
        && same_values(a.order(), a.current()) && same_values(b.order(), b.current())
        && party_of(a) == party_of(b)
        && std::memcmp(&la, &lb, sizeof la) == 0
        && a.sampled_dag() == b.sampled_dag();
}

// Calls f; true when it throws std::invalid_argument.
template <class F>
bool throws_invalid(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

void S2_set_current_other_sequence_throws() {
    std::printf("\n--- S2: stepped block: set_current(v != current()) throws ---\n");
    order_mcmc_block blk(make_cfg(6, true)), twin(make_cfg(6, true));
    block_context ctx;
    std::mt19937_64 rng(11), rng_twin(11);
    for (int s = 0; s < 300; ++s) {
        blk.set_context(ctx); blk.step(rng);
        twin.set_context(ctx); twin.step(rng_twin);
    }
    const arma::vec cur = blk.current();
    const std::size_t n = cur.n_elem;
    arma::vec rot(n), rev(n);
    for (std::size_t k = 0; k < n; ++k) { rot[k] = cur[(k + 1) % n]; rev[k] = cur[n - 1 - k]; }
    std::printf("    element sizes:");
    for (std::size_t k : party_of(blk)) std::printf(" %zu", k);
    std::printf("\n");
    const bool t1 = throws_invalid([&] { blk.set_current(rot); });
    const bool t2 = throws_invalid([&] { blk.set_current(rev); });
    check(t1 && t2, "S2 set_current throws for the rotated and the reversed node sequence");
    check(same_state(blk, twin), "S2 the state is unchanged (node sequence, sizes, log-score, DAG)");
    int mismatch = 0;
    for (int s = 0; s < 200; ++s) {
        blk.set_context(ctx); blk.step(rng);
        twin.set_context(ctx); twin.step(rng_twin);
        if (!same_state(blk, twin)) ++mismatch;
    }
    check(mismatch == 0, "S2 the next 200 sweeps equal the untouched twin",
          "mismatching sweeps = " + std::to_string(mismatch));
}

void S3_composite_round_trip() {
    std::printf("\n--- S3: composite_block::set_current(current()) leaves the state unchanged ---\n");
    composite_block comp("bn");
    comp.add_child(std::make_unique<order_mcmc_block>(make_cfg(6, true)));
    const auto* blk = dynamic_cast<const order_mcmc_block*>(&comp.child(0));
    std::mt19937_64 rng(5);
    for (int s = 0; s < 200; ++s) comp.step(rng);
    for (int s = 0; s < 5000 && party_of(*blk).size() < 2; ++s) comp.step(rng);
    const arma::vec theta = comp.current();
    const double ls0 = blk->current_log_score();
    const auto dag0 = blk->sampled_dag();
    check(party_of(*blk).size() >= 2, "S3 precondition: the partition has two or more elements");
    comp.set_current(theta);
    check(blk->current_log_score() == ls0,
          "S3 log-score unchanged by the round trip",
          std::to_string(ls0) + " -> " + std::to_string(blk->current_log_score()));
    check(arma::all(blk->current() == theta), "S3 current() unchanged by the round trip");
    check(blk->sampled_dag() == dag0, "S3 sampled_dag() unchanged by the round trip");
}

void S4_held_history() {
    std::printf("\n--- S4: frozen block, keep_history: held rows store the held state ---\n");
    composite_block comp("bn");
    comp.add_child(std::make_unique<order_mcmc_block>(make_cfg(6, true)));
    comp.set_keep_history(true);
    const auto* blk = dynamic_cast<const order_mcmc_block*>(&comp.child(0));
    std::mt19937_64 rng(5);
    // Freeze at a node sequence other than 0..n-1, the sequence of the
    // constructor's one-element partition, so that held rows of 0..n-1 or of
    // the constructor state differ from the held state.
    const std::size_t n = blk->current().n_elem;
    arma::vec id(n);
    for (std::size_t k = 0; k < n; ++k) id[k] = static_cast<double>(k);
    std::size_t sweeps = 0;
    while (sweeps < 100 || (sweeps < 5000 && arma::all(blk->current() == id))) {
        comp.step(rng);
        ++sweeps;
    }
    const arma::vec held = blk->current();
    const double held_ls = blk->current_log_score();
    std::printf("    held state after %zu sweeps:", sweeps);
    for (double x : held) std::printf(" %g", x);
    std::printf(" | sizes:");
    for (std::size_t k : party_of(*blk)) std::printf(" %zu", k);
    std::printf("\n");
    check(!arma::all(held == id), "S4 precondition: the held node sequence differs from 0..n-1");
    comp.freeze({"order"}, true);
    for (int s = 0; s < 10; ++s) comp.step(rng);
    const auto h = comp.get_history();
    const arma::mat& H = h.at("order");
    const arma::mat& L = h.at("order_log_score");
    int bad = 0;
    for (std::size_t r = sweeps; r < sweeps + 10 && r < H.n_rows; ++r) {
        if (arma::any(H.row(r).t() != held) || L(r, 0) != held_ls) ++bad;
    }
    check(H.n_rows == sweeps + 10 && bad == 0, "S4 the 10 held rows equal the held state",
          "bad rows = " + std::to_string(bad));
}

void S5_history_without_buffer() {
    std::printf("\n--- S5: keep_history off: get_history() returns current() ---\n");
    order_mcmc_block blk(make_cfg(6, true));
    block_context ctx;
    std::mt19937_64 rng(3);
    for (int s = 0; s < 100; ++s) { blk.set_context(ctx); blk.step(rng); }
    const arma::mat H = blk.get_history().at("order");
    check(H.n_rows == 1 && arma::all(H.row(0).t() == blk.current()),
          "S5 the single history row equals current()");
}

void S6_order_mode_sampled_dag() {
    std::printf("\n--- S6: method = order: set_current redraws the sampled DAG ---\n");
    order_mcmc_block blk(make_cfg(6, false));
    block_context ctx;
    std::mt19937_64 rng(3);
    for (int s = 0; s < 500; ++s) { blk.set_context(ctx); blk.step(rng); }
    const arma::uvec o = blk.order();
    std::size_t edges = 0;
    for (std::uint64_t m : blk.sampled_dag())
        for (; m != 0; m &= m - 1) ++edges;
    arma::vec rev(o.n_elem);
    for (std::size_t q = 0; q < o.n_elem; ++q) rev[q] = static_cast<double>(o[o.n_elem - 1 - q]);
    blk.set_current(rev);
    check(edges > 0 && dag_respects_order(blk.sampled_dag(), blk.order()),
          "S6 sampled_dag() respects the loaded order",
          "edges before = " + std::to_string(edges));
    // Loading the current order again leaves the DAG unchanged.
    const auto dag1 = blk.sampled_dag();
    blk.set_current(blk.current());
    check(blk.sampled_dag() == dag1, "S6 set_current(current()) leaves sampled_dag() unchanged");
}

void S7_new_block_set_current_throws() {
    std::printf("\n--- S7: new block (one-element partition): set_current throws ---\n");
    order_mcmc_block blk(make_cfg(6, true)), twin(make_cfg(6, true));
    const std::size_t n = blk.current().n_elem;
    arma::vec id(n), rev(n), rot(n);
    for (std::size_t k = 0; k < n; ++k) {
        id[k] = static_cast<double>(k);
        rev[k] = static_cast<double>(n - 1 - k);
        rot[k] = static_cast<double>((k + 1) % n);
    }
    check(arma::all(blk.current() == id) && party_of(blk).size() == 1,
          "S7 precondition: a new block holds the one-element partition, current() = 0..n-1");
    int thrown = 0;
    for (const arma::vec* v : {&id, &rev, &rot})
        if (throws_invalid([&] { blk.set_current(*v); })) ++thrown;
    check(thrown == 3, "S7 set_current throws for 0..n-1, its reverse and a rotation",
          "thrown = " + std::to_string(thrown) + " of 3");
    check(same_state(blk, twin), "S7 the state is unchanged");
    composite_block comp("bn");
    comp.add_child(std::make_unique<order_mcmc_block>(make_cfg(6, true)));
    const bool tc = throws_invalid([&] { comp.set_current(rev); });
    const auto* cb = dynamic_cast<const order_mcmc_block*>(&comp.child(0));
    check(tc && same_state(*cb, twin),
          "S7 composite_block::set_current throws and leaves the state unchanged");
}

void S8_restore_into_new_composite() {
    std::printf("\n--- S8: checkpoint restore of saved states into a new composite ---\n");
    for (std::size_t n : {std::size_t(4), std::size_t(6)}) {
        // Saved values: 0..n-1, the node sequence of every partition that
        // cuts 0..n-1 into consecutive elements, and 40 states of a chain.
        std::vector<arma::vec> saved(1, arma::vec(n));
        for (std::size_t k = 0; k < n; ++k) saved[0][k] = static_cast<double>(k);
        composite_block src("bn");
        src.add_child(std::make_unique<order_mcmc_block>(make_cfg(n, true)));
        std::mt19937_64 rng(21 + n);
        for (int s = 0; s < 100; ++s) src.step(rng);
        int identity_seq = 0;
        for (int s = 0; s < 400; ++s) {
            src.step(rng);
            if (s % 10 != 0) continue;
            saved.push_back(src.current());
            if (arma::all(saved.back() == saved[0])) ++identity_seq;
        }
        const order_mcmc_block fresh(make_cfg(n, true));
        int thrown = 0, changed = 0;
        for (const arma::vec& theta : saved) {
            composite_block dst("bn");
            dst.add_child(std::make_unique<order_mcmc_block>(make_cfg(n, true)));
            if (throws_invalid([&] { dst.set_current(theta); })) ++thrown;
            const auto* db = dynamic_cast<const order_mcmc_block*>(&dst.child(0));
            if (!same_state(*db, fresh)) ++changed;
        }
        check(thrown == static_cast<int>(saved.size()) && changed == 0,
              "S8 n=" + std::to_string(n)
              + ": every restore throws and leaves the new composite unchanged",
              "restores = " + std::to_string(saved.size()) + ", thrown = "
              + std::to_string(thrown) + ", changed = " + std::to_string(changed)
              + ", chain states with sequence 0..n-1 = " + std::to_string(identity_seq));
    }
}

void S9_initial_order() {
    std::printf("\n--- S9: initial_order ---\n");
    const std::size_t n = 6;
    arma::uvec id(n), rev(n);
    for (std::size_t k = 0; k < n; ++k) { id[k] = k; rev[k] = n - 1 - k; }
    auto cp = make_cfg(n, true);
    cp.initial_order = id;
    const bool t1 = throws_invalid([&] { order_mcmc_block b(cp); });
    cp.initial_order = rev;
    const bool t2 = throws_invalid([&] { order_mcmc_block b(cp); });
    check(t1 && t2, "S9 method = partition: a non-empty initial_order throws (0..n-1 and its reverse)");
    auto co = make_cfg(n, false);
    co.initial_order = rev;
    const order_mcmc_block ob(co);
    check(same_values(rev, ob.current()), "S9 method = order: the chain starts from initial_order");
}
}  // namespace

int main() {
    std::printf("=== test_fix_order-mcmc_state ===\n");
    S1_order_tracks_chain();
    S2_set_current_other_sequence_throws();
    S3_composite_round_trip();
    S4_held_history();
    S5_history_without_buffer();
    S6_order_mode_sampled_dag();
    S7_new_block_set_current_throws();
    S8_restore_into_new_composite();
    S9_initial_order();
    std::printf("\n=== %d PASS / %d FAIL ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

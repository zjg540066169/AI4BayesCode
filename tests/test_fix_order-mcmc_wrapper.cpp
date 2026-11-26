// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  test_fix_order-mcmc_wrapper.cpp -- the OrderMCMCBN wrapper
//  (examples/OrderMCMCBN.cpp, standalone C++ build) in its partition method.
//
//  get_current() returns the partition: "order" (node sequence, 1-based,
//  element 0 first) and "order_party" (element sizes). set_current() with
//  both keys loads that partition; "order" alone loads nothing.
//
//    W1  at construction and in 300 sweeps, (order, order_party) is a
//        partition, sampled_DAG is a DAG of it, and log_score equals its
//        partition log-score bit for bit
//    W2  round trip into a stepped second model with the same seed: after
//        the second model is loaded with the same node sequence and other
//        element sizes ("order" alone then changes nothing), loading the
//        saved (order, order_party) makes the next 300 sweeps equal the
//        source's; without that load they differ
//    W3  round trip into a new model with the source's seed: the loaded
//        partition and log_score equal the saved ones, and the next 300
//        sweeps equal the source's
//    W4  "order" alone: an error that names order_party on a new model and
//        for another node sequence (state unchanged, next 200 sweeps equal an
//        untouched twin); no change for the current node sequence
//    W5  a non-empty initial_order is an error that points to
//        set_current(list(order = ..., order_party = ...))
//    W6  invalid input is an error and leaves the state unchanged
//    W7  get_history(): "order" is 1-based, as in get_current(): with
//        keep_history, row t equals get_current()$order after sweep t (also
//        after a load), and rows t of order_log_score and order_sampled_DAG
//        equal get_current()$log_score and $sampled_DAG; without, the single
//        row is get_current()$order
//    W8  the composite inside the wrapper: at construction and after loads,
//        the shared data keys order, order_party, order_sampled_DAG and
//        order_log_score hold the state of get_current() (the DAG is
//        sampled_DAG), and after a load its current() is the loaded node
//        sequence also when current() was called before the load
// ============================================================================

#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"
#include "AI4BayesCode/order_mcmc_block.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/shared_data.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Access to the wrapper's composite for W8. OrderMCMCBN declares
// kernel_control_mixin<OrderMCMCBN> a friend. This explicit specialization,
// the base class of OrderMCMCBN here in place of the primary template, has
// only the accessor, so this OrderMCMCBN has no clear_history() (the test
// calls none).
class OrderMCMCBN;
namespace AI4BayesCode {
template <>
class kernel_control_mixin<OrderMCMCBN> {
public:
    static const composite_block& composite_of(const OrderMCMCBN& m);
};
}  // namespace AI4BayesCode

// The example's own int main() is renamed; this file provides main().
#define main ordermcmcbn_example_main
#include "../examples/OrderMCMCBN.cpp"
#undef main

const AI4BayesCode::composite_block&
AI4BayesCode::kernel_control_mixin<OrderMCMCBN>::composite_of(const OrderMCMCBN& m) {
    return *m.impl_;
}

using namespace AI4BayesCode;

namespace {
int g_pass = 0, g_fail = 0;
void check(bool ok, const std::string& tag, const std::string& d = "") {
    std::printf("  %s  %s %s\n", ok ? "PASS" : "FAIL", tag.c_str(), d.c_str());
    if (ok) ++g_pass; else ++g_fail;
}

const std::size_t kN = 5;

// Binary chain X0 -> X1 -> ... ; each node copies its parent w.p. 1 - flip.
arma::mat chain_data(std::size_t N, std::size_t n, double flip, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    arma::mat X(N, n, arma::fill::zeros);
    for (std::size_t r = 0; r < N; ++r) {
        int prev = (U(g) < 0.5) ? 1 : 0;
        X(r, 0) = prev;
        for (std::size_t j = 1; j < n; ++j) {
            const int v = (U(g) < flip) ? 1 - prev : prev;
            X(r, j) = v;
            prev = v;
        }
    }
    return X;
}

const arma::mat& data() {
    static const arma::mat X = chain_data(1500, kN, 0.2, 20261123u);
    return X;
}

std::unique_ptr<OrderMCMCBN> make_model(int seed, int max_parents = 3,
                                        const arma::vec& init = arma::vec(),
                                        bool keep_history = false) {
    return std::make_unique<OrderMCMCBN>(data(), arma::vec(kN, arma::fill::value(2.0)),
                                         "fk_eq2", 1.0, max_parents, 20, 4000, 10.0, 0.5,
                                         init, seed, keep_history);
}

// The block the wrapper builds (same data, prior and settings), for the
// partition log-score of a state.
const order_mcmc_block& reference_block() {
    static const order_mcmc_block blk = [] {
        order_mcmc_block_config cfg;
        cfg.name = "order";
        cfg.data = arma::conv_to<arma::imat>::from(data());
        cfg.cardinalities = arma::uvec(kN, arma::fill::value(2));
        cfg.structure_prior = dag_prior::FK_EQ2;
        cfg.max_parents = 3;
        return order_mcmc_block(cfg);
    }();
    return blk;
}

arma::vec value_of(const state_map& m, const std::string& key) {
    const auto it = m.find(key);
    return it == m.end() ? arma::vec() : it->second;
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem
        && (a.n_elem == 0 || std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0);
}

// order, order_party, sampled_DAG and log_score are equal bit for bit.
bool same_current(const state_map& a, const state_map& b) {
    for (const char* k : {"order", "order_party", "sampled_DAG", "log_score"})
        if (!same_bits(value_of(a, k), value_of(b, k))) return false;
    return !value_of(a, "order").is_empty();
}

// The partition state of get_current() (0-based node sequence, sizes);
// false when the values are not a partition of 0..n-1.
bool state_of(const state_map& cur, partition_state& s) {
    const arma::vec o = value_of(cur, "order"), p = value_of(cur, "order_party");
    if (o.n_elem != kN || p.n_elem == 0) return false;
    s.permy.assign(kN, 0);
    std::vector<int> seen(kN, 0);
    for (std::size_t i = 0; i < kN; ++i) {
        const double v = o[i] - 1.0;
        if (!(v >= 0.0 && v < static_cast<double>(kN)) || std::floor(v) != v) return false;
        s.permy[i] = static_cast<std::size_t>(v);
        if (seen[s.permy[i]]++) return false;
    }
    s.party.clear();
    std::size_t total = 0;
    for (double d : p) {
        if (!(d >= 1.0) || std::floor(d) != d) return false;
        s.party.push_back(static_cast<std::size_t>(d));
        total += s.party.back();
    }
    return total == kN;
}

// Every parent of a node lies in a later element, and a node outside the
// last element has a parent in the next element. A[i + j*n] = 1 <=> j is a
// parent of i.
bool dag_in_partition(const arma::vec& A, const partition_state& s) {
    if (A.n_elem != kN * kN) return false;
    std::vector<std::size_t> elem(kN, 0);
    std::size_t p = 0;
    for (std::size_t t = 0; t < s.m(); ++t)
        for (std::size_t c = 0; c < s.party[t]; ++c) elem[s.permy[p++]] = t;
    for (std::size_t i = 0; i < kN; ++i) {
        bool next = false;
        for (std::size_t j = 0; j < kN; ++j) {
            if (A[i + j * kN] == 0.0) continue;
            if (elem[j] <= elem[i]) return false;
            if (elem[j] == elem[i] + 1) next = true;
        }
        if (elem[i] + 1 < s.m() && !next) return false;
    }
    return true;
}

state_map partition_of(const state_map& cur) {
    state_map m;
    m["order"] = value_of(cur, "order");
    m["order_party"] = value_of(cur, "order_party");
    return m;
}

// Calls f; the message of the std::exception it throws, or "" when it
// returns.
template <class F>
std::string error_of(F&& f) {
    try { f(); } catch (const std::exception& e) { return std::string("E:") + e.what(); }
    return "";
}

bool contains(const std::string& s, const std::string& part) {
    return s.find(part) != std::string::npos;
}

// Element sizes other than `party` for which the node sequence `order`
// (1-based) is sorted within each element: all singletons, or, when `party`
// is all singletons, the first adjacent ascending pair merged. Empty when no
// such sizes exist.
arma::vec other_sizes(const arma::vec& order, const arma::vec& party) {
    const std::size_t n = order.n_elem;
    if (party.n_elem != n) return arma::vec(n, arma::fill::ones);
    for (std::size_t p = 0; p + 1 < n; ++p) {
        if (order[p] < order[p + 1]) {
            arma::vec r(n - 1, arma::fill::ones);
            r[p] = 2.0;
            return r;
        }
    }
    return arma::vec();
}

// ---------------------------------------------------------------------------
void W1_current_is_the_partition() {
    std::printf("\n--- W1: get_current() describes the partition ---\n");
    auto m = make_model(3);
    const score_cache& cache = reference_block().cache();
    int bad_state = 0, bad_dag = 0, bad_score = 0, multi = 0;
    for (int s = 0; s <= 300; ++s) {
        if (s > 0) m->step(1);
        const state_map cur = m->get_current();
        partition_state st;
        if (!state_of(cur, st)) { ++bad_state; continue; }
        partition_state canon = st;
        canonicalize(canon);
        if (canon.permy != st.permy) ++bad_state;
        if (st.m() >= 2) ++multi;
        if (!dag_in_partition(value_of(cur, "sampled_DAG"), st)) ++bad_dag;
        const arma::vec ls = value_of(cur, "log_score");
        if (ls.n_elem != 1 || !same_bits(ls, arma::vec{partition_log_score(cache, st)})) ++bad_score;
    }
    check(bad_state == 0 && multi > 0,
          "W1 (order, order_party) is a canonical partition at construction and in 300 sweeps",
          "violations = " + std::to_string(bad_state) + ", sweeps with >= 2 elements = "
          + std::to_string(multi));
    check(bad_dag == 0, "W1 sampled_DAG is a DAG of (order, order_party)",
          "violations = " + std::to_string(bad_dag));
    check(bad_score == 0, "W1 log_score equals the partition log-score of (order, order_party)",
          "violations = " + std::to_string(bad_score));
}

void W2_round_trip_into_stepped_model() {
    std::printf("\n--- W2: round trip into a stepped model with the same seed ---\n");
    auto a = make_model(7), b = make_model(7), c = make_model(7);
    a->step(400); b->step(400); c->step(400);
    arma::vec other;
    for (int s = 0; s < 5000; ++s) {
        const state_map cur = a->get_current();
        if (value_of(cur, "order_party").n_elem >= 2) {
            other = other_sizes(value_of(cur, "order"), value_of(cur, "order_party"));
            if (!other.is_empty()) break;
        }
        a->step(1); b->step(1); c->step(1);
    }
    const state_map saved = a->get_current();
    std::printf("    saved order:");
    for (double x : value_of(saved, "order")) std::printf(" %g", x);
    std::printf(" | sizes:");
    for (double x : value_of(saved, "order_party")) std::printf(" %g", x);
    std::printf(" | other sizes:");
    for (double x : other) std::printf(" %g", x);
    std::printf("\n");
    const std::string e1 = error_of([&] {
        state_map p;
        p["order"] = value_of(saved, "order");
        p["order_party"] = other;
        b->set_current(p);
        c->set_current(p);
    });
    const state_map moved = b->get_current();
    const bool perturbed = e1.empty() && !other.is_empty()
        && same_bits(value_of(moved, "order"), value_of(saved, "order"))
        && same_bits(value_of(moved, "order_party"), other);
    check(perturbed, "W2 precondition: the second model holds the saved node sequence "
                     "with other element sizes", e1);
    state_map order_only;
    order_only["order"] = value_of(saved, "order");
    const std::string e2 = error_of([&] { b->set_current(order_only); });
    check(e2.empty() && same_current(b->get_current(), moved),
          "W2 'order' alone (the current node sequence) changes nothing", e2);
    const std::string e3 = error_of([&] { b->set_current(partition_of(saved)); });
    const state_map loaded = b->get_current();
    check(e3.empty() && same_bits(value_of(loaded, "order"), value_of(saved, "order"))
          && same_bits(value_of(loaded, "order_party"), value_of(saved, "order_party"))
          && same_bits(value_of(loaded, "log_score"), value_of(saved, "log_score")),
          "W2 set_current(order, order_party) loads the saved partition and log_score", e3);
    int mismatch = 0, control_mismatch = 0;
    for (int s = 0; s < 300; ++s) {
        a->step(1); b->step(1); c->step(1);
        if (!same_current(a->get_current(), b->get_current())) ++mismatch;
        if (!same_current(a->get_current(), c->get_current())) ++control_mismatch;
    }
    check(mismatch == 0, "W2 the next 300 sweeps of the restored model equal the source's",
          "mismatching sweeps = " + std::to_string(mismatch));
    check(control_mismatch > 0,
          "W2 control: the model left at the other element sizes differs from the source",
          "mismatching sweeps = " + std::to_string(control_mismatch));
}

void W3_round_trip_into_new_model() {
    std::printf("\n--- W3: round trip into a new model with the source's seed ---\n");
    auto src = make_model(99);
    src->step(500);
    for (int s = 0; s < 5000 && value_of(src->get_current(), "order_party").n_elem < 2; ++s)
        src->step(1);
    const state_map s0 = src->get_current();
    auto a = make_model(7), b = make_model(7);
    const std::string e1 = error_of([&] { a->set_current(partition_of(s0)); });
    const state_map la = a->get_current();
    check(e1.empty() && same_bits(value_of(la, "order"), value_of(s0, "order"))
          && same_bits(value_of(la, "order_party"), value_of(s0, "order_party"))
          && same_bits(value_of(la, "log_score"), value_of(s0, "log_score")),
          "W3 a new model loads (order, order_party); log_score equals the saved one", e1);
    const std::string e2 = error_of([&] { b->set_current(partition_of(a->get_current())); });
    check(e2.empty() && same_current(a->get_current(), b->get_current()),
          "W3 get_current -> set_current into a second new model gives the same values", e2);
    int mismatch = 0;
    for (int s = 0; s < 300; ++s) {
        a->step(1); b->step(1);
        if (!same_current(a->get_current(), b->get_current())) ++mismatch;
    }
    check(mismatch == 0, "W3 the next 300 sweeps of the two models are equal",
          "mismatching sweeps = " + std::to_string(mismatch));
}

void W4_order_alone() {
    std::printf("\n--- W4: set_current with 'order' alone ---\n");
    auto fresh = make_model(5), fresh_twin = make_model(5);
    state_map id;
    id["order"] = arma::regspace<arma::vec>(1, static_cast<double>(kN));
    const std::string e1 = error_of([&] { fresh->set_current(id); });
    check(contains(e1, "order_party") && same_current(fresh->get_current(), fresh_twin->get_current()),
          "W4 new model: an error that names order_party; state unchanged",
          e1.substr(0, 90));
    auto m = make_model(11), twin = make_model(11);
    m->step(300); twin->step(300);
    for (int s = 0; s < 5000 && value_of(m->get_current(), "order_party").n_elem < 2; ++s) {
        m->step(1); twin->step(1);
    }
    const state_map before = m->get_current();
    state_map same_seq;
    same_seq["order"] = value_of(before, "order");
    const std::string e2 = error_of([&] { m->set_current(same_seq); });
    check(e2.empty() && same_current(m->get_current(), before),
          "W4 stepped model: the current node sequence changes nothing", e2);
    state_map rev;
    rev["order"] = arma::reverse(value_of(before, "order"));
    const std::string e3 = error_of([&] { m->set_current(rev); });
    check(contains(e3, "order_party") && same_current(m->get_current(), before),
          "W4 stepped model: another node sequence is an error that names order_party; "
          "state unchanged", e3.substr(0, 90));
    int mismatch = 0;
    for (int s = 0; s < 200; ++s) {
        m->step(1); twin->step(1);
        if (!same_current(m->get_current(), twin->get_current())) ++mismatch;
    }
    check(mismatch == 0, "W4 the next 200 sweeps equal an untouched twin",
          "mismatching sweeps = " + std::to_string(mismatch));
}

void W5_initial_order() {
    std::printf("\n--- W5: initial_order ---\n");
    const arma::vec id = arma::regspace<arma::vec>(1, static_cast<double>(kN));
    const std::string e1 = error_of([&] { make_model(1, 3, id); });
    const std::string e2 = error_of([&] { make_model(1, 3, arma::reverse(id)); });
    const std::string hint = "set_current(list(order = ..., order_party = ...))";
    check(contains(e1, hint) && contains(e2, hint),
          "W5 a non-empty initial_order is an error that points to " + hint,
          e1.substr(0, 90));
    const std::string e3 = error_of([&] { make_model(1, 3, arma::vec()); });
    check(e3.empty(), "W5 an empty initial_order constructs the model", e3);
}

void W6_invalid_input() {
    std::printf("\n--- W6: invalid input is an error and leaves the state unchanged ---\n");
    auto m = make_model(13);
    m->step(200);
    const state_map before = m->get_current();
    const arma::vec ord = value_of(before, "order");
    const double n = static_cast<double>(kN);
    int errors = 0, changed = 0, cases = 0;
    auto expect_error = [&](const state_map& p) {
        ++cases;
        if (!error_of([&] { m->set_current(p); }).empty()) ++errors;
        if (!same_current(m->get_current(), before)) ++changed;
    };
    state_map p;
    p["order_party"] = arma::vec{n};
    expect_error(p);                                            // order_party without order
    p.clear(); p["order"] = ord; p["order_party"] = arma::vec{n - 1.0};
    expect_error(p);                                            // sizes summing to n - 1
    p.clear(); p["order"] = ord; p["order_party"] = arma::vec{0.0, n};
    expect_error(p);                                            // empty element
    p.clear(); p["order"] = ord; p["order_party"] = arma::vec{n, 1.0};
    expect_error(p);                                            // sizes summing to n + 1
    p.clear(); p["order"] = ord; p["order_party"] = arma::vec();
    expect_error(p);                                            // no elements
    p.clear(); p["order"] = ord; p["order_party"] = arma::vec{std::nan(""), n};
    expect_error(p);                                            // NaN size
    arma::vec dup = ord; dup[1] = dup[0];
    p.clear(); p["order"] = dup; p["order_party"] = arma::vec{n};
    expect_error(p);                                            // repeated node
    arma::vec out = ord; out[0] = n + 1.0;
    p.clear(); p["order"] = out; p["order_party"] = arma::vec{n};
    expect_error(p);                                            // node out of range
    p.clear(); p["order"] = ord; p["order_party"] = arma::vec{n};
    p["log_score"] = arma::vec{0.0};
    expect_error(p);                                            // derived key
    check(errors == cases && changed == 0,
          "W6 invalid (order, order_party) input: an error, state unchanged",
          "errors = " + std::to_string(errors) + " of " + std::to_string(cases)
          + ", changed = " + std::to_string(changed));
    // max_parents = 0: every partition with two or more elements has
    // log-score -inf.
    auto z = make_model(13, 0), z_twin = make_model(13, 0);
    state_map singles;
    singles["order"] = arma::regspace<arma::vec>(1, n);
    singles["order_party"] = arma::vec(kN, arma::fill::ones);
    const std::string e = error_of([&] { z->set_current(singles); });
    check(contains(e, "-inf") && same_current(z->get_current(), z_twin->get_current()),
          "W6 a partition with log-score -inf (max_parents = 0): an error, state unchanged",
          e.substr(0, 90));
}

// A model with seed `seed` after `burn` sweeps and then until its partition
// has two or more elements (at most 5000 more sweeps).
std::unique_ptr<OrderMCMCBN> stepped_model(int seed, int burn) {
    auto m = make_model(seed);
    m->step(burn);
    for (int s = 0; s < 5000 && value_of(m->get_current(), "order_party").n_elem < 2; ++s)
        m->step(1);
    return m;
}

void W7_history_is_one_based() {
    std::printf("\n--- W7: get_history() 'order' is 1-based, as in get_current() ---\n");
    const state_map saved = stepped_model(41, 300)->get_current();
    // keep_history: one row per sweep; a partition is loaded before sweep 201.
    auto m = make_model(17, 3, arma::vec(), true);
    std::vector<arma::vec> orders, dags;
    std::vector<double> scores;
    std::string e;
    for (int s = 0; s < 400; ++s) {
        if (s == 200) e = error_of([&] { m->set_current(partition_of(saved)); });
        m->step(1);
        const state_map cur = m->get_current();
        orders.push_back(value_of(cur, "order"));
        scores.push_back(value_of(cur, "log_score")[0]);
        dags.push_back(value_of(cur, "sampled_DAG"));
    }
    const history_map h = m->get_history();
    const auto ho = h.find("order"), hl = h.find("order_log_score"),
               hd = h.find("order_sampled_DAG");
    const bool shape = ho != h.end() && hl != h.end() && hd != h.end()
        && ho->second.n_rows == orders.size() && ho->second.n_cols == kN
        && hl->second.n_rows == orders.size() && hl->second.n_cols == 1
        && hd->second.n_rows == orders.size() && hd->second.n_cols == kN * kN;
    int bad_order = 0, bad_score = 0, bad_dag = 0;
    for (std::size_t t = 0; shape && t < orders.size(); ++t) {
        const arma::vec row = ho->second.row(t).t();
        if (!same_bits(row, orders[t])) ++bad_order;
        if (!same_bits(arma::vec{hl->second(t, 0)}, arma::vec{scores[t]})) ++bad_score;
        const arma::vec drow = hd->second.row(t).t();
        if (!same_bits(drow, dags[t])) ++bad_dag;
    }
    check(e.empty() && shape && bad_order == 0,
          "W7 keep_history: row t of 'order' equals get_current()$order (1-based) after "
          "sweep t, before and after a load",
          e + "rows = " + std::to_string(shape ? ho->second.n_rows : 0)
          + ", mismatching rows = " + std::to_string(bad_order));
    check(shape && bad_score == 0 && bad_dag == 0 && h.size() == 3,
          "W7 keep_history: the keys are order, order_log_score and order_sampled_DAG, "
          "and rows t of order_log_score and order_sampled_DAG are get_current()$log_score "
          "and get_current()$sampled_DAG (column-major) after sweep t",
          "mismatching rows = " + std::to_string(bad_score) + " (log_score), "
          + std::to_string(bad_dag) + " (sampled_DAG)");
    // keep_history off: one row, the current node sequence.
    auto q = make_model(17);
    auto single_row_ok = [&]() {
        const history_map hq = q->get_history();
        const auto it = hq.find("order");
        if (it == hq.end() || it->second.n_rows != 1) return false;
        const arma::vec row = it->second.row(0).t();
        return same_bits(row, value_of(q->get_current(), "order"));
    };
    int bad_single = single_row_ok() ? 0 : 1;               // construction
    q->step(150);
    if (!single_row_ok()) ++bad_single;                     // after sweeps
    const std::string eq = error_of([&] { q->set_current(partition_of(saved)); });
    if (!eq.empty() || !single_row_ok()) ++bad_single;      // after a load
    check(bad_single == 0,
          "W7 keep_history off: the single 'order' row is get_current()$order at "
          "construction, after sweeps and after a load",
          "mismatches = " + std::to_string(bad_single) + " of 3 " + eq);
}

using wrapper_access = kernel_control_mixin<OrderMCMCBN>;

// The keys of the wrapper's shared data that do not hold the state of
// get_current(): order (0-based there), order_party, order_sampled_DAG
// (row-major there: element i*n + j; get_current() is column-major) and
// order_log_score. Empty when all four hold it.
std::string shared_mismatch(const OrderMCMCBN& m) {
    const shared_data_t& d = wrapper_access::composite_of(m).data();
    const state_map cur = m.get_current();
    std::string bad;
    for (const char* k : {"order", "order_party", "order_sampled_DAG", "order_log_score"})
        if (!d.has(k)) bad += std::string(" missing ") + k;
    if (!bad.empty()) return bad;
    if (!same_bits(d.get("order"), value_of(cur, "order") - 1.0)) bad += " order";
    if (!same_bits(d.get("order_party"), value_of(cur, "order_party"))) bad += " order_party";
    const arma::vec& dag = d.get("order_sampled_DAG");
    const arma::vec a = value_of(cur, "sampled_DAG");
    bool dag_ok = dag.n_elem == kN * kN && a.n_elem == kN * kN;
    for (std::size_t i = 0; dag_ok && i < kN; ++i)
        for (std::size_t j = 0; j < kN; ++j)
            if (dag[i * kN + j] != a[i + j * kN]) dag_ok = false;
    if (!dag_ok) bad += " order_sampled_DAG";
    if (!same_bits(d.get("order_log_score"), value_of(cur, "log_score"))) bad += " order_log_score";
    return bad;
}

// Is the composite's current() get_current()$order - 1?
bool composite_current_ok(const OrderMCMCBN& m) {
    return same_bits(wrapper_access::composite_of(m).current(),
                     value_of(m.get_current(), "order") - 1.0);
}

void W8_composite_after_loads() {
    std::printf("\n--- W8: the wrapper's composite at construction and after loads ---\n");
    auto m = make_model(29);
    const std::string s0 = shared_mismatch(*m);
    check(s0.empty() && composite_current_ok(*m),
          "W8 at construction, the shared data keys and the composite's current() hold the "
          "state of get_current()", s0);
    m->step(300);
    // The reversed node sequence as single-node elements: a partition with
    // another node sequence and a finite log-score.
    state_map singles;
    singles["order"] = arma::reverse(value_of(m->get_current(), "order"));
    singles["order_party"] = arma::vec(kN, arma::fill::ones);
    (void)wrapper_access::composite_of(*m).current();   // the composite stores the concatenation
    const std::string e1 = error_of([&] { m->set_current(singles); });
    const bool loaded = e1.empty()
        && same_bits(value_of(m->get_current(), "order"), singles["order"]);
    check(loaded && composite_current_ok(*m),
          "W8 after a load, the composite's current() is the loaded node sequence (current() "
          "was called before the load)", e1);
    const std::string s1 = shared_mismatch(*m);
    check(loaded && s1.empty(), "W8 after a load, the shared data keys hold the loaded partition", s1);
    // 'order' alone, the current node sequence: no change.
    state_map seq;
    seq["order"] = value_of(m->get_current(), "order");
    (void)wrapper_access::composite_of(*m).current();
    const std::string e2 = error_of([&] { m->set_current(seq); });
    const std::string s2 = shared_mismatch(*m);
    check(e2.empty() && s2.empty() && composite_current_ok(*m),
          "W8 after 'order' alone (no change), the shared data keys and the composite's "
          "current() hold the state", e2 + s2);
}
}  // namespace

int main() {
    std::printf("=== test_fix_order-mcmc_wrapper ===\n");
    W1_current_is_the_partition();
    W2_round_trip_into_stepped_model();
    W3_round_trip_into_new_model();
    W4_order_alone();
    W5_initial_order();
    W6_invalid_input();
    W7_history_is_one_based();
    W8_composite_after_loads();
    std::printf("\n=== %d PASS / %d FAIL ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

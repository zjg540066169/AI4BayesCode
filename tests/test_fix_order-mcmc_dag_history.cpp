// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_order-mcmc_dag_history.cpp -- order_mcmc_block stores the DAG
//  that step() draws in each stored sweep (in a held sweep of a frozen
//  block, the held DAG), and the OrderMCMCBN wrapper
//  (examples/OrderMCMCBN.cpp, standalone C++ build) returns the stored DAGs
//  from get_history() and predict_at().
//
//  Block, method = partition and method = order, n = 5:
//    H1  keep_history, in a composite_block: after T sweeps, get_history()
//        has the keys order, order_log_score and order_sampled_DAG, each
//        with T rows, and history_size() is T for the block and the
//        composite; row t of order_sampled_DAG (element i*n + j = 1 when j
//        is a parent of i) and masks t*n .. t*n + n - 1 of
//        sampled_dag_history() equal sampled_dag() after sweep t, rows t of
//        order and order_log_score equal current() and current_log_score()
//        after sweep t, and the DAG of row t fits the node sequence of row t
//        (parents before the node in method = order, after it in method =
//        partition)
//    H2  the history draws no random numbers: a block with keep_history and
//        a block without, stepped with generators of the same seed, have
//        equal order, log-score and sampled DAG after each sweep, and the
//        two generators give the same next number
//    H3  frozen in a composite: each held sweep appends a row that repeats
//        the held order, log-score and DAG (sampled_dag()); after the
//        unfreeze the rows are the sweeps again, and every buffer holds one
//        entry per composite sweep
//    H4  clear_history() (clear_history_contract.hpp), with the masks of
//        sampled_dag_history() as an extra per-draw buffer; right after the
//        call get_history() has the single key order (one row) and
//        sampled_dag_history() is empty
//  Wrapper OrderMCMCBN (method = partition):
//    P1  keep_history: predict_at({}) has one row per stored sweep and n*n
//        columns, row t equal to get_current()$sampled_DAG after sweep t and
//        to row t of get_history()$order_sampled_DAG; predict_at({}, false)
//        equals predict_at({}); predict_at({}, true) is one row,
//        get_current()$sampled_DAG; get_current() and get_history() are the
//        same after these calls, and the next 100 sweeps equal those of a
//        twin model without them
//    P2  keep_history off: predict_at({}) and predict_at({}, true) are one
//        row, get_current()$sampled_DAG, at construction and after sweeps,
//        and get_history() has no order_sampled_DAG
//    P3  keep_history: at construction and right after clear_history(),
//        predict_at({}) is one row, get_current()$sampled_DAG; the
//        clear_history() contract (clear_history_contract.hpp) holds for
//        get_history() and predict_at({})
//    P4  a non-empty new_data is an error
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/order_mcmc_block.hpp"

// The example's own int main() is renamed; this file provides main().
#define main ordermcmcbn_example_main
#include "../examples/OrderMCMCBN.cpp"
#undef main

#include "clear_history_contract.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

namespace ab  = AI4BayesCode;
namespace chk = clear_history_check;

constexpr int K1 = 40, K2 = 30;
const std::size_t kN = 5;

// A check counted by clear_history_check::finish(); `detail` is appended to
// the printed line.
void check(bool ok, const std::string& what, const std::string& detail = "") {
    chk::check(ok, detail.empty() ? what : what + " -- " + detail);
}

// Binary chain X0 -> X1 -> ...; each node copies its parent w.p. 1 - flip.
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
    static const arma::mat X = chain_data(1500, kN, 0.2, 20261125u);
    return X;
}

ab::order_mcmc_block_config make_cfg(bool partition) {
    ab::order_mcmc_block_config c;
    c.name = "order";
    c.data = arma::conv_to<arma::imat>::from(data());
    c.cardinalities = arma::uvec(kN, arma::fill::value(2));
    c.structure_prior = ab::dag_prior::FK_EQ2;
    c.max_parents = 3;
    c.init_rng_seed = 9;
    c.method = partition ? ab::order_mcmc_block_config::method_t::partition
                         : ab::order_mcmc_block_config::method_t::order;
    return c;
}

std::string label(const char* id, bool partition) {
    return std::string(id) + " [" + (partition ? "partition" : "order") + "] ";
}

// `dag` (n parent masks) as a row of n*n values: element i*n + j (row_major)
// or i + j*n (column-major) is 1 when j is a parent of i.
arma::rowvec dag_flat(const std::vector<std::uint64_t>& dag, bool row_major) {
    const std::size_t n = dag.size();
    arma::rowvec out(n * n, arma::fill::zeros);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (dag[i] & (1ULL << j)) out[row_major ? i * n + j : i + j * n] = 1.0;
    return out;
}

bool same_row(const arma::mat& M, std::size_t r, const arma::rowvec& v) {
    return r < M.n_rows && chk::same_bits(arma::mat(M.row(r)), arma::mat(v));
}

// Masks t*n .. t*n + n - 1 of a buffer of n masks per sweep.
std::vector<std::uint64_t> masks_of(const std::vector<std::uint64_t>& buf,
                                    std::size_t t, std::size_t n) {
    if ((t + 1) * n > buf.size()) return {};
    return std::vector<std::uint64_t>(buf.begin() + static_cast<long>(t * n),
                                      buf.begin() + static_cast<long>((t + 1) * n));
}

// The masks of a buffer of n masks per sweep as a (sweeps x n) matrix.
arma::mat masks_as_rows(const std::vector<std::uint64_t>& buf, std::size_t n) {
    const std::size_t T = buf.size() / n;
    arma::mat out(T, n);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t i = 0; i < n; ++i) out(t, i) = static_cast<double>(buf[t * n + i]);
    return out;
}

// Does `dag` fit the node sequence `seq` (0-based nodes as doubles): every
// parent of a node lies before it (parents_first) or after it in `seq`?
bool dag_fits_sequence(const std::vector<std::uint64_t>& dag, const arma::rowvec& seq,
                       bool parents_first) {
    const std::size_t n = dag.size();
    if (seq.n_elem != n) return false;
    std::vector<std::size_t> pos(n, 0);
    for (std::size_t p = 0; p < n; ++p) pos[static_cast<std::size_t>(seq[p])] = p;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (dag[i] & (1ULL << j))
                if (parents_first ? pos[j] >= pos[i] : pos[j] <= pos[i]) return false;
    return true;
}

// A composite with one order_mcmc_block; `blk` receives the block.
std::shared_ptr<ab::composite_block> block_in_composite(bool partition,
                                                        ab::order_mcmc_block** blk) {
    auto root = std::make_shared<ab::composite_block>("root");
    auto b = std::make_unique<ab::order_mcmc_block>(make_cfg(partition));
    *blk = b.get();
    root->add_child(std::move(b));
    root->set_keep_history(true);
    return root;
}

// ---- block ------------------------------------------------------------------

void H1_rows_are_the_sweeps(bool partition) {
    std::printf("\n--- %s---\n", label("H1: one row per sweep", partition).c_str());
    ab::order_mcmc_block* blk = nullptr;
    auto root = block_in_composite(partition, &blk);
    std::mt19937_64 rng(101);
    const std::size_t T = 300;
    std::vector<std::vector<std::uint64_t>> dags;
    std::vector<arma::vec> orders;
    std::vector<double> scores;
    for (std::size_t t = 0; t < T; ++t) {
        root->step(rng);
        dags.push_back(blk->sampled_dag());
        orders.push_back(blk->current());
        scores.push_back(blk->current_log_score());
    }
    const std::set<std::vector<std::uint64_t>> distinct(dags.begin(), dags.end());
    check(distinct.size() > 1, label("H1", partition) + "precondition: the sampled DAG "
               "changes over the sweeps (" + std::to_string(distinct.size()) + " distinct DAGs)");

    const ab::history_map h = root->get_history();
    const auto ho = h.find("order"), hl = h.find("order_log_score"),
               hd = h.find("order_sampled_DAG");
    const bool keys = h.size() == 3 && ho != h.end() && hl != h.end() && hd != h.end();
    const bool shape = keys && ho->second.n_rows == T && ho->second.n_cols == kN &&
                       hl->second.n_rows == T && hl->second.n_cols == 1 &&
                       hd->second.n_rows == T && hd->second.n_cols == kN * kN;
    const std::vector<std::uint64_t>& buf = blk->sampled_dag_history();
    check(shape && blk->history_size() == T && root->history_size() == T &&
               buf.size() == T * kN,
               label("H1", partition) + "keys order, order_log_score, order_sampled_DAG with "
               "T rows; history_size() T; sampled_dag_history() T*n masks",
               "T = " + std::to_string(T) + ", buffer = " + std::to_string(buf.size()));
    int bad_dag = 0, bad_mask = 0, bad_order = 0, bad_score = 0, bad_fit = 0;
    for (std::size_t t = 0; shape && t < T; ++t) {
        if (!same_row(hd->second, t, dag_flat(dags[t], true))) ++bad_dag;
        if (masks_of(buf, t, kN) != dags[t]) ++bad_mask;
        if (!same_row(ho->second, t, orders[t].t())) ++bad_order;
        if (!chk::same_bits(arma::mat(1, 1, arma::fill::value(hl->second(t, 0))),
                            arma::mat(1, 1, arma::fill::value(scores[t])))) ++bad_score;
        if (!dag_fits_sequence(dags[t], ho->second.row(t), !partition)) ++bad_fit;
    }
    check(shape && bad_dag == 0 && bad_mask == 0,
               label("H1", partition) + "row t of order_sampled_DAG and masks t of "
               "sampled_dag_history() are sampled_dag() after sweep t",
               "mismatching rows = " + std::to_string(bad_dag) + ", masks = " +
               std::to_string(bad_mask));
    check(shape && bad_order == 0 && bad_score == 0 && bad_fit == 0,
               label("H1", partition) + "rows t of order and order_log_score are the state "
               "after sweep t, and DAG row t fits node sequence row t",
               "order = " + std::to_string(bad_order) + ", log-score = " +
               std::to_string(bad_score) + ", DAG/sequence = " + std::to_string(bad_fit));
}

void H2_no_random_draws(bool partition) {
    std::printf("\n--- %s---\n", label("H2: the history draws no random numbers",
                                       partition).c_str());
    ab::order_mcmc_block with(make_cfg(partition)), without(make_cfg(partition));
    with.set_keep_history(true);
    std::mt19937_64 rw(202), ro(202);
    const ab::block_context ctx;
    int bad = 0;
    const int T = 300;
    for (int s = 0; s < T; ++s) {
        with.set_context(ctx);    with.step(rw);
        without.set_context(ctx); without.step(ro);
        if (arma::any(with.order() != without.order()) ||
            with.current_log_score() != without.current_log_score() ||
            with.sampled_dag() != without.sampled_dag()) ++bad;
    }
    const bool next_same = rw() == ro();
    check(bad == 0 && next_same &&
               with.sampled_dag_history().size() == static_cast<std::size_t>(T) * kN &&
               without.sampled_dag_history().empty(),
               label("H2", partition) + "keep_history on and off: equal state after each "
               "sweep and equal next generator output",
               "mismatching sweeps = " + std::to_string(bad) +
               (next_same ? "" : ", next generator outputs differ"));
}

void H3_held_rows(bool partition) {
    std::printf("\n--- %s---\n", label("H3: held rows of a frozen block", partition).c_str());
    ab::order_mcmc_block* blk = nullptr;
    auto root = block_in_composite(partition, &blk);
    std::mt19937_64 rng(303);
    const std::size_t free1 = 120, held = 15, free2 = 20;
    for (std::size_t s = 0; s < free1; ++s) root->step(rng);
    const arma::vec held_order = blk->current();
    const double held_score = blk->current_log_score();
    const std::vector<std::uint64_t> held_dag = blk->sampled_dag();
    root->freeze(std::vector<std::string>{"order"}, true);
    for (std::size_t s = 0; s < held; ++s) root->step(rng);
    root->unfreeze_all();
    std::vector<std::vector<std::uint64_t>> after;
    for (std::size_t s = 0; s < free2; ++s) {
        root->step(rng);
        after.push_back(blk->sampled_dag());
    }
    const std::size_t T = free1 + held + free2;
    const ab::history_map h = root->get_history();
    const arma::mat& H = h.at("order");
    const arma::mat& L = h.at("order_log_score");
    const arma::mat& D = h.at("order_sampled_DAG");
    const std::vector<std::uint64_t>& buf = blk->sampled_dag_history();
    const bool sizes = H.n_rows == T && L.n_rows == T && D.n_rows == T &&
                       buf.size() == T * kN && root->history_size() == T;
    int bad_held = 0, bad_after = 0;
    for (std::size_t r = free1; sizes && r < free1 + held; ++r) {
        if (!same_row(H, r, held_order.t()) || L(r, 0) != held_score ||
            !same_row(D, r, dag_flat(held_dag, true)) || masks_of(buf, r, kN) != held_dag)
            ++bad_held;
    }
    for (std::size_t s = 0; sizes && s < free2; ++s)
        if (!same_row(D, free1 + held + s, dag_flat(after[s], true))) ++bad_after;
    check(sizes && bad_held == 0 && bad_after == 0,
               label("H3", partition) + "the " + std::to_string(held) + " held rows repeat "
               "the held order, log-score and DAG; the rows after the unfreeze are the "
               "sweeps; one entry per composite sweep",
               "rows = " + std::to_string(D.n_rows) + " (T = " + std::to_string(T) +
               "), bad held rows = " + std::to_string(bad_held) + ", bad later rows = " +
               std::to_string(bad_after));
}

void H4_clear_history(bool partition) {
    std::printf("\n--- %s---\n", label("H4: clear_history()", partition).c_str());
    {
        ab::order_mcmc_block* blk = nullptr;
        auto root = block_in_composite(partition, &blk);
        std::mt19937_64 rng(404);
        for (int s = 0; s < K1; ++s) root->step(rng);
        root->clear_history();
        const ab::history_map h = root->get_history();
        const auto it = h.find("order");
        check(h.size() == 1 && it != h.end() && it->second.n_rows == 1 &&
                   blk->sampled_dag_history().empty() && root->history_size() == 1,
                   label("H4", partition) + "right after clear_history(): get_history() is "
                   "{order: 1 row}, sampled_dag_history() is empty, history_size() is 1");
    }
    auto make = [partition]() {
        ab::order_mcmc_block* blk = nullptr;
        auto root = block_in_composite(partition, &blk);
        return chk::composite_model(root, 505, [blk](chk::buffers& b) {
            b.mats["order: sampled_dag_history() masks"] =
                masks_as_rows(blk->sampled_dag_history(), kN);
        });
    };
    chk::run(label("H4", partition) + "clear_history_contract", make, K1, K2);
}

// ---- wrapper ----------------------------------------------------------------

std::unique_ptr<OrderMCMCBN> make_model(int seed, bool keep_history) {
    return std::make_unique<OrderMCMCBN>(data(), arma::vec(kN, arma::fill::value(2.0)),
                                         "fk_eq2", 1.0, 3, 20, 4000, 10.0, 0.5,
                                         arma::vec(), seed, keep_history);
}

const ab::state_map kNoData{};

bool same_map(const ab::state_map& a, const ab::state_map& b) {
    if (a.size() != b.size()) return false;
    for (const auto& kv : a) {
        const auto it = b.find(kv.first);
        if (it == b.end() || !chk::same_bits(arma::mat(kv.second), arma::mat(it->second)))
            return false;
    }
    return true;
}

bool same_map(const ab::history_map& a, const ab::history_map& b) {
    if (a.size() != b.size()) return false;
    for (const auto& kv : a) {
        const auto it = b.find(kv.first);
        if (it == b.end() || !chk::same_bits(kv.second, it->second)) return false;
    }
    return true;
}

// The single "sampled_DAG" entry of a predict_at() result (empty when the
// result has other keys).
arma::mat dag_entry(const ab::history_map& p) {
    const auto it = p.find("sampled_DAG");
    return (p.size() == 1 && it != p.end()) ? it->second : arma::mat();
}

void P1_history_mode() {
    std::printf("\n--- P1: OrderMCMCBN, keep_history: one row per stored sweep ---\n");
    auto m = make_model(31, true), twin = make_model(31, true);
    const std::size_t T = 250;
    std::vector<arma::rowvec> cur;
    for (std::size_t t = 0; t < T; ++t) {
        m->step(1);
        twin->step(1);
        cur.push_back(m->get_current().at("sampled_DAG").t());
    }
    const ab::state_map cur_before = m->get_current();
    const ab::history_map hist_before = m->get_history();
    const arma::mat P  = dag_entry(m->predict_at(kNoData));
    const arma::mat Pf = dag_entry(m->predict_at(kNoData, false));
    const arma::mat Pt = dag_entry(m->predict_at(kNoData, true));
    int bad = 0;
    for (std::size_t t = 0; t < T; ++t)
        if (!same_row(P, t, cur[t])) ++bad;
    const auto hd = hist_before.find("order_sampled_DAG");
    check(P.n_rows == T && P.n_cols == kN * kN && bad == 0,
               "P1 predict_at({}): T x n*n, row t is get_current()$sampled_DAG after sweep t",
               "rows = " + std::to_string(P.n_rows) + ", mismatching rows = " +
               std::to_string(bad));
    check(hd != hist_before.end() && chk::same_bits(P, hd->second),
               "P1 predict_at({}) equals get_history()$order_sampled_DAG");
    check(chk::same_bits(Pf, P), "P1 predict_at({}, false) equals predict_at({})");
    check(Pt.n_rows == 1 && same_row(Pt, 0, cur.back()),
               "P1 predict_at({}, true): one row, get_current()$sampled_DAG");
    check(same_map(m->get_current(), cur_before) &&
               same_map(m->get_history(), hist_before),
               "P1 get_current() and get_history() are the same after the predict_at calls");
    int bad_next = 0;
    for (int s = 0; s < 100; ++s) {
        m->step(1);
        twin->step(1);
        if (!same_map(m->get_current(), twin->get_current())) ++bad_next;
    }
    check(bad_next == 0 && same_map(m->get_history(), twin->get_history()),
               "P1 the next 100 sweeps equal those of a twin without the predict_at calls",
               "mismatching sweeps = " + std::to_string(bad_next));
}

void P2_single_draw_mode() {
    std::printf("\n--- P2: OrderMCMCBN, keep_history off: one row, the current DAG ---\n");
    auto m = make_model(37, false);
    int bad = 0;
    for (int k = 0; k < 3; ++k) {
        if (k > 0) m->step(150);
        const arma::rowvec c = m->get_current().at("sampled_DAG").t();
        const arma::mat P  = dag_entry(m->predict_at(kNoData));
        const arma::mat Pt = dag_entry(m->predict_at(kNoData, true));
        if (P.n_rows != 1 || !same_row(P, 0, c) || Pt.n_rows != 1 || !same_row(Pt, 0, c)) ++bad;
    }
    check(bad == 0 && m->get_history().count("order_sampled_DAG") == 0,
               "P2 predict_at({}) and predict_at({}, true) are get_current()$sampled_DAG at "
               "construction and after sweeps; get_history() has no order_sampled_DAG",
               "mismatches = " + std::to_string(bad) + " of 3");
}

chk::model wrapper_case() {
    std::shared_ptr<OrderMCMCBN> w(make_model(47, true).release());
    OrderMCMCBN* m = w.get();
    chk::model out;
    out.step    = [m](int n) { m->step(n); };
    out.clear   = [m] { m->clear_history(); };
    out.history = [m](chk::buffers& b) {
        for (auto& kv : m->get_history()) b.mats[kv.first] = kv.second;
    };
    out.extra   = [m](chk::buffers& b) {
        b.mats["predict_at({}): sampled_DAG"] = dag_entry(m->predict_at(kNoData));
    };
    out.owners.push_back(w);
    return out;
}

void P3_no_stored_sweep() {
    std::printf("\n--- P3: OrderMCMCBN, keep_history, no stored sweep ---\n");
    auto m = make_model(43, true);
    auto current_dag_only = [&]() {
        const arma::mat P = dag_entry(m->predict_at(kNoData));
        return P.n_rows == 1 &&
               same_row(P, 0, m->get_current().at("sampled_DAG").t());
    };
    const bool at_construction = current_dag_only();
    m->step(60);
    m->clear_history();
    const bool after_clear = current_dag_only();
    const ab::history_map h = m->get_history();
    check(at_construction && after_clear && h.size() == 1 && h.count("order") == 1,
               "P3 at construction and right after clear_history(): predict_at({}) is one "
               "row, get_current()$sampled_DAG; get_history() is {order}");
    chk::run("P3 OrderMCMCBN: get_history() and predict_at({}) under clear_history()",
             wrapper_case, K1, K2);
}

void P4_new_data_rejected() {
    std::printf("\n--- P4: OrderMCMCBN, non-empty new_data ---\n");
    auto m = make_model(53, true);
    m->step(10);
    ab::state_map nd;
    nd["X"] = arma::vec{1.0, 2.0};
    int thrown = 0;
    for (int k = 0; k < 2; ++k) {
        try {
            if (k == 0) (void)m->predict_at(nd);
            else        (void)m->predict_at(nd, true);
        } catch (const std::exception&) {
            ++thrown;
        }
    }
    check(thrown == 2, "P4 predict_at with a non-empty new_data is an error (both forms)",
               "errors = " + std::to_string(thrown) + " of 2");
}

}  // namespace

int main() {
    std::printf("=== test_fix_order-mcmc_dag_history ===\n");
    for (bool partition : {true, false}) {
        H1_rows_are_the_sweeps(partition);
        H2_no_random_draws(partition);
        H3_held_rows(partition);
        H4_clear_history(partition);
    }
    P1_history_mode();
    P2_single_draw_mode();
    P3_no_stored_sweep();
    P4_new_data_rejected();
    return chk::finish("test_fix_order-mcmc_dag_history");
}

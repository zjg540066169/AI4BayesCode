// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  clear_history_contract.hpp -- the clear_history() check shared by
//  test_fix_clear_history.cpp, test_fix_clear_history_genbart.cpp and
//  test_fix_clear_history_softbart.cpp. The kernel headers of bart_block,
//  genbart_block and softbart_block share include guards, so two of these
//  blocks cannot be compiled in one translation unit; each has a file.
//
//  A model is built three times from the same seeds:
//      A:      k1 sweeps, clear_history(), [between], k2 sweeps
//      B, B2:  k1 sweeps, [between], k2 sweeps
//  where `between` (optional, e.g. a freeze) is applied to every run. Checks:
//   - B2 equals B, so the model is reproducible for a fixed seed and a
//     difference between A and B comes from clear_history();
//   - right after clear_history(), every get_history() entry has 1 row and,
//     for a model built on a composite_block, history_size() is 1 for the
//     composite and for each child;
//   - after the k2 sweeps, every per-draw buffer of A (each get_history()
//     entry and each buffer the model adds, such as a BART kernel's stored
//     forests) has k2 entries and equals, bit for bit, the last k2 entries of
//     the same buffer of B, which has k1 + k2 entries; the composite's
//     history_size() is k2;
//   - values that are not per-draw rows (fixed) are bit for bit equal in A
//     and B; buffers of random prediction draws (rows_only) have k2 rows.
// ============================================================================
#ifndef AI4BAYESCODE_TEST_CLEAR_HISTORY_CONTRACT_HPP
#define AI4BAYESCODE_TEST_CLEAR_HISTORY_CONTRACT_HPP

#include <armadillo>
#include "AI4BayesCode/composite_block.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace clear_history_check {

inline int& failures() { static int n = 0; return n; }

inline void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures();
}

/// The per-draw buffers of a model, one entry (row) per stored draw.
struct buffers {
    std::map<std::string, arma::mat> mats;
    std::map<std::string, std::vector<std::string>> strings;
    /// Random prediction draws: only the row count is compared.
    std::map<std::string, arma::mat> rows_only;
    /// Values that are not per-draw rows; equal in A and B.
    std::map<std::string, arma::mat> fixed;
};

struct model {
    std::function<void(int)> step;              // runs n sweeps
    std::function<void()>    clear;             // clear_history()
    std::function<void(buffers&)> history;      // get_history() entries
    std::function<void(buffers&)> extra;        // other buffers (optional)
    std::function<void()>    between;           // optional, as in the file header
    /// history_size() of the composite, then of each child (optional).
    std::function<std::vector<std::size_t>()> sizes;
    std::vector<std::shared_ptr<void>> owners;  // keep the objects alive
};

/// A composite_block stepped with std::mt19937_64(seed); `extra` adds the
/// buffers get_history() does not return.
inline model composite_model(std::shared_ptr<AI4BayesCode::composite_block> root,
                             std::uint64_t seed,
                             std::function<void(buffers&)> extra = nullptr) {
    auto rng = std::make_shared<std::mt19937_64>(seed);
    AI4BayesCode::composite_block* c = root.get();
    model m;
    m.step    = [c, rng](int n) { for (int i = 0; i < n; ++i) c->step(*rng); };
    m.clear   = [c] { c->clear_history(); };
    m.history = [c](buffers& b) {
        for (auto& kv : c->get_history()) b.mats[kv.first] = kv.second;
    };
    m.extra   = std::move(extra);
    m.sizes   = [c] {
        std::vector<std::size_t> s{c->history_size()};
        for (std::size_t i = 0; i < c->n_children(); ++i)
            s.push_back(c->child(i).history_size());
        return s;
    };
    m.owners.push_back(std::move(root));
    return m;
}

inline bool same_bits(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           (a.n_elem == 0 ||
            std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0);
}

inline bool same_buffers(const buffers& x, const buffers& y) {
    auto same_map = [](const std::map<std::string, arma::mat>& p,
                       const std::map<std::string, arma::mat>& q) {
        if (p.size() != q.size()) return false;
        for (const auto& kv : p) {
            auto it = q.find(kv.first);
            if (it == q.end() || !same_bits(kv.second, it->second)) return false;
        }
        return true;
    };
    return same_map(x.mats, y.mats) && same_map(x.fixed, y.fixed) &&
           x.strings == y.strings;
}

inline buffers collect(const model& m) {
    buffers b;
    m.history(b);
    if (m.extra) m.extra(b);
    return b;
}

inline std::string shape(const arma::mat& x) {
    return std::to_string(x.n_rows) + "x" + std::to_string(x.n_cols);
}

inline void run_checks(const std::function<model()>& make, int k1, int k2);

/// Runs the check of the file header on the model built by `make`; an
/// exception is a failed check.
inline void run(const std::string& label, const std::function<model()>& make,
                int k1, int k2) {
    std::printf("%s (k1 = %d, k2 = %d)\n", label.c_str(), k1, k2);
    try {
        run_checks(make, k1, k2);
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
}

inline void run_checks(const std::function<model()>& make, int k1, int k2) {
    const std::size_t n1 = static_cast<std::size_t>(k1);
    const std::size_t n2 = static_cast<std::size_t>(k2);

    // A. Every model is stepped to the end before the next one is built: the
    // BART-family kernels draw from one thread-local engine, which `make`
    // seeds.
    model A = make();
    A.step(k1);
    A.clear();
    {
        if (A.sizes) {
            bool ones = true;
            for (std::size_t s : A.sizes()) ones = ones && s == 1;
            check(ones, "right after clear_history(): history_size() is 1 for "
                        "the composite and each child");
        }
        buffers h;
        A.history(h);
        bool rows1 = !h.mats.empty();
        for (const auto& kv : h.mats) rows1 = rows1 && kv.second.n_rows == 1;
        check(rows1, "right after clear_history(): every get_history() entry has 1 row");
    }
    if (A.between) A.between();
    A.step(k2);
    const buffers a = collect(A);
    const std::size_t a_size = A.sizes ? A.sizes().front() : n2;

    auto run_plain = [&]() {
        model M = make();
        M.step(k1);
        if (M.between) M.between();
        M.step(k2);
        return M;
    };
    model B = run_plain();
    const buffers b = collect(B);
    model B2 = run_plain();
    check(same_buffers(b, collect(B2)),
          "control: two runs without clear_history() are identical");

    for (const auto& kv : b.mats) {
        auto it = a.mats.find(kv.first);
        if (it == a.mats.end()) {
            check(false, "'" + kv.first + "': missing after clear_history()");
            continue;
        }
        const arma::mat& x = it->second;
        const arma::mat& y = kv.second;
        const bool ok = y.n_rows == n1 + n2 && x.n_rows == n2 &&
                        same_bits(x, y.rows(n1, n1 + n2 - 1));
        check(ok, "'" + kv.first + "': A " + shape(x) + ", B " + shape(y) +
                  (ok ? ", A = last rows of B" : ""));
    }
    for (const auto& kv : a.mats)
        if (!b.mats.count(kv.first))
            check(false, "'" + kv.first + "': returned only after clear_history()");
    for (const auto& kv : b.strings) {
        auto it = a.strings.find(kv.first);
        const bool ok = it != a.strings.end() && kv.second.size() == n1 + n2 &&
                        it->second.size() == n2 &&
                        std::equal(it->second.begin(), it->second.end(),
                                   kv.second.begin() + static_cast<long>(n1));
        check(ok, "'" + kv.first + "': A " +
                  std::to_string(it == a.strings.end() ? 0 : it->second.size()) +
                  " entries, B " + std::to_string(kv.second.size()) +
                  (ok ? ", A = last entries of B" : ""));
    }
    for (const auto& kv : b.rows_only) {
        auto it = a.rows_only.find(kv.first);
        const bool ok = it != a.rows_only.end() && it->second.n_rows == n2 &&
                        kv.second.n_rows == n1 + n2;
        check(ok, "'" + kv.first + "' (random draws): A " +
                  (it == a.rows_only.end() ? std::string("none") : shape(it->second)) +
                  ", B " + shape(kv.second));
    }
    for (const auto& kv : b.fixed) {
        auto it = a.fixed.find(kv.first);
        check(it != a.fixed.end() && same_bits(it->second, kv.second),
              "'" + kv.first + "': equal in A and B");
    }
    if (A.sizes)
        check(a_size == n2, "history_size() after the k2 sweeps is " +
                            std::to_string(a_size) + " (k2 = " +
                            std::to_string(k2) + ")");
}

inline int finish(const char* name) {
    std::printf("\n%s: %d failed check(s)\n", name, failures());
    return failures() == 0 ? 0 : 1;
}

}  // namespace clear_history_check

#endif  // AI4BAYESCODE_TEST_CLEAR_HISTORY_CONTRACT_HPP

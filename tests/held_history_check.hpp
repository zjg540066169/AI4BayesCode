// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  held_history_check.hpp -- the nested-freeze check shared by
//  test_fix_held_history.cpp, test_fix_held_history_genbart.cpp and
//  test_fix_held_history_softbart.cpp (one file per BART family, as for
//  clear_history_contract.hpp, whose model and buffers it uses).
//
//  The model is a composite_block whose children are a nested composite that
//  holds the block under test and a free sibling; model.between freezes the
//  nested composite. After k1 sweeps, between() and k2 sweeps:
//   - every per-draw buffer (each get_history() entry and each buffer the
//     model adds, such as a BART kernel's stored forests) has k1 + k2
//     entries, and history_size() is k1 + k2 for the root and each child;
//   - in every buffer except those named in free_keys (the free sibling),
//     entries k1 .. k1 + k2 - 1 equal entry k1 - 1 bit for bit: the frozen
//     subtree records one held entry per sweep of the root.
//  clear_history_check::run with the same between() compares the held entries
//  recorded right after clear_history(), when no entry is stored, with those
//  of a run without the call.
// ============================================================================
#ifndef AI4BAYESCODE_TEST_HELD_HISTORY_CHECK_HPP
#define AI4BAYESCODE_TEST_HELD_HISTORY_CHECK_HPP

#include "clear_history_contract.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace held_history_check {

namespace chk = clear_history_check;

inline void run_checks(const std::function<chk::model()>& make, int k1, int k2,
                       const std::vector<std::string>& free_keys) {
    const std::size_t n1 = static_cast<std::size_t>(k1);
    const std::size_t n = static_cast<std::size_t>(k1 + k2);
    chk::model m = make();
    m.step(k1);
    m.between();
    m.step(k2);
    const chk::buffers b = chk::collect(m);
    auto is_free = [&](const std::string& key) {
        return std::find(free_keys.begin(), free_keys.end(), key) != free_keys.end();
    };
    const std::string want = ", expected " + std::to_string(n);
    for (const auto& kv : b.mats) {
        const arma::mat& x = kv.second;
        const bool rows_ok = x.n_rows == n;
        bool held_ok = true;
        if (rows_ok && !is_free(kv.first))
            for (std::size_t r = n1; held_ok && r < n; ++r)
                held_ok = chk::same_bits(arma::mat(x.row(r)), arma::mat(x.row(n1 - 1)));
        std::string what = "'" + kv.first + "': " + chk::shape(x);
        if (!rows_ok) what += want + " rows";
        else if (!is_free(kv.first))
            what += held_ok ? ", rows k1.. = row k1 - 1" : ", held rows differ from row k1 - 1";
        chk::check(rows_ok && held_ok, what);
    }
    for (const auto& kv : b.strings) {
        const std::vector<std::string>& x = kv.second;
        const bool rows_ok = x.size() == n;
        bool held_ok = true;
        if (rows_ok && !is_free(kv.first))
            for (std::size_t r = n1; held_ok && r < n; ++r) held_ok = x[r] == x[n1 - 1];
        std::string what = "'" + kv.first + "': " + std::to_string(x.size()) + " entries";
        if (!rows_ok) what += want;
        else if (!is_free(kv.first))
            what += held_ok ? ", entries k1.. = entry k1 - 1"
                            : ", held entries differ from entry k1 - 1";
        chk::check(rows_ok && held_ok, what);
    }
    for (const auto& kv : b.rows_only)
        chk::check(kv.second.n_rows == n,
                   "'" + kv.first + "': " + chk::shape(kv.second) +
                   (kv.second.n_rows == n ? "" : want + " rows"));
    if (m.sizes) {
        bool all_n = true;
        std::string got;
        for (std::size_t s : m.sizes()) {
            all_n = all_n && s == n;
            got += (got.empty() ? "" : ", ") + std::to_string(s);
        }
        chk::check(all_n, "history_size() of the root and each child: " + got +
                          " (k1 + k2 = " + std::to_string(n) + ")");
    }
}

/// Runs the check of the file header; an exception is a failed check.
inline void run(const std::string& label, const std::function<chk::model()>& make,
                int k1, int k2, const std::vector<std::string>& free_keys) {
    std::printf("%s (k1 = %d, k2 = %d)\n", label.c_str(), k1, k2);
    try {
        run_checks(make, k1, k2, free_keys);
    } catch (const std::exception& e) {
        chk::check(false, std::string("exception: ") + e.what());
    }
}

}  // namespace held_history_check

#endif  // AI4BAYESCODE_TEST_HELD_HISTORY_CHECK_HPP

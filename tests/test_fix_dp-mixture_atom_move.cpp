// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_dp-mixture_atom_move.cpp
//
//  cluster_atom_block drives one univariate slice block per coordinate whose
//  log-density closure refers to the owning block. A moved block must sample
//  its own target: move construction, move assignment, std::swap and
//  vector growth are checked draw for draw against a freshly built block
//  with the same seed, and a moved-from block must be empty.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/cluster_atom_block.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace AI4BayesCode;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}

// Prior-only component target: mu ~ N(m, 0.1^2), lambda ~ Gamma(2, 1).
cluster_atom_block_config make_cfg(const std::string& name, double m) {
    cluster_atom_block_config cfg;
    cfg.name = name;
    cfg.K_trunc = 2;
    cfg.sub_params.push_back(joint_nuts_sub_param{"mu_" + name, 1u, joint_constraint::REAL});
    cfg.sub_params.push_back(joint_nuts_sub_param{"lam_" + name, 1u, joint_constraint::POSITIVE});
    cfg.initial_cat = arma::vec{m, 1.0, m, 1.0};
    cfg.log_density = [m](const arma::vec& th, std::size_t, const block_context&) {
        const double z = (th[0] - m) / 0.1;
        return -0.5 * z * z + std::log(th[1]) - th[1];
    };
    return cfg;
}

cluster_atom_block make_block(const std::string& name, double m) {
    return cluster_atom_block(make_cfg(name, m));
}

// All states of 500 sweeps with a fixed seed; empty if step() throws.
std::vector<double> draws(cluster_atom_block& b) {
    std::mt19937_64 rng(20261122u);
    std::vector<double> out;
    try {
        b.set_context({{"cluster_counts", arma::vec{0.0, 0.0}}});
        for (int s = 0; s < 500; ++s) {
            b.step(rng);
            out.insert(out.end(), b.current().begin(), b.current().end());
        }
    } catch (const std::exception& e) {
        std::printf("    step() threw: %s\n", e.what());
        out.clear();
    }
    return out;
}

// Vector growth moves the elements; it needs a move that is used in place
// of the (unsupported) copy.
template <class T>
bool vector_growth_ok(const std::vector<double>& ref_a, const std::vector<double>& ref_b) {
    if constexpr (std::is_copy_constructible<T>::value) {
        std::printf("    std::vector growth would copy the block, which does not compile\n");
        return false;
    } else {
        std::vector<T> v;
        v.reserve(1);
        v.emplace_back(make_cfg("A", -10.0));
        v.emplace_back(make_cfg("B", +10.0));
        return draws(v[0]) == ref_a && draws(v[1]) == ref_b;
    }
}

}  // namespace

int main() {
    std::printf("=== test_fix_dp-mixture_atom_move ===\n");
    auto fresh_a = make_block("A", -10.0);
    auto fresh_b = make_block("B", +10.0);
    const std::vector<double> ref_a = draws(fresh_a), ref_b = draws(fresh_b);
    check(!ref_a.empty() && !ref_b.empty() && ref_a != ref_b,
          "freshly built blocks sample their own targets");

    bool moves_ok = true;
    {
        auto a = make_block("A", -10.0);
        cluster_atom_block b(std::move(a));
        const bool ok = draws(b) == ref_a;
        check(ok, "move construction: same draws as a fresh block");
        moves_ok = moves_ok && ok;
        const bool empty = a.dim() == 0 && a.current().n_elem == 0;
        check(empty, "move construction: the moved-from block is empty (dim 0)");
        if (empty) {
            bool threw = false;
            std::mt19937_64 rng(1u);
            try {
                a.set_context({{"cluster_counts", arma::vec{0.0, 0.0}}});
                a.step(rng);
            } catch (const std::exception&) { threw = true; }
            check(threw, "move construction: step() of the moved-from block throws");
        }
    }
    {
        auto a = make_block("A", -10.0);
        auto b = make_block("B", +10.0);
        b = std::move(a);
        const bool ok = draws(b) == ref_a;
        check(ok, "move assignment: same draws as a fresh block");
        moves_ok = moves_ok && ok;
    }
    {
        auto a = make_block("A", -10.0);
        auto b = make_block("B", +10.0);
        std::swap(a, b);
        const bool ok = draws(a) == ref_b && draws(b) == ref_a;
        check(ok, "std::swap: each block samples the target it now holds");
        moves_ok = moves_ok && ok;
    }
    // A temporary moved into a heap block: its closures must not refer to the
    // destroyed temporary. Run only when the moves above are correct.
    if (moves_ok) {
        auto p = std::make_unique<cluster_atom_block>(make_block("A", -10.0));
        check(draws(*p) == ref_a, "make_unique from a temporary: same draws as a fresh block");
    } else {
        check(false, "make_unique from a temporary: not run, moves are broken");
    }
    check(vector_growth_ok<cluster_atom_block>(ref_a, ref_b),
          "std::vector growth: elements sample their own targets after reallocation");

    std::printf("\n=== SUMMARY: %d failed ===\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}

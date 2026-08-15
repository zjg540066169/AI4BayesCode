/*================================================================================
 *  test_predict_dag_depth.cpp
 *
 *  Locks the required propagation depth of the predict-DAG refresh order.
 *
 *  `shared_data_t::predict_downstream_of(changed)` walks the predict DAG from
 *  the replaced keys and returns every deterministic node that must be
 *  recomputed before Pass 2 draws the stochastic nodes. Any deterministic
 *  descendant left out of that list is NOT recomputed, so `predict_at` returns
 *  it -- and every stochastic node drawn from it -- at the TRAINING state, with
 *  no error and no NaN. That is a silently wrong posterior predictive.
 *
 *  These cases pin the contract: the returned order must contain EVERY
 *  deterministic descendant whose parents are all available, no matter how deep
 *  the chain or how few children the replaced key has.
 *
 *  Cases 3 and 4 are the two shipped-example topologies (BSplineRegression and
 *  HSGPRegression in history mode, where the per-draw replaced keys are the
 *  sampled parameters rather than the data inputs).
 *
 *  Copyright (C) 2026 AI4BayesCode
 *  SPDX-License-Identifier: GPL-3.0-or-later
 *================================================================================*/
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
#  define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
#  define ARMA_DONT_USE_WRAPPER
#endif

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  [FAIL] %s\n", what.c_str()); }
}

static bool contains(const std::vector<std::string>& v, const std::string& k) {
    return std::find(v.begin(), v.end(), k) != v.end();
}

static std::string join(const std::vector<std::string>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        s += v[i];
        if (i + 1 < v.size()) s += ", ";
    }
    return s + "]";
}

// Registers a key with a value and an identity-ish refresher, so the node is a
// genuine deterministic node (it has a refresher) rather than a bare value.
static void add_derived(AI4BayesCode::shared_data_t& d, const std::string& key,
                        const std::string& parent) {
    d.set(key, arma::vec{0.0});
    d.register_refresher(key, [parent](const AI4BayesCode::shared_data_t& s) {
        return arma::vec{s.get(parent)(0) + 1.0};
    });
}

int main() {
    std::printf("=== predict-DAG refresh-order depth ===\n\n");

    // ---- Case 1: linear chain, one child per level -------------------------
    // X -> A -> B -> C, all deterministic. Replacing X must recompute A, B, C.
    {
        std::printf("Case 1: linear chain X -> A -> B -> C (1 seed child)\n");
        AI4BayesCode::shared_data_t d;
        d.declare_data_input("X");
        d.set("X", arma::vec{1.0});
        add_derived(d, "A", "X");
        add_derived(d, "B", "A");
        add_derived(d, "C", "B");
        d.declare_predict_edges("X", {"A"});
        d.declare_predict_edges("A", {"B"});
        d.declare_predict_edges("B", {"C"});

        std::unordered_set<std::string> changed{"X"};
        const auto order = d.predict_downstream_of(changed);
        std::printf("        order = %s\n", join(order).c_str());
        check(contains(order, "A"), "depth 1 (A) recomputed");
        check(contains(order, "B"), "depth 2 (B) recomputed");
        check(contains(order, "C"), "depth 3 (C) recomputed");
    }

    // ---- Case 2: deeper chain ---------------------------------------------
    {
        std::printf("\nCase 2: linear chain of depth 5 (1 seed child)\n");
        AI4BayesCode::shared_data_t d;
        d.declare_data_input("X");
        d.set("X", arma::vec{1.0});
        const char* keys[] = {"n1", "n2", "n3", "n4", "n5"};
        add_derived(d, keys[0], "X");
        d.declare_predict_edges("X", {keys[0]});
        for (int i = 1; i < 5; ++i) {
            add_derived(d, keys[i], keys[i - 1]);
            d.declare_predict_edges(keys[i - 1], {keys[i]});
        }

        std::unordered_set<std::string> changed{"X"};
        const auto order = d.predict_downstream_of(changed);
        std::printf("        order = %s\n", join(order).c_str());
        for (int i = 0; i < 5; ++i)
            check(contains(order, keys[i]),
                  std::string("depth ") + std::to_string(i + 1) + " (" + keys[i] +
                  ") recomputed");
    }

    // ---- Case 3: BSplineRegression topology, history mode -------------------
    //   log_sds -> s -> f -> mu -> y_rep(stochastic)
    //   z       -> s
    //   Bsp_flat-> f ;  Intercept -> mu ;  log_sigma -> y_rep
    // History mode replaces the sampled parameters per draw, so `changed` is
    // e.g. {log_sds} or {z} -- one seed child (s) each.
    {
        std::printf("\nCase 3: BSplineRegression topology, per-draw replace\n");
        AI4BayesCode::shared_data_t d;
        d.set("log_sds", arma::vec{0.0});
        d.set("z", arma::vec{0.0});
        d.set("Bsp_flat", arma::vec{1.0});
        d.set("Intercept", arma::vec{0.0});
        d.set("log_sigma", arma::vec{0.0});
        add_derived(d, "s", "log_sds");
        add_derived(d, "f", "s");
        add_derived(d, "mu", "f");
        d.set("y_rep", arma::vec{0.0});
        d.register_stochastic_refresher(
            "y_rep", [](const AI4BayesCode::shared_data_t& s, std::mt19937_64&) {
                return arma::vec{s.get("mu")(0)};
            });
        d.declare_predict_edges("log_sds", {"s"});
        d.declare_predict_edges("z", {"s"});
        d.declare_predict_edges("s", {"f"});
        d.declare_predict_edges("Bsp_flat", {"f"});
        d.declare_predict_edges("f", {"mu"});
        d.declare_predict_edges("Intercept", {"mu"});
        d.declare_predict_edges("mu", {"y_rep"});
        d.declare_predict_edges("log_sigma", {"y_rep"});

        for (const char* seed : {"log_sds", "z"}) {
            std::unordered_set<std::string> changed{seed};
            const auto order = d.predict_downstream_of(changed);
            std::printf("        replace %-8s order = %s\n", seed,
                        join(order).c_str());
            check(contains(order, "s"),
                  std::string("replace ") + seed + ": s recomputed");
            check(contains(order, "f"),
                  std::string("replace ") + seed + ": f recomputed");
            check(contains(order, "mu"),
                  std::string("replace ") + seed +
                  ": mu recomputed (else y_rep is drawn from the training mu)");
        }
    }

    // ---- Case 4: HSGPRegression topology, history mode ----------------------
    //   {log_amp, log_ell, lambda} -> sqrt_spd -> f -> mu -> y_rep(stochastic)
    {
        std::printf("\nCase 4: HSGPRegression topology, per-draw replace\n");
        AI4BayesCode::shared_data_t d;
        for (const char* k : {"log_amp", "log_ell", "lambda", "z", "phi_flat",
                              "Intercept", "log_sigma"})
            d.set(k, arma::vec{0.0});
        add_derived(d, "sqrt_spd", "log_amp");
        add_derived(d, "f", "sqrt_spd");
        add_derived(d, "mu", "f");
        d.set("y_rep", arma::vec{0.0});
        d.register_stochastic_refresher(
            "y_rep", [](const AI4BayesCode::shared_data_t& s, std::mt19937_64&) {
                return arma::vec{s.get("mu")(0)};
            });
        d.declare_predict_edges("log_amp", {"sqrt_spd"});
        d.declare_predict_edges("log_ell", {"sqrt_spd"});
        d.declare_predict_edges("lambda", {"sqrt_spd"});
        d.declare_predict_edges("sqrt_spd", {"f"});
        d.declare_predict_edges("z", {"f"});
        d.declare_predict_edges("phi_flat", {"f"});
        d.declare_predict_edges("f", {"mu"});
        d.declare_predict_edges("Intercept", {"mu"});
        d.declare_predict_edges("mu", {"y_rep"});
        d.declare_predict_edges("log_sigma", {"y_rep"});

        for (const char* seed : {"log_amp", "log_ell", "lambda"}) {
            std::unordered_set<std::string> changed{seed};
            const auto order = d.predict_downstream_of(changed);
            std::printf("        replace %-8s order = %s\n", seed,
                        join(order).c_str());
            check(contains(order, "mu"),
                  std::string("replace ") + seed +
                  ": mu recomputed (else y_rep is drawn from the training mu)");
        }
    }

    // ---- Case 5: wide-then-deep (many seed children, then a long tail) ------
    // Confirms the contract does not depend on the seed-child count either way.
    {
        std::printf("\nCase 5: 3 seed children, then a depth-4 tail\n");
        AI4BayesCode::shared_data_t d;
        d.declare_data_input("X");
        d.set("X", arma::vec{1.0});
        for (const char* k : {"w1", "w2", "w3"}) add_derived(d, k, "X");
        d.declare_predict_edges("X", {"w1", "w2", "w3"});
        add_derived(d, "t1", "w1");
        add_derived(d, "t2", "t1");
        add_derived(d, "t3", "t2");
        add_derived(d, "t4", "t3");
        d.declare_predict_edges("w1", {"t1"});
        d.declare_predict_edges("t1", {"t2"});
        d.declare_predict_edges("t2", {"t3"});
        d.declare_predict_edges("t3", {"t4"});

        std::unordered_set<std::string> changed{"X"};
        const auto order = d.predict_downstream_of(changed);
        std::printf("        order = %s\n", join(order).c_str());
        for (const char* k : {"w1", "w2", "w3", "t1", "t2", "t3", "t4"})
            check(contains(order, k), std::string(k) + " recomputed");
    }

    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

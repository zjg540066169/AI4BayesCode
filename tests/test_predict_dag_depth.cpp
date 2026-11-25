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
 *  Cases 6-8 pin the ORDER when the replaced set has several keys. A parent
 *  that is itself recomputed counts as available from the start (it holds a
 *  value in shared_data), so the walk can resolve a child before that
 *  parent; the returned list is reordered so that each node follows its
 *  parents in the list.
 *    Case 6: A -> Q -> P -> C with X -> C, replaced {A, X}: the walk resolves C
 *            in the first round and P in the second, so before the reorder C
 *            preceded P for every iteration order of the replaced set. Also
 *            run through composite_block::predict_at, where C must be computed
 *            from the recomputed P.
 *    Cases 7, 8: the BSplineRegression and HSGPRegression topologies with the
 *            full per-draw replaced set of their history-mode predict_at
 *            (every sampled source plus the basis), through
 *            composite_block::predict_at with the examples' formulas.
 *  Case 5 also checks that a list already in parent-first order is returned
 *  as the walk produced it.
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
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/shared_data.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
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

// Membership is not the contract. composite_block::predict_at iterates the
// returned vector IN ORDER (`for (dk : det_refreshed) scratch.refresh_key(dk)`),
// so a parent listed AFTER its child is recomputed too late and the child keeps
// a value derived from the training state. Reversing the returned order leaves
// every membership check passing while predict_at returns a wrong number, so
// order has to be asserted explicitly.
static bool before(const std::vector<std::string>& v,
                   const std::string& parent, const std::string& child) {
    const auto ip = std::find(v.begin(), v.end(), parent);
    const auto ic = std::find(v.begin(), v.end(), child);
    if (ip == v.end() || ic == v.end()) return false;   // both must be present
    return ip < ic;
}

static void check_order(const std::vector<std::string>& v,
                        const std::string& parent, const std::string& child) {
    check(before(v, parent, child),
          "order: " + parent + " is refreshed before " + child);
}

static void check_absent(const std::vector<std::string>& v,
                         const std::string& k, const std::string& why) {
    check(std::find(v.begin(), v.end(), k) == v.end(),
          k + " is NOT in the refresh order (" + why + ")");
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
        check_order(order, "A", "B");
        check_order(order, "B", "C");
        check_absent(order, "X", "it is the replaced input, not a downstream node");
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
        for (int i = 1; i < 5; ++i) check_order(order, keys[i - 1], keys[i]);
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
            check_order(order, "s", "f");
            check_order(order, "f", "mu");
            check_absent(order, "y_rep", "stochastic nodes are Pass 2, not Pass 1");
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
            check_order(order, "sqrt_spd", "f");
            check_order(order, "f", "mu");
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
        check_order(order, "w1", "t1");
        check_order(order, "t1", "t2");
        check_order(order, "t2", "t3");
        check_order(order, "t3", "t4");
        // The walk resolves w1, w2, w3 (children of X, in declaration order),
        // then t1..t4 one per round: already parent-first, so the reorder
        // leaves it as it is.
        check(order == std::vector<std::string>{"w1", "w2", "w3", "t1", "t2",
                                                "t3", "t4"},
              "a parent-first walk order is returned unchanged");
    }

    // ---- Case 6: a recomputed parent resolved one round after its child -----
    //   A -> Q -> P -> C,  X -> C ;  replaced {A, X}
    // Round 1 resolves Q (parent A) and C (parents P: holds a value, X:
    // replaced); P follows in round 2. C must still be refreshed after P.
    {
        std::printf("\nCase 6: A -> Q -> P -> C and X -> C, replaced {A, X}\n");
        auto build = [](AI4BayesCode::shared_data_t& d) {
            d.declare_data_input("X");
            d.set("A", arma::vec{0.0});
            d.set("X", arma::vec{0.0});
            d.set("Q", arma::vec{1.0});
            d.set("P", arma::vec{2.0});
            d.set("C", arma::vec{2.0});
            d.register_refresher("Q", [](const AI4BayesCode::shared_data_t& s) {
                return arma::vec{s.get("A")(0) + 1.0}; });
            d.register_refresher("P", [](const AI4BayesCode::shared_data_t& s) {
                return arma::vec{2.0 * s.get("Q")(0)}; });
            d.register_refresher("C", [](const AI4BayesCode::shared_data_t& s) {
                return arma::vec{s.get("P")(0) + s.get("X")(0)}; });
            d.declare_predict_edges("A", {"Q"});
            d.declare_predict_edges("Q", {"P"});
            d.declare_predict_edges("P", {"C"});
            d.declare_predict_edges("X", {"C"});
        };
        AI4BayesCode::shared_data_t d;
        build(d);
        std::unordered_set<std::string> changed{"A", "X"};
        const auto order = d.predict_downstream_of(changed);
        std::printf("        order = %s\n", join(order).c_str());
        check(order.size() == 3 && contains(order, "Q") && contains(order, "P") &&
              contains(order, "C"), "Q, P and C recomputed (and nothing else)");
        check_order(order, "Q", "P");
        check_order(order, "P", "C");

        AI4BayesCode::composite_block c("case6");
        build(c.data());
        AI4BayesCode::block_context rep;
        rep["A"] = arma::vec{5.0};
        rep["X"] = arma::vec{100.0};
        std::mt19937_64 rng(1);
        const auto r = c.predict_at(rep, rng);
        const double C = r.count("C") ? r.at("C")(0) : NAN;
        std::printf("        predict_at(A = 5, X = 100): C = %g (expected (5 + 1) * 2 + 100 = 112)\n", C);
        check(C == 112.0, "predict_at computes C from the recomputed P");
    }

    // ---- Case 7: BSplineRegression, full per-draw replaced set ---------------
    //   s = exp(log_sds) z ; f = Bsp s ; mu = Intercept + f ; y_rep ~ N(mu, .)
    // History-mode predict_at replaces {Intercept, log_sds, log_sigma, z, Bsp}.
    {
        std::printf("\nCase 7: BSplineRegression topology, replaced {Intercept, "
                    "log_sds, log_sigma, z, Bsp}\n");
        const std::size_t N = 4, K = 3;
        AI4BayesCode::composite_block c("case7");
        AI4BayesCode::shared_data_t& d = c.data();
        d.set("Bsp", arma::vec(N * K, arma::fill::ones));
        d.set("Intercept", arma::vec{0.0});
        d.set("log_sds", arma::vec{0.0});
        d.set("log_sigma", arma::vec{0.0});
        d.set("z", arma::vec(K, arma::fill::zeros));
        d.set("s", arma::vec(K, arma::fill::zeros));
        d.set("f", arma::vec(N, arma::fill::zeros));
        d.set("mu", arma::vec(N, arma::fill::zeros));
        d.set("y_rep", arma::vec(N, arma::fill::zeros));
        d.register_refresher("s", [](const AI4BayesCode::shared_data_t& t) {
            return arma::vec(std::exp(t.get("log_sds")(0)) * t.get("z")); });
        d.register_refresher("f", [](const AI4BayesCode::shared_data_t& t) {
            const arma::vec& s = t.get("s");
            const arma::vec& B = t.get("Bsp");
            const arma::mat Bm(B.memptr(), B.n_elem / s.n_elem, s.n_elem);
            return arma::vec(Bm * s); });
        d.register_refresher("mu", [](const AI4BayesCode::shared_data_t& t) {
            return arma::vec(t.get("Intercept")(0) + t.get("f")); });
        d.register_stochastic_refresher("y_rep",
            [](const AI4BayesCode::shared_data_t& t, std::mt19937_64&) {
                return arma::vec(t.get("mu")); });
        d.declare_data_input("Bsp");
        d.declare_predict_edges("log_sds", {"s"});
        d.declare_predict_edges("z", {"s"});
        d.declare_predict_edges("s", {"f"});
        d.declare_predict_edges("Bsp", {"f"});
        d.declare_predict_edges("f", {"mu"});
        d.declare_predict_edges("Intercept", {"mu"});
        d.declare_predict_edges("mu", {"y_rep"});
        d.declare_predict_edges("log_sigma", {"y_rep"});

        std::unordered_set<std::string> changed{"Intercept", "log_sds", "log_sigma",
                                                "z", "Bsp"};
        const auto order = d.predict_downstream_of(changed);
        std::printf("        order = %s\n", join(order).c_str());
        check_order(order, "s", "f");
        check_order(order, "f", "mu");

        arma::mat Bm(N, K);
        for (std::size_t i = 0; i < N * K; ++i) Bm(i) = 0.1 * static_cast<double>(i + 1);
        AI4BayesCode::block_context rep;
        rep["Bsp"] = arma::vectorise(Bm);
        rep["Intercept"] = arma::vec{2.0};
        rep["log_sds"] = arma::vec{std::log(3.0)};
        rep["log_sigma"] = arma::vec{0.0};
        rep["z"] = arma::vec{1.0, -1.0, 0.5};
        std::mt19937_64 rng(1);
        const auto r = c.predict_at(rep, rng);
        const arma::vec f_ref = Bm * (3.0 * rep["z"]);
        const arma::vec mu_ref = 2.0 + f_ref;
        const double ef = r.count("f") ? arma::max(arma::abs(r.at("f") - f_ref)) : INFINITY;
        const double em = r.count("mu") ? arma::max(arma::abs(r.at("mu") - mu_ref)) : INFINITY;
        std::printf("        max|f - Bsp exp(log_sds) z| = %.3g, max|mu - (Intercept + f)| = %.3g\n",
                    ef, em);
        check(ef < 1e-12 && em < 1e-12,
              "predict_at: f and mu from the replaced draw (not from the s before the call)");
    }

    // ---- Case 8: HSGPRegression, full per-draw replaced set ------------------
    //   sqrt_spd(log_amp, log_ell, lambda) ; f = phi (sqrt_spd % z) ;
    //   mu = Intercept + f. History mode replaces {Intercept, log_amp, log_ell,
    //   log_sigma, z, phi}.
    {
        std::printf("\nCase 8: HSGPRegression topology, replaced {Intercept, "
                    "log_amp, log_ell, log_sigma, z, phi}\n");
        const std::size_t N = 5, M = 3;
        AI4BayesCode::composite_block c("case8");
        AI4BayesCode::shared_data_t& d = c.data();
        d.set("phi", arma::vec(N * M, arma::fill::ones));
        d.set("lambda", arma::vec{0.5, 2.0, 4.5});
        for (const char* k : {"Intercept", "log_amp", "log_ell", "log_sigma"})
            d.set(k, arma::vec{0.0});
        d.set("z", arma::vec(M, arma::fill::zeros));
        d.set("sqrt_spd", arma::vec(M, arma::fill::zeros));
        d.set("f", arma::vec(N, arma::fill::zeros));
        d.set("mu", arma::vec(N, arma::fill::zeros));
        d.set("y_rep", arma::vec(N, arma::fill::zeros));
        auto spd = [](double amp, double ell, const arma::vec& lam) {
            arma::vec ss(lam.n_elem);
            for (std::size_t m = 0; m < lam.n_elem; ++m)
                ss[m] = std::sqrt(amp * amp * std::sqrt(2.0 * M_PI) * ell *
                                  std::exp(-0.5 * ell * ell * lam[m]));
            return ss;
        };
        d.register_refresher("sqrt_spd", [spd](const AI4BayesCode::shared_data_t& t) {
            return spd(std::exp(t.get("log_amp")(0)), std::exp(t.get("log_ell")(0)),
                       t.get("lambda")); });
        d.register_refresher("f", [](const AI4BayesCode::shared_data_t& t) {
            const arma::vec c_m = t.get("sqrt_spd") % t.get("z");
            const arma::vec& P = t.get("phi");
            const arma::mat Pm(P.memptr(), P.n_elem / c_m.n_elem, c_m.n_elem);
            return arma::vec(Pm * c_m); });
        d.register_refresher("mu", [](const AI4BayesCode::shared_data_t& t) {
            return arma::vec(t.get("Intercept")(0) + t.get("f")); });
        d.register_stochastic_refresher("y_rep",
            [](const AI4BayesCode::shared_data_t& t, std::mt19937_64&) {
                return arma::vec(t.get("mu")); });
        d.declare_data_input("phi");
        d.declare_predict_edges("log_amp", {"sqrt_spd"});
        d.declare_predict_edges("log_ell", {"sqrt_spd"});
        d.declare_predict_edges("lambda", {"sqrt_spd"});
        d.declare_predict_edges("sqrt_spd", {"f"});
        d.declare_predict_edges("z", {"f"});
        d.declare_predict_edges("phi", {"f"});
        d.declare_predict_edges("f", {"mu"});
        d.declare_predict_edges("Intercept", {"mu"});
        d.declare_predict_edges("mu", {"y_rep"});
        d.declare_predict_edges("log_sigma", {"y_rep"});

        std::unordered_set<std::string> changed{"Intercept", "log_amp", "log_ell",
                                                "log_sigma", "z", "phi"};
        const auto order = d.predict_downstream_of(changed);
        std::printf("        order = %s\n", join(order).c_str());
        check_order(order, "sqrt_spd", "f");
        check_order(order, "f", "mu");

        arma::mat Pm(N, M);
        for (std::size_t i = 0; i < N * M; ++i) Pm(i) = std::sin(0.3 * static_cast<double>(i + 1));
        AI4BayesCode::block_context rep;
        rep["phi"] = arma::vectorise(Pm);
        rep["Intercept"] = arma::vec{-1.0};
        rep["log_amp"] = arma::vec{std::log(1.5)};
        rep["log_ell"] = arma::vec{std::log(0.7)};
        rep["log_sigma"] = arma::vec{0.0};
        rep["z"] = arma::vec{0.3, -1.2, 2.0};
        std::mt19937_64 rng(1);
        const auto r = c.predict_at(rep, rng);
        const arma::vec f_ref = Pm * (spd(1.5, 0.7, arma::vec{0.5, 2.0, 4.5}) % rep["z"]);
        const arma::vec mu_ref = -1.0 + f_ref;
        const double ef = r.count("f") ? arma::max(arma::abs(r.at("f") - f_ref)) : INFINITY;
        const double em = r.count("mu") ? arma::max(arma::abs(r.at("mu") - mu_ref)) : INFINITY;
        std::printf("        max|f - phi (sqrt_spd %% z)| = %.3g, max|mu - (Intercept + f)| = %.3g\n",
                    ef, em);
        check(ef < 1e-12 && em < 1e-12,
              "predict_at: f and mu from the replaced draw (not from the sqrt_spd before the call)");
    }

    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

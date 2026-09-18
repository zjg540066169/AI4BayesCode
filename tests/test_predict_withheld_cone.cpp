// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later.
// ============================================================================
//  test_predict_withheld_cone.cpp
//
//  Pins the semantics of predicting at a STRICT SUBSET of the data.
//
//  A model whose data inputs share one observation index cannot be predicted
//  at new values of only some of them: the ones left out still hold their
//  training values, which no longer line up. Before declare_data_input_group()
//  existed, the availability rule counted such a key as usable (it is a
//  non-data-input, or simply "has a value"), so a node combining a replaced
//  input with a withheld one was computed from a MIXTURE of new and training
//  data -- a dimension error when the two sizes differ, and a silently wrong
//  answer when they happen to match.
//
//  The contract pinned here:
//    * a co-indexed input the caller left out, and everything downstream of
//      it, is NOT PREDICTABLE and is absent from the result;
//    * everything reachable WITHOUT it is still computed -- prediction goes as
//      far as the graph allows and stops;
//    * a model that declares no group is completely unaffected;
//    * replacing NO member of a group is unaffected (the whole group is at its
//      training values and is mutually consistent).
// ============================================================================

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  [FAIL] %s\n", what.c_str()); }
}
static bool has(const std::vector<std::string>& v, const std::string& k) {
    return std::find(v.begin(), v.end(), k) != v.end();
}
static bool has(const std::unordered_set<std::string>& v, const std::string& k) {
    return v.count(k) != 0;
}
static std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) { if (i) s += ", "; s += v[i]; }
    return s.empty() ? "(none)" : s;
}

// A derived key whose refresher just echoes one parent, so the test is about
// the traversal rather than about arithmetic.
static void derived(AI4BayesCode::shared_data_t& d,
                    const std::string& key, const std::string& parent) {
    d.set(key, arma::vec{0.0});
    d.register_refresher(key, [parent](const AI4BayesCode::shared_data_t& s) {
        return s.get(parent);
    });
}

// The BKMR shape, reduced to what matters:
//     X -> K -> h_mean -> h -> y_rep      (h_mean also reads training-side keys)
//     Z -> mu_Z -> y_rep
// X and Z are indexed by the same observations.
static AI4BayesCode::shared_data_t bkmr_shape(bool declare_group) {
    AI4BayesCode::shared_data_t d;
    d.set("X", arma::vec{1.0});
    d.set("Z", arma::vec{1.0});
    d.set("beta", arma::vec{1.0});
    d.set("sigma2", arma::vec{1.0});
    derived(d, "K", "X");
    derived(d, "h_mean", "K");
    derived(d, "mu_Z", "Z");
    d.set("h", arma::vec{0.0});
    d.register_stochastic_refresher(
        "h", [](const AI4BayesCode::shared_data_t& s, std::mt19937_64&) {
            return s.get("h_mean");
        });
    d.set("y_rep", arma::vec{0.0});
    d.register_stochastic_refresher(
        "y_rep", [](const AI4BayesCode::shared_data_t& s, std::mt19937_64&) {
            return s.get("mu_Z");
        });
    d.declare_predict_edges("X", {"K"});
    d.declare_predict_edges("K", {"h_mean"});
    d.declare_predict_edges("h_mean", {"h"});
    d.declare_predict_edges("Z", {"mu_Z"});
    d.declare_predict_edges("beta", {"mu_Z"});
    d.declare_predict_edges("h", {"y_rep"});
    d.declare_predict_edges("mu_Z", {"y_rep"});
    d.declare_predict_edges("sigma2", {"y_rep"});
    if (declare_group) d.declare_data_input_group({"X", "Z"});
    else { d.declare_data_input("X"); d.declare_data_input("Z"); }
    return d;
}

int main() {
    std::printf("=== predict_at at a strict subset of the data ===\n\n");

    // ---- 1. The reported case: replace X only -----------------------------
    {
        std::printf("Case 1: co-indexed X,Z declared; replace X only\n");
        auto d = bkmr_shape(true);
        std::unordered_set<std::string> replaced{"X"};
        const auto cone = d.predict_withheld_cone(replaced);
        std::printf("        not predictable = ");
        for (const auto& k : cone) std::printf("%s ", k.c_str());
        std::printf("\n");
        check(has(cone, "Z"),      "the withheld co-indexed input is in the cone");
        check(has(cone, "mu_Z"),   "a node reading it is in the cone");
        check(has(cone, "y_rep"),  "the cone is TRANSITIVE (y_rep reads mu_Z)");
        check(!has(cone, "K"),     "a node not downstream of Z is NOT in the cone");
        check(!has(cone, "h_mean"),"nor is h_mean");
        check(!has(cone, "h"),     "nor is h -- h never depends on Z");

        const auto det = d.predict_downstream_of(replaced, cone);
        std::printf("        pass 1 = %s\n", join(det).c_str());
        check(has(det, "K"),        "pass 1 recomputes K at the new X");
        check(has(det, "h_mean"),   "pass 1 recomputes h_mean");
        check(!has(det, "mu_Z"),    "pass 1 does NOT recompute mu_Z");

        std::unordered_set<std::string> after(replaced);
        for (const auto& k : det) after.insert(k);
        const auto stoch = d.predict_stochastic_sampleable(after, cone);
        std::printf("        pass 2 = %s\n", join(stoch).c_str());
        check(has(stoch, "h"),
              "h IS predictable at new X -- this is what the whole change is for");
        check(!has(stoch, "y_rep"),
              "y_rep is NOT predictable, rather than drawn from the training mu_Z");
    }

    // ---- 2. Replace both: nothing is withheld -----------------------------
    {
        std::printf("\nCase 2: replace BOTH members of the group\n");
        auto d = bkmr_shape(true);
        std::unordered_set<std::string> replaced{"X", "Z"};
        const auto cone = d.predict_withheld_cone(replaced);
        check(cone.empty(), "nothing is withheld");
        const auto det = d.predict_downstream_of(replaced, cone);
        std::unordered_set<std::string> after(replaced);
        for (const auto& k : det) after.insert(k);
        const auto stoch = d.predict_stochastic_sampleable(after, cone);
        std::printf("        pass 1 = %s\n        pass 2 = %s\n",
                    join(det).c_str(), join(stoch).c_str());
        check(has(det, "K") && has(det, "h_mean") && has(det, "mu_Z"),
              "every deterministic node is recomputed");
        check(has(stoch, "h") && has(stoch, "y_rep"),
              "both stochastic nodes are sampled");
    }

    // ---- 3. Replace nothing: predict_at(list()) is unchanged --------------
    {
        std::printf("\nCase 3: replace NOTHING (predict_at at the training data)\n");
        auto d = bkmr_shape(true);
        std::unordered_set<std::string> replaced;
        const auto cone = d.predict_withheld_cone(replaced);
        check(cone.empty(),
              "a group with no member replaced withholds nothing");
        const auto stoch = d.predict_stochastic_sampleable(replaced, cone);
        check(has(stoch, "h") && has(stoch, "y_rep"),
              "both stochastic nodes still sample at the training data");
    }

    // ---- 4. No group declared: behaviour is exactly as before --------------
    {
        std::printf("\nCase 4: SAME graph, no group declared\n");
        auto d = bkmr_shape(false);
        std::unordered_set<std::string> replaced{"X"};
        const auto cone = d.predict_withheld_cone(replaced);
        check(cone.empty(), "no group means nothing is ever withheld");
        const auto det = d.predict_downstream_of(replaced, cone);
        std::unordered_set<std::string> after(replaced);
        for (const auto& k : det) after.insert(k);
        const auto stoch = d.predict_stochastic_sampleable(after, cone);
        std::printf("        pass 1 = %s\n        pass 2 = %s\n",
                    join(det).c_str(), join(stoch).c_str());
        check(has(stoch, "y_rep"),
              "the old, permissive result is preserved for undeclared models");
    }

    // ---- 5. A group member that shares no descendant with the replaced one -
    {
        std::printf("\nCase 5: withheld member feeds a DISJOINT subgraph\n");
        AI4BayesCode::shared_data_t d;
        d.set("A", arma::vec{1.0}); d.set("B", arma::vec{1.0});
        derived(d, "fa", "A"); derived(d, "fb", "B");
        d.declare_predict_edges("A", {"fa"});
        d.declare_predict_edges("B", {"fb"});
        d.declare_data_input_group({"A", "B"});
        const auto cone = d.predict_withheld_cone({"A"});
        check(has(cone, "B") && has(cone, "fb"), "B's own subgraph is withheld");
        check(!has(cone, "fa"), "A's subgraph is untouched");
        const auto det = d.predict_downstream_of({"A"}, cone);
        check(has(det, "fa") && !has(det, "fb"),
              "prediction proceeds on the reachable side and stops at the other");
    }

    // ---- 6. Two independent groups ----------------------------------------
    {
        std::printf("\nCase 6: two independent co-indexed groups\n");
        AI4BayesCode::shared_data_t d;
        for (const char* k : {"X", "Z", "t", "y0"}) d.set(k, arma::vec{1.0});
        derived(d, "mu", "X"); derived(d, "mz", "Z");
        derived(d, "traj", "t");
        d.declare_predict_edges("X", {"mu"});
        d.declare_predict_edges("Z", {"mz"});
        d.declare_predict_edges("t", {"traj"});
        d.declare_predict_edges("y0", {"traj"});
        d.declare_data_input_group({"X", "Z"});
        d.declare_data_input("t");            // not co-indexed with anything
        d.declare_data_input("y0");
        const auto cone = d.predict_withheld_cone({"t"});
        check(cone.empty(),
              "replacing an input that is in NO group withholds nothing "
              "(an initial condition stays valid at a new time grid)");
        const auto cone2 = d.predict_withheld_cone({"X"});
        check(has(cone2, "Z") && has(cone2, "mz"), "the X,Z group still applies");
        check(!has(cone2, "traj") && !has(cone2, "y0"),
              "the ungrouped inputs are unaffected");
    }

    // ---- 7. Repeated declare_predict_edges MERGES ------------------------
    // It used to overwrite, so a source declared in a loop (one edge per
    // varying-coefficient forest) or in two statements (a scale feeding both
    // a random effect and the likelihood) kept only its LAST children, and
    // predict_at walked a graph missing parents the model plainly had.
    {
        std::printf("\nCase 7: declaring the same source twice keeps BOTH edges\n");
        AI4BayesCode::shared_data_t d;
        for (const char* k : {"tau", "z_flat", "R_chol"}) d.set(k, arma::vec{1.0});
        derived(d, "u", "z_flat");
        derived(d, "y_rep_det", "u");
        d.declare_predict_edges("tau",    {"u"});
        d.declare_predict_edges("R_chol", {"u"});
        d.declare_predict_edges("z_flat", {"u"});
        d.declare_predict_edges("tau",    {"y_rep_det"});     // second call
        d.declare_predict_edges("R_chol", {"y_rep_det"});     // second call
        const auto& e = d.predict_edges();
        auto kids = [&](const std::string& k) {
            auto it = e.find(k);
            return it == e.end() ? std::vector<std::string>{} : it->second;
        };
        check(has(kids("tau"), "u") && has(kids("tau"), "y_rep_det"),
              "tau keeps BOTH children across two declarations");
        check(has(kids("R_chol"), "u") && has(kids("R_chol"), "y_rep_det"),
              "R_chol keeps both as well");
        // and a source declared once per item in a loop keeps every item
        AI4BayesCode::shared_data_t d2;
        d2.set("Z", arma::vec{1.0});
        for (int j = 0; j < 3; ++j) {
            derived(d2, "beta_" + std::to_string(j), "Z");
            d2.declare_predict_edges("Z", {"beta_" + std::to_string(j)});
        }
        const auto& e2 = d2.predict_edges();
        const auto zk = e2.at("Z");
        check(zk.size() == 3 && has(zk, "beta_0") && has(zk, "beta_1")
              && has(zk, "beta_2"),
              "a loop over p forests declares p edges, not just the last");
        d2.declare_predict_edges("Z", {"beta_1"});
        check(e2.at("Z").size() == 3, "re-declaring an existing edge is a no-op");
    }

    // ------------------------------------------------------------------
    // Case 8: the groups are part of get_dag(), so a frontend can compute the
    // withheld cone from the same DAG it already reads (predict_edges,
    // data_inputs) instead of guessing which inputs are co-indexed.
    {
        std::printf("\nCase 8: get_dag() carries data_input_groups\n");
        AI4BayesCode::composite_block c;
        auto& d = c.data();
        for (const char* k : {"X", "Z", "W"}) d.set(k, arma::vec{1.0});
        for (const char* k : {"X", "Z", "W"}) d.declare_data_input(k);
        d.declare_data_input_group({"X", "Z"});
        const AI4BayesCode::dag_info dag = c.get_dag();
        check(dag.data_input_groups.size() == 1, "one declared group is reported");
        check(dag.data_input_groups.size() == 1 && dag.data_input_groups[0].size() == 2
              && has(dag.data_input_groups[0], "X") && has(dag.data_input_groups[0], "Z"),
              "the group lists exactly its members");
        check(dag.data_inputs.size() == 3, "data_inputs is unchanged by the group");
        AI4BayesCode::composite_block c0;
        c0.data().set("X", arma::vec{1.0});
        check(c0.get_dag().data_input_groups.empty(), "no group declared -> empty, not absent");
    }

    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

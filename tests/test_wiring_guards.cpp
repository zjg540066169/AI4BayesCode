/*================================================================================
 *  test_wiring_guards.cpp
 *
 *  Pins the guards that turn silent mis-wiring into a loud error. Each case
 *  below used to run to completion and produce a WRONG posterior with no
 *  diagnostic -- the hardest class of bug for a user to notice, because every
 *  convergence check still passes.
 *
 *    A. A CONFIGURED-but-ABSENT context key. `cfg.<x>_key = "k"` means "read k
 *       from the context"; leaving `<x>_key` empty means "this model has no
 *       such term". A configured key missing from the context is neither --
 *       it is a forgotten declare_dependencies entry, and silently skipping
 *       the term drops a covariate effect / left truncation / exposure from
 *       the conditional.
 *
 *    B. predict_at at a node it cannot recompute. A node downstream of the
 *       replaced inputs with no registered refresher and no supplied value
 *       cannot be evaluated at the new data; propagating its TRAINING value
 *       returns a "prediction" computed at the fit.
 *
 *    C. nuts_block::set_current with a wrong length, or with a non-identity
 *       `constrain` but no `unconstrain` (which would restart the chain at
 *       constrain(value) instead of value).
 *
 *    D. refresh_all() priming a derived-reads-derived chain in REGISTRATION
 *       order, not hash order.
 *
 *    E. predict_at's key whitelist. A declared SOURCE of the predict DAG is
 *       a valid replacement key even when it is neither a data input nor a
 *       child block name -- that is the only way to replace ONE sub-param of
 *       a joint block. The whitelist used to omit this, so DirichletSimplex's
 *       history-mode posterior predictive (which installs a per-draw "theta",
 *       a sub-param of the "theta_joint" block) threw on every call. Widening
 *       it must not make a genuine typo silent, so both directions are pinned.
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
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/gamma_gibbs_block.hpp"
#include "AI4BayesCode/piecewise_exponential_gibbs_block.hpp"

#include <cstdio>
#include <memory>
#include <random>
#include <string>

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  [FAIL] %s\n", what.c_str()); }
}

// Runs fn and reports whether it threw with a message containing `needle`.
template <class F>
static void check_throws(F&& fn, const std::string& needle,
                         const std::string& what) {
    try {
        fn();
        check(false, what + " (did NOT throw)");
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        const bool hit = msg.find(needle) != std::string::npos;
        check(hit, what + (hit ? "" : std::string(" -- wrong message: ") + msg));
    }
}

namespace {

double normal_ld(const arma::vec& x, const AI4BayesCode::block_context&,
                 arma::vec* g) {
    if (g) { g->set_size(1); (*g)[0] = -x[0]; }
    return -0.5 * x[0] * x[0];
}

}  // namespace

int main() {
    std::printf("=== wiring guards ===\n\n");
    std::mt19937_64 rng(20260815);

    // ---- A. configured-but-absent context key ------------------------------
    {
        std::printf("A. configured context key missing from the context\n");

        const std::size_t n = 6;
        AI4BayesCode::piecewise_exponential_gibbs_block_config c;
        c.name       = "lambda";
        c.edges      = arma::vec{0.0, 1.0, 10.0};
        c.time_key   = "t_obs";
        c.event_key  = "delta";
        c.initial_lambda = arma::vec{1.0, 1.0};
        // Configured, but deliberately NOT declared as a dependency below.
        c.offset_key = "lp_offset";

        auto root = std::make_unique<AI4BayesCode::composite_block>("root");
        root->data().set("t_obs", arma::vec(n, arma::fill::ones));
        root->data().set("delta", arma::vec(n, arma::fill::ones));
        root->add_child(
            std::make_unique<AI4BayesCode::piecewise_exponential_gibbs_block>(c));
        root->data().declare_dependencies("lambda", {"t_obs", "delta"});

        check_throws([&] { root->step(rng); },
                     "offset_key",
                     "a configured offset_key absent from the context throws");
        check_throws([&] { root->step(rng); },
                     "declare_dependencies",
                     "the message names the fix (declare_dependencies)");
    }

    // ---- A2. the same block wired CORRECTLY still runs ---------------------
    {
        std::printf("\nA2. the same block, offset_key declared -> runs\n");

        const std::size_t n = 6;
        AI4BayesCode::piecewise_exponential_gibbs_block_config c;
        c.name       = "lambda";
        c.edges      = arma::vec{0.0, 1.0, 10.0};
        c.time_key   = "t_obs";
        c.event_key  = "delta";
        c.initial_lambda = arma::vec{1.0, 1.0};
        c.offset_key = "lp_offset";

        auto root = std::make_unique<AI4BayesCode::composite_block>("root");
        root->data().set("t_obs", arma::vec(n, arma::fill::ones));
        root->data().set("delta", arma::vec(n, arma::fill::ones));
        root->data().set("lp_offset", arma::vec(n, arma::fill::ones));
        root->add_child(
            std::make_unique<AI4BayesCode::piecewise_exponential_gibbs_block>(c));
        root->data().declare_dependencies("lambda",
                                          {"t_obs", "delta", "lp_offset"});

        bool ok = true;
        try { root->step(rng); } catch (const std::exception&) { ok = false; }
        check(ok, "correctly wired block still steps (guard is not overbroad)");

        // And leaving the key EMPTY -- "this model has no offset" -- is fine.
        AI4BayesCode::piecewise_exponential_gibbs_block_config c2 = c;
        c2.offset_key = "";
        auto root2 = std::make_unique<AI4BayesCode::composite_block>("root2");
        root2->data().set("t_obs", arma::vec(n, arma::fill::ones));
        root2->data().set("delta", arma::vec(n, arma::fill::ones));
        root2->add_child(
            std::make_unique<AI4BayesCode::piecewise_exponential_gibbs_block>(c2));
        root2->data().declare_dependencies("lambda", {"t_obs", "delta"});
        bool ok2 = true;
        try { root2->step(rng); } catch (const std::exception&) { ok2 = false; }
        check(ok2, "an EMPTY optional key still means 'not used'");
    }

    // ---- B. predict_at at an unrecomputable node ---------------------------
    {
        std::printf("\nB. predict_at at a node with no refresher\n");

        auto root = std::make_unique<AI4BayesCode::composite_block>("root");
        // A block OWNS f_forest, so it is a legal predict_at replacement key --
        // this stands in for the BART / genBART child in a real model, whose
        // fitted ensemble cannot be re-evaluated from shared_data alone.
        AI4BayesCode::gamma_gibbs_block_config fc;
        fc.name = "f_forest";
        fc.params_fn = [](const AI4BayesCode::block_context&) {
            AI4BayesCode::gamma_params p; p.shape = 2.0; p.rate = 1.0; return p;
        };
        root->add_child(
            std::make_unique<AI4BayesCode::gamma_gibbs_block>(std::move(fc)));

        auto& d = root->data();
        d.declare_data_input("X");
        d.set("X", arma::vec{1.0});
        d.set("f_forest", arma::vec{5.0});     // "fitted at training X"
        d.set("mu", arma::vec{0.0});
        d.register_refresher("mu", [](const AI4BayesCode::shared_data_t& s) {
            return arma::vec{s.get("f_forest")(0)};
        });
        d.declare_predict_edges("X", {"f_forest"});
        d.declare_predict_edges("f_forest", {"mu"});

        // Match the guard's own wording, not just the key name -- "f_forest"
        // alone would also be satisfied by any unrelated exception that
        // happens to mention the key.
        check_throws([&] { root->predict_at(AI4BayesCode::block_context{
                               {"X", arma::vec{9.0}}}); },
                     "no registered refresher",
                     "predict_at reports the missing-refresher guard");
        check_throws([&] { root->predict_at(AI4BayesCode::block_context{
                               {"X", arma::vec{9.0}}}); },
                     "f_forest",
                     "...and names the node it cannot recompute");

        // Supplying it directly (the documented per-draw injection) works.
        bool ok = true;
        try {
            auto out = root->predict_at(AI4BayesCode::block_context{
                {"X", arma::vec{9.0}}, {"f_forest", arma::vec{7.0}}});
            ok = out.count("mu") && std::abs(out.at("mu")(0) - 7.0) < 1e-12;
        } catch (const std::exception&) { ok = false; }
        check(ok, "injecting the node per draw recomputes downstream from it");
    }

    // ---- C. nuts_block::set_current validation -----------------------------
    {
        std::printf("\nC. nuts_block::set_current\n");

        AI4BayesCode::nuts_block_config cfg;
        cfg.name             = "theta";
        cfg.initial_unc      = arma::vec{0.0};
        cfg.log_density_grad = &normal_ld;
        AI4BayesCode::nuts_block nb(cfg);

        check_throws([&] { nb.set_current(arma::vec{1.0, 2.0}); },
                     "length",
                     "a wrong-length set_current throws instead of resizing");

        // constrain supplied, unconstrain not -> refuse rather than mis-restart.
        AI4BayesCode::nuts_block_config cfg2;
        cfg2.name             = "sigma";
        cfg2.initial_unc      = arma::vec{0.0};
        cfg2.log_density_grad = &normal_ld;
        cfg2.constrain = [](const arma::vec& x) { return arma::exp(x); };
        AI4BayesCode::nuts_block nb2(cfg2);
        check_throws([&] { nb2.set_current(arma::vec{2.0}); },
                     "unconstrain",
                     "constrain without unconstrain is refused, not silently "
                     "treated as identity");
    }

    // ---- D. refresh_all uses registration order ----------------------------
    {
        std::printf("\nD. refresh_all primes a derived-reads-derived chain\n");

        AI4BayesCode::shared_data_t d;
        d.set("base", arma::vec{2.0});
        // Placeholders, exactly as a model seeds them before refresh_all.
        d.set("first",  arma::vec{0.0});
        d.set("second", arma::vec{0.0});
        // Registration order IS dependency order: first, then second (reads first).
        d.register_refresher("first", [](const AI4BayesCode::shared_data_t& s) {
            return arma::vec{s.get("base")(0) * 10.0};        // -> 20
        });
        d.register_refresher("second", [](const AI4BayesCode::shared_data_t& s) {
            return arma::vec{s.get("first")(0) + 1.0};         // -> 21, not 1
        });
        d.refresh_all();

        check(std::abs(d.get("first")(0) - 20.0) < 1e-12,
              "the first derived key is computed from its input");
        check(std::abs(d.get("second")(0) - 21.0) < 1e-12,
              "the second reads the REFRESHED first, not its placeholder");
    }

    // ---- E. predict_at's key whitelist --------------------------------------
    {
        std::printf("\nE. predict_at key whitelist\n");

        auto root = std::make_unique<AI4BayesCode::composite_block>("root");
        AI4BayesCode::gamma_gibbs_block_config gc;
        gc.name = "theta_joint";              // the BLOCK name
        gc.params_fn = [](const AI4BayesCode::block_context&) {
            AI4BayesCode::gamma_params p; p.shape = 2.0; p.rate = 1.0; return p;
        };
        root->add_child(
            std::make_unique<AI4BayesCode::gamma_gibbs_block>(std::move(gc)));

        auto& d = root->data();
        // "theta" is a SUB-PARAM name: not a data input, not a block name.
        d.set("theta", arma::vec{1.0});
        d.set("y_rep", arma::vec{0.0});
        d.register_refresher("y_rep", [](const AI4BayesCode::shared_data_t& s) {
            return arma::vec{2.0 * s.get("theta")(0)};
        });
        d.declare_predict_edges("theta", {"y_rep"});

        bool ok = false;
        try {
            auto out = root->predict_at(AI4BayesCode::block_context{
                {"theta", arma::vec{3.0}}});
            ok = out.count("y_rep") && std::abs(out.at("y_rep")(0) - 6.0) < 1e-12;
        } catch (const std::exception& e) {
            std::printf("      threw: %s\n", e.what());
        }
        check(ok, "a predict-DAG source that is neither a data input nor a "
                  "block name is accepted, and propagates");

        // The negative direction: widening the whitelist must not swallow typos.
        check_throws([&] { root->predict_at(AI4BayesCode::block_context{
                               {"thetaa", arma::vec{3.0}}}); },
                     "unknown key",
                     "a misspelled key is still rejected");
        check_throws([&] { root->predict_at(AI4BayesCode::block_context{
                               {"thetaa", arma::vec{3.0}}}); },
                     "predict-DAG sources",
                     "...and the message lists the sources it could have been");
    }

    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

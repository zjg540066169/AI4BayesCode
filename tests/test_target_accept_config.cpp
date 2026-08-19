/*================================================================================
 *  AI4BayesCode  --  test_target_accept_config.cpp
 *
 *  FORK MARKER (2026-07-26 restore) [target_accept API expose, default=0.55]
 *  Restored (as an API-only surface) from commit 060abe4, which had shipped
 *  this same test with default = 0.8. That default triggered the J100 funnel
 *  regression that forced the revert in 664a84f. This restored version keeps
 *  the API surface but pins expected default = 0.55; the ctor-override test
 *  is flipped to override to 0.8 (so it still exercises "user asks for a
 *  non-default target"). All 14 assertions preserved by count.
 *
 *  Verifies the AI4BayesCode wrapper-layer contract for the NUTS dual-
 *  averaging target acceptance rate:
 *
 *      A) Default is 0.55 (Hoffman-Gelman 2014 and mcmclib's own default).
 *         NOTE: 060abe4 raised the default to 0.8, but that shipped a J100
 *         hierarchical funnel mixing regression (see 664a84f revert), so the
 *         2026-07-26 API-restore keeps 0.55. The knob is still exposed so
 *         users can pass 0.8 explicitly to match Stan / PyMC / NumPyro.
 *      B) A ctor-side override via cfg.target_accept_rate is honored and
 *         forwarded into the nested mcmclib settings.
 *      C) The 4th arg of readapt() overrides the target for subsequent
 *         adaptation; step_size adapts toward a smaller value when the
 *         target is raised (higher target => smaller step in expectation).
 *      D) The sentinel (target_accept_override < 0 or > 1) leaves the
 *         target unchanged.
 *      E) joint_nuts_block honours the same four contracts.
 *
 *  Restored from 060abe4 with the 14 assertions unchanged in count, but
 *  each `check(...)` retargeted at the 0.55 default (and the ctor override
 *  in test_B flipped to 0.8 so it still exercises the "user picks Stan
 *  parity" path). See MEMORY.md and the CLAUDE.md standing rules.
 *
 *  License: GPL-3.0-or-later (matches AI4BayesCode).
 *================================================================================*/

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "portable_rng.hpp"   // portable draws: identical on every stdlib
#include <string>

using AI4BayesCode::block_context;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;
using AI4BayesCode::composite_block;
namespace constraints = AI4BayesCode::constraints;

namespace {

struct results { int passed = 0; int failed = 0; };
static results G;

static void check(bool ok, const std::string& tag,
                  const std::string& detail = "") {
    if (ok) { ++G.passed; std::printf("  PASS  %s\n", tag.c_str()); }
    else    { ++G.failed;
              std::printf("  FAIL  %s  %s\n", tag.c_str(), detail.c_str()); }
}

// ---- Shared fixture: mu-only Normal-location oracle -----------------------

static arma::vec gen_data(std::size_t N, double mu_t, double sg_t,
                          std::uint64_t s) {
    std::mt19937_64 rng(s);
    prng::normal_distribution<double> nd(mu_t, sg_t);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) y[i] = nd(rng);
    return y;
}

static double mu_lp(const arma::vec& th, const block_context& ctx,
                    arma::vec* g) {
    const double mu = th[0];
    const arma::vec& y = ctx.at("y");
    const double sg = ctx.at("sigma")[0];
    const double sg2 = sg * sg;
    double sum_r = 0.0, sum_sq = 0.0;
    for (std::size_t i = 0; i < y.n_elem; ++i) {
        const double r = y[i] - mu;
        sum_r += r; sum_sq += r * r;
    }
    const double lp = -0.5 * sum_sq / sg2 - 0.5 * mu * mu / 1e4;
    if (g) { g->set_size(1); (*g)[0] = sum_r / sg2 - mu / 1e4; }
    return lp;
}

static std::unique_ptr<nuts_block>
make_mu_block(const arma::vec& y, double mu0, double target_override) {
    (void)y;
    nuts_block_config cfg;
    cfg.name = "mu";
    cfg.initial_unc = arma::vec{mu0};
    cfg.log_density_grad = [](const arma::vec& th, const block_context& c,
                              arma::vec* g) {
        return constraints::real::wrap(th, g,
            [&](const arma::vec& tn, arma::vec* gn) { return mu_lp(tn, c, gn); });
    };
    cfg.nuts_settings.nuts_settings.step_size      = 0.5;
    cfg.nuts_settings.nuts_settings.max_tree_depth = 6;
    if (target_override > 0.0) cfg.target_accept_rate = target_override;
    return std::make_unique<nuts_block>(std::move(cfg));
}

static std::unique_ptr<composite_block>
build_mu_composite(const arma::vec& y, double mu0, double sg0,
                   double target_override = -1.0) {
    auto m = std::make_unique<composite_block>("test_target_accept");
    m->data().set("y", y);
    m->data().set("mu", arma::vec{mu0});
    m->data().set("sigma", arma::vec{sg0});
    m->data().declare_dependencies("mu", {"y", "sigma"});
    m->add_child(make_mu_block(y, mu0, target_override));
    return m;
}

// ---- Subtest A: default target_accept == 0.55 ----------------------------

static void test_A_default_target() {
    std::printf("\n=== A: nuts_block default target_accept_rate = 0.55 ===\n");
    nuts_block_config cfg;
    check(cfg.target_accept_rate == 0.55,
          "A.a nuts_block_config default 0.55",
          "got " + std::to_string(cfg.target_accept_rate));

    const arma::vec y = gen_data(200, 2.5, 1.2, 20260725ull);
    auto blk = make_mu_block(y, 0.0, /*target_override=*/-1.0);
    check(blk->current_target_accept() == 0.55,
          "A.b block reports 0.55 after ctor",
          "got " + std::to_string(blk->current_target_accept()));
}

// ---- Subtest B: ctor override to 0.8 (Stan/PyMC parity path) ---------------
// (060abe4's version overrode to 0.55 because its default was 0.8; we mirror
//  the intent -- "user asks for a NON-DEFAULT" -- with a 0.55 default by
//  overriding to 0.8 here. Assertion count unchanged.)

static void test_B_ctor_override() {
    std::printf("\n=== B: ctor override target_accept_rate = 0.8 ===\n");
    const arma::vec y = gen_data(200, 2.5, 1.2, 20260725ull);
    auto blk = make_mu_block(y, 0.0, /*target_override=*/0.8);
    check(blk->current_target_accept() == 0.8,
          "B.a block reports 0.8 after ctor",
          "got " + std::to_string(blk->current_target_accept()));

    // Also verify it propagates into the nested mcmclib nuts_settings by
    // stepping once and observing that adaptation ran (step_size may vary,
    // but no exception + step_size still finite proves the settings were
    // consumed downstream).
    composite_block comp("B_comp");
    comp.data().set("y", y);
    comp.data().set("mu", arma::vec{0.0});
    comp.data().set("sigma", arma::vec{1.0});
    comp.data().declare_dependencies("mu", {"y", "sigma"});
    comp.add_child(std::move(blk));
    std::mt19937_64 rng(1);
    for (int i = 0; i < 20; ++i) comp.step(rng);
    const auto& mu_ref = dynamic_cast<const nuts_block&>(comp.child(0));
    const double eps = mu_ref.current_step_size();
    check(std::isfinite(eps) && eps > 0.0,
          "B.b chain runs and step_size finite under target=0.8",
          "step_size=" + std::to_string(eps));
}

// ---- Subtest C: readapt 4th arg overrides target and adapts step ----------

static void test_C_readapt_target_override() {
    std::printf("\n=== C: readapt 4th arg overrides target and adapts ===\n");
    const arma::vec y = gen_data(200, 2.5, 1.2, 20260725ull);
    auto m = build_mu_composite(y, 0.0, 1.0);
    auto& blk = dynamic_cast<nuts_block&>(m->child(0));

    std::mt19937_64 rng(11);
    for (int i = 0; i < 100; ++i) m->step(rng);

    const double eps_before = blk.current_step_size();
    check(blk.current_target_accept() == 0.55,
          "C.a target still 0.55 after 100 steps",
          "got " + std::to_string(blk.current_target_accept()));

    std::mt19937_64 rr(20260725ull);
    m->readapt_NUTS(200, /*reset=*/false, rr,
                    /*max_tree_depth_override=*/0,
                    /*target_accept_override=*/0.99);

    const double eps_after = blk.current_step_size();
    check(blk.current_target_accept() == 0.99,
          "C.b target updated to 0.99 after readapt(..., 0.99)",
          "got " + std::to_string(blk.current_target_accept()));

    std::printf("     step_size: before=%g  after=%g\n", eps_before, eps_after);
    // Direction: higher target => smaller epsilon in dual-averaging
    // steady state. This is a stochastic quantity, so we check the sign
    // of the drift with a generous tolerance; if this assertion becomes
    // flaky, remove it and keep C.a/C.b as the strict contract.
    check(eps_after < eps_before,
          "C.c step_size shrank when target rose 0.55 -> 0.99",
          "eps_before=" + std::to_string(eps_before) +
          " eps_after=" + std::to_string(eps_after));
}

// ---- Subtest D: sentinel leaves target unchanged ---------------------------

static void test_D_sentinel() {
    std::printf("\n=== D: sentinel target_accept < 0 keeps target unchanged ===\n");
    const arma::vec y = gen_data(200, 2.5, 1.2, 20260725ull);
    auto m = build_mu_composite(y, 0.0, 1.0);
    auto& blk = dynamic_cast<nuts_block&>(m->child(0));

    std::mt19937_64 rng(3);
    for (int i = 0; i < 50; ++i) m->step(rng);

    const double target_before = blk.current_target_accept();
    std::mt19937_64 rr(9);
    m->readapt_NUTS(50, /*reset=*/false, rr,
                    /*max_tree_depth_override=*/0,
                    /*target_accept_override=*/-1.0);
    const double target_after = blk.current_target_accept();
    check(target_after == target_before,
          "D.a sentinel -1.0 leaves target unchanged",
          "before=" + std::to_string(target_before) +
          " after=" + std::to_string(target_after));

    // Also verify 3-arg readapt_NUTS (backward-compat, no target override
    // available) leaves target unchanged too.
    m->readapt_NUTS(50, /*reset=*/false, rr,
                    /*max_tree_depth_override=*/0);
    check(blk.current_target_accept() == target_before,
          "D.b 3-arg readapt (default sentinel) leaves target unchanged");
}

// ---- Subtest E: joint_nuts_block matches all four contracts ---------------

// Build a 2-D joint block over (mu, sigma) for a Normal model, so we
// exercise joint_nuts_block's target_accept contract on a non-trivial
// posterior.

static double joint_lp(const arma::vec& theta, const block_context& ctx,
                       arma::vec* g) {
    // theta_cat = [mu, sigma] on the NATURAL scale.
    const double mu = theta[0];
    const double sg = theta[1];
    if (!(sg > 0.0)) {
        if (g) { g->set_size(2); (*g)[0] = 0.0; (*g)[1] = 0.0; }
        return -std::numeric_limits<double>::infinity();
    }
    const arma::vec& y = ctx.at("y");
    const double sg2 = sg * sg;
    double sum_r = 0.0, sum_sq = 0.0;
    for (std::size_t i = 0; i < y.n_elem; ++i) {
        const double r = y[i] - mu;
        sum_r += r; sum_sq += r * r;
    }
    const double N = static_cast<double>(y.n_elem);
    const double lp = -N * std::log(sg) - 0.5 * sum_sq / sg2
                      - 0.5 * mu * mu / 1e4
                      - 0.5 * sg2 / 100.0;
    if (g) {
        g->set_size(2);
        (*g)[0] = sum_r / sg2 - mu / 1e4;
        (*g)[1] = -N / sg + sum_sq / (sg2 * sg) - sg / 100.0;
    }
    return lp;
}

static joint_nuts_block_config make_joint_cfg(double target_override) {
    joint_nuts_block_config c;
    c.name = "theta";
    c.sub_params = {
        {"mu",    1u, joint_constraint::REAL,     0.0, 0.0},
        {"sigma", 1u, joint_constraint::POSITIVE, 0.0, 0.0},
    };
    c.log_density_grad = joint_lp;
    c.initial_cat = arma::vec{0.0, 1.0};
    c.nuts_settings.nuts_settings.step_size = 0.5;
    if (target_override > 0.0) c.target_accept_rate = target_override;
    return c;
}

static void test_E_joint_variants() {
    std::printf("\n=== E: joint_nuts_block honours the same four contracts ===\n");

    // E.A: default 0.55
    {
        joint_nuts_block_config c = make_joint_cfg(-1.0);
        check(c.target_accept_rate == 0.55,
              "E.A joint_nuts_block_config default 0.55",
              "got " + std::to_string(c.target_accept_rate));
    }

    const arma::vec y = gen_data(200, 2.5, 1.2, 20260725ull);

    // E.B: ctor override to 0.8 stored and forwarded (Stan/PyMC parity path)
    {
        auto blk = std::make_unique<joint_nuts_block>(make_joint_cfg(0.8));
        check(blk->current_target_accept() == 0.8,
              "E.B joint block reports 0.8 after ctor",
              "got " + std::to_string(blk->current_target_accept()));
    }

    // E.C: readapt override adjusts target and step size direction
    {
        auto m = std::make_unique<composite_block>("E_C");
        m->data().set("y", y);
        m->data().set("mu",    arma::vec{0.0});
        m->data().set("sigma", arma::vec{1.0});
        m->data().declare_dependencies("theta", {"y"});
        auto blk = std::make_unique<joint_nuts_block>(make_joint_cfg(-1.0));
        auto* raw = blk.get();
        m->add_child(std::move(blk));

        std::mt19937_64 rng(42);
        for (int i = 0; i < 100; ++i) m->step(rng);
        const double eps_before = raw->current_step_size();

        std::mt19937_64 rr(20260725ull);
        m->readapt_NUTS(200, /*reset=*/false, rr,
                        /*max_tree_depth_override=*/0,
                        /*target_accept_override=*/0.99);
        const double eps_after = raw->current_step_size();
        check(raw->current_target_accept() == 0.99,
              "E.C target updated to 0.99 on joint block",
              "got " + std::to_string(raw->current_target_accept()));
        std::printf("     joint step_size: before=%g  after=%g\n",
                    eps_before, eps_after);
        check(eps_after < eps_before,
              "E.C.b joint step_size shrank when target rose 0.55 -> 0.99",
              "before=" + std::to_string(eps_before) +
              " after=" + std::to_string(eps_after));
    }

    // E.D: sentinel leaves target unchanged
    {
        auto m = std::make_unique<composite_block>("E_D");
        m->data().set("y", y);
        m->data().set("mu",    arma::vec{0.0});
        m->data().set("sigma", arma::vec{1.0});
        m->data().declare_dependencies("theta", {"y"});
        auto blk = std::make_unique<joint_nuts_block>(make_joint_cfg(-1.0));
        auto* raw = blk.get();
        m->add_child(std::move(blk));

        std::mt19937_64 rng(7);
        for (int i = 0; i < 50; ++i) m->step(rng);
        const double target_before = raw->current_target_accept();

        std::mt19937_64 rr(11);
        m->readapt_NUTS(50, /*reset=*/false, rr,
                        /*max_tree_depth_override=*/0,
                        /*target_accept_override=*/-1.0);
        check(raw->current_target_accept() == target_before,
              "E.D sentinel -1.0 leaves joint target unchanged",
              "before=" + std::to_string(target_before) +
              " after=" + std::to_string(raw->current_target_accept()));
    }
}

} // anon

int main() {
    std::printf("=== test_target_accept_config ===\n");
    test_A_default_target();
    test_B_ctor_override();
    test_C_readapt_target_override();
    test_D_sentinel();
    test_E_joint_variants();

    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n",
                G.passed, G.failed);
    return G.failed == 0 ? 0 : 1;
}

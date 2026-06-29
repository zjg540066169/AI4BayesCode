/*================================================================================
 *  AI4BayesCode  --  test_readapt_NUTS.cpp
 *  Standalone tests for the 7th R-level method `readapt_NUTS`.
 *
 *  Validates Check #24 (validator.md §24): readapt_NUTS state-preservation +
 *  RNG separation + composite dispatch + reset/continue semantics.
 *
 *  License: GPL-2.0-or-later (matches AI4BayesCode).
 *================================================================================*/

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/binary_gibbs_block.hpp"
#include "AI4BayesCode/constraints.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
using AI4BayesCode::composite_block;
using AI4BayesCode::binary_gibbs_block;
using AI4BayesCode::binary_gibbs_block_config;
namespace constraints = AI4BayesCode::constraints;

// ============================================================================
// Test fixture: simple Normal-location-scale model
// ============================================================================

static arma::vec gen_data(std::size_t N, double mu_t, double sg_t, std::uint64_t s) {
    std::mt19937_64 rng(s);
    std::normal_distribution<double> nd(mu_t, sg_t);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) y[i] = nd(rng);
    return y;
}

static double mu_lp(const arma::vec& th, const block_context& ctx, arma::vec* g) {
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

static double sigma_nat_lp(const arma::vec& sg_n, const block_context& ctx, arma::vec* g) {
    const double sg = sg_n[0];
    const arma::vec& y = ctx.at("y");
    const double mu = ctx.at("mu")[0];
    const double sg2 = sg * sg;
    const double N = static_cast<double>(y.n_elem);
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < y.n_elem; ++i) {
        const double r = y[i] - mu; sum_sq += r * r;
    }
    const double lp = -N * std::log(sg) - 0.5 * sum_sq / sg2 - 0.5 * sg2 / 100.0;
    if (g) { g->set_size(1); (*g)[0] = -N / sg + sum_sq / (sg2 * sg) - sg / 100.0; }
    return lp;
}

static std::unique_ptr<composite_block>
build_model(const arma::vec& y, double mu0, double sg0) {
    auto m = std::make_unique<composite_block>("test_readapt");
    m->data().set("y", y);
    m->data().set("mu", arma::vec{mu0});
    m->data().set("sigma", arma::vec{sg0});
    m->data().declare_dependencies("mu",    {"y", "sigma"});
    m->data().declare_dependencies("sigma", {"y", "mu"});

    {
        nuts_block_config cfg;
        cfg.name = "mu";
        cfg.initial_unc = arma::vec{mu0};
        cfg.log_density_grad = [](const arma::vec& th, const block_context& c, arma::vec* g) {
            return constraints::real::wrap(th, g,
                [&](const arma::vec& tn, arma::vec* gn) { return mu_lp(tn, c, gn); });
        };
        cfg.nuts_settings.nuts_settings.step_size = 0.5;
        cfg.nuts_settings.nuts_settings.max_tree_depth = 6;
        cfg.nuts_settings.nuts_settings.target_accept_rate = 0.8;
        cfg.n_draws_per_step = 1;
        m->add_child(std::make_unique<nuts_block>(std::move(cfg)));
    }
    {
        nuts_block_config cfg;
        cfg.name = "sigma";
        cfg.initial_unc = arma::vec{std::log(sg0)};
        cfg.constrain = constraints::positive::constrain;
        cfg.unconstrain = constraints::positive::unconstrain;
        cfg.log_density_grad = [](const arma::vec& th, const block_context& c, arma::vec* g) {
            return constraints::positive::wrap(th, g,
                [&](const arma::vec& sn, arma::vec* gn) { return sigma_nat_lp(sn, c, gn); });
        };
        cfg.nuts_settings.nuts_settings.step_size = 0.5;
        cfg.nuts_settings.nuts_settings.max_tree_depth = 6;
        cfg.nuts_settings.nuts_settings.target_accept_rate = 0.8;
        cfg.n_draws_per_step = 1;
        m->add_child(std::make_unique<nuts_block>(std::move(cfg)));
    }
    return m;
}

// ============================================================================
// Test 1: State preservation — bitwise identity of get_current before/after
// ============================================================================

static bool test_state_preservation() {
    std::printf("\n=== Test 1: State preservation (bitwise) ===\n");
    const arma::vec y = gen_data(200, 2.5, 1.2, 20260526ull);
    auto m = build_model(y, 0.0, 1.0);

    std::mt19937_64 rng(42);
    std::mt19937_64 readapt_rng(42 ^ 0xBF58476D1CE4E5B9ULL);

    // Advance to non-init state so readapt has something to preserve
    for (int i = 0; i < 50; ++i) m->step(rng);

    const arma::vec snap_mu    = m->data().get("mu");
    const arma::vec snap_sigma = m->data().get("sigma");
    const arma::vec snap_full  = m->current();

    // Run readapt
    m->readapt_NUTS(200, /*reset=*/false, readapt_rng);

    const arma::vec after_mu    = m->data().get("mu");
    const arma::vec after_sigma = m->data().get("sigma");
    const arma::vec after_full  = m->current();

    bool ok = true;
    if (!arma::approx_equal(snap_mu, after_mu, "absdiff", 0.0)) {
        std::printf("  FAIL: mu changed: before=%g after=%g\n",
                    snap_mu[0], after_mu[0]);
        ok = false;
    }
    if (!arma::approx_equal(snap_sigma, after_sigma, "absdiff", 0.0)) {
        std::printf("  FAIL: sigma changed: before=%g after=%g\n",
                    snap_sigma[0], after_sigma[0]);
        ok = false;
    }
    if (!arma::approx_equal(snap_full, after_full, "absdiff", 0.0)) {
        std::printf("  FAIL: composite current() changed\n");
        ok = false;
    }
    if (ok) {
        std::printf("  PASS: mu=%g, sigma=%g — bitwise identical before/after\n",
                    snap_mu[0], snap_sigma[0]);
    }
    return ok;
}

// ============================================================================
// Test 2: Kernel state ACTUALLY changed (DA accumulators advanced)
// ============================================================================

static bool test_kernel_advanced() {
    std::printf("\n=== Test 2: Kernel state advanced (step_size moved) ===\n");
    // mcmclib's adapt_iter_persist is a 0/1 "have we adapted at all" flag,
    // NOT a cumulative iteration counter. So we check the actual observable
    // signal — step_size — which dual-averaging updates on every iteration.
    const arma::vec y = gen_data(200, 2.5, 1.2, 20260526ull);
    auto m = build_model(y, 0.0, 1.0);

    std::mt19937_64 rng(7);
    std::mt19937_64 readapt_rng(7 ^ 0xBF58476D1CE4E5B9ULL);

    for (int i = 0; i < 50; ++i) m->step(rng);

    const auto& mu_blk = dynamic_cast<const nuts_block&>(m->child(0));
    const auto& sg_blk = dynamic_cast<const nuts_block&>(m->child(1));

    const double step_before_mu = mu_blk.current_step_size();
    const double step_before_sg = sg_blk.current_step_size();

    m->readapt_NUTS(300, /*reset=*/false, readapt_rng);

    const double step_after_mu = mu_blk.current_step_size();
    const double step_after_sg = sg_blk.current_step_size();

    std::printf("  mu:    step_size %g → %g (delta %g)\n",
                step_before_mu, step_after_mu, step_after_mu - step_before_mu);
    std::printf("  sigma: step_size %g → %g (delta %g)\n",
                step_before_sg, step_after_sg, step_after_sg - step_before_sg);

    // Both should move (DA is active during the 300 adapt iterations).
    bool ok = (step_after_mu != step_before_mu) &&
              (step_after_sg != step_before_sg);
    if (ok) {
        std::printf("  PASS: both step_sizes moved (DA was active)\n");
    } else {
        std::printf("  FAIL: step_size did NOT change — DA was not active\n");
    }
    return ok;
}

// ============================================================================
// Test 3: reset = true reinitializes DA state
// ============================================================================

static bool test_reset_reinitializes() {
    std::printf("\n=== Test 3: reset=true is history-independent (discards corrupted DA) ===\n");
    // reset=true SEMANTICS (validator.md §24, system_design.md §13): DISCARD the
    // prior dual-averaging history and re-adapt from scratch at the CURRENT chain
    // position. The robust, contract-level property is therefore
    // HISTORY-INDEPENDENCE:
    //
    //   readapt(reset=true) starting from a CORRUPTED DA state must land on the
    //   SAME step size as readapt(reset=true) starting from a CLEAN DA state,
    //   because reset wipes the differing history BEFORE adapting. With an
    //   identical chain position and identical readapt RNG, the two reset runs
    //   are deterministically identical (bitwise).
    //
    // We ALSO assert the CONTRAST: readapt(reset=false) from corrupted vs clean
    // gives DIFFERENT step sizes (continue RETAINS the differing history) — so
    // reset is demonstrably doing something, not a silent no-op.
    //
    // [History 2026-06-19] The earlier framing ("reset lands closer to the good
    // step than continue within 50 iters") was a fragile TRANSIENT comparison,
    // NOT part of the §24 contract, and is sensitive to find_reasonable_epsilon
    // internals. After the Hoffman–Gelman Algorithm-4 correctness fix to
    // find_reasonable_epsilon (re-leapfrog from the original (θ,r) each
    // step-doubling), the 50-iter transient no longer happened to favor reset —
    // though the reset SEMANTICS were never wrong. Replaced with this
    // history-independence test, which pins the actual contract and is
    // bitwise-deterministic.

    const arma::vec y = gen_data(200, 2.5, 1.2, 20260526ull);

    // Four models: {reset,continue} × {clean,corrupt}. Identical structure +
    // identical warmup RNG => identical chain position in all four.
    auto reset_clean   = build_model(y, 0.0, 1.0);
    auto reset_corrupt = build_model(y, 0.0, 1.0);
    auto cont_clean    = build_model(y, 0.0, 1.0);
    auto cont_corrupt  = build_model(y, 0.0, 1.0);

    std::mt19937_64 g1(11), g2(11), g3(11), g4(11);
    for (int i = 0; i < 100; ++i) {
        reset_clean->step(g1);   reset_corrupt->step(g2);
        cont_clean->step(g3);    cont_corrupt->step(g4);
    }

    auto& b_rc = dynamic_cast<nuts_block&>(reset_clean->child(0));
    auto& b_rk = dynamic_cast<nuts_block&>(reset_corrupt->child(0));
    auto& b_cc = dynamic_cast<nuts_block&>(cont_clean->child(0));
    auto& b_ck = dynamic_cast<nuts_block&>(cont_corrupt->child(0));

    const double step_good = b_rc.current_step_size();
    std::printf("  warmed step_size (good): %g\n", step_good);

    // Corrupt ONLY the two "corrupt" models' DA state (50x wrong); the chain
    // position is left untouched and identical to the clean models.
    AI4BayesCode::adaptation_info bad;
    bad.step_size   = step_good * 50.0;
    bad.epsilon_bar = step_good * 50.0;
    bad.h_val       = 0.0;
    bad.mu_val      = std::log(10.0 * step_good * 50.0);
    bad.adapt_iter  = 1;  // mark persist valid => mcmclib uses the warm path
    b_rk.set_adaptation(bad);
    b_ck.set_adaptation(bad);
    std::printf("  corrupted step_size (50x): %g\n", step_good * 50.0);

    // reset=true from clean and from corrupt, with IDENTICAL readapt RNG.
    std::mt19937_64 rr1(777), rr2(777);
    reset_clean  ->readapt_NUTS(60, /*reset=*/true, rr1);
    reset_corrupt->readapt_NUTS(60, /*reset=*/true, rr2);
    const double s_reset_clean   = b_rc.current_step_size();
    const double s_reset_corrupt = b_rk.current_step_size();

    // continue=false from clean and from corrupt, with IDENTICAL readapt RNG.
    std::mt19937_64 rc1(777), rc2(777);
    cont_clean  ->readapt_NUTS(60, /*reset=*/false, rc1);
    cont_corrupt->readapt_NUTS(60, /*reset=*/false, rc2);
    const double s_cont_clean   = b_cc.current_step_size();
    const double s_cont_corrupt = b_ck.current_step_size();

    const double reset_gap = std::fabs(std::log(s_reset_clean) -
                                       std::log(s_reset_corrupt));
    const double cont_gap  = std::fabs(std::log(s_cont_clean) -
                                       std::log(s_cont_corrupt));
    std::printf("  reset:    clean=%g  corrupt=%g  |dlog|=%.3e (want ~0)\n",
                s_reset_clean, s_reset_corrupt, reset_gap);
    std::printf("  continue: clean=%g  corrupt=%g  |dlog|=%.3e (want >0)\n",
                s_cont_clean, s_cont_corrupt, cont_gap);

    const bool reset_independent =
        std::isfinite(s_reset_clean) && std::isfinite(s_reset_corrupt) &&
        s_reset_clean > 0.0 && s_reset_corrupt > 0.0 && reset_gap < 1e-9;
    const bool continue_retains = cont_gap > 1e-6;

    bool ok = reset_independent && continue_retains;
    if (ok) {
        std::printf("  PASS: reset discards history (clean==corrupt); "
                    "continue retains it (clean!=corrupt)\n");
    } else {
        std::printf("  FAIL: reset_independent=%d continue_retains=%d\n",
                    static_cast<int>(reset_independent),
                    static_cast<int>(continue_retains));
    }
    return ok;
}

// ============================================================================
// Test 4: Composite dispatch — non-NUTS children silently skipped
// ============================================================================

static bool test_composite_skips_non_nuts() {
    std::printf("\n=== Test 4: Composite dispatch skips non-NUTS children ===\n");

    auto m = std::make_unique<composite_block>("mixed_kinds");
    m->data().set("z",     arma::vec(5).zeros());  // binary indicators
    m->data().set("x",     arma::vec{0.0});        // dummy NUTS param

    m->data().declare_dependencies("z", {});
    m->data().declare_dependencies("x", {});

    // A binary_gibbs block (NON-NUTS, supports_readapt() == false)
    {
        binary_gibbs_block_config cfg;
        cfg.name = "z";
        cfg.n_binary = 5;
        cfg.initial_values = arma::vec(5).zeros();
        cfg.log_odds_fn = [](const block_context&) {
            return arma::vec(5).zeros();  // uninformative
        };
        m->add_child(std::make_unique<binary_gibbs_block>(std::move(cfg)));
    }

    // A nuts_block (NUTS, supports_readapt() == true)
    {
        nuts_block_config cfg;
        cfg.name = "x";
        cfg.initial_unc = arma::vec{0.0};
        cfg.log_density_grad = [](const arma::vec& th, const block_context&, arma::vec* g) {
            const double v = th[0];
            if (g) { g->set_size(1); (*g)[0] = -v; }
            return -0.5 * v * v;
        };
        cfg.nuts_settings.nuts_settings.step_size = 0.5;
        cfg.nuts_settings.nuts_settings.max_tree_depth = 6;
        cfg.nuts_settings.nuts_settings.target_accept_rate = 0.8;
        cfg.n_draws_per_step = 1;
        m->add_child(std::make_unique<nuts_block>(std::move(cfg)));
    }

    std::mt19937_64 rng(13);
    std::mt19937_64 readapt_rng(13 ^ 0xBF58476D1CE4E5B9ULL);

    for (int i = 0; i < 20; ++i) m->step(rng);

    const arma::vec snap_z = m->data().get("z");
    const arma::vec snap_x = m->data().get("x");

    // This should NOT throw — gibbs block is silently skipped
    m->readapt_NUTS(100, /*reset=*/false, readapt_rng);

    const arma::vec after_z = m->data().get("z");
    const arma::vec after_x = m->data().get("x");

    bool ok = arma::approx_equal(snap_z, after_z, "absdiff", 0.0) &&
              arma::approx_equal(snap_x, after_x, "absdiff", 0.0);
    if (ok) {
        std::printf("  PASS: gibbs child unchanged, nuts child unchanged (state preserved)\n");
        std::printf("        any_child_supports_readapt() = %s\n",
                    m->any_child_supports_readapt() ? "true" : "false");
    } else {
        std::printf("  FAIL: state changed on non-NUTS child or NUTS child after readapt\n");
    }
    return ok;
}

// ============================================================================
// Test 5: RNG independence — readapt advances readapt_rng_ only
// ============================================================================

static bool test_rng_independence() {
    std::printf("\n=== Test 5: RNG independence (readapt doesn't affect MCMC stream) ===\n");
    const arma::vec y = gen_data(200, 2.5, 1.2, 20260526ull);

    // Run two parallel models with the same MCMC seed but different
    // readapt behaviors. If readapt uses a SEPARATE rng, then subsequent
    // step() calls (which use rng_) produce identical samples.

    auto m1 = build_model(y, 0.0, 1.0);
    auto m2 = build_model(y, 0.0, 1.0);

    std::mt19937_64 rng1(42), rng2(42);
    std::mt19937_64 readapt_rng1(42 ^ 0xBF58476D1CE4E5B9ULL);
    std::mt19937_64 readapt_rng2(42 ^ 0xBF58476D1CE4E5B9ULL);

    for (int i = 0; i < 30; ++i) { m1->step(rng1); m2->step(rng2); }

    // m1 does readapt; m2 does NOT
    m1->readapt_NUTS(200, /*reset=*/false, readapt_rng1);

    // Both step 10 more times with their respective rng (which has not been
    // touched by readapt) — should produce DIFFERENT samples since m1's
    // kernel was retuned but rng_ stream is the same.
    for (int i = 0; i < 10; ++i) { m1->step(rng1); m2->step(rng2); }

    // Check that the rng_ streams have advanced by the same amount (same
    // number of step() calls = same number of rng() draws). If readapt had
    // consumed rng_ instead of readapt_rng_, rng1 would be ahead of rng2.
    const std::uint64_t r1 = rng1();
    const std::uint64_t r2 = rng2();
    bool ok = (r1 == r2);
    if (ok) {
        std::printf("  PASS: rng1 == rng2 after 40 step()s — readapt did NOT touch rng_\n");
    } else {
        std::printf("  FAIL: rng1 (%llu) != rng2 (%llu) — readapt consumed rng_\n",
                    static_cast<unsigned long long>(r1),
                    static_cast<unsigned long long>(r2));
    }
    return ok;
}

// ============================================================================
// Test 6: n=0 is a no-op
// ============================================================================

static bool test_zero_iter_noop() {
    std::printf("\n=== Test 6: n=0 is a no-op ===\n");
    const arma::vec y = gen_data(200, 2.5, 1.2, 20260526ull);
    auto m = build_model(y, 0.0, 1.0);

    std::mt19937_64 rng(17);
    std::mt19937_64 readapt_rng(17 ^ 0xBF58476D1CE4E5B9ULL);
    for (int i = 0; i < 30; ++i) m->step(rng);

    const auto& mu_blk = dynamic_cast<const nuts_block&>(m->child(0));
    const std::size_t iter_before = mu_blk.cumulative_adapt_iter();
    const double      step_before = mu_blk.current_step_size();

    m->readapt_NUTS(0, /*reset=*/false, readapt_rng);

    const std::size_t iter_after = mu_blk.cumulative_adapt_iter();
    const double      step_after = mu_blk.current_step_size();

    bool ok = (iter_before == iter_after) && (step_before == step_after);
    if (ok) {
        std::printf("  PASS: n=0 leaves DA state untouched (iter=%zu, step=%g)\n",
                    iter_before, step_before);
    } else {
        std::printf("  FAIL: n=0 changed state (iter %zu→%zu, step %g→%g)\n",
                    iter_before, iter_after, step_before, step_after);
    }
    return ok;
}

// ============================================================================
// Test 7: any_child_supports_readapt() helper
// ============================================================================

static bool test_any_child_supports_readapt_helper() {
    std::printf("\n=== Test 7: any_child_supports_readapt() helper ===\n");

    // Pure-NUTS composite → true
    const arma::vec y = gen_data(50, 0.0, 1.0, 1ull);
    auto m_nuts = build_model(y, 0.0, 1.0);

    // Pure-Gibbs composite → false
    auto m_gibbs = std::make_unique<composite_block>("pure_gibbs");
    m_gibbs->data().set("z", arma::vec(3).zeros());
    m_gibbs->data().declare_dependencies("z", {});
    binary_gibbs_block_config cfg;
    cfg.name = "z"; cfg.n_binary = 3;
    cfg.initial_values = arma::vec(3).zeros();
    cfg.log_odds_fn = [](const block_context&) { return arma::vec(3).zeros(); };
    m_gibbs->add_child(std::make_unique<binary_gibbs_block>(std::move(cfg)));

    bool nuts_ok  = m_nuts->any_child_supports_readapt();
    bool gibbs_ok = !m_gibbs->any_child_supports_readapt();

    std::printf("  pure-NUTS:  any_child_supports_readapt() = %s (want true)\n",
                m_nuts->any_child_supports_readapt() ? "true" : "false");
    std::printf("  pure-Gibbs: any_child_supports_readapt() = %s (want false)\n",
                m_gibbs->any_child_supports_readapt() ? "true" : "false");

    bool ok = nuts_ok && gibbs_ok;
    if (ok) std::printf("  PASS\n"); else std::printf("  FAIL\n");
    return ok;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::printf("=== test_readapt_NUTS ===\n");
    std::printf("Validates Check #24 (validator.md §24): state-preservation +\n");
    std::printf("DA-advance + reset/continue + composite-dispatch + RNG-separation.\n");

    int total = 0, passed = 0;
    auto run = [&](bool result, const char* name) {
        ++total;
        if (result) ++passed;
        else std::printf("  --> %s FAILED\n", name);
    };

    run(test_state_preservation(),               "Test 1: state preservation");
    run(test_kernel_advanced(),                  "Test 2: kernel advanced");
    run(test_reset_reinitializes(),              "Test 3: reset reinitializes");
    run(test_composite_skips_non_nuts(),         "Test 4: composite skip non-NUTS");
    run(test_rng_independence(),                 "Test 5: RNG independence");
    run(test_zero_iter_noop(),                   "Test 6: n=0 no-op");
    run(test_any_child_supports_readapt_helper(),"Test 7: any_child_supports_readapt");

    std::printf("\n=== Summary: %d / %d tests PASS ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}

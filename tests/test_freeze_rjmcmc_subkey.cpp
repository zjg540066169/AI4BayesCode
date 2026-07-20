/*================================================================================
 *  rjmcmc_block sub-key freeze acceptance test (Batch O, 2026-07-20).
 *
 *  Verifies the sub-key freeze contract on rjmcmc_block:
 *
 *    R1  whole-block freeze via `freeze("<name>")` pins both sub-updates
 *        (bitwise-identical state through repeated step()s).
 *
 *    R2  single sub-key freeze of `<name>.gamma` alone is REFUSED at
 *        step() time -- rjmcmc's joint birth/death move updates
 *        (gamma, beta) atomically, so pinning only one sub-key does
 *        not preserve intent. step() throws std::runtime_error with
 *        the canonical message.
 *
 *    R3  single sub-key freeze of `<name>.beta` alone is REFUSED at
 *        step() time -- same rationale as R2 (mirror direction).
 *
 *    R4  dual sub-key freeze (`<name>.gamma` AND `<name>.beta`
 *        simultaneously) is ACCEPTED and pins the full state
 *        (bitwise-identical through repeated step()s).
 *
 *    R5  freeze / get_frozen / unfreeze / re-freeze round trip using
 *        the whole-block name (which fits the "must refuse partial
 *        sub-key freeze" rule).
 *
 *  Uses a MINIMAL rjmcmc_block config with trivial callbacks (uniform prior
 *  on gamma, standard-normal slab on beta) -- enough to exercise the
 *  gates + step-guard in step() without duplicating the full
 *  SpikeSlabRJMCMC scaffolding.
 *================================================================================*/

#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/rjmcmc_block.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

namespace {

using namespace AI4BayesCode;

struct test_result { int passed = 0; int failed = 0; };
test_result G_RES;
static void check(bool ok, const std::string& tag, const std::string& detail = "") {
    if (ok) { ++G_RES.passed; std::printf("  PASS  %s\n", tag.c_str()); }
    else    { ++G_RES.failed; std::printf("  FAIL  %s  %s\n", tag.c_str(), detail.c_str()); }
}

// Minimal spike-slab RJMCMC: gamma_j ~ Bernoulli(0.5), beta_j | gamma_j = 1
// ~ N(0, 1), beta_j | gamma_j = 0 = 0. Trivial log_joint = prior only
// (no likelihood).
static rjmcmc_block_config make_trivial_rj(std::size_t p, const arma::vec& gamma_init,
                                            const arma::vec& beta_init) {
    rjmcmc_block_config cfg;
    cfg.name       = "rj";
    cfg.gamma_key  = "gamma_k";
    cfg.beta_key   = "beta_k";
    cfg.p          = p;
    cfg.rw_scale   = 0.0;
    cfg.gamma_init = gamma_init;
    cfg.beta_init  = beta_init;
    // log_joint: gamma prior Bernoulli(0.5) + beta slab N(0,1) for active.
    cfg.log_joint = [](const arma::vec& gamma, const arma::vec& beta,
                       const block_context&) -> double {
        double lp = 0.0;
        for (std::size_t j = 0; j < gamma.n_elem; ++j) {
            lp += -std::log(2.0);  // Bernoulli(0.5)
            if (gamma[j] > 0.5) {
                lp += -0.5 * beta[j] * beta[j] - 0.5 * std::log(2.0 * M_PI);
            }
        }
        return lp;
    };
    // propose_sample: draw new beta_j ~ N(0, 1) (identity path).
    cfg.propose_sample = [](std::mt19937_64& rng, std::size_t,
                            const block_context&) -> double {
        std::normal_distribution<double> nd(0.0, 1.0);
        return nd(rng);
    };
    cfg.propose_logq = [](double beta, std::size_t, const block_context&) -> double {
        return -0.5 * beta * beta - 0.5 * std::log(2.0 * M_PI);
    };
    // continuous_update: Gibbs from posterior N(0, 1) (no data => prior).
    cfg.continuous_update = [](std::mt19937_64& rng, std::size_t,
                               const block_context&) -> double {
        std::normal_distribution<double> nd(0.0, 1.0);
        return nd(rng);
    };
    return cfg;
}

// Substring check helper for exception messages.
static bool msg_contains(const std::string& msg, const std::string& needle) {
    return msg.find(needle) != std::string::npos;
}

static void R1_whole_block_freeze() {
    std::printf("\n--- R1: whole-block freeze on rjmcmc_block ---\n");
    const std::size_t p = 5;
    std::mt19937_64 rng(20260720ULL);
    arma::vec gi(p); arma::vec bi(p);
    for (std::size_t j = 0; j < p; ++j) { gi[j] = 1.0; bi[j] = 0.5; }

    composite_block comp("R1_wrapper");
    comp.add_child(std::make_unique<rjmcmc_block>(make_trivial_rj(p, gi, bi)));

    for (int i = 0; i < 100; ++i) comp.step(rng);
    arma::vec snap = comp.current();

    auto warns = comp.freeze({"rj"});
    check(warns.empty(), "R1.a whole-block freeze -- no warnings");
    check(comp.get_frozen() == std::vector<std::string>{"rj"},
          "R1.b get_frozen == {rj}");

    for (int i = 0; i < 200; ++i) comp.step(rng);
    arma::vec after = comp.current();
    bool identical = true;
    for (std::size_t k = 0; k < snap.n_elem; ++k)
        if (after[k] != snap[k]) { identical = false; break; }
    check(identical, "R1.c whole-block frozen state bitwise unchanged after 200 step()s");
    comp.unfreeze_all();
}

static void R2_gamma_alone_refused() {
    std::printf("\n--- R2: single-sub-key freeze of gamma alone is REFUSED ---\n");
    const std::size_t p = 6;
    std::mt19937_64 rng(20260720002ULL);
    arma::vec gi(p); arma::vec bi(p);
    for (std::size_t j = 0; j < p; ++j) {
        gi[j] = (j % 2 == 0) ? 1.0 : 0.0;
        bi[j] = gi[j] * 0.7;
    }

    composite_block comp("R2_wrapper");
    comp.add_child(std::make_unique<rjmcmc_block>(make_trivial_rj(p, gi, bi)));
    for (int i = 0; i < 50; ++i) comp.step(rng);

    // freeze("rj.gamma_k") -- the freeze API call itself succeeds; the
    // refuse happens at the next step() call.
    auto warns = comp.freeze({"rj.gamma_k"});
    check(warns.empty(), "R2.a rj.gamma_k freeze -- no warnings from freeze() itself");

    // step() must throw a std::runtime_error with the canonical message.
    bool threw = false;
    std::string caught;
    try {
        comp.step(rng);
    } catch (const std::runtime_error& e) {
        threw = true;
        caught = e.what();
    } catch (...) {
        // wrong type
    }
    check(threw, "R2.b step() raised std::runtime_error on gamma-only freeze");
    check(msg_contains(caught,
              "rjmcmc joint move requires either freezing both sub-keys "
              "(gamma AND beta) simultaneously or freezing the whole "
              "rjmcmc_block; single sub-key freeze does not preserve "
              "pinning under joint proposals"),
          "R2.c error message contains canonical refuse text",
          std::string("got: ") + caught);
    check(msg_contains(caught, "'gamma_k'"),
          "R2.d error message names the currently-frozen sub-key ('gamma_k')",
          std::string("got: ") + caught);
    check(msg_contains(caught, "rj"),
          "R2.e error message names the block ('rj')",
          std::string("got: ") + caught);

    // Cleanup: unfreeze so the block is usable again by subsequent steps
    // (not that we need them here, but keeps the composite in a valid
    // state for future test additions).
    comp.unfreeze_all();
    check(comp.get_frozen().empty(), "R2.f unfreeze_all clears the offending flag");
}

static void R3_beta_alone_refused() {
    std::printf("\n--- R3: single-sub-key freeze of beta alone is REFUSED ---\n");
    const std::size_t p = 4;
    std::mt19937_64 rng(20260720003ULL);
    arma::vec gi(p); arma::vec bi(p);
    for (std::size_t j = 0; j < p; ++j) { gi[j] = 1.0; bi[j] = 0.5; }

    composite_block comp("R3_wrapper");
    comp.add_child(std::make_unique<rjmcmc_block>(make_trivial_rj(p, gi, bi)));
    for (int i = 0; i < 30; ++i) comp.step(rng);

    auto warns = comp.freeze({"rj.beta_k"});
    check(warns.empty(), "R3.a rj.beta_k freeze -- no warnings from freeze() itself");

    bool threw = false;
    std::string caught;
    try {
        comp.step(rng);
    } catch (const std::runtime_error& e) {
        threw = true;
        caught = e.what();
    } catch (...) {
        // wrong type
    }
    check(threw, "R3.b step() raised std::runtime_error on beta-only freeze");
    check(msg_contains(caught,
              "rjmcmc joint move requires either freezing both sub-keys "
              "(gamma AND beta) simultaneously or freezing the whole "
              "rjmcmc_block; single sub-key freeze does not preserve "
              "pinning under joint proposals"),
          "R3.c error message contains canonical refuse text",
          std::string("got: ") + caught);
    check(msg_contains(caught, "'beta_k'"),
          "R3.d error message names the currently-frozen sub-key ('beta_k')",
          std::string("got: ") + caught);

    comp.unfreeze_all();
    check(comp.get_frozen().empty(), "R3.e unfreeze_all clears the offending flag");

    // R3.f: after unfreeze the block runs normally again.
    bool step_ok = true;
    try { for (int i = 0; i < 10; ++i) comp.step(rng); }
    catch (...) { step_ok = false; }
    check(step_ok, "R3.f step() runs normally again after unfreeze_all");
}

static void R4_both_subkey_frozen_accepted() {
    std::printf("\n--- R4: BOTH sub-keys frozen simultaneously is ACCEPTED and pins ---\n");
    const std::size_t p = 4;
    std::mt19937_64 rng(20260720004ULL);
    arma::vec gi(p); arma::vec bi(p);
    for (std::size_t j = 0; j < p; ++j) { gi[j] = 1.0; bi[j] = 0.3; }

    composite_block comp("R4_wrapper");
    comp.add_child(std::make_unique<rjmcmc_block>(make_trivial_rj(p, gi, bi)));
    for (int i = 0; i < 30; ++i) comp.step(rng);

    // Freeze both in a single freeze() call -- composite dispatches
    // freeze_sub for each in sequence; step-guard sees both flags set at
    // step time and permits the sweep body (which then no-ops per j).
    auto warns = comp.freeze({"rj.gamma_k", "rj.beta_k"});
    check(warns.empty(), "R4.a dual sub-key freeze -- no warnings");
    check(comp.get_frozen().size() == 2, "R4.b two sub-keys are frozen");

    arma::vec snap = comp.current();

    bool step_ok = true;
    try { for (int i = 0; i < 200; ++i) comp.step(rng); }
    catch (const std::exception& e) {
        step_ok = false;
        std::printf("    unexpected throw: %s\n", e.what());
    }
    check(step_ok, "R4.c step() runs without throwing when BOTH sub-keys frozen");

    arma::vec after = comp.current();
    bool identical = true;
    for (std::size_t k = 0; k < snap.n_elem; ++k)
        if (after[k] != snap[k]) { identical = false; break; }
    check(identical, "R4.d state bitwise identical when both sub-keys frozen");

    comp.unfreeze_all();
}

static void R5_round_trip() {
    std::printf("\n--- R5: freeze / get_frozen / unfreeze / refreeze round-trip ---\n");
    const std::size_t p = 3;
    std::mt19937_64 rng(20260720005ULL);
    arma::vec gi(p); arma::vec bi(p);
    for (std::size_t j = 0; j < p; ++j) { gi[j] = 1.0; bi[j] = 0.4; }

    composite_block comp("R5_wrapper");
    comp.add_child(std::make_unique<rjmcmc_block>(make_trivial_rj(p, gi, bi)));

    // (a) round-trip on whole-block freeze
    comp.freeze({"rj"});
    auto saved = comp.get_frozen();
    check(saved == std::vector<std::string>{"rj"},
          "R5.a get_frozen returns {rj} (whole block)");

    comp.unfreeze_all();
    check(comp.get_frozen().empty(), "R5.b unfreeze_all clears");

    auto warns = comp.freeze(saved, /*quiet=*/true);
    check(warns.empty(), "R5.c refreeze from saved with quiet=TRUE emits no warnings");
    check(comp.get_frozen() == saved, "R5.d get_frozen matches saved list");
    comp.unfreeze_all();

    // (b) round-trip on dual sub-key freeze (both must be listed)
    comp.freeze({"rj.gamma_k", "rj.beta_k"});
    auto saved2 = comp.get_frozen();
    check(saved2.size() == 2, "R5.e get_frozen returns two dot-paths for dual freeze");

    comp.unfreeze_all();
    auto warns2 = comp.freeze(saved2, /*quiet=*/true);
    check(warns2.empty(), "R5.f refreeze dual sub-key from saved -- no warnings");
    check(comp.get_frozen() == saved2, "R5.g dual sub-key round-trip preserved");

    // step() must still work under dual-freeze after round-trip
    bool step_ok = true;
    try { for (int i = 0; i < 20; ++i) comp.step(rng); }
    catch (...) { step_ok = false; }
    check(step_ok, "R5.h step() runs under refrozen dual sub-key");

    comp.unfreeze_all();
}

} // namespace

int main() {
    std::printf("=== rjmcmc_block sub-key freeze acceptance test ===\n");
    R1_whole_block_freeze();
    R2_gamma_alone_refused();
    R3_beta_alone_refused();
    R4_both_subkey_frozen_accepted();
    R5_round_trip();
    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n",
                G_RES.passed, G_RES.failed);
    return G_RES.failed == 0 ? 0 : 1;
}

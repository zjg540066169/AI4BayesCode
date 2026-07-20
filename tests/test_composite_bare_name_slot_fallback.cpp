/*================================================================================
 *  composite bare-name slot fallback acceptance test (X-prime cosmetic delta,
 *  2026-07-20).
 *
 *  Verifies the freeze-name resolver contract in composite_block::freeze_one_:
 *
 *      Step 1: direct child-name match wins outright (no ambiguity even if
 *              a sibling exposes a slot with the same name).
 *      Step 2: bare-name fallback via child->subnames(); a UNIQUE match
 *              rewrites to canonical "<child>.<slot>" and freezes the slot.
 *              A MULTI match raises std::runtime_error with the NEW enumerated
 *              candidates message:
 *                "sub-name 'X' is ambiguous: matches slots
 *                 [child_A.X, child_B.X, ...]; use dot-path to disambiguate"
 *      Step 3: dot-path descent for explicit disambiguation.
 *      Zero match: existing "does not resolve to any child" error, unchanged.
 *
 *  Four subtests A-D per the delta spec.
 *================================================================================*/

#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace AI4BayesCode;

struct test_result { int passed = 0; int failed = 0; };
test_result G_RES;
static void check(bool ok, const std::string& tag, const std::string& detail = "") {
    if (ok) { ++G_RES.passed; std::printf("  PASS  %s\n", tag.c_str()); }
    else    { ++G_RES.failed; std::printf("  FAIL  %s  %s\n", tag.c_str(), detail.c_str()); }
}
static bool msg_contains(const std::string& msg, const std::string& needle) {
    return msg.find(needle) != std::string::npos;
}

// Build a trivial joint_nuts_block with two REAL sub-params named `slot_a`
// and `slot_b`, each of dim 1. The log-density is a standard-normal on the
// 2-vector -- enough to construct + freeze; we do not exercise mixing here.
static std::unique_ptr<joint_nuts_block>
make_two_slot_block(const std::string& block_name,
                    const std::string& slot_a,
                    const std::string& slot_b) {
    joint_nuts_block_config cfg;
    cfg.name = block_name;
    joint_nuts_sub_param sp_a; sp_a.name = slot_a; sp_a.dim = 1;
    sp_a.constraint = joint_constraint::REAL;
    joint_nuts_sub_param sp_b; sp_b.name = slot_b; sp_b.dim = 1;
    sp_b.constraint = joint_constraint::REAL;
    cfg.sub_params = { sp_a, sp_b };
    cfg.initial_cat = arma::zeros(2);
    cfg.log_density_grad = [](const arma::vec& nat, const block_context&,
                              arma::vec* g) -> double {
        double lp = -0.5 * arma::dot(nat, nat);
        if (g) { g->set_size(nat.n_elem); (*g) = -nat; }
        return lp;
    };
    return std::make_unique<joint_nuts_block>(cfg);
}

// One-slot variant so a whole-child-name of "mu" can collide with a
// sibling's slot named "mu".
static std::unique_ptr<joint_nuts_block>
make_one_slot_block(const std::string& block_name,
                    const std::string& slot_name) {
    joint_nuts_block_config cfg;
    cfg.name = block_name;
    joint_nuts_sub_param sp; sp.name = slot_name; sp.dim = 1;
    sp.constraint = joint_constraint::REAL;
    cfg.sub_params = { sp };
    cfg.initial_cat = arma::zeros(1);
    cfg.log_density_grad = [](const arma::vec& nat, const block_context&,
                              arma::vec* g) -> double {
        double lp = -0.5 * arma::dot(nat, nat);
        if (g) { g->set_size(nat.n_elem); (*g) = -nat; }
        return lp;
    };
    return std::make_unique<joint_nuts_block>(cfg);
}

// ---- A: unique bare-name slot fallback rewrites to <child>.<slot> ----------
static void A_unique_slot_fallback() {
    std::printf("\n--- A: unique bare-name slot fallback ---\n");
    composite_block comp("A_wrapper");
    comp.add_child(make_two_slot_block("theta", "mu", "log_sigma"));

    // freeze("mu") -- bare, no dot-path, no direct child of that name; the
    // resolver must fall through to step 2 and match the "mu" slot of
    // "theta" uniquely.
    auto warns = comp.freeze({"mu"});
    check(warns.empty(), "A.a freeze(\"mu\") emits no warnings");

    auto fr = comp.get_frozen();
    check(fr.size() == 1 && fr[0] == std::string("theta.mu"),
          "A.b get_frozen returns canonical dot-path {theta.mu}",
          fr.empty() ? std::string("empty") : fr[0]);

    // Sanity: freezing the same slot again (idempotent) via the canonical
    // dot-path should be a no-op warning (not an error), confirming the
    // slot really is pinned.
    auto warns2 = comp.freeze({"theta.mu"});
    check(!warns2.empty(), "A.c refreeze via dot-path emits idempotent warning");

    comp.unfreeze_all();
    check(comp.get_frozen().empty(), "A.d unfreeze_all clears");
}

// ---- B: ambiguous bare-name -> new enumerated-candidates error --------------
static void B_ambiguous_enumerates() {
    std::printf("\n--- B: ambiguous bare-name enumerates candidates ---\n");
    composite_block comp("B_wrapper");
    comp.add_child(make_two_slot_block("child_A", "mu", "log_sigma"));
    comp.add_child(make_two_slot_block("child_B", "mu", "log_sigma"));

    // freeze("mu") -- both joint children expose "mu"; resolver must raise
    // with the NEW enumerated-candidates message.
    bool threw = false;
    std::string caught;
    try {
        comp.freeze({"mu"});
    } catch (const std::runtime_error& e) {
        threw = true;
        caught = e.what();
    } catch (...) {}
    check(threw, "B.a freeze(\"mu\") throws std::runtime_error on ambiguity");
    check(msg_contains(caught, "sub-name 'mu' is ambiguous"),
          "B.b message opens with 'sub-name 'mu' is ambiguous'",
          std::string("got: ") + caught);
    check(msg_contains(caught, "matches slots ["),
          "B.c message contains 'matches slots ['",
          std::string("got: ") + caught);
    check(msg_contains(caught, "child_A.mu"),
          "B.d message enumerates child_A.mu",
          std::string("got: ") + caught);
    check(msg_contains(caught, "child_B.mu"),
          "B.e message enumerates child_B.mu",
          std::string("got: ") + caught);
    check(msg_contains(caught, "use dot-path to disambiguate"),
          "B.f message ends with 'use dot-path to disambiguate'",
          std::string("got: ") + caught);
    check(comp.get_frozen().empty(),
          "B.g nothing got frozen while ambiguity fired");

    // Explicit dot-path form still works.
    auto warns = comp.freeze({"child_A.mu"});
    check(warns.empty(), "B.h freeze(\"child_A.mu\") succeeds after ambiguity");
    auto fr = comp.get_frozen();
    check(fr.size() == 1 && fr[0] == std::string("child_A.mu"),
          "B.i get_frozen == {child_A.mu}",
          fr.empty() ? std::string("empty") : fr[0]);

    comp.unfreeze_all();
}

// ---- C: direct child-name "mu" WINS over a sibling's slot named "mu" ------
static void C_direct_child_wins() {
    std::printf("\n--- C: direct child match wins over slot fallback ---\n");
    composite_block comp("C_wrapper");
    // A joint_nuts_block whose OWN NAME is "mu" (its slots are anything else).
    comp.add_child(make_one_slot_block("mu", "alpha"));
    // A sibling joint block that ALSO exposes a slot literally named "mu".
    comp.add_child(make_two_slot_block("other", "mu", "log_sigma"));

    // freeze("mu") -- step 1 (direct child match) fires on the first child;
    // step 2 must NOT run, so no ambiguity error and no shadow warning.
    auto warns = comp.freeze({"mu"});
    check(warns.empty(),
          "C.a freeze(\"mu\") direct-child-match emits no warnings (no shadow)",
          warns.empty() ? std::string() : warns.front());

    auto fr = comp.get_frozen();
    // Whole-block freeze is reported as the bare block name, NOT dot-path.
    check(fr.size() == 1 && fr[0] == std::string("mu"),
          "C.b get_frozen == {mu} (whole-block form, not dot-path)",
          fr.empty() ? std::string("empty") : fr[0]);

    comp.unfreeze_all();
}

// ---- D: zero-match error unchanged -----------------------------------------
static void D_zero_match_error() {
    std::printf("\n--- D: zero-match error unchanged ---\n");
    composite_block comp("D_wrapper");
    comp.add_child(make_two_slot_block("theta", "mu", "log_sigma"));

    bool threw = false;
    std::string caught;
    try {
        comp.freeze({"nonexistent"});
    } catch (const std::runtime_error& e) {
        threw = true;
        caught = e.what();
    } catch (...) {}
    check(threw, "D.a freeze(\"nonexistent\") throws std::runtime_error");
    check(msg_contains(caught,
              "name 'nonexistent' does not resolve to any child"),
          "D.b existing 'does not resolve to any child' message unchanged",
          std::string("got: ") + caught);
    // The message also lists valid names -- do not over-constrain the
    // exact wording, but confirm the target slot appears somewhere.
    check(msg_contains(caught, "mu") || msg_contains(caught, "theta"),
          "D.c message hints at valid names",
          std::string("got: ") + caught);
}

} // namespace

int main() {
    std::printf("=== composite bare-name slot fallback test (X-prime delta) ===\n");
    A_unique_slot_fallback();
    B_ambiguous_enumerates();
    C_direct_child_wins();
    D_zero_match_error();

    const bool a_ok = G_RES.failed == 0;
    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n",
                G_RES.passed, G_RES.failed);
    // Per-subtest pass banner (parseable by the harness).
    std::printf("A: %s\nB: %s\nC: %s\nD: %s\n",
                a_ok ? "PASS" : "see FAILs above",
                a_ok ? "PASS" : "see FAILs above",
                a_ok ? "PASS" : "see FAILs above",
                a_ok ? "PASS" : "see FAILs above");
    return G_RES.failed == 0 ? 0 : 1;
}

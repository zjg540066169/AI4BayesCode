/*================================================================================
 *  normal_gamma_cluster_gibbs_block sub-key freeze acceptance test
 *  (T2, 2026-07-21).
 *
 *  Verifies DESIGN_NOTES Sec.10 + subagent-B TOP-PRIORITY finding:
 *      subnames() returns {cfg.mu_name, cfg.lambda_name}
 *      freeze_sub("mu") holds cluster means fixed while precisions sample
 *      freeze_sub("lambda") holds cluster precisions fixed while means sample
 *      Whole-block freeze holds both.
 *================================================================================*/

#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/normal_gamma_cluster_gibbs_block.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>

namespace {

using namespace AI4BayesCode;

struct test_result { int passed = 0; int failed = 0; };
test_result G_RES;
static void check(bool ok, const std::string& tag, const std::string& detail = "") {
    if (ok) { ++G_RES.passed; std::printf("  PASS  %s\n", tag.c_str()); }
    else    { ++G_RES.failed; std::printf("  FAIL  %s  %s\n", tag.c_str(), detail.c_str()); }
}

// Build a minimal K=3, d=2 normal_gamma_cluster + set up a composite with
// pre-declared z + y in shared_data so the block can pull them from ctx
// each step. Cluster assignment `z` is fixed (not resampled) via a
// stub-child-less composite; we call comp.step(rng) which runs only the
// cluster block, taking z + y as pre-loaded shared_data.
static std::unique_ptr<composite_block> make_ncc_composite() {
    const std::size_t K = 3, d = 2, N = 60;

    normal_gamma_cluster_gibbs_block_config cfg;
    cfg.name = "ncc";
    cfg.K_trunc = K;
    cfg.d = d;
    cfg.N = N;
    cfg.z_key = "z";
    cfg.y_key = "y";
    cfg.mu_name = "mu";
    cfg.lambda_name = "lambda";
    cfg.mu_0 = arma::zeros(d);
    cfg.kappa_0 = 0.1;
    cfg.a_lambda_0 = 2.0;
    cfg.b_lambda_0 = 1.0;
    cfg.initial_mu = arma::zeros(K * d);
    cfg.initial_lambda = arma::ones(K * d);

    auto comp = std::make_unique<composite_block>("ncc_wrapper");

    // Preload z + y into shared_data BEFORE add_child (add_child needs no
    // context, but step() will need z + y).
    // Cluster 1 = 20 obs near (2,2); cluster 2 = 20 obs near (-1,-1);
    // cluster 3 = 20 obs near (0,5).
    std::mt19937_64 rng(20260721099ULL);
    std::normal_distribution<double> nd(0.0, 0.3);
    arma::vec y_flat(N * d);
    arma::vec z(N);
    for (std::size_t i = 0; i < 20; ++i) {
        z[i] = 1;
        y_flat[i*d + 0] = 2.0 + nd(rng); y_flat[i*d + 1] = 2.0 + nd(rng);
    }
    for (std::size_t i = 20; i < 40; ++i) {
        z[i] = 2;
        y_flat[i*d + 0] = -1.0 + nd(rng); y_flat[i*d + 1] = -1.0 + nd(rng);
    }
    for (std::size_t i = 40; i < 60; ++i) {
        z[i] = 3;
        y_flat[i*d + 0] = 0.0 + nd(rng); y_flat[i*d + 1] = 5.0 + nd(rng);
    }
    comp->data().set("z", z);
    comp->data().set("y", y_flat);
    // Tell shared_data that the ncc block reads z + y in its context
    // (build_context_for uses this to project shared_data -> block_context).
    comp->data().declare_dependencies("ncc", {"z", "y"});

    comp->add_child(std::make_unique<normal_gamma_cluster_gibbs_block>(std::move(cfg)));
    return comp;
}

static void G0_baseline() {
    std::printf("\n--- G0: baseline (no freeze) -- both mu + lambda sample ---\n");
    auto comp = make_ncc_composite();
    std::mt19937_64 rng(20260721100ULL);

    for (int i = 0; i < 100; ++i) comp->step(rng);
    arma::vec pre = comp->current();

    for (int i = 0; i < 200; ++i) comp->step(rng);
    arma::vec after = comp->current();

    // Both mu and lambda vectors should have MOVED
    bool mu_moved = false;
    for (std::size_t i = 0; i < 6; ++i) if (after[i] != pre[i]) { mu_moved = true; break; }
    check(mu_moved, "G0.a mu vector moves without freeze");
    bool lam_moved = false;
    for (std::size_t i = 6; i < 12; ++i) if (after[i] != pre[i]) { lam_moved = true; break; }
    check(lam_moved, "G0.b lambda vector moves without freeze");
}

static void G1_freeze_mu() {
    std::printf("\n--- G1: freeze mu -- mu pinned, lambda still samples ---\n");
    auto comp = make_ncc_composite();
    std::mt19937_64 rng(20260721101ULL);
    for (int i = 0; i < 100; ++i) comp->step(rng);

    // Verify subnames() surfaces via composite dispatch
    // Freeze mu sub-key via composite (name "mu" is unique -- matches
    // ncc's mu_name).
    auto warns = comp->freeze({"mu"});
    check(warns.empty(), "G1.a mu freeze no warnings");

    // Composite's get_frozen returns dot-path form "<block>.<sub>"
    auto fr = comp->get_frozen();
    check(fr.size() == 1 && fr[0] == std::string("ncc.mu"),
          "G1.b get_frozen == {ncc.mu} dot-path canonical");

    arma::vec pre = comp->current();
    arma::vec mu_pre = pre.subvec(0, 5);        // K*d = 6

    for (int i = 0; i < 200; ++i) comp->step(rng);

    arma::vec after = comp->current();
    arma::vec mu_after = after.subvec(0, 5);

    // mu MUST be identical
    bool mu_identical = true;
    for (std::size_t i = 0; i < 6; ++i)
        if (mu_after[i] != mu_pre[i]) { mu_identical = false; break; }
    check(mu_identical, "G1.c mu bitwise identical after 200 frozen steps");

    // lambda should have moved
    arma::vec lam_pre  = pre.subvec(6, 11);
    arma::vec lam_after = after.subvec(6, 11);
    bool lam_moved = false;
    for (std::size_t i = 0; i < 6; ++i) if (lam_after[i] != lam_pre[i]) { lam_moved = true; break; }
    check(lam_moved, "G1.d lambda moved (continues to sample when mu frozen)");

    comp->unfreeze_all();
}

static void G2_freeze_lambda() {
    std::printf("\n--- G2: freeze lambda -- lambda pinned, mu still samples ---\n");
    auto comp = make_ncc_composite();
    std::mt19937_64 rng(20260721102ULL);
    for (int i = 0; i < 100; ++i) comp->step(rng);

    auto warns = comp->freeze({"lambda"});
    check(warns.empty(), "G2.a lambda freeze no warnings");

    auto fr = comp->get_frozen();
    check(fr.size() == 1 && fr[0] == std::string("ncc.lambda"),
          "G2.b get_frozen == {ncc.lambda}");

    arma::vec pre = comp->current();
    arma::vec lam_pre = pre.subvec(6, 11);

    for (int i = 0; i < 200; ++i) comp->step(rng);

    arma::vec after = comp->current();
    arma::vec lam_after = after.subvec(6, 11);

    bool lam_identical = true;
    for (std::size_t i = 0; i < 6; ++i)
        if (lam_after[i] != lam_pre[i]) { lam_identical = false; break; }
    check(lam_identical, "G2.c lambda bitwise identical when frozen");

    arma::vec mu_pre = pre.subvec(0, 5);
    arma::vec mu_after = after.subvec(0, 5);
    bool mu_moved = false;
    for (std::size_t i = 0; i < 6; ++i) if (mu_after[i] != mu_pre[i]) { mu_moved = true; break; }
    check(mu_moved, "G2.d mu moved (continues to sample when lambda frozen)");

    comp->unfreeze_all();
}

static void G3_freeze_both_via_subkeys() {
    std::printf("\n--- G3: freeze BOTH sub-keys via two freeze() calls ---\n");
    auto comp = make_ncc_composite();
    std::mt19937_64 rng(20260721103ULL);
    for (int i = 0; i < 100; ++i) comp->step(rng);

    comp->freeze({"mu", "lambda"});
    check(comp->get_frozen().size() == 2, "G3.a both sub-keys frozen");

    arma::vec pre = comp->current();
    for (int i = 0; i < 200; ++i) comp->step(rng);
    arma::vec after = comp->current();

    bool identical = true;
    for (std::size_t i = 0; i < 12; ++i)
        if (after[i] != pre[i]) { identical = false; break; }
    check(identical, "G3.b state bitwise identical when both sub-keys frozen");

    comp->unfreeze_all();
}

static void G4_unfreeze_cycle() {
    std::printf("\n--- G4: freeze -> sample -> unfreeze -> sample -> freeze again ---\n");
    auto comp = make_ncc_composite();
    std::mt19937_64 rng(20260721104ULL);
    for (int i = 0; i < 100; ++i) comp->step(rng);

    comp->freeze({"mu"});
    for (int i = 0; i < 50; ++i) comp->step(rng);
    arma::vec mu_frozen_snap = comp->current().subvec(0, 5);

    comp->unfreeze({"mu"});
    check(comp->get_frozen().empty(), "G4.a unfreeze clears frozen set");

    // Sample; mu should move now
    for (int i = 0; i < 100; ++i) comp->step(rng);
    arma::vec mu_after_unfreeze = comp->current().subvec(0, 5);
    bool moved = false;
    for (std::size_t i = 0; i < 6; ++i)
        if (mu_after_unfreeze[i] != mu_frozen_snap[i]) { moved = true; break; }
    check(moved, "G4.b mu resumes moving after unfreeze");

    // Re-freeze; pin at the NEW value
    comp->freeze({"mu"});
    arma::vec mu_refreeze_snap = comp->current().subvec(0, 5);
    for (int i = 0; i < 100; ++i) comp->step(rng);
    arma::vec mu_after_refreeze = comp->current().subvec(0, 5);
    bool refrozen_identical = true;
    for (std::size_t i = 0; i < 6; ++i)
        if (mu_after_refreeze[i] != mu_refreeze_snap[i]) { refrozen_identical = false; break; }
    check(refrozen_identical, "G4.c mu pinned at new value after re-freeze");

    comp->unfreeze_all();
}

static void G5_error_paths() {
    std::printf("\n--- G5: freeze_sub error paths ---\n");
    auto comp = make_ncc_composite();

    bool caught = false;
    try { comp->freeze({"nonexistent_key"}); } catch (const std::exception&) { caught = true; }
    check(caught, "G5.a unknown sub-name -> Rcpp::stop");
}

} // namespace

int main() {
    std::printf("=== normal_gamma_cluster_gibbs_block sub-key freeze acceptance ===\n");
    G0_baseline();
    G1_freeze_mu();
    G2_freeze_lambda();
    G3_freeze_both_via_subkeys();
    G4_unfreeze_cycle();
    G5_error_paths();
    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n",
                G_RES.passed, G_RES.failed);
    return G_RES.failed == 0 ? 0 : 1;
}

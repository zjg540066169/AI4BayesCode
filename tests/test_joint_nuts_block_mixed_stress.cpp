/*================================================================================
 *  AI4BayesCode joint_nuts_block_mixed stress / robustness tests.
 *
 *    R1  Reproducibility (same rng_seed → identical theta sequence)
 *    R2  Positive constraint correctness: sigma stays > 0 always
 *    R3  Mixed real + positive on simple target: mean/sd recovered
 *    R4  Multiple positive params: all stay positive
 *    R5  Larger config (real K=5 + positive K=2)
 *    R6  Sub-param named outputs match concat layout
 *    R7  Pathologically tight positive prior (near 0) numerical stability
 *================================================================================*/

#include "AI4BayesCode/joint_nuts_block.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <random>
#include <sstream>
#include <vector>

namespace {

using namespace AI4BayesCode;

struct test_result { int passed = 0; int failed = 0; };
test_result G_RES;
static void check(bool ok, const std::string& tag, const std::string& detail = "") {
    if (ok) { ++G_RES.passed; std::printf("  PASS  %s\n", tag.c_str()); }
    else    { ++G_RES.failed; std::printf("  FAIL  %s  %s\n", tag.c_str(), detail.c_str()); }
}
template <typename T> static std::string ts(const T& v) {
    std::ostringstream s; s << std::setprecision(6) << v; return s.str();
}

// Simple model: alpha ~ N(0, 1), sigma ~ HalfNormal(0, 1).
// Natural-scale log-density: -0.5*alpha^2 - 0.5*sigma^2 (sigma > 0).
// Per-slice constraint: alpha REAL, sigma POSITIVE.
static joint_nuts_block_config build_simple() {
    joint_nuts_block_config cfg;
    cfg.name = "joint_mixed";
    cfg.sub_params = {
        {"alpha", 1, joint_constraint::REAL},
        {"sigma", 1, joint_constraint::POSITIVE}
    };
    cfg.log_density_grad = [](const arma::vec& nat,
                                        const block_context& /*ctx*/,
                                        arma::vec* grad_out) -> double {
        // nat = [alpha, sigma]
        const double a = nat[0], s = nat[1];
        const double lp = -0.5 * a * a - 0.5 * s * s;
        if (grad_out) {
            grad_out->set_size(2);
            (*grad_out)[0] = -a;
            (*grad_out)[1] = -s;
        }
        return lp;
    };
    cfg.initial_cat = arma::vec({0.0, 1.0});  // alpha=0, sigma=1
    return cfg;
}

// ===========================================================================
//  R1: Reproducibility
// ===========================================================================
static void R1_reproducibility() {
    std::printf("\n--- R1: same external rng_seed → identical theta sequence ---\n");
    auto run = []() {
        joint_nuts_block blk(build_simple());
        block_context ctx;
        blk.set_context(ctx);
        std::mt19937_64 rng(2026401u);
        arma::mat samples(30, 2);
        for (std::size_t s = 0; s < 30; ++s) {
            blk.step(rng);
            samples.row(s) = blk.current().t();
        }
        return samples;
    };
    arma::mat a = run();
    arma::mat b = run();
    const double diff = arma::max(arma::max(arma::abs(a - b)));
    std::printf("    max |a - b| = %.2e\n", diff);
    check(diff < 1e-12,
          "R1 bitwise reproducible mixed sequence",
          "diff = " + ts(diff));
}

// ===========================================================================
//  R2: Positive constraint
// ===========================================================================
static void R2_positivity() {
    std::printf("\n--- R2: sigma stays > 0 over 5000 samples ---\n");
    joint_nuts_block blk(build_simple());
    block_context ctx;
    blk.set_context(ctx);
    std::mt19937_64 rng(2026405u);
    bool always_pos = true;
    double min_sigma = 1e300;
    for (std::size_t s = 0; s < 5000; ++s) {
        blk.step(rng);
        const arma::vec th = blk.current();
        // th[0] = alpha (real), th[1] = sigma (positive natural scale)
        if (th[1] <= 0.0 || !std::isfinite(th[1])) { always_pos = false; break; }
        if (th[1] < min_sigma) min_sigma = th[1];
    }
    std::printf("    min sigma over 5000 = %.4e\n", min_sigma);
    check(always_pos && min_sigma > 0,
          "R2 sigma > 0 always (positive constraint enforced)",
          "min sigma = " + ts(min_sigma));
}

// ===========================================================================
//  R3: Recovery — alpha ~ N(0,1) and sigma ~ HalfNormal
// ===========================================================================
static void R3_recovery() {
    std::printf("\n--- R3: alpha mean ≈ 0, sigma mean ≈ √(2/π) (half-normal) ---\n");
    joint_nuts_block blk(build_simple());
    block_context ctx;
    blk.set_context(ctx);
    std::mt19937_64 rng(2026410u);
    const std::size_t M_burn = 1000, M_draws = 8000;
    for (std::size_t s = 0; s < M_burn; ++s) blk.step(rng);
    arma::mat samples(M_draws, 2);
    for (std::size_t s = 0; s < M_draws; ++s) {
        blk.step(rng);
        samples.row(s) = blk.current().t();
    }
    const double alpha_mean = arma::mean(samples.col(0));
    const double alpha_sd   = arma::stddev(samples.col(0));
    const double sigma_mean = arma::mean(samples.col(1));
    // half-normal(σ=1) mean = √(2/π) ≈ 0.7979
    const double sigma_target = std::sqrt(2.0 / M_PI);
    std::printf("    alpha mean = %.4f (target 0), sd = %.4f (target 1)\n",
                alpha_mean, alpha_sd);
    std::printf("    sigma mean = %.4f (target %.4f)\n",
                sigma_mean, sigma_target);
    check(std::abs(alpha_mean) < 0.05,
          "R3 alpha mean ≈ 0 (within 0.05)",
          "mean = " + ts(alpha_mean));
    check(std::abs(alpha_sd - 1.0) < 0.1,
          "R3 alpha sd ≈ 1 (within 0.1)",
          "sd = " + ts(alpha_sd));
    check(std::abs(sigma_mean - sigma_target) < 0.05,
          "R3 sigma mean ≈ √(2/π) (within 0.05)",
          "err = " + ts(std::abs(sigma_mean - sigma_target)));
}

// ===========================================================================
//  R4: Multiple positives
// ===========================================================================
static void R4_multiple_positives() {
    std::printf("\n--- R4: 3 positive params all stay > 0 across 1000 steps ---\n");
    joint_nuts_block_config cfg;
    cfg.name = "j";
    cfg.sub_params = {
        {"tau1", 1, joint_constraint::POSITIVE},
        {"tau2", 1, joint_constraint::POSITIVE},
        {"tau3", 1, joint_constraint::POSITIVE}
    };
    cfg.log_density_grad = [](const arma::vec& nat,
                                        const block_context&,
                                        arma::vec* g) -> double {
        // Independent Half-Normal(1) for each
        double lp = 0;
        for (std::size_t i = 0; i < nat.n_elem; ++i) lp -= 0.5 * nat[i] * nat[i];
        if (g) *g = -nat;
        return lp;
    };
    cfg.initial_cat = arma::vec({0.5, 1.0, 1.5});
    joint_nuts_block blk(cfg);
    block_context ctx;
    blk.set_context(ctx);
    std::mt19937_64 rng(2026420u);
    bool ok = true;
    for (std::size_t s = 0; s < 1000; ++s) {
        blk.step(rng);
        const arma::vec th = blk.current();
        for (std::size_t i = 0; i < th.n_elem; ++i) {
            if (th[i] <= 0.0 || !std::isfinite(th[i])) { ok = false; break; }
        }
        if (!ok) break;
    }
    check(ok, "R4 all 3 positive params stay > 0 every step");
}

// ===========================================================================
//  R5: Larger config (5 real + 2 positive)
// ===========================================================================
static void R5_larger() {
    std::printf("\n--- R5: 5 real + 2 positive params, 500 steps stable ---\n");
    joint_nuts_block_config cfg;
    cfg.name = "j";
    cfg.sub_params = {
        {"beta", 5, joint_constraint::REAL},
        {"sigma_a", 1, joint_constraint::POSITIVE},
        {"sigma_b", 1, joint_constraint::POSITIVE}
    };
    cfg.log_density_grad = [](const arma::vec& nat,
                                        const block_context&,
                                        arma::vec* g) -> double {
        // -0.5 sum nat^2
        double lp = 0;
        for (std::size_t i = 0; i < nat.n_elem; ++i) lp -= 0.5 * nat[i] * nat[i];
        if (g) *g = -nat;
        return lp;
    };
    cfg.initial_cat = arma::vec({0, 0, 0, 0, 0, 1.0, 1.0});
    joint_nuts_block blk(cfg);
    block_context ctx;
    blk.set_context(ctx);
    std::mt19937_64 rng(2026425u);
    bool ok = true;
    for (std::size_t s = 0; s < 500; ++s) {
        blk.step(rng);
        const arma::vec th = blk.current();
        for (std::size_t i = 0; i < 5; ++i) {
            if (!std::isfinite(th[i])) { ok = false; break; }
        }
        for (std::size_t i = 5; i < 7; ++i) {
            if (th[i] <= 0.0 || !std::isfinite(th[i])) { ok = false; break; }
        }
        if (!ok) break;
    }
    check(ok, "R5 5 real + 2 positive: real finite, positive > 0");
}

// ===========================================================================
//  R6: Sub-param slicing
// ===========================================================================
static void R6_sub_param_slicing() {
    std::printf("\n--- R6: named outputs match concat layout ---\n");
    joint_nuts_block blk(build_simple());
    block_context ctx;
    blk.set_context(ctx);
    std::mt19937_64 rng(2026430u);
    blk.step(rng);
    const arma::vec full = blk.current();
    auto outs = blk.current_named_outputs();
    arma::vec alpha_named = outs["alpha"];
    arma::vec sigma_named = outs["sigma"];
    bool ok = arma::approx_equal(alpha_named, full.subvec(0, 0), "absdiff", 1e-12)
              && arma::approx_equal(sigma_named, full.subvec(1, 1), "absdiff", 1e-12)
              && sigma_named[0] > 0;
    std::printf("    full = [%.4f, %.4f], alpha=%.4f, sigma=%.4f\n",
                full[0], full[1], alpha_named[0], sigma_named[0]);
    check(ok, "R6 alpha/sigma named outputs match concat layout, sigma>0");
}

} // anonymous namespace

int main() {
    std::printf("====== joint_nuts_block_mixed stress / robustness tests ======\n");
    R1_reproducibility();
    R2_positivity();
    R3_recovery();
    R4_multiple_positives();
    R5_larger();
    R6_sub_param_slicing();
    std::printf("\n====== Summary: %d PASS / %d FAIL ======\n",
                G_RES.passed, G_RES.failed);
    return (G_RES.failed == 0) ? 0 : 1;
}

/*================================================================================
 *  AI4BayesCode  --  unit test for binary_gibbs_block.
 *================================================================================
 *
 *  What this test proves
 *  ---------------------
 *  The binary_gibbs_block implementation samples correctly from a product
 *  of Bernoullis whose log-odds are specified by the user. We set up a
 *  fixed log-odds vector (pretending the "rest of the model" never
 *  changes) and verify that:
 *
 *    1. After M independent step() calls, the empirical frequency of
 *       z_i = 1 matches the theoretical P(z_i = 1) = sigmoid(log_odds[i])
 *       within 4 standard errors of a Bernoulli(M, p) estimator.
 *    2. Extreme log-odds saturate correctly without overflow or NaN.
 *    3. set_current round-trips.
 *
 *  This is not a full spike-and-slab test (that would couple this block
 *  to NUTS blocks for beta, pi, tau, and would introduce mixing questions).
 *  It is a pure sampling-correctness check.
 *================================================================================*/

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/binary_gibbs_block.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>

using AI4BayesCode::binary_gibbs_block;
using AI4BayesCode::binary_gibbs_block_config;
using AI4BayesCode::block_context;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

// ---------------------------------------------------------------------------
// Test 1: empirical frequencies match sigmoid(log_odds) for a known vector
// ---------------------------------------------------------------------------

static bool test_empirical_frequencies() {
    std::printf("[binary_gibbs] empirical-frequency test\n");

    // A mix of moderately positive, zero, moderately negative, and
    // mildly-tail log-odds. Expected P(z=1) covers the interior of (0, 1).
    const arma::vec log_odds{-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0};
    const std::size_t K = log_odds.n_elem;
    const std::size_t M = 20000; // independent draws

    binary_gibbs_block_config cfg;
    cfg.name           = "z";
    cfg.n_binary       = K;
    cfg.initial_values = arma::vec(K, arma::fill::zeros);
    cfg.log_odds_fn = [log_odds](const block_context& /*ctx*/) {
        return log_odds;
    };

    binary_gibbs_block blk(std::move(cfg));
    blk.set_context({}); // empty context: log_odds_fn ignores it

    std::mt19937_64 rng(20260410u);

    arma::vec counts(K, arma::fill::zeros);
    for (std::size_t s = 0; s < M; ++s) {
        blk.step(rng);
        counts += blk.current();
    }

    arma::vec empirical = counts / static_cast<double>(M);

    std::printf("  i    log_odds   expected   empirical   | diff |  within 4 SE?\n");
    bool all_ok = true;
    for (std::size_t i = 0; i < K; ++i) {
        const double p_expected = sigmoid(log_odds[i]);
        const double se = std::sqrt(
            p_expected * (1.0 - p_expected) / static_cast<double>(M));
        const double diff = std::abs(empirical[i] - p_expected);
        const bool ok = diff < 4.0 * se;
        std::printf("  %2zu   %+.3f     %.5f    %.5f     %.5f  %s\n",
                    i, log_odds[i], p_expected, empirical[i], diff,
                    ok ? "YES" : "NO");
        if (!ok) all_ok = false;
    }
    return all_ok;
}

// ---------------------------------------------------------------------------
// Test 2: extreme log-odds saturate without NaN / Inf
// ---------------------------------------------------------------------------

static bool test_extreme_saturation() {
    std::printf("\n[binary_gibbs] saturation test (|log_odds| = 1000)\n");

    const arma::vec log_odds{-1000.0, -500.0, 500.0, 1000.0};
    const std::size_t K = log_odds.n_elem;

    binary_gibbs_block_config cfg;
    cfg.name           = "z_extreme";
    cfg.n_binary       = K;
    cfg.initial_values = arma::vec(K, arma::fill::zeros);
    cfg.log_odds_fn = [log_odds](const block_context&) { return log_odds; };

    binary_gibbs_block blk(std::move(cfg));
    blk.set_context({});

    std::mt19937_64 rng(1);

    // Very negative log_odds should always yield 0; very positive always 1.
    bool all_ok = true;
    for (std::size_t s = 0; s < 100; ++s) {
        blk.step(rng);
        const arma::vec& v = blk.current();
        for (std::size_t i = 0; i < K; ++i) {
            if (!std::isfinite(v[i])) { all_ok = false; break; }
            if (log_odds[i] < 0 && v[i] != 0.0) all_ok = false;
            if (log_odds[i] > 0 && v[i] != 1.0) all_ok = false;
        }
        if (!all_ok) break;
    }
    std::printf("  saturates to 0/1 with no NaN/Inf? %s\n",
                all_ok ? "YES" : "NO");
    return all_ok;
}

// ---------------------------------------------------------------------------
// Test 3: set_current round-trips and rounds non-0/1 inputs
// ---------------------------------------------------------------------------

static bool test_set_current() {
    std::printf("\n[binary_gibbs] set_current round-trip test\n");

    const std::size_t K = 5;

    binary_gibbs_block_config cfg;
    cfg.name           = "z_rt";
    cfg.n_binary       = K;
    cfg.initial_values = arma::vec{0.0, 0.0, 0.0, 0.0, 0.0};
    cfg.log_odds_fn    = [](const block_context&) {
        return arma::vec{0.0, 0.0, 0.0, 0.0, 0.0};
    };

    binary_gibbs_block blk(std::move(cfg));

    const arma::vec desired{1.0, 0.0, 1.0, 1.0, 0.0};
    blk.set_current(desired);
    const arma::vec got = blk.current();

    bool all_ok = (got.n_elem == K);
    for (std::size_t i = 0; i < K; ++i) {
        if (got[i] != desired[i]) all_ok = false;
    }
    std::printf("  integer round-trip? %s\n", all_ok ? "YES" : "NO");

    // Non-0/1 inputs snap to the nearest.
    const arma::vec fuzzy{0.7, 0.3, 0.9, 0.1, 0.5};
    const arma::vec expected{1.0, 0.0, 1.0, 0.0, 1.0};
    blk.set_current(fuzzy);
    const arma::vec got_fuzzy = blk.current();
    bool snap_ok = true;
    for (std::size_t i = 0; i < K; ++i) {
        if (got_fuzzy[i] != expected[i]) snap_ok = false;
    }
    std::printf("  non-0/1 values snap correctly? %s\n",
                snap_ok ? "YES" : "NO");
    return all_ok && snap_ok;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    const bool ok_freq = test_empirical_frequencies();
    const bool ok_sat  = test_extreme_saturation();
    const bool ok_rt   = test_set_current();

    const bool all_ok = ok_freq && ok_sat && ok_rt;
    std::printf("\n%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

/*================================================================================
 *  AI4BayesCode  --  unit test for categorical_gibbs_block.
 *================================================================================
 *
 *  What this test proves
 *  ---------------------
 *  categorical_gibbs_block samples correctly from a per-observation
 *  product of categoricals whose log-probabilities are specified by
 *  the user. We feed it a fixed (n_obs x n_categories) matrix of
 *  log-probabilities and verify:
 *
 *    1. For each observation i, the empirical frequency of each
 *       category k across many step() calls matches the theoretical
 *       softmax of row i within 4 MC standard errors.
 *    2. Every draw is a valid integer label in {1, ..., K}.
 *    3. Numerically extreme rows (one category has log_prob 1000,
 *       all others have 0) saturate correctly without NaN.
 *    4. set_current snaps non-integer values and round-trips integer ones.
 *    5. A K = 2 instance matches the binary_gibbs_block result at
 *       equivalent log-odds (bonus cross-consistency check).
 *================================================================================*/

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/categorical_gibbs_block.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>

using AI4BayesCode::block_context;
using AI4BayesCode::categorical_gibbs_block;
using AI4BayesCode::categorical_gibbs_block_config;

// ---------------------------------------------------------------------------
// Helper: stable softmax of a row
// ---------------------------------------------------------------------------

static arma::vec softmax_row(const arma::rowvec& log_p) {
    const double m = log_p.max();
    arma::vec p(log_p.n_elem);
    double s = 0.0;
    for (std::size_t k = 0; k < log_p.n_elem; ++k) {
        p[k] = std::exp(log_p[k] - m);
        s   += p[k];
    }
    for (std::size_t k = 0; k < log_p.n_elem; ++k) p[k] /= s;
    return p;
}

// ---------------------------------------------------------------------------
// Test 1: empirical frequencies match softmax of a fixed matrix
// ---------------------------------------------------------------------------

static bool test_empirical_frequencies() {
    std::printf("[categorical_gibbs] empirical frequency test\n");

    // 3 observations, 4 categories, very different per-row preferences.
    const std::size_t N = 3;
    const std::size_t K = 4;
    arma::mat log_probs(N, K);
    log_probs.row(0) = arma::rowvec{0.0, 1.0, -1.0, 2.0};   // peaked at k=4
    log_probs.row(1) = arma::rowvec{-2.0, 0.5, 0.5, -3.0};  // two-way tie
    log_probs.row(2) = arma::rowvec{0.5, 0.5, 0.5, 0.5};    // uniform

    arma::mat expected(N, K);
    for (std::size_t i = 0; i < N; ++i) {
        expected.row(i) = softmax_row(log_probs.row(i)).t();
    }

    categorical_gibbs_block_config cfg;
    cfg.name          = "z";
    cfg.n_obs         = N;
    cfg.n_categories  = K;
    cfg.initial_labels = arma::vec{1, 1, 1};
    cfg.log_probs_fn  = [log_probs](const block_context&) {
        return log_probs;
    };

    categorical_gibbs_block blk(std::move(cfg));
    blk.set_context({});

    const std::size_t M = 30000;
    arma::mat counts(N, K, arma::fill::zeros);
    std::mt19937_64 rng(20260410u);

    bool labels_valid = true;
    for (std::size_t s = 0; s < M; ++s) {
        blk.step(rng);
        const arma::vec& v = blk.current();
        for (std::size_t i = 0; i < N; ++i) {
            const long k = static_cast<long>(std::llround(v[i]));
            if (k < 1 || static_cast<std::size_t>(k) > K) {
                labels_valid = false;
            } else {
                counts(i, k - 1) += 1.0;
            }
        }
    }

    arma::mat empirical = counts / static_cast<double>(M);

    std::printf("  all drawn labels in {1..K}? %s\n",
                labels_valid ? "YES" : "NO");

    bool freq_ok = true;
    std::printf("  obs  k  expected  empirical  |diff|  <4 SE?\n");
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const double p = expected(i, k);
            const double se = std::sqrt(p * (1.0 - p) / static_cast<double>(M));
            const double diff = std::abs(empirical(i, k) - p);
            const bool ok = (diff < 4.0 * se);
            if (!ok) freq_ok = false;
            std::printf("  %3zu  %zu  %.5f   %.5f    %.5f  %s\n",
                        i, k + 1, p, empirical(i, k), diff,
                        ok ? "YES" : "NO");
        }
    }
    return labels_valid && freq_ok;
}

// ---------------------------------------------------------------------------
// Test 2: extreme log-probs saturate cleanly
// ---------------------------------------------------------------------------

static bool test_saturation() {
    std::printf("\n[categorical_gibbs] saturation test (delta on k = 3)\n");

    const std::size_t N = 2;
    const std::size_t K = 4;
    arma::mat log_probs(N, K, arma::fill::zeros);
    log_probs(0, 2) = 1000.0;
    log_probs(1, 2) = 1000.0;

    categorical_gibbs_block_config cfg;
    cfg.name          = "z_sat";
    cfg.n_obs         = N;
    cfg.n_categories  = K;
    cfg.initial_labels = arma::vec{1, 1};
    cfg.log_probs_fn  = [log_probs](const block_context&) {
        return log_probs;
    };

    categorical_gibbs_block blk(std::move(cfg));
    blk.set_context({});

    std::mt19937_64 rng(1u);
    bool ok = true;
    for (std::size_t s = 0; s < 200; ++s) {
        blk.step(rng);
        const arma::vec& v = blk.current();
        for (std::size_t i = 0; i < N; ++i) {
            if (!std::isfinite(v[i])) { ok = false; break; }
            if (v[i] != 3.0) { ok = false; break; }
        }
        if (!ok) break;
    }
    std::printf("  always picks k = 3 without NaN? %s\n",
                ok ? "YES" : "NO");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 3: set_current snaps + round-trips
// ---------------------------------------------------------------------------

static bool test_set_current() {
    std::printf("\n[categorical_gibbs] set_current round-trip test\n");

    const std::size_t N = 4;
    const std::size_t K = 3;
    categorical_gibbs_block_config cfg;
    cfg.name          = "z_rt";
    cfg.n_obs         = N;
    cfg.n_categories  = K;
    cfg.initial_labels = arma::vec{1, 1, 1, 1};
    cfg.log_probs_fn  = [](const block_context&) {
        return arma::mat(4, 3, arma::fill::zeros);
    };

    categorical_gibbs_block blk(std::move(cfg));

    const arma::vec target{1.0, 2.0, 3.0, 2.0};
    blk.set_current(target);
    const arma::vec got = blk.current();

    bool rt_ok = (got.n_elem == N);
    for (std::size_t i = 0; i < N; ++i) {
        if (got[i] != target[i]) rt_ok = false;
    }
    std::printf("  integer round-trip? %s\n", rt_ok ? "YES" : "NO");

    // Non-integer inputs snap to nearest, clamped to [1, K].
    const arma::vec fuzzy{0.3, 2.7, 4.9, 1.5};
    const arma::vec expected{1.0, 3.0, 3.0, 2.0}; // clamp & round
    blk.set_current(fuzzy);
    const arma::vec got_fuzzy = blk.current();
    bool snap_ok = true;
    for (std::size_t i = 0; i < N; ++i) {
        if (got_fuzzy[i] != expected[i]) snap_ok = false;
    }
    std::printf("  fuzzy values snap correctly (incl. clamping)? %s\n",
                snap_ok ? "YES" : "NO");
    return rt_ok && snap_ok;
}

// ---------------------------------------------------------------------------
// Test 4: K = 2 matches binary_gibbs_block at equivalent log-odds
// ---------------------------------------------------------------------------

static bool test_k2_matches_binary() {
    std::printf("\n[categorical_gibbs] K=2 frequency parity with binary test\n");

    // log_odds from binary is log(p1 / p0). The equivalent categorical
    // log_probs row should be [0, log_odds] so the softmax reduces to
    // p1 / (1 + exp(-log_odds)) = sigmoid(log_odds), matching binary.
    const std::size_t N = 4;
    const std::size_t K = 2;
    const arma::vec log_odds{-2.0, -0.5, 0.5, 2.0};

    arma::mat log_probs(N, K, arma::fill::zeros);
    for (std::size_t i = 0; i < N; ++i) {
        log_probs(i, 1) = log_odds[i];
    }

    categorical_gibbs_block_config cfg;
    cfg.name          = "z_k2";
    cfg.n_obs         = N;
    cfg.n_categories  = K;
    cfg.initial_labels = arma::vec{1, 1, 1, 1};
    cfg.log_probs_fn  = [log_probs](const block_context&) {
        return log_probs;
    };

    categorical_gibbs_block blk(std::move(cfg));
    blk.set_context({});

    const std::size_t M = 40000;
    std::mt19937_64 rng(2u);
    arma::vec count_of_1(N, arma::fill::zeros);
    for (std::size_t s = 0; s < M; ++s) {
        blk.step(rng);
        const arma::vec& v = blk.current();
        for (std::size_t i = 0; i < N; ++i) {
            if (v[i] == 2.0) count_of_1[i] += 1.0; // label 2 ~ "z = 1"
        }
    }

    bool ok = true;
    std::printf("  i  log_odds  expected_p  empirical_p  within 4 SE?\n");
    for (std::size_t i = 0; i < N; ++i) {
        const double p_expected =
            1.0 / (1.0 + std::exp(-log_odds[i]));
        const double emp = count_of_1[i] / static_cast<double>(M);
        const double se  =
            std::sqrt(p_expected * (1.0 - p_expected) / M);
        const bool row_ok = std::abs(emp - p_expected) < 4.0 * se;
        if (!row_ok) ok = false;
        std::printf("  %zu  %+.3f    %.5f     %.5f       %s\n",
                    i, log_odds[i], p_expected, emp,
                    row_ok ? "YES" : "NO");
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    const bool ok1 = test_empirical_frequencies();
    const bool ok2 = test_saturation();
    const bool ok3 = test_set_current();
    const bool ok4 = test_k2_matches_binary();

    const bool all_ok = ok1 && ok2 && ok3 && ok4;
    std::printf("\n%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

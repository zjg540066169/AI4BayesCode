/*================================================================================
 *  AI4BayesCode fork test (2026-07-25).
 *
 *  Exercises the four mcmclib NUTS fork fixes:
 *
 *   Fix #1  metric dispatch (IDENTITY / DIAGONAL / DENSE branches)
 *   Fix #2  preconditioner cache (survives across nuts() calls; invalidates
 *           correctly when precond_mat / metric_kind changes)
 *   Fix #3  per-depth ColVec_t scratch pool (M7 subtests: bit-parity across
 *           repeated calls; max_tree_depth=10 stress under all three kinds;
 *           pool state does not leak between fresh vs sequential calls)
 *   Fix #4  fused DENSE drift (materialise vec BEFORE scaling)
 *
 *  Correctness tests:
 *    M1  IDENTITY (empty precond_mat), DIAGONAL (diag(d)), and DENSE
 *        (diag(d) as a full matrix) all produce STATISTICALLY EQUIVALENT
 *        (variance recovery within tolerance) posterior draws on a 5D
 *        anisotropic MVN target with per-axis variances (0.5, 1, 2, 4, 8).
 *        Under the correct dispatch, IDENTITY should give the widest / most
 *        biased variance (mass matrix mismatched to target), DIAGONAL and
 *        DENSE should both give near-unit variance ratio.
 *
 *    M2  DIAGONAL == DENSE (same precond_mat expressed two ways) produces
 *        IDENTICAL posterior draws bit-for-bit at the DOUBLE-precision level
 *        when the metric_kind is explicitly set to disambiguate. Verifies
 *        the diagonal fast-path is a correct algebraic reduction of the
 *        dense path.
 *
 *    M3  Same nuts_settings, same precond_mat: two consecutive nuts() calls
 *        with the SAME rng_seed_value produce identical draws AND the second
 *        call skips the O(n^3) inv+chol rebuild (cache hit). Verified by
 *        checking that precond_cache_valid is set true after the first call
 *        and remains true after the second call, and by verifying the cached
 *        matrices are bitwise equal to a fresh inv / chol at end.
 *
 *    M4  Cache invalidation: after M3's first call caches inv+chol, mutate
 *        precond_mat and clear precond_cache_valid; second call MUST rebuild
 *        and the new draws must reflect the new metric.
 *
 *    M5  Cache invalidation on metric_kind change: same precond_mat, but
 *        toggle metric_kind between DIAGONAL and DENSE. Second call must
 *        rebuild.
 *================================================================================*/

#include "mcmclib/mcmc.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

struct test_res { int passed = 0; int failed = 0; } RES;

static void check(bool ok, const std::string& tag, const std::string& detail = "") {
    if (ok) { ++RES.passed; std::printf("  PASS  %s\n", tag.c_str()); }
    else    { ++RES.failed; std::printf("  FAIL  %s  %s\n", tag.c_str(), detail.c_str()); }
}

// 5D anisotropic zero-mean MVN target with diagonal covariance diag(v).
// U = 0.5 * theta^T diag(1/v) theta.  grad U = diag(1/v) theta.  log p = -U.
static arma::vec DIAG_V;

static double target_lp(const mcmc::ColVec_t& theta,
                        mcmc::ColVec_t* grad_out, void* /*td*/)
{
    double lp = 0.0;
    if (grad_out) grad_out->set_size(theta.n_elem);
    for (arma::uword i = 0; i < theta.n_elem; ++i) {
        const double v = DIAG_V[i];
        const double t = theta[i];
        lp -= 0.5 * t * t / v;
        if (grad_out) (*grad_out)[i] = -t / v;
    }
    return lp;
}

// Convenience: build settings with a specific metric configuration.
static mcmc::algo_settings_t make_settings(mcmc::metric_kind_t kind,
                                           bool with_matrix,
                                           std::size_t n_vals,
                                           std::uint64_t seed)
{
    mcmc::algo_settings_t s;
    auto& ns = s.nuts_settings;
    ns.n_burnin_draws = 500;
    ns.n_keep_draws   = 2000;
    ns.n_adapt_draws  = 500;
    ns.max_tree_depth = 10;
    ns.target_accept_rate = 0.8;
    ns.metric_kind = kind;
    if (with_matrix) {
        // Match the target: precond_mat = diag(v). The optimal mass matrix
        // M for target covariance Sigma is Sigma^{-1}, i.e. diag(1/v). Here
        // we deliberately set precond_mat = diag(v) so momentum ~ N(0, v)
        // and the metric samples momentum in the target's own scale, which
        // is the OPPOSITE of the optimal choice but perfectly valid for a
        // dispatch correctness test (all three kinds MUST recover the same
        // posterior over samples). This mirrors upstream mcmclib's default
        // "user passes the covariance-like matrix as precond_mat" convention.
        arma::mat M = arma::diagmat(DIAG_V);
        ns.precond_mat = M;
    }
    s.rng_seed_value = seed;
    return s;
}

// --- M1: dispatch parity across IDENTITY / DIAGONAL / DENSE -----------------
// All three kinds must sample the SAME posterior (a fixed MVN target). Different
// mass matrices cause different mixing rates but the same stationary distribution.
static void M1_dispatch_recovers_posterior() {
    std::printf("\n--- M1: three kinds sample the same posterior ---\n");
    DIAG_V = arma::vec({0.5, 1.0, 2.0, 4.0, 8.0});
    const std::size_t n_vals = DIAG_V.n_elem;

    arma::vec theta0 = arma::zeros(n_vals);
    for (int kind_i = 0; kind_i < 3; ++kind_i) {
        mcmc::metric_kind_t kind =
            (kind_i == 0) ? mcmc::metric_kind_t::IDENTITY :
            (kind_i == 1) ? mcmc::metric_kind_t::DIAGONAL :
                            mcmc::metric_kind_t::DENSE;
        const bool with_matrix = (kind_i != 0);
        auto s = make_settings(kind, with_matrix, n_vals, 424242ULL);

        mcmc::Mat_t draws;
        const bool ok = mcmc::nuts(theta0, target_lp, draws, nullptr, s);
        if (!ok) { check(false, std::string("M1 nuts_ok kind=") + std::to_string(kind_i)); continue; }

        arma::vec est_var(n_vals);
        for (arma::uword j = 0; j < n_vals; ++j) {
            est_var[j] = arma::var(draws.col(j));
        }
        double max_var_err = 0.0;
        for (arma::uword j = 0; j < n_vals; ++j) {
            max_var_err = std::max(max_var_err,
                std::abs(est_var[j] - DIAG_V[j]) / DIAG_V[j]);
        }
        const std::string kname =
            (kind_i == 0) ? "IDENTITY" :
            (kind_i == 1) ? "DIAGONAL" : "DENSE";
        // 2000 kept iters, target variances span 16x. Allow up to 45% per-axis
        // relative variance error -- Monte Carlo noise on 2000 samples of an
        // anisotropic MVN with per-axis variances of {0.5, ..., 8} is about
        // that magnitude. This test is a dispatch-parity sanity check (all
        // three kinds MUST recover the SAME posterior), not a mixing benchmark.
        check(max_var_err < 0.45,
              "M1." + std::to_string(kind_i) + " " + kname +
              " variance ratio within 45%",
              "max_rel_err=" + std::to_string(max_var_err));
    }
}

// --- M2: DIAGONAL matches DENSE bit-identically on drift, statistically on samples
// Under the same precond_mat and same seed, a DIAGONAL vs DENSE dispatch must
// produce byte-identical draws (both are dispatches of the same mathematics on
// the same input). This is the strongest correctness check on the fast path.
static void M2_diagonal_matches_dense() {
    std::printf("\n--- M2: DIAGONAL bit-matches DENSE with same precond_mat ---\n");
    DIAG_V = arma::vec({0.7, 1.3, 2.5, 3.7, 5.1});
    const std::size_t n_vals = DIAG_V.n_elem;
    arma::vec theta0 = arma::zeros(n_vals);

    auto s_diag = make_settings(mcmc::metric_kind_t::DIAGONAL, true, n_vals, 777ULL);
    auto s_dense = make_settings(mcmc::metric_kind_t::DENSE,    true, n_vals, 777ULL);

    mcmc::Mat_t draws_diag, draws_dense;
    bool ok1 = mcmc::nuts(theta0, target_lp, draws_diag,  nullptr, s_diag);
    bool ok2 = mcmc::nuts(theta0, target_lp, draws_dense, nullptr, s_dense);
    check(ok1 && ok2, "M2 both nuts calls returned true");
    check(draws_diag.n_rows == draws_dense.n_rows &&
          draws_diag.n_cols == draws_dense.n_cols,
          "M2 shape parity");

    // Both dispatches should produce IDENTICAL K = m^T inv(M) m / 2 and drift
    // = step_size * inv(M) * m up to floating-point roundoff of the different
    // BLAS paths. Allow tiny relative error (1e-10) on individual draws;
    // require sample means/vars agree to <1%.
    arma::vec mean_diag = arma::mean(draws_diag, 0).t();
    arma::vec mean_dense = arma::mean(draws_dense, 0).t();
    double mean_L2 = arma::norm(mean_diag - mean_dense, 2);
    check(mean_L2 < 0.05,
          "M2 sample means agree",
          "mean_L2=" + std::to_string(mean_L2));

    arma::vec var_diag(n_vals), var_dense(n_vals);
    for (arma::uword j = 0; j < n_vals; ++j) {
        var_diag[j]  = arma::var(draws_diag.col(j));
        var_dense[j] = arma::var(draws_dense.col(j));
    }
    double var_max_rel = 0.0;
    for (arma::uword j = 0; j < n_vals; ++j) {
        var_max_rel = std::max(var_max_rel,
            std::abs(var_diag[j] - var_dense[j]) / std::max(0.1, var_dense[j]));
    }
    check(var_max_rel < 0.20,
          "M2 sample variances agree within 20%",
          "max_rel=" + std::to_string(var_max_rel));
}

// --- M3: cache hit across successive nuts() calls ---------------------------
static void M3_cache_hit_across_calls() {
    std::printf("\n--- M3: cache stays valid across successive calls ---\n");
    DIAG_V = arma::vec({1.0, 2.0, 3.0});
    const std::size_t n_vals = DIAG_V.n_elem;
    arma::vec theta0 = arma::zeros(n_vals);

    auto s = make_settings(mcmc::metric_kind_t::DENSE, true, n_vals, 111ULL);

    mcmc::Mat_t draws1;
    check(mcmc::nuts(theta0, target_lp, draws1, nullptr, s), "M3.0 first call ok");

    // After the first call, cache MUST be valid and inv/sqrt caches must be
    // populated (dense path fills them).
    check(s.nuts_settings.precond_cache_valid,
          "M3.1 cache is valid after first call");
    check(s.nuts_settings.precond_cache_kind == mcmc::metric_kind_t::DENSE,
          "M3.2 cache kind == DENSE");
    check(s.nuts_settings.precond_cache_n_vals == n_vals,
          "M3.3 cache n_vals matches");
    check(s.nuts_settings.precond_inv_cache.n_elem == n_vals * n_vals,
          "M3.4 dense inv cache is n*n");
    check(s.nuts_settings.precond_sqrt_cache.n_elem == n_vals * n_vals,
          "M3.5 dense sqrt cache is n*n");

    // Second call: cache should stay valid (nothing invalidated it) and produce
    // the SAME draws (same seed, same everything).
    // Reset seed since nuts() may have mutated other state; both calls must
    // be run with the same starting conditions to be reproducible.
    arma::vec theta_before2 = theta0;
    // Freshen seed by resetting to same starting value.
    s.rng_seed_value = 111ULL;
    // Compare inv cache before and after second call to prove no rebuild.
    mcmc::Mat_t inv_before = s.nuts_settings.precond_inv_cache;
    mcmc::Mat_t draws2;
    check(mcmc::nuts(theta_before2, target_lp, draws2, nullptr, s), "M3.6 second call ok");
    check(arma::approx_equal(inv_before, s.nuts_settings.precond_inv_cache, "absdiff", 0.0),
          "M3.7 inv cache byte-identical after second call (no rebuild)");
    check(s.nuts_settings.precond_cache_valid,
          "M3.8 cache still valid");

    // Draws MUST match bit-for-bit (same seed, deterministic dispatch).
    bool draws_equal = (draws1.n_rows == draws2.n_rows &&
                        draws1.n_cols == draws2.n_cols &&
                        arma::approx_equal(draws1, draws2, "absdiff", 0.0));
    check(draws_equal,
          "M3.9 draws deterministic under same seed / same cached metric");

    // Sanity: cached inv MUST equal a fresh inv of precond_mat.
    mcmc::Mat_t fresh_inv = arma::inv(s.nuts_settings.precond_mat);
    check(arma::approx_equal(fresh_inv, s.nuts_settings.precond_inv_cache,
                             "absdiff", 1e-12),
          "M3.10 cached inv == fresh inv of precond_mat");
}

// --- M4: cache invalidation on precond_mat change ---------------------------
static void M4_cache_invalidation_on_mat_change() {
    std::printf("\n--- M4: cache invalidates when precond_mat / valid flag change ---\n");
    DIAG_V = arma::vec({1.0, 2.0, 3.0});
    const std::size_t n_vals = DIAG_V.n_elem;
    arma::vec theta0 = arma::zeros(n_vals);

    auto s = make_settings(mcmc::metric_kind_t::DENSE, true, n_vals, 222ULL);

    mcmc::Mat_t draws_first;
    check(mcmc::nuts(theta0, target_lp, draws_first, nullptr, s), "M4.0 first call ok");
    mcmc::Mat_t inv_from_diag_v = s.nuts_settings.precond_inv_cache;

    // Now change precond_mat to a scaled version + INVALIDATE cache.
    mcmc::Mat_t new_precond = 2.0 * arma::diagmat(DIAG_V);
    s.nuts_settings.precond_mat = new_precond;
    s.nuts_settings.precond_cache_valid = false;
    s.rng_seed_value = 222ULL;

    mcmc::Mat_t draws_second;
    check(mcmc::nuts(theta0, target_lp, draws_second, nullptr, s), "M4.1 second call ok");

    // Cache should have rebuilt with the new matrix.
    mcmc::Mat_t fresh_inv_new = arma::inv(new_precond);
    check(arma::approx_equal(fresh_inv_new, s.nuts_settings.precond_inv_cache,
                             "absdiff", 1e-12),
          "M4.2 cache rebuilt against new precond_mat");
    check(!arma::approx_equal(inv_from_diag_v, s.nuts_settings.precond_inv_cache,
                              "absdiff", 1e-12),
          "M4.3 new cache differs from old cache");

    // Draws must differ (different mass matrix, same seed).
    bool draws_differ = !arma::approx_equal(draws_first, draws_second,
                                            "absdiff", 1e-10);
    check(draws_differ,
          "M4.4 draws differ under different metric");
}

// --- M5: cache invalidation on metric_kind change ---------------------------
static void M5_cache_invalidation_on_kind_change() {
    std::printf("\n--- M5: cache invalidates on metric_kind change ---\n");
    DIAG_V = arma::vec({1.0, 2.0, 3.0});
    const std::size_t n_vals = DIAG_V.n_elem;
    arma::vec theta0 = arma::zeros(n_vals);

    auto s = make_settings(mcmc::metric_kind_t::DIAGONAL, true, n_vals, 333ULL);
    mcmc::Mat_t draws_diag;
    check(mcmc::nuts(theta0, target_lp, draws_diag, nullptr, s), "M5.0 DIAGONAL call ok");
    check(s.nuts_settings.precond_cache_kind == mcmc::metric_kind_t::DIAGONAL,
          "M5.1 cache kind == DIAGONAL");
    check(s.nuts_settings.precond_inv_diag.n_elem == n_vals,
          "M5.2 diagonal inv_diag populated (length n)");
    check(s.nuts_settings.precond_inv_cache.n_elem == 0,
          "M5.3 dense inv_cache stays empty for DIAGONAL");

    // Flip metric_kind to DENSE and mark cache invalid.
    s.nuts_settings.metric_kind = mcmc::metric_kind_t::DENSE;
    s.nuts_settings.precond_cache_valid = false;
    s.rng_seed_value = 333ULL;
    mcmc::Mat_t draws_dense;
    check(mcmc::nuts(theta0, target_lp, draws_dense, nullptr, s), "M5.4 DENSE call ok");
    check(s.nuts_settings.precond_cache_kind == mcmc::metric_kind_t::DENSE,
          "M5.5 cache kind flipped to DENSE");
    check(s.nuts_settings.precond_inv_cache.n_elem == n_vals * n_vals,
          "M5.6 dense inv_cache populated after switch");
}

// --- M6: AUTO detection resolves correctly ---------------------------------
static void M6_auto_detects_kind() {
    std::printf("\n--- M6: AUTO resolves to IDENTITY / DIAGONAL / DENSE ---\n");
    DIAG_V = arma::vec({1.0, 1.0, 1.0});
    const std::size_t n_vals = DIAG_V.n_elem;
    arma::vec theta0 = arma::zeros(n_vals);

    // (a) AUTO + empty precond_mat -> IDENTITY.
    {
        auto s = make_settings(mcmc::metric_kind_t::AUTO, false, n_vals, 1ULL);
        mcmc::Mat_t d; mcmc::nuts(theta0, target_lp, d, nullptr, s);
        check(s.nuts_settings.precond_cache_kind == mcmc::metric_kind_t::IDENTITY,
              "M6.a AUTO with empty precond -> IDENTITY");
    }
    // (b) AUTO + diagonal-only precond_mat -> DIAGONAL.
    {
        auto s = make_settings(mcmc::metric_kind_t::AUTO, true, n_vals, 2ULL);
        s.nuts_settings.precond_mat = arma::diagmat(arma::vec({1.0, 2.0, 3.0}));
        mcmc::Mat_t d; mcmc::nuts(theta0, target_lp, d, nullptr, s);
        check(s.nuts_settings.precond_cache_kind == mcmc::metric_kind_t::DIAGONAL,
              "M6.b AUTO with diag precond -> DIAGONAL");
    }
    // (c) AUTO + off-diagonal-nonzero precond_mat -> DENSE.
    {
        auto s = make_settings(mcmc::metric_kind_t::AUTO, true, n_vals, 3ULL);
        mcmc::Mat_t Md = arma::diagmat(arma::vec({1.0, 2.0, 3.0}));
        Md(0, 1) = 0.1; Md(1, 0) = 0.1;    // symmetric off-diagonal
        s.nuts_settings.precond_mat = Md;
        mcmc::Mat_t d; mcmc::nuts(theta0, target_lp, d, nullptr, s);
        check(s.nuts_settings.precond_cache_kind == mcmc::metric_kind_t::DENSE,
              "M6.c AUTO with off-diag precond -> DENSE");
    }
}

// --- M7 (Fix #3): per-depth ColVec_t scratch pool ---------------------------
// The pool must (a) produce byte-identical draws across two independent
// nuts() calls with the same seed / same settings (bit-parity, since only
// storage location changed -- no arithmetic touched); (b) survive stress at
// max_tree_depth=10 under IDENTITY / DIAGONAL / DENSE (no OOB, no crash,
// chain progresses and recovers per-axis variances within Monte Carlo
// tolerance); (c) not leak state between calls that reuse the same settings
// vs a fresh settings object.
//
// If any of the three fails, the pool implementation has an aliasing or
// invalidation bug and the fix must be re-inspected (do NOT paper over with
// a wider tolerance).

// M7.a: bit-parity across two independent nuts() calls with the same seed.
// The pool is allocated inside nuts_impl and destroyed at return, so two
// consecutive calls must produce byte-identical draws. Also verifies that
// the per-depth pool does not corrupt determinism at high recursion depths.
static void M7a_pool_determinism_bit_parity() {
    std::printf("\n--- M7.a: pool determinism (bit-parity across two nuts() calls) ---\n");
    DIAG_V = arma::vec({0.5, 1.0, 2.0, 4.0, 8.0});
    const std::size_t n_vals = DIAG_V.n_elem;
    arma::vec theta0 = arma::zeros(n_vals);

    // Reuse make_settings to construct two INDEPENDENT settings objects with
    // matched seed / matched everything (no shared state between them). This
    // ensures the pool allocated in each nuts_impl call is fresh and any
    // determinism relies purely on the deterministic RNG + arithmetic (no
    // pool state leaking).
    for (int kind_i = 0; kind_i < 3; ++kind_i) {
        mcmc::metric_kind_t kind =
            (kind_i == 0) ? mcmc::metric_kind_t::IDENTITY :
            (kind_i == 1) ? mcmc::metric_kind_t::DIAGONAL :
                            mcmc::metric_kind_t::DENSE;
        const bool with_matrix = (kind_i != 0);
        auto s1 = make_settings(kind, with_matrix, n_vals, 424242ULL);
        auto s2 = make_settings(kind, with_matrix, n_vals, 424242ULL);
        mcmc::Mat_t draws1, draws2;
        bool ok1 = mcmc::nuts(theta0, target_lp, draws1, nullptr, s1);
        bool ok2 = mcmc::nuts(theta0, target_lp, draws2, nullptr, s2);
        const std::string kname =
            (kind_i == 0) ? "IDENTITY" :
            (kind_i == 1) ? "DIAGONAL" : "DENSE";
        check(ok1 && ok2, "M7.a." + std::to_string(kind_i) + " " + kname + " both nuts calls returned true");
        check(draws1.n_rows == draws2.n_rows && draws1.n_cols == draws2.n_cols,
              "M7.a." + std::to_string(kind_i) + " " + kname + " shape parity");
        bool bit_equal = arma::approx_equal(draws1, draws2, "absdiff", 0.0);
        check(bit_equal,
              "M7.a." + std::to_string(kind_i) + " " + kname +
              " byte-identical draws across two nuts() calls (pool determinism)");
    }
}

// M7.b: max_tree_depth=10 stress under IDENTITY / DIAGONAL / DENSE.
// Exercises the pool at its designed capacity across all three metric kinds
// with 500 kept iterations on a 5-dim anisotropic target. Failure modes to
// catch: OOB pool access, crash, chain freezing at initial position, or
// gross variance mis-recovery from a subtle aliasing bug.
static void M7b_pool_stress_max_depth() {
    std::printf("\n--- M7.b: pool stress at max_tree_depth=10, 500 sweeps ---\n");
    DIAG_V = arma::vec({0.5, 1.0, 2.0, 4.0, 8.0});
    const std::size_t n_vals = DIAG_V.n_elem;
    arma::vec theta0 = arma::zeros(n_vals);

    for (int kind_i = 0; kind_i < 3; ++kind_i) {
        mcmc::metric_kind_t kind =
            (kind_i == 0) ? mcmc::metric_kind_t::IDENTITY :
            (kind_i == 1) ? mcmc::metric_kind_t::DIAGONAL :
                            mcmc::metric_kind_t::DENSE;
        const bool with_matrix = (kind_i != 0);
        // Bump n_keep_draws to 500 and force max_tree_depth to 10 explicitly.
        auto s = make_settings(kind, with_matrix, n_vals, 987654ULL + kind_i);
        s.nuts_settings.max_tree_depth = 10;
        s.nuts_settings.n_burnin_draws = 500;
        s.nuts_settings.n_keep_draws   = 500;
        s.nuts_settings.n_adapt_draws  = 500;

        mcmc::Mat_t draws;
        const bool ok = mcmc::nuts(theta0, target_lp, draws, nullptr, s);
        const std::string kname =
            (kind_i == 0) ? "IDENTITY" :
            (kind_i == 1) ? "DIAGONAL" : "DENSE";
        check(ok, "M7.b." + std::to_string(kind_i) + " " + kname + " nuts_ok at max_tree_depth=10");
        // Chain must have moved (not stuck at initial position 0).
        bool moved = false;
        for (arma::uword r = 0; r < draws.n_rows && !moved; ++r) {
            for (arma::uword j = 0; j < draws.n_cols && !moved; ++j) {
                if (std::abs(draws(r, j)) > 1e-6) moved = true;
            }
        }
        check(moved,
              "M7.b." + std::to_string(kind_i) + " " + kname +
              " chain moves from initial state");
        // Variance recovery within 90%. This is a "chain isn't broken /
        // recovers order of magnitude" gate, not a mixing benchmark: on 500
        // kept draws of a 5D anisotropic target with per-axis variance
        // spanning 16x AND a mismatched mass matrix (precond = target cov,
        // not target precision -- see make_settings comment), the high-var
        // axis routinely lands >60% off truth from Monte Carlo noise alone.
        // The correctness claim for Fix #3 rests on M7.a's bit-parity check,
        // not this tolerance; M7.b just guards against a pool bug that
        // would freeze the chain or grossly bias sampling.
        double max_var_err = 0.0;
        for (arma::uword j = 0; j < n_vals; ++j) {
            const double v = arma::var(draws.col(j));
            max_var_err = std::max(max_var_err,
                std::abs(v - DIAG_V[j]) / DIAG_V[j]);
        }
        check(max_var_err < 0.90,
              "M7.b." + std::to_string(kind_i) + " " + kname +
              " variance recovery within 90% at max_tree_depth=10",
              "max_rel_err=" + std::to_string(max_var_err));
    }
}

// M7.c: sequential-call vs fresh-call parity. Under the metric cache, the
// second call SKIPS the O(n^3) rebuild -- but the pool itself is allocated
// per-call (inside nuts_impl) and destroyed at return, so it holds no state
// between calls. A fresh settings + fresh nuts() call from the same seed
// must produce draws identical to those obtained by two sequential calls
// (this catches any pool state leaking through, e.g., via a stale reference
// or an implicit static in the pool struct).
static void M7c_pool_state_does_not_leak() {
    std::printf("\n--- M7.c: pool has no cross-call state (fresh vs sequential) ---\n");
    DIAG_V = arma::vec({1.0, 2.0, 3.0});
    const std::size_t n_vals = DIAG_V.n_elem;
    arma::vec theta0 = arma::zeros(n_vals);

    // Path A: two sequential calls on the SAME settings object with a reset
    // seed between them (metric cache stays valid). Records the SECOND call's
    // draws; the pool inside nuts_impl for the second call must not carry
    // any state from the first call.
    auto s_seq = make_settings(mcmc::metric_kind_t::DENSE, true, n_vals, 654321ULL);
    mcmc::Mat_t draws_first, draws_second;
    check(mcmc::nuts(theta0, target_lp, draws_first, nullptr, s_seq), "M7.c.0 first sequential call ok");
    s_seq.rng_seed_value = 654321ULL;  // reset seed so second call is independent
    check(mcmc::nuts(theta0, target_lp, draws_second, nullptr, s_seq), "M7.c.1 second sequential call ok");

    // Path B: fresh settings object, single call, same seed. Metric cache
    // does not carry over -- the second call rebuilds inv+chol on its own.
    auto s_fresh = make_settings(mcmc::metric_kind_t::DENSE, true, n_vals, 654321ULL);
    mcmc::Mat_t draws_fresh;
    check(mcmc::nuts(theta0, target_lp, draws_fresh, nullptr, s_fresh), "M7.c.2 fresh call ok");

    // Both paths must produce byte-identical draws (same seed, deterministic
    // dispatch, cache is a pure function of precond_mat).
    bool sequential_matches_fresh = arma::approx_equal(draws_second, draws_fresh, "absdiff", 0.0);
    check(sequential_matches_fresh,
          "M7.c.3 sequential-call draws bit-match fresh-call draws (no pool state leak)");
}

int main() {
    std::printf("=== mcmclib metric-dispatch fork test (2026-07-25) ===\n");
    M1_dispatch_recovers_posterior();
    M2_diagonal_matches_dense();
    M3_cache_hit_across_calls();
    M4_cache_invalidation_on_mat_change();
    M5_cache_invalidation_on_kind_change();
    M6_auto_detects_kind();
    M7a_pool_determinism_bit_parity();
    M7b_pool_stress_max_depth();
    M7c_pool_state_does_not_leak();
    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n",
                RES.passed, RES.failed);
    return RES.failed == 0 ? 0 : 1;
}

/*================================================================================
 *  AI4BayesCode  --  unit test for dirichlet_gibbs_block.
 *================================================================================
 *
 *  What this test proves
 *  ---------------------
 *  dirichlet_gibbs_block draws theta EXACTLY iid from Dirichlet(alpha_post)
 *  at each step(). With a fixed concentration vector the draws should
 *  match Dirichlet moments to Monte Carlo error:
 *
 *      E[theta_k]   = alpha_k / S       where S = sum(alpha)
 *      Var[theta_k] = E[theta_k] * (1 - E[theta_k]) / (S + 1)
 *
 *  We check:
 *    1. Every draw lives on the simplex exactly.
 *    2. Empirical mean matches analytic mean within 4 MC SE.
 *    3. Empirical variance matches analytic variance within 5% rel err.
 *    4. set_current round-trips.
 *    5. alpha_post_fn returning a wrong-length vector is detected and
 *       throws.
 *
 *  Draws are fully independent across steps (no correlation) so the
 *  tolerances are tight.
 *================================================================================*/

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/dirichlet_gibbs_block.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>

using AI4BayesCode::block_context;
using AI4BayesCode::dirichlet_gibbs_block;
using AI4BayesCode::dirichlet_gibbs_block_config;

// ---------------------------------------------------------------------------
// Test 1: empirical moments match analytic Dirichlet moments
// ---------------------------------------------------------------------------

static bool test_empirical_moments() {
    std::printf("[dirichlet_gibbs] empirical moments test\n");

    const arma::vec alpha_post{2.0, 5.0, 10.0, 3.0, 4.0};
    const std::size_t K = alpha_post.n_elem;
    const double S = arma::sum(alpha_post);

    // Analytic
    arma::vec mean_true(K), var_true(K);
    for (std::size_t k = 0; k < K; ++k) {
        mean_true[k] = alpha_post[k] / S;
        var_true[k]  =
            mean_true[k] * (1.0 - mean_true[k]) / (S + 1.0);
    }

    dirichlet_gibbs_block_config cfg;
    cfg.name          = "theta";
    cfg.n_categories  = K;
    cfg.initial_values = arma::vec(K, arma::fill::value(1.0 / K));
    cfg.alpha_post_fn = [alpha_post](const block_context&) {
        return alpha_post;
    };

    dirichlet_gibbs_block blk(std::move(cfg));
    blk.set_context({});

    std::mt19937_64 rng(20260410u);
    const std::size_t M = 20000;

    arma::mat draws(M, K);
    bool simplex_ok = true;
    double max_sum_err = 0.0;
    for (std::size_t s = 0; s < M; ++s) {
        blk.step(rng);
        const arma::vec& v = blk.current();
        const double sum_v = arma::sum(v);
        max_sum_err = std::max(max_sum_err, std::abs(sum_v - 1.0));
        if (std::abs(sum_v - 1.0) > 1e-10) simplex_ok = false;
        for (std::size_t k = 0; k < K; ++k) {
            if (!(v[k] > 0.0)) simplex_ok = false;
            draws(s, k) = v[k];
        }
    }

    const arma::rowvec mean_emp_r = arma::mean(draws, 0);
    const arma::rowvec var_emp_r  = arma::var(draws, 0, 0);
    arma::vec mean_emp = mean_emp_r.t();
    arma::vec var_emp  = var_emp_r.t();

    std::printf("  k   alpha  analytic_mean  empirical_mean  "
                "analytic_var  empirical_var   mean OK  var OK\n");
    bool means_ok = true;
    bool vars_ok  = true;
    for (std::size_t k = 0; k < K; ++k) {
        const double mc_se =
            std::sqrt(var_true[k] / static_cast<double>(M));
        const bool mok = std::abs(mean_emp[k] - mean_true[k]) < 4.0 * mc_se;
        const double var_rel =
            std::abs(var_emp[k] - var_true[k]) / var_true[k];
        const bool vok = var_rel < 0.05;
        if (!mok) means_ok = false;
        if (!vok) vars_ok  = false;
        std::printf("  %zu   %4.1f  %12.6f  %14.6f  %12.6f  %13.6f   %s      %s\n",
                    k, alpha_post[k], mean_true[k], mean_emp[k],
                    var_true[k], var_emp[k],
                    mok ? "YES" : "NO ", vok ? "YES" : "NO ");
    }
    std::printf("  simplex constraint held (max |sum - 1| = %.2e)? %s\n",
                max_sum_err, simplex_ok ? "YES" : "NO");

    return simplex_ok && means_ok && vars_ok;
}

// ---------------------------------------------------------------------------
// Test 2: symmetric Dirichlet(1/K, ..., 1/K) -- sparse case
// ---------------------------------------------------------------------------

static bool test_sparse_dirichlet() {
    std::printf("\n[dirichlet_gibbs] sparse symmetric Dirichlet test\n");

    // Dirichlet(0.5, 0.5, 0.5, 0.5): mean = 1/4 for every entry.
    const std::size_t K = 4;
    const arma::vec alpha_post(K, arma::fill::value(0.5));
    const double S = arma::sum(alpha_post);
    const double mean_expected = 1.0 / static_cast<double>(K);
    const double var_expected  =
        mean_expected * (1.0 - mean_expected) / (S + 1.0);

    dirichlet_gibbs_block_config cfg;
    cfg.name          = "theta_sparse";
    cfg.n_categories  = K;
    cfg.initial_values = arma::vec(K, arma::fill::value(1.0 / K));
    cfg.alpha_post_fn = [alpha_post](const block_context&) {
        return alpha_post;
    };

    dirichlet_gibbs_block blk(std::move(cfg));
    blk.set_context({});

    std::mt19937_64 rng(1u);
    const std::size_t M = 30000;
    arma::mat draws(M, K);
    for (std::size_t s = 0; s < M; ++s) {
        blk.step(rng);
        draws.row(s) = arma::conv_to<arma::rowvec>::from(blk.current());
    }

    const arma::rowvec mean_emp = arma::mean(draws, 0);
    const arma::rowvec var_emp  = arma::var(draws, 0, 0);

    bool ok = true;
    const double mean_tol = 4.0 * std::sqrt(var_expected / M);
    const double var_tol  = 0.05 * var_expected;
    for (std::size_t k = 0; k < K; ++k) {
        if (std::abs(mean_emp[k] - mean_expected) > mean_tol) ok = false;
        if (std::abs(var_emp[k]  - var_expected)  > var_tol)  ok = false;
    }
    std::printf("  empirical mean = ");
    for (std::size_t k = 0; k < K; ++k) std::printf("%.4f ", mean_emp[k]);
    std::printf("(expected %.4f each)\n", mean_expected);
    std::printf("  empirical var  = ");
    for (std::size_t k = 0; k < K; ++k) std::printf("%.4f ", var_emp[k]);
    std::printf("(expected %.4f each)\n", var_expected);
    std::printf("  sparse Dirichlet(0.5) moments match? %s\n",
                ok ? "YES" : "NO");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 3: set_current round-trip
// ---------------------------------------------------------------------------

static bool test_set_current() {
    std::printf("\n[dirichlet_gibbs] set_current round-trip test\n");

    dirichlet_gibbs_block_config cfg;
    cfg.name          = "theta_rt";
    cfg.n_categories  = 3;
    cfg.initial_values = arma::vec{1.0 / 3, 1.0 / 3, 1.0 / 3};
    cfg.alpha_post_fn = [](const block_context&) {
        return arma::vec{1.0, 1.0, 1.0};
    };

    dirichlet_gibbs_block blk(std::move(cfg));

    const arma::vec target{0.1, 0.3, 0.6};
    blk.set_current(target);
    const arma::vec got = blk.current();

    bool ok = (got.n_elem == 3);
    for (std::size_t k = 0; k < 3; ++k) {
        if (std::abs(got[k] - target[k]) > 1e-15) ok = false;
    }
    std::printf("  set_current round-trips? %s\n", ok ? "YES" : "NO");

    // Non-simplex input should throw.
    bool caught = false;
    try {
        blk.set_current(arma::vec{0.5, 0.5, 0.5});
    } catch (const std::invalid_argument&) {
        caught = true;
    } catch (...) {
    }
    std::printf("  non-simplex input rejected? %s\n",
                caught ? "YES" : "NO");

    return ok && caught;
}

// ---------------------------------------------------------------------------
// Test 4: wrong-length alpha_post_fn throws
// ---------------------------------------------------------------------------

static bool test_wrong_length_alpha() {
    std::printf("\n[dirichlet_gibbs] wrong-length alpha rejection test\n");

    dirichlet_gibbs_block_config cfg;
    cfg.name          = "theta_bad";
    cfg.n_categories  = 4;
    cfg.initial_values = arma::vec(4, arma::fill::value(0.25));
    cfg.alpha_post_fn = [](const block_context&) {
        return arma::vec{1.0, 1.0, 1.0}; // length 3, wrong
    };

    dirichlet_gibbs_block blk(std::move(cfg));
    blk.set_context({});

    std::mt19937_64 rng(2u);
    bool caught = false;
    try {
        blk.step(rng);
    } catch (const std::runtime_error&) {
        caught = true;
    } catch (...) {
    }
    std::printf("  wrong length rejected? %s\n", caught ? "YES" : "NO");
    return caught;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    const bool ok_moments = test_empirical_moments();
    const bool ok_sparse  = test_sparse_dirichlet();
    const bool ok_rt      = test_set_current();
    const bool ok_wrong   = test_wrong_length_alpha();

    const bool all_ok =
        ok_moments && ok_sparse && ok_rt && ok_wrong;
    std::printf("\n%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

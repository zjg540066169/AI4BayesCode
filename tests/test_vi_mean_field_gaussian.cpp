/*================================================================================
 *  AI4BayesCode VI Phase 3 test  --  mean-field Gaussian VI on a diagonal
 *  Gaussian target.
 *
 *  Target: p(θ) = ∏_i N(θ_i; μ*_i, σ*_i²) on R^K, K = 5.
 *
 *  This target is exact mean-field: the true posterior decomposes as a
 *  product of independent univariate Gaussians, so mean-field VI should
 *  recover μ̄ ≈ μ* and σ̄ ≈ σ* to within Monte Carlo noise. The PSIS-k̂
 *  diagnostic should be very small (near 0) because q ≡ p.
 *
 *  PASS criteria:
 *    - |μ̄_i - μ*_i| < 5 σ̄_i / sqrt(M_eff)   for every i  (well-recovered mean)
 *    - 0.5 ≤ σ̄_i / σ*_i ≤ 2.0                for every i  (within 2x of truth)
 *    - final_khat < 0.5                                     (PSIS PASS tier)
 *    - converged in fewer than max_epochs
 *
 *  These are loose — RAABBVI-lite single-sample reparam grad has
 *  meaningful stochasticity; the test is meant to catch correctness
 *  bugs (wrong gradient signs, wrong reparam, etc.), not calibration
 *  drift.
 *================================================================================*/

#define MCMC_USE_RCPP_ARMADILLO   // not really — we use standalone armadillo
#undef MCMC_USE_RCPP_ARMADILLO

#include "AI4BayesCode/mean_field_gaussian_vi_block.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include <armadillo>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

namespace AI4Bayes = AI4BayesCode;

int main() {
    using namespace AI4BayesCode;

    constexpr std::size_t K = 5;

    // True posterior parameters.
    arma::vec mu_star(K), sd_star(K);
    mu_star = arma::vec({-2.0, 0.0, 1.0, 3.0, -1.0});
    sd_star = arma::vec({ 0.5, 1.0, 2.0, 0.3, 0.7});
    const arma::vec var_star = arma::square(sd_star);

    // Log-density oracle on UNCONSTRAINED scale (= natural since real-valued).
    auto log_density_grad =
        [&](const arma::vec& eta, const block_context& /*ctx*/,
            arma::vec* grad_out) -> double {
        const double log_2pi = std::log(2.0 * M_PI);
        double lp = 0.0;
        for (std::size_t i = 0; i < K; ++i) {
            const double d = eta[i] - mu_star[i];
            lp += -0.5 * (log_2pi + 2.0 * std::log(sd_star[i])
                          + d * d / var_star[i]);
        }
        if (grad_out != nullptr) {
            grad_out->set_size(K);
            for (std::size_t i = 0; i < K; ++i) {
                (*grad_out)[i] = -(eta[i] - mu_star[i]) / var_star[i];
            }
        }
        return lp;
    };

    mean_field_gaussian_vi_block_config cfg;
    cfg.name             = "theta";
    cfg.initial_unc      = arma::zeros(K);
    cfg.initial_log_sd   = arma::vec(K, arma::fill::value(0.0));  // σ = 1 init
    cfg.log_density_grad = log_density_grad;
    cfg.dependencies     = {};
    cfg.optimizer.gamma_0                = 0.1;
    cfg.optimizer.rho                    = 0.5;
    cfg.optimizer.tau                    = 0.05;   // tighter for the test
    cfg.optimizer.inner_iter_per_epoch   = 500;
    cfg.optimizer.max_epochs             = 30;
    cfg.optimizer.S_khat                 = 1000;
    cfg.n_mc_per_step                    = 4;      // M=4 samples per grad (reduce noise)

    auto blk = std::make_unique<mean_field_gaussian_vi_block>(cfg);

    // Wrap in a composite so we exercise the composite_block path too.
    composite_block comp;
    comp.add_child(std::move(blk));

    std::mt19937_64 rng(12345);

    std::printf("\n=== test_vi_mean_field_gaussian ===\n");
    arma::cout << "True mu* = " << mu_star.t();
    arma::cout << "True sd* = " << sd_star.t();
    std::printf("Initial gamma = %.4f, rho = %.2f, tau = %.4f\n",
                cfg.optimizer.gamma_0, cfg.optimizer.rho, cfg.optimizer.tau);
    std::printf("inner_iter_per_epoch = %zu, max_epochs = %zu\n",
                cfg.optimizer.inner_iter_per_epoch, cfg.optimizer.max_epochs);

    // Loop step() until convergence (or max_outer iterations as a safety).
    const std::size_t max_outer = cfg.optimizer.max_epochs *
                                  cfg.optimizer.inner_iter_per_epoch + 100;
    std::size_t outer = 0;
    auto* leaf = dynamic_cast<const mean_field_gaussian_vi_block*>(&comp.child(0));
    while (outer < max_outer && !leaf->converged()) {
        comp.step(rng);
        outer += 1;
    }

    std::printf("\nConverged after %zu step() calls (%zu epochs reached).\n",
                outer, leaf->epoch());

    const arma::vec mu_fit  = leaf->current();
    const arma::vec lsd_fit = leaf->get_log_sd();
    const arma::vec sd_fit  = arma::exp(lsd_fit);

    arma::cout << "Fitted mu  = " << mu_fit.t();
    arma::cout << "Fitted sd  = " << sd_fit.t();
    std::printf("Final ELBO = %.4f\n", leaf->current_elbo());
    std::printf("Final khat = %.4f\n", leaf->vi_history().final_khat);

    // -------- Checks --------
    bool ok = true;

    // (1) Mean recovery: |μ̄ - μ*| < max(0.05, 0.1 * σ̄)
    //     ORIGINAL threshold was 5 * σ̄ — calibrated to actuals it was
    //     200-5000× over actual (e.g. err 0.005 vs threshold 2.5 for σ=0.5).
    //     That would only catch catastrophic algorithm failure, not
    //     calibration drift. Tightened to 10% of σ̄ floor 0.05 (≈ 0.1 σ̄
    //     budget — still allows RAABBVI stochasticity yet catches
    //     real-world correctness regressions).
    for (std::size_t i = 0; i < K; ++i) {
        const double err = std::abs(mu_fit[i] - mu_star[i]);
        const double thresh = std::max(0.05, 0.1 * sd_fit[i]);
        if (err > thresh) {
            std::printf("  FAIL mu[%zu]: |%.4f - %.4f| = %.4f > %.4f\n",
                        i, mu_fit[i], mu_star[i], err, thresh);
            ok = false;
        }
    }

    // (2) SD recovery within factor 1.3 (was factor 2 — actuals show
    //     ratio is typically 0.97-1.03; factor 1.3 still safely catches
    //     real drift while accommodating optimizer noise).
    for (std::size_t i = 0; i < K; ++i) {
        const double ratio = sd_fit[i] / sd_star[i];
        if (ratio < 0.77 || ratio > 1.3) {
            std::printf("  FAIL sd[%zu]: ratio %.4f not in [0.77, 1.3]\n",
                        i, ratio);
            ok = false;
        }
    }

    // (3) PSIS k̂ < 0.5 (PASS tier — q is exact mean-field for this target).
    if (!(leaf->vi_history().final_khat < 0.5)) {
        std::printf("  FAIL final_khat = %.4f, expected < 0.5\n",
                    leaf->vi_history().final_khat);
        ok = false;
    }

    // (4) Converged within budget.
    if (!leaf->converged()) {
        std::printf("  FAIL not converged within budget\n");
        ok = false;
    }

    // (5) current_sample is repeatable-different each call (rng advances).
    std::mt19937_64 rng_test(999);
    const arma::vec s1 = leaf->current_sample(rng_test);
    const arma::vec s2 = leaf->current_sample(rng_test);
    if (arma::approx_equal(s1, s2, "absdiff", 1e-12)) {
        std::printf("  FAIL current_sample(rng) gave identical results twice\n");
        ok = false;
    }

    if (ok) {
        std::printf("\nPASS\n");
        return 0;
    } else {
        std::printf("\nFAIL\n");
        return 1;
    }
}

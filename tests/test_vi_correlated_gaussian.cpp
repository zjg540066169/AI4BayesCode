/*================================================================================
 *  AI4BayesCode VI Phase 3 test #2  --  mean-field VI on a correlated
 *  bivariate Gaussian target (where mean-field is NOT exact).
 *
 *  Target: p(θ) = N(θ; 0, Σ),  Σ = [[1, ρ], [ρ, 1]],  ρ = 0.95.
 *
 *  The true posterior has correlated coordinates. Mean-field VI factorizes
 *  q(θ_1, θ_2) = q_1(θ_1) q_2(θ_2) — it CANNOT capture ρ. The expected
 *  outcome (Bishop §10.1.2):
 *
 *    - μ̄ ≈ 0 (mean still recovered)
 *    - σ̄_1, σ̄_2 ≈ sqrt(1 - ρ²) = sqrt(0.0975) ≈ 0.312
 *      (the conditional standard deviation, NOT the marginal sd=1.0
 *       — mean-field UNDERESTIMATES the marginal variance, this is
 *       the textbook caveat)
 *    - PSIS-k̂ in the CAUTION (0.5 ≤ k̂ < 0.7) or FAIL (≥ 0.7) range,
 *      depending on how thin the conditional tails are
 *
 *  PASS criteria (validates the VI algorithm correctness, not the
 *  posterior recovery):
 *    - |μ̄| < 0.2 for both coordinates (well-recovered mean)
 *    - σ̄_i in [0.2, 0.5] (close to conditional sd, NOT marginal)
 *    - PSIS-k̂ is a FINITE number (sanity: algorithm produced an output)
 *    - converged within max_epochs
 *
 *  This test exists to PROVE the mean-field caveat behaviorally and
 *  validate the PSIS-k̂ implementation produces meaningful values.
 *================================================================================*/

#include "AI4BayesCode/mean_field_gaussian_vi_block.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <random>

int main() {
    using namespace AI4BayesCode;

    constexpr std::size_t K = 2;
    const double rho = 0.95;

    // Σ = [[1, rho], [rho, 1]]; Σ^{-1} = (1/(1-rho²)) * [[1, -rho], [-rho, 1]]
    const double det_Sigma = 1.0 - rho * rho;     // 0.0975 for rho=0.95
    const double inv_det   = 1.0 / det_Sigma;
    const double cond_sd   = std::sqrt(det_Sigma);// 0.312...

    // Log-density: -0.5 * θ^T Σ^{-1} θ + const
    auto log_density_grad =
        [&](const arma::vec& eta, const block_context& /*ctx*/,
            arma::vec* grad_out) -> double {
        // Q = Σ^{-1}
        const double q00 = inv_det * 1.0;
        const double q11 = inv_det * 1.0;
        const double q01 = inv_det * (-rho);
        const double e0 = eta[0], e1 = eta[1];
        // log p = -0.5 (q00 e0² + 2 q01 e0 e1 + q11 e1²) + const
        const double quad = q00 * e0 * e0 + 2.0 * q01 * e0 * e1 + q11 * e1 * e1;
        const double lp = -0.5 * quad;
        if (grad_out != nullptr) {
            grad_out->set_size(K);
            // grad_log_p = -Q θ
            (*grad_out)[0] = -(q00 * e0 + q01 * e1);
            (*grad_out)[1] = -(q01 * e0 + q11 * e1);
        }
        return lp;
    };

    mean_field_gaussian_vi_block_config cfg;
    cfg.name             = "theta";
    cfg.initial_unc      = arma::zeros(K);
    cfg.initial_log_sd   = arma::vec(K, arma::fill::value(0.0));
    cfg.log_density_grad = log_density_grad;
    cfg.optimizer.gamma_0              = 0.05;
    cfg.optimizer.rho                  = 0.5;
    cfg.optimizer.tau                  = 0.05;
    cfg.optimizer.inner_iter_per_epoch = 500;
    cfg.optimizer.max_epochs           = 20;
    cfg.optimizer.S_khat               = 1000;
    cfg.n_mc_per_step                  = 4;

    auto blk = std::make_unique<mean_field_gaussian_vi_block>(cfg);

    composite_block comp;
    comp.add_child(std::move(blk));

    std::mt19937_64 rng(54321);

    std::printf("\n=== test_vi_correlated_gaussian ===\n");
    std::printf("Target: bivariate N(0, [[1, %.2f], [%.2f, 1]])\n", rho, rho);
    std::printf("Marginal sd = 1.0; conditional sd = sqrt(1 - rho^2) = %.4f\n",
                cond_sd);
    std::printf("Mean-field VI cannot capture rho; expects sd ~ conditional sd.\n");

    const auto* leaf =
        dynamic_cast<const mean_field_gaussian_vi_block*>(&comp.child(0));
    const std::size_t max_outer = cfg.optimizer.max_epochs *
                                  cfg.optimizer.inner_iter_per_epoch + 100;
    std::size_t outer = 0;
    while (outer < max_outer && !leaf->converged()) {
        comp.step(rng);
        outer += 1;
    }

    std::printf("\nConverged in %zu step() calls (%zu epochs).\n",
                outer, leaf->epoch());

    const arma::vec mu_fit  = leaf->current();
    const arma::vec lsd_fit = leaf->get_log_sd();
    const arma::vec sd_fit  = arma::exp(lsd_fit);

    arma::cout << "Fitted mu = " << mu_fit.t();
    arma::cout << "Fitted sd = " << sd_fit.t();
    std::printf("Final ELBO = %.4f, khat = %.4f\n",
                leaf->current_elbo(), leaf->vi_history().final_khat);

    bool ok = true;

    // (1) Mean recovered close to zero.
    for (std::size_t i = 0; i < K; ++i) {
        if (std::abs(mu_fit[i]) > 0.2) {
            std::printf("  FAIL |mu[%zu]| = %.4f > 0.2\n", i, std::abs(mu_fit[i]));
            ok = false;
        }
    }

    // (2) sd_fit ∈ [0.2, 0.5] (close to conditional sd 0.312, NOT marginal 1.0).
    //    This validates the mean-field underestimation caveat.
    for (std::size_t i = 0; i < K; ++i) {
        if (!(sd_fit[i] >= 0.2 && sd_fit[i] <= 0.5)) {
            std::printf("  FAIL sd[%zu] = %.4f not in [0.2, 0.5]\n", i, sd_fit[i]);
            ok = false;
        }
    }

    // (3) khat is finite (PSIS produced a meaningful number).
    const double kh = leaf->vi_history().final_khat;
    if (!std::isfinite(kh)) {
        std::printf("  FAIL final_khat = %.4f, expected finite\n", kh);
        ok = false;
    } else {
        const char* tier = (kh < 0.5) ? "PASS"
                          : (kh < 0.7) ? "CAUTION"
                          : "FAIL";
        std::printf("  khat = %.4f → tier = %s (informational)\n", kh, tier);
    }

    // (4) Converged within budget.
    if (!leaf->converged()) {
        std::printf("  FAIL not converged\n");
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

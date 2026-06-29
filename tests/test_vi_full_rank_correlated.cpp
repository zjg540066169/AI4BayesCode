/*================================================================================
 *  AI4BayesCode VI Phase 4 test — full-rank Gaussian VI on a correlated
 *  bivariate Gaussian target.
 *
 *  Target: p(θ) = N(θ; 0, Σ),  Σ = [[1, ρ], [ρ, 1]],  ρ = 0.95.
 *
 *  This is the SAME target as test_vi_correlated_gaussian.cpp (which uses
 *  mean-field VI and recovers sd ≈ 0.31 = conditional sd, NOT the
 *  marginal sd = 1.0). Full-rank VI must capture the full posterior
 *  covariance — recovering BOTH the marginal sd = 1.0 AND the correlation
 *  ρ ≈ 0.95.
 *
 *  This pair of tests demonstrates the "mean-field underestimates variance
 *  vs full-rank captures correlation" textbook example (Bishop §10.1.2).
 *
 *  PASS criteria:
 *    - 0.8 ≤ marginal_sd ≤ 1.3  for both coordinates (recovers near 1.0)
 *    - 0.85 ≤ |corr(θ_1, θ_2)| ≤ 0.99  (recovers near 0.95)
 *    - PSIS-k̂ < 0.7  (full-rank should achieve PASS or CAUTION on this
 *                       Gaussian target since q ≡ p when L L^T = Σ)
 *    - converged within max_epochs
 *================================================================================*/

#include "AI4BayesCode/full_rank_gaussian_vi_block.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <random>

int main() {
    using namespace AI4BayesCode;
    constexpr std::size_t K = 2;
    const double rho = 0.95;

    // Σ = [[1, ρ], [ρ, 1]]; Σ^{-1} = (1/(1-ρ²)) [[1, -ρ], [-ρ, 1]]
    const double det_Sigma = 1.0 - rho * rho;
    const double inv_det   = 1.0 / det_Sigma;

    auto log_density_grad =
        [&](const arma::vec& eta, const block_context& /*ctx*/,
            arma::vec* grad_out) -> double {
        const double q00 = inv_det;
        const double q11 = inv_det;
        const double q01 = inv_det * (-rho);
        const double e0 = eta[0], e1 = eta[1];
        const double quad = q00 * e0 * e0 + 2.0 * q01 * e0 * e1 + q11 * e1 * e1;
        const double lp = -0.5 * quad;
        if (grad_out) {
            grad_out->set_size(K);
            (*grad_out)[0] = -(q00 * e0 + q01 * e1);
            (*grad_out)[1] = -(q01 * e0 + q11 * e1);
        }
        return lp;
    };

    full_rank_gaussian_vi_block_config cfg;
    cfg.name             = "theta";
    cfg.initial_unc      = arma::zeros(K);
    cfg.initial_L        = 0.5 * arma::eye(K, K);  // initially uncorrelated
    cfg.log_density_grad = log_density_grad;
    cfg.optimizer.gamma_0              = 0.02;
    cfg.optimizer.rho                  = 0.5;
    cfg.optimizer.tau                  = 0.01;
    cfg.optimizer.inner_iter_per_epoch = 500;
    cfg.optimizer.max_epochs           = 15;
    cfg.optimizer.S_khat               = 1000;
    cfg.n_mc_per_step                  = 4;

    auto blk = std::make_unique<full_rank_gaussian_vi_block>(cfg);
    composite_block comp;
    comp.add_child(std::move(blk));

    std::mt19937_64 rng(54321);
    const auto* leaf =
        dynamic_cast<const full_rank_gaussian_vi_block*>(&comp.child(0));

    std::printf("\n=== test_vi_full_rank_correlated ===\n");
    std::printf("Target: N(0, [[1, %.2f], [%.2f, 1]]) — marginal sd=1, ρ=%.2f\n",
                rho, rho, rho);
    std::printf("Mean-field VI would give sd ≈ sqrt(1-ρ²) = %.4f  (conditional, not marginal)\n",
                std::sqrt(1.0 - rho*rho));

    const std::size_t cap = cfg.optimizer.max_epochs *
                            cfg.optimizer.inner_iter_per_epoch + 100;
    std::size_t outer = 0;
    while (outer < cap && !leaf->converged()) {
        comp.step(rng);
        outer++;
    }

    const arma::vec mu_fit = leaf->current();
    const arma::mat L_fit  = leaf->L();
    const arma::mat Sigma_fit = L_fit * L_fit.t();
    const double sd0 = std::sqrt(Sigma_fit(0, 0));
    const double sd1 = std::sqrt(Sigma_fit(1, 1));
    const double corr = Sigma_fit(0, 1) / (sd0 * sd1);
    const double khat = leaf->vi_history().final_khat;

    std::printf("\nConverged at iter %zu, %zu epochs.\n", outer, leaf->epoch());
    std::printf("Fitted μ      = (%.4f, %.4f)   (truth (0, 0))\n", mu_fit[0], mu_fit[1]);
    std::printf("Fitted Σ_11   = %.4f, Σ_22 = %.4f, Σ_12 = %.4f\n",
                Sigma_fit(0,0), Sigma_fit(1,1), Sigma_fit(0,1));
    std::printf("Fitted sd     = (%.4f, %.4f)   (target 1.0)\n", sd0, sd1);
    std::printf("Fitted corr   = %.4f             (target %.2f)\n", corr, rho);
    std::printf("Final ELBO    = %.4f, khat = %.4f\n",
                leaf->current_elbo(), khat);

    bool ok = true;
    if (!leaf->converged()) { std::printf("  FAIL: not converged\n"); ok = false; }
    if (!(sd0 >= 0.8 && sd0 <= 1.3)) {
        std::printf("  FAIL: sd0 = %.4f not in [0.8, 1.3]\n", sd0); ok = false;
    }
    if (!(sd1 >= 0.8 && sd1 <= 1.3)) {
        std::printf("  FAIL: sd1 = %.4f not in [0.8, 1.3]\n", sd1); ok = false;
    }
    if (!(std::abs(corr) >= 0.85 && std::abs(corr) <= 0.99)) {
        std::printf("  FAIL: |corr| = %.4f not in [0.85, 0.99]\n", std::abs(corr));
        ok = false;
    }
    if (!std::isfinite(khat) && khat != -1.0) {
        std::printf("  FAIL: khat = %.4f not finite\n", khat); ok = false;
    }

    if (ok) {
        std::printf("\nPASS — full-rank VI recovers marginal sd ≈ 1.0 AND corr ≈ ρ\n");
        std::printf("       (cf. mean-field which only recovers conditional sd ≈ %.4f)\n",
                   std::sqrt(1.0 - rho*rho));
        return 0;
    } else {
        std::printf("\nFAIL\n");
        return 1;
    }
}

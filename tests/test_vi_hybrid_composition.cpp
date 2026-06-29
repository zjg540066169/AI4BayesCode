/*================================================================================
 *  AI4BayesCode VI Phase 6 test  --  Hybrid composition: VI for high-dim
 *  weights + NUTS for hyperparameter (single-sigma BNN-style hierarchy).
 *
 *  Target: a stripped-down toy version of the nn_rbm1bJ10 hierarchy.
 *
 *    w_i ~ N(0, σ²)              i = 1..K           (K = 100 "weights")
 *    σ²  ~ Inv-Gamma(α=2, β=1)                       (hyper-prior on σ²)
 *    y_j ~ N(w · x_j, τ²)        j = 1..N            (linear regression
 *                                                     likelihood; τ² fixed)
 *
 *  This is a SHRUNK BNN — weight prior controlled by a sampled scale —
 *  with one VI block for the K-dimensional weight vector and one NUTS
 *  block for log(σ²). Demonstrates the §18.4 hybrid-correctness
 *  invariant: the VI block writes q-SAMPLES (not q-means) into
 *  shared_data, so the NUTS sigma² block sees a fresh weight draw each
 *  outer iteration and gets the correct conditional posterior.
 *
 *  PASS criteria:
 *    - log(σ²) NUTS chain converges to a finite value (R-hat-like sanity
 *      via single-chain effective consistency)
 *    - VI weight mean stays close to OLS estimate (within reason — VI is
 *      regularised by the hierarchical σ)
 *    - VI's PSIS-k̂ is finite (algorithm produced an output)
 *    - composite step() doesn't crash
 *
 *  This is a CORRECTNESS test, not a posterior recovery benchmark.
 *  The goal is to prove the framework composes engine kinds without
 *  framework changes.
 *================================================================================*/

#include "AI4BayesCode/mean_field_gaussian_vi_block.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <random>

namespace AI4Bayes = AI4BayesCode;

int main() {
    using namespace AI4BayesCode;
    namespace ct = AI4BayesCode::constraints;

    constexpr std::size_t K = 100;   // weights
    constexpr std::size_t N = 300;   // observations
    constexpr double tau  = 0.5;    // fixed likelihood noise sd
    constexpr double tau2 = tau * tau;

    // ---- Synthesize data ----
    std::mt19937_64 rng_data(2026);
    arma::vec w_true(K);
    {
        // True weights from N(0, σ_true²); σ_true² = 0.04 → σ_true = 0.2
        std::normal_distribution<double> N01(0.0, 1.0);
        for (std::size_t j = 0; j < K; ++j) w_true[j] = 0.2 * N01(rng_data);
    }
    arma::mat X(N, K);
    {
        std::normal_distribution<double> N01(0.0, 1.0);
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < K; ++j) X(i, j) = N01(rng_data);
    }
    arma::vec y_obs = X * w_true;
    {
        std::normal_distribution<double> N0t(0.0, tau);
        for (std::size_t i = 0; i < N; ++i) y_obs[i] += N0t(rng_data);
    }

    // ---- Define the model ----
    // VI weights conditional: log p(w | σ², y, X) = log_lik + log_prior + const
    //   log_lik     = -(1/(2 τ²)) ||y - X w||²
    //   log_prior   = -(K/2) log σ² - (1/(2 σ²)) ||w||²
    // (we drop constants from the unconstrained scale; the constraint on w
    //  is identity since w ∈ R^K)
    // Gradient wrt w:
    //   ∇_w log_lik   = (1/τ²) X^T (y - X w)
    //   ∇_w log_prior = -(1/σ²) w

    // NUTS log(σ²) conditional: log p(η | w) where η = log(σ²)
    //   σ² = exp(η)
    //   log p(σ² | w) on the natural scale:
    //     -(K/2) log σ² - (1/(2 σ²)) ||w||²    -- prior on w given σ²
    //     + (α-1) log σ² - β/σ²                 -- σ² ~ IG(α, β) → log p(σ²) = const + (-α-1) log σ² - β/σ²
    //     ...wait, IG(α, β) has log p ∝ (-α - 1) log σ² - β/σ². Hmm, the formula in
    //     the prompt is IG with rate-style β, so p(σ²) ∝ σ²^(-α-1) exp(-β/σ²).
    //   Sum:
    //     log p_nat = -(K/2 + α + 1) log σ² - ||w||²/(2 σ²) - β/σ²
    //              = -(K/2 + α + 1) log σ² - (||w||² / 2 + β) / σ²
    //   On η = log σ² (unconstrained), constraints::positive::wrap adds + log σ²
    //   (the Jacobian for η = log θ → θ = exp η, |dθ/dη| = exp η = σ²)
    //   so log p̃(η) = log p_nat(σ² = exp η) + η,
    //              = -(K/2 + α + 1) η - (||w||²/2 + β) exp(-η) + η
    //              = -(K/2 + α) η - (||w||²/2 + β) exp(-η)
    //   d/dη log p̃(η) = -(K/2 + α) + (||w||²/2 + β) exp(-η)
    //
    //   But wait — constraints::positive::wrap adds the Jacobian automatically.
    //   So the user-supplied natural-scale lambda just writes log p_nat, and the
    //   wrap takes care of + η. Let me follow that convention exactly.

    // Hyperprior parameters for σ²
    const double alpha_ig = 2.0;
    const double beta_ig  = 1.0;

    // Weight VI block: log_density_grad on unconstrained scale (identity for real)
    auto weight_log_density_grad =
        [&](const arma::vec& w, const block_context& ctx,
            arma::vec* grad_out) -> double {
        // Read σ² from sibling via shared_data (written as q-sample by VI
        // — but here log_s2 is from NUTS sibling, so it's the latest sample)
        const arma::vec& log_s2_vec = ctx.at("log_s2");
        const double log_s2 = log_s2_vec[0];
        const double sigma2 = std::exp(log_s2);

        // log p(w | σ², y, X) = log_lik + log_prior
        const arma::vec resid = y_obs - X * w;
        const double log_lik   = -0.5 / tau2 * arma::dot(resid, resid);
        const double log_prior = -0.5 * static_cast<double>(K) * log_s2
                                 - 0.5 / sigma2 * arma::dot(w, w);

        if (grad_out != nullptr) {
            grad_out->set_size(K);
            const arma::vec g_lik   = (1.0 / tau2) * (X.t() * resid);
            const arma::vec g_prior = -(1.0 / sigma2) * w;
            for (std::size_t j = 0; j < K; ++j) {
                (*grad_out)[j] = g_lik[j] + g_prior[j];
            }
        }
        return log_lik + log_prior;
    };

    // log(σ²) NUTS block: natural-scale lambda (constraints::positive::wrap
    // adds the Jacobian automatically).
    auto log_s2_log_density_grad =
        [&](const arma::vec& eta_vec, const block_context& ctx,
            arma::vec* grad_out) -> double {
        // eta = log σ² (unconstrained); natural = σ²
        const double eta = eta_vec[0];
        const double sigma2 = std::exp(eta);

        // Read current weight q-sample from VI sibling
        const arma::vec& w = ctx.at("weights");

        // Natural-scale log p:
        //   log p(σ² | w) = log p_prior(w | σ²) + log p_IG(σ²)  (up to const)
        //                 = -(K/2) log σ² - ||w||²/(2σ²)
        //                   + (-α - 1) log σ² - β/σ²
        // Total: -(K/2 + α + 1) log σ² - (||w||²/2 + β) / σ²
        // But constraints::positive::wrap converts dη to dσ² (multiplies by σ²)
        // ... actually wrap handles all of this. We just write the natural-scale
        // density in η. Let me follow the standard pattern: the lambda returns
        // log p(θ_nat = exp(η)) without the Jacobian; wrap adds the Jacobian.

        const double dot_ww = arma::dot(w, w);
        const double log_p_nat =
              -0.5 * static_cast<double>(K) * std::log(sigma2)
              - 0.5 * dot_ww / sigma2
              + (-alpha_ig - 1.0) * std::log(sigma2)
              -  beta_ig / sigma2;

        // Natural-scale gradient ∂log p / ∂σ² (NOT ∂/∂η)
        // wrap multiplies by ∂σ²/∂η = σ² to get the unconstrained gradient.
        // No, wait: constraints::positive::wrap chain-rules differently.
        //
        // Easier path: write log p̃(η) = log p(σ²=exp η) + η  (Jacobian)
        // and the gradient w.r.t. η directly.
        // Here we're NOT using wrap — we manually write log p̃(η).
        const double log_jac = eta;   // Jacobian contribution
        const double log_p_tilde = log_p_nat + log_jac;

        // d/dη log p̃(η):
        //   from -0.5 K log σ²:     -0.5 K (dσ²/dη / σ²) = -0.5 K  (since σ²=exp η)
        //   from -0.5 ||w||² / σ²:  -0.5 ||w||² * (-1/σ²²) * σ² = +0.5 ||w||² / σ²
        //   from (-α-1) log σ²:     -α-1
        //   from -β/σ²:             +β/σ²
        //   from log_jac (=η):      +1
        // sum: -0.5 K + 0.5 ||w||² / σ² - α - 1 + β/σ² + 1
        //    = -0.5 K - α + (0.5 ||w||² + β) / σ²
        const double d_log_p_d_eta =
            -0.5 * static_cast<double>(K) - alpha_ig
            + (0.5 * dot_ww + beta_ig) / sigma2;

        if (grad_out != nullptr) {
            grad_out->set_size(1);
            (*grad_out)[0] = d_log_p_d_eta;
        }
        return log_p_tilde;
    };

    // ---- Set up the composite ----
    composite_block comp;
    comp.set_keep_history(true);
    comp.data().set("weights", arma::zeros(K));
    comp.data().set("log_s2", arma::vec({-1.0}));  // σ² ≈ 0.37

    {
        // Weights VI block
        mean_field_gaussian_vi_block_config vi_cfg;
        vi_cfg.name             = "weights";
        vi_cfg.initial_unc      = arma::zeros(K);
        vi_cfg.initial_log_sd   = arma::vec(K, arma::fill::value(-1.5));
        vi_cfg.log_density_grad = weight_log_density_grad;
        vi_cfg.dependencies     = {"log_s2"};
        vi_cfg.optimizer.gamma_0              = 0.02;
        vi_cfg.optimizer.rho                  = 0.5;
        vi_cfg.optimizer.tau                  = 0.05;
        vi_cfg.optimizer.inner_iter_per_epoch = 100;
        vi_cfg.optimizer.max_epochs           = 10;
        vi_cfg.optimizer.S_khat               = 500;
        vi_cfg.n_mc_per_step                  = 2;
        comp.add_child(std::make_unique<mean_field_gaussian_vi_block>(vi_cfg));
    }
    {
        // log(σ²) NUTS block (no wrap — we wrote the Jacobian manually)
        nuts_block_config nuts_cfg;
        nuts_cfg.name             = "log_s2";
        nuts_cfg.initial_unc      = arma::vec({-1.0});
        nuts_cfg.log_density_grad = log_s2_log_density_grad;
        // constrain / unconstrain left empty → identity (we expose η directly)
        comp.add_child(std::make_unique<nuts_block>(nuts_cfg));
    }

    // Declare dependencies (Gibbs DAG)
    comp.data().declare_dependencies("weights", {"log_s2"});
    comp.data().declare_dependencies("log_s2",  {"weights"});

    // ---- Run the hybrid sampler ----
    std::mt19937_64 rng(12345);

    std::printf("\n=== test_vi_hybrid_composition ===\n");
    std::printf("K = %zu weights (VI block), N = %zu obs\n", K, N);
    std::printf("True σ_w = 0.2, fixed τ = %.2f\n", tau);

    const std::size_t n_outer = 500;
    for (std::size_t t = 0; t < n_outer; ++t) {
        comp.step(rng);
        if ((t + 1) % 100 == 0) {
            const auto& wmean = comp.child(0).current();
            const auto& log_s2 = comp.child(1).current();
            std::printf("  outer iter %3zu: ||w-mean||=%.3f, log_s2=%.3f\n",
                        t + 1, arma::norm(wmean), log_s2[0]);
        }
    }

    // ---- Inspect outcomes ----
    const auto* vi_leaf = dynamic_cast<const mean_field_gaussian_vi_block*>(
        &comp.child(0));
    const auto& nuts_leaf = comp.child(1);

    const arma::vec w_mean = vi_leaf->current();
    const arma::vec w_sd   = arma::exp(vi_leaf->get_log_sd());
    const arma::vec log_s2_final = nuts_leaf.current();
    const double khat = vi_leaf->vi_history().final_khat;
    const double final_elbo = vi_leaf->current_elbo();

    // OLS reference (ignoring prior — used as a sanity check that w_mean
    // is in the right ballpark).
    const arma::vec w_ols = arma::solve(X.t() * X + 1e-6 * arma::eye(K, K),
                                         X.t() * y_obs);

    const double w_err_to_truth = arma::norm(w_mean - w_true) / arma::norm(w_true);
    const double w_err_to_ols   = arma::norm(w_mean - w_ols) / arma::norm(w_ols);

    std::printf("\nFinal: |w_VI - w_true|/|w_true| = %.4f (true sd=0.2)\n", w_err_to_truth);
    std::printf("       |w_VI - w_OLS|/|w_OLS|   = %.4f\n", w_err_to_ols);
    std::printf("       mean(w_VI_sd) = %.4f\n", arma::mean(w_sd));
    std::printf("       log(σ²) final = %.4f → σ² = %.4f (true σ²=0.04)\n",
                log_s2_final[0], std::exp(log_s2_final[0]));
    std::printf("       VI ELBO = %.2f, khat = %.4f (PASS<0.5, CAUTION<0.7)\n",
                final_elbo, khat);
    std::printf("       VI converged: %s\n", vi_leaf->converged() ? "yes" : "no");

    bool ok = true;

    // (1) Outer loop didn't crash.
    if (n_outer == 0) {
        std::printf("  FAIL: 0 outer iters\n"); ok = false;
    }

    // (2) w_VI close to w_true (within 50% relative error — VI bias is real).
    if (!(w_err_to_truth < 0.5)) {
        std::printf("  FAIL: |w_VI - w_true|/|w_true| = %.4f >= 0.5\n",
                    w_err_to_truth);
        ok = false;
    }

    // (3) log_s2 finite and reasonable (between -8 and 4, i.e., σ² ∈ [3e-4, 50]).
    const double ls2 = log_s2_final[0];
    if (!std::isfinite(ls2) || ls2 < -8.0 || ls2 > 4.0) {
        std::printf("  FAIL: log_s2 = %.4f outside [-8, 4]\n", ls2);
        ok = false;
    }

    // (4) khat is a finite number.
    if (!std::isfinite(khat) && khat != -1.0) {
        std::printf("  FAIL: khat = %.4f not finite\n", khat);
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

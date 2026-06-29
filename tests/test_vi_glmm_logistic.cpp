/*================================================================================
 *  AI4BayesCode VI showcase #1 — Hierarchical Bayesian logistic GLMM
 *  with random intercepts.
 *
 *  Model (non-centered parameterization — codegen_priors.md §2f.(2)
 *         strongly recommends non-centering for VI hierarchical models;
 *         centered mean-field VI on a hierarchical scale routinely
 *         underestimates σ_α via the q(α)-q(σ_α) funnel coupling):
 *    y_{ij} ∈ {0, 1}                            i = 1..n_j, j = 1..J
 *    y_{ij} ~ Bernoulli(σ(α_0 + σ_α z_{j(i)} + X_{ij}^T β))
 *    z_j ~ N(0, 1)                               j = 1..J  (non-centered standard)
 *    β_p ~ N(0, 10²)                             p = 1..P  (fixed effects)
 *    α_0 ~ N(0, 10²)                             grand mean
 *    σ_α ~ half-Normal(0, 2.5)                   between-group sd
 *  Derived quantity:
 *    α_j = α_0 + σ_α z_j                         random intercept
 *
 *  Hybrid sampler:
 *    vi_block          over z = (z_1, ..., z_J)            D_VI = J
 *    nuts_block (real) over [β; α_0; log σ_α]              D_NUTS = P + 2
 *
 *  This is the canonical applied-stats GLMM (medical multi-site trials,
 *  educational multi-school analyses, marketing multi-region A/B tests).
 *  When J is large, VI on α_j is much faster than full MCMC. Hybrid mode
 *  keeps the hyperparameters (β, α_0, σ_α) on MCMC for honest CIs.
 *
 *  PASS criteria (framework correctness, not MAP-fitting):
 *    - Composite runs n_outer steps without crash, NaN, or segfault
 *    - VI converges within its budget
 *    - σ_α recovered within factor 3 of truth (loose — small N)
 *    - α_0 recovered within ±1.0 of truth
 *    - β recovered to within ±0.5 of truth on each component
 *================================================================================*/

#include "AI4BayesCode/mean_field_gaussian_vi_block.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <random>

int main() {
    using namespace AI4BayesCode;

    // ---- Dimensions ----
    constexpr std::size_t J     = 30;  // groups
    constexpr std::size_t n_per = 10;  // observations per group
    constexpr std::size_t N     = J * n_per;
    constexpr std::size_t P     = 3;   // covariates
    constexpr std::size_t D_hyp = P + 2;     // β + α_0 + log σ_α
    const std::size_t off_beta  = 0;
    const std::size_t off_a0    = P;
    const std::size_t off_lsa   = P + 1;

    // ---- Truth ----
    const double sigma_alpha_true = 0.7;
    const double alpha_0_true     = -0.2;
    const arma::vec beta_true     = arma::vec({1.0, -0.5, 0.3});

    // ---- Simulate data ----
    std::mt19937_64 rng_data(42);
    std::normal_distribution<double> N01(0.0, 1.0);
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    arma::ivec group_id(N);
    arma::mat  X(N, P);
    arma::vec  alpha_true(J);

    for (std::size_t j = 0; j < J; ++j) {
        alpha_true[j] = alpha_0_true + sigma_alpha_true * N01(rng_data);
    }
    for (std::size_t i = 0; i < N; ++i) {
        group_id[i] = static_cast<int>(i / n_per);
        for (std::size_t p = 0; p < P; ++p) X(i, p) = N01(rng_data);
    }
    arma::ivec y_obs(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double eta_i = alpha_true[group_id[i]] +
                             arma::dot(X.row(i).t(), beta_true);
        const double p_i = 1.0 / (1.0 + std::exp(-eta_i));
        y_obs[i] = (U01(rng_data) < p_i) ? 1 : 0;
    }

    // ---- VI block: log p(z | y, X, β, α_0, σ_α)  (non-centered) ----
    // z_j ~ N(0, 1) prior. eta_i = α_0 + σ_α * z_{group(i)} + X_i^T β.
    // Reads "hyp" from ctx (NUTS sibling's draw).
    auto vi_log_density_grad =
        [&](const arma::vec& z, const block_context& ctx,
            arma::vec* grad_out) -> double {
        const arma::vec& hyp = ctx.at("hyp");
        const arma::vec beta(const_cast<double*>(hyp.memptr() + off_beta),
                              P, false, true);
        const double alpha_0 = hyp[off_a0];
        const double log_sa  = hyp[off_lsa];
        const double sigma_a = std::exp(log_sa);

        // log_lik with non-centered α_j = α_0 + σ_α z_j
        double log_lik = 0.0;
        arma::vec p_vec(N);
        for (std::size_t i = 0; i < N; ++i) {
            const double e = alpha_0 + sigma_a * z[group_id[i]]
                             + arma::dot(X.row(i).t(), beta);
            const double log_1pe = (e > 0.0)
                ? e + std::log1p(std::exp(-e))
                : std::log1p(std::exp(e));
            log_lik += static_cast<double>(y_obs[i]) * e - log_1pe;
            p_vec[i] = 1.0 / (1.0 + std::exp(-e));
        }

        // log_prior_z = -0.5 sum z²  (standard normal)
        const double log_p_z = -0.5 * arma::dot(z, z);
        const double log_p = log_lik + log_p_z;

        if (grad_out != nullptr) {
            grad_out->set_size(J);
            grad_out->zeros();
            // d(log_lik) / d(z_j) = sum_{i in group j} (y_i - p_i) * σ_α
            for (std::size_t i = 0; i < N; ++i) {
                (*grad_out)[group_id[i]] +=
                    (static_cast<double>(y_obs[i]) - p_vec[i]) * sigma_a;
            }
            // d(log_p_z) / d(z_j) = -z_j
            for (std::size_t j = 0; j < J; ++j) (*grad_out)[j] += -z[j];
        }
        return log_p;
    };

    // ---- NUTS block: log p(β, α_0, log σ_α | y, X, α) ----
    // Reads "alpha" from ctx (VI sibling's q-sample).
    const double prior_sd_beta = 10.0;
    const double prior_sd_a0   = 10.0;
    const double half_n_sd_sa  = 2.5;     // σ_α ~ half-Normal(0, 2.5)

    // ---- NUTS block: log p(β, α_0, log σ_α | z, y, X)  (non-centered) ----
    // z fixed via context (VI sibling). No α prior term here — z prior
    // is in the VI block.
    auto nuts_log_density_grad =
        [&](const arma::vec& hyp, const block_context& ctx,
            arma::vec* grad_out) -> double {
        const arma::vec beta(const_cast<double*>(hyp.memptr() + off_beta),
                              P, false, true);
        const double alpha_0 = hyp[off_a0];
        const double log_sa  = hyp[off_lsa];
        const double sigma_a = std::exp(log_sa);
        const double s2a     = sigma_a * sigma_a;

        const arma::vec& z = ctx.at("alpha");   // VI block writes z under key "alpha"

        // log_lik with non-centered α_j = α_0 + σ_α z_j
        double log_lik = 0.0;
        arma::vec p_vec(N);
        for (std::size_t i = 0; i < N; ++i) {
            const double e = alpha_0 + sigma_a * z[group_id[i]]
                             + arma::dot(X.row(i).t(), beta);
            const double log_1pe = (e > 0.0)
                ? e + std::log1p(std::exp(-e))
                : std::log1p(std::exp(e));
            log_lik += static_cast<double>(y_obs[i]) * e - log_1pe;
            p_vec[i] = 1.0 / (1.0 + std::exp(-e));
        }

        const double sum_beta_sq = arma::dot(beta, beta);
        const double log_p_beta = -0.5 / (prior_sd_beta * prior_sd_beta) * sum_beta_sq;
        const double log_p_a0   = -0.5 / (prior_sd_a0 * prior_sd_a0) * alpha_0 * alpha_0;
        // σ_α ~ half-Normal(0, A) on natural; +log σ_α Jacobian for log scale
        const double log_p_sigma_a =
            -0.5 * s2a / (half_n_sd_sa * half_n_sd_sa) + log_sa;

        const double log_p_tilde =
            log_lik + log_p_beta + log_p_a0 + log_p_sigma_a;

        if (grad_out != nullptr) {
            grad_out->set_size(D_hyp);
            grad_out->zeros();
            // d(log_lik) / d(β_p) = sum_i (y_i - p_i) X_{i,p}
            for (std::size_t i = 0; i < N; ++i) {
                const double err = static_cast<double>(y_obs[i]) - p_vec[i];
                for (std::size_t pp = 0; pp < P; ++pp)
                    (*grad_out)[off_beta + pp] += err * X(i, pp);
            }
            for (std::size_t pp = 0; pp < P; ++pp)
                (*grad_out)[off_beta + pp] +=
                    -beta[pp] / (prior_sd_beta * prior_sd_beta);

            // d(log_lik) / d(α_0) = sum_i (y_i - p_i)
            double grad_a0_lik = 0.0;
            for (std::size_t i = 0; i < N; ++i)
                grad_a0_lik += (static_cast<double>(y_obs[i]) - p_vec[i]);
            (*grad_out)[off_a0] = grad_a0_lik
                                  - alpha_0 / (prior_sd_a0 * prior_sd_a0);

            // d(log_lik) / d(log σ_α):
            //   d(eta_i)/d(log σ_α) = σ_α * z_{group(i)}
            //   d(log_lik)/d(log σ_α) = sum_i (y_i - p_i) σ_α z_{group(i)}
            double grad_lsa_lik = 0.0;
            for (std::size_t i = 0; i < N; ++i)
                grad_lsa_lik += (static_cast<double>(y_obs[i]) - p_vec[i])
                                 * sigma_a * z[group_id[i]];
            (*grad_out)[off_lsa] =
                grad_lsa_lik
                - s2a / (half_n_sd_sa * half_n_sd_sa)
                + 1.0;   // Jacobian
        }
        return log_p_tilde;
    };

    // ---- Compose ----
    composite_block comp;
    comp.data().set("alpha", arma::zeros(J));
    arma::vec hyp_init(D_hyp, arma::fill::zeros);
    hyp_init[off_lsa] = std::log(0.5);   // σ_α ≈ 0.5 to start
    comp.data().set("hyp", hyp_init);

    {
        mean_field_gaussian_vi_block_config vi_cfg;
        vi_cfg.name             = "alpha";
        vi_cfg.initial_unc      = arma::zeros(J);
        vi_cfg.initial_log_sd   = arma::vec(J, arma::fill::value(-2.0));
        vi_cfg.log_density_grad = vi_log_density_grad;
        vi_cfg.dependencies     = {"hyp"};
        vi_cfg.optimizer.gamma_0              = 0.02;
        vi_cfg.optimizer.rho                  = 0.5;
        vi_cfg.optimizer.tau                  = 0.01;
        vi_cfg.optimizer.inner_iter_per_epoch = 300;
        vi_cfg.optimizer.max_epochs           = 10;
        vi_cfg.optimizer.S_khat               = 500;
        vi_cfg.n_mc_per_step                  = 2;
        comp.add_child(std::make_unique<mean_field_gaussian_vi_block>(vi_cfg));
    }
    {
        nuts_block_config nuts_cfg;
        nuts_cfg.name             = "hyp";
        nuts_cfg.initial_unc      = hyp_init;
        nuts_cfg.log_density_grad = nuts_log_density_grad;
        comp.add_child(std::make_unique<nuts_block>(nuts_cfg));
    }
    comp.data().declare_dependencies("alpha", {"hyp"});
    comp.data().declare_dependencies("hyp",   {"alpha"});

    // ---- Run ----
    std::mt19937_64 rng(12345);
    std::printf("\n=== test_vi_glmm_logistic ===\n");
    std::printf("J=%zu groups, n_per=%zu, P=%zu covariates, N=%zu obs\n",
                J, n_per, P, N);
    std::printf("Truth: σ_α=%.3f, α_0=%.3f, β=(%.3f, %.3f, %.3f)\n",
                sigma_alpha_true, alpha_0_true,
                beta_true[0], beta_true[1], beta_true[2]);

    const std::size_t n_outer = 3000;
    for (std::size_t t = 0; t < n_outer; ++t) {
        comp.step(rng);
        if ((t + 1) % 500 == 0) {
            const auto& a = comp.child(0).current();
            const auto& h = comp.child(1).current();
            std::printf("  iter %4zu: ||α||=%.2f, β=(%.2f,%.2f,%.2f), α_0=%.2f, σ_α=%.2f\n",
                        t + 1, arma::norm(a),
                        h[off_beta], h[off_beta+1], h[off_beta+2],
                        h[off_a0], std::exp(h[off_lsa]));
        }
    }

    const auto* vi_leaf = dynamic_cast<const mean_field_gaussian_vi_block*>(
        &comp.child(0));
    const arma::vec a_fit  = vi_leaf->current();
    const arma::vec h_fit  = comp.child(1).current();
    const arma::vec b_fit  = h_fit.subvec(off_beta, off_beta + P - 1);
    const double a0_fit    = h_fit[off_a0];
    const double sa_fit    = std::exp(h_fit[off_lsa]);
    const double khat      = vi_leaf->vi_history().final_khat;

    std::printf("\nFinal:\n");
    std::printf("  σ_α fit = %.3f (true %.3f)\n", sa_fit, sigma_alpha_true);
    std::printf("  α_0 fit = %.3f (true %.3f)\n", a0_fit, alpha_0_true);
    std::printf("  β   fit = (%.3f, %.3f, %.3f) (true (%.3f, %.3f, %.3f))\n",
                b_fit[0], b_fit[1], b_fit[2],
                beta_true[0], beta_true[1], beta_true[2]);
    std::printf("  ||α_fit - α_true|| / ||α_true|| = %.3f\n",
                arma::norm(a_fit - alpha_true) / arma::norm(alpha_true));
    std::printf("  ELBO=%.2f, khat=%.4f, VI converged=%s\n",
                vi_leaf->current_elbo(), khat,
                vi_leaf->converged() ? "yes" : "no");

    // PASS gates
    bool ok = true;
    if (!vi_leaf->converged()) {
        std::printf("  FAIL: VI did not converge\n"); ok = false;
    }
    // Tightened from "factor 3" / |err|<1.0 / |err|<0.5 to factor 2 /
    // |err|<0.5 / |err|<0.4. MF VI on hierarchical GLMM has known bias
    // (Bishop §10.1.2 underestimate of variance) but with N=300 + J=30
    // the recovered values should be within these tighter bounds:
    //   σ_α fit observed ≈ 0.47 (factor 1.49 of true 0.70)
    //   α_0 fit observed ≈ -0.02 (|err| ≈ 0.18 of true -0.20)
    //   β   fit max |err| ≈ 0.34 (β[1]: VI bias dominant)
    // Tightened thresholds catch real correctness regressions without
    // false alarms.
    if (!std::isfinite(sa_fit) || sa_fit < sigma_alpha_true / 2.0 || sa_fit > sigma_alpha_true * 2.0) {
        std::printf("  FAIL: σ_α %.3f outside 2x factor of truth %.3f\n",
                    sa_fit, sigma_alpha_true); ok = false;
    }
    if (std::abs(a0_fit - alpha_0_true) > 0.5) {
        std::printf("  FAIL: α_0 fit %.3f differs by >0.5 from truth %.3f\n",
                    a0_fit, alpha_0_true); ok = false;
    }
    for (std::size_t p = 0; p < P; ++p) {
        if (std::abs(b_fit[p] - beta_true[p]) > 0.4) {
            std::printf("  FAIL: β[%zu]=%.3f differs by >0.4 from truth %.3f\n",
                        p, b_fit[p], beta_true[p]);
            ok = false;
        }
    }
    if (ok) {
        std::printf("\nPASS\n");
        return 0;
    } else {
        std::printf("\nFAIL\n");
        return 1;
    }
}

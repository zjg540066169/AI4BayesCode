/*================================================================================
 *  AI4BayesCode VI showcase #2 — Bayesian Neural Network for REGRESSION.
 *
 *  A 1-hidden-layer BNN for continuous targets — distinct from the
 *  multi-class softmax-classification BNN family (nn_rbm1bJ10 et al.).
 *  Uses non-centered weights + half-Normal hyperpriors instead of the
 *  Lampinen-Vehtari 2001 IG(0.25, (0.05/M^2)^2 / 4) parameterization.
 *
 *  Model:
 *    y_i ~ N(f(x_i), σ_y²)                      i = 1..N
 *    f(x) = β_0 + h(x)^T β
 *    h(x) = tanh(α_0 + A^T x)                    ∈ R^J
 *    α_0_j ~ N(0, 1)                             j = 1..J  (hidden intercepts)
 *    β_0   ~ N(0, 1)                             output intercept
 *    A_{mj} = σ_α * Ã_{mj},  Ã_{mj} ~ N(0, 1)    (non-centered input-to-hidden)
 *    β_j   = σ_β * β̃_j,     β̃_j  ~ N(0, 1)      (non-centered hidden-to-output)
 *    σ_α, σ_β, σ_y ~ half-Normal(0, 2.5)         weakly-informative scales
 *
 *  Hybrid sampler:
 *    vi_block         over [α_0; Ã; β_0; β̃]            D_VI = J + M*J + 1 + J
 *    nuts_block (real) over [log σ_α; log σ_β; log σ_y]  D_NUTS = 3
 *
 *  This is the BNN-regression-with-VI showcase. VI is the standard
 *  approach for neural networks; non-centered reparam handles the
 *  hierarchical weight-scale funnel that defeats centered VI.
 *
 *  PASS criteria (framework correctness):
 *    - Composite runs n_outer steps without crash
 *    - VI converges within its budget
 *    - σ_y fit recovers true σ_y to within factor 2
 *    - Training MSE < variance of y (model fits at all)
 *    - All final values finite
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
    constexpr std::size_t M = 5;     // input dim
    constexpr std::size_t J = 8;     // hidden units
    constexpr std::size_t N = 200;   // training obs

    // Weight packing: alpha0 (J) | A_tilde (M*J, col-major) | beta0 (1) | beta_tilde (J)
    const std::size_t off_a0 = 0;
    const std::size_t off_At = off_a0 + J;
    const std::size_t off_b0 = off_At + M * J;
    const std::size_t off_bt = off_b0 + 1;
    const std::size_t D_w    = off_bt + J;        // J + MJ + 1 + J

    // Hyperparam packing: log_sa | log_sb | log_sy
    const std::size_t off_lsa = 0;
    const std::size_t off_lsb = 1;
    const std::size_t off_lsy = 2;
    const std::size_t D_hyp  = 3;

    const double half_n_sd = 2.5;   // half-Normal(0, 2.5) on σ_α, σ_β, σ_y

    // ---- Synthesize data ----
    std::mt19937_64 rng_data(123);
    std::normal_distribution<double> N01(0.0, 1.0);

    // Truth: a smooth 1-hidden-layer regression function f(x).
    const double sigma_y_true = 0.3;
    arma::vec a0_true(J), b_true(J);
    arma::mat A_true(M, J);
    const double b0_true = 0.2;
    for (std::size_t j = 0; j < J; ++j) {
        a0_true[j] = 0.3 * N01(rng_data);
        b_true[j]  = 0.6 * N01(rng_data);
    }
    for (std::size_t m = 0; m < M; ++m)
        for (std::size_t j = 0; j < J; ++j)
            A_true(m, j) = 0.5 * N01(rng_data);

    arma::mat X(N, M);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t m = 0; m < M; ++m) X(i, m) = N01(rng_data);
        const arma::vec z = a0_true + A_true.t() * X.row(i).t();
        const arma::vec h = arma::tanh(z);
        const double f_x = b0_true + arma::dot(h, b_true);
        y[i] = f_x + sigma_y_true * N01(rng_data);
    }
    const double y_var = arma::var(y);

    // ---- VI block log_density: weights | hyperparams ----
    auto vi_log_density_grad =
        [&](const arma::vec& w, const block_context& ctx,
            arma::vec* grad_out) -> double {
        const arma::vec& hyp = ctx.at("hyp");
        const double sigma_a = std::exp(hyp[off_lsa]);
        const double sigma_b = std::exp(hyp[off_lsb]);
        const double sigma_y = std::exp(hyp[off_lsy]);
        const double s2y     = sigma_y * sigma_y;

        // Unpack
        const arma::vec a0(const_cast<double*>(w.memptr() + off_a0), J, false, true);
        const arma::mat At(const_cast<double*>(w.memptr() + off_At), M, J, false, true);
        const double b0 = w[off_b0];
        const arma::vec bt(const_cast<double*>(w.memptr() + off_bt), J, false, true);

        // Forward pass
        double log_lik = 0.0;
        arma::mat H(N, J);
        arma::vec res(N);
        for (std::size_t i = 0; i < N; ++i) {
            const arma::vec z = a0 + sigma_a * At.t() * X.row(i).t();
            const arma::vec h = arma::tanh(z);
            H.row(i) = h.t();
            const double f_x = b0 + sigma_b * arma::dot(h, bt);
            res[i] = y[i] - f_x;
            log_lik += -0.5 * res[i] * res[i] / s2y;
        }

        // Standard normal priors (non-centered)
        const double log_p_a0 = -0.5 * arma::dot(a0, a0);
        const double log_p_At = -0.5 * arma::accu(arma::square(At));
        const double log_p_b0 = -0.5 * b0 * b0;
        const double log_p_bt = -0.5 * arma::dot(bt, bt);

        const double log_p = log_lik + log_p_a0 + log_p_At + log_p_b0 + log_p_bt;

        if (grad_out != nullptr) {
            grad_out->set_size(D_w);
            grad_out->zeros();

            // Backprop: for each i, dL/d(f_x) = res / s2y
            for (std::size_t i = 0; i < N; ++i) {
                const double dL_dfx = res[i] / s2y;
                const arma::rowvec h_i = H.row(i);

                // d(f_x)/d(b0) = 1
                (*grad_out)[off_b0] += dL_dfx;
                // d(f_x)/d(bt_j) = sigma_b * h_j
                for (std::size_t j = 0; j < J; ++j)
                    (*grad_out)[off_bt + j] += dL_dfx * sigma_b * h_i[j];

                // d(f_x)/d(h_j) = sigma_b * bt_j
                arma::vec dL_dh(J, arma::fill::zeros);
                for (std::size_t j = 0; j < J; ++j)
                    dL_dh[j] = dL_dfx * sigma_b * bt[j];
                arma::vec sech2(J);
                for (std::size_t j = 0; j < J; ++j) sech2[j] = 1.0 - h_i[j] * h_i[j];
                arma::vec dL_dz = dL_dh % sech2;

                // d(z_j)/d(a0_j) = 1
                for (std::size_t j = 0; j < J; ++j)
                    (*grad_out)[off_a0 + j] += dL_dz[j];
                // d(z_j)/d(At_{mj}) = sigma_a * x_m
                for (std::size_t m = 0; m < M; ++m) {
                    const double x_im = X(i, m);
                    for (std::size_t j = 0; j < J; ++j)
                        (*grad_out)[off_At + j * M + m] += dL_dz[j] * sigma_a * x_im;
                }
            }

            // Priors
            for (std::size_t j = 0; j < J; ++j) (*grad_out)[off_a0 + j] += -a0[j];
            for (std::size_t k = 0; k < M*J; ++k) (*grad_out)[off_At + k] += -At[k];
            (*grad_out)[off_b0] += -b0;
            for (std::size_t j = 0; j < J; ++j) (*grad_out)[off_bt + j] += -bt[j];
        }
        return log_p;
    };

    // ---- NUTS block log_density: hyperparams | weights ----
    auto nuts_log_density_grad =
        [&](const arma::vec& hyp, const block_context& ctx,
            arma::vec* grad_out) -> double {
        const double lsa = hyp[off_lsa];
        const double lsb = hyp[off_lsb];
        const double lsy = hyp[off_lsy];
        const double sigma_a = std::exp(lsa);
        const double sigma_b = std::exp(lsb);
        const double sigma_y = std::exp(lsy);
        const double s2a = sigma_a * sigma_a;
        const double s2b = sigma_b * sigma_b;
        const double s2y = sigma_y * sigma_y;

        const arma::vec& w = ctx.at("weights");
        const arma::vec a0(const_cast<double*>(w.memptr() + off_a0), J, false, true);
        const arma::mat At(const_cast<double*>(w.memptr() + off_At), M, J, false, true);
        const double b0 = w[off_b0];
        const arma::vec bt(const_cast<double*>(w.memptr() + off_bt), J, false, true);

        // log_lik
        double log_lik_sum = 0.0;
        arma::vec res(N);
        for (std::size_t i = 0; i < N; ++i) {
            const arma::vec z = a0 + sigma_a * At.t() * X.row(i).t();
            const arma::vec h = arma::tanh(z);
            const double f_x = b0 + sigma_b * arma::dot(h, bt);
            res[i] = y[i] - f_x;
        }
        const double sum_res_sq = arma::dot(res, res);
        // log_lik = -N/2 log(s2y) - 0.5/s2y sum_res_sq (on natural)
        const double log_lik = -0.5 * static_cast<double>(N) * lsy * 2.0
                                - 0.5 / s2y * sum_res_sq;

        // half-Normal priors (natural) + Jacobian for log scale
        const double log_p_sa = -0.5 * s2a / (half_n_sd * half_n_sd) + lsa;
        const double log_p_sb = -0.5 * s2b / (half_n_sd * half_n_sd) + lsb;
        const double log_p_sy = -0.5 * s2y / (half_n_sd * half_n_sd) + lsy;

        const double log_p_tilde = log_lik + log_p_sa + log_p_sb + log_p_sy;

        if (grad_out != nullptr) {
            grad_out->set_size(D_hyp);

            // d(log_lik)/d(lsy): -N - 0.5 sum_res_sq * (-2/s2y) * s2y ...
            //   log_lik = -N lsy - 0.5/s2y sum_res_sq
            //   d/d(lsy) (-N lsy) = -N
            //   d/d(lsy) (-0.5 e^{-2 lsy} sum_res_sq) = sum_res_sq / s2y
            // half-Normal Jacobian:  d/d(lsy) (-0.5 s2y/A² + lsy) = -s2y/A² + 1
            (*grad_out)[off_lsy] =
                -static_cast<double>(N)
                + sum_res_sq / s2y
                - s2y / (half_n_sd * half_n_sd)
                + 1.0;

            // d(log_lik)/d(lsa): chain rule through h_i and z_i
            //   d(f_x_i)/d(sigma_a) = sigma_b sum_j bt_j (1-h_j²) (At^T x_i)_j
            //   d(sigma_a)/d(lsa) = sigma_a
            //   d(log_lik)/d(lsa) = sigma_a * sum_i (res_i/s2y) * d(f_x_i)/d(sigma_a)
            // d(log_lik)/d(lsb): d(f_x)/d(sigma_b) = sum_j h_j bt_j (note: f_x = b0 + sigma_b h^T bt)
            //   d(sigma_b)/d(lsb) = sigma_b
            //   d(log_lik)/d(lsb) = sigma_b * sum_i (res_i/s2y) * (h_i^T bt)
            double g_lsa = 0.0, g_lsb = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                const arma::vec z = a0 + sigma_a * At.t() * X.row(i).t();
                const arma::vec h = arma::tanh(z);
                const double dfx_d_sigma_b = arma::dot(h, bt);
                g_lsb += (res[i] / s2y) * dfx_d_sigma_b;
                // For sigma_a: d(z_j)/d(sigma_a) = (At^T x_i)_j; chain through tanh
                arma::vec dz_d_sa = At.t() * X.row(i).t();
                arma::vec dh_d_sa = (1.0 - arma::square(h)) % dz_d_sa;
                const double dfx_d_sigma_a = sigma_b * arma::dot(dh_d_sa, bt);
                g_lsa += (res[i] / s2y) * dfx_d_sigma_a;
            }
            g_lsa *= sigma_a;
            g_lsb *= sigma_b;
            // half-Normal + Jacobian contributions
            (*grad_out)[off_lsa] = g_lsa - s2a / (half_n_sd * half_n_sd) + 1.0;
            (*grad_out)[off_lsb] = g_lsb - s2b / (half_n_sd * half_n_sd) + 1.0;
        }
        return log_p_tilde;
    };

    // ---- Compose ----
    composite_block comp;
    comp.data().set("weights", arma::zeros(D_w));
    arma::vec hyp_init(D_hyp, arma::fill::value(std::log(0.5)));   // σ≈0.5 init
    comp.data().set("hyp", hyp_init);

    {
        mean_field_gaussian_vi_block_config vi_cfg;
        vi_cfg.name             = "weights";
        vi_cfg.initial_unc      = arma::zeros(D_w);
        vi_cfg.initial_log_sd   = arma::vec(D_w, arma::fill::value(-2.5));
        vi_cfg.log_density_grad = vi_log_density_grad;
        vi_cfg.dependencies     = {"hyp"};
        vi_cfg.optimizer.gamma_0              = 0.01;
        vi_cfg.optimizer.rho                  = 0.5;
        vi_cfg.optimizer.tau                  = 0.01;
        vi_cfg.optimizer.inner_iter_per_epoch = 500;
        vi_cfg.optimizer.max_epochs           = 10;
        vi_cfg.optimizer.S_khat               = 500;
        vi_cfg.n_mc_per_step                  = 4;
        comp.add_child(std::make_unique<mean_field_gaussian_vi_block>(vi_cfg));
    }
    {
        nuts_block_config nuts_cfg;
        nuts_cfg.name             = "hyp";
        nuts_cfg.initial_unc      = hyp_init;
        nuts_cfg.log_density_grad = nuts_log_density_grad;
        comp.add_child(std::make_unique<nuts_block>(nuts_cfg));
    }
    comp.data().declare_dependencies("weights", {"hyp"});
    comp.data().declare_dependencies("hyp",     {"weights"});

    // ---- Run ----
    std::mt19937_64 rng(2026);
    std::printf("\n=== test_vi_bnn_regression ===\n");
    std::printf("M=%zu, J=%zu, N=%zu, D_w=%zu, σ_y_true=%.3f\n",
                M, J, N, D_w, sigma_y_true);

    const std::size_t n_outer = 4000;
    for (std::size_t t = 0; t < n_outer; ++t) {
        comp.step(rng);
        if ((t + 1) % 500 == 0) {
            const auto& w = comp.child(0).current();
            const auto& h = comp.child(1).current();
            std::printf("  iter %4zu: ||w||=%.2f, σ_α=%.3f, σ_β=%.3f, σ_y=%.3f\n",
                        t + 1, arma::norm(w),
                        std::exp(h[off_lsa]), std::exp(h[off_lsb]),
                        std::exp(h[off_lsy]));
        }
    }

    const auto* vi_leaf = dynamic_cast<const mean_field_gaussian_vi_block*>(
        &comp.child(0));
    const arma::vec w_fit = vi_leaf->current();
    const arma::vec h_fit = comp.child(1).current();
    const double sigma_y_fit = std::exp(h_fit[off_lsy]);
    const double sigma_a_fit = std::exp(h_fit[off_lsa]);
    const double sigma_b_fit = std::exp(h_fit[off_lsb]);

    // Compute training MSE at the fitted mean.
    arma::vec a0_f(const_cast<double*>(w_fit.memptr() + off_a0), J, false, true);
    arma::mat At_f(const_cast<double*>(w_fit.memptr() + off_At), M, J, false, true);
    const double b0_f = w_fit[off_b0];
    arma::vec bt_f(const_cast<double*>(w_fit.memptr() + off_bt), J, false, true);
    double sse = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const arma::vec z = a0_f + sigma_a_fit * At_f.t() * X.row(i).t();
        const arma::vec h = arma::tanh(z);
        const double f_x = b0_f + sigma_b_fit * arma::dot(h, bt_f);
        sse += (y[i] - f_x) * (y[i] - f_x);
    }
    const double mse = sse / static_cast<double>(N);

    std::printf("\nFinal:\n");
    std::printf("  σ_α=%.3f, σ_β=%.3f, σ_y=%.3f (true σ_y=%.3f)\n",
                sigma_a_fit, sigma_b_fit, sigma_y_fit, sigma_y_true);
    std::printf("  Training MSE = %.4f (var(y) = %.4f, σ_y²_true = %.4f)\n",
                mse, y_var, sigma_y_true * sigma_y_true);
    std::printf("  ELBO=%.2f, khat=%.4f, VI converged=%s\n",
                vi_leaf->current_elbo(), vi_leaf->vi_history().final_khat,
                vi_leaf->converged() ? "yes" : "no");

    bool ok = true;
    if (!vi_leaf->converged()) {
        std::printf("  FAIL: VI did not converge\n"); ok = false;
    }
    if (!w_fit.is_finite() || !h_fit.is_finite()) {
        std::printf("  FAIL: non-finite outputs\n"); ok = false;
    }
    if (!(sigma_y_fit > sigma_y_true / 2.0 && sigma_y_fit < sigma_y_true * 2.0)) {
        std::printf("  FAIL: σ_y %.3f outside [%.3f, %.3f]\n",
                    sigma_y_fit, sigma_y_true / 2.0, sigma_y_true * 2.0);
        ok = false;
    }
    if (!(mse < y_var)) {
        std::printf("  FAIL: MSE %.4f >= var(y) %.4f (model doesn't fit)\n",
                    mse, y_var);
        ok = false;
    }
    std::printf("  NOTE: khat informational; mean-field BNN is known-biased\n");

    if (ok) { std::printf("\nPASS\n"); return 0; }
    else    { std::printf("\nFAIL\n"); return 1; }
}

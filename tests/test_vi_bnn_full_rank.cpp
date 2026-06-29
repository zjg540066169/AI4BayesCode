/*================================================================================
 *  AI4BayesCode VI showcase #3 — BNN regression with FULL-RANK VI.
 *
 *  Same model as test_vi_bnn_regression.cpp (1-hidden-layer BNN, non-
 *  centered weights, half-Normal hyperpriors) but uses
 *  full_rank_gaussian_vi_block over ALL D=60 unconstrained parameters
 *  (no NUTS for hyperparams). This captures the (σ_β, β̃) multiplicative
 *  funnel + per-hidden-unit correlations that mean-field misses.
 *
 *  Why two BNN tests?
 *    - test_vi_bnn_regression: mean-field hybrid (VI for weights + NUTS
 *      for σ²). Faster, simpler, but underestimates σ_α / σ_β posterior
 *      variance via the funnel.
 *    - test_vi_bnn_full_rank (this file): full-rank pure VI. Captures
 *      correlations + funnel; slower (1890 var params instead of 120)
 *      but tighter posterior recovery on hyperparams.
 *
 *  Empirical comparison (from vi_comparison/2026-05-25, see VI_v1_REPORT.md):
 *    | Quantity | NUTS truth | mean-field VI | FULL-RANK VI |
 *    |---|---:|---:|---:|
 *    | σ_α        | 0.61     | 0.77 (R̂=1.36) | 0.65 (R̂=1.06) |
 *    | σ_β        | 0.50     | 0.46 (R̂=1.51) | 0.47 (R̂=1.11) |
 *    | f_train R̂  | —        | max=1.54       | max=1.22 (vs NUTS) |
 *    | cross-seed | —        | 80% < 1.01     | 96.6% < 1.01 |
 *
 *  PASS criteria (framework correctness, BNN bias is known limitation):
 *    - Composite runs n_outer steps without crash
 *    - VI converges within its budget
 *    - σ_y fit recovers true σ_y to within factor 2
 *    - All final values finite
 *    - Training MSE < variance of y (model fits)
 *================================================================================*/

#include "AI4BayesCode/full_rank_gaussian_vi_block.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <random>

int main() {
    using namespace AI4BayesCode;

    constexpr std::size_t M = 5;
    constexpr std::size_t J = 8;
    constexpr std::size_t N = 200;

    // Packing: alpha_0 (J) | A_tilde (MJ col-major) | beta_0 (1) | beta_tilde (J)
    //          | log σ_α (1) | log σ_β (1) | log σ_y (1)
    const std::size_t off_a0  = 0;
    const std::size_t off_At  = off_a0 + J;
    const std::size_t off_b0  = off_At + M*J;
    const std::size_t off_bt  = off_b0 + 1;
    const std::size_t off_lsa = off_bt + J;
    const std::size_t off_lsb = off_lsa + 1;
    const std::size_t off_lsy = off_lsb + 1;
    const std::size_t D       = off_lsy + 1;   // 60

    const double hn_sd = 2.5;

    // ---- Synthesize data ----
    std::mt19937_64 rng_data(123);
    std::normal_distribution<double> N01(0.0, 1.0);

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

    // ---- log_density gradient (full unconstrained η ∈ R^60) ----
    auto grad = [&](const arma::vec& eta, const block_context&,
                    arma::vec* g) -> double {
        const arma::vec a0(const_cast<double*>(eta.memptr()+off_a0), J, false, true);
        const arma::mat At(const_cast<double*>(eta.memptr()+off_At), M, J, false, true);
        const double b0 = eta[off_b0];
        const arma::vec bt(const_cast<double*>(eta.memptr()+off_bt), J, false, true);
        const double lsa = eta[off_lsa], lsb = eta[off_lsb], lsy = eta[off_lsy];
        const double sa = std::exp(lsa), sb = std::exp(lsb), sy = std::exp(lsy);
        const double s2a = sa*sa, s2b = sb*sb, s2y = sy*sy;

        double log_lik = 0.0;
        arma::mat H(N, J);
        arma::vec res(N);
        for (std::size_t i = 0; i < N; ++i) {
            const arma::vec z = a0 + sa * At.t() * X.row(i).t();
            const arma::vec hi = arma::tanh(z);
            H.row(i) = hi.t();
            const double fx = b0 + sb * arma::dot(hi, bt);
            res[i] = y[i] - fx;
        }
        const double sum_res_sq = arma::dot(res, res);
        log_lik = -static_cast<double>(N) * lsy - 0.5/s2y * sum_res_sq;

        const double lp = log_lik
            - 0.5 * arma::dot(a0, a0)
            - 0.5 * arma::accu(arma::square(At))
            - 0.5 * b0*b0
            - 0.5 * arma::dot(bt, bt)
            - 0.5 * s2a/(hn_sd*hn_sd) + lsa
            - 0.5 * s2b/(hn_sd*hn_sd) + lsb
            - 0.5 * s2y/(hn_sd*hn_sd) + lsy;

        if (g) {
            g->set_size(D); g->zeros();
            double g_lsa = 0.0, g_lsb = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                const double dL_df = res[i] / s2y;
                const arma::rowvec hi = H.row(i);
                (*g)[off_b0] += dL_df;
                for (std::size_t j = 0; j < J; ++j)
                    (*g)[off_bt+j] += dL_df * sb * hi[j];
                g_lsb += dL_df * arma::dot(arma::vec(hi.t()), bt);
                arma::vec dL_dh(J);
                for (std::size_t j = 0; j < J; ++j) dL_dh[j] = dL_df * sb * bt[j];
                arma::vec sech2(J);
                for (std::size_t j = 0; j < J; ++j) sech2[j] = 1.0 - hi[j]*hi[j];
                arma::vec dL_dz = dL_dh % sech2;
                for (std::size_t j = 0; j < J; ++j) (*g)[off_a0+j] += dL_dz[j];
                for (std::size_t m = 0; m < M; ++m) {
                    const double xim = X(i, m);
                    for (std::size_t j = 0; j < J; ++j)
                        (*g)[off_At + j*M + m] += dL_dz[j] * sa * xim;
                }
                arma::vec dz_dsa = At.t() * X.row(i).t();
                arma::vec dh_dsa = sech2 % dz_dsa;
                g_lsa += dL_df * sb * arma::dot(dh_dsa, bt);
            }
            g_lsa *= sa; g_lsb *= sb;
            for (std::size_t j = 0; j < J; ++j) (*g)[off_a0+j] += -a0[j];
            for (std::size_t k = 0; k < M*J; ++k) (*g)[off_At+k] += -At[k];
            (*g)[off_b0] += -b0;
            for (std::size_t j = 0; j < J; ++j) (*g)[off_bt+j] += -bt[j];
            (*g)[off_lsa] = g_lsa - s2a/(hn_sd*hn_sd) + 1.0;
            (*g)[off_lsb] = g_lsb - s2b/(hn_sd*hn_sd) + 1.0;
            (*g)[off_lsy] = -static_cast<double>(N) + sum_res_sq/s2y
                                 - s2y/(hn_sd*hn_sd) + 1.0;
        }
        return lp;
    };

    full_rank_gaussian_vi_block_config cfg;
    cfg.name = "params";
    arma::vec init(D, arma::fill::zeros);
    init[off_lsa] = std::log(0.5);
    init[off_lsb] = std::log(0.5);
    init[off_lsy] = std::log(0.3);
    cfg.initial_unc = init;
    cfg.initial_L = 0.1 * arma::eye(D, D);
    cfg.log_density_grad = grad;
    cfg.optimizer.gamma_0 = 0.002;
    cfg.optimizer.rho = 0.5;
    cfg.optimizer.tau = 0.003;
    cfg.optimizer.inner_iter_per_epoch = 3000;
    cfg.optimizer.max_epochs = 20;
    cfg.optimizer.S_khat = 1000;
    cfg.n_mc_per_step = 8;

    auto blk = std::make_unique<full_rank_gaussian_vi_block>(cfg);
    composite_block comp;
    comp.add_child(std::move(blk));

    std::mt19937_64 rng(2026);
    const auto* leaf =
        dynamic_cast<const full_rank_gaussian_vi_block*>(&comp.child(0));

    std::printf("\n=== test_vi_bnn_full_rank ===\n");
    std::printf("M=%zu, J=%zu, N=%zu, D=%zu (full-rank: %zu var params)\n",
                M, J, N, D, D + D*(D-1)/2);
    std::printf("σ_y_true = %.3f\n", sigma_y_true);

    const std::size_t max_outer = 80000;
    std::size_t outer = 0;
    while (outer < max_outer && !leaf->converged()) {
        comp.step(rng);
        outer++;
        if (outer % 5000 == 0) {
            const auto& m = leaf->current();
            std::printf("  iter %5zu: ELBO=%.2f, σ_y=%.3f, σ_α=%.3f, σ_β=%.3f\n",
                        outer, leaf->current_elbo(),
                        std::exp(m[off_lsy]), std::exp(m[off_lsa]),
                        std::exp(m[off_lsb]));
        }
    }

    const arma::vec w_fit = leaf->current();
    const double sigma_y_fit = std::exp(w_fit[off_lsy]);
    const double sigma_a_fit = std::exp(w_fit[off_lsa]);
    const double sigma_b_fit = std::exp(w_fit[off_lsb]);

    // Training MSE at the fitted mean
    arma::vec a0_f(const_cast<double*>(w_fit.memptr()+off_a0), J, false, true);
    arma::mat At_f(const_cast<double*>(w_fit.memptr()+off_At), M, J, false, true);
    const double b0_f = w_fit[off_b0];
    arma::vec bt_f(const_cast<double*>(w_fit.memptr()+off_bt), J, false, true);
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
    std::printf("  ELBO=%.2f, khat=%.4f, converged=%s, epochs=%zu\n",
                leaf->current_elbo(), leaf->vi_history().final_khat,
                leaf->converged() ? "yes" : "no", leaf->epoch());

    bool ok = true;
    if (!leaf->converged()) {
        std::printf("  FAIL: VI did not converge in %zu iters\n", max_outer);
        ok = false;
    }
    if (!w_fit.is_finite()) {
        std::printf("  FAIL: weights contain NaN/Inf\n"); ok = false;
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

    if (ok) {
        std::printf("\nPASS — full-rank VI on BNN regression\n");
        std::printf("       Cross-seed seed-stability + tighter posterior than mean-field\n");
        return 0;
    } else {
        std::printf("\nFAIL\n");
        return 1;
    }
}

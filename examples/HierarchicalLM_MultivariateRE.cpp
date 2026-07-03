// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  HierarchicalLM_MultivariateRE.cpp
//
//  REFERENCE TEMPLATE for hierarchical linear regression with *multivariate*
//  random effects (random intercept + random slope per group) under an
//  LKJ(eta=2) prior on the correlation matrix of the random-effect vector.
//
//  PRIMARY SHOWCASE: constraints::cholesky_corr::wrap (no other shipped
//  example exercises this constraint).
//
//  MODEL (d = 2 random-effect dimensions)
//  --------------------------------------
//      y_i     ~ Normal(mu_i, sigma^2),                    i = 1..N
//      mu_i    = X_i' beta + Z_i' u_{g(i)}
//      u_j     = diag(tau) . L . z_j                        (non-centered)
//      z_j     ~ Normal_d(0, I_d),                          j = 1..J
//      beta    ~ Normal(0, 10^2 . I_p)
//      tau_k   ~ half-Normal(0, 2.5),                        k = 1..d
//      L       ~ Cholesky factor, R = L L', R ~ LKJ(eta=2)
//      sigma   ~ Jeffreys,  p(sigma) ~ 1/sigma
//
//  KEY DESIGN DECISIONS (informed by 2026-04-22 debug + convergence trials)
//  ------------------------------------------------------------------------
//  (a) NON-CENTERED on (tau, L, z): u_j = diag(tau) L z_j avoids the
//      funnel geometry.
//
//  (b) JOINT NUTS on (beta, z, log_tau, log_sigma) via joint_nuts_block
//      v0 with T11 DENSE METRIC adaptation.
//      Rationale: modular sampling of (tau) separately from (beta, z)
//      causes SLOW MIXING because tau's Gibbs conditional shifts
//      between sweeps (tau and z/beta are correlated through the mean
//      structure). Even at 10k+10k, modular tau R-hat sits at ~1.03.
//      Joint NUTS with dense metric on all 4 continuous groups
//      recovers R-hat <= 1.01.
//
//  (c) Manual log-transform on tau and sigma so the whole joint vector is
//      REAL and the dense-metric (Welford) pilot of joint_nuts_block
//      applies. (joint_nuts_block also supports POSITIVE slices directly,
//      but a log-REAL slice plays better with the dense metric here.) We
//      include the half-Normal prior Jacobian and the Jeffreys-cancellation
//      explicitly in the log-density lambda.
//
//  (d) R_chol stays modular (constraints::cholesky_corr not supported
//      in joint_nuts_block). Its conditional given (beta, z,
//      tau, sigma) is well-identified so modular sampling is fine.
//
//  (e) Every block sets n_warmup_per_step = 0 (validator Check #20).
//      A nonzero value re-runs dual-averaging on the KEPT draws, which
//      collapses the step size and FREEZES the block (R_chol stuck at its
//      warmup endpoint, within-chain sd 0 -> rank-R-hat = Inf). The
//      persistent between-call adaptation already carries the tuned
//      metric/step forward across sweeps, so per-step re-warmup is both
//      unnecessary and incorrect.
//
//  BLOCK COMPOSITION (2 blocks, down from 5)
//  -----------------------------------------
//    1. joint (beta, z, log_tau, log_sigma) : joint_nuts_block v0
//         dim = p + J*d + d + 1
//         DENSE METRIC adaptation (pilot 500 + adapt 1500 iters)
//    2. R_chol                              : nuts_block cholesky_corr
//
//  UNCONSTRAINED-SCALE LOG-DENSITY MATH
//  ------------------------------------
//  Let theta_cat = [beta; z; log_tau; log_sigma].
//  Natural-scale params: tau = exp(log_tau), sigma = exp(log_sigma).
//
//  Priors (natural scale + transform Jacobians on unconstrained scale):
//    beta      : -||beta||^2 / (2 * 100)                 (Normal prior on natural; identity transform)
//    z         : -||z||^2 / 2                             (N(0, I); identity)
//    log_tau_k : -tau_k^2 / (2 * 2.5^2) + log_tau_k       (half-Normal + Jacobian)
//    log_sigma : 0                                         (Jeffreys cancels Jacobian exactly)
//
//  Likelihood on unconstrained scale (sigma = exp(log_sigma)):
//    lik = -N * log_sigma - SSE / (2 * exp(2 * log_sigma))
//         (i.e., -N log sigma - SSE/(2 sigma^2) after substitution; the
//          +log_sigma Jacobian of the log-transform is ABSORBED into
//          the sigma-Jeffreys term, leaving -N instead of -(N+1).)
//
//  Gradients:
//    d / d beta_k      : -beta_k/100 + sum_i r_i X_{i,k} / sigma^2
//    d / d z_{j,k}     : -z_{j,k} + sum_{i:g(i)=j} r_i * sum_m Z_{i,m} tau_m L_{m,k} / sigma^2
//    d / d log_tau_l   : 1 - tau_l^2/6.25 + (tau_l/sigma^2) * sum_i r_i Z_{i,l} (Lz_{g(i)})_l
//    d / d log_sigma   : -N + SSE / sigma^2
//  (validator Check #11 must audit these offsets; verified 2026-04-22.)
//
//  R_chol block reads current (beta, z, tau, sigma) from ctx and uses
//  the LKJ(eta=2) prior on L's diagonal: log p(L | eta) = sum_k (d - k +
//  2*eta - 2) log L_{kk}. For d=2, eta=2: 3 log L_{0,0} + 2 log L_{1,1}.
// ============================================================================

// Frontend-independent standalone demo: this file builds as a plain C++
// program (no Rcpp, no pybind). It has an int main() at the bottom that
// simulates hierarchical data from a known truth, fits the model, and checks
// posterior recovery. No R / Python binding is built or required.

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/constraints.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <array>
#include <unordered_map>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
namespace constraints = AI4BayesCode::constraints;

namespace {

constexpr std::size_t D = 2;

inline arma::mat L_view_from_flat(const arma::vec& L_flat, std::size_t d) {
    return arma::mat(const_cast<double*>(L_flat.memptr()), d, d,
                     /*copy_aux_mem=*/false, /*strict=*/true);
}

// ============================================================================
// Joint log-density on theta_cat = [beta; z_flat; log_tau; log_sigma]
//   Length: p + J*d + d + 1
//   All entries are on UNCONSTRAINED scale. tau = exp(log_tau), sigma =
//   exp(log_sigma) are computed inside.
// ============================================================================
double joint_log_density(const arma::vec& theta_cat,
                         const block_context& ctx,
                         arma::vec* grad_out) {
    const arma::vec& y      = ctx.at("y");
    const arma::vec& X_flat = ctx.at("X");
    const arma::vec& Z_flat = ctx.at("Z");
    const arma::vec& group  = ctx.at("group_idx");
    const arma::vec& L_flat = ctx.at("R_chol");

    const std::size_t N = y.n_elem;
    const std::size_t d = D;
    const std::size_t p = static_cast<std::size_t>(
        ctx.at("beta_dim")[0] + 0.5);
    const std::size_t J = static_cast<std::size_t>(
        ctx.at("J_dim")[0] + 0.5);

    // Slice theta_cat
    const std::size_t off_z       = p;
    const std::size_t off_log_tau = p + J * d;
    const std::size_t off_log_sig = p + J * d + d;

    const arma::vec beta    = theta_cat.subvec(0,           p - 1);
    const arma::vec z_flat  = theta_cat.subvec(off_z,       off_z + J * d - 1);
    const arma::vec log_tau = theta_cat.subvec(off_log_tau, off_log_tau + d - 1);
    const double log_sigma  = theta_cat[off_log_sig];

    arma::vec tau(d);
    for (std::size_t k = 0; k < d; ++k) tau[k] = std::exp(log_tau[k]);
    const double sigma = std::exp(log_sigma);
    const double sigma2 = sigma * sigma;

    // Build matrices
    arma::mat X(const_cast<double*>(X_flat.memptr()), N, p, false, true);
    arma::mat Z(const_cast<double*>(Z_flat.memptr()), N, d, false, true);
    arma::mat L = L_view_from_flat(L_flat, d);
    arma::mat z_mat(const_cast<double*>(z_flat.memptr()), d, J, false, true);
    arma::mat Lz = L * z_mat;                   // d × J
    arma::mat u_mat = Lz;
    for (std::size_t k = 0; k < d; ++k) u_mat.row(k) *= tau[k];  // u = diag(tau) L z

    // Compute mu_i
    arma::vec mu(N);
    for (std::size_t i = 0; i < N; ++i) {
        double mu_i = 0.0;
        for (std::size_t k = 0; k < p; ++k) mu_i += X(i, k) * beta[k];
        const std::size_t g_i = static_cast<std::size_t>(
            std::llround(group[i])) - 1u;
        for (std::size_t k = 0; k < d; ++k) mu_i += Z(i, k) * u_mat(k, g_i);
        mu[i] = mu_i;
    }

    // SSE
    double sse = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double r = y[i] - mu[i];
        sse += r * r;
    }

    // Log-density (unconstrained scale)
    const double tau_prior_var = 2.5 * 2.5;       // half-Normal variance
    double lp = 0.0;

    // beta: -||beta||^2 / (2*100)
    lp -= 0.5 * arma::dot(beta, beta) / 100.0;

    // z: -||z||^2 / 2
    lp -= 0.5 * arma::dot(z_flat, z_flat);

    // log_tau: -tau^2/(2*6.25) + log_tau  (half-Normal + Jacobian)
    for (std::size_t k = 0; k < d; ++k) {
        lp -= 0.5 * tau[k] * tau[k] / tau_prior_var;
        lp += log_tau[k];
    }

    // log_sigma: 0 (Jeffreys cancels Jacobian)
    // Likelihood: -N * log_sigma - SSE/(2*sigma^2)
    lp -= static_cast<double>(N) * log_sigma;
    lp -= 0.5 * sse / sigma2;

    if (grad_out) {
        grad_out->set_size(theta_cat.n_elem);
        grad_out->zeros();

        const double inv_s2 = 1.0 / sigma2;

        // ---- d / d beta
        for (std::size_t k = 0; k < p; ++k) {
            double g = -beta[k] / 100.0;
            for (std::size_t i = 0; i < N; ++i) {
                g += (y[i] - mu[i]) * X(i, k) * inv_s2;
            }
            (*grad_out)[k] = g;
        }

        // ---- d / d z_{j,k}
        for (std::size_t idx = 0; idx < J * d; ++idx)
            (*grad_out)[off_z + idx] = -z_flat[idx];
        for (std::size_t i = 0; i < N; ++i) {
            const double r_i = y[i] - mu[i];
            const std::size_t g_i = static_cast<std::size_t>(
                std::llround(group[i])) - 1u;
            for (std::size_t k = 0; k < d; ++k) {
                double coef = 0.0;
                for (std::size_t m = 0; m < d; ++m) {
                    coef += Z(i, m) * tau[m] * L(m, k);
                }
                (*grad_out)[off_z + g_i * d + k] += r_i * coef * inv_s2;
            }
        }

        // ---- d / d log_tau_l = 1 - tau_l^2/6.25 + (tau_l/sigma^2) * sum_i r_i Z_{i,l} (Lz_{g(i)})_l
        for (std::size_t l = 0; l < d; ++l) {
            double g = 1.0 - tau[l] * tau[l] / tau_prior_var;
            for (std::size_t i = 0; i < N; ++i) {
                const double r_i = y[i] - mu[i];
                const std::size_t g_i = static_cast<std::size_t>(
                    std::llround(group[i])) - 1u;
                g += tau[l] * r_i * Z(i, l) * Lz(l, g_i) * inv_s2;
            }
            (*grad_out)[off_log_tau + l] = g;
        }

        // ---- d / d log_sigma = -N + SSE / sigma^2
        (*grad_out)[off_log_sig] = -static_cast<double>(N) + sse * inv_s2;
    }
    return lp;
}

// ============================================================================
// R_chol block log-density (stays separate; cholesky_corr constraint)
//   Reads beta, z, tau, sigma from shared_data (written by joint block)
// ============================================================================
double R_chol_log_density(const arma::vec& L_nat,
                          const block_context& ctx,
                          arma::vec* grad_nat) {
    const arma::vec& y      = ctx.at("y");
    const arma::vec& X_flat = ctx.at("X");
    const arma::vec& Z_flat = ctx.at("Z");
    const arma::vec& group  = ctx.at("group_idx");
    const arma::vec& beta   = ctx.at("beta");
    const arma::vec& z_flat = ctx.at("z_flat");
    const arma::vec& tau    = ctx.at("tau");
    const double sigma      = ctx.at("sigma")[0];

    const std::size_t N = y.n_elem;
    const std::size_t p = beta.n_elem;
    const std::size_t d = D;
    const std::size_t J = z_flat.n_elem / d;

    arma::mat X(const_cast<double*>(X_flat.memptr()), N, p, false, true);
    arma::mat Z(const_cast<double*>(Z_flat.memptr()), N, d, false, true);
    arma::mat L = L_view_from_flat(L_nat, d);
    arma::mat z_mat(const_cast<double*>(z_flat.memptr()), d, J, false, true);
    arma::mat Lz = L * z_mat;
    arma::mat u_mat = Lz;
    for (std::size_t k = 0; k < d; ++k) u_mat.row(k) *= tau[k];

    arma::vec mu(N);
    for (std::size_t i = 0; i < N; ++i) {
        double mu_i = 0.0;
        for (std::size_t k = 0; k < p; ++k) mu_i += X(i, k) * beta[k];
        const std::size_t g_i = static_cast<std::size_t>(
            std::llround(group[i])) - 1u;
        for (std::size_t k = 0; k < d; ++k) mu_i += Z(i, k) * u_mat(k, g_i);
        mu[i] = mu_i;
    }

    const double eta = 2.0;
    double lp = 0.0;
    for (std::size_t k = 0; k < d; ++k) {
        const double weight = static_cast<double>(d) -
                              (static_cast<double>(k) + 1.0) +
                              2.0 * eta - 2.0;
        lp += weight * std::log(L(k, k));
    }

    const double inv_s2 = 1.0 / (sigma * sigma);
    double sse = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double r = y[i] - mu[i];
        sse += r * r;
    }
    lp -= 0.5 * sse * inv_s2;

    if (grad_nat) {
        grad_nat->set_size(d * d);
        grad_nat->zeros();
        for (std::size_t k = 0; k < d; ++k) {
            const double weight = static_cast<double>(d) -
                                  (static_cast<double>(k) + 1.0) +
                                  2.0 * eta - 2.0;
            (*grad_nat)[k * d + k] += weight / L(k, k);
        }
        for (std::size_t i = 0; i < N; ++i) {
            const double r_i = y[i] - mu[i];
            const std::size_t g_i = static_cast<std::size_t>(
                std::llround(group[i])) - 1u;
            for (std::size_t k = 0; k < d; ++k) {
                for (std::size_t l = 0; l < d; ++l) {
                    const double coef = Z(i, k) * tau[k] * z_mat(l, g_i);
                    (*grad_nat)[l * d + k] += r_i * coef * inv_s2;
                }
            }
        }
    }
    return lp;
}

}  // anonymous namespace

class HierarchicalLM_MultivariateRE {
public:
    HierarchicalLM_MultivariateRE(const arma::vec& y_obs,
                                  const arma::mat& X_fixed,
                                  const arma::ivec& group_idx_1indexed,
                                  int rng_seed,
                                  bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          readapt_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0xBF58476D1CE4E5B9ULL}),
          impl_(std::make_unique<composite_block>("HierarchicalLM_MultivariateRE")),
          keep_history_(keep_history)
    {
        if (y_obs.n_elem < 2) throw std::runtime_error("N must be >= 2");
        if (X_fixed.n_rows != y_obs.n_elem)
            throw std::runtime_error("X rows must equal y length");
        if (X_fixed.n_cols < static_cast<arma::uword>(D))
            throw std::runtime_error("X must have at least d columns");
        if (group_idx_1indexed.n_elem != y_obs.n_elem)
            throw std::runtime_error("group_idx length must equal y length");

        N_ = y_obs.n_elem;
        p_ = X_fixed.n_cols;
        d_ = D;

        int J_raw = 0;
        for (arma::uword i = 0; i < group_idx_1indexed.n_elem; ++i) {
            if (group_idx_1indexed[i] < 1) throw std::runtime_error("group_idx must be 1-indexed");
            if (group_idx_1indexed[i] > J_raw) J_raw = group_idx_1indexed[i];
        }
        J_ = static_cast<std::size_t>(J_raw);

        // Install fixed data + dimension metadata the joint lambda needs.
        impl_->data().set("y", y_obs);
        impl_->data().set("X", arma::vectorise(X_fixed));
        arma::mat Z_fixed = X_fixed.cols(0, D - 1);
        impl_->data().set("Z", arma::vectorise(Z_fixed));
        arma::vec group_idx_dbl(N_);
        for (std::size_t i = 0; i < N_; ++i)
            group_idx_dbl[i] = static_cast<double>(group_idx_1indexed[i]);
        impl_->data().set("group_idx", group_idx_dbl);
        impl_->data().set("beta_dim", arma::vec{static_cast<double>(p_)});
        impl_->data().set("J_dim",    arma::vec{static_cast<double>(J_)});

        // Initial values
        arma::vec beta_init;
        if (N_ > p_)
            beta_init = arma::solve(X_fixed, y_obs);
        else
            beta_init = arma::vec(p_, arma::fill::zeros);
        impl_->data().set("beta", beta_init);
        impl_->data().set("z_flat", arma::vec(J_ * d_, arma::fill::zeros));
        impl_->data().set("tau", arma::vec(d_, arma::fill::ones));

        arma::mat L_init = arma::eye<arma::mat>(d_, d_);
        impl_->data().set("R_chol", arma::vectorise(L_init));

        arma::vec r_init = y_obs - X_fixed * beta_init;
        const double sigma_init = std::max(arma::stddev(r_init), 1e-2);
        impl_->data().set("sigma", arma::vec{sigma_init});
        impl_->data().set("y_rep", arma::vec(N_, arma::fill::zeros));

        // Dependencies
        impl_->data().declare_dependencies(
            "joint_block", {"y", "X", "Z", "group_idx", "beta_dim", "J_dim",
                             "R_chol"});
        impl_->data().declare_dependencies(
            "R_chol", {"y", "X", "Z", "group_idx",
                        "beta", "z_flat", "tau", "sigma"});

        // ---- Q7=A full predict-DAG reconstruction --------------------
        // Deterministic chain reconstructed as first-class nodes (so
        // ai4bayescode_plot_dag shows the generative story, not 8 sources collapsed
        // into y_rep):
        //   mu_fixed = X*beta                      (fixed effects, len N)
        //   u        = diag(tau) @ L @ z           (realized RE, D x J,
        //                                           flat col-major)
        //   y_rep ~ N(mu_fixed_i + Z_i' u[:,g_i], sigma^2)
        // RE rule (Q1/Q7=A): a KNOWN group uses sampled u[:,g]
        // (deterministic; in-sample bit-identical to the old collapsed
        // path — same value, same per-i obs-noise RNG order, no extra
        // draws). An UNSEEN group draws z_g ~ N(0, I_D) ONCE then
        // u_g = diag(tau) L z_g (the intrinsically-stochastic part,
        // confined to y_rep).
        impl_->data().set("mu_fixed", arma::vec(N_, arma::fill::zeros));
        impl_->data().set("u", arma::vec(D * J_, arma::fill::zeros));
        {
            const std::size_t Nl = N_, pl = p_, Jl = J_;
            impl_->data().register_refresher(
                "mu_fixed",
                [Nl, pl](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                    const arma::vec& X_flat = d.get("X");
                    const arma::vec& beta   = d.get("beta");
                    arma::mat X(const_cast<double*>(X_flat.memptr()),
                                Nl, pl, false, true);
                    arma::vec mf(Nl);
                    for (std::size_t i = 0; i < Nl; ++i) {
                        double xb = 0.0;
                        for (std::size_t k = 0; k < pl; ++k)
                            xb += X(i, k) * beta[k];
                        mf[i] = xb;
                    }
                    return mf;
                });
            impl_->data().register_refresher(
                "u",
                [Jl](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                    const arma::vec& z_flat = d.get("z_flat");
                    const arma::vec& tau    = d.get("tau");
                    const arma::vec& L_flat = d.get("R_chol");
                    arma::mat L(const_cast<double*>(L_flat.memptr()),
                                D, D, false, true);
                    arma::mat zm(const_cast<double*>(z_flat.memptr()),
                                 D, Jl, false, true);
                    arma::mat um = L * zm;                 // D x J
                    for (std::size_t k = 0; k < D; ++k) um.row(k) *= tau[k];
                    return arma::vectorise(um);            // col-major D*J
                });
        }

        // Predict DAG: X,beta -> mu_fixed ; z_flat,tau,R_chol -> u ;
        // mu_fixed,Z,group_idx,u,tau,R_chol,sigma -> y_rep.
        impl_->data().declare_data_input("X");
        impl_->data().declare_data_input("Z");
        impl_->data().declare_data_input("group_idx");
        impl_->data().declare_predict_edges("X",         {"mu_fixed"});
        impl_->data().declare_predict_edges("beta",      {"mu_fixed"});
        impl_->data().declare_predict_edges("z_flat",    {"u"});
        impl_->data().declare_predict_edges("tau",       {"u"});
        impl_->data().declare_predict_edges("R_chol",    {"u"});
        impl_->data().declare_predict_edges("mu_fixed",  {"y_rep"});
        impl_->data().declare_predict_edges("Z",         {"y_rep"});
        impl_->data().declare_predict_edges("group_idx", {"y_rep"});
        impl_->data().declare_predict_edges("u",         {"y_rep"});
        impl_->data().declare_predict_edges("tau",       {"y_rep"});
        impl_->data().declare_predict_edges("R_chol",    {"y_rep"});
        impl_->data().declare_predict_edges("sigma",     {"y_rep"});

        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                // Reads the reconstructed deterministic nodes mu_fixed
                // (=X*beta) and u (=diag(tau) L z, flat col-major D*J)
                // plus Z, group, tau, R_chol, sigma. KNOWN group g (0-
                // based < J): RE = Z_i' u[:,g] (no RNG -> in-sample
                // bit-identical to old). UNSEEN group: draw z_g ~
                // N(0,I_D) ONCE then u_g = diag(tau) L z_g (Q7=A).
                const arma::vec& mf     = d.get("mu_fixed");
                const arma::vec& Z_flat = d.get("Z");
                const arma::vec& group  = d.get("group_idx");
                const arma::vec& u_flat = d.get("u");        // D*J col-major
                const arma::vec& tau    = d.get("tau");
                const arma::vec& L_flat = d.get("R_chol");
                const double sigma      = d.get("sigma")[0];
                const std::size_t Nc    = group.n_elem;
                const std::size_t J_cur = u_flat.n_elem / D;
                arma::mat L(const_cast<double*>(L_flat.memptr()),
                            D, D, false, true);
                std::normal_distribution<double> norm(0.0, 1.0);
                std::unordered_map<std::size_t, std::array<double, D>> unseen;
                arma::vec y_rep_out(Nc);
                for (std::size_t i = 0; i < Nc; ++i) {
                    const std::size_t g = static_cast<std::size_t>(
                        std::llround(group[i])) - 1u;
                    double re = 0.0;
                    if (g < J_cur) {
                        for (std::size_t k = 0; k < D; ++k)
                            re += Z_flat[i + k * Nc] * u_flat[k + g * D];
                    } else {
                        auto it = unseen.find(g);
                        if (it == unseen.end()) {
                            arma::vec zg(D);
                            for (std::size_t k = 0; k < D; ++k)
                                zg[k] = norm(rng);
                            arma::vec ug = L * zg;            // D
                            std::array<double, D> ua;
                            for (std::size_t k = 0; k < D; ++k)
                                ua[k] = tau[k] * ug[k];
                            it = unseen.emplace(g, ua).first;
                        }
                        for (std::size_t k = 0; k < D; ++k)
                            re += Z_flat[i + k * Nc] * it->second[k];
                    }
                    y_rep_out[i] = mf[i] + re + sigma * norm(rng);
                }
                return y_rep_out;
            });

        // -------- Joint block (beta, z, log_tau, log_sigma) with dense metric
        {
            joint_nuts_block_config cfg;
            cfg.name = "joint_block";
            cfg.sub_params.push_back({"beta",      p_});
            cfg.sub_params.push_back({"z_flat",    J_ * d_});
            cfg.sub_params.push_back({"log_tau",   d_});
            cfg.sub_params.push_back({"log_sigma", 1});
            // Concatenate initial values (log of positive for last two).
            arma::vec init_cat(p_ + J_ * d_ + d_ + 1);
            init_cat.subvec(0, p_ - 1) = beta_init;
            init_cat.subvec(p_, p_ + J_ * d_ - 1).zeros();       // z init = 0
            init_cat.subvec(p_ + J_ * d_, p_ + J_ * d_ + d_ - 1).zeros();   // log_tau init = 0 (tau = 1)
            init_cat[p_ + J_ * d_ + d_] = std::log(sigma_init);
            cfg.initial_cat = init_cat;
            cfg.log_density_grad = &joint_log_density;

            // Dense metric essential for this joint's correlation structure.
            cfg.use_dense_metric         = true;
            cfg.dense_metric_pilot_iters = 500;
            cfg.dense_metric_adapt_iters = 1500;
            cfg.n_warmup_first_call      = 500;
            cfg.n_warmup_per_step        = 0;   // MUST be 0 (validator Check #20):
            // per-step re-adaptation runs dual-averaging on the KEPT draws, which
            // collapses the step size toward 0 and freezes the block. Persistent
            // between-call adaptation already carries the tuned metric/step forward.

            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
        }

        // -------- R_chol block (cholesky_corr — PRIMARY SHOWCASE)
        {
            nuts_block_config cfg;
            cfg.name        = "R_chol";
            cfg.initial_unc = arma::vec(d_ * (d_ - 1) / 2, arma::fill::zeros);
            cfg.constrain   = constraints::cholesky_corr::constrain;
            cfg.unconstrain = constraints::cholesky_corr::unconstrain;
            cfg.initial_step_size   = 0.05;
            cfg.n_warmup_first_call = 500;
            cfg.n_warmup_per_step   = 0;   // MUST be 0 (Check #20): nonzero re-adapts
            // on kept draws -> epsilon collapses -> R_chol frozen (sd 0) -> R-hat=Inf.
            cfg.log_density_grad =
                [](const arma::vec& t_unc, const block_context& ctx,
                   arma::vec* grad) {
                    return constraints::cholesky_corr::wrap(t_unc, grad,
                        [&](const arma::vec& t_nat, arma::vec* g_nat) {
                            return R_chol_log_density(t_nat, ctx, g_nat);
                        });
                };
            impl_->add_child(std::make_unique<nuts_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);

        // After each step the joint block writes log_tau / log_sigma to
        // shared_data (sub-param keys). We also need tau = exp(log_tau) and
        // sigma = exp(log_sigma) visible to the R_chol block and y_rep
        // refresher under the "tau" / "sigma" keys. Register a post-step
        // deterministic refresher that converts.
        impl_->data().register_refresher(
            "tau",
            [](const AI4BayesCode::shared_data_t& d) {
                const arma::vec& log_tau = d.get("log_tau");
                arma::vec tau(log_tau.n_elem);
                for (arma::uword k = 0; k < log_tau.n_elem; ++k)
                    tau[k] = std::exp(log_tau[k]);
                return tau;
            });
        impl_->data().register_refresher(
            "sigma",
            [](const AI4BayesCode::shared_data_t& d) {
                return arma::vec{std::exp(d.get("log_sigma")[0])};
            });
        // Order matters: tau before u (u reads tau); mu_fixed reads
        // beta; u reads z_flat,tau,R_chol. Also refresh u after the
        // R_chol block updates the Cholesky factor.
        impl_->data().declare_invalidates(
            "joint_block", {"tau", "sigma", "mu_fixed", "u"});
        impl_->data().declare_invalidates("R_chol", {"u"});
        // Prime initial tau / sigma from the initial log_tau / log_sigma.
        // The joint block's write-back on first step will populate these
        // keys; until then we keep the tau=1 / sigma=sigma_init defaults
        // already set above.
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        if (n_steps < 0) throw std::runtime_error("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Neutral-typed snapshot of the current draw. Random-effect z and the
    // Cholesky factor L are returned flattened (column-major) under the
    // "z_flat" / "L" / "R" keys; R = L L' is the implied correlation matrix.
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        const arma::vec& L_flat = impl_->data().get("R_chol");
        out["beta"]   = impl_->data().get("beta");
        out["z_flat"] = impl_->data().get("z_flat");
        out["tau"]    = impl_->data().get("tau");
        out["L"]      = L_flat;
        out["sigma"]  = impl_->data().get("sigma");
        {
            arma::mat L(const_cast<double*>(L_flat.memptr()), d_, d_, false, true);
            out["R"] = arma::vectorise(L * L.t());
        }
        return out;
    }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    /// 7th R-level method: re-tune NUTS metric (mass matrix + step size +
    /// dual averaging) without advancing chain state. Available because
    /// the composite contains NUTS-family children. See system_design.md
    /// §13 NUTS-family + validator.md §24.
    void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1) {
        if (n < 0) {
            throw std::runtime_error("readapt_NUTS: n must be non-negative");
        }
        impl_->readapt_NUTS(static_cast<std::size_t>(n),
                            reset, readapt_rng_, max_tree_depth < 0 ? std::size_t(0) : static_cast<std::size_t>(max_tree_depth));
    }


private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_; // readapt_NUTS() advances it (7th method)
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      N_ = 0;
    std::size_t                      p_ = 0;
    std::size_t                      d_ = D;
    std::size_t                      J_ = 0;
};

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulates hierarchical linear data with multivariate (d=2) random effects
//  from a KNOWN truth:
//      y_i = X_i' beta + Z_i' u_{g(i)} + sigma * eps_i,
//      u_j = diag(tau) . chol(R) . z_j,   z_j ~ N(0, I_2),
//  with X_{i,0} = 1 (intercept) and Z_i = [X_{i,0}, X_{i,1}] (random
//  intercept + random slope). Fits via joint-NUTS (beta, z, log_tau,
//  log_sigma) + a modular cholesky_corr block on R, then checks:
//    (1) posterior-mean fixed effects beta recover the truth, and
//    (2) the model BEATS a naive pooled OLS-residual sigma (i.e. accounting
//        for the random effects reduces residual scale toward sigma_true).
//==============================================================================
#include <cstdio>
int main() {
    const std::size_t J = 30;          // number of groups
    const std::size_t per_group = 20;  // observations per group
    const std::size_t N = J * per_group;
    const std::size_t p = 3;           // fixed-effect dimension (incl. intercept)

    const arma::vec beta_true = {1.5, -2.0, 0.8};
    const double    sigma_true = 0.6;
    const arma::vec tau_true   = {0.9, 0.7};   // RE scales (intercept, slope)
    // True RE correlation matrix R_true (rho = 0.5) and its Cholesky factor.
    const double rho = 0.5;
    arma::mat R_true = {{1.0, rho}, {rho, 1.0}};
    arma::mat L_true = arma::chol(R_true, "lower");

    std::mt19937_64 sim_rng(2024);
    std::normal_distribution<double> snorm(0.0, 1.0);

    // Design matrix: column 0 = intercept, columns 1..2 = covariates.
    arma::mat X(N, p, arma::fill::ones);
    for (std::size_t i = 0; i < N; ++i) {
        X(i, 1) = snorm(sim_rng);
        X(i, 2) = snorm(sim_rng);
    }
    arma::ivec group(N);
    for (std::size_t i = 0; i < N; ++i)
        group[i] = static_cast<int>(i / per_group) + 1;  // 1-indexed

    // Random effects per group: u_j = diag(tau) L z_j. The model assumes
    // E[u] = 0; we center the simulated RE columns to exactly zero mean so
    // the fixed intercept stays identifiable (a non-zero RE sample mean is
    // otherwise indistinguishable from a shift in the fixed intercept).
    arma::mat u_true(2, J);
    for (std::size_t j = 0; j < J; ++j) {
        arma::vec z(2);
        z[0] = snorm(sim_rng);
        z[1] = snorm(sim_rng);
        arma::vec u = L_true * z;
        u_true(0, j) = tau_true[0] * u[0];
        u_true(1, j) = tau_true[1] * u[1];
    }
    u_true.row(0) -= arma::mean(u_true.row(0));
    u_true.row(1) -= arma::mean(u_true.row(1));

    // Response. Z_i = [X_{i,0}=1, X_{i,1}] -> random intercept + random slope.
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t g = static_cast<std::size_t>(group[i]) - 1u;
        double mu = arma::dot(X.row(i), beta_true);
        mu += X(i, 0) * u_true(0, g) + X(i, 1) * u_true(1, g);
        y[i] = mu + sigma_true * snorm(sim_rng);
    }

    // Naive baseline: pooled OLS ignoring random effects -> residual std.
    arma::vec beta_ols = arma::solve(X, y);
    arma::vec r_ols = y - X * beta_ols;
    const double sigma_naive = arma::stddev(r_ols);

    // Fit the hierarchical model.
    HierarchicalLM_MultivariateRE model(y, X, group, /*rng_seed=*/7,
                                        /*keep_history=*/false);
    model.step(500);   // warmup

    arma::vec beta_bar(p, arma::fill::zeros);
    double    sigma_bar = 0.0;
    double    rho_bar = 0.0;
    const int M = 1500;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        beta_bar  += cur.at("beta");
        sigma_bar += cur.at("sigma")[0];
        // R is the 2x2 implied correlation (col-major flat): off-diagonal.
        rho_bar   += cur.at("R")[1];
    }
    beta_bar  /= static_cast<double>(M);
    sigma_bar /= static_cast<double>(M);
    rho_bar   /= static_cast<double>(M);

    std::printf("HierarchicalLM_MultivariateRE demo (N=%zu, J=%zu groups)\n",
                N, J);
    std::printf("  beta_hat  = [% .3f % .3f % .3f]  (truth [% .3f % .3f % .3f])\n",
                beta_bar[0], beta_bar[1], beta_bar[2],
                beta_true[0], beta_true[1], beta_true[2]);
    std::printf("  sigma_hat = %.3f  (truth %.3f)   sigma_naive(pooled OLS) = %.3f\n",
                sigma_bar, sigma_true, sigma_naive);
    std::printf("  rho_hat   = %.3f  (truth %.3f)\n", rho_bar, rho);

    const double beta_err = arma::norm(beta_bar - beta_true, "inf");
    const bool beta_ok  = beta_err < 0.20;
    const bool sigma_ok = std::abs(sigma_bar - sigma_true) < 0.15;
    // Accounting for the random effects must beat the naive pooled sigma.
    const bool beats_naive = sigma_bar < sigma_naive - 0.10;
    const bool ok = beta_ok && sigma_ok && beats_naive;

    std::printf("  |beta_hat - beta_true|_inf = %.3f (< 0.20 ? %s)\n",
                beta_err, beta_ok ? "yes" : "no");
    std::printf("  beats naive sigma by %.3f (> 0.10 ? %s)\n",
                sigma_naive - sigma_bar, beats_naive ? "yes" : "no");
    std::printf("%s\n",
                ok ? "[demo PASS] joint-NUTS + cholesky_corr recovers "
                     "(beta, sigma) and beats pooled OLS"
                   : "[demo FAIL]");
    return ok ? 0 : 1;
}

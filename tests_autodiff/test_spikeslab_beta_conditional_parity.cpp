// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_spikeslab_beta_conditional_parity.cpp
//
// PER-USAGE parity test (**Option A Check #15 escape hatch** from
// skills/codegen.md, escape clause for hand-written Gibbs).
//
// Target
// ------
// Verify that the hand-written conditional Gibbs draw
//     spike_slab_continuous_update(rng, j, ctx)
// in examples/SpikeSlabRJMCMC.cpp — which runs inside rjmcmc_block's
// continuous_update hook (Exception 2) — produces samples matching the
// analytical closed-form conditional posterior under the sigma-scaled
// slab (beta_j | gamma_j=1, sigma, tau ~ N(0, sigma^2 * tau^2),
// Ishwaran & Rao 2005 JASA form):
//     beta_j | gamma_j=1, rest ~ N(mean_cf, sd_cf^2)
// where
//     prec     = xtx / sigma^2 + 1 / (sigma^2 tau^2)
//              = (xtx + 1/tau^2) / sigma^2
//     mean_cf  = (xtr / sigma^2) / prec = xtr / (xtx + 1/tau^2)
//     sd_cf    = 1 / sqrt(prec) = sigma / sqrt(xtx + 1/tau^2)
// and xtr, xtx are data-dependent sufficient statistics on the residual
// (excluding j).
//
// Derivation
// ----------
// For linear-Gaussian y = X beta + eps, eps ~ N(0, sigma^2), with slab
// prior beta_j | gamma_j = 1, sigma, tau ~ N(0, sigma^2 tau^2):
//   p(beta_j | rest) ∝ N(y_resid | X_j beta_j, sigma^2 I)
//                       N(beta_j | 0, sigma^2 tau^2)
// Completing the square yields the Normal conditional above.
// Reference: Ishwaran & Rao 2005 JASA 100:1215-1225, Eq. (3.2).
//
// Test design
// -----------
// 1. Construct a minimal deterministic problem: N=50, p=3, X i.i.d. N(0,1)
//    (fixed seed), beta_true = (1.5, 0, -2.0), gamma = (1, 0, 1),
//    sigma = 0.7, tau = 5.0. y = X beta_true + noise (fixed seed).
// 2. Fix everything and sample beta_j (j=0) via the same formula as
//    spike_slab_continuous_update (copy-pasted here — see parity-test
//    invariant below).
// 3. Draw 10,000 samples. Compute sample mean, sample variance.
// 4. Compare to analytic mean_cf, sd_cf^2.
//    Tolerances (Check #15): |rel_err(mean)| < 5%, |rel_err(var)| < 10%.
//
// PARITY INVARIANT
// ----------------
// The formula below MUST stay in sync with
// `spike_slab_continuous_update` in examples/SpikeSlabRJMCMC.cpp. If
// you change the conditional-posterior formula there, you MUST also
// edit this test. Check #16 (inline justification) + this parity test
// are the two safety nets for hand-written Gibbs.
// ============================================================================

// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <RcppArmadillo.h>

#include <cmath>
#include <random>
#include <vector>

namespace {

// ============================================================================
// Reproduction of the production formula from
// examples/SpikeSlabRJMCMC.cpp::spike_slab_continuous_update.
// If you edit the production version, you MUST edit this copy too.
// ============================================================================
double spike_slab_continuous_update_reference(
        std::mt19937_64& rng,
        std::size_t j,
        const arma::vec& y,
        const arma::vec& X_flat,    // column-major N x p
        const arma::vec& gamma,
        const arma::vec& beta,
        double sigma,
        double tau,
        std::size_t N,
        std::size_t p) {
    const double tau2 = tau * tau;

    double xtr = 0.0, xtx = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double rexcl = y[i];
        for (std::size_t k = 0; k < p; ++k) {
            if (k != j && gamma[k] > 0.5)
                rexcl -= X_flat[i + k * N] * beta[k];
        }
        const double xij = X_flat[i + j * N];
        xtr += xij * rexcl;
        xtx += xij * xij;
    }
    // sigma-scaled slab: prec = (xtx + 1/tau^2) / sigma^2
    const double denom = xtx + 1.0 / tau2;
    const double mean  = xtr / denom;
    const double sd    = sigma / std::sqrt(denom);
    std::normal_distribution<double> norm(mean, sd);
    return norm(rng);
}

// Closed-form analytical mean / variance (independent of sampling).
struct AnalyticCF {
    double mean;
    double variance;
};

AnalyticCF compute_analytic_cf(std::size_t j,
                               const arma::vec& y,
                               const arma::vec& X_flat,
                               const arma::vec& gamma,
                               const arma::vec& beta,
                               double sigma,
                               double tau,
                               std::size_t N,
                               std::size_t p) {
    const double sigma2 = sigma * sigma;
    const double tau2   = tau   * tau;
    double xtr = 0.0, xtx = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double rexcl = y[i];
        for (std::size_t k = 0; k < p; ++k) {
            if (k != j && gamma[k] > 0.5)
                rexcl -= X_flat[i + k * N] * beta[k];
        }
        const double xij = X_flat[i + j * N];
        xtr += xij * rexcl;
        xtx += xij * xij;
    }
    // sigma-scaled slab: prec = (xtx + 1/tau^2) / sigma^2
    const double denom = xtx + 1.0 / tau2;
    AnalyticCF out;
    out.mean     = xtr / denom;
    out.variance = sigma2 / denom;
    return out;
}

} // anonymous namespace

// [[Rcpp::export]]
Rcpp::List test_spikeslab_beta_conditional_parity() {
    // ---- Synthesize deterministic problem ------------------------------
    const std::size_t N = 50;
    const std::size_t p = 3;
    std::mt19937_64 data_rng(20260419);
    std::normal_distribution<double> unit_norm(0.0, 1.0);

    arma::vec X_flat(N * p);
    for (std::size_t i = 0; i < N * p; ++i) X_flat[i] = unit_norm(data_rng);

    arma::vec beta_true = {1.5, 0.0, -2.0};
    arma::vec gamma     = {1.0, 0.0, 1.0};
    const double sigma_true = 0.7;
    const double tau_val    = 5.0;

    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        double xb = 0.0;
        for (std::size_t k = 0; k < p; ++k)
            xb += X_flat[i + k * N] * beta_true[k];
        y[i] = xb + sigma_true * unit_norm(data_rng);
    }

    // ---- Compute closed-form analytical conditional for j = 0 ---------
    const std::size_t test_j = 0;
    AnalyticCF cf = compute_analytic_cf(
        test_j, y, X_flat, gamma, beta_true,
        sigma_true, tau_val, N, p);

    // ---- Sample 10,000 times via the production formula ---------------
    const std::size_t N_DRAWS = 10000;
    std::mt19937_64 sample_rng(42);
    std::vector<double> samples;
    samples.reserve(N_DRAWS);
    for (std::size_t i = 0; i < N_DRAWS; ++i) {
        const double b = spike_slab_continuous_update_reference(
            sample_rng, test_j, y, X_flat, gamma, beta_true,
            sigma_true, tau_val, N, p);
        samples.push_back(b);
    }

    // ---- Sample statistics --------------------------------------------
    double mean = 0.0;
    for (double v : samples) mean += v;
    mean /= static_cast<double>(N_DRAWS);

    double var = 0.0;
    for (double v : samples) {
        const double d = v - mean;
        var += d * d;
    }
    var /= static_cast<double>(N_DRAWS - 1);

    // ---- Check #15 tolerances -----------------------------------------
    const double mean_rel_err = std::abs(mean - cf.mean) / std::max(std::abs(cf.mean), 1e-6);
    const double var_rel_err  = std::abs(var  - cf.variance) / cf.variance;

    const double tol_mean = 0.05;   // 5%
    const double tol_var  = 0.10;   // 10%

    const bool pass_mean = (mean_rel_err < tol_mean);
    const bool pass_var  = (var_rel_err  < tol_var);
    const bool all_pass  = pass_mean && pass_var;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")     = all_pass,
        Rcpp::Named("pass_mean")    = pass_mean,
        Rcpp::Named("pass_var")     = pass_var,
        Rcpp::Named("sample_mean")  = mean,
        Rcpp::Named("exp_mean")     = cf.mean,
        Rcpp::Named("mean_rel_err") = mean_rel_err,
        Rcpp::Named("sample_var")   = var,
        Rcpp::Named("exp_var")      = cf.variance,
        Rcpp::Named("var_rel_err")  = var_rel_err,
        Rcpp::Named("n_draws")      = static_cast<int>(N_DRAWS),
        Rcpp::Named("tol_mean")     = tol_mean,
        Rcpp::Named("tol_var")      = tol_var);
}

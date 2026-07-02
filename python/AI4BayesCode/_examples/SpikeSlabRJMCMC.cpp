// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  SpikeSlabRJMCMC.cpp -- Dirac spike-and-slab variable selection via
//                         rjmcmc_block v0, following AI4BayesCode's "NUTS by
//                         default, Gibbs only under documented exception"
//                         and "scale-invariant Jeffreys" disciplines.
//
//  THE CANONICAL reference template for Class 2b Dirac spike-and-slab.
//
//  Model — Ishwaran & Rao 2005 JASA "sigma-scaled slab" form
//  ----------------------------------------------------------
//      y_i           = X_i' beta + eps_i,   eps_i ~ N(0, sigma^2)   i = 1..N
//      gamma_j       ~ Bernoulli(pi)                                j = 1..p
//      beta_j | gamma_j = 0 = 0              (Dirac spike)
//      beta_j | gamma_j = 1, sigma, tau ~ N(0, sigma^2 * tau^2)
//              (Gaussian slab, SIGMA-SCALED — tau is dimensionless,
//               representing signal-to-noise ratio)
//
//  Why sigma^2 * tau^2 and not just tau^2?
//  ---------------------------------------
//  (i)  Scale-invariance: rescaling (y, X) -> (c*y, c*X) rescales sigma
//       but leaves tau unchanged. tau is therefore a data-scale-free
//       quantity measuring "signal strength in units of noise sd" — a
//       natural interpretable quantity.
//  (ii) Decorrelated posterior: sigma and tau are approximately
//       posterior-independent under this parametrization (their
//       conditional posteriors only weakly couple through sum_active
//       beta^2). Modular NUTS mixes dramatically faster because each
//       block sees a stable target instead of one that shifts as the
//       other updates.
//  (iii) Jeffreys p(tau) ∝ 1/tau becomes the natural noninformative
//       choice (tau is a scale parameter and Jeffreys is the
//       scale-invariant prior). See skills/codegen.md §2a.
//
//  Priors
//  ------
//      pi    ~ Beta(a_pi, b_pi)                 (user-configured)
//      sigma ~ Jeffreys:  p(sigma) ∝ 1/sigma    (scale-invariant)
//      tau   ~ Jeffreys:  p(tau)   ∝ 1/tau      (scale-invariant;
//                                                tau dimensionless)
//
//  Ishwaran & Rao 2005 JASA 100:1215-1225 Eq. (3.1)-(3.2) uses this
//  exact sigma-scaled slab form, cited as the canonical modern Dirac
//  spike-and-slab formulation.
//  Gelman 2006 Bayesian Analysis 1(3):515-533 justifies the Jeffreys
//  choice on sigma and tau (vs. the critiqued InverseGamma(eps, eps)).
//
//  Block decomposition (FOUR blocks)
//  ---------------------------------
//    1. beta_gibbs_block  : pi | gamma      (**Exception 3**: scalar
//                           textbook Beta-Bernoulli conjugate; library
//                           parity test Check #15)
//
//    2. nuts_block        : sigma (log-transformed, Jeffreys prior +
//                           likelihood incl. slab-prior contribution
//                           from active beta_j ~ N(0, sigma^2 tau^2))
//
//    3. nuts_block        : tau   (log-transformed, Jeffreys prior +
//                           slab likelihood contribution;
//                           dimensionless so no hyperparameter)
//
//    4. rjmcmc_block      : (gamma, beta) via identity-coord RJMCMC
//                           with hand-written Gibbs continuous_update
//                           for beta_j | gamma_j=1 (**Exception 2**:
//                           kernel contract; per-usage parity test
//                           Check #15).
//
//  Order: rjmcmc_block runs LAST so pi, sigma, tau read from ctx are
//  the freshest draws from the current sweep.
//
//  k=0 guard (Ishwaran-Rao 2005 §3.1 warm-start)
//  ----------------------------------------------
//  Under Jeffreys on tau with sigma-scaled slab, the conditional
//  posterior of tau at sum_active gamma = 0 is the improper prior
//  itself. To avoid the chain wandering in that state, gamma_init
//  is set by marginal-OLS screening: j_init = argmax_j |X_j' y|, and
//  gamma_init[j_init] = 1. This ensures k >= 1 from iteration 1.
//  If the chain subsequently visits k=0 (rare), NUTS takes a random
//  walk on log(tau) for that single sweep (Jeffreys + Jacobian = 0
//  on eta-scale) until a gamma_j=1 move is accepted.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("SpikeSlabRJMCMC")
//   set.seed(20260621); N <- 300L; p <- 8L; sigma_true <- 1.0
//   beta_true <- numeric(p); beta_true[c(1,4,7)] <- c(2.5, -1.8, 1.2)  # actives {1,4,7}
//   X <- matrix(rnorm(N * p), N, p); X <- scale(X, center = TRUE, scale = FALSE)  # center cols (no intercept)
//   y <- as.numeric(X %*% beta_true + rnorm(N, 0, sigma_true)); y <- y - mean(y)  # center y
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- AI4BayesCode_run_chains(
//       function(seed) new(SpikeSlabRJMCMC, X, y, 1.0, 1.0, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4b_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(SpikeSlabRJMCMC, X, y, 1.0, 1.0, 7L, TRUE)  # X (N x p), y, a_pi=1, b_pi=1 (Beta prior on pi), seed=7, keep_history=TRUE
//   m$step(2500); str(m$get_current())
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(20260621); N, p, sigma_true = 300, 8, 1.0
//   beta_true = np.zeros(p); beta_true[[0, 3, 6]] = [2.5, -1.8, 1.2]   # actives {1,4,7}
//   X = rng.standard_normal((N, p)); X = X - X.mean(axis=0)             # center cols (no intercept)
//   y = X @ beta_true + rng.normal(0.0, sigma_true, N); y = y - y.mean()  # center y
//   Mod = AI4BayesCode.source("SpikeSlabRJMCMC.cpp")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.SpikeSlabRJMCMC(X, y, 1.0, 1.0, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.SpikeSlabRJMCMC(X, y, 1.0, 1.0, 7, True)  # (X, y, a_pi=1, b_pi=1, seed=7, keep_history=True)
//   m.step(2500); print(m.get_current())              # gamma ~ {0,3,6} active, sigma ~ 1
// @example:end
// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#ifdef AI4BAYESCODE_RCPP_MODULE
#  include <RcppArmadillo.h>
#else
#  include <armadillo>
#endif

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/beta_gibbs_block.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/rjmcmc_block.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::beta_gibbs_block;
using AI4BayesCode::beta_gibbs_block_config;
using AI4BayesCode::beta_dist_params;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
using AI4BayesCode::rjmcmc_block;
using AI4BayesCode::rjmcmc_block_config;

namespace constraints = AI4BayesCode::constraints;

namespace {

// ===========================================================================
// NATURAL-SCALE log-density functions for NUTS blocks (sigma and tau).
// Jacobian (+log(param)) is added by constraints::positive::wrap — these
// functions MUST NOT include it.
//
// Convention: the Jeffreys prior p(param) ∝ 1/param appears as -log(param)
// in the natural-scale density and is exactly cancelled by the wrap
// Jacobian, so the eta-scale density the user sees is just the likelihood
// (plus the Gaussian normalization's own -N log(sigma) term, which is
// likelihood, not prior — see skills/codegen.md §2a).
// ===========================================================================

// p(sigma | y, X, beta, tau, gamma) — natural-scale, sigma-scaled slab.
//
// Pieces, in sigma:
//   (a) Gaussian likelihood: L(sigma; y, beta, X)
//       log L = -N log(sigma) - SSE / (2 sigma^2)
//   (b) Slab prior on active beta: beta_j | gamma_j=1, sigma, tau ~ N(0, sigma^2 tau^2)
//       log p(beta_active | sigma, tau) = -0.5 k log(2 pi sigma^2 tau^2)
//                                         - sum_active beta^2 / (2 sigma^2 tau^2)
//       sigma-dependent part: -k log(sigma) - sum_b2 / (2 sigma^2 tau^2)
//   (c) Jeffreys prior: log p(sigma) = -log(sigma)
//
// lp_natural(sigma) = -(N + k + 1) log(sigma)
//                     - SSE / (2 sigma^2)
//                     - sum_b2 / (2 sigma^2 tau^2)
//
// grad (d/d sigma)  = -(N + k + 1)/sigma
//                     + SSE / sigma^3
//                     + sum_b2 / (sigma^3 tau^2)
double sigma_natural_log_density(const arma::vec& sigma_nat,
                                 const block_context& ctx,
                                 arma::vec* grad_nat) {
    const double sigma = sigma_nat[0];
    const arma::vec& y      = ctx.at("y");
    const arma::vec& X_flat = ctx.at("X");
    const arma::vec& beta   = ctx.at("beta");
    const arma::vec& gamma  = ctx.at("gamma");
    const double     tau    = ctx.at("tau")[0];
    const std::size_t N = y.n_elem;
    const std::size_t p = beta.n_elem;

    // SSE = ||y - X*beta||^2 via armadillo / BLAS (skill §6.1).
    const arma::mat X(const_cast<double*>(X_flat.memptr()),
                      N, p, /*copy_aux_mem=*/false, /*strict=*/true);
    arma::vec res = y - X * beta;                                  // length N
    const double sse = arma::dot(res, res);

    // Active-set count and ||beta_active||^2: irregular index pattern
    // (mask by gamma); scalar loop is correct.
    std::size_t k = 0;
    double sum_b2 = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        if (gamma[j] > 0.5) {
            ++k;
            sum_b2 += beta[j] * beta[j];
        }
    }

    const double N_d    = static_cast<double>(N);
    const double k_d    = static_cast<double>(k);
    const double sig2   = sigma * sigma;
    const double sig3   = sig2 * sigma;
    const double tau2   = tau * tau;

    const double lp =
        -(N_d + k_d + 1.0) * std::log(sigma)
        - 0.5 * sse / sig2
        - 0.5 * sum_b2 / (sig2 * tau2);

    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] =
            -(N_d + k_d + 1.0) / sigma
            + sse / sig3
            + sum_b2 / (sig3 * tau2);
    }
    return lp;
}

// p(tau | beta, gamma, sigma) — natural-scale with Jeffreys prior.
//
// Pieces, in tau:
//   (a) Slab likelihood (sigma-scaled):
//       log p(beta_active | sigma, tau) tau-dependent part
//         = -k log(tau) - sum_b2 / (2 sigma^2 tau^2)
//   (b) Jeffreys prior: log p(tau) = -log(tau)
//
// lp_natural(tau) = -(k + 1) log(tau) - sum_b2 / (2 sigma^2 tau^2)
// grad (d/d tau)   = -(k + 1)/tau + sum_b2 / (sigma^2 tau^3)
//
// k = 0 guard
// ------------
// When sum_gamma = 0, tau is irrelevant (no active beta_j for the slab
// likelihood to apply to) and the posterior under Jeffreys would be
// improper. Fall back to a soft pin near the reference scale tau = 1
// via half-Normal(0, 1): log p(tau | k=0) = -0.5 * tau^2, grad = -tau.
// This is a transient state — once any gamma_j = 1 is proposed and
// accepted, the slab likelihood dominates at the next sweep and tau's
// posterior concentrates on the data-driven value. The reference scale
// tau = 1 is the natural choice under the sigma^2 tau^2 parametrization
// (tau=1 means signal scale = noise scale).
double tau_natural_log_density(const arma::vec& tau_nat,
                               const block_context& ctx,
                               arma::vec* grad_nat) {
    const double tau = tau_nat[0];
    const arma::vec& gamma  = ctx.at("gamma");
    const arma::vec& beta   = ctx.at("beta");
    const double     sigma  = ctx.at("sigma")[0];
    const std::size_t p = beta.n_elem;

    std::size_t k = 0;
    double sum_b2 = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        if (gamma[j] > 0.5) {
            ++k;
            sum_b2 += beta[j] * beta[j];
        }
    }

    // Degenerate-state guard: when there is no slab-likelihood info on tau
    // (either k=0 = no active signals, OR k>=1 but all active beta_j are
    // exactly zero, which happens when a user's set_current overrides
    // gamma without setting beta), the Jeffreys-only conditional is
    // improper. Fall back to half-Normal(0, 1) pin near the reference
    // scale tau = 1. This is transient: once the continuous_update hook
    // draws nonzero beta_j (which it will on the next rjmcmc sweep given
    // xtx > 0), the main density takes over.
    if (k == 0 || sum_b2 == 0.0) {
        if (grad_nat) {
            grad_nat->set_size(1);
            (*grad_nat)[0] = -tau;
        }
        return -0.5 * tau * tau;
    }

    const double k_d   = static_cast<double>(k);
    const double tau2  = tau * tau;
    const double tau3  = tau2 * tau;
    const double sig2  = sigma * sigma;

    const double lp =
        -(k_d + 1.0) * std::log(tau)
        - 0.5 * sum_b2 / (sig2 * tau2);

    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] =
            -(k_d + 1.0) / tau
            + sum_b2 / (sig2 * tau3);
    }
    return lp;
}

// ===========================================================================
// rjmcmc_block hooks — sigma-scaled slab (σ² τ² variance).
// ===========================================================================
double spike_slab_log_joint(const arma::vec& gamma,
                            const arma::vec& beta,
                            const block_context& ctx) {
    const arma::vec& y       = ctx.at("y");
    const arma::vec& X_flat  = ctx.at("X");
    const double     sigma   = ctx.at("sigma")[0];
    const double     tau     = ctx.at("tau")[0];
    const double     pi_val  = ctx.at("pi")[0];

    const std::size_t N = y.n_elem;
    const std::size_t p = beta.n_elem;

    if (!(sigma > 0.0) || !(tau > 0.0)) {
        return -std::numeric_limits<double>::infinity();
    }
    if (!(pi_val > 0.0 && pi_val < 1.0)) {
        return -std::numeric_limits<double>::infinity();
    }

    const double sigma2  = sigma * sigma;
    const double tau2    = tau   * tau;
    const double slabvar = sigma2 * tau2;    // <-- σ²τ²

    // SSE = ||y - X*beta||^2 via armadillo / BLAS (skill §6.1).
    const arma::mat X(const_cast<double*>(X_flat.memptr()),
                      N, p, /*copy_aux_mem=*/false, /*strict=*/true);
    arma::vec res = y - X * beta;                                  // length N
    const double sse = arma::dot(res, res);
    double lp = -0.5 * static_cast<double>(N)
                   * std::log(2.0 * M_PI * sigma2)
                - sse / (2.0 * sigma2);

    std::size_t k = 0;
    for (std::size_t j = 0; j < p; ++j) {
        if (gamma[j] > 0.5) {
            ++k;
            lp += -0.5 * std::log(2.0 * M_PI * slabvar)
                  - beta[j] * beta[j] / (2.0 * slabvar);
        }
    }
    lp +=   static_cast<double>(k)       * std::log(pi_val)
          + static_cast<double>(p - k)   * std::log(1.0 - pi_val);
    return lp;
}

// Check #17 whitelist: std::normal_distribution inside rjmcmc_block's
// `propose_sample` hook (one of the three whitelisted contexts from
// skills/codegen.md §2e). The proposal distribution is slab-prior-matched
// N(0, sigma^2 tau^2), part of the RJ kernel contract.
double spike_slab_propose_sample(std::mt19937_64& rng,
                                 std::size_t /*j*/,
                                 const block_context& ctx) {
    const double sigma = ctx.at("sigma")[0];
    const double tau   = ctx.at("tau")[0];
    std::normal_distribution<double> norm(0.0, sigma * tau);   // σ·τ standard dev
    return norm(rng);
}

double spike_slab_propose_logq(double beta_new,
                               std::size_t /*j*/,
                               const block_context& ctx) {
    const double sigma   = ctx.at("sigma")[0];
    const double tau     = ctx.at("tau")[0];
    const double slabvar = sigma * sigma * tau * tau;
    return -0.5 * std::log(2.0 * M_PI * slabvar)
           - beta_new * beta_new / (2.0 * slabvar);
}

// Hand-written conditional Gibbs for beta[j] | gamma[j]=1. Closed-form
// under sigma-scaled slab:
//   beta_j | rest ~ N(mu, v) with
//     prec = xtx / sigma^2 + 1 / (sigma^2 tau^2) = (xtx tau^2 + 1) / (sigma^2 tau^2)
//          = (xtx + 1/tau^2) / sigma^2
//     mu   = (xtr / sigma^2) / prec = xtr / (xtx + 1/tau^2)
//     v    = 1 / prec = sigma^2 / (xtx + 1/tau^2)
//     sd   = sigma / sqrt(xtx + 1/tau^2)
//
// JUSTIFICATION (Check #16): Exception 2 (codegen.md §2b) — rjmcmc_block's
// continuous_update hook requires a direct conditional iid draw to preserve
// detailed balance. A NUTS step would not be conditionally iid and would
// break the RJ chain.
// Per-usage parity test: tests_autodiff/test_spikeslab_beta_conditional_parity.cpp.
double spike_slab_continuous_update(std::mt19937_64& rng,
                                    std::size_t j,
                                    const block_context& ctx) {
    const arma::vec& y       = ctx.at("y");
    const arma::vec& X_flat  = ctx.at("X");
    const arma::vec& gamma   = ctx.at("gamma");
    const arma::vec& beta    = ctx.at("beta");
    const double     sigma   = ctx.at("sigma")[0];
    const double     tau     = ctx.at("tau")[0];
    const std::size_t N = y.n_elem;
    const std::size_t p = beta.n_elem;

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
    const double denom = xtx + 1.0 / tau2;
    const double mean  = xtr / denom;
    const double sd    = sigma / std::sqrt(denom);
    std::normal_distribution<double> norm(mean, sd);
    return norm(rng);
}

} // anonymous namespace

// ============================================================================
//  SpikeSlabRJMCMC -- user-facing wrapper class
// ============================================================================

class SpikeSlabRJMCMC {
public:
    SpikeSlabRJMCMC(const arma::mat& X,
                    const arma::vec& y,
                    double           a_pi_prior,
                    double           b_pi_prior,
                    int              rng_seed,
                    bool             keep_history = false)
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
          impl_(std::make_unique<composite_block>("SpikeSlabRJMCMC")),
          keep_history_(keep_history)
    {
        const std::size_t N = X.n_rows;
        const std::size_t p = X.n_cols;
        if (N == 0)       ai4b::stop("X must have at least 1 row");
        if (p == 0)       ai4b::stop("X must have at least 1 column");
        if (y.n_elem != N) ai4b::stop("y length must equal X row count");
        if (!(a_pi_prior > 0.0) || !(b_pi_prior > 0.0))
            ai4b::stop("a_pi / b_pi must be > 0");
        N_ = N; p_ = p;

        // Data in shared_data.
        impl_->data().set("y",    y);
        impl_->data().set("X",    arma::vectorise(X));
        impl_->data().set("N",    arma::vec{static_cast<double>(N)});
        impl_->data().set("p",    arma::vec{static_cast<double>(p)});
        impl_->data().set("a_pi", arma::vec{a_pi_prior});
        impl_->data().set("b_pi", arma::vec{b_pi_prior});

        // ---- Marginal-OLS screening for gamma_init guard --------------
        // k=0 is a pathological state under Jeffreys on tau (conditional
        // becomes improper). Initialize with the single most-correlated
        // predictor active (Ishwaran & Rao 2005 §3.1 warm-start).
        std::size_t j_screen = 0;
        double best_xty = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            double xty = 0.0;
            for (std::size_t i = 0; i < N; ++i)
                xty += X(i, j) * y[i];
            if (std::abs(xty) > best_xty) {
                best_xty = std::abs(xty);
                j_screen = j;
            }
        }

        arma::vec gamma_init(p, arma::fill::zeros);
        arma::vec beta_init (p, arma::fill::zeros);
        gamma_init[j_screen] = 1.0;
        // beta_init on the screened index: OLS estimate restricted to j_screen:
        // beta_j = (X_j'y) / (X_j'X_j).
        double xtx_screen = 0.0;
        for (std::size_t i = 0; i < N; ++i)
            xtx_screen += X(i, j_screen) * X(i, j_screen);
        if (xtx_screen > 0.0) {
            double xty_screen = 0.0;
            for (std::size_t i = 0; i < N; ++i)
                xty_screen += X(i, j_screen) * y[i];
            beta_init[j_screen] = xty_screen / xtx_screen;
        }

        const double ybar = arma::mean(y);
        double sample_var = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double d = y[i] - ybar;
            sample_var += d * d;
        }
        sample_var /= static_cast<double>(std::max<std::size_t>(N - 1, 1));
        const double sigma_init = std::max(std::sqrt(sample_var), 1e-3);
        // tau (dimensionless) init near 1: signal-to-noise scale = 1 at start.
        const double tau_init = 1.0;
        const double pi_init =
            a_pi_prior / (a_pi_prior + b_pi_prior);

        impl_->data().set("gamma", gamma_init);
        impl_->data().set("beta",  beta_init);
        impl_->data().set("sigma", arma::vec{sigma_init});
        impl_->data().set("tau",   arma::vec{tau_init});
        impl_->data().set("pi",    arma::vec{pi_init});

        // Dependency DAG.
        // Under sigma-scaled slab, sigma's conditional depends on tau, gamma,
        // beta (slab prior contributes to sigma's density). Similarly for tau.
        impl_->data().declare_dependencies(
            "pi",    {"gamma", "a_pi", "b_pi", "p"});
        impl_->data().declare_dependencies(
            "sigma", {"y", "X", "beta", "gamma", "tau"});
        impl_->data().declare_dependencies(
            "tau",   {"beta", "gamma", "sigma"});
        impl_->data().declare_dependencies(
            "gamma_beta_rj",
            {"y", "X", "sigma", "tau", "pi"});

        // Predict DAG + y_rep refresher.
        // X, beta, sigma all directly produce y_rep in the generative
        // model y_rep ~ N(X*beta, sigma^2). All three are declared as
        // predict-edge parents so the DAG visualization (plot_dag) shows
        // X → y_rep. X is also registered as a data_input so it can be
        // replaced via predict_at(list(X = X_new)).
        //
        // Pass-2 availability rule (shared_data.hpp): a parent is
        // available if (in changed) OR (has any value in shared_data),
        // i.e. data_inputs with their training default value count as
        // available. predict_at(list()) therefore samples y_rep at
        // training X; predict_at(list(X = X_new)) samples at the new X
        // (refresher reads scratch.X uniformly via d.get("X")).
        // ---- Full predict-DAG reconstruction (no collapsed eta). ------
        //   eta   = X * beta        (deterministic linear predictor)
        //   y_rep ~ N(eta, sigma^2) (stochastic; reads ONLY eta, sigma)
        // eta kept current during sampling via
        // declare_invalidates("gamma_beta_rj", {"eta"}) (the rjmcmc
        // block writes beta), so stateful predict_at(list()) (empty
        // changed-set, no Pass-1 recompute) reads a fresh eta.
        // Behaviour-preserving: same X*beta and same per-i normal draw
        // order ⇒ bit-identical y_rep under a fixed predict RNG.
        impl_->data().set("eta", arma::vec(N, arma::fill::zeros));
        impl_->data().register_refresher(
            "eta",
            [p](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& beta_cur = d.get("beta");
                const arma::vec& X_flat   = d.get("X");
                const std::size_t N_cur   = X_flat.n_elem / p;
                arma::vec eta(N_cur);
                for (std::size_t i = 0; i < N_cur; ++i) {
                    double xb = 0.0;
                    for (std::size_t j = 0; j < p; ++j)
                        xb += X_flat[i + j * N_cur] * beta_cur[j];
                    eta[i] = xb;
                }
                return eta;
            });
        impl_->data().declare_invalidates("gamma_beta_rj", {"eta"});

        // Predict DAG (full generative chain): X,beta -> eta ;
        // eta,sigma -> y_rep.
        impl_->data().declare_predict_edges("X",     {"eta"});
        impl_->data().declare_predict_edges("beta",  {"eta"});
        impl_->data().declare_predict_edges("eta",   {"y_rep"});
        impl_->data().declare_predict_edges("sigma", {"y_rep"});
        impl_->data().declare_data_input("X");

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). Ishwaran-Rao sigma-scaled slab:
        //      pi ~ Beta(a_pi, b_pi); gamma_j ~ Bernoulli(pi);
        //      beta_j | gamma_j, sigma, tau ~ N(0, sigma^2 tau^2).
        //      tau, sigma ~ Jeffreys (no hyperparam slot). Drawn faded
        //      by plot_dag.
        impl_->data().declare_context_edges("a_pi",  {"pi"});
        impl_->data().declare_context_edges("b_pi",  {"pi"});
        impl_->data().declare_context_edges("pi",    {"gamma"});
        impl_->data().declare_context_edges("gamma", {"beta"});
        impl_->data().declare_context_edges("sigma", {"beta"});
        impl_->data().declare_context_edges("tau",   {"beta"});
        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                // Reads ONLY its direct generative parents eta, sigma.
                // Same per-i normal draw order as the old collapsed
                // y_rep ⇒ bit-identical under a fixed predict RNG.
                const arma::vec& eta       = d.get("eta");
                const double     sigma_cur = d.get("sigma")[0];
                std::normal_distribution<double> norm(0.0, 1.0);
                arma::vec y_rep(eta.n_elem);
                for (std::size_t i = 0; i < eta.n_elem; ++i)
                    y_rep[i] = eta[i] + sigma_cur * norm(rng);
                return y_rep;
            });

        // ================================================================
        // Block 1: pi (beta_gibbs_block)
        // JUSTIFICATION (Check #16): Exception 3 from codegen.md §2b —
        // scalar textbook Beta-Bernoulli conjugate. NUTS on 1-D tight-
        // posterior scalar is warmup-dominated and wasteful.
        // Covered by library parity test
        // tests_autodiff/block_tests/test_beta_gibbs_block.cpp (Check #15).
        // ================================================================
        {
            beta_gibbs_block_config cfg;
            cfg.name          = "pi";
            cfg.initial_value = pi_init;
            cfg.params_fn     = [p](const block_context& ctx) -> beta_dist_params {
                const arma::vec& gamma = ctx.at("gamma");
                const double     a_pi  = ctx.at("a_pi")[0];
                const double     b_pi  = ctx.at("b_pi")[0];
                const double     sum_g = arma::sum(gamma);
                return beta_dist_params{
                    a_pi + sum_g,
                    b_pi + static_cast<double>(p) - sum_g};
            };
            impl_->add_child(std::make_unique<beta_gibbs_block>(std::move(cfg)));
        }

        // ================================================================
        // Block 2: sigma (nuts_block, positive-constrained, Jeffreys)
        // Natural-scale density in sigma_natural_log_density(). Accounts
        // for the sigma-dependent slab-prior contribution from active beta.
        // ================================================================
        {
            nuts_block_config cfg;
            cfg.name        = "sigma";
            cfg.initial_unc = arma::vec{std::log(sigma_init)};
            cfg.constrain   = constraints::positive::constrain;
            cfg.unconstrain = constraints::positive::unconstrain;
            cfg.log_density_grad =
                [](const arma::vec& theta_unc, const block_context& ctx,
                   arma::vec* grad) {
                    return constraints::positive::wrap(
                        theta_unc, grad,
                        [&](const arma::vec& sigma_nat,
                            arma::vec* grad_nat) {
                            return sigma_natural_log_density(
                                sigma_nat, ctx, grad_nat);
                        });
                };
            impl_->add_child(std::make_unique<nuts_block>(std::move(cfg)));
        }

        // ================================================================
        // Block 3: tau (nuts_block, positive-constrained, Jeffreys)
        // Dimensionless scale; natural-scale density in
        // tau_natural_log_density().
        // ================================================================
        {
            nuts_block_config cfg;
            cfg.name        = "tau";
            cfg.initial_unc = arma::vec{std::log(tau_init)};
            cfg.constrain   = constraints::positive::constrain;
            cfg.unconstrain = constraints::positive::unconstrain;
            cfg.log_density_grad =
                [](const arma::vec& theta_unc, const block_context& ctx,
                   arma::vec* grad) {
                    return constraints::positive::wrap(
                        theta_unc, grad,
                        [&](const arma::vec& tau_nat,
                            arma::vec* grad_nat) {
                            return tau_natural_log_density(
                                tau_nat, ctx, grad_nat);
                        });
                };
            impl_->add_child(std::make_unique<nuts_block>(std::move(cfg)));
        }

        // ================================================================
        // Block 4: rjmcmc_block for (gamma, beta), sigma-scaled slab.
        // continuous_update is a HAND-WRITTEN Gibbs step. JUSTIFICATION
        // (Check #16): Exception 2 — rjmcmc_block kernel contract mandates
        // direct conditional draw; NUTS would break detailed balance.
        // Covered by per-usage parity test
        // tests_autodiff/test_spikeslab_beta_conditional_parity.cpp
        // (Check #15).
        // ================================================================
        {
            rjmcmc_block_config cfg;
            cfg.name              = "gamma_beta_rj";
            cfg.gamma_key         = "gamma";
            cfg.beta_key          = "beta";
            cfg.p                 = p;
            cfg.rw_scale          = 0.0;
            cfg.log_joint         = &spike_slab_log_joint;
            cfg.propose_sample    = &spike_slab_propose_sample;
            cfg.propose_logq      = &spike_slab_propose_logq;
            cfg.continuous_update = &spike_slab_continuous_update;
            cfg.gamma_init        = gamma_init;
            cfg.beta_init         = beta_init;
            impl_->add_child(std::make_unique<rjmcmc_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    // ---- Canonical 6-method R interface ----

    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        // Each value is an arma::vec; scalars are length-1 (the frontend
        // converts state_map -> R named list / Python dict; length-1 vectors
        // surface as 1-element numeric vectors / numpy arrays).
        AI4BayesCode::state_map out;
        out["gamma"] = impl_->data().get("gamma");   // length p
        out["beta"]  = impl_->data().get("beta");    // length p
        out["sigma"] = impl_->data().get("sigma");   // length 1
        out["tau"]   = impl_->data().get("tau");     // length 1
        out["pi"]    = impl_->data().get("pi");      // length 1
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        // Backend-neutral I/O: every value is an arma::vec (frontend already
        // converted R list / Python dict). X arrives as a VECTORISED N*p
        // arma::vec, column-major (element (i,j) at index i + j*N), matching
        // how X is stored in data() (arma::vectorise of the N x p design)
        // and how predict_at receives it.
        auto* pi_blk = dynamic_cast<beta_gibbs_block*>(&impl_->child(0));
        auto* sg_blk = dynamic_cast<nuts_block*>(&impl_->child(1));
        auto* tu_blk = dynamic_cast<nuts_block*>(&impl_->child(2));
        auto* rj_blk = dynamic_cast<rjmcmc_block*>(&impl_->child(3));

        auto it_g = params.find("gamma");
        auto it_b = params.find("beta");
        const bool has_g = (it_g != params.end());
        const bool has_b = (it_b != params.end());
        if (has_g || has_b) {
            arma::vec g_new = impl_->data().get("gamma");
            arma::vec b_new = impl_->data().get("beta");
            if (has_g) {
                const arma::vec& gv = it_g->second;
                if (static_cast<std::size_t>(gv.n_elem) != p_)
                    ai4b::stop("set_current: gamma length mismatch");
                for (std::size_t j = 0; j < p_; ++j)
                    g_new[j] = (gv[j] >= 0.5) ? 1.0 : 0.0;
            }
            if (has_b) {
                const arma::vec& bv = it_b->second;
                if (static_cast<std::size_t>(bv.n_elem) != p_)
                    ai4b::stop("set_current: beta length mismatch");
                for (std::size_t j = 0; j < p_; ++j)
                    b_new[j] = bv[j];
            }
            for (std::size_t j = 0; j < p_; ++j)
                if (g_new[j] == 0.0) b_new[j] = 0.0;
            arma::vec cat(2 * p_);
            for (std::size_t j = 0; j < p_; ++j) {
                cat[j]      = g_new[j];
                cat[p_ + j] = b_new[j];
            }
            rj_blk->set_current(cat);
            impl_->data().set("gamma", g_new);
            impl_->data().set("beta",  b_new);
        }
        auto it_sigma = params.find("sigma");
        if (it_sigma != params.end()) {
            const double sg = it_sigma->second[0];
            if (!(sg > 0.0)) ai4b::stop("sigma must be > 0");
            sg_blk->set_current(arma::vec{sg});
            impl_->data().set("sigma", arma::vec{sg});
        }
        auto it_tau = params.find("tau");
        if (it_tau != params.end()) {
            const double tu = it_tau->second[0];
            if (!(tu > 0.0)) ai4b::stop("tau must be > 0");
            tu_blk->set_current(arma::vec{tu});
            impl_->data().set("tau", arma::vec{tu});
        }
        auto it_pi = params.find("pi");
        if (it_pi != params.end()) {
            const double pv = it_pi->second[0];
            if (!(pv > 0.0 && pv < 1.0))
                ai4b::stop("pi must be in (0, 1)");
            pi_blk->set_current(arma::vec{pv});
            impl_->data().set("pi", arma::vec{pv});
        }
        // ---- X / y branches: canonical dynamic-N pattern
        // (codegen_cpp.md §7a; system_design.md §7 rules 4 + 6).
        // p is strict (gamma/beta length); N is dynamic and updated
        // atomically with X. y_rep refresher reads N from X.n_elem/p.
        // X is a vectorised N*p column-major arma::vec; p_ is fixed, so
        // N_new = x_flat.n_elem / p_.
        auto it_X = params.find("X");
        auto it_y = params.find("y");
        const bool has_X_in = (it_X != params.end());
        const bool has_y_in = (it_y != params.end());
        if (has_X_in) {
            const arma::vec& x_flat = it_X->second;
            if (p_ == 0 || x_flat.n_elem % p_ != 0)
                ai4b::stop("set_current: X must be a vectorised N*p "
                           "(column-major) arma::vec; p is fixed by "
                           "internal block state at p = %zu. Reconstruct "
                           "to change p.", p_);
            const std::size_t N_new = x_flat.n_elem / p_;
            if (has_y_in) {
                const arma::vec& y_chk = it_y->second;
                if (static_cast<std::size_t>(y_chk.n_elem) != N_new)
                    ai4b::stop("set_current: X has %zu rows but y has "
                               "length %zu.", N_new,
                               (std::size_t)y_chk.n_elem);
            }
            impl_->data().set("X", x_flat);
            if (N_new != N_) {
                impl_->data().set("y_rep",
                                  arma::vec(N_new, arma::fill::zeros));
                if (keep_history_ && impl_->history_size() > 1) {
                    // Non-fatal warning. backend_neutral.hpp's installed R
                    // copy only guarantees ai4b::stop (not ai4b::warning),
                    // so guard the warning by backend: Rcpp::warning under R
                    // (identical to the original behavior), stderr under
                    // pybind/standalone.
#ifdef AI4BAYESCODE_RCPP_MODULE
                    Rcpp::warning("set_current: X row count changed from "
                                  "%zu to %zu; clearing history (mixed-N "
                                  "history is unsupported).", N_, N_new);
#else
                    std::fprintf(stderr,
                                 "warning: set_current: X row count changed "
                                 "from %zu to %zu; clearing history (mixed-N "
                                 "history is unsupported).\n", N_, N_new);
#endif
                    impl_->clear_history();
                }
            }
            N_ = N_new;
        }
        if (has_y_in) {
            const arma::vec& y_new = it_y->second;
            if (y_new.n_elem != N_)
                ai4b::stop("set_current: y length %zu != current N = %zu",
                           (std::size_t)y_new.n_elem, N_);
            impl_->data().set("y", y_new);
        }
    }

    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        // Backend-neutral I/O. Supports:
        //   predict_at({})              -> posterior-predictive y_rep
        //   predict_at({"X" = X_new})   -> y_rep at new covariates
        //
        //   * INPUT  : new_data["X"] is a VECTORISED N_new*p arma::vec,
        //              column-major (element (i,j) at index i + j*N_new),
        //              matching how X is stored in data() (arma::vectorise of
        //              the N x p design). R/Python callers pass
        //              as.vector(X_new) / X_new flattened column-major.
        //   * OUTPUT : every key is an arma::mat. keep_history = FALSE returns
        //              1-row matrices (single predict at current draw);
        //              keep_history = TRUE returns n_draws-row matrices
        //              (posterior predictive over all draws).

        // ---- Parse X input once (shared across both modes) ----------------
        bool has_X = !new_data.empty();
        arma::vec x_flat;
        if (has_X) {
            for (const auto& kv : new_data) {
                if (kv.first != "X")
                    ai4b::stop("SpikeSlabRJMCMC::predict_at: unknown key '%s'. "
                               "Valid keys: 'X' (or empty map/list).",
                               kv.first.c_str());
            }
            auto it_X = new_data.find("X");
            x_flat = it_X->second;
            if (p_ == 0 || x_flat.n_elem % p_ != 0) {
                ai4b::stop("SpikeSlabRJMCMC::predict_at: X must be a vectorised "
                           "N_new*p (column-major) arma::vec; training X has "
                           "p = %zu columns.", p_);
            }
        }

        AI4BayesCode::history_map out;

        if (!keep_history_) {
            // ---- Stateful mode: single predict at current draw ------------
            block_context replaced;
            if (has_X) replaced["X"] = x_flat;
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result) {
                arma::mat m(1, kv.second.n_elem);
                for (std::size_t j = 0; j < kv.second.n_elem; ++j)
                    m(0, j) = kv.second[j];
                out.emplace(kv.first, std::move(m));
            }
            return out;
        }

        // ---- History mode: loop over all posterior draws ------------------
        // We bypass composite_block::predict_at here because `beta` is a
        // sub-output of the rjmcmc block (block name "gamma_beta_rj", which
        // writes both `gamma` and `beta` to shared_data). composite_block's
        // input-validation only accepts (data_input keys ∪ block names) as
        // valid `replaced` keys, so `replaced["beta"]` is rejected. The
        // semantically correct pattern (mirrors ARDLasso) is to read the
        // sampled history directly and compute y_rep manually per draw.
        AI4BayesCode::history_map hist = impl_->get_history();
        if (!hist.count("beta") || !hist.count("sigma")) {
            ai4b::stop("SpikeSlabRJMCMC::predict_at: keep_history_ requires "
                       "beta and sigma history, but get_history() lacks them. "
                       "Did you forget to construct with keep_history = TRUE?");
        }
        const arma::mat& beta_hist  = hist.at("beta");   // n_draws × p
        const arma::mat& sigma_hist = hist.at("sigma");  // n_draws × 1
        const std::size_t n_draws   = beta_hist.n_rows;
        if (sigma_hist.n_rows != n_draws) {
            ai4b::stop("SpikeSlabRJMCMC::predict_at: inconsistent history "
                       "sizes (beta n_draws=%zu, sigma n_draws=%zu)",
                       (std::size_t)beta_hist.n_rows,
                       (std::size_t)sigma_hist.n_rows);
        }

        // Pick X for this prediction round (training X by default, new X if
        // user supplied one). shared_data stores X as a column-major flat
        // arma::vec of length N*p.
        const arma::vec& X_use = has_X ? x_flat : impl_->data().get("X");
        const std::size_t N_pred = X_use.n_elem / p_;

        // y_rep matrix: n_draws × N_pred. For each draw d:
        //   y_rep[d, i] = sum_j X[i + j*N_pred] * beta_d[j] + sigma_d * N(0,1)
        arma::mat yrep_mat(n_draws, N_pred);
        std::normal_distribution<double> norm01(0.0, 1.0);
        for (std::size_t d = 0; d < n_draws; ++d) {
            const double sigma_d = sigma_hist(d, 0);
            for (std::size_t i = 0; i < N_pred; ++i) {
                double xb = 0.0;
                for (std::size_t j = 0; j < p_; ++j) {
                    xb += X_use[i + j * N_pred] * beta_hist(d, j);
                }
                yrep_mat(d, i) = xb + sigma_d * norm01(predict_rng_);
            }
        }

        out.emplace("y_rep", std::move(yrep_mat));
        return out;
    }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    /// 7th R-level method: re-tune NUTS metric (mass matrix + step size +
    /// dual averaging) without advancing chain state. Available because
    /// the composite contains NUTS-family children. See system_design.md
    /// §13 NUTS-family + validator.md §24.
    void readapt_NUTS(int n, bool reset = false) {
        if (n < 0) {
            ai4b::stop("readapt_NUTS: n must be non-negative");
        }
        impl_->readapt_NUTS(static_cast<std::size_t>(n),
                            reset, readapt_rng_);
    }


private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_; // readapt_NUTS() advances it (7th method)
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      N_ = 0;
    std::size_t                      p_ = 0;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(SpikeSlabRJMCMC_module) {
    Rcpp::class_<SpikeSlabRJMCMC>("SpikeSlabRJMCMC")
        .constructor<arma::mat, arma::vec,
                     double, double,
                     int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::mat, arma::vec,
                     double, double,
                     int, bool>(
            "Construct with: X (N x p, centered), y (length N, centered), "
            "a_pi_prior, b_pi_prior (Beta prior for inclusion probability "
            "pi), rng_seed, keep_history. "
            "Priors on scale parameters are Jeffreys "
            "(p(sigma) ∝ 1/sigma, p(tau) ∝ 1/tau); no hyperparameters "
            "needed. Slab variance is sigma^2 * tau^2 (Ishwaran-Rao 2005 "
            "form), making tau dimensionless and the posterior scale-"
            "invariant. gamma_init uses marginal-OLS screening to ensure "
            "k >= 1 active signal at iteration 1, avoiding the improper "
            "conditional on tau at k = 0.")
        .method("step",        &SpikeSlabRJMCMC::step)
        .method("get_current", &SpikeSlabRJMCMC::get_current)
        .method("set_current", &SpikeSlabRJMCMC::set_current)
        .method("predict_at",  &SpikeSlabRJMCMC::predict_at)
        .method("get_dag",     &SpikeSlabRJMCMC::get_dag)
        .method("get_history", &SpikeSlabRJMCMC::get_history)
        .method("readapt_NUTS", &SpikeSlabRJMCMC::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(SpikeSlabRJMCMC, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<SpikeSlabRJMCMC>(m, "SpikeSlabRJMCMC")
        .def(pybind11::init<arma::mat, arma::vec, double, double, int, bool>(),
             pybind11::arg("X"),
             pybind11::arg("y"),
             pybind11::arg("a_pi_prior") = 1.0,
             pybind11::arg("b_pi_prior") = 1.0,
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step",         &SpikeSlabRJMCMC::step,  pybind11::arg("n_steps"))
        .def("get_current",  &SpikeSlabRJMCMC::get_current)
        .def("set_current",  &SpikeSlabRJMCMC::set_current, pybind11::arg("params"))
        .def("predict_at",   &SpikeSlabRJMCMC::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &SpikeSlabRJMCMC::get_dag)
        .def("get_history",  &SpikeSlabRJMCMC::get_history)
        .def("readapt_NUTS", &SpikeSlabRJMCMC::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false);
}
#endif

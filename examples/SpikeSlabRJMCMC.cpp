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

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that simulates spike-and-slab
// data, drives the composite block directly, and checks variable-selection /
// coefficient recovery. No R / Python binding is built or required.

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
#include "AI4BayesCode/beta_gibbs_block.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rjmcmc_block.hpp"

#include <cmath>
#include <cstdint>
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
//  SpikeSlabRJMCMC -- frontend-independent driver class.
//
//  Same model wiring as the original Rcpp-bound class (priors, log-densities,
//  block configs, DAG, hyperparameters preserved EXACTLY); only the frontend
//  binding (Rcpp::List get/set/predict, RCPP_MODULE) has been removed. step()
//  drives the composite; get_current() returns a neutral AI4BayesCode::state_map.
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
          readapt_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0xBF58476D1CE4E5B9ULL}),
          impl_(std::make_unique<composite_block>("SpikeSlabRJMCMC")),
          keep_history_(keep_history)
    {
        const std::size_t N = X.n_rows;
        const std::size_t p = X.n_cols;
        if (N == 0)       throw std::runtime_error("X must have at least 1 row");
        if (p == 0)       throw std::runtime_error("X must have at least 1 column");
        if (y.n_elem != N) throw std::runtime_error("y length must equal X row count");
        if (!(a_pi_prior > 0.0) || !(b_pi_prior > 0.0))
            throw std::runtime_error("a_pi / b_pi must be > 0");
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
        // predict-edge parents so the DAG visualization (ai4bayescode_plot_dag) shows
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
        //      by ai4bayescode_plot_dag.
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


    // ---- Neutral-typed driver interface (no Rcpp) ----

    void step(int n_steps) {
        if (n_steps < 0) throw std::runtime_error("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Returns the current draw as a neutral AI4BayesCode::state_map.
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["gamma"] = impl_->data().get("gamma");
        out["beta"]  = impl_->data().get("beta");
        out["sigma"] = impl_->data().get("sigma");
        out["tau"]   = impl_->data().get("tau");
        out["pi"]    = impl_->data().get("pi");
        return out;
    }

    AI4BayesCode::dag_info    get_dag()     const { return impl_->get_dag(); }
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    // Re-tune NUTS metric (mass matrix + step size + dual averaging) without
    // advancing chain state. Available because the composite contains
    // NUTS-family children.
    void readapt_NUTS(int n, bool reset = false) {
        if (n < 0) throw std::runtime_error("readapt_NUTS: n must be non-negative");
        impl_->readapt_NUTS(static_cast<std::size_t>(n), reset, readapt_rng_);
    }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      N_ = 0;
    std::size_t                      p_ = 0;
};

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulates a sparse linear-regression spike-and-slab problem from KNOWN
//  truth (only a few predictors active), fits via the four-block sampler
//  (pi | gamma; sigma; tau; (gamma, beta) RJMCMC), then checks:
//    (1) variable selection: posterior inclusion prob (PIP) high on the true
//        active set, low on the true null set;
//    (2) coefficient recovery: posterior-mean beta close to truth on actives;
//    (3) sigma recovery.
//  PASS is derived from these computed comparisons (never hard-coded).
//==============================================================================
#include <cstdio>
int main() {
    // -------- Known truth: p predictors, only a few active --------------
    const std::size_t N = 300;
    const std::size_t p = 8;
    const double      sigma_true = 1.0;

    // True coefficients: actives {0, 3, 6}, the rest exactly zero.
    arma::vec beta_true(p, arma::fill::zeros);
    beta_true[0] =  2.5;
    beta_true[3] = -1.8;
    beta_true[6] =  1.2;
    std::vector<bool> active_true(p, false);
    active_true[0] = active_true[3] = active_true[6] = true;
    std::size_t k_true = 3;

    // -------- Simulate standardized design + Gaussian noise -------------
    std::mt19937_64 sim_rng(20260621u);
    std::normal_distribution<double> gen_x(0.0, 1.0);
    std::normal_distribution<double> gen_e(0.0, sigma_true);

    arma::mat X(N, p);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < p; ++j)
            X(i, j) = gen_x(sim_rng);
    // Center each column (model expects centered X; spike-slab has no
    // intercept).
    for (std::size_t j = 0; j < p; ++j)
        X.col(j) -= arma::mean(X.col(j));

    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        double xb = 0.0;
        for (std::size_t j = 0; j < p; ++j) xb += X(i, j) * beta_true[j];
        y[i] = xb + gen_e(sim_rng);
    }
    y -= arma::mean(y);   // center y as well

    // -------- Fit: Beta(1, 1) prior on inclusion probability ------------
    SpikeSlabRJMCMC model(X, y, /*a_pi=*/1.0, /*b_pi=*/1.0,
                          /*rng_seed=*/7, /*keep_history=*/false);

    model.step(1500);   // warmup

    const int    M = 4000;
    arma::vec    pip(p, arma::fill::zeros);   // posterior inclusion prob
    arma::vec    beta_sum(p, arma::fill::zeros);
    double       sigma_sum = 0.0;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        const arma::vec& g = cur.at("gamma");
        const arma::vec& b = cur.at("beta");
        for (std::size_t j = 0; j < p; ++j) {
            if (g[j] > 0.5) pip[j] += 1.0;
            beta_sum[j] += b[j];
        }
        sigma_sum += cur.at("sigma")[0];
    }
    pip      /= static_cast<double>(M);
    beta_sum /= static_cast<double>(M);
    const double sigma_hat = sigma_sum / static_cast<double>(M);

    // -------- Report --------------------------------------------------
    std::printf("SpikeSlabRJMCMC demo  (N=%zu, p=%zu, true actives={0,3,6})\n",
                N, p);
    std::printf("  j  truth_active  beta_true  PIP     beta_postmean\n");
    for (std::size_t j = 0; j < p; ++j) {
        std::printf("  %zu      %d        %+6.2f    %.3f   %+7.3f\n",
                    j, active_true[j] ? 1 : 0, beta_true[j],
                    pip[j], beta_sum[j]);
    }
    std::printf("  sigma_hat=%.3f (truth %.2f)\n", sigma_hat, sigma_true);

    // -------- Derive PASS from computed comparisons -------------------
    // (a) every TRUE active selected with high PIP;
    // (b) every TRUE null mostly excluded (low PIP);
    // (c) coefficient recovery on actives within tolerance;
    // (d) sigma recovered.
    bool sel_ok = true;
    for (std::size_t j = 0; j < p; ++j) {
        if (active_true[j] && pip[j] < 0.80) sel_ok = false;   // miss
        if (!active_true[j] && pip[j] > 0.50) sel_ok = false;  // false sel
    }
    double max_beta_err = 0.0;
    for (std::size_t j = 0; j < p; ++j)
        if (active_true[j])
            max_beta_err = std::max(max_beta_err,
                                    std::abs(beta_sum[j] - beta_true[j]));
    const bool beta_ok  = max_beta_err < 0.35;
    const bool sigma_ok = std::abs(sigma_hat - sigma_true) < 0.25;

    // Compare against a NAIVE baseline (full OLS, no selection): does the
    // spike-and-slab actually zero-out the true nulls that OLS would not?
    arma::vec beta_ols = arma::solve(X.t() * X, X.t() * y);
    double naive_null_mass = 0.0, ss_null_mass = 0.0;
    for (std::size_t j = 0; j < p; ++j)
        if (!active_true[j]) {
            naive_null_mass += std::abs(beta_ols[j]);
            ss_null_mass    += std::abs(beta_sum[j]);
        }
    std::printf("  null-set |coef| mass: OLS=%.3f  spike-slab=%.3f "
                "(spike-slab should be smaller)\n",
                naive_null_mass, ss_null_mass);
    const bool beats_naive = ss_null_mass < naive_null_mass;

    const bool ok = sel_ok && beta_ok && sigma_ok && beats_naive;
    std::printf("  selection_ok=%d  beta_ok=%d (max_err=%.3f)  "
                "sigma_ok=%d  beats_naive=%d\n",
                (int)sel_ok, (int)beta_ok, max_beta_err,
                (int)sigma_ok, (int)beats_naive);
    std::printf("%s\n", ok ? "[demo PASS] spike-and-slab RJMCMC recovers "
                             "the sparse support + coefficients"
                           : "[demo FAIL]");
    (void)k_true;
    return ok ? 0 : 1;
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  HSGPRegression.cpp
//
//  REFERENCE TEMPLATE for 1-D Hilbert-space Gaussian Process (HSGP)
//  regression. This is the canonical reduced-rank GP basis approximation
//  (Solin & Sarkka 2020; Riutort-Mayol, Burkner, Andersen, Solin, Vehtari
//  2023) used by brms and Stan generated code. THIS IS THE PATTERN ANY
//  HSGP-FLAVORED MODEL SHOULD FOLLOW.
//
//  Why an example
//  --------------
//  A naive HSGP has THREE entangled hyperparameters (amp, ell, zs[M]) +
//  Intercept that NUTS cannot navigate with the diagonal mass matrix:
//  (1) (amp, ell) form a banana ridge through the spectral-density
//      formula; (2) (amp, zs) form a Neal funnel because f scales as
//      amp * zs; (3) (Intercept, sum(zs * phi)) is a marginal ridge.
//  Any agent generating an HSGP cpp without these three guards will get
//  rhat = 2+ and ESS = 1-3 on amp/ell — exactly what we observed in the
//  GP-class test set under sim3 ablation.
//
//  Architectural choices DEMONSTRATED here (must transfer)
//  -------------------------------------------------------
//   1. Reparameterize EVERY positive scalar (amp, ell, sigma) to the log
//      scale and treat them as REAL parameters. This keeps the whole
//      joint vector REAL so the dense-metric (Welford) pilot phase of
//      `joint_nuts_block` applies; the identity metric alone cannot
//      navigate the banana / funnel geometry.
//   2. Put ALL real parameters into ONE `joint_nuts_block` and set
//      `cfg.use_dense_metric = true`. The Welford pilot covariance learns
//      the (amp, ell) banana, the (amp, zs) funnel, and the (Intercept,
//      f-row-sum) ridge in a single 700-iteration warmup phase.
//   3. Manually add log|Jacobian| for each log-transformed scalar inside
//      the log-density (the block does NOT add log|J| for real-valued
//      sub-params; the user is responsible).
//
//  Model
//  -----
//      y_n            ~ Normal(Intercept + f(x_n), sigma)        n=1..N
//      f(x)           = sum_m sqrt(spd_m(amp, ell)) * z_m * phi_m(x)
//      spd_m(amp,ell) = amp^2 * sqrt(2 pi) * ell * exp(-0.5 ell^2 lambda_m)
//      lambda_m       = (m * pi / (2 L))^2          m = 1..M
//      phi_m(x)       = sqrt(1/L) * sin(sqrt(lambda_m) * (x + L))
//
//      L      = 1.5 * max(|x|)        domain-half-width with edge margin
//      M      = number of basis functions (passed in; typically 20-50)
//
//      Intercept ~ Normal(0, 10)
//      amp       ~ Half-Normal(0, sd(y))            (weakly informative)
//      ell       ~ Inverse-Gamma(5, x_range / 4)    (weakly informative)
//      sigma     ~ Jeffreys: p(sigma) oc 1/sigma      (improper)
//      z_m       ~ Normal(0, 1)                     non-centered
//
//  Block decomposition
//  -------------------
//   ONE block, all parameters REAL on the unconstrained scale, dense
//   metric. Sub-param order:
//      (Intercept, log_amp, log_ell, log_sigma, z[M])
//   total dim = 4 + M.
//
//  Synthetic test fixture (1-D, generic)
//  -------------------------------------
//   x in [-3, 3]; y = 1.5*sin(2*x) + 0.3*x + N(0, 0.4); N=80, M=25.
//   This is intentionally GENERIC -- not accelerometer / time-series /
//   any specific applied domain -- to teach the HSGP pattern without
//   leaking the solution to applied test cases.
// ============================================================================

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that simulates 1-D data from a
// known smooth function, fits the HSGP block, and checks that the posterior
// predictive mean recovers the truth better than a naive intercept baseline.
// No R / Python binding is built or required.

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;

// ============================================================================
// HSGP basis precomputation (cached at construction time)
// ============================================================================

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kSqrt2Pi = 2.5066282746310002;

// Joint log-density on the UNCONSTRAINED REAL scale.
//
// theta_cat layout: (Intercept, log_amp, log_ell, log_sigma, z[M])
// All sub_params are joint_constraint-free (REAL). Manual log|J|:
//   log_amp:    Half-Normal prior on amp; transform p(log_amp) = p(amp) * amp
//               -> +log(amp) Jacobian.
//   log_ell:    InvGamma prior on ell; +log(ell) Jacobian.
//   log_sigma:  Jeffreys p(sigma) oc 1/sigma -> p(log_sigma) const,
//               improper uniform on log scale, NO Jacobian (cancels).
double joint_log_density(const arma::vec& theta_cat,
                         const block_context& ctx,
                         arma::vec* grad) {
    const arma::vec& y           = ctx.at("y");
    const arma::vec& phi_flat    = ctx.at("phi_flat");      // (N*M) col-major
    const arma::vec& lambda      = ctx.at("lambda");        // (M)
    const double     amp_prior_sd = ctx.at("amp_prior_sd")[0];
    const double     ell_invG_a   = ctx.at("ell_invG_a")[0];
    const double     ell_invG_b   = ctx.at("ell_invG_b")[0];

    const std::size_t N = y.n_elem;
    const std::size_t M = lambda.n_elem;

    if (theta_cat.n_elem != 4 + M) {
        return -std::numeric_limits<double>::infinity();
    }

    const double Intercept = theta_cat[0];
    const double log_amp   = theta_cat[1];
    const double log_ell   = theta_cat[2];
    const double log_sigma = theta_cat[3];
    const double amp       = std::exp(log_amp);
    const double ell       = std::exp(log_ell);
    const double sigma     = std::exp(log_sigma);
    const arma::vec z      = theta_cat.subvec(4, 4 + M - 1);  // (M)

    // Spectral density per basis: spd_m = amp^2 * sqrt(2*pi) * ell *
    // exp(-0.5 * ell^2 * lambda_m). Square root used for the additive form.
    arma::vec sqrt_spd(M);
    for (std::size_t m = 0; m < M; ++m) {
        const double sd_m = amp * amp * kSqrt2Pi * ell
                            * std::exp(-0.5 * ell * ell * lambda[m]);
        sqrt_spd[m] = std::sqrt(sd_m);  // amp * (...)^(1/2)
    }

    // f_n = sum_m sqrt_spd_m * z_m * phi_{n,m}.
    // phi is col-major (N rows, M cols) flattened: phi[n + m*N].
    arma::vec coeffs = sqrt_spd % z;  // length-M
    arma::vec f(N, arma::fill::zeros);
    for (std::size_t m = 0; m < M; ++m) {
        const double c = coeffs[m];
        for (std::size_t n = 0; n < N; ++n) {
            f[n] += c * phi_flat[n + m * N];
        }
    }

    // -- Gaussian log-likelihood ---------------------------------------------
    arma::vec resid = y - Intercept - f;
    const double sigma2 = sigma * sigma;
    double lp = -0.5 * static_cast<double>(N) * std::log(kTwoPi)
                - static_cast<double>(N) * log_sigma
                - 0.5 * arma::dot(resid, resid) / sigma2;

    // -- Priors --------------------------------------------------------------
    // Intercept ~ Normal(0, 10)
    constexpr double intercept_var = 100.0;
    lp -= 0.5 * Intercept * Intercept / intercept_var;

    // amp ~ Half-Normal(0, amp_prior_sd) + log|J| from log_amp
    const double amp_var = amp_prior_sd * amp_prior_sd;
    lp += -0.5 * amp * amp / amp_var + log_amp;

    // ell ~ InvGamma(a, b): p(ell) ∝ ell^(-a-1) * exp(-b/ell). With log|J|:
    // log p(log_ell) = -a*log_ell - b/ell + const.
    lp += -ell_invG_a * log_ell - ell_invG_b / ell;

    // sigma ~ Jeffreys: p(log_sigma) const (improper); skip.

    // z_m ~ Normal(0, 1)
    lp -= 0.5 * arma::dot(z, z);

    if (!std::isfinite(lp)) {
        return -std::numeric_limits<double>::infinity();
    }

    if (grad) {
        grad->set_size(4 + M);
        // dlp / dIntercept:
        const double sum_resid_over_sigma2 = arma::sum(resid) / sigma2;
        (*grad)[0] = sum_resid_over_sigma2 - Intercept / intercept_var;

        // d f / d log_amp = f (because sqrt_spd_m oc amp -> d log sqrt_spd_m
        // / d log_amp = 1). resid = y - Intercept - f; d resid / d log_amp
        // = -d f / d log_amp = -f. Likelihood contribution to gradient:
        //   d log_lik / d log_amp = -sum(resid * (-f)) / sigma^2
        //                         = sum(resid * f) / sigma^2.
        // Wait: d/d log_amp of -0.5 sum(resid^2)/sigma^2
        //     = -sum(resid * d resid / d log_amp)/sigma^2
        //     = -sum(resid * (-f))/sigma^2
        //     = sum(resid * f)/sigma^2  -- but with one neg from chain rule
        // CHECK: d (resid^2)/d log_amp = 2*resid*d resid/d log_amp
        //                              = 2*resid*(-f).
        // So d (-0.5 sum)/d log_amp = -0.5 * sum(2*resid*(-f))/sigma^2
        //                           = sum(resid * f)/sigma^2.
        // Hmm that's positive. Let me redo: y_pred = Intercept + f.
        // resid = y - y_pred. d resid / d log_amp = -d y_pred / d log_amp
        // = -d f / d log_amp = -f. log_lik = -0.5 sum(resid^2)/sigma^2.
        // d log_lik / d log_amp = (-1/sigma^2) * sum(resid * d resid /
        // d log_amp) = (-1/sigma^2) * sum(resid * (-f))
        // = sum(resid * f) / sigma^2.  POSITIVE.
        const double d_lp_d_log_amp_lik = arma::dot(resid, f) / sigma2;
        (*grad)[1] = d_lp_d_log_amp_lik
                     - amp * amp / amp_var   // d/d log_amp of -0.5 amp^2/var
                     + 1.0;                  // log|J|

        // d sqrt_spd_m / d log_ell = sqrt_spd_m * (0.5 - 0.5 ell^2 lambda_m)
        arma::vec d_sqrt_spd_d_logell(M);
        for (std::size_t m = 0; m < M; ++m) {
            d_sqrt_spd_d_logell[m] = sqrt_spd[m]
                * (0.5 - 0.5 * ell * ell * lambda[m]);
        }
        arma::vec d_f_d_logell(N, arma::fill::zeros);
        for (std::size_t m = 0; m < M; ++m) {
            const double c = z[m] * d_sqrt_spd_d_logell[m];
            for (std::size_t n = 0; n < N; ++n) {
                d_f_d_logell[n] += c * phi_flat[n + m * N];
            }
        }
        // Same chain-rule pattern as log_amp: d log_lik / d log_ell
        // = sum(resid * d_f_d_logell) / sigma2.
        (*grad)[2] = arma::dot(resid, d_f_d_logell) / sigma2
                     // InvGamma + log|J| pieces (log_ell prior):
                     // log p(ell) = -(a+1)*log(ell) - b/ell + const
                     // log p(log_ell) = log p(ell) + log_ell
                     //                = -(a+1)*log_ell - b/ell + log_ell
                     //                = -a*log_ell - b/ell + const
                     // d/d log_ell  = -a + b/ell
                     - ell_invG_a + ell_invG_b / ell;

        // d/d log_sigma:
        // d log_lik / d log_sigma = -N + sum(resid^2) / sigma^2
        (*grad)[3] = -static_cast<double>(N)
                     + arma::dot(resid, resid) / sigma2;

        // d/d z_m: lik part = sqrt_spd_m * sum_n resid_n * phi_{n,m} / sigma^2
        for (std::size_t m = 0; m < M; ++m) {
            double phi_resid = 0.0;
            for (std::size_t n = 0; n < N; ++n) {
                phi_resid += resid[n] * phi_flat[n + m * N];
            }
            (*grad)[4 + m] = sqrt_spd[m] * phi_resid / sigma2 - z[m];
        }
    }

    return lp;
}

}  // anonymous namespace

// ============================================================================
// Class
// ============================================================================

class HSGPRegression {
public:
    HSGPRegression(const arma::vec& y,
                   const arma::vec& x,
                   int M,
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
          impl_(std::make_unique<composite_block>("HSGPRegression")),
          keep_history_(keep_history)
    {
        const std::size_t N = y.n_elem;
        if (x.n_elem != N) {
            throw std::invalid_argument("x and y must have the same length");
        }
        if (M < 1) throw std::invalid_argument("M must be >= 1");

        const std::size_t M_sz = static_cast<std::size_t>(M);

        // Domain half-width L = 1.5 * max(|x|) (edge margin guards basis at boundary).
        double L = 1.5 * arma::max(arma::abs(x));
        if (!(L > 0.0)) L = 1.0;

        // Eigenvalues lambda_m = (m * pi / (2 L))^2.
        arma::vec lambda(M_sz);
        for (std::size_t m = 0; m < M_sz; ++m) {
            const double w = (static_cast<double>(m + 1) * M_PI) / (2.0 * L);
            lambda[m] = w * w;
        }

        // Basis matrix phi(N, M), col-major flattened.
        arma::vec phi_flat(N * M_sz);
        const double sqrt_inv_L = std::sqrt(1.0 / L);
        for (std::size_t m = 0; m < M_sz; ++m) {
            const double w = std::sqrt(lambda[m]);
            for (std::size_t n = 0; n < N; ++n) {
                phi_flat[n + m * N] = sqrt_inv_L * std::sin(w * (x[n] + L));
            }
        }

        // Prior hyperparameters set from data scale.
        const double y_sd      = arma::stddev(y);
        const double amp_prior_sd = (y_sd > 0.0) ? y_sd : 1.0;
        const double x_range   = arma::max(x) - arma::min(x);
        const double ell_invG_a = 5.0;
        const double ell_invG_b = (x_range > 0.0) ? 0.25 * x_range : 1.0;

        // shared_data
        impl_->data().set("y",            y);
        impl_->data().set("x",            x);
        impl_->data().set("phi_flat",     phi_flat);
        impl_->data().set("lambda",       lambda);
        impl_->data().set("L",            arma::vec{L});
        impl_->data().set("M",            arma::vec{static_cast<double>(M_sz)});
        impl_->data().set("N",            arma::vec{static_cast<double>(N)});
        impl_->data().set("amp_prior_sd", arma::vec{amp_prior_sd});
        impl_->data().set("ell_invG_a",   arma::vec{ell_invG_a});
        impl_->data().set("ell_invG_b",   arma::vec{ell_invG_b});

        // Initial values (data-driven where possible, prior-mean otherwise):
        const double Intercept_init = arma::mean(y);
        const double log_amp_init   = std::log(0.5 * amp_prior_sd);
        const double log_ell_init   = std::log(ell_invG_b / (ell_invG_a - 1.0));
        const double log_sigma_init = std::log(0.5 * amp_prior_sd);
        arma::vec z_init(M_sz, arma::fill::zeros);

        // Cache initial scalars in shared_data so get_history records them
        // with the right names. The joint block writes back the full
        // theta_cat under each sub-param name.
        impl_->data().set("Intercept", arma::vec{Intercept_init});
        impl_->data().set("log_amp",   arma::vec{log_amp_init});
        impl_->data().set("log_ell",   arma::vec{log_ell_init});
        impl_->data().set("log_sigma", arma::vec{log_sigma_init});
        impl_->data().set("z",         z_init);

        // ====================================================================
        // ONE joint block: (Intercept, log_amp, log_ell, log_sigma, z[M]).
        // Dense metric: Welford-pilot adapts the (amp, ell, z, Intercept)
        // covariance during the FIRST step() call.
        // ====================================================================
        joint_nuts_block_config cfg;
        cfg.name = "hsgp_joint";
        cfg.sub_params.push_back({"Intercept", 1});
        cfg.sub_params.push_back({"log_amp",   1});
        cfg.sub_params.push_back({"log_ell",   1});
        cfg.sub_params.push_back({"log_sigma", 1});
        cfg.sub_params.push_back({"z",         M_sz});

        cfg.initial_cat = arma::vec(4 + M_sz);
        cfg.initial_cat[0] = Intercept_init;
        cfg.initial_cat[1] = log_amp_init;
        cfg.initial_cat[2] = log_ell_init;
        cfg.initial_cat[3] = log_sigma_init;
        for (std::size_t m = 0; m < M_sz; ++m) cfg.initial_cat[4 + m] = 0.0;

        cfg.log_density_grad      = &joint_log_density;
        cfg.use_dense_metric      = true;
        cfg.n_warmup_first_call   = 1500;

        impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));

        // Dependency DAG: the joint block reads fixed data + hyper-prior
        // scalars from shared_data. Sub-param keys are auto-included.
        impl_->data().declare_dependencies(
            "hsgp_joint", {"y", "phi_flat", "lambda",
                           "amp_prior_sd", "ell_invG_a", "ell_invG_b"});

        // ---- Full predict-DAG: real posterior predictive ---------------
        // (Replaces the old stub predict_at that echoed the training y.)
        // Parameterization read verbatim from joint_log_density:
        //   sqrt_spd_m = sqrt( amp^2 * sqrt(2pi) * ell *
        //                      exp(-0.5 * ell^2 * lambda_m) )   (len M)
        //   f_n  = sum_m (sqrt_spd_m * z_m) * phi[n + m*N]       (len N)
        //   mu   = Intercept + f                                (len N)
        //   y_rep ~ N(mu, exp(log_sigma)^2)                     (stoch)
        // sqrt_spd/f/mu kept current during sampling via
        // declare_invalidates (order sqrt_spd -> f -> mu) so stateful
        // predict_at(list()) (empty changed-set) reads fresh values.
        impl_->data().set("sqrt_spd", arma::vec(M_sz, arma::fill::zeros));
        impl_->data().set("f",        arma::vec(N,    arma::fill::zeros));
        impl_->data().set("mu",       arma::vec(N,    arma::fill::zeros));
        impl_->data().set("y_rep",    arma::vec(N,    arma::fill::zeros));

        impl_->data().register_refresher(
            "sqrt_spd",
            [](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const double amp = std::exp(d.get("log_amp")[0]);
                const double ell = std::exp(d.get("log_ell")[0]);
                const arma::vec& lam = d.get("lambda");
                const double kSqrt2Pi = std::sqrt(2.0 * M_PI);
                arma::vec ss(lam.n_elem);
                for (std::size_t m = 0; m < lam.n_elem; ++m) {
                    const double sd_m = amp * amp * kSqrt2Pi * ell
                        * std::exp(-0.5 * ell * ell * lam[m]);
                    ss[m] = std::sqrt(sd_m);
                }
                return ss;
            });
        impl_->data().register_refresher(
            "f",
            [](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& ss  = d.get("sqrt_spd");
                const arma::vec& z   = d.get("z");
                const arma::vec& phi = d.get("phi_flat");   // (N*M) col-major
                const std::size_t M = ss.n_elem;
                const std::size_t Nn = phi.n_elem / M;
                arma::vec f(Nn, arma::fill::zeros);
                for (std::size_t m = 0; m < M; ++m) {
                    const double c = ss[m] * z[m];
                    for (std::size_t n = 0; n < Nn; ++n)
                        f[n] += c * phi[n + m * Nn];
                }
                return f;
            });
        impl_->data().register_refresher(
            "mu",
            [](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                return d.get("Intercept")[0] + d.get("f");
            });
        impl_->data().declare_invalidates("hsgp_joint",
                                          {"sqrt_spd", "f", "mu"});

        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                const arma::vec& mu = d.get("mu");
                const double sigma  = std::exp(d.get("log_sigma")[0]);
                std::normal_distribution<double> nd(0.0, 1.0);
                arma::vec yr(mu.n_elem);
                for (std::size_t n = 0; n < mu.n_elem; ++n)
                    yr[n] = mu[n] + sigma * nd(rng);
                return yr;
            });

        // Predict DAG (full generative chain). phi_flat is the
        // PRECOMPUTED spectral basis at (new) x supplied R-side (Q3=A).
        impl_->data().declare_data_input("phi_flat");
        impl_->data().declare_predict_edges("log_amp",  {"sqrt_spd"});
        impl_->data().declare_predict_edges("log_ell",  {"sqrt_spd"});
        impl_->data().declare_predict_edges("lambda",   {"sqrt_spd"});
        impl_->data().declare_predict_edges("sqrt_spd", {"f"});
        impl_->data().declare_predict_edges("z",        {"f"});
        impl_->data().declare_predict_edges("phi_flat", {"f"});
        impl_->data().declare_predict_edges("f",        {"mu"});
        impl_->data().declare_predict_edges("Intercept",{"mu"});
        impl_->data().declare_predict_edges("mu",       {"y_rep"});
        impl_->data().declare_predict_edges("log_sigma",{"y_rep"});

        // Generative-DAG context: amp ~ Half-Normal(0, amp_prior_sd);
        // ell^2 ~ InverseGamma(ell_invG_a, ell_invG_b). amp=exp(log_amp),
        // ell=exp(log_ell) -> prior parents on the log params. Intercept
        // ~ N(0,10) and Jeffreys sigma are hardcoded (no slot).
        impl_->data().declare_context_edges("amp_prior_sd", {"log_amp"});
        impl_->data().declare_context_edges("ell_invG_a",   {"log_ell"});
        impl_->data().declare_context_edges("ell_invG_b",   {"log_ell"});

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        if (n_steps < 0) throw std::runtime_error("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["Intercept"] = impl_->data().get("Intercept");
        out["log_amp"]   = impl_->data().get("log_amp");
        out["log_ell"]   = impl_->data().get("log_ell");
        out["log_sigma"] = impl_->data().get("log_sigma");
        out["z"]         = impl_->data().get("z");
        // Convenience: also expose natural-scale scalars.
        const double a = std::exp(impl_->data().get("log_amp")[0]);
        const double e = std::exp(impl_->data().get("log_ell")[0]);
        const double s = std::exp(impl_->data().get("log_sigma")[0]);
        out["amp"]   = arma::vec{a};
        out["ell"]   = arma::vec{e};
        out["sigma"] = arma::vec{s};
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        for (const auto& [k, v] : params) {
            if (k == "amp"   || k == "ell"   || k == "sigma")  continue;
            impl_->data().set(k, v);
        }
    }

    // Real posterior predictive via the framework predict DAG (replaces
    // the old stub that echoed the training y). Single current-draw
    // sample (state_map vector contract).
    //   predict_at(list())                -> sqrt_spd? f,mu,y_rep at x
    //   predict_at(list(phi_flat=...))     -> at a NEW precomputed
    //       spectral basis (phi_flat = vectorise(Phi_new), N_new*M
    //       column-major; Q3=A: basis evaluated R-side at new x).
    AI4BayesCode::state_map predict_at(
        const AI4BayesCode::state_map& new_data) const {
        block_context replaced;
        auto it = new_data.find("phi_flat");
        if (it != new_data.end()) {
            const arma::vec& pf = it->second;
            const std::size_t M = impl_->data().get("sqrt_spd").n_elem;
            if (pf.n_elem == 0 || pf.n_elem % M != 0) {
                throw std::runtime_error(
                    "HSGPRegression::predict_at: phi_flat length " +
                    std::to_string(pf.n_elem) +
                    " is not a positive multiple of M = " +
                    std::to_string(M) +
                    " (pass vectorise(Phi_new), N_new*M col-major).");
            }
            replaced["phi_flat"] = pf;
        } else {
            for (const auto& kv : new_data) {
                throw std::runtime_error(
                    "HSGPRegression::predict_at: unknown key '" +
                    kv.first + "'. Valid: 'phi_flat' (or empty).");
            }
        }
        block_context r = impl_->predict_at(replaced, predict_rng_);
        AI4BayesCode::state_map out;
        for (const auto& kv : r) out[kv.first] = kv.second;
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
    std::mt19937_64 rng_;
    mutable std::mt19937_64 predict_rng_;   // predict_at() const advances it
    mutable std::mt19937_64 readapt_rng_; // readapt_NUTS() advances it (7th method)
    std::unique_ptr<composite_block> impl_;
    bool keep_history_ = false;
};

// ============================================================================
// Standalone demo
// ============================================================================
//
//  Simulate 1-D data from a KNOWN smooth function g(x) = 1.5*sin(2x) + 0.3*x
//  plus Gaussian noise. Fit the HSGP block, then compute the posterior-mean
//  fit mu_hat(x_n) by averaging the (Intercept + f) curve over posterior
//  draws. The in-sample fit is reconstructed in-place from each draw's
//  parameters using the SAME HSGP basis/spectral-density formulas the model
//  uses (L = 1.5*max|x|, lambda_m = (m*pi/2L)^2, phi_m, spd_m). Compare
//  against a NAIVE intercept-only baseline (predict mean(y) everywhere). The
//  HSGP must explain a large share of the signal variance to PASS.
// ============================================================================

int main() {
    const std::size_t N = 80;
    const int         M = 25;
    const std::size_t M_sz = static_cast<std::size_t>(M);
    const double      sigma_true = 0.4;

    // ---- simulate from a known smooth truth ----
    auto g_true = [](double x) { return 1.5 * std::sin(2.0 * x) + 0.3 * x; };

    std::mt19937_64 sim_rng(20240601ULL);
    std::uniform_real_distribution<double> x_gen(-3.0, 3.0);
    std::normal_distribution<double>       noise(0.0, sigma_true);

    arma::vec x(N), y(N), g(N);
    for (std::size_t n = 0; n < N; ++n) {
        x[n] = x_gen(sim_rng);
        g[n] = g_true(x[n]);
        y[n] = g[n] + noise(sim_rng);
    }

    // ---- precompute the HSGP basis EXACTLY as the model does (so the
    //      reconstructed in-sample fit matches the block's internal f). ----
    double L = 1.5 * arma::max(arma::abs(x));
    if (!(L > 0.0)) L = 1.0;
    arma::vec lambda(M_sz);
    for (std::size_t m = 0; m < M_sz; ++m) {
        const double w = (static_cast<double>(m + 1) * M_PI) / (2.0 * L);
        lambda[m] = w * w;
    }
    arma::vec phi_flat(N * M_sz);          // col-major: phi[n + m*N]
    const double sqrt_inv_L = std::sqrt(1.0 / L);
    for (std::size_t m = 0; m < M_sz; ++m) {
        const double w = std::sqrt(lambda[m]);
        for (std::size_t n = 0; n < N; ++n)
            phi_flat[n + m * N] = sqrt_inv_L * std::sin(w * (x[n] + L));
    }
    const double kSqrt2Pi = std::sqrt(2.0 * M_PI);

    // ---- fit ----
    HSGPRegression model(y, x, M, /*rng_seed=*/7, /*keep_history=*/false);
    model.step(800);   // warmup (first call runs the dense-metric pilot)

    // Posterior-mean fit: accumulate the in-sample mu(x_n) = Intercept + f(x_n)
    // curve over draws, reconstructing f from each draw's (amp, ell, z).
    const int    n_draws = 1500;
    arma::vec    mu_hat(N, arma::fill::zeros);
    double       sigma_bar = 0.0, amp_bar = 0.0, ell_bar = 0.0;
    for (int s = 0; s < n_draws; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        const double Intercept = cur.at("Intercept")[0];
        const double amp       = cur.at("amp")[0];
        const double ell       = cur.at("ell")[0];
        const arma::vec& z     = cur.at("z");

        // spd_m -> sqrt_spd_m, then f_n = sum_m sqrt_spd_m * z_m * phi_{n,m}.
        for (std::size_t n = 0; n < N; ++n) {
            double f_n = 0.0;
            for (std::size_t m = 0; m < M_sz; ++m) {
                const double sd_m = amp * amp * kSqrt2Pi * ell
                                    * std::exp(-0.5 * ell * ell * lambda[m]);
                f_n += std::sqrt(sd_m) * z[m] * phi_flat[n + m * N];
            }
            mu_hat[n] += Intercept + f_n;
        }
        sigma_bar += cur.at("sigma")[0];
        amp_bar   += amp;
        ell_bar   += ell;
    }
    mu_hat    /= static_cast<double>(n_draws);
    sigma_bar /= static_cast<double>(n_draws);
    amp_bar   /= static_cast<double>(n_draws);
    ell_bar   /= static_cast<double>(n_draws);

    // ---- evaluate recovery against the noise-free truth g(x) ----
    const double y_mean = arma::mean(y);
    double sse_model = 0.0, sse_naive = 0.0, sst = 0.0;
    for (std::size_t n = 0; n < N; ++n) {
        const double d_model = mu_hat[n] - g[n];   // fit vs noise-free truth
        const double d_naive = y_mean   - g[n];    // intercept-only baseline
        sse_model += d_model * d_model;
        sse_naive += d_naive * d_naive;
        sst       += (g[n] - arma::mean(g)) * (g[n] - arma::mean(g));
    }
    const double rmse_model = std::sqrt(sse_model / static_cast<double>(N));
    const double rmse_naive = std::sqrt(sse_naive / static_cast<double>(N));
    // R^2 of the HSGP fit against the true signal.
    const double r2 = 1.0 - sse_model / sst;

    std::printf("HSGPRegression demo (N=%zu, M=%d):\n", N, M);
    std::printf("  posterior-mean hyperparams: amp=%.3f  ell=%.3f  "
                "sigma=%.3f (truth %.2f)\n",
                amp_bar, ell_bar, sigma_bar, sigma_true);
    std::printf("  RMSE vs true signal: HSGP=%.3f  naive(mean-y)=%.3f\n",
                rmse_model, rmse_naive);
    std::printf("  HSGP R^2 vs true signal = %.3f\n", r2);

    // PASS: the HSGP fit must (a) explain most of the signal variance and
    // (b) beat the naive intercept-only baseline by a clear margin, with a
    // recovered noise scale near the truth.
    const bool ok = (r2 > 0.85) &&
                    (rmse_model < 0.5 * rmse_naive) &&
                    (std::abs(sigma_bar - sigma_true) < 0.2);
    std::printf("%s\n",
                ok ? "[demo PASS] HSGP recovers the smooth signal"
                   : "[demo FAIL] HSGP did not recover the signal");
    return ok ? 0 : 1;
}

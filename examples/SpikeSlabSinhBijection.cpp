// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  SpikeSlabSinhBijection.cpp -- minimal RJMCMC demo using a sinh-stretched
//                                heavy-tailed birth proposal.
//
//  Reference template for the rjmcmc_block custom-bijection path
//  (`make_templated_bijection_1d`, runtime AD Jacobian; see validator.md
//  Check #14 for the gen-time sanity probes).
//
//  WHY THIS EXAMPLE EXISTS
//  =======================
//  rjmcmc_block ships three families of birth proposals:
//
//    (a) Identity-coordinate proposals (no transform): beta_new IS the
//        auxiliary variable.
//    (b) Library-provided 1D transforms (`rjmcmc_transforms.hpp`):
//        identity / linear / affine.
//    (c) Custom user-supplied bijection (`rjmcmc_custom_bijection.hpp`):
//        for non-linear monotone maps the previous two cannot fit. The
//        user supplies one TEMPLATED forward map plus an analytic
//        inverse, and the framework computes |dbeta/du| via runtime
//        autodiff. No Jacobian formula is ever written by user code.
//
//  This example exercises path (c) end-to-end with a non-linear
//  bijection (sinh / asinh) on a small but real Dirac spike-and-slab
//  model.
//
//  Bijection (sinh-stretched proposal)
//  -----------------------------------
//      Forward:  T(u)        = scale * sinh(u)
//      Inverse:  T^{-1}(beta) = asinh(beta / scale)
//      |dT/du|             = scale * cosh(u)            (auto-computed)
//
//  Toy model (single coefficient, fixed hyperparameters)
//  -----------------------------------------------------
//      y_i           ~ N(beta * x_i, sigma^2),   sigma = 1     (fixed)
//      gamma         ~ Bernoulli(pi),            pi    = 0.5   (fixed)
//      beta | gamma=0 = 0                        (Dirac spike)
//      beta | gamma=1 ~ N(0, slab_sd^2),         slab_sd = 5  (fixed)
//
//  Closed-form posterior (used by the audit)
//  -----------------------------------------
//  P(gamma=1 | y) = pi * BF / (pi * BF + (1 - pi))
//  BF = sqrt(sigma^2 / (sigma^2 + slab_sd^2 * X'X))
//          * exp(0.5 * slab_sd^2 * (X'y)^2 /
//                (sigma^2 * (sigma^2 + slab_sd^2 * X'X)))
//  beta | gamma=1, y ~ N(m, v) with
//      v = 1 / (X'X / sigma^2 + 1 / slab_sd^2)
//      m = v * X'y / sigma^2
//  See `tests_autodiff/audit_rjmcmc_custom_bijection.R` for the comparison.
// ============================================================================

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that simulates data from a
// known truth, drives the rjmcmc_block composite, and checks that the sampled
// inclusion probability and conditional posterior mean of beta match the
// closed-form spike-and-slab posterior. No R / Python binding is built.

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/rjmcmc_block.hpp"
#include "AI4BayesCode/rjmcmc_custom_bijection.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::rjmcmc_block;
using AI4BayesCode::rjmcmc_block_config;
using AI4BayesCode::rjmcmc_transforms::make_templated_bijection_1d;

// ---------------------------------------------------------------------------
// TEMPLATED BIJECTION
//
// Forward must satisfy: T(u) is callable for both T = double (sampling)
// and T = autodiff::var (auto-Jacobian). The cleanest way is a struct
// with a templated operator().
// ---------------------------------------------------------------------------
struct SinhForward {
    double scale;
    template <typename T>
    T operator()(T u) const {
        // Both std::sinh (double) and autodiff::sinh (var) are reachable
        // via ADL: the unqualified `sinh` resolves to whichever overload
        // matches T. autodiff::var overloads for sinh / cosh / etc. are
        // pulled in by the include of <autodiff/reverse/var.hpp>.
        return T(scale) * sinh(u);
    }
};

struct AsinhInverse {
    double scale;
    double operator()(double beta) const {
        return std::asinh(beta / scale);
    }
};

// ---------------------------------------------------------------------------
// Build the rjmcmc spike-and-slab composite for the toy model and return it.
// The model is fully self-contained inside the rjmcmc_block config lambdas
// (they capture y, x, sigma, slab_sd, pi by value); the block does NOT read
// from data(), so no data().set / declare_dependencies wiring is needed.
//
// Returns histories of gamma (0/1 inclusion) and beta (0 when gamma=0) via
// the composite's get_history(). Drives the chain directly with comp->step.
// ---------------------------------------------------------------------------
static std::unique_ptr<composite_block>
make_spike_slab_sinh(const arma::vec& y,
                     const arma::vec& x,
                     double sigma,
                     double slab_sd,
                     double pi_inclusion)
{
    if (y.n_elem != x.n_elem) {
        throw std::runtime_error("y and x must have the same length");
    }
    if (!(sigma > 0.0 && slab_sd > 0.0)) {
        throw std::runtime_error("sigma and slab_sd must be > 0");
    }
    if (!(pi_inclusion > 0.0 && pi_inclusion < 1.0)) {
        throw std::runtime_error("pi_inclusion must be in (0, 1)");
    }

    const double xtx = arma::dot(x, x);
    const double xty = arma::dot(x, y);

    // ---- custom bijection transform ----
    auto bij = make_templated_bijection_1d(
        SinhForward{slab_sd}, AsinhInverse{slab_sd});

    // ---- rjmcmc_block config ----
    rjmcmc_block_config cfg;
    cfg.name      = "rj";
    cfg.gamma_key = "gamma";
    cfg.beta_key  = "beta";
    cfg.p         = 1;
    cfg.transform = bij;

    cfg.gamma_init = arma::vec({0.0});
    cfg.beta_init  = arma::vec({0.0});

    // continuous_update: closed-form Gibbs for beta | gamma=1, y, sigma, slab_sd.
    const double sigma2 = sigma * sigma;
    const double slab2  = slab_sd * slab_sd;
    cfg.continuous_update =
        [sigma2, slab2, xtx, xty]
        (std::mt19937_64& rng, std::size_t /*j*/, const block_context& /*ctx*/)
        -> double {
            const double prec = xtx / sigma2 + 1.0 / slab2;
            const double v    = 1.0 / prec;
            const double m    = v * (xty / sigma2);
            std::normal_distribution<double> nrm(m, std::sqrt(v));
            return nrm(rng);
        };

    // log_joint(gamma_1, beta_1).
    const arma::vec  y_loc = y;
    const arma::vec  x_loc = x;
    const double slab_sd_loc = slab_sd;
    const double sigma_loc   = sigma;
    const double pi_loc      = pi_inclusion;
    cfg.log_joint =
        [y_loc, x_loc, sigma_loc, slab_sd_loc, pi_loc]
        (const arma::vec& gamma, const arma::vec& beta,
         const block_context& /*ctx*/) -> double {
            const double g = gamma[0];
            const double b = beta[0];
            double lp = 0.0;
            // Prior on gamma
            lp += (g > 0.5) ? std::log(pi_loc) : std::log(1.0 - pi_loc);
            // Prior on beta | gamma
            if (g > 0.5) {
                lp += -0.5 * b * b / (slab_sd_loc * slab_sd_loc)
                      - 0.5 * std::log(2.0 * M_PI)
                      - std::log(slab_sd_loc);
            } else {
                if (std::abs(b) > 0.0) {
                    return -std::numeric_limits<double>::infinity();
                }
                // Dirac at 0; contribute log 1 = 0.
            }
            // Likelihood: y_i ~ N(b * x_i, sigma^2).
            const double sigma2_loc = sigma_loc * sigma_loc;
            arma::vec    resid      = y_loc - b * x_loc;
            const double rss        = arma::dot(resid, resid);
            lp += -0.5 * rss / sigma2_loc
                  - 0.5 * static_cast<double>(y_loc.n_elem)
                          * std::log(2.0 * M_PI * sigma2_loc);
            return lp;
        };

    // Birth aux: u ~ N(0, 1).
    cfg.propose_sample =
        [](std::mt19937_64& rng, std::size_t /*j*/, const block_context& /*ctx*/)
        -> double {
            std::normal_distribution<double> nrm(0.0, 1.0);
            return nrm(rng);
        };
    cfg.propose_logq =
        [](double u, std::size_t /*j*/, const block_context& /*ctx*/)
        -> double {
            return -0.5 * u * u - 0.5 * std::log(2.0 * M_PI);
        };

    // ---- composite + chain ----
    auto comp = std::make_unique<composite_block>("spike_slab_sinh");
    comp->add_child(std::make_unique<rjmcmc_block>(std::move(cfg)));
    return comp;
}

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulate from a KNOWN truth (beta_true != 0 so the model should strongly
//  prefer inclusion), drive the rjmcmc composite directly, then compare the
//  Monte-Carlo summaries against the CLOSED-FORM spike-and-slab posterior:
//    - sampled P(gamma=1)              vs  closed-form inclusion prob
//    - sampled E[beta | gamma=1]       vs  closed-form conditional mean m
//==============================================================================
#include <cstdio>
int main() {
    // ---- known truth ----
    const std::size_t N         = 80;
    const double      beta_true = 1.5;     // nonzero -> inclusion favored
    const double      sigma     = 1.0;
    const double      slab_sd   = 5.0;
    const double      pi_incl   = 0.5;

    // ---- simulate data ----
    std::mt19937_64 sim_rng(123);
    std::normal_distribution<double> xgen(0.0, 1.0);
    std::normal_distribution<double> egen(0.0, sigma);
    arma::vec x(N), y(N);
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = xgen(sim_rng);
        y[i] = beta_true * x[i] + egen(sim_rng);
    }

    // ---- closed-form posterior targets ----
    const double xtx    = arma::dot(x, x);
    const double xty    = arma::dot(x, y);
    const double sigma2 = sigma * sigma;
    const double slab2  = slab_sd * slab_sd;

    const double v_cf = 1.0 / (xtx / sigma2 + 1.0 / slab2);   // posterior var | gamma=1
    const double m_cf = v_cf * (xty / sigma2);                // posterior mean | gamma=1

    // Bayes factor (gamma=1 vs gamma=0) and inclusion probability.
    const double log_bf = 0.5 * std::log(sigma2 / (sigma2 + slab2 * xtx))
                          + 0.5 * slab2 * (xty * xty)
                                / (sigma2 * (sigma2 + slab2 * xtx));
    const double bf       = std::exp(log_bf);
    const double incl_cf  = pi_incl * bf / (pi_incl * bf + (1.0 - pi_incl));

    // ---- run the sampler ----
    auto comp = make_spike_slab_sinh(y, x, sigma, slab_sd, pi_incl);
    std::mt19937_64 rng(7);

    const int n_burn = 2000;
    const int n_keep = 20000;

    comp->set_keep_history(false);
    for (int i = 0; i < n_burn; ++i) comp->step(rng);

    comp->set_keep_history(true);
    for (int i = 0; i < n_keep; ++i) comp->step(rng);

    AI4BayesCode::history_map h = comp->get_history();

    // ---- Monte-Carlo summaries ----
    double sum_gamma     = 0.0;
    double sum_beta_incl = 0.0;
    long   n_incl        = 0;
    long   T             = 0;
    if (h.count("gamma") && h.count("beta")) {
        const arma::mat& g_hist = h.at("gamma");
        const arma::mat& b_hist = h.at("beta");
        T = static_cast<long>(std::min(g_hist.n_rows, b_hist.n_rows));
        for (long t = 0; t < T; ++t) {
            const double g = g_hist(static_cast<arma::uword>(t), 0);
            sum_gamma += (g > 0.5) ? 1.0 : 0.0;
            if (g > 0.5) {
                sum_beta_incl += b_hist(static_cast<arma::uword>(t), 0);
                ++n_incl;
            }
        }
    }

    const double incl_hat = (T > 0) ? sum_gamma / static_cast<double>(T) : 0.0;
    const double beta_hat = (n_incl > 0)
                                ? sum_beta_incl / static_cast<double>(n_incl)
                                : 0.0;

    std::printf("SpikeSlabSinhBijection demo (N=%zu, beta_true=%.2f)\n",
                N, beta_true);
    std::printf("  inclusion prob : sampled=%.4f   closed-form=%.4f\n",
                incl_hat, incl_cf);
    std::printf("  E[beta|gamma=1]: sampled=%.4f   closed-form=%.4f  "
                "(truth=%.2f)\n",
                beta_hat, m_cf, beta_true);

    // ---- pass criteria (derived from real computed comparisons) ----
    // Posterior strongly favors inclusion here; check both the inclusion
    // probability and the conditional mean against the closed form.
    const bool ok_incl = std::abs(incl_hat - incl_cf) < 0.03;
    const bool ok_beta = std::abs(beta_hat - m_cf)
                             < 0.05 + 0.05 * std::abs(m_cf);
    const bool ok = (T > 0) && ok_incl && ok_beta;

    std::printf("%s\n",
                ok ? "[demo PASS] rjmcmc custom sinh bijection matches "
                     "closed-form spike-and-slab posterior"
                   : "[demo FAIL]");
    return ok ? 0 : 1;
}

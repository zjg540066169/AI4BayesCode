// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_rjmcmc_block_transform.cpp
//
// T5 v0.5 tests — rjmcmc_block transform plumbing + parity verification.
//
// Scope
// -----
// (a) Transform wrappers (identity_transform_1d / diagonal_linear_transform_1d
//     / diagonal_affine_transform_1d) forward/reverse arithmetic + Jacobians.
// (b) rjmcmc_block with `transform = identity_transform_1d` — MUST produce
//     IDENTICAL samples to rjmcmc_block with `transform = nullptr` (v0 path)
//     for the same RNG seed, confirming the transform plumbing is
//     semantically transparent at identity.
// (c) rjmcmc_block with `transform = diagonal_linear_transform_1d(scale=1.0)`
//     — also identical to identity / no-transform.
// (d) rjmcmc_block with `transform = diagonal_linear_transform_1d(scale=2.0)`
//     — DIFFERENT samples, but a 2x scale on the aux variable is compensated
//     by the Jacobian; the resulting posterior should still be correct.
//     Check that the targeted Normal(0, 4) slab is achieved by seeding
//     beta ~ N(0, 1) on the aux scale.
//
// Returned Rcpp::List surfaces a single all_pass flag + details.
// ============================================================================

// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <RcppArmadillo.h>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/rjmcmc_transforms.hpp"
#include "AI4BayesCode/rjmcmc_block.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::rjmcmc_block;
using AI4BayesCode::rjmcmc_block_config;
namespace Tx = AI4BayesCode::rjmcmc_transforms;

namespace {

// A trivial 2-parameter spike-and-slab-ish target for the plumbing tests.
// p = 2, gamma_j in {0, 1}, beta_j ~ N(0, tau^2) slab.
//   log p(gamma, beta | y) ~ sum_i log N(y_i | X_i' beta, sigma^2)
//                           + sum_{j: gamma_j=1} log N(beta_j | 0, tau^2)
//                           + sum_j [gamma_j log pi + (1-gamma_j) log(1-pi)]
// Fixed hyperparameters: sigma=1, tau=1, pi=0.5, N=10 draws from a seed.
struct Fixture {
    arma::vec y;
    arma::vec X_flat;  // N x 2 column-major
    std::size_t N;
    double sigma2 = 1.0;
    double tau2   = 1.0;
    double pi_val = 0.5;
    Fixture() {
        std::mt19937_64 rng(42);
        std::normal_distribution<double> nd(0.0, 1.0);
        N = 10;
        X_flat.set_size(2 * N);
        y.set_size(N);
        for (std::size_t i = 0; i < N; ++i) {
            const double x1 = nd(rng);
            const double x2 = nd(rng);
            X_flat[i]       = x1;
            X_flat[N + i]   = x2;
            // truth: beta = (1, 0), sigma=1
            y[i] = 1.0 * x1 + 0.0 * x2 + nd(rng);
        }
    }
};

Fixture fx;

double tgt_log_joint(const arma::vec& gamma,
                     const arma::vec& beta,
                     const block_context& /*ctx*/) {
    const std::size_t N = fx.N;
    const std::size_t p = 2;
    double lp = 0.0;
    // likelihood
    double sse = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double xb = 0.0;
        for (std::size_t j = 0; j < p; ++j)
            xb += fx.X_flat[i + j * N] * beta[j];
        const double r = fx.y[i] - xb;
        sse += r * r;
    }
    lp += -0.5 * N * std::log(2.0 * M_PI * fx.sigma2)
          - sse / (2.0 * fx.sigma2);
    // slab on active beta
    std::size_t k = 0;
    for (std::size_t j = 0; j < p; ++j) {
        if (gamma[j] > 0.5) {
            ++k;
            lp += -0.5 * std::log(2.0 * M_PI * fx.tau2)
                  - beta[j] * beta[j] / (2.0 * fx.tau2);
        }
    }
    lp += k * std::log(fx.pi_val) + (p - k) * std::log(1.0 - fx.pi_val);
    return lp;
}

double tgt_propose_sample(std::mt19937_64& rng,
                          std::size_t /*j*/,
                          const block_context& /*ctx*/) {
    // aux u ~ N(0, 1). With identity transform, this becomes beta ~ N(0, 1);
    // with scale=2 transform, beta_new = 2 u ~ N(0, 4).
    std::normal_distribution<double> nd(0.0, 1.0);
    return nd(rng);
}

double tgt_propose_logq(double u,
                        std::size_t /*j*/,
                        const block_context& /*ctx*/) {
    // log q(u) = log N(u | 0, 1)
    return -0.5 * std::log(2.0 * M_PI) - 0.5 * u * u;
}

// Build a fresh rjmcmc_block with the given transform (or nullptr).
std::unique_ptr<rjmcmc_block> make_block(
    std::shared_ptr<Tx::transform_1d_base> transform) {
    rjmcmc_block_config cfg;
    cfg.name         = "gamma_beta_rj";
    cfg.gamma_key    = "gamma";
    cfg.beta_key     = "beta";
    cfg.p            = 2;
    cfg.rw_scale     = 0.0;
    cfg.log_joint    = &tgt_log_joint;
    cfg.propose_sample = &tgt_propose_sample;
    cfg.propose_logq   = &tgt_propose_logq;
    cfg.gamma_init   = arma::vec{0.0, 0.0};
    cfg.beta_init    = arma::vec{0.0, 0.0};
    cfg.transform    = transform;
    return std::make_unique<rjmcmc_block>(std::move(cfg));
}

// Run N_STEPS sweeps of a block and return pooled (gamma, beta) history.
struct ChainSummary {
    std::vector<arma::vec> gamma_hist;
    std::vector<arma::vec> beta_hist;
};
ChainSummary run_chain(rjmcmc_block& blk, std::mt19937_64& rng,
                       std::size_t n_steps) {
    block_context ctx;
    blk.set_context(ctx);
    ChainSummary out;
    out.gamma_hist.reserve(n_steps);
    out.beta_hist.reserve(n_steps);
    for (std::size_t s = 0; s < n_steps; ++s) {
        blk.step(rng);
        const auto names = blk.current_named_outputs();
        out.gamma_hist.push_back(names.at("gamma"));
        out.beta_hist.push_back (names.at("beta"));
    }
    return out;
}

// Compute inclusion frequencies per component.
arma::vec incl_of(const ChainSummary& cs) {
    const std::size_t p = cs.gamma_hist[0].n_elem;
    arma::vec incl(p, arma::fill::zeros);
    for (const auto& g : cs.gamma_hist)
        for (std::size_t j = 0; j < p; ++j)
            if (g[j] > 0.5) incl[j] += 1.0;
    return incl / static_cast<double>(cs.gamma_hist.size());
}

// Conditional-on-active posterior mean/var of beta_j.
void beta_active_stats(const ChainSummary& cs, std::size_t j,
                       double& mean_out, double& var_out, std::size_t& n_out) {
    mean_out = 0.0; var_out = 0.0; n_out = 0;
    for (std::size_t t = 0; t < cs.gamma_hist.size(); ++t) {
        if (cs.gamma_hist[t][j] > 0.5) {
            mean_out += cs.beta_hist[t][j];
            ++n_out;
        }
    }
    if (n_out > 0) mean_out /= n_out;
    for (std::size_t t = 0; t < cs.gamma_hist.size(); ++t) {
        if (cs.gamma_hist[t][j] > 0.5) {
            const double d = cs.beta_hist[t][j] - mean_out;
            var_out += d * d;
        }
    }
    if (n_out > 1) var_out /= (n_out - 1);
}

} // anonymous namespace

// [[Rcpp::export]]
Rcpp::List test_rjmcmc_block_transform() {
    std::vector<std::string> lines;
    bool all_ok = true;
    auto record = [&](const std::string& name, bool ok, const std::string& msg) {
        lines.push_back(name + ": " + (ok ? "PASS" : "FAIL") + " — " + msg);
        if (!ok) all_ok = false;
    };

    // =======================================================================
    // (a) Transform wrappers direct tests
    // =======================================================================
    {
        Tx::identity_transform_1d id;
        double out_b, out_u;
        double J_fwd = id.apply_forward(2.5, out_b);
        double J_rev = id.apply_reverse(-3.14, out_u);
        bool ok = (std::abs(out_b - 2.5) < 1e-15) &&
                  (std::abs(out_u + 3.14) < 1e-15) &&
                  (std::abs(J_fwd - 1.0) < 1e-15) &&
                  (std::abs(J_rev - 1.0) < 1e-15);
        record("identity_transform_1d", ok,
               "forward/reverse pass-through + |J|=1");
    }
    {
        Tx::diagonal_linear_transform_1d t(2.0);
        double out_b, out_u;
        double J_fwd = t.apply_forward(1.25, out_b);  // 2 * 1.25 = 2.5
        double J_rev = t.apply_reverse(5.0, out_u);   // 5 / 2 = 2.5
        bool ok = (std::abs(out_b - 2.5) < 1e-15) &&
                  (std::abs(out_u - 2.5) < 1e-15) &&
                  (std::abs(J_fwd - 2.0) < 1e-15) &&
                  (std::abs(J_rev - 0.5) < 1e-15);
        record("diagonal_linear_transform_1d(2)", ok,
               "forward/reverse + |J_fwd|=2 |J_rev|=0.5");
    }
    {
        Tx::diagonal_affine_transform_1d t(3.0, 1.0);  // beta = 3u + 1
        double out_b, out_u;
        double J_fwd = t.apply_forward(2.0, out_b);    // 3*2 + 1 = 7
        double J_rev = t.apply_reverse(10.0, out_u);   // (10-1)/3 = 3
        bool ok = (std::abs(out_b - 7.0) < 1e-15) &&
                  (std::abs(out_u - 3.0) < 1e-15) &&
                  (std::abs(J_fwd - 3.0) < 1e-15) &&
                  (std::abs(J_rev - (1.0/3.0)) < 1e-14);
        record("diagonal_affine_transform_1d(3,1)", ok,
               "forward/reverse + |J_fwd|=3 |J_rev|=1/3");
    }

    // =======================================================================
    // (b) Parity: no-transform vs identity_transform_1d at same seed
    //     Must produce byte-for-byte identical chains.
    // =======================================================================
    {
        auto blk0 = make_block(nullptr);
        auto blk1 = make_block(std::make_shared<Tx::identity_transform_1d>());
        std::mt19937_64 rng0(20260420);
        std::mt19937_64 rng1(20260420);
        auto cs0 = run_chain(*blk0, rng0, 2000);
        auto cs1 = run_chain(*blk1, rng1, 2000);
        bool same = true;
        for (std::size_t t = 0; t < 2000 && same; ++t) {
            for (std::size_t j = 0; j < 2; ++j) {
                if (cs0.gamma_hist[t][j] != cs1.gamma_hist[t][j] ||
                    std::abs(cs0.beta_hist[t][j] - cs1.beta_hist[t][j]) > 0.0) {
                    same = false; break;
                }
            }
        }
        record("parity identity vs no-transform", same,
               "identity-via-transform must be byte-identical to v0 path");
    }

    // =======================================================================
    // (c) diagonal_linear_transform_1d(scale=1) must also match no-transform.
    // =======================================================================
    {
        auto blk0 = make_block(nullptr);
        auto blk1 = make_block(
            std::make_shared<Tx::diagonal_linear_transform_1d>(1.0));
        std::mt19937_64 rng0(20260420);
        std::mt19937_64 rng1(20260420);
        auto cs0 = run_chain(*blk0, rng0, 2000);
        auto cs1 = run_chain(*blk1, rng1, 2000);
        bool same = true;
        for (std::size_t t = 0; t < 2000 && same; ++t) {
            for (std::size_t j = 0; j < 2; ++j) {
                if (cs0.gamma_hist[t][j] != cs1.gamma_hist[t][j] ||
                    std::abs(cs0.beta_hist[t][j] - cs1.beta_hist[t][j]) > 0.0) {
                    same = false; break;
                }
            }
        }
        record("parity scale=1 vs no-transform", same,
               "diagonal_linear(1) must be byte-identical to v0 path");
    }

    // =======================================================================
    // (d) diagonal_linear_transform_1d(scale=2): u ~ N(0,1), beta = 2*u
    //     makes the EFFECTIVE proposal on beta = N(0, 4). Under our slab
    //     prior N(0, tau^2=1), this is OVER-dispersed (stronger rejection
    //     of large-β births). The MH accept ratio correctly compensates
    //     via the Jacobian, so the posterior on (gamma, beta) converges
    //     to the SAME target. Analytical comparison is the most rigorous
    //     test — each chain's sample mean should match the posterior
    //     mean computed directly from the Bayesian-linear-regression
    //     closed form:
    //        E[beta_1 | gamma_1=1, gamma_2=0, y] =
    //            (X_1' y) / (X_1' X_1 + sigma^2 / tau^2)
    //     Both sigma^2 = tau^2 = 1, so this simplifies to:
    //        mu_post = X_1'y / (X_1'X_1 + 1)
    //        sd_post = 1 / sqrt(X_1'X_1 + 1)
    //     Tolerance = 3 * sd_post (3-sigma band on the posterior mean).
    // =======================================================================
    {
        // Analytical posterior mean for beta_1 | gamma_1=1, gamma_2=0.
        double X1tX1 = 0.0, X1ty = 0.0;
        for (std::size_t i = 0; i < fx.N; ++i) {
            X1tX1 += fx.X_flat[i] * fx.X_flat[i];
            X1ty  += fx.X_flat[i] * fx.y[i];
        }
        const double mu_post = X1ty / (X1tX1 + 1.0);
        const double sd_post = 1.0 / std::sqrt(X1tX1 + 1.0);

        auto blk_id = make_block(std::make_shared<Tx::identity_transform_1d>());
        auto blk_2x = make_block(
            std::make_shared<Tx::diagonal_linear_transform_1d>(2.0));
        std::mt19937_64 rng0(20260420);
        std::mt19937_64 rng1(20260421);
        const std::size_t N_STEPS = 20000;
        auto cs_id = run_chain(*blk_id, rng0, N_STEPS);
        auto cs_2x = run_chain(*blk_2x, rng1, N_STEPS);
        auto incl_id = incl_of(cs_id);
        auto incl_2x = incl_of(cs_2x);
        bool ok_incl_1 = std::abs(incl_id[0] - incl_2x[0]) < 0.10;
        bool ok_incl_2 = std::abs(incl_id[1] - incl_2x[1]) < 0.10;
        double m_id_1, v_id_1; std::size_t n_id_1;
        double m_2x_1, v_2x_1; std::size_t n_2x_1;
        beta_active_stats(cs_id, 0, m_id_1, v_id_1, n_id_1);
        beta_active_stats(cs_2x, 0, m_2x_1, v_2x_1, n_2x_1);

        // Each chain's beta_1 mean must be within 3 * sd_post of analytical
        // mu_post. sd_post accounts for the N=10 data variability; at 20k
        // samples, MC noise on the mean is ~ sd_post / sqrt(ESS) which is
        // much smaller than sd_post itself.
        const double tol_mean = 3.0 * sd_post;
        bool ok_mean_id = n_id_1 >= 500 && std::abs(m_id_1 - mu_post) < tol_mean;
        bool ok_mean_2x = n_2x_1 >= 500 && std::abs(m_2x_1 - mu_post) < tol_mean;

        record("distribution parity scale=2 inclusions",
               ok_incl_1 && ok_incl_2,
               "incl_id=[" + std::to_string(incl_id[0]) + "," +
               std::to_string(incl_id[1]) + "] vs incl_2x=[" +
               std::to_string(incl_2x[0]) + "," +
               std::to_string(incl_2x[1]) + "]");
        record("beta[1] mean identity vs analytical posterior",
               ok_mean_id,
               "mu_post=" + std::to_string(mu_post) +
               " (tol=" + std::to_string(tol_mean) + ")  m_id=" +
               std::to_string(m_id_1));
        record("beta[1] mean scale=2 vs analytical posterior",
               ok_mean_2x,
               "mu_post=" + std::to_string(mu_post) +
               " (tol=" + std::to_string(tol_mean) + ")  m_2x=" +
               std::to_string(m_2x_1));
    }

    // =======================================================================
    // Return summary
    // =======================================================================
    Rcpp::CharacterVector details(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i)
        details[i] = lines[i];
    return Rcpp::List::create(
        Rcpp::Named("all_pass") = all_ok,
        Rcpp::Named("details")  = details);
}

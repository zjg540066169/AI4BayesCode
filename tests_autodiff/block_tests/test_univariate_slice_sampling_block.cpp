// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_univariate_slice_sampling_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::univariate_slice_sampling_block
// (**Check #15** from skills/codegen.md + skills/validator.md).
//
// Purpose
// -------
// Verify that univariate_slice_sampling_block's sampling mechanism
// (Neal 2003 section 4.1 stepping-out + shrinkage) correctly targets
// the stationary distribution described by a fixed user-supplied
// unconstrained-scale log-density. Each fixture pins the sampler against
// a known analytic target and checks sample mean/variance agreement.
//
// Any example using univariate_slice_sampling_block with a textbook
// log_density (using constraints::<kind>::wrap and any analytically
// correct natural-scale lp) inherits correctness from this test.
//
// Fixtures
// --------
//   1. Normal(mu=2, sigma=1.5) via identity (real) constraint.
//      analytic:  E[X] = 2.0, Var[X] = 2.25
//
//   2. Gamma(shape=3, rate=2) via positive constraint.
//      natural-scale log p (up to constants): (shape-1)*log(x) - rate*x
//      analytic:  E[X] = shape/rate = 1.5, Var[X] = shape/rate^2 = 0.75
//
//   3. Beta(alpha=2, beta=5) via interval(0, 1) constraint.
//      natural-scale log p: (alpha-1)*log(x) + (beta-1)*log(1-x)
//      analytic:  E[X] = 2/7 ~ 0.2857, Var[X] = 10/392 ~ 0.02551
//
// Tolerances (Check #15): |rel_err(mean)| < 5% AND |rel_err(var)| < 10%.
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
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/univariate_slice_sampling_block.hpp"

#include <cmath>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::univariate_slice_sampling_block;
using AI4BayesCode::univariate_slice_sampling_block_config;
namespace cs = AI4BayesCode::constraints;

namespace {

// Run one fixture: 5000 burnin + N_DRAWS kept, return (mean, var).
struct fixture_result {
    double sample_mean;
    double sample_var;
    double exp_mean;
    double exp_var;
    double mean_rel_err;
    double var_rel_err;
    bool   pass_mean;
    bool   pass_var;
    bool   pass_all;
};

fixture_result run_fixture(univariate_slice_sampling_block_config cfg,
                           double expected_mean,
                           double expected_var,
                           double initial_natural,
                           double tol_mean,
                           double tol_var,
                           std::size_t n_burn,
                           std::size_t n_draws,
                           std::uint64_t seed) {
    univariate_slice_sampling_block blk(std::move(cfg));

    // Force initial state to a sensible natural-scale value so burnin
    // doesn't start at a pathological point.
    arma::vec init_nat(1);
    init_nat[0] = initial_natural;
    blk.set_current(init_nat);

    // Minimal empty context.
    block_context ctx;
    blk.set_context(ctx);

    std::mt19937_64 rng(seed);

    // Burnin
    for (std::size_t i = 0; i < n_burn; ++i) blk.step(rng);

    // Kept draws
    std::vector<double> samples;
    samples.reserve(n_draws);
    for (std::size_t i = 0; i < n_draws; ++i) {
        blk.step(rng);
        samples.push_back(blk.current()[0]);
    }

    double mean = 0.0;
    for (double v : samples) mean += v;
    mean /= static_cast<double>(n_draws);

    double var = 0.0;
    for (double v : samples) {
        const double d = v - mean;
        var += d * d;
    }
    var /= static_cast<double>(n_draws - 1);

    fixture_result r;
    r.sample_mean = mean;
    r.sample_var  = var;
    r.exp_mean    = expected_mean;
    r.exp_var     = expected_var;
    r.mean_rel_err = std::abs(mean - expected_mean) /
                     std::max(std::abs(expected_mean), 1e-10);
    r.var_rel_err  = std::abs(var  - expected_var)  /
                     std::max(std::abs(expected_var),  1e-10);
    r.pass_mean = (r.mean_rel_err < tol_mean);
    r.pass_var  = (r.var_rel_err  < tol_var);
    r.pass_all  = r.pass_mean && r.pass_var;
    return r;
}

} // namespace

// [[Rcpp::export]]
Rcpp::List test_univariate_slice_sampling_block() {
    const std::size_t N_BURN  = 5000;
    const std::size_t N_DRAWS = 10000;
    const double TOL_MEAN = 0.05;  // Check #15
    const double TOL_VAR  = 0.10;  // Check #15

    // ----------------------------------------------------------------------
    // Fixture 1: Normal(2, 1.5) via identity (real) constraint.
    // ----------------------------------------------------------------------
    const double mu_N    = 2.0;
    const double sd_N    = 1.5;
    const double var_N   = sd_N * sd_N;   // 2.25
    fixture_result r_normal;
    {
        univariate_slice_sampling_block_config cfg;
        cfg.name        = "x_normal";
        cfg.initial_unc = arma::vec{0.0};  // identity, nat = unc = 0 initially
        cfg.log_density =
            [mu_N, var_N](const arma::vec& t_unc, const block_context&) {
                return cs::real::wrap(t_unc, nullptr,
                    [mu_N, var_N](const arma::vec& t_nat,
                                  arma::vec* /*grad_unused*/) -> double {
                        const double x = t_nat[0];
                        return -0.5 * (x - mu_N) * (x - mu_N) / var_N;
                    });
            };
        // constrain/unconstrain: identity (default)
        r_normal = run_fixture(std::move(cfg), mu_N, var_N,
                               /*initial_natural=*/mu_N,
                               TOL_MEAN, TOL_VAR, N_BURN, N_DRAWS, 101ULL);
    }

    // ----------------------------------------------------------------------
    // Fixture 2: Gamma(shape=3, rate=2) via positive constraint.
    // ----------------------------------------------------------------------
    const double sh_G   = 3.0;
    const double rt_G   = 2.0;
    const double mean_G = sh_G / rt_G;                  // 1.5
    const double var_G  = sh_G / (rt_G * rt_G);         // 0.75
    fixture_result r_gamma;
    {
        univariate_slice_sampling_block_config cfg;
        cfg.name        = "x_gamma";
        cfg.initial_unc = arma::vec{std::log(mean_G)};
        cfg.constrain   = cs::positive::constrain;
        cfg.unconstrain = cs::positive::unconstrain;
        cfg.log_density =
            [sh_G, rt_G](const arma::vec& t_unc, const block_context&) {
                return cs::positive::wrap(t_unc, nullptr,
                    [sh_G, rt_G](const arma::vec& t_nat,
                                 arma::vec* /*grad_unused*/) -> double {
                        const double x = t_nat[0];
                        return (sh_G - 1.0) * std::log(x) - rt_G * x;
                    });
            };
        r_gamma = run_fixture(std::move(cfg), mean_G, var_G,
                              /*initial_natural=*/mean_G,
                              TOL_MEAN, TOL_VAR, N_BURN, N_DRAWS, 202ULL);
    }

    // ----------------------------------------------------------------------
    // Fixture 3: Beta(alpha=2, beta=5) via interval(0, 1) constraint.
    // ----------------------------------------------------------------------
    const double al_B  = 2.0;
    const double be_B  = 5.0;
    const double mean_B = al_B / (al_B + be_B);                      // 2/7
    const double var_B  = al_B * be_B /
        ((al_B + be_B) * (al_B + be_B) * (al_B + be_B + 1.0));       // 10/392
    fixture_result r_beta;
    {
        univariate_slice_sampling_block_config cfg;
        cfg.name        = "x_beta";
        const double lo = 0.0, up = 1.0;
        cfg.initial_unc = arma::vec{0.0};   // sigmoid(0) = 0.5
        cfg.constrain   = cs::interval::constrain_fn(lo, up);
        cfg.unconstrain = cs::interval::unconstrain_fn(lo, up);
        cfg.log_density =
            [al_B, be_B, lo, up](const arma::vec& t_unc,
                                 const block_context&) {
                return cs::interval::wrap(t_unc, nullptr, lo, up,
                    [al_B, be_B](const arma::vec& t_nat,
                                 arma::vec* /*grad_unused*/) -> double {
                        const double x = t_nat[0];
                        return (al_B - 1.0) * std::log(x)
                             + (be_B - 1.0) * std::log1p(-x);
                    });
            };
        r_beta = run_fixture(std::move(cfg), mean_B, var_B,
                             /*initial_natural=*/mean_B,
                             TOL_MEAN, TOL_VAR, N_BURN, N_DRAWS, 303ULL);
    }

    const bool all_pass =
        r_normal.pass_all && r_gamma.pass_all && r_beta.pass_all;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")        = all_pass,
        Rcpp::Named("n_burn")          = static_cast<int>(N_BURN),
        Rcpp::Named("n_draws")         = static_cast<int>(N_DRAWS),
        Rcpp::Named("tol_mean")        = TOL_MEAN,
        Rcpp::Named("tol_var")         = TOL_VAR,

        Rcpp::Named("normal_pass")     = r_normal.pass_all,
        Rcpp::Named("normal_mean")     = r_normal.sample_mean,
        Rcpp::Named("normal_exp_mean") = r_normal.exp_mean,
        Rcpp::Named("normal_var")      = r_normal.sample_var,
        Rcpp::Named("normal_exp_var")  = r_normal.exp_var,
        Rcpp::Named("normal_mean_re")  = r_normal.mean_rel_err,
        Rcpp::Named("normal_var_re")   = r_normal.var_rel_err,

        Rcpp::Named("gamma_pass")      = r_gamma.pass_all,
        Rcpp::Named("gamma_mean")      = r_gamma.sample_mean,
        Rcpp::Named("gamma_exp_mean")  = r_gamma.exp_mean,
        Rcpp::Named("gamma_var")       = r_gamma.sample_var,
        Rcpp::Named("gamma_exp_var")   = r_gamma.exp_var,
        Rcpp::Named("gamma_mean_re")   = r_gamma.mean_rel_err,
        Rcpp::Named("gamma_var_re")    = r_gamma.var_rel_err,

        Rcpp::Named("beta_pass")       = r_beta.pass_all,
        Rcpp::Named("beta_mean")       = r_beta.sample_mean,
        Rcpp::Named("beta_exp_mean")   = r_beta.exp_mean,
        Rcpp::Named("beta_var")        = r_beta.sample_var,
        Rcpp::Named("beta_exp_var")    = r_beta.exp_var,
        Rcpp::Named("beta_mean_re")    = r_beta.mean_rel_err,
        Rcpp::Named("beta_var_re")     = r_beta.var_rel_err);
}

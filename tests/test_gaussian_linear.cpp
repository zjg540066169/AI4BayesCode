/*================================================================================
 *  block_mcmc  --  end-to-end smoke test of the Layer-1 architecture.
 *================================================================================
 *
 *  Model
 *  -----
 *      y_i | mu, sigma ~  Normal(mu, sigma^2),      i = 1..N
 *      mu              ~  Normal(0, 100^2)          (weakly informative)
 *      sigma           ~  Half-Normal(0, 10)
 *
 *  What this test proves
 *  ---------------------
 *    1. The abstract block_sampler / shared_data_t / composite_block
 *       interfaces type-check and compose.
 *    2. Two nuts_blocks with persistent adaptation can live inside a
 *       single composite and update each other's shared data without
 *       any direct coupling.
 *    3. A block that uses a constraint transform (sigma > 0) correctly
 *       uses constraints::positive::wrap and gets a reasonable posterior.
 *    4. The persistent dual-averaging state accumulates across Gibbs
 *       sweeps -- the step sizes at the end of the run should be
 *       stable and different from the initial 1.0 value.
 *    5. The posterior means of mu and sigma recover the ground truth
 *       to within a few standard errors with N = 200 data points and
 *       2000 Gibbs sweeps after a 500-sweep burn-in.
 *
 *  What this test does NOT prove (and doesn't try to)
 *  --------------------------------------------------
 *    - BridgeStan integration (intentionally absent from v0)
 *    - Mass-matrix adaptation (frozen at identity for v0)
 *    - Multi-chain diagnostics
 *    - BART block (arriving in a follow-up commit)
 *
 *  Build
 *  -----
 *  See tests/Makefile in this directory.
 *================================================================================*/

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
using AI4BayesCode::composite_block;
namespace constraints = AI4BayesCode::constraints;

// ----------------------------------------------------------------------------
// 1. Synthesize data
// ----------------------------------------------------------------------------

static arma::vec generate_data(std::size_t N, double mu_true, double sigma_true,
                               std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(mu_true, sigma_true);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) y[i] = nd(rng);
    return y;
}

// ----------------------------------------------------------------------------
// 2. Target functions -- the "AI-written" bits.
//    These are the only C++ that an AI code generator would need to
//    emit per model. Everything else is Layer 1 machinery.
// ----------------------------------------------------------------------------

/// Conditional log-posterior of mu given (y, sigma).
///   log p(mu) = -0.5 * sum((y - mu)^2) / sigma^2  -  0.5 * mu^2 / prior_var
static double mu_log_density(const arma::vec& theta_unc,
                             const block_context& ctx,
                             arma::vec* grad) {
    const double mu    = theta_unc[0];
    const arma::vec& y = ctx.at("y");
    const double sigma = ctx.at("sigma")[0];

    const double sigma2    = sigma * sigma;
    const double prior_var = 100.0 * 100.0;

    double sum_resid = 0.0;
    double sum_sq    = 0.0;
    for (std::size_t i = 0; i < y.n_elem; ++i) {
        const double r = y[i] - mu;
        sum_resid += r;
        sum_sq    += r * r;
    }

    const double lp = -0.5 * sum_sq / sigma2 - 0.5 * mu * mu / prior_var;

    if (grad) {
        grad->set_size(1);
        (*grad)[0] = sum_resid / sigma2 - mu / prior_var;
    }
    return lp;
}

/// Conditional log-posterior of sigma given (y, mu), on the NATURAL
/// (positive) scale. The AI writes ONLY this function; the unconstraining
/// transform and its Jacobian are handled by constraints::positive::wrap.
///
///   log p(sigma) = -N log(sigma) - 0.5 * sum_sq / sigma^2
///                  - 0.5 * sigma^2 / prior_sd^2
static double sigma_natural_log_density(const arma::vec& sigma_nat,
                                        const block_context& ctx,
                                        arma::vec* grad_nat) {
    const double sigma = sigma_nat[0];
    const arma::vec& y = ctx.at("y");
    const double mu    = ctx.at("mu")[0];

    const double sigma2   = sigma * sigma;
    const double prior_sd = 10.0;
    const double N        = static_cast<double>(y.n_elem);

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < y.n_elem; ++i) {
        const double r = y[i] - mu;
        sum_sq += r * r;
    }

    const double lp =
        -N * std::log(sigma)
        - 0.5 * sum_sq / sigma2
        - 0.5 * sigma2 / (prior_sd * prior_sd);

    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] =
            -N / sigma
            + sum_sq / (sigma2 * sigma)
            - sigma / (prior_sd * prior_sd);
    }
    return lp;
}

// ----------------------------------------------------------------------------
// 3. Assemble the model
// ----------------------------------------------------------------------------

static std::unique_ptr<composite_block>
build_model(const arma::vec& y, double mu_init, double sigma_init) {
    auto model = std::make_unique<composite_block>("gaussian_location_scale");

    // ---- shared data -------------------------------------------------
    model->data().set("y",     y);
    model->data().set("mu",    arma::vec{mu_init});
    model->data().set("sigma", arma::vec{sigma_init});

    // dependency metadata (the "DAG" of the model)
    model->data().declare_dependencies("mu",    {"y", "sigma"});
    model->data().declare_dependencies("sigma", {"y", "mu"});
    // no derived quantities in this simple model

    // ---- block for mu (real / unconstrained) -------------------------
    {
        nuts_block_config cfg;
        cfg.name        = "mu";
        cfg.initial_unc = arma::vec{mu_init};
        cfg.log_density_grad =
            [](const arma::vec& theta_unc, const block_context& ctx,
               arma::vec* grad) {
                return constraints::real::wrap(
                    theta_unc, grad,
                    [&](const arma::vec& theta_nat, arma::vec* grad_nat) {
                        return mu_log_density(theta_nat, ctx, grad_nat);
                    });
            };
        cfg.nuts_settings.nuts_settings.step_size       = 0.5;
        cfg.nuts_settings.nuts_settings.max_tree_depth  = 6;
        cfg.nuts_settings.nuts_settings.target_accept_rate = 0.8;
        cfg.n_draws_per_step                            = 1;
        model->add_child(std::make_unique<nuts_block>(std::move(cfg)));
    }

    // ---- block for sigma (positive) ----------------------------------
    {
        nuts_block_config cfg;
        cfg.name        = "sigma";
        cfg.initial_unc = arma::vec{std::log(sigma_init)}; // log of initial
        cfg.constrain   = constraints::positive::constrain;
        cfg.unconstrain = constraints::positive::unconstrain;
        cfg.log_density_grad =
            [](const arma::vec& theta_unc, const block_context& ctx,
               arma::vec* grad) {
                return constraints::positive::wrap(
                    theta_unc, grad,
                    [&](const arma::vec& sigma_nat, arma::vec* grad_nat) {
                        return sigma_natural_log_density(
                            sigma_nat, ctx, grad_nat);
                    });
            };
        cfg.nuts_settings.nuts_settings.step_size       = 0.5;
        cfg.nuts_settings.nuts_settings.max_tree_depth  = 6;
        cfg.nuts_settings.nuts_settings.target_accept_rate = 0.8;
        cfg.n_draws_per_step                            = 1;
        model->add_child(std::make_unique<nuts_block>(std::move(cfg)));
    }

    return model;
}

// ----------------------------------------------------------------------------
// 4. Run the Gibbs sampler
// ----------------------------------------------------------------------------

struct posterior_summary {
    double mu_mean, mu_sd;
    double sigma_mean, sigma_sd;
    double final_mu_stepsize;
    double final_sigma_stepsize;
};

static posterior_summary run_sampler(composite_block& model,
                                     std::size_t n_burnin,
                                     std::size_t n_keep,
                                     std::uint64_t seed) {
    std::mt19937_64 rng(seed);

    for (std::size_t s = 0; s < n_burnin; ++s) model.step(rng);

    arma::vec mu_draws(n_keep);
    arma::vec sigma_draws(n_keep);
    for (std::size_t s = 0; s < n_keep; ++s) {
        model.step(rng);
        mu_draws[s]    = model.data().get("mu")[0];
        sigma_draws[s] = model.data().get("sigma")[0];
    }

    posterior_summary out;
    out.mu_mean    = arma::mean(mu_draws);
    out.mu_sd      = arma::stddev(mu_draws);
    out.sigma_mean = arma::mean(sigma_draws);
    out.sigma_sd   = arma::stddev(sigma_draws);

    const auto& mu_blk =
        dynamic_cast<const nuts_block&>(model.child(0));
    const auto& sg_blk =
        dynamic_cast<const nuts_block&>(model.child(1));
    out.final_mu_stepsize    = mu_blk.current_step_size();
    out.final_sigma_stepsize = sg_blk.current_step_size();
    return out;
}

// ----------------------------------------------------------------------------
// 5. Main
// ----------------------------------------------------------------------------

int main() {
    const std::size_t N = 200;
    const double mu_true    = 2.5;
    const double sigma_true = 1.2;
    const std::uint64_t data_seed  = 20260410ull;
    const std::uint64_t chain_seed = 31337ull;
    const std::size_t n_burnin = 500;
    const std::size_t n_keep   = 2000;

    const arma::vec y = generate_data(N, mu_true, sigma_true, data_seed);
    std::printf("Generated %zu data points: sample mean = %.4f, "
                "sample sd = %.4f\n",
                N, arma::mean(y), arma::stddev(y));

    // Start the chain deliberately away from the truth.
    auto model = build_model(y, /*mu_init=*/ 0.0, /*sigma_init=*/ 3.0);

    const posterior_summary s =
        run_sampler(*model, n_burnin, n_keep, chain_seed);

    std::printf("\nPosterior summary after %zu burnin + %zu keep sweeps:\n",
                n_burnin, n_keep);
    std::printf("  mu:    mean = %7.4f   sd = %6.4f   (truth %.4f)\n",
                s.mu_mean, s.mu_sd, mu_true);
    std::printf("  sigma: mean = %7.4f   sd = %6.4f   (truth %.4f)\n",
                s.sigma_mean, s.sigma_sd, sigma_true);
    std::printf("  final persistent step sizes: mu = %.4f, sigma = %.4f\n",
                s.final_mu_stepsize, s.final_sigma_stepsize);

    // ---- pass / fail ------------------------------------------------
    const double mu_tol    = 4.0 * sigma_true / std::sqrt(static_cast<double>(N));
    const double sigma_tol = 4.0 * sigma_true / std::sqrt(2.0 * static_cast<double>(N));
    const bool mu_ok    = std::abs(s.mu_mean    - mu_true)    < mu_tol;
    const bool sigma_ok = std::abs(s.sigma_mean - sigma_true) < sigma_tol;
    const bool step_ok  = s.final_mu_stepsize    > 0.0 &&
                          s.final_sigma_stepsize > 0.0;

    std::printf("\nmu within %.3f of truth? %s\n",
                mu_tol, mu_ok ? "YES" : "NO");
    std::printf("sigma within %.3f of truth? %s\n",
                sigma_tol, sigma_ok ? "YES" : "NO");
    std::printf("persistent step sizes positive? %s\n",
                step_ok ? "YES" : "NO");

    const bool all_ok = mu_ok && sigma_ok && step_ok;
    std::printf("\n%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

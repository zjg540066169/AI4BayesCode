// Copyright (C) 2026 AI4BayesCode contributors.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
//
// PoissonStateSpace.cpp -- frontend-independent worked example for
// particle_gibbs_block: a Poisson state-space (count) model, demonstrating that
// the SAME path sampler that serves stochastic volatility also serves a
// nonlinear non-Gaussian count model -- only the plug-in densities change.
//
//  MODEL (latent log-rate x_t, AR(1); parameters mu, phi, sigma_eta):
//      x_1         ~ Normal(mu, sigma_eta^2 / (1 - phi^2))
//      x_t | x_{t-1} ~ Normal(mu + phi (x_{t-1} - mu), sigma_eta^2),  t >= 2
//      y_t | x_t   ~ Poisson(exp(x_t))
//  PRIORS (auxiliary -- they live in the example, NOT the block):
//      mu          ~ Normal(0, 5^2)
//      phi         ~ Uniform(-1, 1)                 (stationary region)
//      sigma_eta^2 ~ Inverse-Gamma(2, 0.05)         (proper, weakly informative;
//                                                    NOT IG(eps,eps), per Gelman 2006)
//
//  BLOCK DECOMPOSITION (full particle Gibbs):
//      composite_block "PoissonStateSpace"
//        +-- particle_gibbs_block "x"   <-- samples the latent path x_{1:T}|y,theta
//                                           by conditional SMC with ancestor sampling.
//      theta = (mu, phi, sigma_eta) is sampled by the inline updates in main()
//      (mu, sigma_eta^2: conjugate Gibbs; phi: random-walk Metropolis), so the two
//      halves alternate -- path | theta, then theta | path -- the full particle
//      Gibbs scheme. (In the codegen flow those theta updates would be sibling
//      blocks; here they are inline to keep the demo self-contained.)
//
//  WHY THIS BLOCK: a nonlinear / non-Gaussian state-space time series with a
//  continuous latent path -- exactly the particle_gibbs_block SelectWhen. The
//  Poisson observation has no closed-form filter (no Kalman / forward-backward),
//  and the kernel is gradient-free, so neither gmrf_* nor joint_nuts_block fits.
//
//  The demo simulates from a KNOWN truth, fits, and checks: (1) the posterior
//  mean path beats a naive per-time baseline log(y_t + 0.5) that ignores the
//  AR(1) smoothing; (2) theta recovery (truth within 3 posterior SDs); (3) a
//  posterior-predictive p-value for Var(y) is non-extreme. Returns 0 on pass.
//
//  Defaults (override if you care): T=200, truth (mu,phi,sigma_eta)=(1.0,0.9,0.3),
//  N=64 particles, 1000 warmup + 4000 sampling sweeps.

#include "particle_gibbs_block.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::particle_gibbs_block;
using AI4BayesCode::particle_gibbs_block_config;
using AI4BayesCode::pg_resampling_scheme;

namespace {

arma::vec sv1(double x) { arma::vec v(1); v[0] = x; return v; }

// ---- Poisson state-space plug-ins (read theta & y_t from ctx) --------------
arma::vec pois_init(const block_context& ctx, std::mt19937_64& rng) {
    const double mu = ctx.at("mu")[0], phi = ctx.at("phi")[0], se = ctx.at("sigma_eta")[0];
    std::normal_distribution<double> n01(0.0, 1.0);
    return sv1(mu + (se / std::sqrt(1.0 - phi * phi)) * n01(rng));
}
arma::vec pois_trans(int /*t*/, const arma::vec& xp, const block_context& ctx,
                     std::mt19937_64& rng) {
    const double mu = ctx.at("mu")[0], phi = ctx.at("phi")[0], se = ctx.at("sigma_eta")[0];
    std::normal_distribution<double> n01(0.0, 1.0);
    return sv1(mu + phi * (xp[0] - mu) + se * n01(rng));
}
double pois_obs(int t, const arma::vec& x, const block_context& ctx) {
    const double y = ctx.at("y")[t]; const double xt = x[0];
    const double rate = std::exp(xt);
    if (!std::isfinite(rate)) return -std::numeric_limits<double>::infinity();
    return y * xt - rate - std::lgamma(y + 1.0);   // log Poisson(y; exp(x))
}
double pois_tlpdf(int /*t*/, const arma::vec& x, const arma::vec& xp,
                  const block_context& ctx) {
    const double mu = ctx.at("mu")[0], phi = ctx.at("phi")[0], se = ctx.at("sigma_eta")[0];
    const double q = se * se;
    const double d = x[0] - (mu + phi * (xp[0] - mu));
    return -0.5 * std::log(2.0 * M_PI * q) - d * d / (2.0 * q);
}

void simulate_poisson_ssm(double mu, double phi, double se, int T,
                          std::mt19937_64& rng, arma::vec& x, arma::vec& y) {
    x.set_size(T); y.set_size(T);
    std::normal_distribution<double> n01(0.0, 1.0);
    x[0] = mu + (se / std::sqrt(1.0 - phi * phi)) * n01(rng);
    for (int t = 1; t < T; ++t) x[t] = mu + phi * (x[t - 1] - mu) + se * n01(rng);
    for (int t = 0; t < T; ++t) {
        std::poisson_distribution<int> pd(std::exp(x[t]));
        y[t] = static_cast<double>(pd(rng));
    }
}

// ---- theta full-conditional updates given the current path -----------------
//  sigma_eta^2 | x, mu, phi : Inverse-Gamma conjugate from the AR(1) innovations
//  (including the stationary first term). Prior IG(a0,b0).
double sample_sigma2(const arma::vec& x, double mu, double phi,
                     double a0, double b0, std::mt19937_64& rng) {
    const int T = static_cast<int>(x.n_elem);
    const double e1 = (x[0] - mu) * std::sqrt(1.0 - phi * phi);
    double ss = e1 * e1;
    for (int t = 1; t < T; ++t) {
        const double e = x[t] - mu - phi * (x[t - 1] - mu);
        ss += e * e;
    }
    const double a = a0 + 0.5 * T, b = b0 + 0.5 * ss;
    std::gamma_distribution<double> g(a, 1.0 / b);   // precision ~ Gamma(a, rate=b)
    return 1.0 / g(rng);
}
//  mu | x, phi, sigma_eta^2 : Normal conjugate. Prior N(m0, s0_2).
double sample_mu(const arma::vec& x, double phi, double se2,
                 double m0, double s0_2, std::mt19937_64& rng) {
    const int T = static_cast<int>(x.n_elem);
    const double v1 = se2 / (1.0 - phi * phi);
    double P = 1.0 / s0_2 + 1.0 / v1;
    double num = m0 / s0_2 + x[0] / v1;
    const double w = (1.0 - phi) * (1.0 - phi) / se2;
    for (int t = 1; t < T; ++t) {
        const double r = x[t] - phi * x[t - 1];
        P += w;
        num += (1.0 - phi) * r / se2;
    }
    std::normal_distribution<double> nrm(num / P, std::sqrt(1.0 / P));
    return nrm(rng);
}
//  AR(1) log-likelihood as a function of phi (for the Metropolis step).
double ar1_loglik_phi(const arma::vec& x, double mu, double se2, double phi) {
    if (std::abs(phi) >= 1.0) return -std::numeric_limits<double>::infinity();
    const int T = static_cast<int>(x.n_elem);
    const double v1 = se2 / (1.0 - phi * phi);
    double ll = -0.5 * std::log(2.0 * M_PI * v1) - (x[0] - mu) * (x[0] - mu) / (2.0 * v1);
    for (int t = 1; t < T; ++t) {
        const double d = x[t] - (mu + phi * (x[t - 1] - mu));
        ll += -0.5 * std::log(2.0 * M_PI * se2) - d * d / (2.0 * se2);
    }
    return ll;   // flat U(-1,1) prior -> prior term cancels in the MH ratio
}
double sample_phi_mh(const arma::vec& x, double mu, double se2, double phi,
                     double prop_sd, std::mt19937_64& rng) {
    std::normal_distribution<double> q(0.0, prop_sd);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double phi_prop = phi + q(rng);
    if (std::abs(phi_prop) >= 1.0) return phi;
    const double ll_cur = ar1_loglik_phi(x, mu, se2, phi);
    const double ll_prop = ar1_loglik_phi(x, mu, se2, phi_prop);
    return (std::log(u(rng)) < ll_prop - ll_cur) ? phi_prop : phi;
}

double mean_v(const std::vector<double>& v) {
    double s = 0.0; for (double e : v) s += e; return s / v.size();
}
double sd_v(const std::vector<double>& v) {
    const double m = mean_v(v); double s = 0.0;
    for (double e : v) s += (e - m) * (e - m);
    return std::sqrt(s / (v.size() - 1));
}
double var_arma(const arma::vec& v) {
    const double m = arma::mean(v); double s = 0.0;
    for (arma::uword i = 0; i < v.n_elem; ++i) s += (v[i] - m) * (v[i] - m);
    return s / (v.n_elem - 1);
}

}  // namespace

int main() {
    std::printf("=== PoissonStateSpace example for particle_gibbs_block ===\n");

    // ---- 1. simulate from a KNOWN truth ----
    const double mu_true = 1.0, phi_true = 0.9, se_true = 0.3;
    const int T = 200;
    std::mt19937_64 sim(20260624ULL);
    arma::vec xtrue, y;
    simulate_poisson_ssm(mu_true, phi_true, se_true, T, sim, xtrue, y);
    const double var_y_obs = var_arma(y);

    // ---- 2. composite holding the path block; theta seeded AWAY from truth ----
    composite_block comp("PoissonStateSpace");
    comp.data().set("mu", sv1(0.0));
    comp.data().set("phi", sv1(0.5));
    comp.data().set("sigma_eta", sv1(0.5));
    comp.data().set("y", y);
    comp.data().declare_dependencies("x", {"mu", "phi", "sigma_eta", "y"});

    particle_gibbs_block_config cfg;
    cfg.name = "x"; cfg.T = T; cfg.state_dim = 1; cfg.n_particles = 64;
    cfg.ancestor_sampling = true; cfg.resampling = pg_resampling_scheme::SYSTEMATIC;
    cfg.init_sample = pois_init; cfg.transition_sample = pois_trans;
    cfg.obs_loglik = pois_obs; cfg.transition_logpdf = pois_tlpdf;
    comp.add_child(std::make_unique<particle_gibbs_block>(cfg));

    // ---- 3. full particle-Gibbs scan: path | theta, then theta | path ----
    const int burn = 1000, M = 4000;
    const double m0 = 0.0, s0_2 = 25.0;          // mu ~ N(0,5^2)
    const double a0 = 2.0, b0 = 0.05;            // sigma_eta^2 ~ IG(2,0.05)
    const double phi_prop_sd = 0.04;
    double mu = 0.0, phi = 0.5, se = 0.5;
    std::mt19937_64 rng(2026ULL);

    std::vector<double> mu_draw, phi_draw, se_draw;
    mu_draw.reserve(M); phi_draw.reserve(M); se_draw.reserve(M);
    arma::vec path_sum(T, arma::fill::zeros);
    int ppc_ge = 0;  // posterior-predictive: #{ Var(y_rep) >= Var(y_obs) }
    std::normal_distribution<double> n01(0.0, 1.0);

    for (int s = 0; s < burn + M; ++s) {
        comp.step(rng);                              // path | theta (the block)
        const arma::vec x = comp.data().get("x");    // current path (copy)

        const double se2 = sample_sigma2(x, mu, phi, a0, b0, rng);
        se = std::sqrt(se2);
        mu = sample_mu(x, phi, se2, m0, s0_2, rng);
        phi = sample_phi_mh(x, mu, se2, phi, phi_prop_sd, rng);
        comp.data().set("mu", sv1(mu));
        comp.data().set("phi", sv1(phi));
        comp.data().set("sigma_eta", sv1(se));

        if (s >= burn) {
            mu_draw.push_back(mu); phi_draw.push_back(phi); se_draw.push_back(se);
            path_sum += x;
            // posterior-predictive replicate from the current path draw
            arma::vec y_rep(T);
            for (int t = 0; t < T; ++t) {
                std::poisson_distribution<int> pd(std::exp(x[t]));
                y_rep[t] = static_cast<double>(pd(rng));
            }
            if (var_arma(y_rep) >= var_y_obs) ++ppc_ge;
        }
    }

    // ---- 4. checks ----
    const arma::vec path_hat = path_sum / static_cast<double>(M);
    double sse_block = 0.0, sse_naive = 0.0;
    for (int t = 0; t < T; ++t) {
        const double naive = std::log(y[t] + 0.5);   // per-time baseline, no smoothing
        sse_block += (path_hat[t] - xtrue[t]) * (path_hat[t] - xtrue[t]);
        sse_naive += (naive - xtrue[t]) * (naive - xtrue[t]);
    }
    const double rmse_block = std::sqrt(sse_block / T);
    const double rmse_naive = std::sqrt(sse_naive / T);

    const double mu_m = mean_v(mu_draw),  mu_s = sd_v(mu_draw);
    const double phi_m = mean_v(phi_draw), phi_s = sd_v(phi_draw);
    const double se_m = mean_v(se_draw),  se_s = sd_v(se_draw);
    const double ppc_p = static_cast<double>(ppc_ge) / M;

    const bool path_ok  = (rmse_block < 0.9 * rmse_naive);
    const bool theta_ok = (std::abs(mu_m - mu_true)   < 3.0 * mu_s) &&
                          (std::abs(phi_m - phi_true) < 3.0 * phi_s) &&
                          (std::abs(se_m - se_true)   < 3.0 * se_s);
    const bool ppc_ok   = (ppc_p > 0.05 && ppc_p < 0.95);
    const bool ok = path_ok && theta_ok && ppc_ok;

    std::printf("  path recovery : RMSE block=%.3f vs naive log(y+0.5)=%.3f  "
                "[block < 0.9*naive]  %s\n",
                rmse_block, rmse_naive, path_ok ? "PASS" : "FAIL");
    std::printf("  theta recovery: mu  %.3f+-%.3f (true %.2f)\n", mu_m, mu_s, mu_true);
    std::printf("                  phi %.3f+-%.3f (true %.2f)\n", phi_m, phi_s, phi_true);
    std::printf("                  se  %.3f+-%.3f (true %.2f)  [truth within 3 SD]  %s\n",
                se_m, se_s, se_true, theta_ok ? "PASS" : "FAIL");
    std::printf("  PPC (Var y)   : p=%.3f  [non-extreme 0.05-0.95]  %s\n",
                ppc_p, ppc_ok ? "PASS" : "FAIL");
    std::printf("=== %s ===\n", ok ? "EXAMPLE PASS" : "EXAMPLE FAILED");
    return ok ? 0 : 1;
}

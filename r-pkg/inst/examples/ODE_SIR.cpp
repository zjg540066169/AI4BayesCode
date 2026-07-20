// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  ODE_SIR.cpp
//
//  JOINT-NUTS rewrite of ODE_SIR.cpp. Same model, same priors, same posterior,
//  but (beta, gamma, sigma) are sampled by ONE joint_nuts_block instead of two
//  separate nuts_blocks alternated Gibbs-style.
//
//  Model (identical to ODE_SIR.cpp)
//  ---------------------------------
//      dS/dt = -beta * S * I / N
//      dI/dt =  beta * S * I / N  -  gamma * I
//      dR/dt =  gamma * I
//
//  Observation: I_obs(t_k) ~ LogNormal(log I(t_k; beta, gamma), sigma^2)
//
//  Priors:
//      beta  ~ half-Normal(0, 1)    =>  -0.5 * beta^2
//      gamma ~ half-Normal(0, 1)    =>  -0.5 * gamma^2
//      sigma ~ Jeffreys             =>  -log(sigma)
//
//  JOINT natural-scale log-density over x = [beta, gamma, sigma] (each model
//  term exactly ONCE — no double-counting):
//
//    log p(beta, gamma, sigma | data) =
//        -N * log(sigma) - 0.5 * SSE / sigma^2        (LogNormal likelihood)
//      - log(sigma)                                     (Jeffreys prior on sigma)
//      - 0.5 * (beta^2 + gamma^2)                       (half-Normal priors on beta, gamma)
//    = -(N+1) * log(sigma) - 0.5 * SSE / sigma^2 - 0.5 * (beta^2 + gamma^2)
//
//  where SSE = sum_k (log I_obs[k] - log I_pred[k])^2.
//
//  Gradient:
//      d/d beta  = central FD on ODE re-solve (mirrors ODE_SIR.cpp)
//      d/d gamma = central FD on ODE re-solve
//      d/d sigma = -(N+1)/sigma + SSE/sigma^3    (analytic)
//
//  ODE is solved ONCE per eval (center point); SSE is reused by both analytic
//  and FD gradient. FD for beta/gamma re-solves ODE at ±h (4 extra solves for
//  2 params).
//
//  Block decomposition
//  -------------------
//      (beta, gamma, sigma) : ONE joint_nuts_block, sub_params
//          [{ "theta", 2, POSITIVE }, { "sigma", 1, POSITIVE }]
//      (theta = [beta, gamma]; naming matches ODE_SIR.cpp data() layout.)
//
//  Cross-validated against ODE_SIR.cpp (same posterior, R-hat).
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("ODE_SIR")
//   set.seed(1)
//   beta_t <- 0.6; gamma_t <- 0.2; sigma_t <- 0.05         # ground truth
//   S0 <- 990; I0 <- 10; R0 <- 0                            # initial compartments at t=0
//   t_obs <- 0:14                                           # 15 obs times (t_obs[0]=0)
//   sir <- function(y, p) { N <- sum(y)
//     c(-p[1]*y[1]*y[2]/N, p[1]*y[1]*y[2]/N - p[2]*y[2], p[2]*y[2]) }
//   y <- c(S0, I0, R0); I_true <- numeric(15); I_true[1] <- I0   # RK4 ODE solve
//   for (k in 2:15) { h <- 1
//     k1<-sir(y,c(beta_t,gamma_t)); k2<-sir(y+h/2*k1,c(beta_t,gamma_t))
//     k3<-sir(y+h/2*k2,c(beta_t,gamma_t)); k4<-sir(y+h*k3,c(beta_t,gamma_t))
//     y <- y + h/6*(k1+2*k2+2*k3+k4); I_true[k] <- y[2] }
//   I_obs <- I_true * exp(sigma_t * rnorm(15))              # LogNormal noise
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(ODE_SIR, S0, I0, R0, t_obs, I_obs, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(ODE_SIR, S0, I0, R0, t_obs, I_obs, 7L, TRUE)   # S0,I0,R0,t_obs,I_obs,seed,keep_history
//   m$step(2500); str(m$get_current())
// @example:python
//   import numpy as np, AI4BayesCode
//   beta_t, gamma_t, sigma_t = 0.6, 0.2, 0.05              # ground truth
//   S0, I0, R0 = 990.0, 10.0, 0.0                          # initial compartments at t=0
//   t_obs = np.arange(15.0)                                # 15 obs times (t_obs[0]=0)
//   def sir(y, p):
//       N = y.sum()
//       return np.array([-p[0]*y[0]*y[1]/N, p[0]*y[0]*y[1]/N - p[1]*y[1], p[1]*y[1]])
//   y = np.array([S0, I0, R0]); I_true = np.empty(15); I_true[0] = I0  # RK4 ODE solve
//   for k in range(1, 15):
//       k1=sir(y,[beta_t,gamma_t]); k2=sir(y+0.5*k1,[beta_t,gamma_t])
//       k3=sir(y+0.5*k2,[beta_t,gamma_t]); k4=sir(y+k3,[beta_t,gamma_t])
//       y = y + (k1+2*k2+2*k3+k4)/6.0; I_true[k] = y[1]
//   rng = np.random.default_rng(1); I_obs = I_true * np.exp(sigma_t * rng.standard_normal(15))
//   Mod = AI4BayesCode.example("ODE_SIR")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.ODE_SIR(S0, I0, R0, t_obs, I_obs, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.ODE_SIR(S0, I0, R0, t_obs, I_obs, 7, True); m.step(2500); print(m.get_current())
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
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/ode_rk45.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;

namespace constraints = AI4BayesCode::constraints;
namespace ode         = AI4BayesCode::ode;

namespace {

// SIR RHS — identical to ODE_SIR.cpp.
inline arma::vec sir_rhs(double /*t*/, const arma::vec& y,
                         const arma::vec& theta_ode) {
    const double beta  = theta_ode[0];
    const double gamma = theta_ode[1];
    const double S = y[0], I = y[1], R = y[2];
    const double N = S + I + R;
    arma::vec dy(3);
    dy[0] = -beta * S * I / N;
    dy[1] =  beta * S * I / N - gamma * I;
    dy[2] =  gamma * I;
    return dy;
}

// Solve SIR and return I(t_obs). Reads y0 and t_obs from ctx.
inline arma::vec solve_sir_I(double beta, double gamma,
                             const block_context& ctx) {
    const arma::vec& y0    = ctx.at("y0");
    const arma::vec& t_obs = ctx.at("t_obs");
    const arma::vec theta_ode = {beta, gamma};
    arma::mat y_traj = ode::rk45(sir_rhs, y0, t_obs, theta_ode,
                                  /*rtol=*/1e-8, /*atol=*/1e-8);
    return y_traj.col(1);   // infected compartment
}

// ----------------------------------------------------------------------------
//  Joint natural-scale log-density for (beta, gamma, sigma).
//
//  theta_cat layout: [beta (nat[0]), gamma (nat[1]), sigma (nat[2])]
//
//  lp = -(N+1)*log(sigma) - 0.5*SSE/sigma^2 - 0.5*(beta^2 + gamma^2)
//
//  Gradient:
//    d/d beta, d/d gamma : central finite differences (re-solve ODE at ±h)
//    d/d sigma           : analytic = -(N+1)/sigma + SSE/sigma^3
//
//  joint_nuts_block adds the POSITIVE-slice Jacobians (+log(beta), +log(gamma),
//  +log(sigma)) internally. This function stays on the NATURAL scale.
// ----------------------------------------------------------------------------
double sir_joint_log_density(const arma::vec& theta_cat,
                             const block_context& ctx,
                             arma::vec* grad_nat) {
    const double beta  = theta_cat[0];
    const double gamma = theta_cat[1];
    const double sigma = theta_cat[2];

    // All three must be strictly positive (POSITIVE constraint ensures this
    // in practice, but guard defensively for the unconstrained path).
    if (!(beta > 0.0) || !(gamma > 0.0) || !(sigma > 0.0)) {
        if (grad_nat) grad_nat->set_size(3);
        return -std::numeric_limits<double>::infinity();
    }

    const arma::vec& I_obs = ctx.at("I_obs");
    const std::size_t N    = I_obs.n_elem;

    // --- Center ODE solve (reused by gradient too) -------------------------
    arma::vec I_pred;
    try {
        I_pred = solve_sir_I(beta, gamma, ctx);
    } catch (const std::exception&) {
        if (grad_nat) grad_nat->set_size(3);
        return -std::numeric_limits<double>::infinity();
    }

    // Guard: I_pred must be strictly positive everywhere.
    for (arma::uword k = 0; k < N; ++k) {
        if (!(I_pred[k] > 0.0)) {
            if (grad_nat) grad_nat->set_size(3);
            return -std::numeric_limits<double>::infinity();
        }
    }

    // Compute SSE = sum_k (log I_obs[k] - log I_pred[k])^2
    double sse = 0.0;
    for (arma::uword k = 0; k < N; ++k) {
        const double r = std::log(I_obs[k]) - std::log(I_pred[k]);
        sse += r * r;
    }

    const double Nd     = static_cast<double>(N);
    const double sigma2 = sigma * sigma;
    const double sigma3 = sigma2 * sigma;

    // Joint log-density (each term exactly once):
    //   likelihood: -N*log(sigma) - 0.5*SSE/sigma^2
    //   Jeffreys:   -log(sigma)
    //   theta prior: -0.5*(beta^2 + gamma^2)
    const double lp =
          -(Nd + 1.0) * std::log(sigma)
        - 0.5 * sse / sigma2
        - 0.5 * (beta * beta + gamma * gamma);

    if (!grad_nat) return lp;

    grad_nat->set_size(3);

    // --- Analytic gradient w.r.t. sigma ------------------------------------
    // d/d sigma = -(N+1)/sigma + SSE/sigma^3
    (*grad_nat)[2] = -(Nd + 1.0) / sigma + sse / sigma3;

    // --- Central FD gradient w.r.t. beta and gamma -------------------------
    // Mirrors ODE_SIR.cpp's theta_log_density gradient, but uses the FULL
    // joint log-density at ±h (so the SSE and prior are both included).
    const double h = 1e-5;

    // Helper: evaluate the full joint lp at a perturbed (beta, gamma).
    auto eval_lp_at = [&](double b, double g) -> double {
        if (b <= 0.0 || g <= 0.0) return -std::numeric_limits<double>::infinity();
        arma::vec I_p;
        try {
            I_p = solve_sir_I(b, g, ctx);
        } catch (...) {
            return -std::numeric_limits<double>::infinity();
        }
        double sse_p = 0.0;
        for (arma::uword k = 0; k < N; ++k) {
            if (!(I_p[k] > 0.0)) return -std::numeric_limits<double>::infinity();
            const double r = std::log(I_obs[k]) - std::log(I_p[k]);
            sse_p += r * r;
        }
        return -(Nd + 1.0) * std::log(sigma)
               - 0.5 * sse_p / sigma2
               - 0.5 * (b * b + g * g);
    };

    // d/d beta: perturb beta, keep gamma fixed.
    {
        const double b_plus  = beta + h;
        double       b_minus = beta - h;
        if (b_minus <= 0.0) b_minus = 1e-10;
        const double lp_plus  = eval_lp_at(b_plus,  gamma);
        const double lp_minus = eval_lp_at(b_minus, gamma);
        if (std::isfinite(lp_plus) && std::isfinite(lp_minus)) {
            (*grad_nat)[0] = (lp_plus - lp_minus) / (b_plus - b_minus);
        } else {
            (*grad_nat)[0] = 0.0;
        }
    }

    // d/d gamma: perturb gamma, keep beta fixed.
    {
        const double g_plus  = gamma + h;
        double       g_minus = gamma - h;
        if (g_minus <= 0.0) g_minus = 1e-10;
        const double lp_plus  = eval_lp_at(beta, g_plus);
        const double lp_minus = eval_lp_at(beta, g_minus);
        if (std::isfinite(lp_plus) && std::isfinite(lp_minus)) {
            (*grad_nat)[1] = (lp_plus - lp_minus) / (g_plus - g_minus);
        } else {
            (*grad_nat)[1] = 0.0;
        }
    }

    return lp;
}

} // anonymous namespace

class ODE_SIR : public AI4BayesCode::kernel_control_mixin<ODE_SIR> {
    friend class AI4BayesCode::kernel_control_mixin<ODE_SIR>;
public:
    ODE_SIR(double S0, double I0, double R0,
                  const arma::vec& t_obs,
                  const arma::vec& I_obs,
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
          impl_(std::make_unique<composite_block>("ODE_SIR")),
          keep_history_(keep_history)
    {
        if (!(S0 > 0) || !(I0 > 0) || !(R0 >= 0))
            throw std::runtime_error("S0, I0 must be > 0, R0 must be >= 0");
        if (t_obs.n_elem != I_obs.n_elem)
            throw std::runtime_error("t_obs and I_obs must have same length");
        if (t_obs.n_elem < 2)
            throw std::runtime_error("need at least 2 observation times");
        if (t_obs[0] != 0.0)
            throw std::runtime_error("t_obs[0] must be 0 (initial time)");
        for (arma::uword i = 1; i < t_obs.n_elem; ++i)
            if (t_obs[i] <= t_obs[i - 1])
                throw std::runtime_error("t_obs must be strictly increasing");
        for (arma::uword i = 0; i < I_obs.n_elem; ++i)
            if (!(I_obs[i] > 0))
                throw std::runtime_error("I_obs must be strictly positive");

        // Fixed data.
        impl_->data().set("y0",    arma::vec{S0, I0, R0});
        impl_->data().set("t_obs", t_obs);
        impl_->data().set("I_obs", I_obs);

        // Initial natural-scale parameter values.
        const arma::vec theta_init = {0.3, 0.1};   // [beta, gamma]
        const double    sigma_init = 0.1;

        impl_->data().set("theta", theta_init);
        impl_->data().set("sigma", arma::vec{sigma_init});

        // y_rep and I_traj (identical to ODE_SIR.cpp).
        impl_->data().set("y_rep",  arma::vec(t_obs.n_elem, arma::fill::zeros));
        impl_->data().set("I_traj", arma::vec(t_obs.n_elem, arma::fill::zeros));

        // ---- Dependency DAG -----------------------------------------------
        // The joint block's dependencies are keyed under the JOINT BLOCK NAME
        // ("theta_sigma_joint"). They are the union of what the original two
        // blocks read from data() — only the true external data() inputs:
        //   theta block read: y0, t_obs, I_obs  (sigma came from ctx/block state)
        //   sigma block read: y0, t_obs, I_obs  (theta came from ctx/block state)
        // Together: {y0, t_obs, I_obs}. theta and sigma now come from the
        // joint block's own concatenated draw, not from data() cross-reads.
        impl_->data().declare_dependencies(
            "theta_sigma_joint", {"y0", "t_obs", "I_obs"});

        // Fixed data inputs for predict_at BFS.
        impl_->data().declare_data_input("y0");
        impl_->data().declare_data_input("t_obs");

        // ---- Predict DAG (same structure as ODE_SIR.cpp) ------------------
        //   y0, t_obs, theta -> I_traj  (deterministic ODE solve)
        //   I_traj, sigma    -> y_rep   (LogNormal noise draw)
        // declare_predict_edges and declare_context_edges are keyed by
        // SUB-PARAM NAME (unchanged from the original).
        impl_->data().declare_predict_edges("y0",     {"I_traj"});
        impl_->data().declare_predict_edges("t_obs",  {"I_traj"});
        impl_->data().declare_predict_edges("theta",  {"I_traj"});
        impl_->data().declare_predict_edges("I_traj", {"y_rep"});
        impl_->data().declare_predict_edges("sigma",  {"y_rep"});

        // I_traj refresher: identical to ODE_SIR.cpp.
        impl_->data().register_refresher(
            "I_traj",
            [](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& y0    = d.get("y0");
                const arma::vec& t_obs = d.get("t_obs");
                const arma::vec& theta = d.get("theta");
                arma::mat y_traj = ode::rk45(sir_rhs, y0, t_obs, theta,
                                             1e-8, 1e-8);
                return y_traj.col(1);
            });
        // Keyed under the JOINT BLOCK NAME (the joint block updates theta+sigma
        // together; composite_block calls refresh_derived_for(block_name)).
        impl_->data().declare_invalidates("theta_sigma_joint", {"I_traj"});

        // y_rep stochastic refresher: identical to ODE_SIR.cpp.
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                const arma::vec& I_traj = d.get("I_traj");
                const double sigma = d.get("sigma")[0];
                std::normal_distribution<double> nd(0.0, 1.0);
                arma::vec I_rep(I_traj.n_elem);
                for (arma::uword k = 0; k < I_traj.n_elem; ++k)
                    I_rep[k] = I_traj[k] * std::exp(sigma * nd(rng));
                return I_rep;
            });

        // ---- ONE joint_nuts_block over (theta=[beta,gamma], sigma) --------
        {
            joint_nuts_block_config cfg;
            cfg.name = "theta_sigma_joint";

            // sub_params: theta (beta, gamma) = POSITIVE x 2; sigma = POSITIVE x 1.
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "theta", 2, joint_constraint::POSITIVE });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "sigma", 1, joint_constraint::POSITIVE });

            // initial_cat on the NATURAL scale: [beta, gamma, sigma].
            cfg.initial_cat = arma::vec{ theta_init[0], theta_init[1], sigma_init };

            cfg.log_density_grad = &sir_joint_log_density;

            // ODE curvature is strong; give the joint block generous warmup.
            // Diagonal metric required: beta/gamma live in (0,2) while sigma
            // lives in (0, 0.5), very different scales.
            cfg.initial_step_size   = 0.01;
            cfg.n_warmup_first_call = 800;
            cfg.use_diagonal_metric = true;

            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        if (n_steps < 0) throw std::runtime_error("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        const arma::vec& theta = impl_->data().get("theta");
        AI4BayesCode::state_map out;
        out["beta"]  = arma::vec{theta[0]};
        out["gamma"] = arma::vec{theta[1]};
        out["sigma"] = impl_->data().get("sigma");
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto& jblk = dynamic_cast<joint_nuts_block&>(impl_->child(0));
        arma::vec cat_new = jblk.current();   // [beta, gamma, sigma]
        bool touched = false;

        auto it_beta = params.find("beta");
        if (it_beta != params.end()) {
            const double b = it_beta->second[0];
            if (!(b > 0.0)) throw std::runtime_error("beta must be > 0");
            cat_new[0] = b; touched = true;
        }
        auto it_gamma = params.find("gamma");
        if (it_gamma != params.end()) {
            const double g = it_gamma->second[0];
            if (!(g > 0.0)) throw std::runtime_error("gamma must be > 0");
            cat_new[1] = g; touched = true;
        }
        auto it_sigma = params.find("sigma");
        if (it_sigma != params.end()) {
            const double s = it_sigma->second[0];
            if (!(s > 0.0)) throw std::runtime_error("sigma must be > 0");
            cat_new[2] = s; touched = true;
        }
        if (touched) {
            jblk.set_current(cat_new);
            impl_->data().set("theta", arma::vec{cat_new[0], cat_new[1]});
            impl_->data().set("sigma", arma::vec{cat_new[2]});
        }
    }

    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty()) {
            throw std::runtime_error(
                "ODE_SIR.predict_at() takes an empty map/list "
                "(returns I_rep at training t_obs)");
        }
        AI4BayesCode::history_map out;

        if (!keep_history_) {
            block_context replaced;
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result) {
                arma::mat m(1, kv.second.n_elem);
                for (std::size_t j = 0; j < kv.second.n_elem; ++j)
                    m(0, j) = kv.second[j];
                out.emplace(kv.first, std::move(m));
            }
            return out;
        }

        // History mode: theta and sigma are sub-outputs of the joint block
        // (keyed by sub-param name in get_history(), NOT by block name).
        // Compute y_rep manually per draw (cf. IRT1PL_joint.cpp, GaussianLocationScale.cpp).
        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& theta_hist = hist.at("theta");   // n_draws x 2: [beta, gamma]
        const arma::mat& sigma_hist = hist.at("sigma");   // n_draws x 1
        const std::size_t n_draws   = theta_hist.n_rows;

        const arma::vec y0_tr    = impl_->data().get("y0");
        const arma::vec t_obs_tr = impl_->data().get("t_obs");
        const arma::vec& I_obs   = impl_->data().get("I_obs");
        const std::size_t n_obs  = t_obs_tr.n_elem;

        std::unordered_map<std::string, std::vector<arma::vec>> collected;
        for (std::size_t d = 0; d < n_draws; ++d) {
            // Inject sub-param draws so the predict DAG BFS can run.
            // theta and sigma are keyed by sub-param name (not block name).
            block_context replaced;
            replaced["y0"]    = y0_tr;
            replaced["t_obs"] = t_obs_tr;
            replaced["theta"] = theta_hist.row(d).t();
            replaced["sigma"] = arma::vec{sigma_hist(d, 0)};
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result)
                collected[kv.first].push_back(kv.second);
        }

        for (const auto& kv : collected) {
            if (kv.second.empty()) continue;
            const std::size_t dim = kv.second[0].n_elem;
            arma::mat m(n_draws, dim);
            for (std::size_t d = 0; d < n_draws; ++d)
                for (std::size_t j = 0; j < dim; ++j)
                    m(d, j) = kv.second[d][j];
            out.emplace(kv.first, std::move(m));
        }
        return out;
    }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1) {
        if (n < 0) ai4b::stop("readapt_NUTS: n must be non-negative");
        impl_->readapt_NUTS(static_cast<std::size_t>(n), reset, readapt_rng_, max_tree_depth < 0 ? std::size_t(0) : static_cast<std::size_t>(max_tree_depth));
    }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
};

// ============================================================================
//  Host-language module declarations.
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(ODE_SIR_module) {
    Rcpp::class_<ODE_SIR>("ODE_SIR")
        .constructor<double, double, double, arma::vec, arma::vec, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<double, double, double, arma::vec, arma::vec, int, bool>(
            "Joint-NUTS SIR ODE model. Args: S0, I0, R0 (initial compartments "
            "at t=0), t_obs (strictly increasing, t_obs[0]=0), I_obs (observed "
            "infected counts > 0 at each t_obs), rng_seed, keep_history. "
            "Infers (beta, gamma, sigma) jointly via one joint_nuts_block. "
            "Priors: beta,gamma ~ half-Normal(0,1); sigma ~ Jeffreys. "
            "Gradient w.r.t. beta/gamma via central FD; w.r.t. sigma analytic.")
        .method("step", (void (ODE_SIR::*)())    &ODE_SIR::step, "Run one sweep.")
        .method("step", (void (ODE_SIR::*)(int)) &ODE_SIR::step, "Run n sweeps.")
        .method("get_current",  &ODE_SIR::get_current)
        .method("set_current",  &ODE_SIR::set_current)
        .method("predict_at",   &ODE_SIR::predict_at)
        .method("get_dag",      &ODE_SIR::get_dag)
        .method("get_history",  &ODE_SIR::get_history)
        .method("readapt_NUTS", &ODE_SIR::readapt_NUTS)
        AI4BAYESCODE_BIND_KERNEL_CONTROL(ODE_SIR);
}
#endif  // AI4BAYESCODE_RCPP_MODULE

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"

PYBIND11_MODULE(ODE_SIR, m) {
    AI4BayesCode::register_ai4bayescode_types(m);

    pybind11::class_<ODE_SIR>(m, "ODE_SIR")
        .def(pybind11::init<double, double, double, arma::vec, arma::vec,
                            int, bool>(),
             pybind11::arg("S0"), pybind11::arg("I0"), pybind11::arg("R0"),
             pybind11::arg("t_obs"), pybind11::arg("I_obs"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false,
             "Joint-NUTS SIR ODE model. Infers (beta, gamma, sigma) in one "
             "joint_nuts_block. Priors: beta,gamma ~ half-Normal(0,1); "
             "sigma ~ Jeffreys. Gradient w.r.t. beta/gamma via central FD; "
             "w.r.t. sigma analytic.")
        .def("step", (void (ODE_SIR::*)())    &ODE_SIR::step, "Run one sweep.")
        .def("step", (void (ODE_SIR::*)(int)) &ODE_SIR::step, pybind11::arg("n_steps"))
        .def("get_current",  &ODE_SIR::get_current)
        .def("set_current",  &ODE_SIR::set_current, pybind11::arg("params"))
        .def("predict_at",   &ODE_SIR::predict_at, pybind11::arg("new_data"))
        .def("get_dag",      &ODE_SIR::get_dag)
        .def("get_history",  &ODE_SIR::get_history)
        .def("readapt_NUTS", &ODE_SIR::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false, pybind11::arg("max_tree_depth") = -1)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(ODE_SIR);
}
#endif  // AI4BAYESCODE_PYBIND_MODULE

// ============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (compiled only when NOT a host module).
// ============================================================================
#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulates an SIR epidemic from a KNOWN (beta, gamma, sigma):
//    - solve the SIR ODE forward with the rk45 integrator to get I(t),
//    - corrupt with LogNormal(log I(t), sigma^2) noise to get I_obs,
//  then fits the model via the joint-NUTS block and checks that the posterior
//  means recover (beta, gamma, sigma). The recovered fit is also compared to a
//  naive baseline (the prior means under the half-Normal / fixed sigma guess)
//  to confirm the data actually informs the estimate.
//==============================================================================
#include <cstdio>

int main() {
    // ---- Known ground truth -------------------------------------------------
    const double beta_true  = 0.6;
    const double gamma_true = 0.2;
    const double sigma_true = 0.05;     // small LogNormal observation noise

    const double S0 = 990.0, I0 = 10.0, R0 = 0.0;

    // Observation times: t=0 plus 14 daily-ish observations.
    const std::size_t n_obs = 15;
    arma::vec t_obs(n_obs);
    for (std::size_t k = 0; k < n_obs; ++k)
        t_obs[k] = static_cast<double>(k);   // 0,1,2,...,14

    // ---- Simulate the true infected trajectory via the ODE solver ----------
    auto sir_rhs_sim = [](double /*t*/, const arma::vec& y,
                          const arma::vec& th) -> arma::vec {
        const double b = th[0], g = th[1];
        const double S = y[0], I = y[1], R = y[2];
        const double N = S + I + R;
        arma::vec dy(3);
        dy[0] = -b * S * I / N;
        dy[1] =  b * S * I / N - g * I;
        dy[2] =  g * I;
        return dy;
    };
    const arma::vec y0_sim{S0, I0, R0};
    const arma::vec th_sim{beta_true, gamma_true};
    arma::mat traj = AI4BayesCode::ode::rk45(sir_rhs_sim, y0_sim, t_obs, th_sim,
                                             1e-9, 1e-9);
    arma::vec I_true = traj.col(1);   // true infected at each t_obs

    // ---- Add LogNormal observation noise -----------------------------------
    std::mt19937_64 sim_rng(20260621ULL);
    std::normal_distribution<double> noise(0.0, 1.0);
    arma::vec I_obs(n_obs);
    for (std::size_t k = 0; k < n_obs; ++k)
        I_obs[k] = I_true[k] * std::exp(sigma_true * noise(sim_rng));

    // ---- Fit the model ------------------------------------------------------
    ODE_SIR model(S0, I0, R0, t_obs, I_obs, /*rng_seed=*/7,
                  /*keep_history=*/false);

    model.step(1000);   // warmup (joint block also self-warms 800 on first call)

    double beta_bar = 0.0, gamma_bar = 0.0, sigma_bar = 0.0;
    const int M = 2000;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        beta_bar  += cur.at("beta")[0];
        gamma_bar += cur.at("gamma")[0];
        sigma_bar += cur.at("sigma")[0];
    }
    beta_bar  /= static_cast<double>(M);
    gamma_bar /= static_cast<double>(M);
    sigma_bar /= static_cast<double>(M);

    // ---- Report -------------------------------------------------------------
    std::printf("ODE_SIR demo (joint-NUTS over beta, gamma, sigma)\n");
    std::printf("  beta : post mean %.4f   (truth %.3f)\n", beta_bar,  beta_true);
    std::printf("  gamma: post mean %.4f   (truth %.3f)\n", gamma_bar, gamma_true);
    std::printf("  sigma: post mean %.4f   (truth %.3f)\n", sigma_bar, sigma_true);

    // Recovery tolerances (relative for beta/gamma; absolute floor for sigma).
    const double beta_err  = std::abs(beta_bar  - beta_true);
    const double gamma_err = std::abs(gamma_bar - gamma_true);
    const double sigma_err = std::abs(sigma_bar - sigma_true);

    // Naive baseline: prior mean for the half-Normal(0,1) on beta/gamma is
    // sqrt(2/pi) ~ 0.7979 — the no-data guess. The posterior must beat it.
    const double prior_mean = std::sqrt(2.0 / M_PI);
    const double naive_beta_err  = std::abs(prior_mean - beta_true);
    const double naive_gamma_err = std::abs(prior_mean - gamma_true);

    const bool beats_naive =
        beta_err  < naive_beta_err &&
        gamma_err < naive_gamma_err;

    const bool ok =
        beta_err  < 0.08 &&        // ~13% of truth 0.6
        gamma_err < 0.04 &&        // ~20% of truth 0.2
        sigma_err < 0.05 &&        // sigma small; loose absolute tol
        beats_naive;

    std::printf("  |err| beta=%.4f gamma=%.4f sigma=%.4f  "
                "(naive prior-mean err beta=%.4f gamma=%.4f)\n",
                beta_err, gamma_err, sigma_err,
                naive_beta_err, naive_gamma_err);
    std::printf("%s\n",
                ok ? "[demo PASS] joint-NUTS recovers (beta, gamma, sigma) "
                     "and beats the prior-mean baseline"
                   : "[demo FAIL] posterior did not recover within tolerance");
    return ok ? 0 : 1;
}
#endif  // standalone demo guard

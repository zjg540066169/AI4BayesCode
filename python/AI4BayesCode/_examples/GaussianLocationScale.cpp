// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  GaussianLocationScale.cpp
//
//  JOINT-NUTS rewrite of GaussianLocationScale.cpp. Same model, same priors,
//  same posterior — but mu and sigma are sampled by ONE joint_nuts_block
//  instead of two separate single nuts_blocks updated Gibbs-style. This is the
//  "collapse a bunch of single nuts_blocks into one joint_nuts_block" refactor.
//
//  Model (identical to GaussianLocationScale.cpp)
//  ---------------------------------------------
//      y_i | mu, sigma  ~ Normal(mu, sigma^2),      i = 1..N
//      mu               ~ Normal(0, 100^2)
//      sigma            ~ Jeffreys, p(sigma) ∝ 1/sigma
//
//  Block decomposition
//  -------------------
//      (mu, sigma) : ONE joint_nuts_block, sub_params
//                    [{ "mu", 1, REAL }, { "sigma", 1, POSITIVE }].
//      The joint log-density is the FULL joint log p(mu, sigma | y) on the
//      NATURAL scale (the sum of every model term, each appearing exactly
//      once); joint_nuts_block adds the POSITIVE-slice Jacobian (+log sigma)
//      internally. This matches GaussianLocationScale.cpp's per-conditional
//      decomposition exactly:
//        - mu conditional:    -0.5 sum_sq/sigma^2 - 0.5 mu^2/prior_var
//        - sigma conditional: -(N+1) log sigma   - 0.5 sum_sq/sigma^2
//        joint = likelihood (-N log sigma - sum_sq/(2 sigma^2))
//              + mu prior   (-0.5 mu^2/prior_var)
//              + Jeffreys   (-log sigma)
//              = -(N+1) log sigma - 0.5 sum_sq/sigma^2 - 0.5 mu^2/prior_var.
//
//  This file is for cross-validation against GaussianLocationScale.cpp (it uses
//  a distinct class/module name so both can be loaded in one R session). Once
//  the two posteriors are confirmed equal (R-hat + means/sd/ESS), this becomes
//  the canonical GaussianLocationScale.
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("GaussianLocationScale")
//   set.seed(42); N <- 200
//   y <- rnorm(N, mean = 3, sd = 2)                # DGP: true mu=3, sigma=2
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(GaussianLocationScale, y, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(GaussianLocationScale, y, 7L, TRUE)   # (y, rng_seed, keep_history)
//   m$step(2500); str(m$get_current())             # mu_hat ~ 3, sigma_hat ~ 2
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(42); N = 200
//   y = rng.normal(3.0, 2.0, N)                    # DGP: true mu=3, sigma=2
//   Mod = AI4BayesCode.example("GaussianLocationScale")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.GaussianLocationScale(y, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.GaussianLocationScale(y, 7, True); m.step(2500); print(m.get_current())
// @example:end
// ============================================================================

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
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;

namespace constraints = AI4BayesCode::constraints;

namespace {

// ----------------------------------------------------------------------------
//  Joint natural-scale log-density of (mu, sigma) given y.
//
//  theta_cat = [ mu, sigma ]  (NATURAL scale; sigma > 0 guaranteed by the
//  POSITIVE sub-param transform inside joint_nuts_block).
//
//  lp(mu, sigma) = -(N+1) log(sigma) - 0.5 sum_sq/sigma^2 - 0.5 mu^2/prior_var
//  d/d mu        =  sum_resid/sigma^2 - mu/prior_var
//  d/d sigma     = -(N+1)/sigma + sum_sq/sigma^3
//
//  sum_resid = sum(y - mu),  sum_sq = sum((y - mu)^2).
//
//  The POSITIVE Jacobian (+log sigma) for the sigma slice is added by
//  joint_nuts_block; this function must NOT include it (exactly as the
//  single-block version's sigma_natural_log_density omits it).
// ----------------------------------------------------------------------------
double location_scale_joint_log_density(const arma::vec& theta_cat,
                                        const block_context& ctx,
                                        arma::vec* grad_nat) {
    const double mu    = theta_cat[0];
    const double sigma = theta_cat[1];
    const arma::vec& y = ctx.at("y");

    const double sigma2    = sigma * sigma;
    const double sigma3    = sigma2 * sigma;
    const double prior_var = 100.0 * 100.0;
    const double N         = static_cast<double>(y.n_elem);

    double sum_resid = 0.0;
    double sum_sq    = 0.0;
    for (std::size_t i = 0; i < y.n_elem; ++i) {
        const double r = y[i] - mu;
        sum_resid += r;
        sum_sq    += r * r;
    }

    const double lp =
        -(N + 1.0) * std::log(sigma)
        - 0.5 * sum_sq / sigma2
        - 0.5 * mu * mu / prior_var;

    if (grad_nat) {
        grad_nat->set_size(2);
        (*grad_nat)[0] = sum_resid / sigma2 - mu / prior_var;        // d/d mu
        (*grad_nat)[1] = -(N + 1.0) / sigma + sum_sq / sigma3;       // d/d sigma
    }
    return lp;
}

} // anonymous namespace

class GaussianLocationScale {
public:
    GaussianLocationScale(const arma::vec& y,
                               int    rng_seed,
                               bool   keep_history = false)
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
          impl_(std::make_unique<composite_block>("GaussianLocationScale")),
          keep_history_(keep_history)
    {
        const double mu_init    = arma::mean(y);
        const double sigma_init = std::max(arma::stddev(y), 1e-3);

        impl_->data().set("y",     y);
        impl_->data().set("mu",    arma::vec{mu_init});
        impl_->data().set("sigma", arma::vec{sigma_init});

        // The joint block's dependencies are keyed under the JOINT BLOCK NAME
        // ("mu_sigma_joint"), NOT the sub-param names — the composite builds the
        // block's context by looking up dependencies under the block name (cf.
        // IRT1PL_joint). It is the union of what mu and sigma each read from
        // data(): just y (mu, sigma come from the block's own concatenated draw).
        impl_->data().declare_dependencies("mu_sigma_joint", {"y"});

        impl_->data().declare_predict_edges("mu",    {"y_rep"});
        impl_->data().declare_predict_edges("sigma", {"y_rep"});
        impl_->data().set("y_rep", arma::vec(y.n_elem, arma::fill::zeros));

        const std::size_t N_obs = y.n_elem;
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [N_obs](const AI4BayesCode::shared_data_t& d,
                    std::mt19937_64& rng) {
                const double mu = d.get("mu")[0];
                const double s  = d.get("sigma")[0];
                std::normal_distribution<double> norm(0.0, 1.0);
                arma::vec y_rep(N_obs);
                for (std::size_t i = 0; i < N_obs; ++i)
                    y_rep[i] = mu + s * norm(rng);
                return y_rep;
            });

        // ---- ONE joint_nuts_block over (mu, sigma) -----------------------
        {
            joint_nuts_block_config cfg;
            cfg.name = "mu_sigma_joint";
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "mu", 1, joint_constraint::REAL });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "sigma", 1, joint_constraint::POSITIVE });
            // initial_cat is NATURAL-scale: [mu_init, sigma_init].
            cfg.initial_cat = arma::vec{ mu_init, sigma_init };
            cfg.log_density_grad = &location_scale_joint_log_density;
            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["mu"]    = impl_->data().get("mu");
        out["sigma"] = impl_->data().get("sigma");
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto& jblk = dynamic_cast<joint_nuts_block&>(impl_->child(0));
        arma::vec cat_new = jblk.current();    // [mu, sigma]
        bool touched = false;

        auto it_mu = params.find("mu");
        if (it_mu != params.end()) { cat_new[0] = it_mu->second[0]; touched = true; }
        auto it_sg = params.find("sigma");
        if (it_sg != params.end()) {
            const double sg = it_sg->second[0];
            if (sg <= 0.0) throw std::runtime_error("sigma must be strictly positive");
            cat_new[1] = sg; touched = true;
        }
        if (touched) {
            jblk.set_current(cat_new);
            impl_->data().set("mu",    arma::vec{cat_new[0]});
            impl_->data().set("sigma", arma::vec{cat_new[1]});
        }
    }

    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty()) {
            throw std::runtime_error(
                "GaussianLocationScale has no covariate inputs. "
                "predict_at() takes an empty map/list.");
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

        // History mode: mu, sigma are sub-outputs of the joint block (not block
        // names), so compute y_rep manually per draw (cf. IRT1PL_joint).
        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& mu_hist    = hist.at("mu");      // n_draws x 1
        const arma::mat& sigma_hist = hist.at("sigma");   // n_draws x 1
        const std::size_t n_draws = mu_hist.n_rows;
        const std::size_t N = impl_->data().get("y").n_elem;

        arma::mat yrep(n_draws, N);
        std::normal_distribution<double> norm(0.0, 1.0);
        for (std::size_t d = 0; d < n_draws; ++d) {
            const double mu = mu_hist(d, 0);
            const double s  = sigma_hist(d, 0);
            for (std::size_t i = 0; i < N; ++i)
                yrep(d, i) = mu + s * norm(predict_rng_);
        }
        out.emplace("y_rep", std::move(yrep));
        return out;
    }

    AI4BayesCode::dag_info      get_dag()     const { return impl_->get_dag(); }
    AI4BayesCode::history_map   get_history() const { return impl_->get_history(); }

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

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(GaussianLocationScale_module) {
    Rcpp::class_<GaussianLocationScale>("GaussianLocationScale")
        .constructor<arma::vec, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::vec, int, bool>(
            "Construct with data vector y, RNG seed (0 = random), and "
            "keep_history (record every draw; default FALSE).")
        .method("step", (void (GaussianLocationScale::*)())    &GaussianLocationScale::step, "Run one sweep.")
        .method("step", (void (GaussianLocationScale::*)(int)) &GaussianLocationScale::step, "Run n sweeps.")
        .method("get_current",  &GaussianLocationScale::get_current)
        .method("set_current",  &GaussianLocationScale::set_current)
        .method("predict_at",   &GaussianLocationScale::predict_at)
        .method("get_dag",      &GaussianLocationScale::get_dag)
        .method("get_history",  &GaussianLocationScale::get_history)
        .method("readapt_NUTS", &GaussianLocationScale::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(GaussianLocationScale, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<GaussianLocationScale>(m, "GaussianLocationScale")
        .def(pybind11::init<arma::vec, int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step", (void (GaussianLocationScale::*)())    &GaussianLocationScale::step, "Run one sweep.")
        .def("step", (void (GaussianLocationScale::*)(int)) &GaussianLocationScale::step,  pybind11::arg("n_steps"))
        .def("get_current",  &GaussianLocationScale::get_current)
        .def("set_current",  &GaussianLocationScale::set_current, pybind11::arg("params"))
        .def("predict_at",   &GaussianLocationScale::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &GaussianLocationScale::get_dag)
        .def("get_history",  &GaussianLocationScale::get_history)
        .def("readapt_NUTS", &GaussianLocationScale::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false, pybind11::arg("max_tree_depth") = -1);
}
#endif

// ============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//  Active ONLY when built as a plain binary (neither module macro defined).
//  Simulates Gaussian data from a known (mu, sigma), fits via the joint-NUTS
//  block, and checks posterior-mean recovery. This is the SAME DGP mirrored
//  into the @example:R / @example:python header blocks above.
// ============================================================================
#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>
int main() {
    const std::size_t N          = 200;
    const double      mu_true    = 3.0;
    const double      sigma_true = 2.0;

    std::mt19937_64 sim_rng(42);
    std::normal_distribution<double> gen(mu_true, sigma_true);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) y[i] = gen(sim_rng);

    GaussianLocationScale model(y, /*rng_seed=*/7, /*keep_history=*/false);
    model.step(500);   // warmup

    double mu_bar = 0.0, sigma_bar = 0.0;
    const int M = 2000;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        mu_bar    += cur.at("mu")[0];
        sigma_bar += cur.at("sigma")[0];
    }
    mu_bar    /= static_cast<double>(M);
    sigma_bar /= static_cast<double>(M);

    std::printf("GaussianLocationScale demo: mu_hat=%.3f (truth %.1f)  "
                "sigma_hat=%.3f (truth %.1f)\n",
                mu_bar, mu_true, sigma_bar, sigma_true);
    const bool ok = std::abs(mu_bar - mu_true)    < 0.3 &&
                    std::abs(sigma_bar - sigma_true) < 0.3;
    std::printf("%s\n", ok ? "[demo PASS] joint-NUTS recovers (mu, sigma)"
                           : "[demo FAIL]");
    return ok ? 0 : 1;
}
#endif

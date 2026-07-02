// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  BSplineRegression.cpp
//
//  REFERENCE TEMPLATE for Bayesian penalized B-spline regression with
//  Gaussian observations. This is the canonical "smoothing spline /
//  P-spline" pattern used by brms, mgcv, and Wood 2017 §5.2.
//
//  Why an example
//  --------------
//  Without the joint dense-metric pattern, agents writing penalized
//  splines hit two pathologies:
//   1. (Intercept, b_smooth) RIDGE: the Intercept and the basis-coefficient
//      mean are confounded; identity-metric NUTS step size shrinks to the
//      narrow ridge direction and ESS collapses 100x.
//   2. (sds, zs) FUNNEL: smoothing scale times non-centered z is the
//      classic Neal funnel; identity-metric NUTS misses the wide-zs /
//      small-sds part of posterior mass.
//
//  Architectural choices DEMONSTRATED here (must transfer)
//  -------------------------------------------------------
//   1. EVERY positive scalar (sds, sigma) is reparameterized to log scale
//      and treated as a REAL parameter -- this unlocks `joint_nuts_block`
//      (real-only) and its dense-metric pilot phase.
//   2. ALL real parameters in ONE `joint_nuts_block` with `use_dense_metric
//      = true`. The Welford pilot covariance learns the (Intercept, b_smooth)
//      ridge AND the (sds, zs) funnel in a single warmup phase.
//   3. log|J| terms manually added inside the log-density (block adds 0
//      Jacobian for REAL sub-params; user is responsible).
//
//  Model
//  -----
//      y_n              ~ Normal(Intercept + Bsp_n . s,  sigma)        n=1..N
//      s_k              = sds * z_k                       (non-centered,
//                                                          k = 1..K_s)
//      z_k              ~ Normal(0, 1)
//      Intercept        ~ Normal(0, 10)
//      sds              ~ Half-Normal(0, sd(y))            (smoothing scale)
//      sigma (residual) ~ Jeffreys: p(sigma) oc 1/sigma
//
//  Bsp is a precomputed basis matrix (N x K_s). Caller supplies it via the
//  constructor. K_s is the smoothing-spline dimension (e.g. 8-20). See
//  the smoke test for a synthetic generator (truncated power basis on an
//  evenly-spaced knot grid).
//
//  Block decomposition
//  -------------------
//   ONE `joint_nuts_block` with sub-params:
//      (Intercept, log_sds, log_sigma, z[K_s])
//   total dim = 3 + K_s, all REAL on unconstrained scale, dense metric.
//
//  Synthetic test fixture (1-D, generic)
//  -------------------------------------
//   x in [0, 1], N=100, K_s=10, y = smooth(x) + N(0, 0.3) where
//   smooth(x) = 0.4*sin(8*x) + 0.2*x^2. Generic synthetic, no applied
//   domain.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("BSplineRegression")
//   set.seed(123); N <- 100L; K_s <- 10L                       # data + basis dim
//   x <- (0:(N-1))/(N-1); knots <- (1:K_s)/(K_s+1)             # grid + interior knots
//   Bsp <- outer(x, knots, function(a,b) pmax(a-b, 0))         # truncated-power basis (N x K_s)
//   y <- 0.4*sin(8*x) + 0.2*x^2 + rnorm(N, 0, 0.3)             # smooth(x) + N(0,0.3)
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(BSplineRegression, y, Bsp, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(BSplineRegression, y, Bsp, 7L, TRUE)              # y, basis, seed, keep_history
//   m$step(2500); str(m$get_current())
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(123); N, K_s = 100, 10
//   x = np.arange(N)/(N-1); knots = (np.arange(1,K_s+1))/(K_s+1)
//   Bsp = np.maximum(x[:,None] - knots[None,:], 0.0)           # truncated-power basis (N x K_s)
//   y = 0.4*np.sin(8*x) + 0.2*x**2 + rng.normal(0, 0.3, N)     # smooth(x) + N(0,0.3)
//   Mod = AI4BayesCode.source("BSplineRegression.cpp")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.BSplineRegression(y, Bsp, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.BSplineRegression(y, Bsp, 7, True); m.step(2500); print(m.get_current())
// @example:end

// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#ifdef AI4BAYESCODE_RCPP_MODULE
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#ifdef AI4BAYESCODE_RCPP_MODULE
#include <Rcpp.h>
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#endif

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// Joint log-density on the UNCONSTRAINED REAL scale.
// theta_cat layout: (Intercept, log_sds, log_sigma, z[K_s])
double joint_log_density(const arma::vec& theta_cat,
                         const block_context& ctx,
                         arma::vec* grad) {
    const arma::vec& y           = ctx.at("y");
    const arma::vec& Bsp_flat    = ctx.at("Bsp_flat");      // (N*K_s) col-major
    const double     sds_prior_sd = ctx.at("sds_prior_sd")[0];

    const std::size_t N   = y.n_elem;
    const std::size_t K_s = Bsp_flat.n_elem / N;

    if (theta_cat.n_elem != 3 + K_s) {
        return -std::numeric_limits<double>::infinity();
    }

    const double Intercept = theta_cat[0];
    const double log_sds   = theta_cat[1];
    const double log_sigma = theta_cat[2];
    const double sds       = std::exp(log_sds);
    const double sigma     = std::exp(log_sigma);
    const arma::vec z      = theta_cat.subvec(3, 3 + K_s - 1);  // (K_s)

    // s = sds * z; f_n = sum_k Bsp_{n,k} * s_k = sds * sum_k Bsp_{n,k} z_k.
    arma::vec f(N, arma::fill::zeros);
    for (std::size_t k = 0; k < K_s; ++k) {
        const double sk = sds * z[k];
        for (std::size_t n = 0; n < N; ++n) {
            f[n] += sk * Bsp_flat[n + k * N];
        }
    }

    arma::vec resid = y - Intercept - f;
    const double sigma2 = sigma * sigma;

    // log lik
    double lp = -0.5 * static_cast<double>(N) * std::log(kTwoPi)
                - static_cast<double>(N) * log_sigma
                - 0.5 * arma::dot(resid, resid) / sigma2;

    // priors
    constexpr double intercept_var = 100.0;  // Normal(0, 10)
    lp -= 0.5 * Intercept * Intercept / intercept_var;

    // sds ~ Half-Normal(0, sds_prior_sd) + log|J|
    const double sds_var = sds_prior_sd * sds_prior_sd;
    lp += -0.5 * sds * sds / sds_var + log_sds;

    // sigma ~ Jeffreys: p(log_sigma) const, no contribution.

    // z ~ Normal(0, 1)
    lp -= 0.5 * arma::dot(z, z);

    if (!std::isfinite(lp)) return -std::numeric_limits<double>::infinity();

    if (grad) {
        grad->set_size(3 + K_s);

        // d log_lik / d Intercept = sum(resid)/sigma^2; prior = -Intercept/var
        (*grad)[0] = arma::sum(resid) / sigma2 - Intercept / intercept_var;

        // d f / d log_sds = f (since f oc sds). Likelihood:
        //   d log_lik / d log_sds = sum(resid * f) / sigma^2.
        // Prior: -0.5 sds^2 / sds_var + log_sds  ->  d/d log_sds = -sds^2/var + 1.
        (*grad)[1] = arma::dot(resid, f) / sigma2
                     - sds * sds / sds_var + 1.0;

        // d log_lik / d log_sigma = -N + sum(resid^2) / sigma^2 (Jeffreys
        // prior contributes 0).
        (*grad)[2] = -static_cast<double>(N)
                     + arma::dot(resid, resid) / sigma2;

        // d f_n / d z_k = sds * Bsp_{n,k}.
        // d log_lik / d z_k = sds * sum_n resid_n * Bsp_{n,k} / sigma^2.
        // Prior: -z_k.
        for (std::size_t k = 0; k < K_s; ++k) {
            double dot_phi_resid = 0.0;
            for (std::size_t n = 0; n < N; ++n) {
                dot_phi_resid += resid[n] * Bsp_flat[n + k * N];
            }
            (*grad)[3 + k] = sds * dot_phi_resid / sigma2 - z[k];
        }
    }

    return lp;
}

}  // anonymous namespace

class BSplineRegression {
public:
    BSplineRegression(const arma::vec& y,
                      const arma::mat& Bsp,
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
          impl_(std::make_unique<composite_block>("BSplineRegression")),
          keep_history_(keep_history)
    {
        const std::size_t N   = y.n_elem;
        const std::size_t K_s = Bsp.n_cols;
        if (Bsp.n_rows != N) {
            throw std::invalid_argument("Bsp rows must equal length(y)");
        }
        if (K_s < 1) throw std::invalid_argument("K_s must be >= 1");

        // Flatten Bsp column-major.
        arma::vec Bsp_flat = arma::vectorise(Bsp);

        const double y_sd        = arma::stddev(y);
        const double sds_prior_sd = (y_sd > 0.0) ? y_sd : 1.0;

        // shared_data
        impl_->data().set("y",            y);
        impl_->data().set("Bsp_flat",     Bsp_flat);
        impl_->data().set("N",            arma::vec{static_cast<double>(N)});
        impl_->data().set("K_s",          arma::vec{static_cast<double>(K_s)});
        impl_->data().set("sds_prior_sd", arma::vec{sds_prior_sd});

        const double Intercept_init = arma::mean(y);
        const double log_sds_init   = std::log(0.5 * sds_prior_sd);
        const double log_sigma_init = std::log(0.5 * sds_prior_sd);
        arma::vec z_init(K_s, arma::fill::zeros);

        impl_->data().set("Intercept", arma::vec{Intercept_init});
        impl_->data().set("log_sds",   arma::vec{log_sds_init});
        impl_->data().set("log_sigma", arma::vec{log_sigma_init});
        impl_->data().set("z",         z_init);

        // ONE joint block.
        joint_nuts_block_config cfg;
        cfg.name = "bspline_joint";
        cfg.sub_params.push_back({"Intercept", 1});
        cfg.sub_params.push_back({"log_sds",   1});
        cfg.sub_params.push_back({"log_sigma", 1});
        cfg.sub_params.push_back({"z",         K_s});

        cfg.initial_cat = arma::vec(3 + K_s);
        cfg.initial_cat[0] = Intercept_init;
        cfg.initial_cat[1] = log_sds_init;
        cfg.initial_cat[2] = log_sigma_init;
        for (std::size_t k = 0; k < K_s; ++k) cfg.initial_cat[3 + k] = 0.0;

        cfg.log_density_grad    = &joint_log_density;
        cfg.use_dense_metric    = true;
        cfg.n_warmup_first_call = 1500;

        impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));

        impl_->data().declare_dependencies(
            "bspline_joint", {"y", "Bsp_flat", "sds_prior_sd"});

        // ---- Full predict-DAG: real posterior predictive ---------------
        // (Replaces the old stub predict_at that echoed the training y.)
        // Generative chain (parameterization read verbatim from
        // joint_log_density):
        //   s   = exp(log_sds) * z                 (basis coefficients)
        //   f_n = sum_k Bsp[n,k] * s_k             (spline contribution)
        //   mu  = Intercept + f                    (fitted mean)
        //   y_rep ~ N(mu, exp(log_sigma)^2)        (stochastic)
        // s/f/mu kept current during sampling via declare_invalidates
        // (order: s -> f -> mu) so stateful predict_at(list()) (empty
        // changed-set, no Pass-1 recompute) reads fresh values.
        impl_->data().set("s",     arma::vec(K_s, arma::fill::zeros));
        impl_->data().set("f",     arma::vec(N,   arma::fill::zeros));
        impl_->data().set("mu",    arma::vec(N,   arma::fill::zeros));
        impl_->data().set("y_rep", arma::vec(N,   arma::fill::zeros));

        impl_->data().register_refresher(
            "s",
            [](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const double sds = std::exp(d.get("log_sds")[0]);
                const arma::vec& z = d.get("z");
                return sds * z;
            });
        impl_->data().register_refresher(
            "f",
            [](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& Bsp_flat = d.get("Bsp_flat");
                const arma::vec& s        = d.get("s");
                const std::size_t K = s.n_elem;
                const std::size_t Nn = Bsp_flat.n_elem / K;
                arma::vec f(Nn, arma::fill::zeros);
                for (std::size_t k = 0; k < K; ++k) {
                    const double sk = s[k];
                    for (std::size_t n = 0; n < Nn; ++n)
                        f[n] += sk * Bsp_flat[n + k * Nn];
                }
                return f;
            });
        impl_->data().register_refresher(
            "mu",
            [](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                return d.get("Intercept")[0] + d.get("f");
            });
        impl_->data().declare_invalidates("bspline_joint",
                                          {"s", "f", "mu"});

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

        // Predict DAG (full generative chain). Bsp_flat is the
        // PRECOMPUTED basis at (new) x supplied R-side (Q3=A).
        impl_->data().declare_data_input("Bsp_flat");
        impl_->data().declare_predict_edges("log_sds",  {"s"});
        impl_->data().declare_predict_edges("z",        {"s"});
        impl_->data().declare_predict_edges("s",        {"f"});
        impl_->data().declare_predict_edges("Bsp_flat", {"f"});
        impl_->data().declare_predict_edges("f",        {"mu"});
        impl_->data().declare_predict_edges("Intercept",{"mu"});
        impl_->data().declare_predict_edges("mu",       {"y_rep"});
        impl_->data().declare_predict_edges("log_sigma",{"y_rep"});

        // Generative-DAG context: sds ~ Half-Normal(0, sds_prior_sd);
        // Intercept ~ N(0,10) and Jeffreys sigma are hardcoded (no
        // slot). sds is exp(log_sds) so the prior parent is on log_sds.
        impl_->data().declare_context_edges("sds_prior_sd", {"log_sds"});

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step(int n_steps) {
        if (n_steps < 0) throw std::runtime_error("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["Intercept"] = impl_->data().get("Intercept");
        out["log_sds"]   = impl_->data().get("log_sds");
        out["log_sigma"] = impl_->data().get("log_sigma");
        out["z"]         = impl_->data().get("z");
        const double s = std::exp(impl_->data().get("log_sds")[0]);
        const double r = std::exp(impl_->data().get("log_sigma")[0]);
        out["sds"]   = arma::vec{s};
        out["sigma"] = arma::vec{r};
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        for (const auto& [k, v] : params) {
            if (k == "sds" || k == "sigma") continue;
            impl_->data().set(k, v);
        }
    }

    // Real posterior predictive via the framework predict DAG (replaces
    // the old stub that echoed the training y). Single current-draw
    // sample, consistent with the state_map (vector) return contract.
    //   predict_at(list())              -> s,f,mu,y_rep at training x
    //   predict_at(list(Bsp_flat=...))  -> at a NEW precomputed basis
    //       (Bsp_flat = vectorise(Bsp_new), N_new*K_s column-major;
    //        Q3=A: the spline basis is evaluated R-side at new x).
    AI4BayesCode::state_map predict_at(
        const AI4BayesCode::state_map& new_data) const {
        block_context replaced;
        auto it = new_data.find("Bsp_flat");
        if (it != new_data.end()) {
            const arma::vec& bf = it->second;
            const std::size_t K =
                impl_->data().get("s").n_elem;          // K_s
            if (bf.n_elem == 0 || bf.n_elem % K != 0) {
                throw std::runtime_error(
                    "BSplineRegression::predict_at: Bsp_flat length " +
                    std::to_string(bf.n_elem) +
                    " is not a positive multiple of K_s = " +
                    std::to_string(K) +
                    " (pass vectorise(Bsp_new), N_new*K_s col-major).");
            }
            replaced["Bsp_flat"] = bf;
        } else {
            for (const auto& kv : new_data) {
                throw std::runtime_error(
                    "BSplineRegression::predict_at: unknown key '" +
                    kv.first + "'. Valid: 'Bsp_flat' (or empty).");
            }
        }
        block_context r = impl_->predict_at(replaced, predict_rng_);
        AI4BayesCode::state_map out;
        for (const auto& kv : r) out[kv.first] = kv.second;
        return out;
    }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    // History is augmented with the derived positive-scale parameters
    // sds = exp(log_sds) and sigma = exp(log_sigma) so that get_history()
    // is consistent with get_current() (which exposes both).
    AI4BayesCode::history_map get_history() const {
        AI4BayesCode::history_map h = impl_->get_history();
        auto it_sds = h.find("log_sds");
        if (it_sds != h.end()) h["sds"] = arma::exp(it_sds->second);
        auto it_sig = h.find("log_sigma");
        if (it_sig != h.end()) h["sigma"] = arma::exp(it_sig->second);
        return h;
    }

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
    std::mt19937_64 rng_;
    mutable std::mt19937_64 predict_rng_;   // predict_at() const advances it
    mutable std::mt19937_64 readapt_rng_; // readapt_NUTS() advances it (7th method)
    std::unique_ptr<composite_block> impl_;
    bool keep_history_ = false;
};

// ============================================================================
// Bindings
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(BSplineRegression_module) {
    Rcpp::class_<BSplineRegression>("BSplineRegression")
        .constructor<arma::vec, arma::mat, int, bool>(
            "1-D penalized B-spline regression.\n"
            "Args: y (N), Bsp (N x K_s basis matrix), seed, keep_history.")
        .method("step",        &BSplineRegression::step)
        .method("get_current", &BSplineRegression::get_current)
        .method("set_current", &BSplineRegression::set_current)
        .method("predict_at",  &BSplineRegression::predict_at)
        .method("get_dag",     &BSplineRegression::get_dag)
        .method("get_history", &BSplineRegression::get_history)
        .method("readapt_NUTS", &BSplineRegression::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"

PYBIND11_MODULE(BSplineRegression_module, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<BSplineRegression>(m, "BSplineRegression")
        .def(pybind11::init<arma::vec, arma::mat, int, bool>(),
             pybind11::arg("y"), pybind11::arg("Bsp"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false,
             "1-D penalized B-spline regression.")
        .def("step",        &BSplineRegression::step,
             pybind11::arg("n_steps"))
        .def("get_current", &BSplineRegression::get_current)
        .def("set_current", &BSplineRegression::set_current,
             pybind11::arg("params"))
        .def("predict_at",  &BSplineRegression::predict_at,
             pybind11::arg("new_data"))
        .def("get_dag",     &BSplineRegression::get_dag)
        .def("get_history", &BSplineRegression::get_history)
        .def("readapt_NUTS", &BSplineRegression::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false);
}
#endif

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Synthetic fixture exactly per the header comment: x in [0,1], N=100,
//  K_s=10, y = smooth(x) + N(0, 0.3) with smooth(x) = 0.4*sin(8*x) + 0.2*x^2.
//  Basis: truncated power (linear) basis on an evenly-spaced interior-knot
//  grid -- a generic, self-contained B-spline-style penalized smoother.
//
//  Success criterion: the fitted spline (Intercept + Bsp . posterior-mean s)
//  must recover the latent smooth(x) far better than the naive constant
//  baseline mean(y). We compare RMSE-to-truth of the fit vs the baseline.
//==============================================================================
#include <cstdio>
int main() {
    const std::size_t N   = 100;
    const std::size_t K_s = 10;      // interior knots -> basis dimension
    const double      noise_sd = 0.3;

    auto smooth = [](double x) {
        return 0.4 * std::sin(8.0 * x) + 0.2 * x * x;
    };

    // ---- x grid and latent truth -----------------------------------------
    arma::vec x(N), truth(N);
    for (std::size_t n = 0; n < N; ++n) {
        x[n]     = static_cast<double>(n) / static_cast<double>(N - 1); // [0,1]
        truth[n] = smooth(x[n]);
    }

    // ---- truncated power (linear) basis: phi_k(x) = (x - knot_k)_+ -------
    // Evenly-spaced interior knots in (0,1). This is a standard piecewise-
    // linear spline basis; the penalty comes from the z ~ N(0,1) prior.
    arma::vec knots(K_s);
    for (std::size_t k = 0; k < K_s; ++k)
        knots[k] = static_cast<double>(k + 1) / static_cast<double>(K_s + 1);

    arma::mat Bsp(N, K_s, arma::fill::zeros);
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t k = 0; k < K_s; ++k) {
            const double d = x[n] - knots[k];
            Bsp(n, k) = (d > 0.0) ? d : 0.0;       // (x - knot)_+
        }
    }

    // ---- simulate observations -------------------------------------------
    std::mt19937_64 sim_rng(123);
    std::normal_distribution<double> eps(0.0, noise_sd);
    arma::vec y(N);
    for (std::size_t n = 0; n < N; ++n) y[n] = truth[n] + eps(sim_rng);

    // ---- fit -------------------------------------------------------------
    BSplineRegression model(y, Bsp, /*rng_seed=*/7, /*keep_history=*/false);
    model.step(800);   // warmup (1500-iter dense-metric pilot on first call)

    // Posterior mean of (Intercept, s = sds * z) over kept draws.
    const int M = 2000;
    double      Intercept_bar = 0.0;
    arma::vec   s_bar(K_s, arma::fill::zeros);
    for (int it = 0; it < M; ++it) {
        model.step(1);
        const auto cur = model.get_current();
        Intercept_bar += cur.at("Intercept")[0];
        const double sds = cur.at("sds")[0];
        const arma::vec& z = cur.at("z");
        for (std::size_t k = 0; k < K_s; ++k) s_bar[k] += sds * z[k];
    }
    Intercept_bar /= static_cast<double>(M);
    s_bar         /= static_cast<double>(M);

    // ---- fitted mean and RMSE vs latent truth ----------------------------
    arma::vec fit = Intercept_bar + Bsp * s_bar;
    double sse_fit = 0.0, sse_base = 0.0;
    const double y_mean = arma::mean(y);             // naive constant baseline
    for (std::size_t n = 0; n < N; ++n) {
        const double e_fit  = fit[n]  - truth[n];
        const double e_base = y_mean  - truth[n];
        sse_fit  += e_fit  * e_fit;
        sse_base += e_base * e_base;
    }
    const double rmse_fit  = std::sqrt(sse_fit  / static_cast<double>(N));
    const double rmse_base = std::sqrt(sse_base / static_cast<double>(N));

    std::printf("BSplineRegression demo: N=%zu K_s=%zu noise_sd=%.2f\n",
                N, K_s, noise_sd);
    std::printf("  fitted-spline RMSE-to-truth = %.4f\n", rmse_fit);
    std::printf("  naive-mean    RMSE-to-truth = %.4f\n", rmse_base);
    std::printf("  Intercept_hat = %.4f\n", Intercept_bar);

    // The spline must (a) clearly beat the constant baseline and (b) recover
    // the smooth to comfortably within the observation noise level.
    const bool ok = (rmse_fit < 0.5 * rmse_base) && (rmse_fit < noise_sd);
    std::printf("%s\n",
        ok ? "[demo PASS] joint-NUTS B-spline recovers smooth(x)"
           : "[demo FAIL] fitted spline did not recover the truth");
    return ok ? 0 : 1;
}
#endif

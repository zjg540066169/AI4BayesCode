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
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("SpikeSlabSinhBijection")
//   set.seed(123)
//   N <- 80; beta_true <- 1.5; sigma <- 1.0; slab_sd <- 5.0; pi_incl <- 0.5
//   x <- rnorm(N)
//   y <- beta_true * x + rnorm(N, 0, sigma)        # known nonzero truth -> inclusion favored
//   m <- new(SpikeSlabSinhBijection,
//            y, x,                                 # data: response y, predictor x
//            sigma, slab_sd, pi_incl,              # fixed hypers
//            7L,                                   # rng_seed (0 = random)
//            TRUE)                                 # keep_history
//   m$step(2000); str(m$get_current())            # gamma (~1 here), beta (~1.5)
//   mean(m$get_history()$gamma)                    # sampled inclusion prob (~1)
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(123); N = 80
//   beta_true, sigma, slab_sd, pi_incl = 1.5, 1.0, 5.0, 0.5
//   x = rng.normal(size=N)
//   y = beta_true * x + rng.normal(0.0, sigma, N)   # known nonzero truth -> inclusion favored
//   Mod = AI4BayesCode.source("SpikeSlabSinhBijection.cpp")
//   m = Mod.SpikeSlabSinhBijection(y, x, sigma, slab_sd, pi_incl, 7, True)
//   m.step(2000); print(m.get_current())            # dict: gamma (~1), beta (~1.5)
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
#include "AI4BayesCode/rjmcmc_block.hpp"
#include "AI4BayesCode/rjmcmc_custom_bijection.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>

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

// ============================================================================
//  User-facing class exposed to R / Python.
//
//  Same Dirac spike-and-slab model and the SAME custom sinh-bijection RJMCMC
//  kernel as the original R-only free function `spike_slab_sinh_chain`. The
//  model logic (closed-form Gibbs continuous_update, log_joint, N(0,1) birth
//  proposal, sinh/asinh templated bijection) is UNCHANGED; only the
//  Rcpp::List entry point has been replaced by a stateful class with the
//  standard backend-neutral method surface so it builds in BOTH R and Python.
// ============================================================================
class SpikeSlabSinhBijection {
public:
    SpikeSlabSinhBijection(const arma::vec& y,
                           const arma::vec& x,
                           double sigma,
                           double slab_sd,
                           double pi_inclusion,
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
          impl_(std::make_unique<composite_block>("SpikeSlabSinhBijection")),
          sigma_(sigma),
          keep_history_(keep_history)
    {
        if (y.n_elem != x.n_elem) {
            ai4b::stop("y and x must have the same length");
        }
        if (!(sigma > 0.0 && slab_sd > 0.0)) {
            ai4b::stop("sigma and slab_sd must be > 0");
        }
        if (!(pi_inclusion > 0.0 && pi_inclusion < 1.0)) {
            ai4b::stop("pi_inclusion must be in (0, 1)");
        }

        N_ = static_cast<std::size_t>(y.n_elem);

        const double xtx = arma::dot(x, x);
        const double xty = arma::dot(x, y);

        // ---- Install data (x stored column-major flat, length N) ----------
        impl_->data().set("x",     x);
        impl_->data().set("gamma", arma::vec{0.0});
        impl_->data().set("beta",  arma::vec{0.0});

        // ---- custom bijection transform ----
        auto bij = make_templated_bijection_1d(
            SinhForward{slab_sd}, AsinhInverse{slab_sd});

        // ---- rjmcmc_block config (IDENTICAL model logic to the original) --
        rjmcmc_block_config cfg;
        cfg.name      = "gamma_beta_rj";
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

        // ---- Predict DAG + y_rep stochastic refresher ---------------------
        // Posterior-predictive: y_rep_i ~ N(beta * x_i, sigma^2) (length N).
        impl_->data().declare_data_input("x");
        impl_->data().declare_predict_edges("beta", {"y_rep"});
        impl_->data().declare_predict_edges("x",    {"y_rep"});
        // Generative-DAG context (VIZ-ONLY; predict_at BFS never reads it).
        impl_->data().declare_context_edges("gamma", {"beta"});

        impl_->data().set("y_rep", arma::vec(N_, arma::fill::zeros));
        const double sigma_ref = sigma;
        // Check #17 whitelist: std::normal_distribution used inside
        // register_stochastic_refresher (one of the whitelisted contexts).
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [sigma_ref](const AI4BayesCode::shared_data_t& d,
                        std::mt19937_64& rng) {
                const arma::vec& x = d.get("x");
                const double     b = d.get("beta")[0];
                std::normal_distribution<double> norm01(0.0, 1.0);
                arma::vec out(x.n_elem);
                for (std::size_t i = 0; i < x.n_elem; ++i)
                    out[i] = b * x[i] + sigma_ref * norm01(rng);
                return out;
            });

        // ---- composite child ----------------------------------------------
        impl_->add_child(std::make_unique<rjmcmc_block>(std::move(cfg)));

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["gamma"] = impl_->data().get("gamma");   // length-1 (0/1)
        out["beta"]  = impl_->data().get("beta");    // length-1 (0 when gamma=0)
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        // rjmcmc_block::set_current takes the concatenated length-2p vector
        // [gamma(0..p-1), beta(0..p-1)] (here p = 1) and enforces the
        // gamma=0 => beta=0 coupling.
        auto& rj_blk = dynamic_cast<rjmcmc_block&>(impl_->child(0));
        arma::vec cat = rj_blk.current();   // [gamma, beta], length 2
        bool touched = false;

        auto it_g = params.find("gamma");
        if (it_g != params.end()) {
            const double g = (it_g->second[0] >= 0.5) ? 1.0 : 0.0;
            cat[0] = g;
            touched = true;
        }
        auto it_b = params.find("beta");
        if (it_b != params.end()) {
            cat[1] = it_b->second[0];
            touched = true;
        }
        if (touched) {
            rj_blk.set_current(cat);   // validates gamma=0 => beta=0
            impl_->data().set("gamma", arma::vec{cat[0]});
            impl_->data().set("beta",  arma::vec{cat[1]});
        }
    }

    // Posterior-predictive y_rep_i ~ N(beta * x_i, sigma^2).
    //   * INPUT  : new_data["x"] is a length-N_new arma::vec of predictors
    //              (frontend passes as.vector(x_new) / x_new). Empty map =>
    //              predict at the training x.
    //   * OUTPUT : every key is an arma::mat. keep_history = FALSE returns a
    //              1-row matrix (single predict at the current draw);
    //              keep_history = TRUE returns an n_draws-row matrix
    //              (posterior predictive over all kept draws).
    //
    // NOTE: the original R-only free function had no predict_at; this is a
    // minimal posterior-predictive added to give the dual class the standard
    // method surface. The model logic / sampler is untouched.
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        // ---- Parse optional x (length N_new) ------------------------------
        bool has_x = false;
        arma::vec x_use;
        for (const auto& kv : new_data) {
            if (kv.first != "x")
                ai4b::stop("SpikeSlabSinhBijection::predict_at: unknown key '%s'. "
                           "Valid keys: 'x' (or empty map).", kv.first.c_str());
        }
        auto it_x = new_data.find("x");
        if (it_x != new_data.end()) {
            x_use = it_x->second;
            has_x = true;
        } else {
            x_use = impl_->data().get("x");
        }
        const std::size_t N_pred = x_use.n_elem;

        AI4BayesCode::history_map out;
        std::normal_distribution<double> norm01(0.0, 1.0);

        if (!keep_history_) {
            // Single predict at the current draw.
            const double b = impl_->data().get("beta")[0];
            arma::mat yrep_mat(1, N_pred);
            for (std::size_t i = 0; i < N_pred; ++i)
                yrep_mat(0, i) = b * x_use[i] + sigma_ * norm01(predict_rng_);
            out.emplace("y_rep", std::move(yrep_mat));
            return out;
        }

        // History mode: beta is a sub-output of the rjmcmc block (block name
        // "gamma_beta_rj" writes both gamma and beta to shared_data, keyed by
        // sub-param name in get_history()). Read the sampled beta history and
        // draw one y_rep per posterior draw (cf. SpikeSlabRJMCMC / ARDLasso).
        AI4BayesCode::history_map hist = impl_->get_history();
        if (!hist.count("beta")) {
            ai4b::stop("SpikeSlabSinhBijection::predict_at: keep_history = TRUE "
                       "requires beta history, but get_history() lacks it.");
        }
        const arma::mat& beta_hist = hist.at("beta");   // n_draws x 1
        const std::size_t n_draws  = beta_hist.n_rows;

        arma::mat yrep_mat(n_draws, N_pred);
        for (std::size_t d = 0; d < n_draws; ++d) {
            const double b = beta_hist(d, 0);
            for (std::size_t i = 0; i < N_pred; ++i)
                yrep_mat(d, i) = b * x_use[i] + sigma_ * norm01(predict_rng_);
        }
        out.emplace("y_rep", std::move(yrep_mat));
        return out;
    }

    AI4BayesCode::dag_info    get_dag()     const { return impl_->get_dag(); }
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    /// Re-tune NUTS metric for any NUTS-family children. This composite holds
    /// only an rjmcmc_block (closed-form Gibbs continuous_update, no NUTS
    /// child), so the dispatch is a no-op here; exposed for a uniform method
    /// surface across examples.
    void readapt_NUTS(int n, bool reset = false) {
        if (n < 0) ai4b::stop("readapt_NUTS: n must be non-negative");
        impl_->readapt_NUTS(static_cast<std::size_t>(n), reset, readapt_rng_);
    }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    double                           sigma_ = 1.0;
    std::size_t                      N_ = 0;
    bool                             keep_history_ = false;
};

// ============================================================================
//  Rcpp module
// ============================================================================
#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(SpikeSlabSinhBijection_module) {
    Rcpp::class_<SpikeSlabSinhBijection>("SpikeSlabSinhBijection")
        .constructor<arma::vec, arma::vec, double, double, double, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::vec, arma::vec, double, double, double, int, bool>(
            "Dirac spike-and-slab via rjmcmc_block with a custom sinh/asinh "
            "birth bijection (runtime-AD Jacobian). Construct with: y, x "
            "(equal length), sigma, slab_sd (both > 0), pi_inclusion in (0,1), "
            "rng_seed (0 = random), and keep_history (default FALSE).")
        .method("step",         &SpikeSlabSinhBijection::step)
        .method("get_current",  &SpikeSlabSinhBijection::get_current)
        .method("set_current",  &SpikeSlabSinhBijection::set_current)
        .method("predict_at",   &SpikeSlabSinhBijection::predict_at)
        .method("get_dag",      &SpikeSlabSinhBijection::get_dag)
        .method("get_history",  &SpikeSlabSinhBijection::get_history)
        .method("readapt_NUTS", &SpikeSlabSinhBijection::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(SpikeSlabSinhBijection, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<SpikeSlabSinhBijection>(m, "SpikeSlabSinhBijection")
        .def(pybind11::init<arma::vec, arma::vec, double, double, double,
                            int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("x"),
             pybind11::arg("sigma"),
             pybind11::arg("slab_sd"),
             pybind11::arg("pi_inclusion"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step",         &SpikeSlabSinhBijection::step,  pybind11::arg("n_steps"))
        .def("get_current",  &SpikeSlabSinhBijection::get_current)
        .def("set_current",  &SpikeSlabSinhBijection::set_current, pybind11::arg("params"))
        .def("predict_at",   &SpikeSlabSinhBijection::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &SpikeSlabSinhBijection::get_dag)
        .def("get_history",  &SpikeSlabSinhBijection::get_history)
        .def("readapt_NUTS", &SpikeSlabSinhBijection::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false);
}
#endif

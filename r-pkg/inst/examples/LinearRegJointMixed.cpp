// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  LinearRegJointMixed_v2.cpp
//
//  Mechanical migration of LinearRegJointMixed.cpp off the legacy
//  joint_nuts_block_mixed shim onto the UNIFIED joint_nuts_block class.
//  Coexists with the original (class LinearRegJointMixed, distinct module)
//  for cross-validation: both files expose the SAME posterior and the SAME
//  natural-scale log-density (copied verbatim here); only the wrapper
//  types/fields change.
//
//  Changes vs. LinearRegJointMixed.cpp
//  ------------------------------------
//    joint_nuts_block_mixed        -> joint_nuts_block
//    joint_nuts_block_mixed_config -> joint_nuts_block_config
//    joint_nuts_sub_param_mixed    -> joint_nuts_sub_param
//    cfg.log_density_grad_natural  -> cfg.log_density_grad
//    cfg.initial_nat               -> cfg.initial_cat
//    cfg.n_warmup_first_call       1000 -> 800
//    + cfg.use_diagonal_metric = true   (new; the unified block supports it)
//
//  Model (identical to LinearRegJointMixed.cpp)
//  --------------------------------------------
//      y_n | alpha, beta, sigma  ~ Normal(alpha + X_n' beta, sigma^2)
//      alpha                      ~ Normal(0, 10^2)
//      beta_k                     ~ Normal(0, 10^2)
//      sigma                      ~ Half-Normal(0, 5)
//
//  Layout in theta_cat (natural-scale):
//      [alpha (1); beta (p); sigma (1)]
//      -- alpha: REAL
//      -- beta:  REAL
//      -- sigma: POSITIVE (block adds log|J| = log sigma internally)
//
//  VALIDATOR CHECK #11 (mixed): sigma is written on the NATURAL scale
//  (sigma > 0). The block adds log(sigma) Jacobian automatically.
//  The log-density MUST NOT include it.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("LinearRegJointMixed")
//   set.seed(2026)
//   N <- 300L; p <- 3L                                  # N >> p: well-identified
//   alpha <- 1.5; beta <- c(2.0, -1.0, 0.5); sigma <- 0.8
//   X <- matrix(rnorm(N * p), N, p)                     # iid N(0,1) design
//   y <- as.numeric(alpha + X %*% beta + rnorm(N, 0, sigma))
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- AI4BayesCode_run_chains(
//       function(seed) new(LinearRegJointMixed, y, X, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4b_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(LinearRegJointMixed, y, X, 7L, TRUE)       # ctor: y, X, seed, keep_history
//   m$step(2500); str(m$get_current())                 # -> alpha, beta(p), sigma
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2026)
//   N, p = 300, 3                                       # N >> p: well-identified
//   alpha, beta, sigma = 1.5, np.array([2.0, -1.0, 0.5]), 0.8
//   X = rng.standard_normal((N, p))                     # iid N(0,1) design
//   y = alpha + X @ beta + rng.standard_normal(N) * sigma
//   Mod = AI4BayesCode.source("LinearRegJointMixed.cpp")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.LinearRegJointMixed(y, X, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.ai4b_diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.LinearRegJointMixed(y, X, 7, True)          # ctor: y, X, seed, keep_history
//   m.step(2500); print(m.get_current())
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
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/constraints.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;

namespace constraints = AI4BayesCode::constraints;

namespace {

// Natural-scale joint log-density. Input layout:
//   theta_cat[0]            = alpha
//   theta_cat[1 .. p]       = beta_1 .. beta_p
//   theta_cat[p + 1]        = sigma (natural, > 0)
//
// Output grad_nat:
//   grad_nat[0]         = d lp / d alpha
//   grad_nat[1..p]      = d lp / d beta
//   grad_nat[p+1]       = d lp / d sigma  (NATURAL; the block
//                         chain-rules it back to d/d log(sigma))
//
// The log|Jacobian| for the positive sigma slice is added automatically by
// joint_nuts_block; DO NOT include it here.
// (Copied verbatim from LinearRegJointMixed.cpp.)
double joint_log_density(const arma::vec& cat_nat,
                         const block_context& ctx,
                         arma::vec* grad_nat) {
    const arma::vec& y = ctx.at("y");
    const arma::vec& X = ctx.at("X");  // N x p, column-major flat
    const std::size_t N = static_cast<std::size_t>(y.n_elem);
    const std::size_t p = static_cast<std::size_t>(ctx.at("p")[0]);

    if (cat_nat.n_elem != 1 + p + 1) {
        return -std::numeric_limits<double>::infinity();
    }
    const double alpha = cat_nat[0];
    auto beta          = cat_nat.subvec(1, 1 + p - 1);
    const double sigma = cat_nat[1 + p];
    if (sigma <= 0.0) {
        if (grad_nat) grad_nat->set_size(1 + p + 1);
        return -std::numeric_limits<double>::infinity();
    }
    const double sigma2 = sigma * sigma;

    constexpr double prior_sd_ab    = 10.0;
    constexpr double prior_sd_sigma = 5.0;
    const double prior_var_ab       = prior_sd_ab * prior_sd_ab;
    const double prior_var_sigma    = prior_sd_sigma * prior_sd_sigma;

    double lp = 0.0;
    if (grad_nat) { grad_nat->set_size(1 + p + 1); grad_nat->zeros(); }

    // Priors
    //   alpha ~ N(0, 10^2)
    lp += -0.5 * alpha * alpha / prior_var_ab;
    if (grad_nat) (*grad_nat)[0] += -alpha / prior_var_ab;
    //   beta_k ~ N(0, 10^2)
    for (std::size_t k = 0; k < p; ++k) {
        const double bk = beta[k];
        lp += -0.5 * bk * bk / prior_var_ab;
        if (grad_nat) (*grad_nat)[1 + k] += -bk / prior_var_ab;
    }
    //   sigma ~ Half-Normal(0, 5). Natural-scale density:
    //     p(sigma) \propto exp(-sigma^2 / (2 * 5^2))
    //   d/d sigma = -sigma / prior_var_sigma
    //   NOTE: the block will ADD log(sigma) Jacobian automatically when it
    //   transforms the gradient from d/d sigma to d/d log(sigma). We do NOT
    //   add -log(sigma)*J_something here.
    lp += -0.5 * sigma2 / prior_var_sigma;
    if (grad_nat) (*grad_nat)[1 + p] += -sigma / prior_var_sigma;

    // Likelihood: y_n ~ N(alpha + X_n' beta, sigma^2)
    //   log p = -N log(sigma) - 0.5 sum_n (y_n - mu_n)^2 / sigma^2
    //   d/d mu_n    = (y_n - mu_n) / sigma^2
    //   d/d alpha   = sum_n d/d mu_n
    //   d/d beta_k  = sum_n d/d mu_n * X_{n,k}
    //   d/d sigma   = -N/sigma + sum_n (y_n - mu_n)^2 / sigma^3
    double sum_sq = 0.0;
    for (std::size_t n = 0; n < N; ++n) {
        double xb = 0.0;
        for (std::size_t k = 0; k < p; ++k) {
            xb += X[n + k * N] * beta[k];
        }
        const double mu_n = alpha + xb;
        const double r    = y[n] - mu_n;
        sum_sq += r * r;
        if (grad_nat) {
            const double gmu = r / sigma2;
            (*grad_nat)[0]              += gmu;
            for (std::size_t k = 0; k < p; ++k) {
                (*grad_nat)[1 + k]      += gmu * X[n + k * N];
            }
        }
    }
    lp += -static_cast<double>(N) * std::log(sigma) - 0.5 * sum_sq / sigma2;
    if (grad_nat) {
        (*grad_nat)[1 + p] +=
            -static_cast<double>(N) / sigma + sum_sq / (sigma2 * sigma);
    }

    if (!std::isfinite(lp)) return -std::numeric_limits<double>::infinity();
    return lp;
}

} // anonymous namespace

class LinearRegJointMixed {
public:
    LinearRegJointMixed(const arma::vec& y,
                          const arma::mat& X,
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
          impl_(std::make_unique<composite_block>("LinearRegJointMixed")),
          keep_history_(keep_history)
    {
        const std::size_t N = y.n_elem;
        const std::size_t p = X.n_cols;
        if (X.n_rows != N) throw std::runtime_error("X rows must equal length(y)");

        impl_->data().set("y", y);
        impl_->data().set("X", arma::vectorise(X));
        impl_->data().set("p", arma::vec{static_cast<double>(p)});

        // Initial values: OLS-like.
        arma::mat XtX = X.t() * X + 1e-6 * arma::eye<arma::mat>(p, p);
        arma::vec Xty = X.t() * y;
        arma::vec beta_init = arma::solve(XtX, Xty);
        const double alpha_init = arma::mean(y - X * beta_init);
        const double resid_sd   = std::max(
            arma::stddev(y - X * beta_init - alpha_init), 1e-3);

        impl_->data().set("alpha", arma::vec{alpha_init});
        impl_->data().set("beta",  beta_init);
        impl_->data().set("sigma", arma::vec{resid_sd});

        // Dependencies and invalidates keyed under the JOINT BLOCK NAME.
        impl_->data().declare_dependencies("abs_joint", {"y", "X", "p"});

        // mu: derived deterministic node, invalidated whenever abs_joint fires.
        impl_->data().set("mu",    arma::vec(N, arma::fill::zeros));
        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));
        impl_->data().register_refresher(
            "mu",
            [p](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const double     alpha_cur = d.get("alpha")[0];
                const arma::vec& beta_cur  = d.get("beta");
                const arma::vec& X_flat    = d.get("X");
                const std::size_t N_cur    = X_flat.n_elem / p;
                arma::vec mu(N_cur);
                for (std::size_t i = 0; i < N_cur; ++i) {
                    double xb = 0.0;
                    for (std::size_t j = 0; j < p; ++j)
                        xb += X_flat[i + j * N_cur] * beta_cur[j];
                    mu[i] = alpha_cur + xb;
                }
                return mu;
            });
        impl_->data().declare_invalidates("abs_joint", {"mu"});

        // Predict DAG (full generative chain):
        //   X,alpha,beta -> mu ;  mu,sigma -> y_rep
        impl_->data().declare_predict_edges("X",     {"mu"});
        impl_->data().declare_predict_edges("alpha", {"mu"});
        impl_->data().declare_predict_edges("beta",  {"mu"});
        impl_->data().declare_predict_edges("mu",    {"y_rep"});
        impl_->data().declare_predict_edges("sigma", {"y_rep"});
        impl_->data().declare_data_input("X");
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                const arma::vec& mu        = d.get("mu");
                const double     sigma_cur = d.get("sigma")[0];
                std::normal_distribution<double> norm(0.0, 1.0);
                arma::vec y_rep(mu.n_elem);
                for (std::size_t i = 0; i < mu.n_elem; ++i)
                    y_rep[i] = mu[i] + sigma_cur * norm(rng);
                return y_rep;
            });

        // ---- ONE joint_nuts_block over (alpha REAL, beta REAL, sigma POSITIVE)
        {
            joint_nuts_block_config cfg;
            cfg.name = "abs_joint";
            cfg.sub_params.push_back(
                joint_nuts_sub_param{"alpha", 1, joint_constraint::REAL});
            cfg.sub_params.push_back(
                joint_nuts_sub_param{"beta",  p, joint_constraint::REAL});
            cfg.sub_params.push_back(
                joint_nuts_sub_param{"sigma", 1, joint_constraint::POSITIVE});

            // initial_cat is NATURAL-scale: [alpha, beta (p), sigma].
            arma::vec init_cat(1 + p + 1);
            init_cat[0] = alpha_init;
            for (std::size_t k = 0; k < p; ++k) init_cat[1 + k] = beta_init[k];
            init_cat[1 + p] = resid_sd;
            cfg.initial_cat = init_cat;

            cfg.log_density_grad    = &joint_log_density;
            cfg.n_warmup_first_call = 800;
            cfg.use_diagonal_metric = true;

            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
        }

        N_ = N; p_ = p;

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["alpha"] = impl_->data().get("alpha");
        out["beta"]  = impl_->data().get("beta");
        out["sigma"] = impl_->data().get("sigma");
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto& joint_blk = dynamic_cast<joint_nuts_block&>(impl_->child(0));

        auto it_a = params.find("alpha");
        auto it_b = params.find("beta");
        auto it_s = params.find("sigma");

        if (it_a != params.end() || it_b != params.end() || it_s != params.end()) {
            arma::vec alpha_cur = impl_->data().get("alpha");
            arma::vec beta_cur  = impl_->data().get("beta");
            arma::vec sigma_cur = impl_->data().get("sigma");
            if (it_a != params.end()) alpha_cur[0] = it_a->second[0];
            if (it_b != params.end()) {
                const arma::vec& beta_new = it_b->second;
                if (beta_new.n_elem != p_)
                    throw std::runtime_error("set_current: beta length must equal p");
                beta_cur = beta_new;
            }
            if (it_s != params.end()) {
                const double sg = it_s->second[0];
                if (!(sg > 0.0)) throw std::runtime_error("sigma must be > 0");
                sigma_cur[0] = sg;
            }
            arma::vec cat(1 + p_ + 1);
            cat[0] = alpha_cur[0];
            for (std::size_t k = 0; k < p_; ++k) cat[1 + k] = beta_cur[k];
            cat[1 + p_] = sigma_cur[0];
            joint_blk.set_current(cat);
            impl_->data().set("alpha", alpha_cur);
            impl_->data().set("beta",  beta_cur);
            impl_->data().set("sigma", sigma_cur);
        }
        // Dynamic-N canonical pattern (codegen_cpp.md §7a).
        auto it_X = params.find("X");
        auto it_y = params.find("y");
        if (it_X != params.end()) {
            const arma::vec& X_new = it_X->second;
            if (X_new.n_elem == 0 || X_new.n_elem % p_ != 0)
                throw std::runtime_error("set_current: X length " +
                    std::to_string(X_new.n_elem) +
                    " is not a positive multiple of p = " +
                    std::to_string(p_));
            const std::size_t N_new = X_new.n_elem / p_;
            if (it_y != params.end() && it_y->second.n_elem != N_new)
                throw std::runtime_error("set_current: X implies N=" +
                    std::to_string(N_new) + " but y has length " +
                    std::to_string(it_y->second.n_elem));
            impl_->data().set("X", X_new);
            if (N_new != N_) {
                impl_->data().set("y_rep",
                                  arma::vec(N_new, arma::fill::zeros));
                if (keep_history_ && impl_->history_size() > 1)
                    impl_->clear_history();
            }
            N_ = N_new;
        }
        if (it_y != params.end()) {
            const arma::vec& y_new = it_y->second;
            if (y_new.n_elem != N_)
                throw std::runtime_error("set_current: y length " +
                    std::to_string(y_new.n_elem) +
                    " != current N = " + std::to_string(N_));
            impl_->data().set("y", y_new);
        }
    }

    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        // Parse optional X input once.
        bool has_X = false;
        arma::vec x_flat;
        auto it_X = new_data.find("X");
        if (it_X != new_data.end()) {
            x_flat = it_X->second;
            if (x_flat.n_elem % p_ != 0)
                throw std::runtime_error(
                    "predict_at: X must be vectorised N_test*p (column-major)");
            has_X = true;
        }

        AI4BayesCode::history_map out;

        if (!keep_history_) {
            block_context replaced;
            if (has_X) replaced["X"] = x_flat;
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result) {
                arma::mat m(1, kv.second.n_elem);
                for (std::size_t j = 0; j < kv.second.n_elem; ++j)
                    m(0, j) = kv.second[j];
                out.emplace(kv.first, std::move(m));
            }
            return out;
        }

        // History mode: alpha / beta / sigma are sub-outputs of the
        // "abs_joint" joint_nuts_block — NOT block names. Use
        // manual-compute pattern (cf. SpikeSlabRJMCMC.cpp).
        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& alpha_hist = hist.at("alpha");  // n_draws x 1
        const arma::mat& beta_hist  = hist.at("beta");   // n_draws x p
        const arma::mat& sigma_hist = hist.at("sigma");  // n_draws x 1
        const std::size_t n_draws   = alpha_hist.n_rows;

        const arma::vec& X_use = has_X ? x_flat : impl_->data().get("X");
        const std::size_t N_pred = X_use.n_elem / p_;

        arma::mat yrep_mat(n_draws, N_pred);
        std::normal_distribution<double> norm01(0.0, 1.0);
        for (std::size_t d = 0; d < n_draws; ++d) {
            const double alpha_d = alpha_hist(d, 0);
            const double sigma_d = sigma_hist(d, 0);
            for (std::size_t i = 0; i < N_pred; ++i) {
                double xb = 0.0;
                for (std::size_t j = 0; j < p_; ++j)
                    xb += X_use[i + j * N_pred] * beta_hist(d, j);
                yrep_mat(d, i) = alpha_d + xb + sigma_d * norm01(predict_rng_);
            }
        }
        out.emplace("y_rep", std::move(yrep_mat));
        return out;
    }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    void readapt_NUTS(int n, bool reset = false) {
        if (n < 0) {
            ai4b::stop("readapt_NUTS: n must be non-negative");
        }
        impl_->readapt_NUTS(static_cast<std::size_t>(n),
                            reset, readapt_rng_);
    }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      N_ = 0;
    std::size_t                      p_ = 0;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(LinearRegJointMixed_module) {
    Rcpp::class_<LinearRegJointMixed>("LinearRegJointMixed")
        .constructor<arma::vec, arma::mat, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::vec, arma::mat, int, bool>(
            "Unified-class migration of LinearRegJointMixed. "
            "Construct with y, X, seed, keep_history.")
        .method("step",         &LinearRegJointMixed::step)
        .method("get_current",  &LinearRegJointMixed::get_current)
        .method("set_current",  &LinearRegJointMixed::set_current)
        .method("get_history",  &LinearRegJointMixed::get_history)
        .method("predict_at",   &LinearRegJointMixed::predict_at)
        .method("get_dag",      &LinearRegJointMixed::get_dag)
        .method("readapt_NUTS", &LinearRegJointMixed::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(LinearRegJointMixed, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<LinearRegJointMixed>(m, "LinearRegJointMixed")
        .def(pybind11::init<arma::vec, arma::mat, int, bool>(),
             pybind11::arg("y"), pybind11::arg("X"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false,
             "Linear regression with (alpha, beta, sigma) sampled jointly "
             "via unified joint_nuts_block (REAL + POSITIVE per-slice). "
             "Migration of LinearRegJointMixed off the legacy mixed shim.")
        .def("step",        &LinearRegJointMixed::step, pybind11::arg("n_steps"))
        .def("get_current", &LinearRegJointMixed::get_current)
        .def("set_current", &LinearRegJointMixed::set_current, pybind11::arg("params"))
        .def("predict_at",  &LinearRegJointMixed::predict_at, pybind11::arg("new_data"))
        .def("get_dag",     &LinearRegJointMixed::get_dag)
        .def("get_history", &LinearRegJointMixed::get_history)
        .def("readapt_NUTS", &LinearRegJointMixed::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false);
}
#endif

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>
// ============================================================================
//  Standalone demo: simulate from a KNOWN truth, fit, recover (alpha, beta,
//  sigma) posterior means, and check against the truth (and a naive OLS-free
//  baseline). No R / Python binding required.
// ============================================================================
int main() {
    const std::size_t N = 300;
    const std::size_t p = 3;

    const double      alpha_true = 1.5;
    const arma::vec   beta_true  = {2.0, -1.0, 0.5};
    const double      sigma_true = 0.8;

    // ---- Simulate data from the known truth ---------------------------------
    std::mt19937_64 sim_rng(123);
    std::normal_distribution<double> xgen(0.0, 1.0);
    std::normal_distribution<double> egen(0.0, sigma_true);

    arma::mat X(N, p);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < p; ++j)
            X(i, j) = xgen(sim_rng);

    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        double mu = alpha_true;
        for (std::size_t j = 0; j < p; ++j) mu += X(i, j) * beta_true[j];
        y[i] = mu + egen(sim_rng);
    }

    // ---- Fit ----------------------------------------------------------------
    LinearRegJointMixed model(y, X, /*rng_seed=*/7, /*keep_history=*/false);
    model.step(800);   // warmup beyond the block's internal first-call warmup

    arma::vec beta_bar(p, arma::fill::zeros);
    double alpha_bar = 0.0, sigma_bar = 0.0;
    const int M = 2000;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        alpha_bar += cur.at("alpha")[0];
        sigma_bar += cur.at("sigma")[0];
        beta_bar  += cur.at("beta");
    }
    alpha_bar /= static_cast<double>(M);
    sigma_bar /= static_cast<double>(M);
    beta_bar  /= static_cast<double>(M);

    // ---- Report -------------------------------------------------------------
    std::printf("LinearRegJointMixed demo (N=%zu, p=%zu)\n", N, p);
    std::printf("  alpha_hat = %+.3f  (truth %+.3f)\n", alpha_bar, alpha_true);
    for (std::size_t j = 0; j < p; ++j)
        std::printf("  beta[%zu]_hat = %+.3f  (truth %+.3f)\n",
                    j, beta_bar[j], beta_true[j]);
    std::printf("  sigma_hat = %+.3f  (truth %+.3f)\n", sigma_bar, sigma_true);

    bool ok = std::abs(alpha_bar - alpha_true) < 0.15 &&
              std::abs(sigma_bar - sigma_true) < 0.15;
    for (std::size_t j = 0; j < p; ++j)
        ok = ok && std::abs(beta_bar[j] - beta_true[j]) < 0.15;

    std::printf("%s\n", ok
        ? "[demo PASS] joint-NUTS recovers (alpha, beta, sigma)"
        : "[demo FAIL] posterior mean off from truth");
    return ok ? 0 : 1;
}
#endif

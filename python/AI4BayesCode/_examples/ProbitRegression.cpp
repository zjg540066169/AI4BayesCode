// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  ProbitRegression.cpp
//
//  Bayesian probit linear regression via Albert-Chib (1993) data
//  augmentation. Reference example for `probit_aug_block` (the closed-
//  form Gibbs leaf for the latent z step).
//
//  Model
//  -----
//      y_i  ~ Bernoulli(p_i),  p_i = Phi(X_i' beta)        i = 1..N
//      beta ~ N(0, prior_sd^2 I)                            (weakly inf.)
//      z_i  ~ N(X_i' beta, 1) truncated by sign(2 y_i - 1)  (Albert-Chib)
//      sigma fixed at 1 (probit identifiability).
//
//  Block decomposition
//  -------------------
//      composite "ProbitRegression":
//        child(0) z    probit_aug_block
//                      reads y, mu_lin (= X * beta refresher)
//                      writes z (length N)
//        child(1) beta nuts_block (real, length p)
//                      log_density: Gaussian likelihood `z ~ N(X beta, 1)`
//                                   plus Gaussian prior `beta ~ N(0, sigma^2)`
//                      analytic gradient
//
//      Refresher: mu_lin = X * beta (deterministic, invalidated by beta).
//
//  WHY this composition
//  --------------------
//  Albert-Chib's z step is conditionally independent across i and is a
//  textbook closed-form vector conjugate sample (Exception 3 of
//  codegen_priors §2b). We therefore use the library-blessed
//  `probit_aug_block` instead of inlining the truncated-normal sampling
//  in this wrapper's step() method. The beta step uses NUTS on a clean
//  Gaussian-likelihood log-density (prior + Gaussian "z ~ N(X beta, 1)"
//  likelihood) so no custom Gibbs is required. This is the canonical
//  pattern for any "binary probit + linear / non-linear mean" model;
//  swap nuts_block for bart_block (with binary=true) to get probit BART.
//
//  CHECK #16 (inline justification)
//  --------------------------------
//  - z step is Exception 3 (closed-form vector conjugate sample,
//    NUTS-wasteful for N independent truncated normals). Library-blessed
//    via probit_aug_block.
//  - beta step uses NUTS on the natural-scale Gaussian log-density. No
//    Jacobian (real-valued, identity transform). Gradient is analytic and
//    derived in this file (see beta_log_density).
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("ProbitRegression")
//   set.seed(20260621)
//   N <- 800L; p <- 3L                       # N >> p: beta well-identified
//   beta_true <- c(-0.4, 1.2, -0.8)
//   X <- cbind(1, matrix(rnorm(N * (p - 1)), N, p - 1))   # intercept + 2 covars
//   eta <- as.numeric(X %*% beta_true)
//   y <- as.numeric(runif(N) < pnorm(eta))   # y ~ Bernoulli(Phi(X beta))
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(ProbitRegression, X, y, 10, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(ProbitRegression, X, y, 10, 7L, TRUE)  # X, y, prior_sd, seed, keep_history
//   m$step(2500); str(m$get_current())
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(20260621)
//   N, p = 800, 3                            # N >> p: beta well-identified
//   beta_true = np.array([-0.4, 1.2, -0.8])
//   X = np.column_stack([np.ones(N), rng.standard_normal((N, p - 1))])
//   y = (X @ beta_true + rng.standard_normal(N) > 0).astype(float)  # probit latent DGP = Bernoulli(Phi(Xb))
//   Mod = AI4BayesCode.example("ProbitRegression")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.ProbitRegression(X, y, 10, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.ProbitRegression(X, y, 10, 7, True)  # X, y, prior_sd, seed, keep_history
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
#include "AI4BayesCode/types.hpp"

#ifdef AI4BAYESCODE_RCPP_MODULE
#  include "AI4BayesCode/rcpp_wrap.hpp"
#endif

#include "AI4BayesCode/probit_aug_block.hpp"
#include "AI4BayesCode/nuts_block.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::probit_aug_block;
using AI4BayesCode::probit_aug_block_config;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;

namespace {

// log p(beta | z, X) = sum_i [ -0.5 (z_i - x_i' beta)^2 ]
//                      + sum_j [ -0.5 beta_j^2 / sigma_prior^2 ]
//                      + const
// gradient wrt beta = X' (z - X beta) - beta / sigma_prior^2
//
// NUTS sees natural-scale beta (real, identity transform), no Jacobian.
double beta_log_density(const arma::vec& beta,
                         const block_context& ctx,
                         arma::vec* grad) {
    const arma::vec& z       = ctx.at("z");
    const arma::vec& X_flat  = ctx.at("X");
    const double prior_sd2   = ctx.at("prior_sd")[0] * ctx.at("prior_sd")[0];
    const std::size_t N = z.n_elem;
    const std::size_t p = beta.n_elem;
    if (X_flat.n_elem != N * p) {
        if (grad) { grad->set_size(p); grad->zeros(); }
        return -std::numeric_limits<double>::infinity();
    }

    // X is stored column-major: X_flat[i + j * N] = X(i, j).
    // Compute residual r_i = z_i - sum_j X(i, j) * beta_j.
    arma::vec r(N);
    for (std::size_t i = 0; i < N; ++i) {
        double xb = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            xb += X_flat[i + j * N] * beta[j];
        }
        r[i] = z[i] - xb;
    }

    double lp = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        lp -= 0.5 * r[i] * r[i];
    }
    for (std::size_t j = 0; j < p; ++j) {
        lp -= 0.5 * beta[j] * beta[j] / prior_sd2;
    }

    if (grad) {
        grad->set_size(p);
        grad->zeros();
        // X' r contribution.
        for (std::size_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                s += X_flat[i + j * N] * r[i];
            }
            (*grad)[j] = s - beta[j] / prior_sd2;
        }
    }
    return lp;
}

}  // anonymous namespace

class ProbitRegression {
public:
    ProbitRegression(const arma::mat& X,
                     const arma::vec& y,
                     double prior_sd,
                     int rng_seed,
                     bool keep_history)
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
          impl_(std::make_unique<composite_block>("ProbitRegression")),
          keep_history_(keep_history)
    {
        if (X.n_rows == 0) throw std::runtime_error("X must have at least 1 row");
        if (X.n_cols == 0) throw std::runtime_error("X must have at least 1 column");
        if (y.n_elem != X.n_rows)
            throw std::runtime_error("y length must equal X row count");
        if (!(prior_sd > 0.0))
            throw std::runtime_error("prior_sd must be > 0");
        for (std::size_t i = 0; i < y.n_elem; ++i) {
            if (y[i] != 0.0 && y[i] != 1.0)
                throw std::runtime_error("y must be 0/1 (Bernoulli)");
        }
        N_ = X.n_rows;
        p_ = X.n_cols;

        // ---- Static data + hyperparameters ---------------------------
        impl_->data().set("y", y);
        impl_->data().set("X", arma::vectorise(X));   // column-major flat
        impl_->data().set("prior_sd", arma::vec{prior_sd});

        // ---- Initial state -------------------------------------------
        impl_->data().set("beta", arma::vec(p_, arma::fill::zeros));
        // Initial z = sign(2y-1) * 1.0 (a finite valid first draw).
        arma::vec z_init(N_);
        for (std::size_t i = 0; i < N_; ++i) {
            z_init[i] = (y[i] >= 0.5) ? 1.0 : -1.0;
        }
        impl_->data().set("z", z_init);
        // Initial mu_lin = X * 0 = 0, refresher will recompute.
        impl_->data().set("mu_lin", arma::vec(N_, arma::fill::zeros));

        // ---- Refresher: mu_lin = X * beta ----------------------------
        // Deterministic refresher invalidated whenever beta changes.
        // N_cur derived from X dynamically so the refresher adapts to
        // predict_at(list(X = X_new)) with new sample size.
        impl_->data().register_refresher(
            "mu_lin",
            [p = p_]
            (const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& X_flat = d.get("X");
                const arma::vec& beta   = d.get("beta");
                const std::size_t N_cur = X_flat.n_elem / p;
                arma::vec mu(N_cur);
                for (std::size_t i = 0; i < N_cur; ++i) {
                    double xb = 0.0;
                    for (std::size_t j = 0; j < p; ++j) {
                        xb += X_flat[i + j * N_cur] * beta[j];
                    }
                    mu[i] = xb;
                }
                return mu;
            });

        // ---- prob = Phi(mu_lin)  (deterministic intermediate) --------
        // Full predict-DAG reconstruction: the success probability is a
        // first-class node so predict_at exposes it (not inlined in
        // y_rep). Phi via the same 0.5*(1+erf(./sqrt2)) used by the old
        // collapsed y_rep -> behaviour-preserving.
        impl_->data().set("prob", arma::vec(N_, arma::fill::value(0.5)));
        impl_->data().register_refresher(
            "prob",
            [](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& mu = d.get("mu_lin");
                arma::vec prob(mu.n_elem);
                for (std::size_t i = 0; i < mu.n_elem; ++i)
                    prob[i] = 0.5 * (1.0 + std::erf(mu[i] / std::sqrt(2.0)));
                return prob;
            });

        // ---- Gibbs DAG dependencies ----------------------------------
        // z reads y and mu_lin; mu_lin (then prob) invalidated by beta;
        // beta reads z and X. Order in the invalidates list matters:
        // mu_lin before prob (prob's refresher reads mu_lin).
        impl_->data().declare_dependencies("z",      {"y", "mu_lin"});
        impl_->data().declare_dependencies("beta",   {"z", "X", "prior_sd"});
        impl_->data().declare_invalidates("beta",    {"mu_lin", "prob"});

        // ---- Predict DAG (full generative chain) ---------------------
        //   X,beta -> mu_lin -> prob -> y_rep
        // (Albert-Chib z is a sampling-augmentation latent, NOT a
        // generative node, so it is absent from the predict DAG.)
        impl_->data().declare_data_input("X");
        impl_->data().declare_predict_edges("X",      {"mu_lin"});
        impl_->data().declare_predict_edges("beta",   {"mu_lin"});
        impl_->data().declare_predict_edges("mu_lin", {"prob"});
        impl_->data().declare_predict_edges("prob",   {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). beta ~ N(0, prior_sd^2 I): the
        //      prior sd is beta's prior parent. (Albert-Chib z is a
        //      sampling-augmentation latent, not a generative node.)
        //      Drawn faded by ai4bayescode_plot_dag.
        impl_->data().declare_context_edges("prior_sd", {"beta"});
        impl_->data().set("y_rep", arma::vec(N_, arma::fill::zeros));
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                // Reads ONLY its direct generative parent `prob`. Same
                // per-i uniform draw order as the old collapsed y_rep, so
                // bit-identical under a fixed predict RNG.
                const arma::vec& prob = d.get("prob");
                std::uniform_real_distribution<double> ud(0.0, 1.0);
                arma::vec y_rep(prob.n_elem);
                for (std::size_t i = 0; i < prob.n_elem; ++i)
                    y_rep[i] = (ud(rng) < prob[i]) ? 1.0 : 0.0;
                return y_rep;
            });

        impl_->data().refresh_all();

        // ---- Children ------------------------------------------------
        // child(0): z (probit_aug_block, reads y + mu_lin)
        {
            probit_aug_block_config cfg;
            cfg.name      = "z";
            cfg.n_obs     = N_;
            cfg.y_key     = "y";
            cfg.mu_key    = "mu_lin";
            // No offset (offset_key empty).
            impl_->add_child(
                std::make_unique<probit_aug_block>(std::move(cfg)));
        }

        // child(1): beta (nuts_block, real, length p_)
        {
            nuts_block_config cfg;
            cfg.name        = "beta";
            cfg.initial_unc = arma::vec(p_, arma::fill::zeros);
            // Real-valued: identity transform (no constrain/unconstrain).
            cfg.log_density_grad =
                [](const arma::vec& beta_unc, const block_context& ctx,
                   arma::vec* grad) -> double {
                    return beta_log_density(beta_unc, ctx, grad);
                };
            impl_->add_child(std::make_unique<nuts_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep

    void step(int n_steps) {
        if (n_steps < 0) throw std::runtime_error("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["beta"] = impl_->data().get("beta");
        out["z"]    = impl_->data().get("z");
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto it_b = params.find("beta");
        if (it_b != params.end()) {
            const arma::vec& bnew = it_b->second;
            if (bnew.n_elem != p_)
                throw std::runtime_error("beta length must equal p");
            dynamic_cast<nuts_block&>(impl_->child(1)).set_current(bnew);
            impl_->data().set("beta", bnew);
            impl_->data().refresh_derived_for("beta");
        }
        auto it_z = params.find("z");
        if (it_z != params.end()) {
            const arma::vec& zn = it_z->second;
            if (zn.n_elem != N_)
                throw std::runtime_error("z length must equal N");
            dynamic_cast<probit_aug_block&>(impl_->child(0)).set_current(zn);
            impl_->data().set("z", zn);
        }
        // ProbitRegression uses Albert-Chib data augmentation: the
        // latent z vector of length N is allocated inside
        // probit_aug_block at construction. Changing N requires
        // reconstructing the aug block, so this wrapper enforces
        // STRICT-N (validator Check #21 legitimate use case —
        // codegen_cpp.md §7a). Same-N replacement of X/y is fine.
        auto it_X = params.find("X");
        if (it_X != params.end()) {
            const arma::vec& Xnew = it_X->second;
            if (Xnew.n_elem != N_ * p_)
                throw std::runtime_error("set_current: ProbitRegression "
                    "fixes N at construction (probit_aug_block holds the "
                    "length-N latent z). Supplied X has length " +
                    std::to_string(Xnew.n_elem) + " but requires N*p = " +
                    std::to_string(N_ * p_) +
                    ". To change N, reconstruct the wrapper.");
            impl_->data().set("X", Xnew);
            impl_->data().refresh_derived_for("beta");  // refresh mu_lin
        }
        auto it_y = params.find("y");
        if (it_y != params.end()) {
            const arma::vec& ynew = it_y->second;
            if (ynew.n_elem != N_)
                throw std::runtime_error("set_current: y length " +
                    std::to_string(ynew.n_elem) +
                    " != construction-time N = " + std::to_string(N_) +
                    ". ProbitRegression fixes N (probit_aug latent z); "
                    "reconstruct to change N.");
            for (std::size_t i = 0; i < N_; ++i) {
                if (ynew[i] != 0.0 && ynew[i] != 1.0)
                    throw std::runtime_error("y must be 0/1");
            }
            impl_->data().set("y", ynew);
        }
    }

    // predict_at: optional X = arma::vec (vectorised N*p_test).
    //   * keep_history = FALSE: single predict at current draw — each
    //     refreshed key returned as a 1-row arma::mat.
    //   * keep_history = TRUE:  loops over all posterior draws of beta —
    //     each refreshed key returned as an (n_draws x N) arma::mat (full
    //     posterior predictive).
    AI4BayesCode::history_map predict_at(
        const AI4BayesCode::state_map& new_data) const {
        // Parse X input once (shared across both modes).
        bool has_X = false;
        arma::vec x_flat;
        auto it_X = new_data.find("X");
        if (it_X != new_data.end()) {
            const arma::vec& X_test = it_X->second;
            if (X_test.n_elem % p_ != 0)
                throw std::runtime_error(
                    "predict_at: X length must be a multiple of p (vectorised)");
            x_flat = X_test;
            has_X  = true;
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

        // History mode: loop over all posterior draws of beta.
        AI4BayesCode::history_map hist = impl_->get_history();
        auto it_b = hist.find("beta");
        if (it_b == hist.end()) {
            throw std::runtime_error(
                "ProbitRegression::predict_at: keep_history_ requires beta "
                "history, but get_history() lacks it. Did you forget to "
                "construct with keep_history = TRUE?");
        }
        const arma::mat& beta_hist = it_b->second;
        const std::size_t n_draws  = beta_hist.n_rows;

        // Set X EVERY draw (new x_flat or training X) so predict_at
        // Pass-1 recomputes the deterministic chain mu_lin->prob for each
        // historical beta draw. Pass-1's strict availability rule treats
        // the data_input X as unavailable unless it is explicitly in the
        // replaced/changed set, which would otherwise leave mu_lin stale.
        const arma::vec X_for_draw = has_X ? x_flat : impl_->data().get("X");
        std::unordered_map<std::string, std::vector<arma::vec>> collected;
        for (std::size_t d = 0; d < n_draws; ++d) {
            block_context replaced;
            replaced["X"]    = X_for_draw;
            replaced["beta"] = beta_hist.row(d).t();
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result) {
                collected[kv.first].push_back(kv.second);
            }
        }

        for (const auto& kv : collected) {
            if (kv.second.empty()) continue;
            const std::size_t dim = kv.second[0].n_elem;
            arma::mat m(n_draws, dim);
            for (std::size_t i = 0; i < n_draws; ++i)
                for (std::size_t j = 0; j < dim; ++j)
                    m(i, j) = kv.second[i][j];
            out.emplace(kv.first, std::move(m));
        }
        return out;
    }

    AI4BayesCode::dag_info     get_dag()     const { return impl_->get_dag(); }
    AI4BayesCode::history_map  get_history() const { return impl_->get_history(); }

    /// 7th R-level method: re-tune NUTS metric (mass matrix + step size +
    /// dual averaging) without advancing chain state. Available because
    /// the composite contains NUTS-family children. See system_design.md
    /// §13 NUTS-family + validator.md §24.
    void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1) {
        if (n < 0) {
            ai4b::stop("readapt_NUTS: n must be non-negative");
        }
        impl_->readapt_NUTS(static_cast<std::size_t>(n),
                            reset, readapt_rng_, max_tree_depth < 0 ? std::size_t(0) : static_cast<std::size_t>(max_tree_depth));
    }


private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_; // readapt_NUTS() advances it (7th method)
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      N_ = 0;
    std::size_t                      p_ = 0;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(ProbitRegression_module) {
    Rcpp::class_<ProbitRegression>("ProbitRegression")
        .constructor<arma::mat, arma::vec, double, int, bool>(
            "Construct ProbitRegression(X, y, prior_sd, seed, keep_history). "
            "X is N x p, y is length N (0/1). beta ~ N(0, prior_sd^2 I). "
            "Albert-Chib data augmentation with NUTS on beta.")
        .method("step", (void (ProbitRegression::*)())    &ProbitRegression::step, "Run one sweep.")
        .method("step", (void (ProbitRegression::*)(int)) &ProbitRegression::step, "Run n sweeps.")
        .method("get_current", &ProbitRegression::get_current)
        .method("set_current", &ProbitRegression::set_current)
        .method("predict_at",  &ProbitRegression::predict_at)
        .method("get_dag",     &ProbitRegression::get_dag)
        .method("get_history", &ProbitRegression::get_history)
        .method("readapt_NUTS", &ProbitRegression::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(ProbitRegression, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<ProbitRegression>(m, "ProbitRegression")
        .def(pybind11::init<arma::mat, arma::vec, double, int, bool>(),
             pybind11::arg("X"), pybind11::arg("y"),
             pybind11::arg("prior_sd") = 10.0,
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false,
             "Bayesian probit regression via Albert-Chib data augmentation. "
             "y_i ~ Bernoulli(Phi(X_i' beta)), beta ~ N(0, prior_sd^2).")
        .def("step", (void (ProbitRegression::*)())    &ProbitRegression::step, "Run one sweep.")
        .def("step", (void (ProbitRegression::*)(int)) &ProbitRegression::step,
             pybind11::arg("n_steps"))
        .def("get_current", &ProbitRegression::get_current)
        .def("set_current", &ProbitRegression::set_current,
             pybind11::arg("params"))
        .def("predict_at",  &ProbitRegression::predict_at,
             pybind11::arg("new_data"))
        .def("get_dag",     &ProbitRegression::get_dag)
        .def("get_history", &ProbitRegression::get_history)
        .def("readapt_NUTS", &ProbitRegression::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false, pybind11::arg("max_tree_depth") = -1);
}
#endif

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
// ============================================================================
//  Standalone demo: simulate probit data from a known beta, fit, and check
//  posterior recovery of beta + in-sample classification accuracy vs a naive
//  majority-class baseline. Derives PASS/FAIL from actual computed numbers.
// ============================================================================
#include <cstdio>
int main() {
    const std::size_t N = 800;
    const std::size_t p = 3;             // intercept + 2 covariates
    const int seed      = 20260621;

    // ---- Known truth -------------------------------------------------------
    const arma::vec beta_true = {-0.4, 1.2, -0.8};

    // ---- Simulate X (intercept + N(0,1) covariates) and y ~ Bernoulli(Phi) -
    std::mt19937_64 sim_rng(static_cast<std::uint64_t>(seed));
    std::normal_distribution<double>      norm(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    arma::mat X(N, p);
    arma::vec y(N);
    std::size_t n_pos = 0;
    for (std::size_t i = 0; i < N; ++i) {
        X(i, 0) = 1.0;                   // intercept
        for (std::size_t j = 1; j < p; ++j) X(i, j) = norm(sim_rng);
        double eta = arma::dot(X.row(i).t(), beta_true);
        double prob = 0.5 * (1.0 + std::erf(eta / std::sqrt(2.0)));  // Phi(eta)
        y[i] = (unif(sim_rng) < prob) ? 1.0 : 0.0;
        n_pos += static_cast<std::size_t>(y[i]);
    }

    std::printf("Probit regression demo: N=%zu, p=%zu, positives=%zu\n",
                N, p, n_pos);
    std::printf("beta_true = [% .3f % .3f % .3f]\n",
                beta_true[0], beta_true[1], beta_true[2]);

    // ---- Fit ---------------------------------------------------------------
    const double prior_sd = 10.0;        // weakly informative
    ProbitRegression model(X, y, prior_sd, seed, /*keep_history=*/false);

    const int n_warmup = 1500;
    const int n_keep   = 2000;
    model.step(n_warmup);                // warmup (NUTS adapts during warmup)

    // Accumulate posterior mean of beta over kept draws.
    arma::vec beta_sum(p, arma::fill::zeros);
    for (int s = 0; s < n_keep; ++s) {
        model.step(1);
        AI4BayesCode::state_map cur = model.get_current();
        beta_sum += cur.at("beta");
    }
    arma::vec beta_hat = beta_sum / static_cast<double>(n_keep);

    std::printf("beta_hat  = [% .3f % .3f % .3f]\n",
                beta_hat[0], beta_hat[1], beta_hat[2]);

    // ---- Recovery metric: RMSE(beta_hat, beta_true) ------------------------
    double sse = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        double d = beta_hat[j] - beta_true[j];
        sse += d * d;
    }
    double rmse = std::sqrt(sse / static_cast<double>(p));

    // ---- In-sample classification accuracy of fitted model vs naive --------
    std::size_t correct_fit = 0, correct_naive = 0;
    double naive_pred = (n_pos >= N - n_pos) ? 1.0 : 0.0;  // majority class
    for (std::size_t i = 0; i < N; ++i) {
        double eta = arma::dot(X.row(i).t(), beta_hat);
        double phat = 0.5 * (1.0 + std::erf(eta / std::sqrt(2.0)));
        double yhat = (phat >= 0.5) ? 1.0 : 0.0;
        if (yhat == y[i])        ++correct_fit;
        if (naive_pred == y[i])  ++correct_naive;
    }
    double acc_fit   = static_cast<double>(correct_fit)   / static_cast<double>(N);
    double acc_naive = static_cast<double>(correct_naive) / static_cast<double>(N);

    std::printf("beta RMSE                 = %.4f\n", rmse);
    std::printf("fitted accuracy           = %.4f\n", acc_fit);
    std::printf("naive (majority) accuracy = %.4f\n", acc_naive);

    // ---- PASS criteria (all derived from computed numbers) -----------------
    //   (1) beta recovered to within RMSE < 0.20 of truth, and
    //   (2) fitted model beats the naive majority baseline by >= 0.05.
    bool ok = (rmse < 0.20) && (acc_fit >= acc_naive + 0.05);
    if (ok) std::printf("[demo PASS]\n");
    else    std::printf("[demo FAIL]\n");
    return ok ? 0 : 1;
}
#endif

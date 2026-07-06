// ============================================================================
//  VCBart.cpp
//
//  Copyright (C) 2026 AI4BayesCode.
//  Licensed under GPL-3.0-or-later (uses bart_block.hpp, which is GPL-2.0-or-later)
//
//  REFERENCE TEMPLATE for the AI4BayesCode skill -- VARYING-COEFFICIENT BART
//  (VC-BART, Deshpande et al.) built entirely from `bart_block` via BACKFITTING.
//  This is the worked example for the "Composite / varying-coefficient BART via
//  BACKFITTING" recipe in the `bart_block` catalogue card and the `genbart_block`
//  reduction ladder: a model that looks like it needs a custom genbart likelihood
//  is actually p+1 fast CONJUGATE Gaussian bart_block ensembles, Gibbs-backfit.
//
//  Model
//  -----
//      y_i | ...      ~ Normal(mu_i, sigma^2),   i = 1..N
//      mu_i           = beta_0(Z_i) + sum_{j=1..p} beta_j(Z_i) * x_ij
//      beta_j(.)      ~ BART tree-ensemble prior over the modifiers Z, j = 0..p
//      sigma^2        ~ InverseGamma(nu/2, nu*lambda/2)  (BART calibrated default)
//
//  Why bart_block and NOT genbart_block
//  ------------------------------------
//  The observation is Gaussian, so each ensemble keeps bart_block's conjugate
//  closed-form Gaussian leaves. The per-observation multiplier x_ij is handled by
//  a WEIGHTED sweep, not a custom likelihood. Condition ensemble j on the others
//  via the partial residual and fit the SCALED response:
//
//      r_i^(j)  = y_i - sum_{k != j} x_ik * beta_k(Z_i)     (partial residual)
//      ytilde_i = r_i^(j) / x_ij                            (working response)
//      ytilde_i ~ N( beta_j(Z_i),  sigma^2 / x_ij^2 )
//
//  so ensemble j is a bart_block over Z with working_response_key = ytilde^(j)
//  and weights_key w_i = 1/|x_ij| (=> per-obs noise sd w_i*sigma, forwarded to
//  bart_model::update_step_weighted). That reproduces VC-BART's weighted leaf
//  sufficient statistics (sum x_ij^2, sum x_ij r_i) EXACTLY -- p+1 conjugate BART
//  sweeps, far faster than a genbart RJMCMC over a custom VC likelihood.
//
//  Block decomposition
//  -------------------
//      beta_0..beta_p : one bart_block each (child 0..p), over the modifiers Z.
//                       Backfitting is native to composite_block: after each
//                       ensemble updates, refresh_derived_for() recomputes the
//                       OTHER ensembles' working responses (wr_k) and the full
//                       mean mu -- a deterministic systematic-scan Gibbs backfit.
//      sigma          : OWNED by beta_0 (the unweighted intercept ensemble),
//                       drawn each sweep by BART's calibrated conjugate Inv-Chi^2
//                       whose residual is exactly y - mu; beta_1..p read it as
//                       their weighted noise scale. No separate sigma block --
//                       a NUTS-on-sigma freezes under the BART mean/noise
//                       non-identifiability.
//
//  BACKEND-NEUTRAL DUAL MODULE
//  ---------------------------
//  Compiles under three backends from a single source:
//    * AI4BAYESCODE_RCPP_MODULE   -> RcppArmadillo, RCPP_MODULE (R)
//    * AI4BAYESCODE_PYBIND_MODULE -> pure armadillo, PYBIND11_MODULE (Python)
//    * neither                    -> standalone int main() demo
// ============================================================================
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("VCBart")
//   set.seed(1); N <- 600L; p <- 2L
//   Z  <- runif(N, -1, 1)                                  # single effect modifier
//   X  <- matrix(runif(N * p, 0.5, 1.5), N, p)             # predictors (bounded away from 0)
//   b0 <- sin(3 * Z); b1 <- Z^2; b2 <- -Z                  # true coefficient functions of Z
//   y  <- b0 + b1 * X[, 1] + b2 * X[, 2] + rnorm(N, 0, 0.3)
//   m  <- new(VCBart, X, matrix(Z, N, 1), y, 50L, 2.0, 2.0, 0.95, 3.0, 100L, 1L)
//   m$step(1500L); cur <- m$get_current()                 # $beta_0..$beta_2, $mu, $sigma
//   cor(cur$beta_0, sin(3 * Z)); cor(cur$beta_1, Z^2)      # recover the coefficient functions
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(1); N, p = 600, 2
//   Z  = rng.uniform(-1, 1, size=N)
//   X  = rng.uniform(0.5, 1.5, size=(N, p))
//   b0 = np.sin(3 * Z); b1 = Z**2; b2 = -Z
//   y  = b0 + b1 * X[:, 0] + b2 * X[:, 1] + rng.normal(0, 0.3, size=N)
//   Mod = AI4BayesCode.example("VCBart")
//   m = Mod.VCBart(X, Z.reshape(-1, 1), y, 50, 2.0, 2.0, 0.95, 3.0, 100, 1)
//   m.step(1500); cur = m.get_current()
//   print(np.corrcoef(cur["beta_1"], Z**2)[0, 1])         # recovers beta_1(Z) = Z^2
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

// AI4BayesCode core (pure C++)
#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"

// BART block (Rcpp-free, transitively pulls in the GPL BART kernel)
#include "AI4BayesCode/bart_block.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
using AI4BayesCode::bart_block;
using AI4BayesCode::bart_block_config;

namespace constraints = AI4BayesCode::constraints;

// (sigma is drawn by beta_0's BART kernel via its conjugate Inv-Chi^2; there is
//  no hand-written sigma log-density any more.)

// ============================================================================
//  User-facing class. Backend-neutral: arma containers, state_map/history_map.
// ============================================================================
class VCBart {
public:
    VCBart(const arma::mat& X,      // N x p predictors
           const arma::mat& Z,      // N x q effect modifiers
           const arma::vec& y,      // length N response
           int    ntrees,
           double k,
           double power,
           double base,
           double nu,
           int    numcut,
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
          impl_(std::make_unique<composite_block>("VCBart")),
          p_(static_cast<int>(X.n_cols)),
          Z_(Z),
          x_ncol_(static_cast<int>(X.n_cols)),
          z_ncol_(static_cast<int>(Z.n_cols)),
          keep_history_(keep_history)
    {
        if (X.n_rows != y.n_elem || Z.n_rows != y.n_elem) {
            ai4b::stop("VCBart: X, Z and y must have matching row counts");
        }
        const std::size_t N = y.n_elem;
        const int J = p_ + 1;                 // number of ensembles (intercept + p)

        // xcols_[j] is the per-observation multiplier for ensemble j:
        //   j == 0  -> ones (the varying intercept beta_0(Z))
        //   j >= 1  -> X.col(j-1)             (the varying coefficient beta_j(Z))
        auto xcols = std::make_shared<std::vector<arma::vec>>();
        xcols->reserve(J);
        xcols->push_back(arma::ones<arma::vec>(N));
        for (int j = 1; j < J; ++j) xcols->push_back(X.col(j - 1));
        xcols_ = xcols;

        impl_->data().set("y", y);

        // ---- One bart_block per coefficient function -------------------------
        for (int j = 0; j < J; ++j) {
            const std::string bj  = beta_key(j);
            const std::string wrj = wr_key(j);
            const std::string wj  = w_key(j);

            // Initial working response (all other betas = 0): ytilde = y / xcol_j.
            arma::vec wr0 = y / (*xcols)[j];
            // Per-observation weight w_i = 1/|x_ij|  (=> per-obs sd w_i*sigma).
            arma::vec w   = 1.0 / arma::abs((*xcols)[j]);

            impl_->data().set(bj,  arma::zeros<arma::vec>(N));  // beta_j(Z_i)
            impl_->data().set(wrj, wr0);                        // working response
            impl_->data().set(wj,  w);                          // constant weights

            // wr_j refresher: partial residual (y - sum_{k!=j} x_ik beta_k) / x_ij.
            impl_->data().register_refresher(
                wrj,
                [xcols, J, j](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                    const arma::vec& yv = d.get("y");
                    arma::vec wr = yv;
                    for (int kk = 0; kk < J; ++kk) {
                        if (kk == j) continue;
                        const arma::vec& bk = d.get(VCBart::beta_key(kk));
                        const arma::vec& xk = (*xcols)[kk];
                        for (std::size_t i = 0; i < wr.n_elem; ++i)
                            wr[i] -= xk[i] * bk[i];
                    }
                    const arma::vec& xj = (*xcols)[j];
                    for (std::size_t i = 0; i < wr.n_elem; ++i) wr[i] /= xj[i];
                    return wr;
                });

            // beta_0 is the intercept (x_i0 == 1): UNWEIGHTED, and it OWNS the
            // shared sigma -- it draws sigma internally via BART's calibrated
            // conjugate Inv-Chi^2 (BartNoise's native path), and its residual is
            // exactly the full VC-BART residual y - mu. beta_1..p are weighted
            // (w_i = 1/|x_ij|) and READ that sigma. Using BART's conjugate draw
            // (not a nuts_block on sigma) avoids the BART mean/noise non-
            // identifiability that freezes a NUTS-on-sigma.
            if (j == 0) impl_->data().declare_dependencies(bj, {wrj});
            else        impl_->data().declare_dependencies(bj, {wrj, wj, "sigma"});
            // When beta_j updates, every OTHER ensemble's working response is
            // stale, and so is the full mean mu. Refresh them (systematic-scan
            // Gibbs backfitting). beta_0 also re-publishes the sigma it just drew.
            std::vector<std::string> inval;
            for (int kk = 0; kk < J; ++kk) if (kk != j) inval.push_back(wr_key(kk));
            inval.push_back("mu");
            if (j == 0) inval.push_back("sigma");
            impl_->data().declare_invalidates(bj, inval);

            bart_block_config bc;
            bc.name                 = bj;
            bc.x_train              = Z;              // ensembles are BARTs over Z
            bc.y_init               = wr0;            // for the leaf-scale prior
            bc.working_response_key = wrj;
            // beta_0: unweighted + draws sigma internally (empty sigma_key).
            // beta_1..p: weighted by 1/|x_ij|, reading the shared sigma.
            bc.weights_key          = (j == 0) ? std::string() : wj;
            bc.sigma_key            = (j == 0) ? std::string() : std::string("sigma");
            bc.ntrees               = ntrees;
            bc.k                    = k;
            bc.power                = power;
            bc.base                 = base;
            bc.nu                   = nu;
            bc.numcut               = numcut;
            bc.sigma_init           = arma::stddev(y);
            impl_->add_child(std::make_unique<bart_block>(std::move(bc)));
        }

        // ---- Full mean mu = sum_j x_ij beta_j(Z_i) (refresher) ---------------
        impl_->data().set("mu", arma::zeros<arma::vec>(N));
        impl_->data().register_refresher(
            "mu",
            [xcols, J](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                arma::vec mu = arma::zeros<arma::vec>(d.get("y").n_elem);
                for (int kk = 0; kk < J; ++kk) {
                    const arma::vec& bk = d.get(VCBart::beta_key(kk));
                    const arma::vec& xk = (*xcols)[kk];
                    for (std::size_t i = 0; i < mu.n_elem; ++i) mu[i] += xk[i] * bk[i];
                }
                return mu;
            });

        // ---- Shared sigma is OWNED by beta_0 (child 0) ----------------------
        // beta_0 draws sigma each sweep via BART's calibrated conjugate Inv-Chi^2
        // (nu passed through to its bart_model). A refresher publishes beta_0's
        // current sigma to the "sigma" key so beta_1..p (weighted) read it.
        impl_->data().set("sigma",
            arma::vec{dynamic_cast<bart_block&>(impl_->child(0)).current_sigma()});
        {
            composite_block* comp = impl_.get();
            impl_->data().register_refresher(
                "sigma",
                [comp](const AI4BayesCode::shared_data_t&) -> arma::vec {
                    return arma::vec{
                        dynamic_cast<bart_block&>(comp->child(0)).current_sigma()};
                });
        }

        // Prime all derived keys (wr_j, mu) before the first sweep.
        impl_->data().refresh_all();

        // sigma is drawn by beta_0 (child 0) via BART's internal conjugate
        // Inv-Chi^2 -- there is NO separate sigma block (see the ensemble loop).

        // ---- predict DAG (generative direction) ------------------------------
        impl_->data().declare_data_input("X");
        impl_->data().declare_data_input("Z");
        for (int j = 0; j < J; ++j) {
            impl_->data().declare_predict_edges("Z", {beta_key(j)});
            impl_->data().declare_predict_edges(beta_key(j), {"mu"});
        }
        impl_->data().declare_predict_edges("X",     {"mu"});
        impl_->data().declare_predict_edges("mu",    {"y_rep"});
        impl_->data().declare_predict_edges("sigma", {"y_rep"});

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg: one sweep (Rcpp/pybind default)
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) {
            impl_->step(rng_);
            // sigma is a refresher-derived key (drawn by beta_0), NOT a child
            // block, so the composite does not record it -- buffer it here so
            // get_history() can surface the sigma chain for diagnostics.
            if (keep_history_)
                sigma_hist_.push_back(impl_->data().get("sigma")[0]);
        }
    }

    // Current draw: each coefficient function at the training Z, the full mean
    // mu, and sigma.
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        for (int j = 0; j <= p_; ++j)
            out[beta_key(j)] = impl_->data().get(beta_key(j));
        out["mu"]    = impl_->data().get("mu");
        out["sigma"] = impl_->data().get("sigma");
        return out;
    }

    // Update from an outer sampler. Supported keys:
    //   sigma : scalar > 0.
    //   y     : length-N response; refreshes every working response + mu.
    void set_current(const AI4BayesCode::state_map& params) {
        auto it_s = params.find("sigma");
        if (it_s != params.end()) impl_->data().set("sigma", it_s->second);
        auto it_y = params.find("y");
        if (it_y != params.end()) {
            impl_->data().set("y", it_y->second);
            impl_->data().refresh_all();   // wr_j and mu depend on y
        }
    }

    // Predict at new data. Pass list(Z = as.vector(Z_new), X = as.vector(X_new))
    // (flattened column-major). Returns the predicted mean mu at the new points.
    AI4BayesCode::state_map predict_at(const AI4BayesCode::state_map& new_data) {
        auto it_z = new_data.find("Z");
        auto it_x = new_data.find("X");
        if (it_z == new_data.end() || it_x == new_data.end())
            ai4b::stop("VCBart::predict_at: needs both 'Z' and 'X' (flattened "
                       "column-major new-data matrices).");
        const std::size_t n_new = it_z->second.n_elem / static_cast<std::size_t>(z_ncol_);
        arma::mat Z_new = arma::reshape(it_z->second, n_new, z_ncol_);
        arma::mat X_new = arma::reshape(it_x->second, n_new, x_ncol_);

        arma::vec mu_new(n_new, arma::fill::zeros);
        for (int j = 0; j <= p_; ++j) {
            auto& bj = dynamic_cast<bart_block&>(impl_->child(static_cast<std::size_t>(j)));
            arma::vec beta_j = bj.predict(Z_new);          // beta_j(Z_new)
            for (std::size_t i = 0; i < n_new; ++i) {
                const double xij = (j == 0) ? 1.0 : X_new(i, j - 1);
                mu_new[i] += xij * beta_j[i];
            }
        }
        AI4BayesCode::state_map out;
        out["mu"] = mu_new;
        return out;
    }

    AI4BayesCode::dag_info    get_dag()     const { return impl_->get_dag(); }
    AI4BayesCode::history_map get_history() const {
        AI4BayesCode::history_map h = impl_->get_history();
        if (!sigma_hist_.empty()) {
            arma::mat sm(sigma_hist_.size(), 1);
            for (std::size_t i = 0; i < sigma_hist_.size(); ++i) sm(i, 0) = sigma_hist_[i];
            h["sigma"] = sm;
        }
        return h;
    }

    void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1) {
        if (n < 0) ai4b::stop("readapt_NUTS: n must be non-negative");
        impl_->readapt_NUTS(static_cast<std::size_t>(n), reset, readapt_rng_, max_tree_depth < 0 ? std::size_t(0) : static_cast<std::size_t>(max_tree_depth));
    }

    // shared_data key helpers (also used by the refresher closures).
    static std::string beta_key(int j) { return "beta_" + std::to_string(j); }
    static std::string wr_key(int j)   { return "wr_"   + std::to_string(j); }
    static std::string w_key(int j)    { return "w_"    + std::to_string(j); }

private:
    std::mt19937_64 rng_;
    std::mt19937_64 predict_rng_;
    std::mt19937_64 readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    int  p_;
    arma::mat Z_;
    int  x_ncol_;
    int  z_ncol_;
    bool keep_history_;
    std::shared_ptr<std::vector<arma::vec>> xcols_;
    std::vector<double> sigma_hist_;   // per-sweep sigma (recorded when keep_history_)
};

// ============================================================================
//  Rcpp module.
// ============================================================================
#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(VCBart_module) {
    Rcpp::class_<VCBart>("VCBart")
        .constructor<arma::mat, arma::mat, arma::vec,
                     int, double, double, double, double, int, int>(
            "Short constructor; keep_history defaults FALSE.")
        .constructor<arma::mat, arma::mat, arma::vec,
                     int, double, double, double, double, int, int, bool>(
            "Construct with: X (N x p predictors), Z (N x q effect modifiers), "
            "y (length N), ntrees, k (leaf prior scale, default 2), power "
            "(tree depth penalty, default 2), base (tree depth base, default "
            "0.95), nu (sigma prior df, default 3), numcut (cutpoints per var, "
            "default 100), seed (RNG seed, 0 = random), keep_history (record "
            "per-step buffers; cheap; default FALSE).")
        .method("step", (void (VCBart::*)())    &VCBart::step, "Run one Gibbs backfitting sweep.")
        .method("step", (void (VCBart::*)(int)) &VCBart::step, "Run n Gibbs backfitting sweeps.")
        .method("get_current",  &VCBart::get_current,
                "Named list: $beta_0..$beta_p (coefficient functions at the "
                "training Z), $mu (full mean), $sigma.")
        .method("set_current",  &VCBart::set_current,
                "Overwrite sigma and/or y (named list). y refreshes every "
                "working response + the full mean.")
        .method("predict_at",   &VCBart::predict_at,
                "Predict mu at new data. Pass list(Z = as.vector(Z_new), "
                "X = as.vector(X_new)) (flattened column-major).")
        .method("get_dag",      &VCBart::get_dag,     "Predict DAG as edge list.")
        .method("get_history",  &VCBart::get_history, "History as named matrices.")
        .method("readapt_NUTS", &VCBart::readapt_NUTS);
}
#endif

// ============================================================================
//  pybind11 module.
// ============================================================================
#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(VCBart, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<VCBart>(m, "VCBart")
        .def(pybind11::init<arma::mat, arma::mat, arma::vec,
                            int, double, double, double, double, int, int, bool>(),
             pybind11::arg("X"),
             pybind11::arg("Z"),
             pybind11::arg("y"),
             pybind11::arg("ntrees")       = 50,
             pybind11::arg("k")            = 2.0,
             pybind11::arg("power")        = 2.0,
             pybind11::arg("base")         = 0.95,
             pybind11::arg("nu")           = 3.0,
             pybind11::arg("numcut")       = 100,
             pybind11::arg("rng_seed")     = 1,
             pybind11::arg("keep_history") = false)
        .def("step", (void (VCBart::*)())    &VCBart::step, "Run one Gibbs backfitting sweep.")
        .def("step", (void (VCBart::*)(int)) &VCBart::step, pybind11::arg("n_steps"))
        .def("get_current",  &VCBart::get_current)
        .def("set_current",  &VCBart::set_current, pybind11::arg("params"))
        .def("predict_at",   &VCBart::predict_at, pybind11::arg("new_data"))
        .def("get_dag",      &VCBart::get_dag)
        .def("get_history",  &VCBart::get_history)
        .def("readapt_NUTS", &VCBart::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false, pybind11::arg("max_tree_depth") = -1);
}
#endif

// ============================================================================
//  Standalone demo: simulate VC-BART data, fit, print coefficient recovery.
// ============================================================================
#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>

int main() {
    const std::size_t N = 600;
    const int p = 2;
    std::mt19937_64 g(1);
    std::uniform_real_distribution<double> uZ(-1.0, 1.0), uX(0.5, 1.5);
    std::normal_distribution<double> eps(0.0, 0.3);

    arma::mat X(N, p), Z(N, 1);
    arma::vec y(N), b0t(N), b1t(N), b2t(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double z = uZ(g);
        Z(i, 0) = z;
        X(i, 0) = uX(g); X(i, 1) = uX(g);
        b0t[i] = std::sin(3.0 * z);
        b1t[i] = z * z;
        b2t[i] = -z;
        y[i]   = b0t[i] + b1t[i] * X(i, 0) + b2t[i] * X(i, 1) + eps(g);
    }

    VCBart m(X, Z, y, 50, 2.0, 2.0, 0.95, 3.0, 100, 1, false);
    m.step(1500);
    auto cur = m.get_current();

    auto corr = [](const arma::vec& a, const arma::vec& b) {
        return arma::as_scalar(arma::cor(a, b));
    };
    std::printf("VC-BART recovery (correlation of fitted beta_j(Z) with truth):\n");
    std::printf("  beta_0 (sin 3Z): %.3f\n", corr(cur["beta_0"], b0t));
    std::printf("  beta_1 (Z^2)   : %.3f\n", corr(cur["beta_1"], b1t));
    std::printf("  beta_2 (-Z)    : %.3f\n", corr(cur["beta_2"], b2t));
    std::printf("  sigma (true 0.3): %.3f\n", cur["sigma"][0]);
    return 0;
}
#endif

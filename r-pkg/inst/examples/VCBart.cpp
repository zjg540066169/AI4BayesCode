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
//  Prediction (predict_at)
//  -----------------------
//  Output keys mu and y_rep, in this order in the R list and the Python dict
//  (mu first, as in 1.1.2). new_data is empty (training data) or holds both
//  Z (N_new x q) and X (N_new x p), flattened column-major; x_i0 = 1.
//    * single draw (keep_history = FALSE, or last_draw_only = TRUE), 1 x N:
//        mu_i = sum_{j=0..p} x_ij beta_j(Z_i), each beta_j evaluated from the
//        current forest of ensemble j at Z; y_rep_i = mu_i + sigma * eps_i
//        with the current sigma.
//    * keep_history = TRUE, training data, one row per stored draw d:
//        mu_{d,i} = sum_{j=0..p} x_ij beta_{j,d}(Z_i), from the stored rows of
//        beta_0..beta_p (get_history()).
//    * keep_history = TRUE, new (X, Z), one row per stored draw d: beta_{j,d}
//        is the forest of ensemble j stored at sweep d, evaluated at Z_new,
//        plus the fmean of that sweep (bart_block::predict_history). The
//        forests are stored only with keep_tree = TRUE; without it the call
//        raises an error when more than one draw is stored, and with at most
//        one stored draw the one row comes from the current forests.
//    In both history modes y_rep_{d,i} = mu_{d,i} + sigma_d * eps_{d,i}, with
//    sigma_d the sigma of sweep d (row d of get_history()$sigma). With no
//    sweep stored since construction or clear_history(), history mode gives
//    one row, from the current forests and the current sigma.
//    eps are iid N(0, 1) draws from predict_rng_, a std::mt19937_64 seeded at
//    construction (from rng_seed; from std::random_device when rng_seed = 0)
//    and used only by predict_at. Each call advances it, so y_rep differs
//    between calls; the MCMC draws of step() do not depend on it.
//
//  BACKEND-NEUTRAL DUAL MODULE
//  ---------------------------
//  Compiles under three backends from a single source:
//    * AI4BAYESCODE_RCPP_MODULE   -> RcppArmadillo, RCPP_MODULE (R)
//    * AI4BAYESCODE_PYBIND_MODULE -> pure armadillo, PYBIND11_MODULE (Python)
//    * neither                    -> standalone int main() demo
//
//  RNG / SEEDING
//  -------------
//  The bart_blocks draw from the BART kernel stream, not from the
//  std::mt19937_64 passed to step(). Under the pybind / standalone backend
//  that stream is bart_rng::engine() (bart_pure_cpp/src/r_compat.h), a
//  thread-local std::mt19937_64 shared by the BART-family models of the
//  thread in one compiled binary (each pybind extension module has its
//  own), and a non-zero rng_seed seeds it (bart_rng::set_seed) in the
//  constructor. A model reproduces the chain of its seed, and models with
//  different seeds give different chains, when no other BART-family model
//  of the thread in the same binary is constructed or stepped between the
//  model's construction and its last sweep (run_chains constructs and runs
//  one chain at a time in each process). Under the Rcpp backend the kernel
//  draws from R's RNG; call set.seed() before construction for a
//  reproducible R run.
// ============================================================================
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("VCBart")
//   set.seed(1); N <- 600L; p <- 2L
//   Z  <- runif(N, -1, 1)                                  # single effect modifier
//   X  <- matrix(runif(N * p, 0.5, 1.5), N, p)             # predictors (bounded away from 0)
//   b0 <- 2 * pmax(Z, 0); b1 <- Z^2; b2 <- -Z              # true coefficient functions of Z
//   y  <- b0 + b1 * X[, 1] + b2 * X[, 2] + rnorm(N, 0, 0.3)
//   m  <- new(VCBart, X, matrix(Z, N, 1), y, rng_seed = 1L)  # knobs defaulted
//   m$step(1500L); cur <- m$get_current()                 # $beta_0..$beta_2, $mu, $sigma
//   cor(cur$beta_0, pmax(Z, 0)); cor(cur$beta_1, Z^2)      # recover the coefficient functions
//   # ---- Posterior of mu: keep_history, plus keep_tree for new (X, Z) ----
//   mh <- new(VCBart, X, matrix(Z, N, 1), y, rng_seed = 1L,
//             keep_history = TRUE, keep_tree = TRUE)
//   mh$step(1000L); mh$clear_history(); mh$step(500L)     # 500 stored draws
//   Zn <- matrix(seq(-1, 1, length.out = 21), 21, 1); Xn <- matrix(1, 21, p)  # new (X, Z)
//   dim(mh$predict_at(list())$mu)                         # 500 x 600: per stored draw, training data
//   dim(mh$predict_at(list(Z = Zn, X = Xn))$mu)           # 500 x 21: per stored draw, new (X, Z)
//   dim(mh$predict_at(list(Z = Zn, X = Xn), TRUE)$mu)     # 1 x 21: current forests (last_draw_only)
//   dim(mh$predict_at(list(Z = Zn, X = Xn))$y_rep)        # 500 x 21: row d = mu_d + sigma_d * N(0, 1) noise
//   pp <- mh$predict_at(list())                           # training data: mu and y_rep, 500 x 600 each
//   mean(apply(pp$y_rep - pp$mu, 1, sd) / mh$get_history()$sigma)  # about 1: the noise of row d has sd sigma_d
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(1); N, p = 600, 2
//   Z  = rng.uniform(-1, 1, size=N)
//   X  = rng.uniform(0.5, 1.5, size=(N, p))
//   b0 = 2 * np.maximum(Z, 0); b1 = Z**2; b2 = -Z
//   y  = b0 + b1 * X[:, 0] + b2 * X[:, 1] + rng.normal(0, 0.3, size=N)
//   Mod = AI4BayesCode.example("VCBart")
//   m = Mod.VCBart(X, Z.reshape(-1, 1), y, rng_seed=1)   # every tuning knob defaulted
//   m.step(1500); cur = m.get_current()
//   print(np.corrcoef(cur["beta_1"], Z**2)[0, 1])         # recovers beta_1(Z) = Z^2
//   # ---- Posterior of mu: keep_history, plus keep_tree for new (X, Z) ----
//   mh = Mod.VCBart(X, Z.reshape(-1, 1), y, rng_seed=1, keep_history=True, keep_tree=True)
//   mh.step(1000); mh.clear_history(); mh.step(500)       # 500 stored draws
//   Zn = np.linspace(-1, 1, 21); Xn = np.ones((21, p))    # new (X, Z)
//   nd = {"Z": Zn, "X": Xn.flatten(order="F")}            # matrices flattened column-major
//   print(mh.predict_at({})["mu"].shape)                  # (500, 600): per stored draw, training data
//   print(mh.predict_at(nd)["mu"].shape)                  # (500, 21): per stored draw, new (X, Z)
//   print(mh.predict_at(nd, last_draw_only=True)["mu"].shape)  # (1, 21): current forests
//   print(mh.predict_at(nd)["y_rep"].shape)               # (500, 21): row d = mu_d + sigma_d * N(0, 1) noise
//   pp = mh.predict_at({})                                # training data: mu and y_rep, (500, 600) each
//   sd = (pp["y_rep"] - pp["mu"]).std(axis=1, ddof=1)
//   print(np.mean(sd / mh.get_history()["sigma"][:, 0]))  # about 1: the noise of row d has sd sigma_d
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
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"

// BART block (Rcpp-free, transitively pulls in the GPL BART kernel)
#include "AI4BayesCode/bart_block.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::bart_block;
using AI4BayesCode::bart_block_config;

namespace constraints = AI4BayesCode::constraints;

// (sigma is drawn by beta_0's BART kernel via its conjugate Inv-Chi^2; there is
//  no hand-written sigma log-density any more.)

// ============================================================================
//  User-facing class. Backend-neutral: arma containers, state_map/history_map.
// ============================================================================
class VCBart : public AI4BayesCode::kernel_control_mixin<VCBart> {
    friend class AI4BayesCode::kernel_control_mixin<VCBart>;
public:
    VCBart(const arma::mat& X,      // N x p predictors
           const arma::mat& Z,      // N x q effect modifiers
           const arma::vec& y,      // length N response
           int    ntrees = 50,
           double k = 2.0,
           double power = 2.0,
           double base = 0.95,
           double nu = 3.0,
           int    numcut = 100,
           int    rng_seed = 1,
           bool   keep_history = false,
           bool   keep_tree = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          impl_(std::make_unique<composite_block>("VCBart")),
          p_(static_cast<int>(X.n_cols)),
          Z_(Z),
          x_ncol_(static_cast<int>(X.n_cols)),
          z_ncol_(static_cast<int>(Z.n_cols)),
          keep_history_(keep_history),
          keep_tree_(keep_tree)
    {
        if (X.n_rows != y.n_elem || Z.n_rows != y.n_elem) {
            ai4b::stop("VCBart: X, Z and y must have matching row counts");
        }

        // ---- Seed the kernel RNG so a construction seed yields an
        //      independent chain (pybind / standalone backend only; the
        //      Rcpp backend uses R's global RNG instead). ----------------
#if !defined(AI4BAYESCODE_RCPP_MODULE)
        if (rng_seed != 0) {
            bart_rng::set_seed(static_cast<std::uint64_t>(rng_seed));
        }
#endif

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
        // NOT declare_data_input_group: this wrapper's predict_at never
        // routes through composite_block::predict_at, because the varying
        // coefficients beta_j(Z) are BART forests that have to be RE-EVALUATED
        // at the new Z (the current forests, or the forest stored at each
        // sweep in history mode) rather than refreshed from shared_data. The
        // co-indexing of X and Z is therefore enforced by predict_at's own check, which is
        // also the more useful answer here: mu combines beta_j(Z_i) with X_ij,
        // so with only one of the two supplied NOTHING is predictable, and an
        // explicit message beats an empty result.
        impl_->data().declare_data_input("X");
        impl_->data().declare_data_input("Z");
        for (int j = 0; j < J; ++j) {
            impl_->data().declare_predict_edges("Z", {beta_key(j)});
            impl_->data().declare_predict_edges(beta_key(j), {"mu"});
        }
        impl_->data().declare_predict_edges("X",     {"mu"});
        impl_->data().declare_predict_edges("mu",    {"y_rep"});
        impl_->data().declare_predict_edges("sigma", {"y_rep"});

        // keep_tree (the forest of every sweep, in each bart_block's kernel;
        // needed for predict_at at new (X, Z) in history mode) and
        // keep_history (numeric per-sweep buffers) are independent switches.
        if (keep_tree_)    impl_->set_keep_tree(true);
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
    //   sigma : scalar, equal to the current sigma (sigma is drawn inside
    //           the BART kernel of beta_0, which has no loader for it, so
    //           another value is rejected).
    //   y     : length-N response; refreshes every working response + mu.
    // Both keys are checked before anything is written.
    void set_current(const AI4BayesCode::state_map& params) {
        auto it_s = params.find("sigma");
        if (it_s != params.end() &&
            (it_s->second.n_elem != 1 ||
             it_s->second[0] != impl_->data().get("sigma")[0]))
            ai4b::stop("VCBart::set_current: sigma is drawn inside the BART "
                       "kernel of beta_0 and cannot be loaded; only its "
                       "current value is accepted");
        auto it_y = params.find("y");
        if (it_y != params.end() &&
            it_y->second.n_elem != impl_->data().get("y").n_elem)
            ai4b::stop("VCBart::set_current: y has length %d but N = %d",
                       (int)it_y->second.n_elem,
                       (int)impl_->data().get("y").n_elem);
        if (it_y != params.end()) {
            impl_->data().set("y", it_y->second);
            impl_->data().refresh_all();   // wr_j and mu depend on y
        }
    }

    // Predict mu and y_rep at new data, or at the training data (empty map).
    // new_data: Z (N_new x q) and X (N_new x p), each flattened column-major
    // (Rcpp flattens an R matrix that way), or an empty map for the training
    // Z and X. Output keys mu and y_rep, with x_i0 = 1:
    //   * keep_history = FALSE, or last_draw_only = TRUE: 1 x N_new,
    //       mu_i = sum_{j=0..p} x_ij beta_j(Z_i), each beta_j evaluated from
    //       the current forest of ensemble j; y_rep_i = mu_i + sigma * eps_i
    //       with the current sigma.
    //   * keep_history = TRUE, empty map: n_draws x N; row d of mu is
    //       computed from row d of the stored beta_0..beta_p (get_history()).
    //   * keep_history = TRUE, Z and X given: n_draws x N_new; row d of mu is
    //       computed from the forests stored at sweep d
    //       (bart_block::predict_history: stored forest d at Z_new plus the
    //       fmean of sweep d). Requires keep_tree = TRUE when more than one
    //       draw is stored.
    //   In both history modes y_rep_{d,i} = mu_{d,i} + sigma_d * eps_{d,i},
    //   with sigma_d = row d of the stored sigma (get_history()["sigma"]); with
    //   no sweep stored, the one row is computed with the current sigma.
    // eps_{d,i} are iid N(0, 1), drawn from predict_rng_ row by row (d outer,
    // i inner). The MCMC state and the MCMC random streams (rng_, the BART
    // kernel stream) are not changed.
    /// const and returning history_map per the core-six contract
    /// (interface.md): predict_at never mutates MCMC state, and its outputs
    /// are matrices so the stateful and history modes share one shape.
    /// One-argument form: predict_at(new_data) is
    /// predict_at(new_data, last_draw_only = false).
    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        return predict_at(new_data, /*last_draw_only=*/false);
    }

    /// last_draw_only = true takes the SINGLE-DRAW path -- the one this
    /// function already takes when keep_history is false -- without giving up
    /// the retained history.
    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data,
                                         bool last_draw_only) const {
        const bool use_history = keep_history_ && !last_draw_only;
        for (const auto& kv : new_data)
            if (kv.first != "Z" && kv.first != "X")
                ai4b::stop("VCBart::predict_at: unknown key '%s'", kv.first.c_str());
        auto it_z = new_data.find("Z");
        auto it_x = new_data.find("X");
        if ((it_z == new_data.end()) != (it_x == new_data.end()))
            ai4b::stop("VCBart::predict_at: supply BOTH 'Z' and 'X', or neither "
                       "(neither = predict at the training data).");
        const bool has_new = it_z != new_data.end();
        const std::size_t J = static_cast<std::size_t>(p_) + 1;   // ensembles

        arma::mat Z_new, X_new;
        if (!has_new) {                               // training data
            Z_new = Z_;
            X_new.set_size(Z_.n_rows, static_cast<arma::uword>(x_ncol_));
            // xcols_[0] is the intercept column; X.col(j) is xcols_[j + 1].
            for (int j = 0; j < x_ncol_; ++j) X_new.col(j) = (*xcols_)[j + 1];
        } else {
            const std::size_t q = static_cast<std::size_t>(z_ncol_);
            const std::size_t p = static_cast<std::size_t>(x_ncol_);
            if (it_z->second.n_elem % q != 0)
                ai4b::stop("VCBart::predict_at: Z has %d elements, not a multiple "
                           "of q = %d (Z is N_new x q, flattened column-major)",
                           (int)it_z->second.n_elem, (int)q);
            const std::size_t n_in = it_z->second.n_elem / q;
            if (it_x->second.n_elem != n_in * p)
                ai4b::stop("VCBart::predict_at: X has %d elements; Z gives "
                           "N_new = %d and p = %d (X is N_new x p, flattened "
                           "column-major)",
                           (int)it_x->second.n_elem, (int)n_in, (int)p);
            Z_new = arma::reshape(it_z->second, n_in, q);
            X_new = arma::reshape(it_x->second, n_in, p);
        }
        const std::size_t n_new = Z_new.n_rows;
        // x_ij: the multiplier of ensemble j at point i (x_i0 = 1).
        auto x_of = [&X_new](std::size_t j, std::size_t i) {
            return (j == 0) ? 1.0 : X_new(i, j - 1);
        };

        AI4BayesCode::history_map out;
        std::normal_distribution<double> norm01(0.0, 1.0);   // eps of y_rep
        if (!use_history) {
            // ---- Single draw: the current forest of each ensemble --------
            arma::mat mu(1, n_new, arma::fill::zeros);
            for (std::size_t j = 0; j < J; ++j) {
                auto& bj = dynamic_cast<bart_block&>(impl_->child(j));
                const arma::vec beta_j = bj.predict(Z_new);       // beta_j(Z_new)
                for (std::size_t i = 0; i < n_new; ++i) mu(0, i) += x_of(j, i) * beta_j[i];
            }
            // y_rep_i = mu_i + sigma * eps_i, sigma the current sigma.
            const double sigma = impl_->data().get("sigma")[0];
            arma::mat y_rep(1, n_new);
            for (std::size_t i = 0; i < n_new; ++i)
                y_rep(0, i) = mu(0, i) + sigma * norm01(predict_rng_);
            // The R list and the Python dict are built in the iteration order
            // of history_map (an unordered_map). With libc++ and libstdc++ the
            // iteration order of two keys inserted into an empty map is the
            // reverse of the insertion order, so y_rep is inserted first and
            // mu is the first element, as in 1.1.2, where mu was the only key.
            out.emplace("y_rep", std::move(y_rep));
            out.emplace("mu", std::move(mu));
            return out;
        }

        // ---- History mode: one row per stored draw ------------------------
        // B[j] (n_draws x n_new): beta_j of each stored draw at the points.
        std::vector<arma::mat> B(J);
        if (!has_new) {
            // Training data: the stored rows of beta_0..beta_p.
            const AI4BayesCode::history_map h = impl_->get_history();
            for (std::size_t j = 0; j < J; ++j) B[j] = h.at(beta_key(static_cast<int>(j)));
        } else {
            for (std::size_t j = 0; j < J; ++j) {
                auto& bj = dynamic_cast<bart_block&>(impl_->child(j));
                B[j] = bj.predict_history(Z_new);   // forest d at Z_new + fmean of sweep d
            }
        }
        const std::size_t n_draws = B[0].n_rows;
        for (std::size_t j = 0; j < J; ++j)
            if (B[j].n_rows != n_draws || B[j].n_cols != n_new)
                ai4b::stop("VCBart::predict_at: beta_%d has %d x %d stored values "
                           "and beta_0 has %d x %d",
                           (int)j, (int)B[j].n_rows, (int)B[j].n_cols,
                           (int)n_draws, (int)B[0].n_cols);
        // sigma_d: row d of the stored sigma, one per stored sweep. With no
        // sweep stored, B holds one row from the current forests, and the
        // y_rep of that row is computed with the current sigma.
        const bool none_stored = sigma_hist_.empty();
        if (none_stored ? n_draws != 1 : sigma_hist_.size() != n_draws) {
            // At new (X, Z), row d is computed from the forests stored at
            // sweep d. The forests are stored only with keep_tree = TRUE;
            // without them predict_history returns one row (the current
            // forests), which differs from the number of stored draws once
            // more than one draw is stored.
            if (has_new && !keep_tree_)
                ai4b::stop("VCBart::predict_at: prediction at new (X, Z) with "
                           "keep_history = TRUE and %d stored draws requires "
                           "keep_tree = TRUE at construction (the forests of "
                           "the stored draws are stored only then). Construct "
                           "with keep_tree = TRUE, or pass last_draw_only = "
                           "TRUE to predict from the current forests.",
                           (int)sigma_hist_.size());
            ai4b::stop("VCBart::predict_at: %d stored draws of the betas and "
                       "%d of sigma", (int)n_draws, (int)sigma_hist_.size());
        }
        // mu_{d,i} = sum_{j=0..p} x_ij beta_{j,d}(Z_i), summed over j in the
        // order of the "mu" refresher.
        arma::mat mu(n_draws, n_new, arma::fill::zeros);
        for (std::size_t j = 0; j < J; ++j)
            for (std::size_t i = 0; i < n_new; ++i) {
                const double xij = x_of(j, i);
                for (std::size_t d = 0; d < n_draws; ++d) mu(d, i) += xij * B[j](d, i);
            }
        // y_rep_{d,i} = mu_{d,i} + sigma_d * eps_{d,i}.
        const double sigma_now = impl_->data().get("sigma")[0];
        arma::mat y_rep(n_draws, n_new);
        for (std::size_t d = 0; d < n_draws; ++d) {
            const double sigma_d = none_stored ? sigma_now : sigma_hist_[d];
            for (std::size_t i = 0; i < n_new; ++i)
                y_rep(d, i) = mu(d, i) + sigma_d * norm01(predict_rng_);
        }
        // y_rep first, so mu is the first element of the R list and the
        // Python dict (see the single-draw branch).
        out.emplace("y_rep", std::move(y_rep));
        out.emplace("mu", std::move(mu));
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

    // Called by clear_history() after impl_ is cleared: empties sigma_hist_.
    void clear_extra_history_() { sigma_hist_.clear(); }

    // shared_data key helpers (also used by the refresher closures).
    static std::string beta_key(int j) { return "beta_" + std::to_string(j); }
    static std::string wr_key(int j)   { return "wr_"   + std::to_string(j); }
    static std::string w_key(int j)    { return "w_"    + std::to_string(j); }

private:
    std::mt19937_64 rng_;
    // eps of y_rep. mutable: predict_at() const advances it. Seeded at
    // construction and used by nothing else, so the MCMC draws do not depend
    // on it.
    mutable std::mt19937_64 predict_rng_;
    std::unique_ptr<composite_block> impl_;
    int  p_;
    arma::mat Z_;
    int  x_ncol_;
    int  z_ncol_;
    bool keep_history_;
    bool keep_tree_;
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
            "Short constructor; keep_history and keep_tree default FALSE.")
        .constructor<arma::mat, arma::mat, arma::vec,
                     int, double, double, double, double, int, int, bool>(
            "Construct with: X (N x p predictors), Z (N x q effect modifiers), "
            "y (length N), ntrees, k (leaf prior scale, default 2), power "
            "(tree depth penalty, default 2), base (tree depth base, default "
            "0.95), nu (sigma prior df, default 3), numcut (cutpoints per var, "
            "default 100), seed (RNG seed, 0 = random), keep_history (record "
            "per-step buffers; cheap; default FALSE); keep_tree defaults FALSE.")
        .constructor<arma::mat, arma::mat, arma::vec,
                     int, double, double, double, double, int, int, bool, bool>(
            "As the 11-argument constructor, plus keep_tree (store the forest "
            "of every sweep, required by predict_at at new (X, Z) with "
            "keep_history = TRUE and more than one stored draw; EXPENSIVE; "
            "default FALSE).")
        .method("step", (void (VCBart::*)())    &VCBart::step, "Run one Gibbs backfitting sweep.")
        .method("step", (void (VCBart::*)(int)) &VCBart::step, "Run n Gibbs backfitting sweeps.")
        .method("get_current",  &VCBart::get_current,
                "Named list: $beta_0..$beta_p (coefficient functions at the "
                "training Z), $mu (full mean), $sigma.")
        .method("set_current",  &VCBart::set_current,
                "Overwrite sigma and/or y (named list). y refreshes every "
                "working response + the full mean.")
        .method("predict_at",
                (AI4BayesCode::history_map (VCBart::*)(const AI4BayesCode::state_map&) const)
                    &VCBart::predict_at,
                "Predict mu and y_rep (mu plus sigma times N(0, 1) noise). "
                "Pass list(Z = Z_new, X = X_new) (matrices, or their "
                "column-major vectors), or list() for the training data. "
                "keep_history = TRUE: one row per stored draw, y_rep with "
                "the sigma of that draw (new (X, Z) with more than one stored "
                "draw requires keep_tree = TRUE); else one row from the "
                "current forests and sigma.")
        .method("predict_at",
                (AI4BayesCode::history_map (VCBart::*)(const AI4BayesCode::state_map&, bool) const)
                    &VCBart::predict_at,
                "predict_at(new_data, last_draw_only): TRUE takes the "
                "single-draw path without discarding the history.")
        .method("get_dag",      &VCBart::get_dag,     "Predict DAG as edge list.")
        .method("get_history",  &VCBart::get_history, "History as named matrices.")
        AI4BAYESCODE_BIND_KERNEL_CONTROL(VCBart);
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
                            int, double, double, double, double, int, int, bool, bool>(),
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
             pybind11::arg("keep_history") = false,
             pybind11::arg("keep_tree")    = false)
        .def("step", (void (VCBart::*)())    &VCBart::step, "Run one Gibbs backfitting sweep.")
        .def("step", (void (VCBart::*)(int)) &VCBart::step, pybind11::arg("n_steps"))
        .def("get_current",  &VCBart::get_current)
        .def("set_current",  &VCBart::set_current, pybind11::arg("params"))
        .def("predict_at",
             (AI4BayesCode::history_map (VCBart::*)(const AI4BayesCode::state_map&, bool) const)
                 &VCBart::predict_at,
             pybind11::arg("new_data"),
             pybind11::arg("last_draw_only") = false,
             "Predict mu and y_rep (mu plus sigma times N(0, 1) noise). "
             "new_data: {'Z': Z_new, 'X': X_new} (each flattened "
             "column-major), or {} for the training data. keep_history=True: "
             "one row per stored draw, y_rep with the sigma of that draw "
             "(new (X, Z) with more than one stored draw requires "
             "keep_tree=True); last_draw_only=True or "
             "keep_history=False: one row from the current forests and sigma.")
        .def("get_dag",      &VCBart::get_dag)
        .def("get_history",  &VCBart::get_history)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(VCBart);
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
        b0t[i] = 2.0 * std::max(z, 0.0);
        b1t[i] = z * z;
        b2t[i] = -z;
        y[i]   = b0t[i] + b1t[i] * X(i, 0) + b2t[i] * X(i, 1) + eps(g);
    }

    // keep_history so the recovery check uses a POSTERIOR MEAN over kept draws
    // rather than whatever a single final draw happened to be.
    VCBart m(X, Z, y, 50, 2.0, 2.0, 0.95, 3.0, 100, 1, true);
    m.step(500);                                  // burn
    const int M = 1000;
    arma::vec b0(b0t.n_elem, arma::fill::zeros);
    arma::vec b1(b1t.n_elem, arma::fill::zeros);
    arma::vec b2(b2t.n_elem, arma::fill::zeros);
    double sig = 0.0;
    for (int d = 0; d < M; ++d) {
        m.step();
        auto cur = m.get_current();
        b0 += cur["beta_0"]; b1 += cur["beta_1"]; b2 += cur["beta_2"];
        sig += cur["sigma"][0];
    }
    b0 /= M; b1 /= M; b2 /= M; sig /= M;

    auto corr = [](const arma::vec& a, const arma::vec& b) {
        return arma::as_scalar(arma::cor(a, b));
    };
    const double c0 = corr(b0, b0t), c1 = corr(b1, b1t), c2 = corr(b2, b2t);
    std::printf("VC-BART recovery (correlation of posterior-mean beta_j(Z) with truth):\n");
    std::printf("  beta_0 (2 max(Z,0)): %.3f\n", c0);
    std::printf("  beta_1 (Z^2)   : %.3f\n", c1);
    std::printf("  beta_2 (-Z)    : %.3f\n", c2);
    std::printf("  sigma (true 0.3): %.3f\n", sig);

    // Gate, so a regression is visible. The thresholds are deliberately loose:
    // varying-coefficient BART recovers the SHAPE of each beta_j(Z) well before
    // it recovers the scale, and sigma absorbs what the trees miss.
    const bool ok = std::isfinite(sig) && sig > 0.0 && sig < 1.0 &&
                    c0 > 0.7 && c1 > 0.5 && c2 > 0.7;
    if (!ok) {
        std::printf("FAIL: recovery below tolerance\n");
        return 1;
    }
    std::printf("OK: recovery within tolerance\n");
    return 0;
}
#endif

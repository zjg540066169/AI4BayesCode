// ============================================================================
//  GBartPoisson.cpp
//
//  Copyright (C) 2026 AI4BayesCode.
//  Licensed under GPL-3.0-or-later (uses genbart_block.hpp, which is GPL-2.0-or-later)
//
//  REFERENCE TEMPLATE for the AI4BayesCode skill -- genbart_block variant
//  with the shipped `genbart::lik::poisson_lik`. Generated samplers that
//  fit a Poisson regression with a BART-style mean function should follow
//  the STRUCTURE of this file exactly. Only the likelihood-selection line
//  and the summary-statistic choices in the y_rep refresher change when
//  swapping to a different response family (Negative binomial -> swap
//  poisson_lik for negative_binomial_lik; heteroscedastic -> swap for
//  heteroscedastic_lik + refresher; etc.).
//
//  Model
//  -----
//      y_i | r, offset_i  ~ Poisson(exp(r(x_i) + offset_i)),   i = 1..N
//      r                   ~ generalized-BART prior (Linero 2022)
//
//      i.e. log rate_i = r(x_i) + offset_i; r is the tree-ensemble output
//      on the LOG-RATE scale.  For pure Poisson with no exposure, the
//      offset is zero.  For rate regression with exposure e_i, pass
//      offset_i = log(e_i).
//
//  Block decomposition
//  -------------------
//      r  : sampled by genbart_block (one RJMCMC tree-ensemble sweep per
//           composite step).  Nuisance parameters (none for Poisson) are
//           owned by the likelihood subclass.
//
//  BACKEND-NEUTRAL DUAL MODULE
//  ---------------------------
//  This file compiles under three backends from a single source:
//    * AI4BAYESCODE_RCPP_MODULE   -> RcppArmadillo, RCPP_MODULE (R)
//    * AI4BAYESCODE_PYBIND_MODULE -> pure armadillo, PYBIND11_MODULE (Python)
//    * neither defined            -> standalone int main() demo
//  All shared code names only backend-neutral types (arma::vec/mat,
//  AI4BayesCode::state_map / history_map) and routes errors through
//  ai4b::stop.
//
//  RNG NOTE
//  --------
//  The generalized-BART kernel draws from its OWN seedable RNG stream
//  (bart_pure_cpp/src/r_compat.h, `bart_rng::engine()`, a thread-local
//  std::mt19937_64) under the pybind / standalone backend; the
//  std::mt19937_64& argument passed to step() is ignored. The
//  construction-time rng_seed therefore seeds the kernel stream
//  (bart_rng::set_seed) under that backend, so two GBartPoisson objects
//  built with DIFFERENT seeds produce independent chains. Under the Rcpp
//  (R) backend the kernel uses R's global RNG instead, so for reproducible
//  R runs call set.seed() before step(); the seed argument there only
//  drives the mutable predict_at RNG stream.
//
//  LICENSE WARNING
//  ---------------
//  This file #includes "AI4BayesCode/genbart_block.hpp", which pulls in
//  the vendored genBART kernel under bart_pure_cpp/src/GENBART/ (GPL-2.0+
//  upstream BART lineage). Any binary or source distribution of this
//  translation unit is therefore subject to GPL-2.0-or-later.
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("GBartPoisson")
//   set.seed(1)                                          # reproducible R run
//   N <- 600L; p <- 3L                                   # well-identified DGP
//   X <- matrix(runif(N * p, -1.5, 1.5), N, p)           # covariates ~ U(-1.5,1.5)
//   r <- 0.8 * sin(2 * X[,1]) + 0.6 * X[,2] - 0.4 * X[,3]# true log-rate surface
//   y <- rpois(N, exp(r))                                # y ~ Poisson(exp(r))
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(GBartPoisson, X, y, 50L, seed, FALSE, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(GBartPoisson, X, y, 50L, 42L, TRUE)         # X, y, ntrees, seed, keep_tree
//   m$step(2000); str(m$get_current())                  # $r (log rate), $rate=exp(r)
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(1)                       # well-identified DGP
//   N, p = 600, 3
//   X = rng.uniform(-1.5, 1.5, size=(N, p))              # covariates ~ U(-1.5,1.5)
//   r = 0.8 * np.sin(2 * X[:,0]) + 0.6 * X[:,1] - 0.4 * X[:,2]  # true log-rate
//   y = rng.poisson(np.exp(r)).astype(float)             # y ~ Poisson(exp(r))
//   Mod = AI4BayesCode.example("GBartPoisson")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.GBartPoisson(X, y, 50, seed, False, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.GBartPoisson(X, y, 50, 42, True); m.step(2000); print(m.get_current())
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

// AI4BayesCode core (pure C++)
#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"

// genBART block (Rcpp-free, transitively pulls in the GPL genBART kernel)
#include "AI4BayesCode/genbart_block.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::genbart_block;
using AI4BayesCode::genbart_block_config;

// ============================================================================
//  User-facing class. Backend-neutral signatures: arma containers,
//  state_map / history_map, ai4b::stop.
// ============================================================================

class GBartPoisson : public AI4BayesCode::kernel_control_mixin<GBartPoisson> {
public:
    GBartPoisson(const arma::mat& X,
                 const arma::vec& y,
                 int  ntrees,
                 int  rng_seed,
                 bool keep_tree    = false,
                 bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                       ? std::mt19937_64{std::random_device{}()}
                       : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                         ^ 0x9E3779B97F4A7C15ULL}),
          impl_(std::make_unique<composite_block>("GBartPoisson")),
          keep_tree_(keep_tree),
          keep_history_(keep_history)
    {
        if (X.n_rows != y.n_elem) {
            ai4b::stop("GBartPoisson: X and y must have matching row counts");
        }
        // y must be non-negative integer counts (we accept doubles but
        // don't strictly enforce integrality -- any non-negative double
        // is mathematically valid in the Poisson likelihood).
        for (std::size_t i = 0; i < y.n_elem; ++i) {
            if (!(y[i] >= 0.0) || !std::isfinite(y[i])) {
                ai4b::stop("GBartPoisson: y must be non-negative and finite");
            }
        }

        // ---- Seed the kernel RNG so a construction seed yields an
        //      independent chain (pybind / standalone backend only; the
        //      Rcpp backend uses R's global RNG instead). ----------------
#if !defined(AI4BAYESCODE_RCPP_MODULE)
        if (rng_seed != 0) {
            bart_rng::set_seed(static_cast<std::uint64_t>(rng_seed));
        }
#endif

        const std::size_t N = static_cast<std::size_t>(X.n_rows);

        // ---- Install fixed data into shared_data -----------------------
        impl_->data().set("y", y);

        // r starts at zero everywhere (initial log rate = offset = 0
        // => initial rate = 1; neutral starting point that genBART
        // itself uses when trees are stumps).
        impl_->data().set("r", arma::vec(N, arma::fill::zeros));

        // Store X as a flattened (column-major) vector in shared_data so
        // the predict_at DAG knows about it (the actual matrix lives in
        // genbart_block_config / the kernel).
        impl_->data().set("X", arma::vec(arma::vectorise(X)));
        impl_->data().declare_data_input("X");
        x_ncol_ = static_cast<std::size_t>(X.n_cols);

        // ---- Full predict-DAG reconstruction (no inlined exp link). ---
        //   rate  = exp(r)           (deterministic intermediate; the
        //                             Poisson mean, natural to report)
        //   y_rep ~ Poisson(rate)    (stochastic; reads ONLY rate)
        // rate kept current during sampling via
        // declare_invalidates("r", {"rate"}) so stateful
        // predict_at(empty) reads a fresh rate.
        impl_->data().set("rate",  arma::vec(N, arma::fill::ones));
        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));
        impl_->data().register_refresher(
            "rate",
            [](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& r = d.get("r");
                arma::vec rate(r.n_elem);
                for (std::size_t i = 0; i < r.n_elem; ++i)
                    rate[i] = std::exp(r[i]);
                return rate;
            });
        impl_->data().declare_invalidates("r", {"rate"});

        // ---- Declare predict DAG (generative direction) ----------------
        //   X -> r -> rate -> y_rep
        impl_->data().declare_predict_edges("X",    {"r"});
        impl_->data().declare_predict_edges("r",    {"rate"});
        impl_->data().declare_predict_edges("rate", {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). r ~ generalized-BART prior. ----
        impl_->data().declare_context_edges("genBART", {"r"});

        // Stochastic refresher: y_rep ~ Poisson(rate), length matches rate.
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                const arma::vec& rate = d.get("rate");
                arma::vec y_rep(rate.n_elem);
                for (std::size_t i = 0; i < rate.n_elem; ++i) {
                    std::poisson_distribution<int> pois(rate[i]);
                    y_rep[i] = static_cast<double>(pois(rng));
                }
                return y_rep;
            });

        // ---- Build the Poisson likelihood ------------------------------
        likelihood_ = std::make_unique<genbart::lik::poisson_lik>();

        // ---- Build and add the genbart_block ---------------------------
        genbart_block_config cfg;
        cfg.name         = "r";
        cfg.x_train      = X;
        cfg.y_init       = y;
        cfg.offset_init  = arma::vec();               // no offset (zeros)
        cfg.lik          = likelihood_.get();
        cfg.ntrees       = static_cast<std::size_t>(ntrees);
        cfg.y_key        = "";                         // y is fixed training data
        cfg.offset_key   = "";
        // Hypers default to Linero 2022 §3.3 (half-Cauchy c = 1/sqrt(T),
        // adaptive sigma_mu, no DART).

        impl_->add_child(std::make_unique<genbart_block>(std::move(cfg)));

        // ---- Enable history recording if requested ---------------------
        if (keep_tree_)    impl_->set_keep_tree(true);
        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Return the current draw. "r" is the ensemble output (log rate,
    // since the Poisson link is log); "rate" = exp(r) is the natural-
    // scale rate. The serialized BART forest is available separately via
    // get_tree() (a std::string, outside the numeric state_map).
    AI4BayesCode::state_map get_current() const {
        const arma::vec& r = impl_->data().get("r");
        arma::vec rate(r.n_elem);
        for (std::size_t i = 0; i < r.n_elem; ++i) rate[i] = std::exp(r[i]);
        AI4BayesCode::state_map out;
        out["r"]    = r;
        out["rate"] = std::move(rate);
        return out;
    }

    // Serialized BART forest snapshot (round-trips through set_tree(...)).
    // Kept as a separate passthrough so the forest state stays reachable
    // from both backends without putting a string into the numeric
    // state_map.
    std::string get_tree() const {
        auto& blk = dynamic_cast<genbart_block&>(impl_->child(0));
        return blk.get_tree();
    }

    // Update data inputs from an outer sampler. Takes a state_map.
    // Supported keys (any subset, order independent):
    //   y : length-n response.
    //   X : FLATTENED column-major n x p matrix (length n*p).
    //   r : SILENTLY IGNORED -- tree forest has no unique inverse. Use
    //       the tree round-trip (get_tree / set_tree) instead.
    // Unknown keys are silently ignored per system_design.md §7 so that
    // set_current(get_current()) round-trips cleanly.
    void set_current(const AI4BayesCode::state_map& params) {
        auto* blk = dynamic_cast<genbart_block*>(&impl_->child(0));

        // r is read-only output; silently ignored on input so that
        // round-trip set_current(get_current()) is supported per
        // system_design.md §7 / §16. Use set_tree() to restore the forest.

        const auto it_X = params.find("X");
        const auto it_y = params.find("y");
        const bool has_X = it_X != params.end();
        const bool has_y = it_y != params.end();

        // Reshape flattened X (column-major) back into an n x p matrix.
        auto reshape_X = [&](const arma::vec& x_flat) -> arma::mat {
            const std::size_t p = x_ncol_;
            if (p == 0 || (x_flat.n_elem % p) != 0) {
                ai4b::stop("GBartPoisson::set_current: flattened X length %d "
                           "is not a multiple of p = %d",
                           (int)x_flat.n_elem, (int)p);
            }
            return arma::reshape(x_flat, x_flat.n_elem / p, p);
        };

        if (has_X && has_y) {
            arma::mat X_new = reshape_X(it_X->second);
            const arma::vec& y_new = it_y->second;
            blk->set_data(X_new, y_new, arma::vec());
            impl_->data().set("X", arma::vec(arma::vectorise(X_new)));
            impl_->data().set("y", y_new);
        } else if (has_X) {
            arma::mat X_new = reshape_X(it_X->second);
            blk->set_X(X_new);
            impl_->data().set("X", arma::vec(arma::vectorise(X_new)));
        } else if (has_y) {
            const arma::vec& y_new = it_y->second;
            blk->set_Y(y_new);
            impl_->data().set("y", y_new);
        }
    }

    // Restore the full BART forest from a serialized snapshot previously
    // obtained from get_tree(). Separate from set_current because a tree
    // is a std::string, not a numeric state vector. Refreshes shared_data
    // r so downstream blocks see the restored forest immediately.
    void set_tree(const std::string& tree_s) {
        auto* blk = dynamic_cast<genbart_block*>(&impl_->child(0));
        blk->set_tree(tree_s);
        impl_->data().set("r", blk->current());
    }

    // Full model DAG (gibbs_reads + gibbs_invalidates + predict_edges +
    // data_inputs + context_edges). See composite_block::get_dag().
    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    // Predict at new data. Takes a state_map with optional key "X"
    // (FLATTENED column-major N_new x p matrix); empty map => posterior-
    // predictive y_rep at training X.
    //
    // Behavior depends on construction-time keep_history flag and whether
    // X was supplied:
    //   * keep_history = FALSE, empty map:
    //       r/rate/y_rep at TRAINING X from the current draw.
    //   * keep_history = FALSE, X supplied:
    //       r/rate at X_new from the current trees.
    //   * keep_history = TRUE,  empty map:
    //       r/rate/y_rep: n_draws x N_train, one row per stored draw.
    //   * keep_history = TRUE,  X supplied: not implemented (errors).
    //
    // Uses predict_rng_ (const-preserving, persistent, seeded at
    // construction) so posterior-predictive draws are reproducible given
    // the same seed. Does NOT modify MCMC state in any mode.
    AI4BayesCode::history_map
    predict_at(const AI4BayesCode::state_map& new_data) const {
        const bool has_X = new_data.find("X") != new_data.end();
        for (const auto& kv : new_data) {
            if (kv.first != "X") {
                ai4b::stop("GBartPoisson::predict_at: unknown key '%s'. "
                           "Valid keys: 'X' (flattened column-major "
                           "N_new x p matrix) or empty map for posterior "
                           "predictive y_rep at training X.",
                           kv.first.c_str());
            }
        }

        // ---- Empty map: posterior-predictive at training X -------------
        if (!has_X) {
            if (keep_history_) {
                // History mode: per-draw r/rate/y_rep from r_hist using
                // the Poisson likelihood (documented shadow loop -- genBART
                // history does not flow through composite predict_at).
                AI4BayesCode::history_map hist = impl_->get_history();
                const arma::mat& r_hist = hist.at("r");
                const std::size_t n_draws = r_hist.n_rows;
                const std::size_t N_local = r_hist.n_cols;
                arma::mat r_mat(n_draws, N_local);
                arma::mat rate_mat(n_draws, N_local);
                arma::mat yrep_mat(n_draws, N_local);
                for (std::size_t d = 0; d < n_draws; ++d) {
                    for (std::size_t i = 0; i < N_local; ++i) {
                        const double r_di = r_hist(d, i);
                        const double lam  = std::exp(r_di);
                        r_mat(d, i)    = r_di;
                        rate_mat(d, i) = lam;
                        std::poisson_distribution<int> pois(lam);
                        yrep_mat(d, i) = static_cast<double>(pois(predict_rng_));
                    }
                }
                AI4BayesCode::history_map out;
                out.emplace("r",     std::move(r_mat));
                out.emplace("rate",  std::move(rate_mat));
                out.emplace("y_rep", std::move(yrep_mat));
                return out;
            }

            // Stateful mode: single-draw predictive at training X.
            const block_context empty;
            block_context result = impl_->predict_at(empty, predict_rng_);
            AI4BayesCode::history_map out;
            for (const auto& kv : result) {
                arma::mat m(1, kv.second.n_elem);
                for (std::size_t i = 0; i < kv.second.n_elem; ++i)
                    m(0, i) = kv.second[i];
                out.emplace(kv.first, std::move(m));
            }
            return out;
        }

        // ---- Keyed branch: X-override (test-set prediction) ------------
        if (keep_history_) {
            ai4b::stop("GBartPoisson::predict_at: new-X prediction in "
                       "history mode is not yet implemented. Use empty map "
                       "or construct with keep_history=FALSE.");
        }

        const arma::vec& x_flat = new_data.at("X");
        if (x_ncol_ == 0 || (x_flat.n_elem % x_ncol_) != 0) {
            ai4b::stop("GBartPoisson::predict_at: flattened X length %d "
                       "is not a multiple of p = %d",
                       (int)x_flat.n_elem, (int)x_ncol_);
        }
        const std::size_t n_new = x_flat.n_elem / x_ncol_;
        arma::mat X_new = arma::reshape(x_flat, n_new, x_ncol_);

        auto& blk = dynamic_cast<genbart_block&>(impl_->child(0));
        arma::vec r = blk.predict_r(X_new);

        arma::mat r_mat(1, r.n_elem);
        arma::mat rate_mat(1, r.n_elem);
        for (std::size_t i = 0; i < r.n_elem; ++i) {
            r_mat(0, i)    = r[i];
            rate_mat(0, i) = std::exp(r[i]);
        }
        AI4BayesCode::history_map out;
        out.emplace("r",    std::move(r_mat));
        out.emplace("rate", std::move(rate_mat));
        return out;
    }

    // ---- History access --------------------------------------------------
    // The composite's neutral history_map (numeric matrices keyed by block
    // name: r [n_draws x N]). Per-draw serialized BART forests are reachable
    // separately via get_tree_history() (a vector<string>, outside the map).
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    // Per-draw serialized BART forest snapshots. In stateful mode this is
    // length 1 (the current forest); in history mode (keep_tree=true) it
    // has one entry per stored draw.
    std::vector<std::string> get_tree_history() const {
        auto& blk = dynamic_cast<genbart_block&>(impl_->child(0));
        return blk.get_tree_history();
    }

private:
    std::mt19937_64                      rng_;
    // predict_rng_ is mutable so `predict_at() const` can advance it when
    // sampling y_rep. Seeded once at construction (derived from rng_seed
    // XOR'd with the golden-ratio constant) so posterior predictive draws
    // are reproducible given a stable construction seed without stealing
    // entropy from the main MCMC RNG.
    mutable std::mt19937_64              predict_rng_;
    std::unique_ptr<composite_block>     impl_;
    std::unique_ptr<genbart::likelihood> likelihood_;
    std::size_t                          x_ncol_       = 0;
    bool                                 keep_tree_    = false;
    bool                                 keep_history_ = false;
};

// ============================================================================
//  Rcpp module: make the class visible to R.
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(GBartPoisson_module) {
    Rcpp::class_<GBartPoisson>("GBartPoisson")
        // Short constructor: defaults keep_tree=FALSE, keep_history=FALSE.
        .constructor<arma::mat, arma::vec, int, int>(
            "Short ctor: X (N x p), y (non-negative counts), ntrees, seed. "
            "keep_tree and keep_history default FALSE.")
        // Constructor with keep_tree only (keep_history defaults FALSE).
        .constructor<arma::mat, arma::vec, int, int, bool>(
            "Ctor with keep_tree only. X, y, ntrees, seed, keep_tree "
            "(forest snapshots per step; EXPENSIVE, required only for "
            "predict_history); keep_history defaults FALSE.")
        // Full constructor: keep_tree + keep_history.
        .constructor<arma::mat, arma::vec, int, int, bool, bool>(
            "Full ctor: X (N x p), y (non-negative counts), ntrees, "
            "seed (note: under R the genBART kernel uses R's global RNG; "
            "seed here drives only the mutable predict_at RNG stream -- "
            "for reproducible MCMC, call set.seed() in R before step()), "
            "keep_tree (forest snapshots per step; EXPENSIVE; default "
            "FALSE), keep_history (numeric per-step buffers; cheap; "
            "default FALSE).")
        .method("step", (void (GBartPoisson::*)())    &GBartPoisson::step, "Run one sweep.")
        .method("step", (void (GBartPoisson::*)(int)) &GBartPoisson::step,
                "Run n generalized-BART RJMCMC sweeps.")
        .method("get_current", &GBartPoisson::get_current,
                "Return the current draw as a named list with $r (log "
                "rate) and $rate (= exp(r)). The serialized BART forest is "
                "available separately via get_tree().")
        .method("get_tree",    &GBartPoisson::get_tree,
                "Return the serialized BART forest as a string "
                "(round-trips through set_tree()).")
        .method("set_current", &GBartPoisson::set_current,
                "Overwrite X and/or y from a named list. Supported keys: "
                "y, X (flattened column-major n x p). r is read-only; use "
                "set_tree() to restore the forest.")
        .method("set_tree",    &GBartPoisson::set_tree,
                "Restore the BART forest from a serialized snapshot string "
                "previously obtained from get_tree().")
        .method("predict_at",  &GBartPoisson::predict_at,
                "Predict log rate and rate at training or new X. Pass "
                "list(X = as.vector(X_new)) (flattened column-major) for "
                "test-set prediction, or an empty list for posterior-"
                "predictive y_rep at training X. Const, no state mutation.")
        .method("get_dag",     &GBartPoisson::get_dag,
                "Return the predict DAG as a named list of edges.")
        .method("get_history", &GBartPoisson::get_history,
                "Return the history as a named list of matrices "
                "(r [n_draws x N]).")
        .method("get_tree_history", &GBartPoisson::get_tree_history,
                "Return per-draw serialized BART forests (one per stored "
                "draw when keep_tree=TRUE; else the current forest).")
        AI4BAYESCODE_BIND_KERNEL_CONTROL(GBartPoisson);
}
#endif

// ============================================================================
//  pybind11 module: make the class visible to Python.
// ============================================================================

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(GBartPoisson, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<GBartPoisson>(m, "GBartPoisson")
        .def(pybind11::init<arma::mat, arma::vec, int, int, bool, bool>(),
             pybind11::arg("X"),
             pybind11::arg("y"),
             pybind11::arg("ntrees")       = 50,
             pybind11::arg("rng_seed")     = 1,
             pybind11::arg("keep_tree")    = false,
             pybind11::arg("keep_history") = false)
        .def("step", (void (GBartPoisson::*)())    &GBartPoisson::step, "Run one sweep.")
        .def("step", (void (GBartPoisson::*)(int)) &GBartPoisson::step, pybind11::arg("n_steps"))
        .def("get_current",      &GBartPoisson::get_current)
        .def("get_tree",         &GBartPoisson::get_tree)
        .def("set_current",      &GBartPoisson::set_current, pybind11::arg("params"))
        .def("set_tree",         &GBartPoisson::set_tree, pybind11::arg("tree_s"))
        .def("predict_at",       &GBartPoisson::predict_at, pybind11::arg("new_data"))
        .def("get_dag",          &GBartPoisson::get_dag)
        .def("get_history",      &GBartPoisson::get_history)
        .def("get_tree_history", &GBartPoisson::get_tree_history)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(GBartPoisson);
}
#endif

// ============================================================================
//  Standalone demo: simulate Poisson counts with a smooth log-rate, fit,
//  print finite recovery of the rate surface.
// ============================================================================

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>

int main() {
    // ---- Simulate a well-identified Poisson DGP -------------------------
    //   smooth, low-dimensional log-rate:
    //       r(x) = 0.8 sin(2 x1) + 0.6 x2 - 0.4 x3
    //   y_i ~ Poisson(exp(r(x_i)))
    //   N >= 400 with well-spread covariates so the rate surface is
    //   identified (per-point f has intrinsic RJMCMC tree-multimodality;
    //   the IDENTIFIED quantity is the fitted rate, which converges).
    const std::size_t N = 600;
    const std::size_t p = 3;

    std::mt19937_64 gen(20260622ULL);
    std::uniform_real_distribution<double> unif(-1.5, 1.5);

    arma::mat X(N, p);
    arma::vec y(N);
    arma::vec rate_true(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double x1 = unif(gen);
        const double x2 = unif(gen);
        const double x3 = unif(gen);
        X(i, 0) = x1; X(i, 1) = x2; X(i, 2) = x3;
        const double rr   = 0.8 * std::sin(2.0 * x1) + 0.6 * x2 - 0.4 * x3;
        const double lam  = std::exp(rr);
        rate_true[i] = lam;
        std::poisson_distribution<int> pois(lam);
        y[i] = static_cast<double>(pois(gen));
    }

    // ---- Fit -------------------------------------------------------------
    GBartPoisson model(X, y,
                       /*ntrees=*/50, /*rng_seed=*/42,
                       /*keep_tree=*/false, /*keep_history=*/false);

    const int n_burn = 500;
    const int n_keep = 500;
    model.step(n_burn);

    // Accumulate posterior-mean rate over kept draws.
    arma::vec rate_sum(N, arma::fill::zeros);
    for (int it = 0; it < n_keep; ++it) {
        model.step(1);
        AI4BayesCode::state_map cur = model.get_current();
        rate_sum += cur.at("rate");
    }
    const arma::vec rate_hat = rate_sum / static_cast<double>(n_keep);

    // ---- Report recovery -------------------------------------------------
    // Compare fitted rate to the true rate surface (the identified
    // quantity). RMSE on the rate scale + correlation.
    const double rmse_rate =
        std::sqrt(arma::mean(arma::square(rate_hat - rate_true)));
    const double cor_rate  = arma::as_scalar(arma::cor(rate_hat, rate_true));
    bool finite_ok = std::isfinite(rmse_rate)
                     && std::isfinite(cor_rate)
                     && rate_hat.is_finite();

    std::printf("GBartPoisson standalone demo\n");
    std::printf("  N = %zu, p = %zu, ntrees = 50\n", N, p);
    std::printf("  mean(y)                   = %.4f\n", arma::mean(y));
    std::printf("  mean(rate_true)           = %.4f\n", arma::mean(rate_true));
    std::printf("  mean(rate_hat)            = %.4f\n", arma::mean(rate_hat));
    std::printf("  RMSE(rate_hat, rate_true) = %.4f\n", rmse_rate);
    std::printf("  cor(rate_hat, rate_true)  = %.4f\n", cor_rate);
    std::printf("  all finite                = %s\n",
                finite_ok ? "YES" : "NO");

    if (!finite_ok) {
        std::printf("FAIL: non-finite recovery\n");
        return 1;
    }
    std::printf("OK: finite recovery\n");
    return 0;
}
#endif

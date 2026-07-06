// ============================================================================
//  GBartLogistic.cpp
//
//  Copyright (C) 2026 AI4BayesCode.
//  Licensed under GPL-3.0-or-later (uses genbart_block.hpp, which is GPL-2.0-or-later)
//
//  REFERENCE TEMPLATE for Bayesian BART binary classification via
//  genBART's direct sigmoid likelihood. No data-augmentation latent
//  required: `genbart::lik::logistic_lik` implements
//  p(y | lambda) = sigmoid(lambda)^y * (1 - sigmoid(lambda))^(1-y)
//  directly and genBART's RJMCMC samples the trees under that non-
//  conjugate likelihood via Laplace proposals (Linero 2022 §4.1).
//
//  This is the SIMPLER path for C = 2 outcomes. For multinomial
//  (C >= 2 generic, or when you specifically want the C-1 coupled-
//  ensemble identified form), use GBartMultinomial.cpp instead.
//
//  Model
//  -----
//      y_i | r  ~ Bernoulli(sigmoid(r(x_i))),   y_i in {0, 1}
//      r        ~ generalized-BART prior (Linero 2022)
//
//  Block decomposition
//  -------------------
//      r : sampled by genbart_block (one RJMCMC sweep per step).
//
//  BACKEND-NEUTRAL DUAL MODULE
//  ---------------------------
//  This file compiles under three backends from a single source:
//    * AI4BAYESCODE_RCPP_MODULE   -> RcppArmadillo, RCPP_MODULE (R)
//    * AI4BAYESCODE_PYBIND_MODULE -> pure armadillo, PYBIND11_MODULE (Python)
//    * neither defined            -> standalone int main() demo
//  All shared code names only backend-neutral types (arma::vec/mat,
//  AI4BayesCode::state_map / history_map) and routes errors through
//  ai4b::stop. genbart_block is itself Rcpp-free, so no R:: or Rcpp::
//  symbol appears outside the guarded module blocks.
//
//  GENBART RNG NOTE
//  ----------------
//  The genBART kernel draws from its OWN thread-local seedable stream
//  (bart_rng in bart_pure_cpp/src/r_compat.h, via the `arn` adapter); the
//  std::mt19937_64 passed to step() is ignored by genbart_block, and the
//  constructor's rng_seed only seeds the (kernel-unused) wrapper RNG +
//  predict_rng_ y_rep refresher. To get INDEPENDENT chains, seed the
//  kernel stream BEFORE constructing the model: under pybind via the
//  module-level set_kernel_seed(seed) free function, under R via the usual
//  set.seed(). The per-point r(x_i) carries genBART's intrinsic RJMCMC
//  tree-multimodality; the IDENTIFIED quantity is the fitted probability
//  sigmoid(r) at a fixed grid, which is what converges.
//
//  LICENSE WARNING
//  ---------------
//  This file #includes "AI4BayesCode/genbart_block.hpp", which in turn
//  includes the vendored genBART tree kernel under bart_pure_cpp/src/,
//  derived from the CRAN BART R package (GPL-2.0-or-later). Any binary or
//  source distribution of this translation unit is therefore subject to
//  GPL-2.0-or-later. Do NOT slap a permissive license header on this file
//  or on any downstream sampler that #includes genbart_block.hpp.
// ============================================================================
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("GBartLogistic")
//   set.seed(42); N <- 800L; p <- 3L            # well-identified DGP (= int main)
//   X <- matrix(runif(N * p, -1.5, 1.5), N, p)  # X ~ Uniform(-1.5, 1.5)
//   eta <- 1.5 * X[, 1] - 1.0 * X[, 2] + 0.8 * X[, 3]   # smooth linear predictor
//   y <- as.numeric(runif(N) < 1 / (1 + exp(-eta)))     # y ~ Bernoulli(sigmoid(eta))
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(GBartLogistic, X, y, 50L, seed, FALSE, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(GBartLogistic, X, y, 50L, 42L)     # X, y, ntrees=50, seed=42 (single chain)
//   m$step(2000L); str(m$get_current())         # $r linear predictor, $p fitted prob
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(42); N, p = 800, 3          # well-identified DGP (= int main)
//   X = rng.uniform(-1.5, 1.5, size=(N, p))                 # X ~ Uniform(-1.5, 1.5)
//   eta = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * X[:, 2]     # smooth linear predictor
//   y = (rng.uniform(size=N) < 1 / (1 + np.exp(-eta))).astype(float)  # Bernoulli(sigmoid)
//   Mod = AI4BayesCode.example("GBartLogistic")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.GBartLogistic(X, y, 50, seed, False, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.GBartLogistic(X, y, 50, 42, False, False)       # X, y, ntrees=50, seed=42
//   m.step(2000); print(m.get_current())                   # 'r' linear predictor, 'p' fitted prob
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
#include "AI4BayesCode/rcpp_wrap.hpp"

// genBART block (Rcpp-free, transitively pulls in the GPL genBART kernel)
#include "AI4BayesCode/genbart_block.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::genbart_block;
using AI4BayesCode::genbart_block_config;

namespace {
inline double safe_sigmoid(double x) {
    if (x >  36.0) return 1.0 - 1e-16;
    if (x < -36.0) return 1e-16;
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}
} // namespace

// ============================================================================
//  User-facing class. Backend-neutral signatures: arma containers,
//  state_map / history_map, ai4b::stop.
// ============================================================================

class GBartLogistic {
public:
    GBartLogistic(const arma::mat& X,
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
          impl_(std::make_unique<composite_block>("GBartLogistic")),
          keep_tree_(keep_tree),
          keep_history_(keep_history)
    {
        if (X.n_rows != y.n_elem) {
            ai4b::stop("GBartLogistic: X and y must have matching row counts");
        }
        for (std::size_t i = 0; i < y.n_elem; ++i) {
            if (!(y[i] == 0.0 || y[i] == 1.0)) {
                ai4b::stop("GBartLogistic: y must be 0/1 binary");
            }
        }
        const std::size_t N = static_cast<std::size_t>(X.n_rows);

        impl_->data().set("y", y);
        impl_->data().set("r", arma::vec(N, arma::fill::zeros));

        // Store X as a flattened (column-major) vector in shared_data so the
        // predict_at DAG knows about it (the actual matrix lives in the
        // genbart_block_config).
        impl_->data().set("X", arma::vec(arma::vectorise(X)));
        impl_->data().declare_data_input("X");
        x_ncol_ = static_cast<std::size_t>(X.n_cols);

        // ---- Full predict-DAG reconstruction (no inlined sigmoid). ----
        //   prob = sigmoid(r)        (deterministic intermediate)
        //   y_rep ~ Bernoulli(prob)  (stochastic; reads ONLY prob)
        // prob kept current during sampling via
        // declare_invalidates("r", {"prob"}) so stateful
        // predict_at(empty) reads a fresh prob. Behaviour-preserving:
        // same sigmoid(r), same per-i Bernoulli draw order => bit-identical
        // y_rep under a fixed predict RNG. (genBART history mode stays a
        // documented shadow loop per Ruling A -- tree forests cannot be
        // injected through predict_at's replaced-key validation.)
        impl_->data().set("prob",  arma::vec(N, arma::fill::value(0.5)));
        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));
        impl_->data().register_refresher(
            "prob",
            [](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& r = d.get("r");
                arma::vec p(r.n_elem);
                for (std::size_t i = 0; i < r.n_elem; ++i)
                    p[i] = safe_sigmoid(r[i]);
                return p;
            });
        impl_->data().declare_invalidates("r", {"prob"});

        impl_->data().declare_predict_edges("X",    {"r"});
        impl_->data().declare_predict_edges("r",    {"prob"});
        impl_->data().declare_predict_edges("prob", {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). r ~ generalized-BART prior.
        impl_->data().declare_context_edges("genBART", {"r"});

        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                const arma::vec& prob = d.get("prob");
                arma::vec y_rep(prob.n_elem);
                for (std::size_t i = 0; i < prob.n_elem; ++i) {
                    std::bernoulli_distribution bern(prob[i]);
                    y_rep[i] = bern(rng) ? 1.0 : 0.0;
                }
                return y_rep;
            });

        // ---- Add the genBART block --------------------------------------
        likelihood_ = std::make_unique<genbart::lik::logistic_lik>();

        genbart_block_config cfg;
        cfg.name        = "r";
        cfg.x_train     = X;
        cfg.y_init      = y;
        cfg.offset_init = arma::vec();   // no offset
        cfg.lik         = likelihood_.get();
        cfg.ntrees      = static_cast<std::size_t>(ntrees);
        cfg.y_key       = "";            // y fixed (training data)
        cfg.offset_key  = "";

        impl_->add_child(std::make_unique<genbart_block>(std::move(cfg)));

        if (keep_tree_)    impl_->set_keep_tree(true);
        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Current draw: r (linear predictor) + prob (fitted probability). The
    // serialized genBART forest snapshot is reachable separately via
    // get_tree() (a std::string, not a numeric state vector).
    AI4BayesCode::state_map get_current() const {
        const arma::vec& r = impl_->data().get("r");
        arma::vec p(r.n_elem);
        for (std::size_t i = 0; i < r.n_elem; ++i) p[i] = safe_sigmoid(r[i]);
        AI4BayesCode::state_map out;
        out["r"] = r;
        out["p"] = std::move(p);
        return out;
    }

    // Serialized genBART forest snapshot (round-trips through set_tree(...)).
    // Kept as a separate passthrough so the forest state stays reachable from
    // both backends without putting a string into the numeric state_map.
    std::string get_tree() const {
        auto& blk = dynamic_cast<genbart_block&>(impl_->child(0));
        return blk.get_tree();
    }

    // Update data inputs from an outer sampler. Supported keys (any subset;
    // order independent):
    //   X : FLATTENED column-major N x p matrix (length N*p). Replaces the
    //       design matrix (genBART preserves N; col count must match).
    //   y : length-N 0/1 vector. New working response.
    //   r : SILENTLY IGNORED -- read-only output derived from the trees (no
    //       unique inverse). Use set_tree(...) to restore a forest.
    // Unknown keys are silently ignored per system_design.md §7 so that
    // set_current(get_current()) round-trips cleanly.
    void set_current(const AI4BayesCode::state_map& params) {
        auto* blk = dynamic_cast<genbart_block*>(&impl_->child(0));

        const auto it_X = params.find("X");
        const auto it_y = params.find("y");
        const bool has_X = it_X != params.end();
        const bool has_y = it_y != params.end();

        auto reshape_X = [&](const arma::vec& x_flat) -> arma::mat {
            const std::size_t p = x_ncol_;
            if (p == 0 || (x_flat.n_elem % p) != 0) {
                ai4b::stop("GBartLogistic::set_current: flattened X length %d "
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
        // r is read-only output; silently ignored on input so that round-trip
        // set_current(get_current()) is supported. Use set_tree() to restore
        // the forest.
    }

    // Restore the full genBART forest from a serialized snapshot previously
    // obtained from get_tree(). Separate from set_current because a tree is a
    // std::string, not a numeric state vector. Refreshes shared_data r so
    // downstream nodes see the restored forest immediately.
    void set_tree(const std::string& tree_s) {
        auto* blk = dynamic_cast<genbart_block*>(&impl_->child(0));
        blk->set_tree(tree_s);
        impl_->data().set("r", blk->current());
    }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    // Predict at new data. Takes a state_map with optional key "X".
    //
    //   * empty map, keep_history = FALSE:
    //       y_rep: posterior-predictive at TRAINING X from the current draw.
    //   * empty map, keep_history = TRUE:
    //       y_rep: n_draws x N_train posterior-predictive from each stored
    //              draw's r history (documented genBART shadow loop -- forests
    //              cannot be injected through predict_at's replaced-key
    //              validation, so this iterates the stored r directly).
    //   * X supplied, keep_history = FALSE:
    //       r, p:  linear predictor + fitted probability at X_new from the
    //              CURRENT forest.
    //   * X supplied, keep_history = TRUE: not implemented (see error).
    //
    // X is passed as a FLATTENED column-major vector under key "X"
    // (length N_new * p); reshaped here back into an N_new x p matrix.
    // Uses predict_rng_ (const-preserving, persistent, seeded at
    // construction). Does NOT modify MCMC state in any mode.
    AI4BayesCode::history_map
    predict_at(const AI4BayesCode::state_map& new_data) const {
        const bool has_X = new_data.find("X") != new_data.end();
        for (const auto& kv : new_data) {
            if (kv.first != "X") {
                ai4b::stop("GBartLogistic::predict_at: unknown key '%s'. "
                           "Valid keys: 'X' (flattened column-major "
                           "N_new x p matrix) or empty map for posterior "
                           "predictive at training X.",
                           kv.first.c_str());
            }
        }

        if (!has_X) {
            // ---- Posterior-predictive at training X ---------------------
            if (keep_history_) {
                // History mode: documented genBART shadow loop over the
                // recorded r draws (tree forests cannot be injected through
                // predict_at's replaced-key validation).
                AI4BayesCode::history_map hist = impl_->get_history();
                const arma::mat& r_hist = hist.at("r");
                const std::size_t n_draws = r_hist.n_rows;
                const std::size_t N_local = r_hist.n_cols;
                arma::mat yrep(n_draws, N_local);
                for (std::size_t d = 0; d < n_draws; ++d) {
                    for (std::size_t i = 0; i < N_local; ++i) {
                        const double p = safe_sigmoid(r_hist(d, i));
                        std::bernoulli_distribution bern(p);
                        yrep(d, i) = bern(predict_rng_) ? 1.0 : 0.0;
                    }
                }
                AI4BayesCode::history_map out;
                out.emplace("y_rep", std::move(yrep));
                return out;
            }

            // Stateful mode: walk the DAG once from the current draw.
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

        // ---- New-X prediction ------------------------------------------
        if (keep_history_) {
            ai4b::stop("GBartLogistic::predict_at: new-X prediction in "
                       "history mode is not yet implemented. Use an empty map "
                       "or construct with keep_history = FALSE.");
        }
        const arma::vec& x_flat = new_data.at("X");
        if (x_ncol_ == 0 || (x_flat.n_elem % x_ncol_) != 0) {
            ai4b::stop("GBartLogistic::predict_at: flattened X length %d "
                       "is not a multiple of p = %d",
                       (int)x_flat.n_elem, (int)x_ncol_);
        }
        const std::size_t n_new = x_flat.n_elem / x_ncol_;
        arma::mat X_new = arma::reshape(x_flat, n_new, x_ncol_);

        auto& blk = dynamic_cast<genbart_block&>(impl_->child(0));
        arma::vec r = blk.predict_r(X_new);

        arma::mat r_mat(1, r.n_elem), p_mat(1, r.n_elem);
        for (std::size_t i = 0; i < r.n_elem; ++i) {
            r_mat(0, i) = r[i];
            p_mat(0, i) = safe_sigmoid(r[i]);
        }
        AI4BayesCode::history_map out;
        out.emplace("r", std::move(r_mat));
        out.emplace("p", std::move(p_mat));
        return out;
    }

    // The composite's neutral history_map (numeric matrices keyed by block
    // name: r [n_draws x N]). Per-draw serialized genBART forests are
    // reachable separately via get_tree_history() (a vector<string>, outside
    // the numeric map).
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    // Per-draw serialized genBART forest snapshots. In stateful mode this is
    // length 1 (the current forest); in history mode (keep_tree = true) it has
    // one entry per stored draw.
    std::vector<std::string> get_tree_history() const {
        auto& blk = dynamic_cast<genbart_block&>(impl_->child(0));
        return blk.get_tree_history();
    }

private:
    std::mt19937_64                      rng_;
    // predict_rng_ is mutable so predict_at() const can advance it when
    // sampling y_rep. Seeded once at construction (rng_seed XOR golden ratio)
    // so posterior-predictive draws are reproducible without stealing entropy
    // from the main RNG. NOTE: the genBART kernel ignores both RNGs (it draws
    // from its own seedable arn stream); these only drive the y_rep Bernoulli
    // refresher and DAG bookkeeping.
    mutable std::mt19937_64              predict_rng_;
    std::unique_ptr<composite_block>     impl_;
    std::unique_ptr<genbart::likelihood> likelihood_;
    std::size_t                          x_ncol_ = 0;
    bool                                 keep_tree_    = false;
    bool                                 keep_history_ = false;
};

// ============================================================================
//  Rcpp module: make the class visible to R.
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(GBartLogistic_module) {
    Rcpp::class_<GBartLogistic>("GBartLogistic")
        .constructor<arma::mat, arma::vec, int, int>(
            "Short ctor: X, y, ntrees, seed; keep_tree and keep_history "
            "default FALSE.")
        .constructor<arma::mat, arma::vec, int, int, bool>(
            "Ctor with keep_tree only: X, y, ntrees, seed, keep_tree "
            "(EXPENSIVE; required for predict_history).")
        .constructor<arma::mat, arma::vec, int, int, bool, bool>(
            "Full ctor: X (N x p), y (0/1), ntrees, seed, keep_tree "
            "(forest snapshots per step; EXPENSIVE; default FALSE), "
            "keep_history (numeric per-step buffers; cheap; default FALSE).")
        .method("step", (void (GBartLogistic::*)())    &GBartLogistic::step, "Run one sweep.")
        .method("step", (void (GBartLogistic::*)(int)) &GBartLogistic::step,
                "Run n genBART RJMCMC sweeps.")
        .method("get_current", &GBartLogistic::get_current,
                "Return the current draw as a named list with $r (linear "
                "predictor) and $p (fitted probability). The serialized "
                "genBART forest is available separately via get_tree().")
        .method("get_tree",    &GBartLogistic::get_tree,
                "Return the serialized genBART forest as a length-1 string "
                "(round-trips through set_tree()).")
        .method("set_current", &GBartLogistic::set_current,
                "Overwrite working response (y) and/or X from a named list. "
                "Supported keys: y, X (flattened column-major N x p). r is "
                "read-only; use set_tree() to restore a forest.")
        .method("set_tree",    &GBartLogistic::set_tree,
                "Restore the genBART forest from a serialized snapshot string "
                "previously obtained from get_tree().")
        .method("predict_at",  &GBartLogistic::predict_at,
                "Predict at new data. Pass list(X = as.vector(X_new)) "
                "(flattened column-major) for r/p at X_new, or list() for "
                "posterior-predictive y_rep at training X. Const, no state "
                "mutation.")
        .method("get_dag",     &GBartLogistic::get_dag,
                "Return the predict DAG as a named list of edges.")
        .method("get_history", &GBartLogistic::get_history,
                "Return the history as a named list of matrices "
                "(r [n_draws x N]).")
        .method("get_tree_history", &GBartLogistic::get_tree_history,
                "Return per-draw serialized genBART forests (one per stored "
                "draw when keep_tree=TRUE; else the current forest).");
}
#endif

// ============================================================================
//  pybind11 module: make the class visible to Python.
// ============================================================================

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(GBartLogistic, m) {
    AI4BayesCode::register_ai4bayescode_types(m);

    // genBART draws from its OWN thread-local kernel RNG stream
    // (bart_rng in bart_pure_cpp/src/r_compat.h); the std::mt19937 passed
    // to step() is ignored, and the constructor's rng_seed only seeds the
    // (unused) wrapper RNG + predict_rng_. To reproduce a run or to get
    // INDEPENDENT chains, seed the kernel stream BEFORE constructing the
    // model. This free function exposes that seed hook (kernel plumbing,
    // not model substance) -- mirrors the documented
    // bart_rng::set_seed(seed) convention in genbart_block.hpp.
    m.def("set_kernel_seed",
          [](std::uint64_t s) { bart_rng::set_seed(s); },
          pybind11::arg("seed"),
          "Seed the genBART kernel RNG stream (call before constructing a "
          "model to get reproducible / independent chains).");

    pybind11::class_<GBartLogistic>(m, "GBartLogistic")
        .def(pybind11::init<arma::mat, arma::vec, int, int, bool, bool>(),
             pybind11::arg("X"),
             pybind11::arg("y"),
             pybind11::arg("ntrees")       = 50,
             pybind11::arg("rng_seed")     = 1,
             pybind11::arg("keep_tree")    = false,
             pybind11::arg("keep_history") = false)
        .def("step", (void (GBartLogistic::*)())    &GBartLogistic::step, "Run one sweep.")
        .def("step", (void (GBartLogistic::*)(int)) &GBartLogistic::step, pybind11::arg("n_steps"))
        .def("get_current",      &GBartLogistic::get_current)
        .def("get_tree",         &GBartLogistic::get_tree)
        .def("set_current",      &GBartLogistic::set_current, pybind11::arg("params"))
        .def("set_tree",         &GBartLogistic::set_tree, pybind11::arg("tree_s"))
        .def("predict_at",       &GBartLogistic::predict_at, pybind11::arg("new_data"))
        .def("get_dag",          &GBartLogistic::get_dag)
        .def("get_history",      &GBartLogistic::get_history)
        .def("get_tree_history", &GBartLogistic::get_tree_history);
}
#endif

// ============================================================================
//  Standalone demo: simulate a well-identified logistic DGP, fit, print
//  finite recovery of the fitted probability surface.
// ============================================================================

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>

int main() {
    // ---- Simulate a well-identified binary DGP --------------------------
    //   smooth, low-dimensional linear predictor:
    //     eta(x) = 1.5 x1 - 1.0 x2 + 0.8 x3
    //   y_i ~ Bernoulli(sigmoid(eta(x_i)))
    //   N large enough for the IDENTIFIED probability surface to converge
    //   (the per-point r carries genBART's intrinsic tree-multimodality).
    const std::size_t N = 800;
    const std::size_t p = 3;

    std::mt19937_64 gen(20260622ULL);
    std::uniform_real_distribution<double> unif(-1.5, 1.5);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    arma::mat X(N, p);
    arma::vec y(N);
    arma::vec eta_true(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double x1 = unif(gen);
        const double x2 = unif(gen);
        const double x3 = unif(gen);
        X(i, 0) = x1; X(i, 1) = x2; X(i, 2) = x3;
        const double eta = 1.5 * x1 - 1.0 * x2 + 0.8 * x3;
        eta_true[i] = eta;
        const double prob = safe_sigmoid(eta);
        y[i] = (u01(gen) < prob) ? 1.0 : 0.0;
    }
    arma::vec p_true(N);
    for (std::size_t i = 0; i < N; ++i) p_true[i] = safe_sigmoid(eta_true[i]);

    // ---- Fit -------------------------------------------------------------
    GBartLogistic model(X, y,
                        /*ntrees=*/50, /*rng_seed=*/42,
                        /*keep_tree=*/false, /*keep_history=*/false);

    const int n_burn = 500;
    const int n_keep = 500;
    model.step(n_burn);

    // Accumulate posterior-mean fitted probability over kept draws.
    arma::vec p_sum(N, arma::fill::zeros);
    for (int it = 0; it < n_keep; ++it) {
        model.step(1);
        AI4BayesCode::state_map cur = model.get_current();
        p_sum += cur.at("p");
    }
    const arma::vec p_hat = p_sum / static_cast<double>(n_keep);

    // ---- Report recovery -------------------------------------------------
    // RMSE / correlation on the IDENTIFIED probability surface, plus
    // classification accuracy of p_hat > 0.5 vs y.
    const double rmse_p = std::sqrt(arma::mean(arma::square(p_hat - p_true)));
    const double cor_p  = arma::as_scalar(arma::cor(p_hat, p_true));
    std::size_t correct = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const double yhat = (p_hat[i] > 0.5) ? 1.0 : 0.0;
        if (yhat == y[i]) ++correct;
    }
    const double accuracy = static_cast<double>(correct) / static_cast<double>(N);

    const bool finite_ok = p_hat.is_finite()
                           && std::isfinite(rmse_p)
                           && std::isfinite(cor_p);

    std::printf("GBartLogistic standalone demo\n");
    std::printf("  N = %zu, p = %zu, ntrees = 50\n", N, p);
    std::printf("  RMSE(p_hat, p_true)      = %.4f\n", rmse_p);
    std::printf("  cor(p_hat, p_true)       = %.4f\n", cor_p);
    std::printf("  classification accuracy  = %.4f\n", accuracy);
    std::printf("  all finite               = %s\n",
                finite_ok ? "YES" : "NO");

    if (!finite_ok) {
        std::printf("FAIL: non-finite recovery\n");
        return 1;
    }
    std::printf("OK: finite recovery\n");
    return 0;
}
#endif

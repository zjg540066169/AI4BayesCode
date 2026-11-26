// ============================================================================
//  GBartHeteroscedastic.cpp
//
//  Copyright (C) 2026 AI4BayesCode.
//  Licensed under GPL-3.0-or-later (uses genbart_block.hpp, which is GPL-2.0-or-later).
//
//  REFERENCE TEMPLATE for Linero 2022 Sec.4.2 heteroscedastic BART. This
//  is the marquee example where RJMCMC beats conjugate backfitting
//  (paper reports RMSE 3.25 vs 5.71 for rbart vs 6.67 for bartMachine).
//
//  Model
//  -----
//      y_i | r, phi ~ Normal(mean = exp(r(x_i)),
//                             var  = phi * exp(r(x_i)))
//      r            ~ generalized-BART prior
//      phi > 0      ~ Inv-Gamma prior (tau = 1/phi ~ Gamma; conjugate,
//                     updated by the likelihood each step via
//                     heteroscedastic_lik::update_nuisance)
//
//  Both the mean AND the variance are modelled jointly via r(x) -- the
//  variance scales with the mean through the (g = exp, V = identity)
//  choice of Linero 2022 eqs Sec.4.2. This is a simple mean-variance
//  relationship that does remarkably well on skewed / heavy-tail data
//  that breaks ordinary BART.
//
//  Block decomposition
//  -------------------
//      r  : sampled by genbart_block (one RJMCMC sweep per step; phi
//           updated inside via heteroscedastic_lik::update_nuisance).
//
//  BACKEND-NEUTRAL DUAL MODULE
//  ---------------------------
//  This file compiles under three backends from a single source:
//    * AI4BAYESCODE_RCPP_MODULE   -> RcppArmadillo, RCPP_MODULE (R)
//    * AI4BAYESCODE_PYBIND_MODULE -> pure armadillo, PYBIND11_MODULE (Python)
//    * neither defined            -> standalone int main() demo
//  All shared code names only backend-neutral types (arma::vec/mat,
//  AI4BayesCode::state_map / history_map) and routes errors through
//  ai4b::stop. The genBART kernel + heteroscedastic likelihood are pure
//  C++; no R:: math calls are needed here (the conjugate Gamma update
//  lives inside the likelihood and draws from the kernel RNG stream).
//
//  RNG NOTE
//  --------
//  The genBART kernel draws from its OWN seedable RNG stream under the
//  pybind / standalone backend (bart_pure_cpp/src/r_compat.h,
//  bart_rng::engine(), a thread-local std::mt19937_64); the
//  std::mt19937_64 passed to step() is ignored. A non-zero
//  construction-time rng_seed therefore seeds the kernel stream
//  (bart_rng::set_seed) under that backend, so two GBartHeteroscedastic
//  objects built with DIFFERENT seeds produce independent chains. Under
//  the Rcpp (R) backend the kernel uses R's global RNG instead; R runs call
//  set.seed() before construction.
//
//  LICENSE WARNING
//  ---------------
//  This file #includes "AI4BayesCode/genbart_block.hpp", which in turn
//  includes the vendored genBART tree kernel under bart_pure_cpp/src/,
//  derived from the CRAN BART R package (GPL-2.0-or-later). Any binary
//  or source distribution of this translation unit is therefore subject
//  to GPL-2.0-or-later. Do NOT slap a permissive license header on this
//  file or on any downstream sampler that #includes genbart_block.hpp.
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("GBartHeteroscedastic")
//   set.seed(42); N <- 600L                          # well-identified DGP
//   X  <- matrix(runif(N * 2L, -1, 1), N, 2L)        # x1,x2 ~ Unif(-1,1)
//   r  <- 1.0 + 0.8 * pmax(X[,1], 0) + 0.6 * X[,2]   # log-mean with a kink at x1 = 0
//   m  <- exp(r)                                      # mean in ~[1.5, 11.0]
//   y  <- m + sqrt(0.5 * m) * rnorm(N)               # y~N(m, phi*m), phi=0.5
//   # ---- Parallel chains + convergence diagnosis (default) ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(GBartHeteroscedastic, X, y, rng_seed = seed, keep_history = TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   print(ai4bayescode_rhat_summary(run))          # CROSS-chain R-hat / ESS
//   ai4bayescode_diagnose(run$histories[[1]])      # chain 1: summary + plots
//   # ---- Advanced: stateful single-chain control ----
//   mod <- new(GBartHeteroscedastic, X, y, rng_seed = 42L)   # ntrees 50, phi_init 1.0
//   #          X,  y, ntrees, phi_init, seed, keep_tree
//   mod$step(2000); str(mod$get_current())           # $r, $mean, $phi
//   # ---- Posterior at new X from every stored draw: keep_tree = TRUE ----
//   # (predict(run, list(X = Xn)) requires keep_tree = TRUE in the factory as well)
//   mh <- new(GBartHeteroscedastic, X, y, rng_seed = 42L, keep_tree = TRUE, keep_history = TRUE)
//   mh$step(500L); mh$clear_history(); mh$step(500L)  # 500 stored draws, forest and phi of each
//   Xn <- cbind(seq(-1, 1, by = 0.25), 0)             # 9 new points on the x1 axis
//   pn <- mh$predict_at(list(X = Xn)); dim(pn$mean)   # $r, $mean, $mean_r, $y_rep: 500 x 9 each
//   rbind(colMeans(pn$mean), exp(1 + 0.8 * pmax(Xn[, 1], 0)))  # posterior mean vs truth
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(42); N = 600          # well-identified DGP
//   X = rng.uniform(-1, 1, size=(N, 2))               # x1,x2 ~ Unif(-1,1)
//   r = 1.0 + 0.8 * np.maximum(X[:,0], 0) + 0.6 * X[:,1]  # log-mean with a kink at x1 = 0
//   m = np.exp(r)                                      # mean in ~[1.5, 11.0]
//   y = m + np.sqrt(0.5 * m) * rng.standard_normal(N) # y~N(m, phi*m), phi=0.5
//   Mod = AI4BayesCode.example("GBartHeteroscedastic")
//   # ---- Parallel chains + diagnosis (default) ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.GBartHeteroscedastic(X, y, rng_seed=seed, keep_history=True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   print(AI4BayesCode.rhat_summary(chains))   # CROSS-chain R-hat / ESS
//   AI4BayesCode.diagnose(chains[0]["hist"])   # chain 1: summary + plots
//   # ---- Advanced: stateful single-chain control ----
//   m_ = Mod.GBartHeteroscedastic(X, y, rng_seed=42)   # ntrees 50, phi_init 1.0
//   m_.step(2000); print(m_.get_current())            # r, mean, phi
//   # ---- Posterior at new X from every stored draw: keep_tree=True ----
//   # (AI4BayesCode.predict(chains, {"X": ...}) requires keep_tree=True in the factory as well)
//   mh = Mod.GBartHeteroscedastic(X, y, rng_seed=42, keep_tree=True, keep_history=True)
//   mh.step(500); mh.clear_history(); mh.step(500)     # 500 stored draws, forest and phi of each
//   Xn = np.column_stack([np.linspace(-1, 1, 9), np.zeros(9)])  # 9 new points on the x1 axis
//   pn = mh.predict_at({"X": Xn.flatten(order="F")})   # r, mean, mean_r, y_rep: (500, 9) each
//   print(pn["mean"].mean(axis=0), np.exp(1 + 0.8 * np.maximum(Xn[:, 0], 0)))  # posterior mean vs truth
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
#include "AI4BayesCode/rcpp_predict_guard.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"

// genBART block (Rcpp-free, transitively pulls in the GPL genBART kernel)
#include "AI4BayesCode/genbart_block.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

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

// ============================================================================
//  User-facing class. Backend-neutral signatures: arma containers,
//  state_map / history_map, ai4b::stop.
// ============================================================================

class GBartHeteroscedastic : public AI4BayesCode::kernel_control_mixin<GBartHeteroscedastic> {
    friend class AI4BayesCode::kernel_control_mixin<GBartHeteroscedastic>;
public:
    GBartHeteroscedastic(const arma::mat& X,
                         const arma::vec& y,
                         int    ntrees = 50,
                         double phi_init = 1.0,
                         int    rng_seed = 1,
                         bool   keep_tree    = false,
                         bool   keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          impl_(std::make_unique<composite_block>("GBartHeteroscedastic")),
          keep_tree_(keep_tree),
          keep_history_(keep_history)
    {
        if (X.n_rows != y.n_elem) {
            ai4b::stop("GBartHeteroscedastic: X and y must have matching "
                       "row counts");
        }
        for (std::size_t i = 0; i < y.n_elem; ++i) {
            if (!std::isfinite(y[i])) {
                ai4b::stop("GBartHeteroscedastic: y must be finite");
            }
        }
        // Note: the heteroscedastic Normal likelihood does NOT require
        // y > 0 (only the mean m = exp(r) > 0, which is automatic).
        // Negative Y is mathematically valid but typically indicates
        // model misspecification for the (g=exp, V=identity) link;
        // most real datasets in Linero 2022 Sec.4.2 are positive by
        // construction (rates, counts-as-continuous, etc.).
        if (!(phi_init > 0.0)) {
            ai4b::stop("GBartHeteroscedastic: phi_init must be positive");
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

        // ---- Install fixed data and initial parameter values ------------
        impl_->data().set("y", y);
        impl_->data().set("r", arma::vec(N, arma::fill::zeros));

        // Store X as a flattened (column-major) vector in shared_data so
        // the predict_at DAG knows about it (the actual matrix lives in
        // genbart_block_config).
        impl_->data().set("X", arma::vec(arma::vectorise(X)));
        impl_->data().declare_data_input("X");
        x_ncol_ = static_cast<std::size_t>(X.n_cols);

        likelihood_ = std::make_unique<genbart::lik::heteroscedastic_lik>(
            phi_init);
        auto* lik_ptr = static_cast<genbart::lik::heteroscedastic_lik*>(
            likelihood_.get());

        // ---- Full predict-DAG reconstruction --------------------------
        //   mean_r = exp(r)                    (conditional mean, det)
        //   phi    = lik_ptr->phi()            (dispersion, synced from
        //            the likelihood object into a first-class slot so
        //            ai4bayescode_plot_dag shows phi -> y_rep and predict_at can see
        //            it; updated by the conjugate Gamma inside the r
        //            block each step)
        //   y_rep ~ N(mean_r, phi * mean_r)    (stochastic; reads ONLY
        //            mean_r, phi)
        // mean_r/phi kept current via declare_invalidates("r", {...}).
        // Behaviour-preserving: same exp(r) and phi, same per-i normal
        // draw order => bit-identical y_rep under a fixed predict RNG.
        // genBART history mode stays a documented shadow loop (Ruling
        // A -- tree forests cannot pass predict_at replaced-validation).
        impl_->data().set("mean_r", arma::vec(N, arma::fill::ones));
        impl_->data().set("phi",    arma::vec{phi_init});
        impl_->data().set("y_rep",  arma::vec(N, arma::fill::zeros));
        impl_->data().register_refresher(
            "mean_r",
            [](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& r = d.get("r");
                arma::vec m(r.n_elem);
                for (std::size_t i = 0; i < r.n_elem; ++i)
                    m[i] = std::exp(r[i]);
                return m;
            });
        impl_->data().register_refresher(
            "phi",
            [lik_ptr](const AI4BayesCode::shared_data_t&) -> arma::vec {
                return arma::vec{lik_ptr->phi()};
            });
        impl_->data().declare_invalidates("r", {"mean_r", "phi"});

        // ---- Declare predict DAG (generative direction) -----------------
        //   X      -> r       (genBART tree traversal)
        //   r      -> mean_r  (exp link)
        //   mean_r -> y_rep   (observation layer)
        //   phi    -> y_rep   (dispersion enters the observation layer)
        impl_->data().declare_predict_edges("X",      {"r"});
        impl_->data().declare_predict_edges("r",      {"mean_r"});
        impl_->data().declare_predict_edges("mean_r", {"y_rep"});
        impl_->data().declare_predict_edges("phi",    {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). r ~ generalized-BART prior.
        impl_->data().declare_context_edges("genBART", {"r"});

        // Stochastic refresher: y_rep ~ N(mean_r, phi * mean_r).
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                const arma::vec& mean_r = d.get("mean_r");
                const double phi = d.get("phi")[0];
                std::normal_distribution<double> norm(0.0, 1.0);
                arma::vec y_rep(mean_r.n_elem);
                for (std::size_t i = 0; i < mean_r.n_elem; ++i) {
                    const double m  = mean_r[i];
                    const double sd = std::sqrt(phi * m);
                    y_rep[i] = m + sd * norm(rng);
                }
                return y_rep;
            });

        // ---- Add the genBART block --------------------------------------
        genbart_block_config cfg;
        cfg.name        = "r";
        cfg.x_train     = X;
        cfg.y_init      = y;
        cfg.offset_init = arma::vec();   // empty = no offset
        cfg.lik         = likelihood_.get();
        cfg.ntrees      = static_cast<std::size_t>(ntrees);
        cfg.y_key       = "";
        cfg.offset_key  = "";

        impl_->add_child(std::make_unique<genbart_block>(std::move(cfg)));

        // ---- Enable history recording if requested ----------------------
        if (keep_tree_)    impl_->set_keep_tree(true);
        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) {
            impl_->step(rng_);
            // phi is drawn inside the likelihood and has no block history, so
            // the phi of each sweep is stored here for the history-mode
            // posterior predictive.
            if (keep_history_)
                phi_hist_.push_back(static_cast<genbart::lik::heteroscedastic_lik*>(
                    likelihood_.get())->phi());
        }
    }

    // Current draw: r + mean (= exp(r)) + phi (dispersion). The serialized
    // genBART forest snapshot stays reachable via get_tree(); it is not a
    // numeric state vector so it lives outside the state_map.
    AI4BayesCode::state_map get_current() const {
        const arma::vec& r = impl_->data().get("r");
        arma::vec mean_out(r.n_elem);
        for (std::size_t i = 0; i < r.n_elem; ++i) {
            mean_out[i] = std::exp(r[i]);
        }
        auto* lik_ptr = static_cast<genbart::lik::heteroscedastic_lik*>(
            likelihood_.get());
        AI4BayesCode::state_map out;
        out["r"]    = r;
        out["mean"] = mean_out;
        out["phi"]  = arma::vec{lik_ptr->phi()};
        return out;
    }

    // Serialized genBART forest snapshot (round-trips through set_tree()).
    // Kept as a separate passthrough so the forest state stays reachable
    // from both backends without putting a string into the numeric
    // state_map.
    std::string get_tree() const {
        auto& blk = dynamic_cast<genbart_block&>(impl_->child(0));
        return blk.get_tree();
    }

    // Update parameters / data inputs from an outer sampler.
    // Supported keys (any subset; order independent):
    //   y : length-N vector. New working response for subsequent sweeps.
    //   X : FLATTENED column-major N x p matrix (length N*p). Replaces the
    //       design matrix. Row count and column count must match the
    //       construction-time dimensions.
    //   r : SILENTLY IGNORED -- derived from the trees (no unique inverse).
    //       Use set_tree(...) to restore a forest.
    // Unknown keys are silently ignored per system_design.md Sec.7 so that
    // set_current(get_current()) round-trips cleanly.
    void set_current(const AI4BayesCode::state_map& params) {
        auto* blk = dynamic_cast<genbart_block*>(&impl_->child(0));

        // r is read-only output; silently ignored on input so that
        // round-trip set_current(get_current()) is supported per
        // system_design.md Sec.7 / Sec.16. Use set_tree() to restore the forest.

        const auto it_X = params.find("X");
        const auto it_y = params.find("y");
        const bool has_X = it_X != params.end();
        const bool has_y = it_y != params.end();

        // Reshape flattened X (column-major) back into an N x p matrix.
        auto reshape_X = [&](const arma::vec& x_flat) -> arma::mat {
            const std::size_t p = x_ncol_;
            if (p == 0 || (x_flat.n_elem % p) != 0) {
                ai4b::stop("GBartHeteroscedastic::set_current: flattened X "
                           "length %d is not a multiple of p = %d",
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
        // The forest's r at the training X, and "mean_r" derived from it, in
        // the shared data (get_current and predict_at({}) use them).
        if (has_X || has_y) impl_->sync_child(0);
    }

    // Restore the full genBART forest from a serialized snapshot previously
    // obtained from get_tree(). Separate from set_current because a tree is
    // a std::string, not a numeric state vector. Sets shared_data r and
    // "mean_r" from the restored forest.
    void set_tree(const std::string& tree_s) {
        auto* blk = dynamic_cast<genbart_block*>(&impl_->child(0));
        blk->set_tree(tree_s);
        impl_->sync_child(0);
    }
#ifdef AI4BAYESCODE_RCPP_MODULE
    // R-facing predict_at. An R matrix loses its dim crossing into state_map
    // (Rcpp flattens it column-major), so the column count is checked HERE,
    // while the SEXP still has it. See rcpp_predict_guard.hpp.
    AI4BayesCode::history_map predict_at_r(Rcpp::List new_data) const {
        return predict_at(ai4b::predict_input(new_data, "X", x_ncol_));
    }

    /// Same trailing switch as predict_at, for the R entry point.
    AI4BayesCode::history_map predict_at_r(Rcpp::List new_data,
                                           bool last_draw_only) const {
        return predict_at(ai4b::predict_input(new_data, "X", x_ncol_), last_draw_only);
    }
#endif


    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    // Predict at new data. Takes a state_map with optional key "X".
    //
    //   * empty map, keep_history = FALSE:
    //       y_rep: posterior-predictive at TRAINING X from current draw.
    //   * empty map, keep_history = TRUE:
    //       y_rep: n_draws x N posterior-predictive at training X, one
    //              row per stored draw (documented shadow loop -- tree
    //              forests cannot pass predict_at replaced-validation).
    //   * X supplied, keep_history = FALSE (or last_draw_only = TRUE):
    //       r, mean: from current trees at X_new.
    //       mean_r, y_rep: routed through the predict DAG (real
    //              posterior-predictive at X_new from the current draw).
    //   * X supplied, keep_history = TRUE: r, mean, mean_r, y_rep, each
    //       n_draws x N_new, one row per stored draw d. r_d = the forest
    //       stored at sweep d evaluated at X_new
    //       (genbart_block::predict_history; no offset), mean_d = mean_r_d =
    //       exp(r_d), y_rep_d ~ N(mean_d, phi_d * mean_d) per point, with
    //       phi_d the phi of sweep d. Requires keep_tree = TRUE (the forests
    //       of the stored draws are stored only then); without it the call
    //       raises an error.
    //
    // X is passed as a FLATTENED column-major vector under key "X"
    // (length N_new * p); reshaped here back into an N_new x p matrix.
    /// Unchanged one-argument form: predict_at(new_data) means exactly what
    /// it always did. The overload below adds the trailing switch.
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        return predict_at(new_data, /*last_draw_only=*/false);
    }

    /// last_draw_only = true takes the SINGLE-DRAW path -- the one this
    /// function already takes when keep_history is false -- without giving up
    /// the retained history.
    AI4BayesCode::history_map
    predict_at(const AI4BayesCode::state_map& new_data,
            bool last_draw_only) const {
        const bool use_history = keep_history_ && !last_draw_only;

        const bool has_X = new_data.find("X") != new_data.end();
        for (const auto& kv : new_data) {
            if (kv.first != "X") {
                ai4b::stop("GBartHeteroscedastic::predict_at: unknown key "
                           "'%s'. Valid keys: 'X' (flattened column-major "
                           "N_new x p matrix) or empty map for posterior "
                           "predictive at training X.",
                           kv.first.c_str());
            }
        }

        if (!has_X) {
            // ---- Training-X posterior predictive ------------------------
            if (use_history) {
                // History mode at training X: per-draw y_rep_d from r_d and
                // the phi of the same sweep.
                AI4BayesCode::history_map hist = impl_->get_history();
                const arma::mat& r_hist = hist.at("r");   // n_draws x N
                const std::size_t n_draws = r_hist.n_rows;
                const std::size_t N_local = r_hist.n_cols;
                // Before the first sweep the history is the current state
                // (one row) and no phi has been stored: that row uses the
                // current phi.
                if (!phi_hist_.empty() && phi_hist_.size() != n_draws) {
                    ai4b::stop("GBartHeteroscedastic::predict_at: inconsistent "
                               "history sizes (r=%d, phi=%d)",
                               (int)n_draws, (int)phi_hist_.size());
                }
                auto* lik_ptr = static_cast<genbart::lik::heteroscedastic_lik*>(
                    likelihood_.get());
                arma::mat yrep_mat(n_draws, N_local);
                std::normal_distribution<double> norm(0.0, 1.0);
                for (std::size_t d = 0; d < n_draws; ++d) {
                    const double phi =
                        phi_hist_.empty() ? lik_ptr->phi() : phi_hist_[d];
                    for (std::size_t i = 0; i < N_local; ++i) {
                        const double m  = std::exp(r_hist(d, i));
                        const double sd = std::sqrt(phi * m);
                        yrep_mat(d, i)  = m + sd * norm(predict_rng_);
                    }
                }
                AI4BayesCode::history_map out;
                out.emplace("y_rep", std::move(yrep_mat));
                return out;
            }

            // Stateful mode: route through the framework predict DAG so the
            // training-X path returns a posterior-predictive y_rep.
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

        // ---- New-X prediction -------------------------------------------
        // Reconstruct X_new (N_new x p) from the flattened input.
        const arma::vec& x_flat = new_data.at("X");
        if (x_ncol_ == 0 || (x_flat.n_elem % x_ncol_) != 0) {
            ai4b::stop("GBartHeteroscedastic::predict_at: flattened X length "
                       "%d is not a multiple of p = %d",
                       (int)x_flat.n_elem, (int)x_ncol_);
        }
        const std::size_t n_new = x_flat.n_elem / x_ncol_;
        arma::mat X_new = arma::reshape(x_flat, n_new, x_ncol_);

        auto& blk = dynamic_cast<genbart_block&>(impl_->child(0));

        if (use_history) {
            // History mode: one row per stored draw d, computed from the
            // forest stored at sweep d and the phi of the same sweep.
            if (!keep_tree_) {
                ai4b::stop("GBartHeteroscedastic::predict_at: prediction at "
                           "new X with keep_history = TRUE requires keep_tree "
                           "= TRUE at construction (the forests of the stored "
                           "draws are stored only then). Construct with "
                           "keep_tree = TRUE, or pass last_draw_only = TRUE to "
                           "predict from the current forest.");
            }
            arma::mat r_hist = blk.predict_history(X_new);  // n_draws x N_new
            const std::size_t n_draws = r_hist.n_rows;
            if (n_draws != blk.history_size()) {
                ai4b::stop("GBartHeteroscedastic::predict_at: inconsistent "
                           "history sizes (forests=%d, r=%d)",
                           (int)n_draws, (int)blk.history_size());
            }
            // Before the first sweep, or right after clear_history(), the
            // history is the current state (one row) and no phi is stored:
            // that row uses the current phi.
            if (!phi_hist_.empty() && phi_hist_.size() != n_draws) {
                ai4b::stop("GBartHeteroscedastic::predict_at: inconsistent "
                           "history sizes (r=%d, phi=%d)",
                           (int)n_draws, (int)phi_hist_.size());
            }
            auto* lik_ptr = static_cast<genbart::lik::heteroscedastic_lik*>(
                likelihood_.get());
            arma::mat mean_mat(n_draws, n_new), yrep_mat(n_draws, n_new);
            std::normal_distribution<double> norm(0.0, 1.0);
            for (std::size_t d = 0; d < n_draws; ++d) {
                const double phi =
                    phi_hist_.empty() ? lik_ptr->phi() : phi_hist_[d];
                for (std::size_t i = 0; i < n_new; ++i) {
                    const double m  = std::exp(r_hist(d, i));
                    const double sd = std::sqrt(phi * m);
                    mean_mat(d, i) = m;
                    yrep_mat(d, i) = m + sd * norm(predict_rng_);
                }
            }
            AI4BayesCode::history_map out;
            out.emplace("r",      std::move(r_hist));
            out.emplace("mean",   mean_mat);
            out.emplace("mean_r", std::move(mean_mat));
            out.emplace("y_rep",  std::move(yrep_mat));
            return out;
        }

        arma::vec r = blk.predict_r(X_new);

        // Route through the framework predict DAG so the new-X path returns
        // a REAL posterior-predictive y_rep. Inject r (block name "r" ->
        // valid replaced key); Pass-1 recomputes mean_r=exp(r); Pass-2
        // samples y_rep ~ N(mean_r, phi*mean_r) using the current phi.
        block_context replaced;
        replaced["r"] = r;
        block_context res = impl_->predict_at(replaced, predict_rng_);

        AI4BayesCode::history_map out;
        // Echo r and mean (mean = exp(r)).
        {
            arma::mat rm(1, r.n_elem), mm(1, r.n_elem);
            for (std::size_t i = 0; i < r.n_elem; ++i) {
                rm(0, i) = r[i];
                mm(0, i) = std::exp(r[i]);
            }
            out.emplace("r",    std::move(rm));
            out.emplace("mean", std::move(mm));
        }
        for (const auto& kv : res) {
            arma::mat m(1, kv.second.n_elem);
            for (std::size_t i = 0; i < kv.second.n_elem; ++i)
                m(0, i) = kv.second[i];
            out.emplace(kv.first, std::move(m));
        }
        return out;
    }

    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    // Called by clear_history() after impl_ is cleared: empties phi_hist_.
    void clear_extra_history_() { phi_hist_.clear(); }

    // Per-draw serialized genBART forest snapshots. In stateful mode this
    // is length 1 (the current forest); in tree-history mode (keep_tree=
    // true) it has one entry per stored draw.
    std::vector<std::string> get_tree_history() const {
        auto& blk = dynamic_cast<genbart_block&>(impl_->child(0));
        return blk.get_tree_history();
    }

private:
    std::mt19937_64                      rng_;
    // predict_rng_ is mutable so `predict_at() const` can advance it when
    // sampling y_rep. Seeded once at construction (derived from rng_seed
    // XOR'd with the golden-ratio constant) so posterior predictive draws
    // are reproducible given a stable construction seed.
    mutable std::mt19937_64              predict_rng_;
    std::unique_ptr<composite_block>     impl_;
    std::unique_ptr<genbart::likelihood> likelihood_;
    std::size_t                          x_ncol_ = 0;
    bool                                 keep_tree_    = false;
    bool                                 keep_history_ = false;
    std::vector<double>                  phi_hist_;   // phi of each sweep (keep_history_)
};

// ============================================================================
//  Rcpp module: make the class visible to R.
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(GBartHeteroscedastic_module) {
    Rcpp::class_<GBartHeteroscedastic>("GBartHeteroscedastic")
        // Short ctor: defaults keep_tree=FALSE, keep_history=FALSE.
        .constructor<arma::mat, arma::vec, int, double, int>(
            "Short ctor: X (N x p matrix), y (length N), ntrees, phi_init "
            "(> 0), seed; keep_tree and keep_history default FALSE.")
        // Ctor with keep_tree only (keep_history defaults FALSE).
        .constructor<arma::mat, arma::vec, int, double, int, bool>(
            "Ctor with keep_tree only: keep_tree (forest snapshots per "
            "step; EXPENSIVE; required by predict_at at new X with "
            "keep_history = TRUE); keep_history defaults FALSE.")
        // Full ctor: keep_tree + keep_history.
        .constructor<arma::mat, arma::vec, int, double, int, bool, bool>(
            "Full ctor: X (N x p), y (length N), ntrees, phi_init (> 0), "
            "seed (RNG seed, 0 = random), keep_tree (forest snapshots; "
            "EXPENSIVE; default FALSE; required by predict_at at new X "
            "with keep_history = TRUE), keep_history (numeric per-step "
            "buffers for trace analysis; cheap; default FALSE).")
        .method("step", (void (GBartHeteroscedastic::*)())    &GBartHeteroscedastic::step, "Run one sweep.")
        .method("step", (void (GBartHeteroscedastic::*)(int)) &GBartHeteroscedastic::step,
                "Run n RJMCMC sweeps.")
        .method("get_current", &GBartHeteroscedastic::get_current,
                "Return the current draw as a named list with $r, $mean "
                "(= exp(r)) and $phi (dispersion). The serialized genBART "
                "forest is available separately via get_tree().")
        .method("get_tree",    &GBartHeteroscedastic::get_tree,
                "Return the serialized genBART forest as a length-1 string "
                "(round-trips through set_tree()).")
        .method("set_current", &GBartHeteroscedastic::set_current,
                "Overwrite working response (y) and/or X from a named "
                "list. Supported keys: y, X (flattened column-major N x p). "
                "r is read-only; use set_tree() to restore the forest.")
        .method("set_tree",    &GBartHeteroscedastic::set_tree,
                "Restore the genBART forest from a serialized snapshot "
                "string previously obtained from get_tree().")
        .method("predict_at",
                (AI4BayesCode::history_map (GBartHeteroscedastic::*)(Rcpp::List) const)
                    &GBartHeteroscedastic::predict_at_r,
                "Predict at new data. Pass list(X = as.vector(X_new)) "
                "(flattened column-major) for new-X, or an empty list for "
                "training-X posterior predictive. keep_history = TRUE: one "
                "row per stored draw (new X requires keep_tree = TRUE). "
                "Const, no state mutation.")
        .method("predict_at",
                (AI4BayesCode::history_map (GBartHeteroscedastic::*)(Rcpp::List, bool) const)
                    &GBartHeteroscedastic::predict_at_r,
                "predict_at(new_data, last_draw_only): TRUE takes the "
                "single-draw path without discarding the history.")
        .method("get_dag",     &GBartHeteroscedastic::get_dag,
                "Return the predict DAG as a named list of edges.")
        .method("get_history", &GBartHeteroscedastic::get_history,
                "Return the history as a named list of matrices "
                "(r [n_draws x N]).")
        .method("get_tree_history", &GBartHeteroscedastic::get_tree_history,
                "Return per-draw serialized genBART forests (one per stored "
                "draw when keep_tree=TRUE; else the current forest).")
        AI4BAYESCODE_BIND_KERNEL_CONTROL(GBartHeteroscedastic);
}
#endif

// ============================================================================
//  pybind11 module: make the class visible to Python.
// ============================================================================

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(GBartHeteroscedastic, m) {
    AI4BayesCode::register_ai4bayescode_types(m);

    // genBART draws from its OWN thread-local kernel RNG stream
    // (bart_rng in bart_pure_cpp/src/r_compat.h); the std::mt19937 passed
    // to step() is ignored. The constructor seeds the kernel stream with a
    // non-zero rng_seed (bart_rng::set_seed), which replaces any earlier
    // seed. This free function seeds the stream directly (kernel plumbing,
    // not model substance), e.g. for a model built with rng_seed = 0 --
    // mirrors the documented bart_rng::set_seed(seed) convention in
    // genbart_block.hpp.
    m.def("set_kernel_seed",
          [](std::uint64_t s) { bart_rng::set_seed(s); },
          pybind11::arg("seed"),
          "Seed the genBART kernel RNG stream of the calling thread. A "
          "constructor with rng_seed != 0 seeds the stream itself, which "
          "replaces a seed set before the construction.");

    pybind11::class_<GBartHeteroscedastic>(m, "GBartHeteroscedastic")
        .def(pybind11::init<arma::mat, arma::vec,
                            int, double, int, bool, bool>(),
             pybind11::arg("X"),
             pybind11::arg("y"),
             pybind11::arg("ntrees")       = 50,
             pybind11::arg("phi_init")     = 1.0,
             pybind11::arg("rng_seed")     = 1,
             pybind11::arg("keep_tree")    = false,
             pybind11::arg("keep_history") = false)
        .def("step", (void (GBartHeteroscedastic::*)())    &GBartHeteroscedastic::step, "Run one sweep.")
        .def("step", (void (GBartHeteroscedastic::*)(int)) &GBartHeteroscedastic::step,
             pybind11::arg("n_steps"))
        .def("get_current",     &GBartHeteroscedastic::get_current)
        .def("get_tree",        &GBartHeteroscedastic::get_tree)
        .def("set_current",     &GBartHeteroscedastic::set_current,
             pybind11::arg("params"))
        .def("set_tree",        &GBartHeteroscedastic::set_tree,
             pybind11::arg("tree_s"))
        .def("predict_at",
             (AI4BayesCode::history_map (GBartHeteroscedastic::*)(const AI4BayesCode::state_map&, bool) const)
                 &GBartHeteroscedastic::predict_at,
             pybind11::arg("new_data"),
             pybind11::arg("last_draw_only") = false)
        .def("get_dag",         &GBartHeteroscedastic::get_dag)
        .def("get_history",     &GBartHeteroscedastic::get_history)
        .def("get_tree_history", &GBartHeteroscedastic::get_tree_history)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(GBartHeteroscedastic);
}
#endif

// ============================================================================
//  Standalone demo: simulate Linero 2022 Sec.4.2 heteroscedastic data, fit,
//  print finite recovery of the dispersion phi and the fitted mean.
// ============================================================================

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>

int main() {
    // ---- Simulate a well-identified heteroscedastic DGP -----------------
    //   log-mean with a kink at x1 = 0: r(x) = 1.0 + 0.8 max(x1, 0) + 0.6 x2
    //   mean  m(x)  = exp(r(x))   (in roughly [1.5, 11.0], well above 0)
    //   y     ~ Normal(m, phi_true * m),  phi_true = 0.5
    // N >= 400 with well-spread covariates so the identified quantities
    // (phi, and the fitted mean at a grid) converge. genBART's per-point r
    // has intrinsic RJMCMC tree-multimodality -- that is expected and not a
    // defect; we report recovery of the IDENTIFIED mean and phi instead.
    const std::size_t N = 600;
    const std::size_t p = 2;
    const double phi_true = 0.5;

    std::mt19937_64 gen(20260622ULL);
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::normal_distribution<double>       znorm(0.0, 1.0);

    arma::mat X(N, p);
    arma::vec y(N);
    arma::vec m_true(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double x1 = unif(gen);
        const double x2 = unif(gen);
        X(i, 0) = x1; X(i, 1) = x2;
        const double r = 1.0 + 0.8 * std::max(x1, 0.0) + 0.6 * x2;
        const double m = std::exp(r);
        m_true[i] = m;
        const double sd = std::sqrt(phi_true * m);
        y[i] = m + sd * znorm(gen);
    }

    // ---- Fit -------------------------------------------------------------
    GBartHeteroscedastic model(X, y,
                               /*ntrees=*/50, /*phi_init=*/1.0,
                               /*rng_seed=*/42,
                               /*keep_tree=*/false, /*keep_history=*/false);

    const int n_burn = 1000;
    const int n_keep = 1000;
    model.step(n_burn);

    // Accumulate posterior-mean fitted mean and phi over kept draws.
    arma::vec mean_sum(N, arma::fill::zeros);
    double    phi_sum = 0.0;
    for (int it = 0; it < n_keep; ++it) {
        model.step(1);
        AI4BayesCode::state_map cur = model.get_current();
        mean_sum += cur.at("mean");
        phi_sum  += cur.at("phi")[0];
    }
    const arma::vec mean_hat = mean_sum / static_cast<double>(n_keep);
    const double    phi_hat  = phi_sum  / static_cast<double>(n_keep);

    // ---- Report recovery -------------------------------------------------
    const double rmse_m = std::sqrt(arma::mean(arma::square(mean_hat - m_true)));
    const double cor_m  = arma::as_scalar(arma::cor(mean_hat, m_true));
    const bool finite_ok = std::isfinite(phi_hat)
                           && std::isfinite(rmse_m)
                           && std::isfinite(cor_m)
                           && mean_hat.is_finite();

    std::printf("GBartHeteroscedastic standalone demo\n");
    std::printf("  N = %zu, p = %zu, ntrees = 50\n", N, p);
    std::printf("  phi_true = %.4f   phi_hat = %.4f\n", phi_true, phi_hat);
    std::printf("  RMSE(mean_hat, m_true)   = %.4f\n", rmse_m);
    std::printf("  cor(mean_hat, m_true)    = %.4f\n", cor_m);
    std::printf("  all finite               = %s\n",
                finite_ok ? "YES" : "NO");

    // Finiteness alone would pass a model that returned a constant. Gate on the
    // recovery quantities this demo already computes: cor(mean_hat, m_true) > 0.70.
    const bool recovery_ok = cor_m > 0.70;

    if (!finite_ok) {
        std::printf("FAIL: non-finite recovery\n");
        return 1;
    }
    if (!recovery_ok) {
        std::printf("FAIL: recovery below tolerance\n");
        return 1;
    }
    std::printf("OK: recovery within tolerance\n");
    return 0;
}
#endif

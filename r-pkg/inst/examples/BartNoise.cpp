// ============================================================================
//  BartNoise.cpp
//
//  Copyright (C) 2026 AI4BayesCode.
//  Licensed under GPL-2.0-or-later (inherits from bart_block.hpp)
//
//  REFERENCE TEMPLATE for the AI4BayesCode skill -- bart_block variant.
//  Generated samplers that involve a BART mean component should follow the
//  STRUCTURE of this file exactly. Only the model-specific bits change.
//
//  Model
//  -----
//      y_i | mu, sigma  ~ Normal(f_bart(x_i), sigma^2),   i = 1..N
//      f_bart           ~ BART tree-ensemble prior
//      sigma^2          ~ InverseGamma(nu/2, nu*lambda/2)
//                         with lambda = sigest^2 * qchisq(0.1, nu) / nu,
//                         the BART::wbart default (sigquant = 0.9).
//                         This is the default — the simple
//                         model y ~ N(BART(X), sigma^2) has non-
//                         identifiability issues under a non-informative
//                         prior (a larger tree fit can be compensated
//                         by a smaller sigma), so we use BART's
//                         calibrated conjugate IG. Sampled via NUTS
//                         (same kernel as every other continuous block).
//
//  Block decomposition
//  -------------------
//      f_bart           : sampled by bart_block (one tree sweep per
//                         composite step)
//      sigma            : sampled by a positive nuts_block with the
//                         standard Gaussian log-likelihood conditional
//
//  BACKEND-NEUTRAL DUAL MODULE
//  ---------------------------
//  This file compiles under three backends from a single source:
//    * AI4BAYESCODE_RCPP_MODULE   -> RcppArmadillo, RCPP_MODULE (R)
//    * AI4BAYESCODE_PYBIND_MODULE -> pure armadillo, PYBIND11_MODULE (Python)
//    * neither defined            -> standalone int main() demo
//  All shared code names only backend-neutral types (arma::vec/mat,
//  AI4BayesCode::state_map / history_map) and routes errors through
//  ai4b::stop. The BART IG-prior calibration `R::qchisq(0.1, nu, ...)`
//  resolves to R's math library under the Rcpp backend and to the
//  kernel's `namespace R { qchisq(...) }` (bart_pure_cpp/src/r_compat.h,
//  ~5e-14 vs R) under pybind / standalone — same calibrated lambda
//  across all three.
//
//  LICENSE WARNING
//  ---------------
//  This file #includes "AI4BayesCode/bart_block.hpp", which in turn
//  includes the vendored BART tree kernel under bart_pure_cpp/src/,
//  derived from the CRAN BART R package (GPL-2.0-or-later). Any
//  binary or source distribution of this translation unit is therefore
//  subject to GPL-2.0-or-later. Do NOT slap a permissive license
//  header on this file or on any downstream sampler that #includes
//  bart_block.hpp.
// ============================================================================
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("BartNoise")
//   set.seed(42); N <- 600L; p <- 3L                      # well-identified DGP (= int main)
//   X <- matrix(runif(N * p, -1, 1), N, p)                # X ~ Uniform(-1, 1)
//   f <- sin(3 * X[, 1]) + 0.5 * X[, 2]^2 - X[, 3]        # smooth low-dim mean
//   y <- f + rnorm(N, 0, 0.5)                             # y ~ N(f, sigma_true^2), sigma_true = 0.5
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(BartNoise, X, y, 50L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE, seed, FALSE, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(BartNoise, X, y, 50L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE, 42L)
//                                                          # ntrees=50,k=2,power=2,base=.95,nu=3,numcut=100,dart=F,aug=F,seed=42
//   m$step(2000L); str(m$get_current())                  # $f_bart fitted mean, $sigma noise SD
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(42); N, p = 600, 3        # well-identified DGP (= int main)
//   X = rng.uniform(-1, 1, size=(N, p))                   # X ~ Uniform(-1, 1)
//   f = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1]**2 - X[:, 2]  # smooth low-dim mean
//   y = f + rng.normal(0, 0.5, size=N)                    # y ~ N(f, sigma_true^2), sigma_true = 0.5
//   Mod = AI4BayesCode.example("BartNoise")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.BartNoise(X, y, 50, 2.0, 2.0, 0.95, 3.0, 100, False, False, seed, False, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.BartNoise(X, y, 50, 2.0, 2.0, 0.95, 3.0, 100, False, False, 42, False, False)
//   m.step(2000); print(m.get_current())                 # 'f_bart' fitted mean, 'sigma' noise SD
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
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
using AI4BayesCode::bart_block;
using AI4BayesCode::bart_block_config;

namespace constraints = AI4BayesCode::constraints;

// ============================================================================
//  Natural-scale log-density for sigma, given the BART fit.
//
//  Prior (default — BART::wbart calibration, sigquant = 0.9):
//      sigma^2 ~ InverseGamma(nu/2, nu*lambda/2)
//      lambda  = sigest^2 * qchisq(0.1, nu) / nu
//  The combined log-density of sigma (after the d sigma^2 / d sigma
//  Jacobian embedded in the IG->sigma transform) is:
//
//      log p(sigma | y, f_bart, nu, lambda)
//          = -(N + nu + 1) * log(sigma)
//            - (sum_sq + nu*lambda) / (2 * sigma^2)               (+ const)
//
//      d/dsigma log p = -(N + nu + 1)/sigma + (sum_sq + nu*lambda)/sigma^3
//
//  The positive-constraint Jacobian (+ log(sigma)) is added automatically
//  by constraints::positive::wrap, which transforms to the unconstrained
//  theta_unc = log(sigma) scale that NUTS operates on.
// ============================================================================

namespace {

double sigma_natural_log_density(const arma::vec& sigma_nat,
                                 const block_context& ctx,
                                 arma::vec* grad_nat) {
    const double sigma      = sigma_nat[0];
    const arma::vec& y      = ctx.at("y");
    const arma::vec& f_bart = ctx.at("f_bart");
    const double nu         = ctx.at("sigma_nu")[0];
    const double lambda     = ctx.at("sigma_lambda")[0];

    const double sigma2 = sigma * sigma;
    const double N      = static_cast<double>(y.n_elem);

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < y.n_elem; ++i) {
        const double r = y[i] - f_bart[i];
        sum_sq += r * r;
    }

    const double rate = sum_sq + nu * lambda;
    const double lp   = -(N + nu + 1.0) * std::log(sigma)
                        - 0.5 * rate / sigma2;

    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = -(N + nu + 1.0) / sigma + rate / (sigma2 * sigma);
    }
    return lp;
}

} // anonymous namespace

// ============================================================================
//  User-facing class. Backend-neutral signatures: arma containers,
//  state_map / history_map, ai4b::stop.
// ============================================================================

class BartNoise {
public:
    BartNoise(const arma::mat& X,
              const arma::vec& y,
              int    ntrees,
              double k,
              double power,
              double base,
              double nu,
              int    numcut,
              bool   dart,
              bool   aug,
              int    rng_seed,
              bool   keep_tree    = false,
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
          impl_(std::make_unique<composite_block>("BartNoise")),
          keep_tree_(keep_tree),
          keep_history_(keep_history)
    {
        if (X.n_rows != y.n_elem) {
            ai4b::stop("BartNoise: X and y must have matching row counts");
        }

        const std::size_t N = static_cast<std::size_t>(X.n_rows);

        // ---- Install fixed data and initial parameter values ------------
        impl_->data().set("y", y);

        // For a pure BART + noise model the BART working response IS y,
        // so we set "bart_target" to y and never refresh it.
        impl_->data().set("bart_target", y);

        // Store X as a flattened (column-major) vector in shared_data so
        // the predict_at DAG knows about it (the actual matrix lives in
        // bart_block_config).
        impl_->data().set("X", arma::vec(arma::vectorise(X)));
        x_ncol_ = static_cast<std::size_t>(X.n_cols);

        // Placeholder sigma; after constructing bart_block below, we
        // overwrite with bart_model's OLS-computed sigest.
        impl_->data().set("sigma",  arma::vec{arma::stddev(y)});
        impl_->data().set("f_bart", arma::vec(N, arma::fill::zeros));

        // ---- Declare the dependency DAG ----------------------------------
        impl_->data().declare_dependencies(
            "f_bart", {"bart_target", "sigma"});
        impl_->data().declare_dependencies(
            "sigma",  {"y", "f_bart", "sigma_nu", "sigma_lambda"});

        // ---- Declare predict-at data inputs ------------------------------
        impl_->data().declare_data_input("X");

        // ---- Declare predict DAG (generative direction) -----------------
        // X → f_bart  (BART tree traversal)
        // f_bart → y_rep, sigma → y_rep  (observation layer, y ~ N(f_bart, sigma^2))
        impl_->data().declare_predict_edges("X",      {"f_bart"});
        impl_->data().declare_predict_edges("f_bart", {"y_rep"});
        impl_->data().declare_predict_edges("sigma",  {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). sigma^2 ~ IG(nu/2, nu*lambda/2)
        //      (calibrated): sigma_nu / sigma_lambda are sigma's prior
        //      parents. The BART forest is itself a sampled generative
        //      parent of f_bart alongside X (cf. MetaRegBartSpline). All
        //      drawn faded by ai4bayescode_plot_dag.
        impl_->data().declare_context_edges("sigma_nu",     {"sigma"});
        impl_->data().declare_context_edges("sigma_lambda", {"sigma"});
        impl_->data().declare_context_edges("BART",         {"f_bart"});

        // Initialize y_rep slot so downstream refreshers have a slot to
        // write into. Value is overwritten every predict_at call.
        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));

        // Stochastic refresher: y_rep ~ N(f_bart, sigma^2), length matches f_bart.
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d, std::mt19937_64& rng) {
                const arma::vec& f = d.get("f_bart");
                const double s     = d.get("sigma")[0];
                std::normal_distribution<double> norm(0.0, 1.0);
                arma::vec y_rep(f.n_elem);
                for (std::size_t i = 0; i < f.n_elem; ++i) {
                    y_rep[i] = f[i] + s * norm(rng);
                }
                return y_rep;
            });

        // ---- Add the BART block -----------------------------------------
        bart_block_config bart_cfg;
        bart_cfg.name                 = "f_bart";
        bart_cfg.x_train              = X;
        bart_cfg.y_init               = y;
        bart_cfg.working_response_key = "bart_target";
        bart_cfg.sigma_key            = "sigma";
        bart_cfg.ntrees               = ntrees;
        bart_cfg.k                    = k;
        bart_cfg.power                = power;
        bart_cfg.base                 = base;
        bart_cfg.nu                   = nu;
        bart_cfg.numcut               = numcut;
        bart_cfg.dart                 = dart;
        bart_cfg.aug                  = aug;
        bart_cfg.sigma_init           = arma::stddev(y);
        impl_->add_child(
            std::make_unique<bart_block>(std::move(bart_cfg)));

        // Sync shared_data sigma with bart_model's OLS-computed sigest,
        // and install BART's default calibrated IG prior:
        //     sigma^2 ~ IG(nu/2, nu*lambda/2)
        //     lambda  = sigest^2 * qchisq(0.1, nu) / nu    (sigquant = 0.9)
        {
            auto& bblk = dynamic_cast<bart_block&>(impl_->child(0));
            const double sigest = bblk.current_sigma();
            impl_->data().set("sigma", arma::vec{sigest});

            // R::qchisq resolves to R's math library under the Rcpp
            // backend and to the kernel's namespace R { qchisq } (from
            // bart_pure_cpp/src/r_compat.h, ~5e-14 vs R) under
            // pybind / standalone. Same calibrated lambda in all three.
            const double qchi   = R::qchisq(0.1, nu, /*lower.tail=*/1,
                                            /*log.p=*/0);
            const double lambda = sigest * sigest * qchi / nu;
            impl_->data().set("sigma_nu",     arma::vec{nu});
            impl_->data().set("sigma_lambda", arma::vec{lambda});
        }

        // ---- Add the sigma block (positive NUTS) ------------------------
        // Use bart_model's OLS sigest as initial sigma for NUTS, not
        // the user-provided sigma_init (which is typically sd(y) and
        // includes signal).
        const double sigest_for_nuts =
            dynamic_cast<bart_block&>(impl_->child(0)).current_sigma();
        nuts_block_config sg_cfg;
        sg_cfg.name        = "sigma";
        sg_cfg.initial_unc = arma::vec{std::log(sigest_for_nuts)};
        sg_cfg.constrain   = constraints::positive::constrain;
        sg_cfg.unconstrain = constraints::positive::unconstrain;
        sg_cfg.log_density_grad =
            [](const arma::vec& theta_unc, const block_context& ctx,
               arma::vec* grad) {
                return constraints::positive::wrap(
                    theta_unc, grad,
                    [&](const arma::vec& sigma_nat,
                        arma::vec* grad_nat) {
                        return sigma_natural_log_density(
                            sigma_nat, ctx, grad_nat);
                    });
            };
        sg_cfg.nuts_settings.nuts_settings.max_tree_depth     = 6;
        sg_cfg.nuts_settings.nuts_settings.target_accept_rate = 0.8;
        impl_->add_child(
            std::make_unique<nuts_block>(std::move(sg_cfg)));

        // ---- Enable history recording if requested ----------------------
        // keep_tree (BART forest snapshots, expensive) and keep_history
        // (numeric per-step buffers, cheap) are independent toggles.
        if (keep_tree_)    impl_->set_keep_tree(true);
        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Current draw: f_bart + sigma. (The serialized BART forest snapshot
    // stays reachable via get_tree(); it is not a numeric state vector so
    // it lives outside the state_map.)
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["f_bart"] = impl_->data().get("f_bart");
        out["sigma"]  = impl_->data().get("sigma");
        return out;
    }

    // Serialized BART forest snapshot (round-trips through
    // set_tree(...)). Kept as a separate passthrough so the forest state
    // stays reachable from both backends without putting a string into
    // the numeric state_map.
    std::string get_tree() const {
        auto& bart_child = dynamic_cast<bart_block&>(impl_->child(0));
        return bart_child.get_tree();
    }

    // Predict at new data. Takes a state_map with optional key "X".
    //
    // Behavior depends on construction-time keep_history flag and whether
    // X was supplied:
    //   * keep_history = FALSE, X supplied:
    //       f_bart: length N_new, from current trees.
    //       y_rep:  posterior-predictive at X_new from CURRENT draw.
    //   * keep_history = FALSE, empty map:
    //       y_rep:  posterior-predictive at TRAINING X from current draw.
    //   * keep_history = TRUE,  X supplied:
    //       f_bart: n_draws x N_new from each stored draw's trees.
    //       y_rep:  n_draws x N_new posterior-predictive samples.
    //   * keep_history = TRUE,  empty map:
    //       y_rep:  n_draws x N_train posterior-predictive at training X.
    //
    // X is passed as a FLATTENED column-major vector under key "X"
    // (length N_new * p); reshaped here back into an N_new x p matrix.
    // Uses predict_rng_ (const-preserving, persistent, seeded at
    // construction) so posterior-predictive draws are reproducible given
    // the same seed. Does NOT modify MCMC state in any mode.
    AI4BayesCode::history_map
    predict_at(const AI4BayesCode::state_map& new_data) const {
        // Validate keys.
        const bool has_X = new_data.find("X") != new_data.end();
        for (const auto& kv : new_data) {
            if (kv.first != "X") {
                ai4b::stop("BartNoise::predict_at: unknown key '%s'. "
                           "Valid keys: 'X' (flattened column-major "
                           "N_new x p matrix) or empty map for posterior "
                           "predictive at training X.",
                           kv.first.c_str());
            }
        }

        // Reconstruct X_new (N_new x p) from the flattened input if given.
        arma::mat X_new;
        if (has_X) {
            const arma::vec& x_flat = new_data.at("X");
            if (x_ncol_ == 0 || (x_flat.n_elem % x_ncol_) != 0) {
                ai4b::stop("BartNoise::predict_at: flattened X length %d "
                           "is not a multiple of p = %d",
                           (int)x_flat.n_elem, (int)x_ncol_);
            }
            const std::size_t n_new = x_flat.n_elem / x_ncol_;
            X_new = arma::reshape(x_flat, n_new, x_ncol_);
        }

        auto& bart_child = dynamic_cast<bart_block&>(impl_->child(0));

        if (!keep_history_) {
            // ---- Stateful mode ------------------------------------------
            block_context replaced;
            arma::vec f_pred;           // only set if has_X
            if (has_X) {
                f_pred = bart_child.predict(X_new);
                replaced["X"]      = arma::vec(arma::vectorise(X_new));
                replaced["f_bart"] = f_pred;
            }
            block_context result = impl_->predict_at(replaced, predict_rng_);

            AI4BayesCode::history_map out;
            if (has_X) {
                // Echo f_bart prediction (it was in `replaced`, so
                // composite_block::predict_at strips it from `result`).
                arma::mat fm(1, f_pred.n_elem);
                for (std::size_t i = 0; i < f_pred.n_elem; ++i) fm(0, i) = f_pred[i];
                out.emplace("f_bart", std::move(fm));
            }
            for (const auto& kv : result) {
                arma::mat m(1, kv.second.n_elem);
                for (std::size_t i = 0; i < kv.second.n_elem; ++i)
                    m(0, i) = kv.second[i];
                out.emplace(kv.first, std::move(m));
            }
            return out;
        }

        // ---- History mode ---------------------------------------------
        auto& nuts_child = dynamic_cast<nuts_block&>(impl_->child(1));
        AI4BayesCode::history_map sigma_list = nuts_child.get_history();
        const arma::mat& sigma_hist = sigma_list.at("sigma");
        const std::size_t n_draws = sigma_hist.n_rows;

        // F_pred[d, :] = f_bart prediction from draw d's trees.
        // When X supplied: predict at X_new. Otherwise: use stored training
        // f_bart from each draw's get_history.
        arma::mat F_pred;
        arma::vec x_flat;
        bool echo_f_bart = has_X;  // echo f_bart in output only when X given
        if (has_X) {
            F_pred = bart_child.predict_history(X_new);
            x_flat = arma::vec(arma::vectorise(X_new));
        } else {
            // Empty map: use training-X f_bart history from the block's
            // own get_history(). f_bart returned as arma::mat
            // (n_draws x N_train).
            AI4BayesCode::history_map bart_hist_list = bart_child.get_history();
            F_pred = bart_hist_list.at("f_bart");
        }

        if (n_draws != F_pred.n_rows) {
            ai4b::stop("BartNoise::predict_at: inconsistent history sizes "
                       "(bart=%d, sigma=%d)",
                       (int)F_pred.n_rows, (int)n_draws);
        }

        std::unordered_map<std::string, std::vector<arma::vec>> collected;

        for (std::size_t d = 0; d < n_draws; ++d) {
            block_context replaced;
            if (has_X) replaced["X"] = x_flat;
            replaced["f_bart"] = F_pred.row(d).t();
            replaced["sigma"]  = arma::vec{sigma_hist(d, 0)};

            block_context result = impl_->predict_at(replaced, predict_rng_);

            for (const auto& kv : result) {
                collected[kv.first].push_back(kv.second);
            }
            if (echo_f_bart) {
                collected["f_bart"].push_back(F_pred.row(d).t());
            }
        }

        // Aggregate: each key -> arma::mat (n_draws x dim). Scalars become
        // n_draws x 1.
        AI4BayesCode::history_map out;
        for (const auto& kv : collected) {
            if (kv.second.empty()) continue;
            const std::size_t dim = kv.second[0].n_elem;
            arma::mat m(n_draws, dim);
            for (std::size_t i = 0; i < n_draws; ++i) {
                for (std::size_t j = 0; j < dim; ++j) {
                    m(i, j) = kv.second[i][j];
                }
            }
            out.emplace(kv.first, std::move(m));
        }
        return out;
    }

    // Full model DAG: $gibbs_reads, $gibbs_invalidates, $predict_edges,
    // $data_inputs. See composite_block::get_dag() for the semantics.
    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    // Update parameters / data inputs from an outer sampler.
    // Supported keys (any subset; order independent):
    //   sigma  : scalar > 0. Overwrites the current noise SD.
    //   y      : length-n vector. Pushed into bart_block as the new
    //            working response for subsequent sweeps. Typical use:
    //            outer Gibbs computes y_raw - X_other %*% beta_other
    //            (partial residual) and passes the vector in here.
    //   X      : FLATTENED column-major n x p matrix (length n*p).
    //            Replaces the design matrix. Row count and column count
    //            must match the construction-time dimensions. Cutpoints
    //            are NOT rebuilt; new X must have comparable range to the
    //            original (typical in missing-data / measurement-error
    //            Gibbs where each sweep imputes a slightly different X).
    //   f_bart : SILENTLY IGNORED -- derived from trees + fmean (no
    //            unique inverse). Use the tree round-trip
    //            (get_tree / set_tree at the block level) instead.
    // Unknown keys are silently ignored per system_design.md §7 so
    // that set_current(get_current()) round-trips cleanly.
    //
    // NOTE: tree restoration is exposed via the block-level get_tree() /
    // bart_block::set_tree string API (a std::string, not a numeric
    // state), so it is intentionally NOT part of the numeric state_map
    // round-trip here.
    void set_current(const AI4BayesCode::state_map& params) {
        auto* bart_blk =
            dynamic_cast<bart_block*>(&impl_->child(0));
        auto* sg_blk =
            dynamic_cast<nuts_block*>(&impl_->child(1));

        // f_bart is read-only output; silently ignored on input so
        // that round-trip set_current(get_current()) is supported per
        // system_design.md §7 / §16.

        const auto it_X = params.find("X");
        const auto it_y = params.find("y");
        const bool has_X = it_X != params.end();
        const bool has_y = it_y != params.end();

        // Reshape flattened X (column-major) back into an n x p matrix.
        auto reshape_X = [&](const arma::vec& x_flat) -> arma::mat {
            const std::size_t p = x_ncol_;
            if (p == 0 || (x_flat.n_elem % p) != 0) {
                ai4b::stop("BartNoise::set_current: flattened X length %d "
                           "is not a multiple of p = %d",
                           (int)x_flat.n_elem, (int)p);
            }
            return arma::reshape(x_flat, x_flat.n_elem / p, p);
        };

        if (has_X && has_y) {
            arma::mat X_new = reshape_X(it_X->second);
            const arma::vec& y_new = it_y->second;
            bart_blk->set_data(X_new, y_new);
            // Push into shared_data so working_response_key refresh on
            // the next set_context is a no-op / consistent.
            impl_->data().set("bart_target", y_new);
            impl_->data().set("X", arma::vec(arma::vectorise(X_new)));
        } else if (has_X) {
            arma::mat X_new = reshape_X(it_X->second);
            bart_blk->set_X(X_new);
            impl_->data().set("X", arma::vec(arma::vectorise(X_new)));
        } else if (has_y) {
            const arma::vec& y_new = it_y->second;
            bart_blk->set_Y(y_new);
            impl_->data().set("bart_target", y_new);
        }

        const auto it_s = params.find("sigma");
        if (it_s != params.end()) {
            const double s = it_s->second[0];
            if (!(s > 0.0)) ai4b::stop("sigma must be strictly positive");
            sg_blk->set_current(arma::vec{s});
            impl_->data().set("sigma", arma::vec{s});
        }
    }

    // Restore the full BART forest from a serialized snapshot previously
    // obtained from get_tree(). Separate from set_current because a tree
    // is a std::string, not a numeric state vector. Refreshes shared_data
    // f_bart so downstream blocks see the restored forest immediately.
    void set_tree(const std::string& tree_s) {
        auto* bart_blk = dynamic_cast<bart_block*>(&impl_->child(0));
        bart_blk->set_tree(tree_s);
        const arma::vec& f_new = bart_blk->current();
        impl_->data().set("f_bart", f_new);
    }

    // ---- History access --------------------------------------------------
    // The composite's neutral history_map (numeric matrices keyed by
    // block name: f_bart [n_draws x N], sigma [n_draws x 1]). The
    // per-draw serialized BART forests are reachable separately via
    // get_tree_history() (a vector<string>, outside the numeric map).
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    // Per-draw serialized BART forest snapshots. In stateful mode this is
    // length 1 (the current forest); in history mode (keep_tree=true) it
    // has one entry per stored draw.
    std::vector<std::string> get_tree_history() const {
        auto& bart_child = dynamic_cast<bart_block&>(impl_->child(0));
        return bart_child.get_tree_history();
    }

    /// 7th method: re-tune NUTS metric (mass matrix + step size + dual
    /// averaging) without advancing chain state. Available because the
    /// composite contains NUTS-family children. See system_design.md §13
    /// NUTS-family + validator.md §24.
    void readapt_NUTS(int n, bool reset = false) {
        if (n < 0) {
            ai4b::stop("readapt_NUTS: n must be non-negative");
        }
        impl_->readapt_NUTS(static_cast<std::size_t>(n),
                            reset, readapt_rng_);
    }


private:
    std::mt19937_64                  rng_;
    // predict_rng_ is mutable so `predict_at() const` can advance it when
    // sampling y_rep. Seeded once at construction (derived from rng_seed
    // XOR'd with the golden-ratio constant) so posterior predictive draws
    // are reproducible given a stable construction seed without stealing
    // entropy from the main MCMC RNG.
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_; // readapt_NUTS() advances it (7th method)
    std::unique_ptr<composite_block> impl_;
    std::size_t                      x_ncol_ = 0;
    bool                             keep_tree_    = false;
    bool                             keep_history_ = false;
};

// ============================================================================
//  Rcpp module: make the class visible to R.
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(BartNoise_module) {
    Rcpp::class_<BartNoise>("BartNoise")
        // Short constructor: defaults keep_tree=FALSE, keep_history=FALSE.
        .constructor<arma::mat, arma::vec,
                     int, double, double, double, double,
                     int, bool, bool, int>(
            "Short constructor; keep_tree and keep_history default FALSE.")
        // Constructor with keep_tree only (keep_history defaults FALSE).
        .constructor<arma::mat, arma::vec,
                     int, double, double, double, double,
                     int, bool, bool, int, bool>(
            "Constructor with keep_tree (BART forest snapshots per step; "
            "EXPENSIVE, required only for predict_history(X_new)); "
            "keep_history defaults FALSE.")
        // Full constructor: keep_tree + keep_history.
        .constructor<arma::mat, arma::vec,
                     int, double, double, double, double,
                     int, bool, bool, int, bool, bool>(
            "Construct with: X (N x p matrix), y (length N), ntrees, "
            "k (prior SD scale, default 2), power (tree depth penalty, "
            "default 2), base (tree depth base, default 0.95), "
            "nu (sigma prior df, default 3), numcut (cutpoints per var, "
            "default 100), dart (DART sparsity, default FALSE), "
            "aug (DART augmentation, default FALSE), "
            "seed (RNG seed, 0 = random), "
            "keep_tree (record forest snapshots per step for "
            "predict_history; EXPENSIVE; default FALSE), "
            "keep_history (record numeric per-step buffers for trace "
            "analysis; cheap; default FALSE).")
        .method("step", (void (BartNoise::*)())    &BartNoise::step, "Run one sweep.")
        .method("step", (void (BartNoise::*)(int)) &BartNoise::step,
                "Run n Gibbs sweeps.")
        .method("get_current", &BartNoise::get_current,
                "Return the current draw as a named list with $f_bart and "
                "$sigma. The serialized BART forest is available "
                "separately via get_tree().")
        .method("get_tree",    &BartNoise::get_tree,
                "Return the serialized BART forest as a length-1 string "
                "(round-trips through set_tree()).")
        .method("set_current", &BartNoise::set_current,
                "Overwrite sigma, working response (y), and/or X from a "
                "named list. Supported keys: sigma, y, X (flattened "
                "column-major n x p). f_bart is read-only.")
        .method("set_tree",    &BartNoise::set_tree,
                "Restore the BART forest from a serialized snapshot "
                "string previously obtained from get_tree().")
        .method("predict_at",  &BartNoise::predict_at,
                "Predict at new data. Pass list(X = as.vector(X_new)) "
                "(flattened column-major). Returns a named list with "
                "$f_bart and any derived quantities. Const, no state "
                "mutation.")
        .method("get_dag",     &BartNoise::get_dag,
                "Return the predict DAG as a named list of edges.")
        .method("get_history", &BartNoise::get_history,
                "Return the history as a named list of matrices "
                "(f_bart [n_draws x N], sigma [n_draws x 1]).")
        .method("get_tree_history", &BartNoise::get_tree_history,
                "Return per-draw serialized BART forests (one per stored "
                "draw when keep_tree=TRUE; else the current forest).")
        .method("readapt_NUTS", &BartNoise::readapt_NUTS);
}
#endif

// ============================================================================
//  pybind11 module: make the class visible to Python.
// ============================================================================

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(BartNoise, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<BartNoise>(m, "BartNoise")
        .def(pybind11::init<arma::mat, arma::vec,
                            int, double, double, double, double,
                            int, bool, bool, int, bool, bool>(),
             pybind11::arg("X"),
             pybind11::arg("y"),
             pybind11::arg("ntrees")       = 200,
             pybind11::arg("k")            = 2.0,
             pybind11::arg("power")        = 2.0,
             pybind11::arg("base")         = 0.95,
             pybind11::arg("nu")           = 3.0,
             pybind11::arg("numcut")       = 100,
             pybind11::arg("dart")         = false,
             pybind11::arg("aug")          = false,
             pybind11::arg("rng_seed")     = 1,
             pybind11::arg("keep_tree")    = false,
             pybind11::arg("keep_history") = false)
        .def("step", (void (BartNoise::*)())    &BartNoise::step, "Run one sweep.")
        .def("step", (void (BartNoise::*)(int)) &BartNoise::step, pybind11::arg("n_steps"))
        .def("get_current",     &BartNoise::get_current)
        .def("get_tree",        &BartNoise::get_tree)
        .def("set_current",     &BartNoise::set_current, pybind11::arg("params"))
        .def("set_tree",        &BartNoise::set_tree, pybind11::arg("tree_s"))
        .def("predict_at",      &BartNoise::predict_at, pybind11::arg("new_data"))
        .def("get_dag",         &BartNoise::get_dag)
        .def("get_history",     &BartNoise::get_history)
        .def("get_tree_history", &BartNoise::get_tree_history)
        .def("readapt_NUTS",    &BartNoise::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false);
}
#endif

// ============================================================================
//  Standalone demo: simulate y = f(x) + noise, fit, print finite recovery.
// ============================================================================

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>

int main() {
    // ---- Simulate a well-identified DGP ---------------------------------
    //   smooth, low-dimensional mean: f(x) = sin(3 x1) + 0.5 x2^2 - x3
    //   y = f(x) + N(0, sigma_true^2),  sigma_true = 0.5
    const std::size_t N = 600;
    const std::size_t p = 3;
    const double sigma_true = 0.5;

    std::mt19937_64 gen(20260622ULL);
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::normal_distribution<double>       noise(0.0, sigma_true);

    arma::mat X(N, p);
    arma::vec y(N);
    arma::vec f_true(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double x1 = unif(gen);
        const double x2 = unif(gen);
        const double x3 = unif(gen);
        X(i, 0) = x1; X(i, 1) = x2; X(i, 2) = x3;
        const double f = std::sin(3.0 * x1) + 0.5 * x2 * x2 - x3;
        f_true[i] = f;
        y[i]      = f + noise(gen);
    }

    // ---- Fit -------------------------------------------------------------
    BartNoise model(X, y,
                    /*ntrees=*/50, /*k=*/2.0, /*power=*/2.0, /*base=*/0.95,
                    /*nu=*/3.0, /*numcut=*/100, /*dart=*/false, /*aug=*/false,
                    /*rng_seed=*/42, /*keep_tree=*/false, /*keep_history=*/false);

    const int n_burn = 500;
    const int n_keep = 500;
    model.step(n_burn);

    // Accumulate posterior-mean f and sigma over kept draws.
    arma::vec f_sum(N, arma::fill::zeros);
    double    sigma_sum = 0.0;
    for (int it = 0; it < n_keep; ++it) {
        model.step(1);
        AI4BayesCode::state_map cur = model.get_current();
        f_sum    += cur.at("f_bart");
        sigma_sum += cur.at("sigma")[0];
    }
    const arma::vec f_hat = f_sum / static_cast<double>(n_keep);
    const double sigma_hat = sigma_sum / static_cast<double>(n_keep);

    // ---- Report recovery -------------------------------------------------
    const double rmse_f = std::sqrt(arma::mean(arma::square(f_hat - f_true)));
    const double cor_f  = arma::as_scalar(arma::cor(f_hat, f_true));
    bool finite_ok = std::isfinite(sigma_hat)
                     && std::isfinite(rmse_f)
                     && std::isfinite(cor_f)
                     && f_hat.is_finite();

    std::printf("BartNoise standalone demo\n");
    std::printf("  N = %zu, p = %zu, ntrees = 50\n", N, p);
    std::printf("  sigma_true = %.4f   sigma_hat = %.4f\n",
                sigma_true, sigma_hat);
    std::printf("  RMSE(f_hat, f_true)      = %.4f\n", rmse_f);
    std::printf("  cor(f_hat, f_true)       = %.4f\n", cor_f);
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

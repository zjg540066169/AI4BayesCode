// ============================================================================
//  SoftBartNoise.cpp
//
//  Copyright (C) 2026 AI4BayesCode.
//  Licensed under GPL-2.0-or-later (inherits from softbart_block.hpp)
//
//  REFERENCE EXAMPLE for the AI4BayesCode skill -- softbart_block variant.
//
//  Model
//  -----
//      y_i | f_softbart, sigma  ~ Normal(f_softbart(x_i), sigma^2)
//      f_softbart                ~ Soft BART tree-ensemble prior
//                                  (Linero & Yang 2018, JRSSB)
//      sigma^2                   ~ InverseGamma(nu/2, nu * lambda/2)
//                                  with lambda calibrated from OLS sigest
//                                  and sigquant = 0.9 (BART::wbart default).
//                                  Sampled via NUTS on the log scale.
//
//  Block decomposition
//  -------------------
//      f_softbart  : sampled by softbart_block (one SoftBart sweep per
//                    composite step)
//      sigma       : sampled by a positive nuts_block (same kernel as
//                    every other continuous block; the SoftBart kernel
//                    also samples its own sigma internally, but we
//                    override it via softbart_block's sigma_key so the
//                    outer Gibbs stays in control).
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
//  ~5e-14 vs R) under pybind / standalone -- same calibrated lambda
//  across all three.
//
//  RNG / SEEDING
//  -------------
//  The pure-C++ SoftBart kernel draws from its OWN seedable RNG stream
//  (bart_pure_cpp/src/r_compat.h, bart_rng::engine(), a thread-local
//  std::mt19937_64) under the pybind / standalone backend. The
//  construction-time rng_seed therefore seeds the kernel stream
//  (bart_rng::set_seed) under that backend, so two SoftBartNoise objects
//  built with DIFFERENT seeds produce independent chains. Under the Rcpp
//  backend the kernel uses R's global RNG instead; R runs call set.seed()
//  before step() and the seed argument there only drives the mutable
//  predict_at RNG stream.
//
//  LICENSE WARNING
//  ---------------
//  This file #includes "AI4BayesCode/softbart_block.hpp", which in turn
//  includes the vendored SoftBart kernel. SoftBart is GPL-2.0-or-later.
//  Any binary or source distribution of this translation unit is
//  therefore subject to GPL-2.0-or-later.
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("SoftBartNoise")
//   set.seed(42); N <- 400L                       # well-identified smooth DGP
//   X <- matrix(runif(N * 3L, -1, 1), N, 3L)      # x1,x2,x3 ~ Unif(-1,1)
//   f <- sin(3 * X[,1]) + 0.5 * X[,2]^2 - X[,3]   # smooth low-dim mean
//   y <- f + rnorm(N, 0, 0.5)                      # sigma_true = 0.5
//   m <- new(SoftBartNoise, X, y, 50L, 2.0, 10.0, FALSE, 42L)
//   #          X,  y, ntrees, k, tau_rate, dart, seed
//   m$step(2000); str(m$get_current())             # $f_softbart, $sigma
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(42); N = 400        # well-identified smooth DGP
//   X = rng.uniform(-1, 1, size=(N, 3))             # x1,x2,x3 ~ Unif(-1,1)
//   f = np.sin(3 * X[:,0]) + 0.5 * X[:,1]**2 - X[:,2]   # smooth low-dim mean
//   y = f + rng.normal(0, 0.5, size=N)              # sigma_true = 0.5
//   Mod = AI4BayesCode.example("SoftBartNoise")
//   m = Mod.SoftBartNoise(X, y, 50, 2.0, 10.0, False, 42)  # X,y,ntrees,k,tau_rate,dart,seed
//   m.step(2000); print(m.get_current())            # f_softbart, sigma
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
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"

// SoftBart block (Rcpp-free, transitively pulls in the GPL SoftBart kernel)
#include "AI4BayesCode/softbart_block.hpp"

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
using AI4BayesCode::softbart_block;
using AI4BayesCode::softbart_block_config;

namespace constraints = AI4BayesCode::constraints;

// ============================================================================
//  Natural-scale log-density for sigma | y, f_softbart, nu, lambda
//  (same Inverse-Gamma sigma prior as BartNoise.cpp).
// ============================================================================

namespace {

double sigma_natural_log_density(const arma::vec& sigma_nat,
                                 const block_context& ctx,
                                 arma::vec* grad_nat) {
    const double sigma       = sigma_nat[0];
    const arma::vec& y       = ctx.at("y");
    const arma::vec& f_sb    = ctx.at("f_softbart");
    const double nu          = ctx.at("sigma_nu")[0];
    const double lambda      = ctx.at("sigma_lambda")[0];

    const double sigma2 = sigma * sigma;
    const double N      = static_cast<double>(y.n_elem);

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < y.n_elem; ++i) {
        const double r = y[i] - f_sb[i];
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

class SoftBartNoise {
public:
    SoftBartNoise(const arma::mat& X,
                  const arma::vec& y,
                  int    ntrees,
                  double k,
                  double tau_rate,
                  bool   dart,
                  int    rng_seed,
                  double nu           = 3.0,
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
          impl_(std::make_unique<composite_block>("SoftBartNoise")),
          keep_tree_(keep_tree),
          keep_history_(keep_history)
    {
        if (X.n_rows != y.n_elem) {
            ai4b::stop("SoftBartNoise: X and y must have matching row counts");
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

        // ---- Install fixed data and initial values ---------------------
        impl_->data().set("y", y);

        // For a pure SoftBart + noise model the working response IS y, so
        // we set "softbart_target" to y and never refresh it.
        impl_->data().set("softbart_target", y);

        // Store X as a flattened (column-major) vector in shared_data so
        // the predict_at DAG knows about it (the actual matrix lives in
        // softbart_block_config).
        impl_->data().set("X", arma::vec(arma::vectorise(X)));
        x_ncol_ = static_cast<std::size_t>(X.n_cols);

        impl_->data().set("sigma",      arma::vec{arma::stddev(y)});
        impl_->data().set("f_softbart", arma::vec(N, arma::fill::zeros));

        // ---- Declare the dependency DAG --------------------------------
        impl_->data().declare_dependencies(
            "f_softbart", {"softbart_target", "sigma"});
        impl_->data().declare_dependencies(
            "sigma",      {"y", "f_softbart", "sigma_nu", "sigma_lambda"});

        // ---- Declare predict-at edges (generative direction) -----------
        impl_->data().declare_data_input("X");
        impl_->data().declare_predict_edges("X",          {"f_softbart"});
        impl_->data().declare_predict_edges("f_softbart", {"y_rep"});
        impl_->data().declare_predict_edges("sigma",      {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). sigma^2 ~ IG(nu/2, nu*lambda/2):
        //      sigma_nu / sigma_lambda are sigma's prior parents. The
        //      SoftBART forest is a sampled generative parent of
        //      f_softbart alongside X. Drawn faded by ai4bayescode_plot_dag.
        impl_->data().declare_context_edges("sigma_nu",     {"sigma"});
        impl_->data().declare_context_edges("sigma_lambda", {"sigma"});
        impl_->data().declare_context_edges("SoftBART",     {"f_softbart"});

        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));

        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d, std::mt19937_64& rng) {
                const arma::vec& f = d.get("f_softbart");
                const double s     = d.get("sigma")[0];
                std::normal_distribution<double> norm(0.0, 1.0);
                arma::vec y_rep(f.n_elem);
                for (std::size_t i = 0; i < f.n_elem; ++i) {
                    y_rep[i] = f[i] + s * norm(rng);
                }
                return y_rep;
            });

        // ---- Add the SoftBart block ------------------------------------
        softbart_block_config sb_cfg;
        sb_cfg.name                 = "f_softbart";
        sb_cfg.x_train              = X;
        sb_cfg.y_init               = y;
        sb_cfg.working_response_key = "softbart_target";
        sb_cfg.sigma_key            = "sigma";
        sb_cfg.ntrees               = ntrees;
        sb_cfg.k                    = k;
        sb_cfg.tau_rate             = tau_rate;
        sb_cfg.dart                 = dart;
        sb_cfg.center_Y             = true;
        impl_->add_child(
            std::make_unique<softbart_block>(std::move(sb_cfg)));

        // ---- Calibrate sigma prior (BART::wbart sigquant=0.9 default) --
        {
            auto& sb_blk = dynamic_cast<softbart_block&>(impl_->child(0));
            const double sigest = sb_blk.current_sigma();
            // SoftBart's initial sigma may be 0 before any step; if so,
            // fall back to sd(y).
            const double sigest_eff =
                (sigest > 0.0) ? sigest : arma::stddev(y);
            impl_->data().set("sigma", arma::vec{sigest_eff});

            // R::qchisq resolves to R's math library under the Rcpp
            // backend and to the kernel's namespace R { qchisq } (from
            // bart_pure_cpp/src/r_compat.h, ~5e-14 vs R) under
            // pybind / standalone. Same calibrated lambda in all three.
            const double qchi   = R::qchisq(0.1, nu, /*lower.tail=*/1,
                                            /*log.p=*/0);
            const double lambda = sigest_eff * sigest_eff * qchi / nu;
            impl_->data().set("sigma_nu",     arma::vec{nu});
            impl_->data().set("sigma_lambda", arma::vec{lambda});
        }

        // ---- Add the sigma block (positive NUTS) -----------------------
        const double sigest_for_nuts =
            dynamic_cast<softbart_block&>(impl_->child(0)).current_sigma();
        const double sigest_log_init =
            std::log(sigest_for_nuts > 0.0
                       ? sigest_for_nuts
                       : arma::stddev(y));
        nuts_block_config sg_cfg;
        sg_cfg.name        = "sigma";
        sg_cfg.initial_unc = arma::vec{sigest_log_init};
        sg_cfg.constrain   = constraints::positive::constrain;
        sg_cfg.unconstrain = constraints::positive::unconstrain;
        sg_cfg.log_density_grad =
            [](const arma::vec& theta_unc, const block_context& ctx,
               arma::vec* grad) {
                return constraints::positive::wrap(
                    theta_unc, grad,
                    [&](const arma::vec& sigma_nat, arma::vec* grad_nat) {
                        return sigma_natural_log_density(
                            sigma_nat, ctx, grad_nat);
                    });
            };
        sg_cfg.nuts_settings.nuts_settings.max_tree_depth     = 6;
        sg_cfg.nuts_settings.nuts_settings.target_accept_rate = 0.8;
        impl_->add_child(
            std::make_unique<nuts_block>(std::move(sg_cfg)));

        // ---- Enable history recording if requested ---------------------
        // keep_tree (SoftBart forest snapshots, expensive) and keep_history
        // (numeric per-step buffers, cheap) are independent toggles.
        if (keep_tree_)    impl_->set_keep_tree(true);
        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Current draw: f_softbart + sigma. (The serialized SoftBart forest
    // snapshot stays reachable via get_tree(); it is not a numeric state
    // vector so it lives outside the state_map.)
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["f_softbart"] = impl_->data().get("f_softbart");
        out["sigma"]      = impl_->data().get("sigma");
        return out;
    }

    // Serialized SoftBart forest snapshot (round-trips through set_tree()).
    // Kept as a separate passthrough so the forest state stays reachable
    // from both backends without putting a string into the numeric
    // state_map.
    std::string get_tree() const {
        auto& sb_child = dynamic_cast<softbart_block&>(impl_->child(0));
        return sb_child.get_tree();
    }

    // Update parameters / data inputs from an outer sampler.
    // Supported keys (any subset; order independent):
    //   sigma  : scalar > 0. Overwrites the current noise SD.
    //   y      : length-n vector. Pushed into softbart_block as the new
    //            working response for subsequent sweeps.
    //   X      : FLATTENED column-major n x p matrix (length n*p).
    //            Replaces the design matrix. Row/col count must match the
    //            construction-time dimensions.
    //   f_softbart : SILENTLY IGNORED -- derived from trees (no unique
    //            inverse). Use the tree round-trip (get_tree / set_tree)
    //            instead.
    // Unknown keys are silently ignored per system_design.md §7 so that
    // set_current(get_current()) round-trips cleanly.
    void set_current(const AI4BayesCode::state_map& params) {
        auto* sb_blk = dynamic_cast<softbart_block*>(&impl_->child(0));
        auto* sg_blk = dynamic_cast<nuts_block*>(&impl_->child(1));

        // f_softbart is read-only output; silently ignored on input so
        // that round-trip set_current(get_current()) is supported per
        // system_design.md §7 / §16. Use set_tree() to restore the forest.

        const auto it_X = params.find("X");
        const auto it_y = params.find("y");
        const bool has_X = it_X != params.end();
        const bool has_y = it_y != params.end();

        // Reshape flattened X (column-major) back into an n x p matrix.
        auto reshape_X = [&](const arma::vec& x_flat) -> arma::mat {
            const std::size_t p = x_ncol_;
            if (p == 0 || (x_flat.n_elem % p) != 0) {
                ai4b::stop("SoftBartNoise::set_current: flattened X length "
                           "%d is not a multiple of p = %d",
                           (int)x_flat.n_elem, (int)p);
            }
            return arma::reshape(x_flat, x_flat.n_elem / p, p);
        };

        if (has_X && has_y) {
            arma::mat X_new = reshape_X(it_X->second);
            const arma::vec& y_new = it_y->second;
            sb_blk->set_data(X_new, y_new);
            impl_->data().set("softbart_target", y_new);
            impl_->data().set("y", y_new);
            impl_->data().set("X", arma::vec(arma::vectorise(X_new)));
        } else if (has_X) {
            arma::mat X_new = reshape_X(it_X->second);
            sb_blk->set_X(X_new);
            impl_->data().set("X", arma::vec(arma::vectorise(X_new)));
        } else if (has_y) {
            const arma::vec& y_new = it_y->second;
            sb_blk->set_Y(y_new);
            impl_->data().set("softbart_target", y_new);
            impl_->data().set("y", y_new);
        }

        const auto it_s = params.find("sigma");
        if (it_s != params.end()) {
            const double s = it_s->second[0];
            if (!(s > 0.0)) ai4b::stop("sigma must be strictly positive");
            sg_blk->set_current(arma::vec{s});
            impl_->data().set("sigma", arma::vec{s});
        }
    }

    // Restore the full SoftBart forest from a serialized snapshot
    // previously obtained from get_tree(). Separate from set_current
    // because a tree is a std::string, not a numeric state vector.
    // Refreshes shared_data f_softbart so downstream blocks see the
    // restored forest immediately.
    void set_tree(const std::string& tree_s) {
        auto* sb_blk = dynamic_cast<softbart_block*>(&impl_->child(0));
        sb_blk->set_tree(tree_s);
        impl_->data().set("f_softbart", sb_blk->current());
    }

    // Predict at new data. Takes a state_map with optional key "X".
    //
    // Behavior depends on construction-time keep_history flag and whether
    // X was supplied (mirrors BartNoise::predict_at):
    //   * keep_history = FALSE, X supplied:
    //       f_softbart: length N_new, from current trees.
    //       y_rep:      posterior-predictive at X_new from CURRENT draw.
    //   * keep_history = FALSE, empty map:
    //       y_rep:      posterior-predictive at TRAINING X from current draw.
    //   * keep_history = TRUE,  X supplied:
    //       f_softbart: n_draws x N_new from each stored draw's trees.
    //       y_rep:      n_draws x N_new posterior-predictive samples.
    //   * keep_history = TRUE,  empty map:
    //       y_rep:      n_draws x N_train posterior-predictive at training X.
    //
    // X is passed as a FLATTENED column-major vector under key "X"
    // (length N_new * p); reshaped here back into an N_new x p matrix.
    // Uses predict_rng_ (const-preserving, persistent, seeded at
    // construction). Does NOT modify MCMC state in any mode.
    AI4BayesCode::history_map
    predict_at(const AI4BayesCode::state_map& new_data) const {
        // Validate keys.
        const bool has_X = new_data.find("X") != new_data.end();
        for (const auto& kv : new_data) {
            if (kv.first != "X") {
                ai4b::stop("SoftBartNoise::predict_at: unknown key '%s'. "
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
                ai4b::stop("SoftBartNoise::predict_at: flattened X length %d "
                           "is not a multiple of p = %d",
                           (int)x_flat.n_elem, (int)x_ncol_);
            }
            const std::size_t n_new = x_flat.n_elem / x_ncol_;
            X_new = arma::reshape(x_flat, n_new, x_ncol_);
        }

        auto& sb_child = dynamic_cast<softbart_block&>(impl_->child(0));

        if (!keep_history_) {
            // ---- Stateful mode -----------------------------------------
            block_context replaced;
            arma::vec f_pred;           // only set if has_X
            if (has_X) {
                f_pred = sb_child.predict(X_new);
                replaced["X"]          = arma::vec(arma::vectorise(X_new));
                replaced["f_softbart"] = f_pred;
            }
            block_context result = impl_->predict_at(replaced, predict_rng_);

            AI4BayesCode::history_map out;
            if (has_X) {
                arma::mat fm(1, f_pred.n_elem);
                for (std::size_t i = 0; i < f_pred.n_elem; ++i) fm(0, i) = f_pred[i];
                out.emplace("f_softbart", std::move(fm));
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

        arma::mat F_pred;
        arma::vec x_flat;
        bool echo_f = has_X;
        if (has_X) {
            F_pred = sb_child.predict_history(X_new);
            x_flat = arma::vec(arma::vectorise(X_new));
        } else {
            AI4BayesCode::history_map sb_hist_list = sb_child.get_history();
            F_pred = sb_hist_list.at("f_softbart");
        }

        if (n_draws != F_pred.n_rows) {
            ai4b::stop("SoftBartNoise::predict_at: inconsistent history "
                       "sizes (softbart=%d, sigma=%d)",
                       (int)F_pred.n_rows, (int)n_draws);
        }

        std::unordered_map<std::string, std::vector<arma::vec>> collected;

        for (std::size_t d = 0; d < n_draws; ++d) {
            block_context replaced;
            if (has_X) replaced["X"] = x_flat;
            replaced["f_softbart"] = F_pred.row(d).t();
            replaced["sigma"]      = arma::vec{sigma_hist(d, 0)};

            block_context result = impl_->predict_at(replaced, predict_rng_);

            for (const auto& kv : result) {
                collected[kv.first].push_back(kv.second);
            }
            if (echo_f) {
                collected["f_softbart"].push_back(F_pred.row(d).t());
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

    // ---- History access --------------------------------------------------
    // The composite's neutral history_map (numeric matrices keyed by block
    // name: f_softbart [n_draws x N], sigma [n_draws x 1]). The per-draw
    // serialized SoftBart forests are reachable separately via
    // get_tree_history() (a vector<string>, outside the numeric map).
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    // Per-draw serialized SoftBart forest snapshots. In stateful mode this
    // is length 1 (the current forest); in history mode (keep_tree=true) it
    // has one entry per stored draw.
    std::vector<std::string> get_tree_history() const {
        auto& sb_child = dynamic_cast<softbart_block&>(impl_->child(0));
        return sb_child.get_tree_history();
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
RCPP_MODULE(SoftBartNoise_module) {
    Rcpp::class_<SoftBartNoise>("SoftBartNoise")
        // Short constructor: nu=3, keep_tree=FALSE, keep_history=FALSE.
        .constructor<arma::mat, arma::vec,
                     int, double, double, bool, int>(
            "Construct (short): X (N x p), y (length N), ntrees, k "
            "(prior SD scale, default 2), tau_rate (bandwidth Gamma "
            "rate, default 10), dart (DART sparsity, default FALSE), "
            "seed (RNG seed, 0 = random). nu defaults to 3, keep_tree "
            "and keep_history default FALSE.")
        // Full constructor: nu + keep_tree + keep_history.
        .constructor<arma::mat, arma::vec,
                     int, double, double, bool, int, double, bool, bool>(
            "Construct (full): X, y, ntrees, k, tau_rate, dart, seed, "
            "nu (sigma prior df, default 3), keep_tree (record forest "
            "snapshots per step for predict_history; EXPENSIVE; default "
            "FALSE), keep_history (record numeric per-step buffers; "
            "cheap; default FALSE).")
        .method("step", (void (SoftBartNoise::*)())    &SoftBartNoise::step, "Run one sweep.")
        .method("step", (void (SoftBartNoise::*)(int)) &SoftBartNoise::step,
                "Run n Gibbs sweeps.")
        .method("get_current", &SoftBartNoise::get_current,
                "Return the current draw as a named list with $f_softbart "
                "and $sigma. The serialized SoftBart forest is available "
                "separately via get_tree().")
        .method("get_tree",    &SoftBartNoise::get_tree,
                "Return the serialized SoftBart forest as a length-1 "
                "string (round-trips through set_tree()).")
        .method("set_current", &SoftBartNoise::set_current,
                "Overwrite sigma, working response (y), and/or X from a "
                "named list. Supported keys: sigma, y, X (flattened "
                "column-major n x p). f_softbart is read-only.")
        .method("set_tree",    &SoftBartNoise::set_tree,
                "Restore the SoftBart forest from a serialized snapshot "
                "string previously obtained from get_tree().")
        .method("predict_at",  &SoftBartNoise::predict_at,
                "Predict at new data. Pass list(X = as.vector(X_new)) "
                "(flattened column-major) or list(). Returns a named list "
                "with $f_softbart and any derived quantities. Const, no "
                "state mutation.")
        .method("get_dag",     &SoftBartNoise::get_dag,
                "Return the predict DAG as a named list of edges.")
        .method("get_history", &SoftBartNoise::get_history,
                "Return the history as a named list of matrices "
                "(f_softbart [n_draws x N], sigma [n_draws x 1]).")
        .method("get_tree_history", &SoftBartNoise::get_tree_history,
                "Return per-draw serialized SoftBart forests (one per "
                "stored draw when keep_tree=TRUE; else the current forest).")
        .method("readapt_NUTS", &SoftBartNoise::readapt_NUTS);
}
#endif

// ============================================================================
//  pybind11 module: make the class visible to Python.
// ============================================================================

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(SoftBartNoise, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<SoftBartNoise>(m, "SoftBartNoise")
        .def(pybind11::init<arma::mat, arma::vec,
                            int, double, double, bool, int,
                            double, bool, bool>(),
             pybind11::arg("X"),
             pybind11::arg("y"),
             pybind11::arg("ntrees")       = 50,
             pybind11::arg("k")            = 2.0,
             pybind11::arg("tau_rate")     = 10.0,
             pybind11::arg("dart")         = false,
             pybind11::arg("rng_seed")     = 1,
             pybind11::arg("nu")           = 3.0,
             pybind11::arg("keep_tree")    = false,
             pybind11::arg("keep_history") = false)
        .def("step", (void (SoftBartNoise::*)())    &SoftBartNoise::step, "Run one sweep.")
        .def("step", (void (SoftBartNoise::*)(int)) &SoftBartNoise::step, pybind11::arg("n_steps"))
        .def("get_current",     &SoftBartNoise::get_current)
        .def("get_tree",        &SoftBartNoise::get_tree)
        .def("set_current",     &SoftBartNoise::set_current, pybind11::arg("params"))
        .def("set_tree",        &SoftBartNoise::set_tree, pybind11::arg("tree_s"))
        .def("predict_at",      &SoftBartNoise::predict_at, pybind11::arg("new_data"))
        .def("get_dag",         &SoftBartNoise::get_dag)
        .def("get_history",     &SoftBartNoise::get_history)
        .def("get_tree_history", &SoftBartNoise::get_tree_history)
        .def("readapt_NUTS",    &SoftBartNoise::readapt_NUTS,
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
    const std::size_t N = 400;
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
    SoftBartNoise model(X, y,
                        /*ntrees=*/50, /*k=*/2.0, /*tau_rate=*/10.0,
                        /*dart=*/false, /*rng_seed=*/42, /*nu=*/3.0,
                        /*keep_tree=*/false, /*keep_history=*/false);

    // SoftBart converges more weakly on short chains than hard BART, so
    // give it a longer warmup + keep window in the standalone demo.
    const int n_burn = 1000;
    const int n_keep = 1000;
    model.step(n_burn);

    // Accumulate posterior-mean f and sigma over kept draws.
    arma::vec f_sum(N, arma::fill::zeros);
    double    sigma_sum = 0.0;
    for (int it = 0; it < n_keep; ++it) {
        model.step(1);
        AI4BayesCode::state_map cur = model.get_current();
        f_sum     += cur.at("f_softbart");
        sigma_sum += cur.at("sigma")[0];
    }
    const arma::vec f_hat   = f_sum / static_cast<double>(n_keep);
    const double sigma_hat  = sigma_sum / static_cast<double>(n_keep);

    // ---- Report recovery -------------------------------------------------
    const double rmse_f = std::sqrt(arma::mean(arma::square(f_hat - f_true)));
    const double cor_f  = arma::as_scalar(arma::cor(f_hat, f_true));
    bool finite_ok = std::isfinite(sigma_hat)
                     && std::isfinite(rmse_f)
                     && std::isfinite(cor_f)
                     && f_hat.is_finite();

    std::printf("SoftBartNoise standalone demo\n");
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

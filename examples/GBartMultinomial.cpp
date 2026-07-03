// ============================================================================
//  GBartMultinomial.cpp
//
//  Copyright (C) 2026 AI4BayesCode.
//  Licensed under GPL-2.0-or-later (inherits from genbart_block.hpp).
//
//  REFERENCE TEMPLATE for multinomial logistic BART via the
//  Baker 1994 / Forster 2010 Poisson-multinomial gamma augmentation
//  applied to genBART (Linero 2022 RJMCMC tree kernel) in the C-1
//  identified reference-category parameterization (architecture
//  originally proposed by Murray 2021 Sec 3.1; implemented here on
//  genBART's pluggable-likelihood RJMCMC engine).
//
//  Model
//  -----
//      y_i | r ~ Categorical(pi_0(x_i), pi_1(x_i), ..., pi_{C-1}(x_i))
//      pi_0(x)    = 1      / (1 + sum_{l=1..C-1} exp(r_l(x)))    [ref]
//      pi_j(x>=1) = exp(r_j(x)) / (1 + sum_{l=1..C-1} exp(r_l(x)))
//      r_j        ~ independent generalized-BART priors
//
//  Reference-category identification: f^(0) := 1 (so r_0 := 0); log-
//  odds of class j vs reference = r_j(x). This is mathematically
//  equivalent to Murray 2021's unidentified C-tree form for
//  inference on identified quantities (probabilities, odds ratios)
//  while being faster (C-1 tree ensembles instead of C), more
//  interpretable per-parameter, and safer against cross-chain
//  unidentification R-hat pathology.
//
//  Gamma augmentation (see poisson_multinomial_aug_block.hpp header
//  for full derivation and attribution):
//      phi_i | y, r  ~  Gamma(n_i, 1 + sum_{l=1..C-1} exp(r_l(x_i)))
//  For n_i = 1 (common case): phi_i ~ Exp(1 + sum exp(r_l)).
//
//  Given phi_i, the augmented likelihood factorises into C-1
//  conditionally independent Poisson-like likelihoods per observation:
//      genbart_block_j with lik = poisson_lik,
//                           y_key = "u_j" = [y == j],
//                           offset_key = "log_phi_aug".
//
//  Block decomposition
//  -------------------
//      log_phi_aug (alias "phi_aug")  : poisson_multinomial_aug_block
//                                       (writes log_phi + u_1..u_{C-1})
//      r_1  : genbart_block(poisson_lik, offset = log_phi)
//      r_2  : genbart_block(poisson_lik, offset = log_phi)
//      ...
//      r_{C-1} : genbart_block(poisson_lik, offset = log_phi)
//  Gibbs sweep order: aug block first, then each genbart child.
//
//  Predict DAG
//  -----------
//      X     -> r_j for each j = 1..C-1
//      r_1, ..., r_{C-1}  -> probs  (softmax with f^(0) := 1)
//      probs -> y_rep (Categorical)
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
//   ai4bayescode_example("GBartMultinomial")
//   set.seed(42); N <- 900L                       # well-identified C=3 DGP
//   X <- matrix(runif(N * 3L, -1.5, 1.5), N, 3L)  # x1,x2,x3 ~ Unif(-1.5,1.5)
//   r1 <-  1.2 * X[,1] - 0.8 * X[,2]              # class 1 vs ref log-odds
//   r2 <- -1.0 * X[,1] + 1.5 * X[,3]              # class 2 vs ref log-odds
//   E  <- cbind(1, exp(r1), exp(r2)); P <- E / rowSums(E)   # softmax(0,r1,r2)
//   y  <- apply(P, 1L, function(p) sample.int(3L, 1L, prob = p)) - 1L  # 0..2
//   m  <- new(GBartMultinomial, X, as.numeric(y), 3L, 50L, 42L, TRUE)
//   #          X,    y,         C,  ntrees, seed, keep_history
//   m$step(2000); str(m$get_current())            # $r, $probs, $log_phi
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(42); N = 900       # well-identified C=3 DGP
//   X  = rng.uniform(-1.5, 1.5, size=(N, 3))        # x1,x2,x3 ~ Unif(-1.5,1.5)
//   r1 =  1.2 * X[:,0] - 0.8 * X[:,1]; r2 = -1.0 * X[:,0] + 1.5 * X[:,2]
//   E  = np.column_stack([np.ones(N), np.exp(r1), np.exp(r2)])  # softmax(0,r1,r2)
//   P  = E / E.sum(1, keepdims=True)
//   y  = np.array([rng.choice(3, p=P[i]) for i in range(N)], float)  # 0..2
//   Mod = AI4BayesCode.example("GBartMultinomial")
//   m = Mod.GBartMultinomial(X, y, 3, 50, 42, False, True)  # X,y,C,ntrees,seed,keep_tree,keep_history
//   m.step(2000); print(m.get_current())            # r, probs, log_phi
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

// Augmentation + genBART blocks (Rcpp-free; transitively pull in the
// GPL genBART kernel).
#include "AI4BayesCode/poisson_multinomial_aug_block.hpp"
#include "AI4BayesCode/genbart_block.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::genbart_block;
using AI4BayesCode::genbart_block_config;
using AI4BayesCode::poisson_multinomial_aug_block;
using AI4BayesCode::poisson_multinomial_aug_block_config;

namespace {

inline std::string r_key(std::size_t j) {
    return std::string("r_") + std::to_string(j);
}
inline std::string u_key(std::size_t j) {
    return std::string("u_") + std::to_string(j);
}

// probs[i*C + j] = exp(r_j - max) / sum with f^(0) = 1 (r_0 := 0).
inline void softmax_with_ref(std::size_t N, std::size_t C,
                             const std::vector<const arma::vec*>& r_ptrs,
                             arma::vec& probs) {
    probs.set_size(N * C);
    for (std::size_t i = 0; i < N; ++i) {
        double max_r = 0.0;
        for (std::size_t j = 0; j < C - 1; ++j) {
            const double rj = (*r_ptrs[j])[i];
            if (rj > max_r) max_r = rj;
        }
        double denom = std::exp(0.0 - max_r);        // ref f^(0) = 1
        for (std::size_t j = 0; j < C - 1; ++j) {
            denom += std::exp((*r_ptrs[j])[i] - max_r);
        }
        probs[i * C + 0] = std::exp(0.0 - max_r) / denom;
        for (std::size_t j = 0; j < C - 1; ++j) {
            probs[i * C + (j + 1)] =
                std::exp((*r_ptrs[j])[i] - max_r) / denom;
        }
    }
}

// Sentinel separating the C-1 per-class serialized forests inside the
// single get_tree()/set_tree() round-trip string.
const char* const kTreeSep = "\n===GBARTCLASS===\n";

} // anonymous namespace

// ============================================================================
//  User-facing class. Backend-neutral signatures: arma containers,
//  state_map / history_map, ai4b::stop.
// ============================================================================

class GBartMultinomial {
public:
    GBartMultinomial(const arma::mat& X,
                     const arma::vec& y,
                     int    C,
                     int    ntrees,
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
          impl_(std::make_unique<composite_block>("GBartMultinomial")),
          C_(static_cast<std::size_t>(C)),
          keep_tree_(keep_tree),
          keep_history_(keep_history)
    {
        if (C < 2) {
            ai4b::stop("GBartMultinomial: C must be >= 2 "
                       "(for C = 2 you may prefer GBartLogistic).");
        }
        if (X.n_rows != y.n_elem) {
            ai4b::stop("GBartMultinomial: X and y must have matching "
                       "row counts");
        }
        for (std::size_t i = 0; i < y.n_elem; ++i) {
            const double yi = y[i];
            if (!(yi >= 0.0) || !(yi <= static_cast<double>(C - 1))
                || yi != std::floor(yi))
            {
                ai4b::stop("GBartMultinomial: y[%d] = %g; must be "
                           "integer in {0, ..., %d}", (int)i, yi, C - 1);
            }
        }
        if (ntrees <= 0) {
            ai4b::stop("GBartMultinomial: ntrees must be > 0");
        }

        const std::size_t N = static_cast<std::size_t>(X.n_rows);
        x_ncol_ = static_cast<std::size_t>(X.n_cols);

        // ---- Install fixed data and initial parameter values --------
        impl_->data().set("y", y);

        // Store X as a flattened (column-major) vector in shared_data so
        // the predict_at DAG knows about it (the actual matrix lives in
        // each genbart_block_config).
        impl_->data().set("X", arma::vec(arma::vectorise(X)));

        for (std::size_t j = 0; j < C_ - 1; ++j) {
            impl_->data().set(r_key(j + 1), arma::vec(N, arma::fill::zeros));
        }
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            arma::vec uj(N);
            const double target = static_cast<double>(j + 1);
            for (std::size_t i = 0; i < N; ++i) {
                uj[i] = (y[i] == target) ? 1.0 : 0.0;
            }
            impl_->data().set(u_key(j + 1), uj);
        }
        impl_->data().set("log_phi_aug", arma::vec(N, arma::fill::zeros));

        impl_->data().set("probs",
            arma::vec(N * C_, arma::fill::value(1.0 / C_)));
        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));

        // ---- Gibbs DAG ----------------------------------------------
        std::vector<std::string> aug_deps = { "y" };
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            aug_deps.push_back(r_key(j + 1));
        }
        impl_->data().declare_dependencies("log_phi_aug", aug_deps);

        for (std::size_t j = 0; j < C_ - 1; ++j) {
            impl_->data().declare_dependencies(
                r_key(j + 1),
                { u_key(j + 1), "log_phi_aug" });
        }

        // ---- Predict-at data inputs ---------------------------------
        impl_->data().declare_data_input("X");

        // ---- Predict DAG -------------------------------------------
        std::vector<std::string> r_all;
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            r_all.push_back(r_key(j + 1));
        }
        impl_->data().declare_predict_edges("X", r_all);
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            impl_->data().declare_predict_edges(r_key(j + 1), {"probs"});
        }
        impl_->data().declare_predict_edges("probs", { "y_rep" });

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). Each r_j ~ independent
        //      generalized-BART prior: the shared genBART forest is a
        //      sampled generative parent of every r_j alongside X. Drawn
        //      faded by ai4bayescode_plot_dag.
        impl_->data().declare_context_edges("genBART", r_all);

        // Deterministic refresher: probs = softmax(0, r_1, ..., r_{C-1}).
        const std::size_t C_cap = C_;
        impl_->data().register_refresher(
            "probs",
            [C_cap](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                std::vector<const arma::vec*> ptrs(C_cap - 1);
                for (std::size_t j = 0; j < C_cap - 1; ++j) {
                    ptrs[j] = &d.get(std::string("r_") +
                                     std::to_string(j + 1));
                }
                arma::vec probs;
                const std::size_t N_here = ptrs[0]->n_elem;
                softmax_with_ref(N_here, C_cap, ptrs, probs);
                return probs;
            });

        // Stochastic refresher: y_rep[i] ~ Categorical(probs[i, :]).
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [C_cap](const AI4BayesCode::shared_data_t& d,
                    std::mt19937_64& rng) {
                const arma::vec& probs = d.get("probs");
                const std::size_t N_here = probs.n_elem / C_cap;
                std::uniform_real_distribution<double> unif(0.0, 1.0);
                arma::vec yrep(N_here);
                for (std::size_t i = 0; i < N_here; ++i) {
                    const double u = unif(rng);
                    double acc = 0.0;
                    std::size_t chosen = C_cap - 1;
                    for (std::size_t j = 0; j < C_cap; ++j) {
                        acc += probs[i * C_cap + j];
                        if (u < acc) { chosen = j; break; }
                    }
                    yrep[i] = static_cast<double>(chosen);
                }
                return yrep;
            });

        // ---- Add aug block (FIRST) ---------------------------------
        poisson_multinomial_aug_block_config aug_cfg;
        aug_cfg.name  = "log_phi_aug";
        aug_cfg.N     = N;
        aug_cfg.C     = C_;
        aug_cfg.y_key = "y";
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            aug_cfg.r_keys.push_back(r_key(j + 1));
            aug_cfg.u_keys.push_back(u_key(j + 1));
        }
        aug_cfg.initial_y = y;
        impl_->add_child(std::make_unique<poisson_multinomial_aug_block>(
            std::move(aug_cfg)));

        // ---- Add C-1 genbart blocks --------------------------------
        // Each needs its own likelihood instance (poisson_lik has no
        // per-class state but owning separate instances keeps lifetimes
        // explicit).
        lik_pool_.reserve(C_ - 1);
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            lik_pool_.push_back(
                std::make_unique<genbart::lik::poisson_lik>());

            genbart_block_config bart_cfg;
            bart_cfg.name        = r_key(j + 1);
            bart_cfg.x_train     = X;
            // y_init for genbart = indicator u_j (read via y_key at
            // each sweep from shared_data).
            arma::vec uj_init(N);
            const double target = static_cast<double>(j + 1);
            for (std::size_t i = 0; i < N; ++i) {
                uj_init[i] = (y[i] == target) ? 1.0 : 0.0;
            }
            bart_cfg.y_init      = uj_init;
            bart_cfg.offset_init = arma::vec(N, arma::fill::zeros);
            bart_cfg.lik         = lik_pool_.back().get();
            bart_cfg.ntrees      = static_cast<std::size_t>(ntrees);
            bart_cfg.y_key       = u_key(j + 1);
            bart_cfg.offset_key  = "log_phi_aug";

            impl_->add_child(
                std::make_unique<genbart_block>(std::move(bart_cfg)));
        }

        if (keep_tree_)    impl_->set_keep_tree(true);
        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Current draw. Backend-neutral state_map:
    //   r       : flattened column-major N x (C-1) matrix of r_j at the
    //             training inputs (length N*(C-1)).
    //   probs   : flattened per-observation N x C probabilities
    //             (probs[i*C + j], col 0 = reference) -- same layout the
    //             refresher / predict_at use.
    //   log_phi : length-N log augmentation variable.
    // The per-class serialized BART forests are NOT numeric state; they
    // are reachable separately via get_tree() (a single string holding
    // all C-1 forests, separated by a sentinel) round-tripping through
    // set_tree().
    AI4BayesCode::state_map get_current() const {
        const std::size_t N = impl_->data().get("y").n_elem;

        // r flattened column-major (N x (C-1)).
        arma::vec r_flat(N * (C_ - 1));
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            const arma::vec& rj = impl_->data().get(r_key(j + 1));
            for (std::size_t i = 0; i < N; ++i) {
                r_flat[j * N + i] = rj[i];
            }
        }

        std::vector<const arma::vec*> ptrs(C_ - 1);
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            ptrs[j] = &impl_->data().get(r_key(j + 1));
        }
        arma::vec probs;
        softmax_with_ref(N, C_, ptrs, probs);

        AI4BayesCode::state_map out;
        out["r"]       = std::move(r_flat);
        out["probs"]   = std::move(probs);
        out["log_phi"] = impl_->data().get("log_phi_aug");
        return out;
    }

    // Serialized per-class BART forests, concatenated into ONE string
    // (the C-1 forests joined by a sentinel). Round-trips through
    // set_tree(...). Kept as a separate passthrough so the forest state
    // stays reachable from both backends without putting strings into the
    // numeric state_map.
    std::string get_tree() const {
        std::ostringstream os;
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            auto& g = dynamic_cast<genbart_block&>(impl_->child(1 + j));
            if (j > 0) os << kTreeSep;
            os << g.get_tree();
        }
        return os.str();
    }

    // Restore the C-1 BART forests from a single serialized snapshot
    // previously obtained from get_tree(). Refreshes shared_data r_j so
    // downstream blocks see the restored forests immediately.
    void set_tree(const std::string& tree_s) {
        // Split on the sentinel.
        std::vector<std::string> parts;
        const std::string sep = kTreeSep;
        std::size_t pos = 0;
        while (true) {
            std::size_t nxt = tree_s.find(sep, pos);
            if (nxt == std::string::npos) {
                parts.push_back(tree_s.substr(pos));
                break;
            }
            parts.push_back(tree_s.substr(pos, nxt - pos));
            pos = nxt + sep.size();
        }
        if (parts.size() != C_ - 1) {
            ai4b::stop("GBartMultinomial::set_tree: snapshot holds %d "
                       "forests but C - 1 = %d",
                       (int)parts.size(), (int)(C_ - 1));
        }
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            auto* g = dynamic_cast<genbart_block*>(&impl_->child(1 + j));
            g->set_tree(parts[j]);
            impl_->data().set(r_key(j + 1), g->current());
        }
    }

    // Update parameters / data inputs from an outer sampler. Supported
    // keys (any subset; order independent):
    //   y : length-N vector in {0,...,C-1}. Recomputes the u_j indicators
    //       and pushes them into the aug block and each genbart child.
    //   X : FLATTENED column-major N x p matrix (length N*p). Replaces the
    //       design matrix for all C-1 forests (row + column count must
    //       match construction).
    // r / probs / log_phi are read-only outputs; silently ignored on
    // input so that round-trip set_current(get_current()) is supported per
    // system_design.md §7 / §16. Use set_tree(...) for forest restoration.
    void set_current(const AI4BayesCode::state_map& params) {
        auto* aug_blk =
            dynamic_cast<poisson_multinomial_aug_block*>(&impl_->child(0));

        const auto it_y = params.find("y");
        const auto it_X = params.find("X");
        const bool has_y = it_y != params.end();
        const bool has_X = it_X != params.end();

        if (has_y) {
            const arma::vec& y_new = it_y->second;
            const std::size_t N =
                static_cast<std::size_t>(impl_->data().get("y").n_elem);
            if (y_new.n_elem != N) {
                ai4b::stop("GBartMultinomial::set_current: y length %d "
                           "does not match N = %d",
                           (int)y_new.n_elem, (int)N);
            }
            aug_blk->set_y(y_new);
            impl_->data().set("y", y_new);
            // Recompute u_j indicators and push to each genbart_block's Y.
            for (std::size_t j = 0; j < C_ - 1; ++j) {
                arma::vec uj(y_new.n_elem);
                const double target = static_cast<double>(j + 1);
                for (std::size_t i = 0; i < y_new.n_elem; ++i) {
                    uj[i] = (y_new[i] == target) ? 1.0 : 0.0;
                }
                impl_->data().set(u_key(j + 1), uj);
                auto* g = dynamic_cast<genbart_block*>(&impl_->child(1 + j));
                g->set_Y(uj);
            }
        }

        if (has_X) {
            const arma::vec& x_flat = it_X->second;
            if (x_ncol_ == 0 || (x_flat.n_elem % x_ncol_) != 0) {
                ai4b::stop("GBartMultinomial::set_current: flattened X "
                           "length %d is not a multiple of p = %d",
                           (int)x_flat.n_elem, (int)x_ncol_);
            }
            const std::size_t n_new = x_flat.n_elem / x_ncol_;
            arma::mat X_new = arma::reshape(x_flat, n_new, x_ncol_);
            for (std::size_t j = 0; j < C_ - 1; ++j) {
                auto* g = dynamic_cast<genbart_block*>(&impl_->child(1 + j));
                g->set_X(X_new);
            }
            impl_->data().set("X", arma::vec(arma::vectorise(X_new)));
        }
    }

    // Full model DAG: $gibbs_reads, $gibbs_invalidates, $predict_edges,
    // $data_inputs. See composite_block::get_dag() for the semantics.
    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    // Predict at new data. Takes a state_map with optional key "X"
    // (flattened column-major N_new x p matrix). Returns a history_map.
    //
    //   * keep_history = FALSE, X supplied:
    //       r     : 1 x (N_new*(C-1)) flattened column-major r_j(X_new)
    //       probs : 1 x (N_new*C)     flattened-per-observation probs
    //   * keep_history = FALSE, empty map:
    //       probs : 1 x (N_train*C) at training X from the current draw
    //       y_rep : 1 x N_train posterior-predictive draw
    //   * keep_history = TRUE, empty map:
    //       y_rep : n_draws x N_train posterior-predictive samples, drawn
    //               from each stored draw's softmaxed r history.
    //   New-X prediction in history mode is not implemented.
    //
    // Uses predict_rng_ (const-preserving, persistent, seeded at
    // construction). Does NOT modify MCMC state in any mode.
    AI4BayesCode::history_map
    predict_at(const AI4BayesCode::state_map& new_data) const {
        const bool has_X = new_data.find("X") != new_data.end();
        for (const auto& kv : new_data) {
            if (kv.first != "X") {
                ai4b::stop("GBartMultinomial::predict_at: unknown key "
                           "'%s'. Valid keys: 'X' (flattened column-major "
                           "N_new x p) or empty map for posterior "
                           "predictive at training X.",
                           kv.first.c_str());
            }
        }

        if (!has_X) {
            // ---- Empty-map prediction ----------------------------------
            if (keep_history_) {
                // History mode: pull each class's r history (block names
                // "r_1", ..., "r_{C-1}"), softmax to probs (ref f^(0)=1),
                // sample categorical y_rep per draw.
                AI4BayesCode::history_map hist = impl_->get_history();
                std::vector<arma::mat> r_class_hist;
                r_class_hist.reserve(C_ - 1);
                for (std::size_t j = 1; j < C_; ++j) {
                    auto it = hist.find(std::string("r_") +
                                        std::to_string(j));
                    if (it == hist.end()) {
                        ai4b::stop("GBartMultinomial::predict_at: "
                                   "keep_history requires r_%d history "
                                   "(missing).", (int)j);
                    }
                    r_class_hist.push_back(it->second);
                }
                const std::size_t n_draws = r_class_hist[0].n_rows;
                const std::size_t N_local = r_class_hist[0].n_cols;
                std::uniform_real_distribution<double> unif(0.0, 1.0);
                arma::mat yrep_mat(n_draws, N_local);
                for (std::size_t d = 0; d < n_draws; ++d) {
                    for (std::size_t i = 0; i < N_local; ++i) {
                        // Softmax over (ref f^(0)=1, r_1..r_{C-1}), class
                        // index 0 = reference (r = 0). Same convention as
                        // softmax_with_ref / the refresher.
                        double max_r = 0.0;
                        for (std::size_t c = 0; c + 1 < C_; ++c)
                            max_r = std::max(max_r,
                                             r_class_hist[c](d, i));
                        std::vector<double> probs_i(C_);
                        double denom = std::exp(0.0 - max_r);  // ref
                        probs_i[0] = std::exp(0.0 - max_r);
                        for (std::size_t c = 0; c + 1 < C_; ++c) {
                            probs_i[c + 1] =
                                std::exp(r_class_hist[c](d, i) - max_r);
                            denom += probs_i[c + 1];
                        }
                        const double u = unif(predict_rng_);
                        double acc = 0.0;
                        std::size_t chosen = C_ - 1;
                        for (std::size_t c = 0; c < C_; ++c) {
                            acc += probs_i[c] / denom;
                            if (u < acc) { chosen = c; break; }
                        }
                        yrep_mat(d, i) = static_cast<double>(chosen);
                    }
                }
                AI4BayesCode::history_map out;
                out.emplace("y_rep", std::move(yrep_mat));
                return out;
            }

            // Stateful mode: refresh probs + y_rep from the current draw.
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

        // ---- New-X prediction --------------------------------------
        if (keep_history_) {
            ai4b::stop("GBartMultinomial::predict_at: new-X prediction in "
                       "history mode is not yet implemented. Use empty map "
                       "or construct with keep_history = FALSE.");
        }

        const arma::vec& x_flat = new_data.at("X");
        if (x_ncol_ == 0 || (x_flat.n_elem % x_ncol_) != 0) {
            ai4b::stop("GBartMultinomial::predict_at: flattened X length "
                       "%d is not a multiple of p = %d",
                       (int)x_flat.n_elem, (int)x_ncol_);
        }
        const std::size_t N_new = x_flat.n_elem / x_ncol_;
        arma::mat X_new = arma::reshape(x_flat, N_new, x_ncol_);

        std::vector<arma::vec> r_new(C_ - 1);
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            auto& g = dynamic_cast<genbart_block&>(impl_->child(1 + j));
            r_new[j] = g.predict_r(X_new);
        }
        std::vector<const arma::vec*> ptrs(C_ - 1);
        for (std::size_t j = 0; j < C_ - 1; ++j) ptrs[j] = &r_new[j];
        arma::vec probs;
        softmax_with_ref(N_new, C_, ptrs, probs);

        // r flattened column-major (N_new x (C-1)).
        arma::mat r_mat(1, N_new * (C_ - 1));
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            for (std::size_t i = 0; i < N_new; ++i) {
                r_mat(0, j * N_new + i) = r_new[j][i];
            }
        }
        arma::mat probs_mat(1, probs.n_elem);
        for (std::size_t i = 0; i < probs.n_elem; ++i)
            probs_mat(0, i) = probs[i];

        AI4BayesCode::history_map out;
        out.emplace("r", std::move(r_mat));
        out.emplace("probs", std::move(probs_mat));
        return out;
    }

    // ---- History access --------------------------------------------------
    // The composite's neutral history_map (numeric matrices keyed by block
    // name: log_phi_aug [n_draws x N], r_1..r_{C-1} [n_draws x N]).
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    // Per-class serialized BART forest snapshots. In stateful mode each
    // child contributes one string (its current forest); in history mode
    // (keep_tree=true) each child contributes its full tree-string
    // history. Concatenated class-major (class 1's draws, then class 2's,
    // ...). Callers re-split using C-1 and the known per-class draw count.
    std::vector<std::string> get_tree_history() const {
        std::vector<std::string> out;
        for (std::size_t j = 0; j < C_ - 1; ++j) {
            auto& g = dynamic_cast<genbart_block&>(impl_->child(1 + j));
            std::vector<std::string> hj = g.get_tree_history();
            for (auto& s : hj) out.push_back(std::move(s));
        }
        return out;
    }

private:
    std::mt19937_64                                   rng_;
    mutable std::mt19937_64                           predict_rng_;
    std::unique_ptr<composite_block>                  impl_;
    std::vector<std::unique_ptr<genbart::likelihood>> lik_pool_; // C-1 entries
    std::size_t                                       C_;
    std::size_t                                       x_ncol_ = 0;
    bool                                              keep_tree_    = false;
    bool                                              keep_history_ = false;
};

// ============================================================================
//  Rcpp module: make the class visible to R.
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(GBartMultinomial_module) {
    Rcpp::class_<GBartMultinomial>("GBartMultinomial")
        .constructor<arma::mat, arma::vec, int, int, int>(
            "Short ctor: X (N x p), y in {0..C-1}, C, ntrees, seed; "
            "keep_tree and keep_history default FALSE.")
        .constructor<arma::mat, arma::vec, int, int, int, bool>(
            "Ctor with keep_tree only (EXPENSIVE; for predict_history).")
        .constructor<arma::mat, arma::vec, int, int, int, bool, bool>(
            "Full ctor: X (N x p), y in {0..C-1}, C (>= 2), ntrees, "
            "seed, keep_tree (forest snapshots; EXPENSIVE; default "
            "FALSE), keep_history (numeric buffers; cheap; default "
            "FALSE).")
        .method("step", (void (GBartMultinomial::*)())    &GBartMultinomial::step, "Run one sweep.")
        .method("step", (void (GBartMultinomial::*)(int)) &GBartMultinomial::step,
                "Run n Gibbs sweeps.")
        .method("get_current", &GBartMultinomial::get_current,
                "Return the current draw as a named list with $r "
                "(flattened N x (C-1)), $probs (flattened N x C), and "
                "$log_phi. Forests are available via get_tree().")
        .method("get_tree",    &GBartMultinomial::get_tree,
                "Return all C-1 serialized BART forests joined into one "
                "string (round-trips through set_tree()).")
        .method("set_tree",    &GBartMultinomial::set_tree,
                "Restore the C-1 BART forests from a serialized snapshot "
                "string previously obtained from get_tree().")
        .method("set_current", &GBartMultinomial::set_current,
                "Overwrite y and/or X from a named list. Supported keys: "
                "y, X (flattened column-major N x p). r/probs/log_phi are "
                "read-only.")
        .method("predict_at",  &GBartMultinomial::predict_at,
                "Predict at new data. Pass list(X = as.vector(X_new)) "
                "(flattened column-major) or an empty list for posterior "
                "predictive at training X. Const, no state mutation.")
        .method("get_dag",     &GBartMultinomial::get_dag,
                "Return the predict DAG as a named list of edges.")
        .method("get_history", &GBartMultinomial::get_history,
                "Return the history as a named list of matrices "
                "(log_phi_aug, r_1..r_{C-1}, each n_draws x N).")
        .method("get_tree_history", &GBartMultinomial::get_tree_history,
                "Return per-class serialized BART forests (latest per "
                "class, or per-draw when keep_tree=TRUE).");
}
#endif

// ============================================================================
//  pybind11 module: make the class visible to Python.
// ============================================================================

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(GBartMultinomial, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<GBartMultinomial>(m, "GBartMultinomial")
        .def(pybind11::init<arma::mat, arma::vec, int, int, int, bool, bool>(),
             pybind11::arg("X"),
             pybind11::arg("y"),
             pybind11::arg("C"),
             pybind11::arg("ntrees")       = 50,
             pybind11::arg("rng_seed")     = 1,
             pybind11::arg("keep_tree")    = false,
             pybind11::arg("keep_history") = false)
        .def("step", (void (GBartMultinomial::*)())    &GBartMultinomial::step, "Run one sweep.")
        .def("step", (void (GBartMultinomial::*)(int)) &GBartMultinomial::step, pybind11::arg("n_steps"))
        .def("get_current",      &GBartMultinomial::get_current)
        .def("get_tree",         &GBartMultinomial::get_tree)
        .def("set_tree",         &GBartMultinomial::set_tree, pybind11::arg("tree_s"))
        .def("set_current",      &GBartMultinomial::set_current, pybind11::arg("params"))
        .def("predict_at",       &GBartMultinomial::predict_at, pybind11::arg("new_data"))
        .def("get_dag",          &GBartMultinomial::get_dag)
        .def("get_history",      &GBartMultinomial::get_history)
        .def("get_tree_history", &GBartMultinomial::get_tree_history);
}
#endif

// ============================================================================
//  Standalone demo: simulate a 3-class multinomial logistic DGP, fit,
//  print finite recovery + classification accuracy.
// ============================================================================

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>

int main() {
    // ---- Simulate a well-identified C = 3 multinomial DGP ---------------
    //   true log-odds vs reference class 0:
    //     r_1(x) =  1.2 * x1 - 0.8 * x2       (class 1 vs 0)
    //     r_2(x) = -1.0 * x1 + 1.5 * x3       (class 2 vs 0)
    //   y_i ~ Categorical( softmax(0, r_1, r_2) )
    //   N = 900 (>= ~300 per class on average) for identifiability.
    const std::size_t N = 900;
    const std::size_t p = 3;
    const std::size_t C = 3;

    std::mt19937_64 gen(20260622ULL);
    std::uniform_real_distribution<double> unif(-1.5, 1.5);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    arma::mat X(N, p);
    arma::vec y(N);
    std::vector<std::size_t> class_count(C, 0);
    for (std::size_t i = 0; i < N; ++i) {
        const double x1 = unif(gen);
        const double x2 = unif(gen);
        const double x3 = unif(gen);
        X(i, 0) = x1; X(i, 1) = x2; X(i, 2) = x3;
        const double r1 =  1.2 * x1 - 0.8 * x2;
        const double r2 = -1.0 * x1 + 1.5 * x3;
        const double e0 = 1.0;            // ref f^(0) = 1
        const double e1 = std::exp(r1);
        const double e2 = std::exp(r2);
        const double Z  = e0 + e1 + e2;
        const double p0 = e0 / Z, p1 = e1 / Z;
        const double u  = u01(gen);
        std::size_t cls = (u < p0) ? 0 : (u < p0 + p1 ? 1 : 2);
        y[i] = static_cast<double>(cls);
        class_count[cls]++;
    }

    // ---- Fit -------------------------------------------------------------
    GBartMultinomial model(X, y,
                           /*C=*/static_cast<int>(C),
                           /*ntrees=*/50,
                           /*rng_seed=*/42,
                           /*keep_tree=*/false, /*keep_history=*/false);

    const int n_burn = 400;
    const int n_keep = 400;
    model.step(n_burn);

    // Accumulate posterior-mean class probabilities at the training X.
    arma::vec probs_sum(N * C, arma::fill::zeros);
    double    log_phi_check = 0.0;
    for (int it = 0; it < n_keep; ++it) {
        model.step(1);
        AI4BayesCode::state_map cur = model.get_current();
        probs_sum += cur.at("probs");
        log_phi_check += cur.at("log_phi")[0];
    }
    const arma::vec probs_hat = probs_sum / static_cast<double>(n_keep);

    // ---- Report recovery -------------------------------------------------
    // Posterior-mean MAP classification accuracy on the training data.
    std::size_t correct = 0;
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t argmax = 0;
        double best = probs_hat[i * C + 0];
        for (std::size_t j = 1; j < C; ++j) {
            if (probs_hat[i * C + j] > best) {
                best = probs_hat[i * C + j];
                argmax = j;
            }
        }
        if (argmax == static_cast<std::size_t>(y[i])) correct++;
    }
    const double acc = static_cast<double>(correct) / static_cast<double>(N);

    // Mean predicted probability assigned to the TRUE class (a proper
    // identified summary: should be well above 1/C = 0.333 if r_j fit).
    double mean_true_prob = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        mean_true_prob += probs_hat[i * C + static_cast<std::size_t>(y[i])];
    }
    mean_true_prob /= static_cast<double>(N);

    bool finite_ok = probs_hat.is_finite()
                     && std::isfinite(acc)
                     && std::isfinite(mean_true_prob)
                     && std::isfinite(log_phi_check);

    std::printf("GBartMultinomial standalone demo\n");
    std::printf("  N = %zu, p = %zu, C = %zu, ntrees = 50\n", N, p, C);
    std::printf("  class counts: %zu / %zu / %zu\n",
                class_count[0], class_count[1], class_count[2]);
    std::printf("  training MAP accuracy        = %.4f  (chance ~%.3f)\n",
                acc, 1.0 / static_cast<double>(C));
    std::printf("  mean P(true class)           = %.4f  (chance ~%.3f)\n",
                mean_true_prob, 1.0 / static_cast<double>(C));
    std::printf("  all finite                   = %s\n",
                finite_ok ? "YES" : "NO");

    if (!finite_ok) {
        std::printf("FAIL: non-finite recovery\n");
        return 1;
    }
    if (!(acc > 1.0 / static_cast<double>(C))
        || !(mean_true_prob > 1.0 / static_cast<double>(C))) {
        std::printf("FAIL: no signal recovery (accuracy/true-prob at "
                    "or below chance)\n");
        return 1;
    }
    std::printf("OK: finite recovery, signal above chance\n");
    return 0;
}
#endif

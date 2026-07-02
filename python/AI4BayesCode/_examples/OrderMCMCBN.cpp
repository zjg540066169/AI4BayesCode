// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  OrderMCMCBN.cpp
//
//  Tier A demo for v1.2 Block 3 (order_mcmc_block).
//
//  Model
//  -----
//  Bayesian network structure learning via Friedman-Koller 2003 order
//  MCMC. Given a categorical dataset D, samples from the posterior
//  P(≺ | D) over orderings ≺ of n variables, with closed-form marginal
//  likelihood
//
//      P(D | ≺) = ∏_i Σ_{U ⊆ pred_i(≺), |U| ≤ k} score(X_i, U | D)
//
//  Score is BDe (Heckerman-Geiger-Chickering 1995 Eq 28) with BDeu
//  pseudocounts (Buntine 1991): N'_ijk = α / (r_i × q_i). Per-family
//  structure prior is the FK Eq 2 uniform over |U|.
//
//  MH proposal: mixture of any-pair swap + adjacent swap (symmetric;
//  see order_mcmc_block.hpp).
//
//  Per-step outputs (via current_named_outputs()):
//    "order"             length-n permutation
//    "order_sampled_DAG" length-n² flattened adjacency (row-major,
//                        A[i*n+j]=1 ⇔ j is parent of i)
//    "order_log_score"   length-1 log P(D | ≺)
//
//  NAMING CLASH WARNING
//  --------------------
//  get_dag() returns AI4BayesCode's INTERNAL Gibbs/Predict DAG (one node,
//  "order_mcmc"), NOT the learned Bayesian network DAG. The learned BN
//  DAG is in `get_current()$sampled_DAG`.
//
//  KNOWN BIAS (FK 2003 §4.1)
//  -------------------------
//  Order MCMC's induced structure prior is NOT hypothesis-equivalent
//  (Markov-equivalent DAGs receive different prior weights). Fix is
//  Kuipers-Moffa 2017 partition MCMC, deferred to v1.2.1.
//
//  JUSTIFICATION (Check #16): Discrete latent (BN structure) with
//  combinatorial state space. system_design.md §11.2(b) lists this as
//  the canonical Block 3. order_mcmc_block is the v1.2 implementation.
//
//  DUAL-BACKEND (R + Python)
//  -------------------------
//  Backend-neutral state I/O via AI4BayesCode::state_map / history_map:
//    * get_current()  -> state_map { "order" (1-based, flat arma::vec),
//                                    "sampled_DAG" (n*n flattened, COLUMN-
//                                    major arma::vec; element (i,j) at index
//                                    i + j*n; A[i,j]=1 ⇔ j is parent of i),
//                                    "log_score" (length-1) }.
//    * set_current(const state_map&) reads "order" (1-based length-n) via
//                                    .find(); sampled_DAG / log_score are
//                                    derived and rejected if passed.
//    * predict_at(const state_map&) -> history_map { "sampled_DAG" } : v1
//                                    stub returning the current sampled DAG
//                                    as a 1 x n² matrix (column-major). Full
//                                    posterior-predictive simulation deferred
//                                    to v1.2.1 per spec §5.5.
//  Constructor takes arma::mat data (N x n, integer-valued, rounded to
//  category indices 0..r-1), arma::vec cardinalities (length n), and an
//  arma::vec initial_order (EMPTY = random init; otherwise a 1-based length-n
//  permutation). This replaces the Rcpp-only IntegerMatrix / IntegerVector /
//  Nullable signature so the SAME class compiles under both RCPP_MODULE and
//  PYBIND11_MODULE.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("OrderMCMCBN")
//   set.seed(123); n <- 4L; N <- 4000L; flip <- 0.10   # binary chain X1->X2->X3->X4
//   data <- matrix(0L, N, n)
//   data[, 1] <- as.integer(runif(N) < 0.5)             # X1 ~ Bernoulli(0.5)
//   for (j in 2:n) data[, j] <- ifelse(runif(N) < flip, 1L - data[, j-1], data[, j-1])
//   cards <- rep(2, n)                                  # all variables binary
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- AI4BayesCode_run_chains(
//       function(seed) new(OrderMCMCBN, data, cards, 1.0, 3L, 20L, 4000L, 10.0, 0.5, numeric(0), seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4b_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(OrderMCMCBN, data, cards, 1.0, 3L, 20L, 4000L, 10.0, 0.5,
//            numeric(0), 7L, TRUE)   # bdeu_alpha,max_parents,top_C,cache_F,prune,p_adj,init_order=empty,seed,keep_history
//   m$step(2500L); cur <- m$get_current(); str(cur); print(cur$sampled_DAG)
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(123); n, N, flip = 4, 4000, 0.10   # binary chain X1->...->X4
//   data = np.zeros((N, n), dtype=float)
//   data[:, 0] = (rng.uniform(size=N) < 0.5).astype(float)         # X1 ~ Bernoulli(0.5)
//   for j in range(1, n):
//       flipmask = rng.uniform(size=N) < flip
//       data[:, j] = np.where(flipmask, 1.0 - data[:, j-1], data[:, j-1])
//   cards = np.full(n, 2.0)                                        # all binary
//   Mod = AI4BayesCode.source("OrderMCMCBN.cpp")
//   # bdeu_alpha, max_parents, top_C, cache_F, prune, p_adj, init_order(empty), seed, keep_history
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.OrderMCMCBN(data, cards, 1.0, 3, 20, 4000, 10.0, 0.5, np.zeros(0), seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.OrderMCMCBN(data, cards, 1.0, 3, 20, 4000, 10.0, 0.5,
//                       np.zeros(0), 7, True)
//   m.step(2500); cur = m.get_current(); print(cur["sampled_DAG"].reshape(n, n, order="F"))
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
#include "AI4BayesCode/order_mcmc_block.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::order_mcmc_block;
using AI4BayesCode::order_mcmc_block_config;

// ============================================================================
//  User-facing class exposed to BOTH R (RCPP_MODULE) and Python
//  (PYBIND11_MODULE). All state I/O is backend-neutral (state_map /
//  history_map); the constructor uses arma types only.
// ============================================================================

class OrderMCMCBN {
public:
    OrderMCMCBN(const arma::mat& data,            // N x n, integer-valued
                const arma::vec& cardinalities,   // length n, r_i per variable
                double bdeu_alpha,
                int max_parents,
                int candidate_top_C,
                int family_cache_F,
                double gamma_prune_nats,
                double prob_adjacent_swap,
                const arma::vec& initial_order,    // EMPTY = random; else 1-based
                int rng_seed,
                bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          impl_(std::make_unique<composite_block>("OrderMCMCBN")),
          keep_history_(keep_history)
    {
        const int N = static_cast<int>(data.n_rows);
        const int n = static_cast<int>(data.n_cols);
        if (n < 2)        ai4b::stop("data must have >= 2 columns");
        if (N < 1)        ai4b::stop("data must have >= 1 row");
        if (n > 64)       ai4b::stop("n > 64 not supported");
        if (static_cast<int>(cardinalities.n_elem) != n)
            ai4b::stop("cardinalities length must equal ncol(data)");
        if (!(bdeu_alpha > 0)) ai4b::stop("bdeu_alpha must be > 0");
        if (max_parents < 0 || max_parents > 64)
            ai4b::stop("max_parents must be in [0, 64]");
        if (candidate_top_C < 1)
            ai4b::stop("candidate_top_C must be >= 1");
        if (family_cache_F < 1)
            ai4b::stop("family_cache_F must be >= 1");
        if (gamma_prune_nats < 0)
            ai4b::stop("gamma_prune_nats must be >= 0");
        if (prob_adjacent_swap < 0.0 || prob_adjacent_swap > 1.0)
            ai4b::stop("prob_adjacent_swap must be in [0, 1]");

        n_ = static_cast<std::size_t>(n);

        // Convert (integer-valued) arma::mat → arma::imat by rounding.
        arma::imat data_arma(N, n);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < n; ++j)
                data_arma(i, j) =
                    static_cast<arma::sword>(std::llround(data(i, j)));

        arma::uvec cards_arma(n);
        for (int j = 0; j < n; ++j)
            cards_arma[j] =
                static_cast<arma::uword>(std::llround(cardinalities[j]));

        // ---- Build order_mcmc_block config -----------------------------
        order_mcmc_block_config cfg;
        cfg.name = "order";
        cfg.data = data_arma;
        cfg.cardinalities = cards_arma;
        cfg.bdeu_alpha = bdeu_alpha;
        cfg.max_parents = static_cast<std::size_t>(max_parents);
        cfg.candidate_top_C = static_cast<std::size_t>(candidate_top_C);
        cfg.family_cache_F = static_cast<std::size_t>(family_cache_F);
        cfg.gamma_prune_nats = gamma_prune_nats;
        cfg.prob_adjacent_swap = prob_adjacent_swap;
        cfg.use_structure_prior = true;
        cfg.init_rng_seed = static_cast<std::uint64_t>(rng_seed);
        // EMPTY initial_order = random init; otherwise it is a 1-based
        // (R/Python-style) length-n permutation converted to 0-based.
        if (initial_order.n_elem > 0) {
            if (static_cast<int>(initial_order.n_elem) != n)
                ai4b::stop("initial_order length must equal ncol(data)");
            cfg.initial_order = arma::uvec(n);
            for (int i = 0; i < n; ++i) {
                const long long v = std::llround(initial_order[i]);
                if (v < 1 || v > n)
                    ai4b::stop("initial_order entries must be in 1..n "
                               "(1-based)");
                cfg.initial_order[i] = static_cast<arma::uword>(v - 1);
            }
        }

        // ---- shared_data setup -----------------------------------------
        // No upstream data dependencies for the block.
        impl_->add_child(std::make_unique<order_mcmc_block>(std::move(cfg)));

        if (keep_history_) impl_->set_keep_history(true);

        // composite_block.step() writes child.current() to shared_data
        // under the block's name automatically; no explicit invalidates
        // declarations needed here.
    }

    // ---- backend-neutral interface ---------------------------------

    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        const auto* blk = dynamic_cast<const order_mcmc_block*>(
            &impl_->child(0));
        if (!blk) ai4b::stop("internal: child 0 is not order_mcmc_block");

        AI4BayesCode::state_map out;

        // Order (1-based, flat length-n arma::vec).
        const arma::uvec& order = blk->order();
        arma::vec order_v(order.n_elem);
        for (std::size_t i = 0; i < order.n_elem; ++i)
            order_v[i] = static_cast<double>(order[i]) + 1.0;  // 1-based
        out["order"] = std::move(order_v);

        // Sampled DAG (n x n adjacency, FLATTENED COLUMN-MAJOR: element
        // (i,j) is stored at index i + j*n; A[i,j]=1 ⇔ j is parent of i;
        // indices refer to 1-based variable names on the frontend).
        const auto& dag = blk->sampled_dag();
        arma::vec dag_v(n_ * n_, arma::fill::zeros);
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j)
                dag_v[i + j * n_] = (dag[i] & (1ULL << j)) ? 1.0 : 0.0;
        out["sampled_DAG"] = std::move(dag_v);

        // Log score (length-1).
        out["log_score"] = arma::vec{blk->current_log_score()};
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto* blk = dynamic_cast<order_mcmc_block*>(&impl_->child(0));
        if (!blk) ai4b::stop("internal: child 0 is not order_mcmc_block");

        // sampled_DAG and log_score are derived; reject set on them.
        if (params.count("sampled_DAG"))
            ai4b::stop("set_current: 'sampled_DAG' is derived from order; "
                       "pass 'order' to update state");
        if (params.count("log_score"))
            ai4b::stop("set_current: 'log_score' is derived; pass 'order'");

        auto it_order = params.find("order");
        if (it_order != params.end()) {
            const arma::vec& ord = it_order->second;
            if (ord.n_elem != n_)
                ai4b::stop("set_current: order length must equal n");
            arma::vec order_arma(n_);
            for (std::size_t i = 0; i < n_; ++i)
                order_arma[i] =
                    static_cast<double>(std::llround(ord[i]) - 1);  // 1->0 based
            blk->set_current(order_arma);
            impl_->data().set("order",
                arma::conv_to<arma::vec>::from(blk->order()));
        }
    }

    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        // v1: stub. Returns the current sampled_DAG as a 1 x n² matrix
        // (column-major flattening, matching get_current()'s "sampled_DAG").
        // Full posterior-predictive simulation deferred to v1.2.1 per
        // spec §5.5. OrderMCMCBN has no covariate inputs.
        if (!new_data.empty())
            ai4b::stop("OrderMCMCBN has no covariate inputs. "
                       "predict_at() takes an empty map/list.");

        const auto* blk = dynamic_cast<const order_mcmc_block*>(
            &impl_->child(0));
        if (!blk) ai4b::stop("internal: child 0 is not order_mcmc_block");
        const auto& dag = blk->sampled_dag();

        arma::mat dag_m(1, n_ * n_, arma::fill::zeros);
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j)
                dag_m(0, i + j * n_) = (dag[i] & (1ULL << j)) ? 1.0 : 0.0;

        AI4BayesCode::history_map out;
        out.emplace("sampled_DAG", std::move(dag_m));
        return out;
    }

    AI4BayesCode::dag_info     get_dag()     const { return impl_->get_dag(); }
    AI4BayesCode::history_map  get_history() const { return impl_->get_history(); }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      n_ = 0;
};

// ============================================================================
//  Rcpp module (R)
// ============================================================================
#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(OrderMCMCBN_module) {
    Rcpp::class_<OrderMCMCBN>("OrderMCMCBN")
        .constructor<arma::mat, arma::vec, double, int,
                      int, int, double, double,
                      arma::vec, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::mat, arma::vec, double, int,
                      int, int, double, double,
                      arma::vec, int, bool>(
            "Bayesian-network structure learning via Friedman-Koller 2003 "
            "order MCMC with BDeu scoring. Args: data (N x n integer 0..r-1), "
            "cardinalities (length n, r_i for each variable), bdeu_alpha "
            "(default 1), max_parents (default 5), candidate_top_C (default "
            "20), family_cache_F (default 4000), gamma_prune_nats (default "
            "10), prob_adjacent_swap (default 0.5), initial_order (EMPTY "
            "numeric(0) -> random; else 1-based length-n permutation), "
            "rng_seed, keep_history.")
        .method("step",        &OrderMCMCBN::step)
        .method("get_current", &OrderMCMCBN::get_current,
                "Returns list(order = length-n 1-based permutation, "
                "sampled_DAG = n*n flattened COLUMN-MAJOR adjacency "
                "(A[i,j]=1 ⇔ j is parent of i; 1-based indices), "
                "log_score = log P(D|order)).")
        .method("set_current", &OrderMCMCBN::set_current,
                "Overwrite the order (1-based length-n permutation). "
                "sampled_DAG and log_score are derived — must pass 'order'.")
        .method("predict_at",  &OrderMCMCBN::predict_at,
                "v1: stub returning current sampled_DAG (1 x n² matrix). "
                "Full posterior-predictive simulation deferred to v1.2.1.")
        .method("get_dag",     &OrderMCMCBN::get_dag)
        .method("get_history", &OrderMCMCBN::get_history);
}
#endif

// ============================================================================
//  Pybind11 module (Python)
// ============================================================================
#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(OrderMCMCBN, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<OrderMCMCBN>(m, "OrderMCMCBN")
        .def(pybind11::init<arma::mat, arma::vec, double, int,
                            int, int, double, double,
                            arma::vec, int, bool>(),
             pybind11::arg("data"),
             pybind11::arg("cardinalities"),
             pybind11::arg("bdeu_alpha")        = 1.0,
             pybind11::arg("max_parents")       = 5,
             pybind11::arg("candidate_top_C")   = 20,
             pybind11::arg("family_cache_F")    = 4000,
             pybind11::arg("gamma_prune_nats")  = 10.0,
             pybind11::arg("prob_adjacent_swap")= 0.5,
             pybind11::arg("initial_order")     = arma::vec(),
             pybind11::arg("rng_seed")          = 1,
             pybind11::arg("keep_history")      = false)
        .def("step",        &OrderMCMCBN::step,        pybind11::arg("n_steps"))
        .def("get_current", &OrderMCMCBN::get_current)
        .def("set_current", &OrderMCMCBN::set_current, pybind11::arg("params"))
        .def("predict_at",  &OrderMCMCBN::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",     &OrderMCMCBN::get_dag)
        .def("get_history", &OrderMCMCBN::get_history);
}
#endif

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  OrderMCMCBN.cpp
//
//  Bayesian-network structure learning by partition MCMC (order_mcmc_block,
//  default method).
//
//  Model
//  -----
//  Bayesian network structure learning with order_mcmc_block in its default
//  method, Kuipers-Moffa 2017 partition MCMC. Given a categorical dataset D,
//  samples from the posterior P(Λ | D) over labelled partitions Λ of the n
//  variables (an ordered sequence of elements; a node's parents lie in later
//  elements, at least one in the next element), with closed-form marginal
//  likelihood
//
//      P(D | Λ) = ∏_i Σ_{U permissible for i in Λ, |U| <= k} score(X_i, U | D)
//
//  Score is BDe (Heckerman-Geiger-Chickering 1995 Eq 28) with BDeu
//  pseudocounts (Buntine 1991): N'_ijk = α / (r_i × q_i). Per-family
//  structure prior: the structure_prior argument ("uniform" or "fk_eq2").
//
//  MH proposal: mixture of split/join, swap, single-node and edge-reversal
//  moves (see partition_state.hpp). The order method (Friedman-Koller 2003,
//  P(≺ | D) over orders ≺) is a block option that this wrapper does not set.
//
//  Per-step outputs (via current_named_outputs()):
//    "order"             length-n partition node sequence, element 0 first
//    "order_party"       element sizes of the partition
//    "order_sampled_DAG" length-n² flattened adjacency (row-major,
//                        A[i*n+j]=1 ⇔ j is parent of i)
//    "order_log_score"   length-1 log P(D | Λ)
//
//  NAMING CLASH WARNING
//  --------------------
//  get_dag() returns AI4BayesCode's INTERNAL Gibbs/Predict DAG (one node,
//  "order_mcmc"), NOT the learned Bayesian network DAG. The learned BN
//  DAG is in `get_current()$sampled_DAG`.
//
//  KNOWN BIAS (FK 2003 Sec.4.1)
//  -------------------------
//  Order MCMC's induced structure prior is NOT hypothesis-equivalent
//  (Markov-equivalent DAGs receive different prior weights). Fix is
//  Kuipers-Moffa 2017 partition MCMC,
//  cfg.method = order_mcmc_block_config::method_t::partition, the block's
//  default and the method this wrapper runs.
//
//  Sampling note: Discrete latent (BN structure) with
//  combinatorial state space. system_design.md Sec.11.2(b) lists this as
//  the canonical implementation.
//
//  DUAL-BACKEND (R + Python)
//  -------------------------
//  Backend-neutral state I/O via AI4BayesCode::state_map / history_map:
//    * get_current()  -> state_map { "order" (the partition node sequence,
//                                    1-based, flat arma::vec; element 0
//                                    first, so children come before parents
//                                    and rev(order) lists parents first),
//                                    "order_party" (the element sizes,
//                                    summing to n),
//                                    "sampled_DAG" (n*n flattened, COLUMN-
//                                    major arma::vec; element (i,j) at index
//                                    i + j*n; A[i,j]=1 ⇔ j is parent of i),
//                                    "log_score" (length-1, log P(D | Λ)) }.
//    * set_current(const state_map&) with "order" and "order_party" loads
//                                    that partition exactly
//                                    (order_mcmc_block::set_partition; nodes
//                                    are sorted within each element). "order"
//                                    alone loads nothing: it is accepted only
//                                    when it equals the current node sequence
//                                    of a partition with two or more
//                                    elements, and raises an error otherwise,
//                                    including on a new model (the
//                                    one-element partition). sampled_DAG /
//                                    log_score are derived and rejected if
//                                    passed. After a load, get_current(),
//                                    predict_at() and the shared data keys
//                                    order / order_party / order_sampled_DAG
//                                    / order_log_score describe the loaded
//                                    partition; sampled_DAG is a DAG of it
//                                    (a load that changes the partition
//                                    draws it from a generator seeded by
//                                    rng_seed, not from the step generator).
//    * freeze / unfreeze accept "order_party" as a name of the "order"
//                                    block, so ai4bayescode_new_frozen(...,
//                                    fixed = list(order = ..., order_party =
//                                    ...)) (Python: new_frozen(..., fixed =
//                                    {"order": ..., "order_party": ...}))
//                                    loads the partition and freezes the
//                                    block at it.
//    * predict_at(const state_map&) -> history_map { "sampled_DAG" } : v1
//                                    stub returning the current sampled DAG
//                                    as a 1 x n² matrix (column-major). Full
//                                    posterior-predictive simulation deferred
//                                    to v1.2.1 per spec Sec.5.5.
//    * get_history()  -> history_map { "order" (node sequences, 1-based as
//                                    in get_current(); one row per sweep
//                                    with keep_history, else the current
//                                    sequence), "order_log_score" (one row
//                                    per sweep with keep_history) }.
//  Constructor takes arma::mat data (N x n, integer-valued, rounded to
//  category indices 0..r-1), arma::vec cardinalities (length n), and an
//  arma::vec initial_order that must be EMPTY: partition MCMC starts from the
//  one-element partition, and a node sequence does not give the element
//  sizes; a non-empty initial_order raises an error that points to
//  set_current(list(order = ..., order_party = ...)). This replaces the
//  Rcpp-only IntegerMatrix / IntegerVector / Nullable signature so the SAME
//  class compiles under both RCPP_MODULE and PYBIND11_MODULE.
//  candidate_top_C, family_cache_F, gamma_prune_nats and prob_adjacent_swap
//  are validated and passed to the block but do not change the draws of
//  partition MCMC: the block uses every family up to max_parents without
//  gamma pruning, and prob_adjacent_swap applies only to the order method.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("OrderMCMCBN")
//   set.seed(123); n <- 4L; N <- 4000L; flip <- 0.10   # binary chain X1->X2->X3->X4
//   data <- matrix(0L, N, n)
//   data[, 1] <- as.integer(runif(N) < 0.5)             # X1 ~ Bernoulli(0.5)
//   for (j in 2:n) data[, j] <- ifelse(runif(N) < flip, 1L - data[, j-1], data[, j-1])
//   cards <- rep(2, n)                                  # all variables binary
//   # ---- Parallel chains + convergence diagnosis (default) ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(OrderMCMCBN, data, cards, "fk_eq2", 1.0, 3L, 20L, 4000L, 10.0, 0.5, numeric(0), seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 10000)
//   print(ai4bayescode_rhat_summary(run))          # CROSS-chain R-hat / ESS
//   ai4bayescode_diagnose(run$histories[[1]])      # chain 1: summary + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(OrderMCMCBN, data, cards, "fk_eq2", 1.0, 3L, 20L, 4000L, 10.0, 0.5,
//            numeric(0), 7L, TRUE)   # structure_prior,bdeu_alpha,max_parents,top_C,cache_F,prune,p_adj,init_order=empty,seed,keep_history
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
//   Mod = AI4BayesCode.example("OrderMCMCBN")
//   # structure_prior ("uniform" | "fk_eq2"), then bdeu_alpha, max_parents, top_C,
//   # cache_F, prune, p_adj, init_order(empty), seed, keep_history
//   # ---- Parallel chains + diagnosis (default) ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.OrderMCMCBN(data, cards, "fk_eq2", 1.0, 3, 20, 4000, 10.0, 0.5, np.zeros(0), seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=10000, n_jobs=1)
//   print(AI4BayesCode.rhat_summary(chains))   # CROSS-chain R-hat / ESS
//   AI4BayesCode.diagnose(chains[0]["hist"])   # chain 1: summary + plots
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.OrderMCMCBN(data, cards, "fk_eq2", 1.0, 3, 20, 4000, 10.0, 0.5,
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
using AI4BayesCode::order_mcmc_block;
using AI4BayesCode::order_mcmc_block_config;

// ============================================================================
//  User-facing class exposed to BOTH R (RCPP_MODULE) and Python
//  (PYBIND11_MODULE). All state I/O is backend-neutral (state_map /
//  history_map); the constructor uses arma types only.
// ============================================================================

class OrderMCMCBN : public AI4BayesCode::kernel_control_mixin<OrderMCMCBN> {
    friend class AI4BayesCode::kernel_control_mixin<OrderMCMCBN>;
public:
    OrderMCMCBN(const arma::mat& data,            // N x n, integer-valued
                const arma::vec& cardinalities,   // length n, r_i per variable
                const std::string& structure_prior, // "uniform" | "fk_eq2" -- see below
                double bdeu_alpha = 1.0,
                int max_parents = 5,
                int candidate_top_C = 20,
                int family_cache_F = 4000,
                double gamma_prune_nats = 10.0,
                double prob_adjacent_swap = 0.5,
                const arma::vec& initial_order = arma::vec(),    // must be EMPTY (see below)
                int rng_seed = 1,
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

        // Convert (integer-valued) arma::mat -> arma::imat by rounding.
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
        // The DAG prior is a MODELLING choice with no safe default, so it is a
        // constructor argument rather than something buried in this file: the
        // two priors give different posteriors and the caller has to say which
        // one their model means. Note that neither of the two neighbouring
        // knobs decides it -- `max_parents` above is an in-degree CAP applied
        // under both priors, and picking Friedman-Koller order MCMC as the
        // SAMPLER does not pick the Friedman-Koller PRIOR.
        if (structure_prior == "uniform") {
            // P(G) proportional to 1 over DAGs obeying the in-degree cap.
            // Matches BiDAG (edgepf = 1) and bnlearn defaults; this is what a
            // spec saying "G ~ Uniform(DAGs)" or "P(G) propto 1" asks for.
            cfg.structure_prior = AI4BayesCode::dag_prior::UNIFORM;
        } else if (structure_prior == "fk_eq2") {
            // Friedman-Koller 2003 Eq 2 per-family balancing:
            // P(G) proportional to prod_j 1 / C(p-1, |Pa_j|). Penalises high
            // fan-in; the FK paper's own recommended default.
            cfg.structure_prior = AI4BayesCode::dag_prior::FK_EQ2;
        } else {
            ai4b::stop("OrderMCMCBN: structure_prior must be \"uniform\" "
                       "(P(G) proportional to 1) or \"fk_eq2\" "
                       "(Friedman-Koller 2003 Eq 2 per-family balancing); got \"%s\"",
                       structure_prior.c_str());
        }
        cfg.init_rng_seed = static_cast<std::uint64_t>(rng_seed);
        // The block runs its default method, partition MCMC, which starts
        // from the one-element partition. A node sequence does not give the
        // element sizes of a partition, so a non-empty initial_order is
        // rejected; set_current(list(order = ..., order_party = ...)) loads a
        // partition after construction.
        if (initial_order.n_elem > 0)
            ai4b::stop(std::string(
                "OrderMCMCBN: initial_order must be empty (numeric(0) in R, "
                "np.zeros(0) in Python). The sampler is partition MCMC, which "
                "starts from the one-element partition, and a node sequence "
                "does not give the element sizes of a partition. To start "
                "from another partition, construct the model with an empty "
                "initial_order and call set_current(list(order = ..., "
                "order_party = ...)) (Python: set_current({'order': ..., "
                "'order_party': ...})) with values as get_current() returns "
                "them."));

        // ---- shared_data setup -----------------------------------------
        // No upstream data dependencies for the block.
        impl_->add_child(std::make_unique<order_mcmc_block>(std::move(cfg)));

        if (keep_history_) impl_->set_keep_history(true);

        // composite_block.step() writes child.current() to shared_data
        // under the block's name automatically; no explicit invalidates
        // declarations needed here.
    }

    // ---- backend-neutral interface ---------------------------------

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        const auto* blk = dynamic_cast<const order_mcmc_block*>(
            &impl_->child(0));
        if (!blk) ai4b::stop("internal: child 0 is not order_mcmc_block");

        AI4BayesCode::state_map out;

        // Partition node sequence (1-based, flat length-n arma::vec),
        // element 0 first: a node's parents lie in later elements.
        const arma::uvec& order = blk->order();
        arma::vec order_v(order.n_elem);
        for (std::size_t i = 0; i < order.n_elem; ++i)
            order_v[i] = static_cast<double>(order[i]) + 1.0;  // 1-based
        out["order"] = std::move(order_v);

        // Element sizes of the partition (they sum to n).
        const std::vector<std::size_t>& party = blk->partition().party;
        arma::vec party_v(party.size());
        for (std::size_t t = 0; t < party.size(); ++t)
            party_v[t] = static_cast<double>(party[t]);
        out["order_party"] = std::move(party_v);

        // Sampled DAG (n x n adjacency, FLATTENED COLUMN-MAJOR: element
        // (i,j) is stored at index i + j*n; A[i,j]=1 ⇔ j is parent of i;
        // indices refer to 1-based variable names on the frontend).
        const auto& dag = blk->sampled_dag();
        arma::vec dag_v(n_ * n_, arma::fill::zeros);
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j)
                dag_v[i + j * n_] = (dag[i] & (1ULL << j)) ? 1.0 : 0.0;
        out["sampled_DAG"] = std::move(dag_v);

        // Log score (length-1): log P(D | partition).
        out["log_score"] = arma::vec{blk->current_log_score()};
        return out;
    }

    // "order" + "order_party": load that partition (set_partition).
    // "order" alone: order_mcmc_block::set_current, which loads nothing; it
    // returns when "order" equals the current node sequence of a partition
    // with two or more elements and throws otherwise.
    void set_current(const AI4BayesCode::state_map& params) {
        auto* blk = dynamic_cast<order_mcmc_block*>(&impl_->child(0));
        if (!blk) ai4b::stop("internal: child 0 is not order_mcmc_block");

        // sampled_DAG and log_score are derived; reject set on them.
        if (params.count("sampled_DAG"))
            ai4b::stop("set_current: 'sampled_DAG' is derived from the "
                       "partition; pass 'order' and 'order_party' to update "
                       "state");
        if (params.count("log_score"))
            ai4b::stop("set_current: 'log_score' is derived; pass 'order' "
                       "and 'order_party'");

        auto it_order = params.find("order");
        auto it_party = params.find("order_party");
        if (it_order == params.end()) {
            if (it_party != params.end())
                ai4b::stop("set_current: 'order_party' needs 'order', the "
                           "node sequence of the same partition");
            return;
        }
        const arma::vec& ord = it_order->second;
        if (ord.n_elem != n_)
            ai4b::stop("set_current: order length must equal n");
        arma::uvec permy(n_);
        std::vector<char> seen(n_, 0);
        for (std::size_t i = 0; i < n_; ++i) {
            const double d = ord[i];
            // d in [0.5, n + 0.5) <=> llround(d) in 1..n (1-based)
            if (!(d >= 0.5 && d < static_cast<double>(n_) + 0.5))
                ai4b::stop("set_current: 'order' must be a permutation of "
                           "1..n (1-based)");
            const std::size_t v = static_cast<std::size_t>(std::llround(d)) - 1;
            if (seen[v])
                ai4b::stop("set_current: 'order' must be a permutation of "
                           "1..n (1-based)");
            seen[v] = 1;
            permy[i] = v;
        }

        if (it_party != params.end()) {
            const arma::vec& pv = it_party->second;
            arma::uvec party(pv.n_elem);
            std::size_t total = 0;
            for (std::size_t t = 0; t < pv.n_elem; ++t) {
                const double d = pv[t];
                if (!(d >= 0.5 && d < static_cast<double>(n_ - total) + 0.5))
                    ai4b::stop("set_current: 'order_party' must hold element "
                               "sizes >= 1 that sum to n");
                party[t] = static_cast<arma::uword>(std::llround(d));
                total += party[t];
            }
            if (total != n_)
                ai4b::stop("set_current: 'order_party' must hold element "
                           "sizes >= 1 that sum to n");
            try {
                blk->set_partition(permy, party);
            } catch (const std::invalid_argument& e) {
                ai4b::stop(std::string("set_current: ") + e.what());
            }
        } else {
            try {
                blk->set_current(arma::conv_to<arma::vec>::from(permy));
            } catch (const std::invalid_argument&) {
                ai4b::stop(std::string(
                    "set_current: a node sequence does not give the element "
                    "sizes of a partition, so 'order' alone is accepted only "
                    "when it equals the current node sequence of a partition "
                    "with two or more elements (nothing changes). To load a "
                    "partition, pass 'order' and 'order_party' together, as "
                    "get_current() returns them: set_current(list(order = "
                    "..., order_party = ...)) (Python: set_current({'order': "
                    "..., 'order_party': ...}))."));
            }
        }
        // Writes the block's outputs to the shared data keys order,
        // order_party, order_sampled_DAG and order_log_score (the DAG is the
        // block's sampled_dag()) and marks the composite's current() for
        // rebuild.
        impl_->sync_child(0);
    }

    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        // v1: stub. Returns the current sampled_DAG as a 1 x n² matrix
        // (column-major flattening, matching get_current()'s "sampled_DAG").
        // Full posterior-predictive simulation deferred to v1.2.1 per
        // spec Sec.5.5. OrderMCMCBN has no covariate inputs.
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

    // The composite's history with "order" 1-based, as in get_current().
    AI4BayesCode::history_map  get_history() const {
        AI4BayesCode::history_map h = impl_->get_history();
        const auto it = h.find("order");
        if (it != h.end()) it->second += 1.0;
        return h;
    }

    // freeze / unfreeze: "order_party" names the "order" block (the node
    // sequence and the element sizes are one block state), so
    // freeze(c("order", "order_party")), as ai4bayescode_new_frozen /
    // new_frozen call it, freezes that block once.
#ifdef AI4BAYESCODE_RCPP_MODULE
    void freeze_names(const Rcpp::CharacterVector& names) {
        mixin_::freeze_names(block_names_(names));
    }
    void freeze_names_quiet(const Rcpp::CharacterVector& names, bool quiet) {
        mixin_::freeze_names_quiet(block_names_(names), quiet);
    }
    void unfreeze_names(Rcpp::Nullable<Rcpp::CharacterVector> names) {
        if (names.isNull()) { mixin_::unfreeze_names(names); return; }
        mixin_::unfreeze_names(Rcpp::Nullable<Rcpp::CharacterVector>(
            block_names_(Rcpp::CharacterVector(names.get()))));
    }
#endif
#ifdef AI4BAYESCODE_PYBIND_MODULE
    void py_freeze(const std::vector<std::string>& names, bool quiet) {
        mixin_::py_freeze(block_names_(names), quiet);
    }
    void py_unfreeze(pybind11::object names) {
        if (names.is_none()) { mixin_::py_unfreeze(names); return; }
        mixin_::py_unfreeze(pybind11::cast(
            block_names_(names.cast<std::vector<std::string>>())));
    }
#endif

private:
    using mixin_ = AI4BayesCode::kernel_control_mixin<OrderMCMCBN>;

    // Freeze names with "order_party" replaced by "order"; the replacement
    // is dropped when "order" is also given or was already added.
    static std::vector<std::string> block_names_(
            const std::vector<std::string>& names) {
        bool has_order = false;
        for (const auto& s : names) if (s == "order") has_order = true;
        std::vector<std::string> out;
        for (const auto& s : names) {
            if (s != "order_party") { out.push_back(s); continue; }
            if (has_order) continue;
            out.push_back("order");
            has_order = true;
        }
        return out;
    }
#ifdef AI4BAYESCODE_RCPP_MODULE
    static Rcpp::CharacterVector block_names_(const Rcpp::CharacterVector& names) {
        return Rcpp::CharacterVector(Rcpp::wrap(block_names_(
            Rcpp::as<std::vector<std::string>>(names))));
    }
#endif

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
        .constructor<arma::mat, arma::vec, std::string, double, int,
                      int, int, double, double,
                      arma::vec, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::mat, arma::vec, std::string, double, int,
                      int, int, double, double,
                      arma::vec, int, bool>(
            "Bayesian-network structure learning via Kuipers-Moffa 2017 "
            "partition MCMC (order_mcmc_block, default method) with BDeu "
            "scoring. Args: data (N x n integer 0..r-1), "
            "cardinalities (length n, r_i for each variable), structure_prior "
            "(\"uniform\" | \"fk_eq2\"), bdeu_alpha "
            "(default 1), max_parents (default 5), candidate_top_C (default "
            "20), family_cache_F (default 4000), gamma_prune_nats (default "
            "10), prob_adjacent_swap (default 0.5), initial_order (must be "
            "EMPTY numeric(0); load a partition with set_current(list(order "
            "= ..., order_party = ...))), rng_seed, keep_history. "
            "candidate_top_C, family_cache_F, gamma_prune_nats and "
            "prob_adjacent_swap are validated and passed to the block but do "
            "not change the draws of partition MCMC: the block uses every "
            "family up to max_parents without gamma pruning, and "
            "prob_adjacent_swap applies only to the order method.")
        .method("step", (void (OrderMCMCBN::*)())    &OrderMCMCBN::step, "Run one sweep.")
        .method("step", (void (OrderMCMCBN::*)(int)) &OrderMCMCBN::step, "Run n sweeps.")
        .method("get_current", &OrderMCMCBN::get_current,
                "Returns list(order = partition node sequence, 1-based, "
                "element 0 first (children before parents; rev(order) lists "
                "parents first), order_party = element sizes, "
                "sampled_DAG = n*n flattened COLUMN-MAJOR adjacency "
                "(A[i,j]=1 ⇔ j is parent of i; 1-based indices), "
                "log_score = log P(D | partition)).")
        .method("set_current", &OrderMCMCBN::set_current,
                "set_current(list(order = ..., order_party = ...)) loads that "
                "partition exactly (values as get_current() returns them; "
                "nodes are sorted within each element). 'order' alone loads "
                "nothing: it is accepted only when it equals the current node "
                "sequence of a partition with two or more elements, and is an "
                "error otherwise, including on a new model. sampled_DAG and "
                "log_score are derived and rejected. freeze() and unfreeze() "
                "accept 'order_party' as a name of the 'order' block, so "
                "ai4bayescode_new_frozen(..., fixed = list(order = ..., "
                "order_party = ...)) loads the partition and freezes the "
                "block at it.")
        .method("predict_at",  &OrderMCMCBN::predict_at,
                "v1: stub returning current sampled_DAG (1 x n² matrix). "
                "Full posterior-predictive simulation deferred to v1.2.1.")
        .method("get_dag",     &OrderMCMCBN::get_dag)
        .method("get_history", &OrderMCMCBN::get_history,
                "Returns list(order = node sequences, 1-based as in "
                "get_current(), one row per sweep when keep_history = TRUE, "
                "else the current sequence; order_log_score = log P(D | "
                "partition), one row per sweep when keep_history = TRUE).")
        AI4BAYESCODE_BIND_KERNEL_CONTROL(OrderMCMCBN);
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
        .def(pybind11::init<arma::mat, arma::vec, std::string, double, int,
                            int, int, double, double,
                            arma::vec, int, bool>(),
             pybind11::arg("data"),
             pybind11::arg("cardinalities"),
             pybind11::arg("structure_prior"),          // REQUIRED: "uniform" | "fk_eq2"
             pybind11::arg("bdeu_alpha")        = 1.0,
             pybind11::arg("max_parents")       = 5,
             pybind11::arg("candidate_top_C")   = 20,
             pybind11::arg("family_cache_F")    = 4000,
             pybind11::arg("gamma_prune_nats")  = 10.0,
             pybind11::arg("prob_adjacent_swap")= 0.5,
             pybind11::arg("initial_order")     = arma::vec(),
             pybind11::arg("rng_seed")          = 1,
             pybind11::arg("keep_history")      = false,
             "Bayesian-network structure learning via Kuipers-Moffa 2017 "
             "partition MCMC (order_mcmc_block, default method) with BDeu "
             "scoring. initial_order must be empty (np.zeros(0)): load a "
             "partition with set_current({'order': ..., 'order_party': ...}) "
             "after construction. candidate_top_C, family_cache_F, "
             "gamma_prune_nats and prob_adjacent_swap are validated and "
             "passed to the block but do not change the draws of partition "
             "MCMC: the block uses every family up to max_parents without "
             "gamma pruning, and prob_adjacent_swap applies only to the order "
             "method.")
        .def("step", (void (OrderMCMCBN::*)())    &OrderMCMCBN::step, "Run one sweep.")
        .def("step", (void (OrderMCMCBN::*)(int)) &OrderMCMCBN::step,        pybind11::arg("n_steps"))
        .def("get_current", &OrderMCMCBN::get_current,
             "Returns {'order': partition node sequence, 1-based, element 0 "
             "first (children before parents; order[::-1] lists parents "
             "first), 'order_party': element sizes, 'sampled_DAG': n*n "
             "flattened COLUMN-MAJOR adjacency (A[i,j]=1 <=> j is parent of "
             "i), 'log_score': log P(D | partition)}.")
        .def("set_current", &OrderMCMCBN::set_current, pybind11::arg("params"),
             "set_current({'order': ..., 'order_party': ...}) loads that "
             "partition exactly (values as get_current() returns them; nodes "
             "are sorted within each element). 'order' alone loads nothing: "
             "it is accepted only when it equals the current node sequence "
             "of a partition with two or more elements, and is an error "
             "otherwise, including on a new model. 'sampled_DAG' and "
             "'log_score' are derived and rejected. freeze() and unfreeze() "
             "accept 'order_party' as a name of the 'order' block, so "
             "new_frozen(..., fixed={'order': ..., 'order_party': ...}) "
             "loads the partition and freezes the block at it.")
        .def("predict_at",  &OrderMCMCBN::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",     &OrderMCMCBN::get_dag)
        .def("get_history", &OrderMCMCBN::get_history,
             "Returns {'order': node sequences, 1-based as in get_current(), "
             "one row per sweep when keep_history=True, else the current "
             "sequence; 'order_log_score': log P(D | partition), one row per "
             "sweep when keep_history=True}.")
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(OrderMCMCBN);
}
#endif

// ============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Honest recovery check for a STRUCTURE-LEARNING model: there is no scalar
//  parameter to recover, so we test that the sampler concentrates posterior
//  mass on the TRUE skeleton.
//
//  Truth: a 4-variable binary chain  X0 -> X1 -> X2 -> X3  (a strong Markov
//  chain). Each child copies its parent w.p. 0.9, flips w.p. 0.1; X0 is a
//  fair coin. The true skeleton (undirected edges) is exactly the three
//  chain links {0-1, 1-2, 2-3}; the three non-adjacent pairs {0-2, 0-3,
//  1-3} are NON-edges.
//
//  We run partition MCMC, sample one DAG per kept step, accumulate the
//  posterior SKELETON marginal P(i~j | D) for every unordered pair, and PASS
//  iff every true-edge marginal clears every non-edge marginal by a clear gap.
//  State is read via the FULL contract: get_current()["sampled_DAG"] (n*n
//  flattened, COLUMN-MAJOR, element (i,j) at index i + j*n; A[i,j]=1 ⇔ j is
//  parent of i).
// ============================================================================
#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

int main() {
    const std::size_t n   = 4;       // variables: chain X0->X1->X2->X3
    const std::size_t N   = 4000;    // samples
    const double      flip = 0.10;   // child disagrees with parent w.p. flip

    // ---- Simulate the chain BN -------------------------------------------
    std::mt19937_64 sim(123);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    arma::mat data(N, n, arma::fill::zeros);   // integer-valued (0/1) doubles
    for (std::size_t r = 0; r < N; ++r) {
        int prev = (U(sim) < 0.5) ? 1 : 0;   // X0 ~ Bernoulli(0.5)
        data(r, 0) = static_cast<double>(prev);
        for (std::size_t j = 1; j < n; ++j) {
            int xj = (U(sim) < flip) ? (1 - prev) : prev;  // copy parent, flip
            data(r, j) = static_cast<double>(xj);
            prev = xj;
        }
    }
    arma::vec cards(n);
    cards.fill(2.0);   // all binary

    // ---- Fit partition MCMC ----------------------------------------------
    OrderMCMCBN model(data, cards,
                      // The DAG prior is a required argument -- there is no
                      // default, because the two priors give different
                      // posteriors. This demo asks for the Friedman-Koller
                      // Eq 2 per-family prior; pass "uniform" for P(G)
                      // proportional to 1.
                      /*structure_prior=*/"fk_eq2",
                      /*bdeu_alpha=*/1.0,
                      /*max_parents=*/3,
                      /*candidate_top_C=*/20,
                      /*family_cache_F=*/4000,
                      /*gamma_prune_nats=*/10.0,
                      /*prob_adjacent_swap=*/0.5,
                      /*initial_order=*/arma::vec(),   // must be empty
                      /*rng_seed=*/7,
                      /*keep_history=*/false);

    model.step(2000);   // burn-in

    // Accumulate the posterior SKELETON marginal for each unordered pair.
    const int M = 8000;
    arma::mat skel(n, n, arma::fill::zeros);
    for (int s = 0; s < M; ++s) {
        model.step(1);
        // Read the sampled DAG via the FULL contract (column-major flatten:
        // element (i,j) at index i + j*n; A[i,j]=1 ⇔ j is parent of i).
        const auto gc = model.get_current();
        const arma::vec& dag_v = gc.at("sampled_DAG");
        auto A = [&](std::size_t i, std::size_t j) -> double {
            return dag_v[i + j * n];
        };
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
                if (A(i, j) != 0.0 || A(j, i) != 0.0) skel(i, j) += 1.0;
    }
    skel /= static_cast<double>(M);

    // ---- Score against the known skeleton --------------------------------
    // True skeleton edges of the chain: (0,1), (1,2), (2,3).
    auto is_true_edge = [](std::size_t i, std::size_t j) {
        return (j == i + 1);   // adjacent in the chain
    };

    std::printf("OrderMCMCBN demo: posterior skeleton marginals P(i~j | D)\n");
    double min_true_edge = 1.0;
    double max_non_edge  = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double p = skel(i, j);
            const bool te = is_true_edge(i, j);
            std::printf("   %zu ~ %zu : %.3f  [%s]\n", i, j, p,
                        te ? "TRUE edge" : "non-edge");
            if (te) min_true_edge = std::min(min_true_edge, p);
            else    max_non_edge  = std::max(max_non_edge,  p);
        }
    }
    std::printf("   min(true-edge marginal) = %.3f\n", min_true_edge);
    std::printf("   max(non-edge marginal)  = %.3f\n", max_non_edge);

    const bool ok = (min_true_edge > 0.80) &&
                    (max_non_edge  < 0.50) &&
                    (min_true_edge - max_non_edge > 0.30);
    std::printf("%s\n", ok
        ? "[demo PASS] order MCMC recovers the chain skeleton"
        : "[demo FAIL] skeleton not separated from non-edges");
    return ok ? 0 : 1;
}
#endif

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
//  DAG is in `get_current()` under "sampled_DAG".
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
// ============================================================================

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that simulates a known chain
// Bayesian network, fits the order_mcmc_block, accumulates posterior edge
// marginals, and checks skeleton recovery. No R / Python binding is built or
// required.

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/order_mcmc_block.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::order_mcmc_block;
using AI4BayesCode::order_mcmc_block_config;

// ----------------------------------------------------------------------------
//  Neutral-typed Tier-A wrapper. The underlying order_mcmc_block already
//  speaks AI4BayesCode::state_map / arma types, so the class carries the same
//  model wiring as the original Rcpp module (priors, BDeu score, block config)
//  with zero frontend types. main() drives it directly.
// ----------------------------------------------------------------------------
class OrderMCMCBN {
public:
    OrderMCMCBN(const arma::imat&  data,
                const arma::uvec&  cardinalities,
                double             bdeu_alpha,
                int                max_parents,
                int                candidate_top_C,
                int                family_cache_F,
                double             gamma_prune_nats,
                double             prob_adjacent_swap,
                int                rng_seed,
                bool               keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          impl_(std::make_unique<composite_block>("OrderMCMCBN")),
          keep_history_(keep_history)
    {
        const std::size_t N = data.n_rows;
        const std::size_t n = data.n_cols;
        if (n < 2)        throw std::runtime_error("data must have >= 2 columns");
        if (N < 1)        throw std::runtime_error("data must have >= 1 row");
        if (n > 64)       throw std::runtime_error("n > 64 not supported");
        if (cardinalities.n_elem != n)
            throw std::runtime_error("cardinalities length must equal ncol(data)");
        if (!(bdeu_alpha > 0)) throw std::runtime_error("bdeu_alpha must be > 0");
        if (max_parents < 0 || max_parents > 64)
            throw std::runtime_error("max_parents must be in [0, 64]");
        if (candidate_top_C < 1)
            throw std::runtime_error("candidate_top_C must be >= 1");
        if (family_cache_F < 1)
            throw std::runtime_error("family_cache_F must be >= 1");
        if (gamma_prune_nats < 0)
            throw std::runtime_error("gamma_prune_nats must be >= 0");
        if (prob_adjacent_swap < 0.0 || prob_adjacent_swap > 1.0)
            throw std::runtime_error("prob_adjacent_swap must be in [0, 1]");

        n_ = n;

        // ---- Build order_mcmc_block config -----------------------------
        order_mcmc_block_config cfg;
        cfg.name = "order";
        cfg.data = data;
        cfg.cardinalities = cardinalities;
        cfg.bdeu_alpha = bdeu_alpha;
        cfg.max_parents = static_cast<std::size_t>(max_parents);
        cfg.candidate_top_C = static_cast<std::size_t>(candidate_top_C);
        cfg.family_cache_F = static_cast<std::size_t>(family_cache_F);
        cfg.gamma_prune_nats = gamma_prune_nats;
        cfg.prob_adjacent_swap = prob_adjacent_swap;
        cfg.use_structure_prior = true;
        cfg.init_rng_seed = static_cast<std::uint64_t>(rng_seed);

        // No upstream data dependencies for the block.
        impl_->add_child(std::make_unique<order_mcmc_block>(std::move(cfg)));

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        if (n_steps < 0) throw std::runtime_error("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // current() exposes the order (0-based) and the per-step sampled DAG as a
    // dense n×n adjacency (A(i,j)=1 ⇔ j is parent of i), matching the original
    // module's semantics (minus the 1-based R re-indexing).
    const order_mcmc_block& block() const {
        const auto* blk = dynamic_cast<const order_mcmc_block*>(&impl_->child(0));
        if (!blk) throw std::runtime_error("internal: child 0 is not order_mcmc_block");
        return *blk;
    }

    arma::uvec order() const { return block().order(); }

    arma::umat sampled_DAG() const {
        const auto& dag = block().sampled_dag();
        arma::umat A(n_, n_, arma::fill::zeros);
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j)
                A(i, j) = (dag[i] & (1ULL << j)) ? 1u : 0u;
        return A;
    }

    double log_score() const { return block().current_log_score(); }

    AI4BayesCode::dag_info     get_dag()     const { return impl_->get_dag(); }
    AI4BayesCode::history_map  get_history() const { return impl_->get_history(); }

private:
    std::mt19937_64                  rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      n_ = 0;
};

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Honest recovery check for a STRUCTURE-LEARNING model: there is no scalar
//  parameter to recover, so we test that the sampler concentrates posterior
//  mass on the TRUE skeleton.
//
//  Truth: a 4-variable binary chain  X0 -> X1 -> X2 -> X3  (a strong
//  Markov chain). Each child copies its parent w.p. 0.9, flips w.p. 0.1;
//  X0 is a fair coin. The true skeleton (undirected edges) is exactly the
//  three chain links {0-1, 1-2, 2-3}; the three non-adjacent pairs
//  {0-2, 0-3, 1-3} are NON-edges.
//
//  We run order MCMC, sample one DAG per kept step, accumulate the
//  posterior SKELETON marginal P(i~j | D) for every unordered pair, and
//  PASS iff every true-edge marginal clears every non-edge marginal by a
//  clear gap. This is the standard order-MCMC inclusion-probability check
//  (cf. order_mcmc_block.hpp diagnostics D1/D2), not a fudged constant.
//==============================================================================
#include <cstdio>
#include <algorithm>

int main() {
    const std::size_t n   = 4;       // variables: chain X0->X1->X2->X3
    const std::size_t N   = 4000;    // samples
    const double      flip = 0.10;   // child disagrees with parent w.p. flip

    // ---- Simulate the chain BN -------------------------------------------
    std::mt19937_64 sim(123);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    arma::imat data(N, n, arma::fill::zeros);
    for (std::size_t r = 0; r < N; ++r) {
        int prev = (U(sim) < 0.5) ? 1 : 0;   // X0 ~ Bernoulli(0.5)
        data(r, 0) = prev;
        for (std::size_t j = 1; j < n; ++j) {
            int xj = (U(sim) < flip) ? (1 - prev) : prev;  // copy parent, flip
            data(r, j) = xj;
            prev = xj;
        }
    }
    arma::uvec cards(n);
    cards.fill(2);   // all binary

    // ---- Fit order MCMC --------------------------------------------------
    OrderMCMCBN model(data, cards,
                      /*bdeu_alpha=*/1.0,
                      /*max_parents=*/3,
                      /*candidate_top_C=*/20,
                      /*family_cache_F=*/4000,
                      /*gamma_prune_nats=*/10.0,
                      /*prob_adjacent_swap=*/0.5,
                      /*rng_seed=*/7,
                      /*keep_history=*/false);

    model.step(2000);   // burn-in

    // Accumulate the posterior SKELETON marginal for each unordered pair.
    const int M = 8000;
    arma::mat skel(n, n, arma::fill::zeros);
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const arma::umat A = model.sampled_DAG();   // A(i,j)=1 ⇔ j parent of i
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
                if (A(i, j) || A(j, i)) skel(i, j) += 1.0;
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

    // PASS: every true edge strongly included, every non-edge clearly
    // separated. The chain signal is strong (flip=0.10), so the true links
    // should sit near 1.0 and the non-adjacent pairs well below.
    const bool ok = (min_true_edge > 0.80) &&
                    (max_non_edge  < 0.50) &&
                    (min_true_edge - max_non_edge > 0.30);
    std::printf("%s\n", ok
        ? "[demo PASS] order MCMC recovers the chain skeleton"
        : "[demo FAIL] skeleton not separated from non-edges");
    return ok ? 0 : 1;
}

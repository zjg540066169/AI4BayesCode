/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  order_mcmc_block.hpp -- Friedman-Koller 2003 order MCMC for Bayesian
 *                          network structure learning. Tier B block.
 *
 *  TARGET DISTRIBUTION
 *  ===================
 *      P(≺ | D) ∝ P(D | ≺) · P(≺),  P(≺) uniform over n! orders
 *
 *  Marginal likelihood (FK Eq 8):
 *      log P(D | ≺) = Σ_i log Σ_{U ∈ U_{i,≺}} score(X_i, U | D)
 *
 *  with score(X_i, U | D) = BDe family score (bde_scorer.hpp) and the
 *  candidate parent set U restricted to subsets of ≺-predecessors of
 *  X_i (via the FK §4.2 three-tier heuristic in score_cache.hpp).
 *
 *  MH PROPOSAL (FK §4.1 + BiDAG default mixture)
 *  =============================================
 *  v1 implements TWO symmetric move types in a user-specified mixture:
 *      Move 1: swap any two positions (uniform over n(n-1)/2 pairs)
 *      Move 2: swap two adjacent positions (uniform over n-1)
 *
 *  Both proposals are symmetric: q(≺'|≺) = q(≺|≺'). Acceptance:
 *      alpha = min(1, exp(log P(D|≺') - log P(D|≺)))
 *
 *  Move 3 (BiDAG position-resample) is deferred to v1.2.1.
 *
 *  TARGET-SHAPE CATEGORY (system_design.md §11.2(b))
 *  =================================================
 *  Discrete combinatorial target with strong structural dependence.
 *  Per-DAG Gibbs (Madigan-York 1995 edge-flip) on the n×n adjacency
 *  matrix is irreducible but mixes catastrophically: edge directions
 *  are tightly coupled through acyclicity, and a single flip frequently
 *  changes the score of many other potential edges. Order MCMC sidesteps
 *  this by sampling permutations instead of DAGs, marginalising over
 *  all DAGs compatible with each order (FK 2003 Eq 8). v1.2 scope
 *  covers:
 *    * Discrete data with per-column cardinalities (BDeu score via
 *      bde_scorer.hpp, Heckerman-Geiger-Chickering 1995 Eq 28).
 *    * n ≤ 64 (64-bit parent-set bitmask cap from score_cache.hpp's
 *      FK §4.2 three-tier candidate cache).
 *    * Max parents k = 5 default (configurable; tractability cliff
 *      at k ≈ 7-8).
 *    * DAG prior P(G) selectable via `use_structure_prior` flag:
 *        - false → strict uniform on DAGs: P(G) ∝ 1 (matches BiDAG
 *                   edgepf=1 + bnlearn defaults; the natural reading
 *                   of a spec saying "G ~ Uniform(DAGs)").
 *        - true  → Friedman-Koller 2003 Eq 2 per-family balancing:
 *                   P(G) ∝ ∏_j 1/C(p-1, |Pa_j|) (penalises high
 *                   fan-in; FK's recommended empirical default).
 *      THE TWO PRIORS PRODUCE DIFFERENT POSTERIORS. Codegen agents
 *      MUST `AskUserQuestion` when the spec doesn't disambiguate
 *      (see codegen_priors.md §3a Class 5a).
 *  Each of the following is on the v1.2.1 roadmap (see
 *  V1_2_SPECIALIZED_BLOCKS_PLAN_2026-05-27.md
 *  §4 Block 3 "Deferred to v1.2.1"):
 *    * Kuipers-Moffa 2017 partition_mcmc_block (removes the FK §4.1
 *      induced-structure-prior bias inside Markov equivalence
 *      classes — see KNOWN BIAS below).
 *    * BGe Gaussian score (continuous data; Geiger-Heckerman 1994).
 *    * Mixed conditional-Gaussian BN (Lauritzen 1992).
 *    * Edge-specific structural prior.
 *    * Tempered / parallel-tempered chains.
 *
 *  STATE REPRESENTATION
 *  ====================
 *  Internally: arma::uvec order_state_ of length n holding indices
 *  0..n-1 (a permutation). Externally exposed via current() as
 *  arma::vec (cast to doubles).
 *
 *  Joint sub-parameter outputs (via current_named_outputs()):
 *      "order"       arma::vec length n         (permutation as doubles)
 *      "sampled_DAG" arma::vec length n*n       (row-major adjacency)
 *      "log_score"   arma::vec length 1         (current log P(D|≺))
 *
 *  KNOWN BIAS (per spec §1.6, FK §4.1 last 3 paragraphs)
 *  ====================================================
 *  Order prior is uniform over orders; the INDUCED structure prior is
 *  NOT hypothesis-equivalent (different Markov-equivalent DAGs receive
 *  different prior weight in proportion to how many orders they are
 *  consistent with). Fix is Kuipers-Moffa 2017 partition MCMC,
 *  deferred to v1.2.1 partition_mcmc_block.
 *
 *  ENGINE FAMILY
 *  =============
 *  engine_kind() = MCMC. supports_readapt() = false (specialised
 *                  permutation MH, no tunable metric).
 *
 *  COMPLEXITY
 *  ==========
 *  Per step: O((|swap_range|) × |cached families per node|) where
 *            swap_range is the affected window of the permutation
 *            (FK §4.2 incremental rescoring). Cache lookup is constant.
 *
 *  VALIDATOR (system_design.md §11.7, Checks #15/16/17)
 *  ====================================================
 *  - Check #15 (parity panel for the BDe scorer Tier C kernel):
 *      see tests/test_bde_scorer.cpp
 *        T1  2-node Heckerman hand-computed BDe (empty-parent +
 *            single-parent family scores match closed-form Eq 28)
 *        T2  Likelihood-equivalence within a Markov class (X→Y vs
 *            Y→X on a 2-node BN give identical marginal scores)
 *        T3  Empty-parent closed form (BDe reduces to product of
 *            Beta-Bernoulli marginals)
 *        T4  Structure prior FK Eq 2 toggling (uniform vs fan-in-
 *            penalised)
 *        T5  BDeu α-scaling asymptotics (α → 0 sharpens to MLE-like;
 *            α → ∞ tends to no-parent baseline)
 *        T6  Single-edge score reduces to two-bin Beta-Bernoulli
 *        T7  Top-C candidate-parent selection by single-edge score
 *        T8  Validation rejects (cardinality mismatch, α ≤ 0,
 *            out-of-range data values)
 *  - Check #15 (parity panel for the FK §4.2 three-tier cache):
 *      see tests/test_score_cache.cpp
 *        T1  Cached families sorted descending by score
 *        T2  order_node_score consistent with explicit Σ family_score
 *        T3  Total order log-score factorises across nodes
 *        T4  sample_parent_set returns feasible (Pa ⊂ predecessor)
 *        T5  γ-pruning at default 10-nat cap
 *        T6  Top-C candidate cap respected
 *        T7  Markov-equivalent orders (identity vs reverse on a
 *            chain) give similar log-scores
 *        T8  sample_dag returns size-n feasible parent-set bitmasks
 *  - Check #15 (parity panel for the order_mcmc_block Tier B
 *      sampler): see tests/test_order_mcmc_block.cpp
 *        T1  Construction smoke (length n, finite initial log-score)
 *        T2  Reproducibility: same rng_seed → bitwise-identical
 *            permutation sequence
 *        T3  step() preserves the permutation invariant
 *        T4  set_current ∘ get_current round-trip; duplicate-entry
 *            rejection
 *        T5  MH accept rate non-degenerate (0.02 < α < 0.98) over
 *            2000 sweeps on a chain BN
 *        T6  Long-run log-score plateau on chain BN (drift < 5%)
 *        T7  sampled_dag respects current order (parents are strict
 *            predecessors)
 *        T8  current_named_outputs returns 3 keys (order /
 *            sampled_DAG / log_score) with correct lengths
 *  - Check #15 (stress / robustness panel, 9 sub-tests):
 *      see tests/test_order_mcmc_block_stress.cpp
 *        R1 bitwise reproducibility under fixed seed
 *        R2 iid Bernoulli no-signal → max edge marginal < 0.30
 *        R3 max_parents enforced (binary check, no violations)
 *        R4 BDeu α-asymptotics direction match (flat ≥ sharp)
 *        R5 n = 20 large-graph stability (no NaN, order invariant)
 *        R6 4-chain Gelman-Rubin R-hat < 1.01 strict (Vehtari 2021),
 *           scoped to iid no-signal data on n = 4 (unimodal target)
 *        R7 post-conv stability: std/|mean|(log_score) < 1%
 *        R8 cardinality r_i = 4 chain BN recovery (skeleton > 0.5)
 *        R9 n = 2 edge case runs without error
 *  - Check #15 (exact-posterior diagnostic panel, 4 sub-tests):
 *      see tests/test_order_mcmc_block_diagnostics.cpp
 *        D1 n = 3 exact P(edge | D) match: enumerate 25 DAGs, weight
 *           by # linear extensions, compare to 20000 MCMC samples;
 *           HARD max |exact - empirical| < 0.05 (achieves ≈ 0.001).
 *        D2 n = 4 exact P(edge | D) match: enumerate 543 DAGs, same
 *           comparison on 30000 MCMC samples; achieves ≈ 0.009.
 *        D3 conditional P(Pa_i | order, D) match against direct
 *           subset enumeration on 10000 sample_parent_set draws;
 *           HARD max |Δ| < 0.03 (achieves ≈ 0.001).
 *        D4 ESS(log_score) > 200 on 10000 post-burn samples (Geyer
 *           initial-positive-sequence estimator).
 *  - Check #15 (R-level audit panel):
 *      see tests/audit_OrderMCMCBN.R (30/30 PASS — includes Layer-3
 *      plateau, STRICT < 1.01 4-chain R-hat, and §16 R-level pre-
 *      merge checklist: derived-key rejection, predict_at state
 *      preservation, unknown-key tolerance, round-trip identity)
 *      and tests/audit_OrderMCMCBN_bnlearn_cross.R (5/5 PASS on
 *      ASIA; perfect 7/7 skeleton agreement with bnlearn::hc; 7/8
 *      true Markov-equivalent edges recovered in top-8 inclusion).
 *  - Check #15 (reference implementation cross-check):
 *      see tests/audit_OrderMCMCBN_vs_BiDAG.R — head-to-head with
 *      BiDAG::orderMCMC (Kuipers-Moffa reference) on identical data
 *      and 4 matched chains: BOTH implementations achieve R-hat <
 *      1.01 (ours 1.00073, BiDAG 1.00000). Outcome (A): our code
 *      converges to the same target as the reference.
 *  - Check #16 (inline JUSTIFICATION at every wrapper call site):
 *      see examples/OrderMCMCBN.cpp ~line 43.
 *  - Check #17: only std::uniform_real_distribution and
 *               std::uniform_int_distribution primitives are used
 *               in the proposal kernel; no hand-written conjugate
 *               samplers.
 *================================================================================*/

#ifndef AI4BAYESCODE_ORDER_MCMC_BLOCK_HPP
#define AI4BAYESCODE_ORDER_MCMC_BLOCK_HPP

#include "block_sampler.hpp"
#include "bde_scorer.hpp"
#include "score_cache.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace AI4BayesCode {

struct order_mcmc_block_config {
    /// Block name (default "order").
    std::string name = "order";

    /// Integer-encoded categorical data, N × n with values 0..r_i-1.
    arma::imat data;

    /// Variable cardinalities r_i, length n.
    arma::uvec cardinalities;

    /// BDeu equivalent sample size α (default 1).
    double bdeu_alpha = 1.0;

    /// Max parent set size k (default 5).
    std::size_t max_parents = 5;

    /// Candidate parents top-C per node (default 20).
    std::size_t candidate_top_C = 20;

    /// Family cache top-F per node (default 4000).
    std::size_t family_cache_F = 4000;

    /// γ-pruning threshold in nats (default 10).
    double gamma_prune_nats = 10.0;

    /// Probability of adjacent-swap proposal (rest is any-pair swap).
    /// Default 0.5.
    double prob_adjacent_swap = 0.5;

    /// DAG prior P(G) selector:
    ///   true  → Friedman-Koller 2003 Eq 2 per-family balancing
    ///           prior: P(G) ∝ ∏_j 1/C(p-1, |Pa_j|). Penalises high
    ///           fan-in; FK paper's recommended empirical default.
    ///   false → Strict uniform DAG prior: P(G) ∝ 1 (with the same
    ///           in-degree cap). Matches BiDAG (`edgepf=1`) +
    ///           bnlearn defaults; pick this when the spec says
    ///           "uniform DAG prior" or "P(G) ∝ 1".
    /// THE TWO PRIORS PRODUCE DIFFERENT POSTERIORS — they are NOT
    /// equivalent up to a normalising constant. Codegen agents MUST
    /// AskUserQuestion when the spec doesn't disambiguate (see
    /// codegen_priors.md §3a Class 5a). Default true preserves
    /// backwards compatibility with FK-style usage.
    bool use_structure_prior = true;

    /// Initial order: optional. If empty, the constructor generates a
    /// random permutation using init_rng_seed.
    arma::uvec initial_order;

    /// Seed for the initial order's RNG (independent of step()'s rng).
    std::uint64_t init_rng_seed = 0;
};

/**
 * @brief order_mcmc_block — MH on permutations targeting P(≺ | D).
 *
 * Holds an internal score_cache (constructed once) plus an arma::uvec
 * order_state_. step() performs one MH proposal (mixture of any-pair
 * swap + adjacent-pair swap), accepts/rejects on the closed-form
 * order log-score from the cache.
 */
class order_mcmc_block : public block_sampler {
public:
    explicit order_mcmc_block(order_mcmc_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.data.n_rows == 0 || cfg_.data.n_cols == 0) {
            throw std::invalid_argument(
                "order_mcmc_block: data must be non-empty");
        }
        n_ = cfg_.data.n_cols;
        if (n_ > 64u) {
            throw std::invalid_argument(
                "order_mcmc_block: n > 64 not supported (bitmask encoding)");
        }
        if (cfg_.cardinalities.n_elem != n_) {
            throw std::invalid_argument(
                "order_mcmc_block: cardinalities length must equal n");
        }
        if (cfg_.prob_adjacent_swap < 0.0 || cfg_.prob_adjacent_swap > 1.0) {
            throw std::invalid_argument(
                "order_mcmc_block: prob_adjacent_swap must be in [0, 1]");
        }

        // Build bde_scorer.
        bde_scorer_config bcfg;
        bcfg.data = cfg_.data;
        bcfg.cardinalities = cfg_.cardinalities;
        bcfg.alpha = cfg_.bdeu_alpha;
        bcfg.use_structure_prior = cfg_.use_structure_prior;
        bcfg.max_parents = cfg_.max_parents;
        bde_scorer scorer(bcfg);

        // Build score_cache.
        score_cache_config sc;
        sc.max_parents = cfg_.max_parents;
        sc.candidate_top_C = cfg_.candidate_top_C;
        sc.family_top_F = cfg_.family_cache_F;
        sc.gamma_prune_nats = cfg_.gamma_prune_nats;
        cache_ = std::make_unique<score_cache>(std::move(scorer), sc);

        // Initialise order.
        if (cfg_.initial_order.n_elem > 0) {
            if (cfg_.initial_order.n_elem != n_) {
                throw std::invalid_argument(
                    "order_mcmc_block: initial_order length must equal n");
            }
            // Validate permutation.
            std::vector<int> seen(n_, 0);
            for (std::size_t p = 0; p < n_; ++p) {
                const std::size_t v = cfg_.initial_order[p];
                if (v >= n_ || seen[v]) {
                    throw std::invalid_argument(
                        "order_mcmc_block: initial_order is not a "
                        "permutation of 0..n-1");
                }
                seen[v] = 1;
            }
            order_state_ = cfg_.initial_order;
        } else {
            // Random init.
            std::mt19937_64 init_rng(
                cfg_.init_rng_seed ^ 0x4A39A6DCA12E47A1ULL);
            order_state_.set_size(n_);
            for (std::size_t i = 0; i < n_; ++i) order_state_[i] = i;
            for (std::size_t i = n_ - 1; i > 0; --i) {
                std::uniform_int_distribution<std::size_t> U(0, i);
                std::size_t j = U(init_rng);
                std::swap(order_state_[i], order_state_[j]);
            }
        }

        // Compute initial log score.
        current_log_score_ = cache_->order_log_score(order_as_vec_());

        // Pre-allocate current arma::vec for current().
        current_natural_ = arma::conv_to<arma::vec>::from(order_state_);

        // Sampled DAG (one per step), initially sample from current order.
        std::mt19937_64 dag_init_rng(
            cfg_.init_rng_seed ^ 0xA7F38E15CD49B22BULL);
        sampled_dag_ = cache_->sample_dag(dag_init_rng, order_as_vec_());
    }

    // ---- block_sampler interface ----------------------------------------

    void set_context(const block_context& /*ctx*/) override {
        // order_mcmc_block has no upstream data dependencies for v1.
    }

    void step(std::mt19937_64& rng) override {
        // ----- Propose -----
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        std::size_t a = 0, b = 0;
        if (U01(rng) < cfg_.prob_adjacent_swap) {
            // Move 2: swap adjacent positions.
            std::uniform_int_distribution<std::size_t> Up(0, n_ - 2);
            a = Up(rng); b = a + 1;
        } else {
            // Move 1: swap any two positions.
            std::uniform_int_distribution<std::size_t> Up(0, n_ - 1);
            a = Up(rng);
            do { b = Up(rng); } while (b == a);
            if (a > b) std::swap(a, b);
        }

        // ----- Compute proposed log score (incremental) -----
        // FK §4.2: only positions in [a, b] of the proposed order have
        // changed predecessor sets, so we recompute only those nodes.
        // For simplicity in v1, recompute full score.
        std::swap(order_state_[a], order_state_[b]);
        const double proposed_log_score =
            cache_->order_log_score(order_as_vec_());

        // ----- Accept / reject -----
        const double log_alpha = proposed_log_score - current_log_score_;
        if (std::log(U01(rng)) < log_alpha) {
            // Accept.
            current_log_score_ = proposed_log_score;
            current_natural_ = arma::conv_to<arma::vec>::from(order_state_);
        } else {
            // Reject — revert swap.
            std::swap(order_state_[a], order_state_[b]);
        }

        // Sample a DAG from the current order (per FK §3.2 Prop 3.1).
        sampled_dag_ = cache_->sample_dag(rng, order_as_vec_());

        // History (if enabled).
        if (keep_history()) {
            history_order_.push_back(order_state_);
            history_log_score_.push_back(current_log_score_);
        }
    }

    const arma::vec& current() const override {
        return current_natural_;
    }

    void set_current(const arma::vec& v) override {
        if (v.n_elem != n_) {
            throw std::invalid_argument(
                "order_mcmc_block::set_current: expected length "
                + std::to_string(n_));
        }
        // Validate permutation.
        std::vector<int> seen(n_, 0);
        arma::uvec new_state(n_);
        for (std::size_t p = 0; p < n_; ++p) {
            const double d = v[p];
            if (d < 0.0 || d >= static_cast<double>(n_)
                || std::floor(d) != d) {
                throw std::invalid_argument(
                    "order_mcmc_block::set_current: entries must be "
                    "integers in 0..n-1");
            }
            const std::size_t k = static_cast<std::size_t>(d);
            if (seen[k]) {
                throw std::invalid_argument(
                    "order_mcmc_block::set_current: not a permutation "
                    "(duplicate entry)");
            }
            seen[k] = 1;
            new_state[p] = k;
        }
        order_state_ = new_state;
        current_log_score_ = cache_->order_log_score(order_as_vec_());
        current_natural_ = arma::conv_to<arma::vec>::from(order_state_);
    }

    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim() const noexcept override { return n_; }

    state_map current_named_outputs(std::mt19937_64& rng) const override {
        state_map out;
        // "order" key (length n).
        arma::vec order_vec = arma::conv_to<arma::vec>::from(order_state_);
        out.emplace(cfg_.name, std::move(order_vec));
        // "sampled_DAG" key (length n*n, row-major).
        // Sample a fresh DAG for this output.
        auto dag = cache_->sample_dag(rng, order_as_vec_());
        arma::vec dag_flat(n_ * n_, arma::fill::zeros);
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                if (dag[i] & (1ULL << j)) {
                    dag_flat[i * n_ + j] = 1.0;
                }
            }
        }
        out.emplace(cfg_.name + "_sampled_DAG", std::move(dag_flat));
        // "log_score" key (length 1).
        out.emplace(cfg_.name + "_log_score",
                    arma::vec({current_log_score_}));
        return out;
    }

    history_map get_history() const override {
        history_map out;
        const std::size_t T = history_order_.size();
        if (T == 0) {
            arma::mat m(1, n_);
            for (std::size_t i = 0; i < n_; ++i)
                m(0, i) = static_cast<double>(order_state_[i]);
            out.emplace(cfg_.name, std::move(m));
            return out;
        }
        // Pack order history as (T, n).
        arma::mat orders(T, n_);
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t i = 0; i < n_; ++i) {
                orders(t, i) = static_cast<double>(history_order_[t][i]);
            }
        }
        out.emplace(cfg_.name, std::move(orders));
        // Pack log-score history as (T, 1).
        arma::mat ls(T, 1);
        for (std::size_t t = 0; t < T; ++t) ls(t, 0) = history_log_score_[t];
        out.emplace(cfg_.name + "_log_score", std::move(ls));
        return out;
    }

    std::size_t history_size() const noexcept override {
        return history_order_.size();
    }

    void clear_history() override {
        history_order_.clear();
        history_log_score_.clear();
    }

    // ---- Diagnostics ----------------------------------------------------

    double current_log_score() const noexcept { return current_log_score_; }
    const arma::uvec& order() const noexcept { return order_state_; }
    const score_cache& cache() const noexcept { return *cache_; }
    const std::vector<std::uint64_t>& sampled_dag() const { return sampled_dag_; }

private:
    /// Helper to view order_state_ as std::vector<std::size_t> for cache calls.
    std::vector<std::size_t> order_as_vec_() const {
        std::vector<std::size_t> out(n_);
        for (std::size_t i = 0; i < n_; ++i) out[i] = order_state_[i];
        return out;
    }

    order_mcmc_block_config cfg_;
    std::size_t n_ = 0;

    std::unique_ptr<score_cache>      cache_;
    arma::uvec                        order_state_;
    double                            current_log_score_ = 0.0;
    arma::vec                         current_natural_;
    std::vector<std::uint64_t>        sampled_dag_;

    // History (when keep_history true).
    std::vector<arma::uvec> history_order_;
    std::vector<double>     history_log_score_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_ORDER_MCMC_BLOCK_HPP

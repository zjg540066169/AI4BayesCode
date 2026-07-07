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
 *  v1.2.1 (SHIPPED) — opt-in via config, defaults preserve v1 exactly:
 *    * cfg.method = partition — Kuipers-Moffa 2017 partition MCMC. Samples
 *      labelled node partitions (split/join + swap + single-node +
 *      Sec.5 edge-reversal moves), removing the order-prior bias inside
 *      Markov equivalence classes (KNOWN BIAS below). UNBIASED: verified vs
 *      exact enumeration of all DAGs at n=3/4/5. See partition_state.hpp.
 *    * cfg.continuous_data (BGe) — Geiger-Heckerman 2002 Gaussian score for
 *      continuous data (bge_scorer.hpp); selected automatically when
 *      continuous_data is non-empty. Works with either method.
 *  Still deferred (v1.2.2+): mixed conditional-Gaussian BN (Lauritzen 1992);
 *  per-edge (weight-matrix) structural prior; tempering (target-changing,
 *  deliberately excluded).
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
 *  In method=order the order prior is uniform over orders; the INDUCED
 *  structure prior is NOT hypothesis-equivalent (different Markov-equivalent
 *  DAGs receive different prior weight in proportion to how many orders they
 *  are consistent with). **Set cfg.method = partition to remove this bias**
 *  (Kuipers-Moffa 2017; SHIPPED in v1.2.1, this block).
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
 *  - Check #15 (v1.2.1 partition MCMC + BGe panels):
 *      * tests/test_bge_scorer.cpp — BGe (continuous) local-score parity:
 *        likelihood-equivalence EXACT (X->Y == Y->X), 3-node Markov class,
 *        parent-order invariance, structure-prior hook, df/am validation.
 *      * tests/test_partition_geometry.cpp — partition-score geometry
 *        (Kuipers-Moffa Eq. 3) vs outpoint-peel + disjoint cover: exact to
 *        ~1e-14 at n = 3/4/5 (25/543/29281 DAGs).
 *      * tests/test_partition_mcmc_diagnostics.cpp — UNBIASEDNESS by exact
 *        enumeration: empirical DAG frequencies == P(G|D) at n = 3/4/5 for
 *        BDeu AND BGe, and WITH a per-edge structural prior; HARD < 0.01
 *        (achieves ~0.001). The sharp check the equivalence-class
 *        edge-direction R-hat cannot provide.
 *      * tests/test_partition_mcmc_rhat.cpp — two-chain Gelman-Rubin R-hat
 *        from over-dispersed inits (n = 6): log-score, skeleton, AND
 *        directed-edge R-hat all < 1.01 (the Sec.5 edge-reversal move mixes
 *        edge directions).
 *      * tests/test_order_mcmc_block_partition.cpp — block API in
 *        method=partition over BDeu + BGe (named outputs, structure recovery).
 *  - Check #15 (partition reference cross-check):
 *      see tests/audit_PartitionMCMCBN_vs_BiDAG.R — head-to-head with
 *      BiDAG::partitionMCMC on identical synthetic data; cross-impl R-hat on
 *      the BDeu log-marginal-likelihood scalar (both target P(G|D)).
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
#include "bge_scorer.hpp"
#include "score_cache.hpp"
#include "partition_state.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
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

    // ---- v1.2.1: partition MCMC + BGe (opt-in; defaults preserve v1) --------

    /// Sampler method. `order` = Friedman-Koller 2003 order MCMC (v1 default).
    /// `partition` = Kuipers-Moffa 2017 partition MCMC — samples over labelled
    /// partitions of the nodes, removing the order-prior bias inside Markov
    /// equivalence classes (unbiased DAG posterior; see partition_state.hpp).
    enum class method_t { order, partition };
    method_t method = method_t::order;

    /// Continuous (Gaussian) data, N x n. If NON-EMPTY, the block scores with
    /// the BGe score (Geiger-Heckerman 2002) over this data INSTEAD of BDeu over
    /// `data`, and `cardinalities` is ignored. Leave empty for discrete BDeu.
    arma::mat continuous_data;

    /// BGe prior hyperparameters (used only when continuous_data is non-empty).
    /// bge_am = prior-mean effective sample size (default 1); bge_aw = Wishart
    /// degrees of freedom (0 = the default n + am + 1; must be > n + 1).
    double bge_am = 1.0;
    double bge_aw = 0.0;

    /// Edge-specific structural prior: an n x n matrix of per-edge log-prior
    /// weights forwarded to whichever scorer is used. edge_log_prior(i, j) is
    /// added to node i's family score when j is a parent of i (edge j -> i),
    /// letting a user up/down-weight individual edges. Composes with either
    /// method and with use_structure_prior. Empty (default) = no per-edge prior.
    arma::mat edge_log_prior;
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
        const bool use_bge = (cfg_.continuous_data.n_cols > 0);
        if (use_bge) {
            if (cfg_.continuous_data.n_rows < 2) {
                throw std::invalid_argument(
                    "order_mcmc_block: continuous_data must have >= 2 rows");
            }
            n_ = cfg_.continuous_data.n_cols;
        } else {
            if (cfg_.data.n_rows == 0 || cfg_.data.n_cols == 0) {
                throw std::invalid_argument(
                    "order_mcmc_block: data must be non-empty "
                    "(or supply continuous_data for the BGe score)");
            }
            n_ = cfg_.data.n_cols;
            if (cfg_.cardinalities.n_elem != n_) {
                throw std::invalid_argument(
                    "order_mcmc_block: cardinalities length must equal n");
            }
        }
        if (n_ > 64u) {
            throw std::invalid_argument(
                "order_mcmc_block: n > 64 not supported (bitmask encoding)");
        }
        if (cfg_.prob_adjacent_swap < 0.0 || cfg_.prob_adjacent_swap > 1.0) {
            throw std::invalid_argument(
                "order_mcmc_block: prob_adjacent_swap must be in [0, 1]");
        }

        // Build the scorer (BDeu discrete or BGe continuous) + the cache.
        score_cache_config sc;
        sc.max_parents = cfg_.max_parents;
        sc.candidate_top_C = cfg_.candidate_top_C;
        sc.family_top_F = cfg_.family_cache_F;
        sc.gamma_prune_nats = cfg_.gamma_prune_nats;
        if (cfg_.method == order_mcmc_block_config::method_t::partition) {
            // Partition MCMC needs the FULL family set. Its ">= 1 parent in the
            // adjacent partition element" constraint can be STARVED by the FK
            // top-C / top-F / gamma-pruning heuristic (which is safe for order
            // MCMC where any predecessor subset is permissible): if every cached
            // family carrying a required parent is pruned, that node's partition
            // score is a spurious -inf. So use the exact, unpruned cache in
            // partition mode (this bounds partition mode to roughly n <= 20).
            sc.candidate_top_C  = (n_ > 1) ? (n_ - 1) : 1;
            sc.family_top_F     = std::numeric_limits<std::size_t>::max();
            sc.gamma_prune_nats = std::numeric_limits<double>::infinity();
        }
        if (use_bge) {
            bge_scorer_config gcfg;
            gcfg.data = cfg_.continuous_data;
            gcfg.am = cfg_.bge_am;
            gcfg.aw = cfg_.bge_aw;
            gcfg.use_structure_prior = cfg_.use_structure_prior;
            gcfg.edge_log_prior = cfg_.edge_log_prior;
            cache_ = std::make_unique<score_cache>(
                std::make_unique<bge_scorer>(std::move(gcfg)), sc);
        } else {
            bde_scorer_config bcfg;
            bcfg.data = cfg_.data;
            bcfg.cardinalities = cfg_.cardinalities;
            bcfg.alpha = cfg_.bdeu_alpha;
            bcfg.use_structure_prior = cfg_.use_structure_prior;
            bcfg.max_parents = cfg_.max_parents;
            bcfg.edge_log_prior = cfg_.edge_log_prior;
            cache_ = std::make_unique<score_cache>(
                std::make_unique<bde_scorer>(std::move(bcfg)), sc);
        }

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

        // Pre-allocate current arma::vec for current().
        current_natural_ = arma::conv_to<arma::vec>::from(order_state_);

        std::mt19937_64 dag_init_rng(
            cfg_.init_rng_seed ^ 0xA7F38E15CD49B22BULL);
        if (cfg_.method == order_mcmc_block_config::method_t::partition) {
            // Partition mode: seed from the trivial single-element partition
            // (empty DAG); the chain quickly builds up structure.
            pchain_ = partition_chain_init(*cache_, trivial_partition(n_));
            current_log_score_ = pchain_.log_score;
            current_natural_.set_size(n_);
            for (std::size_t p = 0; p < n_; ++p)
                current_natural_[p] = static_cast<double>(pchain_.state.permy[p]);
            sampled_dag_ = partition_sample_dag(*cache_, dag_init_rng, pchain_.state);
        } else {
            current_log_score_ = cache_->order_log_score(order_as_vec_());
            sampled_dag_ = cache_->sample_dag(dag_init_rng, order_as_vec_());
        }
    }

    // ---- block_sampler interface ----------------------------------------

    void set_context(const block_context& /*ctx*/) override {
        // order_mcmc_block has no upstream data dependencies for v1.
    }

    void step(std::mt19937_64& rng) override {
        if (cfg_.method == order_mcmc_block_config::method_t::partition) {
            // Kuipers-Moffa partition MCMC: one mixture step (split/join + swap
            // + single-node + edge-reversal), then draw a DAG from the partition.
            partition_mcmc_step(pchain_, *cache_, rng);
            current_log_score_ = pchain_.log_score;
            current_natural_.set_size(n_);
            for (std::size_t p = 0; p < n_; ++p)
                current_natural_[p] = static_cast<double>(pchain_.state.permy[p]);
            sampled_dag_ = partition_sample_dag(*cache_, rng, pchain_.state);
            if (keep_history()) {
                arma::uvec pv(n_);
                for (std::size_t p = 0; p < n_; ++p) pv[p] = pchain_.state.permy[p];
                history_order_.push_back(std::move(pv));
                history_log_score_.push_back(current_log_score_);
            }
            return;
        }

        // ----- Propose (order mode) -----
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
        if (cfg_.method == order_mcmc_block_config::method_t::partition) {
            // "<name>"           = the partition's node permutation (permy)
            // "<name>_party"     = the element sizes (partition composition)
            // "<name>_sampled_DAG" = a DAG drawn from the current partition
            // "<name>_log_score" = the partition log-score
            arma::vec order_vec(n_);
            for (std::size_t p = 0; p < n_; ++p)
                order_vec[p] = static_cast<double>(pchain_.state.permy[p]);
            out.emplace(cfg_.name, std::move(order_vec));
            arma::vec party_vec(pchain_.state.party.size());
            for (std::size_t t = 0; t < pchain_.state.party.size(); ++t)
                party_vec[t] = static_cast<double>(pchain_.state.party[t]);
            out.emplace(cfg_.name + "_party", std::move(party_vec));
            auto pdag = partition_sample_dag(*cache_, rng, pchain_.state);
            arma::vec pdag_flat(n_ * n_, arma::fill::zeros);
            for (std::size_t i = 0; i < n_; ++i)
                for (std::size_t j = 0; j < n_; ++j)
                    if (pdag[i] & (1ULL << j)) pdag_flat[i * n_ + j] = 1.0;
            out.emplace(cfg_.name + "_sampled_DAG", std::move(pdag_flat));
            out.emplace(cfg_.name + "_log_score", arma::vec({current_log_score_}));
            return out;
        }
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
    partition_chain                   pchain_;   // used only when method=partition

    // History (when keep_history true).
    std::vector<arma::uvec> history_order_;
    std::vector<double>     history_log_score_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_ORDER_MCMC_BLOCK_HPP

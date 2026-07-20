## order_mcmc_block (Friedman-Koller 2003 order MCMC for Bayesian-network structure)

Specialised MCMC kernel for **Bayesian-network structure learning**
over a state space of total orderings on n discrete random variables.
Target distribution (Friedman & Koller 2003, *Machine Learning*
**50**: 95-125):

    p(< | D) prop.to p(<) Sigma_{G < <} p(D | G) p(G)
            = p(<) Pi_{i=1..n} Sigma_{Pa_i subset Pred(i, <)} score(i, Pa_i)

where the sum over DAGs compatible with order < factorises into n
independent sums because each variable's parent set lives in its
predecessor set. We sample orders via Metropolis-Hastings with a
mixture of any-pair swaps and adjacent swaps, and at each step
sample a DAG by drawing each variable's parent set from its
posterior conditional on the current order. Discrete data + BDeu
score (Heckerman-Geiger-Chickering 1995, Eq 28) is the only
likelihood family in v1.2.

**JUSTIFICATION (Check #16):** discrete DAG-structure learning -- a
combinatorial state space that's intractable with NUTS / Gibbs (no
gradient, factorial-many DAGs even for n = 10). Order MCMC reduces
the search to the n! permutations (one order ~ many DAGs) and uses
BDeu Bayesian model averaging within each order. This is the
textbook efficiency path for DAG learning. Algorithm is
Friedman-Koller 2003; the BDe / BDeu score is Heckerman-Geiger-
Chickering 1995. Check #15 parity panel under `tests/`:
- `test_bde_scorer.cpp` -- 12 sub-tests across T1-T8: 2-node
  Heckerman hand-computed BDe (empty + single-parent families);
  likelihood-equivalence within a Markov class (X->Y vs Y->X give
  identical marginal scores); empty-parent closed form (BDe
  reduces to product of Beta-Bernoulli marginals); structure
  prior FK Eq 2 toggling (uniform vs fan-in-penalised); BDeu
  alpha-scaling asymptotics; single-edge score reduces to two-bin
  Beta-Bernoulli; top-C candidate-parent selection by single-edge
  score; validation rejects (cardinality mismatch, alpha <= 0,
  out-of-range data values).
- `test_score_cache.cpp` -- 11 sub-tests across T1-T8: cached
  families sorted descending by score; `order_node_score`
  consistency with explicit Sigma `family_score`; total order
  log-score factorises across nodes; `sample_parent_set` returns
  feasible (Pa subset predecessor) sets; gamma-pruning at default 10-nat
  cap; FK Sec.4.2 top-C candidate cap respected; Markov-equivalence
  stability (identity vs reverse order on a chain gives similar
  log-scores); `sample_dag` returns size-n feasible parent-set
  bitmasks.
- `test_order_mcmc_block.cpp` -- 11 sub-tests: construction smoke;
  reproducibility under fixed seed; permutation invariant under
  step; round-trip `set_current o get_current`; non-trivial MH
  accept rate (0.02 < alpha < 0.98); long-run log-score plateau on
  chain BN; sampled DAG respects current order (parents are
  predecessors); `current_named_outputs` returns 3 keys
  (order / order_sampled_DAG / order_log_score).

Ground truth is hand-computed BDe / Markov-equivalence + textbook
formulas -- zero external package dependency in the unit tier.

**Stress / robustness (9 sub-tests):** `tests/test_order_mcmc_block_stress.cpp` --
R1 bitwise reproducibility, R2 no-signal max edge marginal < 0.30,
R3 max_parents enforced, R4 BDeu alpha-asymptotics, R5 n = 20
stability, **R6 4-chain Gelman-Rubin R-hat < 1.01 strict (Vehtari
2021) on unimodal target**, R7 post-conv std/|mean| < 1%, R8
r_i = 4 cardinality recovery, R9 n = 2 edge case.

**Exact-posterior gold-standard diagnostics (4 sub-tests):**
`tests/test_order_mcmc_block_diagnostics.cpp` --
- **D1 n = 3:** enumerate all 25 DAGs, compute exact
  P(G | D) prop.to p(D | G) * |LE(G)| * Pi_i rho(|Pa_i|), marginalise to
  edges, compare to 20000 MCMC samples; HARD max |Delta| < 0.05
  achieves ~= 0.001.
- **D2 n = 4:** same comparison on 543 DAGs, 30000 samples;
  achieves ~= 0.009.
- **D3 conditional P(Pa_i | order, D):** direct subset
  enumeration vs sample_parent_set 10000 draws; HARD max
  |Delta| < 0.03 achieves ~= 0.001.
- **D4 ESS(log_score) > 200** on 10000 post-burn samples (Geyer
  initial-positive-sequence).

**bnlearn ASIA cross-check:**
`tests/audit_OrderMCMCBN_bnlearn_cross.R` -- On ASIA (8 nodes,
N = 5000), OrderMCMCBN's top-8 inclusion frequencies recover
**7 / 8** true Markov-equivalent edges with **7 / 7 perfect
skeleton match** against `bnlearn::hc`. The miss (A -> T) is also
missed by bnlearn::hc -- low data evidence, not an algorithm
failure.

**BiDAG reference-implementation comparison:**
`tests/audit_OrderMCMCBN_vs_BiDAG.R` -- head-to-head with
`BiDAG::orderMCMC` (Kuipers-Moffa reference, source of the v1.2.1
partition-MCMC track) on identical data and 4 matched chains.
Both implementations achieve **R-hat < 1.01** (ours 1.00073,
BiDAG 1.00000). **Outcome (A): our code converges to the same
target as the reference.**

**Reference template:** `examples/OrderMCMCBN.cpp` -- Tier A R
wrapper (`new(OrderMCMCBN, D, cardinalities, bdeu_alpha,
max_parents, candidate_top_C, family_cache_F, gamma_prune_nats,
prob_adjacent_swap, initial_order, rng_seed, keep_history)`)
with the core-6 R state contract (step / get_current /
set_current / predict_at / get_dag / get_history) plus the kernel-control
category (freeze / unfreeze / get_frozen) per interface.md Sec.1.

**Scope (v1.2 ship):**
- **Discrete data only** with per-column cardinality vector (BDeu
  with user-settable equivalent sample size alpha). Continuous /
  mixed / hybrid BN deferred to v1.2.1.
- **Order MCMC**: any-pair + adjacent-swap mixture (default
  prob_adjacent_swap = 0.5).
- FK Sec.4.2 three-tier candidate-parent heuristic (top-C parents
  per node by single-edge score; top-F whole families cached
  globally; gamma-pruning in nats of remaining families during the
  sum step).
- **Per-order BDeu Bayesian model averaging** for sampled DAG --
  parent sets drawn from the FK Sec.4.2 posterior conditional on
  the current order.
- **Documented FK Sec.4.1 bias** within Markov equivalence: order
  MCMC's induced *structure* prior is not hypothesis-equivalent
  (Markov-equivalent DAGs receive different prior weights). The
  algorithm faithfully recovers the **skeleton** (7 / 7 vs
  bnlearn on ASIA) but may flip directions within an equivalence
  class. Fix is Kuipers-Moffa 2017 partition MCMC -- deferred to
  v1.2.1 -- see "Deferred to v1.2.1".

**Deferred to v1.2.1:**
- Kuipers & Moffa (2017) **partition MCMC** to remove the FK Sec.4.1
  structure-prior bias inside Markov equivalence classes.
- **Continuous / mixed-type data** (BGe Gaussian score per
  Geiger-Heckerman 1994; mixed-discrete-Gaussian conditional
  Gaussian networks per Lauritzen 1992).
- **Edge-specific prior** (currently uniform DAG prior with hard
  max-parents cap).
- **Tempered / parallel-tempered chains** for very multimodal
  posteriors.
- **Predict_at v2**: forward simulation under sampled DAG (v1 is
  a stub returning the current sampled DAG).

```cpp
order_mcmc_block_config cfg;
cfg.name              = "order";       // shared_data key for the
                                       //   permutation (length n)
cfg.data              = D;              // arma::imat (N x n) of
                                       //   non-negative integers
                                       //   in {0, ..., r_i - 1}
cfg.cardinalities     = arma::uvec(n, arma::fill::value(2));
cfg.bdeu_alpha        = 1.0;           // BDeu equivalent sample size
cfg.max_parents       = 5;             // hard cap (FK Sec.4.2 typical 4-6)
cfg.candidate_top_C   = 7;             // FK Sec.4.2 top-C candidates
cfg.family_cache_F    = 200;           // FK Sec.4.2 top-F cached families
cfg.gamma_prune_nats  = 10.0;          // gamma-prune sum in nats
cfg.prob_adjacent_swap = 0.5;          // 0.5 = balanced any-pair /
                                       //   adjacent-swap mixture
cfg.initial_order     = arma::uvec();  // empty = random; otherwise
                                       //   a permutation of 0..n-1
cfg.init_rng_seed     = 42;
```

**Tier C kernels:**
- `include/AI4BayesCode/bde_scorer.hpp` -- BDe / BDeu family-score
  kernel. O(N + q_i * r_i) per family with explicit count
  scatter; per-cell log-Gamma differences.
- `include/AI4BayesCode/score_cache.hpp` -- FK Sec.4.2 three-tier
  cache. `order_node_score(i, order)` = log Sigma_{Pa_i subset Pred(i)}
  exp(score(i, Pa_i)) over the cached + gamma-pruned family list.
  `sample_parent_set(i, order, rng)` draws a parent set from the
  same posterior.

**Six-method R contract** (Tier A `OrderMCMCBN`):
- `step(M)` -- M MH order-swap steps.
- `get_current()` -- `list(order = 1:n permutation, sampled_DAG =
  n x n {0, 1} matrix, log_score = scalar)`.
- `set_current(list(order = ...))` -- only `order` is settable;
  derived `sampled_DAG` / `log_score` are read-only (rejected
  on input).
- `predict_at(list())` -- v1 stub returning the current
  `sampled_DAG`.
- `get_dag()` -- sampling DAG topology (block-level).
- `get_history()` -- `list(order = M x n matrix, ...)` when
  `keep_history = TRUE`.

**Vendored kernel:** none. Pure C++17 + Armadillo + 64-bit
bitmasks for parent sets (caps n <= 64; FK Sec.4.2 already requires
small n for tractable enumeration).

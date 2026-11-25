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

**bnlearn ASIA cross-check:** on ASIA (8 nodes,
N = 5000), OrderMCMCBN's top-8 inclusion frequencies recover
**7 / 8** true Markov-equivalent edges with **7 / 7 perfect
skeleton match** against `bnlearn::hc`. The miss (A -> T) is also
missed by bnlearn::hc -- low data evidence, not an algorithm
failure.

**BiDAG reference-implementation comparison:** head-to-head with
`BiDAG::orderMCMC` (Kuipers-Moffa reference, the same source as our
shipped partition method) on identical data and 4 matched chains.
Both implementations achieve **R-hat < 1.01** (ours 1.00073,
BiDAG 1.00000). **Outcome (A): our code converges to the same
target as the reference.**

**Reference template:** `examples/OrderMCMCBN.cpp` -- Tier A R
wrapper (`new(OrderMCMCBN, D, cardinalities, structure_prior,
bdeu_alpha, max_parents, candidate_top_C, family_cache_F,
gamma_prune_nats, prob_adjacent_swap, initial_order, rng_seed,
keep_history)` -- `structure_prior` is the 3rd argument and is
REQUIRED: `"uniform"` or `"fk_eq2"`)
with the core-6 R state contract (step / get_current /
set_current / predict_at / get_dag / get_history) plus the kernel-control
category (freeze / unfreeze / get_frozen / clear_history) per interface.md Sec.1.

**Scope (v1.2 ship):**
- **Discrete data** with per-column cardinality vector (BDeu with
  user-settable equivalent sample size alpha), OR **continuous data**
  via the BGe Gaussian score (`cfg.continuous_data` non-empty;
  Geiger-Heckerman, tuned by `cfg.bge_am` / `cfg.bge_aw`). Works with
  either sampling method. MIXED discrete + continuous (conditional
  Gaussian networks, Lauritzen 1992) is still not implemented.
- **Order MCMC**: any-pair + adjacent-swap mixture (default
  prob_adjacent_swap = 0.5).
- FK Sec.4.2 three-tier candidate-parent heuristic (top-C parents
  per node by single-edge score; top-F whole families cached
  globally; gamma-pruning in nats of remaining families during the
  sum step). The block disables gamma-pruning, and in method =
  partition it uses every family up to max_parents.
- **Per-order BDeu Bayesian model averaging** for sampled DAG --
  parent sets drawn from the FK Sec.4.2 posterior conditional on
  the current order.
- **Documented FK Sec.4.1 bias** within Markov equivalence: order
  MCMC's induced *structure* prior is not hypothesis-equivalent
  (Markov-equivalent DAGs receive different prior weights). The
  algorithm faithfully recovers the **skeleton** (7 / 7 vs
  bnlearn on ASIA) but may flip directions within an equivalence
  class. Fix: set `cfg.method = method_t::partition` (Kuipers-Moffa
  2017 partition MCMC, SHIPPED) -- it samples over labelled partitions
  and removes this bias.

**Not implemented:**
- **MIXED discrete + continuous data** (conditional Gaussian networks
  per Lauritzen 1992). Pure-discrete (BDeu) and pure-continuous (BGe)
  both ship.
- **Tempered / parallel-tempered chains** for very multimodal
  posteriors.
- **Forward simulation of data under the sampled DAG**: predict_at
  returns the sampled DAGs themselves (see the contract below).

```cpp
order_mcmc_block_config cfg;
cfg.name              = "order";       // shared_data key for the
                                       //   permutation (length n)
cfg.data              = D;              // arma::imat (N x n) of
                                       //   non-negative integers
                                       //   in {0, ..., r_i - 1}
cfg.cardinalities     = arma::uvec(n, arma::fill::value(2));
// MANDATORY -- no default; the constructor THROWS on dag_prior::UNSET.
//   dag_prior::UNIFORM -> P(G) proportional to 1 (BiDAG edgepf=1 / bnlearn)
//   dag_prior::FK_EQ2  -> Friedman-Koller 2003 Eq 2 per-family balancing
// Take it from the spec's own words. Note that choosing FK order MCMC as the
// SAMPLER does not choose the FK prior, and max_parents below is an in-degree
// CAP, not a prior shape -- both have been mistaken for this field.
cfg.structure_prior   = dag_prior::UNIFORM;   // quote the deciding spec line
cfg.bdeu_alpha        = 1.0;           // BDeu equivalent sample size
cfg.max_parents       = 5;             // hard cap (FK Sec.4.2 typical 4-6)
cfg.candidate_top_C   = 7;             // FK Sec.4.2 top-C candidates
                                       //   (method = order only)
cfg.family_cache_F    = 200;           // FK Sec.4.2 top-F cached families
                                       //   (method = order only)
cfg.gamma_prune_nats  = 10.0;          // no effect in either method: the
                                       //   block scores without gamma
                                       //   pruning
cfg.prob_adjacent_swap = 0.5;          // 0.5 = balanced any-pair /
                                       //   adjacent-swap mixture
                                       //   (method = order only)
cfg.initial_order     = arma::uvec();  // method = order: empty = random,
                                       //   otherwise a permutation of
                                       //   0..n-1. method = partition (the
                                       //   default): must be empty (the
                                       //   constructor throws otherwise);
                                       //   the chain starts from the
                                       //   one-element partition
cfg.init_rng_seed     = 42;
```

**Block state and loaders.** `cfg.method = method_t::partition` (the
default): the state is a labelled partition. `current()` / the `<name>`
output is its node sequence, element 0 first (a node's parents lie in later
elements, so children come before parents and the reversed sequence lists
parents first); `<name>_party` holds the element sizes; `<name>_log_score`
is log P(D | partition). `set_partition(permy, party)` loads a partition
(nodes are sorted within each element; it throws for invalid input or a
partition with log-score -inf); on a child of a `composite_block`, call
the composite's `sync_child(i)` after it to write the shared data keys.
`set_current(v)` loads nothing: it returns without a change when `v`
equals `current()` on a partition with two or more elements, or on the
one-element partition when n = 1 (the element sizes are those of the
current partition), and throws `std::invalid_argument` otherwise, also,
for n >= 2, whenever the current partition is the one-element partition.
`cfg.method = method_t::order`: the state is an order, `set_current(v)`
loads it, and `<name>_log_score` is log P(D | order).

**Block history.** With `keep_history`, each `step()` and each held sweep
of a frozen block (`record_held_history()`) appends the order (node
sequence), its log-score and `sampled_dag()`, so `get_history()`
returns `<name>` (T x n), `<name>_log_score` (T x 1) and
`<name>_sampled_DAG` (T x n*n; row t is the DAG of stored sweep t, element
i*n + j is 1 when j is a parent of i, as in the `<name>_sampled_DAG`
output). After a `step()`, `sampled_dag()` is the DAG sampled in it from
the sweep's order or partition with its rng. In a held sweep `step()` is
not called, and the stored DAG is the held DAG (`sampled_dag()` at the
freeze or after a later load), so the rows of held sweeps are copies of
one DAG, not draws from P(G | partition, D). No random number is drawn
for the history. The DAGs are stored as n 64-bit parent masks per sweep
(`sampled_dag_history()`); `get_history()` builds the T x n*n matrix of
doubles on each call (n = 20, T = 40000: 128 MB). `clear_history()`
empties the three buffers, and with no stored sweep `get_history()`
returns only `<name>`, the current order, as one row. Tested in
`tests/test_fix_order-mcmc_dag_history.cpp`.

**Tier C kernels:**
- `include/AI4BayesCode/bde_scorer.hpp` -- BDe / BDeu family-score
  kernel. O(N + q_i * r_i) per family with explicit count
  scatter; per-cell log-Gamma differences.
- `include/AI4BayesCode/score_cache.hpp` -- FK Sec.4.2 three-tier
  cache. `order_node_score(i, order)` = log Sigma_{Pa_i subset Pred(i)}
  exp(score(i, Pa_i)) over the cached + gamma-pruned family list.
  `sample_parent_set(i, order, rng)` draws a parent set from the
  same posterior.

**Six-method R contract** (Tier A `OrderMCMCBN`, which runs the default
partition method):
- `step(M)` -- M sweeps (one partition-MCMC move each).
- `get_current()` -- `list(order = partition node sequence (1-based,
  element 0 first: children before parents, so rev(order) lists parents
  first), order_party = element sizes, sampled_DAG = n*n {0, 1} vector
  (column-major), log_score = log P(D | partition))`.
- `set_current(list(order = ..., order_party = ...))` -- loads that
  partition exactly (values as `get_current()` returns them). `order`
  alone loads nothing: it is accepted only when it equals the current
  node sequence of a partition with two or more elements, and is an
  error otherwise (always on a new model). Derived `sampled_DAG` /
  `log_score` are read-only (rejected on input).
- `initial_order` must be `numeric(0)`; a non-empty value is an error
  that points to `set_current(list(order = ..., order_party = ...))`.
- `freeze()` / `unfreeze()` accept `"order_party"` as a name of the
  `"order"` block, so `ai4bayescode_new_frozen(OrderMCMCBN, ..., fixed =
  list(order = ..., order_party = ...))` (Python: `new_frozen(...,
  fixed={"order": ..., "order_party": ...})`) loads the partition and
  freezes the block at it.
- `predict_at(list())` -- `list(sampled_DAG = ...)`, one sampled DAG
  per row, n*n column-major as in `get_current()`. With `keep_history =
  TRUE`: one row per stored sweep, the DAG stored for that sweep (read
  from the block's DAG history; nothing is drawn): the DAG sampled in that
  sweep, or, in a held sweep of the frozen block (`freeze()`,
  `ai4bayescode_new_frozen()`), the held DAG, `get_current()$sampled_DAG`
  at the freeze or after a later load, so the rows of held sweeps are
  copies of one DAG, not draws from P(G | partition, D); one row, the
  current DAG, before the first sweep or right after `clear_history()`.
  With `keep_history = FALSE`, or `predict_at(list(), TRUE)`
  (`last_draw_only`): one row, the current DAG.
- `get_dag()` -- sampling DAG topology (block-level).
- `get_history()` -- `list(order = M x n matrix, 1-based as in
  get_current(), order_log_score = M x 1, order_sampled_DAG = M x n*n,
  each row column-major as in get_current())` when `keep_history = TRUE`;
  row t of `order_sampled_DAG` is the DAG stored for sweep t, as in
  `predict_at(list())` above (held sweeps: the held DAG).

**Vendored kernel:** none. Pure C++17 + Armadillo + 64-bit
bitmasks for parent sets (caps n <= 64; FK Sec.4.2 already requires
small n for tractable enumeration).

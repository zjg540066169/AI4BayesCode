## structured_categorical_vi_block (v1.2 SHIPPED 2026-05-31)

```cpp
structured_categorical_vi_block_config cfg;
cfg.name             = "z";
cfg.cardinalities    = arma::uvec{3, 3, 3, 3};
cfg.clique_partition = { {0, 1}, {2, 3} };    // partition of {0..n-1}
cfg.log_density      = /* same callback as Block 4 */;
cfg.dependencies     = {"y", "X"};
cfg.exact_enumeration = false;
cfg.n_mc_samples      = 16;
cfg.optimizer         = vi_optimizer::raabbvi_config{};
auto blk = std::make_shared<structured_categorical_vi_block>(cfg);
impl_->add_child(std::move(blk));
```

**What it does**: implements **Saul-Jordan 1996** "structured" (a.k.a.
**partially-factorised**) mean-field VI. Refines Block 4 by preserving
intra-clique correlation exactly while factorising ACROSS user-defined
cliques:

```
Block 4 (fully factorised):    q(z) = prod_i Cat(z_i; phi_i)
Block 5 (structured):          q(z) = prod_C Cat(z_C; phi_C)
```

For clique C with joint state count `S_C = prod_{i in C} K_i`, q_C is an
anchored-softmax Categorical over `S_C` states. Total free parameters
= `sum_C (S_C - 1)` -- exponential in clique size, so cliques must be
small (4-5 nodes is typical).

**Gradient**: analytical sum-over-CLIQUE-state (vs Block 4's
sum-over-variable-state). Exact enumeration mode for small joint state
spaces; Monte Carlo otherwise. Closed-form chain rule through the
anchored softmax exactly mirrors Block 4.

**JUSTIFICATION (Check #16)**: discrete latents with strong local
dependence (`system_design.md` Sec.11.2(b)). Structured MF gives
dramatically better approximations than Block 4 when the user can
identify strong-coupling clusters; degenerates exactly to Block 4
under singleton cliques, and to exact inference under a single
all-encompassing clique (both unit-tested).

**Degeneracies (correctness invariants, unit-tested)**:
- **Singleton cliques** `{{0},{1},...,{n-1}}` => block IS Block 4.
  Test `S2` verifies per-node marginals match Block 4 to < 0.01.
- **Single grand-clique** `{{0,1,...,n-1}}` => block performs EXACT
  inference (joint phi matches normalised `exp(log p~)`). Test `S3`
  verifies max joint diff < 0.001 over 81 states.

**Empirical head-to-head with Block 4** (4-node K=3 chain with strong
beta_intra=1.5 and weak beta_inter=0.3, cliques {{0,1},{2,3}}):

| Metric | Block 4 (MF) | Block 5 (structured) | Improvement |
|---|---:|---:|---:|
| KL(joint || exact) | 0.465 | 0.010 | **46x** |
| KL(per-node || exact) | 0.025 | 5e-6 | **~5000x** |
| KL(pairwise || exact) | 0.219 | 5e-7 | **~400000x** |
| ELBO | 5.958 | **6.414** | +0.456 |
| log Z (truth) | -- | -- | 6.425 |

Block 5's ELBO is **0.011 below log Z** -- the variational bound is
essentially tight. Within-clique pairwise marginal `P(z_0, z_1)` is
captured EXACTLY in clique C1, vs Block 4's forced product-of-marginals
that washes out the correlation.

**Public interface** (`vi_block` contract):
- `current()` returns concatenated per-clique joint phi_C (length `sum_C S_C`)
- `current_sample(rng)` returns one z draw (length n integer indices)
- `per_node_marginals()` helper computes per-node marginal matrix
  by marginalising the clique joint phi
- `set_current(phi_concat)` accepts per-clique joint probabilities and
  recovers eta via inverse anchored softmax
- composite_block writes integer-indexed q-samples to shared_data
  under `cfg.name` (per Sec.18.4 hybrid-correctness invariant)

**Reference template**:
- `examples/StructuredPottsVI.cpp` -- n-node K-state Potts on arbitrary
  graph (user-supplied edges + per-edge couplings + per-node external
  field + clique partition).
- R audit `tests/audit_StructuredPottsVI.R` -- Pearson cor 1.00000 vs
  exact 81-state enumeration over both cliques.

**Validation against independent paths** (`tests/test_structured_*.cpp`):
| Path | Algorithm | Status |
|---|---|---|
| #1 | RAABBVI gradient (block under test) | reference |
| #2 | Bishop CAVI lifted to clique level (closed-form fixed point) | matches VI to 1e-4 |
| #3 | Single-site Gibbs MCMC + R-hat across 4 chains | R-hat 1.00003 |
| #4 | Exact 81-state enumeration | matches VI to 2e-4 |

VI multi-init consistency across 4 init seeds: max deviation **0.000026**.

**Scope (v1.2 ship)**:
- Arbitrary user-specified clique partition (each node in exactly one
  clique)
- Cardinality `K_i >= 2` per node, arbitrary mix
- Generic `log_density` callback -- no exponential-family assumption
- Both exact and Monte Carlo gradient modes
- RAABBVI optimizer + PSIS-k-hat Layer-3 diagnostic

**Deferred to v1.2.1**:
- Saul-Jordan Sec.2.3 "inducing partial factorizability" -- auxiliary
  variables W_ij to decouple intra-clique intractable couplings
- Overlapping cliques (junction-tree style) -- current v1 requires a
  proper partition
- Closed-form CAVI alternative when `log_density` is exp-family

**Constraints + validator obligations**: same as
`mean_field_gaussian_vi_block` (Checks #21-#22 plus the Layer-3 R2-VI
PSIS-k-hat diagnostic).

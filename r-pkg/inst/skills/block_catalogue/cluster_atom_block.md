## cluster_atom_block (per-component atoms, NON-conjugate)

**Use when** a mixture model has explicit cluster allocations `z` and the
per-component parameters have **no conjugate closed form**. This is the
non-conjugate member of the cluster-atom family; prefer the exact blocks when
they apply:

| per-component prior | block |
|---|---|
| diagonal Normal-Gamma | `normal_gamma_cluster_gibbs_block` (exact, cheaper) |
| Normal-Inverse-Wishart | `niw_cluster_gibbs_block` (exact, cheaper) |
| **anything else** | **`cluster_atom_block`** |

**What it does.** Ishwaran & James (2001) blocked Gibbs step (a): given `z`,
each component's atom is drawn from its OWN conditional,

    p(theta_k | z, y)  prop  pi(theta_k) * prod_{i: z_i = k+1} f(y_i | theta_k)

Components are conditionally independent given `z` -- that is the target's own
factorisation (Eq. 18), not an approximation -- so each gets its own update.

**Empty components need no special case.** With `n_k = 0` the product is empty
and the target reduces to the prior exactly, which is what step (a) prescribes
for an unoccupied component. There is no branch to forget and no prior-sampler
callback to supply.

**Mechanism: univariate slice** (Neal 2003 stepping-out + shrinkage), delegated
to internally held `univariate_slice_sampling_block` instances -- the kernel,
including the Eq. 5 random split of the step-out budget that reversibility
requires, is that block's and is covered by its own Check #15 parity test.

**Why not NUTS here.** The target moves every sweep: as `z` reassigns
observations a component flips between prior-wide and data-narrow. NUTS must
freeze its step size after warmup to stay valid (`n_warmup_per_step` must stay
0 -- Check #20), and a frozen step size cannot follow that. Measured on a
truncated-DP mixture, K = 10, 20 datasets, 20k+20k, the fraction of sweeps in
which the location vector did not move at all: one `joint_nuts_block` over all
components **74%**, per-component `joint_nuts_block` **15%**, this block **0%**;
median cross-chain rank R-hat 1.1018 / 1.0866 / 1.0018. Slice needs no tuning,
so a per-sweep change of target shape costs it nothing -- and it needs no
gradient, so there is no hand-derived artifact for Check #12 to police.

```cpp
cluster_atom_block_config cfg;
cfg.name       = "cluster_params";
cfg.K_trunc    = K;
cfg.counts_key = "cluster_counts";            // as stick_breaking_block
// same element type and field name as joint_nuts_block_config::sub_params,
// but `dim` is the PER-COMPONENT dimension of that slice
cfg.sub_params.push_back(joint_nuts_sub_param{"shape", 1u, joint_constraint::POSITIVE});
cfg.sub_params.push_back(joint_nuts_sub_param{"rate",  1u, joint_constraint::POSITIVE});
cfg.initial_cat = init;                       // K_trunc * sum(dim), cluster-major
cfg.log_density = &atom_log_density;          // the ONE user function
impl_->add_child(std::make_unique<cluster_atom_block>(std::move(cfg)));
```

`log_density(theta_k, k, ctx)` returns component k's conditional on the
**NATURAL** scale up to a constant -- the same contract as `joint_nuts_block`'s
log-density. The block adds the `log|Jacobian|` of each constraint transform
itself; never hand-write one. Declare `cluster_counts` (and whatever the
density reads) in the block's `declare_dependencies` or they will not be in its
context.

**Outputs.** One shared_data key per slice, each `K_trunc * dim` in
cluster-major order -- the layout the conjugate cluster blocks use, so this
block is a drop-in for them.

**Constraint scope (v1).** Per-element kinds only: REAL, POSITIVE,
LOWER_BOUNDED, UPPER_BOUNDED, INTERVAL. A coupled kind (ORDERED, SIMPLEX,
CHOLESKY_*, CORR_MATRIX, COV_MATRIX) throws from the constructor -- a
coordinate-wise sweep over them is well defined but has no ground-truth
coverage yet, and an untested constraint path is how a block gets silently
wrong.

**Reference example:** `examples/DPGammaMixture.cpp` (truncated DP mixture of
Gamma densities; the shape parameter is non-conjugate under any prior because
the likelihood contributes `-n_k * lgamma(shape_k)`).

**Tests:** `tests/test_cluster_atom_block.cpp` (T0-T4 ladder: analytic
Normal-Gamma parity, 3-component parity, recovery from truth, cross-chain rank
R-hat 1.00013 from over-dispersed inits, 18 empty components matching the
analytic prior) and `tests_autodiff/block_tests/test_cluster_atom_block.cpp`
(Check #15 library parity, 40000 draws).

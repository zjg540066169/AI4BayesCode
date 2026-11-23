## split_merge_block (Jain-Neal 2004 cluster-partition MH)

Metropolis-Hastings proposal block that toggles a cluster allocation
vector `z` between **merged** and **split** versions in a single
proposal. Accelerates mixing on partitions vs single-i Gibbs.

**Algorithm**: Jain & Neal 2004 "A Split-Merge Markov Chain Monte Carlo
Procedure for the Dirichlet Process Mixture Model". This block
implements the algorithm in the **truncated SBP regime** (NOT
CRP-marginal): `pi` is held fixed during the MH ratio, and the prior on
`z` is product-of-categorical Cat(pi).

**Algorithm summary** (per step):
1. Sample (i, j) uniformly w/o replacement from {0..N-1}.
2. If `z[i] == z[j]` -> SPLIT proposal:
   - Pick `c_new` uniformly from EMPTY slots.
   - Initialise S = {k : z[k] == z[i], k != i, j} randomly to `{s_i, c_new}`.
   - Run T restricted-Gibbs scans; final iteration accumulates log q(z*|z).
   - log A = log q(z|z*) - log q(z*|z) + log_prior + log_lik
3. Else -> MERGE proposal:
   - Set z*[k] = s_i for all k in {j}  U  S.
   - Compute reverse-split q via launch state + T restricted scans
     accumulating at the final iter where target = z[k].

**Empty-slot rule**: SPLIT proposals when no empty slots are available
return immediately as rejected. So K_active is bounded by K_trunc.

```cpp
#include "AI4BayesCode/split_merge_block.hpp"

split_merge_block_config cfg;
cfg.name = "split_merge_z";        // child label
cfg.N = N;
cfg.K_trunc = K_trunc;
cfg.d = d;
cfg.z_name = "z";                  // writes back to shared_data["z"]
cfg.y_key = "y";
cfg.pi_key = "pi";
cfg.mu_key = "mu";
cfg.lambda_key = "lambda";         // exactly ONE of lambda_key / sigma_key
// cfg.sigma_key = "sigma";        // (full covariance path)
cfg.n_restricted_gibbs_iters = 5;  // Jain-Neal Sec.3.1 typical
cfg.initial_z = z_init;
impl_->add_child(std::make_unique<split_merge_block>(std::move(cfg)));

// REQUIRED: "z" in the dependencies, so each proposal starts from the shared
// allocation (this sweep's categorical draw) instead of the block's own
// previous z, and cluster_counts invalidated, so the counts refresher reruns
// after an accepted split or merge.
impl_->data().declare_dependencies("split_merge_z", {"y", "pi", "mu", "lambda", "z"});
impl_->data().declare_invalidates("split_merge_z", {"cluster_counts"});

// REQUIRED: `categorical_gibbs_block` and `split_merge_block` both
// drive -- and both record to history -- the SAME allocation `z`.
// shared_data writes coexist fine (last writer in the sweep wins),
// but composite_block::get_history() rejects a history key emitted by
// two children UNLESS it is explicitly declared shared. Declare it,
// using the SAME key both blocks write (categorical_gibbs `cfg.name`
// == split_merge `cfg.z_name`, here "z"):
impl_->declare_shared_history("z");
```

**Composite integration**: add `split_merge_block` as a child AFTER
`categorical_gibbs_block` in the Gibbs sweep, and call
`impl_->declare_shared_history("z")` (see above). Both blocks write
shared_data key "z" (coexist fine -- last writer in the sweep wins) AND
both record "z" to history; the `declare_shared_history` call makes
`get_history()` keep the LAST contributor in scan order -- i.e. the
post-split-merge, scan-END `z` draw, which is the correct posterior
chain. Without the declaration `get_history()` throws
`duplicate key 'z' contributed by multiple children`. Each step
alternates: per-i Gibbs -> split-merge proposal -> cluster_params
update -> pi update.

**Shared `z` and counts.** Both wiring lines of the recipe are required. The
block keeps its own copy of `z` and loads the shared `z` in `set_context`
when `"z"` is listed. Without that entry each proposal starts from the
block's previous `z`, which discards the sweep's categorical draw and, with
`stick_label_swap_block` in the sweep, undoes the relabelling of `z` while
the atoms and weights stay relabelled. Without the invalidation, the
stick-breaking weights of the same sweep are drawn from the counts of the
allocation before an accepted split or merge. On exactly enumerated
mixtures (4 slots and 6 observations; 3 slots and 7), with the label swap,
the total variation distance over partitions is 0.48-0.54 when `"z"` is not
listed; with `"z"` listed but no invalidation the error of P(K_active) is
0.003-0.007, against 0.0002-0.0016 with both lines, which matches a
composition without split_merge. With `"z"` listed, the shared `z` is the
block's starting state at every sweep, so a restore of this block must also
write the shared key (`data().set`, as the shipped wrappers do) or go
through `composite_block::set_current`; a `set_current` of this block alone
is replaced by the shared value at the next sweep.

**Acceptance asymmetry note**: Jain-Neal's q-density is asymmetric
(merge is deterministic; split is random over restricted Gibbs). So
SPLIT acceptance rates can be high (favoured) and MERGE acceptance rates
can be near 0 even when statistically equivalent. This is **textbook
behavior**, not a bug. The block is designed to accelerate breaking
into well-fitted sub-clusters; merging away of redundant clusters
relies on the categorical Gibbs sweeps depopulating clusters.

**JUSTIFICATION (Check #16): Exception 1** (discrete-allocation MH that
NUTS cannot target). Library-blessed block; user writes no MH
acceptance code.

**Check #15** parity: NO dedicated parity test ships for this block. When
Check #15 fires, write one covering (a) smoke + valid output on an
identical-cluster fixture; (b)
truth-recovery K_active >= 2 on a 2-cluster split-favored fixture;
(c) full-covariance sigma_key path runs.

**No hand-written Jacobian** (Check #5 vacuous -- discrete-allocation MH).

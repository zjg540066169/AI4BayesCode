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

**Check #15** parity: `tests_autodiff/block_tests/test_split_merge_block.cpp`
verifies (a) smoke + valid output on identical-cluster fixture; (b)
truth-recovery K_active >= 2 on a 2-cluster split-favored fixture;
(c) full-covariance sigma_key path runs.

**No hand-written Jacobian** (Check #5 vacuous -- discrete-allocation MH).

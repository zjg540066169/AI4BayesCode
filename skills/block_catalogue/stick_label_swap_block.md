## stick_label_swap_block (label-switching MH moves for truncated stick-breaking)

Metropolis-Hastings block that relabels whole clusters of a TRUNCATED
stick-breaking mixture (Dirichlet process or Pitman-Yor) in one step. It
leaves the joint posterior of (z, atoms, sticks) unchanged.

**Why it is needed**: under stick-breaking the slot weights are
stochastically ordered, so a cluster's weight depends on which slot holds
it. The blocked Gibbs sampler (`categorical_gibbs_block` for z, an atom
block, `stick_breaking_block`) moves a cluster to another slot only by
emptying it one observation at a time. Chains then keep different slot
arrangements, and the slot weights `pi` and sticks `stick_V` do not mix
across chains even when every label-invariant quantity has converged.
Any composition that samples z, atoms and sticks of a truncated
stick-breaking mixture includes this block.

**Moves** (deterministic scan, each an exact MH step):
1. Neighbour swap (Papaspiliopoulos & Roberts 2008): exchange labels c and
   c+1 in z and the atoms and exchange V_c and V_{c+1}, for every pair with
   both sticks free; the last pair, whose final stick is fixed at 1,
   exchanges the two weights instead. Valid for DP and Pitman-Yor.
2. Weight-preserving swap (Hastie, Liverani & Richardson 2015, Sec. 3.2):
   exchange labels c and c+1 and re-split pi_c + pi_{c+1}. Dirichlet process
   only (skipped when `discount > 0`). The acceptance ratio is derived for
   the truncated representation and includes the change-of-variables
   Jacobian; the formula as stated for the slice representation does not
   leave the truncated posterior invariant.

```cpp
#include "AI4BayesCode/stick_label_swap_block.hpp"

AI4BayesCode::stick_label_swap_block_config cfg;
cfg.name      = "label_swap";
cfg.K_trunc   = K;                                   // same truncation as the stick block
cfg.atom_keys = {{"mu", d}, {"lambda", d}};          // (key, width), cluster-major layout
cfg.discount  = 0.0;                                 // Pitman-Yor d in [0, 1); 0 = DP
impl_->data().declare_dependencies("label_swap", {"z", "mu", "lambda", "pi", "stick_V", "alpha"});
impl_->data().declare_invalidates("label_swap", {"cluster_counts"});
impl_->add_child(std::make_unique<AI4BayesCode::stick_label_swap_block>(std::move(cfg)));
```

**Contract**:
- Reads and writes z (1-based labels), every atom key, `pi` and `stick_V`;
  reads the concentration from `alpha_key` (default `"alpha"`).
- Each atom key holds component k in entries [k * width, (k + 1) * width).
  Every per-component parameter must be listed, or the swap would pair a
  cluster with another cluster's parameters.
- Declare `invalidates("label_swap", {"cluster_counts"})` so the counts
  refresher reruns after a relabelling.
- Its position in the sweep does not affect correctness; appending it as
  the last child keeps the indices of the existing children.
- `current()` and its history are the accepted counts of the two moves per
  step, one row per step, so the history stays aligned with the other
  blocks. History key: the block name.
- Per-slot summaries are no longer stable under this block: summarise
  `mu`, `pi`, `stick_V` and the other atom keys with label-invariant
  statistics (`rhat_summary(..., order_components = TRUE)`), and match
  components to targets within each draw, never by averaging a slot over
  the chain.

**Verified by**: `tests/test_stick_label_swap_exact.cpp` -- exact
enumeration of a 4-slot, 6-observation mixture (DP with each move and
both, Pitman-Yor); every marginal P(z_i = k) within 0.002 of the exact
value, and deliberately wrong acceptance ratios fail by 0.09-0.17.

**Reference examples**: `examples/DPGaussianMixture.cpp`,
`examples/DPGaussianMixture_DerivedAlpha.cpp`,
`examples/PYGaussianMixture.cpp`, `examples/DPGammaMixture.cpp`.

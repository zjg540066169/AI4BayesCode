## binary_gibbs_block

**Reference example:** NONE shipped -- library-only. The classical use
case (continuous spike-and-slab with binary indicator gamma_j) is
intentionally covered in `examples/SpikeSlabRJMCMC.cpp` via the
`rjmcmc_block` (Dirac spike-and-slab is the geometry the project
targets), not via `binary_gibbs_block`. If a codegen agent needs the
continuous-spike Class 2a pattern (continuous relaxation instead of
Dirac), it should compose `binary_gibbs_block` + `nuts_block` inline
following the header-comment pattern in `binary_gibbs_block.hpp`.

Closed-form Bernoulli for z in {0,1}. Provide `log_odds_fn(ctx) -> arma::vec`.
Use for spike-and-slab, variable selection.

**Target-geometry caveat:** `binary_gibbs_block` is the right tool for
*continuous* spike-and-slab (both spike and slab have positive Gaussian
density) and for plain binary indicators. It is **NOT** the right tool
for **Dirac** spike-and-slab (beta_j = 0 exactly when gamma_j = 0) -- that
target has a dimension-changing state space and Gibbs on it is not
irreducible. See `skills/system_design.md` Sec.11 and
`skills/codegen_priors.md` Sec.3a. For Dirac spike-and-slab, use the
`rjmcmc_block` identity path (see the rjmcmc_block entry below and
`examples/SpikeSlabRJMCMC.cpp` reference template). Alternatively,
a continuous relaxation (tight-Gaussian-spike) with `nuts_block` on
beta works too when the point-mass is not semantically required.

```cpp
binary_gibbs_block_config cfg;
cfg.name     = "gamma";
cfg.n_binary = p;
cfg.initial_values = arma::vec(p, arma::fill::zeros);
cfg.log_odds_fn = [](const block_context& ctx) -> arma::vec { ... };
```

**Oracle calls and the block's own indicators.** If `log_odds_fn` uses the
block's own indicators (Ising / MRF couplings, a prior on their count), list
the block's own name in its dependency list, e.g.
`impl_->data().declare_dependencies("gamma", {"gamma", ...})`; `step()` then
calls the oracle once per component, each time after writing the previous
indicator into that context entry. Otherwise the context has no entry under
the block's name and the oracle is called once per `step()` for all
components, so it must not keep per-call state (a call counter, a test for
its own key). When the own key is in the context, each step starts from its
shared value: indicators written there by another block are the starting
state of the sweep, so a restore of this block must also write the shared key
(`data().set`) or go through `composite_block::set_current`, and an entry
whose length is not `n_binary` makes `step()` throw.

## binary_gibbs_block

**Reference example:** NONE shipped — library-only. The classical use
case (continuous spike-and-slab with binary indicator γ_j) is
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
for **Dirac** spike-and-slab (β_j = 0 exactly when γ_j = 0) — that
target has a dimension-changing state space and Gibbs on it is not
irreducible. See `skills/system_design.md` §11 and
`skills/codegen_priors.md` §3a. For Dirac spike-and-slab, use the
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

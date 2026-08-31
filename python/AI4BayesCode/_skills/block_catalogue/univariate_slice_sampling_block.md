## univariate_slice_sampling_block

Neal 2003 univariate slice sampler (stepping-out + shrinkage,
section 4.1, with random step-out budget split for reversibility per
eq. 5). **Strictly 1-D / scalar.** User writes ONLY a
natural-scale log-density lambda; sampler machinery is textbook and
library-owned -- and unlike the `nuts_block` lambda it does not also have
to produce a gradient, so Check #12 does not apply. No
conditional-posterior derivation either (unlike Gibbs).

**When to use vs nuts_block.** For a SCALAR parameter sampled in its own
block, this is the DEFAULT -- prefer it over a standalone `nuts_block`.
`joint_nuts_block` is unchanged and remains the default for continuous
parameters generally; this block is about what a parameter becomes once
it is broken out on its own.

Three reasons, in order of how often they bite:
  1. **No step size to freeze.** NUTS must keep its step size frozen after
     warmup (Check #20). A scalar block usually sits in a Gibbs sweep
     whose siblings keep moving its conditional, and a frozen step cannot
     follow that. Measured on a truncated-DP mixture: per-component 1-D
     NUTS froze 15% of sweeps (rank R-hat 1.0866), this block 0% (1.0018).
  2. **It cannot lock up.** Shrinkage is guaranteed to accept in finite
     iterations, so there is no rejection-stall state (cf. the lock-up at
     `nuts_block.hpp:242`).
  3. **No gradient.** One natural-scale log-density returning a double;
     Check #12 does not apply.

Keep a scalar on `nuts_block` only when its gradient shares most of its
computation with the density, so the gradient is nearly free. Multi-dim
parameters are out of scope: `initial_unc` MUST be length 1.

**Reference example**: `examples/GPTimeSeries.cpp` (using slice sampling:
hyperparameters amp/tau/sigma sampled via slice on celerite's
marginal log-likelihood).

**JUSTIFICATION (Check #16):** library-provided specialized sampler --
the tuning-free default for a SCALAR continuous block. (Not "Exception 1":
that entry in the Sec.2b table is the DISCRETE-parameter exception.)

**Check #15** is satisfied by the library parity test at
`tests_autodiff/block_tests/test_univariate_slice_sampling_block.cpp`
(three fixtures: Normal via identity, Gamma via positive, Beta via
interval; 10k draws each, mean/variance within 5%/10% of analytical).
A usage inherits it -- no per-usage parity test is required.

```cpp
#include "AI4BayesCode/univariate_slice_sampling_block.hpp"

univariate_slice_sampling_block_config cfg;
cfg.name         = "amp";
cfg.initial_unc  = arma::vec{std::log(amp_init)};  // unc scale, length 1
cfg.constrain    = constraints::positive::constrain;
cfg.unconstrain  = constraints::positive::unconstrain;
cfg.w            = 1.0;    // initial slice-bracket width on unc scale
cfg.log_density  = [](const arma::vec& t_unc, const block_context& ctx) {
    return constraints::positive::wrap(t_unc, nullptr,
        [&](const arma::vec& t_nat, arma::vec* /*unused*/) -> double {
            // natural-scale lp (prior + likelihood); NO Jacobian, NO grad.
            return lp;
        });
};
```

## univariate_slice_sampling_block

Neal 2003 univariate slice sampler (stepping-out + shrinkage,
section 4.1, with random step-out budget split for reversibility per
eq. 5). **Strictly 1-D / scalar.** User writes ONLY a
natural-scale log-density lambda; sampler machinery is textbook and
library-owned. Same AI-safety profile as `nuts_block` (no
conditional-posterior derivation, unlike Gibbs).

**When to use vs nuts_block.** ALWAYS prefer `nuts_block` for
smooth differentiable targets -- NUTS mixes better and is the
default. Choose `univariate_slice_sampling_block` only when NUTS
cannot:
- log p is non-differentiable (piecewise, kink, floor/ceil)
- log p is a black-box library call whose gradient would require
  re-implementing that library with autodiff::var (e.g., celerite
  marginal log-likelihood via `celerite_marginal_likelihood.hpp`)
- gradient prohibitively expensive relative to lp eval.

See `skills/codegen_priors.md` §2b.1 for the decision tree.

**Reference example**: `examples/GPTimeSeries.cpp` (using slice sampling:
hyperparameters amp/tau/sigma sampled via slice on celerite's
marginal log-likelihood).

**JUSTIFICATION (Check #16): Exception 1** -- specialized sampler for
1-D continuous parameters whose log-density lacks an accessible
gradient (violates NUTS's prerequisite). Library parity test at
`tests_autodiff/block_tests/test_univariate_slice_sampling_block.cpp`
(three fixtures: Normal via identity, Gamma via positive, Beta via
interval; 10k draws each, mean/variance within 5%/10% of analytical).

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

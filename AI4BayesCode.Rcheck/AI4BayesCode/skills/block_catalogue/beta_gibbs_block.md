## beta_gibbs_block

Exact draw from Beta(alpha, beta) via the Gamma trick. Use when the full
conditional is exactly Beta — typically mixing proportions in
spike-and-slab, or probabilities in Beta-Binomial models.

```cpp
#include "AI4BayesCode/beta_gibbs_block.hpp"

beta_gibbs_block_config cfg;
cfg.name = "pi";
cfg.initial_value = 0.5;
cfg.params_fn = [](const block_context& ctx) -> beta_dist_params {
    const arma::vec& gamma = ctx.at("gamma");
    double a = ctx.at("a_pi")[0] + arma::sum(gamma);
    double b = ctx.at("b_pi")[0] + gamma.n_elem - arma::sum(gamma);
    // Beta(a, b), E[pi] = a / (a + b)
    return {a, b};
};
```

**Do NOT use** when the conditional is not exactly Beta. Fall back to
`nuts_block` + `constraints::interval::wrap(0, 1)`.

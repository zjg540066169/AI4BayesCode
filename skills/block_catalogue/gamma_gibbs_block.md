## gamma_gibbs_block

**Reference example:** library-only initially; designed as a drop-in
replacement for `nuts_block` on log(alpha) inside the BNP examples
(`DPGaussianMixture.cpp` etc.) for users who want exact iid alpha draws
under the truncated-SBP DP closed-form posterior. The shipped DP
example uses NUTS for simplicity; switching to this block is a 1-2
line edit.

Closed-form Gibbs leaf for a SCALAR positive parameter whose full
conditional is exactly Gamma(shape, rate). Companion / dual of
`inv_gamma_gibbs_block`. Use when the conditional posterior is an
exact Gamma -- common cases:

- DP concentration alpha under truncated stick-breaking with
  `alpha ~ Gamma(a, b)` prior -> posterior `alpha | V_1, ..., V_{T-1}
  ~ Gamma(a + T - 1, b - Sigma log(1 - V_k))`.
- Scalar precision lambda when the Normal-Gamma joint is integrated to
  the marginal Gamma on lambda.
- Any scalar positive parameter with Gamma posterior conjugate.

```cpp
#include "AI4BayesCode/gamma_gibbs_block.hpp"

gamma_gibbs_block_config cfg;
cfg.name          = "alpha";
cfg.initial_value = a_prior / b_prior;
cfg.params_fn = [a_prior, b_prior](const block_context& ctx) {
    const arma::vec& V = ctx.at("stick_V");
    const std::size_t T = V.n_elem;
    double Tsum = 0.0;
    for (std::size_t k = 0; k + 1 < T; ++k) Tsum += std::log(1.0 - V[k]);
    return AI4BayesCode::gamma_params{
        a_prior + static_cast<double>(T - 1),
        b_prior - Tsum
    };
};
```

**Parameterization**: shape-RATE (matches `inv_gamma_gibbs_block`,
R's `rgamma(n, shape, rate)`, JAGS / NIMBLE / Stan). Internally
draws via `std::gamma_distribution<double>(shape, scale = 1/rate)`
which is shape-SCALE in C++. The conversion is commented at the
call site (`rcpp_api.md Sec.11`).

**JUSTIFICATION (Check #16): Exception 3** (scalar textbook conjugate
with NUTS-wasteful efficiency profile).
**Check #15** parity:
`tests_autodiff/block_tests/test_gamma_gibbs_block.cpp` covers three
regimes (small shape, large shape, DP-style closure) at 10 000 draws,
within 5 % mean / 10 % variance tolerance.

**Do NOT use** when the conditional is not exactly Gamma (e.g., alpha with
non-conjugate likelihood factors). Fall back to `nuts_block` with
`constraints::positive::wrap`.

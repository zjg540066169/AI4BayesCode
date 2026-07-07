## inv_gamma_gibbs_block

**Reference example:** NONE shipped, and none planned -- this block is
intentionally library-only because it is DISCOURAGED as default (see
below). The preferred pattern for scale parameters is Jeffreys on
`nuts_block` with inline k=0 half-Normal(0,1) fallback; see
`examples/SpikeSlabRJMCMC.cpp` tau block for the reference template.

WARNING **STRONGLY DISCOURAGED AS DEFAULT.** See `codegen_priors.md` Sec.2a "Variance
/ scale parameter prior discipline (Gelman 2006)": the default prior
for variance/scale is Jeffreys `p(sigma) prop.to 1/sigma` implemented via
`nuts_block` + `constraints::positive::wrap`, NOT InverseGamma.
`Gamma(epsilon, epsilon)` as "noninformative" is the specific prior Gelman 2006
Bayesian Analysis 1(3):515-533 refuted.

**Valid use cases** (both require explicit documented justification
inline per Check #16):
- Exception 3 with genuinely heavy-tail-pathological NUTS conditional --
  but first try the Jeffreys + k=0 half-Normal(0, 1) fallback
  pattern (`codegen_priors.md Sec.2a`), which implements weakly-informative on
  natural scale without the IG(epsilon, epsilon) critique. If even that pattern
  fails to mix, only then consider IG with documented justification.
- INFORMATIVE IG prior where the hyperparameters come from external
  knowledge (e.g., BART's calibrated IG via `sigest`, where the prior
  is deliberately informative to resolve an identifiability issue).

**If you are tempted to use this block, first reconsider whether your
goal is actually better served by `nuts_block` with Jeffreys
`p(sigma) prop.to 1/sigma` on the natural scale.** Most of the time, yes.
The pattern in `examples/SpikeSlabRJMCMC.cpp` is the canonical
template:
- sigma: `nuts_block` with Jeffreys prior
- tau  : `nuts_block` with Jeffreys prior + k=0 fallback to
  half-Normal(0, 1) pin (inside the natural-scale log-density)
under the Ishwaran & Rao 2005 sigma-scaled slab
(`beta_j | gamma_j=1, sigma, tau ~ N(0, sigma^2 tau^2)`).

Closed-form Gibbs for a scalar positive parameter whose conditional
is InverseGamma(shape, rate). Provide `params_fn(ctx) ->
inv_gamma_params{shape, rate}`.

```cpp
inv_gamma_gibbs_block_config cfg;
cfg.name          = "sigma2";
cfg.initial_value = var(y);
cfg.params_fn     = [N](const block_context& ctx) {
    const arma::vec& y    = ctx.at("y");
    const arma::vec& Xb   = ctx.at("Xbeta_cache");
    const double     a_pr = ctx.at("a_sigma_prior")[0];
    const double     b_pr = ctx.at("b_sigma_prior")[0];
    double sse = arma::accu(arma::square(y - Xb));
    // Conditional: IG(a + N/2, b + SSE/2). E[sigma^2] = rate / (shape-1).
    return inv_gamma_params{ a_pr + N/2.0, b_pr + sse/2.0 };
};
```

**Parameterization:** shape-RATE. To sample, the block draws
g ~ Gamma(shape, 1/rate) internally and inverts.

**When to use:** any scalar with a closed-form InverseGamma
conditional -- sigma^2 in Gaussian regression, tau^2 in a slab prior,
any scale parameter with conjugate InvGamma prior. Do NOT use for
vector-valued positive parameters; that's `nuts_block` with
`constraints::positive::wrap` territory.

## dirichlet_gibbs_block

**Reference example:** `examples/FiniteGaussianMixture.cpp` (shipped
2026-05-03) uses `dirichlet_gibbs_block` for the symmetric Dirichlet
posterior on the K-mixing-weight vector π. The DirichletSimplex
example uses `nuts_block` + `constraints::simplex::wrap` instead — that
is the more general option for non-conjugate factors on the simplex.
`dirichlet_gibbs_block` is also the right choice for the "A rows / pi"
of a full-Bayesian HMM (mentioned in `HMMGaussian2State.cpp`'s header
comments; the minimal 2-state example ships with fixed A / pi so it
doesn't actually use this block).
Codegen agents targeting Dirichlet-Categorical / LDA should compose
this block inline.

Exact iid draws via gamma-normalization. Use when the full conditional is
a clean Dirichlet (Dirichlet-Categorical, LDA proportions, etc.).

```cpp
dirichlet_gibbs_block_config cfg;
cfg.name           = "theta";
cfg.n_categories   = K;
cfg.initial_values = arma::vec(K, arma::fill::value(1.0 / K));
cfg.alpha_post_fn  = [](const block_context& ctx) -> arma::vec {
    return ctx.at("alpha") + counts;  // posterior concentration
};
```

**Do NOT use** when theta has non-conjugate factors. Fall back to
`nuts_block` + `constraints::simplex::wrap`.

### R-hat caveat on near-zero simplex components

`dirichlet_gibbs_block` is exact, so R-hat on each theta component is
clean regardless of posterior mass. But for **`nuts_block` +
`constraints::simplex::wrap`** (e.g. the Linero / sparse-Dirichlet
pattern in `DirichletSparse`), any component with near-zero posterior
mean is susceptible to a well-known R-hat artifact: the sampler
explores the flat/unidentified tail of the unconstrained stick-
breaking coordinate, and on the natural scale both chains produce
"effectively zero" values (e.g. 1e-6 in one chain, 1e-10 in another)
that give a legitimate R-hat of 2+ because *between-chain variance /
within-chain variance is numerically large* — even though both chains
agree the component is 0. Always report per-component R-hat, not just
the max; the max is dominated by unidentified / data-less components
in sparse-Dirichlet settings.

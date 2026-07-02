## poisson_multinomial_aug_block

Gamma-augmentation block for **multinomial / binary logistic BART via
genBART** under the **reference-category identified parameterization**
(category 0 fixed as reference, f^(0)(x) := 1, with C-1 non-reference
generalized-BART functions r_1, ..., r_{C-1}).

Implements the classical Poisson-multinomial gamma trick:

    phi_i | y, r  ~  Gamma(n_i, 1 + sum_{j=1..C-1} exp(r_j(x_i)))

For single-observation data (n_i = 1) this reduces to
Exp(rate_i). Given phi_i, the augmented likelihood factorises into
C-1 conditionally independent Poisson-like likelihoods that each
match `genbart::lik::poisson_lik` with **offset = log(phi_i)**:

    likelihood_j ~ prod_i exp(u_{i,j} * r_j(x_i) - phi_i * exp(r_j(x_i)))

where u_{i,j} = [y_i == j]. The block writes log_phi under its own
name (so downstream genbart_blocks reference it via offset_key) and
each u_j under the configured u_keys[j-1].

### Attribution

The gamma-trick augmentation is classical (predates BART):
**Baker 1994** (Statistician 43(4):495-504), **Forster 2010**
(Stat. Meth. 7(3):210-224), **Walker 2011**, **Caron & Doucet 2012**.
**Murray 2021** (JASA 116(534):756-769 §3.1) introduced the C-1
reference-category identified multinomial parameterization for tree
ensembles. This implementation preserves that architecture but uses
Linero 2022's RJMCMC tree kernel (via `genbart_block`) as the
per-ensemble sampler rather than backfitting + GIG conjugate prior.

**JUSTIFICATION (Check #16): Exception 1** (discrete/augmented
measure; NUTS cannot target the continuous phi with its y-dependent
rate).

### Use with `genbart_block` children

Reference templates:
- `examples/GBartLogistic.cpp` -- binary (C = 2) via **direct**
  `genbart_block + logistic_lik` (simpler; no augmentation).
- `examples/GBartMultinomial.cpp` -- C >= 2 via C-1
  `genbart_block(poisson_lik, offset_key = "log_phi_aug")` +
  one `poisson_multinomial_aug_block`.

```cpp
poisson_multinomial_aug_block_config cfg;
cfg.name   = "log_phi_aug";   // downstream offset_key = this name
cfg.N      = N;
cfg.C      = C;               // total classes incl. reference (>= 2)
cfg.y_key  = "y";
cfg.n_key  = "";              // empty = n_i = 1 per obs
cfg.r_keys = { "r_1", ..., "r_{C-1}" };
cfg.u_keys = { "u_1", ..., "u_{C-1}" };
cfg.initial_y = y_arma;       // integer labels 0..C-1 as doubles
// cfg.initial_log_phi defaults to zeros (phi = 1, neutral start).
```

Gibbs sweep order inside the composite:
`poisson_multinomial_aug_block` FIRST (writes u_j and log_phi from
current r_j), then each `genbart_block` in sequence (each reads its
own u_j via y_key and the shared log_phi via offset_key).

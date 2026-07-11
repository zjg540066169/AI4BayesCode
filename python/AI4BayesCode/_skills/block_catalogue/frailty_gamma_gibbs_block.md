## frailty_gamma_gibbs_block

**Reference example:** compose with `piecewise_exponential_gibbs_block` (baseline
hazard) and a small regression block (e.g. `nuts_block("beta")`) for a
shared-frailty proportional-hazards survival model -- see the USAGE block below.

Exact Gamma-conjugate Gibbs update for a length-G vector of per-group Gamma
frailties `w_g` in a shared-frailty proportional-hazards survival model, driven
by an externally-supplied piecewise-exponential baseline hazard `lambda[K]`,
covariate offset `exp(f_i)`, and (fixed or externally-sampled) concentration
`theta`.

**Model** (Clayton 1991 JRSS-B 53(1):45-73; Ibrahim, Chen & Sinha 2001 Sec.4.3;
Vaupel-Manton-Stallard 1979 Demography 16(3):439-454; Aslanidou-Dey-Sinha
1998): subject i in group g(i) has hazard
```
h_i(t) = w_{g(i)} * lambda_{k(t)} * exp(f_i)
w_g   ~  Gamma(theta, theta)     (E[w_g] = 1, Var[w_g] = 1/theta)
```

**Full conditional** (independent Gamma-Gamma conjugate step across groups):
```
D_g = sum_{i in g} delta_i                             (events in group g)
H_g = sum_{i in g} exp(f_i) * H_0(t_i, v_i)            (weighted at-risk time)
H_0(t, v) = sum_k lambda_k * max(0, min(t, e_k) - max(v, e_{k-1}))

w_g | rest ~ Gamma(theta + D_g, theta + H_g)           (independent across g)
```
One exact Gibbs step draws all G frailties. O(n + G) per step.

```cpp
#include "AI4BayesCode/frailty_gamma_gibbs_block.hpp"

frailty_gamma_gibbs_block_config cfg;
cfg.name              = "w";                          // shared_data + history key (writes length-G)
cfg.G                 = 30;                           // number of groups
cfg.edges             = edges;                        // MUST match sibling PEH block
cfg.theta             = 2.0;                          // FIXED theta; OR set cfg.theta_key = "theta"
// group_key="z", time_key="t", event_key="delta", lambda_key="lambda" (defaults).
// offset_key: OPTIONAL exp(x_i^T beta) from a sibling regression block.
cfg.initial_frailties = arma::vec(cfg.G, arma::fill::ones);
composite->add_child(std::make_unique<frailty_gamma_gibbs_block>(std::move(cfg)));
```

**theta hyperparameter**: `theta` is NOT Gamma-conjugate under the frailty-with-
Gamma prior (the true marginal is an intractable Antoniak sum). Supported patterns:
1. **FIXED theta**: `cfg.theta > 0`, `cfg.theta_key = ""`. Sensible when the user
   has a domain-informed guess or is doing sensitivity analysis over `theta`.
2. **INFERRED theta**: `cfg.theta_key = "theta"`. A sibling block (e.g.
   `nuts_block` on `log theta`, or `univariate_slice_sampling_block`) samples
   `theta` each sweep and writes it to `shared_data["theta"]`; this block reads
   it fresh each step.

**USAGE (shared-frailty PEH survival with Cox linear regression)**:

```cpp
// Sibling blocks:
//   nuts_block("beta") writes "exp_lp" = exp(X * beta) each sweep (length n).
//   frailty_gamma_gibbs_block("w") writes "w" (length G).
//   nuts_block("log_theta") writes "theta" (length 1) each sweep.
//   [refresher] writes "full_offset[i]" = w[z[i]] * exp_lp[i]  (length n).
//   piecewise_exponential_gibbs_block("lambda") reads "full_offset" via its offset_key.
```

The frailty block does NOT combine `w[g(i)]` with the regression offset itself
-- that combination lives in a shared_data refresher (a couple of lines in the
user's composite) so that (a) each of the two blocks has ONE clean output, and
(b) alternative frailty-distribution primitives (log-normal, Positive Stable,
Inverse-Gaussian) can drop in without changing the PEH block's contract.

**JUSTIFICATION (Check #16): Exception 3** (textbook multivariate conjugate
with NUTS-wasteful profile). `joint_nuts_block` on `log w[G]` for G=30..500
mixes slowly under any diagonal metric because the per-group posterior is
Gamma-tailed and inter-group posterior is weakly correlated only via `theta`;
the exact Gibbs update is ~50-100x faster and iid across groups.

**Check #15** parity:
`tests_autodiff/block_tests/test_frailty_gamma_gibbs_block.cpp` verifies BOTH
D_g / H_g accounting AND the Gamma(theta + D_g, theta + H_g) sample distribution
on a hand-computed 2-group / 4-subject / 3-bin fixture at 20 000 draws, within
5 % mean / 15 % variance tolerance.

**Do NOT use** when:
- frailty distribution is not Gamma (log-normal / Positive Stable / Inverse-
  Gaussian): those have different conjugate structures; distinct primitive is
  needed;
- the frailty enters non-multiplicatively (e.g. accelerated failure time
  `T_i = exp(f_i + eta_g) * epsilon`): different model class, closed-form breaks;
- theta itself must be inferred WITHOUT a sibling block sampling it: this block
  requires theta as either a fixed number or a sampled scalar in shared_data;
  it does NOT sample theta.

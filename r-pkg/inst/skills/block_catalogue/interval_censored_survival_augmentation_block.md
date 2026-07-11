## interval_censored_survival_augmentation_block

**Reference example:** compose with `piecewise_exponential_gibbs_block` (baseline
hazard) for interval-censored PEH survival regression -- see the "USAGE" block below.

Tanner-Wong (1987) data augmentation for INTERVAL-censored survival data under a
piecewise-exponential (PEH) baseline hazard. For each subject i whose event time
is only known to lie in `(L_i, U_i]`, this block samples the latent event time
`T_i` from the truncated PEH conditional
```
T_i | lambda, f_i  ~  f_PEH( . )  restricted to (L_i, U_i]
```
given the current baseline hazard `lambda[K]` (sampled by a sibling
`piecewise_exponential_gibbs_block`) and per-subject log-relative-hazard `f_i`
(fixed or produced by another sibling block). Right-censoring is a supported
special case: set `U_i = std::numeric_limits<double>::infinity()`.

**Model** (Ibrahim, Chen & Sinha 2001 Sec.3.5; Tanner & Wong 1987 JASA 82:528-540;
Sinha, Chen & Ghosh 1999 Biometrics 55(2):585; Lin & Wang 2010 Stat. Med.
29(29):3059-3072): let `H_0(t)` be the PEH cumulative baseline hazard and
`S_i(t) = exp(-exp(f_i) * H_0(t))`. Given `L_i < T_i <= U_i`, the conditional
CDF `F(t | L < T <= U) = [S(L) - S(t)] / [S(L) - S(U)]` inverts EXACTLY in
closed form for PEH -- no rejection sampling, no autodiff:
```
Draw u ~ Uniform(0, 1)
Set   H_target = H_0(L) - log(1 - u * (1 - exp(-w * (H_0(U) - H_0(L))))) / w        (w = exp(f_i))
Locate bin k such that cumH[k-1] <= H_target < cumH[k]
Return T = e_{k-1} + (H_target - cumH[k-1]) / lambda_k
```
For `U = +Inf` this degenerates to `H_target = H_0(L) - log(1-u) / w` (the exponential
tail). O(n + K) per step; all `n` subjects updated independently.

**Why augmentation?** The observed-data likelihood for an interval-censored
subject is `integral_{L}^{U} f(t) dt = S(L) - S(U)`, which BREAKS the
Gamma-conjugacy that makes PEH Gibbs cheap. Introducing `T_i` as a latent
augmentation parameter restores full conjugacy: on the augmented complete data,
the sibling `piecewise_exponential_gibbs_block` runs its usual Gamma-Poisson
step on `lambda[K]` treating `T_i` as observed with `delta_i = 1`. The joint
`(T_latent, lambda)` sweep is exact MCMC on the interval-censored posterior for
`lambda`.

```cpp
#include "AI4BayesCode/interval_censored_survival_augmentation_block.hpp"

interval_censored_survival_augmentation_block_config cfg;
cfg.name          = "t";                             // shared_data + history key (writes length-n)
cfg.edges         = edges;                           // MUST match sibling PEH block
// L_key="L", U_key="U", lambda_key="lambda" (defaults) -- must be in shared_data.
// offset_key: OPTIONAL exp(f_i); "" or missing => all exp(f_i) = 1.
cfg.initial_times = arma::vec{ /* each in (L[i], U[i]] */ };
composite->add_child(std::make_unique<interval_censored_survival_augmentation_block>(std::move(cfg)));
```

**USAGE (interval-censored PEH survival regression):**

```cpp
// Sibling blocks:
//   piecewise_exponential_gibbs_block("lambda") reads "t","delta","exp_lp" from shared_data
//   nuts_block("beta") writes "exp_lp" = exp(X * beta) each sweep
//   interval_censored_survival_augmentation_block("t") reads "L","U","lambda","exp_lp"
// Note: "delta" is fixed at 1.0 for the interval-censored subjects (event
// occurred by construction inside (L, U]); pre-processing separates them from
// exactly-observed subjects, which are shared_data'd in with fixed t, delta.
```

**Right-censoring vs interval-censoring**:
- Interval-censored (`U_i < +Inf`): T_i lands in `(L_i, U_i]` by construction.
- Right-censored (`U_i = +Inf`, encoded via `std::numeric_limits<double>::infinity()`):
  T_i sampled from the PEH tail beyond L_i. **Set `edges[K]` MUCH LARGER than
  the plausible right-tail** (a T_i sampled beyond `edges[K]` is clipped to
  `edges[K]`, biasing the tail); typically set `edges[K] = 10x` the median
  observed follow-up.

**Exactly-observed subjects**: DO NOT include them in this block's data. Pass
them straight to the sibling PEH-Gibbs block as `(t = observed_time, delta = 1)`;
this block handles ONLY the interval- / right-censored subset.

**JUSTIFICATION (Check #16): Exception 3** (exact augmentation with a
closed-form truncated survival distribution that NUTS/HMC cannot mimic
efficiently: the interval boundary is a hard discontinuity in the marginal
observed-data likelihood).

**Check #15** parity:
`tests_autodiff/block_tests/test_interval_censored_survival_augmentation_block.cpp`
verifies both a finite interval-censored case (analytic truncated-exponential
moments) and a right-censored case (memoryless-tail moments) at 30 000 draws,
within 5 % mean / 15 % variance tolerance.

**Do NOT use** when:
- observations are fully observed (`t_i` exact, `delta_i` observed): pass them
  directly to `piecewise_exponential_gibbs_block`;
- the baseline hazard is not piecewise-exponential (e.g., Weibull, splines):
  the closed-form truncated CDF inversion in this block only works under PEH.

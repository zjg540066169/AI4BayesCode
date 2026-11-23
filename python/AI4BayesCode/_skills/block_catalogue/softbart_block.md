---
name: softbart_block
description: Soft BART (Linero & Yang 2018) -- Gaussian-mean tree ensemble with smooth logistic splits, for smooth response surfaces
---

# `softbart_block`

Wrapper around the vendored `softbart::softbart_model` kernel
(`bart_pure_cpp/src/softbart_model.h`), implementing Soft BART
(Linero & Yang 2018, JRSSB 80(5): 1087-1110). One `step()` runs one
tree-ensemble sweep.

Soft BART replaces BART's hard cutpoints with a smooth logistic activation
around a learned bandwidth `tau`, so a datum contributes to BOTH sides of a
split with weights that vary continuously with its position. The fitted
surface is differentiable, which matters when the true mean is smooth: hard
BART approximates a smooth curve with a staircase and needs many more trees
to hide the steps.

## Routing row

```
| Gaussian mean f, constant noise, response surface expected to be **SMOOTH** in the covariates | **`softbart_block`** (Linero & Yang 2018 Soft BART) | **(none -- f is unconstrained)** |
```

## WHEN to use

A real-valued response with constant variance whose mean is expected to vary
SMOOTHLY with the covariates -- dose-response curves, growth curves, spatial
trends, any setting where you would otherwise reach for a spline or a GP but
want BART's automatic interaction detection and variable selection.

## WHEN NOT to use

- **Constant variance, no smoothness expectation** -> `bart_block`. It is the
  faster conjugate BART and the default for a real-valued response; Soft BART's
  per-datum weighting costs more per sweep.
- **Non-Gaussian response** (Poisson / NB / logistic / heteroscedastic / AFT /
  beta / ...) -> `genbart_block`. `softbart_block` has a Gaussian likelihood
  only; there is no Soft-BART analogue of the generalized-BART ladder.
- **A genuinely step-like or discontinuous surface** (treatment thresholds,
  regime switches) -> `bart_block`. Smoothing across a real discontinuity
  costs accuracy exactly where it matters.
- **You need a non-Gaussian likelihood AND sparsity** -> `genbart_block`.
  (`cfg.dart = true` DOES work here -- unlike `bart_block`, the SoftBart
  kernel honours it via `update_s` / `update_alpha` -- so a Gaussian response
  needing variable selection can stay on `softbart_block`.)

## Geometry class

Not applicable: this is a tree-ensemble kernel, not a continuous target on
R^d. It sits outside the `geometry.md` Sec.11.1 taxonomy in the same way
`bart_block` does -- there is no unconstrained parameterization to give NUTS,
and no metric or warmup decision to make.

## Config

```cpp
AI4BayesCode::softbart_block_config cfg;
cfg.name                  = "f_softbart";   // shared_data key for f(X)
cfg.x_train               = X;              // n x p design matrix
cfg.y_init                = y;              // length n, initial response
cfg.working_response_key  = "softbart_target";  // refreshed each set_context
cfg.sigma_key             = "sigma";        // forwarded to update_step if present

cfg.ntrees                = 50;    // ensemble size
cfg.k                     = 2.0;   // leaf-prior scale (as in BART)
cfg.sigma_hat             = -1.0;  // <= 0 -> sqrt(RSS/n) of the OLS fit of y on [1, X]; scale of y
cfg.y_scale               = -1.0;  // <= 0 -> max(y_init) - min(y_init) (1 if 0); see below
cfg.alpha                 = 1.0;   // tree-prior base
cfg.beta                  = 2.0;   // tree-prior power
cfg.gamma                 = 0.95;
cfg.width                 = 0.1;   // initial branch bandwidth tau
cfg.shape                 = 1.0;
cfg.tau_rate              = 10.0;  // Gamma rate on tau
cfg.center_Y              = true;  // m = mean(Y): f excludes it; false: m = midrange(y_init)
cfg.dart                  = false; // true -> Dirichlet split prior (honoured)
```

Pass `X` and `y` on their own scale. As SoftBart does, the block maps each
column of `X` to `[0, 1]` by the ranks of its distinct training values (new
`X` linearly between them, clamped to `[0, 1]`; `set_X` makes the map again)
and fits `Z = (Y - m) / y_scale`, so the leaf prior has SoftBart's
calibration; `f`, `sigma`, `predict()` and the context `sigma` are on the
scale of `y`. When `y_init` is a placeholder (e.g. a latent variable started at
0), set `y_scale` to the expected range of the response.

Typical composite: `softbart_block` for the mean, plus an
`inv_gamma_gibbs_block` (or a `nuts_block` on `log sigma`) for the noise
scale, wired through `sigma_key` / `working_response_key` exactly as
`bart_block` is.

## Reference example

`examples/SoftBartNoise.cpp` -- Gaussian response with a smooth mean,
tri-module (standalone `int main` + Rcpp + pybind11).

## Failure modes specific to this block

- **Freeze is BLACKLISTED**, same as `bart_block` / `genbart_block`: a fitted
  ensemble cannot be restored from a stored value. `freeze()` raises with
  "freezing softbart_block not supported ... use predict_at()". Use
  `predict_at()` to score new data without changing the fit.
- **Stale working response.** `working_response_key` is refreshed on every
  `set_context`, so the block MUST appear in the composite's
  `declare_dependencies` for the residual key. Omit it and the ensemble keeps
  fitting the previous sweep's residual -- it converges to the wrong surface
  with no error.
- **`center_Y = true` is on by default** and subtracts `mean(Y)`. If a sibling
  block also models an intercept, the two absorb the same location and the
  pair is only weakly identified.
- **Bandwidth `tau` and `ntrees` trade off.** A small `width` with few trees
  gives a nearly-hard ensemble (no smoothness gain, higher cost than
  `bart_block`); if the fit looks like hard BART, that is the first thing to
  check.

## License

GPL-2.0-or-later, matching the upstream SoftBart R package. The wrapper is
backend-neutral (no `Rcpp::` / `R::` symbols; errors via `ai4b::stop`).

## CITE

- Linero, A. R. and Yang, Y. (2018). Bayesian regression tree ensembles that
  adapt to smoothness and sparsity. *JRSS-B* 80(5): 1087-1110.
- `system_design Sec.13` (`families.md`) for the BART-family conventions this
  block shares.

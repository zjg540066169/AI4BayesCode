## Reduced-rank / spline / spatial patterns (HSGP, B-spline, ICAR)

The recipes above cover full-rank GPs (Cholesky of `K(X, X)` per draw).
For **basis-expanded / non-centered structured priors** -- HSGP, splines,
ICAR, BYM2 -- a different architectural pattern applies:

| Example | Pattern | When to use |
|---|---|---|
| `HSGPRegression.cpp` | 1-D Hilbert-space reduced-rank GP. `f(x) = sum_m sqrt(spd(ell)_m) * z_m * phi_m(x)`. | Smooth function regression with O(M) cost (vs O(N^3) full GP). M = 20-50 basis. brms-style. |
| `BSplineRegression.cpp` | Penalised B-spline. `f(x) = Bsp(x) . (sds * z)` non-centered. | 1-D smoothing without GP cost. Smoothing parameter `sds` learned from data. |

**Architectural rule for these patterns** (different from the full-rank
GP recipes above):

1. Reparameterise EVERY positive scalar to log scale. Treat them as
   REAL parameters of `joint_nuts_block`.
2. Put ALL parameters (`Intercept`, all log-scales, all latent vectors)
   into ONE `joint_nuts_block`.
3. Set `cfg.use_dense_metric = true`. The Welford pilot covariance
   handles the (Intercept, basis-coef-mean) ridge AND the
   (positive-scale, latent vec) funnel in one pass.
4. Manually add `log|Jacobian| = +log_scale` for each log-transformed
   positive scalar inside the user log-density. The block does NOT add
   Jacobians for REAL sub-params.

**Why all-REAL + dense metric (not per-slice POSITIVE + identity)**:
on the funnel/ridge geometry of HSGP/spline/ICAR an identity metric
consistently produces ESS = 1-3 on the smoothing scale or chain-stuck
behaviour at moderate chain lengths. Log-transform the positive scalars
to REAL and turn on the dense metric so the Welford pilot covariance can
capture the off-diagonal coupling.

**Why not separate blocks per parameter**: blocking
`(Intercept | log_amp | log_ell | z)` separately preserves the
within-block strong correlations as cross-block conditionals, and the
per-block step size adapts to those tight correlations. The chain
mixes within each block but takes O(rho) sweeps to traverse the joint
posterior, where rho is the cross-block correlation; for HSGP/spline/
ICAR rho is large enough that this dominates the wall time.

**Pattern check before writing the cpp** (metric is a Check #18 escalation,
MEASURED -- there is no "Check #11.7"):
- Joint-block dim >= 5 with at least one POSITIVE scalar
  AND a basis-coefficient or latent vector?
  -> use `joint_nuts_block` (real-only), log-transform the POSITIVE scalars
     (manually add log|J|); START DIAGONAL and escalate to
     `use_dense_metric = true` (Check #18) ONLY if R2/R3 shows diagonal is
     inadequate -- do NOT gate dense on the dimension.
- Joint-block dim < 5, all POSITIVE, no latent vector?
  -> `joint_nuts_block` with identity metric is fine
     (this is what `GPRegression.cpp` and `GPClassification.cpp` do
     for the 2-3 hyperparameters).

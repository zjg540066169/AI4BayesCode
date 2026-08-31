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

**This family is a MEASURED exception to the "START DIAGONAL" default**
(`codegen_cpp.md` Sec.4a). Adjacent spline bases have overlapping
support, so the coefficients are strongly correlated by construction --
the coupling is OFF-DIAGONAL and a per-axis metric cannot represent it.
Measured on the shipped `BSplineRegression` example (N = 100, 10 interior
knots, 13 parameters; 2 chains at different seeds, 1500 warmup + 2000
kept draws, cross-chain rank R-hat):

| metric | max rank R-hat | min ess_ratio | wall (2 chains) |
|---|---|---|---|
| adapted diagonal | 1.0288 (`z[8]`) | 0.0059 (`z[8]`) | 3.3s + 2.9s |
| **dense (shipped)** | **1.0017** (`sds`) | **0.3488** (`sds`) | **0.9s + 0.8s** |

Dense wins on all three axes: **59x the effective sample size**, R-hat
inside the strict 1.01 bar where diagonal misses it, and 3.6x FASTER --
diagonal mixes so poorly that NUTS builds much deeper trees per draw.
Note the diagonal `ess_ratio = 0.0059` sits right at the 0.005
escalation floor, i.e. diagonal is not merely slower, it is close to
failing the runtime gate outright. Start dense here; do NOT "start
diagonal and escalate" for this pattern.

**Why not separate blocks per parameter**: blocking
`(Intercept | log_amp | log_ell | z)` separately preserves the
within-block strong correlations as cross-block conditionals, and the
per-block step size adapts to those tight correlations. The chain
mixes within each block but takes O(rho) sweeps to traverse the joint
posterior, where rho is the cross-block correlation; for HSGP/spline/
ICAR rho is large enough that this dominates the wall time.

**Pattern check before writing the cpp.** The general rule elsewhere is
START DIAGONAL and escalate on measured evidence; THIS family is the
documented exception -- start dense (see the measurement table above).
Never gate the metric on dimension alone.
- Joint-block dim >= 5 with at least one POSITIVE scalar
  AND a basis-coefficient or latent vector?
  -> use `joint_nuts_block` (real-only), log-transform the POSITIVE scalars
     (manually add log|J|), and set `use_dense_metric = true` from the
     start: the basis coefficients are correlated by construction, so the
     diagonal metric has nothing to adapt to.
- Joint-block dim < 5, all POSITIVE, no latent vector?
  -> `joint_nuts_block` with identity metric is fine
     (this is what `GPRegression.cpp` does for its three
     hyperparameters). Once a latent vector joins the block --
     `GPClassification.cpp` carries `z[N]` alongside its two
     hyperparameters -- switch to the diagonal metric.

---
name: AI4BayesCode-constraints
description: |
  Reference for the 16 constraint transforms in AI4BayesCode. Consult this when
  choosing which transform to use and whether step-size seeding is needed.
---

# Constraint transforms

## Transform table

| Constraint | Natural space | Unconstrained via | Needs step-size seed? |
|------------|--------------|-------------------|----------------------|
| `real` | R | identity | No |
| `positive` | R+ | log | No |
| `simplex` | sum=1, all>0 | stick-breaking | **Yes** (0.05) |
| `lower_bounded` | (L, +inf) | log(x - L) | No |
| `upper_bounded` | (-inf, U) | log(U - x) | No |
| `interval` | (L, U) | logit | Tight intervals: Yes |
| `ordered` | x1 < x2 < ... | log-increments | **Yes** (0.05) |
| `cholesky_corr` | Cholesky of corr matrix | tanh partial correlations | **Yes** (0.05) |
| `unit_vector` | `norm(x) = 1` | auxiliary Gaussian | No |
| `positive_ordered` | 0 < x1 < x2 < ... | log first + log-increments | **Yes** (0.05) |
| `offset_multiplier` | R (affine reparam) | offset + multiplier * u | No |
| `sum_to_zero` | sum(x) = 0, K-1 free dims | orthogonal sum-to-zero basis | No |
| `cholesky_factor_cov` | lower-triangular Cholesky of a cov matrix | log-diagonal + free off-diagonal | **Yes** (0.05) |
| `corr_matrix` | K x K correlation matrix R = L L^T | composes `cholesky_corr` | **Yes** (0.05) |
| `cov_matrix` | K x K SPD covariance matrix | `cholesky_corr` + log scales | **Yes** (0.05) |
| `stochastic_column` | M x N, every COLUMN a simplex | per-column stick-breaking | **Yes** (0.05) |

## Usage pattern

Every constraint provides three functions: `constrain`, `unconstrain`, `wrap`.

```cpp
// In nuts_block_config:
cfg.constrain   = constraints::positive::constrain;
cfg.unconstrain = constraints::positive::unconstrain;
cfg.initial_unc = constraints::positive::unconstrain(init_natural);

// In log_density_grad lambda:
return constraints::positive::wrap(theta_unc, grad,
    [&](const arma::vec& theta_nat, arma::vec* grad_nat) {
        // Write natural-scale log p and gradient here.
        // DO NOT include Jacobian -- wrap handles it.
        return lp;
    });
```

For `real` blocks, omit `constrain`/`unconstrain` (identity transform).

## Step-size seeding

For `simplex`, `ordered`, `positive_ordered`, `cholesky_corr`, and tight
`interval` -- and for the matrix-valued transforms built on top of them
(`cholesky_factor_cov`, `corr_matrix`, `cov_matrix`, `stochastic_column`,
which reuse the same stick-breaking / tanh-partial-correlation geometry):

```cpp
cfg.initial_step_size   = 0.05;   // bypass FindReasonableEpsilon
cfg.n_warmup_first_call = 400;    // give dual averaging more runway
```

**Why:** the paper's `mu = log(10 * epsilon_1)` bias overshoots the stable
region for tight constrained geometries, causing NaN step sizes.

**Symptom:** chain stuck at initial value, step size is NaN or huge.

For `real`, `positive`, `unit_vector`, `lower_bounded`, `upper_bounded`,
`offset_multiplier`, and `sum_to_zero`: leave `initial_step_size` at default 0
(auto path works fine). The last two are affine / linear maps, so they do not
introduce the tight curvature that defeats the epsilon heuristic.

## Special notes

### simplex

- Unconstrained dimension is K-1 (stick-breaking).
- `constrain` returns a K-vector summing to 1.
- High-dimensional simplices (P > 100) are expensive per NUTS iter.

### cholesky_corr

- Unconstrained: K(K-1)/2 partial correlations.
- Natural: K^2 vector (column-major lower-triangular L, L*L' = corr matrix).
- LKJ(eta) prior: `lp = sum_{i=1}^{K-1} (K-i-1+2*eta-2) * log(L[i,i])`.

### unit_vector

- Same unconstrained dimension as natural (K).
- Auxiliary Gaussian convention: `log|J| = -0.5 * ||y||^2`.
- Well-conditioned geometry -- no step-size seeding needed.

## joint_constraint (for joint_nuts_block per-slice constraints)

When you use `joint_nuts_block` (see `block_catalogue/index.md`) the
per-slice constraint is declared with the `joint_constraint` enum,
NOT via the `constraints::<kind>::wrap` helpers. Supported:

**15 per-slice constraint kinds are supported** (validated 2026-06-17; see
`system_design.md` Sec.10 "joint_nuts_block per-slice constraints (current scope)"):

| `joint_constraint` | Semantics | dim |
|--------------------|-----------|-----|
| `REAL`                | identity transform | preserving |
| `POSITIVE`            | `nat = exp(unc)` (log transform) | preserving |
| `LOWER_BOUNDED`       | `nat = lb + exp(unc)` | preserving |
| `UPPER_BOUNDED`       | `nat = ub - exp(unc)` | preserving |
| `INTERVAL`            | `nat = lb + (ub-lb)*sigmoid(unc)` | preserving |
| `ORDERED`             | ordered (cumulative-exp) | preserving |
| `POSITIVE_ORDERED`    | positive AND ordered | preserving |
| `UNIT_VECTOR`         | unit-norm vector | preserving |
| `OFFSET_MULTIPLIER`   | `nat = offset + mult*unc` | preserving |
| `SIMPLEX`             | stick-breaking (K-1 -> K) | changing |
| `SUM_TO_ZERO`         | zero-sum (K-1 -> K) | changing |
| `CHOLESKY_CORR`       | Cholesky factor of a correlation matrix | changing |
| `CHOLESKY_FACTOR_COV` | Cholesky factor of a covariance matrix | changing |
| `CORR_MATRIX`         | correlation matrix | changing |
| `COV_MATRIX`          | covariance matrix | changing |

The block applies each per-slice transform and adds `log|J|` internally (via
`constraints::<kind>::wrap`); the user-written log-density stays on the **natural
scale** throughout and MUST NOT include any manual Jacobian terms (validator Check
#11.3 / #11.4). Dimension-CHANGING kinds (unconstrained width != natural width) use
a dual offset scheme internally; matrix-valued kinds (`CHOLESKY_*`, `CORR_MATRIX`,
`COV_MATRIX`) auto-enable the diagonal metric. Because all these kinds are
supported per-slice, do NOT split a constrained parameter (simplex, cholesky,
ordered, ...) into its own block -- keep it as a slice of the `joint_nuts_block`.

Standalone NUTS blocks keep using the full `constraints::*::wrap`
machinery above -- nothing about the `joint_constraint` enum changes
the API for `nuts_block`.

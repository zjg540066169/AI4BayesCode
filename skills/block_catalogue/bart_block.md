## bart_block (GPL-2.0+)

For `y = f(x) + epsilon`. One tree-sweep per `step()`. Sigma is handled
by a sibling block.

**Vendor (2026-05-03 onward):** uses the unified BART_unified library at
`bart_pure_cpp/src/`. Internally wraps `stdbart::bart_model` (the namespace name
is `stdbart` to avoid clash with the engine class also named `bart`;
the project-facing alias `bart::bart_model` is also valid). The Tier B
plug-in API (`set_X / set_Y / set_data / set_offset / set_s / get_s /
update_step / current_sigma / current_var_probs / current_var_counts /
get_invchi / startdart`) is uniform with `genbart_block`. softBART
(`softbart::softbart_model`) is also vendored under the same `bart_pure_cpp/`
tree (`bart_pure_cpp/src/SOFTBART_VENDOR`), and `softbart_block` ships with the same uniform interface --
see `examples/SoftBartNoise.cpp` for a worked example.

**For nested MCMC**: `BartNoise::set_current(Rcpp::List)` already
accepts any subset of `{X, y, sigma}` and routes them into the block
-- use this when composing BART inside an outer Gibbs. No other R
state methods are exposed; the unified core-6 state contract + the
kernel-control category (freeze / unfreeze / get_frozen + readapt_NUTS
iff any NUTS-family sibling in the composite) + the BART tree-serialization
carve-out (get_tree / set_tree / get_tree_history) is the whole API
per interface.md Sec.1. (System-design agents modifying `bart_block` itself should read
`skills/system_design.md`; code-generation agents do NOT need
to -- see the top of `skills/codegen.md`.)

### Composite / varying-coefficient BART via BACKFITTING -- do NOT reach for `genbart_block`

When the mean is an ADDITIVE combination of tree ensembles -- **VC-BART**
(Deshpande et al.: `mu_i = beta_0(Z_i) + sum_j beta_j(Z_i) * x_ij`, each
`beta_j` its own BART over a modifier `Z`), additive-BART, or any
`mu = sum_j g_j(Z) * x_j` -- the observation is **still Gaussian**, so compose
ONE `bart_block` per ensemble and Gibbs-**backfit** them. The AI's instinct to
switch to `genbart_block` with a custom likelihood is WRONG here: it throws away
`bart_block`'s conjugate leaves for a much slower RJMCMC, for no modelling gain.

The per-observation multiplier `x_ij` is handled by **`weights_key`**, NOT a
custom likelihood. Condition ensemble `j` on the others via the partial residual
and fit the SCALED response:

    r_i^(j) = y_i - sum_{k != j} x_ik * beta_k(Z_i)     (partial residual)
    ytilde_i = r_i^(j) / x_ij                            (working response)
    ytilde_i ~ N( beta_j(Z_i),  sigma^2 / x_ij^2 )

so ensemble `j` is a `bart_block` with:
- `x_train = Z`                       (the modifier -- NOT the predictors x)
- `working_response_key` <- `ytilde^(j)`, refreshed each sweep (the `/ x_ij` step)
- `weights_key`          <- `w_i = 1/|x_ij|`  (=> per-obs noise sd `w_i*sigma` via
                           `update_step_weighted`; reproduces VC-BART's weighted
                           leaf sufficient stats `sum x_ij^2` and `sum x_ij r_i` exactly)
- `sigma_key`            <- the shared scalar noise sd
The intercept ensemble `beta_0` has `x_i0 == 1` => `w_i = 1`, `ytilde = r^(0)`:
a plain unweighted `bart_block`.

Each sweep: for `j = 0..p` recompute `r^(j)`, `ytilde^(j)`, `w^(j)`, push to
shared_data, `step()` ensemble `j`; then one shared-`sigma` draw. That is `p+1`
CONJUGATE BART sweeps -- far faster than one genbart RJMCMC over a custom VC
likelihood, and statistically exact (identical weighted posterior).

**Edge case:** an observation with `x_ij == 0` carries no information about
`beta_j` (`beta_j(Z_i)*0 = 0`) and would give `w_i = 1/0`. Drop such rows from
ensemble `j`'s fit (equivalently their `x_ij^2` leaf weight is 0) -- never feed
`Inf` weights to the block.

**Worked example:** `examples/VCBart.cpp` compiles this recipe end-to-end (p+1 backfit ensembles, `weights_key` = 1/|x_ij|, shared sigma) and recovers the coefficient surfaces.

```cpp
bart_block_config cfg;
cfg.name                 = "f_bart";
cfg.x_train              = X;            // Rcpp::NumericMatrix
cfg.y_init               = y;            // Rcpp::NumericVector
cfg.working_response_key = "bart_target";
cfg.sigma_key            = "sigma";
cfg.ntrees               = 50;
cfg.sigma_init           = sd(y);   // placeholder, see note below

// --- Optional: sparse variable selection (Linero 2018 DART) ---
cfg.dart                 = false;   // enable DART sparsity prior
cfg.aug                  = false;   // Linero 2018 Section 4 augmentation

// --- Optional: probit-BART leaf prior tau formula (Albert-Chib regime) ---
cfg.binary               = false;   // see "Binary leaf prior" below
```

### Binary leaf prior (Albert-Chib probit BART)

`bart_block_config::binary` controls which leaf-prior tau formula
`bart_model` uses internally:

| `cfg.binary` | tau formula | When to use |
|--------------|-------------|-------------|
| `false` (default) | `(max(y_init) - min(y_init)) / (2 * k * sqrt(ntrees))` | Gaussian regression `y ~ N(BART(X), sigma^2)`. The data range gives a sensible leaf scale. |
| `true` | `3 / (k * sqrt(ntrees))` | **Probit BART via Albert-Chib data augmentation.** The BART working response is a latent z whose effective range under truncation is ~6 (~= +/-3 SD). Using the Gaussian formula on an arbitrary y_init scale (e.g. +/-1) over-shrinks the leaves by a factor of 3. This formula matches `BART::pbart` / `cpbart.cpp` line 244 exactly. |

Non-Gaussian likelihoods that are NOT data augmentation (e.g. logistic
BART via `genbart_block + genbart::lik::logistic_lik`) do not use
`bart_block` at all -- see the `genbart_block` section below.

When generating a probit-BART wrapper that uses Albert-Chib (the user
prompt says "data augmentation"), set `cfg.binary = true`. The
recipe is to compose `probit_aug_block` (latent z) with `bart_block`
(`cfg.binary = true`) in a single composite. The shipped
`examples/ProbitRegression.cpp` shows the Albert-Chib half; the
`bart_block` docstring describes the binary mode flag.

Uses R's RNG. `set.seed()` for reproducibility.

### Sparse variable selection (DART)

`bart_block_config` exposes two Linero (2018) flags for **implicit
variable selection** in high-p settings:

- **`dart = true`** -- replaces BART's uniform split-variable probabilities
  with a sparse Dirichlet prior. Variables with no signal have their
  split probability shrunk toward zero. Posterior-mean split frequency
  per variable gives a usable variable-importance / inclusion score
  without a separate spike-and-slab block.
- **`aug = true`** -- only active when `dart = true`. Enables the data-
  augmentation scheme in Linero 2018 Section 4 that accelerates mixing
  of the sparse-Dirichlet concentration parameter. Adds a small
  per-sweep cost and is not always beneficial -- leave OFF by default;
  flip ON only if DART's concentration parameter is visibly slow to mix.

**When to recommend this to the user:** the model is of the form
`y = f(X) + noise` with p moderate-to-large (say p >= 10) AND the user
indicates they care about which features matter (variable selection,
feature importance, interpretability). DART is often a cleaner fit than
wrapping BART inside a spike-and-slab selector -- it gets sparsity "for
free" from the tree prior.

**Decision-flow recipe** for the codegen skill:
- If the user's model description includes words like "variable selection",
  "which features matter", "sparse", "feature importance", or "high-
  dimensional", AskUserQuestion whether to enable DART.
- Recommended defaults when the user opts in: **`dart = TRUE, aug = FALSE`**.
  Describe `aug` in one sentence (mixing accelerator for the
  concentration parameter, adds per-sweep cost, turn on only if DART
  mixing is slow) and let the user pick. Offer `dart = FALSE` when the
  goal is pure prediction.
- Expose `dart` and `aug` as constructor arguments to the generated
  R wrapper **only if the user opts into DART** -- otherwise hardcode
  both to `false` inside the class body and do NOT surface them to R.
  See `examples/BartNoise.cpp` -- they are positional args 9 and 10 in
  that reference template's constructor.

All other BART hyperparameters (`ntrees`, `k`, `power`, `base`, `nu`,
`numcut`) keep BART::wbart's defaults and are passed through unchanged.

### Sigma initialization in bart_model (automatic)

**You do NOT need to compute hat_sigma in R or in AI-generated code.**
The `bart_model` constructor automatically:

1. Fits OLS internally (`arma::solve(X, y)`) when `p < n`
2. Computes `sigest = sqrt(RSS / (n - p))` from OLS residuals
3. Falls back to `sd(y)` only when `p >= n` (high-dimensional case)
4. Sets `lambda = sigest^2 * qchisq(0.1, nu) / nu` for the conjugate
   sigma prior (using a pure C++ lookup table, no R function calls)
5. Uses `sigest` as the initial sigma value

`cfg.sigma_init` is NOT used to override this -- `bart_block` does not
call `set_sigma` after construction. The internal OLS-based estimate
matches `BART::wbart`'s `sigest` default exactly.

### Recommended sigma prior for BART-structured models

In any model with likelihood `y ~ N(BART(X), sigma^2)`, sigma has a soft
identifiability problem -- BART's flexible f(x) can absorb signal that
should go into the noise variance, and with a weakly informative prior
sigma can drift anywhere the under-fit bias permits. The
default for this class of model is BART::wbart's **calibrated conjugate
inverse-gamma** prior (Chipman, George & McCulloch 2010):

    sigma^2 ~ InverseGamma(nu/2, nu*lambda/2)
    lambda  = sigest^2 * qchisq(0.1, nu) / nu      (sigquant = 0.9)
    nu = 3 (default)

`sigest` is the OLS residual SD, computed automatically by `bart_model`
and retrievable via `bart_block::current_sigma()` after the BART child
has been added to the composite.

**Decision-flow recipe** (follow `skills/codegen.md` Section 2):
- When parsing a user model with a BART mean and no explicit sigma prior,
  run the AskUserQuestion decision flow exactly as for any other
  parameter, but substitute this calibrated IG in place of the generic
  Jeffreys `p(sigma) prop.to 1/sigma` scale default (the default-priors table).
- Present it as option (b) "Default" and tell the user WHY
  (identifiability / under-fit absorption).
- Offer `HalfNormal(0, 10)` or a user-specified alternative as option (c).
- Still accept an explicit user-supplied prior verbatim.

**Sampling stays uniform:** sigma is still a standard `nuts_block` -- only
the prior term in the log-density changes. Do NOT introduce a custom
conjugate-Gibbs block for sigma; mixing closed-form conjugate draws with
a NUTS-first generator is explicitly prohibited (see
`skills/codegen.md` Hard Rules). The only Gibbs blocks the generator may
use are the `*_gibbs_block` headers already shipped in
`include/AI4BayesCode/` -- see the "Gibbs blocks" sections of this catalogue
for the current roster.

See `examples/BartNoise.cpp` for the reference template -- both the
`R::qchisq` calibration in the constructor and the IG log-density in the
`nuts_block` lambda.

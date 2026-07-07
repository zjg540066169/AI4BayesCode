---
name: AI4BayesCode-hierarchical-re
audience: |
  Code-generating agent. Triggered from `codegen.md` when the user
  prompt contains a hierarchical random-effects pattern. ALWAYS read
  this in full before emitting a hierarchical sampler.
description: |
  Authoritative recipe for code-generating samplers for any
  hierarchical model with group-level Gaussian random effects
  (theta_j ~ Normal(mu, sigma)). Covers pattern recognition,
  the funnel geometry problem, the MANDATORY non-centered
  reparameterization, joint_nuts_block slice ordering,
  log-density and gradient structure, dense-metric escalation
  rule, crossed-effects (multiple hierarchies), and a
  validator checklist. NO new blocks are required for any
  Gaussian-RE model -- `joint_nuts_block` already in the
  library handles every case described here.
---

# Hierarchical random effects -- codegen recipe

## 1. When this skill applies

Trigger on any user prompt containing all of:

1. A vector of group-level effects sampled from a normal:
   `theta_j ~ Normal(mu, sigma_alpha)` for `j = 1, ..., J`.
2. The group effects entering the likelihood through a group-index
   lookup: e.g. `y_n ~ Normal(theta_{group(n)} + ..., sigma_y)`,
   `y_n ~ Bernoulli(invlogit(theta_{group(n)} + ...))`,
   `y_n ~ Poisson(exp(theta_{group(n)} + ...))`.
3. Hyperpriors on `(mu, sigma_alpha)` that are NOT trivially
   constant in the prompt (i.e. some prior is specified, even if
   improper / noninformative).

Pattern names you will see in prompts:
- "varying-intercept regression with partial pooling"
- "random intercept" / "random slope" / "random effect"
- "exchangeable" group effects
- "multilevel model"
- "hospital effect" / "subject effect" / "school effect" / etc.
- "crossed effects" (apply this skill TWICE -- see Sec.6)

Concrete examples in this benchmark:
- `radon_variable_intercept_centered` (1 RE level, 1 fixed slope)
- `radon_hierarchical_intercept_centered` (1 RE level, 2 fixed slopes)
- `radon_hierarchical_intercept_noncentered` (same model, prompt
  explicitly notes NC -- but the codegen rule below is NC regardless)
- `pilots` (CROSSED effects: groups + scenarios -- see Sec.6)
- `seeds_model`, `surgical_model` (binomial likelihood; same recipe
  applies -- see Sec.7)

## 2. The funnel geometry -- why naive composition silently fails

The naive composition is to assign one `nuts_block` per parameter:

```
nuts_block(mu)             reads (alpha, sigma_alpha)
nuts_block(sigma_alpha)    reads (alpha, mu)
nuts_block(alpha[1..J])    reads (y, mu, sigma_alpha, sigma_y)
nuts_block(sigma_y)        reads (y, alpha, ...)
```

This converges to the correct posterior in the limit but mixes
catastrophically because of the **funnel** (Neal 2003,
Betancourt-Girolami 2015):

- When `sigma_alpha` is small, `alpha_j | (mu, sigma_alpha)` is
  highly concentrated around `mu`. The chain's NUTS step on
  `alpha_j` is over-confident; the next `sigma_alpha` proposal
  is rejected; `sigma_alpha` stays small. Funnel.
- The joint posterior has thin regions in `(sigma_alpha, alpha_j)`
  that no per-block NUTS can step across with a diagonal mass
  matrix. Even a properly-tuned single-parameter NUTS step is
  wrong-scaled for the geometry.

**Empirical evidence (sim1 cross-impl rhat against Stan reference):**

| Model | Naive composition | rhat | cov vs truth |
|---|---|---|---|
| radon_variable_intercept_centered | modular | 1.64 | **AI 0.75 vs Stan 0.94** |
| radon_hierarchical_intercept_noncentered | modular (with NC reparam) | 2.23 | **AI 0.12 vs Stan 0.94** |
| surgical_model | modular | 2.02 | **AI 0.07 vs Stan 0.93** |

The "noncentered" variant fails just as badly as the centered one,
because **the funnel returns when alpha_raw and sigma_alpha are in
separate Gibbs blocks** -- the conditional `alpha_raw | sigma_alpha`
still has `sigma_alpha`-dependent geometry under the data likelihood
(the inverse funnel; Betancourt-Girolami Sec.3.3). Block separation IS
the bug, not the parameterization choice.

## 3. The MANDATORY rule

For any hierarchical model matching Sec.1, the wrapper MUST:

1. Use a **single `joint_nuts_block`** block targeting all of
   `(mu, sigma_alpha, sigma_y, alpha_raw[1..J], beta)` together --
   never split these across multiple blocks.
2. Use the **non-centered reparameterization**:
   `alpha_j = mu + sigma_alpha * alpha_raw_j`,
   with `alpha_raw_j ~ Normal(0, 1)` as the SAMPLED parameter.
3. Compute `alpha_j` deterministically AFTER each step in the
   wrapper and write it to shared_data, so y_rep and downstream
   generated quantities can read it.

This rule applies REGARDLESS of:
- Whether the user prompt uses centered or non-centered notation.
  Always implement NC.
- The number of groups J. There is no "small enough J" exception.
- The number of fixed-effects coefficients P. Always include `beta`
  in the same joint block.

## 4. Slice ordering for `joint_nuts_block`

Build the joint vector in this order (this is a convention; the
log-density function does its own slicing, but consistency makes
debugging much easier):

```
slice 0: mu_alpha           dim = 1, type = real_constraint
slice 1: sigma_alpha        dim = 1, type = positive_constraint
slice 2: sigma_y            dim = 1, type = positive_constraint
slice 3: alpha_raw          dim = J, type = real_constraint
slice 4: beta               dim = P, type = real_constraint   (omit if P = 0)
```

Total joint dim: `3 + J + P`. For radon_variable_intercept (J=12, P=1)
this is 16-dim. For radon_hierarchical (J=12, P=2) it is 17-dim.

**Composite ordering**: this joint block is the ONLY child block of
the wrapper's `composite_block`. Do NOT add separate sibling blocks
for `mu`, `sigma_alpha`, `sigma_y`, or any subset of `alpha_raw`.

## 5. Log-density structure

The user supplies `joint_nuts_block_config::log_density_grad`
on the **natural-scale** concatenated vector. The block applies
per-slice constraint transforms internally (log for positive,
identity for real) and adds `log|J|` for each -- your function ONLY
writes the natural-scale density and natural-scale gradient.

```cpp
cfg.log_density_grad = [&, J, P, group_idx_key, y_key, x_key,
                       mu_prior_mean, mu_prior_sd,
                       sigma_alpha_prior_scale, sigma_y_prior_scale,
                       beta_prior_sd]
    (const arma::vec& theta_cat,
     const block_context& ctx,
     arma::vec* grad_nat) -> double {

    // 1. Slice the joint vector.
    const double mu          = theta_cat[0];
    const double sigma_alpha = theta_cat[1];     // > 0 (constraint enforced)
    const double sigma_y     = theta_cat[2];     // > 0 (constraint enforced)
    const arma::vec alpha_raw = theta_cat.subvec(3, 3 + J - 1);
    arma::vec beta;
    if (P > 0) beta = theta_cat.subvec(3 + J, 3 + J + P - 1);

    // 2. Reconstruct alpha = mu + sigma_alpha * alpha_raw.
    const arma::vec alpha = mu + sigma_alpha * alpha_raw;

    // 3. Read data + group index from ctx.
    const arma::vec& y    = ctx.at(y_key);
    const arma::vec& gidx = ctx.at(group_idx_key);   // 1-indexed double
    const std::size_t N = y.n_elem;
    arma::mat X;
    if (P > 0) {
        const arma::vec& X_flat = ctx.at(x_key);
        X = arma::mat(const_cast<double*>(X_flat.memptr()), N, P, false, true);
    }

    // 4. Likelihood: y_n ~ Normal(alpha_{g(n)} + (X*beta)_n, sigma_y).
    double log_lik = -static_cast<double>(N) * std::log(sigma_y);
    arma::vec resid(N);
    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t g = static_cast<std::size_t>(std::lround(gidx[n])) - 1;
        double mu_n = alpha[g];
        if (P > 0) {
            for (std::size_t p = 0; p < P; ++p) mu_n += X(n, p) * beta[p];
        }
        resid[n] = y[n] - mu_n;
        log_lik -= 0.5 * resid[n] * resid[n] / (sigma_y * sigma_y);
    }

    // 5. Priors.
    // mu: Normal(mu_prior_mean, mu_prior_sd^2). Skip if flat (mu_prior_sd = +inf).
    double log_prior_mu = 0.0;
    if (std::isfinite(mu_prior_sd)) {
        const double dmu = (mu - mu_prior_mean) / mu_prior_sd;
        log_prior_mu = -0.5 * dmu * dmu;
    }

    // sigma_alpha: HalfNormal(0, sigma_alpha_prior_scale).
    // log p(sigma_alpha) = const - 0.5 * sigma_alpha^2 / scale^2
    // (Jeffreys 1/sigma_alpha would be log p = -log(sigma_alpha); covered by
    //  using sigma_alpha_prior_scale = +inf and adding -log(sigma_alpha).)
    double log_prior_sigma_alpha;
    if (std::isfinite(sigma_alpha_prior_scale)) {
        log_prior_sigma_alpha =
            -0.5 * sigma_alpha * sigma_alpha
            / (sigma_alpha_prior_scale * sigma_alpha_prior_scale);
    } else {
        // Jeffreys: log p prop.to -log(sigma_alpha). NB: positive_constraint adds
        // +log(sigma_alpha) Jacobian, so the natural-scale log_p is just
        // -log(sigma_alpha). Total target on unconstrained scale = 0
        // contribution from the prior -- Jeffreys cancels.
        log_prior_sigma_alpha = -std::log(sigma_alpha);
    }

    // sigma_y: HalfNormal(0, sigma_y_prior_scale) (same pattern).
    double log_prior_sigma_y;
    if (std::isfinite(sigma_y_prior_scale)) {
        log_prior_sigma_y =
            -0.5 * sigma_y * sigma_y
            / (sigma_y_prior_scale * sigma_y_prior_scale);
    } else {
        log_prior_sigma_y = -std::log(sigma_y);
    }

    // alpha_raw ~ Normal(0, 1).
    double log_prior_alpha_raw = 0.0;
    for (std::size_t j = 0; j < J; ++j) {
        log_prior_alpha_raw -= 0.5 * alpha_raw[j] * alpha_raw[j];
    }

    // beta ~ Normal(0, beta_prior_sd^2). Skip if flat.
    double log_prior_beta = 0.0;
    if (std::isfinite(beta_prior_sd) && P > 0) {
        for (std::size_t p = 0; p < P; ++p) {
            const double db = beta[p] / beta_prior_sd;
            log_prior_beta -= 0.5 * db * db;
        }
    }

    const double lp = log_lik + log_prior_mu
                    + log_prior_sigma_alpha + log_prior_sigma_y
                    + log_prior_alpha_raw + log_prior_beta;

    if (grad_nat) {
        // Hand-computed natural-scale gradient.
        // 5a. d/dmu of log_lik: sum_n (y_n - mu_n) / sigma_y^2 (the same
        //     residual contributes to alpha_g). d alpha_g / d mu = 1.
        //     So d log_lik / d mu = sum_n resid[n] / sigma_y^2.
        // 5b. d/dsigma_alpha: d alpha_g / d sigma_alpha = alpha_raw[g_n].
        //     d log_lik / d sigma_alpha
        //       = sum_n (resid[n] / sigma_y^2) * alpha_raw[g_n].
        // 5c. d/dsigma_y: -N/sigma_y + sum_n resid[n]^2 / sigma_y^3.
        // 5d. d/dalpha_raw_j: sigma_alpha * sum_{n: g(n)=j} (resid[n]/sigma_y^2)
        //                  - alpha_raw_j (from prior).
        // 5e. d/dbeta_p: sum_n (resid[n] / sigma_y^2) * X(n, p)
        //                - beta_p / beta_prior_sd^2 (if proper).

        grad_nat->zeros(theta_cat.n_elem);
        const double inv_sy2 = 1.0 / (sigma_y * sigma_y);
        double dmu = 0.0;
        double dsa = 0.0;
        double dsy_lik_part = 0.0;
        for (std::size_t n = 0; n < N; ++n) {
            const std::size_t g = static_cast<std::size_t>(std::lround(gidx[n])) - 1;
            const double w_n = resid[n] * inv_sy2;
            dmu               += w_n;
            dsa               += w_n * alpha_raw[g];
            dsy_lik_part      += resid[n] * resid[n];
            (*grad_nat)[3 + g] += sigma_alpha * w_n;
            if (P > 0) {
                for (std::size_t p = 0; p < P; ++p) {
                    (*grad_nat)[3 + J + p] += w_n * X(n, p);
                }
            }
        }
        (*grad_nat)[0] = dmu;
        (*grad_nat)[1] = dsa;
        (*grad_nat)[2] = -static_cast<double>(N) / sigma_y
                       + dsy_lik_part / (sigma_y * sigma_y * sigma_y);

        // Add prior contributions to gradient.
        if (std::isfinite(mu_prior_sd)) {
            (*grad_nat)[0] -= (mu - mu_prior_mean)
                            / (mu_prior_sd * mu_prior_sd);
        }
        if (std::isfinite(sigma_alpha_prior_scale)) {
            (*grad_nat)[1] -= sigma_alpha
                            / (sigma_alpha_prior_scale
                               * sigma_alpha_prior_scale);
        } else {
            (*grad_nat)[1] -= 1.0 / sigma_alpha;
        }
        if (std::isfinite(sigma_y_prior_scale)) {
            (*grad_nat)[2] -= sigma_y
                            / (sigma_y_prior_scale * sigma_y_prior_scale);
        } else {
            (*grad_nat)[2] -= 1.0 / sigma_y;
        }
        for (std::size_t j = 0; j < J; ++j) {
            (*grad_nat)[3 + j] -= alpha_raw[j];
        }
        if (std::isfinite(beta_prior_sd) && P > 0) {
            for (std::size_t p = 0; p < P; ++p) {
                (*grad_nat)[3 + J + p] -=
                    beta[p] / (beta_prior_sd * beta_prior_sd);
            }
        }
    }
    return lp;
};
```

**Check #12 (autodiff gradient verification) is REQUIRED** for this
hand-coded gradient. Before shipping, the codegen agent writes
`tests_autodiff/verify_<wrapper>.cpp` with a templated `autodiff::var`
version of the same density and verifies max_diff < 1e-8 on 5-20
random `(mu, sigma_alpha, sigma_y, alpha_raw, beta)` points, then
deletes the verify file. See `validator.md Sec.12`.

## 6. Crossed effects (pilots-style)

For two independent hierarchies sharing `sigma_y`:

```
y_n ~ Normal(a_{g(n)} + b_{s(n)}, sigma_y)
a_g ~ Normal(mu_a, sigma_a)    g = 1..G
b_s ~ Normal(mu_b, sigma_b)    s = 1..S
```

Use TWO `joint_nuts_block` instances:

- **Block 1** (slice ordering): `(mu_a, sigma_a, sigma_y, a_raw[G])`
- **Block 2** (slice ordering): `(mu_b, sigma_b, b_raw[S])`
  -- Block 2 reads `sigma_y` from `ctx`, treats it as a constant
  during its own NUTS trajectory.

Composite Gibbs sweep:
1. Block 1 steps -> updates `(mu_a, sigma_a, sigma_y, a_raw)`.
2. Block 2 steps -> reads new `sigma_y` from ctx, updates
   `(mu_b, sigma_b, b_raw)` against current `sigma_y`.

Both blocks compute their own residuals using the OTHER hierarchy's
current `a` (or `b`) values. Wrapper computes `a = mu_a + sigma_a *
a_raw` and `b = mu_b + sigma_b * b_raw` after each step, writes to
shared_data.

This is correct under the standard Gibbs-on-conditional argument:
each block targets its own conditional given the other block's
current state. **DO NOT** put both hierarchies into ONE
`joint_nuts_block` with all `(mu_a, sigma_a, mu_b, sigma_b,
sigma_y, a_raw, b_raw)` jointly -- the joint dim becomes
`3 + G + S`, a needlessly large block that would likely need the
dense metric. Two-block Gibbs avoids that and is mathematically
equivalent because the two RE structures are conditionally
independent given `sigma_y`.

## 7. Non-Gaussian likelihoods (binomial, Poisson, etc.)

The recipe in Sec.3-5 generalizes: replace the Gaussian log-likelihood
with the appropriate family. For example, binomial:

```
y_n ~ Binomial(n_n, p_n)
logit(p_n) = alpha_{g(n)} + (X * beta)_n
alpha_j    = mu + sigma_alpha * alpha_raw_j  (NC, same as Sec.3)
```

The joint slice now has NO `sigma_y`:
```
slice 0: mu_alpha           dim = 1, type = real
slice 1: sigma_alpha        dim = 1, type = positive
slice 2: alpha_raw          dim = J, type = real
slice 3: beta               dim = P, type = real
```

Log-likelihood becomes
`log p(y | alpha, beta) = sum_n [y_n * eta_n - n_n * log1p(exp(eta_n))]`
where `eta_n = alpha_{g(n)} + (X * beta)_n`. Gradient is hand-computable
and structurally identical to Sec.5 (replace `resid_n / sigma_y^2` with
the binomial residual `y_n - n_n * sigmoid(eta_n)`).

For Poisson `y_n ~ Poisson(exp(eta_n))`, the residual becomes
`y_n - exp(eta_n)`. Same structure, same recipe.

This covers `surgical_model` (binomial RE), `seeds_model` (binomial RE
with interaction terms), and any GLMM with a single hierarchy.

## 8. Dense metric escalation (Check #18)

If 2-chain rhat > 1.05 at full 20k+20k budget on the joint posterior
even with the rule above:

```cpp
cfg.use_dense_metric = true;
```

and add a Check #18 inline justification:

```cpp
// JUSTIFICATION (Check #18): hierarchical funnel residual after NC
// reparameterization, J=<n>, observed rhat = <r> at full budget on
// diagonal metric. Dense metric required to capture (mu_alpha,
// alpha_raw) cross-correlation in the data-rich-group regime.
```

Common signals that dense metric MAY be needed (escalate via Check #18 only when
diagnostics confirm -- start diagonal, measure, do NOT gate on dimension):
- large `J` (dense is frequently needed, but confirm via R-hat -- not automatic)
- Highly variable group sizes (some `n_j` >> others, creating
  scale heterogeneity within `alpha_raw`)
- Heavy-tail random effects where `sigma_alpha_prior_scale` is
  large and the chain visits the tail

## 9. Validator checklist for hierarchical RE wrappers

Before declaring a hierarchical wrapper complete, verify:

- [ ] Single `joint_nuts_block` (or two for crossed effects);
      NO separate `nuts_block` for any of
      `(mu_alpha, sigma_alpha, sigma_y, alpha_raw, beta)`.
- [ ] Slice ordering matches Sec.4 (or Sec.6 for crossed).
- [ ] `alpha_raw` is the sampled parameter; `alpha = mu + sigma_alpha
      * alpha_raw` is computed deterministically in the wrapper after
      each step and written to `shared_data["alpha"]`.
- [ ] `y_rep` stochastic refresher reads `alpha` (not `alpha_raw`)
      and `beta` from shared_data.
- [ ] Hand-coded gradient + autodiff verification (Check #12) PASS.
- [ ] 2-chain rhat at 20k+20k < 1.05 on `(mu_alpha, sigma_alpha,
      sigma_y, alpha_raw)`. If not: enable `use_dense_metric` with
      Check #18 justification.
- [ ] Cross-impl rhat against Stan reference (sim1) < 1.05 on
      `(mu_alpha, sigma_alpha, sigma_y, alpha)` with appropriate
      relabeling (none needed -- alpha_raw + recovered alpha are
      identifiable; no label switching at the group-effect level).

## 10. Anti-patterns to reject in review

These are the top failure modes that produced cov_AI ~= 0 on the
broken radon and surgical models. Reject these patterns at review:

1. `nuts_block(sigma_alpha)` and `nuts_block(alpha)` as separate
   children. The funnel is between these two -- they MUST be in
   the same joint block.
2. `categorical_gibbs` or any other Gibbs sampling on the group
   index `g(n)`. The group index is a fixed observed feature, NOT
   a latent.
3. Centered parameterization (`alpha ~ Normal(mu, sigma)` directly,
   without the alpha_raw layer) inside the joint block. Even joint
   NUTS can struggle with the centered funnel under diagonal metric;
   NC is structurally cleaner.
4. Putting `alpha` (the RE values) in shared_data as a SAMPLED key.
   `alpha` is DERIVED from `(mu, sigma_alpha, alpha_raw)`. Register
   it via `register_refresher` if downstream blocks need it; do
   NOT ship a `nuts_block(alpha)`.
5. Multiple Gibbs sweeps over disjoint subsets of
   `(mu, sigma_alpha, alpha_raw)`. Even if each conditional is
   correct, the funnel reasserts across sweeps. Joint update
   ALWAYS.

---

This skill exists because the code-gen agent's natural choice
("one nuts_block per parameter, that's the default in
codegen_cpp.md") silently produces a wrong sampler for hierarchical
models. The correct answer is in the library -- `joint_nuts_block`
and the NC reparameterization are both shipped -- but the agent has
to be TOLD to use them as the default for this pattern. That's
what this skill does.

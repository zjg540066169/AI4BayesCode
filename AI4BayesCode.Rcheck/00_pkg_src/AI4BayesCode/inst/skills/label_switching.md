---
name: AI4BayesCode-label-switching
description: |
  Reference guide for handling label switching in sim1 cross-implementation
  comparison scripts. Covers detection, the simple-sort shortcut, the full
  Stephens 2000 pipeline, model-family-specific allocation probability
  formulas, Hungarian truth matching, and fallback strategies.
---

# Label Switching — sim1 reference

## 1. What it is and when it matters

A Bayesian model has **label switching** when the likelihood is invariant
under any permutation of K component labels (e.g. swapping "component 1"
and "component 2" does not change `p(y | theta)`). Without identifiability
constraints, MCMC chains can explore multiple modes corresponding to
different label permutations, making per-component posterior summaries and
cross-implementation R-hat meaningless.

**Affected model families**

| Family | Label-switching risk |
|--------|---------------------|
| Gaussian / Poisson / any finite mixture | Yes — unless ordered constraint enforced |
| Hidden Markov Model (HMM) | Yes — hidden states are exchangeable |
| LDA / topic model | Yes — topics are exchangeable |
| Dirichlet process mixture | Yes — exchangeable clusters |
| Changepoint model (Poisson/Gaussian with K segments ordered in time) | **No** — segments are identified by position |
| Regression / hierarchical with named groups | No — groups are fixed |

**Key diagnostic**: if `<id>.txt` says "K components", "K latent states",
or "K topics" and the **reference implementation** does not impose an
ordering constraint, assume label switching applies. Reference type
matters here:

- **Stan reference** (`<id>.stan`): check `parameters` block for
  `ordered[K] mu` or `mu[1] < mu[2]`-style constraint; if absent → label
  switching applies.
- **R-package reference** (`<id>.r` calls e.g. `bayesm::rmixGibbs`,
  `mixtools::normalmixEM`, NIMBLE mixtures): nearly all R-package
  mixture fitters are unconstrained — assume label switching applies
  unless the package's docs explicitly say otherwise.
- **R-package reference for non-mixture models** (e.g.,
  `BART::wbart`, `SoftBart::softbart`): no exchangeable components,
  no label switching, **skip this skill entirely**.

---

## 2. AI4BayesCode vs reference convention

For all label-switching-affected models in this benchmark:

- **AI4BayesCode** enforces an ascending ordering constraint (e.g. `mu[1] <
  mu[2] < ... < mu[K]`) to resolve label switching structurally. The Gibbs
  sampler samples within the ordered support.
- **Reference** (Stan or R-package) typically has no such constraint by
  default. Each reference draw can use any permutation.

This asymmetry means:

- AI draws: already in the canonical ("component 1 = smallest mean") label.
- Reference draws: must be relabeled before computing cross-impl R-hat and
  coverage.

If the reference also has an ordering constraint (Stan's `ordered[K] mu`,
or an R package that internally orders components after the Gibbs sweep),
label switching is already resolved and you can skip all relabeling.

The recipes in §3-§5 below show Stan-reference syntax (most existing
benchmark models use Stan); the **same logic applies to R-package
references** — only the draw-extraction line changes:

| Reference type | Draw extraction line in `align_REF_draws` |
|---|---|
| Stan (`cmdstanr`) | `d <- posterior::as_draws_matrix(stan_fit$draws())` |
| `bayesm::rmixGibbs` | extract `$compdraws[[k]]$mu`, etc., from the package's nested-list output and stack |
| `mixtools::normalmixEM` | `fit$mu`, `fit$sigma`, `fit$lambda` (point estimates only — EM, not MCMC; not suitable as a reference for label-switching cross-impl) |
| Other | check the package's `?<fitter>` Value section for the draw structure |

Once draws are extracted into the same matrix shape (n_keep × K),
the per-draw permutation (Stephens or simple-sort) is identical
regardless of reference type.

---

## 3. Simple-sort shortcut (K = 2 or any K with well-separated ordering)

When the convention is **ascending by key parameter** (e.g. `mu[1] < mu[2]`
for a Gaussian mixture), per-draw sorting of reference draws is both
correct and fast. Use this when K ≤ 3 and the model imposes an ascending
ordering on AI.

**Naming convention.** The function name should track the reference type:
`align_Stan_draws(stan_fit)` for Stan, `align_REF_draws(ref_run)` for
R-package references. The body's draw-extraction line is the only piece
that differs between reference types; the per-draw permutation logic
shown below is identical for both.

### K = 2 example (normal_mixture)

```r
align_Stan_draws <- function(stan_fit) {
    d         <- posterior::as_draws_matrix(stan_fit$draws())
    mu_mat    <- cbind(as.numeric(d[, "mu[1]"]), as.numeric(d[, "mu[2]"]))
    theta_vec <- as.numeric(d[, "theta"])
    for (i in seq_len(nrow(mu_mat))) {
        if (mu_mat[i, 1L] > mu_mat[i, 2L]) {
            tmp           <- mu_mat[i, 1L]
            mu_mat[i, 1L] <- mu_mat[i, 2L]
            mu_mat[i, 2L] <- tmp
            theta_vec[i]  <- 1 - theta_vec[i]   # flip mixing weight
        }
    }
    list(mu = mu_mat, theta = matrix(theta_vec, ncol = 1L))
}
```

The DGP truth must use the same convention:

```r
if (mu[1] > mu[2]) { mu <- rev(mu); theta <- 1 - theta }
```

### K = 3 example (normal_mixture_k)

```r
align_Stan_draws <- function(stan_fit, K) {
    d         <- posterior::as_draws_matrix(stan_fit$draws())
    mu_mat    <- unname(d[, paste0("mu[",    seq_len(K), "]"), drop = FALSE])
    sigma_mat <- unname(d[, paste0("sigma[", seq_len(K), "]"), drop = FALSE])
    theta_mat <- unname(d[, paste0("theta[", seq_len(K), "]"), drop = FALSE])
    for (i in seq_len(nrow(mu_mat))) {
        ord <- order(mu_mat[i, ])
        if (!identical(ord, seq_len(K))) {
            mu_mat[i, ]    <- mu_mat[i,    ord]
            sigma_mat[i, ] <- sigma_mat[i, ord]
            theta_mat[i, ] <- theta_mat[i, ord]
        }
    }
    list(mu = mu_mat, sigma = sigma_mat, theta = theta_mat)
}
```

DGP truth:

```r
ord <- order(mu)
mu <- mu[ord]; sigma <- sigma[ord]; theta <- theta[ord]
```

**When sort is sufficient**: components are well-separated so that no draw
has ambiguous membership. If two components have nearly identical means the
sort may still produce a misleading alignment; use Stephens in that case.

---

## 4. Stephens 2000 pipeline (robust, any K)

Use when K > 2 or when components may overlap heavily. Requires packages
`label.switching` and `clue`.

### Package dependencies

```r
suppressPackageStartupMessages({
    library(label.switching)
    library(clue)
})
```

### Step 1 — Compute per-draw allocation probabilities

`stephens()` needs an `[iter × N × K]` array of allocation probabilities.

#### Gaussian mixture

```r
compute_alloc_probs <- function(draws, y) {
    iter <- nrow(draws$mu); N <- length(y); K <- ncol(draws$mu)
    p <- array(NA_real_, dim = c(iter, N, K))
    for (d in seq_len(iter)) {
        lp <- sweep(
            mapply(function(k) dnorm(y, draws$mu[d, k], draws$sigma[d, k], log = TRUE),
                   seq_len(K)),
            1, log(draws$theta[d, ]), "+")
        # rows = observations, cols = components
        lp <- lp - apply(lp, 1, function(r) {
            mx <- max(r); mx + log(sum(exp(r - mx)))  # log-sum-exp
        })
        p[d, , ] <- exp(lp)
    }
    p
}
```

#### Poisson mixture

Same as above but replace `dnorm` with `dpois`.

#### LDA / topic model

For LDA the per-token topic-assignment posterior is
`p(z_n = k | doc_n, w_n, theta_d, phi_d) ∝ theta_d[doc_n, k] * phi_d[k, w_n]`.

```r
compute_alloc_probs_lda <- function(theta_arr, phi_arr, doc, w, K) {
    # theta_arr: [S, M, K], phi_arr: [S, K, V], doc/w: integer vectors length N
    S <- dim(theta_arr)[1L]; N <- length(doc)
    p <- array(NA_real_, dim = c(S, N, K))
    for (s in seq_len(S)) {
        thd <- theta_arr[s, doc, , drop = TRUE]   # N x K
        phd <- t(phi_arr[s, , w, drop = TRUE])     # N x K
        u   <- thd * phd
        rs  <- rowSums(u); rs[rs <= 0] <- 1
        p[s, , ] <- u / rs
    }
    p
}
```

#### HMM

Use the **forward–backward algorithm** to get smoothing probabilities
`gamma[t, k] = p(z_t = k | y_1..T, theta)`. The result is an
`[iter × T × K]` array where T plays the role of N.

```r
compute_alloc_probs_hmm <- function(draws, y) {
    iter <- nrow(draws$A); T <- length(y); K <- ncol(draws$lambda)
    p <- array(NA_real_, dim = c(iter, T, K))
    for (d in seq_len(iter)) {
        A_d      <- draws$A[d, , ]     # K x K transition matrix
        lambda_d <- draws$lambda[d, ]  # K emission rates (Poisson)
        pi_d     <- draws$pi[d, ]      # K initial distribution
        # Forward pass
        alpha <- matrix(NA_real_, T, K)
        alpha[1, ] <- pi_d * dpois(y[1], lambda_d)
        alpha[1, ] <- alpha[1, ] / sum(alpha[1, ])
        for (t in 2:T) {
            alpha[t, ] <- (alpha[t-1, ] %*% A_d) * dpois(y[t], lambda_d)
            alpha[t, ] <- alpha[t, ] / sum(alpha[t, ])
        }
        # Backward pass
        beta <- matrix(1, T, K)
        for (t in (T-1):1) {
            for (k in seq_len(K))
                beta[t, k] <- sum(A_d[k, ] * dpois(y[t+1], lambda_d) * beta[t+1, ])
            beta[t, ] <- beta[t, ] / sum(beta[t, ])
        }
        gamma <- alpha * beta
        p[d, , ] <- gamma / rowSums(gamma)
    }
    p
}
```

### Step 2 — Run Stephens on pooled chains

```r
# ai_draws, ref_draws: aligned output from align_AI_draws / align_Stan_draws
# replicate_data: dgp$data (used to compute allocation probs)

p_AI  <- compute_alloc_probs(ai_draws,  replicate_data$y)
p_REF <- compute_alloc_probs(ref_draws, replicate_data$y)
p_stk <- abind::abind(p_AI, p_REF, along = 1)
perm  <- label.switching::stephens(p_stk)$permutations  # (nAI+nREF) x K

n_AI     <- nrow(p_AI)
perm_AI  <- perm[1:n_AI, ]
perm_REF <- perm[(n_AI + 1):nrow(perm), ]

apply_perm <- function(mat, perm_mat) {
    out <- mat
    for (d in seq_len(nrow(out))) out[d, ] <- mat[d, perm_mat[d, ]]
    out
}

label_switching_params <- c("mu", "sigma", "theta")  # adjust per model
for (nm in label_switching_params) {
    ai_draws[[nm]]  <- apply_perm(ai_draws[[nm]],  perm_AI)
    ref_draws[[nm]] <- apply_perm(ref_draws[[nm]], perm_REF)
}
```

### Step 3 — Hungarian match to truth

After Stephens, both chains share an internal labeling that need not
match the DGP truth. Apply a Hungarian match so coverage is computed
against the right truth component.

```r
# key_param: the parameter used as the distance pivot (typically mu or lambda)
ai_mean     <- colMeans(ai_draws$mu)
truth_mu    <- as.numeric(truth$mu)
cost_mat    <- abs(outer(ai_mean, truth_mu, "-"))
match_perm  <- as.integer(clue::solve_LSAP(cost_mat))

for (nm in label_switching_params) {
    ai_draws[[nm]]  <- ai_draws[[nm]] [, match_perm, drop = FALSE]
    ref_draws[[nm]] <- ref_draws[[nm]][, match_perm, drop = FALSE]
}
```

---

## 5. Where to call relabeling in the worker

The relabeling step sits **between alignment and metrics**:

```r
worker <- function(r) {
    dgp <- simulate_data_with_truth(seed = r)
    ai_run  <- run_AI_chain(dgp$data, seed = r)
    ref_run <- run_Stan_chain(dgp$data, seed = r)
    # ... fail-fast checks ...
    aligned <- tryCatch(
        list(ai = align_AI_draws(ai_run$model_obj),
             ref = align_Stan_draws(ref_run$fit)),
        error = function(e) list(error = conditionMessage(e)))
    if ("error" %in% names(aligned)) { ... return(base_out) }

    # ---- LABEL SWITCHING ----
    aligned <- tryCatch({
        p_AI  <- compute_alloc_probs(aligned$ai,  dgp$data$y)
        p_REF <- compute_alloc_probs(aligned$ref, dgp$data$y)
        p_stk <- abind::abind(p_AI, p_REF, along = 1)
        perm  <- label.switching::stephens(p_stk)$permutations
        n_AI  <- nrow(p_AI)
        for (nm in label_switching_params) {
            aligned$ai[[nm]]  <- apply_perm(aligned$ai[[nm]],
                                            perm[1:n_AI, ])
            aligned$ref[[nm]] <- apply_perm(aligned$ref[[nm]],
                                            perm[(n_AI+1):nrow(perm), ])
        }
        # Hungarian to truth
        cost_mat   <- abs(outer(colMeans(aligned$ai$mu), as.numeric(dgp$truth$mu), "-"))
        match_perm <- as.integer(clue::solve_LSAP(cost_mat))
        for (nm in label_switching_params) {
            aligned$ai[[nm]]  <- aligned$ai[[nm]] [, match_perm, drop = FALSE]
            aligned$ref[[nm]] <- aligned$ref[[nm]][, match_perm, drop = FALSE]
        }
        aligned
    }, error = function(e) list(error = conditionMessage(e)))
    if ("error" %in% names(aligned)) {
        base_out$status <- "align_failed"; base_out$err_align <- aligned$error
        return(base_out) }
    # ---- END LABEL SWITCHING ----

    # ... compute metrics, return base_out ...
}
```

---

## 6. Decision tree

```
Does the model have K exchangeable components / states / topics?
│
├── No  → no relabeling needed; use standard sim1 template
│
└── Yes → Does AI4BayesCode enforce an ascending ordering constraint
          (mu[1] < mu[2] < ... < mu[K])?
          │
          ├── No  → Stephens pipeline (§4) for both chains
          │
          └── Yes → Are components well-separated in the key parameter?
                    │
                    ├── Yes, K ≤ 3  → simple sort shortcut (§3) for Stan;
                    │                  AI draws are already ordered; no
                    │                  Stephens needed. Hungarian match to
                    │                  truth is automatic because truth is
                    │                  sorted by the same key.
                    │
                    └── Possibly overlapping or K > 3
                                     → Stephens pipeline (§4) for Stan;
                                       AI draws can skip Stephens since
                                       they are already structured;
                                       run Hungarian match (§4 Step 3)
                                       using the key parameter
```

---

## 7. Package checklist

| Package | Purpose | Install |
|---------|---------|---------|
| `label.switching` | `stephens()` algorithm | `install.packages("label.switching")` |
| `clue` | `solve_LSAP()` Hungarian | `install.packages("clue")` |
| `abind` | `abind()` array concat | `install.packages("abind")` |

---

## 8. Worked model inventory

| Model | K | Label switching? | Strategy |
|-------|---|-----------------|----------|
| `normal_mixture` | 2 | Yes (Stan); No (AI: mu ordered) | Simple sort on Stan draws; truth sorted ascending |
| `normal_mixture_k` | 3 | Yes (Stan); No (AI: mu ordered) | Simple sort on Stan draws; truth sorted ascending |
| `poisson_changepoint` | 1 cp → 2 segments | No — segments identified by time | No relabeling |
| `poisson_k_changepoint` | K_cp+1 segments | No — ordered by changepoint position | No relabeling |
| HMM (if added) | K states | Yes | Stephens with forward–backward alloc probs |
| LDA / topic model | K topics | Yes | **Stephens with per-token topic-assignment posterior + Hungarian to truth.** Simple sort is NOT sufficient even at K=2 when the AI sampler is z-augmented (slow-mixing): the per-draw sort matches the wrong labels often enough to inflate cross-impl rhat. At K ≥ 3, simple sort completely fails (K! permutations); Stephens is mandatory. |

---

## 9. Pitfalls

1. **Applying permutation only to some parameters** — if `mu` is relabeled
   but `sigma` or `theta` are not, the alignment is inconsistent and R-hat
   will be inflated. Always relabel ALL K-dependent parameters together.

2. **Using different label conventions for AI and Stan** — Stephens resolves
   within-chain label switching jointly by pooling both chains. If you run
   Stephens on each chain independently, there is no guarantee the two
   chains end up with the same labeling, and cross-impl R-hat will still be
   inflated.

3. **Confusing component ordering with time ordering** — in a K-segment
   changepoint model, segment 1 is always the earliest in time, not the one
   with the smallest rate. Do NOT sort these by lambda — that would destroy
   the temporal alignment.

4. **sim1 convergence with label switching** — even after relabeling,
   if two components are very close in parameter space, ESS for those
   components will be low because the sampler mixes slowly. This is a
   posterior geometry issue, not a relabeling failure.

5. **Hungarian match sensitivity** — `clue::solve_LSAP` uses absolute
   difference in the key parameter as the cost. If two truth components have
   the same key value (degenerate case), the Hungarian assignment may be
   arbitrary. This rarely happens with DGP draws from a continuous prior,
   but is worth checking in debug runs.

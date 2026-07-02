# Tests for BartNoise / GBartPoisson set_current(list(X=..., y=..., sigma=..., offset=...))
#
# 1. set_current(list(y=...)) after step: BART sees new working response
# 2. set_current(list(X=...)) after step: BART sees new design matrix
# 3. set_current(list(X=..., y=...)) atomic update
# 4. State preservation: only the keys passed change; untouched parameters
#    are stable across the set_current call
# 5. Nested-MCMC sanity: simulate an outer residual update loop and verify
#    BART chain still converges to truth

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

cat("=========================================\n")
cat("  BART / genBART set_current(X, y)      \n")
cat("=========================================\n")

ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "BartNoise.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

# ============================================================================
# Test 1: set_current(list(y = r)) — working-response update
# ============================================================================
cat("\n[T1] BartNoise: set_current(list(y = new_y)) pushes working response\n")
set.seed(1); N <- 80; p <- 3
X <- matrix(rnorm(N*p), N, p)
f_true <- 3*X[,1] + X[,2]^2
y_raw <- as.numeric(f_true + rnorm(N, 0, 0.8))

m <- new(BartNoise, X, y_raw, 50L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE,
         101L, TRUE)
m$step(200L)
sigma_before <- m$get_current()$sigma
f_before     <- m$get_current()$f_bart

# Push a very different working response (pure noise around 0, no signal)
y_new <- rnorm(N, 0, 0.8)
m$set_current(list(y = y_new))

# 200 more steps on the new y; BART fit should drift toward new_y
m$step(200L)
f_after <- m$get_current()$f_bart
rmse_old_y <- sqrt(mean((f_after - y_raw)^2))
rmse_new_y <- sqrt(mean((f_after - y_new)^2))
cat(sprintf("  wall test: f_after RMSE vs y_raw = %.3f, vs y_new = %.3f\n",
    rmse_old_y, rmse_new_y))
stopifnot(rmse_new_y < rmse_old_y)
cat("  [PASS] after set_current(y=new_y) + steps, fit tracks new_y\n")

# ============================================================================
# Test 2: set_current(list(X = X_new)) — design-matrix swap
# ============================================================================
cat("\n[T2] BartNoise: set_current(list(X = X_new)) swaps design matrix\n")
set.seed(2)
m2 <- new(BartNoise, X, y_raw, 50L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE,
         102L, TRUE)
m2$step(200L)
f_before2 <- m2$get_current()$f_bart

# Produce a permuted X (same rows, same columns, just reshuffled)
perm <- sample.int(N)
X_perm <- X[perm, , drop=FALSE]
# Corresponding truth would be f_true[perm]; set working response to match
m2$set_current(list(X = X_perm, y = y_raw[perm]))
m2$step(300L)
f_after2 <- m2$get_current()$f_bart
# After sweeping on permuted data, f_after2[i] should track f_true[perm[i]]
cor_permuted <- cor(f_after2, f_true[perm])
cor_original <- cor(f_after2, f_true)
cat(sprintf("  cor(f_after2, f_true_perm) = %.3f, cor(f_after2, f_true_orig) = %.3f\n",
    cor_permuted, cor_original))
stopifnot(cor_permuted > 0.6)            # should now track permuted truth
stopifnot(cor_permuted > cor_original + 0.15)  # and be clearly better than un-permuted
cat("  [PASS] after set_current(X=X_perm), fit tracks permuted truth\n")

# ============================================================================
# Test 3: only-passed-keys change — sigma untouched if not in list
# ============================================================================
cat("\n[T3] BartNoise: sigma untouched when set_current(list(y=...)) only\n")
m3 <- new(BartNoise, X, y_raw, 50L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE,
         103L, TRUE)
m3$step(200L)
sigma_a <- m3$get_current()$sigma
m3$set_current(list(y = rnorm(N, 0, 1)))  # only y; sigma not passed
sigma_b <- m3$get_current()$sigma
stopifnot(identical(sigma_a, sigma_b))
cat(sprintf("  sigma unchanged: %.6f == %.6f\n", sigma_a, sigma_b))
cat("  [PASS] untouched keys preserved across set_current\n")

# ============================================================================
# Test 4: rejects f_bart
# ============================================================================
cat("\n[T4] BartNoise: set_current(list(f_bart=...)) is rejected\n")
m4 <- new(BartNoise, X, y_raw, 50L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE,
         104L, TRUE)
err <- tryCatch({
    m4$set_current(list(f_bart = rnorm(N)))
    "NO ERROR"
}, error = function(e) conditionMessage(e))
stopifnot(grepl("f_bart", err))
cat(sprintf("  err msg: %s\n", err))
cat("  [PASS] f_bart is correctly rejected\n")

# ============================================================================
# Test 5: nested-MCMC sanity — imputed-X loop (missing-data BART)
# ============================================================================
# Model: X[,1] has missing entries drawn from N(0, 1); outer sampler imputes
# them from their marginal each sweep; BART sees an evolving X while y is
# fixed. Verifies set_current(list(X = X_imp)) plays nice with repeated
# sweeps and the fit tracks a sensible average.
cat("\n[T5] BartNoise with imputed-X outer loop (missing-data pattern)\n")
set.seed(77); N5 <- 100; p5 <- 3
X_full <- matrix(rnorm(N5 * p5), N5, p5)
f_t    <- 3 * X_full[,1] + X_full[,2]^2
y_full <- as.numeric(f_t + rnorm(N5, 0, 0.8))
miss_idx <- sample(N5, 20)          # 20% of X[,1] missing
X_obs <- X_full; X_obs[miss_idx, 1] <- NA

# Initial imputation: draw from marginal N(0,1)
X_imp <- X_obs; X_imp[miss_idx, 1] <- rnorm(length(miss_idx), 0, 1)

m5 <- new(BartNoise, X_imp, y_full, 50L, 2.0, 2.0, 0.95, 3.0, 100L,
          FALSE, FALSE, 105L, TRUE)
m5$step(300L)       # BART burn-in on initial imputation

# 300 outer iterations: at each, re-draw missing X from their marginal,
# push new X via set_current, step BART once.
for (it in 1:300) {
    X_imp[miss_idx, 1] <- rnorm(length(miss_idx), 0, 1)
    m5$set_current(list(X = X_imp))
    m5$step(1L)
}

# After the loop, BART's fit on observed-X rows should still recover truth
# there (since those rows never changed).
f_post <- m5$get_current()$f_bart
rmse_obs  <- sqrt(mean((f_post[-miss_idx] - f_t[-miss_idx])^2))
sigma_end <- m5$get_current()$sigma
cat(sprintf("  RMSE on observed rows = %.3f  sigma_end = %.3f (truth 0.8)\n",
    rmse_obs, sigma_end))
stopifnot(rmse_obs < 2.0)           # should be small — BART tracks truth
stopifnot(sigma_end > 0.3 && sigma_end < 2.5)  # sensible range
# After 300 set_current(X) cycles nothing should have crashed or NaN'd.
stopifnot(all(is.finite(f_post)))
stopifnot(is.finite(sigma_end))
cat("  [PASS] 300 imputed-X cycles: fit finite, observed-row recovery OK\n")

# ============================================================================
# GBartPoisson tests
# ============================================================================
cat("\n\n=========================================\n")
cat("  GBartPoisson set_current(X, y)         \n")
cat("=========================================\n")

ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GBartPoisson.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

# ============================================================================
# Test 6: set_current(list(y=...)) updates response, r drifts toward new rate
# ============================================================================
cat("\n[T6] GBartPoisson: set_current(list(y=...)) drifts r toward new rate\n")
set.seed(3); N <- 80; p <- 3
X_l <- matrix(runif(N*p), N, p)
r_true <- 1 + 2*X_l[,1] - X_l[,2]
y_l <- rpois(N, exp(r_true))

m6 <- new(GBartPoisson, X_l, as.numeric(y_l), 50L, 106L, TRUE)
m6$step(100L)
r_before <- m6$get_current()$r

# Push new y under a SHIFTED rate.
r_new_rate <- 0.1 + 0.5*X_l[,1]
y_new      <- rpois(N, exp(r_new_rate))
m6$set_current(list(y = as.numeric(y_new)))
m6$step(500L)
r_after <- m6$get_current()$r
before_err <- mean((r_before - r_new_rate)^2)
after_err  <- mean((r_after  - r_new_rate)^2)
cat(sprintf("  pre-update MSE vs new rate = %.3f, post-update = %.3f\n",
    before_err, after_err))
stopifnot(after_err < before_err)
cat("  [PASS] set_current(y) shifts r toward new rate\n")

# ============================================================================
# Test 7: set_current(list(X=X_new, y=y_new)) — atomic design swap
# ============================================================================
cat("\n[T7] GBartPoisson: set_current(list(X = X_new, y = y_new)) works\n")
m7 <- new(GBartPoisson, X_l, as.numeric(y_l), 50L, 107L, TRUE)
m7$step(200L)
r0 <- m7$get_current()$r

# Permute rows and repush
perm_l <- sample.int(N)
X_l_perm <- X_l[perm_l, , drop=FALSE]
y_l_perm <- y_l[perm_l]
m7$set_current(list(X = X_l_perm, y = as.numeric(y_l_perm)))
m7$step(300L)
r_after7 <- m7$get_current()$r
# Should track permuted true log-rate
cor_p <- cor(r_after7, r_true[perm_l])
cor_o <- cor(r_after7, r_true)
cat(sprintf("  cor(r, r_true_perm) = %.3f,  cor vs un-perm = %.3f\n",
    cor_p, cor_o))
stopifnot(cor_p > 0.4)
stopifnot(cor_p > cor_o + 0.15)
cat("  [PASS] set_current(X, y) atomic update works\n")

# ============================================================================
# Test 8: rejects r (tree forest has no unique inverse)
# ============================================================================
cat("\n[T8] GBartPoisson: set_current(list(r=...)) rejected\n")
m8 <- new(GBartPoisson, X_l, as.numeric(y_l), 50L, 108L, TRUE)
err <- tryCatch({
    m8$set_current(list(r = rnorm(N)))
    "NO ERROR"
}, error = function(e) conditionMessage(e))
stopifnot(grepl("r", err))
cat(sprintf("  err msg: %s\n", err))
cat("  [PASS] r is correctly rejected\n")

# ============================================================================
# Summary
# ============================================================================
cat("\n\n=============================================\n")
cat("  ALL 8 TESTS PASSED\n")
cat("=============================================\n")

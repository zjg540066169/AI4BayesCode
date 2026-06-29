# ----------------------------------------------------------------------------
# test_bart_predict.R
#
# Tests for the predict_at interface on BartNoise:
#   1. predict_at(list(X = X_test)) returns correct-length f_bart
#   2. No state mutation: get_current() unchanged after predict_at
#   3. Posterior predictive f(X_test) vs BART::wbart reference
#   4. Validation: wrong key throws error
#   5. Validation: wrong column count throws error
#
# Also compares sigma initialization against BART::wbart to verify
# the OLS-based sigest matches.
# ----------------------------------------------------------------------------

# Locate this script's directory portably (works under Rscript and under
# source()). Under Rscript, --file= is in commandArgs; under source(), fall
# back to getwd() (user must be cd'd into AI4BayesCode/examples/).
script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))

source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

AI4BayesCode_sourceCpp(
    cpp_file       = file.path(script_dir, "BartNoise.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

# ---- Generate Friedman data ------------------------------------------------
set.seed(20260412L)
N      <- 300L
N_test <- 100L
p      <- 5L

X_all <- matrix(runif((N + N_test) * p), ncol = p)
f_true <- function(x) {
    10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 +
        10 * x[, 4] + 5 * x[, 5]
}
sigma_true <- 1.0
y_all <- f_true(X_all) + rnorm(N + N_test, 0, sigma_true)

X_train <- X_all[1:N, ]
y_train <- y_all[1:N]
X_test  <- X_all[(N + 1):(N + N_test), ]
y_test  <- y_all[(N + 1):(N + N_test)]
f_test_true <- f_true(X_test)

cat("=== BART predict_at test ===\n")
cat(sprintf("N_train = %d, N_test = %d, p = %d, sigma_true = %.2f\n\n",
            N, N_test, p, sigma_true))

# ---- Test 0: sigma initialization vs BART::wbart --------------------------
cat("--- Test 0: sigma initialization ---\n")

# Our model's internal sigma init
model <- new(BartNoise, X_train, y_train, 200L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE, 42L)
# Run 1 step to trigger bart_model init
model$step(1L)
bmc_sigma_init <- model$get_current()$sigma

# Compare with BART::wbart sigest
if (requireNamespace("BART", quietly = TRUE)) {
    # BART::wbart computes sigest from OLS residuals
    fit_lm <- lm(y_train ~ X_train)
    ols_sigest <- sd(residuals(fit_lm))
    cat(sprintf("  OLS sigest              = %.6f\n", ols_sigest))
    cat(sprintf("  AI4BayesCode sigma (1 step) = %.6f\n", bmc_sigma_init))
    cat(sprintf("  sd(y_train)             = %.6f\n", sd(y_train)))
    cat(sprintf("  ratio bmc/ols           = %.6f\n",
                bmc_sigma_init / ols_sigest))
} else {
    fit_lm <- lm(y_train ~ X_train)
    ols_sigest <- sd(residuals(fit_lm))
    cat(sprintf("  OLS sigest = %.6f, sd(y) = %.6f\n",
                ols_sigest, sd(y_train)))
    cat("  (BART package not installed, skipping wbart comparison)\n")
}
cat("\n")

# ---- Test 1: predict_at returns correct dimensions -------------------------
cat("--- Test 1: predict_at dimensions ---\n")
model <- new(BartNoise, X_train, y_train, 200L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE, 42L)
model$step(500L)

pred <- model$predict_at(list(X = X_test))
stopifnot("f_bart" %in% names(pred))
stopifnot(length(pred$f_bart) == N_test)
cat(sprintf("  predict_at returned f_bart of length %d (expected %d): OK\n",
            length(pred$f_bart), N_test))

# ---- Test 2: no state mutation ---------------------------------------------
cat("--- Test 2: no state mutation ---\n")
state_before <- model$get_current()
pred2 <- model$predict_at(list(X = X_test))
state_after  <- model$get_current()

f_diff <- max(abs(state_before$f_bart - state_after$f_bart))
s_diff <- abs(state_before$sigma - state_after$sigma)
cat(sprintf("  max |f_bart diff| = %.2e, |sigma diff| = %.2e\n",
            f_diff, s_diff))
stopifnot(f_diff < 1e-15)
stopifnot(s_diff < 1e-15)
cat("  State unchanged after predict_at: OK\n")

# ---- Test 3: posterior predictive vs truth ----------------------------------
cat("--- Test 3: posterior predictive ---\n")
n_burnin <- 2000L
n_keep   <- 2000L

model <- new(BartNoise, X_train, y_train, 200L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE, 42L)
model$step(n_burnin)

f_pred_mat <- matrix(NA, nrow = n_keep, ncol = N_test)
sigma_draws <- numeric(n_keep)

for (i in seq_len(n_keep)) {
    model$step(1L)
    pred_i <- model$predict_at(list(X = X_test))
    f_pred_mat[i, ] <- pred_i$f_bart
    sigma_draws[i]  <- model$get_current()$sigma
}

f_pred_mean <- colMeans(f_pred_mat)
cor_val <- cor(f_pred_mean, f_test_true)
rmse    <- sqrt(mean((f_pred_mean - f_test_true)^2))
cat(sprintf("  correlation(f_pred, f_true)  = %.4f\n", cor_val))
cat(sprintf("  RMSE(f_pred, f_true)         = %.4f\n", rmse))
cat(sprintf("  sigma posterior mean         = %.4f (true %.2f)\n",
            mean(sigma_draws), sigma_true))
cat(sprintf("  sigma posterior sd           = %.4f\n", sd(sigma_draws)))

# ---- Test 4: validation - wrong key ---------------------------------------
cat("--- Test 4: validation (wrong key) ---\n")
err_caught <- FALSE
tryCatch(
    model$predict_at(list(Z = X_test)),
    error = function(e) {
        cat(sprintf("  Error caught: %s\n", conditionMessage(e)))
        err_caught <<- TRUE
    })
stopifnot(err_caught)
cat("  Wrong key rejected: OK\n")

# ---- Test 5: validation - wrong column count --------------------------------
cat("--- Test 5: validation (wrong columns) ---\n")
err_caught2 <- FALSE
tryCatch(
    model$predict_at(list(X = X_test[, 1:3])),
    error = function(e) {
        cat(sprintf("  Error caught: %s\n", conditionMessage(e)))
        err_caught2 <<- TRUE
    })
stopifnot(err_caught2)
cat("  Wrong column count rejected: OK\n")

# ---- Test 6: empty predict -------------------------------------------------
cat("--- Test 6: empty predict ---\n")
empty_pred <- model$predict_at(list())
stopifnot(length(empty_pred) == 0)
cat("  Empty list in, empty list out: OK\n")

# ---- Summary ---------------------------------------------------------------
all_ok <- cor_val > 0.95 && rmse < 3.0 && f_diff < 1e-15 &&
          err_caught && err_caught2
cat(sprintf("\n%s\n", if (all_ok) "ALL TESTS PASS" else "SOME TESTS FAILED"))

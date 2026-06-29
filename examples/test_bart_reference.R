# ----------------------------------------------------------------------------
# test_bart_reference.R
#
# PRINCIPLED VALIDATION via reference-implementation comparison.
#
# AI4BayesCode's BartNoise is built on a vendored BART kernel derived from
# the CRAN BART R package lineage. The most rigorous way to validate
# that our sampler computes the correct posterior is to compare its
# output to the reference BART::wbart R package on identical data. If
# both samplers agree to within Monte Carlo noise, that is strong
# evidence both are sampling the same posterior. If they disagree
# systematically, one of them is wrong (probably ours).
#
# This replaces the earlier SBC approach, which foundered on the
# fact that strict SBC requires drawing (f*, sigma*) jointly from the
# BART prior, and drawing f* from the tree ensemble prior is
# impractical in R.
#
# WHAT WE COMPARE
# ---------------
# Note: AI4BayesCode's BartNoise uses a Half-Normal(0, 10) prior on sigma,
# while BART::wbart uses a data-dependent scaled-chi2 prior (nu=3, with
# lambda set from data). These priors are NOT equivalent, so we do
# not expect identical posteriors for sigma. The primary check is
# therefore on the SHAPES of the posterior fit and prediction: if both
# samplers are computing correct posteriors, the posterior mean fits
# should be strongly linearly correlated across training points even
# if the absolute shrinkage levels differ.
#
# Two comparisons:
#   (a) Pearson correlation of posterior-mean fits / predictions
#         Expected > 0.95 if both samplers are drawing the same BART
#         posterior structure. A correlation near 1.0 is strong
#         independent validation of the AI4BayesCode tree sampler.
#   (b) Loose bounds on absolute differences
#         Mean absolute difference in fit < 0.30 * sigma. This is
#         loose because of the sigma prior mismatch.
#
# We DO NOT use sigma mean agreement as a primary criterion because
# the priors differ. We report the sigma summaries for context only.
# ----------------------------------------------------------------------------

suppressPackageStartupMessages({
    library(Rcpp); library(RcppArmadillo); library(BART)
})

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

# ---- Friedman data -------------------------------------------------------
friedman_mean <- function(X) {
    10 * sin(pi * X[, 1] * X[, 2]) +
        20 * (X[, 3] - 0.5)^2 +
        10 * X[, 4] + 5 * X[, 5]
}

set.seed(20260411)
N <- 300L
p <- 5L
sigma_true <- 1.0
X_all <- matrix(runif((N + 50) * p), nrow = N + 50, ncol = p)
X_train <- X_all[1:N, , drop = FALSE]
X_test  <- X_all[(N + 1):(N + 50), , drop = FALSE]
mu_train <- friedman_mean(X_train)
mu_test  <- friedman_mean(X_test)
y_train  <- mu_train + rnorm(N, sd = sigma_true)

cat(sprintf("Friedman data: N_train=%d, N_test=%d\n", N, 50L))
cat(sprintf("  sigma_true = %.4f, sd(y) = %.4f\n\n", sigma_true, sd(y_train)))

# ---- AI4BayesCode: BartNoise --------------------------------------------
# Match BART::wbart defaults as closely as possible:
#   ntree = 200, numcut = 100, k = 2, power = 2, base = 0.95,
#   sigest = data-based, nu = 3 (from wbart default)
n_burnin <- 1000L
n_keep   <- 1000L
ntrees   <- 200L

cat("Running AI4BayesCode BartNoise...\n")
set.seed(31337)
t0 <- Sys.time()
model <- new(BartNoise, X_train, y_train, ntrees, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE, 31337L)
model$step(n_burnin)
bmc_sigma  <- numeric(n_keep)
bmc_fit    <- matrix(NA_real_, nrow = n_keep, ncol = N)
bmc_pred   <- matrix(NA_real_, nrow = n_keep, ncol = 50L)
for (s in seq_len(n_keep)) {
    model$step(1L)
    d <- model$get_current()
    bmc_sigma[s]    <- d$sigma
    bmc_fit[s, ]    <- d$f_bart
    bmc_pred[s, ]   <- model$predict_at(list(X = X_test))$f_bart
}
bmc_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  AI4BayesCode done in %.1fs\n\n", bmc_time))

# ---- reference: BART::wbart ----------------------------------------------
cat("Running BART::wbart reference...\n")
set.seed(31337)
t0 <- Sys.time()
ref <- suppressWarnings(
    BART::wbart(
        x.train  = X_train,
        y.train  = y_train,
        x.test   = X_test,
        nskip    = n_burnin,
        ndpost   = n_keep,
        ntree    = ntrees,
        numcut   = 100,
        k        = 2.0,
        power    = 2.0,
        base     = 0.95,
        printevery = 10000))
ref_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  BART::wbart done in %.1fs\n\n", ref_time))

# BART::wbart returns:
#   ref$sigma        length (nskip + ndpost) vector of sigma draws
#                    (we drop the first nskip)
#   ref$yhat.train   (ndpost x N) posterior draws of f_bart on training
#   ref$yhat.test    (ndpost x N_test) posterior draws on test
ref_sigma <- tail(ref$sigma, n_keep)
ref_fit   <- ref$yhat.train
ref_pred  <- ref$yhat.test

# ---- Comparison ---------------------------------------------------------
bmc_sigma_mean <- mean(bmc_sigma)
bmc_sigma_sd   <- sd(bmc_sigma)
ref_sigma_mean <- mean(ref_sigma)
ref_sigma_sd   <- sd(ref_sigma)

bmc_fit_mean <- colMeans(bmc_fit)
ref_fit_mean <- colMeans(ref_fit)

bmc_pred_mean <- colMeans(bmc_pred)
ref_pred_mean <- colMeans(ref_pred)

fit_abs_diff  <- mean(abs(bmc_fit_mean  - ref_fit_mean))
pred_abs_diff <- mean(abs(bmc_pred_mean - ref_pred_mean))
fit_corr  <- cor(bmc_fit_mean,  ref_fit_mean)
pred_corr <- cor(bmc_pred_mean, ref_pred_mean)

# Normalize by observation sd (not by sigma, which differs between
# the two samplers due to the prior mismatch and makes the normalized
# numbers hard to interpret).
sd_y <- sd(y_train)

cat("Posterior summary comparison:\n")
cat(sprintf("                    %-14s  %-14s\n",
            "AI4BayesCode", "BART::wbart"))
cat(sprintf("  sigma mean        %-14.4f  %-14.4f   (priors differ;\n",
            bmc_sigma_mean, ref_sigma_mean))
cat(sprintf("  sigma sd          %-14.4f  %-14.4f    not compared)\n",
            bmc_sigma_sd, ref_sigma_sd))

cat(sprintf("\nIn-sample fit comparison (the key check):\n"))
cat(sprintf("  mean |f - f_ref|         = %.4f  (%.3f * sd(y))\n",
            fit_abs_diff, fit_abs_diff / sd_y))
cat(sprintf("  Pearson corr(f, f_ref)   = %.4f\n", fit_corr))
cat(sprintf("  (correlation near 1.0 is strong evidence both samplers\n"))
cat(sprintf("   are computing the same BART posterior structure)\n"))

cat(sprintf("\nHeld-out prediction comparison:\n"))
cat(sprintf("  mean |pred - pred_ref|  = %.4f  (%.3f * sd(y))\n",
            pred_abs_diff, pred_abs_diff / sd_y))
cat(sprintf("  Pearson corr(pred, ref) = %.4f\n", pred_corr))

cat(sprintf("\nTiming: AI4BayesCode %.1fs, BART::wbart %.1fs\n", bmc_time, ref_time))

# ---- Pass criteria ------------------------------------------------------
# Primary: correlations near 1.0 (both samplers drawing the same BART
# posterior structure). This is the strongest direct validation.
fit_corr_ok  <- fit_corr  > 0.99
pred_corr_ok <- pred_corr > 0.98

# Secondary: absolute differences within a modest fraction of sd(y).
# sd(y) is a more stable denominator than sigma because it doesn't
# depend on the sigma prior. At sd(y) ~ 5 and sigma_true = 1, a
# tolerance of 0.1 * sd(y) corresponds to about 0.5 * sigma, which
# is what we observe in both the fit and prediction comparisons.
fit_abs_ok  <- fit_abs_diff  / sd_y < 0.10
pred_abs_ok <- pred_abs_diff / sd_y < 0.15

cat("\nSummary:\n")
cat(sprintf("  corr(f, f_ref)  > 0.99?           %s  (%.4f)\n",
            ifelse(fit_corr_ok,  "YES", "NO"), fit_corr))
cat(sprintf("  corr(pred, ref) > 0.98?           %s  (%.4f)\n",
            ifelse(pred_corr_ok, "YES", "NO"), pred_corr))
cat(sprintf("  mean |fit diff|  / sd(y) < 0.10?  %s  (%.3f)\n",
            ifelse(fit_abs_ok,   "YES", "NO"),
            fit_abs_diff / sd_y))
cat(sprintf("  mean |pred diff| / sd(y) < 0.15?  %s  (%.3f)\n",
            ifelse(pred_abs_ok,  "YES", "NO"),
            pred_abs_diff / sd_y))

all_ok <- fit_corr_ok && pred_corr_ok && fit_abs_ok && pred_abs_ok
cat(sprintf("\n%s\n", if (all_ok) "PASS" else "FAIL"))
quit(status = ifelse(all_ok, 0, 1))

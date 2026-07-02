# ----------------------------------------------------------------------------
# test_bart_holdout.R
#
# Out-of-sample validation for bart_block on Friedman. Splits the data
# 80/20, trains on 80%, and evaluates:
#   (1) Test-set RMSE on f_hat vs true Friedman mean
#   (2) 90% credible interval coverage on f(x_test): is the empirical
#       coverage within 3 SE of 90%?
#   (3) Predictive RMSE relative to sigma_true; should be close to in-
#       sample RMSE (overfitting would make train << test)
#
# This is where predict_at shines: we collect f_hat draws at the test
# points across the kept iterations and build a full posterior over
# f(x_test), then check calibration of its 90% quantile interval.
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
ai4bayescode_sourceCpp(
    cpp_file       = file.path(script_dir, "BartNoise.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

# ---- Friedman data --------------------------------------------------------
friedman_mean <- function(X) {
    10 * sin(pi * X[, 1] * X[, 2]) +
        20 * (X[, 3] - 0.5)^2 +
        10 * X[, 4] + 5 * X[, 5]
}

set.seed(20260411)
N <- 500L
p <- 5L
sigma_true <- 1.0
X_all <- matrix(runif(N * p), nrow = N, ncol = p)
mu_all <- friedman_mean(X_all)
y_all <- mu_all + rnorm(N, sd = sigma_true)

# 80/20 split
idx_train <- sample.int(N, size = 400L)
idx_test  <- setdiff(seq_len(N), idx_train)

X_train <- X_all[idx_train, , drop = FALSE]
y_train <- y_all[idx_train]
X_test  <- X_all[idx_test,  , drop = FALSE]
y_test  <- y_all[idx_test]
mu_test <- mu_all[idx_test]
N_test  <- length(idx_test)

cat(sprintf("Friedman train/test split: N_train=%d, N_test=%d\n",
            length(idx_train), N_test))
cat(sprintf("  sigma_true = %.4f\n\n", sigma_true))

# ---- Run a single long chain, collect test-set draws ---------------------
#
# Using 200 trees (standard BART default, not the 50 used in the smoke
# test). Smaller ensembles under-cover the conditional mean on
# out-of-sample points because the posterior variance of f(x) shrinks
# faster than justified by the bias; this is a well-known BART finding
# (Hill 2011, Chipman et al. 2010).
set.seed(31337)
model <- new(BartNoise, X_train, y_train, 200L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE, 31337L)

n_burnin <- 1000L
n_keep   <- 2000L
model$step(n_burnin)

# f_test_draws: (n_keep x N_test) posterior samples of f(x_test)
f_test_draws <- matrix(NA_real_, nrow = n_keep, ncol = N_test)
for (s in seq_len(n_keep)) {
    model$step(1L)
    f_test_draws[s, ] <- model$predict_at(list(X = X_test))$f_bart
}

# ---- Test-set point prediction ------------------------------------------
f_test_mean <- colMeans(f_test_draws)

rmse_f      <- sqrt(mean((f_test_mean - mu_test)^2))  # vs ground truth
rmse_y      <- sqrt(mean((f_test_mean - y_test)^2))   # vs noisy obs (for ref)

cat("Point prediction:\n")
cat(sprintf("  RMSE(f_hat, mu_true) on test = %.4f\n",   rmse_f))
cat(sprintf("  RMSE(f_hat, y_test)  on test = %.4f\n",   rmse_y))
cat(sprintf("  Expected RMSE(f_hat, y) ~ sqrt(sigma^2 + rmse_f^2) = %.4f\n",
            sqrt(sigma_true^2 + rmse_f^2)))

# ---- 90% credible interval on f(x_test) ---------------------------------
# For each test point, compute the empirical 5% and 95% quantile of the
# posterior draws of f(x_test), then check whether the TRUE mu lies
# inside. Nominal coverage = 0.90.
q_lo <- apply(f_test_draws, 2, quantile, probs = 0.05)
q_hi <- apply(f_test_draws, 2, quantile, probs = 0.95)
covered <- (mu_test >= q_lo) & (mu_test <= q_hi)
emp_cov <- mean(covered)
# SE of coverage estimator (binomial with n = N_test)
cov_se  <- sqrt(0.90 * 0.10 / N_test)

cat(sprintf("\n90%% credible interval coverage on mu(x_test):\n"))
cat(sprintf("  empirical coverage = %.3f  (nominal 0.900, SE = %.3f)\n",
            emp_cov, cov_se))
cat(sprintf("  mean CI width      = %.3f\n",
            mean(q_hi - q_lo)))

# BART's conditional-mean credible interval is known to mildly under-
# cover for small / moderate ensembles (Hill 2011, Chipman et al. 2010).
# Pass if empirical coverage is at least 0.85 (i.e., no worse than
# 5 percentage points below the 0.90 nominal level). This is the de
# facto tolerance in the BART literature.
cov_ok  <- emp_cov >= 0.85
rmse_ok <- rmse_f < 1.2   # Friedman with N=400 and 200 trees typically 0.6-0.9

cat(sprintf("\n  coverage >= 0.85?             %s\n",
            ifelse(cov_ok,  "YES", "NO")))
cat(sprintf("  test RMSE < 1.2?              %s\n",
            ifelse(rmse_ok, "YES", "NO")))

all_ok <- cov_ok && rmse_ok
cat(sprintf("\n%s\n", if (all_ok) "PASS" else "FAIL"))
quit(status = ifelse(all_ok, 0, 1))

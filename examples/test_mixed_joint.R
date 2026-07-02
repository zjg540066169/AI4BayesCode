# ----------------------------------------------------------------------------
# test_mixed_joint.R
#
# Smoke + diagnostic test for LinearRegJointMixed (demonstrates
# joint_nuts_block_mixed with REAL + POSITIVE sub-parameters).
#
# Checks:
#   - compiles and constructs
#   - step() runs
#   - posterior recovery of (alpha, beta, sigma) on simulated data
#   - 2-chain R-hat + bulk/tail ESS
#   - comparison against OLS estimates (should be close)
# ----------------------------------------------------------------------------

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

suppressPackageStartupMessages(library(posterior))

ai4bayescode_sourceCpp(file.path(script_dir, "LinearRegJointMixed.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

# --- Simulate --------------------------------------------------------------
set.seed(1L)
N <- 200L
p <- 5L
alpha_true <- 1.5
beta_true  <- c(2.0, -1.0, 0.5, 0.0, 3.0)
sigma_true <- 1.2
X <- matrix(rnorm(N * p), nrow = N, ncol = p)
y <- as.numeric(alpha_true + X %*% beta_true + rnorm(N, 0, sigma_true))

# --- OLS baseline ----------------------------------------------------------
Xd <- cbind(1, X)
ols <- solve(t(Xd) %*% Xd, t(Xd) %*% y)
ols_alpha <- ols[1]
ols_beta  <- ols[-1]
ols_sigma <- sqrt(sum((y - Xd %*% ols)^2) / (N - p - 1))

# --- Fit ---------------------------------------------------------------
run_chain <- function(seed, n_burnin, n_keep) {
    m <- new(LinearRegJointMixed, y, X, as.integer(seed), TRUE)
    t0 <- Sys.time()
    m$step(n_burnin)
    m$step(n_keep)
    t1 <- Sys.time()
    h <- m$get_history()
    list(alpha = h$alpha[(n_burnin + 1):(n_burnin + n_keep)],
         beta  = h$beta[(n_burnin + 1):(n_burnin + n_keep), , drop = FALSE],
         sigma = h$sigma[(n_burnin + 1):(n_burnin + n_keep)],
         wall_sec = as.numeric(difftime(t1, t0, units = "secs")))
}

cat("Running 2 chains (N=", N, ", p=", p, ")\n")
c1 <- run_chain(101L, 500L, 1000L)
cat(sprintf("  chain1 wall: %.1fs\n", c1$wall_sec))
c2 <- run_chain(202L, 500L, 1000L)
cat(sprintf("  chain2 wall: %.1fs\n", c2$wall_sec))

pack2 <- function(x1, x2) {
    if (is.null(dim(x1))) array(cbind(x1, x2), dim = c(length(x1), 2, 1))
    else {
        arr <- array(NA_real_, dim = c(nrow(x1), 2, ncol(x1)))
        arr[, 1, ] <- x1; arr[, 2, ] <- x2
        arr
    }
}

# --- Diagnostics -----------------------------------------------------------
alpha_arr <- pack2(c1$alpha, c2$alpha)
beta_arr  <- pack2(c1$beta,  c2$beta)
sigma_arr <- pack2(c1$sigma, c2$sigma)

cat("\n=== R-hat / ESS ===\n")
cat(sprintf("  alpha   Rhat=%.3f  ESS_bulk=%.0f  ESS_tail=%.0f\n",
            posterior::rhat(alpha_arr[,,1]),
            posterior::ess_bulk(alpha_arr[,,1]),
            posterior::ess_tail(alpha_arr[,,1])))
beta_rhats <- sapply(seq_len(p), function(k) posterior::rhat(beta_arr[,,k]))
beta_esss  <- sapply(seq_len(p), function(k) posterior::ess_bulk(beta_arr[,,k]))
cat(sprintf("  beta    max Rhat=%.3f  min ESS_bulk=%.0f\n",
            max(beta_rhats), min(beta_esss)))
cat(sprintf("  sigma   Rhat=%.3f  ESS_bulk=%.0f  ESS_tail=%.0f\n",
            posterior::rhat(sigma_arr[,,1]),
            posterior::ess_bulk(sigma_arr[,,1]),
            posterior::ess_tail(sigma_arr[,,1])))

# --- Posterior vs truth vs OLS --------------------------------------------
cat("\n=== Posterior vs truth vs OLS ===\n")
cat(sprintf("  alpha: post=%.3f  OLS=%.3f  truth=%.3f\n",
            mean(c1$alpha), ols_alpha, alpha_true))
cat(sprintf("  beta[1-5]:\n"))
cat(sprintf("    post  : %s\n", paste(sprintf("%.3f", colMeans(c1$beta)), collapse = " ")))
cat(sprintf("    OLS   : %s\n", paste(sprintf("%.3f", ols_beta),          collapse = " ")))
cat(sprintf("    truth : %s\n", paste(sprintf("%.3f", beta_true),         collapse = " ")))
cat(sprintf("  sigma: post=%.3f  OLS=%.3f  truth=%.3f\n",
            mean(c1$sigma), ols_sigma, sigma_true))

# Simple PASS checks
all_rhat_ok <- max(c(posterior::rhat(alpha_arr[,,1]),
                     beta_rhats,
                     posterior::rhat(sigma_arr[,,1]))) < 1.05
alpha_err <- abs(mean(c1$alpha) - alpha_true)
beta_err  <- max(abs(colMeans(c1$beta) - beta_true))
sigma_err <- abs(mean(c1$sigma) - sigma_true)

cat("\n=== Overall ===\n")
cat(sprintf("  R-hat < 1.05:      %s\n", ifelse(all_rhat_ok, "OK", "FAIL")))
cat(sprintf("  |alpha err|:       %.3f   %s\n",
            alpha_err, ifelse(alpha_err < 0.5, "OK", "WARN")))
cat(sprintf("  max |beta err|:    %.3f   %s\n",
            beta_err,  ifelse(beta_err  < 0.5, "OK", "WARN")))
cat(sprintf("  |sigma err|:       %.3f   %s\n",
            sigma_err, ifelse(sigma_err < 0.3, "OK", "WARN")))

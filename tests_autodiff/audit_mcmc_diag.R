# Phase C of audit: 2-chain MCMC diagnostics for each production example.
# For every example, run 2 chains on synthetic data and check:
#   - R-hat (rank-normalised) < 1.05 per parameter
#   - bulk ESS > 200, tail ESS > 200
#   - posterior recovery: |post_mean - truth| < 3 * post_sd per param
#   - LOO via loo::loo() when applicable (pointwise_loglik available)
# Appends results to AUDIT_LOG.md and prints summary table.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
audit_log    <- file.path(AI4BayesCode_dir, "AUDIT_LOG.md")
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

suppressPackageStartupMessages({
    library(posterior)
    library(loo)
})

append_log <- function(...) {
    cat(..., file = audit_log, sep = "", append = TRUE)
}

# Shared helpers
pack2 <- function(x1, x2) {
    if (is.null(dim(x1)))
        array(cbind(x1, x2), dim = c(length(x1), 2, 1))
    else {
        arr <- array(NA_real_, dim = c(nrow(x1), 2, ncol(x1)))
        arr[, 1, ] <- x1; arr[, 2, ] <- x2
        arr
    }
}
diag_arr <- function(arr) {
    p <- dim(arr)[3]
    list(
        rhat_max = max(sapply(seq_len(p), function(k) posterior::rhat(arr[,,k])),
                       na.rm = TRUE),
        ess_bulk_min = min(sapply(seq_len(p), function(k) posterior::ess_bulk(arr[,,k])),
                           na.rm = TRUE),
        ess_tail_min = min(sapply(seq_len(p), function(k) posterior::ess_tail(arr[,,k])),
                           na.rm = TRUE))
}

results <- list()

# ============================================================================
# C.1 GaussianLocationScale
# ============================================================================
cat("\n[C.1] GaussianLocationScale\n")
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GaussianLocationScale.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1)
y <- rnorm(100, mean = 2.0, sd = 1.5)
truth <- list(mu = 2.0, sigma = 1.5)

run_gauss <- function(seed, n_burn, n_keep) {
    m <- new(GaussianLocationScale, y, as.integer(seed), TRUE)
    t0 <- Sys.time()
    m$step(n_burn); m$step(n_keep)
    t1 <- Sys.time()
    h <- m$get_history()
    list(mu = h$mu[(n_burn+1):(n_burn+n_keep)],
         sigma = h$sigma[(n_burn+1):(n_burn+n_keep)],
         wall = as.numeric(difftime(t1, t0, units = "secs")))
}
c1 <- run_gauss(101L, 500L, 2000L)
c2 <- run_gauss(202L, 500L, 2000L)
d_mu    <- diag_arr(pack2(c1$mu,    c2$mu))
d_sigma <- diag_arr(pack2(c1$sigma, c2$sigma))
post_mean <- list(mu=mean(c(c1$mu, c2$mu)), sigma=mean(c(c1$sigma, c2$sigma)))
post_sd   <- list(mu=sd(c(c1$mu, c2$mu)), sigma=sd(c(c1$sigma, c2$sigma)))
rec_mu    <- abs(post_mean$mu - truth$mu)    < 3 * post_sd$mu
rec_sigma <- abs(post_mean$sigma - truth$sigma) < 3 * post_sd$sigma
# LOO
LL1 <- matrix(NA_real_, nrow = length(c1$mu), ncol = length(y))
LL2 <- matrix(NA_real_, nrow = length(c2$mu), ncol = length(y))
for (i in seq_along(c1$mu)) LL1[i, ] <- dnorm(y, c1$mu[i], c1$sigma[i], log = TRUE)
for (i in seq_along(c2$mu)) LL2[i, ] <- dnorm(y, c2$mu[i], c2$sigma[i], log = TRUE)
LLarr <- array(NA_real_, dim = c(nrow(LL1), 2, ncol(LL1)))
LLarr[,1,] <- LL1; LLarr[,2,] <- LL2
rel_eff <- loo::relative_eff(exp(LLarr))
lo <- loo::loo(LLarr, r_eff = rel_eff, cores = 1)
results$Gaussian <- list(
    wall = c1$wall + c2$wall,
    rhat_max = max(d_mu$rhat_max, d_sigma$rhat_max),
    ess_bulk_min = min(d_mu$ess_bulk_min, d_sigma$ess_bulk_min),
    recovery_mu = rec_mu, recovery_sigma = rec_sigma,
    post_mean = post_mean, truth = truth,
    loo_elpd = lo$estimates["elpd_loo", "Estimate"],
    pct_k_ok = 100 * mean(lo$diagnostics$pareto_k < 0.5))

cat(sprintf("  wall=%.1fs  Rhat=%.4f  ESS_bulk=%.0f  rec_mu=%s rec_sigma=%s  LOO k<0.5=%.0f%%\n",
            results$Gaussian$wall, results$Gaussian$rhat_max,
            results$Gaussian$ess_bulk_min,
            results$Gaussian$recovery_mu, results$Gaussian$recovery_sigma,
            results$Gaussian$pct_k_ok))

saveRDS(results, file.path(AI4BayesCode_dir, "audit_results.rds"))
cat("[C.1 DONE]\n")

# Phase F2 of audit — COVERAGE simulation (Cook-Gelman-Rubin style lite).
# For 4 fast examples (Gaussian, BetaBernoulli, ARDLasso, LinearRegJointMixed),
# simulate N_REPS datasets from the true generative model, fit MCMC per
# replicate, and check whether the 95% CI covers each scalar parameter.
# Empirical coverage should be ~95% if sampler correctness holds.
#
# Logs to audit_coverage.log; rds to audit_xl_F2.rds.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

N_REPS <- 80L   # 80 replicates per model
NB <- 2000L
NK <- 2000L
CI_WIDTH <- 0.95

ci_covers <- function(draws, truth, width = CI_WIDTH) {
    # returns TRUE if truth in CI, per-component vector for vector params
    q <- (1 - width) / 2
    if (is.null(dim(draws)))
        truth >= quantile(draws, q, names=FALSE) &
        truth <= quantile(draws, 1 - q, names=FALSE)
    else
        apply(draws, 2, function(col) {
            truth_i <- truth[which(colnames(draws) == names(truth))]
            # fallback: operate index-wise assuming truth aligned with cols
            NA
        })
}

# --- per-parameter CI coverage for scalar and vector params ---
coverage_1 <- function(draws, truth) {
    if (is.null(dim(draws))) {
        q <- quantile(draws, c(0.025, 0.975), names = FALSE)
        truth >= q[1] && truth <= q[2]
    } else {
        stopifnot(length(truth) == ncol(draws))
        sapply(seq_len(ncol(draws)), function(j) {
            q <- quantile(draws[,j], c(0.025, 0.975), names = FALSE)
            truth[j] >= q[1] && truth[j] <= q[2]
        })
    }
}

results_F2 <- list()

# ===== 1. Gaussian — mu, sigma =====
cat("\n[F2:Gaussian] coverage over", N_REPS, "reps\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GaussianLocationScale.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
truth <- list(mu = 2.0, sigma = 1.5)
cov_mu <- logical(N_REPS); cov_sigma <- logical(N_REPS)
t0 <- Sys.time()
for (r in seq_len(N_REPS)) {
    set.seed(1000 + r)
    y <- rnorm(100, truth$mu, truth$sigma)
    m <- new(GaussianLocationScale, y, as.integer(r), TRUE)
    m$step(NB); m$step(NK)
    h <- m$get_history()
    mu_d  <- h$mu[(NB+1):(NB+NK)]
    sig_d <- h$sigma[(NB+1):(NB+NK)]
    cov_mu[r]    <- coverage_1(mu_d,  truth$mu)
    cov_sigma[r] <- coverage_1(sig_d, truth$sigma)
}
t1 <- Sys.time()
results_F2$Gaussian <- list(
    cov_mu = mean(cov_mu), cov_sigma = mean(cov_sigma),
    n_reps = N_REPS,
    wall = as.numeric(difftime(t1, t0, units="secs"))
)
cat(sprintf("  mu coverage = %.1f%% (%.0f/%d)  sigma coverage = %.1f%% (%.0f/%d)  wall=%.1fs\n",
    100*mean(cov_mu), sum(cov_mu), N_REPS,
    100*mean(cov_sigma), sum(cov_sigma), N_REPS,
    as.numeric(difftime(t1, t0, units="secs"))))

# ===== 2. BetaBernoulli — p =====
cat("\n[F2:BetaBernoulli] coverage over", N_REPS, "reps\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "BetaBernoulli.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
truth_p <- 0.3
cov_p <- logical(N_REPS)
t0 <- Sys.time()
for (r in seq_len(N_REPS)) {
    set.seed(2000 + r)
    y <- rbinom(200, 1, truth_p)
    m <- new(BetaBernoulli, as.numeric(y), 1.0, 1.0, as.integer(r), TRUE)
    m$step(NB); m$step(NK)
    h <- m$get_history()
    p_d <- h$p[(NB+1):(NB+NK)]
    cov_p[r] <- coverage_1(p_d, truth_p)
}
t1 <- Sys.time()
results_F2$BetaBernoulli <- list(
    cov_p = mean(cov_p), n_reps = N_REPS,
    wall = as.numeric(difftime(t1, t0, units="secs"))
)
cat(sprintf("  p coverage = %.1f%% (%.0f/%d)  wall=%.1fs\n",
    100*mean(cov_p), sum(cov_p), N_REPS,
    as.numeric(difftime(t1, t0, units="secs"))))

# ===== 3. ARDLasso — alpha, beta (vec), sigma2 =====
cat("\n[F2:ARDLasso] coverage over", N_REPS, "reps\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "ARDLasso.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
truth_ard <- list(alpha = 2.0, beta = c(3, -2, 0, 0, 1.5, 0, 0, 0, 0, 0), sigma = 1.0)
cov_alpha <- logical(N_REPS)
cov_beta  <- matrix(NA, N_REPS, length(truth_ard$beta))
cov_sigma2 <- logical(N_REPS)
t0 <- Sys.time()
for (r in seq_len(N_REPS)) {
    set.seed(3000 + r)
    X <- matrix(rnorm(100*10), 100, 10)
    y <- as.numeric(truth_ard$alpha + X %*% truth_ard$beta + rnorm(100, 0, truth_ard$sigma))
    m <- new(ARDLasso, X, y, as.integer(r), TRUE)
    m$step(NB); m$step(NK)
    h <- m$get_history()
    alpha_d <- h$alpha[(NB+1):(NB+NK)]
    beta_d  <- h$beta[(NB+1):(NB+NK), , drop=FALSE]
    sig2_d  <- h$sigma2[(NB+1):(NB+NK)]
    cov_alpha[r]  <- coverage_1(alpha_d, truth_ard$alpha)
    cov_beta[r,]  <- coverage_1(beta_d,  truth_ard$beta)
    cov_sigma2[r] <- coverage_1(sig2_d,  truth_ard$sigma^2)
}
t1 <- Sys.time()
results_F2$ARDLasso <- list(
    cov_alpha = mean(cov_alpha),
    cov_beta_per_j = colMeans(cov_beta),
    cov_beta_avg = mean(cov_beta),
    cov_sigma2 = mean(cov_sigma2),
    n_reps = N_REPS,
    wall = as.numeric(difftime(t1, t0, units="secs"))
)
cat(sprintf("  alpha cov=%.1f%%  beta avg cov=%.1f%% (per-j: %s)  sigma2 cov=%.1f%%  wall=%.1fs\n",
    100*mean(cov_alpha), 100*mean(cov_beta),
    paste(sprintf("%.0f", 100*colMeans(cov_beta)), collapse="/"),
    100*mean(cov_sigma2),
    as.numeric(difftime(t1, t0, units="secs"))))

# ===== 4. LinearRegJointMixed — alpha, beta (vec), sigma =====
cat("\n[F2:LinearRegJointMixed] coverage over", N_REPS, "reps\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "LinearRegJointMixed.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
truth_lr <- list(alpha = 1.5, beta = c(2, -1, 0.5, 0, 3), sigma = 1.2)
cov_alpha <- logical(N_REPS)
cov_beta  <- matrix(NA, N_REPS, length(truth_lr$beta))
cov_sigma <- logical(N_REPS)
t0 <- Sys.time()
for (r in seq_len(N_REPS)) {
    set.seed(4000 + r)
    X <- matrix(rnorm(200*5), 200, 5)
    y <- as.numeric(truth_lr$alpha + X %*% truth_lr$beta + rnorm(200, 0, truth_lr$sigma))
    m <- new(LinearRegJointMixed, y, X, as.integer(r), TRUE)
    m$step(NB); m$step(NK)
    h <- m$get_history()
    a_d <- h$alpha[(NB+1):(NB+NK)]
    b_d <- h$beta[(NB+1):(NB+NK), , drop=FALSE]
    s_d <- h$sigma[(NB+1):(NB+NK)]
    cov_alpha[r] <- coverage_1(a_d, truth_lr$alpha)
    cov_beta[r,] <- coverage_1(b_d, truth_lr$beta)
    cov_sigma[r] <- coverage_1(s_d, truth_lr$sigma)
}
t1 <- Sys.time()
results_F2$LinearRegJointMixed <- list(
    cov_alpha = mean(cov_alpha),
    cov_beta_per_j = colMeans(cov_beta),
    cov_beta_avg = mean(cov_beta),
    cov_sigma = mean(cov_sigma),
    n_reps = N_REPS,
    wall = as.numeric(difftime(t1, t0, units="secs"))
)
cat(sprintf("  alpha cov=%.1f%%  beta avg cov=%.1f%% (per-j: %s)  sigma cov=%.1f%%  wall=%.1fs\n",
    100*mean(cov_alpha), 100*mean(cov_beta),
    paste(sprintf("%.0f", 100*colMeans(cov_beta)), collapse="/"),
    100*mean(cov_sigma),
    as.numeric(difftime(t1, t0, units="secs"))))

# ===== Save & summary =====
saveRDS(results_F2, file.path(AI4BayesCode_dir, "audit_xl_F2.rds"))
cat("\n========== PHASE F2 SUMMARY ==========\n")
cat("Nominal coverage: 95%.  Empirical coverage over", N_REPS, "replicates:\n")
for (nm in names(results_F2)) {
    r <- results_F2[[nm]]
    cat(sprintf("  %-25s  %s\n", nm,
        paste(names(r)[startsWith(names(r),"cov_")],
              sprintf("%.1f%%", 100 * unlist(r[startsWith(names(r),"cov_")])),
              sep="=", collapse="  ")))
}
cat("\n[F2 DONE] saved to audit_xl_F2.rds\n")

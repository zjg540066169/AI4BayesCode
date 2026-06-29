# RJMCMC v0 (Dirac spike-and-slab) audit — 5 simulation scenarios,
# 2-chain diagnostics each. Strong MCMC-correctness battery.
#
# Scenarios:
#   S1. Easy:      N=100, p=10,  3 nonzero, uncorrelated X, SNR~3
#   S2. Medium:    N=200, p=30,  5 nonzero, uncorrelated X, SNR~2
#   S3. Correlated: N=200, p=30,  5 nonzero, AR-1 X rho=0.5, SNR~2
#   S4. Null:      N=200, p=20,  0 nonzero (truly no signal)
#   S5. High-dim:  N=100, p=50,  3 nonzero (p/2), uncorrelated X, SNR~3
#
# For each scenario: 2 chains × 2000 burnin + 3000 keep. Per-chain and
# across-chain diagnostics:
#   - Inclusion probability per j (posterior P(γ_j=1 | y))
#   - Conditional β_j posterior mean and SD (when γ_j=1)
#   - σ² posterior (R-hat across chains, vs true σ²)
#   - π posterior
#   - Top-k selected features (union vs intersection across chains)
#   - Posterior predictive y_rep mean RMSE vs y (should ~σ^2)
#   - Chain mixing: R-hat on gamma_j (meaningful only for identifiable j),
#                   R-hat on beta_j (conditional on γ_j=1),
#                   R-hat on σ² and π
#
# Script writes to audit_rjmcmc.log; rds to audit_rjmcmc_results.rds.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))
suppressPackageStartupMessages({ library(posterior) })

AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "SpikeSlabRJMCMC.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

# =========================================================================
# Helpers
# =========================================================================

run_chain <- function(X, y, a_pi, b_pi,
                      seed, n_burn, n_keep) {
    m <- new(SpikeSlabRJMCMC, X, y,
             a_pi, b_pi,
             as.integer(seed), TRUE)
    t0 <- Sys.time()
    m$step(as.integer(n_burn))
    m$step(as.integer(n_keep))
    t1 <- Sys.time()
    h <- m$get_history()
    pp <- m$predict_at(list())
    # History keys are now "sigma" and "tau" (natural scale, not sigma2/tau2).
    # We compute sigma2 = sigma^2 here so downstream diagnostics stay the same.
    list(
        gamma  = h$gamma[(n_burn+1):(n_burn+n_keep), , drop=FALSE],
        beta   = h$beta [(n_burn+1):(n_burn+n_keep), , drop=FALSE],
        sigma2 = h$sigma[(n_burn+1):(n_burn+n_keep)]^2,
        tau2   = h$tau  [(n_burn+1):(n_burn+n_keep)]^2,
        pi     = h$pi   [(n_burn+1):(n_burn+n_keep)],
        y_rep  = pp$y_rep,
        wall   = as.numeric(difftime(t1, t0, units="secs"))
    )
}

# Rank-normalized R-hat over 2 chains for a vector parameter.
# For binary gamma components this is still meaningful: if true γ=0 is
# well-identified, both chains will stay at 0 and R-hat≈1 (with potential
# numerical-zero artifact, same as DirichletSparse — see Phase C2).
rhat_2chain_vec <- function(c1_vec, c2_vec) {
    arr <- array(cbind(c1_vec, c2_vec), dim=c(length(c1_vec), 2, 1))
    tryCatch(posterior::rhat(arr[,,1]), error = function(e) NA_real_)
}

rhat_2chain_mat <- function(c1_mat, c2_mat) {
    # Per-column R-hat.
    p <- ncol(c1_mat)
    rh <- numeric(p)
    for (j in 1:p) rh[j] <- rhat_2chain_vec(c1_mat[,j], c2_mat[,j])
    rh
}

summary_run <- function(label, X, y, beta_true, sigma_true,
                        a_pi, b_pi,
                        n_burn, n_keep) {
    cat(sprintf("\n===== %s =====\n", label))
    cat(sprintf("N=%d  p=%d  true_nonzero=%d  sigma_true=%.2f\n",
        nrow(X), ncol(X), sum(beta_true != 0), sigma_true))

    # Two chains with distinct seeds.
    c1 <- run_chain(X, y, a_pi, b_pi,
                    101L, n_burn, n_keep)
    c2 <- run_chain(X, y, a_pi, b_pi,
                    202L, n_burn, n_keep)
    total_wall <- c1$wall + c2$wall
    p <- ncol(X); N <- nrow(X)

    # Inclusion probabilities (pool across 2 chains).
    incl_prob_c1 <- colMeans(c1$gamma)
    incl_prob_c2 <- colMeans(c2$gamma)
    incl_prob <- (incl_prob_c1 + incl_prob_c2) / 2

    # Conditional β̂ (pool active draws from both chains).
    gamma_all <- rbind(c1$gamma, c2$gamma)
    beta_all  <- rbind(c1$beta,  c2$beta)
    n_total <- nrow(gamma_all)
    beta_cond_mean <- numeric(p)
    beta_cond_sd   <- numeric(p)
    for (j in 1:p) {
        active <- which(gamma_all[, j] == 1)
        if (length(active) >= 10) {
            beta_cond_mean[j] <- mean(beta_all[active, j])
            beta_cond_sd[j]   <- sd(beta_all[active, j])
        } else {
            beta_cond_mean[j] <- NA_real_
            beta_cond_sd[j]   <- NA_real_
        }
    }

    # R-hat across chains (per-column for gamma / beta; scalar for σ² / π).
    rhat_gamma <- rhat_2chain_mat(c1$gamma, c2$gamma)
    rhat_beta  <- rhat_2chain_mat(c1$beta,  c2$beta)
    rhat_sigma2 <- rhat_2chain_vec(c1$sigma2, c2$sigma2)
    rhat_pi     <- rhat_2chain_vec(c1$pi,    c2$pi)

    # σ² and π posterior summaries.
    sigma2_pm <- mean(c(c1$sigma2, c2$sigma2))
    sigma2_sd <- sd  (c(c1$sigma2, c2$sigma2))
    pi_pm     <- mean(c(c1$pi, c2$pi))
    pi_sd     <- sd  (c(c1$pi, c2$pi))

    # Posterior predictive y_rep RMSE vs observed y (should ≈ sigma_true).
    # pp$y_rep is a single realization per chain — we compute pooled RMSE.
    yrep_all <- c(c1$y_rep, c2$y_rep)
    y_rep_pooled_sd <- sd(yrep_all - rep(y, 2))
    # Per-chain sigma recovery.
    rmse_pred_c1 <- sqrt(mean((c1$y_rep - y)^2))
    rmse_pred_c2 <- sqrt(mean((c2$y_rep - y)^2))

    # Top-k selection stability across chains.
    true_k <- sum(beta_true != 0)
    top_k <- max(true_k, 3)  # at least 3
    top_c1 <- order(incl_prob_c1, decreasing=TRUE)[1:top_k]
    top_c2 <- order(incl_prob_c2, decreasing=TRUE)[1:top_k]
    true_idx <- which(beta_true != 0)
    top_overlap <- length(intersect(top_c1, top_c2))
    true_hit_c1 <- length(intersect(top_c1, true_idx))
    true_hit_c2 <- length(intersect(top_c2, true_idx))

    # ---- Report -----
    cat(sprintf("wall = %.1fs  (2 chains × %d+%d)\n",
        total_wall, n_burn, n_keep))

    cat("\nInclusion probability per j:\n")
    for (j in 1:p) {
        mark <- if (beta_true[j] != 0) " *TRUE*" else ""
        cat(sprintf("  j=%2d  P(γ=1)=%.3f  β̂_cond=%s  β_true=%.2f%s\n",
            j, incl_prob[j],
            if (is.na(beta_cond_mean[j])) "NA"
                else sprintf("%.3f", beta_cond_mean[j]),
            beta_true[j], mark))
    }

    cat(sprintf("\nσ²: pm=%.3f (true=%.3f)  sd=%.3f  R-hat=%.4f\n",
        sigma2_pm, sigma_true^2, sigma2_sd, rhat_sigma2))
    cat(sprintf("π:  pm=%.3f              sd=%.3f  R-hat=%.4f\n",
        pi_pm, pi_sd, rhat_pi))
    cat(sprintf("y_rep RMSE: chain1=%.3f chain2=%.3f (should ≈ %.2f)\n",
        rmse_pred_c1, rmse_pred_c2, sigma_true))

    # γ R-hat: summarize max (full), and max excluding j where both chains
    # agree identically 0 (near-numerical-zero).
    gamma_any_active <- which(
        colMeans(c1$gamma) > 0.01 | colMeans(c2$gamma) > 0.01)
    rhat_gamma_active_max <- if (length(gamma_any_active) > 0)
        max(rhat_gamma[gamma_any_active], na.rm=TRUE) else NA
    rhat_beta_active_max <- if (length(gamma_any_active) > 0)
        max(rhat_beta[gamma_any_active], na.rm=TRUE) else NA
    cat(sprintf("R-hat max — γ (all): %.3f  γ (active subset): %.3f\n",
        max(rhat_gamma, na.rm=TRUE),
        ifelse(is.na(rhat_gamma_active_max), NA, rhat_gamma_active_max)))
    cat(sprintf("R-hat max — β (active subset): %.3f\n",
        ifelse(is.na(rhat_beta_active_max), NA, rhat_beta_active_max)))

    # Top-k stability.
    cat(sprintf("top-%d stability: chain1 hit %d/%d true, chain2 hit %d/%d, cross-chain overlap %d/%d\n",
        top_k, true_hit_c1, true_k, true_hit_c2, true_k, top_overlap, top_k))

    # Verdict: inclusion-probability-based.
    # Pass criteria:
    #   - every true-nonzero j has P(γ=1) > 0.90
    #   - every true-zero j has P(γ=1) < 0.10
    #   - σ² within ±3 sd of truth
    tpos <- which(beta_true != 0)
    tneg <- which(beta_true == 0)
    hits_tpos <- if (length(tpos) > 0) incl_prob[tpos] > 0.90 else logical(0)
    hits_tneg <- if (length(tneg) > 0) incl_prob[tneg] < 0.10 else logical(0)
    all_tpos_ok <- all(hits_tpos)
    all_tneg_ok <- all(hits_tneg)
    sigma_ok    <- abs(sigma2_pm - sigma_true^2) < 3 * sigma2_sd

    verdict <- if (all_tpos_ok && all_tneg_ok && sigma_ok) "CLEAN PASS" else "ISSUES"
    cat(sprintf("VERDICT: %s  (tpos: %s, tneg: %s, sigma: %s)\n",
        verdict,
        all_tpos_ok, all_tneg_ok, sigma_ok))

    list(
        label = label,
        incl_prob = incl_prob,
        beta_cond_mean = beta_cond_mean,
        beta_cond_sd = beta_cond_sd,
        beta_true = beta_true,
        rhat_gamma_max = max(rhat_gamma, na.rm=TRUE),
        rhat_gamma_active_max = rhat_gamma_active_max,
        rhat_beta_active_max = rhat_beta_active_max,
        rhat_sigma2 = rhat_sigma2,
        rhat_pi = rhat_pi,
        sigma2_pm = sigma2_pm,
        sigma2_sd = sigma2_sd,
        sigma_true = sigma_true,
        pi_pm = pi_pm,
        pi_sd = pi_sd,
        top_k = top_k,
        true_hit_c1 = true_hit_c1,
        true_hit_c2 = true_hit_c2,
        top_overlap = top_overlap,
        true_k = true_k,
        rmse_pred_c1 = rmse_pred_c1,
        rmse_pred_c2 = rmse_pred_c2,
        wall = total_wall,
        all_tpos_ok = all_tpos_ok,
        all_tneg_ok = all_tneg_ok,
        sigma_ok = sigma_ok,
        verdict = verdict
    )
}

# =========================================================================
# Data generation
# =========================================================================

gen_X_iid <- function(N, p, seed) {
    set.seed(seed)
    X <- matrix(rnorm(N*p), N, p)
    X - matrix(colMeans(X), N, p, byrow=TRUE)
}

gen_X_ar1 <- function(N, p, rho, seed) {
    set.seed(seed)
    # AR-1 covariance: Σ_ij = ρ^|i-j|.
    Sigma <- rho^abs(outer(1:p, 1:p, "-"))
    L <- chol(Sigma)
    Z <- matrix(rnorm(N*p), N, p)
    X <- Z %*% L
    X - matrix(colMeans(X), N, p, byrow=TRUE)
}

gen_y <- function(X, beta_true, sigma_true, seed) {
    set.seed(seed + 12345)
    N <- nrow(X)
    y_raw <- as.numeric(X %*% beta_true + rnorm(N, 0, sigma_true))
    y_raw - mean(y_raw)
}

# =========================================================================
# Scenarios
# =========================================================================

results <- list()

# --- S1. Easy: small p, high SNR, uncorrelated ---
cat("\n########## S1: EASY ##########\n")
X1 <- gen_X_iid(100L, 10L, seed=1)
beta_true_1 <- c(3, -2, 0, 0, 1.5, 0, 0, 0, 0, 0)
y1 <- gen_y(X1, beta_true_1, sigma_true=1.0, seed=1)
results$S1 <- summary_run("S1: Easy (N=100, p=10, 3 nonzero, SNR≈3)",
    X1, y1, beta_true_1, sigma_true=1.0,
    a_pi=1, b_pi=1,
    n_burn=2000, n_keep=3000)

# --- S2. Medium: larger p, moderate SNR, uncorrelated ---
cat("\n########## S2: MEDIUM ##########\n")
X2 <- gen_X_iid(200L, 30L, seed=2)
beta_true_2 <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 25))
y2 <- gen_y(X2, beta_true_2, sigma_true=1.5, seed=2)
results$S2 <- summary_run("S2: Medium (N=200, p=30, 5 nonzero, σ=1.5)",
    X2, y2, beta_true_2, sigma_true=1.5,
    a_pi=1, b_pi=1,
    n_burn=2000, n_keep=3000)

# --- S3. Correlated X ---
cat("\n########## S3: CORRELATED X (AR-1 ρ=0.5) ##########\n")
X3 <- gen_X_ar1(200L, 30L, rho=0.5, seed=3)
beta_true_3 <- beta_true_2   # same signal, different X
y3 <- gen_y(X3, beta_true_3, sigma_true=1.5, seed=3)
results$S3 <- summary_run("S3: Correlated X (N=200, p=30, ρ=0.5)",
    X3, y3, beta_true_3, sigma_true=1.5,
    a_pi=1, b_pi=1,
    n_burn=2000, n_keep=3000)

# --- S4. Null model ---
cat("\n########## S4: NULL (no signal) ##########\n")
X4 <- gen_X_iid(200L, 20L, seed=4)
beta_true_4 <- rep(0, 20)
y4 <- gen_y(X4, beta_true_4, sigma_true=1.0, seed=4)
results$S4 <- summary_run("S4: Null (N=200, p=20, all β=0)",
    X4, y4, beta_true_4, sigma_true=1.0,
    a_pi=1, b_pi=1,
    n_burn=2000, n_keep=3000)

# --- S5. High-dim (p = N/2) ---
cat("\n########## S5: HIGH-DIM (p=N/2) ##########\n")
X5 <- gen_X_iid(100L, 50L, seed=5)
beta_true_5 <- c(3, -2, 1.5, rep(0, 47))
y5 <- gen_y(X5, beta_true_5, sigma_true=1.0, seed=5)
results$S5 <- summary_run("S5: High-dim (N=100, p=50, 3 nonzero)",
    X5, y5, beta_true_5, sigma_true=1.0,
    a_pi=1, b_pi=1,
    n_burn=2000, n_keep=3000)

# =========================================================================
# Final summary
# =========================================================================

cat("\n\n========== FINAL SUMMARY (5 scenarios) ==========\n")
for (nm in names(results)) {
    r <- results[[nm]]
    cat(sprintf(" %s  %-50s  wall=%.1fs  sigma2=%.2f(true %.2f)  pi=%.3f\n",
        r$verdict, r$label, r$wall, r$sigma2_pm, r$sigma_true^2, r$pi_pm))
}
saveRDS(results, file.path(AI4BayesCode_dir, "audit_rjmcmc_results.rds"))
cat("\nSaved to audit_rjmcmc_results.rds\n")

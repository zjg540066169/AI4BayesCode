# Extended RJMCMC audit — more scenarios, longer chains, 4-chain R-hat,
# coverage simulation. Uses the unified SpikeSlabRJMCMC (full hierarchical:
# pi, sigma^2, tau^2 all sampled).
#
# Structure:
#   Part 1: 2-chain diagnostics on diverse scenarios (2ch × 2k+3k or 3k+4k)
#     H1: medium-sparsity linear regression
#     H2: widely-varying beta magnitudes (beta = c(10, -5, 1, -0.5, 0.2, 0...))
#         — hierarchical tau^2 should adapt to the wide effect-size range
#     H3: grouped sparsity (adjacent active blocks)
#     H4: high-dim (N=100, p=100, p=N)
#   Part 2: long-chain diagnostic
#     L1: correlated X at 10k+10k — verify convergence isn't artifact
#   Part 3: 4-chain R-hat stability
#     M1: medium scenario with 4 chains, check R-hat < 1.02
#   Part 4: coverage simulation
#     C1: 30 synthetic replications, verify per-coefficient 95% CI
#         empirical coverage on the ACTIVE set (conditional on gamma_j=1)
#
# Writes results to audit_rjmcmc_xl_results.rds.

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

ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "SpikeSlabRJMCMC.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

# =========================================================================
# Helpers
# =========================================================================

rhat_vec <- function(vec_list) {
    # vec_list: list of n_chains numeric vectors (same length)
    nc <- length(vec_list)
    ni <- length(vec_list[[1]])
    arr <- array(unlist(vec_list), dim = c(ni, nc, 1))
    tryCatch(posterior::rhat(arr[,,1]), error = function(e) NA_real_)
}

rhat_mat <- function(mat_list) {
    # mat_list: list of n_chains (n_iter x p) matrices
    p <- ncol(mat_list[[1]])
    rh <- numeric(p)
    for (j in 1:p) {
        rh[j] <- rhat_vec(lapply(mat_list, function(m) m[, j]))
    }
    rh
}

run_spikeslab <- function(X, y, a_pi, b_pi,
                          seed, nb, nk) {
    m <- new(SpikeSlabRJMCMC, X, y,
             a_pi, b_pi,
             as.integer(seed), TRUE)
    m$step(as.integer(nb)); m$step(as.integer(nk))
    h <- m$get_history()
    # History keys now "sigma" and "tau" (natural); square to match diagnostics.
    list(
        gamma  = h$gamma[(nb+1):(nb+nk), , drop=FALSE],
        beta   = h$beta [(nb+1):(nb+nk), , drop=FALSE],
        sigma2 = h$sigma[(nb+1):(nb+nk)]^2,
        tau2   = h$tau  [(nb+1):(nb+nk)]^2,
        pi     = h$pi   [(nb+1):(nb+nk)]
    )
}

# Data generators.
gen_X_iid <- function(N, p, seed) {
    set.seed(seed); X <- matrix(rnorm(N*p), N, p)
    X - matrix(colMeans(X), N, p, byrow=TRUE)
}
gen_X_ar1 <- function(N, p, rho, seed) {
    set.seed(seed)
    Sigma <- rho^abs(outer(1:p, 1:p, "-"))
    X <- matrix(rnorm(N*p), N, p) %*% chol(Sigma)
    X - matrix(colMeans(X), N, p, byrow=TRUE)
}
gen_y <- function(X, beta_true, sigma_true, seed) {
    set.seed(seed + 12345); N <- nrow(X)
    y_raw <- as.numeric(X %*% beta_true + rnorm(N, 0, sigma_true))
    y_raw - mean(y_raw)
}

# Diagnose a 2-chain run.
diag_2chain <- function(label, res_list, beta_true, sigma_true) {
    stopifnot(length(res_list) == 2)
    c1 <- res_list[[1]]; c2 <- res_list[[2]]
    p <- ncol(c1$gamma)
    gamma_all <- rbind(c1$gamma, c2$gamma)
    beta_all  <- rbind(c1$beta,  c2$beta)
    incl <- colMeans(gamma_all)

    beta_cond_mean <- numeric(p)
    beta_cond_sd   <- numeric(p)
    for (j in 1:p) {
        active <- which(gamma_all[, j] == 1)
        if (length(active) >= 10) {
            beta_cond_mean[j] <- mean(beta_all[active, j])
            beta_cond_sd[j]   <- sd(beta_all[active, j])
        } else { beta_cond_mean[j] <- NA; beta_cond_sd[j] <- NA }
    }

    rhat_gamma <- rhat_mat(list(c1$gamma, c2$gamma))
    rhat_beta  <- rhat_mat(list(c1$beta,  c2$beta))
    rhat_sigma2 <- rhat_vec(list(c1$sigma2, c2$sigma2))
    rhat_pi     <- rhat_vec(list(c1$pi, c2$pi))
    rhat_tau2   <- if (!is.null(c1$tau2))
        rhat_vec(list(c1$tau2, c2$tau2)) else NA_real_

    active_set <- which(incl > 0.01)
    rhat_gamma_active <- if (length(active_set) > 0)
        max(rhat_gamma[active_set], na.rm=TRUE) else NA
    rhat_beta_active <- if (length(active_set) > 0)
        max(rhat_beta[active_set], na.rm=TRUE) else NA

    sigma2_pm <- mean(c(c1$sigma2, c2$sigma2))
    sigma2_sd <- sd  (c(c1$sigma2, c2$sigma2))
    pi_pm     <- mean(c(c1$pi, c2$pi))
    tau2_pm   <- if (!is.null(c1$tau2)) mean(c(c1$tau2, c2$tau2)) else NA

    tpos <- which(beta_true != 0); tneg <- which(beta_true == 0)
    all_tpos_ok <- length(tpos) == 0 || all(incl[tpos] > 0.90)
    all_tneg_ok <- length(tneg) == 0 || all(incl[tneg] < 0.15)
    sigma_ok    <- abs(sigma2_pm - sigma_true^2) < 3 * sigma2_sd

    verdict <- if (all_tpos_ok && all_tneg_ok && sigma_ok) "CLEAN" else "ISSUES"

    cat(sprintf("\n--- %s ---\n", label))
    cat(sprintf(" sigma2: pm=%.3f (true=%.3f)  sd=%.3f  R-hat=%.4f\n",
        sigma2_pm, sigma_true^2, sigma2_sd, rhat_sigma2))
    if (!is.na(tau2_pm))
        cat(sprintf(" tau2:   pm=%.3f  sd=%.3f  R-hat=%.4f\n",
            tau2_pm, sd(c(c1$tau2, c2$tau2)), rhat_tau2))
    cat(sprintf(" pi:     pm=%.3f  R-hat=%.4f\n", pi_pm, rhat_pi))
    cat(sprintf(" R-hat max (active subset): gamma=%.3f beta=%.3f\n",
        rhat_gamma_active, rhat_beta_active))
    cat(sprintf(" true-positive incl > 0.90: %s   true-negative incl < 0.15: %s   sigma ok: %s\n",
        all_tpos_ok, all_tneg_ok, sigma_ok))
    # Print first few betas:
    n_show <- min(p, length(tpos) + 3)
    cat(" Top betas:\n")
    for (j in seq_len(n_show)) {
        mark <- if (beta_true[j] != 0) " *TRUE*" else ""
        cat(sprintf("   j=%d  incl=%.3f  beta_cond=%s  true=%.2f%s\n",
            j, incl[j],
            if (is.na(beta_cond_mean[j])) "NA" else sprintf("%.3f", beta_cond_mean[j]),
            beta_true[j], mark))
    }
    cat(sprintf(" VERDICT: %s\n", verdict))

    list(label=label, verdict=verdict, incl=incl,
         beta_cond_mean=beta_cond_mean, beta_cond_sd=beta_cond_sd,
         rhat_sigma2=rhat_sigma2, rhat_pi=rhat_pi, rhat_tau2=rhat_tau2,
         rhat_gamma_active=rhat_gamma_active,
         rhat_beta_active=rhat_beta_active,
         sigma2_pm=sigma2_pm, tau2_pm=tau2_pm, pi_pm=pi_pm,
         all_tpos_ok=all_tpos_ok, all_tneg_ok=all_tneg_ok, sigma_ok=sigma_ok)
}

# =========================================================================
# Part 1: 2-chain diagnostics across diverse scenarios
# =========================================================================

results <- list()

cat("\n================ Part 1: diverse 2-chain scenarios ================\n")

# H1: medium sparsity + hierarchical tau^2.
cat("\n########## H1: Medium (p=30, 5 active) ##########\n")
X_h1 <- gen_X_iid(200L, 30L, seed=2)
beta_h1 <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 25))
y_h1 <- gen_y(X_h1, beta_h1, sigma_true=1.5, seed=2)
ch_h1 <- list(
    run_spikeslab(X_h1, y_h1, 1, 1, 101L, 2000, 3000),
    run_spikeslab(X_h1, y_h1, 1, 1, 202L, 2000, 3000))
results$H1 <- diag_2chain("H1 Medium + hier tau^2 (2ch × 2k+3k)",
                          ch_h1, beta_h1, sigma_true=1.5)

# H2: widely-varying beta magnitudes. Fixed-tau^2 SpikeSlabRJMCMC might
# miss small or shrink large; hierarchical should adapt.
cat("\n########## H2: varying-magnitude beta ##########\n")
X_h2 <- gen_X_iid(200L, 20L, seed=12)
beta_h2 <- c(10, -5, 1.0, -0.5, 0.2, rep(0, 15))
y_h2 <- gen_y(X_h2, beta_h2, sigma_true=0.5, seed=12)
ch_h2 <- list(
    run_spikeslab(X_h2, y_h2, 1, 1, 101L, 3000, 4000),
    run_spikeslab(X_h2, y_h2, 1, 1, 202L, 3000, 4000))
results$H2 <- diag_2chain("H2 Varying-magnitude beta (2ch × 3k+4k)",
                          ch_h2, beta_h2, sigma_true=0.5)

# H3: grouped sparsity — adjacent coefficients in groups, all nonzero or all zero
cat("\n########## H3: grouped sparsity (blocks of 3 active/inactive) ##########\n")
set.seed(13)
X_h3 <- gen_X_iid(200L, 30L, seed=13)
beta_h3 <- c(rep(c(2, -1.5, 1), 1), rep(0, 3), rep(c(-2, 1.5, -1), 1), rep(0, 21))
y_h3 <- gen_y(X_h3, beta_h3, sigma_true=1.0, seed=13)
ch_h3 <- list(
    run_spikeslab(X_h3, y_h3, 1, 1, 101L, 2000, 3000),
    run_spikeslab(X_h3, y_h3, 1, 1, 202L, 2000, 3000))
results$H3 <- diag_2chain("H3 Grouped sparsity (2ch × 2k+3k)",
                          ch_h3, beta_h3, sigma_true=1.0)

# H4: high-dim hierarchical (N = p)
cat("\n########## H4: high-dim hierarchical (N=100, p=100) ##########\n")
X_h4 <- gen_X_iid(100L, 100L, seed=14)
beta_h4 <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 95))
y_h4 <- gen_y(X_h4, beta_h4, sigma_true=1.0, seed=14)
ch_h4 <- list(
    run_spikeslab(X_h4, y_h4, 1, 1, 101L, 3000, 4000),
    run_spikeslab(X_h4, y_h4, 1, 1, 202L, 3000, 4000))
results$H4 <- diag_2chain("H4 high-dim N=p=100 hierarchical (2ch × 3k+4k)",
                          ch_h4, beta_h4, sigma_true=1.0)

# =========================================================================
# Part 2: long-chain diagnostic
# =========================================================================

cat("\n================ Part 2: long-chain diagnostic ================\n")

# L1: correlated X at 10k+10k (was S3 at 2k+3k — verify convergence holds)
cat("\n########## L1: correlated X long run (10k+10k) ##########\n")
X_l1 <- gen_X_ar1(200L, 30L, rho=0.5, seed=3)
beta_l1 <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 25))
y_l1 <- gen_y(X_l1, beta_l1, sigma_true=1.5, seed=3)
ch_l1 <- list(
    run_spikeslab(X_l1, y_l1, 1, 1, 101L, 10000, 10000),
    run_spikeslab(X_l1, y_l1, 1, 1, 202L, 10000, 10000))
results$L1 <- diag_2chain("L1 Correlated X (2ch × 10k+10k)",
                          ch_l1, beta_l1, sigma_true=1.5)

# =========================================================================
# Part 3: 4-chain R-hat stability
# =========================================================================

cat("\n================ Part 3: 4-chain R-hat stability ================\n")

# M1: 4 chains for S2 medium at standard budget, check R-hat stability
cat("\n########## M1: 4-chain R-hat on Medium ##########\n")
X_m1 <- gen_X_iid(200L, 30L, seed=2)
beta_m1 <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 25))
y_m1 <- gen_y(X_m1, beta_m1, sigma_true=1.5, seed=2)
ch_m1 <- lapply(1:4, function(k)
    run_spikeslab(X_m1, y_m1, 1, 1,
                  100L * k, 2000, 3000))

# Compute 4-chain R-hat on sigma2, pi, gamma (active), beta (active).
rhat_sigma2_4ch <- rhat_vec(lapply(ch_m1, `[[`, "sigma2"))
rhat_pi_4ch     <- rhat_vec(lapply(ch_m1, `[[`, "pi"))
rhat_gamma_4ch  <- rhat_mat(lapply(ch_m1, `[[`, "gamma"))
rhat_beta_4ch   <- rhat_mat(lapply(ch_m1, `[[`, "beta"))
incl_4ch <- rowMeans(sapply(ch_m1, function(c) colMeans(c$gamma)))
active_m1 <- which(incl_4ch > 0.01)
rhat_gamma_active_4ch <- if (length(active_m1) > 0)
    max(rhat_gamma_4ch[active_m1], na.rm=TRUE) else NA
rhat_beta_active_4ch <- if (length(active_m1) > 0)
    max(rhat_beta_4ch[active_m1], na.rm=TRUE) else NA
cat(sprintf(" 4-chain R-hat: sigma2=%.4f, pi=%.4f, gamma_active=%.4f, beta_active=%.4f\n",
    rhat_sigma2_4ch, rhat_pi_4ch, rhat_gamma_active_4ch, rhat_beta_active_4ch))
cat(" true indices (1..5) inclusion across 4 chains:\n")
for (j in 1:5) {
    incl_per_ch <- sapply(ch_m1, function(c) mean(c$gamma[, j]))
    cat(sprintf("  j=%d  chains: %.3f %.3f %.3f %.3f  mean %.3f\n",
        j, incl_per_ch[1], incl_per_ch[2], incl_per_ch[3], incl_per_ch[4],
        mean(incl_per_ch)))
}
m1_ok <- rhat_sigma2_4ch < 1.01 && rhat_pi_4ch < 1.01 &&
    (is.na(rhat_gamma_active_4ch) || rhat_gamma_active_4ch < 1.02) &&
    (is.na(rhat_beta_active_4ch)  || rhat_beta_active_4ch  < 1.02)
cat(sprintf(" VERDICT: %s\n", if (m1_ok) "CLEAN (all 4-chain R-hat < 1.02)" else "ISSUES"))
results$M1 <- list(rhat_sigma2=rhat_sigma2_4ch, rhat_pi=rhat_pi_4ch,
                   rhat_gamma_active=rhat_gamma_active_4ch,
                   rhat_beta_active=rhat_beta_active_4ch,
                   verdict=if (m1_ok) "CLEAN" else "ISSUES")

# =========================================================================
# Part 4: coverage simulation
# =========================================================================

cat("\n================ Part 4: coverage simulation ================\n")
cat("\n########## C1: 30 replicates of Easy scenario ##########\n")
cat("For each rep, check 95% CI of (sigma2, pi, beta_j) covers truth.\n")

N_REPS <- 30L
beta_true_c1 <- c(3, -2, 0, 0, 1.5, 0, 0, 0, 0, 0)
sigma_true_c1 <- 1.0

cov_sigma2 <- logical(N_REPS)
cov_pi     <- logical(N_REPS)  # only weakly meaningful (pi has no "truth"); use a reasonable range
cov_beta1  <- logical(N_REPS)
cov_beta2  <- logical(N_REPS)
cov_beta5  <- logical(N_REPS)
true_incl_hit <- integer(N_REPS)  # out of 3 (hit of P > 0.5 on true nonzero)

t0 <- Sys.time()
for (r in 1:N_REPS) {
    set.seed(r)
    X <- gen_X_iid(100L, 10L, seed=r)
    y <- gen_y(X, beta_true_c1, sigma_true=sigma_true_c1, seed=r)
    m <- new(SpikeSlabRJMCMC, X, y, 1, 1,
             as.integer(r), TRUE)
    m$step(2000L); m$step(2000L)
    h <- m$get_history()
    keep <- 2001:4000
    s2 <- h$sigma[keep]^2; pi_v <- h$pi[keep]
    gamma_h <- h$gamma[keep, , drop=FALSE]
    beta_h <- h$beta[keep, , drop=FALSE]
    # sigma2 CI
    q <- quantile(s2, c(0.025, 0.975), names=FALSE)
    cov_sigma2[r] <- sigma_true_c1^2 >= q[1] && sigma_true_c1^2 <= q[2]
    # pi: no true value; just check it's not absurd
    q <- quantile(pi_v, c(0.025, 0.975), names=FALSE)
    cov_pi[r] <- 0.3 >= q[1] && 0.3 <= q[2]  # truth implied by 3/10
    # beta_j conditional on gamma_j = 1
    for (j in c(1, 2, 5)) {
        active <- which(gamma_h[, j] == 1)
        if (length(active) >= 20) {
            q <- quantile(beta_h[active, j], c(0.025, 0.975), names=FALSE)
            hit <- beta_true_c1[j] >= q[1] && beta_true_c1[j] <= q[2]
        } else {
            hit <- FALSE   # variable never selected — cannot cover
        }
        if (j == 1) cov_beta1[r] <- hit
        if (j == 2) cov_beta2[r] <- hit
        if (j == 5) cov_beta5[r] <- hit
    }
    # true-positive inclusion: count number of true-nonzero j with incl > 0.5
    incl <- colMeans(gamma_h)
    true_incl_hit[r] <- sum(incl[c(1, 2, 5)] > 0.5)
}
t1 <- Sys.time()
cat(sprintf(" 30 reps ran in %.1fs\n", as.numeric(difftime(t1, t0, units="secs"))))
cat(sprintf(" coverage: sigma2 %d/%d (%.0f%%, nominal 95)\n",
    sum(cov_sigma2), N_REPS, 100*mean(cov_sigma2)))
cat(sprintf(" coverage: pi     %d/%d (%.0f%%, nominal 95 — pi true ≈ 0.30)\n",
    sum(cov_pi), N_REPS, 100*mean(cov_pi)))
cat(sprintf(" coverage: beta_1 %d/%d (%.0f%%, conditional on gamma_1 = 1)\n",
    sum(cov_beta1), N_REPS, 100*mean(cov_beta1)))
cat(sprintf(" coverage: beta_2 %d/%d (%.0f%%, conditional on gamma_2 = 1)\n",
    sum(cov_beta2), N_REPS, 100*mean(cov_beta2)))
cat(sprintf(" coverage: beta_5 %d/%d (%.0f%%, conditional on gamma_5 = 1)\n",
    sum(cov_beta5), N_REPS, 100*mean(cov_beta5)))
cat(sprintf(" true-inclusion hits (out of 3): mean %.2f, all-3 %d/%d\n",
    mean(true_incl_hit), sum(true_incl_hit == 3), N_REPS))
results$C1 <- list(
    cov_sigma2=mean(cov_sigma2), cov_pi=mean(cov_pi),
    cov_beta1=mean(cov_beta1),   cov_beta2=mean(cov_beta2),
    cov_beta5=mean(cov_beta5),
    true_incl_all3=sum(true_incl_hit == 3) / N_REPS)

# =========================================================================
# Final summary
# =========================================================================

cat("\n\n================ FINAL SUMMARY ================\n")
for (nm in names(results)) {
    r <- results[[nm]]
    if (!is.null(r$verdict)) {
        cat(sprintf(" %-40s  %s\n", nm, r$verdict))
    }
}
cat("\nC1 coverage (nominal 95%):")
cat(sprintf("\n  sigma2 %.0f%%  pi %.0f%%  beta_1 %.0f%%  beta_2 %.0f%%  beta_5 %.0f%%  all3_hit %.0f%%\n",
    100*results$C1$cov_sigma2, 100*results$C1$cov_pi,
    100*results$C1$cov_beta1, 100*results$C1$cov_beta2,
    100*results$C1$cov_beta5, 100*results$C1$true_incl_all3))
saveRDS(results, file.path(AI4BayesCode_dir, "audit_rjmcmc_xl_results.rds"))
cat("\nSaved to audit_rjmcmc_xl_results.rds\n")

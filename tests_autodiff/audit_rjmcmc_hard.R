# Extended-hard RJMCMC audit — harder simulation scenarios at 10k+10k
# per chain to avoid any non-convergence uncertainty. Builds on
# audit_rjmcmc.R and audit_rjmcmc_xl.R. Uses the unified
# SpikeSlabRJMCMC (full hierarchical: pi, sigma^2, tau^2 all sampled).
#
# Scenarios:
#   X1: Small N  (N=50, p=10, 2 nonzero)
#   X2: p >> N   (N=60, p=100, 5 nonzero)
#   X3: Weak signal (SNR ~1)  (N=200, p=20, 3 nonzero, beta around 1.0, sigma=1.0)
#   X4: Strong correlation AR(0.9)  (N=200, p=30)
#   X5: Grouped sparsity with tighter pi prior Beta(1, 5)
#   X6: N=p=100 with tighter pi prior Beta(1, 5)
#
# All scenarios: 2 chains × 10000 burnin + 10000 keep.
# Diagnostics: R-hat on sigma^2 / tau^2 / pi / gamma (active) / beta
# (active); inclusion probabilities; max false-positive P(gamma=1).

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

rhat_vec <- function(vl) {
    arr <- array(unlist(vl), dim=c(length(vl[[1]]), length(vl), 1))
    tryCatch(posterior::rhat(arr[,,1]), error = function(e) NA_real_)
}
rhat_mat <- function(ml) {
    p <- ncol(ml[[1]])
    vapply(seq_len(p),
           function(j) rhat_vec(lapply(ml, function(m) m[, j])),
           numeric(1))
}

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
gen_y <- function(X, beta, sigma, seed) {
    set.seed(seed + 12345); N <- nrow(X)
    y_raw <- as.numeric(X %*% beta + rnorm(N, 0, sigma))
    y_raw - mean(y_raw)
}

run_ss <- function(X, y, a_pi, b_pi,
                   seed, nb, nk) {
    m <- new(SpikeSlabRJMCMC, X, y,
             a_pi, b_pi,
             as.integer(seed), TRUE)
    m$step(as.integer(nb)); m$step(as.integer(nk))
    h <- m$get_history()
    # History keys "sigma" and "tau" (natural scale); square for diagnostics.
    list(gamma=h$gamma[(nb+1):(nb+nk), , drop=FALSE],
         beta =h$beta [(nb+1):(nb+nk), , drop=FALSE],
         sigma2=h$sigma[(nb+1):(nb+nk)]^2,
         tau2  =h$tau  [(nb+1):(nb+nk)]^2,
         pi    =h$pi   [(nb+1):(nb+nk)])
}

diag_2 <- function(label, res_list, beta_true, sigma_true, t0, t1) {
    c1 <- res_list[[1]]; c2 <- res_list[[2]]
    p <- ncol(c1$gamma)
    gamma_all <- rbind(c1$gamma, c2$gamma)
    incl <- colMeans(gamma_all)
    tpos <- which(beta_true != 0); tneg <- which(beta_true == 0)
    sigma2_pm <- mean(c(c1$sigma2, c2$sigma2))
    sigma2_sd <- sd  (c(c1$sigma2, c2$sigma2))
    pi_pm <- mean(c(c1$pi, c2$pi))
    tau2_pm <- if (!is.null(c1$tau2)) mean(c(c1$tau2, c2$tau2)) else NA

    rhat_sigma2 <- rhat_vec(list(c1$sigma2, c2$sigma2))
    rhat_pi     <- rhat_vec(list(c1$pi, c2$pi))
    rhat_tau2   <- if (!is.null(c1$tau2))
        rhat_vec(list(c1$tau2, c2$tau2)) else NA
    rhat_gamma  <- rhat_mat(list(c1$gamma, c2$gamma))
    rhat_beta   <- rhat_mat(list(c1$beta,  c2$beta))
    active <- which(incl > 0.01)
    rh_g_act <- if (length(active) > 0) max(rhat_gamma[active], na.rm=TRUE) else NA
    rh_b_act <- if (length(active) > 0) max(rhat_beta[active],  na.rm=TRUE) else NA

    max_fpos <- if (length(tneg) > 0) max(incl[tneg]) else 0
    min_tpos <- if (length(tpos) > 0) min(incl[tpos]) else 1

    wall <- as.numeric(difftime(t1, t0, units="secs"))

    cat(sprintf("\n--- %s ---  wall=%.1fs\n", label, wall))
    cat(sprintf(" sigma2 pm=%.3f (true=%.3f) sd=%.3f Rhat=%.4f\n",
        sigma2_pm, sigma_true^2, sigma2_sd, rhat_sigma2))
    if (!is.na(tau2_pm))
        cat(sprintf(" tau2   pm=%.3f sd=%.3f Rhat=%.4f\n",
            tau2_pm, sd(c(c1$tau2, c2$tau2)), rhat_tau2))
    cat(sprintf(" pi     pm=%.3f Rhat=%.4f\n", pi_pm, rhat_pi))
    cat(sprintf(" Rhat (active subset): gamma=%.3f beta=%.3f\n",
        rh_g_act, rh_b_act))
    cat(sprintf(" min true-positive incl: %.3f  /  max true-negative incl: %.3f\n",
        min_tpos, max_fpos))

    # Print per-j for true-nonzero plus top-5 false-positives.
    beta_cond <- numeric(p)
    beta_cond_sd <- numeric(p)
    for (j in 1:p) {
        act <- which(gamma_all[, j] == 1)
        if (length(act) >= 20) {
            beta_cond[j] <- mean(rbind(c1$beta, c2$beta)[act, j])
            beta_cond_sd[j] <- sd(rbind(c1$beta, c2$beta)[act, j])
        } else { beta_cond[j] <- NA; beta_cond_sd[j] <- NA }
    }
    cat(" True-nonzero indices:\n")
    for (j in tpos) {
        cat(sprintf("   j=%d  incl=%.3f  beta_cond=%s (true=%.2f)\n",
            j, incl[j],
            if (is.na(beta_cond[j])) "NA" else sprintf("%.3f", beta_cond[j]),
            beta_true[j]))
    }
    top_fp <- order(incl[tneg], decreasing=TRUE)[seq_len(min(5, length(tneg)))]
    if (length(top_fp) > 0) {
        cat(" Top 5 false-positives:\n")
        for (k in seq_along(top_fp)) {
            j <- tneg[top_fp[k]]
            cat(sprintf("   j=%d  incl=%.3f\n", j, incl[j]))
        }
    }

    verdict <- if (rhat_sigma2 < 1.02 &&
                   rh_g_act < 1.05 && rh_b_act < 1.05 &&
                   min_tpos > 0.5 &&     # all true signals get > 50% incl
                   abs(sigma2_pm - sigma_true^2) < 3 * sigma2_sd)
                  "CLEAN" else "ISSUES"
    cat(sprintf(" VERDICT: %s\n", verdict))
    list(label=label, verdict=verdict, incl=incl,
         beta_cond=beta_cond, beta_cond_sd=beta_cond_sd,
         sigma2_pm=sigma2_pm, tau2_pm=tau2_pm, pi_pm=pi_pm,
         rhat_sigma2=rhat_sigma2, rhat_tau2=rhat_tau2, rhat_pi=rhat_pi,
         rhat_gamma_active=rh_g_act, rhat_beta_active=rh_b_act,
         max_false_pos=max_fpos, min_true_pos=min_tpos, wall=wall)
}

results <- list()
NB <- 10000L; NK <- 10000L

# =============================================================================
# X1. Small N (N=50, p=10, 2 nonzero)
# =============================================================================
cat("\n############# X1: Small N (N=50, p=10, 2 nonzero), 10k+10k ##########\n")
X <- gen_X_iid(50L, 10L, seed=21)
beta <- c(3, -2, rep(0, 8)); y <- gen_y(X, beta, sigma=1, seed=21)
t0 <- Sys.time()
ch <- list(run_ss(X, y, 1, 1, 101L, NB, NK),
           run_ss(X, y, 1, 1, 202L, NB, NK))
t1 <- Sys.time()
results$X1 <- diag_2("X1 Small N", ch, beta, 1, t0, t1)

# =============================================================================
# X2. p >> N (N=60, p=100, 5 nonzero)
# =============================================================================
cat("\n############# X2: p >> N (N=60, p=100, 5 nonzero), 10k+10k #########\n")
X <- gen_X_iid(60L, 100L, seed=22)
beta <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 95))
y <- gen_y(X, beta, sigma=0.8, seed=22)
t0 <- Sys.time()
ch <- list(run_ss(X, y, 1, 10, 101L, NB, NK),
           run_ss(X, y, 1, 10, 202L, NB, NK))
t1 <- Sys.time()
results$X2 <- diag_2("X2 p>>N", ch, beta, 0.8, t0, t1)

# =============================================================================
# X3. Weak signal (SNR ~ 1)
# =============================================================================
cat("\n############# X3: Weak signal (N=200, p=20, SNR~1), 10k+10k ########\n")
X <- gen_X_iid(200L, 20L, seed=23)
beta <- c(1.0, -1.0, 0.8, rep(0, 17))   # all |beta| ~ 1
y <- gen_y(X, beta, sigma=1.0, seed=23)
t0 <- Sys.time()
ch <- list(run_ss(X, y, 1, 1, 101L, NB, NK),
           run_ss(X, y, 1, 1, 202L, NB, NK))
t1 <- Sys.time()
results$X3 <- diag_2("X3 Weak signal", ch, beta, 1.0, t0, t1)

# =============================================================================
# X4. Very strong correlation AR(0.9)
# =============================================================================
cat("\n############# X4: Strong corr AR(0.9) (N=200, p=30), 10k+10k ########\n")
X <- gen_X_ar1(200L, 30L, rho=0.9, seed=24)
beta <- c(3, rep(0, 9), -2, rep(0, 9), 1.5, rep(0, 9))
y <- gen_y(X, beta, sigma=1.5, seed=24)
t0 <- Sys.time()
ch <- list(run_ss(X, y, 1, 5, 101L, NB, NK),
           run_ss(X, y, 1, 5, 202L, NB, NK))
t1 <- Sys.time()
results$X4 <- diag_2("X4 Strong corr AR(0.9)", ch, beta, 1.5, t0, t1)

# =============================================================================
# X5. H3 (grouped sparsity) replay with tighter pi prior Beta(1, 5), 10k+10k
# =============================================================================
cat("\n############# X5: H3 replay + Beta(1,5) pi, 10k+10k ###############\n")
X <- gen_X_iid(200L, 30L, seed=13)
beta <- c(2, -1.5, 1, 0, 0, 0, -2, 1.5, -1, rep(0, 21))
y <- gen_y(X, beta, sigma=1.0, seed=13)
t0 <- Sys.time()
ch <- list(run_ss(X, y, 1, 5, 101L, NB, NK),
           run_ss(X, y, 1, 5, 202L, NB, NK))
t1 <- Sys.time()
results$X5 <- diag_2("X5 H3 replay tighter pi prior", ch, beta, 1.0, t0, t1)

# =============================================================================
# X6. H4 (N=p=100) replay with tighter pi prior, 10k+10k
# =============================================================================
cat("\n############# X6: H4 replay + Beta(1,5) pi, 10k+10k ###############\n")
X <- gen_X_iid(100L, 100L, seed=14)
beta <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 95))
y <- gen_y(X, beta, sigma=1.0, seed=14)
t0 <- Sys.time()
ch <- list(run_ss(X, y, 1, 5, 101L, NB, NK),
           run_ss(X, y, 1, 5, 202L, NB, NK))
t1 <- Sys.time()
results$X6 <- diag_2("X6 H4 replay tighter pi prior", ch, beta, 1.0, t0, t1)

# =============================================================================
# Summary
# =============================================================================
cat("\n\n================ HARD AUDIT SUMMARY (6 scenarios, 10k+10k) ================\n")
for (nm in names(results)) {
    r <- results[[nm]]
    cat(sprintf(" %-6s  %-35s  %s  min_tpos=%.3f max_fpos=%.3f  wall=%.1fs\n",
        nm, r$label, r$verdict, r$min_true_pos, r$max_false_pos, r$wall))
}
saveRDS(results, file.path(AI4BayesCode_dir, "audit_rjmcmc_hard_results.rds"))
cat("\nSaved to audit_rjmcmc_hard_results.rds\n")

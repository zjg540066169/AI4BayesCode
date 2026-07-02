# Long-chain experiment for SpikeSlabRJMCMC refactor.
#
# Motivation: after the T3-refactor to sigma-scaled Ishwaran-Rao slab +
# double-Jeffreys, several "ISSUES" verdicts persisted in audit_rjmcmc_xl
# (H1-H4, L1, M1) and audit_rjmcmc_hard (X2, X6). The common failure is
# cross-chain R-hat > 1.02 on sigma^2/tau^2 even though per-chain beta/gamma
# recovery is correct (min_tpos = 1.000 everywhere). Hypothesis: this is
# structural gamma multimodality (different chains settle at different
# gamma patterns with equivalent true-signal recovery). Test:
#
#   L1-extended: S2/S3/H1/L1 at 10k+10k  (vs 2k-3k default)
#   M1-extended: 4-chain R-hat at 10k+10k (vs 2k+3k default)
#   X2/X6-extended: at 30k+30k          (vs 10k+10k default)
#
# If R-hat cleans up with longer chains -> mixing slowness.
# If R-hat persists -> structural multimodality (chains find different
# but equivalent gamma patterns).

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
run_ss <- function(X, y, a_pi, b_pi, seed, nb, nk) {
    m <- new(SpikeSlabRJMCMC, X, y, a_pi, b_pi,
             as.integer(seed), TRUE)
    m$step(as.integer(nb)); m$step(as.integer(nk))
    h <- m$get_history()
    list(gamma=h$gamma[(nb+1):(nb+nk), , drop=FALSE],
         beta =h$beta [(nb+1):(nb+nk), , drop=FALSE],
         sigma=h$sigma[(nb+1):(nb+nk)],
         tau  =h$tau  [(nb+1):(nb+nk)],
         pi   =h$pi   [(nb+1):(nb+nk)])
}
rhat_vec <- function(vl) {
    arr <- array(unlist(vl), dim=c(length(vl[[1]]), length(vl), 1))
    tryCatch(posterior::rhat(arr[,,1]), error=function(e) NA_real_)
}
rhat_mat_active <- function(ml, active_idx) {
    if (length(active_idx) == 0) return(NA_real_)
    max(sapply(active_idx, function(j)
        rhat_vec(lapply(ml, function(m) m[, j]))), na.rm=TRUE)
}

diag_2 <- function(label, chains, beta_true, sigma_true) {
    c1 <- chains[[1]]; c2 <- chains[[2]]
    gamma_all <- rbind(c1$gamma, c2$gamma)
    incl <- colMeans(gamma_all)
    active <- which(incl > 0.01)
    rh_sigma <- rhat_vec(list(c1$sigma, c2$sigma))
    rh_tau   <- rhat_vec(list(c1$tau,   c2$tau))
    rh_pi    <- rhat_vec(list(c1$pi,    c2$pi))
    rh_g_act <- rhat_mat_active(list(c1$gamma, c2$gamma), active)
    rh_b_act <- rhat_mat_active(list(c1$beta,  c2$beta),  active)
    tpos <- which(beta_true != 0); tneg <- which(beta_true == 0)
    min_tpos <- if (length(tpos) > 0) min(incl[tpos]) else 1
    max_tneg <- if (length(tneg) > 0) max(incl[tneg]) else 0
    cat(sprintf("  %-16s  R-hat sigma=%.3f  tau=%.3f  pi=%.3f | active: gamma=%.3f beta=%.3f\n",
        label, rh_sigma, rh_tau, rh_pi, rh_g_act, rh_b_act))
    cat(sprintf("                    min_tpos=%.3f  max_fpos=%.3f  sigma_pm=%.3f (true=%.3f)\n",
        min_tpos, max_tneg, mean(c(c1$sigma, c2$sigma)), sigma_true))
    list(rh_sigma=rh_sigma, rh_tau=rh_tau, rh_pi=rh_pi,
         rh_g_act=rh_g_act, rh_b_act=rh_b_act,
         min_tpos=min_tpos, max_fpos=max_tneg)
}

results <- list()

# --------------------------------------------------------------------------
# Group 1 — S2/S3 at 10k+10k (Phase 4 ran at 2k+3k)
# --------------------------------------------------------------------------
cat("\n================ Group 1: S2/S3 at 10k+10k (was 2k+3k) ================\n")
NB1 <- 10000L; NK1 <- 10000L

cat("\n-- S2 Medium (N=200, p=30, 5 nonzero, sigma=1.5), 10k+10k --\n")
X2 <- gen_X_iid(200L, 30L, seed=2)
beta_2 <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 25))
y2 <- gen_y(X2, beta_2, sigma=1.5, seed=2)
t0 <- Sys.time()
ch <- list(run_ss(X2, y2, 1, 1, 101L, NB1, NK1),
           run_ss(X2, y2, 1, 1, 202L, NB1, NK1))
t1 <- Sys.time()
cat(sprintf("  wall=%.1fs\n", as.numeric(difftime(t1, t0, units="secs"))))
results$S2_10k <- diag_2("S2 @ 10k+10k", ch, beta_2, 1.5)

cat("\n-- S3 Correlated X (N=200, p=30, rho=0.5, sigma=1.5), 10k+10k --\n")
X3 <- gen_X_ar1(200L, 30L, rho=0.5, seed=3)
y3 <- gen_y(X3, beta_2, sigma=1.5, seed=3)
t0 <- Sys.time()
ch <- list(run_ss(X3, y3, 1, 1, 101L, NB1, NK1),
           run_ss(X3, y3, 1, 1, 202L, NB1, NK1))
t1 <- Sys.time()
cat(sprintf("  wall=%.1fs\n", as.numeric(difftime(t1, t0, units="secs"))))
results$S3_10k <- diag_2("S3 @ 10k+10k", ch, beta_2, 1.5)

# --------------------------------------------------------------------------
# Group 2 — H1/H3/H4 at 10k+10k (Phase 5 ran at 2-3k)
# --------------------------------------------------------------------------
cat("\n================ Group 2: H1/H3/H4 at 10k+10k ================\n")

cat("\n-- H1 Medium + hier tau, 10k+10k --\n")
X_h1 <- gen_X_iid(200L, 30L, seed=2)
beta_h1 <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 25))
y_h1 <- gen_y(X_h1, beta_h1, sigma=1.5, seed=2)
t0 <- Sys.time()
ch <- list(run_ss(X_h1, y_h1, 1, 1, 101L, NB1, NK1),
           run_ss(X_h1, y_h1, 1, 1, 202L, NB1, NK1))
t1 <- Sys.time()
cat(sprintf("  wall=%.1fs\n", as.numeric(difftime(t1, t0, units="secs"))))
results$H1_10k <- diag_2("H1 @ 10k+10k", ch, beta_h1, 1.5)

cat("\n-- H3 Grouped sparsity, 10k+10k --\n")
X_h3 <- gen_X_iid(200L, 30L, seed=13)
beta_h3 <- c(rep(c(2, -1.5, 1), 1), rep(0, 3), rep(c(-2, 1.5, -1), 1), rep(0, 21))
y_h3 <- gen_y(X_h3, beta_h3, sigma=1.0, seed=13)
t0 <- Sys.time()
ch <- list(run_ss(X_h3, y_h3, 1, 1, 101L, NB1, NK1),
           run_ss(X_h3, y_h3, 1, 1, 202L, NB1, NK1))
t1 <- Sys.time()
cat(sprintf("  wall=%.1fs\n", as.numeric(difftime(t1, t0, units="secs"))))
results$H3_10k <- diag_2("H3 @ 10k+10k", ch, beta_h3, 1.0)

cat("\n-- H4 N=p=100, 10k+10k --\n")
X_h4 <- gen_X_iid(100L, 100L, seed=14)
beta_h4 <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 95))
y_h4 <- gen_y(X_h4, beta_h4, sigma=1.0, seed=14)
t0 <- Sys.time()
ch <- list(run_ss(X_h4, y_h4, 1, 1, 101L, NB1, NK1),
           run_ss(X_h4, y_h4, 1, 1, 202L, NB1, NK1))
t1 <- Sys.time()
cat(sprintf("  wall=%.1fs\n", as.numeric(difftime(t1, t0, units="secs"))))
results$H4_10k <- diag_2("H4 @ 10k+10k", ch, beta_h4, 1.0)

# --------------------------------------------------------------------------
# Group 3 — 4-chain R-hat at 10k+10k (M1 used 2k+3k × 4)
# --------------------------------------------------------------------------
cat("\n================ Group 3: 4-chain R-hat on Medium at 10k+10k ================\n")
cat("\n-- M1-extended: 4 chains × 10k+10k --\n")
X_m1 <- gen_X_iid(200L, 30L, seed=2)
beta_m1 <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 25))
y_m1 <- gen_y(X_m1, beta_m1, sigma=1.5, seed=2)
t0 <- Sys.time()
ch_m1 <- lapply(1:4, function(k)
    run_ss(X_m1, y_m1, 1, 1, 100L*k, NB1, NK1))
t1 <- Sys.time()
rh_sigma <- rhat_vec(lapply(ch_m1, `[[`, "sigma"))
rh_tau   <- rhat_vec(lapply(ch_m1, `[[`, "tau"))
rh_pi    <- rhat_vec(lapply(ch_m1, `[[`, "pi"))
gamma_all <- do.call(rbind, lapply(ch_m1, `[[`, "gamma"))
active <- which(colMeans(gamma_all) > 0.01)
rh_g_act <- rhat_mat_active(lapply(ch_m1, `[[`, "gamma"), active)
rh_b_act <- rhat_mat_active(lapply(ch_m1, `[[`, "beta"),  active)
cat(sprintf("  wall=%.1fs\n", as.numeric(difftime(t1, t0, units="secs"))))
cat(sprintf("  4-chain R-hat: sigma=%.3f  tau=%.3f  pi=%.3f | active: gamma=%.3f beta=%.3f\n",
    rh_sigma, rh_tau, rh_pi, rh_g_act, rh_b_act))
results$M1_4ch_10k <- list(rh_sigma=rh_sigma, rh_tau=rh_tau, rh_pi=rh_pi,
                           rh_g_act=rh_g_act, rh_b_act=rh_b_act)

# --------------------------------------------------------------------------
# Group 4 — X2/X6 at 30k+30k (were ISSUES at 10k+10k in hard audit)
# --------------------------------------------------------------------------
cat("\n================ Group 4: X2/X6 at 30k+30k (was ISSUES at 10k+10k) ================\n")
NB2 <- 30000L; NK2 <- 30000L

cat("\n-- X2 p>>N (N=60, p=100, 5 nonzero), 30k+30k --\n")
X_x2 <- gen_X_iid(60L, 100L, seed=22)
beta_x2 <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 95))
y_x2 <- gen_y(X_x2, beta_x2, sigma=0.8, seed=22)
t0 <- Sys.time()
ch <- list(run_ss(X_x2, y_x2, 1, 10, 101L, NB2, NK2),
           run_ss(X_x2, y_x2, 1, 10, 202L, NB2, NK2))
t1 <- Sys.time()
cat(sprintf("  wall=%.1fs\n", as.numeric(difftime(t1, t0, units="secs"))))
results$X2_30k <- diag_2("X2 @ 30k+30k", ch, beta_x2, 0.8)

cat("\n-- X6 H4 replay + Beta(1,5) pi (N=p=100), 30k+30k --\n")
X_x6 <- gen_X_iid(100L, 100L, seed=14)
beta_x6 <- c(3, -2.5, 2, -1.5, 1.5, rep(0, 95))
y_x6 <- gen_y(X_x6, beta_x6, sigma=1.0, seed=14)
t0 <- Sys.time()
ch <- list(run_ss(X_x6, y_x6, 1, 5, 101L, NB2, NK2),
           run_ss(X_x6, y_x6, 1, 5, 202L, NB2, NK2))
t1 <- Sys.time()
cat(sprintf("  wall=%.1fs\n", as.numeric(difftime(t1, t0, units="secs"))))
results$X6_30k <- diag_2("X6 @ 30k+30k", ch, beta_x6, 1.0)

# --------------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------------
cat("\n\n================ LONG-CHAIN SUMMARY ================\n")
cat("threshold: clean if R-hat_active < 1.05 AND min_tpos > 0.9\n\n")
for (nm in names(results)) {
    r <- results[[nm]]
    if (!is.null(r$min_tpos)) {
        clean_primary <- !is.na(r$rh_g_act) && !is.na(r$rh_b_act) &&
                         r$rh_g_act < 1.05 && r$rh_b_act < 1.05 &&
                         r$min_tpos > 0.9
        clean_secondary <- !is.na(r$rh_sigma) && !is.na(r$rh_tau) &&
                           r$rh_sigma < 1.05 && r$rh_tau < 1.05
        cat(sprintf(" %-14s  primary(beta/gamma)=%s  secondary(sigma/tau)=%s\n",
            nm,
            if (clean_primary) "CLEAN" else "ISSUES",
            if (clean_secondary) "CLEAN" else "ISSUES"))
    } else {
        clean_primary <- !is.na(r$rh_g_act) && !is.na(r$rh_b_act) &&
                         r$rh_g_act < 1.05 && r$rh_b_act < 1.05
        cat(sprintf(" %-14s  4-chain gamma R-hat=%.3f  beta R-hat=%.3f\n",
            nm, r$rh_g_act, r$rh_b_act))
    }
}

saveRDS(results, file.path(AI4BayesCode_dir, "audit_rjmcmc_longchain.rds"))
cat("\nSaved to audit_rjmcmc_longchain.rds\n")

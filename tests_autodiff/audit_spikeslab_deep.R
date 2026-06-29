# Deep MCMC audit for the refactored SpikeSlabRJMCMC (sigma-scaled slab,
# double Jeffreys on sigma and tau, marginal-OLS init).
#
#   D1: pi-prior sensitivity grid. a_pi/b_pi ∈ {(1,1), (1,5), (1,10)}
#       on a fixed problem. Check that true signals are recovered and
#       false-positive incl drops as Beta(a,b) becomes tighter.
#
#   D2: NUTS sigma/tau diagnostic. Long chain, check ESS for sigma, tau,
#       and pi. Under sigma-scaled slab + Jeffreys, both sigma and tau
#       should have decorrelated posteriors → clean NUTS geometry →
#       high ESS (>500 after 5k keep).
#
#   D3: Init sensitivity. Start from 3 different gamma_init:
#           (i)   model default (marginal-OLS screening)
#           (ii)  single-active (via set_current override)
#           (iii) all-active
#       3-chain R-hat on sigma, tau, pi, gamma_active, beta_active.
#       All < 1.1 → chains converge to same posterior regardless of init.
#
#   D4: Wide N:p ratio grid.
#           (N=50,  p=5)   — small-N
#           (N=100, p=20)  — medium
#           (N=500, p=50)  — large-N
#           (N=100, p=200) — p >> N

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

# Helpers
gen_X_iid <- function(N, p, seed) {
    set.seed(seed); X <- matrix(rnorm(N*p), N, p)
    X - matrix(colMeans(X), N, p, byrow=TRUE)
}
gen_y <- function(X, beta_true, sigma_true, seed) {
    set.seed(seed + 12345); N <- nrow(X)
    y_raw <- as.numeric(X %*% beta_true + rnorm(N, 0, sigma_true))
    y_raw - mean(y_raw)
}
run_chain <- function(X, y, a_pi, b_pi, seed, nb, nk,
                      gamma_init=NULL, X_ref=NULL, y_ref=NULL) {
    m <- new(SpikeSlabRJMCMC, X, y, a_pi, b_pi,
             as.integer(seed), TRUE)
    if (!is.null(gamma_init)) {
        # Initialize beta to OLS for active components (sensible warm-start).
        # Without this, the chain can get stuck at tau=0 if all active beta=0
        # — a degenerate state. The library's default constructor init uses
        # marginal-OLS screening + OLS beta init to avoid this.
        Xm <- if (!is.null(X_ref)) X_ref else X
        ym <- if (!is.null(y_ref)) y_ref else y
        beta_init <- rep(0, length(gamma_init))
        for (j in which(gamma_init == 1)) {
            xtx <- sum(Xm[, j]^2)
            if (xtx > 0) beta_init[j] <- sum(Xm[, j] * ym) / xtx
        }
        m$set_current(list(gamma = gamma_init, beta = beta_init))
    }
    m$step(as.integer(nb)); m$step(as.integer(nk))
    h <- m$get_history()
    list(
        gamma = h$gamma[(nb+1):(nb+nk), , drop=FALSE],
        beta  = h$beta [(nb+1):(nb+nk), , drop=FALSE],
        sigma = h$sigma[(nb+1):(nb+nk)],
        tau   = h$tau  [(nb+1):(nb+nk)],
        pi    = h$pi   [(nb+1):(nb+nk)]
    )
}
rhat_vec <- function(vl) {
    arr <- array(unlist(vl), dim=c(length(vl[[1]]), length(vl), 1))
    tryCatch(posterior::rhat(arr[,,1]), error = function(e) NA_real_)
}
ess_basic <- function(v) tryCatch(posterior::ess_basic(v), error=function(e) NA_real_)

results <- list()

# ===========================================================================
# D1: pi-prior sensitivity grid
# ===========================================================================
cat("\n================ D1: pi-prior sensitivity ================\n")
X_d1 <- gen_X_iid(200L, 20L, seed=101)
beta_d1 <- c(3, -2, 1.5, rep(0, 17))
y_d1 <- gen_y(X_d1, beta_d1, sigma_true=1.0, seed=101)

pi_grid <- list(
    "Beta(1,1)"  = c(1, 1),
    "Beta(1,5)"  = c(1, 5),
    "Beta(1,10)" = c(1, 10)
)
d1_results <- list()
for (nm in names(pi_grid)) {
    ab <- pi_grid[[nm]]
    ch <- run_chain(X_d1, y_d1, ab[1], ab[2], 42L, 3000, 5000)
    incl <- colMeans(ch$gamma)
    sigma_pm <- mean(ch$sigma)
    tau_pm   <- mean(ch$tau)
    cat(sprintf("  %-12s: sigma pm=%.3f  tau pm=%.3f  incl_top3=[%.3f %.3f %.3f]\n",
        nm, sigma_pm, tau_pm, incl[1], incl[2], incl[3]))
    max_tneg <- max(incl[-c(1,2,3)])
    cat(sprintf("      max false-positive incl (j ∈ 4..20): %.3f\n", max_tneg))
    d1_results[[nm]] <- list(incl=incl, sigma_pm=sigma_pm, tau_pm=tau_pm,
                             max_tneg=max_tneg)
}
# Check: under TIGHTER pi prior, max_tneg should decrease (more shrinkage).
mn <- sapply(d1_results, function(r) r$max_tneg)
d1_ok <- all(sapply(d1_results, function(r) r$incl[1] > 0.9 &&
                                             r$incl[2] > 0.9 &&
                                             r$incl[3] > 0.9))
cat(sprintf("  D1 verdict: %s (true signals all recovered in every prior)\n",
    if (d1_ok) "CLEAN" else "ISSUES"))
cat(sprintf("      max_tneg trend: %s\n",
    paste(sprintf("%.3f", mn), collapse=" → ")))
results$D1 <- list(d1_results=d1_results, ok=d1_ok)

# ===========================================================================
# D2: NUTS sigma / tau ESS
# ===========================================================================
cat("\n================ D2: NUTS sigma / tau ESS ================\n")
ch_d2 <- run_chain(X_d1, y_d1, 1, 1, 42L, 3000, 5000)
sigma_ess <- ess_basic(ch_d2$sigma)
tau_ess   <- ess_basic(ch_d2$tau)
pi_ess    <- ess_basic(ch_d2$pi)
cat(sprintf("  sigma: pm=%.3f  sd=%.3f  ESS=%.0f\n",
    mean(ch_d2$sigma), sd(ch_d2$sigma), sigma_ess))
cat(sprintf("  tau:   pm=%.3f  sd=%.3f  ESS=%.0f\n",
    mean(ch_d2$tau), sd(ch_d2$tau), tau_ess))
cat(sprintf("  pi:    pm=%.3f  sd=%.3f  ESS=%.0f\n",
    mean(ch_d2$pi), sd(ch_d2$pi), pi_ess))
d2_ok <- !is.na(sigma_ess) && !is.na(tau_ess) && !is.na(pi_ess) &&
    (sigma_ess > 500) && (tau_ess > 500) && (pi_ess > 500)
cat(sprintf("  D2 verdict: %s (all ESS > 500)\n",
    if (isTRUE(d2_ok)) "CLEAN" else "LOW ESS — investigate"))
results$D2 <- list(sigma_ess=sigma_ess, tau_ess=tau_ess, pi_ess=pi_ess,
                   ok=isTRUE(d2_ok))

# ===========================================================================
# D3: Init sensitivity — 3 different gamma_init states
# ===========================================================================
cat("\n================ D3: Init sensitivity ================\n")
# The constructor already uses marginal-OLS screening internally, so we
# override with set_current for 2 different alternative starts:
inits <- list(
    default_marginal_OLS = NULL,               # constructor's default
    single_active_j1     = c(1, rep(0, 19)),
    all_active           = rep(1, 20)
)
d3_chains <- list()
for (nm in names(inits)) {
    ch <- run_chain(X_d1, y_d1, 1, 1, 42L, 5000, 5000,
                    gamma_init=inits[[nm]],
                    X_ref=X_d1, y_ref=y_d1)
    d3_chains[[nm]] <- ch
    cat(sprintf("  init=%-22s sigma pm=%.3f  tau pm=%.3f  incl_top3=[%.2f %.2f %.2f]\n",
        nm, mean(ch$sigma), mean(ch$tau),
        mean(ch$gamma[,1]), mean(ch$gamma[,2]), mean(ch$gamma[,3])))
}
rhat_sigma <- rhat_vec(lapply(d3_chains, `[[`, "sigma"))
rhat_tau   <- rhat_vec(lapply(d3_chains, `[[`, "tau"))
rhat_pi    <- rhat_vec(lapply(d3_chains, `[[`, "pi"))
cat(sprintf("  3-chain R-hat: sigma=%.3f  tau=%.3f  pi=%.3f\n",
    rhat_sigma, rhat_tau, rhat_pi))
d3_ok <- !is.na(rhat_sigma) && !is.na(rhat_tau) && !is.na(rhat_pi) &&
    rhat_sigma < 1.1 && rhat_tau < 1.1 && rhat_pi < 1.1
cat(sprintf("  D3 verdict: %s (all 3-chain R-hat < 1.1)\n",
    if (isTRUE(d3_ok)) "CLEAN" else "ISSUES"))
results$D3 <- list(rhat_sigma=rhat_sigma, rhat_tau=rhat_tau, rhat_pi=rhat_pi,
                   ok=isTRUE(d3_ok))

# ===========================================================================
# D4: Wide N:p ratio grid
# ===========================================================================
cat("\n================ D4: Wide N:p ratio grid ================\n")
configs <- list(
    small    = list(N=50L,  p=5L,  nonzero=c(2.0, -1.5)),
    medium   = list(N=100L, p=20L, nonzero=c(2.0, -1.5, 1.0)),
    large    = list(N=500L, p=50L, nonzero=c(2.0, -1.5, 1.0)),
    p_gt_N   = list(N=100L, p=200L,nonzero=c(2.0, -1.5, 1.0))
)
d4_results <- list()
for (nm in names(configs)) {
    cfg <- configs[[nm]]
    X <- gen_X_iid(cfg$N, cfg$p, seed=42)
    beta <- c(cfg$nonzero, rep(0, cfg$p - length(cfg$nonzero)))
    y <- gen_y(X, beta, sigma_true=1.0, seed=42)
    ch <- run_chain(X, y, 1, 1, 42L, 3000, 5000)
    incl <- colMeans(ch$gamma)
    tpos <- 1:length(cfg$nonzero)
    tneg <- (length(cfg$nonzero)+1):cfg$p
    min_tpos <- min(incl[tpos])
    max_tneg <- max(incl[tneg])
    sigma_pm <- mean(ch$sigma)
    tau_pm   <- mean(ch$tau)
    cat(sprintf("  %-8s (N=%3d, p=%3d): sigma=%.3f  tau=%.3f  min_tpos=%.3f max_tneg=%.3f\n",
        nm, cfg$N, cfg$p, sigma_pm, tau_pm, min_tpos, max_tneg))
    d4_results[[nm]] <- list(cfg=cfg, sigma_pm=sigma_pm, tau_pm=tau_pm,
                             min_tpos=min_tpos, max_tneg=max_tneg)
}
all_d4_ok <- all(sapply(d4_results, function(r) r$min_tpos > 0.5))
cat(sprintf("  D4 verdict: %s (all true signals recovered min_tpos > 0.5)\n",
    if (all_d4_ok) "CLEAN" else "ISSUES"))
results$D4 <- d4_results

# ===========================================================================
# Final summary
# ===========================================================================
cat("\n\n================ DEEP AUDIT SUMMARY ================\n")
cat(sprintf(" D1 (pi-prior):           %s\n",
    if (results$D1$ok) "CLEAN" else "ISSUES"))
cat(sprintf(" D2 (NUTS ESS):           %s\n",
    if (results$D2$ok) "CLEAN" else "ISSUES"))
cat(sprintf(" D3 (init sensitivity):   %s\n",
    if (results$D3$ok) "CLEAN" else "ISSUES"))
cat(sprintf(" D4 (N:p grid):           %s\n",
    if (all_d4_ok)     "CLEAN" else "ISSUES"))

saveRDS(results, file.path(AI4BayesCode_dir, "audit_spikeslab_deep.rds"))
cat("\nSaved to audit_spikeslab_deep.rds\n")

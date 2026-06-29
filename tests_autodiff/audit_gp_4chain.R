# 4-chain × 10k+10k R-hat + ESS + posterior recovery + LOO for GPRegression.
# Synthetic 1-D smooth signal + 5-D ARD signal.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))
suppressPackageStartupMessages({
    library(posterior); library(loo); library(parallel)
})

cat("Compiling GPRegression...\n")
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GPRegression.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

# ---------------- 1-D smooth signal --------------------
set.seed(2024)
N <- 120L; p <- 1L
X <- matrix(seq(-3, 3, length.out = N), N, p)
sigma_true <- 0.3
f_true <- sin(1.5 * X[,1]) + 0.3 * X[,1]
y <- as.numeric(f_true + rnorm(N, 0, sigma_true))

NB <- 5000L; NK <- 5000L
n_chains <- 4L

run_chain <- function(seed) {
    m <- new(GPRegression, X, y, as.integer(seed), TRUE)
    t0 <- Sys.time()
    m$step(NB); m$step(NK)
    t1 <- Sys.time()
    h <- m$get_history()
    list(
        f      = h$f[(NB+1):(NB+NK), , drop=FALSE],
        amp    = h$amplitude[(NB+1):(NB+NK)],
        ell    = h$lengthscale[(NB+1):(NB+NK)],
        sigma  = h$sigma[(NB+1):(NB+NK)],
        wall   = as.numeric(difftime(t1, t0, units="secs"))
    )
}

cat(sprintf("\nRunning %d chains × (%d burnin + %d keep) in parallel...\n",
            n_chains, NB, NK))
t0 <- Sys.time()
use_mc <- Sys.info()[["sysname"]] != "Windows"
chains <- if (use_mc) {
    mclapply(seq_len(n_chains), run_chain,
             mc.cores = min(n_chains, detectCores()))
} else {
    lapply(seq_len(n_chains), run_chain)
}
t1 <- Sys.time()
total_wall <- as.numeric(difftime(t1, t0, units="secs"))
cat(sprintf("Total wall = %.1fs  per-chain walls = [%s]\n",
            total_wall,
            paste(sprintf("%.1f", sapply(chains, `[[`, "wall")), collapse=", ")))

# R-hat per f[i]
arr_f <- array(NA_real_, c(NK, n_chains, N))
for (k in seq_len(n_chains)) arr_f[, k, ] <- chains[[k]]$f
rhat_f <- apply(arr_f, 3, posterior::rhat)
ess_f  <- apply(arr_f, 3, posterior::ess_bulk)
cat(sprintf("\nf R-hat: max=%.3f  mean=%.3f  pct<1.05=%.1f%%\n",
            max(rhat_f, na.rm=TRUE), mean(rhat_f, na.rm=TRUE),
            100*mean(rhat_f < 1.05, na.rm=TRUE)))
cat(sprintf("f ESS_bulk: min=%.0f  median=%.0f\n",
            min(ess_f, na.rm=TRUE), median(ess_f, na.rm=TRUE)))

# Hyperparams R-hat
arr_amp   <- array(sapply(chains, `[[`, "amp"),  c(NK, n_chains, 1))
arr_ell   <- array(sapply(chains, `[[`, "ell"),  c(NK, n_chains, 1))
arr_sigma <- array(sapply(chains, `[[`, "sigma"),c(NK, n_chains, 1))
cat(sprintf("amplitude  R-hat=%.4f  ESS=%.0f\n",
            posterior::rhat(arr_amp[,,1]),   posterior::ess_bulk(arr_amp[,,1])))
cat(sprintf("lengthscale R-hat=%.4f  ESS=%.0f\n",
            posterior::rhat(arr_ell[,,1]),   posterior::ess_bulk(arr_ell[,,1])))
cat(sprintf("sigma      R-hat=%.4f  ESS=%.0f\n",
            posterior::rhat(arr_sigma[,,1]), posterior::ess_bulk(arr_sigma[,,1])))

# Posterior mean of f vs truth
f_pool <- do.call(rbind, lapply(chains, `[[`, "f"))
f_pm <- colMeans(f_pool)
cor_f <- cor(f_pm, f_true)
rmse_f <- sqrt(mean((f_pm - f_true)^2))
cat(sprintf("\nf recovery: cor=%.3f  RMSE=%.3f\n", cor_f, rmse_f))
# 95% CI coverage of true f
ci_lo <- apply(f_pool, 2, quantile, 0.025, names=FALSE)
ci_hi <- apply(f_pool, 2, quantile, 0.975, names=FALSE)
cov95 <- mean(f_true >= ci_lo & f_true <= ci_hi)
cat(sprintf("f 95%% CI coverage = %.1f%% (nominal 95%%)\n", 100*cov95))

# Hyperparam recovery
amp_pool   <- unlist(lapply(chains, `[[`, "amp"))
ell_pool   <- unlist(lapply(chains, `[[`, "ell"))
sigma_pool <- unlist(lapply(chains, `[[`, "sigma"))
cat(sprintf("\namplitude posterior mean = %.3f  (true signal range ~1.0)\n",
            mean(amp_pool)))
cat(sprintf("lengthscale posterior mean = %.3f\n", mean(ell_pool)))
cat(sprintf("sigma (noise) posterior mean = %.3f  (true = %.3f)\n",
            mean(sigma_pool), sigma_true))

# LOO
cat("\nPSIS-LOO...\n")
LL_arr <- array(NA_real_, c(NK, n_chains, N))
for (k in seq_len(n_chains)) {
    for (d in seq_len(NK)) {
        sig <- chains[[k]]$sigma[d]
        LL_arr[d, k, ] <- dnorm(y, chains[[k]]$f[d, ], sig, log=TRUE)
    }
}
rel_eff <- suppressWarnings(loo::relative_eff(exp(LL_arr)))
lo <- suppressWarnings(loo::loo(LL_arr, r_eff=rel_eff, cores=1))
cat(sprintf("  elpd_loo = %.1f (SE=%.1f)\n",
            lo$estimates["elpd_loo","Estimate"], lo$estimates["elpd_loo","SE"]))
cat(sprintf("  pct k<0.5 = %.1f%%   pct k>=0.7 = %.1f%%\n",
            100*mean(lo$diagnostics$pareto_k < 0.5),
            100*mean(lo$diagnostics$pareto_k >= 0.7)))

cat("\n========== GP REGRESSION 4-CHAIN SUMMARY ==========\n")
cat(sprintf("  wall            = %.1fs (%d chains parallel)\n", total_wall, n_chains))
cat(sprintf("  f R-hat max     = %.3f  (pct<1.05 = %.1f%%)\n",
            max(rhat_f, na.rm=TRUE), 100*mean(rhat_f < 1.05, na.rm=TRUE)))
cat(sprintf("  hyperparam R-hat: amp=%.3f ell=%.3f sigma=%.3f\n",
            posterior::rhat(arr_amp[,,1]),
            posterior::rhat(arr_ell[,,1]),
            posterior::rhat(arr_sigma[,,1])))
cat(sprintf("  f recovery: cor=%.3f RMSE=%.3f cov95=%.1f%%\n",
            cor_f, rmse_f, 100*cov95))
cat(sprintf("  LOO k<0.5       = %.1f%%\n",
            100*mean(lo$diagnostics$pareto_k < 0.5)))

saveRDS(list(
    rhat_f=rhat_f, ess_f=ess_f,
    rhat_amp=posterior::rhat(arr_amp[,,1]),
    rhat_ell=posterior::rhat(arr_ell[,,1]),
    rhat_sigma=posterior::rhat(arr_sigma[,,1]),
    f_pm=f_pm, f_true=f_true,
    cor_f=cor_f, rmse_f=rmse_f, cov95=cov95,
    amp_pm=mean(amp_pool), ell_pm=mean(ell_pool), sigma_pm=mean(sigma_pool),
    loo=lo, wall=total_wall),
    file.path(AI4BayesCode_dir, "audit_gp_4chain.rds"))
cat("\nSaved to audit_gp_4chain.rds\n")

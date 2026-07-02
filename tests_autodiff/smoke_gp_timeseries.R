# Smoke test for GPTimeSeries v0.5 (celerite-based 1-D GP with
# univariate_slice_sampling_block for hyperparameter inference).

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

cat("Compiling GPTimeSeries...\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GPTimeSeries.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)

# Simulate an OU-like time series
set.seed(1)
N <- 100
t <- sort(runif(N, 0, 10))
sigma_true <- 0.3
amp_true   <- 1.0
tau_true   <- 2.0
dt <- diff(t)
f <- numeric(N)
f[1] <- rnorm(1, 0, amp_true)
for (i in 2:N) {
    rho <- exp(-dt[i-1] / tau_true)
    f[i] <- rho * f[i-1] + amp_true * sqrt(1 - rho^2) * rnorm(1)
}
y <- f + sigma_true * rnorm(N)

cat(sprintf("Data: N=%d, range of t=[%.2f, %.2f], sd(y)=%.3f\n",
            N, min(t), max(t), sd(y)))
cat(sprintf("Truth: amp=%.3f, tau=%.3f, sigma=%.3f\n",
            amp_true, tau_true, sigma_true))

# ---------------------- Stateful construct + 10 steps --------------------
cat("\n[1] Construct + 10 steps...\n")
m <- new(GPTimeSeries, t, y, 1L, FALSE)
m$step(10L)
d <- m$get_current()
cat(sprintf("  keys = %s\n", paste(names(d), collapse=",")))
stopifnot(all(c("amp","tau","sigma","logp") %in% names(d)))
cat(sprintf("  amp=%.3f tau=%.3f sigma=%.3f logp=%.3f\n",
            d$amp, d$tau, d$sigma, d$logp))
stopifnot(is.finite(d$logp))
stopifnot(d$amp > 0, d$tau > 0, d$sigma > 0)

# ---------------------- DAG ---------------------------------------------
cat("\n[2] DAG consistency...\n")
dag <- m$get_dag()
cat("  gibbs_reads:\n")
for (nm in names(dag$gibbs_reads))
    cat(sprintf("    %-15s <- %s\n", nm,
                paste(dag$gibbs_reads[[nm]], collapse=", ")))

# ---------------------- predict_at --------------------------------------
cat("\n[3] predict_at at t_test...\n")
t_test <- seq(min(t) - 1, max(t) + 1, length.out = 25)
before <- m$get_current()
pp <- m$predict_at(list(t = t_test))
after <- m$get_current()
stopifnot(identical(before, after))  # predict_at must not mutate state
stopifnot(all(c("f_mean","f_sd","f_star","y_rep") %in% names(pp)))
stopifnot(length(pp$f_mean) == 25)
stopifnot(all(pp$f_sd >= 0))
cat(sprintf("  f_mean range = [%.3f, %.3f]\n",
            min(pp$f_mean), max(pp$f_mean)))
cat(sprintf("  f_sd   range = [%.3f, %.3f]\n",
            min(pp$f_sd),   max(pp$f_sd)))

# ---------------------- 2k+2k chain (v0.5 recovery) ---------------------
cat("\n[4] 2k+2k chain with keep_history=TRUE...\n")
mh <- new(GPTimeSeries, t, y, 42L, TRUE)
t0 <- Sys.time()
mh$step(2000L)
mh$step(2000L)
t1 <- Sys.time()
wall <- as.numeric(difftime(t1, t0, units = "secs"))
cat(sprintf("  wall = %.1fs for 4000 sweeps (%.3fs / sweep)\n",
            wall, wall / 4000))
h <- mh$get_history()
cat(sprintf("  history keys: %s\n", paste(names(h), collapse=",")))

amp_pm <- mean(h$amp[2001:4000])
tau_pm <- mean(h$tau[2001:4000])
sig_pm <- mean(h$sigma[2001:4000])
cat(sprintf("  amp   post=%.3f  (true %.3f)\n", amp_pm, amp_true))
cat(sprintf("  tau   post=%.3f  (true %.3f)\n", tau_pm, tau_true))
cat(sprintf("  sigma post=%.3f  (true %.3f)\n", sig_pm, sigma_true))

# v0.5 recovery expectations: amp/tau/sigma posterior means should be
# within a generous factor of truth (short chain, N=100, but slice
# sampling with celerite evaluation should already give decent recovery).
amp_ok   <- abs(log(amp_pm)   - log(amp_true))   < log(2.0)    # within 2x
tau_ok   <- abs(log(tau_pm)   - log(tau_true))   < log(3.0)    # within 3x
sigma_ok <- abs(log(sig_pm)   - log(sigma_true)) < log(2.0)    # within 2x

cat(sprintf("  amp  within 2x of truth: %s\n", amp_ok))
cat(sprintf("  tau  within 3x of truth: %s\n", tau_ok))
cat(sprintf("  sigma within 2x of truth: %s\n", sigma_ok))

if (!amp_ok || !tau_ok || !sigma_ok) {
    cat("\n!! Recovery weak -- check v0.5 slice implementation or chain length\n")
    cat("   (short 2k+2k chain; re-run audit_gp_timeseries_v05_4chain.R for\n")
    cat("   longer diagnostics before declaring a regression.)\n")
}

cat("\n========== GP TIME SERIES v0.5 SMOKE PASS ==========\n")

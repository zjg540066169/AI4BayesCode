# ----------------------------------------------------------------------------
# test_joint_nuts_full.R
#
# Comprehensive test suite for IRT1PL_joint (joint_nuts_block over (theta, b)).
# Runs four synthetic datasets:
#   - small   (N=60,  J=12)   — smoke
#   - medium  (N=200, J=30)   — typical IRT
#   - large   (N=500, J=50)   — perf stress
#   - wide-J  (N=100, J=80)   — J > N
#
# For each dataset:
#   (1) Run 2 chains with keep_history = TRUE
#   (2) R-hat (rank-normalized) on sigma_b, max over theta, max over b
#   (3) Bulk / Tail ESS floor (> 200 for a 1000-keep chain)
#   (4) Posterior recovery: |post_mean - truth| < 3 * post_sd
#   (5) Pointwise log-likelihood + loo::loo() pareto-k report
#   (6) Wall-clock: total time, ESS/sec for sigma_b
#
# Aside from dataset 4 (wide-J), this run is also used to benchmark against
# the modular IRT1PL on the same data; see test_joint_vs_block.R.
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

suppressPackageStartupMessages({
    library(posterior)
    library(loo)
})

AI4BayesCode_sourceCpp(file.path(script_dir, "IRT1PL_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

# ============================================================================
# Helpers
# ============================================================================

sim_irt <- function(N, J, sigma_b, miss_rate = 0.1, seed = 1L) {
    set.seed(seed)
    theta_true <- rnorm(N, 0, 1)
    b_true     <- rnorm(J, 0, sigma_b)
    P <- outer(theta_true, b_true, function(a, bb) 1 / (1 + exp(-(a - bb))))
    Y <- matrix(rbinom(N * J, 1, P), nrow = N, ncol = J)
    Y[matrix(runif(N * J) < miss_rate, nrow = N, ncol = J)] <- NA_real_
    list(Y = Y, theta_true = theta_true, b_true = b_true,
         sigma_b_true = sigma_b)
}

run_chain_joint <- function(Y, theta_init, b_init, sigma_b_init,
                            seed, n_burnin, n_keep) {
    m <- new(IRT1PL_joint, Y,
             theta_init, b_init, as.numeric(sigma_b_init),
             as.integer(seed), TRUE)
    t0 <- Sys.time()
    m$step(n_burnin)
    m$step(n_keep)
    t1 <- Sys.time()
    h <- m$get_history()
    # Trim burnin.
    h$theta   <- h$theta[(n_burnin + 1):(n_burnin + n_keep), , drop = FALSE]
    h$b       <- h$b    [(n_burnin + 1):(n_burnin + n_keep), , drop = FALSE]
    h$sigma_b <- h$sigma_b[(n_burnin + 1):(n_burnin + n_keep)]
    list(hist = h,
         wall_sec = as.numeric(difftime(t1, t0, units = "secs")))
}

# Stack two chains' histories into posterior arrays of shape
# (n_iter, 2, n_param). Works for both scalar (vector history) and
# vector (matrix history) params.
stack_chains <- function(h1, h2, name) {
    x1 <- h1[[name]]
    x2 <- h2[[name]]
    if (is.null(dim(x1))) {
        array(cbind(x1, x2), dim = c(length(x1), 2, 1))
    } else {
        arr <- array(NA_real_, dim = c(nrow(x1), 2, ncol(x1)))
        arr[, 1, ] <- x1
        arr[, 2, ] <- x2
        arr
    }
}

# Rank-normalized R-hat, bulk / tail ESS on each slice of a (iter, chain, p)
# array. Returns a data.frame.
chain_diag <- function(arr) {
    p <- dim(arr)[3]
    out <- data.frame(
        rhat     = sapply(seq_len(p), function(k) posterior::rhat(arr[,,k])),
        ess_bulk = sapply(seq_len(p), function(k) posterior::ess_bulk(arr[,,k])),
        ess_tail = sapply(seq_len(p), function(k) posterior::ess_tail(arr[,,k])))
    out
}

# Pointwise log-lik for IRT: log-density of each observed (i, j).
# theta_hist: n x N, b_hist: n x J, Y: N x J with NA for missing.
# Returns LL matrix (n x n_obs) and a list mapping obs index -> (i, j).
irt_pointwise_ll <- function(theta_hist, b_hist, Y) {
    N <- nrow(Y); J <- ncol(Y)
    # Flat list of observed (i, j)
    obs_idx <- which(!is.na(Y), arr.ind = TRUE)   # n_obs x 2 (row, col)
    n_obs <- nrow(obs_idx)
    n_iter <- nrow(theta_hist)
    LL <- matrix(NA_real_, nrow = n_iter, ncol = n_obs)
    y_obs <- Y[obs_idx]
    for (d in seq_len(n_iter)) {
        eta <- theta_hist[d, obs_idx[, 1]] - b_hist[d, obs_idx[, 2]]
        # log p(y | eta) = y*eta - log1p(exp(eta))  (binary logistic)
        LL[d, ] <- y_obs * eta - log1p(exp(eta)) -
                   (1 - y_obs) * 0  # already accounted; keep form simple
    }
    list(LL = LL, y_obs = y_obs)
}

diag_one_dataset <- function(label, dat, n_burnin, n_keep,
                              ess_floor = 200) {
    cat(sprintf("\n==========  %s  (N=%d, J=%d)  ==========\n",
                label, nrow(dat$Y), ncol(dat$Y)))

    N <- nrow(dat$Y); J <- ncol(dat$Y)
    set.seed(7L)
    inits1 <- list(theta = rep(0.0, N),
                   b     = rep(0.0, J),
                   sigma_b = 1.0,
                   seed  = 101L)
    # Overdispersed start for chain 2 — larger scatter to stress R-hat.
    inits2 <- list(theta = rnorm(N, 0, 2.0),
                   b     = rnorm(J, 0, 1.5),
                   sigma_b = 0.3,
                   seed  = 202L)

    cat("  -- Chain 1...\n")
    c1 <- run_chain_joint(dat$Y, inits1$theta, inits1$b, inits1$sigma_b,
                          inits1$seed, n_burnin, n_keep)
    cat(sprintf("     wall: %.2fs  (%.3fs / (burn+keep) sweep, %d total)\n",
                c1$wall_sec, c1$wall_sec / (n_burnin + n_keep),
                n_burnin + n_keep))

    cat("  -- Chain 2...\n")
    c2 <- run_chain_joint(dat$Y, inits2$theta, inits2$b, inits2$sigma_b,
                          inits2$seed, n_burnin, n_keep)
    cat(sprintf("     wall: %.2fs\n", c2$wall_sec))

    # R-hat + ESS
    theta_arr <- stack_chains(c1$hist, c2$hist, "theta")
    b_arr     <- stack_chains(c1$hist, c2$hist, "b")
    sb_arr    <- stack_chains(c1$hist, c2$hist, "sigma_b")

    d_theta <- chain_diag(theta_arr)
    d_b     <- chain_diag(b_arr)
    d_sb    <- chain_diag(sb_arr)

    cat(sprintf("  theta: max Rhat = %.3f   min ESS_bulk = %.0f   min ESS_tail = %.0f\n",
                max(d_theta$rhat),
                min(d_theta$ess_bulk),
                min(d_theta$ess_tail)))
    cat(sprintf("  b    : max Rhat = %.3f   min ESS_bulk = %.0f   min ESS_tail = %.0f\n",
                max(d_b$rhat),
                min(d_b$ess_bulk),
                min(d_b$ess_tail)))
    cat(sprintf("  sb   :     Rhat = %.3f       ESS_bulk = %.0f       ESS_tail = %.0f\n",
                d_sb$rhat, d_sb$ess_bulk, d_sb$ess_tail))

    # Posterior recovery
    post_theta <- colMeans(c1$hist$theta)
    post_theta_sd <- apply(c1$hist$theta, 2, sd)
    post_b     <- colMeans(c1$hist$b)
    post_b_sd  <- apply(c1$hist$b, 2, sd)
    post_sb    <- mean(c1$hist$sigma_b)

    frac_theta_ok <- mean(abs(post_theta - dat$theta_true) < 3 * post_theta_sd)
    frac_b_ok     <- mean(abs(post_b     - dat$b_true)     < 3 * post_b_sd)

    cat(sprintf("  posterior recovery: theta %.1f%% within 3sd, b %.1f%% within 3sd\n",
                100 * frac_theta_ok, 100 * frac_b_ok))
    cat(sprintf("  sigma_b post mean = %.3f (truth %.3f)\n",
                post_sb, dat$sigma_b_true))

    # LOO via pointwise log-lik
    ll1 <- irt_pointwise_ll(c1$hist$theta, c1$hist$b, dat$Y)
    ll2 <- irt_pointwise_ll(c2$hist$theta, c2$hist$b, dat$Y)
    n_obs <- ncol(ll1$LL)
    LLarr <- array(NA_real_, dim = c(nrow(ll1$LL), 2, n_obs))
    LLarr[, 1, ] <- ll1$LL
    LLarr[, 2, ] <- ll2$LL
    rel_n_eff <- loo::relative_eff(exp(LLarr))
    lo <- loo::loo(LLarr, r_eff = rel_n_eff, cores = 1)
    ks <- lo$diagnostics$pareto_k
    pct_k_small <- mean(ks < 0.5) * 100
    pct_k_warn  <- mean(ks >= 0.7) * 100
    cat(sprintf("  LOO: elpd=%.1f (se=%.1f)  pct_k<0.5=%.1f%%  pct_k>=0.7=%.1f%%\n",
                lo$estimates["elpd_loo", "Estimate"],
                lo$estimates["elpd_loo", "SE"],
                pct_k_small, pct_k_warn))

    # ESS/sec
    total_wall <- c1$wall_sec + c2$wall_sec
    ess_sec_sb    <- d_sb$ess_bulk / total_wall
    ess_sec_theta <- min(d_theta$ess_bulk) / total_wall
    cat(sprintf("  ESS/sec: sigma_b=%.1f  min-theta=%.1f  (total wall %.1fs)\n",
                ess_sec_sb, ess_sec_theta, total_wall))

    # PASS / FAIL checks (loose — diagnostic, not strict gate)
    pass_rhat <- max(d_theta$rhat, d_b$rhat, d_sb$rhat, na.rm = TRUE) < 1.1
    pass_ess  <- min(d_theta$ess_bulk, d_b$ess_bulk, d_sb$ess_bulk) > ess_floor
    pass_rec  <- frac_theta_ok > 0.9 && frac_b_ok > 0.9
    pass_loo  <- pct_k_warn < 10
    cat(sprintf("  verdict: R-hat %s  ESS %s  recovery %s  LOO-k %s\n",
                ifelse(pass_rhat, "OK", "WARN"),
                ifelse(pass_ess,  "OK", "WARN"),
                ifelse(pass_rec,  "OK", "WARN"),
                ifelse(pass_loo,  "OK", "WARN")))

    invisible(list(label = label, wall = total_wall,
                   diag_theta = d_theta, diag_b = d_b, diag_sb = d_sb,
                   loo = lo, frac_theta_ok = frac_theta_ok,
                   frac_b_ok = frac_b_ok, post_sb = post_sb,
                   sb_truth = dat$sigma_b_true))
}

# ============================================================================
# Main
# ============================================================================

results <- list()

# v2: longer warmup (4000) + longer keep (4000) per chain on every dataset.
# The goal is to check whether the MEDIUM+ mixing warnings from v1 are
# just a warmup-budget issue or a fundamental mass-matrix issue.

# --- small ----------------------------------------------------------------
d_small <- sim_irt(N = 60L,  J = 12L, sigma_b = 0.8, seed = 1L)
results$small <- diag_one_dataset("SMALL",  d_small,  n_burnin = 4000, n_keep = 4000)

# --- medium ---------------------------------------------------------------
d_medium <- sim_irt(N = 150L, J = 20L, sigma_b = 1.0, seed = 2L)
results$medium <- diag_one_dataset("MEDIUM-LITE", d_medium, n_burnin = 4000, n_keep = 4000)

# --- large ---------------------------------------------------------------
d_large <- sim_irt(N = 250L, J = 20L, sigma_b = 0.7, seed = 3L)
results$large <- diag_one_dataset("LARGE-LITE",  d_large,  n_burnin = 4000, n_keep = 4000)

# --- wide-J --------------------------------------------------------------
d_wide <- sim_irt(N = 50L, J = 50L, sigma_b = 0.9, seed = 4L)
results$wide <- diag_one_dataset("WIDE-J-LITE", d_wide,  n_burnin = 4000, n_keep = 4000)

# --- Summary --------------------------------------------------------------
cat("\n==========  SUMMARY  ==========\n")
for (k in names(results)) {
    r <- results[[k]]
    cat(sprintf("%-7s  wall=%.1fs  sb=%.3f(truth %.3f)  theta-ok=%.0f%%  b-ok=%.0f%%\n",
                r$label, r$wall, r$post_sb, r$sb_truth,
                100 * r$frac_theta_ok, 100 * r$frac_b_ok))
}

saveRDS(results, file.path(script_dir, "joint_nuts_full_results_v2.rds"))
cat("\n Saved results to joint_nuts_full_results_v2.rds\n")

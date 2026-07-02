# ----------------------------------------------------------------------------
# test_joint_vs_block.R
#
# Head-to-head speed / ESS-per-sec benchmark:
#   IRT1PL (modular NUTS per parameter)   vs
#   IRT1PL_joint (joint_nuts_block over theta, b; NUTS on sigma_b)
#
# Uses small and medium data sizes only (modular becomes unbearably slow
# at N=500+, which is precisely why joint exists).
#
# Metrics:
#   - wall-clock per sweep (steady-state, post-warmup)
#   - ESS_bulk / sec on sigma_b  (scalar, easy to compare)
#   - ESS_bulk / sec on the worst-case theta component
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

suppressPackageStartupMessages(library(posterior))

ai4bayescode_sourceCpp(file.path(script_dir, "IRT1PL.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
ai4bayescode_sourceCpp(file.path(script_dir, "IRT1PL_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

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

# modular runner: walks current() each iteration (IRT1PL has no history).
run_block_chain <- function(Y, N, J, seed, n_burnin, n_keep) {
    m <- new(IRT1PL, Y, rep(0.0, N), rep(0.0, J), 1.0, as.integer(seed))
    t0 <- Sys.time()
    m$step(n_burnin)
    # Manually collect.
    theta_hist   <- matrix(NA_real_, nrow = n_keep, ncol = N)
    b_hist       <- matrix(NA_real_, nrow = n_keep, ncol = J)
    sigma_b_hist <- numeric(n_keep)
    for (i in seq_len(n_keep)) {
        m$step(1L)
        d <- m$get_current()
        theta_hist[i, ]   <- d$theta
        b_hist[i, ]       <- d$b
        sigma_b_hist[i]   <- d$sigma_b
    }
    t1 <- Sys.time()
    list(theta = theta_hist, b = b_hist, sigma_b = sigma_b_hist,
         wall_sec = as.numeric(difftime(t1, t0, units = "secs")))
}

run_joint_chain <- function(Y, N, J, seed, n_burnin, n_keep) {
    m <- new(IRT1PL_joint, Y, rep(0.0, N), rep(0.0, J),
             1.0, as.integer(seed), TRUE)
    t0 <- Sys.time()
    m$step(n_burnin)
    m$step(n_keep)
    t1 <- Sys.time()
    h <- m$get_history()
    list(theta   = h$theta  [(n_burnin + 1):(n_burnin + n_keep), , drop = FALSE],
         b       = h$b      [(n_burnin + 1):(n_burnin + n_keep), , drop = FALSE],
         sigma_b = h$sigma_b[(n_burnin + 1):(n_burnin + n_keep)],
         wall_sec = as.numeric(difftime(t1, t0, units = "secs")))
}

pack2 <- function(x1, x2) {
    if (is.null(dim(x1)))
        array(cbind(x1, x2), dim = c(length(x1), 2, 1))
    else {
        arr <- array(NA_real_, dim = c(nrow(x1), 2, ncol(x1)))
        arr[, 1, ] <- x1; arr[, 2, ] <- x2
        arr
    }
}

bench_one <- function(label, dat, n_burnin, n_keep) {
    N <- nrow(dat$Y); J <- ncol(dat$Y)
    cat(sprintf("\n==========  %s  (N=%d, J=%d, burn=%d, keep=%d)  ==========\n",
                label, N, J, n_burnin, n_keep))

    cat("  -- MODULAR chain 1...\n")
    b1 <- run_block_chain(dat$Y, N, J, 101L, n_burnin, n_keep)
    cat(sprintf("     wall %.1fs   %.3fs/sweep\n",
                b1$wall_sec, b1$wall_sec / (n_burnin + n_keep)))
    cat("  -- MODULAR chain 2...\n")
    b2 <- run_block_chain(dat$Y, N, J, 202L, n_burnin, n_keep)
    cat(sprintf("     wall %.1fs\n", b2$wall_sec))

    cat("  -- JOINT       chain 1...\n")
    j1 <- run_joint_chain(dat$Y, N, J, 101L, n_burnin, n_keep)
    cat(sprintf("     wall %.1fs   %.3fs/sweep\n",
                j1$wall_sec, j1$wall_sec / (n_burnin + n_keep)))
    cat("  -- JOINT       chain 2...\n")
    j2 <- run_joint_chain(dat$Y, N, J, 202L, n_burnin, n_keep)
    cat(sprintf("     wall %.1fs\n", j2$wall_sec))

    # ESS / sec comparison
    block_ess_sb <- posterior::ess_bulk(pack2(b1$sigma_b, b2$sigma_b))
    joint_ess_sb <- posterior::ess_bulk(pack2(j1$sigma_b, j2$sigma_b))

    # Worst theta ESS
    theta_block_arr <- pack2(b1$theta, b2$theta)
    theta_joint_arr <- pack2(j1$theta, j2$theta)
    block_ess_theta <- min(sapply(seq_len(dim(theta_block_arr)[3]),
        function(k) posterior::ess_bulk(theta_block_arr[,,k])))
    joint_ess_theta <- min(sapply(seq_len(dim(theta_joint_arr)[3]),
        function(k) posterior::ess_bulk(theta_joint_arr[,,k])))

    tot_block <- b1$wall_sec + b2$wall_sec
    tot_joint <- j1$wall_sec + j2$wall_sec

    cat(sprintf("\n  metric                    | modular   joint       ratio(j/b)\n"))
    cat(sprintf("  wall-clock (s, 2 chains)  | %10.1f  %10.1f  %.2fx\n",
                tot_block, tot_joint, tot_joint / tot_block))
    cat(sprintf("  ESS_bulk sigma_b          | %10.0f  %10.0f\n",
                block_ess_sb, joint_ess_sb))
    cat(sprintf("  ESS_bulk min-theta        | %10.0f  %10.0f\n",
                block_ess_theta, joint_ess_theta))
    cat(sprintf("  ESS/sec sigma_b           | %10.2f  %10.2f  %.1fx\n",
                block_ess_sb / tot_block, joint_ess_sb / tot_joint,
                (joint_ess_sb / tot_joint) / (block_ess_sb / tot_block)))
    cat(sprintf("  ESS/sec min-theta         | %10.2f  %10.2f  %.1fx\n",
                block_ess_theta / tot_block, joint_ess_theta / tot_joint,
                (joint_ess_theta / tot_joint) / (block_ess_theta / tot_block)))
}

# --- tiny: N=30, J=8 ---
d_tiny <- sim_irt(30L, 8L, 0.8, seed = 1L)
bench_one("TINY",    d_tiny,   n_burnin = 200, n_keep = 500)

# --- small: N=60, J=12 ---
d_small <- sim_irt(60L, 12L, 0.8, seed = 1L)
bench_one("SMALL",   d_small,  n_burnin = 300, n_keep = 500)

# --- medium: N=200, J=30 --- BLOCKWISE WILL BE SLOW (~15 min at worst)
# Uncomment only if you have time.
# d_medium <- sim_irt(200L, 30L, 1.0, seed = 2L)
# bench_one("MEDIUM", d_medium, n_burnin = 300, n_keep = 300)

cat("\nDONE.\n")

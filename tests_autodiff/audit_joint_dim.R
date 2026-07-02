# Phase F3 of audit — joint NUTS dim stress. Probes the documented
# mass-matrix ceiling (~dim 70 in well-conditioned problems).
# IRT1PL_joint at (N=30,J=8) dim=38, (N=60,J=12) dim=72,
#                (N=100,J=15) dim=115.
# HierarchicalLM_joint at G=10 dim=14, G=20 dim=24, G=50 dim=54.
#
# Logs to audit_joint_dim.log; rds to audit_xl_F3.rds.

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

pack2 <- function(x1, x2) {
    if (is.null(dim(x1)))
        array(cbind(x1, x2), dim = c(length(x1), 2, 1))
    else {
        arr <- array(NA_real_, dim = c(nrow(x1), 2, ncol(x1)))
        arr[,1,] <- x1; arr[,2,] <- x2
        arr
    }
}
diag_arr <- function(arr) {
    p <- dim(arr)[3]
    list(
        rhat = sapply(seq_len(p), function(k) posterior::rhat(arr[,,k])),
        ess  = sapply(seq_len(p), function(k) posterior::ess_bulk(arr[,,k])))
}

N_BURN <- 10000L
N_KEEP <- 10000L

# ===== IRT1PL_joint dim stress =====
cat("\n[F3] IRT1PL_joint dim stress\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "IRT1PL_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

irt_cases <- list(
    list(N=30L,  J=8L,  label="N=30/J=8 (dim 38)"),
    list(N=60L,  J=12L, label="N=60/J=12 (dim 72)"),
    list(N=100L, J=15L, label="N=100/J=15 (dim 115)"))
irt_results <- list()
for (cs in irt_cases) {
    cat(sprintf("\n  [IRT] %s: burning 10k+10k × 2 chains...\n", cs$label))
    set.seed(1)
    theta_t <- rnorm(cs$N, 0, 1); sb_t <- 0.8
    b_t <- rnorm(cs$J, 0, sb_t)
    P <- outer(theta_t, b_t, function(a, bb) 1/(1+exp(-(a-bb))))
    Y <- matrix(rbinom(cs$N*cs$J, 1, P), cs$N, cs$J)
    Y[matrix(runif(cs$N*cs$J) < 0.1, cs$N, cs$J)] <- NA_real_
    run <- function(seed, th0, b0, sb0) {
        m <- new(IRT1PL_joint, Y, th0, b0, as.numeric(sb0), as.integer(seed), TRUE)
        t0 <- Sys.time(); m$step(N_BURN); m$step(N_KEEP); t1 <- Sys.time()
        h <- m$get_history()
        list(theta   = h$theta[(N_BURN+1):(N_BURN+N_KEEP), , drop=FALSE],
             b       = h$b[(N_BURN+1):(N_BURN+N_KEEP), , drop=FALSE],
             sigma_b = h$sigma_b[(N_BURN+1):(N_BURN+N_KEEP)],
             wall    = as.numeric(difftime(t1, t0, units="secs")))
    }
    c1 <- run(101L, rep(0, cs$N), rep(0, cs$J), 1.0)
    c2 <- run(202L, rnorm(cs$N, 0, 0.5), rnorm(cs$J, 0, 0.5), 0.5)
    d_theta <- diag_arr(pack2(c1$theta, c2$theta))
    d_b     <- diag_arr(pack2(c1$b, c2$b))
    d_sb    <- diag_arr(pack2(c1$sigma_b, c2$sigma_b))
    rhat_th_max <- max(d_theta$rhat, na.rm=TRUE)
    rhat_b_max  <- max(d_b$rhat, na.rm=TRUE)
    rhat_sb     <- max(d_sb$rhat, na.rm=TRUE)
    # Per-chain recovery
    pm_th_c1 <- colMeans(c1$theta); pm_th_c2 <- colMeans(c2$theta)
    in_c1 <- mean(abs(pm_th_c1 - theta_t) < 3 * apply(c1$theta, 2, sd))
    in_c2 <- mean(abs(pm_th_c2 - theta_t) < 3 * apply(c2$theta, 2, sd))
    wall <- c1$wall + c2$wall
    cat(sprintf("    wall=%.1fs rhat_theta_max=%.3f rhat_b_max=%.3f rhat_sb=%.3f per-chain recovery theta: c1=%.0f%% c2=%.0f%%\n",
        wall, rhat_th_max, rhat_b_max, rhat_sb, 100*in_c1, 100*in_c2))
    irt_results[[cs$label]] <- list(
        N=cs$N, J=cs$J, dim=cs$N+cs$J,
        rhat_theta_max=rhat_th_max, rhat_b_max=rhat_b_max, rhat_sb=rhat_sb,
        per_chain_rec = c(c1=in_c1, c2=in_c2),
        wall = wall)
}

# ===== HierarchicalLM_joint dim stress =====
cat("\n\n[F3] HierarchicalLM_joint dim stress\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "HierarchicalLM_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

hlm_cases <- list(
    list(G=10L, Np=15L, label="G=10 (dim 14)"),
    list(G=20L, Np=15L, label="G=20 (dim 24)"),
    list(G=50L, Np=15L, label="G=50 (dim 54)"))
hlm_results <- list()
for (cs in hlm_cases) {
    cat(sprintf("\n  [HierLM] %s: burning 10k+10k × 2 chains...\n", cs$label))
    set.seed(1)
    N <- cs$G * cs$Np; p <- 3L
    alpha <- 1.0; beta <- c(2, -1, 0.5); sigma <- 0.8; tau <- 1.5
    X <- matrix(rnorm(N*p), N, p); g_idx <- rep(seq_len(cs$G), each=cs$Np)
    u <- rnorm(cs$G, 0, tau)
    y <- as.numeric(alpha + X %*% beta + u[g_idx] + rnorm(N, 0, sigma))
    run <- function(seed) {
        m <- new(HierarchicalLM_joint, y, X, as.integer(g_idx), cs$G,
                 1.0, 1.0, as.integer(seed), TRUE)
        t0 <- Sys.time(); m$step(N_BURN); m$step(N_KEEP); t1 <- Sys.time()
        h <- m$get_history()
        list(alpha=h$alpha[(N_BURN+1):(N_BURN+N_KEEP)],
             beta =h$beta[(N_BURN+1):(N_BURN+N_KEEP), , drop=FALSE],
             u    =h$u[(N_BURN+1):(N_BURN+N_KEEP), , drop=FALSE],
             sigma=h$sigma[(N_BURN+1):(N_BURN+N_KEEP)],
             tau  =h$tau[(N_BURN+1):(N_BURN+N_KEEP)],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    }
    c1 <- run(101L); c2 <- run(202L)
    d_a <- diag_arr(pack2(c1$alpha, c2$alpha))
    d_b <- diag_arr(pack2(c1$beta, c2$beta))
    d_u <- diag_arr(pack2(c1$u, c2$u))
    d_s <- diag_arr(pack2(c1$sigma, c2$sigma))
    d_t <- diag_arr(pack2(c1$tau, c2$tau))
    rhat_max <- max(d_a$rhat, d_b$rhat, d_u$rhat, d_s$rhat, d_t$rhat)
    wall <- c1$wall + c2$wall
    cat(sprintf("    wall=%.1fs  rhat_max=%.3f  a=%.3f b=%.3f u=%.3f s=%.3f t=%.3f\n",
        wall, rhat_max, max(d_a$rhat), max(d_b$rhat), max(d_u$rhat),
        max(d_s$rhat), max(d_t$rhat)))
    hlm_results[[cs$label]] <- list(
        G=cs$G, dim=1 + p + cs$G,
        rhat_max=rhat_max,
        rhat_alpha=max(d_a$rhat), rhat_beta=max(d_b$rhat), rhat_u=max(d_u$rhat),
        rhat_sigma=max(d_s$rhat), rhat_tau=max(d_t$rhat),
        wall = wall)
}

results_F3 <- list(IRT=irt_results, HierLM=hlm_results)
saveRDS(results_F3, file.path(AI4BayesCode_dir, "audit_xl_F3.rds"))
cat("\n[F3 DONE] saved to audit_xl_F3.rds\n")

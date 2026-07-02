# ---------------------------------------------------------------------------
# test_hierarchical_lm_joint.R
#
# Test HierarchicalLM_joint on a simulated linear mixed model dataset.
#   - Fit with 2 chains, keep_history = TRUE
#   - R-hat (rank-normalized) and bulk/tail ESS
#   - Posterior recovery of (alpha, beta, u, sigma, tau)
# ---------------------------------------------------------------------------

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

ai4bayescode_sourceCpp(file.path(script_dir, "HierarchicalLM_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

# --- Simulate --------------------------------------------------------------
set.seed(1L)
G <- 10L
N_per_group <- 15L
N <- G * N_per_group
p <- 3L
alpha_true <- 1.0
beta_true  <- c(2.0, -1.0, 0.5)
sigma_true <- 0.8
tau_true   <- 1.5

X <- matrix(rnorm(N * p), nrow = N, ncol = p)
g_idx <- rep(seq_len(G), each = N_per_group)
u_true <- rnorm(G, 0, tau_true)
y <- alpha_true + X %*% beta_true + u_true[g_idx] + rnorm(N, 0, sigma_true)
y <- as.numeric(y)

run_chain <- function(seed, n_burnin, n_keep) {
    m <- new(HierarchicalLM_joint,
             y, X, as.integer(g_idx), G,
             1.0, 1.0, as.integer(seed), TRUE)
    t0 <- Sys.time()
    m$step(n_burnin)
    m$step(n_keep)
    t1 <- Sys.time()
    h <- m$get_history()
    list(
        alpha = h$alpha[(n_burnin + 1):(n_burnin + n_keep)],
        beta  = h$beta [(n_burnin + 1):(n_burnin + n_keep), , drop = FALSE],
        u     = h$u    [(n_burnin + 1):(n_burnin + n_keep), , drop = FALSE],
        sigma = h$sigma[(n_burnin + 1):(n_burnin + n_keep)],
        tau   = h$tau  [(n_burnin + 1):(n_burnin + n_keep)],
        wall_sec = as.numeric(difftime(t1, t0, units = "secs")))
}

cat("Running 2 chains (N =", N, ", p =", p, ", G =", G, ")\n")
c1 <- run_chain(101L, 500L, 1000L)
cat(sprintf("  chain1 wall: %.2fs\n", c1$wall_sec))
c2 <- run_chain(202L, 500L, 1000L)
cat(sprintf("  chain2 wall: %.2fs\n", c2$wall_sec))

pack2 <- function(x1, x2) {
    if (is.null(dim(x1))) array(cbind(x1, x2), dim = c(length(x1), 2, 1))
    else {
        arr <- array(NA_real_, dim = c(nrow(x1), 2, ncol(x1)))
        arr[, 1, ] <- x1; arr[, 2, ] <- x2
        arr
    }
}

report <- function(name, arr, truth = NULL) {
    p <- dim(arr)[3]
    rh <- sapply(seq_len(p), function(k) posterior::rhat(arr[,,k]))
    eb <- sapply(seq_len(p), function(k) posterior::ess_bulk(arr[,,k]))
    et <- sapply(seq_len(p), function(k) posterior::ess_tail(arr[,,k]))
    cat(sprintf("  %-8s  max Rhat=%.3f  min ESS_bulk=%.0f  min ESS_tail=%.0f\n",
                name, max(rh), min(eb), min(et)))
    if (!is.null(truth)) {
        post_mean <- sapply(seq_len(p), function(k) mean(arr[, 1, k]))
        post_sd   <- sapply(seq_len(p), function(k) sd(arr[, 1, k]))
        within_3sd <- abs(post_mean - truth) < 3 * post_sd
        cat(sprintf("            %s/%s within 3sd of truth\n",
                    sum(within_3sd), length(truth)))
    }
}

cat("\n=== Diagnostics ===\n")
report("alpha", pack2(c1$alpha, c2$alpha), truth = alpha_true)
report("beta",  pack2(c1$beta,  c2$beta),  truth = beta_true)
report("u",     pack2(c1$u,     c2$u),     truth = u_true)
report("sigma", pack2(c1$sigma, c2$sigma), truth = sigma_true)
report("tau",   pack2(c1$tau,   c2$tau),   truth = tau_true)

cat("\n=== Posterior means vs truth ===\n")
cat(sprintf("  alpha: %.3f  (truth %.3f)\n",   mean(c1$alpha), alpha_true))
cat(sprintf("  beta : %s  (truth %s)\n",
            paste(sprintf("%.3f", colMeans(c1$beta)), collapse = " "),
            paste(sprintf("%.3f", beta_true),         collapse = " ")))
cat(sprintf("  sigma: %.3f  (truth %.3f)\n",   mean(c1$sigma), sigma_true))
cat(sprintf("  tau:   %.3f  (truth %.3f)\n",   mean(c1$tau),   tau_true))
cat(sprintf("  u[1-3]: %s  (truth %s)\n",
            paste(sprintf("%.3f", colMeans(c1$u)[1:3]), collapse = " "),
            paste(sprintf("%.3f", u_true[1:3]),         collapse = " ")))

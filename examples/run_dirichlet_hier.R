# ----------------------------------------------------------------------------
# run_dirichlet_hier.R
#
# Hierarchical Dirichlet model fit with NUTS on (s, kappa, theta), exercising
# four scenarios:
#
#       (P = 20,  sparse s)
#       (P = 20,  non-sparse s)
#       (P = 50,  sparse s)
#       (P = 50,  non-sparse s)
#
# Model
# -----
#   s_k | s, kappa          ~ Dirichlet(kappa * s),  k = 1..K
#   s                       ~ Dirichlet(theta/P, ..., theta/P)
#   kappa                   ~ Gamma(1, 1)
#   theta / (theta + P)     ~ Beta(beta_a, beta_b)
#
# We use Beta(2, 5) here instead of the more permissive Beta(0.5, 1) that
# appeared in the single-level example. Beta(2, 5) has density zero at both
# endpoints so it gives theta a finite-mode, thin-tailed prior. This is
# important for this hierarchical model: under a heavy right-tailed prior,
# theta can drift to infinity (uniform s) and lock the sampler in a
# degenerate mode during warmup.
#
# Diagnostics
# -----------
#   Per scenario, we run 2 overdispersed chains with 5000 burnin + 5000 keep
#   and report per-parameter rank R-hat via the `posterior` package, falling
#   back to a classical split-R-hat if it is missing. Posterior recovery of
#   (kappa, theta, s) is printed against the true generative values.
# ----------------------------------------------------------------------------

# Locate this script's directory portably (works under Rscript and under
# source()). Under Rscript, --file= is in commandArgs; under source(), fall
# back to getwd() (user must be cd'd into AI4BayesCode/examples/).
script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))

source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

ai4bayescode_sourceCpp(
    cpp_file       = file.path(script_dir, "DirichletHierarchical.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

# ============================================================================
#  Utilities
# ============================================================================

# Robust rDirichlet — floors underflowed gamma draws so log(s) stays finite.
rdirichlet <- function(n, alpha) {
    k <- length(alpha)
    x <- matrix(rgamma(n * k, shape = alpha, rate = 1), nrow = n, byrow = TRUE)
    x[x < 1e-300] <- 1e-300
    x / rowSums(x)
}

# Classical split R-hat fallback.
classical_split_rhat <- function(chains_mat) {
    n_full <- nrow(chains_mat)
    M_in   <- ncol(chains_mat)
    n_half <- n_full %/% 2L
    if (n_half < 2) return(NA_real_)
    halves <- matrix(NA_real_, nrow = n_half, ncol = 2 * M_in)
    for (m in seq_len(M_in)) {
        halves[, 2 * m - 1] <- chains_mat[seq_len(n_half), m]
        halves[, 2 * m]     <- chains_mat[seq_len(n_half) + n_half, m]
    }
    chain_means <- colMeans(halves)
    chain_vars  <- apply(halves, 2, var)
    gm          <- mean(chain_means)
    B_over_n    <- sum((chain_means - gm)^2) / (ncol(halves) - 1)
    W           <- mean(chain_vars)
    if (W <= 0) return(NA_real_)
    V_hat <- (n_half - 1) / n_half * W + B_over_n
    sqrt(V_hat / W)
}

rhat_of <- function(draws_by_chain) {
    # draws_by_chain: matrix, rows = iterations, cols = chains
    if (requireNamespace("posterior", quietly = TRUE)) {
        # Build iters x chains array
        arr <- array(draws_by_chain,
                     dim = c(nrow(draws_by_chain), ncol(draws_by_chain), 1))
        as.numeric(posterior::rhat(arr[, , 1, drop = TRUE]))
    } else {
        classical_split_rhat(draws_by_chain)
    }
}

BETA_A <- 2.0
BETA_B <- 5.0

run_chain <- function(S_obs, seed, s_jitter_alpha,
                      kappa_init, theta_init,
                      n_burnin, n_keep) {
    # Initial (s, kappa, theta) are chosen internally from the data +
    # Beta hyperpriors. Scenario args `s_jitter_alpha`, `kappa_init`,
    # `theta_init` (if supplied by the caller) are applied AFTER
    # construction via set_current — this is how one gets overdispersed
    # starts for R-hat diagnostics.
    model <- new(DirichletHierarchical,
                 S_obs, BETA_A, BETA_B, as.integer(seed))
    overrides <- list()
    if (!is.null(kappa_init)) overrides$kappa <- kappa_init
    if (!is.null(theta_init)) overrides$theta <- theta_init
    if (length(overrides) > 0L) model$set_current(overrides)
    model$step(n_burnin)
    P <- ncol(S_obs)
    s_draws     <- matrix(NA_real_, nrow = n_keep, ncol = P)
    kappa_draws <- numeric(n_keep)
    theta_draws <- numeric(n_keep)
    for (it in seq_len(n_keep)) {
        model$step(1L)
        d <- model$get_current()
        s_draws[it, ]   <- d$s
        kappa_draws[it] <- d$kappa
        theta_draws[it] <- d$theta
    }
    list(s = s_draws, kappa = kappa_draws, theta = theta_draws)
}

# ============================================================================
#  Scenario generator
# ============================================================================

make_scenario <- function(P, kind, K = 200L, kappa_true = 15.0, seed = 42L) {
    set.seed(seed)
    # alpha_true = theta_true / P.  "sparse" means concentration < 1 (near
    # simplex vertices); "non_sparse" means concentration > 1 (near centroid).
    if (kind == "sparse") {
        alpha_true <- 0.1                     # theta_true = 0.1 * P
    } else if (kind == "non_sparse") {
        alpha_true <- 5.0                     # theta_true = 5 * P
    } else {
        stop("unknown kind")
    }
    theta_true <- alpha_true * P

    s_true <- rdirichlet(1, rep(alpha_true, P))[1, ]
    s_true <- pmax(s_true, 1e-6)
    s_true <- s_true / sum(s_true)

    S_obs  <- rdirichlet(K, kappa_true * s_true)

    list(P = P, K = K, kind = kind,
         theta_true = theta_true, kappa_true = kappa_true,
         s_true = s_true, S_obs = S_obs)
}

# ============================================================================
#  Per-scenario runner
# ============================================================================

run_scenario <- function(scen, n_burnin, n_keep) {
    P  <- scen$P
    cat(sprintf("\n==================================================================\n"))
    cat(sprintf(" P = %d,  kind = %s,  theta_true = %.2f,  kappa_true = %.2f\n",
                P, scen$kind, scen$theta_true, scen$kappa_true))
    cat(sprintf(" K = %d,  burnin = %d,  keep = %d,  2 chains\n",
                scen$K, n_burnin, n_keep))
    cat(sprintf("==================================================================\n"))

    # Both chains start from a data-driven region but with mild
    # overdispersion on (kappa, theta, s jitter). Initializing too far from
    # the posterior typical set (e.g. near-uniform s with tiny theta) can
    # drive theta into its degenerate right tail during warmup and lock
    # the chain there.
    t0 <- Sys.time()
    chain1 <- run_chain(scen$S_obs, seed = 1001L + P,
                        s_jitter_alpha = 0.01,
                        kappa_init = 5.0,   theta_init = P,
                        n_burnin = n_burnin, n_keep = n_keep)
    chain2 <- run_chain(scen$S_obs, seed = 2002L + P,
                        s_jitter_alpha = 0.10,
                        kappa_init = 30.0,  theta_init = 3 * P,
                        n_burnin = n_burnin, n_keep = n_keep)
    t1 <- Sys.time()
    cat(sprintf("  sampling (2 chains): %.1fs\n",
                as.numeric(difftime(t1, t0, units = "secs"))))

    # ---- R-hat ----
    kappa_mat <- cbind(chain1$kappa, chain2$kappa)
    theta_mat <- cbind(chain1$theta, chain2$theta)
    rhat_kappa <- rhat_of(kappa_mat)
    rhat_theta <- rhat_of(theta_mat)

    s_rhat <- numeric(P)
    for (i in seq_len(P)) {
        s_rhat[i] <- rhat_of(cbind(chain1$s[, i], chain2$s[, i]))
    }

    cat(sprintf("\n  R-hat summary:\n"))
    cat(sprintf("    kappa  R-hat = %.4f\n", rhat_kappa))
    cat(sprintf("    theta  R-hat = %.4f\n", rhat_theta))
    cat(sprintf("    s      R-hat: min = %.4f, median = %.4f, max = %.4f\n",
                min(s_rhat, na.rm = TRUE),
                median(s_rhat, na.rm = TRUE),
                max(s_rhat, na.rm = TRUE)))

    # ---- Posterior recovery (pool both chains) ----
    s_all     <- rbind(chain1$s, chain2$s)
    kappa_all <- c(chain1$kappa, chain2$kappa)
    theta_all <- c(chain1$theta, chain2$theta)

    cat(sprintf("\n  kappa posterior: mean = %.3f, sd = %.3f, 90%% CI [%.3f, %.3f]  (true %.2f)\n",
                mean(kappa_all), sd(kappa_all),
                quantile(kappa_all, 0.05),
                quantile(kappa_all, 0.95),
                scen$kappa_true))
    cat(sprintf("  theta posterior: mean = %.3f, sd = %.3f, 90%% CI [%.3f, %.3f]  (true %.2f)\n",
                mean(theta_all), sd(theta_all),
                quantile(theta_all, 0.05),
                quantile(theta_all, 0.95),
                scen$theta_true))

    s_post_mean <- colMeans(s_all)
    max_sdiff   <- max(abs(s_post_mean - scen$s_true))
    mean_sdiff  <- mean(abs(s_post_mean - scen$s_true))
    cat(sprintf("  s posterior vs true: mean |diff| = %.2e, max |diff| = %.2e\n",
                mean_sdiff, max_sdiff))

    # Top 6 active components
    top6 <- order(-scen$s_true)[1:6]
    cat(sprintf("  top 6 components (ordered by true s):\n"))
    cat(sprintf("    %6s %10s %10s %10s\n", "k", "true", "post_mean", "post_sd"))
    for (i in top6) {
        cat(sprintf("    %6d %10.5f %10.5f %10.5f\n",
                    i, scen$s_true[i], s_post_mean[i], sd(s_all[, i])))
    }

    # pass criteria: R-hat < 1.1 (loose, since simplex NUTS has autocorr)
    ok <- rhat_kappa < 1.1 && rhat_theta < 1.1 &&
          max(s_rhat, na.rm = TRUE) < 1.1

    cat(sprintf("\n  scenario %s\n",
                if (ok) "PASS (all R-hat < 1.1)" else "WARN (some R-hat >= 1.1)"))

    list(scen = scen,
         rhat_kappa = rhat_kappa, rhat_theta = rhat_theta,
         s_rhat_max = max(s_rhat, na.rm = TRUE),
         ok = ok)
}

# ============================================================================
#  Run all four scenarios
# ============================================================================

n_burnin <- 5000L
n_keep   <- 5000L

scenarios <- list(
    make_scenario(20L, "sparse",     K = 200L, seed = 101L),
    make_scenario(20L, "non_sparse", K = 200L, seed = 102L),
    make_scenario(50L, "sparse",     K = 200L, seed = 103L),
    make_scenario(50L, "non_sparse", K = 200L, seed = 104L))

results <- lapply(scenarios, run_scenario,
                  n_burnin = n_burnin, n_keep = n_keep)

# ---- Final summary table --------------------------------------------------
cat("\n==================================================================\n")
cat(" Overall summary\n")
cat("==================================================================\n")
cat(sprintf("%-20s %-12s %-12s %-14s %-8s\n",
            "scenario", "rhat_kappa", "rhat_theta", "max_rhat_s", "status"))
for (r in results) {
    label <- sprintf("P=%d %s", r$scen$P, r$scen$kind)
    cat(sprintf("%-20s %-12.4f %-12.4f %-14.4f %s\n",
                label, r$rhat_kappa, r$rhat_theta, r$s_rhat_max,
                if (r$ok) "PASS" else "WARN"))
}

overall_ok <- all(vapply(results, function(x) x$ok, logical(1)))
cat(sprintf("\n%s\n", if (overall_ok) "ALL SCENARIOS PASS" else "SOME SCENARIOS WARN"))

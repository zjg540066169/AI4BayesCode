# ----------------------------------------------------------------------------
# run_dirichlet.R
#
# End-to-end demo of DirichletSimplex.cpp. Runs 4 independent chains from
# overdispersed initial points and reports Vehtari et al. 2021 rank R-hat
# per component via the `posterior` package if it is installed, falling
# back to a minimal classical split-R-hat if it is not.
#
# This script is also the canonical pattern that the AI4BayesCode SKILL
# (section 12) tells the AI generator to follow when it emits a runner
# alongside a generated sampler class. Any user-facing sampler should
# be checked with multi-chain R-hat before its draws are used for
# inference.
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

AI4BayesCode_sourceCpp(
    cpp_file       = file.path(script_dir, "DirichletSimplex.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

# ---- Synthesize data -------------------------------------------------------
set.seed(20260410)
K           <- 5L
theta_true  <- c(0.10, 0.25, 0.35, 0.20, 0.10)
N           <- 500L
alpha_prior <- rep(1.0, K)

# Multinomial counts
y_counts <- as.numeric(table(
    factor(sample.int(K, N, prob = theta_true, replace = TRUE),
           levels = seq_len(K))))
cat(sprintf("Generated N = %d counts across K = %d categories:\n", N, K))
cat("  y_counts =", paste(y_counts, collapse = " "), "\n")
cat("  theta*   =", sprintf("%.4f", theta_true), "\n\n")

# Analytic conjugate posterior: Dirichlet(alpha + y_counts)
alpha_post  <- alpha_prior + y_counts
S           <- sum(alpha_post)
mean_true   <- alpha_post / S
sd_true     <- sqrt(mean_true * (1 - mean_true) / (S + 1))

# ---- Four chains from overdispersed "corner" initializations ---------------
corner_init <- function(K, heavy_k, heavy_weight = 0.70) {
    light <- (1.0 - heavy_weight) / (K - 1)
    theta <- rep(light, K)
    theta[heavy_k] <- heavy_weight
    theta
}

n_chains  <- 4L
n_burnin  <- 500L
n_keep    <- 2000L

chain_draws <- vector("list", n_chains)
for (m in seq_len(n_chains)) {
    init <- corner_init(K, heavy_k = m)
    cat(sprintf("Chain %d init: %s\n", m,
                paste(sprintf("%.3f", init), collapse = " ")))

    model <- new(DirichletSimplex, y_counts, alpha_prior,
                 as.integer(20260410L + 1000L * m))

    # Seed the chain to its overdispersed starting point.
    model$set_current(list(theta = init))

    model$step(n_burnin)

    draws <- matrix(NA_real_, nrow = n_keep, ncol = K)
    for (s in seq_len(n_keep)) {
        model$step(1L)
        draws[s, ] <- model$get_current()$theta
    }
    chain_draws[[m]] <- draws
}
cat("\n")

# ---- Per-chain summaries ---------------------------------------------------
per_chain_means <- sapply(chain_draws, colMeans)  # K x n_chains
colnames(per_chain_means) <- sprintf("chain%d", seq_len(n_chains))

cat("Per-chain posterior means (component x chain):\n")
summary_mat <- cbind(per_chain_means, analytic = mean_true)
rownames(summary_mat) <- sprintf("theta[%d]", seq_len(K))
print(round(summary_mat, 4))
cat("\n")

# ---- Rank R-hat via posterior, or classical fallback ----------------------

# The posterior package expects draws in an (iterations x chains x vars)
# array. Build that from our per-chain matrices.
draws_array <- array(NA_real_, dim = c(n_keep, n_chains, K))
for (m in seq_len(n_chains)) {
    draws_array[, m, ] <- chain_draws[[m]]
}
dimnames(draws_array) <- list(
    iteration = NULL,
    chain     = sprintf("chain%d", seq_len(n_chains)),
    variable  = sprintf("theta[%d]", seq_len(K)))

rhat_vec <- tryCatch({
    if (requireNamespace("posterior", quietly = TRUE)) {
        cat("Using posterior::rhat (rank-normalized split-R-hat).\n")
        apply(draws_array, 3, posterior::rhat)
    } else {
        stop("posterior not installed")
    }
}, error = function(e) {
    cat("NOTE: `posterior` package not available; using a minimal\n")
    cat("      classical split-R-hat implementation instead. For the\n")
    cat("      rank-normalized variant (Vehtari et al. 2021), install\n")
    cat("      the posterior package: install.packages('posterior').\n")
    classical_split_rhat <- function(chains_mat) {
        # chains_mat: iters x chains (single variable)
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
    vapply(seq_len(K), function(k) {
        classical_split_rhat(t(sapply(chain_draws, function(d) d[, k])))
    }, numeric(1))
})

cat("\nR-hat per component:\n")
print(round(rhat_vec, 4))

max_rhat <- max(rhat_vec, na.rm = TRUE)
cat(sprintf("\nmax R-hat = %.4f  (< 1.05 is the conventional pass threshold)\n",
            max_rhat))

# ---- Pooled summaries ------------------------------------------------------
pooled <- do.call(rbind, chain_draws)
cat("\nPooled posterior summary:\n")
cat("  mean =", sprintf("%.4f", colMeans(pooled)),  "\n")
cat("  sd   =", sprintf("%.4f", apply(pooled, 2, sd)),  "\n")
cat("  analytic mean =", sprintf("%.4f", mean_true), "\n")
cat("  analytic sd   =", sprintf("%.4f", sd_true),   "\n")

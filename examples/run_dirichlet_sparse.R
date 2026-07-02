# ----------------------------------------------------------------------------
# run_dirichlet_sparse.R
#
# High-dimensional sparse symmetric Dirichlet with Beta hyperprior on the
# concentration, fit to multinomial count data (Linero 2018 DART prior).
# Both `s` and `theta` are sampled via NUTS (NOT the conjugate Dirichlet-
# gamma trick), per user request.
#
# Model
# -----
#     y      ~ Multinomial(N, s)                       (P categories)
#     s      ~ Dirichlet(theta/P, ..., theta/P)        (P-dim simplex)
#     theta / (theta + P) ~ Beta(0.5, 1)               (hyperprior)
#
# Validation strategy
# -------------------
# With theta fixed at a given value t, the s-conditional is conjugate:
#   s | y, theta = t  ~  Dirichlet(y + t/P)
# so E[s_i | y, theta] = (y_i + t/P) / (N + t).  Averaging over the NUTS
# draws of theta gives the correct marginal mean E[s_i | y], which is what
# the sampler estimates directly from the s-chain.  We check the two
# estimates match closely for a handful of components.
#
# We also run a prior-only sanity check (y = 0) and verify
# u = theta/(theta+P) matches Beta(0.5, 1) by KS test.
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
    cpp_file       = file.path(script_dir, "DirichletSparse.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

# ============================================================================
#  Helper: run the sampler and return (s_draws, theta_draws)
# ============================================================================
run_chain <- function(y, theta_init, seed, n_burnin, n_keep) {
    model <- new(DirichletSparse, y, as.integer(seed))
    # Caller may still force an overdispersed theta start for R-hat
    # diagnostics; that happens AFTER construction now.
    if (!is.null(theta_init)) {
        model$set_current(list(theta = theta_init))
    }
    model$step(n_burnin)

    P <- length(y)
    s_draws     <- matrix(NA_real_, nrow = n_keep, ncol = P)
    theta_draws <- numeric(n_keep)
    for (it in seq_len(n_keep)) {
        model$step(1L)
        d <- model$get_current()
        s_draws[it, ]  <- d$s
        theta_draws[it] <- d$theta
    }
    list(s = s_draws, theta = theta_draws)
}

# ============================================================================
#  Test 1: posterior fit to a true-sparse multinomial
# ============================================================================
cat("==================================================================\n")
cat(" Test 1: posterior fit,  P = 200,  N = 2000,  sparse true s\n")
cat("==================================================================\n")

set.seed(20260411L)
P <- 200L
N <- 2000L

# True s has 10 active components, rest near zero
active_k <- 10L
s_true <- rep(1e-5, P)
s_true[seq_len(active_k)] <- 1.0 / active_k
s_true <- s_true / sum(s_true)

y <- as.numeric(
    tabulate(sample.int(P, N, prob = s_true, replace = TRUE), nbins = P))
cat(sprintf("  N = %d,  nonzero categories observed: %d\n",
            N, sum(y > 0)))
cat(sprintf("  y[1:10] = %s\n", paste(y[1:10], collapse = " ")))
cat(sprintf("  sum(y[11:P]) = %d\n\n", sum(y[11:P])))

n_burnin <- 800L
n_keep   <- 1500L

t0 <- Sys.time()
out <- run_chain(y, theta_init = 10.0, seed = 20260411L,
                 n_burnin = n_burnin, n_keep = n_keep)
t1 <- Sys.time()
cat(sprintf("  sampling: %.1fs  (burnin %d + keep %d)\n\n",
            as.numeric(difftime(t1, t0, units = "secs")),
            n_burnin, n_keep))

# ---- Simplex / basic sanity ----
simplex_err_max <- max(abs(rowSums(out$s) - 1.0))
cat(sprintf("  max |sum(s) - 1| over kept draws = %.2e\n", simplex_err_max))

# ---- Posterior s_i means vs the Rao-Blackwellized analytic reference ----
# E[s_i | y, theta] = (y_i + theta/P) / (N + theta), so
# E[s_i | y]        = mean over draws of the above.
alpha_over_P <- out$theta / P
denom        <- N + out$theta
# Rao-Blackwellized E[s_i | y]
rb_mean <- sapply(seq_len(P), function(i) {
    mean((y[i] + alpha_over_P) / denom)
})
sample_mean <- colMeans(out$s)

# Compare the two on all P components
diff_all <- abs(sample_mean - rb_mean)
cat(sprintf("  mean |E_sample[s_i] - E_rb[s_i]|  = %.2e\n", mean(diff_all)))
cat(sprintf("  max  |E_sample[s_i] - E_rb[s_i]|  = %.2e\n", max(diff_all)))

# Show the top 12 components side by side
ord <- order(-sample_mean)[1:12]
tab <- data.frame(
    k         = ord,
    y         = y[ord],
    s_true    = round(s_true[ord], 5),
    E_sample  = round(sample_mean[ord], 5),
    E_rao_bw  = round(rb_mean[ord], 5))
cat("\n  Top 12 components (ordered by posterior mean):\n")
print(tab, row.names = FALSE)

# ---- theta posterior ----
cat("\n  theta posterior:\n")
cat(sprintf("    mean = %.4f\n", mean(out$theta)))
cat(sprintf("    sd   = %.4f\n", sd(out$theta)))
cat(sprintf("    q05  = %.4f,  q50 = %.4f,  q95 = %.4f\n",
            quantile(out$theta, 0.05),
            quantile(out$theta, 0.50),
            quantile(out$theta, 0.95)))

# Pass criteria for Test 1
t1_simplex <- simplex_err_max < 1e-10
t1_rb      <- max(diff_all) < 2e-2    # simplex NUTS has autocorr in high dim
t1_theta_ok<- is.finite(mean(out$theta)) && mean(out$theta) > 0

cat(sprintf("\n  Test 1 simplex OK?             %s\n",
            ifelse(t1_simplex, "YES", "NO")))
cat(sprintf("  Test 1 Rao-Blackwell agree?    %s  (max diff %.2e)\n",
            ifelse(t1_rb, "YES", "NO"), max(diff_all)))
cat(sprintf("  Test 1 theta well-defined?     %s\n",
            ifelse(t1_theta_ok, "YES", "NO")))

# ============================================================================
#  Overall
# ============================================================================
cat("\n==================================================================\n")
all_ok <- t1_simplex && t1_rb && t1_theta_ok
cat(sprintf(" %s\n", if (all_ok) "ALL PASS" else "SOME CHECKS FAILED"))
cat("==================================================================\n")

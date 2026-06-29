# ----------------------------------------------------------------------------
# run_bart.R
#
# End-to-end demo of BartNoise.cpp on the Friedman (1991) benchmark:
#   y = 10 sin(pi*x1*x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + epsilon
#
# Runs one chain, checks in-sample RMSE against Friedman-with-zero-noise
# truth, and reports posterior sigma summary. This is the canonical
# pattern generated samplers that include BART should follow.
#
# BART inside AI4BayesCode uses R's RNG via arn in bart/BART/rn.h, so
# set.seed() at the start of the script controls the tree sampler.
# The int seed passed to `new(BartNoise, ..., seed)` controls the
# accompanying nuts_block for sigma.
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
    cpp_file       = file.path(script_dir, "BartNoise.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

# ---- Friedman (1991) data generator ---------------------------------------
friedman_mean <- function(X) {
    10 * sin(pi * X[, 1] * X[, 2]) +
        20 * (X[, 3] - 0.5)^2 +
        10 * X[, 4] +
         5 * X[, 5]
}

set.seed(20260411)
N <- 500L
p <- 5L
# Friedman adds irrelevant covariates if p > 5; we stick to exactly 5.
X <- matrix(runif(N * p), nrow = N, ncol = p)
mu_true   <- friedman_mean(X)
sigma_true <- 1.0
y <- mu_true + rnorm(N, sd = sigma_true)

cat(sprintf("Generated Friedman data: N=%d, p=%d, sample sd(y)=%.4f\n",
            N, p, sd(y)))
cat(sprintf("  true mean range:  [%.2f, %.2f]\n",
            min(mu_true), max(mu_true)))
cat(sprintf("  sigma_true = %.4f\n\n", sigma_true))

# ---- Build and run the sampler --------------------------------------------
n_burnin <- 500L
n_keep   <- 1000L
ntrees   <- 50L

model <- new(BartNoise,
             X,           # design matrix (N x p)
             y,           # response (length N)
             ntrees,      # number of trees
             2.0,         # k (prior SD scale)
             2.0,         # power (tree depth penalty)
             0.95,        # base (tree depth base)
             3.0,         # nu (sigma prior df)
             100L,        # numcut (cutpoints per variable)
             FALSE,       # dart (DART sparsity prior)
             FALSE,       # aug (DART augmentation)
             31337L)      # rng_seed

cat(sprintf("Running %d burnin + %d keep sweeps with %d trees...\n",
            n_burnin, n_keep, ntrees))

t0 <- Sys.time()
model$step(n_burnin)
burnin_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  burnin done in %.1f s\n", burnin_time))

fitted_sum <- numeric(N)
sigma_draws <- numeric(n_keep)
t0 <- Sys.time()
for (s in seq_len(n_keep)) {
    model$step(1L)
    draw <- model$get_current()
    fitted_sum    <- fitted_sum + draw$f_bart
    sigma_draws[s] <- draw$sigma
}
keep_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  keep   done in %.1f s\n\n", keep_time))

# ---- Summaries ------------------------------------------------------------
f_post_mean <- fitted_sum / n_keep
rmse <- sqrt(mean((f_post_mean - mu_true)^2))

cat("Posterior summaries:\n")
cat(sprintf("  sigma: post mean = %.4f  sd = %.4f   (truth %.4f)\n",
            mean(sigma_draws), sd(sigma_draws), sigma_true))
cat(sprintf("  in-sample RMSE (post mean vs Friedman truth) = %.4f\n",
            rmse))
cat(sprintf(paste0("  RMSE / sigma_true = %.3f   ",
                   "(BART on N=500 Friedman typically gets 0.5 - 1.0 ",
                   "for well-tuned ntrees)\n"),
            rmse / sigma_true))

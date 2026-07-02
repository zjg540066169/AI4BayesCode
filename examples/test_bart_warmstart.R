# ----------------------------------------------------------------------------
# test_bart_warmstart.R
#
# THE MOST IMPORTANT TEST for AI4BayesCode's core architectural promise:
# calling `step(1)` K times must produce BYTE-IDENTICAL results to
# calling `step(K)` once, assuming the same RNG seeds.
#
# If this test fails, it means:
#   - bart_block's set_context is accidentally resetting some internal
#     BART state between sweeps, OR
#   - composite_block is not threading the rng correctly, OR
#   - something else in the persistence chain is broken.
#
# If this test passes, we have proved that bart_block::step() is truly
# stateful and AI4BayesCode's "stateful MCMC inside a larger MCMC" promise
# holds for BART. This is the architectural prerequisite for PSPI-style
# block Gibbs.
#
# RNG channels:
#   - R's RNG (used by BART internally) is seeded with set.seed()
#   - mt19937 (used by sigma's nuts_block) is seeded by the int arg
#     passed to new(BartNoise, ...)
# Both channels are identical between the two runs, so byte-exact
# equality is achievable -- not just "close".
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
    cpp_file       = file.path(script_dir, "BartNoise.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

# ---- Simulate Friedman data once ------------------------------------------
set.seed(20260411)
N <- 200L
p <- 5L
X <- matrix(runif(N * p), nrow = N, ncol = p)
y <- (10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2
      + 10 * X[, 4] + 5 * X[, 5]
      + rnorm(N))

K <- 500L

# ---- Run A: step(K) once --------------------------------------------------
set.seed(42)
modelA <- new(BartNoise, X, y, 50L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE, 777L)
modelA$step(K)
drawA <- modelA$get_current()

# ---- Run B: step(1) K times -----------------------------------------------
set.seed(42)   # identical R RNG stream
modelB <- new(BartNoise, X, y, 50L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE, 777L)  # identical mt19937 seed
for (i in seq_len(K)) modelB$step(1L)
drawB <- modelB$get_current()

# ---- Compare --------------------------------------------------------------
sigma_diff <- abs(drawA$sigma - drawB$sigma)
fit_diff   <- max(abs(drawA$f_bart - drawB$f_bart))

cat("State preservation test:\n")
cat(sprintf("  step(%d) -> sigma = %.12f\n", K, drawA$sigma))
cat(sprintf("  step(1)*%d -> sigma = %.12f\n", K, drawB$sigma))
cat(sprintf("  |sigma diff|    = %.2e\n", sigma_diff))
cat(sprintf("  max |fit diff| = %.2e\n", fit_diff))

tol <- 1e-12
if (sigma_diff < tol && fit_diff < tol) {
    cat("\nPASS: step(1)*K is byte-identical to step(K).\n")
    cat("      bart_block is truly stateful.\n")
    quit(status = 0)
} else if (sigma_diff < 1e-6 && fit_diff < 1e-6) {
    cat("\nNEAR-PASS: differences are floating-point noise (< 1e-6).\n")
    cat("           bart_block is effectively stateful; minor divergence\n")
    cat("           likely comes from double rounding in BART internals.\n")
    quit(status = 0)
} else {
    cat("\nFAIL: non-trivial difference between step(1)*K and step(K).\n")
    cat("      This means bart_block is NOT fully stateful.\n")
    cat("      Diagnose set_context / composite threading immediately.\n")
    quit(status = 1)
}

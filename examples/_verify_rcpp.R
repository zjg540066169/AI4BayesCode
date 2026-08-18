# ----------------------------------------------------------------------------
# _verify_rcpp.R
#
# Integration test used during package development to confirm that the
# Rcpp example files under AI4BayesCode/examples/ actually compile and run.
# This is NOT end-user code; end users run run_gaussian.R / run_dirichlet.R.
#
# Invocation (from AI4BayesCode/examples/):
#     Rscript _verify_rcpp.R
#
# Exit code:
#     0  = both examples compiled and produced sane output
#     1+ = at least one example failed (the failure is reported on stderr)
# ----------------------------------------------------------------------------

suppressPackageStartupMessages({
    library(Rcpp)
    library(RcppArmadillo)
})

# Use the shipped helper rather than hand-rolling PKG_CPPFLAGS. The hand-rolled
# version passed its -I paths through shQuote(), which R's make did not honour
# -- the compile line carried no AI4BayesCode include at all and every example
# failed on "'AI4BayesCode/block_sampler.hpp' file not found". It had also
# drifted: it never passed the BaseMatrixOps or eigen directories that every
# other compile path supplies. One code path, maintained in one place.
script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg)) dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

compile_example <- function(f)
    ai4bayescode_source_checkout(file.path(AI4BayesCode_dir, "examples", f),
                                 AI4BayesCode_path = AI4BayesCode_dir,
                                 rebuild = TRUE, verbose = FALSE)

fail <- 0L

# ---- Gaussian location-scale ----------------------------------------------

cat("=== compiling GaussianLocationScale.cpp ===\n")
ok_gauss <- tryCatch({
    compile_example("GaussianLocationScale.cpp")
    TRUE
}, error = function(e) {
    message("GaussianLocationScale.cpp FAILED to compile:\n", conditionMessage(e))
    FALSE
})

if (ok_gauss) {
    set.seed(42)
    y <- rnorm(200, mean = 2.5, sd = 1.2)
    model <- new(GaussianLocationScale, y, 31337L)
    model$step(500L)
    draws <- matrix(NA_real_, nrow = 1000, ncol = 2,
                    dimnames = list(NULL, c("mu", "sigma")))
    for (s in seq_len(1000)) {
        model$step(1L)
        d <- model$get_current()
        draws[s, ] <- c(d$mu, d$sigma)
    }
    mu_mean    <- mean(draws[, "mu"])
    sigma_mean <- mean(draws[, "sigma"])
    cat(sprintf("  mu mean    = %.4f (truth 2.5)\n",    mu_mean))
    cat(sprintf("  sigma mean = %.4f (truth 1.2)\n", sigma_mean))
    if (abs(mu_mean - 2.5) < 0.3 && abs(sigma_mean - 1.2) < 0.3) {
        cat("  GaussianLocationScale PASS\n")
    } else {
        cat("  GaussianLocationScale FAIL (posterior far from truth)\n")
        fail <- fail + 1L
    }
} else {
    fail <- fail + 1L
}

# ---- Dirichlet simplex -----------------------------------------------------

cat("\n=== compiling DirichletSimplex.cpp ===\n")
ok_dir <- tryCatch({
    compile_example("DirichletSimplex.cpp")
    TRUE
}, error = function(e) {
    message("DirichletSimplex.cpp FAILED to compile:\n", conditionMessage(e))
    FALSE
})

if (ok_dir) {
    set.seed(43)
    K <- 5L
    theta_true <- c(0.10, 0.25, 0.35, 0.20, 0.10)
    y_counts   <- as.numeric(table(
        factor(sample.int(K, 500, prob = theta_true, replace = TRUE),
               levels = seq_len(K))))
    alpha      <- rep(1.0, K)

    model <- new(DirichletSimplex, y_counts, alpha, 20260410L)
    model$step(500L)
    draws <- matrix(NA_real_, nrow = 1000, ncol = K)
    for (s in seq_len(1000)) {
        model$step(1L)
        draws[s, ] <- model$get_current()$theta
    }
    post_mean <- colMeans(draws)
    analytic  <- (alpha + y_counts) / sum(alpha + y_counts)
    cat("  post mean  = ", sprintf("%.4f ", post_mean), "\n")
    cat("  analytic   = ", sprintf("%.4f ", analytic),  "\n")

    max_err <- max(abs(post_mean - analytic))
    if (max_err < 0.02) {
        cat(sprintf("  DirichletSimplex PASS (max |mean - analytic| = %.4f)\n",
                    max_err))
    } else {
        cat(sprintf("  DirichletSimplex FAIL (max err = %.4f)\n", max_err))
        fail <- fail + 1L
    }
} else {
    fail <- fail + 1L
}

cat(sprintf("\n=== verification complete: %d failure(s) ===\n", fail))
quit(status = fail)

# ----------------------------------------------------------------------------
# test_beta_bernoulli.R
#
# Test Beta-distributed parameter p in (0, 1) sampled with NUTS.
# Compares against analytic Beta posterior.
#
# This catches common AI agent errors:
#   1. Using constraints::positive instead of constraints::interval(0,1)
#   2. Wrong Jacobian for logit transform
#   3. Missing boundary checks
#   4. Wrong API for constraints::interval (needs lo, hi args)
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
    cpp_file       = file.path(script_dir, "BetaBernoulli.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

# ---- Test 1: p near 0.5 (easy) -------------------------------------------
cat("=== Test 1: p_true = 0.7, N = 100, Beta(1,1) prior ===\n")
set.seed(42)
N <- 100L
p_true <- 0.7
y <- rbinom(N, 1, p_true)
a_prior <- 1.0; b_prior <- 1.0

# Analytic posterior: Beta(a + sum(y), b + N - sum(y))
a_post <- a_prior + sum(y)
b_post <- b_prior + N - sum(y)
post_mean <- a_post / (a_post + b_post)
post_var  <- a_post * b_post / ((a_post + b_post)^2 * (a_post + b_post + 1))
post_sd   <- sqrt(post_var)

cat(sprintf("  sum(y) = %d, analytic: mean = %.4f, sd = %.4f\n",
            sum(y), post_mean, post_sd))

# Run 2 chains
run_chain <- function(seed, n_burnin = 4000, n_keep = 4000) {
    model <- new(BetaBernoulli, as.numeric(y), a_prior, b_prior,
                 as.integer(seed))
    model$step(n_burnin)
    draws <- numeric(n_keep)
    for (i in seq_len(n_keep)) {
        model$step(1L)
        draws[i] <- model$get_current()$p
    }
    draws
}

c1 <- run_chain(101L)
c2 <- run_chain(202L)

sample_mean <- mean(c(c1, c2))
sample_sd   <- sd(c(c1, c2))
cat(sprintf("  NUTS:     mean = %.4f, sd = %.4f\n", sample_mean, sample_sd))

# R-hat
rhat <- if (requireNamespace("posterior", quietly = TRUE)) {
    as.numeric(posterior::rhat(cbind(c1, c2)))
} else {
    n <- length(c1); h <- n %/% 2
    halves <- cbind(c1[1:h], c1[(h+1):(2*h)], c2[1:h], c2[(h+1):(2*h)])
    W <- mean(apply(halves, 2, var))
    B <- var(colMeans(halves)) * h
    sqrt((h-1)/h + B/(h*W))
}
cat(sprintf("  R-hat = %.4f\n", rhat))

# Check mean and sd agree with analytic
mean_ok <- abs(sample_mean - post_mean) < 3 * post_sd / sqrt(8000)
sd_ratio <- sample_sd / post_sd
sd_ok <- sd_ratio > 0.9 && sd_ratio < 1.1

cat(sprintf("  mean agrees: %s (diff = %.4e)\n",
            ifelse(mean_ok, "YES", "NO"), abs(sample_mean - post_mean)))
cat(sprintf("  sd ratio: %.4f %s\n", sd_ratio,
            ifelse(sd_ok, "(OK)", "(BAD)")))

# ---- Test 2: p near boundary (hard) --------------------------------------
cat("\n=== Test 2: p_true = 0.02, N = 200, Beta(0.5, 0.5) prior ===\n")
set.seed(43)
N2 <- 200L
p_true2 <- 0.02
y2 <- rbinom(N2, 1, p_true2)
a2 <- 0.5; b2 <- 0.5

a_post2 <- a2 + sum(y2)
b_post2 <- b2 + N2 - sum(y2)
post_mean2 <- a_post2 / (a_post2 + b_post2)
post_sd2   <- sqrt(a_post2 * b_post2 / ((a_post2 + b_post2)^2 *
                                          (a_post2 + b_post2 + 1)))

cat(sprintf("  sum(y) = %d, analytic: mean = %.4f, sd = %.4f\n",
            sum(y2), post_mean2, post_sd2))

c3 <- run_chain2 <- function(seed) {
    model <- new(BetaBernoulli, as.numeric(y2), a2, b2, as.integer(seed))
    model$step(4000L)
    draws <- numeric(4000L)
    for (i in seq_len(4000L)) {
        model$step(1L)
        draws[i] <- model$get_current()$p
    }
    draws
}
c3 <- run_chain2(301L)
c4 <- run_chain2(402L)

sample_mean2 <- mean(c(c3, c4))
sample_sd2   <- sd(c(c3, c4))
cat(sprintf("  NUTS:     mean = %.4f, sd = %.4f\n", sample_mean2, sample_sd2))

# All draws in (0, 1)?
in_range <- all(c(c3, c4) > 0 & c(c3, c4) < 1)
cat(sprintf("  all draws in (0,1): %s\n", ifelse(in_range, "YES", "NO")))

mean_ok2 <- abs(sample_mean2 - post_mean2) < 3 * post_sd2 / sqrt(8000)
sd_ratio2 <- sample_sd2 / post_sd2
sd_ok2 <- sd_ratio2 > 0.85 && sd_ratio2 < 1.15

cat(sprintf("  mean agrees: %s (diff = %.4e)\n",
            ifelse(mean_ok2, "YES", "NO"), abs(sample_mean2 - post_mean2)))
cat(sprintf("  sd ratio: %.4f %s\n", sd_ratio2,
            ifelse(sd_ok2, "(OK)", "(BAD)")))

# ---- Test 3: extreme prior Beta(100, 2) ----------------------------------
cat("\n=== Test 3: strong prior Beta(100, 2), N = 10, sum(y) = 3 ===\n")
y3 <- c(rep(1, 3), rep(0, 7))
a3 <- 100; b3 <- 2

a_post3 <- a3 + 3; b_post3 <- b3 + 7
post_mean3 <- a_post3 / (a_post3 + b_post3)
post_sd3   <- sqrt(a_post3 * b_post3 / ((a_post3 + b_post3)^2 *
                                          (a_post3 + b_post3 + 1)))

cat(sprintf("  analytic: mean = %.4f, sd = %.4f\n", post_mean3, post_sd3))

c5 <- {
    model <- new(BetaBernoulli, as.numeric(y3), a3, b3, 501L)
    model$step(4000L)
    draws <- numeric(4000L)
    for (i in seq_len(4000L)) {
        model$step(1L)
        draws[i] <- model$get_current()$p
    }
    draws
}
cat(sprintf("  NUTS: mean = %.4f, sd = %.4f\n", mean(c5), sd(c5)))
mean_ok3 <- abs(mean(c5) - post_mean3) < 3 * post_sd3 / sqrt(4000)
cat(sprintf("  mean agrees: %s\n", ifelse(mean_ok3, "YES", "NO")))

# ---- Summary --------------------------------------------------------------
cat("\n=== Summary ===\n")
all_ok <- mean_ok && sd_ok && rhat < 1.05 &&
          mean_ok2 && sd_ok2 && in_range &&
          mean_ok3
cat(sprintf("%s\n", if (all_ok) "ALL TESTS PASS" else "SOME TESTS FAILED"))

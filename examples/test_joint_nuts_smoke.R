# ---------------------------------------------------------------------------
# test_joint_nuts_smoke.R
#
# Quick smoke test to verify IRT1PL_joint:
#   - compiles and constructs
#   - step() runs without error
#   - get_current / get_history return correct shapes
#   - set_current updates parameters consistently
#   - basic posterior recovery on a small dataset
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

AI4BayesCode_sourceCpp(file.path(script_dir, "IRT1PL_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

n_pass <- 0L
n_fail <- 0L
check <- function(cond, msg) {
    if (isTRUE(cond)) {
        cat(sprintf("  PASS: %s\n", msg))
        n_pass <<- n_pass + 1L
    } else {
        cat(sprintf("  FAIL: %s\n", msg))
        n_fail <<- n_fail + 1L
    }
}

# --- Small synthetic IRT data ---------------------------------------------
set.seed(1L)
N <- 30L
J <- 8L
theta_true <- rnorm(N, 0, 1)
sigma_b_true <- 0.8
b_true <- rnorm(J, 0, sigma_b_true)
P <- outer(theta_true, b_true, function(a, bb) 1 / (1 + exp(-(a - bb))))
Y <- matrix(rbinom(N * J, 1, P), nrow = N, ncol = J)
Y[matrix(runif(N * J) < 0.1, nrow = N, ncol = J)] <- NA_real_

# --- Construction ---------------------------------------------------------
cat("\n=== Construction ===\n")
m <- new(IRT1PL_joint, Y,
         rep(0.0, N), rep(0.0, J),
         1.0, 1L, TRUE)
check(TRUE, "construction succeeds")

d0 <- m$get_current()
check(length(d0$theta)   == N, "get_current $theta length = N")
check(length(d0$b)       == J, "get_current $b     length = J")
check(length(d0$sigma_b) == 1, "get_current $sigma_b scalar")

# --- step() and history ---------------------------------------------------
cat("\n=== step() and history ===\n")
t0 <- Sys.time()
m$step(50L)
dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  50 sweeps took %.2fs  (%.3fs/sweep)\n", dt, dt / 50))
check(TRUE, "step(50) runs without error")

h <- m$get_history()
check(is.matrix(h$theta), "$theta is a matrix")
check(nrow(h$theta) == 50 && ncol(h$theta) == N, "$theta shape (50 x N)")
check(is.matrix(h$b),     "$b is a matrix")
check(nrow(h$b) == 50 && ncol(h$b) == J,         "$b shape (50 x J)")
check(length(h$sigma_b) == 50,                   "$sigma_b length 50")

# --- set_current consistency ---------------------------------------------
cat("\n=== set_current ===\n")
new_theta <- rnorm(N, 0, 0.5)
new_b     <- rnorm(J, 0, 0.5)
m$set_current(list(theta = new_theta, b = new_b, sigma_b = 0.7))
dnow <- m$get_current()
check(max(abs(dnow$theta - new_theta)) < 1e-12, "set_current $theta round-trip")
check(max(abs(dnow$b     - new_b))     < 1e-12, "set_current $b round-trip")
check(abs(dnow$sigma_b - 0.7)          < 1e-12, "set_current $sigma_b round-trip")

# --- Posterior recovery (small chain, loose bounds) ----------------------
cat("\n=== Posterior recovery (200 burnin + 500 keep) ===\n")
m2 <- new(IRT1PL_joint, Y,
          rep(0.0, N), rep(0.0, J),
          1.0, 11L, TRUE)
t0 <- Sys.time()
m2$step(200L)
m2$step(500L)
dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  total 700 sweeps: %.2fs\n", dt))

h2 <- m2$get_history()
post_theta <- colMeans(h2$theta[201:700, ])
post_b     <- colMeans(h2$b[201:700, ])
post_sb    <- mean(h2$sigma_b[201:700])

# Correlation with truth (should be high for theta & b)
cor_theta <- cor(post_theta, theta_true)
cor_b     <- cor(post_b,     b_true)

cat(sprintf("  cor(post_theta, true) = %.3f\n", cor_theta))
cat(sprintf("  cor(post_b, true)     = %.3f\n", cor_b))
cat(sprintf("  post_mean sigma_b     = %.3f  (true %.3f)\n", post_sb, sigma_b_true))

check(cor_theta > 0.7, "theta posterior correlates with truth (> 0.7)")
check(cor_b     > 0.7, "b posterior correlates with truth (> 0.7)")
check(post_sb > 0.2 && post_sb < 2.5, "sigma_b in plausible range")

cat(sprintf("\n=== %d passed, %d failed ===\n", n_pass, n_fail))
if (n_fail > 0L) quit(status = 1L)

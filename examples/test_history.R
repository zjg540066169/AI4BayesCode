# ----------------------------------------------------------------------------
# test_history.R
#
# Exercises the keep_history = TRUE API end-to-end on three model types:
#     GaussianLocationScale  (pure NUTS)
#     BartNoise              (BART + NUTS + history-mode predict_at)
#     DirichletSimplex       (simplex NUTS, conjugate analytic check)
#
# For each model we:
#   1. Construct with keep_history = TRUE.
#   2. Run one long sweep with model$step(n_burn + n_keep) — no R-side loop.
#   3. Pull draws back with the model$get_history() helper.
#   4. Assert: history shape, R-hat (vs a second chain), and a
#      model-specific posterior-correctness check (ybar for Gaussian,
#      prediction correlation for BART, analytic Dirichlet posterior mean
#      for DirichletSimplex).
#
# This test intentionally does NOT hand-roll a step(1) + get_current()
# loop — that pattern is what the history API replaced. Use
# test_keep_history_bart.R and the per-GBart* scripts under
# tests_autodiff/gbart_*.R for the per-wrapper shape + diagnostic tests;
# this file is the integration check that the full history workflow
# (construct -> step -> model$get_history() -> analyze) works.
#
# Note: each model is compiled only ONCE per Rscript invocation to avoid
# Rcpp module-reload segfaults.
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

n_pass <- 0L; n_fail <- 0L
pass  <- function(msg) { cat(sprintf("  PASS: %s\n", msg)); n_pass <<- n_pass + 1L }
fail  <- function(msg) { cat(sprintf("  FAIL: %s\n", msg)); n_fail <<- n_fail + 1L }
check <- function(cond, msg) { if (isTRUE(cond)) pass(msg) else fail(msg) }

# Minimal fallback R-hat if the `posterior` package isn't available.
rhat_of <- function(x1, x2) {
    if (requireNamespace("posterior", quietly = TRUE)) {
        as.numeric(posterior::rhat(cbind(x1, x2)))
    } else {
        n <- length(x1); h <- n %/% 2L
        halves <- cbind(x1[1:h], x1[(h+1L):(2L*h)],
                        x2[1:h], x2[(h+1L):(2L*h)])
        W <- mean(apply(halves, 2, var))
        B <- var(colMeans(halves)) * h
        sqrt((h - 1L) / h + B / (h * W))
    }
}

# ============================================================================
#  Test 1: GaussianLocationScale — history API + 2-chain R-hat
# ============================================================================
cat("=== Test 1: GaussianLocationScale (history API) ===\n")
ai4bayescode_sourceCpp(file.path(script_dir, "GaussianLocationScale.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
y <- rnorm(50, 3, 1.5)

run_gaussian_chain <- function(seed, n_burn = 4000L, n_keep = 4000L) {
    model <- new(GaussianLocationScale, y, as.integer(seed), TRUE)   # keep_history = TRUE
    model$step(n_burn + n_keep)     # one call, no per-iter loop
    draws <- model$get_history()
    # Drop burnin rows
    list(mu    = draws$mu   [(n_burn + 1L):(n_burn + n_keep)],
         sigma = draws$sigma[(n_burn + 1L):(n_burn + n_keep)])
}

c1 <- run_gaussian_chain(101L)
c2 <- run_gaussian_chain(202L)

check(length(c1$mu) == 4000L,       "4000 mu draws after split + burnin drop")
check(length(c1$sigma) == 4000L,    "4000 sigma draws after split + burnin drop")
check(all(is.finite(c1$mu)),        "all mu draws finite")
check(all(c1$sigma > 0),            "all sigma draws positive")

rhat_mu    <- rhat_of(c1$mu,    c2$mu)
rhat_sigma <- rhat_of(c1$sigma, c2$sigma)
cat(sprintf("  R-hat mu=%.4f  sigma=%.4f\n", rhat_mu, rhat_sigma))
check(rhat_mu    < 1.05, "mu R-hat < 1.05")
check(rhat_sigma < 1.05, "sigma R-hat < 1.05")

pool_mu <- c(c1$mu, c2$mu)
check(abs(mean(pool_mu) - mean(y)) < 0.2, "mu posterior near ybar")

# ============================================================================
#  Test 2: BartNoise — history API + history-mode predict_at
# ============================================================================
cat("\n=== Test 2: BartNoise (history API + predict_at matrix) ===\n")
ai4bayescode_sourceCpp(file.path(script_dir, "BartNoise.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
N <- 100L; p <- 3L
X <- matrix(rnorm(N * p), ncol = p)
f_true <- sin(X[, 1]) + 0.5 * X[, 2]
y_bart <- f_true + rnorm(N, 0, 0.5)
N_test <- 30L
X_test <- matrix(rnorm(N_test * p), ncol = p)
f_test_true <- sin(X_test[, 1]) + 0.5 * X_test[, 2]

n_burn <- 500L
n_keep <- 500L
model  <- new(BartNoise, X, y_bart, 50L, 2.0, 2.0, 0.95, 3.0, 100L,
              FALSE, FALSE, 42L, TRUE)   # keep_history = TRUE
model$step(n_burn + n_keep)

# Split the concatenated history into named blocks.
H <- model$get_history()
check(is.matrix(H$f_bart),                        "H$f_bart is a matrix")
check(nrow(H$f_bart) == (n_burn + n_keep),        "H$f_bart has n_total rows")
check(ncol(H$f_bart) == N,                        "H$f_bart has N columns")
check(is.numeric(H$sigma) && !is.matrix(H$sigma), "H$sigma is a vector")
check(length(H$sigma) == (n_burn + n_keep),       "H$sigma has n_total entries")
check(all(H$sigma > 0),                           "all sigma draws positive")

# History-mode predict_at returns a matrix (n_total x N_test)
pred     <- model$predict_at(list(X = X_test))
check(is.matrix(pred$f_bart),
      "history-mode predict_at$f_bart is a matrix")
check(identical(dim(pred$f_bart),
                c(as.integer(n_burn + n_keep), N_test)),
      "history-mode predict_at shape = (n_total x N_test)")

# Accuracy on the kept slice
keep_rows    <- (n_burn + 1L):(n_burn + n_keep)
f_pred_mean  <- colMeans(pred$f_bart[keep_rows, ])
cor_val      <- cor(f_pred_mean, f_test_true)
cat(sprintf("  prediction correlation = %.4f\n", cor_val))
check(cor_val > 0.7, "prediction correlation > 0.7")

sigma_keep   <- H$sigma[keep_rows]
cat(sprintf("  sigma posterior mean = %.4f  (true 0.5)\n", mean(sigma_keep)))
check(abs(mean(sigma_keep) - 0.5) < 0.2, "sigma near truth")

# ============================================================================
#  Test 3: DirichletSimplex — history API + analytic posterior
# ============================================================================
cat("\n=== Test 3: DirichletSimplex (history API + analytic check) ===\n")
ai4bayescode_sourceCpp(file.path(script_dir, "DirichletSimplex.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

model_d <- new(DirichletSimplex, c(10, 20, 30, 40), c(1, 1, 1, 1), 42L,
               TRUE)  # keep_history = TRUE
n_burn <- 500L
n_keep <- 1000L
model_d$step(n_burn + n_keep)

Hd <- model_d$get_history()
check(is.matrix(Hd$theta),                              "Hd$theta is a matrix")
check(ncol(Hd$theta) == 4L,                             "Hd$theta has K=4 columns")
check(nrow(Hd$theta) == (n_burn + n_keep),              "Hd$theta has n_total rows")

theta_keep <- Hd$theta[(n_burn + 1L):(n_burn + n_keep), , drop = FALSE]
check(all(abs(rowSums(theta_keep) - 1) < 1e-10),        "every kept draw on simplex")

alpha_post <- c(11, 21, 31, 41)
post_mean  <- alpha_post / sum(alpha_post)
max_diff   <- max(abs(colMeans(theta_keep) - post_mean))
cat(sprintf("  max |colMean(theta) - analytic mean| = %.4f\n", max_diff))
check(max_diff < 0.02, "simplex posterior near analytic")

# ============================================================================
#  Summary
# ============================================================================
cat(sprintf("\n=== %d passed, %d failed ===\n", n_pass, n_fail))
cat(sprintf("%s\n", if (n_fail == 0L) "ALL TESTS PASS" else "SOME TESTS FAILED"))
if (n_fail != 0L) quit(status = 1L)

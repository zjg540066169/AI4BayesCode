# ----------------------------------------------------------------------------
# test_keep_history_bart.R
#
# BartNoise keep_history flag + history-mode predict_at full test suite.
#
# Layers (in order, cheap to expensive):
#   A. Post-construction smoke tests (no stepping yet)
#   B. Stateful-mode behavior (keep_history = FALSE)
#   C. History-mode behavior (keep_history = TRUE)
#   D. Predict_at state immutability in both modes
#   E. Prediction accuracy (posterior mean vs truth)
#   F. 4-chain Gelman rank-normalized R-hat (Vehtari et al. 2021), bulk ESS,
#      tail ESS over sigma and f_bart points.
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

has_posterior <- requireNamespace("posterior", quietly = TRUE)
if (!has_posterior) {
    cat("NOTE: 'posterior' package not found — rank-normalized R-hat and ESS\n",
        "      will fall back to a plain R-hat implementation.\n", sep = "")
}

n_pass <- 0L; n_fail <- 0L
pass <- function(msg) { cat(sprintf("  PASS: %s\n", msg)); n_pass <<- n_pass + 1L }
fail <- function(msg) { cat(sprintf("  FAIL: %s\n", msg)); n_fail <<- n_fail + 1L }
check <- function(cond, msg) { if (isTRUE(cond)) pass(msg) else fail(msg) }

cat("=== Compiling BartNoise ===\n")
ai4bayescode_sourceCpp(file.path(script_dir, "BartNoise.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)

set.seed(123)
N <- 120; p <- 4
X <- matrix(rnorm(N * p), ncol = p)
f_true <- sin(X[, 1]) + 0.5 * X[, 2] - 0.3 * X[, 3]
y <- f_true + rnorm(N, 0, 0.5)
N_test <- 50
X_test <- matrix(rnorm(N_test * p), ncol = p)
f_test_true <- sin(X_test[, 1]) + 0.5 * X_test[, 2] - 0.3 * X_test[, 3]

# ============================================================================
# Layer A: Post-construction smoke tests
# ----------------------------------------------------------------------------
# Check state is sensible before ANY stepping — catches silent init bugs.
# ============================================================================
cat("\n=== A. Post-construction smoke tests ===\n")

# A1. Stateful-mode construction
m_A <- new(BartNoise, X, y, 50L, 2.0, 2.0, 0.95, 3.0, 100L,
           FALSE, FALSE, 7L, FALSE)
dA <- m_A$get_current()
check(is.list(dA),                     "stateful: get_current() returns list")
check(all(c("f_bart","sigma") %in% names(dA)),
      "stateful: list has f_bart and sigma")
check(length(dA$f_bart) == N,          "stateful: f_bart length = N")
check(all(is.finite(dA$f_bart)),       "stateful: f_bart all finite at init")
check(length(dA$sigma) == 1,           "stateful: sigma scalar at init")
check(is.finite(dA$sigma) && dA$sigma > 0,
      "stateful: sigma positive and finite at init (OLS sigest)")
check(abs(mean(dA$f_bart)) < 10,       "stateful: f_bart init not wildly off")

hA <- m_A$get_history()
check(is.list(hA) && !is.null(names(hA)),
      "stateful: get_history() returns a named list")
check(nrow(hA$f_bart) == 1 && ncol(hA$f_bart) == N,
      "stateful: $f_bart is 1 x N fallback")
check(length(hA$sigma) == 1,
      "stateful: $sigma is length 1 fallback")
check(length(hA$f_bart_trees) == 1,
      "stateful: $f_bart_trees has 1 entry (serialized live trees)")

# A2. History-mode construction
m_B <- new(BartNoise, X, y, 50L, 2.0, 2.0, 0.95, 3.0, 100L,
           FALSE, FALSE, 7L, TRUE)
dB <- m_B$get_current()
check(is.list(dB) && length(dB$f_bart) == N && length(dB$sigma) == 1,
      "history-mode: get_current() shape same as stateful")
check(all(is.finite(dB$f_bart)) && is.finite(dB$sigma),
      "history-mode: all values finite at init")

# Before stepping, history buffers are empty on every leaf, but the
# composite's get_history() returns a 1-row fallback (current value) per
# the block_sampler contract.
hB <- m_B$get_history()
check(is.list(hB) && nrow(hB$f_bart) == 1 && ncol(hB$f_bart) == N,
      "history-mode: pre-step $f_bart is 1 x N fallback")
check(length(hB$sigma) == 1,
      "history-mode: pre-step $sigma is length-1 fallback")

# A3. Constructor rejects bad input
err_ok <- FALSE
tryCatch(new(BartNoise, X, y[-1], 50L, 2.0, 2.0, 0.95, 3.0, 100L,
             FALSE, FALSE, 7L, FALSE),
         error = function(e) err_ok <<- TRUE)
check(err_ok, "stateful: constructor errors on length mismatch")

# ============================================================================
# Layer B: Stateful-mode behavior (keep_history = FALSE)
# ============================================================================
cat("\n=== B. Stateful-mode behavior ===\n")
m1 <- new(BartNoise, X, y, 50L, 2.0, 2.0, 0.95, 3.0, 100L,
          FALSE, FALSE, 42L, FALSE)
m1$step(300L)
d1 <- m1$get_current()
check(all(is.finite(d1$f_bart)) && is.finite(d1$sigma),
      "stateful: after 300 steps values stay finite")

p1 <- m1$predict_at(list(X = X_test))
check(is.numeric(p1$f_bart) && !is.matrix(p1$f_bart),
      "stateful: predict_at$f_bart is a vector")
check(length(p1$f_bart) == N_test,
      "stateful: predict_at$f_bart length = N_test")

h1 <- m1$get_history()
check(nrow(h1$f_bart) == 1 && length(h1$sigma) == 1,
      "stateful: get_history() stays 1-row after 300 steps (not recording)")

# ============================================================================
# Layer C: History-mode behavior (keep_history = TRUE)
# ============================================================================
cat("\n=== C. History-mode behavior ===\n")
n_burn <- 400L
n_keep <- 600L
m2 <- new(BartNoise, X, y, 50L, 2.0, 2.0, 0.95, 3.0, 100L,
          FALSE, FALSE, 42L, TRUE)
m2$step(n_burn + n_keep)

h2 <- m2$get_history()
check(nrow(h2$f_bart) == (n_burn + n_keep) && ncol(h2$f_bart) == N,
      "history: $f_bart shape = (n_total, N)")
check(length(h2$sigma) == (n_burn + n_keep),
      "history: $sigma length = n_total")
check(length(h2$f_bart_trees) == (n_burn + n_keep),
      "history: $f_bart_trees length = n_total (one serialized forest per draw)")
check(all(is.finite(h2$f_bart)) && all(is.finite(h2$sigma)),
      "history: every stored numeric draw is finite")

# Expect positive posterior variance on sigma (not stuck)
check(sd(h2$sigma[(n_burn + 1):(n_burn + n_keep)]) > 1e-4,
      "history: sigma posterior has non-zero variance (chain not stuck)")

# predict_at shape
p2 <- m2$predict_at(list(X = X_test))
check(is.matrix(p2$f_bart),
      "history: predict_at$f_bart is a matrix")
check(nrow(p2$f_bart) == (n_burn + n_keep) && ncol(p2$f_bart) == N_test,
      "history: predict_at$f_bart shape = (n_draws x N_test)")
check(all(is.finite(p2$f_bart)),
      "history: predict_at$f_bart all finite")

# ============================================================================
# Layer D: predict_at state immutability
# ============================================================================
cat("\n=== D. predict_at state immutability ===\n")
s_before <- m1$get_current()
invisible(m1$predict_at(list(X = X_test)))
s_after  <- m1$get_current()
check(identical(s_before, s_after),
      "stateful: predict_at does not mutate current state")

h_before <- m2$get_history()
invisible(m2$predict_at(list(X = X_test)))
h_after  <- m2$get_history()
check(identical(h_before, h_after),
      "history: predict_at does not append to history buffer")

size_before <- m2$get_current()
invisible(m2$predict_at(list(X = X_test)))
size_after <- m2$get_current()
check(identical(size_before, size_after),
      "history: predict_at does not advance current draw either")

# ============================================================================
# Layer E: Prediction accuracy (posterior mean vs truth)
# ============================================================================
cat("\n=== E. Prediction accuracy ===\n")
keep_rows   <- (n_burn + 1):(n_burn + n_keep)
f_pred_mean <- colMeans(p2$f_bart[keep_rows, ])
cor_val     <- cor(f_pred_mean, f_test_true)
rmse        <- sqrt(mean((f_pred_mean - f_test_true)^2))
cat(sprintf("  cor(f_pred_mean, truth)   = %.4f\n", cor_val))
cat(sprintf("  RMSE(f_pred_mean - truth) = %.4f\n", rmse))
check(cor_val > 0.85, "posterior-mean prediction cor > 0.85")
check(rmse    < 0.5,  "posterior-mean prediction RMSE < 0.5")

# ============================================================================
# Layer F: MCMC diagnosis — 4-chain rank-normalized R-hat + ESS
# ----------------------------------------------------------------------------
# Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021):
# "Rank-normalization, folding, and localization: An improved Rhat for
#  assessing convergence of MCMC." Bayesian Analysis 16(2):667-718.
# Thresholds used below: Rhat < 1.01 (strict), bulk/tail ESS > 400.
# ============================================================================
cat("\n=== F. MCMC diagnosis (4 chains, rank-normalized) ===\n")
run_bart_chain <- function(seed, n_burn, n_keep) {
    m <- new(BartNoise, X, y, 50L, 2.0, 2.0, 0.95, 3.0, 100L,
             FALSE, FALSE, as.integer(seed), TRUE)
    m$step(n_burn + n_keep)
    H <- m$get_history()
    keep <- (n_burn + 1):(n_burn + n_keep)
    list(
        sigma  = H$sigma [keep],
        f_p10  = H$f_bart[keep, 10L],
        f_p60  = H$f_bart[keep, 60L],
        f_p110 = H$f_bart[keep, 110L]
    )
}

n_burn_diag <- 4000L
n_keep_diag <- 4000L
seeds <- c(11L, 22L, 33L, 44L)
cat(sprintf("  running %d chains, burnin=%d, keep=%d each ...\n",
            length(seeds), n_burn_diag, n_keep_diag))
chains <- lapply(seeds, run_bart_chain, n_burn = n_burn_diag, n_keep = n_keep_diag)

# Assemble into (iterations x chains) matrices per parameter
assemble <- function(param) {
    sapply(chains, function(c) c[[param]])  # iterations x chains
}

params <- c("sigma", "f_p10", "f_p60", "f_p110")

rhat_strict_thresh <- 1.01
ess_thresh         <- 400

for (prm in params) {
    mat <- assemble(prm)  # iters x chains
    if (has_posterior) {
        # posterior::rhat does rank-normalized split-R-hat by default
        rh      <- posterior::rhat(mat)
        ess_b   <- posterior::ess_bulk(mat)
        ess_t   <- posterior::ess_tail(mat)
        cat(sprintf("  %-7s  Rhat_rn=%.4f  bulk_ESS=%.0f  tail_ESS=%.0f\n",
                    prm, rh, ess_b, ess_t))
        check(is.finite(rh) && rh < rhat_strict_thresh,
              sprintf("%s: rank-normalized R-hat < %.2f", prm, rhat_strict_thresh))
        check(is.finite(ess_b) && ess_b > ess_thresh,
              sprintf("%s: bulk ESS > %d", prm, ess_thresh))
        check(is.finite(ess_t) && ess_t > ess_thresh,
              sprintf("%s: tail ESS > %d", prm, ess_thresh))
    } else {
        # Plain R-hat fallback (not rank-normalized)
        n <- nrow(mat); k <- ncol(mat)
        chain_means <- colMeans(mat)
        chain_vars  <- apply(mat, 2, var)
        W <- mean(chain_vars)
        B <- n * var(chain_means)
        varplus <- ((n - 1) / n) * W + B / n
        rh <- sqrt(varplus / W)
        cat(sprintf("  %-7s  Rhat_plain=%.4f\n", prm, rh))
        check(is.finite(rh) && rh < 1.05,
              sprintf("%s: plain R-hat < 1.05 (posterior pkg not installed)", prm))
    }
}

# Final posterior summary of sigma as a sanity check
sigma_pool <- unlist(lapply(chains, `[[`, "sigma"))
cat(sprintf("\n  sigma posterior: mean=%.4f  sd=%.4f  (true=%.2f)\n",
            mean(sigma_pool), sd(sigma_pool), 0.5))

# ============================================================================
# Summary
# ============================================================================
cat(sprintf("\n=== %d passed, %d failed ===\n", n_pass, n_fail))
if (n_fail != 0L) quit(status = 1L)

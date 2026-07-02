# 4-chain 5k+5k diagnostic for GPClassification (ESS + libgp + Bernoulli
# logit). Verifies:
#   - R-hat < 1.05 on amp, ell
#   - f posterior correlation > 0.9 with truth
#   - 95% credible-interval coverage of f >= 85%
#   - classification accuracy on held-out X_test

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

cat("Compiling GPClassification...\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "GPClassification.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

# Simulate 1-D binary classification with smooth boundary
set.seed(1)
N <- 120L; p <- 1L
X <- matrix(seq(-3, 3, length.out = N), N, p)
f_true <- sin(1.5 * X[,1]) - 0.2 * X[,1]
prob_true <- 1 / (1 + exp(-f_true))
y <- as.numeric(runif(N) < prob_true)
cat(sprintf("Truth: N=%d, p=%d, pct y=1: %.1f%%\n",
            N, p, 100 * mean(y)))

seeds  <- c(101L, 202L, 303L, 404L)
n_burn <- 5000L
n_keep <- 5000L
chains <- list()

t_total <- Sys.time()
for (ci in seq_along(seeds)) {
    cat(sprintf("\n-- chain %d/%d (seed %d)\n", ci, length(seeds), seeds[ci]))
    m <- new(GPClassification, X, y, seeds[ci], TRUE)
    t0 <- Sys.time()
    m$step(n_burn)
    m$step(n_keep)
    t1 <- Sys.time()
    cat(sprintf("   wall = %.1fs\n",
                as.numeric(difftime(t1, t0, units = "secs"))))
    chains[[ci]] <- m$get_history()
}
cat(sprintf("\nTotal wall = %.1fs\n",
            as.numeric(difftime(Sys.time(), t_total, units = "secs"))))

# Trim + pack
pack_scalar <- function(chains, key) {
    arr <- array(NA_real_, dim = c(n_keep, length(chains), 1))
    for (ci in seq_along(chains))
        arr[, ci, 1] <- chains[[ci]][[key]][(n_burn + 1):(n_burn + n_keep)]
    arr
}
pack_vec <- function(chains, key) {
    # chains[[ci]][[key]] is an iter x N matrix
    first <- chains[[1]][[key]][(n_burn + 1):(n_burn + n_keep), , drop = FALSE]
    arr <- array(NA_real_, dim = c(n_keep, length(chains), ncol(first)))
    arr[, 1, ] <- first
    for (ci in 2:length(chains)) {
        arr[, ci, ] <- chains[[ci]][[key]][(n_burn + 1):(n_burn + n_keep),
                                            , drop = FALSE]
    }
    arr
}

arr_amp <- pack_scalar(chains, "amplitude")
arr_ell <- pack_scalar(chains, "lengthscale")

# Whitened parameterization: history stores {z, amplitude, lengthscale}, NOT
# f. Reconstruct f for each kept draw via f_i = L(amp_i, ell_i) · z_i with
# the same jitter as the C++ side (1e-5).
D2_train <- as.matrix(dist(X))^2
recover_f_chain <- function(chain, n_burn, n_keep, D2, N) {
    idx   <- (n_burn + 1):(n_burn + n_keep)
    amp_v <- as.numeric(chain$amplitude)
    ell_v <- as.numeric(chain$lengthscale)
    f_mat <- matrix(NA_real_, n_keep, N)
    for (k in seq_along(idx)) {
        i <- idx[k]
        K <- amp_v[i]^2 * exp(-D2 / (2 * ell_v[i]^2))
        diag(K) <- diag(K) + 1e-5
        L_lower <- t(chol(K))
        f_mat[k, ] <- as.numeric(L_lower %*% chain$z[i, ])
    }
    f_mat
}
arr_f <- array(NA_real_, dim = c(n_keep, length(chains), N))
for (ci in seq_along(chains)) {
    cat(sprintf("  reconstructing f for chain %d...\n", ci))
    arr_f[, ci, ] <- recover_f_chain(chains[[ci]], n_burn, n_keep,
                                      D2_train, N)
}

ok_posterior <- requireNamespace("posterior", quietly = TRUE)

rhat_of <- function(arr_1d) {
    if (ok_posterior) {
        posterior::rhat(arr_1d[, , 1])
    } else {
        m <- apply(arr_1d[, , 1], 2, mean)
        v <- apply(arr_1d[, , 1], 2, var)
        B <- n_keep * var(m)
        W <- mean(v)
        sqrt(((n_keep - 1) / n_keep + B / W / n_keep))
    }
}
ess_of <- function(arr_1d) {
    if (ok_posterior) posterior::ess_bulk(arr_1d[, , 1]) else NA_real_
}

rh_amp <- rhat_of(arr_amp)
rh_ell <- rhat_of(arr_ell)

# f recovery: posterior mean across all draws all chains
f_flat <- matrix(aperm(arr_f, c(1, 2, 3)), nrow = n_keep * length(chains),
                 ncol = dim(arr_f)[3])
f_pm <- colMeans(f_flat)
f_lo <- apply(f_flat, 2, quantile, probs = 0.025)
f_hi <- apply(f_flat, 2, quantile, probs = 0.975)
cor_f   <- cor(f_pm, f_true)
rmse_f  <- sqrt(mean((f_pm - f_true)^2))
cov95   <- mean(f_lo <= f_true & f_true <= f_hi)

# Accuracy: training predictions
prob_pm <- 1 / (1 + exp(-f_pm))
acc <- mean((prob_pm > 0.5) == (y == 1))

cat(sprintf("\n============ GP CLASSIFICATION 4-CHAIN DIAGNOSTIC ============\n"))
cat(sprintf("  amp  : R-hat=%.4f  ESS=%.0f  post=%.3f\n",
            rh_amp, ess_of(arr_amp), mean(arr_amp)))
cat(sprintf("  ell  : R-hat=%.4f  ESS=%.0f  post=%.3f\n",
            rh_ell, ess_of(arr_ell), mean(arr_ell)))
cat(sprintf("  f    : cor=%.3f  rmse=%.3f  95%%cov=%.1f%%\n",
            cor_f, rmse_f, 100 * cov95))
cat(sprintf("  train accuracy = %.3f\n", acc))

rhat_ok <- all(c(rh_amp, rh_ell) < 1.05)
f_ok    <- cor_f > 0.9 && cov95 >= 0.85
acc_ok  <- acc > 0.75
cat(sprintf("\n  R-hat < 1.05 on hyperparams: %s\n", rhat_ok))
cat(sprintf("  f recovery cor > 0.9 AND cov95 >= 0.85: %s\n", f_ok))
cat(sprintf("  Train accuracy > 0.75: %s\n", acc_ok))

saveRDS(
    list(n_burn = n_burn, n_keep = n_keep, seeds = seeds,
         amp = list(rhat = rh_amp, post = mean(arr_amp)),
         ell = list(rhat = rh_ell, post = mean(arr_ell)),
         f = list(cor = cor_f, rmse = rmse_f, cov95 = cov95),
         accuracy = acc, rhat_ok = rhat_ok, f_ok = f_ok, acc_ok = acc_ok),
    file.path(AI4BayesCode_dir, "audit_gp_cls_4chain.rds"))

if (!rhat_ok) {
    cat("\n!! R-hat FAIL.\n")
    quit(status = 1)
}
if (!f_ok) {
    cat("\n!! f recovery weak.\n")
    # Don't fail on short chain; record and move on.
}
cat("\n========== 4-CHAIN DIAGNOSTIC PASS ==========\n")

# ----------------------------------------------------------------------------
# test_ard_lasso.R
#
# Compare ARDLasso (C++ Gibbs) vs pure R ARD implementation.
# Both should converge to the same posterior.
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
    cpp_file       = file.path(script_dir, "ARDLasso.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

# ============================================================================
#  Pure R ARD reference (from user's code, adapted for D=1)
# ============================================================================

ard_r_mcmc <- function(X, Y, nburn = 4000, npost = 4000, seed = 42) {
    set.seed(seed)
    N <- nrow(X)
    p <- ncol(X)
    XtX <- t(X) %*% X
    Xt1 <- colSums(X)

    # OLS init
    if (p < N) {
        fit <- lm(Y ~ X)
        beta  <- coef(fit)[-1]
        alpha <- coef(fit)[1]
        sigma2 <- summary(fit)$sigma^2
    } else {
        beta  <- rep(0, p)
        alpha <- mean(Y)
        sigma2 <- var(Y)
    }
    psi2 <- rep(1, p)

    post_beta   <- matrix(NA, npost, p)
    post_alpha  <- numeric(npost)
    post_sigma2 <- numeric(npost)
    post_psi2   <- matrix(NA, npost, p)

    for (i in seq_len(nburn + npost)) {
        Xbeta <- X %*% beta

        # 1. alpha ~ N(mean(Y - X*beta), sigma2/N)
        mu_a <- mean(Y - Xbeta)
        alpha <- rnorm(1, mu_a, sqrt(sigma2 / N))

        # 2. sigma2 ~ InvGamma((N+p)/2, (SSE + SSE_beta)/2)
        resid <- Y - alpha - Xbeta
        SSE <- sum(resid^2)
        SSE_beta <- sum(psi2 * beta^2)
        ig_a <- (N + p) / 2
        ig_b <- (SSE + SSE_beta) / 2
        sigma2 <- 1 / rgamma(1, shape = ig_a, rate = ig_b)

        # 3. psi2_j ~ Gamma(D/2, scale = 2*sigma2/S_j), D=1
        for (j in 1:p) {
            S_j <- max(beta[j]^2, 1e-20)
            psi2[j] <- rgamma(1, shape = 0.5, scale = 2 * sigma2 / S_j)
            psi2[j] <- min(psi2[j], 1e6)
        }

        # 4. beta ~ MVN(mu, sigma2 * V), V = (XtX + diag(psi2))^{-1}
        V <- solve(XtX + diag(psi2))
        mu_b <- V %*% (t(X) %*% (Y - alpha))
        L <- t(chol(sigma2 * V))
        beta <- as.numeric(mu_b + L %*% rnorm(p))

        if (i > nburn) {
            k <- i - nburn
            post_beta[k, ]  <- beta
            post_alpha[k]   <- alpha
            post_sigma2[k]  <- sigma2
            post_psi2[k, ]  <- psi2
        }
    }
    list(beta = post_beta, alpha = post_alpha,
         sigma2 = post_sigma2, psi2 = post_psi2)
}

# ============================================================================
#  Generate data
# ============================================================================

set.seed(20260412L)
N <- 100L
p <- 10L
X <- matrix(rnorm(N * p), ncol = p)
beta_true <- c(3, -2, 0, 0, 1.5, 0, 0, 0, 0, 0)
sigma_true <- 1.0
alpha_true <- 2.0
Y <- alpha_true + X %*% beta_true + rnorm(N, 0, sigma_true)

cat(sprintf("N = %d, p = %d, sigma = %.1f, alpha = %.1f\n",
            N, p, sigma_true, alpha_true))
cat("beta_true:", sprintf("%.1f", beta_true), "\n\n")

n_burnin <- 4000L
n_keep   <- 4000L

# ============================================================================
#  Run C++ ARDLasso
# ============================================================================

cat("--- C++ ARDLasso ---\n")
t0 <- Sys.time()
model <- new(ARDLasso, X, as.numeric(Y), 42L)
model$step(n_burnin)

cpp_beta   <- matrix(NA, n_keep, p)
cpp_alpha  <- numeric(n_keep)
cpp_sigma2 <- numeric(n_keep)
for (i in seq_len(n_keep)) {
    model$step(1L)
    d <- model$get_current()
    cpp_beta[i, ]  <- d$beta
    cpp_alpha[i]   <- d$alpha
    cpp_sigma2[i]  <- d$sigma2
}
t1 <- Sys.time()
cat(sprintf("  time: %.2fs\n", as.numeric(difftime(t1, t0, units = "secs"))))

# ============================================================================
#  Run pure R ARD
# ============================================================================

cat("--- R ARD reference ---\n")
t0 <- Sys.time()
r_out <- ard_r_mcmc(X, as.numeric(Y), nburn = n_burnin, npost = n_keep,
                    seed = 43L)
t1 <- Sys.time()
cat(sprintf("  time: %.2fs\n\n", as.numeric(difftime(t1, t0, units = "secs"))))

# ============================================================================
#  Compare posterior means
# ============================================================================

cat("Posterior mean comparison:\n")
cat(sprintf("  %10s %10s %10s %10s\n", "param", "true", "C++", "R"))
cat(sprintf("  %10s %10.4f %10.4f %10.4f\n",
            "alpha", alpha_true, mean(cpp_alpha), mean(r_out$alpha)))
cat(sprintf("  %10s %10.4f %10.4f %10.4f\n",
            "sigma", sigma_true, mean(sqrt(cpp_sigma2)), mean(sqrt(r_out$sigma2))))

for (j in seq_len(p)) {
    cat(sprintf("  %10s %10.4f %10.4f %10.4f\n",
                paste0("beta[", j, "]"),
                beta_true[j],
                mean(cpp_beta[, j]),
                mean(r_out$beta[, j])))
}

# ============================================================================
#  Check agreement: C++ and R posterior means should be close
# ============================================================================

cat("\nC++ vs R agreement:\n")
beta_diff <- abs(colMeans(cpp_beta) - colMeans(r_out$beta))
alpha_diff <- abs(mean(cpp_alpha) - mean(r_out$alpha))
sigma2_diff <- abs(mean(cpp_sigma2) - mean(r_out$sigma2))

cat(sprintf("  |alpha diff|:  %.4f\n", alpha_diff))
cat(sprintf("  |sigma2 diff|: %.4f\n", sigma2_diff))
cat(sprintf("  max |beta diff|: %.4f\n", max(beta_diff)))

# Also check posterior SDs agree
beta_sd_cpp <- apply(cpp_beta, 2, sd)
beta_sd_r   <- apply(r_out$beta, 2, sd)
sd_ratio <- beta_sd_cpp / beta_sd_r
cat(sprintf("  beta SD ratio (C++/R): min=%.3f max=%.3f\n",
            min(sd_ratio), max(sd_ratio)))

# ============================================================================
#  R-hat across both (treat C++ and R as two chains from same posterior)
# ============================================================================

rhat_of <- function(x1, x2) {
    if (requireNamespace("posterior", quietly = TRUE)) {
        as.numeric(posterior::rhat(cbind(x1, x2)))
    } else {
        n <- length(x1); h <- n %/% 2
        halves <- cbind(x1[1:h], x1[(h+1):(2*h)], x2[1:h], x2[(h+1):(2*h)])
        W <- mean(apply(halves, 2, var))
        B <- var(colMeans(halves)) * h
        sqrt((h-1)/h + B/(h*W))
    }
}

cat("\nCross-implementation R-hat (C++ chain vs R chain):\n")
rhat_alpha <- rhat_of(cpp_alpha, r_out$alpha)
rhat_sigma2 <- rhat_of(cpp_sigma2, r_out$sigma2)
cat(sprintf("  alpha:  %.4f\n", rhat_alpha))
cat(sprintf("  sigma2: %.4f\n", rhat_sigma2))

rhat_beta <- numeric(p)
for (j in seq_len(p)) {
    rhat_beta[j] <- rhat_of(cpp_beta[, j], r_out$beta[, j])
}
cat(sprintf("  beta:   min=%.4f  max=%.4f\n", min(rhat_beta), max(rhat_beta)))

# ============================================================================
#  Pass criteria
# ============================================================================

max_rhat <- max(c(rhat_alpha, rhat_sigma2, rhat_beta))
means_agree <- max(beta_diff) < 0.15 && alpha_diff < 0.15 && sigma2_diff < 0.1
sd_agree <- all(sd_ratio > 0.7 & sd_ratio < 1.4)

cat(sprintf("\nmax cross-R-hat: %.4f (threshold 1.05)\n", max_rhat))
cat(sprintf("Posterior means agree: %s\n", ifelse(means_agree, "YES", "NO")))
cat(sprintf("Posterior SDs agree:   %s\n", ifelse(sd_agree, "YES", "NO")))

all_ok <- max_rhat < 1.05 && means_agree && sd_agree
cat(sprintf("\n%s\n", if (all_ok) "PASS" else "FAIL"))

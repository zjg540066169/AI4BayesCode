# ----------------------------------------------------------------------------
# test_rhat_strict.R
#
# Strict R-hat < 1.01 test across all model types.
# Uses long chains (8000 burnin + 8000 keep) x 2 chains.
# Each model compiled once, then 2 chains run.
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

n_burnin <- 8000L
n_keep   <- 8000L
threshold <- 1.01

results <- list()
all_ok <- TRUE

# ============================================================================
#  1. GaussianLocationScale
# ============================================================================
cat("=== 1. GaussianLocationScale ===\n")
ai4bayescode_sourceCpp(file.path(script_dir, "GaussianLocationScale.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
y_gauss <- rnorm(100, 3, 1.5)

run_gaussian <- function(seed) {
    model <- new(GaussianLocationScale, y_gauss, as.integer(seed))
    model$step(n_burnin)
    mu <- numeric(n_keep); sigma <- numeric(n_keep)
    for (i in seq_len(n_keep)) {
        model$step(1L); d <- model$get_current()
        mu[i] <- d$mu; sigma[i] <- d$sigma
    }
    list(mu = mu, sigma = sigma)
}

t0 <- Sys.time()
g1 <- run_gaussian(101L); g2 <- run_gaussian(202L)
dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

rh_mu <- rhat_of(g1$mu, g2$mu)
rh_sigma <- rhat_of(g1$sigma, g2$sigma)
max_rh <- max(rh_mu, rh_sigma)
ok <- max_rh < threshold

cat(sprintf("  R-hat: mu=%.5f sigma=%.5f  max=%.5f  %s  (%.1fs)\n",
            rh_mu, rh_sigma, max_rh, ifelse(ok, "PASS", "FAIL"), dt))
if (!ok) all_ok <- FALSE
results[["Gaussian"]] <- max_rh

# ============================================================================
#  2. DirichletSimplex (K=5)
# ============================================================================
cat("\n=== 2. DirichletSimplex ===\n")
ai4bayescode_sourceCpp(file.path(script_dir, "DirichletSimplex.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

y_dir <- c(15, 25, 30, 20, 10)
alpha_dir <- rep(1, 5)

run_dirichlet <- function(seed) {
    model <- new(DirichletSimplex, as.numeric(y_dir), alpha_dir, as.integer(seed))
    model$step(n_burnin)
    draws <- matrix(NA, n_keep, 5)
    for (i in seq_len(n_keep)) {
        model$step(1L)
        draws[i, ] <- model$get_current()$theta
    }
    draws
}

t0 <- Sys.time()
d1 <- run_dirichlet(101L); d2 <- run_dirichlet(202L)
dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

rh_dir <- numeric(5)
for (j in 1:5) rh_dir[j] <- rhat_of(d1[, j], d2[, j])
max_rh <- max(rh_dir)
ok <- max_rh < threshold

cat(sprintf("  R-hat per component: %s\n",
            paste(sprintf("%.5f", rh_dir), collapse = " ")))
cat(sprintf("  max=%.5f  %s  (%.1fs)\n", max_rh, ifelse(ok, "PASS", "FAIL"), dt))
if (!ok) all_ok <- FALSE
results[["Dirichlet"]] <- max_rh

# ============================================================================
#  3. BartNoise
# ============================================================================
cat("\n=== 3. BartNoise ===\n")
ai4bayescode_sourceCpp(file.path(script_dir, "BartNoise.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
N_bart <- 200; p_bart <- 5
X_bart <- matrix(rnorm(N_bart * p_bart), ncol = p_bart)
f_true <- 10 * sin(pi * X_bart[, 1] * X_bart[, 2]) + 5 * X_bart[, 3]
y_bart <- f_true + rnorm(N_bart, 0, 1.0)

run_bart <- function(seed) {
    set.seed(seed)  # R's RNG for BART
    model <- new(BartNoise, X_bart, y_bart, 200L, 2.0, 2.0, 0.95, 3.0,
                 100L, FALSE, FALSE, as.integer(seed))
    model$step(n_burnin)
    sigma <- numeric(n_keep)
    # For R-hat on BART, use sigma + a few summary stats of f
    f_mean_draw <- numeric(n_keep)
    f_sd_draw <- numeric(n_keep)
    for (i in seq_len(n_keep)) {
        model$step(1L); d <- model$get_current()
        sigma[i] <- d$sigma
        f_mean_draw[i] <- mean(d$f_bart)
        f_sd_draw[i] <- sd(d$f_bart)
    }
    list(sigma = sigma, f_mean = f_mean_draw, f_sd = f_sd_draw)
}

t0 <- Sys.time()
b1 <- run_bart(101L); b2 <- run_bart(202L)
dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

rh_sigma <- rhat_of(b1$sigma, b2$sigma)
rh_fmean <- rhat_of(b1$f_mean, b2$f_mean)
rh_fsd   <- rhat_of(b1$f_sd, b2$f_sd)
max_rh <- max(rh_sigma, rh_fmean, rh_fsd)
ok <- max_rh < threshold

cat(sprintf("  R-hat: sigma=%.5f f_mean=%.5f f_sd=%.5f\n",
            rh_sigma, rh_fmean, rh_fsd))
cat(sprintf("  max=%.5f  %s  (%.1fs)\n", max_rh, ifelse(ok, "PASS", "FAIL"), dt))
cat(sprintf("  sigma: mean=%.4f (true 1.0)\n", mean(c(b1$sigma, b2$sigma))))
if (!ok) all_ok <- FALSE
results[["BART"]] <- max_rh

# ============================================================================
#  4. BetaBernoulli (interval constraint)
# ============================================================================
cat("\n=== 4. BetaBernoulli ===\n")
ai4bayescode_sourceCpp(file.path(script_dir, "BetaBernoulli.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
y_bern <- rbinom(200, 1, 0.3)

run_beta <- function(seed) {
    model <- new(BetaBernoulli, as.numeric(y_bern), 1.0, 1.0, as.integer(seed))
    model$step(n_burnin)
    p_draws <- numeric(n_keep)
    for (i in seq_len(n_keep)) {
        model$step(1L)
        p_draws[i] <- model$get_current()$p
    }
    p_draws
}

t0 <- Sys.time()
be1 <- run_beta(101L); be2 <- run_beta(202L)
dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

rh_p <- rhat_of(be1, be2)
ok <- rh_p < threshold

# Analytic: Beta(1 + sum(y), 1 + N - sum(y))
a_post <- 1 + sum(y_bern); b_post <- 1 + length(y_bern) - sum(y_bern)
analytic_mean <- a_post / (a_post + b_post)
sample_mean <- mean(c(be1, be2))

cat(sprintf("  R-hat: p=%.5f  %s  (%.1fs)\n", rh_p, ifelse(ok, "PASS", "FAIL"), dt))
cat(sprintf("  posterior mean: %.5f (analytic %.5f)\n", sample_mean, analytic_mean))
if (!ok) all_ok <- FALSE
results[["Beta"]] <- rh_p

# ============================================================================
#  6. ARDLasso (cross-implementation with R)
# ============================================================================
cat("\n=== 6. ARDLasso ===\n")
ai4bayescode_sourceCpp(file.path(script_dir, "ARDLasso.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
N_ard <- 100; p_ard <- 10
X_ard <- matrix(rnorm(N_ard * p_ard), ncol = p_ard)
beta_true_ard <- c(3, -2, 0, 0, 1.5, 0, 0, 0, 0, 0)
y_ard <- 2.0 + X_ard %*% beta_true_ard + rnorm(N_ard, 0, 1.0)

run_ard <- function(seed) {
    model <- new(ARDLasso, X_ard, as.numeric(y_ard), as.integer(seed))
    model$step(n_burnin)
    alpha <- numeric(n_keep); sigma2 <- numeric(n_keep)
    beta_draws <- matrix(NA, n_keep, p_ard)
    for (i in seq_len(n_keep)) {
        model$step(1L); d <- model$get_current()
        alpha[i] <- d$alpha; sigma2[i] <- d$sigma2
        beta_draws[i, ] <- d$beta
    }
    list(alpha = alpha, sigma2 = sigma2, beta = beta_draws)
}

t0 <- Sys.time()
a1 <- run_ard(101L); a2 <- run_ard(202L)
dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

rh_alpha <- rhat_of(a1$alpha, a2$alpha)
rh_sigma2 <- rhat_of(a1$sigma2, a2$sigma2)
rh_beta <- numeric(p_ard)
for (j in seq_len(p_ard)) rh_beta[j] <- rhat_of(a1$beta[, j], a2$beta[, j])
max_rh <- max(rh_alpha, rh_sigma2, rh_beta)
ok <- max_rh < threshold

cat(sprintf("  R-hat: alpha=%.5f sigma2=%.5f  beta_max=%.5f\n",
            rh_alpha, rh_sigma2, max(rh_beta)))
cat(sprintf("  max=%.5f  %s  (%.1fs)\n", max_rh, ifelse(ok, "PASS", "FAIL"), dt))
if (!ok) all_ok <- FALSE
results[["ARD"]] <- max_rh

# ============================================================================
#  Summary
# ============================================================================
cat("\n============================================================\n")
cat(sprintf("  %-20s  max R-hat    status\n", "Model"))
cat("------------------------------------------------------------\n")
for (nm in names(results)) {
    rh <- results[[nm]]
    cat(sprintf("  %-20s  %.5f      %s\n", nm, rh,
                ifelse(rh < threshold, "PASS", "FAIL")))
}
cat("------------------------------------------------------------\n")
cat(sprintf("  Threshold: %.3f\n", threshold))
cat(sprintf("\n%s\n", if (all_ok) "ALL PASS" else "SOME FAILED"))

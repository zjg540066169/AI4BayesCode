# ---------------------------------------------------------------------------
# test_predict_at_additional.R
#
# Additional predict_at + loo tests for models not in
# test_posterior_predictive.R:
#  - DirichletSimplex (Multinomial observation)
#  - DirichletSparse (Multinomial with hyperprior)
#  - DirichletHierarchical (Dirichlet observation)
#  - BartNoise with X_new (predict at test points)
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

library(loo)

# ============================================================================
# 1. DirichletSimplex + loo (Multinomial log-likelihood)
# ============================================================================
cat("\n======== DirichletSimplex ========\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "DirichletSimplex.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

y_dir <- c(15, 25, 30, 20, 10)
alpha_dir <- rep(1, 5)

run_dir <- function(seed, n_burnin, n_keep) {
    m <- new(DirichletSimplex, as.numeric(y_dir), alpha_dir,
             as.integer(seed), TRUE)
    m$step(n_burnin)
    m$step(n_keep)
    list(m = m, pp = m$predict_at(list()), hist = m$get_history())
}

for (seed in c(101L, 202L)) {
    chain <- run_dir(seed, 500L, 2000L)
    y_rep <- chain$pp$y_rep    # (n_draws x 5)
    # Posterior predictive check: column means of y_rep vs y_dir
    pm <- colMeans(y_rep)
    cat(sprintf("  chain %d  obs counts: %s\n",
                seed, paste(y_dir, collapse = " ")))
    cat(sprintf("         y_rep means:  %s\n",
                paste(sprintf("%.1f", pm), collapse = " ")))
    # Bayesian p-value on Chi-square-style discrepancy
    expected <- y_dir
    T_obs <- sum((y_dir - expected)^2 / (expected + 1))
    T_rep <- apply(y_rep, 1, function(r) sum((r - expected)^2 / (expected + 1)))
    pv <- mean(T_rep >= T_obs)
    cat(sprintf("         chi2-style bp=%.3f\n", pv))
    assign(sprintf("dir_c%d", seed), chain)
}

# ============================================================================
# 2. DirichletSparse
# ============================================================================
cat("\n======== DirichletSparse ========\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "DirichletSparse.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
P <- 50L; N_total <- 500L
s_true <- c(rep(1/5, 5), rep(1e-5, P - 5))
s_true <- s_true / sum(s_true)
y_sparse <- as.numeric(tabulate(
    sample.int(P, N_total, prob = s_true, replace = TRUE), nbins = P))

run_sp <- function(seed, n_burnin, n_keep) {
    m <- new(DirichletSparse, y_sparse, as.integer(seed), TRUE)
    m$step(n_burnin)
    m$step(n_keep)
    list(m = m, pp = m$predict_at(list()), hist = m$get_history())
}

for (seed in c(101L, 202L)) {
    chain <- run_sp(seed, 300L, 1000L)
    y_rep <- chain$pp$y_rep    # n_draws x P
    # Just check that the top-5 categories are well-recovered
    pm <- colMeans(y_rep)
    top_obs <- order(y_sparse, decreasing = TRUE)[1:5]
    top_pred <- order(pm, decreasing = TRUE)[1:5]
    overlap <- length(intersect(top_obs, top_pred))
    cat(sprintf("  chain %d  top-5 overlap (obs vs pp mean) = %d/5\n",
                seed, overlap))
}

# ============================================================================
# 3. DirichletHierarchical
# ============================================================================
cat("\n======== DirichletHierarchical ========\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "DirichletHierarchical.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
K <- 4L; P_obs <- 20L
s_true_dh <- c(0.1, 0.3, 0.4, 0.2)
kappa_true <- 20.0
# Generate K x P matrix of simplex observations from Dir(kappa_true * s_true)
S_obs <- matrix(NA, nrow = K, ncol = P_obs)
for (i in seq_len(P_obs)) {
    g <- rgamma(K, shape = kappa_true * s_true_dh)
    S_obs[, i] <- g / sum(g)
}

run_dh <- function(seed, n_burnin, n_keep) {
    m <- new(DirichletHierarchical, S_obs, 1.0, 1.0, as.integer(seed), TRUE)
    m$step(n_burnin)
    m$step(n_keep)
    list(m = m, pp = m$predict_at(list()), hist = m$get_history())
}

for (seed in c(101L, 202L)) {
    chain <- run_dh(seed, 500L, 1000L)
    y_rep <- chain$pp$y_rep    # n_draws x K simplex obs
    pm <- colMeans(y_rep)
    cat(sprintf("  chain %d  true s:        %s\n", seed,
                paste(sprintf("%.3f", s_true_dh), collapse = " ")))
    cat(sprintf("         pp mean y_rep:  %s\n",
                paste(sprintf("%.3f", pm), collapse = " ")))
    cat(sprintf("         posterior s mean: %s\n",
                paste(sprintf("%.3f", colMeans(chain$hist$s)), collapse = " ")))
}

# ============================================================================
# 4. GBartPoisson — predict_at at new X returns r (log-rate) + rate
# ============================================================================
cat("\n======== GBartPoisson (genBART Poisson at test X) ========\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "GBartPoisson.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
N_lbp <- 80; p_lbp <- 3
X_lbp <- matrix(runif(N_lbp * p_lbp), ncol = p_lbp)
r_true <- 1 + 2 * X_lbp[, 1] - X_lbp[, 2]
y_lbp  <- rpois(N_lbp, exp(r_true))

m <- new(GBartPoisson, X_lbp, as.numeric(y_lbp), 50L, 101L, TRUE)
m$step(500L)
m$step(500L)

# predict_at with X_new: returns live-tree r + rate at test points.
X_test <- matrix(runif(20 * p_lbp), ncol = p_lbp)
pred <- m$predict_at(list(X = X_test))
cat(sprintf("  GBart names: %s\n", paste(names(pred), collapse = ", ")))
cat(sprintf("  r length: %d; rate length: %d\n",
            length(pred$r), length(pred$rate)))
stopifnot(length(pred$r) == nrow(X_test))
stopifnot(length(pred$rate) == nrow(X_test))
stopifnot(all(pred$rate > 0))
cat("  r + rate returned correctly at test X.\n")

# ============================================================================
# 5. BartNoise with X_new — posterior predictive at test points
# ============================================================================
cat("\n======== BartNoise w/ X_new ========\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "BartNoise.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
N_b <- 100; p_b <- 3
X_train <- matrix(rnorm(N_b * p_b), ncol = p_b)
f_true_b <- 3 * X_train[, 1] + X_train[, 2]^2
y_train <- f_true_b + rnorm(N_b, 0, 0.8)

# Test points
X_test_b <- matrix(rnorm(30 * p_b), ncol = p_b)
f_true_test <- 3 * X_test_b[, 1] + X_test_b[, 2]^2

# Train with keep_history
set.seed(101); m_b <- new(BartNoise, X_train, y_train, 100L, 2.0, 2.0, 0.95,
                          3.0, 100L, FALSE, FALSE, 101L, TRUE)
m_b$step(500L); m_b$step(1000L)

# predict with X_new
pred_b <- m_b$predict_at(list(X = X_test_b))
cat(sprintf("  names: %s\n", paste(names(pred_b), collapse = ", ")))
cat(sprintf("  f_bart shape: %d x %d  (draws x N_test)\n",
            nrow(pred_b$f_bart), ncol(pred_b$f_bart)))
cat(sprintf("  y_rep  shape: %d x %d\n",
            nrow(pred_b$y_rep), ncol(pred_b$y_rep)))

# Compare posterior mean f_bart at test points to truth
post_mean_f <- colMeans(pred_b$f_bart)
rmse_mean   <- sqrt(mean((post_mean_f - f_true_test)^2))
cat(sprintf("  RMSE(post-mean f_bart, true f(X_new)) = %.3f\n", rmse_mean))

# y_rep should have extra noise ~ sigma
post_sd_yrep <- sd(as.vector(pred_b$y_rep - pred_b$f_bart))
cat(sprintf("  sd(y_rep - f_bart) = %.3f   (true sigma = 0.80)\n", post_sd_yrep))

# predict with empty list (posterior predictive at TRAINING X)
pred_tr <- m_b$predict_at(list())
cat(sprintf("  training-X pp names: %s\n",
            paste(names(pred_tr), collapse = ", ")))
cat(sprintf("  y_rep shape: %d x %d  (draws x N_train)\n",
            nrow(pred_tr$y_rep), ncol(pred_tr$y_rep)))

cat("\n=== DONE ===\n")

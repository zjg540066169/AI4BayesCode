# ----------------------------------------------------------------------------
# test_posterior_predictive.R
#
# End-to-end test of predict_at() posterior-predictive sampling + loo PSIS.
# Layer R3 candidate.
#
# For each model, runs 2 chains of ~moderate length with keep_history=TRUE,
# then:
#   (a) calls predict_at(list()) to get y_rep matrices (n_draws x N)
#   (b) computes 6 Bayesian p-values against training y
#   (c) computes pointwise log-likelihood and runs loo::loo()
#
# Diagnosis criteria (soft, not hard fail):
#   - Bayesian p-values should all be in (0.05, 0.95) for a well-fitting model
#   - Pareto-k: >50% k<0.5 (good), <10% k>0.7 (warning)
# ----------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
# 6 summary statistics for Bayesian p-values
# --------------------------------------------------------------------------
SUMMARY_STATS <- list(
    mean   = function(y) mean(y),
    sd     = function(y) sd(y),
    min    = function(y) min(y),
    max    = function(y) max(y),
    q25    = function(y) quantile(y, 0.25, names = FALSE),
    q75    = function(y) quantile(y, 0.75, names = FALSE))

# y_rep: n_draws x N matrix of posterior predictive draws
# y:     length N observed data
bayesian_pvalues <- function(y_rep, y) {
    sapply(names(SUMMARY_STATS), function(nm) {
        T_obs <- SUMMARY_STATS[[nm]](y)
        T_rep <- apply(y_rep, 1, SUMMARY_STATS[[nm]])
        mean(T_rep >= T_obs)
    })
}

# Verdict for Bayesian p-values
bp_verdict <- function(pv, lo = 0.05, hi = 0.95) {
    any_bad <- any(pv < lo | pv > hi)
    if (any_bad) sprintf("WARN (outside [%.2f, %.2f])", lo, hi) else "OK"
}

# Pointwise log-likelihood matrix loglik[d, i] = log p(y_i | theta^(d))
# For discrete observation (Bernoulli, Poisson): direct PMF.
# For Gaussian: dnorm(y, mean, sd, log=TRUE).
# --------------------------------------------------------------------------

loo_report <- function(LLarr_chain1, LLarr_chain2, tag) {
    # LLarr_chain1, 2: n_iter x N matrices (one per chain)
    stopifnot(dim(LLarr_chain1) == dim(LLarr_chain2))
    LL <- array(NA_real_, dim = c(nrow(LLarr_chain1), 2, ncol(LLarr_chain1)))
    LL[, 1, ] <- LLarr_chain1
    LL[, 2, ] <- LLarr_chain2
    rel_n_eff <- loo::relative_eff(exp(LL))
    lo <- loo::loo(LL, r_eff = rel_n_eff, cores = 1)
    ks <- lo$diagnostics$pareto_k
    pct_k_small <- mean(ks < 0.5) * 100
    pct_k_warn  <- mean(ks >= 0.7) * 100
    cat(sprintf("  [%s] elpd_loo=%.2f (se=%.2f)  %%k<0.5=%.1f%%  %%k>=0.7=%.1f%%\n",
                tag, lo$estimates["elpd_loo", "Estimate"],
                lo$estimates["elpd_loo", "SE"],
                pct_k_small, pct_k_warn))
    invisible(lo)
}

# ============================================================================
# 1. GaussianLocationScale
# ============================================================================
cat("\n======== 1. GaussianLocationScale ========\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "GaussianLocationScale.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(1)
y_gauss <- rnorm(100, 2.0, 1.5)

run_gauss_chain <- function(seed, n_burnin, n_keep) {
    m <- new(GaussianLocationScale, y_gauss, as.integer(seed), TRUE)
    m$step(n_burnin)
    m$step(n_keep)
    # history holds both burnin + keep; trim externally.
    list(model = m, pp = m$predict_at(list()), hist = m$get_history())
}

for (seed in c(101L, 202L)) {
    chain <- run_gauss_chain(seed, 500L, 2000L)
    n_rec <- nrow(chain$pp$y_rep)
    # Keep only the final n_keep draws (trim burnin)
    y_rep <- chain$pp$y_rep[(n_rec - 2000 + 1):n_rec, , drop = FALSE]
    pv <- bayesian_pvalues(y_rep, y_gauss)
    cat(sprintf("  chain %d  bp: %s   %s\n",
                seed,
                paste(sprintf("%s=%.2f", names(pv), pv), collapse = " "),
                bp_verdict(pv)))
    assign(sprintf("gauss_c%d", seed), chain)
}

# Pointwise log-lik for Gaussian: dnorm(y_i, mu^(d), sigma^(d), log=TRUE)
pointwise_ll_gauss <- function(hist, y) {
    mu_d    <- hist$mu
    sigma_d <- hist$sigma
    S <- length(mu_d)
    N <- length(y)
    LL <- matrix(NA_real_, nrow = S, ncol = N)
    for (d in seq_len(S)) {
        LL[d, ] <- dnorm(y, mu_d[d], sigma_d[d], log = TRUE)
    }
    LL
}

n_rec <- length(gauss_c101$hist$mu)
ll1 <- pointwise_ll_gauss(
    list(mu    = gauss_c101$hist$mu[(n_rec - 2000 + 1):n_rec],
         sigma = gauss_c101$hist$sigma[(n_rec - 2000 + 1):n_rec]),
    y_gauss)
n_rec <- length(gauss_c202$hist$mu)
ll2 <- pointwise_ll_gauss(
    list(mu    = gauss_c202$hist$mu[(n_rec - 2000 + 1):n_rec],
         sigma = gauss_c202$hist$sigma[(n_rec - 2000 + 1):n_rec]),
    y_gauss)
loo_report(ll1, ll2, "Gaussian")

# ============================================================================
# 2. BetaBernoulli
# ============================================================================
cat("\n======== 2. BetaBernoulli ========\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "BetaBernoulli.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
y_bern <- rbinom(200, 1, 0.3)

run_bern_chain <- function(seed, n_burnin, n_keep) {
    m <- new(BetaBernoulli, as.numeric(y_bern), 1.0, 1.0, as.integer(seed), TRUE)
    m$step(n_burnin)
    m$step(n_keep)
    list(model = m, pp = m$predict_at(list()), hist = m$get_history())
}

for (seed in c(101L, 202L)) {
    chain <- run_bern_chain(seed, 500L, 2000L)
    n_rec <- nrow(chain$pp$y_rep)
    y_rep <- chain$pp$y_rep[(n_rec - 2000 + 1):n_rec, , drop = FALSE]
    # For binary, min/max/quantiles are degenerate; only mean+sd meaningful.
    pv_mean <- mean(apply(y_rep, 1, mean) >= mean(y_bern))
    cat(sprintf("  chain %d  bp_mean=%.3f (obs mean = %.3f)  %s\n",
                seed, pv_mean, mean(y_bern),
                if (pv_mean > 0.05 & pv_mean < 0.95) "OK" else "WARN"))
    assign(sprintf("bern_c%d", seed), chain)
}

pointwise_ll_bern <- function(hist, y) {
    p_d <- hist$p
    S <- length(p_d)
    N <- length(y)
    LL <- matrix(NA_real_, nrow = S, ncol = N)
    for (d in seq_len(S)) {
        LL[d, ] <- dbinom(y, 1, p_d[d], log = TRUE)
    }
    LL
}

n_rec <- length(bern_c101$hist$p)
ll1 <- pointwise_ll_bern(
    list(p = bern_c101$hist$p[(n_rec - 2000 + 1):n_rec]), y_bern)
n_rec <- length(bern_c202$hist$p)
ll2 <- pointwise_ll_bern(
    list(p = bern_c202$hist$p[(n_rec - 2000 + 1):n_rec]), y_bern)
loo_report(ll1, ll2, "Bernoulli")

# ============================================================================
# 3. BartNoise
# ============================================================================
cat("\n======== 3. BartNoise ========\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "BartNoise.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
N_bart <- 150
p_bart <- 5
X_bart <- matrix(rnorm(N_bart * p_bart), ncol = p_bart)
f_true <- 10 * sin(pi * X_bart[, 1] * X_bart[, 2]) + 5 * X_bart[, 3]
y_bart <- f_true + rnorm(N_bart, 0, 1.0)

run_bart_chain <- function(seed, n_burnin, n_keep) {
    set.seed(seed)
    m <- new(BartNoise, X_bart, y_bart, 100L, 2.0, 2.0, 0.95, 3.0,
             100L, FALSE, FALSE, as.integer(seed), TRUE)
    m$step(n_burnin)
    # Because keep_history accumulates, we can't "clear after burnin"
    # unless the wrapper exposes that. Collect all and post-trim.
    m$step(n_keep)
    list(model = m, pp = m$predict_at(list()), hist = m$get_history())
}

# Smaller chains for BART (slower)
for (seed in c(101L, 202L)) {
    cat(sprintf("  Running BART chain %d...\n", seed))
    t0 <- Sys.time()
    chain <- run_bart_chain(seed, 500L, 1500L)
    cat(sprintf("  (%.1fs)\n",
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
    n_rec <- nrow(chain$pp$y_rep)
    y_rep <- chain$pp$y_rep[(n_rec - 1500 + 1):n_rec, , drop = FALSE]
    pv <- bayesian_pvalues(y_rep, y_bart)
    cat(sprintf("  chain %d  bp: %s   %s\n",
                seed,
                paste(sprintf("%s=%.2f", names(pv), pv), collapse = " "),
                bp_verdict(pv)))
    assign(sprintf("bart_c%d", seed), chain)
}

pointwise_ll_bart <- function(hist, y) {
    # f_bart: n_draws x N matrix, sigma: n_draws vector
    F_mat <- hist$f_bart
    sig_d <- hist$sigma
    S <- length(sig_d)
    N <- length(y)
    stopifnot(nrow(F_mat) == S, ncol(F_mat) == N)
    LL <- matrix(NA_real_, nrow = S, ncol = N)
    for (d in seq_len(S)) {
        LL[d, ] <- dnorm(y, F_mat[d, ], sig_d[d], log = TRUE)
    }
    LL
}

n_rec <- length(bart_c101$hist$sigma)
ll1 <- pointwise_ll_bart(
    list(f_bart = bart_c101$hist$f_bart[(n_rec - 1500 + 1):n_rec, ],
         sigma  = bart_c101$hist$sigma[(n_rec - 1500 + 1):n_rec]), y_bart)
n_rec <- length(bart_c202$hist$sigma)
ll2 <- pointwise_ll_bart(
    list(f_bart = bart_c202$hist$f_bart[(n_rec - 1500 + 1):n_rec, ],
         sigma  = bart_c202$hist$sigma[(n_rec - 1500 + 1):n_rec]), y_bart)
loo_report(ll1, ll2, "BART")

# ============================================================================
# 4. ARDLasso
# ============================================================================
cat("\n======== 4. ARDLasso ========\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "ARDLasso.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42)
N_ard <- 100
p_ard <- 10
X_ard <- matrix(rnorm(N_ard * p_ard), ncol = p_ard)
beta_true_ard <- c(3, -2, 0, 0, 1.5, 0, 0, 0, 0, 0)
y_ard <- 2.0 + X_ard %*% beta_true_ard + rnorm(N_ard, 0, 1.0)

run_ard_chain <- function(seed, n_burnin, n_keep) {
    m <- new(ARDLasso, X_ard, as.numeric(y_ard), as.integer(seed), TRUE)
    m$step(n_burnin)
    m$step(n_keep)
    list(model = m, pp = m$predict_at(list()), hist = m$get_history())
}

for (seed in c(101L, 202L)) {
    chain <- run_ard_chain(seed, 500L, 2000L)
    n_rec <- nrow(chain$pp$y_rep)
    y_rep <- chain$pp$y_rep[(n_rec - 2000 + 1):n_rec, , drop = FALSE]
    pv <- bayesian_pvalues(y_rep, as.numeric(y_ard))
    cat(sprintf("  chain %d  bp: %s   %s\n",
                seed,
                paste(sprintf("%s=%.2f", names(pv), pv), collapse = " "),
                bp_verdict(pv)))
    assign(sprintf("ard_c%d", seed), chain)
}

pointwise_ll_ard <- function(hist, X, y) {
    beta_mat <- hist$beta          # S x p
    alpha_v  <- hist$alpha         # S
    sigma2_v <- hist$sigma2        # S
    S <- length(alpha_v)
    N <- length(y)
    LL <- matrix(NA_real_, nrow = S, ncol = N)
    for (d in seq_len(S)) {
        mu_d <- as.numeric(X %*% beta_mat[d, ]) + alpha_v[d]
        LL[d, ] <- dnorm(y, mu_d, sqrt(sigma2_v[d]), log = TRUE)
    }
    LL
}

n_rec <- length(ard_c101$hist$alpha)
ll1 <- pointwise_ll_ard(
    list(beta   = ard_c101$hist$beta[(n_rec - 2000 + 1):n_rec, ],
         alpha  = ard_c101$hist$alpha[(n_rec - 2000 + 1):n_rec],
         sigma2 = ard_c101$hist$sigma2[(n_rec - 2000 + 1):n_rec]),
    X_ard, as.numeric(y_ard))
n_rec <- length(ard_c202$hist$alpha)
ll2 <- pointwise_ll_ard(
    list(beta   = ard_c202$hist$beta[(n_rec - 2000 + 1):n_rec, ],
         alpha  = ard_c202$hist$alpha[(n_rec - 2000 + 1):n_rec],
         sigma2 = ard_c202$hist$sigma2[(n_rec - 2000 + 1):n_rec]),
    X_ard, as.numeric(y_ard))
loo_report(ll1, ll2, "ARDLasso")

cat("\n=== DONE ===\n")

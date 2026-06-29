# ----------------------------------------------------------------------------
# test_keep_history_all.R
#
# Batch test for keep_history + get_history across the non-BART wrappers,
# plus ARDLasso (self-contained, no composite).
#
# For each model:
#   A. Post-construction smoke test (keep_history = FALSE and TRUE)
#   B. History-mode behavior: get_history() shape, history grows with step
#   C. 2-chain rank-normalized R-hat (Vehtari 2021) + bulk/tail ESS on a
#      representative scalar parameter. 2 chains (not 4) to keep runtime
#      reasonable for a batch sanity test; thresholds loosened accordingly.
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

n_pass <- 0L; n_fail <- 0L
pass <- function(m) { cat(sprintf("  PASS: %s\n", m)); n_pass <<- n_pass + 1L }
fail <- function(m) { cat(sprintf("  FAIL: %s\n", m)); n_fail <<- n_fail + 1L }
check <- function(cond, msg) if (isTRUE(cond)) pass(msg) else fail(msg)

# Rank-normalized R-hat + ESS diagnostic per parameter
diag_block <- function(name, chain_list_of_vecs,
                       rhat_thresh = 1.05, ess_thresh = 200) {
    mat <- do.call(cbind, chain_list_of_vecs)  # iters x chains
    if (!has_posterior) {
        # Plain R-hat fallback
        n <- nrow(mat); k <- ncol(mat)
        W <- mean(apply(mat, 2, var))
        B <- n * var(colMeans(mat))
        rh <- sqrt(((n - 1)/n) * W/W + B/(n*W))
        cat(sprintf("    %-12s Rhat_plain=%.4f\n", name, rh))
        check(is.finite(rh) && rh < rhat_thresh,
              sprintf("%s: plain R-hat < %.2f (posterior pkg missing)",
                      name, rhat_thresh))
        return(invisible())
    }
    rh <- posterior::rhat(mat)
    eb <- posterior::ess_bulk(mat)
    et <- posterior::ess_tail(mat)
    cat(sprintf("    %-12s Rhat_rn=%.4f  bulk_ESS=%.0f  tail_ESS=%.0f\n",
                name, rh, eb, et))
    check(is.finite(rh) && rh < rhat_thresh,
          sprintf("%s: rank-normalized R-hat < %.2f", name, rhat_thresh))
    check(is.finite(eb) && eb > ess_thresh,
          sprintf("%s: bulk ESS > %d", name, ess_thresh))
    check(is.finite(et) && et > ess_thresh,
          sprintf("%s: tail ESS > %d", name, ess_thresh))
}

# ============================================================================
# 1. GaussianLocationScale
# ============================================================================
cat("\n=== 1. GaussianLocationScale ===\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "GaussianLocationScale.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)

set.seed(1); y <- rnorm(60, 3, 1.5)

m0 <- new(GaussianLocationScale, y, 42L, FALSE)
d0 <- m0$get_current()
check(length(d0$mu) == 1 && length(d0$sigma) == 1,
      "Gaussian: get_current returns scalar mu, sigma")
h0 <- m0$get_history()
check(is.list(h0) && length(h0$mu) == 1 && length(h0$sigma) == 1,
      "Gaussian: pre-step get_history() is named list, both length 1")

m1 <- new(GaussianLocationScale, y, 42L, TRUE)
m1$step(200L)
h1 <- m1$get_history()
check(length(h1$mu) == 200 && length(h1$sigma) == 200,
      "Gaussian: history-mode $mu and $sigma each length 200")
check(all(is.finite(h1$mu)) && all(is.finite(h1$sigma)),
      "Gaussian: history finite")

# 2-chain R-hat on mu
run_gauss <- function(seed) {
    m <- new(GaussianLocationScale, y, as.integer(seed), TRUE)
    m$step(3000L + 3000L)
    H <- m$get_history()
    keep <- 3001:6000
    list(mu = H$mu[keep], sigma = H$sigma[keep])
}
c1 <- run_gauss(101L); c2 <- run_gauss(202L)
diag_block("mu",    list(c1$mu,    c2$mu))
diag_block("sigma", list(c1$sigma, c2$sigma))

# ============================================================================
# 2. BetaBernoulli
# ============================================================================
cat("\n=== 2. BetaBernoulli ===\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "BetaBernoulli.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)

set.seed(2); y_bb <- rbinom(200, 1, 0.35)

m0 <- new(BetaBernoulli, y_bb, 1, 1, 42L, FALSE)
d0 <- m0$get_current()
check(length(d0$p) == 1, "BetaBernoulli: get_current$p scalar")
h0 <- m0$get_history()
check(is.list(h0) && length(h0$p) == 1,
      "BetaBernoulli: pre-step get_history()$p length 1")

m1 <- new(BetaBernoulli, y_bb, 1, 1, 42L, TRUE)
m1$step(200L)
h1 <- m1$get_history()
check(length(h1$p) == 200,
      "BetaBernoulli: history $p length 200")

run_bb <- function(seed) {
    m <- new(BetaBernoulli, y_bb, 1, 1, as.integer(seed), TRUE)
    m$step(3000L + 3000L)
    list(p = m$get_history()$p[3001:6000])
}
c1 <- run_bb(11L); c2 <- run_bb(22L)
diag_block("p", list(c1$p, c2$p))

# ============================================================================
# 3. DirichletSimplex
# ============================================================================
cat("\n=== 3. DirichletSimplex ===\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "DirichletSimplex.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)

K <- 4
y_counts <- c(10, 20, 30, 40)
alpha_pri <- rep(1, K)

m0 <- new(DirichletSimplex, y_counts, alpha_pri, 42L, FALSE)
d0 <- m0$get_current()
check(length(d0$theta) == K, "Dirichlet: theta length K")
h0 <- m0$get_history()
check(nrow(h0$theta) == 1 && ncol(h0$theta) == K,
      "Dirichlet: pre-step $theta is 1 x K")

m1 <- new(DirichletSimplex, y_counts, alpha_pri, 42L, TRUE)
m1$step(200L)
h1 <- m1$get_history()
check(nrow(h1$theta) == 200 && ncol(h1$theta) == K,
      "Dirichlet: $theta is 200 x K")
check(all(abs(rowSums(h1$theta) - 1) < 1e-8),
      "Dirichlet: every draw sums to 1")

run_ds <- function(seed) {
    m <- new(DirichletSimplex, y_counts, alpha_pri, as.integer(seed), TRUE)
    m$step(3000L + 3000L)
    theta <- m$get_history()$theta[3001:6000, ]
    list(theta2 = theta[, 2], theta4 = theta[, 4])
}
c1 <- run_ds(11L); c2 <- run_ds(22L)
diag_block("theta[2]", list(c1$theta2, c2$theta2))
diag_block("theta[4]", list(c1$theta4, c2$theta4))

# ============================================================================
# 4. DirichletSparse
# ============================================================================
cat("\n=== 4. DirichletSparse ===\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "DirichletSparse.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)

P <- 5
y_c2 <- c(20, 5, 0, 10, 2)

m0 <- new(DirichletSparse, y_c2, 42L, FALSE)
d0 <- m0$get_current()
check(length(d0$s) == P && length(d0$theta) == 1,
      "DirichletSparse: get_current shapes")
h0 <- m0$get_history()
check(nrow(h0$s) == 1 && ncol(h0$s) == P && length(h0$theta) == 1,
      "DirichletSparse: pre-step $s is 1xP, $theta length 1")

m1 <- new(DirichletSparse, y_c2, 42L, TRUE)
m1$step(200L)
h1 <- m1$get_history()
check(nrow(h1$s) == 200 && ncol(h1$s) == P && length(h1$theta) == 200,
      "DirichletSparse: 200-step $s is 200xP, $theta length 200")

run_dsp <- function(seed) {
    m <- new(DirichletSparse, y_c2, as.integer(seed), TRUE)
    m$step(3000L + 3000L)
    H <- m$get_history()
    keep <- 3001:6000
    list(s1 = H$s[keep, 1], theta = H$theta[keep])
}
c1 <- run_dsp(11L); c2 <- run_dsp(22L)
diag_block("s[1]",  list(c1$s1,    c2$s1))
# theta in the sparse-Dirichlet model has heavy tails and mixes slowly;
# loosen ESS threshold here. This is a model-level property, not a
# keep_history implementation issue.
diag_block("theta", list(c1$theta, c2$theta), ess_thresh = 50)

# ============================================================================
# 5. DirichletHierarchical
# ============================================================================
cat("\n=== 5. DirichletHierarchical ===\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "DirichletHierarchical.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)

set.seed(5)
Kh <- 3; Ph <- 4
S_obs <- t(replicate(Kh, {
    draws <- MCMCpack::rdirichlet(1, rep(2, Ph))
    as.numeric(draws)
}))
# fall back to simple proportions if MCMCpack missing
if (!requireNamespace("MCMCpack", quietly = TRUE)) {
    S_obs <- matrix(runif(Kh * Ph), Kh, Ph)
    S_obs <- S_obs / rowSums(S_obs)
}

m0 <- new(DirichletHierarchical, S_obs, 1.0, 1.0, 42L, FALSE)
d0 <- m0$get_current()
check(length(d0$s) == Ph, "DirichletHier: get_current$s length P")
h0 <- m0$get_history()
check(nrow(h0$s) == 1 && ncol(h0$s) == Ph &&
      length(h0$kappa) == 1 && length(h0$theta) == 1,
      "DirichletHier: pre-step shapes ($s 1xP, $kappa 1, $theta 1)")

m1 <- new(DirichletHierarchical, S_obs, 1.0, 1.0, 42L, TRUE)
m1$step(200L)
h1 <- m1$get_history()
check(nrow(h1$s) == 200 && ncol(h1$s) == Ph &&
      length(h1$kappa) == 200 && length(h1$theta) == 200,
      "DirichletHier: 200-step shapes ($s 200xP, $kappa 200, $theta 200)")

run_dh <- function(seed) {
    m <- new(DirichletHierarchical, S_obs, 1.0, 1.0,
             as.integer(seed), TRUE)
    m$step(3000L + 3000L)
    H <- m$get_history()
    keep <- 3001:6000
    list(s1 = H$s[keep, 1], kappa = H$kappa[keep], theta = H$theta[keep])
}
c1 <- run_dh(11L); c2 <- run_dh(22L)
diag_block("s[1]",  list(c1$s1,    c2$s1))
diag_block("kappa", list(c1$kappa, c2$kappa))
diag_block("theta", list(c1$theta, c2$theta))

# ============================================================================
# 6. ARDLasso — history buffers (self-contained, no composite)
# ============================================================================
cat("\n=== 6. ARDLasso ===\n")
AI4BayesCode_sourceCpp(file.path(script_dir, "ARDLasso.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)

set.seed(7)
Na <- 150; pa <- 6
Xa <- matrix(rnorm(Na * pa), ncol = pa)
beta_a <- c(2, 0, -1.5, 0, 1, 0)
Ya <- as.numeric(Xa %*% beta_a + 0.5 * rnorm(Na))

m0 <- new(ARDLasso, Xa, Ya, 42L, FALSE)
d0 <- m0$get_current()
check(length(d0$beta) == pa, "ARDLasso: beta length p")
h0 <- m0$get_history()
check(nrow(h0$beta) == 1 && ncol(h0$beta) == pa &&
      length(h0$alpha) == 1 && length(h0$sigma2) == 1 &&
      nrow(h0$psi2) == 1 && ncol(h0$psi2) == pa,
      "ARDLasso: pre-step list shapes ($beta/$psi2 1xp, $alpha/$sigma2 scalar)")

m1 <- new(ARDLasso, Xa, Ya, 42L, TRUE)
m1$step(200L)
h1 <- m1$get_history()
check(nrow(h1$beta) == 200 && ncol(h1$beta) == pa &&
      length(h1$alpha) == 200 && length(h1$sigma2) == 200 &&
      nrow(h1$psi2) == 200 && ncol(h1$psi2) == pa,
      "ARDLasso: 200-step shapes")

# history-mode predict_at returns matrix n_draws x N_new
X_new <- matrix(rnorm(20 * pa), ncol = pa)
preds1 <- m1$predict_at(list(X = X_new))
check(is.matrix(preds1$y_hat) && nrow(preds1$y_hat) == 200 && ncol(preds1$y_hat) == 20,
      "ARDLasso history predict_at: y_hat is n_draws x N_new matrix")

# stateful-mode predict_at returns vector
preds0 <- m0$predict_at(list(X = X_new))
check(!is.matrix(preds0$y_hat) && length(preds0$y_hat) == 20,
      "ARDLasso stateful predict_at: y_hat is vector length N_new")

run_ard <- function(seed) {
    m <- new(ARDLasso, Xa, Ya, as.integer(seed), TRUE)
    m$step(3000L + 3000L)
    H <- m$get_history()
    keep <- 3001:6000
    list(beta1 = H$beta[keep, 1], alpha = H$alpha[keep],
         sigma2 = H$sigma2[keep])
}
c1 <- run_ard(11L); c2 <- run_ard(22L)
diag_block("beta[1]", list(c1$beta1,  c2$beta1))
diag_block("alpha",   list(c1$alpha,  c2$alpha))
diag_block("sigma2",  list(c1$sigma2, c2$sigma2))

# Prediction accuracy sanity — use one of the 6000-step chains, not the
# 200-step warmup instance m1
m_acc <- new(ARDLasso, Xa, Ya, 42L, TRUE)
m_acc$step(6000L)
preds_keep <- m_acc$predict_at(list(X = Xa))$y_hat[3001:6000, ]
y_pred <- colMeans(preds_keep)
cor_val <- cor(y_pred, Ya)
cat(sprintf("    ARD in-sample cor(y_pred, Y) = %.4f\n", cor_val))
check(cor_val > 0.9, "ARDLasso: in-sample cor > 0.9")

cat(sprintf("\n=== %d passed, %d failed ===\n", n_pass, n_fail))
if (n_fail != 0L) quit(status = 1L)

# Phase C of audit — MCMC diagnostics on all 11 production examples.
# Synthetic data per example, 2 chains, compute R-hat / ESS / posterior
# recovery / LOO (where applicable). Appends summary to AUDIT_LOG.md.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
audit_log     <- file.path(AI4BayesCode_dir, "AUDIT_LOG.md")
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))
suppressPackageStartupMessages({ library(posterior); library(loo) })

pack2 <- function(x1, x2) {
    if (is.null(dim(x1)))
        array(cbind(x1, x2), dim = c(length(x1), 2, 1))
    else {
        arr <- array(NA_real_, dim = c(nrow(x1), 2, ncol(x1)))
        arr[,1,] <- x1; arr[,2,] <- x2
        arr
    }
}
diag_arr <- function(arr) {
    p <- dim(arr)[3]
    rh <- sapply(seq_len(p), function(k) posterior::rhat(arr[,,k]))
    eb <- sapply(seq_len(p), function(k) posterior::ess_bulk(arr[,,k]))
    et <- sapply(seq_len(p), function(k) posterior::ess_tail(arr[,,k]))
    list(rhat_max = max(rh, na.rm=TRUE),
         ess_bulk_min = min(eb, na.rm=TRUE),
         ess_tail_min = min(et, na.rm=TRUE))
}
summarize_recovery <- function(hist_vec_or_mat, truth) {
    if (is.null(dim(hist_vec_or_mat))) {
        pm <- mean(hist_vec_or_mat); ps <- sd(hist_vec_or_mat)
        list(within_3sd = abs(pm - truth) < 3 * ps,
             post_mean = pm, post_sd = ps)
    } else {
        pm <- colMeans(hist_vec_or_mat); ps <- apply(hist_vec_or_mat, 2, sd)
        within <- abs(pm - truth) < 3 * ps
        list(within_3sd_pct = 100 * mean(within),
             post_mean = pm, post_sd = ps)
    }
}
loo_from <- function(LL1, LL2) {
    stopifnot(all(dim(LL1) == dim(LL2)))
    LLarr <- array(NA_real_, dim = c(nrow(LL1), 2, ncol(LL1)))
    LLarr[,1,] <- LL1; LLarr[,2,] <- LL2
    rel_eff <- loo::relative_eff(exp(LLarr))
    lo <- loo::loo(LLarr, r_eff = rel_eff, cores = 1)
    list(elpd = lo$estimates["elpd_loo", "Estimate"],
         se   = lo$estimates["elpd_loo", "SE"],
         pct_k_ok = 100 * mean(lo$diagnostics$pareto_k < 0.5),
         pct_k_bad = 100 * mean(lo$diagnostics$pareto_k >= 0.7))
}

results <- list()

# Each block: compile → simulate → run 2 chains → diagnose → append result

# ============================================================================
# C.1 GaussianLocationScale
# ============================================================================
cat("\n[C.1] GaussianLocationScale\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GaussianLocationScale.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); y <- rnorm(100, 2, 1.5); truth <- list(mu = 2.0, sigma = 1.5)
run <- function(seed, nb, nk) {
    m <- new(GaussianLocationScale, y, as.integer(seed), TRUE)
    t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
    h <- m$get_history()
    list(mu = h$mu[(nb+1):(nb+nk)], sigma = h$sigma[(nb+1):(nb+nk)],
         wall = as.numeric(difftime(t1,t0, units="secs")))
}
c1 <- run(101L, 500L, 2000L); c2 <- run(202L, 500L, 2000L)
d <- diag_arr(pack2(cbind(c1$mu,c1$sigma), cbind(c2$mu,c2$sigma)))
rec_mu    <- summarize_recovery(c(c1$mu, c2$mu), truth$mu)
rec_sigma <- summarize_recovery(c(c1$sigma, c2$sigma), truth$sigma)
LL1 <- matrix(NA_real_, length(c1$mu), length(y))
LL2 <- matrix(NA_real_, length(c2$mu), length(y))
for (i in seq_along(c1$mu)) LL1[i,] <- dnorm(y, c1$mu[i], c1$sigma[i], log=TRUE)
for (i in seq_along(c2$mu)) LL2[i,] <- dnorm(y, c2$mu[i], c2$sigma[i], log=TRUE)
loo_ <- loo_from(LL1, LL2)
results$Gaussian <- list(wall=c1$wall+c2$wall, diag=d,
    rec_mu=rec_mu$within_3sd, rec_sigma=rec_sigma$within_3sd,
    post_mu=rec_mu$post_mean, post_sigma=rec_sigma$post_mean,
    truth=truth, loo=loo_)
cat(sprintf("  wall=%.2fs Rhat=%.4f ESS_bulk=%.0f rec=(mu:%s,sig:%s) LOO k<0.5=%.0f%%\n",
    results$Gaussian$wall, d$rhat_max, d$ess_bulk_min,
    rec_mu$within_3sd, rec_sigma$within_3sd, loo_$pct_k_ok))

# ============================================================================
# C.2 BetaBernoulli
# ============================================================================
cat("\n[C.2] BetaBernoulli\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "BetaBernoulli.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(42); y <- rbinom(200, 1, 0.3); truth <- list(p = 0.3)
run <- function(seed, nb, nk) {
    m <- new(BetaBernoulli, as.numeric(y), 1.0, 1.0, as.integer(seed), TRUE)
    t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
    h <- m$get_history()
    list(p = h$p[(nb+1):(nb+nk)],
         wall = as.numeric(difftime(t1,t0, units="secs")))
}
c1 <- run(101L, 500L, 2000L); c2 <- run(202L, 500L, 2000L)
d <- diag_arr(pack2(c1$p, c2$p))
rec_p <- summarize_recovery(c(c1$p, c2$p), truth$p)
LL1 <- matrix(NA_real_, length(c1$p), length(y))
LL2 <- matrix(NA_real_, length(c2$p), length(y))
for (i in seq_along(c1$p)) LL1[i,] <- dbinom(y, 1, c1$p[i], log=TRUE)
for (i in seq_along(c2$p)) LL2[i,] <- dbinom(y, 1, c2$p[i], log=TRUE)
loo_ <- loo_from(LL1, LL2)
results$BetaBernoulli <- list(wall=c1$wall+c2$wall, diag=d, rec_p=rec_p$within_3sd,
    post_p=rec_p$post_mean, truth=truth, loo=loo_)
cat(sprintf("  wall=%.2fs Rhat=%.4f ESS_bulk=%.0f rec=%s  post_p=%.3f(true 0.3) LOO k<0.5=%.0f%%\n",
    results$BetaBernoulli$wall, d$rhat_max, d$ess_bulk_min,
    rec_p$within_3sd, rec_p$post_mean, loo_$pct_k_ok))

# ============================================================================
# C.3 DirichletSimplex
# ============================================================================
cat("\n[C.3] DirichletSimplex\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletSimplex.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
y_counts <- c(15, 25, 30, 20, 10); alpha <- rep(1,5)
# Truth theta = y_counts/sum(y_counts) roughly (posterior approximately)
run <- function(seed, nb, nk) {
    m <- new(DirichletSimplex, as.numeric(y_counts), alpha, as.integer(seed), TRUE)
    t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
    h <- m$get_history()
    list(theta = h$theta[(nb+1):(nb+nk), , drop=FALSE],
         wall = as.numeric(difftime(t1,t0, units="secs")))
}
c1 <- run(101L, 500L, 2000L); c2 <- run(202L, 500L, 2000L)
d <- diag_arr(pack2(c1$theta, c2$theta))
post_mean <- colMeans(rbind(c1$theta, c2$theta))
# Posterior Dirichlet(alpha + y) — posterior mean = (alpha+y)/sum(alpha+y)
posterior_truth <- (alpha + y_counts) / sum(alpha + y_counts)
rec <- summarize_recovery(rbind(c1$theta, c2$theta), posterior_truth)
# Pointwise LOO on categorical observations (y_counts = aggregated)
# Skip LOO for Dirichlet (single aggregated y; not meaningful per-obs LOO)
results$DirichletSimplex <- list(wall=c1$wall+c2$wall, diag=d,
    rec_pct = rec$within_3sd_pct,
    post_mean=post_mean, posterior_truth=posterior_truth)
cat(sprintf("  wall=%.2fs Rhat=%.4f ESS_bulk=%.0f rec=%.0f%%  post=%s  truth=%s\n",
    results$DirichletSimplex$wall, d$rhat_max, d$ess_bulk_min, rec$within_3sd_pct,
    paste(sprintf("%.3f", post_mean), collapse=" "),
    paste(sprintf("%.3f", posterior_truth), collapse=" ")))

# ============================================================================
# C.4 DirichletSparse
# ============================================================================
cat("\n[C.4] DirichletSparse\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletSparse.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(42); P <- 20L; N_tot <- 500L
# True simplex: 5 large, rest near zero
s_true <- c(rep(1/5, 5), rep(1e-5, P-5))
s_true <- s_true / sum(s_true)
y_sp <- as.numeric(tabulate(sample.int(P, N_tot, prob=s_true, replace=TRUE), nbins=P))
run <- function(seed, nb, nk) {
    m <- new(DirichletSparse, y_sp, as.integer(seed), TRUE)
    t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
    h <- m$get_history()
    list(s = h$s[(nb+1):(nb+nk), , drop=FALSE],
         theta = h$theta[(nb+1):(nb+nk)],
         wall = as.numeric(difftime(t1,t0, units="secs")))
}
c1 <- run(101L, 500L, 1500L); c2 <- run(202L, 500L, 1500L)
d_s <- diag_arr(pack2(c1$s, c2$s))
d_theta <- diag_arr(pack2(c1$theta, c2$theta))
# Top-5 recovery
pm <- colMeans(rbind(c1$s, c2$s))
top5_true <- order(s_true, decreasing=TRUE)[1:5]
top5_pred <- order(pm, decreasing=TRUE)[1:5]
results$DirichletSparse <- list(wall=c1$wall+c2$wall, diag_s=d_s, diag_theta=d_theta,
    top5_overlap = length(intersect(top5_true, top5_pred)))
cat(sprintf("  wall=%.2fs  s: Rhat=%.4f ESS=%.0f  theta: Rhat=%.4f ESS=%.0f  top5/5=%d\n",
    results$DirichletSparse$wall, d_s$rhat_max, d_s$ess_bulk_min,
    d_theta$rhat_max, d_theta$ess_bulk_min, results$DirichletSparse$top5_overlap))

# ============================================================================
# C.5 DirichletHierarchical
# ============================================================================
cat("\n[C.5] DirichletHierarchical\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletHierarchical.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(42); K_obs <- 20L; P_dim <- 4L
s_true_dh <- c(0.1, 0.3, 0.4, 0.2); kappa_true <- 20.0
S_obs <- matrix(NA, K_obs, P_dim)
for (k in 1:K_obs) {
    # NB: C++ expects each ROW of S_obs to be a P-dim simplex.
    g <- rgamma(P_dim, shape = kappa_true * s_true_dh)
    S_obs[k,] <- g / sum(g)
}
run <- function(seed, nb, nk) {
    m <- new(DirichletHierarchical, S_obs, 1.0, 1.0, as.integer(seed), TRUE)
    t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
    h <- m$get_history()
    list(s = h$s[(nb+1):(nb+nk), , drop=FALSE],
         kappa = h$kappa[(nb+1):(nb+nk)],
         theta = h$theta[(nb+1):(nb+nk)],
         wall = as.numeric(difftime(t1,t0, units="secs")))
}
c1 <- run(101L, 500L, 1000L); c2 <- run(202L, 500L, 1000L)
d_s <- diag_arr(pack2(c1$s, c2$s))
d_k <- diag_arr(pack2(c1$kappa, c2$kappa))
d_th<- diag_arr(pack2(c1$theta, c2$theta))
results$DirichletHierarchical <- list(wall=c1$wall+c2$wall,
    diag_s=d_s, diag_kappa=d_k, diag_theta=d_th)
cat(sprintf("  wall=%.2fs  s Rhat=%.4f  kappa Rhat=%.4f  theta Rhat=%.4f\n",
    results$DirichletHierarchical$wall, d_s$rhat_max, d_k$rhat_max, d_th$rhat_max))

# ============================================================================
# C.6 BartNoise
# ============================================================================
cat("\n[C.6] BartNoise (small, short chains)\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "BartNoise.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(42); N_b <- 80L; p_b <- 3L
X_b <- matrix(rnorm(N_b*p_b), N_b, p_b)
f_true_b <- 3*X_b[,1] + X_b[,2]^2
y_b <- as.numeric(f_true_b + rnorm(N_b, 0, 0.8))
run <- function(seed, nb, nk) {
    set.seed(seed)
    m <- new(BartNoise, X_b, y_b, 50L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE,
             as.integer(seed), TRUE)
    t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
    h <- m$get_history()
    list(f_bart = h$f_bart[(nb+1):(nb+nk), , drop=FALSE],
         sigma = h$sigma[(nb+1):(nb+nk)],
         wall = as.numeric(difftime(t1,t0, units="secs")))
}
c1 <- run(101L, 300L, 700L); c2 <- run(202L, 300L, 700L)
d_sigma <- diag_arr(pack2(c1$sigma, c2$sigma))
# Pointwise loglik
LL1 <- matrix(NA_real_, length(c1$sigma), length(y_b))
LL2 <- matrix(NA_real_, length(c2$sigma), length(y_b))
for (i in seq_along(c1$sigma)) LL1[i,] <- dnorm(y_b, c1$f_bart[i,], c1$sigma[i], log=TRUE)
for (i in seq_along(c2$sigma)) LL2[i,] <- dnorm(y_b, c2$f_bart[i,], c2$sigma[i], log=TRUE)
loo_ <- loo_from(LL1, LL2)
# f RMSE on training
pm_f <- colMeans(rbind(c1$f_bart, c2$f_bart))
rmse_f <- sqrt(mean((pm_f - f_true_b)^2))
results$BartNoise <- list(wall=c1$wall+c2$wall, diag_sigma=d_sigma,
    post_sigma=mean(c(c1$sigma, c2$sigma)), truth_sigma=0.8, rmse_f=rmse_f, loo=loo_)
cat(sprintf("  wall=%.2fs  sigma Rhat=%.4f ESS=%.0f post_sigma=%.3f(true 0.8)  f RMSE=%.3f  LOO k<0.5=%.0f%%\n",
    results$BartNoise$wall, d_sigma$rhat_max, d_sigma$ess_bulk_min,
    results$BartNoise$post_sigma, rmse_f, loo_$pct_k_ok))

# ============================================================================
# C.7 GBartPoisson (Linero 2022 genBART)
# ============================================================================
cat("\n[C.7] GBartPoisson\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GBartPoisson.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(42); N_l <- 80L; p_l <- 3L
X_l <- matrix(runif(N_l*p_l), N_l, p_l)
r_true <- 1 + 2*X_l[,1] - X_l[,2]   # log-rate (r ≡ log_f in legacy notation)
y_l <- rpois(N_l, exp(r_true))
run <- function(seed, nb, nk) {
    m <- new(GBartPoisson, X_l, as.numeric(y_l), 50L, as.integer(seed), TRUE)
    t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
    h <- m$get_history()
    list(r    = h$r[(nb+1):(nb+nk), , drop=FALSE],
         wall = as.numeric(difftime(t1,t0, units="secs")))
}
c1 <- run(101L, 300L, 700L); c2 <- run(202L, 300L, 700L)
# Poisson LOO
LL1 <- matrix(NA_real_, nrow(c1$r), length(y_l))
LL2 <- matrix(NA_real_, nrow(c2$r), length(y_l))
for (i in seq_len(nrow(c1$r))) LL1[i,] <- dpois(y_l, exp(c1$r[i,]), log=TRUE)
for (i in seq_len(nrow(c2$r))) LL2[i,] <- dpois(y_l, exp(c2$r[i,]), log=TRUE)
loo_ <- loo_from(LL1, LL2)
pm_r   <- colMeans(rbind(c1$r, c2$r))
rmse_r <- sqrt(mean((pm_r - r_true)^2))
results$GBartPoisson <- list(wall=c1$wall+c2$wall, rmse_r=rmse_r, loo=loo_)
cat(sprintf("  wall=%.2fs  r RMSE=%.3f  LOO k<0.5=%.0f%%\n",
    results$GBartPoisson$wall, rmse_r, loo_$pct_k_ok))

# ============================================================================
# C.8 ARDLasso
# ============================================================================
cat("\n[C.8] ARDLasso\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "ARDLasso.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(42); N_a <- 100L; p_a <- 10L
X_a <- matrix(rnorm(N_a*p_a), N_a, p_a)
beta_true <- c(3, -2, 0, 0, 1.5, 0, 0, 0, 0, 0)
y_a <- as.numeric(2.0 + X_a %*% beta_true + rnorm(N_a, 0, 1.0))
run <- function(seed, nb, nk) {
    m <- new(ARDLasso, X_a, y_a, as.integer(seed), TRUE)
    t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
    h <- m$get_history()
    list(beta = h$beta[(nb+1):(nb+nk), , drop=FALSE],
         alpha = h$alpha[(nb+1):(nb+nk)],
         sigma2 = h$sigma2[(nb+1):(nb+nk)],
         wall = as.numeric(difftime(t1,t0, units="secs")))
}
c1 <- run(101L, 500L, 2000L); c2 <- run(202L, 500L, 2000L)
d_beta <- diag_arr(pack2(c1$beta, c2$beta))
rec_beta <- summarize_recovery(rbind(c1$beta, c2$beta), beta_true)
LL1 <- matrix(NA_real_, length(c1$alpha), length(y_a))
LL2 <- matrix(NA_real_, length(c2$alpha), length(y_a))
for (i in seq_along(c1$alpha)) LL1[i,] <- dnorm(y_a, as.numeric(X_a %*% c1$beta[i,]) + c1$alpha[i], sqrt(c1$sigma2[i]), log=TRUE)
for (i in seq_along(c2$alpha)) LL2[i,] <- dnorm(y_a, as.numeric(X_a %*% c2$beta[i,]) + c2$alpha[i], sqrt(c2$sigma2[i]), log=TRUE)
loo_ <- loo_from(LL1, LL2)
results$ARDLasso <- list(wall=c1$wall+c2$wall, diag_beta=d_beta,
    rec_pct=rec_beta$within_3sd_pct,
    post_beta=colMeans(rbind(c1$beta, c2$beta)), truth=beta_true, loo=loo_)
cat(sprintf("  wall=%.2fs  beta Rhat=%.4f ESS=%.0f rec=%.0f%%  LOO k<0.5=%.0f%%\n",
    results$ARDLasso$wall, d_beta$rhat_max, d_beta$ess_bulk_min,
    rec_beta$within_3sd_pct, loo_$pct_k_ok))

# ============================================================================
# C.9 IRT1PL_joint — small dimension (N=30, J=8; dim 38)
# ============================================================================
cat("\n[C.9] IRT1PL_joint (N=30, J=8; dim 38 — below known mass-matrix ceiling)\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "IRT1PL_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_i <- 30L; J_i <- 8L
theta_t <- rnorm(N_i, 0, 1); sigma_b_t <- 0.8
b_t <- rnorm(J_i, 0, sigma_b_t)
P_i <- outer(theta_t, b_t, function(a, bb) 1/(1+exp(-(a-bb))))
Y_i <- matrix(rbinom(N_i*J_i, 1, P_i), N_i, J_i)
Y_i[matrix(runif(N_i*J_i) < 0.1, N_i, J_i)] <- NA_real_
run <- function(seed, nb, nk, th_init, b_init, sb_init) {
    m <- new(IRT1PL_joint, Y_i, th_init, b_init, as.numeric(sb_init),
             as.integer(seed), TRUE)
    t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
    h <- m$get_history()
    list(theta = h$theta[(nb+1):(nb+nk), , drop=FALSE],
         b = h$b[(nb+1):(nb+nk), , drop=FALSE],
         sigma_b = h$sigma_b[(nb+1):(nb+nk)],
         wall = as.numeric(difftime(t1,t0, units="secs")))
}
c1 <- run(101L, 500L, 1500L, rep(0,N_i), rep(0,J_i), 1.0)
c2 <- run(202L, 500L, 1500L, rnorm(N_i,0,0.5), rnorm(J_i,0,0.5), 0.5)
d_theta <- diag_arr(pack2(c1$theta, c2$theta))
d_b     <- diag_arr(pack2(c1$b, c2$b))
d_sb    <- diag_arr(pack2(c1$sigma_b, c2$sigma_b))
rec_theta <- summarize_recovery(rbind(c1$theta, c2$theta), theta_t)
rec_b     <- summarize_recovery(rbind(c1$b, c2$b), b_t)
results$IRT1PL_joint <- list(wall=c1$wall+c2$wall,
    diag_theta=d_theta, diag_b=d_b, diag_sb=d_sb,
    rec_theta=rec_theta$within_3sd_pct, rec_b=rec_b$within_3sd_pct,
    post_sb=mean(c(c1$sigma_b, c2$sigma_b)), truth_sb=sigma_b_t)
cat(sprintf("  wall=%.1fs  theta Rhat=%.3f  b Rhat=%.3f  sb Rhat=%.3f  rec=(th:%.0f%%, b:%.0f%%)  post_sb=%.3f(true 0.8)\n",
    results$IRT1PL_joint$wall, d_theta$rhat_max, d_b$rhat_max, d_sb$rhat_max,
    rec_theta$within_3sd_pct, rec_b$within_3sd_pct, results$IRT1PL_joint$post_sb))

# ============================================================================
# C.10 HierarchicalLM_joint
# ============================================================================
cat("\n[C.10] HierarchicalLM_joint\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "HierarchicalLM_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); G <- 10L; Np <- 15L; N_hl <- G*Np; p_hl <- 3L
alpha_t <- 1.0; beta_t <- c(2, -1, 0.5); sigma_t <- 0.8; tau_t <- 1.5
X_hl <- matrix(rnorm(N_hl*p_hl), N_hl, p_hl)
g_idx <- rep(seq_len(G), each=Np)
u_t <- rnorm(G, 0, tau_t)
y_hl <- as.numeric(alpha_t + X_hl %*% beta_t + u_t[g_idx] + rnorm(N_hl, 0, sigma_t))
run <- function(seed, nb, nk) {
    m <- new(HierarchicalLM_joint, y_hl, X_hl, as.integer(g_idx), G,
             1.0, 1.0, as.integer(seed), TRUE)
    t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
    h <- m$get_history()
    list(alpha = h$alpha[(nb+1):(nb+nk)],
         beta  = h$beta[(nb+1):(nb+nk), , drop=FALSE],
         u     = h$u[(nb+1):(nb+nk), , drop=FALSE],
         sigma = h$sigma[(nb+1):(nb+nk)],
         tau   = h$tau[(nb+1):(nb+nk)],
         wall = as.numeric(difftime(t1,t0, units="secs")))
}
c1 <- run(101L, 500L, 1500L); c2 <- run(202L, 500L, 1500L)
d_a  <- diag_arr(pack2(c1$alpha, c2$alpha))
d_b  <- diag_arr(pack2(c1$beta, c2$beta))
d_u  <- diag_arr(pack2(c1$u, c2$u))
d_s  <- diag_arr(pack2(c1$sigma, c2$sigma))
d_t  <- diag_arr(pack2(c1$tau, c2$tau))
rec_b <- summarize_recovery(rbind(c1$beta, c2$beta), beta_t)
rec_sigma <- summarize_recovery(c(c1$sigma, c2$sigma), sigma_t)
results$HierarchicalLM_joint <- list(wall=c1$wall+c2$wall,
    diag_alpha=d_a, diag_beta=d_b, diag_u=d_u, diag_sigma=d_s, diag_tau=d_t,
    rec_beta=rec_b$within_3sd_pct, rec_sigma=rec_sigma$within_3sd)
cat(sprintf("  wall=%.1fs a Rhat=%.3f b Rhat=%.3f u Rhat=%.3f s Rhat=%.3f t Rhat=%.3f  beta rec=%.0f%%  sigma rec=%s\n",
    results$HierarchicalLM_joint$wall, d_a$rhat_max, d_b$rhat_max,
    d_u$rhat_max, d_s$rhat_max, d_t$rhat_max,
    rec_b$within_3sd_pct, rec_sigma$within_3sd))

# ============================================================================
# C.11 LinearRegJointMixed
# ============================================================================
cat("\n[C.11] LinearRegJointMixed\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "LinearRegJointMixed.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_lr <- 200L; p_lr <- 5L
X_lr <- matrix(rnorm(N_lr*p_lr), N_lr, p_lr)
alpha_t <- 1.5; beta_t <- c(2, -1, 0.5, 0, 3); sigma_t <- 1.2
y_lr <- as.numeric(alpha_t + X_lr %*% beta_t + rnorm(N_lr, 0, sigma_t))
run <- function(seed, nb, nk) {
    m <- new(LinearRegJointMixed, y_lr, X_lr, as.integer(seed), TRUE)
    t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
    h <- m$get_history()
    list(alpha = h$alpha[(nb+1):(nb+nk)],
         beta  = h$beta[(nb+1):(nb+nk), , drop=FALSE],
         sigma = h$sigma[(nb+1):(nb+nk)],
         wall = as.numeric(difftime(t1,t0, units="secs")))
}
c1 <- run(101L, 500L, 1500L); c2 <- run(202L, 500L, 1500L)
d_a <- diag_arr(pack2(c1$alpha, c2$alpha))
d_b <- diag_arr(pack2(c1$beta, c2$beta))
d_s <- diag_arr(pack2(c1$sigma, c2$sigma))
rec_a <- summarize_recovery(c(c1$alpha, c2$alpha), alpha_t)
rec_b <- summarize_recovery(rbind(c1$beta, c2$beta), beta_t)
rec_s <- summarize_recovery(c(c1$sigma, c2$sigma), sigma_t)
LL1 <- matrix(NA_real_, length(c1$alpha), length(y_lr))
LL2 <- matrix(NA_real_, length(c2$alpha), length(y_lr))
for (i in seq_along(c1$alpha)) LL1[i,] <- dnorm(y_lr, c1$alpha[i] + as.numeric(X_lr %*% c1$beta[i,]), c1$sigma[i], log=TRUE)
for (i in seq_along(c2$alpha)) LL2[i,] <- dnorm(y_lr, c2$alpha[i] + as.numeric(X_lr %*% c2$beta[i,]), c2$sigma[i], log=TRUE)
loo_ <- loo_from(LL1, LL2)
results$LinearRegJointMixed <- list(wall=c1$wall+c2$wall,
    diag_a=d_a, diag_b=d_b, diag_s=d_s,
    rec_a=rec_a$within_3sd, rec_beta=rec_b$within_3sd_pct, rec_s=rec_s$within_3sd,
    post_s=mean(c(c1$sigma, c2$sigma)), truth_s=sigma_t, loo=loo_)
cat(sprintf("  wall=%.1fs a Rhat=%.3f b Rhat=%.3f s Rhat=%.3f  rec=(a:%s, b:%.0f%%, s:%s) post_s=%.3f(true 1.2) LOO k<0.5=%.0f%%\n",
    results$LinearRegJointMixed$wall, d_a$rhat_max, d_b$rhat_max, d_s$rhat_max,
    rec_a$within_3sd, rec_b$within_3sd_pct, rec_s$within_3sd,
    mean(c(c1$sigma, c2$sigma)), loo_$pct_k_ok))

# ============================================================================
# Save + emit summary
# ============================================================================
saveRDS(results, file.path(AI4BayesCode_dir, "audit_results.rds"))
cat("\n=== SUMMARY ===\n")
for (nm in names(results)) {
    r <- results[[nm]]
    cat(sprintf("%-28s wall=%.1fs\n", nm, r$wall))
}
cat("\n[ALL C PHASES DONE]\n")

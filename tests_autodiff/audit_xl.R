# Phase F1 of audit — LONG CHAIN + MULTI-SEED MCMC diagnostics.
# For each of 13 production examples, run 3 data seeds × 2 chains ×
# 10000 burnin + 10000 keep. Reports R-hat, ESS, posterior recovery,
# LOO per seed. Aggregates "fraction of seeds clean" per example.
# (13 = original 11 + GBartLogistic + GBartMultinomial via genBART.)
#
# Logs to audit_xl.log; rds to audit_xl_F1.rds.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))
suppressPackageStartupMessages({ library(posterior); library(loo) })

# -------- helpers --------
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
         ess_tail_min = min(et, na.rm=TRUE),
         rhat = rh, ess_bulk = eb, ess_tail = et)
}
recovery_pct <- function(post_mean, post_sd, truth) {
    if (length(truth) == 1)
        as.numeric(abs(post_mean - truth) < 3 * post_sd)
    else
        100 * mean(abs(post_mean - truth) < 3 * post_sd)
}
loo_from <- function(LL1, LL2) {
    stopifnot(all(dim(LL1) == dim(LL2)))
    LLarr <- array(NA_real_, dim = c(nrow(LL1), 2, ncol(LL1)))
    LLarr[,1,] <- LL1; LLarr[,2,] <- LL2
    rel_eff <- suppressWarnings(loo::relative_eff(exp(LLarr)))
    lo <- suppressWarnings(loo::loo(LLarr, r_eff = rel_eff, cores = 1))
    list(elpd = lo$estimates["elpd_loo", "Estimate"],
         se   = lo$estimates["elpd_loo", "SE"],
         pct_k_ok  = 100 * mean(lo$diagnostics$pareto_k < 0.5),
         pct_k_bad = 100 * mean(lo$diagnostics$pareto_k >= 0.7))
}

# Seeds used as "data seeds". Each generates a distinct synthetic dataset;
# Rcpp class constructor seeds are set to 101/202 as before for the 2 chains.
DATA_SEEDS <- c(1, 42, 123)
N_BURN <- 10000L
N_KEEP <- 10000L

log_seed <- function(ex, s, msg) {
    cat(sprintf("  [%s seed=%d] %s\n", ex, s, msg))
}

results_F1 <- list()

run_one_example <- function(name, fn_simulate, fn_run, fn_diagnose) {
    cat(sprintf("\n=== [F1:%s] 3 seeds × 2 chains × %d+%d ===\n",
                name, N_BURN, N_KEEP))
    per_seed <- list()
    for (s in DATA_SEEDS) {
        dat <- fn_simulate(s)
        c1 <- fn_run(dat, 101L, N_BURN, N_KEEP)
        c2 <- fn_run(dat, 202L, N_BURN, N_KEEP)
        d <- fn_diagnose(dat, c1, c2)
        per_seed[[as.character(s)]] <- d
        log_seed(name, s, sprintf("wall=%.1fs  %s", c1$wall + c2$wall, d$summary))
    }
    results_F1[[name]] <<- per_seed
    saveRDS(results_F1, file.path(AI4BayesCode_dir, "audit_xl_F1.rds"))
    invisible(NULL)
}

# ====== 1. GaussianLocationScale ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GaussianLocationScale.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("Gaussian",
    fn_simulate = function(s) { set.seed(s); list(y=rnorm(100, 2, 1.5),
                                                  truth=list(mu=2.0, sigma=1.5)) },
    fn_run = function(dat, cs, nb, nk) {
        m <- new(GaussianLocationScale, dat$y, as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(mu = h$mu[(nb+1):(nb+nk)], sigma = h$sigma[(nb+1):(nb+nk)],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        d_mu  <- diag_arr(pack2(c1$mu,    c2$mu))
        d_sig <- diag_arr(pack2(c1$sigma, c2$sigma))
        pm_mu  <- mean(c(c1$mu,    c2$mu));    ps_mu  <- sd(c(c1$mu,    c2$mu))
        pm_sig <- mean(c(c1$sigma, c2$sigma)); ps_sig <- sd(c(c1$sigma, c2$sigma))
        # LOO
        LL1 <- matrix(NA_real_, length(c1$mu), length(dat$y))
        LL2 <- matrix(NA_real_, length(c2$mu), length(dat$y))
        for (i in seq_along(c1$mu)) LL1[i,] <- dnorm(dat$y, c1$mu[i], c1$sigma[i], log=TRUE)
        for (i in seq_along(c2$mu)) LL2[i,] <- dnorm(dat$y, c2$mu[i], c2$sigma[i], log=TRUE)
        lo <- loo_from(LL1, LL2)
        rhat_max <- max(d_mu$rhat_max, d_sig$rhat_max)
        ess_min  <- min(d_mu$ess_bulk_min, d_sig$ess_bulk_min)
        rec_mu  <- abs(pm_mu  - dat$truth$mu)    < 3 * ps_mu
        rec_sig <- abs(pm_sig - dat$truth$sigma) < 3 * ps_sig
        list(rhat_max=rhat_max, ess_min=ess_min,
             post_mu=pm_mu, post_sigma=pm_sig, truth=dat$truth,
             rec_mu=rec_mu, rec_sigma=rec_sig, loo=lo,
             summary=sprintf("Rhat=%.4f ESS=%.0f rec(mu:%s,sig:%s) LOOk<0.5=%.0f%%",
                 rhat_max, ess_min, rec_mu, rec_sig, lo$pct_k_ok))
    }
)

# ====== 2. BetaBernoulli ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "BetaBernoulli.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("BetaBernoulli",
    fn_simulate = function(s) { set.seed(s); list(y=rbinom(200, 1, 0.3),
                                                  truth=list(p=0.3)) },
    fn_run = function(dat, cs, nb, nk) {
        m <- new(BetaBernoulli, as.numeric(dat$y), 1.0, 1.0, as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(p = h$p[(nb+1):(nb+nk)],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        d_p <- diag_arr(pack2(c1$p, c2$p))
        pm_p <- mean(c(c1$p, c2$p)); ps_p <- sd(c(c1$p, c2$p))
        rec_p <- abs(pm_p - dat$truth$p) < 3 * ps_p
        LL1 <- matrix(NA_real_, length(c1$p), length(dat$y))
        LL2 <- matrix(NA_real_, length(c2$p), length(dat$y))
        for (i in seq_along(c1$p)) LL1[i,] <- dbinom(dat$y, 1, c1$p[i], log=TRUE)
        for (i in seq_along(c2$p)) LL2[i,] <- dbinom(dat$y, 1, c2$p[i], log=TRUE)
        lo <- loo_from(LL1, LL2)
        list(rhat_max=d_p$rhat_max, ess_min=d_p$ess_bulk_min,
             post_p=pm_p, truth=dat$truth, rec_p=rec_p, loo=lo,
             summary=sprintf("Rhat=%.4f ESS=%.0f post_p=%.3f(true %.2f) rec=%s LOOk<0.5=%.0f%%",
                 d_p$rhat_max, d_p$ess_bulk_min, pm_p, dat$truth$p, rec_p, lo$pct_k_ok))
    }
)

# ====== 3. DirichletSimplex ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletSimplex.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("DirichletSimplex",
    fn_simulate = function(s) {
        set.seed(s)
        y_counts <- rmultinom(1, 100, c(0.15, 0.25, 0.30, 0.20, 0.10))[,1]
        alpha <- rep(1, 5)
        list(y=as.numeric(y_counts), alpha=alpha,
             posterior_truth=(alpha + y_counts)/sum(alpha + y_counts))
    },
    fn_run = function(dat, cs, nb, nk) {
        m <- new(DirichletSimplex, dat$y, dat$alpha, as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(theta = h$theta[(nb+1):(nb+nk), , drop=FALSE],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        d <- diag_arr(pack2(c1$theta, c2$theta))
        pm <- colMeans(rbind(c1$theta, c2$theta))
        ps <- apply(rbind(c1$theta, c2$theta), 2, sd)
        rec_pct <- 100 * mean(abs(pm - dat$posterior_truth) < 3 * ps)
        list(rhat_max=d$rhat_max, ess_min=d$ess_bulk_min,
             post_mean=pm, truth=dat$posterior_truth, rec_pct=rec_pct,
             summary=sprintf("Rhat=%.4f ESS=%.0f rec=%.0f%%  max|diff|=%.3f",
                 d$rhat_max, d$ess_bulk_min, rec_pct, max(abs(pm - dat$posterior_truth))))
    }
)

# ====== 4. DirichletSparse ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletSparse.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("DirichletSparse",
    fn_simulate = function(s) {
        set.seed(s); P <- 20L; N_tot <- 500L
        s_true <- c(rep(1/5, 5), rep(1e-5, P-5)); s_true <- s_true / sum(s_true)
        y_sp <- as.numeric(tabulate(sample.int(P, N_tot, prob=s_true, replace=TRUE), nbins=P))
        list(y=y_sp, s_true=s_true, P=P)
    },
    fn_run = function(dat, cs, nb, nk) {
        m <- new(DirichletSparse, dat$y, as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(s = h$s[(nb+1):(nb+nk), , drop=FALSE],
             theta = h$theta[(nb+1):(nb+nk)],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        d_s <- diag_arr(pack2(c1$s, c2$s))
        d_th <- diag_arr(pack2(c1$theta, c2$theta))
        pm <- colMeans(rbind(c1$s, c2$s))
        top5_true <- order(dat$s_true, decreasing=TRUE)[1:5]
        top5_pred <- order(pm, decreasing=TRUE)[1:5]
        overlap <- length(intersect(top5_true, top5_pred))
        list(rhat_s=d_s$rhat_max, ess_s=d_s$ess_bulk_min,
             rhat_theta=d_th$rhat_max, ess_theta=d_th$ess_bulk_min,
             top5=overlap,
             summary=sprintf("s Rhat=%.4f ESS=%.0f theta Rhat=%.4f ESS=%.0f top5/5=%d",
                 d_s$rhat_max, d_s$ess_bulk_min, d_th$rhat_max, d_th$ess_bulk_min, overlap))
    }
)

# ====== 5. DirichletHierarchical ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletHierarchical.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("DirichletHierarchical",
    fn_simulate = function(s) {
        # NB: C++ expects S_obs with K rows = K observations, each a P-dim
        # simplex (each ROW sums to 1). Earlier bug had this transposed.
        set.seed(s); K_obs <- 20L; P_dim <- 4L
        s_true <- c(0.1, 0.3, 0.4, 0.2); kappa_true <- 20.0
        S_obs <- matrix(NA, K_obs, P_dim)
        for (k in 1:K_obs) {
            g <- rgamma(P_dim, shape = kappa_true * s_true)
            S_obs[k,] <- g / sum(g)
        }
        list(S=S_obs, truth=list(s=s_true, kappa=kappa_true))
    },
    fn_run = function(dat, cs, nb, nk) {
        m <- new(DirichletHierarchical, dat$S, 1.0, 1.0, as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(s = h$s[(nb+1):(nb+nk), , drop=FALSE],
             kappa = h$kappa[(nb+1):(nb+nk)],
             theta = h$theta[(nb+1):(nb+nk)],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        d_s  <- diag_arr(pack2(c1$s, c2$s))
        d_k  <- diag_arr(pack2(c1$kappa, c2$kappa))
        d_th <- diag_arr(pack2(c1$theta, c2$theta))
        pm_s <- colMeans(rbind(c1$s, c2$s))
        pm_k <- mean(c(c1$kappa, c2$kappa))
        list(rhat_s=d_s$rhat_max, ess_s=d_s$ess_bulk_min,
             rhat_kappa=d_k$rhat_max, ess_kappa=d_k$ess_bulk_min,
             rhat_theta=d_th$rhat_max, ess_theta=d_th$ess_bulk_min,
             post_s=pm_s, post_kappa=pm_k, truth=dat$truth,
             summary=sprintf("s Rhat=%.3f ess=%.0f kappa Rhat=%.3f ess=%.0f theta Rhat=%.3f post_kappa=%.1f(true 20)",
                 d_s$rhat_max, d_s$ess_bulk_min,
                 d_k$rhat_max, d_k$ess_bulk_min,
                 d_th$rhat_max, pm_k))
    }
)

# ====== 6. BartNoise ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "BartNoise.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("BartNoise",
    fn_simulate = function(s) {
        set.seed(s); N <- 80L; p <- 3L
        X <- matrix(rnorm(N*p), N, p)
        f_true <- 3*X[,1] + X[,2]^2
        y <- as.numeric(f_true + rnorm(N, 0, 0.8))
        list(X=X, y=y, f_true=f_true, truth=list(sigma=0.8))
    },
    fn_run = function(dat, cs, nb, nk) {
        set.seed(cs)
        m <- new(BartNoise, dat$X, dat$y, 50L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE,
                 as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(f_bart = h$f_bart[(nb+1):(nb+nk), , drop=FALSE],
             sigma = h$sigma[(nb+1):(nb+nk)],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        d_sig <- diag_arr(pack2(c1$sigma, c2$sigma))
        LL1 <- matrix(NA_real_, length(c1$sigma), length(dat$y))
        LL2 <- matrix(NA_real_, length(c2$sigma), length(dat$y))
        for (i in seq_along(c1$sigma)) LL1[i,] <- dnorm(dat$y, c1$f_bart[i,], c1$sigma[i], log=TRUE)
        for (i in seq_along(c2$sigma)) LL2[i,] <- dnorm(dat$y, c2$f_bart[i,], c2$sigma[i], log=TRUE)
        lo <- loo_from(LL1, LL2)
        pm_f <- colMeans(rbind(c1$f_bart, c2$f_bart))
        rmse_f <- sqrt(mean((pm_f - dat$f_true)^2))
        post_sig <- mean(c(c1$sigma, c2$sigma))
        list(rhat_sigma=d_sig$rhat_max, ess_sigma=d_sig$ess_bulk_min,
             post_sigma=post_sig, truth_sigma=dat$truth$sigma, rmse_f=rmse_f, loo=lo,
             summary=sprintf("sigma Rhat=%.4f ess=%.0f post=%.3f(true 0.8) fRMSE=%.3f LOOk<0.5=%.0f%%",
                 d_sig$rhat_max, d_sig$ess_bulk_min, post_sig, rmse_f, lo$pct_k_ok))
    }
)

# ====== 7. GBartPoisson (genBART RJMCMC) ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GBartPoisson.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("GBartPoisson",
    fn_simulate = function(s) {
        set.seed(s); N <- 80L; p <- 3L
        X <- matrix(runif(N*p), N, p)
        r_true <- 1 + 2*X[,1] - X[,2]   # r ≡ log_f (log-rate)
        y <- rpois(N, exp(r_true))
        list(X=X, y=y, r_true=r_true)
    },
    fn_run = function(dat, cs, nb, nk) {
        m <- new(GBartPoisson, dat$X, as.numeric(dat$y), 50L, as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(r    = h$r[(nb+1):(nb+nk), , drop=FALSE],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        LL1 <- matrix(NA_real_, nrow(c1$r), length(dat$y))
        LL2 <- matrix(NA_real_, nrow(c2$r), length(dat$y))
        for (i in seq_len(nrow(c1$r))) LL1[i,] <- dpois(dat$y, exp(c1$r[i,]), log=TRUE)
        for (i in seq_len(nrow(c2$r))) LL2[i,] <- dpois(dat$y, exp(c2$r[i,]), log=TRUE)
        lo <- loo_from(LL1, LL2)
        pm_r <- colMeans(rbind(c1$r, c2$r))
        rmse <- sqrt(mean((pm_r - dat$r_true)^2))
        list(rmse=rmse, loo=lo,
             summary=sprintf("r RMSE=%.3f LOOk<0.5=%.0f%%", rmse, lo$pct_k_ok))
    }
)

# ====== 8. ARDLasso ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "ARDLasso.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("ARDLasso",
    fn_simulate = function(s) {
        set.seed(s); N <- 100L; p <- 10L
        X <- matrix(rnorm(N*p), N, p)
        beta <- c(3, -2, 0, 0, 1.5, 0, 0, 0, 0, 0)
        y <- as.numeric(2.0 + X %*% beta + rnorm(N, 0, 1.0))
        list(X=X, y=y, truth=list(alpha=2.0, beta=beta, sigma=1.0))
    },
    fn_run = function(dat, cs, nb, nk) {
        m <- new(ARDLasso, dat$X, dat$y, as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(alpha = h$alpha[(nb+1):(nb+nk)],
             beta = h$beta[(nb+1):(nb+nk), , drop=FALSE],
             sigma2 = h$sigma2[(nb+1):(nb+nk)],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        d_a <- diag_arr(pack2(c1$alpha, c2$alpha))
        d_b <- diag_arr(pack2(c1$beta, c2$beta))
        d_s <- diag_arr(pack2(c1$sigma2, c2$sigma2))
        pm_b <- colMeans(rbind(c1$beta, c2$beta))
        ps_b <- apply(rbind(c1$beta, c2$beta), 2, sd)
        rec_b <- 100 * mean(abs(pm_b - dat$truth$beta) < 3 * ps_b)
        LL1 <- matrix(NA_real_, length(c1$alpha), length(dat$y))
        LL2 <- matrix(NA_real_, length(c2$alpha), length(dat$y))
        for (i in seq_along(c1$alpha)) LL1[i,] <- dnorm(dat$y,
            as.numeric(dat$X %*% c1$beta[i,]) + c1$alpha[i], sqrt(c1$sigma2[i]), log=TRUE)
        for (i in seq_along(c2$alpha)) LL2[i,] <- dnorm(dat$y,
            as.numeric(dat$X %*% c2$beta[i,]) + c2$alpha[i], sqrt(c2$sigma2[i]), log=TRUE)
        lo <- loo_from(LL1, LL2)
        list(rhat_alpha=d_a$rhat_max, rhat_beta=d_b$rhat_max, rhat_sigma2=d_s$rhat_max,
             ess_beta=d_b$ess_bulk_min,
             post_beta=pm_b, truth=dat$truth, rec_beta=rec_b, loo=lo,
             summary=sprintf("beta Rhat=%.4f ess=%.0f rec=%.0f%% LOOk<0.5=%.0f%%",
                 d_b$rhat_max, d_b$ess_bulk_min, rec_b, lo$pct_k_ok))
    }
)

# ====== 9. IRT1PL_joint (N=30, J=8 — baseline) ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "IRT1PL_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("IRT1PL_joint",
    fn_simulate = function(s) {
        set.seed(s); N <- 30L; J <- 8L
        theta_t <- rnorm(N, 0, 1); sb_t <- 0.8
        b_t <- rnorm(J, 0, sb_t)
        P <- outer(theta_t, b_t, function(a, bb) 1/(1+exp(-(a-bb))))
        Y <- matrix(rbinom(N*J, 1, P), N, J)
        Y[matrix(runif(N*J) < 0.1, N, J)] <- NA_real_
        list(Y=Y, N=N, J=J,
             truth=list(theta=theta_t, b=b_t, sigma_b=sb_t))
    },
    fn_run = function(dat, cs, nb, nk) {
        th_init <- if (cs == 101L) rep(0, dat$N) else rnorm(dat$N, 0, 0.5)
        b_init  <- if (cs == 101L) rep(0, dat$J) else rnorm(dat$J, 0, 0.5)
        sb_init <- if (cs == 101L) 1.0 else 0.5
        m <- new(IRT1PL_joint, dat$Y, th_init, b_init, as.numeric(sb_init),
                 as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(theta = h$theta[(nb+1):(nb+nk), , drop=FALSE],
             b = h$b[(nb+1):(nb+nk), , drop=FALSE],
             sigma_b = h$sigma_b[(nb+1):(nb+nk)],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        d_th <- diag_arr(pack2(c1$theta, c2$theta))
        d_b  <- diag_arr(pack2(c1$b, c2$b))
        d_sb <- diag_arr(pack2(c1$sigma_b, c2$sigma_b))
        pm_th <- colMeans(rbind(c1$theta, c2$theta))
        ps_th <- apply(rbind(c1$theta, c2$theta), 2, sd)
        rec_th <- 100 * mean(abs(pm_th - dat$truth$theta) < 3 * ps_th)
        pm_b <- colMeans(rbind(c1$b, c2$b))
        ps_b <- apply(rbind(c1$b, c2$b), 2, sd)
        rec_b <- 100 * mean(abs(pm_b - dat$truth$b) < 3 * ps_b)
        pm_sb <- mean(c(c1$sigma_b, c2$sigma_b))
        list(rhat_theta=d_th$rhat_max, rhat_b=d_b$rhat_max, rhat_sb=d_sb$rhat_max,
             rec_theta=rec_th, rec_b=rec_b, post_sb=pm_sb, truth_sb=dat$truth$sigma_b,
             summary=sprintf("th Rhat=%.3f b Rhat=%.3f sb Rhat=%.3f rec(th:%.0f%%,b:%.0f%%) post_sb=%.3f",
                 d_th$rhat_max, d_b$rhat_max, d_sb$rhat_max, rec_th, rec_b, pm_sb))
    }
)

# ====== 10. HierarchicalLM_joint ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "HierarchicalLM_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("HierarchicalLM_joint",
    fn_simulate = function(s) {
        set.seed(s); G <- 10L; Np <- 15L; N <- G*Np; p <- 3L
        alpha <- 1.0; beta <- c(2, -1, 0.5); sigma <- 0.8; tau <- 1.5
        X <- matrix(rnorm(N*p), N, p); g_idx <- rep(seq_len(G), each=Np)
        u <- rnorm(G, 0, tau)
        y <- as.numeric(alpha + X %*% beta + u[g_idx] + rnorm(N, 0, sigma))
        list(X=X, y=y, g_idx=g_idx, G=G,
             truth=list(alpha=alpha, beta=beta, sigma=sigma, tau=tau))
    },
    fn_run = function(dat, cs, nb, nk) {
        m <- new(HierarchicalLM_joint, dat$y, dat$X, as.integer(dat$g_idx), dat$G,
                 1.0, 1.0, as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(alpha = h$alpha[(nb+1):(nb+nk)],
             beta  = h$beta[(nb+1):(nb+nk), , drop=FALSE],
             u     = h$u[(nb+1):(nb+nk), , drop=FALSE],
             sigma = h$sigma[(nb+1):(nb+nk)],
             tau   = h$tau[(nb+1):(nb+nk)],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        d_a  <- diag_arr(pack2(c1$alpha, c2$alpha))
        d_b  <- diag_arr(pack2(c1$beta, c2$beta))
        d_u  <- diag_arr(pack2(c1$u, c2$u))
        d_s  <- diag_arr(pack2(c1$sigma, c2$sigma))
        d_t  <- diag_arr(pack2(c1$tau, c2$tau))
        pm_b <- colMeans(rbind(c1$beta, c2$beta))
        ps_b <- apply(rbind(c1$beta, c2$beta), 2, sd)
        rec_b <- 100 * mean(abs(pm_b - dat$truth$beta) < 3 * ps_b)
        pm_s <- mean(c(c1$sigma, c2$sigma)); ps_s <- sd(c(c1$sigma, c2$sigma))
        rec_s <- abs(pm_s - dat$truth$sigma) < 3 * ps_s
        list(rhat_alpha=d_a$rhat_max, rhat_beta=d_b$rhat_max,
             rhat_u=d_u$rhat_max, rhat_sigma=d_s$rhat_max, rhat_tau=d_t$rhat_max,
             rec_beta=rec_b, rec_sigma=rec_s, post_sigma=pm_s,
             summary=sprintf("a Rhat=%.3f b Rhat=%.3f u Rhat=%.3f s Rhat=%.3f t Rhat=%.3f rec_b=%.0f%%",
                 d_a$rhat_max, d_b$rhat_max, d_u$rhat_max, d_s$rhat_max, d_t$rhat_max, rec_b))
    }
)

# ====== 11. LinearRegJointMixed ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "LinearRegJointMixed.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("LinearRegJointMixed",
    fn_simulate = function(s) {
        set.seed(s); N <- 200L; p <- 5L
        X <- matrix(rnorm(N*p), N, p)
        alpha <- 1.5; beta <- c(2, -1, 0.5, 0, 3); sigma <- 1.2
        y <- as.numeric(alpha + X %*% beta + rnorm(N, 0, sigma))
        list(X=X, y=y, truth=list(alpha=alpha, beta=beta, sigma=sigma))
    },
    fn_run = function(dat, cs, nb, nk) {
        m <- new(LinearRegJointMixed, dat$y, dat$X, as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(alpha = h$alpha[(nb+1):(nb+nk)],
             beta = h$beta[(nb+1):(nb+nk), , drop=FALSE],
             sigma = h$sigma[(nb+1):(nb+nk)],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        d_a <- diag_arr(pack2(c1$alpha, c2$alpha))
        d_b <- diag_arr(pack2(c1$beta, c2$beta))
        d_s <- diag_arr(pack2(c1$sigma, c2$sigma))
        pm_a <- mean(c(c1$alpha, c2$alpha)); ps_a <- sd(c(c1$alpha, c2$alpha))
        pm_s <- mean(c(c1$sigma, c2$sigma)); ps_s <- sd(c(c1$sigma, c2$sigma))
        pm_b <- colMeans(rbind(c1$beta, c2$beta))
        ps_b <- apply(rbind(c1$beta, c2$beta), 2, sd)
        rec_b <- 100 * mean(abs(pm_b - dat$truth$beta) < 3 * ps_b)
        rec_a <- abs(pm_a - dat$truth$alpha) < 3 * ps_a
        rec_s <- abs(pm_s - dat$truth$sigma) < 3 * ps_s
        LL1 <- matrix(NA_real_, length(c1$alpha), length(dat$y))
        LL2 <- matrix(NA_real_, length(c2$alpha), length(dat$y))
        for (i in seq_along(c1$alpha)) LL1[i,] <- dnorm(dat$y,
            c1$alpha[i] + as.numeric(dat$X %*% c1$beta[i,]), c1$sigma[i], log=TRUE)
        for (i in seq_along(c2$alpha)) LL2[i,] <- dnorm(dat$y,
            c2$alpha[i] + as.numeric(dat$X %*% c2$beta[i,]), c2$sigma[i], log=TRUE)
        lo <- loo_from(LL1, LL2)
        list(rhat_alpha=d_a$rhat_max, rhat_beta=d_b$rhat_max, rhat_sigma=d_s$rhat_max,
             rec_alpha=rec_a, rec_beta=rec_b, rec_sigma=rec_s,
             post_sigma=pm_s, loo=lo,
             summary=sprintf("a Rhat=%.3f b Rhat=%.3f s Rhat=%.3f rec(a:%s,b:%.0f%%,s:%s) LOOk<0.5=%.0f%%",
                 d_a$rhat_max, d_b$rhat_max, d_s$rhat_max,
                 rec_a, rec_b, rec_s, lo$pct_k_ok))
    }
)

# ====== 12. GBartLogistic (binary classification via genBART direct sigmoid;
#            replaces archived LogisticBART + multinomial_gamma path) ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GBartLogistic.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("GBartLogistic",
    fn_simulate = function(s) {
        set.seed(s); N <- 150L; p <- 5L
        X <- matrix(rnorm(N * p), N, p)
        beta <- c(1.5, -1.0, 0.5, 0.0, -0.8)
        logit <- X %*% beta
        prob  <- 1 / (1 + exp(-logit))
        y <- as.numeric(runif(N) < prob)
        list(X = X, y = y, prob_true = as.numeric(prob), logit_true = as.numeric(logit))
    },
    fn_run = function(dat, cs, nb, nk) {
        set.seed(cs)
        m <- new(GBartLogistic, dat$X, dat$y, 50L, as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(r    = h$r[(nb+1):(nb+nk), , drop=FALSE],
             wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        d_r   <- diag_arr(pack2(c1$r, c2$r))
        pm_r  <- colMeans(rbind(c1$r, c2$r))
        prob_pm <- 1 / (1 + exp(-pm_r))
        rmse_prob <- sqrt(mean((prob_pm - dat$prob_true)^2))
        cor_prob  <- cor(prob_pm, dat$prob_true)
        # Pointwise log-lik for LOO
        LL1 <- matrix(NA_real_, nrow(c1$r), length(dat$y))
        LL2 <- matrix(NA_real_, nrow(c2$r), length(dat$y))
        for (i in seq_len(nrow(c1$r))) {
            p_i <- 1 / (1 + exp(-c1$r[i, ]))
            LL1[i, ] <- dbinom(dat$y, 1, p_i, log = TRUE)
        }
        for (i in seq_len(nrow(c2$r))) {
            p_i <- 1 / (1 + exp(-c2$r[i, ]))
            LL2[i, ] <- dbinom(dat$y, 1, p_i, log = TRUE)
        }
        lo <- loo_from(LL1, LL2)
        list(rhat_max = d_r$rhat_max, ess_min = d_r$ess_bulk_min,
             rmse_prob = rmse_prob, cor_prob = cor_prob, loo = lo,
             summary = sprintf(
                "r Rhat=%.3f ess=%.0f prob RMSE=%.3f corr=%.3f LOOk<0.5=%.0f%%",
                d_r$rhat_max, d_r$ess_bulk_min, rmse_prob, cor_prob,
                lo$pct_k_ok))
    }
)

# ====== 13. GBartMultinomial (C=4 via genBART + poisson_multinomial_aug_block;
#            replaces archived MultinomialLogisticBART) ======
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "GBartMultinomial.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
run_one_example("GBartMultinomial",
    fn_simulate = function(s) {
        set.seed(s); N <- 120L; p <- 5L; C <- 4L
        X <- matrix(rnorm(N * p), N, p)
        B <- cbind(c(1.5, -1.0, 0.5, 0.0, -0.8),
                   c(-0.8, 1.2, 0.0, -0.5, 0.3),
                   c(0.3, 0.5, 1.0, -1.2, 0.0))
        logits <- X %*% B   # N x (C-1)
        full <- cbind(0, logits)
        full <- exp(full - apply(full, 1, max))
        probs_true <- full / rowSums(full)   # N x C
        y <- apply(probs_true, 1, function(p) sample.int(C, 1, prob = p)) - 1L
        list(X = X, y = y, C = C, probs_true = probs_true)
    },
    fn_run = function(dat, cs, nb, nk) {
        set.seed(cs)
        m <- new(GBartMultinomial, dat$X, as.numeric(dat$y),
                 dat$C, 50L, as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        # Collect non-reference r_j histories into a list of matrices.
        # Key names are "r_1", "r_2", ... (genbart linear predictor scale).
        lf_list <- lapply(seq_len(dat$C - 1), function(j) {
            h[[paste0("r_", j)]][(nb+1):(nb+nk), , drop=FALSE]
        })
        list(lf_list = lf_list, wall = as.numeric(difftime(t1, t0, units="secs")))
    },
    fn_diagnose = function(dat, c1, c2) {
        # R-hat per non-reference class
        rhat_max <- 0; ess_min <- Inf
        for (j in seq_len(dat$C - 1)) {
            d <- diag_arr(pack2(c1$lf_list[[j]], c2$lf_list[[j]]))
            rhat_max <- max(rhat_max, d$rhat_max)
            ess_min  <- min(ess_min,  d$ess_bulk_min)
        }
        # Posterior mean probs at training X
        lf_pm <- sapply(seq_len(dat$C - 1), function(j) {
            colMeans(rbind(c1$lf_list[[j]], c2$lf_list[[j]]))
        })
        full <- cbind(0, lf_pm)
        full <- exp(full - apply(full, 1, max))
        probs_pm <- full / rowSums(full)
        rmse_probs <- sqrt(mean((probs_pm - dat$probs_true)^2))
        # Per-class mean correlation with truth
        cor_per_class <- sapply(seq_len(dat$C), function(c_ix) {
            cor(probs_pm[, c_ix], dat$probs_true[, c_ix])
        })
        # Pointwise log-lik for LOO (Categorical per-obs)
        n_draws_1 <- nrow(c1$lf_list[[1]]); n_draws_2 <- nrow(c2$lf_list[[1]])
        LL1 <- matrix(NA_real_, n_draws_1, length(dat$y))
        LL2 <- matrix(NA_real_, n_draws_2, length(dat$y))
        for (i in seq_len(n_draws_1)) {
            lf_i <- sapply(seq_len(dat$C - 1), function(j) c1$lf_list[[j]][i, ])
            full <- cbind(0, lf_i)
            full <- exp(full - apply(full, 1, max))
            probs_i <- full / rowSums(full)
            LL1[i, ] <- log(probs_i[cbind(seq_along(dat$y), dat$y + 1)])
        }
        for (i in seq_len(n_draws_2)) {
            lf_i <- sapply(seq_len(dat$C - 1), function(j) c2$lf_list[[j]][i, ])
            full <- cbind(0, lf_i)
            full <- exp(full - apply(full, 1, max))
            probs_i <- full / rowSums(full)
            LL2[i, ] <- log(probs_i[cbind(seq_along(dat$y), dat$y + 1)])
        }
        lo <- loo_from(LL1, LL2)
        list(rhat_max = rhat_max, ess_min = ess_min,
             rmse_probs = rmse_probs, cor_per_class = cor_per_class, loo = lo,
             summary = sprintf(
                "log_f Rhat=%.3f ess=%.0f probs RMSE=%.3f cor(min)=%.3f LOOk<0.5=%.0f%%",
                rhat_max, ess_min, rmse_probs, min(cor_per_class),
                lo$pct_k_ok))
    }
)

# ====== F1 summary ======
cat("\n\n========== PHASE F1 FINAL SUMMARY ==========\n")
for (nm in names(results_F1)) {
    cat(sprintf("\n--- %s ---\n", nm))
    for (s in names(results_F1[[nm]])) {
        cat(sprintf("  seed=%s : %s\n", s, results_F1[[nm]][[s]]$summary))
    }
}
saveRDS(results_F1, file.path(AI4BayesCode_dir, "audit_xl_F1.rds"))
cat("\n[F1 DONE] saved to audit_xl_F1.rds\n")

# Verify that the DirichletHierarchical kappa mis-fit was a test-harness
# bug. The C++ treats each ROW of S_obs as a P-dim simplex (K observations
# × P-dim). The previous harness generated 20 COLUMN-simplices of 4-dim
# (P_obs=20, K_rows=4, each col sums to 1). Rows of that matrix do NOT
# sum to 1, so the fit was garbage.
#
# Corrected simulation: K=20 rows (observations) × P=4 cols (simplex dim),
# each ROW sums to 1. kappa_true = 20.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))
suppressPackageStartupMessages({ library(posterior) })

ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletHierarchical.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

pack2 <- function(x1, x2) {
    if (is.null(dim(x1)))
        array(cbind(x1, x2), dim = c(length(x1), 2, 1))
    else {
        arr <- array(NA_real_, dim = c(nrow(x1), 2, ncol(x1)))
        arr[,1,] <- x1; arr[,2,] <- x2
        arr
    }
}

K_obs <- 20L                         # number of observations (rows)
P_dim <- 4L                          # simplex dimension (columns)
s_true <- c(0.1, 0.3, 0.4, 0.2)
kappa_true <- 20.0

for (seed in c(1, 42, 123)) {
    cat(sprintf("\n--- seed = %d ---\n", seed))
    set.seed(seed)
    S_obs <- matrix(NA_real_, K_obs, P_dim)
    for (k in 1:K_obs) {
        g <- rgamma(P_dim, shape = kappa_true * s_true)
        S_obs[k,] <- g / sum(g)               # ROW sums to 1
    }
    stopifnot(all(abs(rowSums(S_obs) - 1) < 1e-12))

    run <- function(cs, nb, nk) {
        m <- new(DirichletHierarchical, S_obs, 1.0, 1.0, as.integer(cs), TRUE)
        t0 <- Sys.time(); m$step(nb); m$step(nk); t1 <- Sys.time()
        h <- m$get_history()
        list(s=h$s[(nb+1):(nb+nk), , drop=FALSE],
             kappa=h$kappa[(nb+1):(nb+nk)],
             theta=h$theta[(nb+1):(nb+nk)],
             wall=as.numeric(difftime(t1, t0, units="secs")))
    }
    c1 <- run(101L, 5000L, 5000L)
    c2 <- run(202L, 5000L, 5000L)

    arr_k <- pack2(c1$kappa, c2$kappa)
    rhat_k <- posterior::rhat(arr_k[,,1])
    ess_k  <- posterior::ess_bulk(arr_k[,,1])
    post_k <- mean(c(c1$kappa, c2$kappa))
    q_k    <- quantile(c(c1$kappa, c2$kappa), c(0.025, 0.5, 0.975), names=FALSE)

    pm_s <- colMeans(rbind(c1$s, c2$s))
    arr_s <- pack2(c1$s, c2$s)
    rhat_s_max <- max(sapply(seq_len(P_dim), function(j) posterior::rhat(arr_s[,,j])))

    cat(sprintf("  wall=%.1fs  kappa: Rhat=%.4f ess=%.0f post_mean=%.1f  95%%CI=[%.1f, %.1f]\n",
        c1$wall + c2$wall, rhat_k, ess_k, post_k, q_k[1], q_k[3]))
    cat(sprintf("             s post_mean=(%s) truth=(%s) Rhat_max=%.4f\n",
        paste(sprintf("%.3f", pm_s), collapse=","),
        paste(sprintf("%.3f", s_true), collapse=","),
        rhat_s_max))
}
cat("\n[DONE]\n")

# ============================================================================
# gbart_poisson_4chain_extended.R
#
# 4-chain extended MCMC run for GBartPoisson: 4000 burnin + 10000 keep
# per chain, with rank-normalized split-R-hat + ESS diagnostics.
# ============================================================================

source("R/AI4BayesCode_helpers.R")
suppressPackageStartupMessages(library(posterior))

AI4BayesCode_sourceCpp("r-pkg/inst/examples/GBartPoisson.cpp",
                       AI4BayesCode_path = ".")

NBURN <- 4000L
NKEEP <- 10000L

set.seed(42)
N <- 200L; p <- 3L
X <- matrix(runif(N*p), N, p)
truth <- -1 + X[,1] - 0.5*X[,2] + 0.3*X[,3]
y <- rpois(N, exp(truth))

run_chain <- function(seed) {
    set.seed(seed)
    m <- new(GBartPoisson, X, as.numeric(y), 200L, seed, TRUE)
    t0 <- Sys.time()
    m$step(NBURN); m$step(NKEEP)
    t1 <- Sys.time()
    list(hist = m$get_history(),
         wall_sec = as.numeric(difftime(t1, t0, units = "secs")))
}

cat(sprintf("=== Running 4 chains x %d burnin + %d keep on N=%d, p=%d ===\n",
            NBURN, NKEEP, N, p))

t_all <- Sys.time()
chains <- lapply(c(101L, 202L, 303L, 404L), run_chain)
t_all_end <- Sys.time()

for (i in seq_along(chains)) {
    cat(sprintf("  chain %d wall = %.1fs (%.2f sec/sweep)\n", i,
                chains[[i]]$wall_sec,
                chains[[i]]$wall_sec / (NBURN + NKEEP)))
}
cat(sprintf("  total wall (sequential) = %.1fs\n",
            as.numeric(difftime(t_all_end, t_all, units="secs"))))

# Stack per-observation r draws: (n_keep, chain, N).
arr <- array(NA_real_, dim = c(NKEEP, 4, N))
for (i in 1:4) {
    h <- chains[[i]]$hist[["r"]]
    arr[, i, ] <- h[(NBURN + 1):(NBURN + NKEEP), ]
}

rh <- apply(arr, 3, posterior::rhat)
eb <- apply(arr, 3, posterior::ess_bulk)
et <- apply(arr, 3, posterior::ess_tail)

cat("\n=== 4-chain diagnostics (N=200 per-obs r) ===\n")
cat(sprintf("  max Rhat       = %.3f (target < 1.05)\n", max(rh)))
cat(sprintf("  median Rhat    = %.3f\n", median(rh)))
cat(sprintf("  min ESS_bulk   = %.0f (target > 400)\n", min(eb)))
cat(sprintf("  median ESS_bulk= %.0f\n", median(eb)))
cat(sprintf("  min ESS_tail   = %.0f (target > 400)\n", min(et)))

# Per-observation pass/fail summary
pass_rh <- sum(rh < 1.05) / length(rh)
pass_eb <- sum(eb > 400) / length(eb)
cat(sprintf("  pass Rhat<1.05 = %.1f%% of obs\n", 100*pass_rh))
cat(sprintf("  pass ESS_bulk>400 = %.1f%% of obs\n", 100*pass_eb))

# Posterior recovery
r_mean_all <- rowMeans(apply(arr, c(1, 3), mean))  # mean over chain-iter
yhat_mean <- colMeans(apply(arr, c(1, 3), mean))    # per-obs posterior mean
cat(sprintf("\n  cor(posterior mean r, truth) = %.3f\n",
            cor(yhat_mean, truth)))
cat(sprintf("  RMSE(posterior mean - truth)  = %.3f\n",
            sqrt(mean((yhat_mean - truth)^2))))

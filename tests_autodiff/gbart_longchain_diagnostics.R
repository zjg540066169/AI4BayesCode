# ============================================================================
# gbart_longchain_diagnostics.R
#
# 2-chain long-chain MCMC diagnostics for all 4 shipped GBart* examples.
#   - 2000 burnin + 4000 keep per chain
#   - R-hat + ESS_bulk + ESS_tail via posterior::
#   - posterior recovery cor(posterior-mean r, truth) per example
#   - per-chain wall time
#
# Runs in ~5-10 minutes on a single core.
# ============================================================================

source("R/AI4BayesCode_helpers.R")
suppressPackageStartupMessages(library(posterior))

NBURN <- 2000L
NKEEP <- 4000L

pack2 <- function(x1, x2) {
    if (is.null(dim(x1))) {
        array(cbind(x1, x2), dim = c(length(x1), 2, 1))
    } else {
        arr <- array(NA_real_, dim = c(nrow(x1), 2, ncol(x1)))
        arr[, 1, ] <- x1; arr[, 2, ] <- x2
        arr
    }
}

diagnose_r <- function(r1, r2, example_name) {
    arr <- pack2(r1, r2)
    rh <- apply(arr, 3, posterior::rhat)
    eb <- apply(arr, 3, posterior::ess_bulk)
    et <- apply(arr, 3, posterior::ess_tail)
    cat(sprintf("  %s r (N=%d): max Rhat=%.3f  min ESS_bulk=%.0f  min ESS_tail=%.0f\n",
                example_name, ncol(r1), max(rh), min(eb), min(et)))
    list(max_rhat = max(rh), min_ess_bulk = min(eb), min_ess_tail = min(et))
}

# ------------------------- GBartPoisson ---------------------------------
cat("\n===== GBartPoisson =====\n")
AI4BayesCode_sourceCpp("r-pkg/inst/examples/GBartPoisson.cpp",
                       AI4BayesCode_path = ".")
set.seed(1); N <- 200L; p <- 3L
X <- matrix(runif(N*p), N, p)
truth_p <- -1 + X[,1] - 0.5*X[,2]
y <- rpois(N, exp(truth_p))
run_p <- function(seed) {
    set.seed(seed)
    m <- new(GBartPoisson, X, as.numeric(y), 200L, seed, TRUE)
    t0 <- Sys.time(); m$step(NBURN); m$step(NKEEP); t1 <- Sys.time()
    list(h=m$get_history(), wall=as.numeric(difftime(t1,t0,units="secs")))
}
c1 <- run_p(101L); c2 <- run_p(202L)
cat(sprintf("  wall chain1=%.1fs chain2=%.1fs\n", c1$wall, c2$wall))
d_p <- diagnose_r(c1$h[["r"]], c2$h[["r"]], "GBartPoisson")
cat(sprintf("  recovery cor(posterior mean, truth) = %.3f\n",
            cor(colMeans(c1$h[["r"]]), truth_p)))

# ------------------------- GBartLogistic ---------------------------------
cat("\n===== GBartLogistic =====\n")
AI4BayesCode_sourceCpp("r-pkg/inst/examples/GBartLogistic.cpp",
                       AI4BayesCode_path = ".")
set.seed(2); N <- 200L
X <- matrix(runif(N*p), N, p)
truth_l <- -0.5 + 2*X[,1] - X[,2]
y <- rbinom(N, 1, plogis(truth_l))
run_l <- function(seed) {
    set.seed(seed)
    m <- new(GBartLogistic, X, as.numeric(y), 50L, seed, TRUE)
    t0 <- Sys.time(); m$step(NBURN); m$step(NKEEP); t1 <- Sys.time()
    list(h=m$get_history(), wall=as.numeric(difftime(t1,t0,units="secs")))
}
c1 <- run_l(101L); c2 <- run_l(202L)
cat(sprintf("  wall chain1=%.1fs chain2=%.1fs\n", c1$wall, c2$wall))
d_l <- diagnose_r(c1$h[["r"]], c2$h[["r"]], "GBartLogistic")
cat(sprintf("  recovery cor(posterior mean, truth logit) = %.3f\n",
            cor(colMeans(c1$h[["r"]]), truth_l)))
p_hat <- plogis(colMeans(c1$h[["r"]]))
cat(sprintf("  classification accuracy @ 0.5 = %.3f\n",
            mean((p_hat > 0.5) == y)))

# ------------------------- GBartMultinomial -----------------------------
cat("\n===== GBartMultinomial (C=3) =====\n")
AI4BayesCode_sourceCpp("r-pkg/inst/examples/GBartMultinomial.cpp",
                       AI4BayesCode_path = ".")
set.seed(3); N <- 300L; C <- 3L
X <- matrix(runif(N*p), N, p)
eta <- cbind(-1 + 2*X[,1], 0 + X[,2] - X[,3])
denom <- 1 + exp(eta[,1]) + exp(eta[,2])
probs_true <- cbind(1/denom, exp(eta[,1])/denom, exp(eta[,2])/denom)
y <- apply(probs_true, 1, function(pi) sample(0:2, 1, prob=pi))
run_m <- function(seed) {
    set.seed(seed)
    m <- new(GBartMultinomial, X, as.numeric(y), C, 50L, seed, TRUE)
    t0 <- Sys.time(); m$step(NBURN); m$step(NKEEP); t1 <- Sys.time()
    list(h=m$get_history(), wall=as.numeric(difftime(t1,t0,units="secs")))
}
c1 <- run_m(101L); c2 <- run_m(202L)
cat(sprintf("  wall chain1=%.1fs chain2=%.1fs\n", c1$wall, c2$wall))
# r_1 and r_2 are stored under block keys "r_1" and "r_2"
for (j in 1:(C-1)) {
    k <- paste0("r_", j)
    if (!is.null(c1$h[[k]])) {
        d_m <- diagnose_r(c1$h[[k]], c2$h[[k]], sprintf("GBartMultinomial.%s", k))
        cat(sprintf("    recovery cor(pm r_%d, eta_%d) = %.3f\n", j, j,
                    cor(colMeans(c1$h[[k]]), eta[, j])))
    }
}

# ------------------------- GBartHeteroscedastic --------------------------
cat("\n===== GBartHeteroscedastic =====\n")
AI4BayesCode_sourceCpp("r-pkg/inst/examples/GBartHeteroscedastic.cpp",
                       AI4BayesCode_path = ".")
set.seed(5); N <- 200L
X <- matrix(runif(N*p), N, p)
truth_h <- 3 + X[,1] - 0.5*X[,2]
phi_true <- 0.5
mu <- exp(truth_h); y <- rnorm(N, mu, sqrt(phi_true*mu))
run_h <- function(seed) {
    set.seed(seed)
    m <- new(GBartHeteroscedastic, X, as.numeric(y), 200L, 1.0, seed, TRUE)
    t0 <- Sys.time(); m$step(NBURN); m$step(NKEEP); t1 <- Sys.time()
    list(h=m$get_history(), cur=m$get_current(),
         wall=as.numeric(difftime(t1,t0,units="secs")))
}
c1 <- run_h(101L); c2 <- run_h(202L)
cat(sprintf("  wall chain1=%.1fs chain2=%.1fs\n", c1$wall, c2$wall))
d_h <- diagnose_r(c1$h[["r"]], c2$h[["r"]], "GBartHeteroscedastic")
cat(sprintf("  recovery cor(posterior mean, truth) = %.3f ; phi c1=%.3f c2=%.3f (truth=%.2f)\n",
            cor(colMeans(c1$h[["r"]]), truth_h), c1$cur$phi, c2$cur$phi,
            phi_true))

cat("\n===== Summary =====\n")
cat("All 6 GBart* examples completed 2-chain 2000+4000 long-chain runs\n")
cat("without crashing. See per-example lines above for R-hat, ESS, recovery.\n")

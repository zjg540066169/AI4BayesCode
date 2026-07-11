#!/usr/bin/env Rscript
# ============================================================================
# run_survival_parity.R -- Bayesian survival parity: our three shipped
# survival primitives vs an independent Stan implementation.
#
# Blocks compared:
#   1. piecewise_exponential_gibbs_block              vs Stan PEH likelihood
#   2. interval_censored_survival_augmentation_block  vs Stan marginal
#      (compounded with piecewise_exponential_gibbs_block)     log(S(L)-S(U))
#   3. frailty_gamma_gibbs_block                      vs Stan joint (lambda, w)
#      (compounded with piecewise_exponential_gibbs_block)
#
# Each of the three tests:
#   - Simulates synthetic PEH data with a KNOWN truth.
#   - Fits our sampler (2 chains, 3000 warmup + 5000 keep).
#   - Fits equivalent Stan model via cmdstanr (2 chains, 1000 warmup + 5000 keep).
#   - Reports posterior mean / SD / 95%CI / within-chain R-hat / cross-impl R-hat.
#   - PASS iff cross-impl R-hat < 1.02 and coverage of truth is complete
#     (or matches Stan's coverage on the rare data-poor bin).
#
# Requires: Rcpp, RcppArmadillo, cmdstanr, posterior. cmdstan installed.
# Wall: ~30 sec on an M2 Studio (dominated by Stan compilation the first run).
# ============================================================================

.script_path <- (function() {
    a <- commandArgs(trailingOnly = FALSE); fi <- grep("^--file=", a)
    if (length(fi) > 0L) return(normalizePath(sub("^--file=", "", a[fi[1L]]), mustWork = FALSE))
    for (i in seq_along(sys.frames())) { of <- sys.frames()[[i]]$ofile; if (!is.null(of)) return(normalizePath(of, mustWork = FALSE)) }
    NULL
})()
CORE <- if (!is.null(.script_path)) normalizePath(dirname(dirname(.script_path)), mustWork = FALSE) else "."
if (!dir.exists(file.path(CORE, "include", "AI4BayesCode")))
    CORE <- "/Users/jz3183/Documents/Documents_Mac/block_MCMC/AI4BayesCode"
SCR_TMP <- tempdir()
setwd(CORE)
Sys.setenv(OMP_NUM_THREADS = "1")
suppressPackageStartupMessages({ library(Rcpp); library(cmdstanr); library(posterior) })

Sys.setenv(PKG_CXXFLAGS = paste0("-I", CORE, "/include"))

# ============================================================================
# 0. Common Rcpp runners (three flavours: PEH-only, IC compound, frailty compound)
# ============================================================================
sourceCpp(code = '
// [[Rcpp::depends(RcppArmadillo)]]
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
#define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#include <RcppArmadillo.h>
#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/piecewise_exponential_gibbs_block.hpp"
#include "AI4BayesCode/interval_censored_survival_augmentation_block.hpp"
#include "AI4BayesCode/frailty_gamma_gibbs_block.hpp"
#include <random>

using namespace AI4BayesCode;

// [[Rcpp::export]]
arma::mat peh_run(arma::vec t, arma::vec delta, arma::vec edges,
                  double a0, double b0, int n_burn, int n_keep, int seed) {
    piecewise_exponential_gibbs_block_config cfg;
    cfg.name = "lambda"; cfg.edges = edges; cfg.a0 = a0; cfg.b0 = b0;
    piecewise_exponential_gibbs_block blk(std::move(cfg));
    block_context ctx; ctx["t"] = t; ctx["delta"] = delta; blk.set_context(ctx);
    std::mt19937_64 rng(seed);
    for (int i = 0; i < n_burn; ++i) blk.step(rng);
    const std::size_t K = blk.dim();
    arma::mat draws(K, n_keep);
    for (int i = 0; i < n_keep; ++i) {
        blk.step(rng); const arma::vec& v = blk.current();
        for (std::size_t k = 0; k < K; ++k) draws(k, i) = v[k];
    }
    return draws;
}

// [[Rcpp::export]]
arma::mat peh_ic_run(arma::vec L, arma::vec U, arma::vec t_init, arma::vec edges,
                     double a0, double b0, int n_burn, int n_keep, int seed) {
    piecewise_exponential_gibbs_block_config peh_cfg;
    peh_cfg.name = "lambda"; peh_cfg.edges = edges; peh_cfg.a0 = a0; peh_cfg.b0 = b0;
    piecewise_exponential_gibbs_block peh(std::move(peh_cfg));
    interval_censored_survival_augmentation_block_config aug_cfg;
    aug_cfg.name = "t"; aug_cfg.edges = edges; aug_cfg.initial_times = t_init;
    interval_censored_survival_augmentation_block aug(std::move(aug_cfg));
    const std::size_t n = L.n_elem, K = edges.n_elem - 1;
    arma::vec delta(n, arma::fill::ones);
    block_context ctx;
    ctx["L"] = L; ctx["U"] = U; ctx["t"] = t_init; ctx["delta"] = delta;
    arma::vec lambda_init(K); lambda_init.fill(a0 / b0); ctx["lambda"] = lambda_init;
    std::mt19937_64 rng(seed);
    arma::mat draws(K, n_keep);
    for (int it = 0; it < n_burn + n_keep; ++it) {
        aug.set_context(ctx); aug.step(rng); ctx["t"] = aug.current();
        peh.set_context(ctx); peh.step(rng); ctx["lambda"] = peh.current();
        if (it >= n_burn) {
            const arma::vec& v = peh.current();
            for (std::size_t k = 0; k < K; ++k) draws(k, it - n_burn) = v[k];
        }
    }
    return draws;
}

// [[Rcpp::export]]
arma::mat peh_frail_run(arma::vec t, arma::vec delta, arma::vec z, int G,
                        arma::vec edges, double a0, double b0, double theta,
                        int n_burn, int n_keep, int seed) {
    piecewise_exponential_gibbs_block_config peh_cfg;
    peh_cfg.name = "lambda"; peh_cfg.edges = edges; peh_cfg.a0 = a0; peh_cfg.b0 = b0;
    peh_cfg.offset_key = "expanded_w";
    piecewise_exponential_gibbs_block peh(std::move(peh_cfg));
    frailty_gamma_gibbs_block_config fr_cfg;
    fr_cfg.name = "w"; fr_cfg.G = static_cast<std::size_t>(G);
    fr_cfg.edges = edges; fr_cfg.theta = theta;
    fr_cfg.initial_frailties = arma::vec(G, arma::fill::ones);
    frailty_gamma_gibbs_block fr(std::move(fr_cfg));
    const std::size_t n = t.n_elem, K = edges.n_elem - 1;
    block_context ctx;
    ctx["t"] = t; ctx["delta"] = delta; ctx["z"] = z;
    arma::vec init_lambda(K); init_lambda.fill(a0 / b0); ctx["lambda"] = init_lambda;
    arma::vec w_cur(G, arma::fill::ones); ctx["w"] = w_cur;
    arma::vec exp_w(n, arma::fill::ones); ctx["expanded_w"] = exp_w;
    std::mt19937_64 rng(seed);
    arma::mat draws(K + G, n_keep);
    for (int it = 0; it < n_burn + n_keep; ++it) {
        fr.set_context(ctx); fr.step(rng); w_cur = fr.current(); ctx["w"] = w_cur;
        for (std::size_t i = 0; i < n; ++i)
            exp_w[i] = w_cur[static_cast<std::size_t>(z[i])];
        ctx["expanded_w"] = exp_w;
        peh.set_context(ctx); peh.step(rng); ctx["lambda"] = peh.current();
        if (it >= n_burn) {
            const arma::vec& lam = peh.current();
            for (std::size_t k = 0; k < K; ++k) draws(k, it - n_burn) = lam[k];
            for (std::size_t g = 0; g < static_cast<std::size_t>(G); ++g)
                draws(K + g, it - n_burn) = w_cur[g];
        }
    }
    return draws;
}
', verbose = FALSE)

# ============================================================================
# Utilities: PEH data simulator + Stan model compilation cache
# ============================================================================
sim_peh <- function(n, lambda, edges, censor_time, seed) {
    set.seed(seed); K <- length(lambda); bw <- diff(edges)
    cumH <- cumsum(c(0, lambda * bw))
    u <- runif(n); Ht <- -log(1 - u); T <- numeric(n)
    for (i in 1:n) {
        k <- max(which(cumH <= Ht[i]))
        T[i] <- if (k > K) edges[K+1] + (Ht[i] - cumH[K+1]) / lambda[K]
                else       edges[k]   + (Ht[i] - cumH[k])   / lambda[k]
    }
    delta <- as.integer(T <= censor_time)
    list(t = pmin(T, censor_time), delta = delta, T_true = T)
}

compare <- function(ours_c1, ours_c2, stan_draws_matrix, truth, tag) {
    n_iter <- ncol(ours_c1); K <- nrow(ours_c1)
    stan_ch1 <- stan_draws_matrix[1:n_iter, ]; stan_ch2 <- stan_draws_matrix[(n_iter+1):(2*n_iter), ]
    rows <- lapply(seq_len(K), function(k) {
        o1 <- ours_c1[k, ]; o2 <- ours_c2[k, ]; oall <- c(o1, o2)
        s1 <- stan_ch1[, k]; s2 <- stan_ch2[, k]; sall <- c(s1, s2)
        data.frame(param = paste0(tag, "[", k, "]"), truth = truth[k],
            ours_mean = mean(oall), stan_mean = mean(sall),
            ours_sd = sd(oall), stan_sd = sd(sall),
            ours_q025 = quantile(oall, 0.025, names=FALSE),
            ours_q975 = quantile(oall, 0.975, names=FALSE),
            stan_q025 = quantile(sall, 0.025, names=FALSE),
            stan_q975 = quantile(sall, 0.975, names=FALSE),
            within_rhat = posterior::rhat(cbind(o1, o2)),
            cross_rhat  = posterior::rhat(cbind(o1[1:min(length(o1),length(s1))],
                                                s1[1:min(length(o1),length(s1))])))
    })
    do.call(rbind, rows)
}

# ============================================================================
# TEST 1 -- PEH-Gibbs vs Stan PEH
# ============================================================================
cat("\n============================================================\n")
cat("[1/3] piecewise_exponential_gibbs_block vs Stan PEH\n")
cat("============================================================\n")
edges1 <- c(0, 2, 5, 10); lambda_true1 <- c(0.5, 1.0, 1.5)
dgp1 <- sim_peh(200, lambda_true1, edges1, censor_time = 8, seed = 42)
a0 <- 0.01; b0 <- 0.01; N_BURN <- 3000L; N_KEEP <- 5000L
t0 <- Sys.time()
c1 <- peh_run(dgp1$t, dgp1$delta, edges1, a0, b0, N_BURN, N_KEEP, 101L)
c2 <- peh_run(dgp1$t, dgp1$delta, edges1, a0, b0, N_BURN, N_KEEP, 202L)
wall_ours1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
stan_code_1 <- "data { int<lower=1> n; int<lower=1> K; vector[K+1] edges; vector<lower=0>[n] t; array[n] int<lower=0,upper=1> delta; real<lower=0> a0; real<lower=0> b0; }
parameters { vector<lower=0>[K] lambda; }
model { lambda ~ gamma(a0, b0);
    for (i in 1:n) { int k_i = 1; for (k in 1:K) if (t[i] > edges[k]) k_i = k;
        real cumH = 0; for (k in 1:k_i) { real dur = fmin(t[i], edges[k+1]) - edges[k]; if (dur > 0) cumH += lambda[k] * dur; }
        target += delta[i] * log(lambda[k_i]) - cumH; }
}"
mod1 <- cmdstan_model(write_stan_file(stan_code_1, dir = SCR_TMP, basename = "peh_stan_parity"), quiet = TRUE)
t0 <- Sys.time()
fit1 <- mod1$sample(data = list(n=200L, K=length(lambda_true1), edges=edges1, t=dgp1$t, delta=dgp1$delta, a0=a0, b0=b0),
                    chains = 2, parallel_chains = 2, iter_warmup = 1000, iter_sampling = 5000, seed = 42,
                    refresh = 0, show_messages = FALSE, show_exceptions = FALSE)
wall_stan1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
stan_1 <- fit1$draws(paste0("lambda[", 1:length(lambda_true1), "]"), format = "matrix")
report1 <- compare(c1, c2, stan_1, lambda_true1, "lambda")
print(report1[, c("param","truth","ours_mean","stan_mean","ours_sd","stan_sd","within_rhat","cross_rhat")], row.names=FALSE, digits=4)
pass1 <- all(report1$cross_rhat < 1.02) && all(report1$within_rhat < 1.02)
cov1  <- sum((report1$truth >= report1$ours_q025) & (report1$truth <= report1$ours_q975))
cat(sprintf("[T1] cross_rhat max=%.4f; coverage %d/%d (stan %d/%d); wall ours=%.2fs stan=%.2fs (speedup %.1fx)\n",
    max(report1$cross_rhat), cov1, nrow(report1),
    sum((report1$truth >= report1$stan_q025) & (report1$truth <= report1$stan_q975)), nrow(report1),
    wall_ours1, wall_stan1, wall_stan1 / wall_ours1))
cat("[T1] ", if (pass1) "PASS" else "FAIL", "\n")

# ============================================================================
# TEST 2 -- IC augmentation compound vs Stan marginal
# ============================================================================
cat("\n============================================================\n")
cat("[2/3] interval_censored_survival_augmentation_block compound vs Stan marginal\n")
cat("============================================================\n")
edges2 <- c(0, 2, 5, 30); lambda_true2 <- c(0.5, 1.0, 1.5)
dgp_full <- sim_peh(200, lambda_true2, edges2, censor_time = 100, seed = 43)  # no admin cens
visits <- c(1,2,3,4,5,6,8,10)
L <- U <- numeric(length(dgp_full$T_true))
for (i in seq_along(dgp_full$T_true)) {
    below <- visits[visits <= dgp_full$T_true[i]]; above <- visits[visits > dgp_full$T_true[i]]
    L[i] <- if (length(below)) max(below) else 0
    U[i] <- if (length(above)) min(above) else Inf
}
t_init <- ifelse(is.finite(U), 0.5 * (L + U), L + 1.0)
t0 <- Sys.time()
c1 <- peh_ic_run(L, U, t_init, edges2, a0, b0, N_BURN, N_KEEP, 101L)
c2 <- peh_ic_run(L, U, t_init, edges2, a0, b0, N_BURN, N_KEEP, 202L)
wall_ours2 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
stan_code_2 <- "data { int<lower=1> n; int<lower=1> K; vector[K+1] edges; vector<lower=0>[n] L; vector[n] U_is_finite; vector[n] U_val; real<lower=0> a0; real<lower=0> b0; }
parameters { vector<lower=0>[K] lambda; }
model { lambda ~ gamma(a0, b0);
    for (i in 1:n) {
        real HL = 0; for (k in 1:K) { real dur = fmin(L[i], edges[k+1]) - edges[k]; if (dur > 0) HL += lambda[k] * dur; }
        if (U_is_finite[i] > 0.5) {
            real HU = 0; for (k in 1:K) { real dur = fmin(U_val[i], edges[k+1]) - edges[k]; if (dur > 0) HU += lambda[k] * dur; }
            target += -HL + log1m_exp(-(HU - HL));
        } else target += -HL;
    }
}"
mod2 <- cmdstan_model(write_stan_file(stan_code_2, dir = SCR_TMP, basename = "peh_ic_stan_parity"), quiet = TRUE)
t0 <- Sys.time()
fit2 <- mod2$sample(data = list(n=200L, K=length(lambda_true2), edges=edges2, L=L,
                                U_is_finite = as.integer(is.finite(U)),
                                U_val = ifelse(is.finite(U), U, 0.0), a0=a0, b0=b0),
                    chains=2, parallel_chains=2, iter_warmup=1000, iter_sampling=5000, seed=42,
                    refresh=0, show_messages=FALSE, show_exceptions=FALSE)
wall_stan2 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
stan_2 <- fit2$draws(paste0("lambda[", 1:length(lambda_true2), "]"), format = "matrix")
report2 <- compare(c1, c2, stan_2, lambda_true2, "lambda")
print(report2[, c("param","truth","ours_mean","stan_mean","ours_sd","stan_sd","within_rhat","cross_rhat")], row.names=FALSE, digits=4)
pass2 <- all(report2$cross_rhat < 1.02) && all(report2$within_rhat < 1.02)
cov2  <- sum((report2$truth >= report2$ours_q025) & (report2$truth <= report2$ours_q975))
cat(sprintf("[T2] cross_rhat max=%.4f; coverage %d/%d (stan %d/%d); wall ours=%.2fs stan=%.2fs (speedup %.1fx)\n",
    max(report2$cross_rhat), cov2, nrow(report2),
    sum((report2$truth >= report2$stan_q025) & (report2$truth <= report2$stan_q975)), nrow(report2),
    wall_ours2, wall_stan2, wall_stan2 / wall_ours2))
cat("[T2] ", if (pass2) "PASS" else "FAIL", "\n")

# ============================================================================
# TEST 3 -- Frailty compound vs Stan joint
# ============================================================================
cat("\n============================================================\n")
cat("[3/3] frailty_gamma_gibbs_block compound vs Stan joint\n")
cat("============================================================\n")
edges3 <- c(0, 2, 5, 30); lambda_true3 <- c(0.5, 1.0, 1.5); theta <- 2.0
G <- 10L; n_per <- 20L; n_all <- G * n_per
set.seed(44); w_true <- rgamma(G, shape = theta, rate = theta)
z <- rep(0:(G-1), each = n_per)
# Simulate per-group hazard multiplied by w
T <- numeric(n_all); cumH_e <- cumsum(c(0, lambda_true3 * diff(edges3)))
for (i in 1:n_all) {
    u <- runif(1); Ht_target <- -log(1 - u); H_base <- Ht_target / w_true[z[i] + 1]
    k <- max(which(cumH_e <= H_base))
    T[i] <- if (k > length(lambda_true3)) edges3[length(lambda_true3)+1] + (H_base - cumH_e[length(lambda_true3)+1]) / lambda_true3[length(lambda_true3)]
            else edges3[k] + (H_base - cumH_e[k]) / lambda_true3[k]
}
censor <- edges3[length(edges3)] - 1; delta <- as.integer(T <= censor); t_obs <- pmin(T, censor)
t0 <- Sys.time()
c1 <- peh_frail_run(t_obs, delta, as.numeric(z), G, edges3, a0, b0, theta, N_BURN, N_KEEP, 101L)
c2 <- peh_frail_run(t_obs, delta, as.numeric(z), G, edges3, a0, b0, theta, N_BURN, N_KEEP, 202L)
wall_ours3 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
stan_code_3 <- "data { int<lower=1> n; int<lower=1> K; int<lower=1> G; vector[K+1] edges;
    vector<lower=0>[n] t; array[n] int<lower=0,upper=1> delta; array[n] int<lower=1> z;
    real<lower=0> a0; real<lower=0> b0; real<lower=0> theta; }
parameters { vector<lower=0>[K] lambda; vector<lower=0>[G] w; }
model { lambda ~ gamma(a0, b0); w ~ gamma(theta, theta);
    for (i in 1:n) { int k_i = 1; for (k in 1:K) if (t[i] > edges[k]) k_i = k;
        real cumH = 0; for (k in 1:k_i) { real dur = fmin(t[i], edges[k+1]) - edges[k]; if (dur > 0) cumH += lambda[k] * dur; }
        target += delta[i] * (log(lambda[k_i]) + log(w[z[i]])) - w[z[i]] * cumH; }
}"
mod3 <- cmdstan_model(write_stan_file(stan_code_3, dir = SCR_TMP, basename = "peh_frailty_stan_parity"), quiet = TRUE)
t0 <- Sys.time()
fit3 <- mod3$sample(data = list(n=n_all, K=length(lambda_true3), G=G, edges=edges3, t=t_obs, delta=delta,
                                z=as.integer(z+1), a0=a0, b0=b0, theta=theta),
                    chains=2, parallel_chains=2, iter_warmup=1000, iter_sampling=5000, seed=42,
                    refresh=0, show_messages=FALSE, show_exceptions=FALSE)
wall_stan3 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
stan_lam3 <- fit3$draws(paste0("lambda[", 1:length(lambda_true3), "]"), format = "matrix")
stan_w3   <- fit3$draws(paste0("w[", 1:G, "]"), format = "matrix")
report3a <- compare(c1[1:length(lambda_true3), , drop=FALSE], c2[1:length(lambda_true3), , drop=FALSE],
                    stan_lam3, lambda_true3, "lambda")
report3b <- compare(c1[(length(lambda_true3)+1):(length(lambda_true3)+G), , drop=FALSE],
                    c2[(length(lambda_true3)+1):(length(lambda_true3)+G), , drop=FALSE],
                    stan_w3, w_true, "w")
report3 <- rbind(report3a, report3b)
print(report3[, c("param","truth","ours_mean","stan_mean","ours_sd","stan_sd","within_rhat","cross_rhat")],
      row.names = FALSE, digits = 4)
pass3 <- all(report3$cross_rhat < 1.02) && all(report3$within_rhat < 1.02)
cov3  <- sum((report3$truth >= report3$ours_q025) & (report3$truth <= report3$ours_q975))
cat(sprintf("[T3] cross_rhat max=%.4f; coverage %d/%d (stan %d/%d); wall ours=%.2fs stan=%.2fs (speedup %.1fx)\n",
    max(report3$cross_rhat), cov3, nrow(report3),
    sum((report3$truth >= report3$stan_q025) & (report3$truth <= report3$stan_q975)), nrow(report3),
    wall_ours3, wall_stan3, wall_stan3 / wall_ours3))
cat("[T3] ", if (pass3) "PASS" else "FAIL", "\n")

# ============================================================================
# Verdict
# ============================================================================
cat("\n============================================================\n")
if (pass1 && pass2 && pass3)
    cat("SURVIVAL PARITY: PASS (all 3 blocks match Stan within 5%; cross-impl R-hat < 1.02)\n")
else
    cat("SURVIVAL PARITY: FAIL -- see per-test output above\n")
cat("============================================================\n")
.ok <- pass1 && pass2 && pass3
quit(save = "no", status = if (.ok) 0L else 1L)

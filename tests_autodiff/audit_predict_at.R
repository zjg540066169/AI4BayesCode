# Phase F5 of audit — predict_at state preservation + DAG consistency.
# For each example:
#   - Run 200 MCMC steps
#   - Snapshot get_current()
#   - Call predict_at(list()) (or with appropriate X_new for predict-enabled models)
#   - Snapshot get_current() again
#   - Assert: snapshots are identical (predict_at is pure)
#   - Record predict DAG edges + keys the output returned.
#
# Logs to audit_predict_at.log; rds to audit_xl_F5.rds.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

check_predict <- function(name, ctor_expr, predict_input = list()) {
    cat(sprintf("\n[F5:%s]\n", name))
    r <- list()
    m <- eval(ctor_expr)
    m$step(200L)
    before <- m$get_current()
    dag <- tryCatch(m$get_dag(), error = function(e) NULL)
    if (!is.null(dag)) {
        r$dag_keys <- names(dag)
        r$predict_edges <- dag$predict_edges
        r$predict_nodes <- dag$predict_nodes
        r$data_inputs   <- dag$data_inputs
    }
    pred_result <- tryCatch(m$predict_at(predict_input),
                            error = function(e) structure(list(err=conditionMessage(e)),
                                                           class="err"))
    has_predict  <- !(inherits(pred_result, "err") &&
                      grepl("could not find|not exist|no applicable method|argument",
                            pred_result$err, ignore.case=TRUE))
    predict_ok   <- !inherits(pred_result, "err")
    r$has_predict <- has_predict
    r$predict_ok  <- predict_ok
    r$predict_err <- if (inherits(pred_result, "err")) pred_result$err else NULL
    if (predict_ok) {
        r$predict_out_keys <- names(pred_result)
        r$predict_out_sizes <- sapply(pred_result, function(x)
            if (is.null(dim(x))) length(x) else paste(dim(x), collapse="x"))
    }
    # Check state preservation after predict
    after <- m$get_current()
    r$state_preserved <- identical(before, after)
    cat(sprintf("  has_predict=%s predict_ok=%s state_preserved=%s\n",
        has_predict, predict_ok, r$state_preserved))
    if (!is.null(r$predict_out_keys))
        cat(sprintf("    predict output: %s  (sizes: %s)\n",
            paste(r$predict_out_keys, collapse=","),
            paste(r$predict_out_sizes, collapse=",")))
    if (!is.null(r$predict_edges) && length(r$predict_edges) > 0) {
        edges_str <- paste(names(r$predict_edges),
                           sapply(r$predict_edges, paste, collapse="+"),
                           sep="->", collapse="  ")
        cat(sprintf("    predict edges: %s\n", edges_str))
    }
    if (!is.null(r$predict_err))
        cat(sprintf("    predict_at err: %s\n", r$predict_err))
    r
}

results <- list()

# 1. Gaussian
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GaussianLocationScale.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); y <- rnorm(100, 2, 1.5)
results$Gaussian <- check_predict("GaussianLocationScale",
    quote(new(GaussianLocationScale, y, 1L, TRUE)))

# 2. BetaBernoulli
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "BetaBernoulli.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); y <- as.numeric(rbinom(200, 1, 0.3))
results$BetaBernoulli <- check_predict("BetaBernoulli",
    quote(new(BetaBernoulli, y, 1.0, 1.0, 1L, TRUE)))

# 3. DirichletSimplex
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletSimplex.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
y_counts <- as.numeric(c(15,25,30,20,10)); alpha_pr <- rep(1,5)
results$DirichletSimplex <- check_predict("DirichletSimplex",
    quote(new(DirichletSimplex, y_counts, alpha_pr, 1L, TRUE)))

# 4. DirichletSparse
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletSparse.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
y_sp <- as.numeric(c(100, 80, 50, 30, 20, rep(2, 15)))
results$DirichletSparse <- check_predict("DirichletSparse",
    quote(new(DirichletSparse, y_sp, 1L, TRUE)))

# 5. DirichletHierarchical
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletHierarchical.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); K <- 4L; P <- 20L
S_obs <- matrix(NA_real_, K, P)
for (i in 1:P) { g <- rgamma(K, shape = 20 * c(0.1, 0.3, 0.4, 0.2)); S_obs[,i] <- g / sum(g) }
results$DirichletHierarchical <- check_predict("DirichletHierarchical",
    quote(new(DirichletHierarchical, S_obs, 1.0, 1.0, 1L, TRUE)))

# 6. BartNoise — predict_at with X_new
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "BartNoise.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); X_b <- matrix(rnorm(80*3), 80, 3); y_b <- as.numeric(3*X_b[,1] + rnorm(80, 0, 0.8))
X_test <- matrix(rnorm(20*3), 20, 3)
results$BartNoise <- check_predict("BartNoise",
    quote(new(BartNoise, X_b, y_b, 50L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE, 1L, TRUE)),
    predict_input = list(X = X_test))

# 7. GBartPoisson
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GBartPoisson.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); X_l <- matrix(runif(80*3), 80, 3); y_l <- as.numeric(rpois(80, exp(1 + 2*X_l[,1])))
X_lt <- matrix(runif(20*3), 20, 3)
results$GBartPoisson <- check_predict("GBartPoisson",
    # keep_history=FALSE — GBart variants don't support new-X predict_at
    # in history mode (documented; deferred per-draw tree forest reproject).
    quote(new(GBartPoisson, X_l, y_l, 50L, 1L, FALSE)),
    predict_input = list(X = X_lt))

# 8. ARDLasso
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "ARDLasso.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); X_a <- matrix(rnorm(100*10), 100, 10)
y_a <- as.numeric(2 + X_a %*% c(3, -2, 0, 0, 1.5, rep(0,5)) + rnorm(100))
X_at <- matrix(rnorm(25*10), 25, 10)
results$ARDLasso <- check_predict("ARDLasso",
    quote(new(ARDLasso, X_a, y_a, 1L, TRUE)),
    predict_input = list(X = X_at))

# 9. IRT1PL_joint — empty predict
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "IRT1PL_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_i <- 30L; J_i <- 8L
theta_t <- rnorm(N_i); b_t <- rnorm(J_i, 0, 0.8)
P_i <- outer(theta_t, b_t, function(a, bb) 1/(1+exp(-(a-bb))))
Y_i <- matrix(rbinom(N_i*J_i, 1, P_i), N_i, J_i)
results$IRT1PL_joint <- check_predict("IRT1PL_joint",
    quote(new(IRT1PL_joint, Y_i, rep(0, N_i), rep(0, J_i), 1.0, 1L, TRUE)))

# 10. HierarchicalLM_joint
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "HierarchicalLM_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); G_hl <- 10L; Np_hl <- 15L; N_hl <- G_hl*Np_hl
X_hl <- matrix(rnorm(N_hl*3), N_hl, 3)
g_idx_hl <- rep(seq_len(G_hl), each=Np_hl)
y_hl <- as.numeric(1 + X_hl %*% c(2, -1, 0.5) + rep(rnorm(G_hl, 0, 1.5), each=Np_hl) + rnorm(N_hl, 0, 0.8))
results$HierarchicalLM_joint <- check_predict("HierarchicalLM_joint",
    quote(new(HierarchicalLM_joint, y_hl, X_hl, as.integer(g_idx_hl), G_hl, 1.0, 1.0, 1L, TRUE)))

# 11. LinearRegJointMixed
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "LinearRegJointMixed.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); X_lr <- matrix(rnorm(200*5), 200, 5)
y_lr <- as.numeric(1.5 + X_lr %*% c(2, -1, 0.5, 0, 3) + rnorm(200, 0, 1.2))
X_lrt <- matrix(rnorm(25*5), 25, 5)
results$LinearRegJointMixed <- check_predict("LinearRegJointMixed",
    quote(new(LinearRegJointMixed, y_lr, X_lr, 1L, TRUE)),
    predict_input = list(X = X_lrt))

# 12. SpikeSlabRJMCMC — Dirac spike-and-slab, all hyperparameters sampled
#     Empty predict (no X replacement yet; DAG marks X as a data input).
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "SpikeSlabRJMCMC.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_ss <- 80L; p_ss <- 10L
X_ss <- matrix(rnorm(N_ss*p_ss), N_ss, p_ss)
X_ss <- scale(X_ss, center=TRUE, scale=FALSE)
beta_ss <- c(3, -2, rep(0, 8))
y_ss <- as.numeric(X_ss %*% beta_ss + rnorm(N_ss))
y_ss <- y_ss - mean(y_ss)
results$SpikeSlabRJMCMC <- check_predict("SpikeSlabRJMCMC",
    quote(new(SpikeSlabRJMCMC, X_ss, y_ss,
              1.0, 1.0, 1L, TRUE)))

# 13. LogisticRegression (T12)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "LogisticRegression.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_lg <- 200L; p_lg <- 5L
X_lg <- matrix(rnorm(N_lg*p_lg), N_lg, p_lg)
beta_lg <- c(1.0, -0.5, 0.8, 0.0, -1.2)
prob_lg <- 1 / (1 + exp(-X_lg %*% beta_lg))
y_lg <- as.numeric(runif(N_lg) < prob_lg)
X_lg_test <- matrix(rnorm(25*p_lg), 25, p_lg)
results$LogisticRegression <- check_predict("LogisticRegression",
    quote(new(LogisticRegression, X_lg, y_lg, 10.0, 1L, TRUE)),
    predict_input = list(X = X_lg_test))

# 13b. ProbitRegression (probit_aug_block + nuts_block on beta)
# predict_at supports optional X (vectorised column-major); empty list
# returns y_rep at training X.
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "ProbitRegression.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_pr <- 100L; p_pr <- 3L
X_pr <- matrix(rnorm(N_pr * p_pr), N_pr, p_pr)
beta_pr <- c(0.7, -1.0, 0.3)
prob_pr <- pnorm(X_pr %*% beta_pr)
y_pr <- as.numeric(runif(N_pr) < prob_pr)
results$ProbitRegression <- check_predict("ProbitRegression",
    quote(new(ProbitRegression, X_pr, y_pr, 10.0, 1L, TRUE)))

# 14. HMMGaussian2State (T10)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "HMMGaussian2State.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); T_hmm <- 50L
A_hmm  <- c(0.8, 0.2, 0.3, 0.7)
pi_hmm <- c(0.5, 0.5)
mu_hmm <- c(0.0, 2.0)
z_true <- integer(T_hmm); z_true[1] <- if (runif(1) < pi_hmm[1]) 0 else 1
for (tt in 2:T_hmm)
    z_true[tt] <- if (runif(1) < A_hmm[z_true[tt-1]*2 + 1]) 0 else 1
y_hmm <- rnorm(T_hmm, mu_hmm[z_true + 1], 1.0)
results$HMMGaussian2State <- check_predict("HMMGaussian2State",
    quote(new(HMMGaussian2State, y_hmm, A_hmm, pi_hmm, mu_hmm, 1.0, 1L, TRUE)))

# 15. GBartLogistic (direct genBART sigmoid; replaces archived LogisticBART)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GBartLogistic.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_lb <- 100L; p_lb <- 5L
X_lb <- matrix(rnorm(N_lb * p_lb), N_lb, p_lb)
beta_lb <- c(1.5, -1.0, 0.0, 0.5, -0.8)
prob_lb <- 1 / (1 + exp(-X_lb %*% beta_lb))
y_lb <- as.numeric(runif(N_lb) < prob_lb)
X_lb_test <- matrix(rnorm(25 * p_lb), 25, p_lb)
results$GBartLogistic <- check_predict("GBartLogistic",
    # keep_history=FALSE (same documented limitation as GBartPoisson)
    quote(new(GBartLogistic, X_lb, y_lb, 50L, 1L, FALSE)),
    predict_input = list(X = X_lb_test))

# 16. GBartMultinomial (C=4; C-1 coupled + poisson_multinomial_aug; replaces archived MultinomialLogisticBART)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "GBartMultinomial.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_ml <- 80L; p_ml <- 5L; C_ml <- 4L
X_ml <- matrix(rnorm(N_ml * p_ml), N_ml, p_ml)
B_ml <- cbind(c(1.5, -1.0, 0.5, 0.0, -0.8),
              c(-0.8, 1.2, 0.0, -0.5, 0.3),
              c(0.3, 0.5, 1.0, -1.2, 0.0))
logits_ml <- X_ml %*% B_ml
probs_ml  <- cbind(0, logits_ml)
probs_ml  <- exp(probs_ml - apply(probs_ml, 1, max))
probs_ml  <- probs_ml / rowSums(probs_ml)
y_ml <- apply(probs_ml, 1, function(p) sample.int(C_ml, 1, prob = p)) - 1L
X_ml_test <- matrix(rnorm(25 * p_ml), 25, p_ml)
results$GBartMultinomial <- check_predict("GBartMultinomial",
    # keep_history=FALSE (same documented limitation as GBartPoisson)
    quote(new(GBartMultinomial, X_ml, as.numeric(y_ml), C_ml,
              50L, 1L, FALSE)),
    predict_input = list(X = X_ml_test))

# 17. GPRegression (ESS + libgp SE kernel)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GPRegression.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_gp <- 60L; p_gp <- 1L
X_gp <- matrix(seq(-3, 3, length.out=N_gp), N_gp, p_gp)
y_gp <- as.numeric(sin(X_gp[,1]) + 0.5*X_gp[,1] + rnorm(N_gp, 0, 0.3))
X_gp_test <- matrix(seq(-4, 4, length.out=20), 20, p_gp)
results$GPRegression <- check_predict("GPRegression",
    quote(new(GPRegression, X_gp, y_gp, 1L, TRUE)),
    predict_input = list(X = X_gp_test))

# 18. GPTimeSeries v0.5 (celerite + slice block for hyperparams;
#     predict_at takes list(t = t_new) rather than X).
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GPTimeSeries.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_gts <- 60L
t_gts <- sort(runif(N_gts, 0, 10))
y_gts <- as.numeric(sin(t_gts) + rnorm(N_gts, 0, 0.3))
t_gts_test <- seq(min(t_gts) - 1, max(t_gts) + 1, length.out = 20)
results$GPTimeSeries <- check_predict("GPTimeSeries",
    quote(new(GPTimeSeries, t_gts, y_gts, 1L, TRUE)),
    predict_input = list(t = t_gts_test))

# 19. GPClassification (ESS + libgp SE kernel + Bernoulli-logit)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GPClassification.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_gc <- 60L; p_gc <- 1L
X_gc <- matrix(seq(-3, 3, length.out = N_gc), N_gc, p_gc)
f_gc_true <- sin(1.5 * X_gc[,1]) - 0.2 * X_gc[,1]
prob_gc <- 1 / (1 + exp(-f_gc_true))
y_gc <- as.numeric(runif(N_gc) < prob_gc)
X_gc_test <- matrix(seq(-4, 4, length.out = 20), 20, p_gc)
results$GPClassification <- check_predict("GPClassification",
    quote(new(GPClassification, X_gc, y_gc, 1L, TRUE)),
    predict_input = list(X = X_gc_test))

# 20-22. BNP DP / PY / DerivedAlpha — predict_at(list()) returns y_rep
# at training X (Q3=a; covariate-dependent BNP = future).
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "DPGaussianMixture.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_dp <- 60L; d_dp <- 2L
mu_t_dp <- rbind(c(-2,-2), c(0,0), c(2,2))
y_dp <- do.call(rbind, lapply(seq_len(nrow(mu_t_dp)), function(k)
    matrix(rnorm(20*d_dp, mu_t_dp[k,], 0.6), 20, d_dp, byrow=TRUE)))
results$DPGaussianMixture <- check_predict("DPGaussianMixture",
    quote(new(DPGaussianMixture, y_dp, 10L, colMeans(y_dp),
              0.1, 2.0, 1.0, 1.0, 1.0, 1L, TRUE)),
    predict_input = list())

AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "PYGaussianMixture.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
results$PYGaussianMixture <- check_predict("PYGaussianMixture",
    quote(new(PYGaussianMixture, y_dp, 10L, 0.3, colMeans(y_dp),
              0.1, 2.0, 1.0, 1.0, 1.0, 1L, TRUE)),
    predict_input = list())

AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "DPGaussianMixture_DerivedAlpha.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
results$DPGaussianMixture_DerivedAlpha <- check_predict(
    "DPGaussianMixture_DerivedAlpha",
    quote(new(DPGaussianMixture_DerivedAlpha, y_dp, 10L, colMeans(y_dp),
              0.1, 2.0, 1.0, 1L, TRUE)),
    predict_input = list())

# 23. FiniteGaussianMixture (finite K, dirichlet_gibbs on π)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "FiniteGaussianMixture.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
results$FiniteGaussianMixture <- check_predict(
    "FiniteGaussianMixture",
    quote(new(FiniteGaussianMixture, y_dp, 3L, colMeans(y_dp),
              0.1, 2.0, 1.0, 1.0, 1L, TRUE)),
    predict_input = list())

# 24. HDPGaussianMixture (truncated HDP across G groups)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "HDPGaussianMixture.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
g_idx_hdp <- as.integer(rep(0:1, each = nrow(y_dp) / 2))
results$HDPGaussianMixture <- check_predict(
    "HDPGaussianMixture",
    quote(new(HDPGaussianMixture, y_dp, g_idx_hdp, 5L,
              colMeans(y_dp), 0.1,
              diag(ncol(y_dp)), 5.0, 1.0, 1.0, 1L, TRUE)),
    predict_input = list())

# ====== Summary ======
saveRDS(results, file.path(AI4BayesCode_dir, "audit_xl_F5.rds"))
cat("\n========== PHASE F5 SUMMARY ==========\n")
for (nm in names(results)) {
    r <- results[[nm]]
    cat(sprintf("  %-25s  has_predict=%s  ok=%s  state_pres=%s%s\n",
        nm, r$has_predict, r$predict_ok, r$state_preserved,
        if (!is.null(r$predict_out_keys)) sprintf(" keys=%s",
            paste(r$predict_out_keys, collapse=",")) else ""))
}
cat("\n[F5 DONE] saved to audit_xl_F5.rds\n")

# Phase F4 of audit — compile + smoke-test every production example.
# For each of 16 examples:
#   - fresh sourceCpp (clean caches) — catches any dormant compile error
#   - 10 MCMC steps from default init — catches immediate crash / NaN
#   - get_current() / get_dag() return sane shapes
#   - history is finite
# Logs to audit_compile_smoke.log; rds to audit_xl_F4.rds.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

results <- list()

smoke_test <- function(name, constructor_expr) {
    cat(sprintf("\n[F4:%s]\n", name))
    res <- list(compile=FALSE, step=FALSE, finite=FALSE, dag=FALSE,
                check24=NA, check24_why=NA_character_,
                sustained=NA, sustained_why=NA_character_)
    tc <- tryCatch({
        m <- eval(constructor_expr)
        res$compile <- TRUE
        m$step(10L)
        res$step <- TRUE
        cur <- m$get_current()
        # Only check finiteness on NUMERIC fields. Structured fields
        # (BART tree forests serialized to character, HMM emissions as
        # list, etc.) are not subject to is.finite — skipping them is
        # the correct behavior, NOT a fail.
        all_finite <- all(sapply(cur, function(x) {
            if (is.numeric(x)) all(is.finite(x)) else TRUE
        }))
        res$finite <- all_finite
        dag <- tryCatch(m$get_dag(), error = function(e) NULL)
        res$dag <- !is.null(dag) && is.list(dag) &&
                   all(c("gibbs_reads","gibbs_invalidates") %in% names(dag))
        res$cur_names <- names(cur)
        res$dag_keys <- if (res$dag) names(dag) else NULL

        # ---- Check #24: readapt_NUTS state preservation -----------------
        has_readapt <- tryCatch({
            !is.null(m$readapt_NUTS)
        }, error = function(e) FALSE)

        if (has_readapt) {
            s_before <- m$get_current()
            ok_cont <- tryCatch({
                m$readapt_NUTS(50L, FALSE)
                identical(s_before, m$get_current())
            }, error = function(e) {
                res$check24_why <<- paste("continue threw:", conditionMessage(e))
                FALSE
            })
            ok_reset <- if (ok_cont) {
                tryCatch({
                    m$readapt_NUTS(20L, TRUE)
                    identical(s_before, m$get_current())
                }, error = function(e) {
                    res$check24_why <<- paste("reset threw:", conditionMessage(e))
                    FALSE
                })
            } else FALSE
            res$check24 <- ok_cont && ok_reset
            if (is.na(res$check24_why) || is.null(res$check24_why))
                res$check24_why <- if (res$check24) "OK (state preserved both modes)"
                                   else "state changed across readapt"

            # ---- SUSTAINED-CHAIN test after readapt --------------------
            # Catch the bug where readapt corrupts the kernel such that
            # subsequent step() produces NaN / errors / chain stuck.
            # Note: skip non-numeric outputs (e.g. BART/SoftBart serialized
            # "tree" string) for the finite check — is.finite(character)
            # returns NA spuriously.
            numeric_finite <- function(lst) {
                num_parts <- lst[sapply(lst, is.numeric)]
                all(sapply(num_parts, function(x) all(is.finite(x))))
            }
            if (res$check24) {
                s_pre <- m$get_current()
                sus_ok <- tryCatch({
                    m$readapt_NUTS(100L, FALSE)  # readapt
                    for (i in 1:200) m$step(1L)   # 200 sustained steps
                    s_post <- m$get_current()
                    finite_ok <- numeric_finite(s_post)
                    moved_ok <- !identical(s_pre, s_post)
                    if (!finite_ok) {
                        res$sustained_why <<- "non-finite (numeric parts) after 200 sustained"
                        FALSE
                    } else if (!moved_ok) {
                        res$sustained_why <<- "chain stuck (didn't advance)"
                        FALSE
                    } else {
                        # 2nd round: readapt again + 50 more
                        m$readapt_NUTS(50L, FALSE)
                        for (i in 1:50) m$step(1L)
                        s_2nd <- m$get_current()
                        if (!numeric_finite(s_2nd)) {
                            res$sustained_why <<- "non-finite (numeric) in 2nd round"
                            FALSE
                        } else TRUE
                    }
                }, error = function(e) {
                    res$sustained_why <<- paste("threw:", conditionMessage(e))
                    FALSE
                })
                res$sustained <- sus_ok
                if (is.na(res$sustained_why) || is.null(res$sustained_why))
                    res$sustained_why <- if (res$sustained)
                        "OK (200 sustained + 2nd round both clean)" else "FAIL"
            } else {
                res$sustained_why <- "skipped (check24 failed)"
            }
        } else {
            res$check24_why    <- "no 7th method (non-NUTS example)"
            res$sustained_why  <- "n/a (non-NUTS example)"
        }
        NULL
    }, error = function(e) {
        res$error <<- conditionMessage(e)
        NULL
    })
    status <- with(res,
        sprintf("compile=%s step=%s finite=%s dag=%s%s",
                compile, step, finite, dag,
                if (!is.null(res$error)) sprintf(" ERR=%s", res$error) else ""))
    cat("  ", status, "\n")
    cat("   check24=", if (is.na(res$check24)) "NA" else as.character(res$check24),
        "  (", res$check24_why, ")\n", sep = "")
    cat("   sustained=", if (is.na(res$sustained)) "NA" else as.character(res$sustained),
        "  (", res$sustained_why, ")\n", sep = "")
    if (!is.null(res$cur_names))
        cat("  current keys:", paste(res$cur_names, collapse=","), "\n")
    if (!is.null(res$dag_keys))
        cat("  dag keys:",     paste(res$dag_keys, collapse=","), "\n")
    results[[name]] <<- res
    invisible(NULL)
}

# 1. GaussianLocationScale
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GaussianLocationScale.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); y <- rnorm(100, 2, 1.5)
smoke_test("GaussianLocationScale",
    quote(new(GaussianLocationScale, y, 1L, TRUE)))

# 2. BetaBernoulli
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "BetaBernoulli.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); y <- as.numeric(rbinom(200, 1, 0.3))
smoke_test("BetaBernoulli",
    quote(new(BetaBernoulli, y, 1.0, 1.0, 1L, TRUE)))

# 3. DirichletSimplex
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletSimplex.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
y_counts <- as.numeric(c(15,25,30,20,10)); alpha_pr <- rep(1,5)
smoke_test("DirichletSimplex",
    quote(new(DirichletSimplex, y_counts, alpha_pr, 1L, TRUE)))

# 4. DirichletSparse
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletSparse.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); y_sp <- as.numeric(c(100, 80, 50, 30, 20, rep(2, 15)))
smoke_test("DirichletSparse",
    quote(new(DirichletSparse, y_sp, 1L, TRUE)))

# 5. DirichletHierarchical
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletHierarchical.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); K <- 4L; P <- 20L
S_obs <- matrix(NA_real_, K, P)
for (i in 1:P) {
    g <- rgamma(K, shape = 20 * c(0.1, 0.3, 0.4, 0.2)); S_obs[,i] <- g / sum(g)
}
smoke_test("DirichletHierarchical",
    quote(new(DirichletHierarchical, S_obs, 1.0, 1.0, 1L, TRUE)))

# 6. BartNoise
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "BartNoise.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); X_b <- matrix(rnorm(80*3), 80, 3); y_b <- as.numeric(3*X_b[,1] + rnorm(80, 0, 0.8))
smoke_test("BartNoise",
    quote(new(BartNoise, X_b, y_b, 50L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE, 1L, TRUE)))

# 7. GBartPoisson (Linero 2022 genBART)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GBartPoisson.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); X_l <- matrix(runif(80*3), 80, 3); y_l <- as.numeric(rpois(80, exp(1 + 2*X_l[,1])))
smoke_test("GBartPoisson",
    quote(new(GBartPoisson, X_l, y_l, 50L, 1L, TRUE)))

# 8. ARDLasso
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "ARDLasso.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); X_a <- matrix(rnorm(100*10), 100, 10)
y_a <- as.numeric(2 + X_a %*% c(3, -2, 0, 0, 1.5, rep(0,5)) + rnorm(100))
smoke_test("ARDLasso",
    quote(new(ARDLasso, X_a, y_a, 1L, TRUE)))

# 9. IRT1PL_joint
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "IRT1PL_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_i <- 30L; J_i <- 8L
theta_t <- rnorm(N_i); b_t <- rnorm(J_i, 0, 0.8)
P_i <- outer(theta_t, b_t, function(a, bb) 1/(1+exp(-(a-bb))))
Y_i <- matrix(rbinom(N_i*J_i, 1, P_i), N_i, J_i)
smoke_test("IRT1PL_joint",
    quote(new(IRT1PL_joint, Y_i, rep(0, N_i), rep(0, J_i), 1.0, 1L, TRUE)))

# 10. HierarchicalLM_joint
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "HierarchicalLM_joint.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); G_hl <- 10L; Np_hl <- 15L; N_hl <- G_hl*Np_hl
X_hl <- matrix(rnorm(N_hl*3), N_hl, 3)
g_idx_hl <- rep(seq_len(G_hl), each=Np_hl)
y_hl <- as.numeric(1 + X_hl %*% c(2, -1, 0.5) + rep(rnorm(G_hl, 0, 1.5), each=Np_hl) + rnorm(N_hl, 0, 0.8))
smoke_test("HierarchicalLM_joint",
    quote(new(HierarchicalLM_joint, y_hl, X_hl, as.integer(g_idx_hl), G_hl, 1.0, 1.0, 1L, TRUE)))

# 11. LinearRegJointMixed
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "LinearRegJointMixed.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); X_lr <- matrix(rnorm(200*5), 200, 5)
y_lr <- as.numeric(1.5 + X_lr %*% c(2, -1, 0.5, 0, 3) + rnorm(200, 0, 1.2))
smoke_test("LinearRegJointMixed",
    quote(new(LinearRegJointMixed, y_lr, X_lr, 1L, TRUE)))

# 12. SpikeSlabRJMCMC (Dirac spike-and-slab, Ishwaran-Rao 2005 sigma-scaled
#     slab form: pi via beta_gibbs (Exception 3 conjugate), sigma and tau
#     via nuts_block with Jeffreys priors, gamma/beta via rjmcmc_block with
#     hand-written Gibbs continuous_update)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "SpikeSlabRJMCMC.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_ss <- 80L; p_ss <- 10L
X_ss <- matrix(rnorm(N_ss*p_ss), N_ss, p_ss)
X_ss <- scale(X_ss, center=TRUE, scale=FALSE)
beta_ss <- c(3, -2, rep(0, 8))
y_ss <- as.numeric(X_ss %*% beta_ss + rnorm(N_ss))
y_ss <- y_ss - mean(y_ss)
smoke_test("SpikeSlabRJMCMC",
    quote(new(SpikeSlabRJMCMC, X_ss, y_ss,
              1.0, 1.0, 1L, TRUE)))

# 13. LogisticRegression (T12 pg_logistic_block via Polya-Gamma)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "LogisticRegression.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_lg <- 200L; p_lg <- 5L
X_lg <- matrix(rnorm(N_lg*p_lg), N_lg, p_lg)
beta_lg <- c(1.0, -0.5, 0.8, 0.0, -1.2)
prob_lg <- 1 / (1 + exp(-X_lg %*% beta_lg))
y_lg <- as.numeric(runif(N_lg) < prob_lg)
smoke_test("LogisticRegression",
    quote(new(LogisticRegression, X_lg, y_lg, 10.0, 1L, TRUE)))

# 13b. ProbitRegression (probit_aug_block + nuts_block on beta;
#      Albert-Chib 1993 data augmentation)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "ProbitRegression.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_pr <- 100L; p_pr <- 3L
X_pr <- matrix(rnorm(N_pr * p_pr), N_pr, p_pr)
beta_pr <- c(0.7, -1.0, 0.3)
prob_pr <- pnorm(X_pr %*% beta_pr)
y_pr <- as.numeric(runif(N_pr) < prob_pr)
smoke_test("ProbitRegression",
    quote(new(ProbitRegression, X_pr, y_pr, 10.0, 1L, TRUE)))

# 14. HMMGaussian2State (T10 hmm_block via forward-backward)
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
smoke_test("HMMGaussian2State",
    quote(new(HMMGaussian2State, y_hmm, A_hmm, pi_hmm, mu_hmm, 1.0, 1L, TRUE)))

# 15. GBartLogistic (binary classification via genBART direct sigmoid;
#     replaces archived LogisticBART + multinomial_gamma_aug path)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GBartLogistic.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_lb <- 100L; p_lb <- 5L
X_lb <- matrix(rnorm(N_lb * p_lb), N_lb, p_lb)
beta_lb <- c(1.5, -1.0, 0.0, 0.5, -0.8)
prob_lb <- 1 / (1 + exp(-X_lb %*% beta_lb))
y_lb <- as.numeric(runif(N_lb) < prob_lb)
smoke_test("GBartLogistic",
    quote(new(GBartLogistic, X_lb, y_lb, 50L, 1L, TRUE)))

# 16. GBartMultinomial (C=4 multi-class; C-1 coupled genbart_blocks +
#     poisson_multinomial_aug_block. Replaces archived
#     MultinomialLogisticBART.)
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
smoke_test("GBartMultinomial",
    quote(new(GBartMultinomial, X_ml, as.numeric(y_ml), C_ml,
              50L, 1L, TRUE)))

# 17. GPRegression (Gaussian Process regression via ESS + libgp SE kernel)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GPRegression.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_gp <- 60L; p_gp <- 1L
X_gp <- matrix(seq(-3, 3, length.out=N_gp), N_gp, p_gp)
y_gp <- as.numeric(sin(X_gp[,1]) + 0.5*X_gp[,1] + rnorm(N_gp, 0, 0.3))
smoke_test("GPRegression",
    quote(new(GPRegression, X_gp, y_gp, 1L, TRUE)))

# 18. GPTimeSeries (v0.5: celerite-based 1-D GP for time-series with
#     univariate_slice_sampling_block for hyperparameter inference
#     on celerite's marginal log-likelihood. Replaces the v0 PoC.)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GPTimeSeries.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_gts <- 80L
t_gts <- sort(runif(N_gts, 0, 10))
y_gts <- as.numeric(sin(t_gts) + rnorm(N_gts, 0, 0.3))
smoke_test("GPTimeSeries",
    quote(new(GPTimeSeries, t_gts, y_gts, 1L, TRUE)))

# 19. GPClassification (binary GP classification via ESS on latent f +
#     NUTS on kernel hyperparameters + libgp SE kernel + Bernoulli-logit
#     likelihood. No sigma block -- Bernoulli has no noise parameter.)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GPClassification.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_gc <- 60L; p_gc <- 1L
X_gc <- matrix(seq(-3, 3, length.out = N_gc), N_gc, p_gc)
f_gc_true <- sin(1.5 * X_gc[,1]) - 0.2 * X_gc[,1]
prob_gc <- 1 / (1 + exp(-f_gc_true))
y_gc <- as.numeric(runif(N_gc) < prob_gc)
smoke_test("GPClassification",
    quote(new(GPClassification, X_gc, y_gc, 1L, TRUE)))

# 20. DPGaussianMixture (DP truncated SBP + Normal-Gamma cluster prior +
#     diagonal Gaussian emissions; alpha sampled by NUTS on log scale)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "DPGaussianMixture.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
set.seed(1); N_dp <- 60L; d_dp <- 2L
mu_t_dp <- rbind(c(-2,-2), c(0,0), c(2,2))
y_dp <- do.call(rbind, lapply(seq_len(nrow(mu_t_dp)), function(k)
    matrix(rnorm(20*d_dp, mu_t_dp[k,], 0.6), 20, d_dp, byrow=TRUE)))
smoke_test("DPGaussianMixture",
    quote(new(DPGaussianMixture, y_dp, 10L, colMeans(y_dp),
              0.1, 2.0, 1.0, 1.0, 1.0, 1L, TRUE)))

# 21. PYGaussianMixture (Pitman-Yor variant; discount fixed)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "PYGaussianMixture.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
smoke_test("PYGaussianMixture",
    quote(new(PYGaussianMixture, y_dp, 10L, 0.3, colMeans(y_dp),
              0.1, 2.0, 1.0, 1.0, 1.0, 1L, TRUE)))

# 22. DPGaussianMixture_DerivedAlpha (alpha = exp(phi) via refresher)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "DPGaussianMixture_DerivedAlpha.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
smoke_test("DPGaussianMixture_DerivedAlpha",
    quote(new(DPGaussianMixture_DerivedAlpha, y_dp, 10L, colMeans(y_dp),
              0.1, 2.0, 1.0, 1L, TRUE)))

# 23. FiniteGaussianMixture (finite K, dirichlet_gibbs_block on π)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "FiniteGaussianMixture.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
smoke_test("FiniteGaussianMixture",
    quote(new(FiniteGaussianMixture, y_dp, 3L, colMeans(y_dp),
              0.1, 2.0, 1.0, 1.0, 1L, TRUE)))

# 24. HDPGaussianMixture (truncated HDP across G groups)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "HDPGaussianMixture.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
g_idx_hdp <- as.integer(rep(0:1, each = nrow(y_dp) / 2))
smoke_test("HDPGaussianMixture",
    quote(new(HDPGaussianMixture, y_dp, g_idx_hdp, 5L,
              colMeans(y_dp), 0.1,
              diag(ncol(y_dp)), 5.0, 1.0, 1.0, 1L, TRUE)))

# 25. LdaCollapsedGibbs (Griffiths-Steyvers 2004 collapsed Gibbs LDA)
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "LdaCollapsedGibbs.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)
{
    .lda_M <- 4L; .lda_V <- 6L; .lda_K <- 2L
    .lda_w   <- as.integer(c(1, 2, 1, 3, 5, 4, 6, 5, 2, 1, 6, 5))
    .lda_doc <- as.integer(c(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4))
    smoke_test("LdaCollapsedGibbs",
        quote(new(LdaCollapsedGibbs, .lda_w, .lda_doc, .lda_M, .lda_V, .lda_K,
                  rep(1.0, .lda_K), rep(1.0, .lda_V), 1L, TRUE)))
}

# ======== Summary ========
saveRDS(results, file.path(AI4BayesCode_dir, "audit_xl_F4.rds"))
cat("\n========== PHASE F4 SUMMARY ==========\n")
pass <- sum(sapply(results, function(r) r$compile && r$step && r$finite && r$dag))
cat(sprintf("Pass: %d/%d (compile + 10-step + finite + dag)\n",
    pass, length(results)))
for (nm in names(results)) {
    r <- results[[nm]]
    ok <- r$compile && r$step && r$finite && r$dag
    cat(sprintf("  %s  %s\n", ifelse(ok, "PASS", "FAIL"), nm))
    if (!ok && !is.null(r$error)) cat("     err:", r$error, "\n")
}
cat("\n[F4 DONE] saved to audit_xl_F4.rds\n")

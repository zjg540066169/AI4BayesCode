# Runs every library parity / per-usage test in tests_autodiff/ and
# reports pass/fail per test. These are the Check #15 library-level
# parity tests and various per-usage tests that aren't in the driver.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

run_parity <- function(cpp, fn_name, data_args = list(),
                       pass_condition = function(r) r$all_pass) {
    cat(sprintf("\n----- %s -----\n", basename(cpp)))
    # Several entries below name a parity test that has not been authored yet
    # (the BLOCK exists; its Rcpp parity harness does not). Crashing here used
    # to abort the whole runner at entry 12 and hide every entry after it.
    # Report the gap, count it on its own line, and carry on -- a missing test
    # must stay visible, not be quietly dropped from the denominator.
    if (!file.exists(cpp)) {
        cat("  NOT AUTHORED -- no parity test at this path\n")
        return(list(name = basename(cpp), pass = NA, wall = 0,
                    details = list(all_pass = NA, missing = TRUE)))
    }
    t0 <- Sys.time()
    ai4bayescode_source_checkout(cpp, AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)
    res <- tryCatch({
        fn <- get(fn_name, envir = globalenv())
        do.call(fn, data_args)
    }, error = function(e) {
        cat("  ERROR: ", conditionMessage(e), "\n")
        list(all_pass = FALSE, error = conditionMessage(e))
    })
    t1 <- Sys.time()
    passed <- tryCatch(pass_condition(res), error = function(e) FALSE)
    cat(sprintf("  wall=%.1fs  %s\n",
                as.numeric(difftime(t1, t0, units="secs")),
                if (passed) "PASS" else "FAIL"))
    list(name = basename(cpp), pass = passed,
         wall = as.numeric(difftime(t1, t0, units="secs")),
         details = res)
}

# Some parity files do not follow the "one zero-argument entry point named
# after the file" contract: test_wrap_autodiff_vs_hand.cpp and
# test_wrap_simplex.cpp each export SEVERAL check_* functions that take
# arguments, so there is no test_wrap_autodiff_vs_hand() to call. They are
# driven by a sweep instead. (run_wrap_check12.R runs the same two files the
# same way; keep the two in step.)
run_parity_sweep <- function(cpp, sweep) {
    cat(sprintf("\n----- %s -----\n", basename(cpp)))
    t0 <- Sys.time()
    ai4bayescode_source_checkout(cpp, AI4BayesCode_path = AI4BayesCode_dir,
                                 verbose = FALSE)
    res <- tryCatch(sweep(), error = function(e) {
        cat("  ERROR: ", conditionMessage(e), "\n")
        list(all_pass = FALSE, error = conditionMessage(e))
    })
    t1 <- Sys.time()
    passed <- isTRUE(res$all_pass)
    cat(sprintf("  wall=%.1fs  %s\n",
                as.numeric(difftime(t1, t0, units = "secs")),
                if (passed) "PASS" else "FAIL"))
    list(name = basename(cpp), pass = passed,
         wall = as.numeric(difftime(t1, t0, units = "secs")), details = res)
}

# Gradient parity tolerance, matching run_wrap_check12.R.
WRAP_TOL <- 1e-8

# max|hand - ad| over the log density and the gradient of one check_* result.
wrap_worst <- function(r) {
    d_lp <- max(abs(r$lp_hand - r$lp_ad))
    d_gr <- max(abs(unlist(r$grad_hand) - unlist(r$grad_ad)))
    c(d_lp = d_lp, d_gr = d_gr)
}

results <- list()

# 1. Beta gibbs block library parity
results$beta_gibbs <- run_parity(
    file.path(AI4BayesCode_dir, "tests_autodiff/block_tests/test_beta_gibbs_block.cpp"),
    "test_beta_gibbs_block", list())

# 1b. cluster_atom_block library parity (Check #15). Non-conjugate slice
# sampler fed a CONJUGATE target so its output has closed-form moments to
# match; the two EMPTY components additionally pin Ishwaran & James (2001)
# step (a) -- an unoccupied component's conditional IS the prior.
results$cluster_atom <- run_parity(
    file.path(AI4BayesCode_dir, "tests_autodiff/block_tests/test_cluster_atom_block.cpp"),
    "test_cluster_atom_block", list())

# 2. Poisson-multinomial aug block library parity
# (parity test to be authored in a follow-up — for now the block is
#  exercised end-to-end via GBartMultinomial compile-smoke + 2-chain
#  longchain diagnostics under tests_autodiff/gbart_*.R)

# 3. HMM block
results$hmm <- run_parity(
    file.path(AI4BayesCode_dir, "tests_autodiff/test_hmm_block.cpp"),
    "test_hmm_block", list())

# 4. joint_nuts dense metric
results$dense_metric <- run_parity(
    file.path(AI4BayesCode_dir, "tests_autodiff/test_joint_nuts_dense_metric.cpp"),
    "test_joint_nuts_dense_metric", list())

# 5. nuts_adaptation
results$nuts_adaptation <- run_parity(
    file.path(AI4BayesCode_dir, "tests_autodiff/test_nuts_adaptation.cpp"),
    "test_nuts_adaptation", list())

# 6. pg_logistic_block
results$pg_logistic <- run_parity(
    file.path(AI4BayesCode_dir, "tests_autodiff/test_pg_logistic_block.cpp"),
    "test_pg_logistic_block", list())

# 7. rjmcmc_block_transform
results$rj_transform <- run_parity(
    file.path(AI4BayesCode_dir, "tests_autodiff/test_rjmcmc_block_transform.cpp"),
    "test_rjmcmc_block_transform", list())

# 8. rjmcmc_transforms
results$rj_transforms <- run_parity(
    file.path(AI4BayesCode_dir, "tests_autodiff/test_rjmcmc_transforms.cpp"),
    "test_rjmcmc_transforms", list())

# 9. spikeslab_beta_conditional_parity
results$spikeslab <- run_parity(
    file.path(AI4BayesCode_dir, "tests_autodiff/test_spikeslab_beta_conditional_parity.cpp"),
    "test_spikeslab_beta_conditional_parity", list())

# 10. wrap_autodiff_vs_hand -- exports check_real_grad / check_positive_grad /
#     check_mixed_grad, each taking arguments.
results$wrap_autodiff <- run_parity_sweep(
    file.path(AI4BayesCode_dir, "tests_autodiff/test_wrap_autodiff_vs_hand.cpp"),
    function() {
        ok <- TRUE
        report <- function(label, r) {
            w <- wrap_worst(r)
            p <- all(!is.na(w)) && all(w < WRAP_TOL)
            cat(sprintf("  %-22s d_lp=%.2e d_grad=%.2e  %s\n",
                        label, w[["d_lp"]], w[["d_gr"]],
                        if (p) "PASS" else "FAIL"))
            if (!p) ok <<- FALSE
        }
        for (mu in c(-1.5, 0.5, 2.0))
            report(sprintf("real(mu=%.2f)", mu), check_real_grad(mu))
        for (sig in c(0.3, 1.0, 3.0))
            report(sprintf("positive(sig=%.2f)", sig), check_positive_grad(sig))
        set.seed(1); y <- rnorm(20)
        for (cfg in list(c(0.0, 1.0), c(1.5, 0.5), c(-0.5, 2.0)))
            report(sprintf("mixed(mu=%.2f,sig=%.2f)", cfg[1], cfg[2]),
                   check_mixed_grad(y, cfg[1], cfg[2]))
        list(all_pass = ok)
    })

# 11. wrap_simplex -- exports check_simplex_grad(y_counts, alpha, n, seed).
results$wrap_simplex <- run_parity_sweep(
    file.path(AI4BayesCode_dir, "tests_autodiff/test_wrap_simplex.cpp"),
    function() {
        ok <- TRUE
        y_counts <- c(10, 25, 30, 20, 15)
        for (alpha_val in c(0.5, 1.0, 2.0)) {
            r <- check_simplex_grad(y_counts, rep(alpha_val, length(y_counts)),
                                    10L, 12345L)
            p <- !is.na(r$max_diff) && r$max_diff < WRAP_TOL
            cat(sprintf("  simplex(alpha=%.2f): max_diff=%.2e  %s\n",
                        alpha_val, r$max_diff, if (p) "PASS" else "FAIL"))
            if (!p) ok <- FALSE
        }
        list(all_pass = ok)
    })

# 12. bnp_utils (counts_from_z, crp/py log-prior, crp/py samplers)
results$bnp_utils <- run_parity(
    file.path(AI4BayesCode_dir,
              "tests_autodiff/block_tests/test_bnp_utils.cpp"),
    "test_bnp_utils", list())

# 13. stick_breaking_block (DP truncated SBP)
results$stick_breaking <- run_parity(
    file.path(AI4BayesCode_dir,
              "tests_autodiff/block_tests/test_stick_breaking_block.cpp"),
    "test_stick_breaking_block", list())

# 14. normal_gamma_cluster_gibbs_block (NIW-diagonal cluster sampler)
results$normal_gamma_cluster <- run_parity(
    file.path(AI4BayesCode_dir,
              "tests_autodiff/block_tests/test_normal_gamma_cluster_gibbs_block.cpp"),
    "test_normal_gamma_cluster_gibbs_block", list())

# 15. gamma_gibbs_block (scalar Gamma conjugate; dual of inv_gamma_gibbs_block)
results$gamma_gibbs <- run_parity(
    file.path(AI4BayesCode_dir,
              "tests_autodiff/block_tests/test_gamma_gibbs_block.cpp"),
    "test_gamma_gibbs_block", list())

# 16. niw_cluster_gibbs_block (full-covariance NIW cluster sampler;
#     Bartlett decomposition for IW)
results$niw_cluster <- run_parity(
    file.path(AI4BayesCode_dir,
              "tests_autodiff/block_tests/test_niw_cluster_gibbs_block.cpp"),
    "test_niw_cluster_gibbs_block", list())

# 17. split_merge_block (Jain-Neal 2004 partition MH proposal)
results$split_merge <- run_parity(
    file.path(AI4BayesCode_dir,
              "tests_autodiff/block_tests/test_split_merge_block.cpp"),
    "test_split_merge_block", list())

# 18. categorical_gibbs_block (used by HMM, DPMM, FiniteMixture, HDP)
results$categorical_gibbs <- run_parity(
    file.path(AI4BayesCode_dir,
              "tests_autodiff/block_tests/test_categorical_gibbs_block.cpp"),
    "test_categorical_gibbs_block", list())

# 19. dirichlet_gibbs_block (used by FiniteMixture, HDP)
results$dirichlet_gibbs <- run_parity(
    file.path(AI4BayesCode_dir,
              "tests_autodiff/block_tests/test_dirichlet_gibbs_block.cpp"),
    "test_dirichlet_gibbs_block", list())

# 20. binary_gibbs_block (closed-form Bernoulli)
results$binary_gibbs <- run_parity(
    file.path(AI4BayesCode_dir,
              "tests_autodiff/block_tests/test_binary_gibbs_block.cpp"),
    "test_binary_gibbs_block", list())

# 21. lda_collapsed_gibbs_block (Griffiths-Steyvers 2004 collapsed LDA)
results$lda_collapsed_gibbs <- run_parity(
    file.path(AI4BayesCode_dir,
              "tests_autodiff/block_tests/test_lda_collapsed_gibbs_block.cpp"),
    "test_lda_collapsed_gibbs_block", list())

# 22. probit_aug_block (Albert-Chib 1993 truncated-normal data augmentation)
results$probit_aug <- run_parity(
    file.path(AI4BayesCode_dir,
              "tests_autodiff/block_tests/test_probit_aug_block.cpp"),
    "test_probit_aug_block", list())

cat("\n\n========== PARITY TEST SUMMARY ==========\n")
pass_cnt <- 0; fail_cnt <- 0; missing_cnt <- 0; missing_names <- character(0)
for (nm in names(results)) {
    r <- results[[nm]]
    if (is.na(r$pass)) {
        status <- "NOT AUTHORED"; missing_cnt <- missing_cnt + 1
        missing_names <- c(missing_names, r$name)
    } else if (isTRUE(r$pass)) {
        status <- "PASS"; pass_cnt <- pass_cnt + 1
    } else {
        status <- "FAIL"; fail_cnt <- fail_cnt + 1
    }
    cat(sprintf("  %-50s  %-12s  (%.1fs)\n", r$name, status, r$wall))
}
cat(sprintf("\n%d PASS, %d FAIL, %d NOT AUTHORED out of %d entries\n",
            pass_cnt, fail_cnt, missing_cnt, length(results)))
if (missing_cnt > 0) {
    cat("\nThese blocks ship WITHOUT an Rcpp parity test:\n")
    for (nm in missing_names) cat(sprintf("  - %s\n", nm))
}
if (fail_cnt > 0) quit(status = 1)

saveRDS(results, file.path(AI4BayesCode_dir, "parity_test_results.rds"))

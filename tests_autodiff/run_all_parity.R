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
    t0 <- Sys.time()
    AI4BayesCode_sourceCpp(cpp, AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)
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

results <- list()

# 1. Beta gibbs block library parity
results$beta_gibbs <- run_parity(
    file.path(AI4BayesCode_dir, "tests_autodiff/block_tests/test_beta_gibbs_block.cpp"),
    "test_beta_gibbs_block", list())

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

# 10. wrap_autodiff_vs_hand
results$wrap_autodiff <- run_parity(
    file.path(AI4BayesCode_dir, "tests_autodiff/test_wrap_autodiff_vs_hand.cpp"),
    "test_wrap_autodiff_vs_hand", list())

# 11. wrap_simplex
results$wrap_simplex <- run_parity(
    file.path(AI4BayesCode_dir, "tests_autodiff/test_wrap_simplex.cpp"),
    "test_wrap_simplex", list())

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
pass_cnt <- 0; fail_cnt <- 0
for (nm in names(results)) {
    r <- results[[nm]]
    status <- if (isTRUE(r$pass)) "PASS" else "FAIL"
    if (isTRUE(r$pass)) pass_cnt <- pass_cnt + 1 else fail_cnt <- fail_cnt + 1
    cat(sprintf("  %-50s  %s  (%.1fs)\n", r$name, status, r$wall))
}
cat(sprintf("\n%d PASS, %d FAIL out of %d total\n",
            pass_cnt, fail_cnt, length(results)))

saveRDS(results, file.path(AI4BayesCode_dir, "parity_test_results.rds"))

# Parity test runner for elliptical_slice_sampling_block (Check #15).

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

cat("Compiling elliptical_slice_sampling_block parity test...\n")
ai4bayescode_sourceCpp(file.path(script_dir, "block_tests",
                              "test_elliptical_slice_sampling_block.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

cat("\nRunning ESS parity test...\n")
r <- test_elliptical_slice_sampling_block()

cat("\n========== ESS PARITY SUMMARY ==========\n")
cat(sprintf("  all_pass             = %s\n", r$all_pass))
cat(sprintf("  n_draws/burn         = %d / %d\n", r$n_draws, r$n_burn))
cat(sprintf("  worst_mean_rel_err   = %.4f (tol %.3f)\n",
            r$worst_mean_err, r$tol_mean))
cat(sprintf("  worst_var_rel_err    = %.4f (tol %.3f)\n",
            r$worst_var_err, r$tol_var))
cat(sprintf("  analytical_var       = %.4f\n", r$analytical_var))
cat(sprintf("  sample_var_first     = %.4f\n", r$sample_var_first))
cat("\n  mean_sample vs mu_post (first 8 entries):\n")
for (i in seq_len(min(8, length(r$sample_mean))))
    cat(sprintf("    i=%2d  sample=%.3f  analytical=%.3f  diff=%+.3f\n",
                i, r$sample_mean[i], r$analytical_mean[i],
                r$sample_mean[i] - r$analytical_mean[i]))

if (!r$all_pass) quit(status = 1)
cat("\n========== PASS ==========\n")

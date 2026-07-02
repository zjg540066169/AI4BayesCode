# Parity test runner for univariate_slice_sampling_block (Check #15).
# Three fixtures: Normal(2, 1.5) via identity, Gamma(3, 2) via positive,
# Beta(2, 5) via interval.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

cat("Compiling univariate_slice_sampling_block parity test...\n")
ai4bayescode_sourceCpp(file.path(script_dir, "block_tests",
                              "test_univariate_slice_sampling_block.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

cat("\nRunning univariate slice parity test (3 fixtures)...\n")
r <- test_univariate_slice_sampling_block()

cat("\n========== SLICE PARITY SUMMARY ==========\n")
cat(sprintf("  all_pass  = %s\n", r$all_pass))
cat(sprintf("  n_burn    = %d, n_draws = %d\n", r$n_burn, r$n_draws))
cat(sprintf("  tolerance = mean<%.0f%%  var<%.0f%%\n",
            100*r$tol_mean, 100*r$tol_var))

print_fx <- function(label, pass, mn, mnexp, vr, vrexp, mre, vre) {
    cat(sprintf("  [%s]  pass=%s\n", label, pass))
    cat(sprintf("    mean: sample=%.4f  analytic=%.4f  rel_err=%.4f\n",
                mn, mnexp, mre))
    cat(sprintf("    var : sample=%.4f  analytic=%.4f  rel_err=%.4f\n",
                vr, vrexp, vre))
}
print_fx("Normal(2, 1.5)",   r$normal_pass,
         r$normal_mean, r$normal_exp_mean,
         r$normal_var,  r$normal_exp_var,
         r$normal_mean_re, r$normal_var_re)
print_fx("Gamma(3, 2)",      r$gamma_pass,
         r$gamma_mean, r$gamma_exp_mean,
         r$gamma_var,  r$gamma_exp_var,
         r$gamma_mean_re, r$gamma_var_re)
print_fx("Beta(2, 5)",       r$beta_pass,
         r$beta_mean, r$beta_exp_mean,
         r$beta_var,  r$beta_exp_var,
         r$beta_mean_re, r$beta_var_re)

if (!r$all_pass) {
    cat("\n========== FAIL ==========\n"); quit(status = 1)
}
cat("\n========== PASS ==========\n")

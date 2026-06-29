# Runs Check #12 gradient-verify tests for the constraint wrap primitives.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

TOL <- 1e-8
all_pass <- TRUE
diag_one <- function(label, r) {
    d_lp <- max(abs(r$lp_hand - r$lp_ad))
    if (!is.null(r$grad_hand) && is.numeric(r$grad_hand)) {
        d_gr <- max(abs(r$grad_hand - r$grad_ad))
    } else {
        # grad might be arma::vec/matrix
        d_gr <- max(abs(unlist(r$grad_hand) - unlist(r$grad_ad)))
    }
    pass <- !is.na(d_lp) && !is.na(d_gr) && d_lp < TOL && d_gr < TOL
    cat(sprintf("  %s: |Δlp|=%.2e |Δgrad|=%.2e  %s\n",
                label, d_lp, d_gr, if (pass) "PASS" else "FAIL"))
    if (!pass) all_pass <<- FALSE
    invisible(NULL)
}

cat("----- test_wrap_autodiff_vs_hand.cpp -----\n")
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "tests_autodiff",
                              "test_wrap_autodiff_vs_hand.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)

for (mu in c(-1.5, 0.5, 2.0)) diag_one(sprintf("real(mu=%.2f)", mu),
                                       check_real_grad(mu))
for (sig in c(0.3, 1.0, 3.0)) diag_one(sprintf("positive(sig=%.2f)", sig),
                                       check_positive_grad(sig))
set.seed(1); y <- rnorm(20)
for (cfg in list(c(0.0, 1.0), c(1.5, 0.5), c(-0.5, 2.0))) {
    diag_one(sprintf("mixed(mu=%.2f,sig=%.2f)", cfg[1], cfg[2]),
             check_mixed_grad(y, cfg[1], cfg[2]))
}

cat("\n----- test_wrap_simplex.cpp -----\n")
AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "tests_autodiff",
                              "test_wrap_simplex.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)
y_counts <- c(10, 25, 30, 20, 15)
for (alpha_val in c(0.5, 1.0, 2.0)) {
    alpha_vec <- rep(alpha_val, length(y_counts))
    r <- check_simplex_grad(y_counts, alpha_vec, 10L, 12345L)
    pass <- !is.na(r$max_diff) && r$max_diff < TOL
    cat(sprintf("  simplex(alpha=%.2f): max_diff=%.2e  %s\n",
                alpha_val, r$max_diff, if (pass) "PASS" else "FAIL"))
    if (!pass) all_pass <- FALSE
}

cat(sprintf("\n========== CHECK #12 WRAP SUITE %s ==========\n",
            if (all_pass) "PASS" else "FAIL"))
if (!all_pass) quit(status = 1)

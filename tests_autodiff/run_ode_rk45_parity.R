# Parity test runner for ode_rk45 (Tier 1 ODE integrator).

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

cat("Compiling ode_rk45 parity test...\n")
ai4bayescode_sourceCpp(file.path(script_dir, "test_ode_rk45.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

cat("\nRunning parity tests (4 canonical ODE problems)...\n")
r <- test_ode_rk45()

cat("\n========== ODE RK45 PARITY SUMMARY ==========\n")
cat(sprintf("  all_pass = %s\n", r$all_pass))

cat("\n  Test 1: Linear ODE dy/dt = -0.5 y, y(0)=1\n")
cat(sprintf("    max_abs_err vs y0*exp(-k*t) = %.2e  (tol 1e-8)  PASS=%s\n",
            r$test1_linear_max_err, r$test1_linear_pass))

cat("\n  Test 2: Harmonic oscillator omega=2, 10 periods\n")
cat(sprintf("    max_abs_err vs analytical    = %.2e  (tol 1e-6)  \n",
            r$test2_harmonic_max_err))
cat(sprintf("    max rel energy deviation     = %.2e  (tol 1e-6)  PASS=%s\n",
            r$test2_harmonic_E_dev, r$test2_harmonic_pass))

cat("\n  Test 3: Lotka-Volterra (alpha=beta=delta=gamma=1), 15 time units\n")
cat(sprintf("    max rel conserved-V deviation = %.2e  (tol 1e-4)  PASS=%s\n",
            r$test3_lv_max_V_dev, r$test3_lv_pass))

cat("\n  Test 4: SIR (beta=0.4, gamma=0.1, N=1000, I0=10)\n")
cat(sprintf("    max rel total-N deviation     = %.2e  (tol 1e-8)\n",
            r$test4_sir_max_N_dev))
cat(sprintf("    S_final=%.2f  R_final=%.2f  (epidemic spread: %s)\n",
            r$test4_sir_S_final, r$test4_sir_R_final,
            r$test4_sir_S_final < 990))
cat(sprintf("    PASS=%s\n", r$test4_sir_pass))

if (!r$all_pass) { cat("\nFAIL\n"); quit(status = 1) }
cat("\nPASS\n")

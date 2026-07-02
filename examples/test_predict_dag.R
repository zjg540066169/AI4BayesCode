# ----------------------------------------------------------------------------
# test_predict_dag.R
#
# Tests for the predict DAG infrastructure:
#   1. BartNoise: predict_at returns f_bart, does NOT return sigma
#   2. BartNoise: get_dag returns correct edges
#   3. BartNoise: state unchanged after predict_at
#   4. GBartPoisson: predict_at returns r and rate (predict DAG smoke)
#   5. GaussianLocationScale: predict_at(list()) returns empty
#   6. GaussianLocationScale: predict_at(list(x=1)) throws error
#   7. DirichletSimplex: predict_at empty, get_dag empty
#   8. DirichletSparse: predict_at empty, get_dag empty
#   9. DirichletHierarchical: predict_at empty, get_dag empty
# ----------------------------------------------------------------------------

# Locate this script's directory portably (works under Rscript and under
# source()). Under Rscript, --file= is in commandArgs; under source(), fall
# back to getwd() (user must be cd'd into AI4BayesCode/examples/).
script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))

source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

n_pass <- 0L
n_fail <- 0L

pass <- function(msg) {
    cat(sprintf("  PASS: %s\n", msg))
    n_pass <<- n_pass + 1L
}
fail <- function(msg) {
    cat(sprintf("  FAIL: %s\n", msg))
    n_fail <<- n_fail + 1L
}
check <- function(cond, msg) {
    if (isTRUE(cond)) pass(msg) else fail(msg)
}

# ============================================================================
#  Test 1-3: BartNoise
# ============================================================================
cat("=== BartNoise predict DAG tests ===\n")
ai4bayescode_sourceCpp(
    file.path(script_dir, "BartNoise.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42L)
N <- 100L; p <- 3L
X <- matrix(rnorm(N * p), ncol = p)
y <- sin(X[,1]) + rnorm(N, 0, 0.5)
X_test <- matrix(rnorm(50 * p), ncol = p)

model <- new(BartNoise, X, y, 50L, 2.0, 2.0, 0.95, 3.0, 100L,
             FALSE, FALSE, 42L)
model$step(200L)

# Test 1: predict_at returns f_bart, NOT sigma
pred <- model$predict_at(list(X = X_test))
check("f_bart" %in% names(pred),
      "predict_at returns f_bart")
check(!("sigma" %in% names(pred)),
      "predict_at does NOT return sigma (unchanged parameter)")
check(length(pred$f_bart) == 50L,
      "f_bart has correct length (50)")
check(all(is.finite(pred$f_bart)),
      "f_bart values are all finite")

# Test 2: get_dag returns correct edges
dag <- model$get_dag()
check("X" %in% dag[["data_inputs"]],
      "get_dag: X is a data input")
check(identical(dag$predict_edges[["X"]], "f_bart"),
      "get_dag: X -> f_bart edge exists")
# sigma IS in predict DAG (sigma -> y_rep edge for the Gaussian observation
# layer). This was previously excluded in the mean-only predict design.
check("sigma" %in% names(dag$predict_edges),
      "get_dag: sigma -> y_rep edge exists (observation layer)")

# Test 3: state unchanged after predict_at
state_before <- model$get_current()
pred2 <- model$predict_at(list(X = X_test))
state_after <- model$get_current()
check(max(abs(state_before$f_bart - state_after$f_bart)) < 1e-15,
      "f_bart unchanged after predict_at")
check(abs(state_before$sigma - state_after$sigma) < 1e-15,
      "sigma unchanged after predict_at")

# Test: empty predict returns posterior-predictive y_rep at training X
pred_empty <- model$predict_at(list())
check("y_rep" %in% names(pred_empty),
      "predict_at(list()) returns posterior-predictive y_rep")
check(is.null(pred_empty$f_bart),
      "predict_at(list()) does not echo unchanged f_bart")

# Test: wrong key
err <- tryCatch({model$predict_at(list(Z = X_test)); FALSE},
                error = function(e) TRUE)
check(err, "predict_at rejects unknown key 'Z'")

# Test: wrong columns
err2 <- tryCatch({model$predict_at(list(X = X_test[,1:2])); FALSE},
                 error = function(e) TRUE)
check(err2, "predict_at rejects wrong column count")

# ============================================================================
#  Test 4: GBartPoisson
# ============================================================================
cat("\n=== GBartPoisson predict DAG tests ===\n")
ai4bayescode_sourceCpp(
    file.path(script_dir, "GBartPoisson.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42L)
X_pois <- matrix(runif(N * p), ncol = p)
y_pois <- rpois(N, exp(X_pois[,1]))

model_p <- new(GBartPoisson, X_pois, as.numeric(y_pois), 50L, 42L, FALSE)
model_p$step(200L)

pred_p <- model_p$predict_at(list(X = X_test))
check("r" %in% names(pred_p),
      "GBart predict_at returns r (log-rate)")
check("rate" %in% names(pred_p),
      "GBart predict_at returns rate (= exp(r))")
check(all(is.finite(pred_p$r)),
      "r values are finite")
check(all(pred_p$rate > 0),
      "rate values are positive")

dag_p <- model_p$get_dag()
check("X" %in% dag_p[["data_inputs"]],
      "GBart get_dag: X is data input")

# ============================================================================
#  Test 5-6: GaussianLocationScale (no covariates)
# ============================================================================
cat("\n=== GaussianLocationScale (no covariates) ===\n")
ai4bayescode_sourceCpp(
    file.path(script_dir, "GaussianLocationScale.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

model_g <- new(GaussianLocationScale, rnorm(50), 42L)
model_g$step(100L)

pred_g <- model_g$predict_at(list())
check("y_rep" %in% names(pred_g),
      "Gaussian predict_at(list()) returns posterior-predictive y_rep")

err_g <- tryCatch({model_g$predict_at(list(x = 1.0)); FALSE},
                  error = function(e) TRUE)
check(err_g, "Gaussian predict_at rejects non-empty input")

dag_g <- model_g$get_dag()
check(length(dag_g[["data_inputs"]]) == 0L,
      "Gaussian get_dag: no data inputs")

# ============================================================================
#  Test 7: DirichletSimplex (no covariates)
# ============================================================================
cat("\n=== DirichletSimplex (no covariates) ===\n")
ai4bayescode_sourceCpp(
    file.path(script_dir, "DirichletSimplex.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

model_ds <- new(DirichletSimplex, c(10, 20, 30), c(1, 1, 1), 42L)
model_ds$step(100L)

pred_ds <- model_ds$predict_at(list())
check("y_rep" %in% names(pred_ds),
      "DirichletSimplex predict_at returns posterior-predictive y_rep")

dag_ds <- model_ds$get_dag()
check(length(dag_ds[["data_inputs"]]) == 0L,
      "DirichletSimplex get_dag: no data inputs")

# ============================================================================
#  Test 8: DirichletSparse (no covariates)
# ============================================================================
cat("\n=== DirichletSparse (no covariates) ===\n")
ai4bayescode_sourceCpp(
    file.path(script_dir, "DirichletSparse.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

model_sp <- new(DirichletSparse, c(10, 5, 3, 2, 0), 42L)
model_sp$step(100L)

pred_sp <- model_sp$predict_at(list())
check("y_rep" %in% names(pred_sp),
      "DirichletSparse predict_at returns posterior-predictive y_rep")

dag_sp <- model_sp$get_dag()
check(length(dag_sp[["data_inputs"]]) == 0L,
      "DirichletSparse get_dag: no data inputs")

# ============================================================================
#  Test 9: DirichletHierarchical (no covariates)
# ============================================================================
cat("\n=== DirichletHierarchical (no covariates) ===\n")
ai4bayescode_sourceCpp(
    file.path(script_dir, "DirichletHierarchical.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir)

# Quick synthetic data
rdirichlet <- function(n, alpha) {
    k <- length(alpha)
    x <- matrix(rgamma(n * k, shape = alpha), nrow = n, byrow = TRUE)
    x[x < 1e-300] <- 1e-300
    x / rowSums(x)
}
S_obs <- rdirichlet(20, rep(1, 5))

model_dh <- new(DirichletHierarchical, S_obs, 2.0, 5.0, 42L)
model_dh$step(100L)

pred_dh <- model_dh$predict_at(list())
check("y_rep" %in% names(pred_dh),
      "DirichletHierarchical predict_at returns posterior-predictive y_rep")

dag_dh <- model_dh$get_dag()
check(length(dag_dh[["data_inputs"]]) == 0L,
      "DirichletHierarchical get_dag: no data inputs")

# ============================================================================
#  Summary
# ============================================================================
cat(sprintf("\n=== %d passed, %d failed ===\n", n_pass, n_fail))
cat(sprintf("%s\n", if (n_fail == 0L) "ALL TESTS PASS" else "SOME TESTS FAILED"))

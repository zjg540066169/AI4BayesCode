# Smoke test for GPRegression.

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

cat("Compiling GPRegression...\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "GPRegression.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)

# Simulate a simple 1D GP-like dataset
set.seed(1)
N <- 80L; p <- 1L
X <- matrix(seq(-3, 3, length.out = N), N, p)
f_true <- sin(X[,1]) + 0.5 * X[,1]
y <- as.numeric(f_true + rnorm(N, 0, 0.3))
cat(sprintf("Data: N=%d, p=%d, sd(y)=%.3f\n", N, p, sd(y)))

# ---------------------- Stateful construct + 10 steps --------------------
cat("\n[1] Construct + 10 steps...\n")
m <- new(GPRegression, X, y, 1L, FALSE)
m$step(10L)
d <- m$get_current()
cat(sprintf("  keys = %s\n", paste(names(d), collapse=",")))
stopifnot(all(c("f","amplitude","lengthscale","sigma") %in% names(d)))
stopifnot(length(d$f) == N)
stopifnot(all(is.finite(d$f)))
stopifnot(d$amplitude > 0, d$lengthscale > 0, d$sigma > 0)
cat(sprintf("  amp=%.3f ell=%.3f sigma=%.3f\n", d$amplitude, d$lengthscale, d$sigma))

# ---------------------- DAG ----------------------------------------------
cat("\n[2] DAG consistency...\n")
dag <- m$get_dag()
cat("  gibbs_reads:\n")
for (nm in names(dag$gibbs_reads))
    cat(sprintf("    %-15s <- %s\n", nm, paste(dag$gibbs_reads[[nm]], collapse=", ")))
cat("  predict_edges:\n")
for (nm in names(dag$predict_edges))
    cat(sprintf("    %-15s -> %s\n", nm, paste(dag$predict_edges[[nm]], collapse=", ")))
cat(sprintf("  data_inputs = {%s}\n", paste(dag$data_inputs, collapse=", ")))
stopifnot("X" %in% dag$data_inputs)

# ---------------------- predict_at state preservation --------------------
cat("\n[3] predict_at state preservation...\n")
before <- m$get_current()
# Empty predict
pp_empty <- m$predict_at(list())
after1 <- m$get_current()
stopifnot(identical(before, after1))
cat("  empty predict: state preserved\n")
stopifnot(all(c("f_mean","y_rep") %in% names(pp_empty)))

# X predict
X_test <- matrix(seq(-4, 4, length.out = 20), 20, p)
pp_x <- m$predict_at(list(X = X_test))
after2 <- m$get_current()
stopifnot(identical(before, after2))
cat("  X predict: state preserved\n")
stopifnot(all(c("f_mean","f_sd","f_star","y_rep") %in% names(pp_x)))
stopifnot(length(pp_x$f_mean) == 20)
stopifnot(length(pp_x$f_sd) == 20)
stopifnot(all(is.finite(pp_x$f_mean)))
stopifnot(all(pp_x$f_sd >= 0))
cat(sprintf("  f_mean range = [%.3f, %.3f]\n", min(pp_x$f_mean), max(pp_x$f_mean)))
cat(sprintf("  f_sd   range = [%.3f, %.3f]\n", min(pp_x$f_sd),   max(pp_x$f_sd)))

# ---------------------- set_current round-trip ---------------------------
cat("\n[4] set_current round-trip...\n")
s1 <- m$get_current()
m$set_current(list(amplitude = 1.0, lengthscale = 0.5, sigma = 0.3))
s2 <- m$get_current()
stopifnot(abs(s2$amplitude - 1.0) < 1e-10)
stopifnot(abs(s2$lengthscale - 0.5) < 1e-10)
stopifnot(abs(s2$sigma - 0.3) < 1e-10)
cat("  hyperparams overwrite OK\n")

# ---------------------- Quick 2k burnin + 2k keep check ------------------
cat("\n[5] Short 2k+2k chain with keep_history=TRUE...\n")
set.seed(1)
mh <- new(GPRegression, X, y, 42L, TRUE)
t0 <- Sys.time(); mh$step(2000L); mh$step(2000L); t1 <- Sys.time()
cat(sprintf("  wall = %.1fs for 4000 sweeps\n",
            as.numeric(difftime(t1, t0, units="secs"))))
h <- mh$get_history()
cat(sprintf("  history keys: %s\n", paste(names(h), collapse=",")))
stopifnot("f" %in% names(h))
stopifnot(nrow(h$f) == 4000)
stopifnot(ncol(h$f) == N)

# Posterior mean of f vs truth
f_pm <- colMeans(h$f[2001:4000, , drop=FALSE])
cor_f <- cor(f_pm, f_true)
rmse_f <- sqrt(mean((f_pm - f_true)^2))
cat(sprintf("  post-mean f vs truth: cor = %.3f, rmse = %.3f\n", cor_f, rmse_f))
stopifnot(cor_f > 0.9)  # should be very high for this smooth signal

# Hyperparam posterior means
amp_post <- mean(h$amplitude[2001:4000])
ell_post <- mean(h$lengthscale[2001:4000])
sig_post <- mean(h$sigma[2001:4000])
cat(sprintf("  post-mean amp=%.3f  ell=%.3f  sigma=%.3f (true sigma=0.3)\n",
            amp_post, ell_post, sig_post))
stopifnot(abs(sig_post - 0.3) < 0.15)  # should recover noise sigma

cat("\n========== GP REGRESSION SMOKE PASS ==========\n")

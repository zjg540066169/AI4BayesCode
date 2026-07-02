# Smoke test for GPClassification (ESS + libgp SE kernel + Bernoulli-logit).

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

cat("Compiling GPClassification...\n")
ai4bayescode_sourceCpp(file.path(AI4BayesCode_dir, "examples",
                              "GPClassification.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)

# Simulate 1D binary classification where y follows a smooth probit-like
# boundary: p(y=1 | x) = sigmoid(sin(2x) - 0.3x).
set.seed(1)
N <- 80L; p <- 1L
X <- matrix(seq(-3, 3, length.out = N), N, p)
f_true <- sin(2 * X[,1]) - 0.3 * X[,1]
prob_true <- 1 / (1 + exp(-f_true))
y <- as.numeric(runif(N) < prob_true)
cat(sprintf("Data: N=%d, p=%d, pct y=1: %.1f%%\n",
            N, p, 100 * mean(y)))

# ---------------------- Construct + 10 steps -----------------------------
cat("\n[1] Construct + 10 steps...\n")
m <- new(GPClassification, X, y, 1L, FALSE)
m$step(10L)
d <- m$get_current()
cat(sprintf("  keys = %s\n", paste(names(d), collapse = ",")))
stopifnot(all(c("f", "amplitude", "lengthscale") %in% names(d)))
stopifnot(length(d$f) == N)
stopifnot(all(is.finite(d$f)))
stopifnot(d$amplitude > 0, d$lengthscale > 0)
cat(sprintf("  amp=%.3f  ell=%.3f\n", d$amplitude, d$lengthscale))

# ---------------------- DAG ----------------------------------------------
cat("\n[2] DAG consistency...\n")
dag <- m$get_dag()
cat("  gibbs_reads:\n")
for (nm in names(dag$gibbs_reads))
    cat(sprintf("    %-15s <- %s\n", nm,
                paste(dag$gibbs_reads[[nm]], collapse = ", ")))
cat("  predict_edges:\n")
for (nm in names(dag$predict_edges))
    cat(sprintf("    %-15s -> %s\n", nm,
                paste(dag$predict_edges[[nm]], collapse = ", ")))
cat(sprintf("  data_inputs = {%s}\n",
            paste(dag$data_inputs, collapse = ", ")))
stopifnot("X" %in% dag$data_inputs)

# ---------------------- predict_at state preservation --------------------
cat("\n[3] predict_at state preservation...\n")
before <- m$get_current()
pp_empty <- m$predict_at(list())
after1 <- m$get_current()
stopifnot(identical(before, after1))
stopifnot(all(c("f_mean", "prob", "y_rep") %in% names(pp_empty)))
stopifnot(length(pp_empty$prob) == N)
stopifnot(all(pp_empty$prob >= 0 & pp_empty$prob <= 1))
stopifnot(all(pp_empty$y_rep %in% c(0, 1)))

X_test <- matrix(seq(-4, 4, length.out = 20), 20, p)
pp_x <- m$predict_at(list(X = X_test))
after2 <- m$get_current()
stopifnot(identical(before, after2))
stopifnot(all(c("f_mean", "prob", "y_rep") %in% names(pp_x)))
stopifnot(length(pp_x$f_mean) == 20)
stopifnot(all(is.finite(pp_x$f_mean)))
stopifnot(all(pp_x$prob >= 0 & pp_x$prob <= 1))
stopifnot(all(pp_x$y_rep %in% c(0, 1)))
cat(sprintf("  f_mean range = [%.3f, %.3f]\n",
            min(pp_x$f_mean), max(pp_x$f_mean)))
cat(sprintf("  prob   range = [%.3f, %.3f]\n",
            min(pp_x$prob),   max(pp_x$prob)))

# ---------------------- set_current round-trip ---------------------------
cat("\n[4] set_current round-trip...\n")
m$set_current(list(amplitude = 1.5, lengthscale = 0.8))
s2 <- m$get_current()
stopifnot(abs(s2$amplitude - 1.5) < 1e-10)
stopifnot(abs(s2$lengthscale - 0.8) < 1e-10)
cat("  hyperparams overwrite OK\n")

# ---------------------- 2k+2k chain recovery -----------------------------
cat("\n[5] Short 2k+2k chain with keep_history=TRUE...\n")
set.seed(1)
mh <- new(GPClassification, X, y, 42L, TRUE)
t0 <- Sys.time()
mh$step(2000L)
mh$step(2000L)
t1 <- Sys.time()
wall <- as.numeric(difftime(t1, t0, units = "secs"))
cat(sprintf("  wall = %.1fs for 4000 sweeps (%.3fs / sweep)\n",
            wall, wall / 4000))
h <- mh$get_history()
cat(sprintf("  history keys: %s\n", paste(names(h), collapse = ",")))
# Whitened parameterization: z is sampled, f is recovered as f = L(amp,ell)·z.
# History therefore stores {z, amplitude, lengthscale}, NOT f directly.
stopifnot(all(c("z", "amplitude", "lengthscale") %in% names(h)))
stopifnot(nrow(h$z) == 4000)
stopifnot(ncol(h$z) == N)
stopifnot(nrow(h$amplitude) == 4000)
stopifnot(nrow(h$lengthscale) == 4000)

# Reconstruct f for each kept draw via f_i = L(amp_i, ell_i) · z_i.
# Match the C++ jitter (1e-5) so that L is comparable.
D2 <- as.matrix(dist(X))^2
recover_f <- function(h, idx) {
    n_kept <- length(idx)
    f_mat  <- matrix(NA_real_, n_kept, N)
    amp_v  <- as.numeric(h$amplitude)
    ell_v  <- as.numeric(h$lengthscale)
    for (k in seq_along(idx)) {
        i <- idx[k]
        K <- amp_v[i]^2 * exp(-D2 / (2 * ell_v[i]^2))
        diag(K) <- diag(K) + 1e-5
        L_lower <- t(chol(K))                  # R's chol returns U; we want L
        f_mat[k, ] <- as.numeric(L_lower %*% h$z[i, ])
    }
    f_mat
}
f_kept <- recover_f(h, 2001:4000)
f_pm   <- colMeans(f_kept)
cor_f  <- cor(f_pm, f_true)
rmse_f <- sqrt(mean((f_pm - f_true)^2))
cat(sprintf("  post-mean f vs truth: cor = %.3f, rmse = %.3f\n",
            cor_f, rmse_f))
stopifnot(cor_f > 0.6)  # classification is harder than regression

# Classification accuracy at training x based on posterior-mean f.
prob_pm <- 1 / (1 + exp(-f_pm))
acc <- mean((prob_pm > 0.5) == (y == 1))
cat(sprintf("  train accuracy = %.3f\n", acc))
stopifnot(acc > 0.7)

amp_post <- mean(as.numeric(h$amplitude)[2001:4000])
ell_post <- mean(as.numeric(h$lengthscale)[2001:4000])
cat(sprintf("  post-mean amp=%.3f  ell=%.3f\n", amp_post, ell_post))

cat("\n========== GP CLASSIFICATION SMOKE PASS ==========\n")

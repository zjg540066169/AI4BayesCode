# Diagnose DirichletSparse R-hat: is the high R-hat driven by (a) the
# sparse / near-zero components (expected flat-posterior identifiability
# issue), or (b) a real sampler bug in the larger components?

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))
suppressPackageStartupMessages({ library(posterior) })

AI4BayesCode_sourceCpp(file.path(AI4BayesCode_dir, "examples", "DirichletSparse.cpp"),
                    AI4BayesCode_path = AI4BayesCode_dir)

set.seed(42); P <- 20L; N_tot <- 500L
s_true <- c(rep(1/5, 5), rep(1e-5, P-5))
s_true <- s_true / sum(s_true)
y_sp <- as.numeric(tabulate(sample.int(P, N_tot, prob=s_true, replace=TRUE), nbins=P))

cat("y_sp (data):", y_sp, "\n")
cat("indices with y > 0:", which(y_sp > 0), "\n")

run <- function(cs, nb, nk) {
    m <- new(DirichletSparse, y_sp, as.integer(cs), TRUE)
    m$step(nb); m$step(nk)
    h <- m$get_history()
    list(s = h$s[(nb+1):(nb+nk), , drop=FALSE],
         theta = h$theta[(nb+1):(nb+nk)])
}
c1 <- run(101L, 5000L, 5000L)
c2 <- run(202L, 5000L, 5000L)

cat("\n-- per-component R-hat on natural-scale s --\n")
for (j in seq_len(P)) {
    arr <- array(NA_real_, dim=c(nrow(c1$s), 2, 1))
    arr[,1,1] <- c1$s[,j]; arr[,2,1] <- c2$s[,j]
    rh <- posterior::rhat(arr[,,1])
    m1 <- mean(c1$s[,j]); m2 <- mean(c2$s[,j])
    s1 <- sd(c1$s[,j]);   s2 <- sd(c2$s[,j])
    cat(sprintf("  j=%2d  y=%3.0f  Rhat=%.3f  chain1: %.4e±%.3e  chain2: %.4e±%.3e  truth=%.4f\n",
        j, y_sp[j], rh, m1, s1, m2, s2, s_true[j]))
}
cat("\ntheta Rhat:", posterior::rhat(array(cbind(c1$theta, c2$theta), dim=c(length(c1$theta), 2, 1))[,,1]), "\n")
cat("posterior mean theta:", mean(c(c1$theta, c2$theta)), "\n")

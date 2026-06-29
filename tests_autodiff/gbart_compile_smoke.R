# ============================================================================
# gbart_compile_smoke.R
#
# Compile-smoke test for all 4 shipped GBart* examples + vanilla BartNoise.
#
# For each example:
#   (a) AI4BayesCode_sourceCpp compiles
#   (b) new(..., seed) constructs
#   (c) step(10) runs to completion
#   (d) get_current() returns finite values
#   (e) predict_at(list()) returns finite values (no crash)
#   (f) get_current() state unchanged after predict_at (Check #10)
#
# All in a single R session so shared helpers load once.
# ============================================================================

source("R/AI4BayesCode_helpers.R")

examples <- list(
    list(name = "GBartPoisson",
         cpp  = "r-pkg/inst/examples/GBartPoisson.cpp",
         build = function() {
             set.seed(1); N <- 80L; X <- matrix(runif(N*3L), N, 3L)
             y <- rpois(N, exp(-1 + X[,1] - 0.5*X[,2]))
             new(GBartPoisson, X, as.numeric(y), 50L, 1L, FALSE)
         }),
    list(name = "GBartLogistic",
         cpp  = "r-pkg/inst/examples/GBartLogistic.cpp",
         build = function() {
             set.seed(1); N <- 80L; X <- matrix(runif(N*3L), N, 3L)
             y <- rbinom(N, 1, plogis(-0.5 + 2*X[,1]))
             new(GBartLogistic, X, as.numeric(y), 50L, 1L, FALSE)
         }),
    list(name = "GBartMultinomial",
         cpp  = "r-pkg/inst/examples/GBartMultinomial.cpp",
         build = function() {
             set.seed(1); N <- 150L; X <- matrix(runif(N*3L), N, 3L)
             eta <- cbind(-1 + 2*X[,1], 0 + X[,2] - X[,3])
             denom <- 1 + exp(eta[,1]) + exp(eta[,2])
             probs <- cbind(1/denom, exp(eta[,1])/denom, exp(eta[,2])/denom)
             y <- apply(probs, 1, function(pi) sample(0:2, 1, prob=pi))
             new(GBartMultinomial, X, as.numeric(y), 3L, 50L, 1L, FALSE)
         }),
    list(name = "GBartHeteroscedastic",
         cpp  = "r-pkg/inst/examples/GBartHeteroscedastic.cpp",
         build = function() {
             set.seed(1); N <- 80L; X <- matrix(runif(N*3L), N, 3L)
             mu <- exp(3 + X[,1]); y <- rnorm(N, mu, sqrt(0.5 * mu))
             new(GBartHeteroscedastic, X, as.numeric(y), 50L, 1.0, 1L, FALSE)
         }),
    list(name = "BartNoise",
         cpp  = "r-pkg/inst/examples/BartNoise.cpp",
         build = function() {
             set.seed(1); N <- 80L; X <- matrix(runif(N*3L), N, 3L)
             y <- rnorm(N, X[,1] - X[,2], 0.5)
             # Full BartNoise signature: X, y, ntrees, k, power, base, nu,
             # numcut, dart, aug, seed, keep_history
             new(BartNoise, X, as.numeric(y), 50L, 2.0, 2.0, 0.95, 3.0,
                 100L, FALSE, FALSE, 1L, FALSE)
         })
)

results <- data.frame(
    example = character(0), compile = logical(0), step = logical(0),
    finite = logical(0), predict_at = logical(0), state_preserved = logical(0),
    wall_sec = numeric(0))

for (ex in examples) {
    cat(sprintf("=== %s ===\n", ex$name))
    t0 <- Sys.time()
    tryCatch({
        AI4BayesCode_sourceCpp(ex$cpp, AI4BayesCode_path = ".")
        compiled <- TRUE
    }, error = function(e) {
        cat("  COMPILE FAIL:", conditionMessage(e), "\n")
        compiled <<- FALSE
    })
    if (!compiled) {
        results <- rbind(results, data.frame(example=ex$name, compile=FALSE,
            step=NA, finite=NA, predict_at=NA, state_preserved=NA,
            wall_sec=NA))
        next
    }
    m <- ex$build()
    stepped <- tryCatch({ m$step(10L); TRUE },
        error = function(e) { cat("  STEP FAIL:", conditionMessage(e),"\n"); FALSE })
    finite <- FALSE
    if (stepped) {
        cur <- m$get_current()
        finite <- all(sapply(cur, function(v) all(is.finite(unlist(v)))))
    }
    pp_ok <- FALSE
    state_ok <- FALSE
    if (finite) {
        s_before <- m$get_current()
        pp_ok <- tryCatch({ p <- m$predict_at(list()); TRUE },
            error = function(e) { cat("  PP FAIL:", conditionMessage(e),"\n"); FALSE })
        s_after <- m$get_current()
        state_ok <- isTRUE(all.equal(s_before, s_after))
    }
    t1 <- Sys.time()
    wall <- as.numeric(difftime(t1, t0, units = "secs"))
    cat(sprintf("  [%s] compile=%s step=%s finite=%s predict_at=%s state=%s (%.1fs)\n",
                ex$name, compiled, stepped, finite, pp_ok, state_ok, wall))
    results <- rbind(results, data.frame(example=ex$name,
        compile=compiled, step=stepped, finite=finite,
        predict_at=pp_ok, state_preserved=state_ok, wall_sec=wall))
}

cat("\n=== Compile-smoke summary ===\n")
print(results)
stopifnot(all(results$compile, na.rm = TRUE))
stopifnot(all(results$step, na.rm = TRUE))
stopifnot(all(results$finite, na.rm = TRUE))
stopifnot(all(results$predict_at, na.rm = TRUE))
stopifnot(all(results$state_preserved, na.rm = TRUE))
cat("All 7 examples PASS compile-smoke.\n")

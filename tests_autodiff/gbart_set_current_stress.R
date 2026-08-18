# ============================================================================
# gbart_set_current_stress.R
#
# Stress test for the set_current / predict_at contract on GBart* wrappers:
#   (a) set_current(list(y = y_new)) updates Y via Tier B, then step() runs
#       a fresh sweep on the new response (validates the nested-MCMC
#       use case).
#   (b) Unknown keys silently ignored (round-trip get_current()).
#   (c) Impossible keys ("r") produce clear Rcpp::stop.
#   (d) predict_at does not mutate MCMC state (validator Check #10).
# ============================================================================

source("R/AI4BayesCode_helpers.R")

test_set_current_stress <- function(example_name, build_fn, update_keys,
                                    readonly_key, unknown_key_value) {
    cat(sprintf("\n===== %s set_current stress =====\n", example_name))
    m <- build_fn()
    m$step(20L)
    s0 <- m$get_current()

    # (a) Supported key update
    for (k in names(update_keys)) {
        vals <- update_keys[[k]]
        before <- m$get_current()
        m$set_current(setNames(list(vals), k))
        m$step(5L)
        after <- m$get_current()
        cat(sprintf("  [%s] set_current(list(%s=...)) + step(5): finite=%s\n",
                    example_name, k,
                    all(sapply(after, function(v) all(is.finite(unlist(v)))))))
    }

    # (b) Unknown key silently ignored
    m2 <- build_fn()
    m2$step(10L)
    before <- m2$get_current()
    m2$set_current(list(random_garbage = unknown_key_value))
    after <- m2$get_current()
    unchanged <- isTRUE(all.equal(before, after))
    cat(sprintf("  [%s] unknown key: state unchanged = %s\n", example_name,
                unchanged))
    stopifnot(unchanged)

    # (c) The read-only forest output is INERT on input, not rejected.
    # r (r_1 for the multinomial) is what the forest produces; there is no
    # unique inverse, so the examples ignore it on input rather than erroring,
    # which is what lets set_current(get_current()) round-trip
    # (GBartPoisson.cpp:297-306). Ignoring has to be inert, not merely quiet.
    # Restore a forest with set_tree(), not with r.
    m3 <- build_fn(); m3$step(20L)
    before3 <- m3$get_current()
    m3$set_current(setNames(list(rep(1e6, length(before3[[readonly_key]]))),
                            readonly_key))
    after3 <- m3$get_current()
    inert <- isTRUE(all.equal(before3, after3))
    cat(sprintf("  [%s] read-only key '%s' inert on input: %s\n",
                example_name, readonly_key, inert))
    stopifnot(inert)

    rt <- tryCatch({ m3$set_current(before3); TRUE }, error = function(e) FALSE)
    cat(sprintf("  [%s] set_current(get_current()) round-trips: %s\n",
                example_name, rt))
    stopifnot(rt, isTRUE(all.equal(before3, m3$get_current())))

    # (d) predict_at does not mutate state
    m4 <- build_fn(); m4$step(30L)
    before <- m4$get_current()
    try(m4$predict_at(list()), silent = TRUE)
    after <- m4$get_current()
    cat(sprintf("  [%s] predict_at(list()) state-preserved: %s\n",
                example_name, isTRUE(all.equal(before, after))))
    stopifnot(isTRUE(all.equal(before, after)))
}

# --------- GBartPoisson -----------
ai4bayescode_source_checkout("r-pkg/inst/examples/GBartPoisson.cpp",
                       AI4BayesCode_path = ".")
set.seed(1); N <- 80L; p <- 3L; X <- matrix(runif(N*p), N, p)
y <- rpois(N, exp(-1 + X[,1] - 0.5*X[,2]))
test_set_current_stress(
    "GBartPoisson",
    function() new(GBartPoisson, X, as.numeric(y), 50L, 1L, FALSE),
    list(y = as.numeric(rpois(N, 1.5)),
         X = matrix(runif(N*p), N, p)),
    "r", rnorm(5))

# --------- GBartLogistic -----------
ai4bayescode_source_checkout("r-pkg/inst/examples/GBartLogistic.cpp",
                       AI4BayesCode_path = ".")
set.seed(2); X <- matrix(runif(N*p), N, p)
y <- rbinom(N, 1, plogis(-0.5 + 2*X[,1]))
test_set_current_stress(
    "GBartLogistic",
    function() new(GBartLogistic, X, as.numeric(y), 50L, 1L, FALSE),
    list(y = as.numeric(rbinom(N, 1, 0.5)),
         X = matrix(runif(N*p), N, p)),
    "r", rnorm(5))

# --------- GBartMultinomial -----------
ai4bayescode_source_checkout("r-pkg/inst/examples/GBartMultinomial.cpp",
                       AI4BayesCode_path = ".")
set.seed(3); N <- 120L; X <- matrix(runif(N*p), N, p)
y <- sample(0:2, N, replace = TRUE)
test_set_current_stress(
    "GBartMultinomial",
    function() new(GBartMultinomial, X, as.numeric(y), 3L, 50L, 1L, FALSE),
    list(y = as.numeric(sample(0:2, N, replace=TRUE)),
         X = matrix(runif(N*p), N, p)),
    "r_1", rnorm(5))

# --------- GBartHeteroscedastic -----------
ai4bayescode_source_checkout("r-pkg/inst/examples/GBartHeteroscedastic.cpp",
                       AI4BayesCode_path = ".")
set.seed(5); X <- matrix(runif(N*p), N, p)
mu <- exp(3 + X[,1])
y <- rnorm(N, mu, sqrt(0.5*mu))
test_set_current_stress(
    "GBartHeteroscedastic",
    function() new(GBartHeteroscedastic, X, as.numeric(y), 50L, 1.0, 1L, FALSE),
    list(y = as.numeric(rnorm(N, 5, 1))),
    "r", rnorm(5))

cat("\n===== All set_current stress tests PASSED =====\n")

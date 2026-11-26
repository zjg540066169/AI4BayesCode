# ai4bayescode_run_chains() calls set.seed(seeds[i]) immediately before
# model_ctor(seeds[i]) for chain i in every path (persistent worker
# processes, mclapply with keep_models = FALSE, sequential, and the
# sequential re-run after a failed parallel attempt), and a sequential run
# and the mclapply call restore the caller's .Random.seed. The fork tests
# are skipped on Windows; the BartNoise test compiles the model and is
# skipped on CRAN.

# A model written in R that draws from R's random-number generator, as the
# BART kernels do under the Rcpp backend: each sweep appends one runif().
rng_ctor <- function(seed) {
    e <- new.env()
    e$draws <- numeric(0)
    e$step <- function(n = 1L) e$draws <- c(e$draws, stats::runif(n))
    e$get_history <- function() list(u = e$draws)
    e$clear_history <- function() e$draws <- numeric(0)
    e
}

rng_seeds <- c(11L, 22L, 33L)

# The kept draws of chain i with n_burn = 2 and n_keep = 5: draws 3 to 7 of
# the stream that set.seed(rng_seeds[i]) starts, under RNG kind `kind`.
rng_expected <- function(kind = NULL)
    lapply(rng_seeds, function(s)
        withr::with_seed(s, stats::runif(7L), .rng_kind = kind)[3:7])

rng_run <- function(ctor = rng_ctor, ...)
    ai4bayescode_run_chains(ctor, n_chains = 3, n_burn = 2, n_keep = 5,
                            seeds = rng_seeds, verbose = FALSE, ...)

draws_u <- function(run) lapply(run$histories, `[[`, "u")

caller_seed <- function()
    get0(".Random.seed", envir = globalenv(), inherits = FALSE)

# BartNoise class generator, compiled on first use.
bart_class <- local({
    env <- NULL
    function() {
        if (is.null(env)) {
            e <- new.env()
            suppressMessages(ai4bayescode_example("BartNoise", env = e,
                                                  quiet = TRUE))
            env <<- e
        }
        env$BartNoise
    }
})

test_that("chain i draws from the stream of set.seed(seeds[i]) in every path", {
    skip_on_os("windows"); skip_if_not_installed("withr")
    withr::local_preserve_seed()
    set.seed(1); run_p <- rng_run(n_cores = 2)
    withr::defer(ai4bayescode_close_models(run_p))
    expect_s3_class(run_p$models[[1]], "ai4bayescode_chain_model")
    set.seed(2); run_m <- rng_run(n_cores = 2, keep_models = FALSE)
    expect_null(run_m$models)
    set.seed(3); run_s <- rng_run(parallel = FALSE)
    set.seed(4); run_1 <- rng_run(n_cores = 1)
    for (run in list(run_p, run_m, run_s, run_1))
        expect_identical(draws_u(run), rng_expected())
    # With RNGkind "L'Ecuyer-CMRG", set.seed() seeds that generator.
    run_l <- withr::with_seed(5, rng_run(n_cores = 2),
                              .rng_kind = "L'Ecuyer-CMRG")
    withr::defer(ai4bayescode_close_models(run_l))
    expect_identical(draws_u(run_l), rng_expected("L'Ecuyer-CMRG"))
    expect_false(identical(draws_u(run_l), rng_expected()))
})

test_that("a sequential run leaves the caller's .Random.seed unchanged", {
    skip_if_not_installed("withr")
    withr::local_preserve_seed()
    set.seed(9); before <- caller_seed()
    run_s <- rng_run(parallel = FALSE)
    expect_identical(caller_seed(), before)
    expect_identical(draws_u(run_s), rng_expected())
    # No .Random.seed before the run: none after it.
    rm(".Random.seed", envir = globalenv())
    run_s <- rng_run(parallel = FALSE, keep_models = FALSE)
    expect_null(caller_seed())
    expect_identical(draws_u(run_s), rng_expected())
    # A chain that fails: .Random.seed is restored and the error is raised.
    set.seed(9)
    bad <- function(seed) if (seed == 22L) stop("chain two fails") else
        rng_ctor(seed)
    expect_error(rng_run(bad, parallel = FALSE), "chain two fails")
    expect_identical(caller_seed(), before)
})

test_that("keep_models = FALSE: chains run in this session restore .Random.seed", {
    skip_on_os("windows"); skip_if_not_installed("withr")
    withr::local_preserve_seed()
    run_k <- function(n_chains, ...)
        ai4bayescode_run_chains(rng_ctor, n_chains = n_chains, n_burn = 2,
                                n_keep = 5, seeds = rng_seeds[seq_len(n_chains)],
                                keep_models = FALSE, ...)
    # One chain runs sequentially in this session, also with n_cores = 2.
    set.seed(9); before <- caller_seed()
    expect_message(run_1 <- run_k(1, n_cores = 2),
                   "running 1 chains sequentially")
    expect_identical(caller_seed(), before)
    expect_identical(draws_u(run_1), rng_expected()[1])
    # Two chains with n_cores = 1.5: parallel::mclapply evaluates them in
    # this session (as.integer(mc.cores) < 2).
    run_2 <- run_k(2, n_cores = 1.5, verbose = FALSE)
    expect_identical(caller_seed(), before)
    expect_identical(draws_u(run_2), rng_expected()[1:2])
    # Two chains in forked processes.
    run_f <- run_k(2, n_cores = 2, verbose = FALSE)
    expect_identical(caller_seed(), before)
    expect_identical(draws_u(run_f), rng_expected()[1:2])
    # The caller's next draws equal the draws without the runs.
    set.seed(9); ref <- stats::runif(3)
    set.seed(9)
    run_k(1, n_cores = 2, verbose = FALSE)
    run_k(2, n_cores = 1.5, verbose = FALSE)
    expect_identical(stats::runif(3), ref)
    # RNGkind "L'Ecuyer-CMRG" and no .Random.seed: mc.reset.stream() in
    # parallel::mclapply() creates .Random.seed; it is removed after the run.
    withr::local_seed(1, .rng_kind = "L'Ecuyer-CMRG")
    for (nc in c(1.5, 2)) {
        if (!is.null(caller_seed())) rm(".Random.seed", envir = globalenv())
        run_l <- run_k(2, n_cores = nc, verbose = FALSE)
        expect_null(caller_seed())
        expect_identical(draws_u(run_l), rng_expected("L'Ecuyer-CMRG")[1:2])
    }
})

test_that("the sequential re-run after a failed parallel attempt uses the same seeds", {
    skip_on_os("windows"); skip_if_not_installed("withr")
    withr::local_preserve_seed()
    parent <- Sys.getpid()
    # The constructor fails in a forked worker process only, so the run
    # falls back to running every chain in this session.
    fork_fails <- function(seed) {
        if (Sys.getpid() != parent) stop("not in this session")
        rng_ctor(seed)
    }
    set.seed(9); before <- caller_seed()
    expect_warning(run_f <- rng_run(fork_fails, n_cores = 2),
                   "re-running all chains sequentially in this session")
    expect_false(inherits(run_f$models[[1]], "ai4bayescode_chain_model"))
    expect_identical(draws_u(run_f), rng_expected())
    expect_identical(caller_seed(), before)
    # The mclapply path (keep_models = FALSE) and its sequential re-run.
    run_g <- suppressWarnings(rng_run(fork_fails, n_cores = 2,
                                      keep_models = FALSE))
    expect_identical(draws_u(run_g), rng_expected())
    expect_identical(caller_seed(), before)
})

test_that("BartNoise chains: same seeds, same draws, in every path", {
    skip_on_cran(); skip_on_os("windows"); skip_if_not_installed("withr")
    withr::local_preserve_seed()
    BN <- bart_class()
    set.seed(17)
    N <- 100L
    X <- matrix(stats::runif(N * 3L, -1, 1), N, 3L)
    y <- 2 * pmax(X[, 1], 0) + 0.5 * X[, 2]^2 - X[, 3] +
        stats::rnorm(N, 0, 0.5)
    # 20 trees, keep_tree = FALSE, keep_history = TRUE.
    ctor <- function(seed)
        methods::new(BN, X, y, 20L, 2.0, 2.0, 0.95, 3.0, 100L, FALSE, FALSE,
                     seed, FALSE, TRUE)
    run <- function(...)
        ai4bayescode_run_chains(ctor, n_chains = 3, n_burn = 20, n_keep = 20,
                                verbose = FALSE, ...)
    # Chain 2 (seed 202) by hand: set.seed(202) immediately before the
    # constructor, 20 warmup sweeps, clear_history(), 20 kept sweeps.
    set.seed(202); m <- ctor(202L)
    m$step(20L); m$clear_history(); m$step(20L)
    h2 <- m$get_history()

    # Two runs in worker processes, after different caller seeds.
    set.seed(1); run_p1 <- run(n_cores = 2)
    withr::defer(ai4bayescode_close_models(run_p1))
    set.seed(2); run_p2 <- run(n_cores = 2)
    withr::defer(ai4bayescode_close_models(run_p2))
    expect_identical(run_p2$histories, run_p1$histories)
    expect_identical(run_p1$histories[[2]], h2)
    set.seed(3); run_m <- run(n_cores = 2, keep_models = FALSE)
    expect_identical(run_m$histories, run_p1$histories)

    set.seed(4); before <- caller_seed()
    run_s <- run(parallel = FALSE)
    expect_identical(caller_seed(), before)
    expect_identical(run_s$histories, run_p1$histories)
    for (i in 1:3)
        expect_identical(run_s$models[[i]]$get_current(),
                         run_p1$models[[i]]$get_current())
    expect_identical(predict(run_s), predict(run_p1))

    # The three chains draw from different streams.
    f <- lapply(run_p1$histories, `[[`, "f_bart")
    expect_false(identical(f[[1]], f[[2]]))
    expect_false(identical(f[[1]], f[[3]]))
    expect_false(identical(f[[2]], f[[3]]))

    # Destroying the last BART model writes R's generator state to
    # .Random.seed (Rcpp::RNGScope); after the run that state is the one
    # restored, so .Random.seed is still the one of set.seed(4).
    rm(run_s, m); invisible(gc())
    expect_identical(caller_seed(), before)

    # One chain with keep_models = FALSE and n_cores = 2 runs in this
    # session: chain 1 of the 3-chain runs (seed 101), .Random.seed restored.
    set.seed(5); before <- caller_seed()
    run_1 <- ai4bayescode_run_chains(ctor, n_chains = 1, n_burn = 20,
                                     n_keep = 20, n_cores = 2,
                                     keep_models = FALSE, verbose = FALSE)
    expect_identical(caller_seed(), before)
    expect_identical(run_1$histories, run_p1$histories[1])
    invisible(gc())
    expect_identical(caller_seed(), before)
})

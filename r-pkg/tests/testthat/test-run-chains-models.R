# ai4bayescode_run_chains(keep_models = TRUE): models kept after the run,
# worker-process handles, ai4bayescode_predict(), ai4bayescode_close_models().
# The compiled-model tests build GaussianLocationScale once (~30 s), so they
# are skipped on CRAN; the fork tests are skipped on Windows.

# GaussianLocationScale class generator, compiled on first use.
gls_class <- local({
    env <- NULL
    function() {
        if (is.null(env)) {
            e <- new.env()
            suppressMessages(ai4bayescode_example("GaussianLocationScale",
                                                  env = e, quiet = TRUE))
            env <<- e
        }
        env$GaussianLocationScale
    }
})

gls_ctor <- function() {
    GLS <- gls_class()
    y <- stats::rnorm(60, mean = 3, sd = 2)
    function(seed) methods::new(GLS, y, seed, TRUE)
}

# A model written in R: theta increases by 1 per sweep, starting at the seed.
# clear = TRUE adds clear_history(); extra = TRUE adds an entry that
# clear_history() does not empty.
fake_ctor <- function(clear = FALSE, extra = FALSE) {
    function(seed) {
        e <- new.env()
        e$state <- seed
        e$draws <- numeric(0)
        e$other <- numeric(0)
        e$step <- function(n = 1L) for (i in seq_len(n)) {
            e$state <- e$state + 1
            e$draws <- c(e$draws, e$state)
            e$other <- c(e$other, -e$state)
        }
        e$get_history <- function() {
            h <- list(theta = e$draws, pair = cbind(e$draws, 2 * e$draws))
            if (extra) h$other <- e$other
            h
        }
        e$predict_at <- function(newdata)
            list(det = cbind(e$draws, e$draws + 0.5))
        e$slow <- function(sec) { Sys.sleep(sec); "slow-result" }
        if (clear) e$clear_history <- function() {
            e$draws <- numeric(0)
            if (!extra) e$other <- numeric(0)
        }
        class(e) <- "FakeModel"
        e
    }
}

pool_of   <- function(run) .subset2(run$models[[1L]], "pool")
pids_of   <- function(run) pool_of(run)$pids
alive     <- function(pids) tools::pskill(pids, 0L)
# TRUE once none of `pids` exists (they are reaped after they exit).
all_exited <- function(pids, timeout = 10) {
    t0 <- Sys.time()
    while (any(alive(pids))) {
        if (difftime(Sys.time(), t0, units = "secs") > timeout) return(FALSE)
        Sys.sleep(0.05)
    }
    TRUE
}
n_conn <- function() nrow(showConnections(all = FALSE))

test_that("parallel keep_models run: handles, histories, predict_at", {
    skip_on_cran(); skip_on_os("windows"); skip_if_not_installed("withr")
    set.seed(1); ctor <- gls_ctor()
    run <- ai4bayescode_run_chains(ctor, n_chains = 2, n_burn = 50,
                                   n_keep = 100, n_cores = 2, verbose = FALSE)
    withr::defer(ai4bayescode_close_models(run))
    expect_s3_class(run, "ai4bayescode_run")
    expect_type(run, "list")
    expect_named(run, c("histories", "seeds", "wall", "models", "n_burn",
                        "n_keep", "history_cleared"))
    expect_identical(run$history_cleared, c(TRUE, TRUE))
    expect_length(run$models, 2L)
    for (i in 1:2) {
        h <- run$models[[i]]
        expect_s3_class(h, "ai4bayescode_chain_model")
        expect_true(is.environment(h))
        hist <- h$get_history()
        expect_identical(hist, run$histories[[i]])
        expect_identical(nrow(hist$mu), 100L)
        expect_identical(nrow(h$predict_at(list())$y_rep), 100L)
    }
    # The two chains are in two different worker processes.
    expect_length(unique(pids_of(run)), 2L)
    expect_false(any(pids_of(run) == Sys.getpid()))

    run_f <- ai4bayescode_run_chains(ctor, n_chains = 2, n_burn = 50,
                                     n_keep = 100, n_cores = 2,
                                     keep_models = FALSE, verbose = FALSE)
    run_s <- ai4bayescode_run_chains(ctor, n_chains = 2, n_burn = 50,
                                     n_keep = 100, parallel = FALSE,
                                     verbose = FALSE)
    expect_null(run_f$models)
    expect_false(is.object(run_f))
    expect_named(run_f, c("histories", "seeds", "wall", "n_burn", "n_keep",
                          "history_cleared"))
    expect_identical(run_f$histories, run$histories)
    expect_identical(run_s$histories, run$histories)
    expect_identical(run_s$seeds, run$seeds)

    # Handle methods: completion names, no assignment, diagnose accepts it.
    expect_true("get_history( " %in% utils::.DollarNames(run$models[[1]], "get"))
    expect_error(run$models[[1]]$x <- 1, "not supported")
    skip_if_not_installed("posterior")
    d <- ai4bayescode_diagnose(run$models[[2]], plot = FALSE)
    expect_identical(nrow(d$summary), 2L)
})

test_that("predict pools the chains and equals the per-chain predict_at", {
    skip_on_cran(); skip_on_os("windows"); skip_if_not_installed("withr")
    set.seed(2); ctor <- gls_ctor()
    run   <- ai4bayescode_run_chains(ctor, n_chains = 2, n_burn = 50,
                                     n_keep = 100, n_cores = 2, verbose = FALSE)
    withr::defer(ai4bayescode_close_models(run))
    run_s <- ai4bayescode_run_chains(ctor, n_chains = 2, n_burn = 50,
                                     n_keep = 100, parallel = FALSE,
                                     verbose = FALSE)
    pooled <- predict(run)
    expect_named(pooled, "y_rep")
    expect_identical(dim(pooled$y_rep), c(200L, 60L))
    # The first predict_at() call of an identically seeded model gives the
    # same draws, so the pooled result equals rbind of the per-chain calls.
    ref <- do.call(rbind, lapply(run_s$models,
                                 function(m) m$predict_at(list())$y_rep))
    expect_identical(pooled$y_rep, ref)

    per <- ai4bayescode_predict(run, pool = FALSE)
    expect_length(per, 2L)
    expect_identical(dim(per[[2]]$y_rep), c(100L, 60L))
    expect_identical(dim(predict(run_s, list())$y_rep), c(200L, 60L))

    # Deterministic key: pooled == rbind of each chain's own predict_at.
    run_d <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 3,
                                     n_burn = 4, n_keep = 6, n_cores = 2,
                                     verbose = FALSE)
    withr::defer(ai4bayescode_close_models(run_d))
    by_chain <- lapply(run_d$models, function(h) h$predict_at(list())$det)
    expect_identical(predict(run_d)$det, do.call(rbind, by_chain))
    expect_identical(dim(predict(run_d)$det), c(18L, 2L))
    expect_identical(ai4bayescode_predict(run_d, pool = FALSE)[[3]]$det,
                     by_chain[[3]])

    run_n <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                     n_burn = 1, n_keep = 2, n_cores = 2,
                                     keep_models = FALSE, verbose = FALSE)
    expect_error(ai4bayescode_predict(run_n), "keep_models = TRUE")
})

test_that("calls through a handle change the model in its worker", {
    skip_on_cran(); skip_on_os("windows"); skip_if_not_installed("withr")
    set.seed(3); ctor <- gls_ctor()
    run <- ai4bayescode_run_chains(ctor, n_chains = 2, n_burn = 20,
                                   n_keep = 30, n_cores = 2, verbose = FALSE)
    withr::defer(ai4bayescode_close_models(run))
    h <- run$models[[2]]
    expect_invisible(h$step(5L))
    expect_identical(nrow(h$get_history()$mu), 35L)
    expect_identical(nrow(run$models[[1]]$get_history()$mu), 30L)
    h$set_current(list(mu = 0.25))
    expect_equal(h$get_current()$mu[1], 0.25)
    # An error in the worker is raised here with the worker's message.
    err <- tryCatch(h$set_current(list(sigma = -1)), error = identity)
    expect_s3_class(err, "error")
    expect_match(conditionMessage(err), "sigma must be strictly positive")
    expect_match(conditionMessage(err), "chain 2")
    expect_output(print(h), "worker process")
})

test_that("ai4bayescode_close_models terminates the workers, idempotently", {
    skip_on_cran(); skip_on_os("windows")
    run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 3,
                                   n_burn = 2, n_keep = 3, n_cores = 2,
                                   verbose = FALSE)
    pids <- pids_of(run)
    expect_length(pids, 2L)
    expect_true(all(alive(pids)))
    expect_output(print(run), "worker process")
    expect_null(ai4bayescode_close_models(run))
    expect_true(all_exited(pids))
    msg <- "the worker processes holding the models of this run were closed (ai4bayescode_close_models)"
    expect_error(run$models[[1]]$get_history(), msg, fixed = TRUE)
    expect_error(predict(run), msg, fixed = TRUE)
    expect_silent(ai4bayescode_close_models(run))
    expect_output(print(run), "closed (ai4bayescode_close_models)", fixed = TRUE)
    expect_output(print(run$models[[3]]), "were closed")
    # The histories are still usable.
    expect_identical(run$histories[[3]]$theta, 303 + 2 + 1:3)
})

test_that("a pool is closed when its run is garbage-collected", {
    skip_on_cran(); skip_on_os("windows")
    run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                   n_burn = 1, n_keep = 2, n_cores = 2,
                                   verbose = FALSE)
    pids <- pids_of(run)
    conn <- n_conn()
    rm(run); invisible(gc())
    expect_true(all_exited(pids))
    expect_identical(n_conn(), conn - 2L)
})

test_that("a sequential run returns the model objects themselves", {
    skip_on_cran()
    set.seed(5); ctor <- gls_ctor()
    run_s <- ai4bayescode_run_chains(ctor, n_chains = 2, n_burn = 40,
                                     n_keep = 60, parallel = FALSE,
                                     verbose = FALSE)
    run_1 <- ai4bayescode_run_chains(ctor, n_chains = 2, n_burn = 40,
                                     n_keep = 60, n_cores = 1,
                                     verbose = FALSE)
    for (run in list(run_s, run_1)) {
        expect_false(any(vapply(run$models, inherits, logical(1),
                                "ai4bayescode_chain_model")))
        expect_true(methods::is(run$models[[1]], "Rcpp_GaussianLocationScale"))
        expect_identical(run$models[[2]]$get_history(), run$histories[[2]])
        expect_identical(nrow(run$models[[1]]$predict_at(list())$y_rep), 60L)
        expect_null(ai4bayescode_close_models(run))
        expect_output(print(run), "in this R session")
    }
    expect_identical(run_1$histories, run_s$histories)
    run_one <- ai4bayescode_run_chains(ctor, n_chains = 1, n_burn = 40,
                                       n_keep = 60, n_cores = 2,
                                       verbose = FALSE)
    expect_true(methods::is(run_one$models[[1]], "Rcpp_GaussianLocationScale"))
    expect_identical(run_one$histories[[1]], run_s$histories[[1]])
})

test_that("a model without clear_history: stripped histories, one message", {
    skip_on_os("windows")
    for (par in c(FALSE, TRUE)) {
        msgs <- character(0)
        run <- withCallingHandlers(
            ai4bayescode_run_chains(fake_ctor(), n_chains = 3, n_burn = 10,
                                    n_keep = 5, parallel = par, n_cores = 2,
                                    verbose = FALSE),
            message = function(m) {
                msgs <<- c(msgs, conditionMessage(m))
                invokeRestart("muffleMessage")
            })
        expect_length(msgs, 1L)
        expect_match(msgs, "FakeModel has no clear_history", fixed = TRUE)
        expect_match(msgs, "AI4BayesCode >= 1.1.3", fixed = TRUE)
        expect_identical(run$history_cleared, rep(FALSE, 3))
        expect_identical(run$histories[[2]]$theta, 202 + 10 + 1:5)
        expect_identical(dim(run$histories[[2]]$pair), c(5L, 2L))
        # The model's own history has the warmup draws; predict drops them.
        expect_length(run$models[[1]]$get_history()$theta, 15L)
        expect_identical(predict(run)$det[, 1],
                         c(101, 202, 303)[rep(1:3, each = 5)] + 10 + 1:5)
        ai4bayescode_close_models(run)
    }
    # No warmup, nothing to report.
    expect_silent(run0 <- ai4bayescode_run_chains(fake_ctor(), n_chains = 2,
                                                  n_burn = 0, n_keep = 3,
                                                  parallel = FALSE,
                                                  verbose = FALSE))
    expect_identical(run0$histories[[1]]$theta, 101 + 1:3)
})

test_that("an entry clear_history() leaves non-empty is stripped, with a warning", {
    skip_on_os("windows")
    for (par in c(FALSE, TRUE)) {
        expect_warning(
            run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE, extra = TRUE),
                                           n_chains = 2, n_burn = 4, n_keep = 3,
                                           parallel = par, n_cores = 2,
                                           verbose = FALSE),
            "entries other of FakeModel have more than n_keep = 3 rows")
        expect_identical(run$history_cleared, c(TRUE, TRUE))
        expect_identical(run$histories[[1]]$theta, 101 + 4 + 1:3)
        expect_identical(run$histories[[1]]$other, -(101 + 4 + 1:3))
        expect_length(run$models[[2]]$get_history()$other, 7L)
        ai4bayescode_close_models(run)
    }
})

test_that("mc.cores is an alias of n_cores", {
    skip_on_os("windows")
    run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                   n_burn = 1, n_keep = 2, mc.cores = 2,
                                   verbose = FALSE)
    expect_s3_class(run$models[[1]], "ai4bayescode_chain_model")
    expect_length(pids_of(run), 2L)
    ai4bayescode_close_models(run)
    run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                   n_burn = 1, n_keep = 2, n_cores = 2,
                                   mc.cores = 2, verbose = FALSE)
    expect_length(pids_of(run), 2L)
    ai4bayescode_close_models(run)
    expect_error(
        ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                n_burn = 1, n_keep = 2, n_cores = 2,
                                mc.cores = 3, verbose = FALSE),
        "n_cores = 2 and mc.cores = 3 differ")
})

test_that("a handle of another session errors without using a connection", {
    skip_on_cran(); skip_on_os("windows"); skip_if_not_installed("withr")
    run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                   n_burn = 1, n_keep = 2, n_cores = 2,
                                   verbose = FALSE)
    withr::defer(ai4bayescode_close_models(run))
    entry <- get(pool_of(run)$token,
                 envir = AI4BayesCode:::.ai4b_rc_registry())
    # A copy made in this session still reaches the same workers.
    same <- unserialize(serialize(run, NULL))
    expect_identical(same$models[[2]]$get_history(), run$histories[[2]])
    # A different session id: an error, and no request is sent.
    other <- unserialize(serialize(run, NULL))
    other_pool <- pool_of(other)
    other_pool$session <- "pid|another tempdir|time"
    n_req <- entry$seq
    expect_error(other$models[[1]]$get_history(), "another R session")
    expect_error(predict(other), "another R session")
    expect_identical(entry$seq, n_req)
    expect_null(ai4bayescode_close_models(other))     # a no-op
    expect_output(print(other), "another R session")
    expect_identical(run$models[[1]]$get_history(), run$histories[[1]])

    # The real case: readRDS() in a new R process.
    rds    <- tempfile(fileext = ".rds")
    out    <- tempfile(fileext = ".txt")
    script <- tempfile(fileext = ".R")
    saveRDS(run, rds)
    writeLines(c(
        sprintf(".libPaths(%s)", paste(deparse(.libPaths()), collapse = "")),
        "suppressPackageStartupMessages(library(AI4BayesCode))",
        sprintf("run <- readRDS(%s)", deparse(rds)),
        "n0  <- nrow(showConnections())",
        "msg <- tryCatch(run$models[[1]]$get_history(), error = conditionMessage)",
        sprintf("writeLines(c(msg, nrow(showConnections()) - n0), %s)",
                deparse(out))), script)
    rscript <- file.path(R.home("bin"), "Rscript")
    status <- system2(rscript, c("--vanilla", shQuote(script)),
                      stdout = FALSE, stderr = FALSE)
    expect_identical(status, 0L)
    res <- readLines(out)
    expect_match(res[1], "created in another R session", fixed = TRUE)
    expect_identical(res[2], "0")
    expect_identical(run$models[[2]]$get_history(), run$histories[[2]])
})

test_that("an R error in a chain stops the run at once, naming the chain", {
    skip_on_os("windows")
    parent <- Sys.getpid()
    log <- tempfile()
    # The constructor records the pid of every process it runs in, as the
    # name of a file in `log`.
    bad <- function(seed) {
        file.create(file.path(log, Sys.getpid()))
        stop("boom in the constructor")
    }
    conn  <- n_conn()
    pools <- ls(AI4BayesCode:::.ai4b_rc_registry())
    for (keep in c(TRUE, FALSE)) {
        unlink(log, recursive = TRUE)
        dir.create(log)
        expect_no_warning(
            expect_error(ai4bayescode_run_chains(bad, n_chains = 2, n_burn = 1,
                                                 n_keep = 1, n_cores = 2,
                                                 keep_models = keep,
                                                 verbose = FALSE),
                         "chain 1 (seed 101): boom in the constructor",
                         fixed = TRUE))
        # No sequential re-run in this session.
        ran_in <- as.integer(list.files(log))
        expect_length(ran_in, 2L)
        expect_false(parent %in% ran_in)
        expect_identical(n_conn(), conn)
        expect_identical(ls(AI4BayesCode:::.ai4b_rc_registry()), pools)
    }
    expect_error(ai4bayescode_run_chains(bad, n_chains = 2, n_burn = 1,
                                         n_keep = 1, parallel = FALSE,
                                         verbose = FALSE),
                 "chain 1 (seed 101): boom in the constructor", fixed = TRUE)
    # A chain that fails in one worker only, in a method of the model.
    half <- function(seed) {
        m <- fake_ctor(clear = TRUE)(seed)
        if (seed == 202) m$step <- function(n = 1L) stop("chain two fails")
        m
    }
    for (keep in c(TRUE, FALSE))
        expect_no_warning(
            expect_error(ai4bayescode_run_chains(half, n_chains = 3, n_burn = 1,
                                                 n_keep = 1, n_cores = 2,
                                                 keep_models = keep,
                                                 verbose = FALSE),
                         "chain 2 (seed 202): chain two fails", fixed = TRUE))
    err <- tryCatch(ai4bayescode_run_chains(half, n_chains = 3, n_burn = 1,
                                            n_keep = 1, parallel = FALSE,
                                            verbose = FALSE),
                    error = identity)
    expect_s3_class(err, "simpleError")
    expect_identical(conditionMessage(err),
                     "ai4bayescode_run_chains: chain 2 (seed 202): chain two fails")
})

test_that("print(run) is compact", {
    skip_on_os("windows")
    run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                   n_burn = 5, n_keep = 400, n_cores = 2,
                                   verbose = FALSE)
    out <- capture.output(print(run))
    expect_lte(length(out), 8L)
    expect_match(out[1], "2 chains, n_burn = 5 warmup + n_keep = 400", fixed = TRUE)
    expect_true(any(grepl("pair +400 x 2", out)))
    expect_true(any(grepl("Models: in 2 worker processes", out, fixed = TRUE)))
    capture.output(vis <- withVisible(print(run)))
    expect_false(vis$visible)
    expect_identical(vis$value, run)
    ai4bayescode_close_models(run)
    # keep_models = FALSE gives the plain list of earlier versions.
    run_f <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                     n_burn = 5, n_keep = 400,
                                     keep_models = FALSE, parallel = FALSE,
                                     verbose = FALSE)
    expect_false(inherits(run_f, "ai4bayescode_run"))
})

test_that("an interrupted call does not shift the results of later calls", {
    skip_on_cran(); skip_on_os("windows"); skip_if_not_installed("withr")
    run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                   n_burn = 3, n_keep = 4, n_cores = 2,
                                   verbose = FALSE)
    withr::defer(ai4bayescode_close_models(run))
    parent <- Sys.getpid()
    # Interrupt this process 0.3 s into a 2 s call; the worker finishes the
    # call and sends its result, which the next call has to discard.
    r <- tryCatch({
        job <- parallel::mcparallel({
            Sys.sleep(0.3)
            tools::pskill(parent, tools::SIGINT)
        })
        run$models[[1]]$slow(2)
    }, interrupt = function(e) "interrupted")
    tryCatch(parallel::mccollect(job), interrupt = function(e) NULL)
    expect_identical(r, "interrupted")
    expect_identical(run$models[[1]]$get_history(), run$histories[[1]])
    expect_identical(run$models[[1]]$slow(0), "slow-result")
    expect_identical(predict(run)$det[, 1], c(101 + 3 + 1:4, 202 + 3 + 1:4))
})

test_that("keep_models = FALSE returns the 1.1.2 list, with positional arguments", {
    skip_on_os("windows")
    ctor <- fake_ctor()           # no clear_history(), as with 1.1.2 headers
    # 1.1.2 call: model_ctor, n_chains, n_burn, n_keep, seeds, parallel,
    # mc.cores, verbose.
    expect_silent(run <- ai4bayescode_run_chains(ctor, 2, 3, 4, NULL, TRUE, 2,
                                                 FALSE, keep_models = FALSE))
    expect_false(is.object(run))
    expect_named(run, c("histories", "seeds", "wall", "n_burn", "n_keep",
                        "history_cleared"))
    expect_identical(run[[1]], run$histories)
    expect_identical(run[[2]], c(101L, 202L))
    expect_identical(run[[3]], run$wall)
    expect_identical(run$histories[[2]]$theta, 202 + 3 + 1:4)
    expect_null(run$models)
    skip_if_not_installed("jsonlite")
    js <- jsonlite::toJSON(run)
    expect_s3_class(js, "json")
    expect_identical(jsonlite::fromJSON(js)$seeds, c(101L, 202L))
})

# A model whose predict_at() returns the shape named in newdata$mode, with
# n_keep draws per chain; `chain_value` is seed / 101.
shape_ctor <- function(seed) {
    m <- fake_ctor(clear = TRUE)(seed)
    m$predict_at <- function(newdata) {
        d <- m$draws
        n <- length(d)
        odd <- (seed / 101) %% 2 == 1
        switch(newdata$mode,
            vec    = list(v = d),
            scalar = list(s = d[1L]),
            arr1   = list(a = array(d)),
            mat    = list(m = cbind(d, -d)),
            arr3   = list(a = array(seq_len(n * 6) + seed, c(n, 2, 3))),
            mixed  = list(x = if (odd) d else matrix(d, ncol = 1L)),
            cols   = list(x = matrix(d, n, if (odd) 2L else 3L)),
            nullkey = list(a = d, b = NULL),
            listkey = list(a = list(1, 2)),
            unnamed = list(d),
            dup    = list(a = d, a = -d),
            bare   = cbind(d, d),
            empty  = list())
    }
    m
}

test_that("predict(): newdata NULL is list(); other non-lists raise an error", {
    skip_on_os("windows")
    for (par in c(FALSE, TRUE)) {
        run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                       n_burn = 1, n_keep = 3, parallel = par,
                                       n_cores = 2, verbose = FALSE)
        expect_identical(predict(run, NULL), predict(run))
        expect_identical(ai4bayescode_predict(run, NULL), predict(run, list()))
        X <- matrix(1, 2, 2)
        expect_error(predict(run, X), "predict(run, list(X = X_new))",
                     fixed = TRUE)
        expect_error(predict(run, X), "got a matrix", fixed = TRUE)
        expect_error(predict(run, data.frame(a = 1:2)), "got a data.frame",
                     fixed = TRUE)
        expect_error(predict(run, list(1, 2)), "unnamed element", fixed = TRUE)
        expect_error(predict(run, 1:3), "got an integer", fixed = TRUE)
        ai4bayescode_close_models(run)
    }
})

test_that("predict(): pooled entries keep their number of dimensions", {
    skip_on_os("windows")
    for (par in c(FALSE, TRUE)) {
        run <- ai4bayescode_run_chains(shape_ctor, n_chains = 3, n_burn = 2,
                                       n_keep = 4, parallel = par, n_cores = 2,
                                       verbose = FALSE)
        d <- lapply(c(101, 202, 303), function(s) s + 2 + 1:4)
        p <- function(mode, ...) predict(run, list(mode = mode), ...)
        expect_identical(p("vec")$v, unlist(d))
        expect_identical(p("scalar")$s, vapply(d, `[`, numeric(1), 1L))
        expect_identical(p("arr1")$a, unlist(d))
        expect_identical(p("mat")$m, do.call(rbind, lapply(d, function(d) cbind(d, -d))))
        a3 <- p("arr3")$a
        expect_identical(dim(a3), c(12L, 2L, 3L))
        by_chain <- ai4bayescode_predict(run, list(mode = "arr3"), pool = FALSE)
        for (i in 1:3)
            expect_identical(a3[(i - 1) * 4 + 1:4, , , drop = FALSE],
                             by_chain[[i]]$a)
        expect_identical(p("empty"), list())
        expect_error(p("mixed"), "'x' has 1 dimension in chain 1 and 2 in chain 2",
                     fixed = TRUE)
        expect_error(p("cols"), "'x' has 2 columns in chain 1 and 3 in chain 2",
                     fixed = TRUE)
        expect_error(p("nullkey"), "'b' of chain 1 is NULL", fixed = TRUE)
        expect_error(p("listkey"), "'a' of chain 1 is a list", fixed = TRUE)
        # The value of predict_at() must be a list with unique non-empty
        # names, also with pool = FALSE.
        for (pl in c(TRUE, FALSE)) {
            expect_error(p("unnamed", pool = pl),
                         "predict_at() of chain 1 returned a list without unique non-empty names",
                         fixed = TRUE)
            expect_error(p("dup", pool = pl), "of chain 1 returned a list without",
                         fixed = TRUE)
            expect_error(p("bare", pool = pl),
                         "predict_at() of chain 1 returned a matrix", fixed = TRUE)
        }
        ai4bayescode_close_models(run)
    }
})

test_that("predict(run, ...) raises an error for an unused argument", {
    skip_on_os("windows")
    run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                   n_burn = 1, n_keep = 2, parallel = FALSE,
                                   verbose = FALSE)
    expect_error(predict(run, nwdata = list(x = 1)), "unused argument 'nwdata'",
                 fixed = TRUE)
    expect_error(predict(run, list(), pooled = FALSE),
                 "unused argument 'pooled'", fixed = TRUE)
    expect_error(predict(run, list(), TRUE, 3, z = 1),
                 "unused arguments '(unnamed)', 'z'", fixed = TRUE)
    # Positional, exact and partial names are accepted.
    expect_identical(predict(run, list(), FALSE), predict(run, pool = FALSE))
    expect_identical(predict(run, newdat = list()), predict(run))
})

test_that("a lost worker process ends only the models of its own chains", {
    skip_on_cran(); skip_on_os("windows")
    run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 4,
                                   n_burn = 1, n_keep = 2, n_cores = 2,
                                   verbose = FALSE)
    pids <- pids_of(run)
    tools::pskill(pids[1L], tools::SIGKILL)       # holds chains 1 and 3
    expect_true(all_exited(pids[1L]))
    lost <- sprintf("the worker process %d holding chains 1, 3 exited unexpectedly",
                    pids[1L])
    expect_error(run$models[[1]]$get_history(), paste("chain 1:", lost),
                 fixed = TRUE)
    expect_error(run$models[[3]]$get_history(), "the models of these chains are lost",
                 fixed = TRUE)
    expect_error(predict(run), lost, fixed = TRUE)
    # The other worker process keeps its models.
    expect_identical(run$models[[2]]$get_history(), run$histories[[2]])
    expect_identical(run$models[[4]]$get_history(), run$histories[[4]])
    expect_true(alive(pids[2L]))
    expect_output(print(run$models[[3]]), sub("^the", "The", lost), fixed = TRUE)
    expect_output(print(run$models[[2]]), "is in worker process", fixed = TRUE)
    out <- capture.output(print(run))
    expect_true(any(grepl(sprintf("in 1 worker process of this R session (pid %d)",
                                  pids[2L]), out, fixed = TRUE)))
    expect_true(any(grepl(lost, out, fixed = TRUE)))
    ai4bayescode_close_models(run)
    expect_true(all_exited(pids[2L]))
})

test_that("a worker process that exits during the run: sequential re-run", {
    skip_on_cran(); skip_on_os("windows")
    parent <- Sys.getpid()
    # The constructor ends its own process (SIGKILL) when it runs in a
    # forked worker process.
    dies <- function(seed) {
        if (Sys.getpid() != parent) tools::pskill(Sys.getpid(), tools::SIGKILL)
        fake_ctor(clear = TRUE)(seed)
    }
    for (keep in c(TRUE, FALSE)) {
        # keep_models = FALSE: parallel::mclapply() warns as well.
        w <- character(0)
        run <- withCallingHandlers(
            ai4bayescode_run_chains(dies, n_chains = 2, n_burn = 1, n_keep = 2,
                                    n_cores = 2, keep_models = keep,
                                    verbose = FALSE),
            warning = function(x) {
                w <<- c(w, conditionMessage(x))
                invokeRestart("muffleWarning")
            })
        expect_true(any(grepl("fork-unsafe multithreaded BLAS", w, fixed = TRUE)))
        expect_identical(run$histories[[2]]$theta, 202 + 1 + 1:2)
        if (keep) expect_false(inherits(run$models[[1]], "ai4bayescode_chain_model"))
    }
})

test_that("a pid that no longer designates a worker is not signalled", {
    skip_on_cran(); skip_on_os("windows")
    run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                   n_burn = 1, n_keep = 2, n_cores = 2,
                                   verbose = FALSE)
    pids  <- pids_of(run)
    entry <- get(pool_of(run)$token, envir = AI4BayesCode:::.ai4b_rc_registry())
    # A stand-in process started later than the worker takes the pid of
    # worker 2 in the registry, as a pid reused by the system would.
    Sys.sleep(1.2)
    standin <- parallel::mcparallel(Sys.sleep(60), mc.set.seed = FALSE)
    on.exit({
        tools::pskill(standin$pid, tools::SIGKILL)
        suppressWarnings(parallel::mccollect(standin))
    }, add = TRUE)
    tools::pskill(pids[2L], tools::SIGKILL)
    expect_true(all_exited(pids[2L]))
    entry$pids[2L] <- standin$pid
    expect_error(run$models[[2]]$get_history(), "exited unexpectedly")
    ai4bayescode_close_models(run)
    expect_true(all_exited(pids[1L]))
    Sys.sleep(0.3)
    expect_true(alive(standin$pid))

    # A busy worker (an interrupted call) is sent SIGTERM at close, also only
    # while its pid designates it.
    run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                   n_burn = 1, n_keep = 2, n_cores = 2,
                                   verbose = FALSE)
    pids  <- pids_of(run)
    entry <- get(pool_of(run)$token, envir = AI4BayesCode:::.ai4b_rc_registry())
    parent <- Sys.getpid()
    job <- parallel::mcparallel({ Sys.sleep(0.3); tools::pskill(parent, tools::SIGINT) })
    done <- FALSE
    tryCatch({ run$models[[2]]$slow(30); done <- TRUE; Sys.sleep(30) },
             interrupt = function(e) NULL)
    suppressWarnings(parallel::mccollect(job))
    expect_false(done)
    expect_true(entry$sent[2L] > entry$recv[2L])
    Sys.sleep(1.2)
    standin2 <- parallel::mcparallel(Sys.sleep(60), mc.set.seed = FALSE)
    on.exit({
        tools::pskill(standin2$pid, tools::SIGKILL)
        suppressWarnings(parallel::mccollect(standin2))
    }, add = TRUE)
    tools::pskill(pids[2L], tools::SIGKILL)
    expect_true(all_exited(pids[2L]))
    entry$pids[2L] <- standin2$pid
    ai4bayescode_close_models(run)
    expect_true(all_exited(pids[1L]))
    Sys.sleep(0.3)
    expect_true(alive(standin2$pid))
})

test_that("the reason a pool was closed is shown by every copy of the run", {
    skip_on_cran(); skip_on_os("windows")
    run <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                   n_burn = 1, n_keep = 2, n_cores = 2,
                                   verbose = FALSE)
    pids <- pids_of(run)
    copy <- unserialize(serialize(run, NULL))
    expect_identical(copy$models[[2]]$get_history(), run$histories[[2]])
    rm(run); invisible(gc())
    expect_true(all_exited(pids))
    expect_error(copy$models[[1]]$get_history(), "were closed when the run and its handles were garbage-collected",
                 fixed = TRUE)
    expect_output(print(copy), "closed (the run that started them was garbage-collected)",
                  fixed = TRUE)
    expect_output(print(copy$models[[2]]), "garbage-collected")
    expect_false(any(grepl("ai4bayescode_close_models)", capture.output(print(copy)),
                           fixed = TRUE)))
})

test_that("a handle called in model_ctor or a forked child: a specific error", {
    skip_on_cran(); skip_on_os("windows")
    pilot <- ai4bayescode_run_chains(fake_ctor(clear = TRUE), n_chains = 2,
                                     n_burn = 1, n_keep = 2, n_cores = 2,
                                     verbose = FALSE)
    on.exit(ai4bayescode_close_models(pilot), add = TRUE)
    warm <- function(seed) {
        m <- fake_ctor(clear = TRUE)(seed)
        m$state <- tail(pilot$models[[match(seed, pilot$seeds)]]$get_history()$theta, 1)
        m
    }
    expect_no_warning(expect_error(
        ai4bayescode_run_chains(warm, n_chains = 2, n_burn = 1, n_keep = 2,
                                n_cores = 2, verbose = FALSE),
        "chain 1 (seed 101): an ai4bayescode_chain_model handle can be called only in the R process",
        fixed = TRUE))
    res <- parallel::mclapply(pilot$models, function(h)
        tryCatch(h$get_history(), error = conditionMessage), mc.cores = 2)
    expect_match(res[[1]], "not in a forked child process", fixed = TRUE)
    # Reading the values first runs in parallel.
    last <- lapply(pilot$models, function(h) tail(h$get_history()$theta, 1))
    warm2 <- function(seed) {
        m <- fake_ctor(clear = TRUE)(seed)
        m$state <- last[[match(seed, pilot$seeds)]]
        m
    }
    run <- ai4bayescode_run_chains(warm2, n_chains = 2, n_burn = 1, n_keep = 2,
                                   n_cores = 2, verbose = FALSE)
    expect_s3_class(run$models[[1]], "ai4bayescode_chain_model")
    expect_identical(run$histories[[2]]$theta, 205 + 1 + 1:2)
    ai4bayescode_close_models(run)
    # The pilot's handles still work here.
    expect_identical(pilot$models[[1]]$get_history(), pilot$histories[[1]])
})

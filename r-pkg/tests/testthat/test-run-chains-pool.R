# The worker processes of kept parallel runs: the bound max_workers and the
# closing of the least recently used runs, the free-connection check,
# interrupts during calls (SIGINT sent from a forked child process), and the
# checkout copy R/AI4BayesCode_helpers.R. The fork tests are skipped on
# Windows and the long ones on CRAN.

# A model written in R: theta increases by 1 per sweep, starting at the seed.
pool_ctor <- function(seed) {
    e <- new.env()
    e$state <- seed
    e$draws <- numeric(0)
    e$step <- function(n = 1L) for (i in seq_len(n)) {
        e$state <- e$state + 1
        e$draws <- c(e$draws, e$state)
    }
    e$get_history <- function() list(theta = e$draws)
    e$clear_history <- function() e$draws <- numeric(0)
    # newdata$sleep: seconds to sleep first; newdata$n: length of `out`;
    # newdata$payload: any value, whose length is returned.
    e$predict_at <- function(newdata) {
        if (!is.null(newdata$sleep)) Sys.sleep(newdata$sleep)
        n <- if (is.null(newdata$n)) 2 else newdata$n
        list(out = rep(e$state, n), len = length(newdata$payload))
    }
    e$slow <- function(sec) { Sys.sleep(sec); "slow-result" }
    e$busy <- function(sec) {
        t0 <- Sys.time()
        while (as.numeric(Sys.time() - t0, units = "secs") < sec) NULL
        "busy-result"
    }
    e$pid <- function() Sys.getpid()
    e
}

kept_run <- function(n_workers = 2L, n_chains = n_workers, ...)
    ai4bayescode_run_chains(pool_ctor, n_chains = n_chains, n_burn = 1,
                            n_keep = 2, n_cores = n_workers, verbose = FALSE, ...)

workers_of <- function(run) .subset2(run$models[[1L]], "pool")$pids
is_kept    <- function(run) inherits(run$models[[1L]], "ai4bayescode_chain_model")
live_workers <- function() {
    reg <- AI4BayesCode:::.ai4b_rc_registry()
    sum(vapply(mget(ls(reg), envir = reg), function(e) sum(is.na(e$lost)),
               numeric(1)))
}
alive <- function(pids) tools::pskill(pids, 0L)
all_exited <- function(pids, timeout = 10) {
    t0 <- Sys.time()
    while (any(alive(pids))) {
        if (difftime(Sys.time(), t0, units = "secs") > timeout) return(FALSE)
        Sys.sleep(0.05)
    }
    TRUE
}
spill_files <- function() list.files(tempdir(), pattern = "^ai4b_rc_")

# SIGINT to process `pid` after `sec` seconds, from a forked child process.
sigint_after <- function(sec, pid = Sys.getpid()) {
    force(pid)
    parallel::mcparallel({ Sys.sleep(sec); tools::pskill(pid, tools::SIGINT) },
                         mc.set.seed = FALSE)
}
collect <- function(job) invisible(suppressWarnings(parallel::mccollect(job)))

# Evaluate `expr` and interrupt this process after `sec` seconds; the
# signal is received inside the tryCatch also when `expr` returns earlier.
# TRUE when `expr` returned, FALSE when it was interrupted.
returned_before <- function(expr, sec) {
    job <- sigint_after(sec)
    done <- FALSE
    tryCatch({ expr; done <- TRUE; Sys.sleep(sec + 30) },
             interrupt = function(e) NULL)
    collect(job)
    done
}

# The value of `expr`, or "HUNG" when it did not return within `limit`
# seconds (it is then interrupted).
within <- function(expr, limit) {
    job <- sigint_after(limit)
    on.exit({ tools::pskill(job$pid, tools::SIGKILL); collect(job) }, add = TRUE)
    tryCatch(list(expr), interrupt = function(e) "HUNG")[[1L]]
}

test_that("max_workers: the least recently used runs are closed, with one warning", {
    skip_on_cran(); skip_on_os("windows"); skip_if_not_installed("withr")
    withr::local_options(ai4bayescode.max_workers = 4)
    invisible(gc())
    a <- kept_run(); b <- kept_run()
    withr::defer({ ai4bayescode_close_models(a); ai4bayescode_close_models(b) })
    expect_identical(a$models[[2]]$pid(), workers_of(a)[2])   # a is used last
    pids_b <- workers_of(b)
    w <- character(0)
    cc <- withCallingHandlers(kept_run(), warning = function(x) {
        w <<- c(w, conditionMessage(x)); invokeRestart("muffleWarning") })
    withr::defer(ai4bayescode_close_models(cc))
    expect_length(w, 1L)
    expect_match(w, "the worker processes of 1 earlier run (2 processes) were closed",
                 fixed = TRUE)
    expect_match(w, "max_workers = 4", fixed = TRUE)
    expect_match(w, "ai4bayescode_close_models(run)", fixed = TRUE)
    expect_true(is_kept(cc))
    expect_true(all_exited(pids_b))
    expect_error(b$models[[1]]$pid(),
                 "were closed to start the worker processes of a later run",
                 fixed = TRUE)
    expect_output(print(b), "Models: closed to start the worker processes of a later run",
                  fixed = TRUE)
    expect_identical(a$models[[1]]$pid(), workers_of(a)[1])
    expect_identical(cc$models[[2]]$pid(), workers_of(cc)[2])
    expect_lte(live_workers(), 4)
    # A run with more worker processes than max_workers runs in this session.
    expect_warning(d <- kept_run(6L), "exceed max_workers = 4")
    expect_false(is_kept(d))
    expect_identical(d$models[[6]]$get_history(), d$histories[[6]])
    # The environment variable applies when the option is not set.
    withr::local_options(ai4bayescode.max_workers = NULL)
    withr::local_envvar(AI4BAYESCODE_MAX_WORKERS = "3")
    expect_warning(e <- kept_run(4L), "exceed max_workers = 3")
    expect_false(is_kept(e))
    withr::local_envvar(AI4BAYESCODE_MAX_WORKERS = "two")
    expect_error(kept_run(), "AI4BAYESCODE_MAX_WORKERS must be a positive whole number")
})

test_that("a run whose worker processes were all lost is not closed to make room", {
    skip_on_cran(); skip_on_os("windows"); skip_if_not_installed("withr")
    withr::local_options(ai4bayescode.max_workers = 4)
    invisible(gc())
    a <- kept_run()
    pids_a <- workers_of(a)
    tools::pskill(pids_a, tools::SIGKILL)
    expect_true(all_exited(pids_a))
    expect_error(predict(a), "exited unexpectedly", fixed = TRUE)
    b <- kept_run(); cc <- kept_run()
    withr::defer({ ai4bayescode_close_models(b); ai4bayescode_close_models(cc) })
    w <- character(0)
    d <- withCallingHandlers(kept_run(), warning = function(x) {
        w <<- c(w, conditionMessage(x)); invokeRestart("muffleWarning") })
    withr::defer(ai4bayescode_close_models(d))
    expect_length(w, 1L)
    expect_match(w, "the worker processes of 1 earlier run (2 processes) were closed",
                 fixed = TRUE)
    expect_match(w, "keep_models = FALSE", fixed = TRUE)
    expect_error(b$models[[1]]$pid(), "a later run", fixed = TRUE)
    expect_error(a$models[[2]]$pid(), "exited unexpectedly", fixed = TRUE)
    expect_identical(cc$models[[1]]$pid(), workers_of(cc)[1])
})

test_that("too few free connections: earlier runs are closed, or the run is sequential", {
    skip_on_cran(); skip_on_os("windows"); skip_if_not_installed("withr")
    withr::local_options(ai4bayescode.max_workers = 1000)
    invisible(gc())
    a <- kept_run()
    withr::defer(ai4bayescode_close_models(a))
    pids_a <- workers_of(a)
    # Fill the connection table up to 3 free connections.
    held <- list()
    withr::defer(for (cn in held) try(close(cn), silent = TRUE))
    while (AI4BayesCode:::.ai4b_rc_free_slots(4L) > 3L)
        held[[length(held) + 1L]] <- textConnection(NULL, "w")
    # 2 worker processes require 4: closing run a frees 2.
    expect_warning(b <- kept_run(), "R had fewer than 4 free connections")
    expect_true(is_kept(b))
    expect_true(all_exited(pids_a))
    expect_error(a$models[[1]]$pid(), "R had too few free connections", fixed = TRUE)
    expect_identical(b$models[[2]]$pid(), workers_of(b)[2])
    ai4bayescode_close_models(b)
    # Without kept runs to close, the chains run in this session.
    while (AI4BayesCode:::.ai4b_rc_free_slots(4L) > 2L)
        held[[length(held) + 1L]] <- textConnection(NULL, "w")
    expect_warning(cc <- kept_run(), "R has fewer than 4 free connections")
    expect_false(is_kept(cc))
    expect_identical(cc$histories[[2]]$theta, 202 + 1 + 1:2)
})

test_that("40 kept 4-worker runs without close stay parallel within max_workers", {
    skip_on_cran(); skip_on_os("windows")
    invisible(gc())
    max_w <- AI4BayesCode:::.ai4b_rc_max_workers()
    conn0 <- nrow(showConnections(all = TRUE))
    fits  <- list()
    n_warn <- 0L
    for (r in 1:40) {
        fits[[r]] <- withCallingHandlers(kept_run(4L),
            warning = function(w) {
                expect_match(conditionMessage(w), "least recently used first")
                n_warn <<- n_warn + 1L
                invokeRestart("muffleWarning")
            })
        expect_true(is_kept(fits[[r]]))
        expect_lte(live_workers(), max_w)
        expect_lt(nrow(showConnections(all = TRUE)), 128L)
    }
    expect_gt(n_warn, 0L)
    n_open <- max_w %/% 4L
    expect_identical(fits[[40]]$models[[4]]$pid(), workers_of(fits[[40]])[4])
    expect_identical(fits[[41 - n_open]]$models[[1]]$pid(),
                     workers_of(fits[[41 - n_open]])[1])
    expect_error(fits[[1]]$models[[1]]$pid(), "a later run", fixed = TRUE)
    pids <- unlist(lapply(fits[(41 - n_open):40], workers_of))
    for (f in fits) ai4bayescode_close_models(f)
    expect_true(all_exited(pids))
    expect_identical(live_workers(), 0)
    expect_identical(nrow(showConnections(all = TRUE)), conn0)
})

test_that("an interrupted call with a large reply, then a large request", {
    skip_on_cran(); skip_on_os("windows")
    run <- kept_run()
    on.exit(ai4bayescode_close_models(run), add = TRUE)
    # 8 MB reply per worker, interrupted while the workers compute it.
    expect_false(returned_before(predict(run, list(sleep = 1, n = 1e6)), 0.3))
    # 8 MB request: the replies of the interrupted call are read first.
    r <- within(suppressWarnings(predict(run, list(payload = stats::rnorm(1e6), n = 5))), 30)
    expect_false(identical(r, "HUNG"))
    expect_equal(r$len, c(1e6, 1e6))
    expect_identical(r$out, rep(c(104, 205), each = 5))
    expect_identical(within(predict(run)$out, 10), rep(c(104, 205), each = 2))
    expect_identical(run$models[[1]]$pid(), workers_of(run)[1])
    expect_identical(run$models[[2]]$pid(), workers_of(run)[2])
    expect_length(spill_files(), 0L)
})

test_that("SIGINT while a large request is written or a large reply is read", {
    skip_on_cran(); skip_on_os("windows")
    run <- kept_run()
    on.exit(ai4bayescode_close_models(run), add = TRUE)
    big <- stats::rnorm(2.5e7)        # 200 MB
    for (d in c(0.05, 0.15, 0.4)) {
        returned_before(predict(run, list(payload = big, n = 1)), d)
        expect_identical(within(predict(run)$out, 30), c(104, 104, 205, 205))
    }
    for (d in c(0.1, 0.3)) {
        returned_before(suppressWarnings(predict(run, list(n = 2.5e7))), d)
        expect_identical(within(predict(run)$out, 30), c(104, 104, 205, 205))
    }
    expect_identical(run$models[[2]]$get_history(), run$histories[[2]])
    expect_length(spill_files(), 0L)
})

test_that("an unread reply leaves no file after close or garbage collection", {
    skip_on_cran(); skip_on_os("windows")
    for (how in c("close", "gc")) {
        run  <- kept_run()
        pids <- workers_of(run)
        # 8 MB reply per worker, written to a file after the call was
        # interrupted.
        expect_false(returned_before(predict(run, list(sleep = 0.5, n = 1e6)), 0.2))
        t0 <- Sys.time()
        while (length(spill_files()) < 2L &&
               difftime(Sys.time(), t0, units = "secs") < 20) Sys.sleep(0.1)
        expect_length(spill_files(), 2L)
        if (how == "close") ai4bayescode_close_models(run)
        else { rm(run); invisible(gc()) }
        expect_true(all_exited(pids))
        expect_length(spill_files(), 0L)
    }
    # Closed while the worker processes compute or write their replies.
    for (d in c(0.3, 0.8)) {
        run  <- kept_run()
        pids <- workers_of(run)
        returned_before(suppressWarnings(predict(run, list(n = 2.5e7))), d)
        ai4bayescode_close_models(run)
        expect_true(all_exited(pids))
        expect_length(spill_files(), 0L)
    }
})

test_that("SIGINT to the worker processes only: every call gets a reply", {
    skip_on_cran(); skip_on_os("windows")
    run <- kept_run()
    on.exit(ai4bayescode_close_models(run), add = TRUE)
    pids <- workers_of(run)
    # In a select() (Sys.sleep) the interrupt ends the call with an error.
    jobs <- lapply(pids, function(p) sigint_after(0.4, p))
    r <- within(tryCatch(predict(run, list(sleep = 3)), error = conditionMessage), 20)
    lapply(jobs, collect)
    expect_identical(r, "chain 1: interrupted in the worker process (SIGINT)")
    # In R code the interrupt is held back and the call completes.
    job <- sigint_after(0.4, pids[1])
    expect_identical(within(run$models[[1]]$busy(1.5), 20), "busy-result")
    collect(job)
    expect_identical(within(run$models[[1]]$pid(), 10), pids[1])
    expect_identical(within(predict(run)$out, 10), c(104, 104, 205, 205))
    # SIGINT to the worker processes during the run itself: held back in
    # the R code of step(), and the run completes.
    busy_ctor <- function(seed) {
        m <- pool_ctor(seed)
        f <- m$step
        m$step <- function(n = 1L) {
            t0 <- Sys.time()
            while (as.numeric(Sys.time() - t0, units = "secs") < 0.004 * n) NULL
            f(n)
        }
        m
    }
    ref <- ai4bayescode_run_chains(busy_ctor, n_chains = 2, n_burn = 20,
                                   n_keep = 300, parallel = FALSE, verbose = FALSE)
    sigint_children <- function(sec) {
        parent <- Sys.getpid()
        parallel::mcparallel({
            Sys.sleep(sec)
            kids <- suppressWarnings(system2("pgrep", c("-P", parent), stdout = TRUE))
            for (k in kids) tools::pskill(as.integer(k), tools::SIGINT)
        }, mc.set.seed = FALSE)
    }
    job  <- sigint_children(0.6)
    run2 <- within(ai4bayescode_run_chains(busy_ctor, n_chains = 2, n_burn = 20,
                                           n_keep = 300, n_cores = 2,
                                           verbose = FALSE), 60)
    collect(job)
    expect_false(identical(run2, "HUNG"))
    expect_identical(run2$histories, ref$histories)
    if (is.list(run2)) ai4bayescode_close_models(run2)
    # In Sys.sleep() (a select()) the interrupt ends the chain with an error,
    # which stops the run.
    sleep_ctor <- function(seed) {
        m <- pool_ctor(seed)
        f <- m$step
        m$step <- function(n = 1L) { Sys.sleep(0.004 * n); f(n) }
        m
    }
    job <- sigint_children(0.6)
    r <- within(tryCatch(ai4bayescode_run_chains(sleep_ctor, n_chains = 2,
                                                 n_burn = 20, n_keep = 300,
                                                 n_cores = 2, verbose = FALSE),
                         error = conditionMessage), 60)
    collect(job)
    expect_match(r, "\\(seed [0-9]+\\): interrupted in the worker process \\(SIGINT\\)")
})

# The checkout copy of the helpers: the path from r-pkg/tests/testthat.
helpers_file <- function() {
    f <- normalizePath(testthat::test_path("..", "..", "..", "R",
                                           "AI4BayesCode_helpers.R"),
                       mustWork = FALSE)
    if (!file.exists(f)) testthat::skip("R/AI4BayesCode_helpers.R is not available")
    f
}

test_that("the checkout copy defines every shared function identically", {
    f <- helpers_file()
    defs <- function(file) {
        out <- list()
        for (e in parse(file, keep.source = FALSE))
            if (is.call(e) && identical(e[[1L]], as.name("<-")) && is.name(e[[2L]]))
                out[[as.character(e[[2L]])]] <- deparse(e[[3L]])
        out
    }
    pkg_files <- list.files(testthat::test_path("..", "..", "R"), full.names = TRUE)
    skip_if(!length(pkg_files), "the package sources are not available")
    a <- do.call(c, lapply(pkg_files, defs))
    b <- defs(f)
    shared <- intersect(names(a), names(b))
    expect_gt(length(shared), 40L)
    differ <- shared[!vapply(shared, function(n) identical(a[[n]], b[[n]]), logical(1))]
    expect_identical(differ, character(0))
})

test_that("the checkout copy sourced into another environment runs chains in workers", {
    skip_on_cran(); skip_on_os("windows")
    f <- helpers_file()
    script <- tempfile(fileext = ".R")
    out    <- tempfile(fileext = ".txt")
    writeLines(c(
        "setTimeLimit(elapsed = 120)",
        "ctor <- function(seed) {",
        "  e <- new.env(); e$s <- seed; e$d <- numeric(0)",
        "  e$step <- function(n = 1L) for (i in seq_len(n)) { e$s <- e$s + 1; e$d <- c(e$d, e$s) }",
        "  e$get_history <- function() list(theta = e$d, pid = rep(Sys.getpid(), length(e$d)))",
        "  e$clear_history <- function() e$d <- numeric(0)",
        "  e$predict_at <- function(newdata) list(v = e$d)",
        "  e }",
        "res <- character(0)",
        "check <- function(env, mode) {",
        "  run <- env$ai4bayescode_run_chains(ctor, n_chains = 3, n_burn = 1, n_keep = 2, n_cores = 2, verbose = FALSE)",
        "  pids <- vapply(run$histories, function(h) h$pid[1], numeric(1))",
        "  ok <- c(all(pids != Sys.getpid()), inherits(run$models[[1]], 'ai4bayescode_chain_model'),",
        "          identical(run$models[[3]]$get_history()$theta, c(305, 306)),",
        "          identical(predict(run)$v, c(103, 104, 204, 205, 305, 306)),",
        "          any(grepl('worker processes', capture.output(print(run)))))",
        "  env$ai4bayescode_close_models(run)",
        "  res <<- c(res, paste(mode, paste(ok, collapse = ' ')))",
        "}",
        sprintf("e <- new.env(); sys.source(%s, envir = e); check(e, 'sys.source')",
                deparse(f)),
        sprintf("g <- function() { source(%s, local = TRUE); check(environment(), 'local') }; g()",
                deparse(f)),
        sprintf("writeLines(res, %s)", deparse(out))), script)
    rscript <- file.path(R.home("bin"), "Rscript")
    status <- system2(rscript, c("--vanilla", shQuote(script)),
                      stdout = FALSE, stderr = FALSE)
    expect_identical(status, 0L)
    expect_identical(readLines(out), c("sys.source TRUE TRUE TRUE TRUE TRUE",
                                       "local TRUE TRUE TRUE TRUE TRUE"))
})

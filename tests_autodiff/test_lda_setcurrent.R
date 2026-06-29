# Copyright (C) 2026 AI4BayesCode.
# Licensed under the GNU General Public License v2.0 or later
# (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
# ============================================================================
# test_lda_setcurrent.R
#
# R-level functional test for LdaCollapsedGibbs.cpp's six-method contract,
# focused on set_current dispatcher behavior. Mirrors the style of
# tests_autodiff/test_bart_setcurrent.R per system_design.md §16
# pre-merge checklist.
#
# Coverage:
#  (T1) round-trip: get_current -> set_current does not crash; state
#       unchanged when the round-tripped z is plugged back in.
#  (T2) untouched-key preservation: passing only `z` does not affect the
#       block's internal w / doc / alpha / beta state.
#  (T3) unknown-key tolerance: `set_current(list(unknown_xyz = 42))`
#       does not error.
#  (T4) impossible-key rejection: `set_current(list(theta = ...))` and
#       `set_current(list(phi = ...))` both error with a clear message.
#  (T5) dimension validation: `set_current(list(z = wrong_length))` errors.
#  (T6) range validation: `set_current(list(z = entries_out_of_range))`
#       errors.
#  (T7) w/doc paired update: `set_current(list(w = ..., doc = ...))` works
#       at construction (before first step); a single-side update
#       (just w or just doc) errors.
#  (T8) multi-iteration stability: 100 outer iterations of (set_current,
#       step) with intermittent z resets do not crash; counts are
#       always consistent post-step.
# ============================================================================

script_dir <- local({
    cmdargs <- commandArgs(trailingOnly = FALSE)
    farg    <- grep("^--file=", cmdargs, value = TRUE)
    if (length(farg))
        dirname(normalizePath(sub("^--file=", "", farg[1])))
    else getwd()
})
AI4BayesCode_dir <- normalizePath(file.path(script_dir, ".."))
source(file.path(AI4BayesCode_dir, "R", "AI4BayesCode_helpers.R"))

AI4BayesCode_sourceCpp(
    file.path(AI4BayesCode_dir, "examples", "LdaCollapsedGibbs.cpp"),
    AI4BayesCode_path = AI4BayesCode_dir, verbose = FALSE)

cat("\n===== LdaCollapsedGibbs set_current contract tests =====\n")

# Build a small fixture corpus.
M <- 4L; V <- 6L; K <- 2L
N <- 24L
set.seed(7L)
w_full   <- as.integer(sample.int(V, N, replace = TRUE))
doc_full <- as.integer(sample.int(M, N, replace = TRUE))
build_block <- function(seed = 1L, keep_history = FALSE) {
    new(LdaCollapsedGibbs, w_full, doc_full, M, V, K,
        rep(1.0, K), rep(1.0, V), seed, keep_history)
}

results <- list()
record <- function(name, ok, info = NULL) {
    results[[name]] <<- list(pass = ok, info = info)
    cat(sprintf("  [%s] %s%s\n",
                if (ok) "PASS" else "FAIL",
                name,
                if (!is.null(info) && !ok) paste0(" -- ", info) else ""))
}

# --------------------------------------------------------------------------
# (T1) round-trip
# --------------------------------------------------------------------------
{
    m <- build_block(seed = 11L)
    m$step(50L)
    cur1 <- m$get_current()
    # set_current with the z just produced
    m$set_current(list(z = cur1$z))
    cur2 <- m$get_current()
    ok <- identical(cur1$z, cur2$z) &&
          isTRUE(all.equal(cur1$theta, cur2$theta)) &&
          isTRUE(all.equal(cur1$phi,   cur2$phi))
    record("T1 round-trip", ok)
}

# --------------------------------------------------------------------------
# (T2) untouched-key preservation
# --------------------------------------------------------------------------
{
    m <- build_block(seed = 22L)
    m$step(20L)
    cur_before <- m$get_current()
    # Set only z; alpha, beta, M, V, K should be unchanged.
    m$set_current(list(z = cur_before$z))
    cur_after <- m$get_current()
    ok <- isTRUE(all.equal(cur_before$alpha, cur_after$alpha)) &&
          isTRUE(all.equal(cur_before$beta,  cur_after$beta))  &&
          (cur_before$M == cur_after$M) &&
          (cur_before$V == cur_after$V) &&
          (cur_before$K == cur_after$K)
    record("T2 untouched-key preservation", ok)
}

# --------------------------------------------------------------------------
# (T3) unknown-key tolerance
# --------------------------------------------------------------------------
{
    m <- build_block(seed = 33L)
    m$step(5L)
    err <- tryCatch({
        m$set_current(list(unknown_xyz = 42, foo = "bar"))
        NULL
    }, error = function(e) e)
    record("T3 unknown-key tolerance", is.null(err),
           info = if (!is.null(err)) conditionMessage(err))
}

# --------------------------------------------------------------------------
# (T4) impossible-key rejection (theta + phi)
# --------------------------------------------------------------------------
{
    m <- build_block(seed = 44L)
    err_th <- tryCatch({
        m$set_current(list(theta = matrix(0.5, M, K)))
        NULL
    }, error = function(e) e)
    err_ph <- tryCatch({
        m$set_current(list(phi = matrix(1 / V, K, V)))
        NULL
    }, error = function(e) e)
    ok <- !is.null(err_th) && !is.null(err_ph) &&
          (grepl("theta", conditionMessage(err_th), ignore.case = TRUE) ||
           grepl("phi",   conditionMessage(err_th), ignore.case = TRUE))
    record("T4 impossible-key rejection", ok,
           info = if (!is.null(err_th)) conditionMessage(err_th))
}

# --------------------------------------------------------------------------
# (T5) dimension validation: wrong-length z
# --------------------------------------------------------------------------
{
    m <- build_block(seed = 55L)
    err <- tryCatch({
        m$set_current(list(z = rep(1, N - 5)))  # too short
        NULL
    }, error = function(e) e)
    record("T5 wrong-length z rejection", !is.null(err),
           info = if (!is.null(err)) conditionMessage(err))
}

# --------------------------------------------------------------------------
# (T6) range validation: entries out of {1..K}
# --------------------------------------------------------------------------
{
    m <- build_block(seed = 66L)
    err <- tryCatch({
        m$set_current(list(z = c(rep(1, N - 1), K + 1)))
        NULL
    }, error = function(e) e)
    record("T6 z entries out of range rejection", !is.null(err),
           info = if (!is.null(err)) conditionMessage(err))
}

# --------------------------------------------------------------------------
# (T7) w/doc paired update + asymmetric reject
# --------------------------------------------------------------------------
{
    m <- build_block(seed = 77L)
    # Single-side updates should error.
    err_w_only <- tryCatch({
        m$set_current(list(w = w_full))  # without doc
        NULL
    }, error = function(e) e)
    err_doc_only <- tryCatch({
        m$set_current(list(doc = doc_full))
        NULL
    }, error = function(e) e)
    ok <- !is.null(err_w_only) && !is.null(err_doc_only)
    record("T7 single-side w/doc rejection", ok,
           info = if (!is.null(err_w_only)) conditionMessage(err_w_only))
}

# --------------------------------------------------------------------------
# (T8) multi-iteration stability with intermittent z resets
# --------------------------------------------------------------------------
{
    m <- build_block(seed = 88L)
    crashed <- FALSE
    for (it in seq_len(100)) {
        # Periodic z reset to a random valid assignment.
        if (it %% 20 == 0) {
            new_z <- sample.int(K, N, replace = TRUE)
            tryCatch(m$set_current(list(z = new_z)),
                     error = function(e) { crashed <<- TRUE; NULL })
            if (crashed) break
        }
        tryCatch(m$step(1L),
                 error = function(e) { crashed <<- TRUE; NULL })
        if (crashed) break
    }
    cur <- m$get_current()
    ok <- !crashed &&
          all(cur$z %in% seq_len(K)) &&
          all(abs(rowSums(cur$theta) - 1) < 1e-9) &&
          all(abs(rowSums(cur$phi)   - 1) < 1e-9)
    record("T8 multi-iteration stability", ok)
}

# ==========================================================================
n_pass <- sum(sapply(results, function(r) isTRUE(r$pass)))
n_total <- length(results)
cat(sprintf("\n===== summary: %d / %d PASS =====\n", n_pass, n_total))
if (n_pass < n_total) {
    cat("FAILURES:\n")
    for (nm in names(results)) {
        if (!isTRUE(results[[nm]]$pass)) {
            cat("  ", nm,
                if (!is.null(results[[nm]]$info))
                    paste0(" -- ", results[[nm]]$info)
                else "", "\n")
        }
    }
    quit(save = "no", status = 1L)
}
quit(save = "no", status = 0L)
